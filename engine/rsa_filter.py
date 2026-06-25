"""
RSA (Relative Solvent Accessibility) filter for CPRED predictions.

For each predicted cleavage site, fetches the AlphaFold PDB structure,
computes per-residue RSA using ShrakeRupley (BioPython) and parses secondary
structure from PDB HELIX/SHEET records, and flags sites where P1 or P1'
falls below the accessibility threshold.

P1' position = start_1based + 4  (5th residue of the P4–P4' octamer)

Fields added to each site record by annotate_and_filter:
  rsa_p1, rsa_p1prime, rsa_mean         — core RSA (backwards-compat)
  rsa_p4 … rsa_p4prime                  — per-position RSA (all 8 positions)
  ss_p1, ss_p1prime                     — secondary structure ('H'/'E'/'C')
  rsa_accessible                        — bool: passed threshold
"""

from __future__ import annotations

import os
from typing import Any

import requests

ALPHAFOLD_API = "https://alphafold.ebi.ac.uk/api/prediction/{accession}"
RSA_THRESHOLD = 0.15
CACHE_DIR = ".af_cache"

# Position labels for the P4–P4' window relative to P1'
_WINDOW_OFFSETS = list(range(-4, 4))   # -4 … +3  (P4 … P4')
_WINDOW_KEYS    = ["p4", "p3", "p2", "p1", "p1prime", "p2prime", "p3prime", "p4prime"]


def fetch_alphafold_pdb(accession: str, cache_dir: str = CACHE_DIR) -> str | None:
    """
    Download AlphaFold PDB and cache locally. Returns local path or None.
    Queries the AlphaFold API first to obtain the correct versioned PDB URL.
    """
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"{accession}.pdb")
    if os.path.exists(path):
        return path
    try:
        api_url = ALPHAFOLD_API.format(accession=accession)
        meta = requests.get(api_url, timeout=15)
        meta.raise_for_status()
        pdb_url = meta.json()[0]["pdbUrl"]
        r = requests.get(pdb_url, timeout=30)
        r.raise_for_status()
        with open(path, "w") as f:
            f.write(r.text)
        return path
    except (requests.RequestException, KeyError, IndexError):
        return None


def compute_rsa(pdb_path: str) -> dict[int, float] | None:
    """
    Run ShrakeRupley on a PDB file and return {residue_seq_id: rsa} (0.0–1.0).
    Returns None on failure.
    """
    try:
        from Bio.PDB import PDBParser, DSSP as BiopythonDSSP

        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("protein", pdb_path)
        model = structure[0]
        dssp = BiopythonDSSP(model, pdb_path, dssp="mkdssp")
        rsa_map: dict[int, float] = {}
        for key in dssp.keys():
            _chain, res = key
            res_id = res[1]
            rsa_val = dssp[key][3]
            if rsa_val is not None:
                rsa_map[res_id] = float(rsa_val)
        return rsa_map
    except Exception:
        return _compute_rsa_pydssp(pdb_path)


def _compute_rsa_pydssp(pdb_path: str) -> dict[int, float] | None:
    """Fallback: use Bio.PDB.SASA.ShrakeRupley (no external executable needed)."""
    try:
        from Bio.PDB import PDBParser
        from Bio.PDB.SASA import ShrakeRupley

        MAX_ASA = {
            "ALA": 106, "ARG": 248, "ASN": 157, "ASP": 163, "CYS": 135,
            "GLN": 198, "GLU": 194, "GLY": 84,  "HIS": 184, "ILE": 169,
            "LEU": 164, "LYS": 205, "MET": 188, "PHE": 197, "PRO": 136,
            "SER": 130, "THR": 142, "TRP": 227, "TYR": 222, "VAL": 142,
        }
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("prot", pdb_path)
        sr = ShrakeRupley()
        sr.compute(structure, level="R")

        rsa_map: dict[int, float] = {}
        for residue in structure[0].get_residues():
            res_id = residue.get_id()[1]
            resname = residue.get_resname()
            asa = residue.sasa
            max_asa = MAX_ASA.get(resname, 200)
            rsa_map[res_id] = min(asa / max_asa, 1.0)
        return rsa_map if rsa_map else None
    except Exception:
        return None


def compute_ss(pdb_path: str) -> dict[int, str]:
    """
    Parse secondary structure from PDB HELIX/SHEET records.

    Returns {res_id: 'H'|'E'} for helix/sheet residues.
    Residues not in any record default to 'C' (coil/loop) at lookup time.

    AlphaFold PDB files include HELIX and SHEET records derived from the
    predicted structure, so no external DSSP executable is needed.
    """
    ss_map: dict[int, str] = {}
    try:
        with open(pdb_path) as f:
            for line in f:
                if line.startswith("HELIX "):
                    try:
                        start = int(line[21:25])
                        end   = int(line[33:37])
                        for i in range(start, end + 1):
                            ss_map[i] = "H"
                    except ValueError:
                        pass
                elif line.startswith("SHEET "):
                    try:
                        start = int(line[22:26])
                        end   = int(line[33:37])
                        for i in range(start, end + 1):
                            ss_map.setdefault(i, "E")  # helix takes priority
                    except ValueError:
                        pass
    except OSError:
        pass
    return ss_map


def _p1prime_1based(rec: dict[str, Any]) -> int:
    """P1' residue number (1-based): 5th residue of the P4–P4' octamer."""
    return int(rec["start_1based"]) + 4


def get_site_rsa(
    rsa_map: dict[int, float],
    p1prime: int,
) -> dict[str, float | None]:
    """Return RSA values for P1, P1', and the full P4–P4' window (backwards compat)."""
    p1 = p1prime - 1
    window_vals = [rsa_map.get(p1prime + off) for off in _WINDOW_OFFSETS]
    window_valid = [v for v in window_vals if v is not None]
    return {
        "rsa_p1":      rsa_map.get(p1),
        "rsa_p1prime": rsa_map.get(p1prime),
        "rsa_mean":    (sum(window_valid) / len(window_valid)) if window_valid else None,
    }


def get_site_features(
    rsa_map: dict[int, float],
    ss_map: dict[int, str],
    p1prime: int,
) -> dict[str, Any]:
    """
    Return all structural features for one cleavage site:
      - rsa_<pos> for each of the 8 positions (P4–P4')
      - rsa_p1, rsa_p1prime, rsa_mean (backwards-compat aliases already in above)
      - ss_p1, ss_p1prime ('H', 'E', or 'C')
    """
    result: dict[str, Any] = {}

    for key, off in zip(_WINDOW_KEYS, _WINDOW_OFFSETS):
        res_id = p1prime + off
        result[f"rsa_{key}"] = rsa_map.get(res_id)

    # backwards-compat aliases (p1 is offset -1 from p1prime)
    result["rsa_p1"]      = result["rsa_p1"]       # already set via offset -1
    result["rsa_p1prime"] = result["rsa_p1prime"]   # offset 0
    window_valid = [v for v in result.values() if isinstance(v, float)]
    result["rsa_mean"] = sum(window_valid) / len(window_valid) if window_valid else None

    result["ss_p1"]      = ss_map.get(p1prime - 1, "C")
    result["ss_p1prime"] = ss_map.get(p1prime, "C")

    return result


def is_accessible(
    rsa_map: dict[int, float],
    p1prime_1based: int,
    threshold: float = RSA_THRESHOLD,
) -> bool:
    """
    True if at least one of P1 or P1' has RSA >= threshold.
    Conservatively keeps sites where both residues are missing from the map.
    """
    p1 = p1prime_1based - 1
    rsa_p1 = rsa_map.get(p1)
    rsa_p1prime = rsa_map.get(p1prime_1based)
    if rsa_p1 is None and rsa_p1prime is None:
        return True
    for rsa in (rsa_p1, rsa_p1prime):
        if rsa is not None and rsa >= threshold:
            return True
    return False


def annotate_and_filter(
    site_results: list[dict[str, Any]],
    threshold: float = RSA_THRESHOLD,
    cache_dir: str = CACHE_DIR,
    verbose: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Annotate each site with RSA + secondary structure and filter by accessibility.

    Adds fields to each passing record:
      rsa_p4 … rsa_p4prime   — per-position RSA (all 8 window positions)
      rsa_p1, rsa_p1prime, rsa_mean  — core RSA (backwards-compat)
      ss_p1, ss_p1prime      — secondary structure at P1 / P1' ('H'/'E'/'C')
      rsa_accessible         — bool

    Returns:
      accessible — sites that passed the filter (or had no structure)
      stats      — n_total, n_passed, n_filtered, n_no_structure
    """
    accessible: list[dict[str, Any]] = []
    stats: dict[str, Any] = {
        "n_total": len(site_results),
        "n_passed": 0,
        "n_filtered": 0,
        "n_no_structure": 0,
    }

    rsa_cache: dict[str, dict[int, float] | None] = {}
    ss_cache:  dict[str, dict[int, str]] = {}

    for rec in site_results:
        pid = rec["protein_id"]

        if pid not in rsa_cache:
            pdb_path = fetch_alphafold_pdb(pid, cache_dir=cache_dir)
            rsa_cache[pid] = compute_rsa(pdb_path) if pdb_path else None
            ss_cache[pid]  = compute_ss(pdb_path)  if pdb_path else {}

        rsa_map = rsa_cache[pid]
        ss_map  = ss_cache[pid]

        if rsa_map is None:
            # No structure: impute neutral values, keep site
            rec["rsa_p1"]      = None
            rec["rsa_p1prime"] = None
            rec["rsa_mean"]    = None
            for key in _WINDOW_KEYS:
                rec[f"rsa_{key}"] = None
            rec["ss_p1"]       = "C"
            rec["ss_p1prime"]  = "C"
            rec["rsa_accessible"] = None
            accessible.append(rec)
            stats["n_no_structure"] += 1
            continue

        p1prime = _p1prime_1based(rec)
        feats = get_site_features(rsa_map, ss_map, p1prime)
        accessible_flag = is_accessible(rsa_map, p1prime, threshold=threshold)

        rec.update(feats)
        rec["rsa_accessible"] = accessible_flag

        if accessible_flag:
            accessible.append(rec)
            stats["n_passed"] += 1
        else:
            stats["n_filtered"] += 1
            if verbose:
                print(
                    f"  [RSA] {pid} site@{rec['start_1based']} "
                    f"P1={feats['rsa_p1']:.2f} P1'={feats['rsa_p1prime']:.2f} -> buried"
                )

    return accessible, stats


__all__ = [
    "annotate_and_filter",
    "compute_rsa",
    "compute_ss",
    "fetch_alphafold_pdb",
    "get_site_features",
    "get_site_rsa",
    "is_accessible",
    "RSA_THRESHOLD",
]
