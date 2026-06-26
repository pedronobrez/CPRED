"""
CLI entry point for CPRED — Protease Cleavage Site Predictor.

Usage:
  cpred.py --protease ADAM10 --organism mouse --genes ADAM10,MMP2 --out ./results
  cpred.py --protease ADAM10 --organism mouse --fasta sequences.fasta --out ./results
"""

import argparse
import sys
from pathlib import Path


def _parse_genes(raw: str) -> list[str]:
    import re
    return [g.strip() for g in re.split(r"[,\n]+", raw) if g.strip()]


def _run_from_genes(
    protease: str, organism: str, genes: list[str], out_dir: Path,
    min_top_hits: int = 4, _write: bool = True,
) -> dict:
    from engine.pipeline import run_job

    print(f"[cpred] protease={protease} organism={organism} genes={genes} min_top_hits={min_top_hits}")
    result = run_job({
        "protease_gene": protease,
        "organism": organism,
        "genes": genes,
        "min_top_hits": min_top_hits,
    })
    if _write:
        _write_outputs(result, out_dir)
    return result


def _run_from_fasta(
    protease: str, organism: str, fasta_path: Path, out_dir: Path,
    min_top_hits: int = 4, _write: bool = True,
) -> dict:
    from engine import protein_match, reference_sources, substrate_bank

    REFERENCE_TO_BANK = {
        "P4": "P4", "P3": "P3", "P2": "P2", "P1": "P1",
        "P1prime": "P1'", "P2prime": "P2'", "P3prime": "P3'", "P4prime": "P4'",
    }

    print(f"[cpred] protease={protease} organism={organism} fasta={fasta_path} min_top_hits={min_top_hits}")

    reference = reference_sources.load_predefined_reference(protease, organism)
    positions = [REFERENCE_TO_BANK[p] for p in reference["positions"]]
    positional_preferences = {
        REFERENCE_TO_BANK[p]: aa
        for p, aa in reference["positional_preferences"].items()
    }

    generated_bank, _ = substrate_bank.generate_weighted_substrate_bank(
        positional_preferences=positional_preferences,
        positions=positions,
        min_top_hits=min_top_hits,
    )

    records = protein_match.load_fasta_records(fasta_path)
    if not records:
        sys.exit(f"[cpred] ERROR: No records found in {fasta_path}")

    print(f"[cpred] loaded {len(records)} sequence(s) from FASTA")
    match_results = protein_match.analyze_protein_records(records, generated_bank)

    result = {
        "site_results":    match_results["site_results"],
        "protein_ranking": match_results["protein_ranking"],
        "summary": {
            **match_results["summary"],
            "protease_gene": protease,
            "organism":      organism,
        },
    }
    if _write:
        _write_outputs(result, out_dir)
    return result


def _apply_rsa_filter(site_results: list, threshold: float, verbose: bool) -> list:
    from engine.rsa_filter import annotate_and_filter

    filtered, stats = annotate_and_filter(site_results, threshold=threshold, verbose=verbose)
    print(
        f"[cpred] RSA filter (threshold={threshold}): "
        f"{stats['n_passed']} accessible, "
        f"{stats['n_filtered']} buried (removed), "
        f"{stats['n_no_structure']} kept (no structure)"
    )
    return filtered


def _apply_rf_scorer(site_results: list, model_path: Path) -> list:
    from engine.ml_scorer import load_model, score_records

    model = load_model(str(model_path))
    site_results = score_records(model, site_results)
    print(f"[cpred] RF scorer applied from {model_path}")
    return site_results


def _write_outputs(result: dict, out_dir: Path) -> None:
    import json
    from engine.protein_match import (
        export_site_results_csv,
        export_protein_ranking_csv,
    )

    out_dir.mkdir(parents=True, exist_ok=True)

    sites_path   = out_dir / "site_details.csv"
    ranking_path = out_dir / "protein_ranking.csv"
    summary_path = out_dir / "summary.json"

    export_site_results_csv(result["site_results"], sites_path)
    export_protein_ranking_csv(result["protein_ranking"], ranking_path)

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(result["summary"], f, indent=2)

    summary = result["summary"]
    print(
        f"[cpred] done — "
        f"total_sites={summary.get('total_sites')} "
        f"proteins_with_sites={summary.get('proteins_with_sites')} "
        f"total_proteins={summary.get('total_proteins')}"
    )
    print(f"[cpred] site_details.csv    -> {sites_path}")
    print(f"[cpred] protein_ranking.csv -> {ranking_path}")
    print(f"[cpred] summary.json        -> {summary_path}")


def main() -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

    parser = argparse.ArgumentParser(
        prog="cpred",
        description="Predict protease cleavage sites in protein sequences.",
    )
    parser.add_argument(
        "--protease", required=True,
        help="Protease gene symbol (e.g. ADAM10, MMP2)",
    )
    parser.add_argument(
        "--organism", required=True,
        help="Organism (e.g. mouse, human, rat, cow)",
    )
    parser.add_argument(
        "--genes", default=None,
        help="Comma-separated gene symbols to fetch from UniProt",
    )
    parser.add_argument(
        "--fasta", default=None,
        help="Path to a FASTA file with protein sequences",
    )
    parser.add_argument(
        "--out", required=True,
        help="Output directory for CSV files",
    )
    parser.add_argument(
        "--rsa-threshold", type=float, default=None, metavar="FLOAT",
        help=(
            "RSA cutoff for structural accessibility filter (0.0–1.0). "
            "Sites with RSA < threshold in both P1 and P1' are removed. "
            "Default: disabled. Recommended: 0.15."
        ),
    )
    parser.add_argument(
        "--rf-model", default=None, metavar="PATH",
        help="Path to a trained RF model (.pkl) for rescoring sites with rf_score.",
    )
    parser.add_argument(
        "--min-top-hits", type=int, default=4, metavar="INT",
        help=(
            "Minimum number of positions (of 8) that must have a top-k amino acid "
            "for a motif to enter the substrate bank. Default: 4. "
            "Lower values (e.g. 3) increase sensitivity at the cost of more candidates."
        ),
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print per-site RSA filter decisions.",
    )
    args = parser.parse_args()

    if args.genes is None and args.fasta is None:
        parser.error("Provide --genes or --fasta (at least one is required).")
    if args.genes is not None and args.fasta is not None:
        parser.error("Use either --genes or --fasta, not both.")

    out_dir = Path(args.out)

    if args.fasta:
        fasta_path = Path(args.fasta)
        if not fasta_path.exists():
            sys.exit(f"[cpred] ERROR: FASTA file not found: {fasta_path}")
        result = _run_from_fasta(args.protease, args.organism, fasta_path, out_dir,
                                 min_top_hits=args.min_top_hits, _write=False)
    else:
        genes = _parse_genes(args.genes)
        if not genes:
            sys.exit("[cpred] ERROR: --genes produced an empty list after parsing.")
        result = _run_from_genes(args.protease, args.organism, genes, out_dir,
                                 min_top_hits=args.min_top_hits, _write=False)

    if args.rsa_threshold is not None:
        result["site_results"] = _apply_rsa_filter(
            result["site_results"], args.rsa_threshold, args.verbose
        )

    if args.rf_model is not None:
        rf_path = Path(args.rf_model)
        if not rf_path.exists():
            sys.exit(f"[cpred] ERROR: RF model not found: {rf_path}")
        result["site_results"] = _apply_rf_scorer(result["site_results"], rf_path)

    _write_outputs(result, out_dir)


if __name__ == "__main__":
    main()
