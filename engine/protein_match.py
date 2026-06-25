import csv
import time
from pathlib import Path
from typing import Any
from typing import Iterable


def match_motif(site: str, motif: str) -> bool:
    """
    Return True when a site matches a motif, treating 'X' as a wildcard.
    """
    assert len(site) == len(motif)
    for amino_acid, motif_amino_acid in zip(site, motif):
        if motif_amino_acid == "X":
            continue
        if amino_acid != motif_amino_acid:
            return False
    return True


def iter_sequence_windows(sequence: str, window_size: int = 8):
    """
    Yield contiguous fixed-size windows across a protein sequence.
    """
    sequence_length = len(sequence)
    for start in range(0, sequence_length - window_size + 1):
        end = start + window_size
        yield start, end, sequence[start:end]


def find_matching_sites_with_scores(
    sequence: str,
    substrate_bank: dict[str, dict[str, Any]],
    window_size: int = 8,
) -> list[dict[str, Any]]:
    """
    Find all motif matches in a sequence using the provided substrate bank.
    """
    site_matches: list[dict[str, Any]] = []
    motif_items = list(substrate_bank.items())

    for start, end, site_sequence in iter_sequence_windows(sequence, window_size=window_size):
        for motif, info in motif_items:
            if match_motif(site_sequence, motif):
                site_matches.append(
                    {
                        "start": start,
                        "end": end,
                        "site_seq": site_sequence,
                        "motif": motif,
                        "top_hits_motif": info["top_hits"],
                        "motif_score": info["motif_score"],
                        "mask_top": info["mask_top"],
                        "motif_details": info["details"],
                    }
                )

    return site_matches


def select_best_site_per_position(site_matches: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Collapse duplicate matches at the same position, keeping the highest motif score.
    """
    best_site_by_position: dict[tuple[int, int], dict[str, Any]] = {}

    for site_match in site_matches:
        key = (site_match["start"], site_match["end"])
        if key not in best_site_by_position:
            best_site_by_position[key] = site_match
        elif site_match["motif_score"] > best_site_by_position[key]["motif_score"]:
            best_site_by_position[key] = site_match

    return list(best_site_by_position.values())


def _record_field(record: Any, field_name: str, default: Any = "") -> Any:
    if isinstance(record, dict):
        return record.get(field_name, default)
    return getattr(record, field_name, default)


def _protein_progress_label(record: Any) -> Any:
    gene = _record_field(record, "gene")
    if gene not in ("", None, "UNKNOWN"):
        return gene
    return _record_field(record, "id")


def analyze_protein_record(
    record: Any,
    substrate_bank: dict[str, dict[str, Any]],
    window_size: int = 8,
) -> dict[str, Any]:
    """
    Analyze one in-memory protein record against the substrate bank.
    """
    sequence = str(_record_field(record, "seq"))
    raw_site_matches = find_matching_sites_with_scores(
        sequence,
        substrate_bank,
        window_size=window_size,
    )
    unique_sites = select_best_site_per_position(raw_site_matches)
    site_count = len(unique_sites)

    if site_count > 0:
        best_site_score = max(site["motif_score"] for site in unique_sites)
        sum_site_score = sum(site["motif_score"] for site in unique_sites)
        average_site_score = sum_site_score / site_count
    else:
        best_site_score = 0.0
        sum_site_score = 0.0
        average_site_score = 0.0

    return {
        "id": _record_field(record, "id"),
        "gene": _record_field(record, "gene"),
        "entry_name": _record_field(record, "entry_name"),
        "description": _record_field(record, "description"),
        "length": len(sequence),
        "num_sites": site_count,
        "best_site_score": best_site_score,
        "sum_site_score": sum_site_score,
        "avg_site_score": average_site_score,
        "sites": unique_sites,
    }


def rank_protein_results(protein_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Rank proteins by total score, then best site score, then number of sites.
    """
    return sorted(
        protein_results,
        key=lambda result: (
            result["sum_site_score"],
            result["best_site_score"],
            result["num_sites"],
        ),
        reverse=True,
    )


def flatten_site_results(protein_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Flatten per-protein sites into a single list for downstream export or inspection.
    """
    site_results: list[dict[str, Any]] = []

    for protein_result in protein_results:
        protein_id = protein_result["id"]
        for site_index, site in enumerate(protein_result["sites"]):
            start_0based = site["start"]
            end_0based = site["end"]
            site_results.append(
                {
                    "protein_id": protein_id,
                    "site_index": site_index,
                    "start_0based": start_0based,
                    "end_0based": end_0based,
                    "start_1based": start_0based + 1,
                    "end_1based": end_0based,
                    "site_seq": site["site_seq"],
                    "motif": site["motif"],
                    "motif_score": site["motif_score"],
                    "top_hits_motif": site["top_hits_motif"],
                    "mask_top": site["mask_top"],
                    "motif_details": site["motif_details"],
                }
            )

    return site_results


def summarize_match_results(
    protein_results: list[dict[str, Any]],
    protein_ranking: list[dict[str, Any]],
    window_size: int,
) -> dict[str, Any]:
    """
    Build a compact summary of the protein matching run.
    """
    proteins_with_sites = sum(1 for result in protein_results if result["num_sites"] > 0)
    total_sites = sum(result["num_sites"] for result in protein_results)

    return {
        "total_proteins": len(protein_results),
        "proteins_with_sites": proteins_with_sites,
        "total_sites": total_sites,
        "window_size": window_size,
        "top_ranked_protein_id": protein_ranking[0]["id"] if protein_ranking else None,
        "top_ranked_protein_name": protein_ranking[0]["description"] if protein_ranking else None,
    }


def analyze_protein_records(
    records: Iterable[Any],
    substrate_bank: dict[str, dict[str, Any]],
    window_size: int = 8,
) -> dict[str, Any]:
    """
    Analyze in-memory protein records and return ranked proteins, sites, and summary.
    """
    records = list(records)
    total_records = len(records)

    matching_started_at = time.perf_counter()
    protein_results = []
    for index, record in enumerate(records, start=1):
        sequence_length = len(str(_record_field(record, "seq")))
        window_count = max(sequence_length - window_size + 1, 0)
        protein_started_at = time.perf_counter()
        protein_results.append(
            analyze_protein_record(record, substrate_bank, window_size=window_size)
        )
        print(
            f"[matching] {index}/{total_records} | protein={_protein_progress_label(record)} "
            f"| length={sequence_length} | windows={window_count} "
            f"| elapsed={time.perf_counter() - protein_started_at:.3f}s"
        )
    print(f"[timing] matching: {time.perf_counter() - matching_started_at:.3f}s")

    ranking_started_at = time.perf_counter()
    protein_ranking = rank_protein_results(protein_results)
    site_results = flatten_site_results(protein_ranking)
    summary = summarize_match_results(
        protein_results,
        protein_ranking,
        window_size=window_size,
    )
    print(f"[timing] ranking/export: {time.perf_counter() - ranking_started_at:.3f}s")

    return {
        "protein_results": protein_results,
        "protein_ranking": protein_ranking,
        "site_results": site_results,
        "summary": summary,
    }


def load_fasta_records(fasta_path: str | Path) -> list[Any]:
    """
    Load Biopython SeqRecord objects from one FASTA file.
    """
    from Bio import SeqIO

    fasta_path = Path(fasta_path)
    return list(SeqIO.parse(str(fasta_path), "fasta"))


def load_fasta_records_from_folder(folder: str | Path, pattern: str = "*.fasta") -> list[Any]:
    """
    Load Biopython SeqRecord objects from all matching FASTA files in a folder.
    """
    from Bio import SeqIO

    folder = Path(folder)
    records: list[Any] = []
    for fasta_file in folder.glob(pattern):
        records.extend(SeqIO.parse(str(fasta_file), "fasta"))
    return records


def analyze_fasta_folder(
    folder: str | Path,
    substrate_bank: dict[str, dict[str, Any]],
    window_size: int = 8,
    pattern: str = "*.fasta",
) -> dict[str, Any]:
    """
    Convenience wrapper that loads FASTA records from a folder before analysis.
    """
    records = load_fasta_records_from_folder(folder, pattern=pattern)
    return analyze_protein_records(records, substrate_bank, window_size=window_size)


def export_protein_ranking_csv(
    protein_ranking: list[dict[str, Any]],
    output_path: str | Path,
) -> None:
    """
    Export one ranked row per protein.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    field_names = [
        "protein_id",
        "gene",
        "description",
        "length",
        "num_sites",
        "sum_site_score",
        "best_site_score",
        "avg_site_score",
    ]

    with output_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(field_names)
        for result in protein_ranking:
            writer.writerow(
                [
                    result["id"],
                    result.get("gene", ""),
                    result["description"],
                    result["length"],
                    result["num_sites"],
                    result["sum_site_score"],
                    result["best_site_score"],
                    result["avg_site_score"],
                ]
            )


def export_site_results_csv(
    site_results: list[dict[str, Any]],
    output_path: str | Path,
) -> None:
    """
    Export one row per matched site.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    base_fields = [
        "protein_id",
        "site_index",
        "start_0based",
        "end_0based",
        "start_1based",
        "end_1based",
        "site_seq",
        "motif",
        "motif_score",
        "top_hits_motif",
    ]
    # Append RSA and RF columns only when present in the first record
    extra_fields: list[str] = []
    if site_results:
        first = site_results[0]
        for col in ("rsa_p1", "rsa_p1prime", "rsa_mean", "rsa_accessible", "rf_score"):
            if col in first:
                extra_fields.append(col)

    field_names = base_fields + extra_fields

    with output_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(field_names)
        for site_result in site_results:
            row = [
                site_result["protein_id"],
                site_result["site_index"],
                site_result["start_0based"],
                site_result["end_0based"],
                site_result["start_1based"],
                site_result["end_1based"],
                site_result["site_seq"],
                site_result["motif"],
                site_result["motif_score"],
                site_result["top_hits_motif"],
            ]
            for col in extra_fields:
                row.append(site_result.get(col, ""))
            writer.writerow(row)


def export_match_results(
    analysis_results: dict[str, Any],
    proteins_output_path: str | Path,
    sites_output_path: str | Path,
) -> None:
    """
    Export ranked proteins and flattened site results to separate CSV files.
    """
    export_protein_ranking_csv(analysis_results["protein_ranking"], proteins_output_path)
    export_site_results_csv(analysis_results["site_results"], sites_output_path)


__all__ = [
    "analyze_fasta_folder",
    "analyze_protein_record",
    "analyze_protein_records",
    "export_match_results",
    "export_protein_ranking_csv",
    "export_site_results_csv",
    "find_matching_sites_with_scores",
    "flatten_site_results",
    "iter_sequence_windows",
    "load_fasta_records",
    "load_fasta_records_from_folder",
    "match_motif",
    "rank_protein_results",
    "select_best_site_per_position",
    "summarize_match_results",
]
