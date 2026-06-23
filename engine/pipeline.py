import time
from pathlib import Path
from typing import Any

from engine import protein_match
from engine import reference_sources
from engine import substrate_bank
from engine import uniprot_service


REFERENCE_TO_BANK_POSITION = {
    "P4": "P4",
    "P3": "P3",
    "P2": "P2",
    "P1": "P1",
    "P1prime": "P1'",
    "P2prime": "P2'",
    "P3prime": "P3'",
    "P4prime": "P4'",
}


def _require_job_input(job_input: dict[str, Any], field_name: str) -> Any:
    value = job_input.get(field_name)
    if value in (None, ""):
        raise ValueError(f"job_input must include '{field_name}'.")
    return value


def _prepare_reference_for_substrate_bank(reference: dict[str, Any]) -> tuple[list[str], dict[str, list[str]]]:
    positions = [
        REFERENCE_TO_BANK_POSITION[position]
        for position in reference["positions"]
    ]
    positional_preferences = {
        REFERENCE_TO_BANK_POSITION[position]: amino_acids
        for position, amino_acids in reference["positional_preferences"].items()
    }
    return positions, positional_preferences


def _prepare_protein_match_records(protein_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "id": protein_record["protein_id"],
            "description": protein_record["description"],
            "seq": protein_record["sequence"],
            "gene": protein_record["gene"],
            "entry_name": protein_record["entry_name"],
            "organism": protein_record["organism"],
        }
        for protein_record in protein_records
    ]


def run_job(job_input: dict) -> dict:
    """
    Run the scientific pipeline from predefined reference to ranked protein matches.
    """
    protease_gene = _require_job_input(job_input, "protease_gene")
    organism = _require_job_input(job_input, "organism")
    genes = job_input.get("genes")
    substrate_genes = job_input.get("substrate_genes")
    if genes is None:
        genes = substrate_genes if substrate_genes is not None else []
    elif substrate_genes is not None and genes != substrate_genes:
        raise ValueError("job_input fields 'genes' and 'substrate_genes' must match when both are provided.")
    if not isinstance(genes, list):
        raise ValueError("job_input field 'genes' must be a list of gene symbols.")

    reviewed = job_input.get("reviewed", True)
    include_isoforms = job_input.get("include_isoforms", False)
    window_size = job_input.get("window_size", 8)
    top_k = job_input.get("top_k", 3)
    min_top_hits = job_input.get("min_top_hits", 4)
    base_score = job_input.get("base_score", 1.0)
    decay = job_input.get("decay", 0.8)
    position_weights = job_input.get("position_weights")

    started_at = time.perf_counter()
    reference = reference_sources.load_predefined_reference(protease_gene, organism)
    print(f"[timing] reference loading: {time.perf_counter() - started_at:.3f}s")
    positions, positional_preferences = _prepare_reference_for_substrate_bank(reference)

    started_at = time.perf_counter()
    generated_bank, positional_scores = substrate_bank.generate_weighted_substrate_bank(
        positional_preferences=positional_preferences,
        positions=positions,
        top_k=top_k,
        min_top_hits=min_top_hits,
        base_score=base_score,
        decay=decay,
        position_weights=position_weights,
    )
    print(
        "[timing] substrate bank generation: "
        f"{time.perf_counter() - started_at:.3f}s | bank_size={len(generated_bank)}"
    )

    started_at = time.perf_counter()
    protein_records = uniprot_service.fetch_uniprot_proteins(
        organism=organism,
        genes=genes,
        reviewed=reviewed,
        include_isoforms=include_isoforms,
    )
    print(
        "[timing] UniProt retrieval: "
        f"{time.perf_counter() - started_at:.3f}s | protein_count={len(protein_records)}"
    )
    match_records = _prepare_protein_match_records(protein_records)

    match_results = protein_match.analyze_protein_records(
        match_records,
        generated_bank,
        window_size=window_size,
    )

    summary = dict(match_results["summary"])
    summary.update(
        {
            "protease_gene": reference.get("protease_gene", protease_gene),
            "organism": reference.get("organism", organism),
            "queried_genes": genes,
            "retrieved_proteins": len(protein_records),
            "substrate_bank_size": len(generated_bank),
        }
    )

    return {
        "reference_bank": {
            "reference": reference,
            "positions": positions,
            "positional_preferences": positional_preferences,
            "positional_scores": positional_scores,
            "substrate_bank": generated_bank,
        },
        "site_results": match_results["site_results"],
        "protein_ranking": match_results["protein_ranking"],
        "summary": summary,
    }


__all__ = ["run_job"]
