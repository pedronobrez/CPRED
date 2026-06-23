import csv
import itertools
from pathlib import Path
from typing import Any


DEFAULT_POSITIONS = ["P4", "P3", "P2", "P1", "P1'", "P2'", "P3'", "P4'"]


def make_position_weights() -> dict[str, float]:
    """
    Return the default relative weights around the cleavage site.
    """
    return {
        "P1": 1.0,
        "P1'": 1.0,
        "P2": 0.8,
        "P2'": 0.8,
        "P3": 0.6,
        "P3'": 0.6,
        "P4": 0.4,
        "P4'": 0.4,
    }


def make_weighted_positional_scores(
    positional_preferences: dict[str, list[str]],
    position_weights: dict[str, float],
    base_score: float = 1.0,
    decay: float = 0.8,
) -> dict[str, dict[str, float]]:
    """
    Build per-position amino-acid scores using rank decay and positional weights.
    """
    positional_scores: dict[str, dict[str, float]] = {}
    for position, amino_acids in positional_preferences.items():
        position_weight = position_weights.get(position, 1.0)
        scores: dict[str, float] = {}
        for rank, amino_acid in enumerate(amino_acids, start=1):
            rank_score = base_score * (decay ** (rank - 1))
            scores[amino_acid] = rank_score * position_weight
        positional_scores[position] = scores
    return positional_scores


def score_weighted_motif(
    motif: str,
    positions: list[str],
    positional_preferences: dict[str, list[str]],
    positional_scores: dict[str, dict[str, float]],
) -> tuple[float, list[dict[str, Any]]]:
    """
    Score one motif against the positional preferences, treating 'X' as zero score.
    """
    assert len(motif) == len(positions)
    total = 0.0
    details: list[dict[str, Any]] = []

    for index, amino_acid in enumerate(motif):
        position = positions[index]
        preference_list = positional_preferences[position]
        score_lookup = positional_scores[position]

        if amino_acid == "X":
            rank = None
            score = 0.0
        elif amino_acid in preference_list:
            rank = preference_list.index(amino_acid) + 1
            score = score_lookup.get(amino_acid, 0.0)
        else:
            rank = None
            score = 0.0

        total += score
        details.append(
            {
                "pos": position,
                "aa": amino_acid,
                "rank": rank,
                "score": score,
            }
        )

    return total, details


def generate_weighted_substrate_bank(
    positional_preferences: dict[str, list[str]],
    positions: list[str],
    top_k: int = 3,
    min_top_hits: int = 4,
    base_score: float = 1.0,
    decay: float = 0.8,
    position_weights: dict[str, float] | None = None,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, float]]]:
    """
    Generate the substrate bank and weighted positional score table.
    """
    if position_weights is None:
        position_weights = make_position_weights()

    positional_scores = make_weighted_positional_scores(
        positional_preferences,
        position_weights,
        base_score=base_score,
        decay=decay,
    )

    top_sets = {
        position: set(positional_preferences[position][:top_k])
        for position in positions
    }

    substrate_bank: dict[str, dict[str, Any]] = {}
    position_count = len(positions)

    for mask in itertools.product([0, 1], repeat=position_count):
        if sum(mask) < min_top_hits:
            continue

        choices_by_position: list[list[str]] = []
        for index, position in enumerate(positions):
            if mask[index] == 1:
                choices_by_position.append(list(top_sets[position]))
            else:
                choices_by_position.append(["X"])

        for combo in itertools.product(*choices_by_position):
            motif = "".join(combo)

            top_hits = sum(
                1
                for index, amino_acid in enumerate(motif)
                if amino_acid in top_sets[positions[index]]
            )

            motif_score, details = score_weighted_motif(
                motif,
                positions,
                positional_preferences,
                positional_scores,
            )

            substrate_bank[motif] = {
                "top_hits": top_hits,
                "mask_top": [bool(flag) for flag in mask],
                "motif_score": motif_score,
                "details": details,
            }

    return substrate_bank, positional_scores


def substrate_bank_to_rows(
    substrate_bank: dict[str, dict[str, Any]],
    positions: list[str],
) -> list[dict[str, Any]]:
    """
    Flatten the substrate bank into row dictionaries for export or inspection.
    """
    rows: list[dict[str, Any]] = []
    for motif, info in substrate_bank.items():
        row: dict[str, Any] = {
            "motif": motif,
            "top_hits": info["top_hits"],
            "motif_score": info["motif_score"],
        }
        for detail in info["details"]:
            position = detail["pos"]
            row[f"{position}_aa"] = detail["aa"]
            row[f"{position}_rank"] = detail["rank"] if detail["rank"] is not None else ""
            row[f"{position}_score"] = detail["score"]
        rows.append(row)

    return rows


def substrate_bank_to_dataframe(
    substrate_bank: dict[str, dict[str, Any]],
    positions: list[str],
):
    """
    Convert the substrate bank into a pandas DataFrame using export-oriented columns.
    """
    import pandas as pd

    columns = ["motif", "top_hits", "motif_score"]
    for position in positions:
        columns.extend([f"{position}_aa", f"{position}_rank", f"{position}_score"])

    return pd.DataFrame(substrate_bank_to_rows(substrate_bank, positions), columns=columns)


def export_substrate_bank_csv(
    substrate_bank: dict[str, dict[str, Any]],
    positions: list[str],
    output_path: str | Path,
) -> None:
    """
    Write the substrate bank rows to a CSV file with the original export schema.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    header = ["motif", "top_hits", "motif_score"]
    for position in positions:
        header.extend([f"{position}_aa", f"{position}_rank", f"{position}_score"])

    with output_path.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter=",")
        writer.writerow(header)

        for motif, info in substrate_bank.items():
            row = [motif, info["top_hits"], info["motif_score"]]
            for detail in info["details"]:
                row.append(detail["aa"])
                row.append(detail["rank"] if detail["rank"] is not None else "")
                row.append(detail["score"])
            writer.writerow(row)


# Backward-compatible aliases for the original script naming.
make_positional_scores_weighted = make_weighted_positional_scores
score_motif_weighted = score_weighted_motif
gerar_banco_substratos_weighted = generate_weighted_substrate_bank
salvar_banco_substratos_csv = export_substrate_bank_csv


__all__ = [
    "DEFAULT_POSITIONS",
    "export_substrate_bank_csv",
    "generate_weighted_substrate_bank",
    "gerar_banco_substratos_weighted",
    "make_position_weights",
    "make_positional_scores_weighted",
    "make_weighted_positional_scores",
    "salvar_banco_substratos_csv",
    "score_motif_weighted",
    "score_weighted_motif",
    "substrate_bank_to_dataframe",
    "substrate_bank_to_rows",
]
