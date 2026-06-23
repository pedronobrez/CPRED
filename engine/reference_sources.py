import json
from functools import lru_cache
from pathlib import Path
from typing import Any


_ORGANISM_ALIASES = {
    "human": "human",
    "homo sapiens": "human",
    "homo_sapiens": "human",
    "mouse": "mouse",
    "mus musculus": "mouse",
    "mus_musculus": "mouse",
    "rat": "rat",
    "rattus norvegicus": "rat",
    "rattus_norvegicus": "rat",
    "cow": "cow",
    "bos taurus": "cow",
    "bos_taurus": "cow",
}

_ORGANISM_LABELS = {
    "cow": "Cow",
    "human": "Human",
    "mouse": "Mouse",
    "rat": "Rat",
}

_PREDEFINED_ROOT = Path(__file__).resolve().parent.parent / "data" / "predefined_proteases"


def normalize_organism_name(organism: str) -> str:
    """
    Map simple organism inputs to a canonical organism key.
    """
    normalized = organism.strip().lower().replace("-", " ").replace("/", " ")
    normalized = " ".join(normalized.split())
    return _ORGANISM_ALIASES.get(normalized, normalized.replace(" ", "_"))


def organism_label(organism: str) -> str:
    """
    Convert a canonical organism key into a UI-friendly label.
    """
    normalized = normalize_organism_name(organism)
    return _ORGANISM_LABELS.get(normalized, normalized.replace("_", " ").title())


def _load_reference_file(reference_path: Path) -> dict[str, Any]:
    with reference_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _resolve_reference_organism(reference: dict[str, Any], fallback_organism: str) -> str:
    raw_organism = reference.get("organism")
    if isinstance(raw_organism, str) and raw_organism.strip():
        return normalize_organism_name(raw_organism)
    return normalize_organism_name(fallback_organism)


@lru_cache(maxsize=1)
def get_predefined_reference_catalog() -> list[dict[str, Any]]:
    """
    Return all predefined references discovered from JSON files on disk.
    """
    catalog: list[dict[str, Any]] = []

    for reference_path in sorted(_PREDEFINED_ROOT.rglob("*.json")):
        reference = _load_reference_file(reference_path)
        fallback_organism = normalize_organism_name(reference_path.parent.name)
        organism = _resolve_reference_organism(reference, fallback_organism)
        protease_gene = str(reference.get("protease_gene") or reference_path.stem).strip().upper()

        catalog.append(
            {
                "protease_gene": protease_gene,
                "organism": organism,
                "organism_label": organism_label(organism),
                "path": reference_path,
            }
        )

    catalog.sort(key=lambda item: (item["organism"], item["protease_gene"]))
    return catalog


def get_available_organisms() -> list[str]:
    """
    Return the canonical organism keys present in the predefined reference catalog.
    """
    return sorted({entry["organism"] for entry in get_predefined_reference_catalog()})


def get_available_proteases_by_organism(organism: str) -> list[str]:
    """
    Return the available protease genes for one organism.
    """
    normalized_organism = normalize_organism_name(organism)
    return sorted(
        {
            entry["protease_gene"]
            for entry in get_predefined_reference_catalog()
            if entry["organism"] == normalized_organism
        }
    )


def get_predefined_reference_options() -> dict[str, Any]:
    """
    Return UI-friendly organism and protease options derived from the catalog.
    """
    organisms = get_available_organisms()
    return {
        "organisms": [
            {
                "value": organism,
                "label": organism_label(organism),
            }
            for organism in organisms
        ],
        "proteases_by_organism": {
            organism: get_available_proteases_by_organism(organism)
            for organism in organisms
        },
    }


def has_predefined_reference(protease_gene: str, organism: str) -> bool:
    """
    Return True when a predefined reference exists for the given protease and organism.
    """
    normalized_organism = normalize_organism_name(organism)
    normalized_gene = protease_gene.strip().upper()
    return any(
        entry["organism"] == normalized_organism and entry["protease_gene"] == normalized_gene
        for entry in get_predefined_reference_catalog()
    )


def load_predefined_reference(protease_gene: str, organism: str) -> dict[str, Any]:
    """
    Load a predefined protease reference JSON by gene and organism.
    """
    normalized_organism = normalize_organism_name(organism)
    normalized_gene = protease_gene.strip().upper()

    for entry in get_predefined_reference_catalog():
        if entry["organism"] == normalized_organism and entry["protease_gene"] == normalized_gene:
            reference = dict(_load_reference_file(entry["path"]))
            reference["protease_gene"] = normalized_gene
            reference["organism"] = normalized_organism
            return reference

    raise FileNotFoundError(
        f"Predefined reference not found for protease_gene='{normalized_gene}' "
        f"and organism='{normalized_organism}'."
    )


__all__ = [
    "get_available_organisms",
    "get_available_proteases_by_organism",
    "get_predefined_reference_catalog",
    "get_predefined_reference_options",
    "has_predefined_reference",
    "load_predefined_reference",
    "normalize_organism_name",
    "organism_label",
]
