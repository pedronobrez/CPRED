import re
from io import StringIO
from pathlib import Path
from typing import Any

import requests
from Bio import SeqIO


UNIPROT_STREAM_URL = "https://rest.uniprot.org/uniprotkb/stream"
_GENE_BATCH_SIZE = 50  # max genes per UniProt query to stay under URL length limits

_ORGANISM_ALIASES = {
    "human": "Homo sapiens",
    "homo sapiens": "Homo sapiens",
    "homo_sapiens": "Homo sapiens",
    "mouse": "Mus musculus",
    "mus musculus": "Mus musculus",
    "mus_musculus": "Mus musculus",
    "rat": "Rattus norvegicus",
    "rattus norvegicus": "Rattus norvegicus",
    "rattus_norvegicus": "Rattus norvegicus",
    "cow": "Bos taurus",
    "bos taurus": "Bos taurus",
    "bos_taurus": "Bos taurus",
}


def _clean_genes(genes: list[str]) -> list[str]:
    return [gene.strip() for gene in genes if gene.strip()]


def _normalize_gene_name(gene: str) -> str:
    return gene.strip().casefold()


def _filter_records_by_requested_genes(
    records: list[dict[str, Any]],
    genes: list[str],
) -> list[dict[str, Any]]:
    cleaned_genes = _clean_genes(genes)
    if not cleaned_genes:
        return records

    requested_genes = {_normalize_gene_name(gene) for gene in cleaned_genes}
    return [
        record
        for record in records
        if _normalize_gene_name(str(record.get("gene", ""))) in requested_genes
    ]


def normalize_organism_name(organism: str) -> str:
    """
    Normalize common organism inputs to a UniProt-friendly organism name.
    """
    normalized = organism.strip().lower().replace("-", " ").replace("/", " ")
    normalized = " ".join(normalized.split())
    return _ORGANISM_ALIASES.get(normalized, organism.strip())


def build_uniprot_query(
    organism: str,
    genes: list[str],
    reviewed: bool = True,
) -> str:
    """
    Build a UniProtKB query for one organism and an optional set of genes.
    """
    normalized_organism = normalize_organism_name(organism)
    organism_filter = f'organism_name:"{normalized_organism}"'
    reviewed_filter = "reviewed:true" if reviewed else None

    cleaned_genes = _clean_genes(genes)
    gene_filter = None
    if cleaned_genes:
        gene_terms = [f"gene:{gene}" for gene in cleaned_genes]
        gene_filter = "(" + " OR ".join(gene_terms) + ")"

    query_parts = [organism_filter]
    if reviewed_filter:
        query_parts.append(reviewed_filter)
    if gene_filter:
        query_parts.append(gene_filter)

    return " AND ".join(query_parts)


def fetch_uniprot_fasta(
    organism: str,
    genes: list[str],
    reviewed: bool = True,
    include_isoforms: bool = False,
    timeout: int = 120,
) -> str:
    """
    Fetch UniProtKB FASTA text for the requested organism and genes.
    """
    query = build_uniprot_query(organism, genes, reviewed=reviewed)
    print(f"[uniprot] final query: {query}")
    params = {
        "query": query,
        "format": "fasta",
        "compressed": "false",
        "includeIsoform": "true" if include_isoforms else "false",
    }

    response = requests.get(UNIPROT_STREAM_URL, params=params, timeout=timeout)
    response.raise_for_status()
    return response.text


def sanitize_filename(name: str) -> str:
    """
    Remove characters that are invalid in Windows filenames.
    """
    return re.sub(r'[<>:"/\\|?*]', "_", name)


def extract_uniprot_accession(record_id: str) -> str:
    """
    Extract the accession from a typical UniProt FASTA record id.
    """
    parts = record_id.split("|")
    if len(parts) >= 3:
        return parts[1]
    return record_id


def extract_entry_name(record_id: str) -> str:
    """
    Extract the UniProt entry name from a typical UniProt FASTA record id.
    """
    parts = record_id.split("|")
    if len(parts) >= 3:
        return parts[2]
    return record_id


def extract_gene_name(description: str) -> str:
    """
    Extract the gene name from the GN= field in a UniProt FASTA header.
    """
    match = re.search(r"\bGN=([^\s]+)", description)
    if match:
        return match.group(1)
    return "UNKNOWN"


def extract_organism_from_description(description: str) -> str:
    """
    Extract the organism from the OS= field in a UniProt FASTA header.
    """
    match = re.search(r"\bOS=(.+?)\sOX=", description)
    if match:
        return match.group(1).strip()

    match = re.search(r"\bOS=(.+?)(?:\sGN=|\sPE=|\sSV=|$)", description)
    if match:
        return match.group(1).strip()

    return ""


def extract_protein_description(record_id: str, description: str) -> str:
    """
    Extract the protein description text from a UniProt FASTA header.
    """
    prefix = f"{record_id} "
    normalized_description = description[len(prefix):] if description.startswith(prefix) else description
    return normalized_description.split(" OS=", 1)[0].strip()


def parse_fasta_records(fasta_text: str) -> list[Any]:
    """
    Parse FASTA text into Biopython SeqRecord objects.
    """
    return list(SeqIO.parse(StringIO(fasta_text), "fasta"))


def normalize_uniprot_record(record: Any) -> dict[str, Any]:
    """
    Convert a UniProt FASTA record into a normalized protein dictionary.
    """
    protein_id = extract_uniprot_accession(record.id)
    return {
        "gene": extract_gene_name(record.description),
        "protein_id": protein_id,
        "entry_name": extract_entry_name(record.id),
        "description": extract_protein_description(record.id, record.description),
        "sequence": str(record.seq),
        "organism": extract_organism_from_description(record.description),
    }


def fetch_uniprot_proteins(
    organism: str,
    genes: list[str],
    reviewed: bool = True,
    include_isoforms: bool = False,
    timeout: int = 120,
) -> list[dict[str, Any]]:
    """
    Fetch UniProt proteins and return normalized in-memory records.
    """
    return fetch_proteins_by_genes(
        organism=organism,
        genes=genes,
        reviewed=reviewed,
        include_isoforms=include_isoforms,
        timeout=timeout,
    )


def fetch_proteins_by_genes(
    organism: str,
    genes: list[str],
    reviewed: bool = True,
    include_isoforms: bool = False,
    timeout: int = 120,
) -> list[dict[str, Any]]:
    """
    Fetch UniProt proteins and return normalized in-memory records.
    Large gene lists are split into batches of _GENE_BATCH_SIZE to avoid
    UniProt 400 errors caused by excessively long query URLs.
    """
    cleaned = _clean_genes(genes)
    batches = [
        cleaned[i: i + _GENE_BATCH_SIZE]
        for i in range(0, max(len(cleaned), 1), _GENE_BATCH_SIZE)
    ]

    all_normalized: list[dict[str, Any]] = []
    for batch_idx, batch in enumerate(batches, start=1):
        print(f"[uniprot] batch {batch_idx}/{len(batches)} ({len(batch)} genes)")
        fasta_text = fetch_uniprot_fasta(
            organism=organism,
            genes=batch,
            reviewed=reviewed,
            include_isoforms=include_isoforms,
            timeout=timeout,
        )
        records = parse_fasta_records(fasta_text)
        all_normalized.extend(normalize_uniprot_record(r) for r in records)

    filtered_records = _filter_records_by_requested_genes(all_normalized, cleaned)
    returned_gene_names = sorted({record["gene"] for record in filtered_records})
    print(f"[uniprot] returned records: {len(filtered_records)}")
    print(f"[uniprot] returned genes: {returned_gene_names}")
    return filtered_records


def download_uniprot_fasta(
    output_fasta_path: str | Path,
    organism: str,
    genes: list[str],
    reviewed: bool = True,
    include_isoforms: bool = False,
    timeout: int = 120,
) -> Path:
    """
    Fetch UniProt FASTA content and save it to disk.
    """
    fasta_text = fetch_uniprot_fasta(
        organism=organism,
        genes=genes,
        reviewed=reviewed,
        include_isoforms=include_isoforms,
        timeout=timeout,
    )

    output_fasta_path = Path(output_fasta_path)
    output_fasta_path.parent.mkdir(parents=True, exist_ok=True)
    output_fasta_path.write_text(fasta_text, encoding="utf-8")
    return output_fasta_path


def split_fasta_to_individual_files(input_fasta: str | Path, output_dir: str | Path) -> list[Path]:
    """
    Split a multi-record FASTA into one FASTA file per protein.
    """
    input_fasta = Path(input_fasta)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    written_paths: list[Path] = []
    for record in SeqIO.parse(str(input_fasta), "fasta"):
        accession = extract_uniprot_accession(record.id)
        gene = extract_gene_name(record.description)
        filename = sanitize_filename(f"{gene}_{accession}.fasta")
        output_path = output_dir / filename

        with output_path.open("w", encoding="utf-8") as handle:
            SeqIO.write(record, handle, "fasta")

        written_paths.append(output_path)

    return written_paths


__all__ = [
    "UNIPROT_STREAM_URL",
    "build_uniprot_query",
    "download_uniprot_fasta",
    "extract_entry_name",
    "extract_gene_name",
    "extract_organism_from_description",
    "extract_protein_description",
    "extract_uniprot_accession",
    "fetch_uniprot_fasta",
    "fetch_proteins_by_genes",
    "fetch_uniprot_proteins",
    "normalize_organism_name",
    "normalize_uniprot_record",
    "parse_fasta_records",
    "sanitize_filename",
    "split_fasta_to_individual_files",
]
