import requests
import re
from difflib import SequenceMatcher
from typing import Optional, Dict

UNIPROT_STREAM = "https://rest.uniprot.org/uniprotkb/stream"
UNIPROT_SEARCH = "https://rest.uniprot.org/uniprotkb/search"


class ProteinNotFound(Exception):
    pass


def _best_hit(
    name_query: str, organism: Optional[str], reviewed: bool, size: int = 10
) -> Dict:
    """
    Query UniProt search API and return the best-matching hit (JSON record).
    """
    # Build query
    parts = []
    # exact phrase match on recommended/alternative names
    parts.append(f'protein_name:"{name_query}"')
    if organism:
        # allow common name or scientific name; UniProt matches both in organism_name
        parts.append(f'organism_name:"{organism}"')
    if reviewed:
        parts.append("reviewed:true")
    # avoid fragments if possible
    parts.append("fragment:false")

    query = " AND ".join(parts)

    params = {
        "query": query,
        "format": "json",
        "size": str(size),
        "fields": ",".join(
            [
                "accession",
                "id",
                "protein_name",
                "gene_primary",
                "organism_name",
                "length",
            ]
        ),
    }

    r = requests.get(UNIPROT_SEARCH, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    results = data.get("results", [])

    if not results:
        # fallback: try gene symbol match if name looked like a gene (e.g., TP53)
        params["query"] = query + f' OR gene_exact:"{name_query}"'
        r = requests.get(UNIPROT_SEARCH, params=params, timeout=30)
        r.raise_for_status()
        results = r.json().get("results", [])

    if not results:
        raise ProteinNotFound(
            f"No UniProt entries found for '{name_query}' (organism={organism}, reviewed={reviewed})."
        )

    # Pick best by fuzzy similarity to protein_name (fall back to id)
    def rec_name(rec):
        # 'protein_name' is a flat field when requested in fields=
        return rec.get("proteinName", rec.get("uniProtkbId", ""))

    scored = sorted(
        results,
        key=lambda rec: SequenceMatcher(
            None, name_query.lower(), (rec_name(rec) or "").lower()
        ).ratio(),
        reverse=True,
    )
    return scored[0]


def _fetch_fasta_by_accession(accession: str) -> str:
    """
    Download FASTA for a single accession using the stream endpoint.
    """
    # Use a query that targets just this accession, format=fasta
    params = {"format": "fasta", "query": f"accession:{accession}"}
    r = requests.get(UNIPROT_STREAM, params=params, timeout=30)
    r.raise_for_status()
    fasta = r.text.strip()
    if not fasta.startswith(">"):
        raise ProteinNotFound(f"FASTA not found for accession {accession}")
    return fasta


def _parse_fasta_sequence(fasta_text: str) -> str:
    """
    Extract the raw sequence (no line breaks) from a FASTA string.
    """
    lines = fasta_text.splitlines()
    seq = "".join(line.strip() for line in lines if not line.startswith(">"))
    # sanity check
    if not re.fullmatch(r"[A-Z]+", seq):
        raise ValueError("Parsed sequence contained unexpected characters.")
    return seq


def get_protein_sequence(
    name_or_accession: str, organism: Optional[str] = None, reviewed: bool = True
) -> Dict[str, str]:
    """
    Resolve a protein name OR a UniProt accession to a canonical sequence.

    Returns:
        {
          "accession": "P05067",
          "uniprot_id": "A4_HUMAN",
          "organism": "Homo sapiens",
          "length": "770",
          "fasta": ">sp|P05067|A4_HUMAN ...\nSEQUENCE...\n",
          "sequence": "SEQUENCEWITHOUTLINEBREAKS"
        }
    """
    # If the user passes a UniProt accession directly (looks like Q9XXXX or P####)
    if re.fullmatch(r"[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9]{5}", name_or_accession):
        # That regex covers Swiss-Prot/TrEMBL style accessions
        acc = name_or_accession
        # Minimal metadata fetch via search to fill fields
        hit = _best_hit(acc, organism=None, reviewed=False, size=1)
    else:
        hit = _best_hit(
            name_or_accession, organism=organism, reviewed=reviewed, size=10
        )

    accession = hit["primaryAccession"]
    fasta = _fetch_fasta_by_accession(accession)
    seq = _parse_fasta_sequence(fasta)

    return {
        "accession": accession,
        "uniprot_id": hit.get("uniProtkbId", ""),
        "organism": hit.get("organism", {}).get(
            "scientificName", hit.get("organism", {}).get("commonName", "")
        )
        or hit.get("organismName", ""),
        "length": str(hit.get("sequence", {}).get("length", hit.get("length", ""))),
        "fasta": fasta,
        "sequence": seq,
    }


# ---------- Examples ----------
if __name__ == "__main__":
    # Example 1: Human TP53 (by common gene symbol)
    print(get_protein_sequence("TP53", organism="Homo sapiens"))

    # Example 2: Human “Amyloid beta A4 protein” (APP)
    print(get_protein_sequence("Amyloid beta A4 protein", organism="Homo sapiens"))

    # Example 3: If you already know the accession:
    print(get_protein_sequence("P05067"))  # APP_HUMAN

    print(get_protein_sequence("Ure2"))
