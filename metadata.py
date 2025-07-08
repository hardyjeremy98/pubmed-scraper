from typing import Optional, Dict
from Bio import Entrez
from utils import ArticleMetadata
from http_session import HTTPSession


class MetadataFetcher:
    """Handles fetching article metadata from various sources."""

    def __init__(self, email: str, http_session: HTTPSession):
        Entrez.email = email
        self.http_session = http_session
        self._cache: Dict[str, ArticleMetadata] = {}

    def get_article_metadata(self, pmid: str) -> ArticleMetadata:
        """
        Retrieve article metadata (PMID, PMCID, title, DOI, journal).
        Uses caching to avoid repeated API calls.
        """
        # Check cache first
        if pmid in self._cache:
            return self._cache[pmid]

        # Try Europe PMC first
        metadata = self._fetch_metadata_from_epmc(pmid)

        # Fallback to PubMed if EPMC fails
        if metadata is None:
            metadata = self._fetch_metadata_from_pubmed(pmid)

        if not metadata.pmcid:
            fallback_pmcid = self._fetch_pmcid_from_ncbi(pmid)
            if fallback_pmcid:
                metadata.pmcid = fallback_pmcid

        # Cache the result
        self._cache[pmid] = metadata
        return metadata

    def _fetch_metadata_from_epmc(self, pmid: str) -> Optional[ArticleMetadata]:
        """Fetch metadata from Europe PMC API."""
        epmc_url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=EXT_ID:{pmid}%20AND%20SRC:MED&format=json"

        try:
            r = self.http_session.get(epmc_url)
            if r.status_code == 200 and r.json().get("hitCount", 0) > 0:
                doc = r.json()["resultList"]["result"][0]
                return ArticleMetadata(
                    pmid=pmid,
                    pmcid=doc.get("pmcid"),
                    title=doc.get("title", ""),
                    doi=doc.get("doi", ""),
                    journal=doc.get("journalTitle", ""),
                )
        except Exception as e:
            print(f"Error fetching from EPMC for PMID {pmid}: {e}")

        return None

    def _fetch_pmcid_from_ncbi(self, pmid: str) -> Optional[str]:
        """Try to fetch the PMCID using NCBI's eLink API as a backup."""
        try:
            handle = Entrez.elink(
                dbfrom="pubmed", db="pmc", id=pmid, linkname="pubmed_pmc"
            )
            records = Entrez.read(handle)
            linksets = records[0].get("LinkSetDb", [])
            for linkset in linksets:
                for link in linkset.get("Link", []):
                    pmcid_num = link.get("Id")
                    pmcid = f"PMC{pmcid_num}"
                    if pmcid:
                        return pmcid
        except Exception as e:
            print(f"NCBI eLink error for PMID {pmid}: {e}")
        return None

    def _fetch_metadata_from_pubmed(self, pmid: str) -> ArticleMetadata:
        """Fetch metadata from PubMed API as fallback."""
        try:
            handle = Entrez.efetch(db="pubmed", id=pmid, rettype="xml", retmode="xml")
            record = Entrez.read(handle)
            article = record["PubmedArticle"][0]["MedlineCitation"]["Article"]

            title = article.get("ArticleTitle", "")
            journal = article.get("Journal", {}).get("Title", "")
            doi = ""

            # Extract DOI if available
            for eloc in article.get("ELocationID", []):
                if (
                    hasattr(eloc, "attributes")
                    and eloc.attributes.get("EIdType") == "doi"
                ):
                    doi = str(eloc)
                    break

            return ArticleMetadata(pmid=pmid, title=title, doi=doi, journal=journal)

        except Exception as e:
            print(f"Error fetching from PubMed for PMID {pmid}: {e}")
            return ArticleMetadata(pmid=pmid)

    def clear_cache(self) -> None:
        """Clear the metadata cache."""
        self._cache.clear()

    def get_cached_article(self, pmid: str) -> Optional[ArticleMetadata]:
        """Get article from cache if available."""
        return self._cache.get(pmid)
