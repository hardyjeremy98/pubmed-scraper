from typing import Optional, Dict
from Bio import Entrez
from bs4 import BeautifulSoup
from utils import ArticleMetadata, create_pmc_url
from http_session import HTTPSession


class MetadataFetcher:
    """Handles fetching article metadata from various sources."""

    def __init__(self, email: str, http_session: HTTPSession):
        Entrez.email = email
        self.http_session = http_session
        self._cache: Dict[str, ArticleMetadata] = {}
        self._html_cache: Dict[str, str] = {}  # Cache HTML content by PMC URL

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

    def _fetch_pubmed_record(self, pmid: str) -> Optional[dict]:
        """Fetch the raw PubMed record for a given PMID."""
        try:
            handle = Entrez.efetch(db="pubmed", id=pmid, rettype="xml", retmode="xml")
            record = Entrez.read(handle)
            return record["PubmedArticle"][0]["MedlineCitation"]["Article"]
        except Exception as e:
            print(f"Error fetching PubMed record for PMID {pmid}: {e}")
            return None

    def _fetch_metadata_from_pubmed(self, pmid: str) -> ArticleMetadata:
        """Fetch metadata from PubMed API as fallback."""
        article_data = self._fetch_pubmed_record(pmid)
        if not article_data:
            return ArticleMetadata(pmid=pmid)

        title = article_data.get("ArticleTitle", "")
        journal = article_data.get("Journal", {}).get("Title", "")
        doi = ""

        # Extract DOI if available
        for eloc in article_data.get("ELocationID", []):
            if hasattr(eloc, "attributes") and eloc.attributes.get("EIdType") == "doi":
                doi = str(eloc)
                break

        return ArticleMetadata(pmid=pmid, title=title, doi=doi, journal=journal)

    def clear_cache(self) -> None:
        """Clear both metadata and HTML caches."""
        self._cache.clear()
        self._html_cache.clear()

    def clear_metadata_cache(self) -> None:
        """Clear only the metadata cache."""
        self._cache.clear()

    def clear_html_cache(self) -> None:
        """Clear only the HTML cache."""
        self._html_cache.clear()

    def get_cached_article(self, pmid: str) -> Optional[ArticleMetadata]:
        """Get article from cache if available."""
        return self._cache.get(pmid)

    def get_full_article_content(self, article: ArticleMetadata) -> ArticleMetadata:
        """
        Get complete article data including full text, abstract, and HTML.
        """
        # Try to get HTML content first if we have a PMCID
        if article.pmcid and not article.html_content:
            html_content = self.get_html_content(article)
            # Extract text content from HTML if we don't have content yet
            if html_content and not article.content:
                content = self._extract_text_from_html(html_content)
                if content:
                    article.content = content
                    article.source = "fulltext"

        # If still no content, try to get abstract
        if not article.content:
            abstract = self._fetch_abstract_from_pubmed(article.pmid)
            if abstract:
                article.content = abstract
                article.source = "abstract"

        return article

    def _fetch_html_from_pmc(self, pmc_url: str) -> Optional[str]:
        """Fetch raw HTML content from PMC article with caching."""
        # Check cache first
        if pmc_url in self._html_cache:
            print("Using cached HTML content")
            return self._html_cache[pmc_url]

        try:
            print(f"Fetching HTML from: {pmc_url}")
            response = self.http_session.get(pmc_url)
            response.raise_for_status()

            if response.status_code == 200:
                print("✓ Successfully fetched HTML content")
                # Cache the HTML content
                self._html_cache[pmc_url] = response.text
                return response.text
            else:
                print(f"Failed to fetch HTML: HTTP {response.status_code}")
                return None

        except Exception as e:
            print(f"Error fetching HTML from {pmc_url}: {e}")
            return None

    def _extract_text_from_html(self, html_content: str) -> Optional[str]:
        """Extract clean text content from HTML."""
        try:
            soup = BeautifulSoup(html_content, "lxml")
            text_parts = [
                sec.get_text(strip=True) for sec in soup.find_all(["title", "p"])
            ]
            return "\n".join(text_parts)
        except Exception as e:
            print(f"Error extracting text from HTML: {e}")
            return None

    def _fetch_abstract_from_pubmed(self, pmid: str) -> Optional[str]:
        """Fetch abstract from PubMed."""
        article_data = self._fetch_pubmed_record(pmid)
        if not article_data:
            return None

        abstract_sections = article_data.get("Abstract", {}).get("AbstractText", [])
        if abstract_sections:
            if isinstance(abstract_sections, list):
                return " ".join(str(section) for section in abstract_sections)
            else:
                return str(abstract_sections)

        return None

    def get_html_content(self, article: ArticleMetadata) -> Optional[str]:
        """Get HTML content for an article, using cache if available."""
        if article.html_content:
            return article.html_content

        if article.pmcid:
            pmc_url = create_pmc_url(article.pmcid)
            html_content = self._fetch_html_from_pmc(pmc_url)
            if html_content:
                article.html_content = html_content
                return html_content

        return None
