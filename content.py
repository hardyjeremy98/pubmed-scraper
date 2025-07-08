from typing import Optional, Dict
from Bio import Entrez
from bs4 import BeautifulSoup
from utils import ArticleMetadata, create_pmc_url
from http_session import HTTPSession


class ContentFetcher:
    """Handles fetching full text content, abstracts, and HTML."""

    def __init__(self, email: str, http_session: HTTPSession):
        Entrez.email = email
        self.http_session = http_session
        self._html_cache: Dict[str, str] = {}  # Cache HTML content by PMC URL

    def get_full_article_content(self, article: ArticleMetadata) -> ArticleMetadata:
        """
        Get complete article data including full text, abstract, and HTML.
        """
        # If we have a PMCID, fetch HTML first and derive content from it
        if article.pmcid and not article.html_content:
            pmc_url = create_pmc_url(article.pmcid)
            html_content = self._fetch_html_from_pmc(pmc_url)
            if html_content:
                article.html_content = html_content
                # Extract text content from HTML if we don't have content yet
                if not article.content:
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

    def _fetch_fulltext_from_pmc(self, pmcid: str) -> Optional[str]:
        """Retrieve fulltext from PMC if available (deprecated - use get_full_article_content)."""
        if not pmcid:
            return None

        print(f"Fetching full text for PMCID: {pmcid}")
        pmc_url = create_pmc_url(pmcid)
        html_content = self._fetch_html_from_pmc(pmc_url)

        if html_content:
            return self._extract_text_from_html(html_content)

        return None

    def _fetch_abstract_from_pubmed(self, pmid: str) -> Optional[str]:
        """Fetch abstract from PubMed."""
        try:
            handle = Entrez.efetch(db="pubmed", id=pmid, rettype="xml", retmode="xml")
            record = Entrez.read(handle)
            article = record["PubmedArticle"][0]["MedlineCitation"]["Article"]

            abstract_sections = article.get("Abstract", {}).get("AbstractText", [])
            if abstract_sections:
                if isinstance(abstract_sections, list):
                    return " ".join(str(section) for section in abstract_sections)
                else:
                    return str(abstract_sections)

        except Exception as e:
            print(f"Error fetching abstract for PMID {pmid}: {e}")

        return None
