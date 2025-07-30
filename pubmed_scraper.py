import os
from typing import List, Optional
from Bio import Entrez

from utils import create_pmc_url, ensure_pmid_directory
from http_session import HTTPSession
from metadata import MetadataFetcher
from pdf_handler import PDFDownloader
from figure_handler import ImageDownloader
from storage import ArticleManager
from utils import ArticleMetadata, Figure


class PubMedClient:
    """Main client class that orchestrates all components."""

    def __init__(self, email: str, base_dir: str = "articles_data"):
        self.base_dir = base_dir
        self.http_session = HTTPSession()
        self.metadata_fetcher = MetadataFetcher(email, self.http_session)
        self.pdf_downloader = PDFDownloader(self.http_session, base_dir)
        self.image_downloader = ImageDownloader(self.http_session, base_dir)
        self.article_manager = ArticleManager()

    def search_pubmed(self, query: str, retmax: int = 100) -> List[str]:
        """Search PubMed and return list of PubMed IDs."""
        handle = Entrez.esearch(db="pubmed", term=query, retmax=retmax)
        record = Entrez.read(handle)
        return record["IdList"]

    def fetch_details(self, id_list: List[str]) -> str:
        """Fetch article details given a list of PubMed IDs."""
        ids = ",".join(id_list)
        handle = Entrez.efetch(db="pubmed", id=ids, rettype="medline", retmode="text")
        return handle.read()

    def fetch_summary(self, id_list: List[str]):
        """Fetch summary metadata (title, journal, date)."""
        ids = ",".join(id_list)
        handle = Entrez.esummary(db="pubmed", id=ids)
        return Entrez.read(handle)

    def get_article_metadata(self, pmid: str) -> ArticleMetadata:
        """Get article metadata."""
        return self.metadata_fetcher.get_article_metadata(pmid)

    def get_full_article(self, pmid: str) -> ArticleMetadata:
        """Get complete article data including full text or abstract."""
        # Get basic metadata first
        article = self.metadata_fetcher.get_article_metadata(pmid)
        # Get content
        return self.metadata_fetcher.get_full_article_content(article)

    def download_pdf_from_pmcid(self, pmcid: str, pmid: str, **kwargs) -> None:
        """Download PDF for a given PMCID."""
        self.pdf_downloader.download_pdf_from_pmcid(pmcid, pmid, **kwargs)

    def download_image(self, img_url: str, filename: str, pmid: str) -> Optional[str]:
        """Download image from URL."""
        return self.image_downloader.download_image(img_url, filename, pmid)

    def get_article_figures(self, pmid: str) -> List[Figure]:
        """Get figures for an article by PMID using cached HTML when possible."""
        article = self.get_article_metadata(pmid)
        if article.pmcid:
            # Get HTML content through ContentFetcher to use caching
            html_content = self.metadata_fetcher.get_html_content(article)
            if html_content:
                pmc_url = create_pmc_url(article.pmcid)
                return self.image_downloader.get_pmc_figures_from_html(
                    html_content, pmc_url
                )
            else:
                print(f"Could not fetch HTML for PMCID {article.pmcid}")
                return []
        else:
            print(f"No PMCID available for PMID {pmid}, cannot extract figures")
            return []

    def get_pmc_figures(self, pmc_url: str) -> List[Figure]:
        """Extract figures from PMC article (legacy method)."""
        return self.image_downloader.get_pmc_figures(pmc_url)

    def get_article_html(self, pmid: str) -> Optional[str]:
        """Extract raw HTML content from PMC article if available."""
        article = self.get_article_metadata(pmid)
        return self.metadata_fetcher.get_html_content(article)

    def save_article_html(
        self, html_content: str, pmid: str, base_dir: str = None
    ) -> Optional[str]:
        """Save article HTML to file in the PMID-specific directory."""
        if base_dir is None:
            base_dir = self.base_dir

        if not html_content:
            print(f"No HTML content to save for PMID {pmid}")
            return None

        # Create PMID-specific directory
        pmid_dir = ensure_pmid_directory(pmid, base_dir)

        # Save HTML file
        filename = f"{pmid}_article.html"
        filepath = os.path.join(pmid_dir, filename)

        try:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(html_content)
            print(f"Saved HTML content to: {filepath}")
            return filepath
        except Exception as e:
            print(f"Error saving HTML for PMID {pmid}: {e}")
            return None

    def save_article_to_json(
        self, article: ArticleMetadata, base_dir: str = None
    ) -> str:
        """Save article to JSON."""
        if base_dir is None:
            base_dir = self.base_dir
        return self.article_manager.save_article_to_json(article, base_dir)

    def save_figures_to_json(
        self, figures: List[Figure], pmid: str, base_dir: str = None
    ) -> str:
        """Save figures metadata to JSON."""
        if base_dir is None:
            base_dir = self.base_dir
        return self.article_manager.save_figures_to_json(figures, pmid, base_dir)

    def load_pmids_from_json(self, json_file: str = "unique_pmids.json") -> List[str]:
        """Load PMIDs from JSON file."""
        return self.article_manager.load_pmids_from_json(json_file)

    def get_cached_article(self, pmid: str) -> Optional[ArticleMetadata]:
        """Get article from cache if available."""
        return self.metadata_fetcher.get_cached_article(pmid)

    def clear_cache(self) -> None:
        """Clear the article cache."""
        self.metadata_fetcher.clear_cache()

    def _candidate_pdf_urls(self, pmcid: str) -> list[str]:
        """Generate candidate PDF URLs for a given PMCID."""
        pmcid = pmcid if pmcid.startswith("PMC") else f"PMC{pmcid}"
        base = create_pmc_url(pmcid)
        return [
            f"{base}/pdf?download=1",  # compiled PDF
            f"{base}/pdf/{pmcid}.pdf",  # common fallback
            f"{base}/pdf",  # HTML wrapper (parse if reached)
        ]
