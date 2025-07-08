"""
PubMed/PMC Literature Mining Toolkit

A comprehensive toolkit for mining literature from PubMed and PMC,
including metadata extraction, content fetching, PDF downloading,
and figure extraction.
"""

from .pubmed_scraper import PubMedClient
from .utils import ArticleMetadata, Figure, create_pmc_url, ensure_pmid_directory
from .http_session import HTTPSession
from .metadata import MetadataFetcher
from .pdf_downloader import PDFDownloader, PDFFinder
from .image_downloader import ImageDownloader
from .storage import ArticleManager
from .figure_scanner import (
    FigureKeywordScanner,
    ScanResult,
    scan_article_figures_for_keywords,
)

__all__ = [
    "PubMedClient",
    "ArticleMetadata",
    "Figure",
    "HTTPSession",
    "MetadataFetcher",
    "PDFDownloader",
    "PDFFinder",
    "ImageDownloader",
    "ArticleManager",
    "FigureKeywordScanner",
    "ScanResult",
    "scan_article_figures_for_keywords",
    "create_pmc_url",
    "ensure_pmid_directory",
]
