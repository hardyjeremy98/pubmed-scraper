"""
PubMed/PMC Literature Mining Toolkit

A comprehensive toolkit for mining literature from PubMed and PMC,
including metadata extraction, content fetching, PDF downloading,
and figure extraction.
"""

from .article_processing.pubmed_scraper import PubMedClient
from .utils.utils import ArticleMetadata, Figure, create_pmc_url, ensure_pmid_directory
from .article_processing.http_session import HTTPSession
from .article_processing.article_fetcher import DataFetcher
from .utils.pdf_handler import PDFDownloader, PDFFinder, PageExtractor
from .article_processing.figure_handler import ImageDownloader
from .utils.storage import ArticleManager
from .article_processing.figure_scanner import (
    FigureKeywordScanner,
    ScanResult,
    scan_article_figures_for_keywords,
)

__all__ = [
    "PubMedClient",
    "ArticleMetadata",
    "Figure",
    "HTTPSession",
    "DataFetcher",
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
