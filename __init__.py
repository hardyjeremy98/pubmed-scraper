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
from .content import ContentFetcher
from .pdf_downloader import PDFDownloader, PDFFinder
from .image_downloader import ImageDownloader
from .storage import ArticleManager

__all__ = [
    "PubMedClient",
    "ArticleMetadata",
    "Figure",
    "HTTPSession",
    "MetadataFetcher",
    "ContentFetcher",
    "PDFDownloader",
    "PDFFinder",
    "ImageDownloader",
    "ArticleManager",
    "create_pmc_url",
    "ensure_pmid_directory",
]
