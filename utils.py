import os
from dataclasses import dataclass
from typing import List, Optional, Dict, Any


def create_pmc_url(pmcid: str) -> str:
    """Create PMC article URL from PMCID."""
    return f"https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}"


def ensure_pmid_directory(pmid: str, base_dir: str = "articles_data") -> str:
    """Create and return PMID-specific directory path."""
    pmid_dir = os.path.join(base_dir, pmid)
    os.makedirs(pmid_dir, exist_ok=True)
    return pmid_dir


@dataclass
class ArticleMetadata:
    """Data class to hold article metadata."""

    pmid: str
    pmcid: Optional[str] = None
    title: str = ""
    doi: str = ""
    journal: str = ""
    source: str = ""
    content: str = ""
    html_content: Optional[str] = None  # Store raw HTML content

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "pmid": self.pmid,
            "pmcid": self.pmcid,
            "title": self.title,
            "doi": self.doi,
            "journal": self.journal,
            "source": self.source,
            "content": self.content,
            "html_content": self.html_content,
        }


@dataclass
class Figure:
    url: str
    alt: str
    caption: str
    element: Optional[any] = None
