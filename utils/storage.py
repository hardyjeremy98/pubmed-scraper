import json
import os
from typing import List
from utils.utils import ArticleMetadata, Figure, ensure_pmid_directory


class ArticleManager:
    """Manages article data storage."""

    def save_article_to_json(
        self, article: ArticleMetadata, base_dir: str = "data/articles_data"
    ) -> str:
        """Save article metadata to JSON file."""
        # Create PMID directory if it doesn't exist
        pmid_dir = ensure_pmid_directory(article.pmid, base_dir)
        filename = f"{article.pmid}_metadata.json"
        filepath = os.path.join(pmid_dir, filename)

        # Save article metadata to JSON
        with open(filepath, "w") as f:
            json.dump(article.to_dict(), f, indent=2)

    def save_figures_to_json(
        self, figures: List[Figure], pmid: str, base_dir: str = "data/articles_data"
    ) -> str:
        """Save figures data to JSON file."""
        # Create PMID directory if it doesn't exist
        pmid_dir = ensure_pmid_directory(pmid, base_dir)
        filename = f"{pmid}_figures.json"
        filepath = os.path.join(pmid_dir, filename)

        # Convert figures to list of dictionaries
        figures_data = [figure.to_dict() for figure in figures]

        # Save figures data to JSON
        with open(filepath, "w") as f:
            json.dump(figures_data, f, indent=2)

        print(f"Saved {len(figures_data)} figures with captions to {filepath}")
        return filepath

    @staticmethod
    def load_pmids_from_json(json_file: str = "data/unique_pmids.json") -> List[str]:
        """Load list of PMIDs from a JSON file."""
        with open(json_file, "r") as f:
            pmids = json.load(f)
        print(f"Loaded {len(pmids)} PMIDs from {json_file}")
        return pmids
