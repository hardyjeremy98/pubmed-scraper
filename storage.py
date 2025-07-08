import json
import os
from typing import List
from utils import ArticleMetadata, Figure, ensure_pmid_directory


class ArticleManager:
    """Manages article storage and retrieval."""

    @staticmethod
    def save_article_to_json(
        article: ArticleMetadata, base_dir: str = "articles_data"
    ) -> str:
        """Save article data to JSON file in the PMID-specific directory."""
        # Create PMID-specific directory
        pmid_dir = ensure_pmid_directory(article.pmid, base_dir)

        # Use PMID as filename
        filename = f"{article.pmid}_metadata.json"
        filepath = os.path.join(pmid_dir, filename)

        # Save article data
        with open(filepath, "w") as f:
            json.dump(article.to_dict(), f, indent=2)
        print(f"Saved article metadata to {filepath}")
        return filepath

    @staticmethod
    def save_figures_to_json(
        figures: List[Figure], pmid: str, base_dir: str = "articles_data"
    ) -> str:
        """Save figures metadata to JSON file in the PMID-specific directory.
        Only saves figures that have non-empty captions."""
        # Create PMID-specific directory
        pmid_dir = ensure_pmid_directory(pmid, base_dir)

        # Prepare figures data for JSON serialization
        # Only include figures with non-empty captions
        figures_data = []
        figure_counter = 1
        for figure in figures:
            # Skip figures without captions or with empty captions
            if not figure.caption or not figure.caption.strip():
                continue

            # Extract file extension from URL or use .jpg as default
            file_ext = (
                figure.url.split(".")[-1] if "." in figure.url.split("/")[-1] else "jpg"
            )

            figures_data.append(
                {
                    "id": figure_counter,
                    "url": figure.url,
                    "alt": figure.alt,
                    "caption": figure.caption,
                    "local_filename": f"figure_{figure_counter}.{file_ext}",
                }
            )
            figure_counter += 1

        filename = f"{pmid}_figures.json"
        filepath = os.path.join(pmid_dir, filename)

        # Save figures data
        with open(filepath, "w") as f:
            json.dump(figures_data, f, indent=2)

        print(f"Saved {len(figures_data)} figures with captions to {filepath}")
        if len(figures_data) != len(figures):
            print(
                f"  Filtered out {len(figures) - len(figures_data)} figures without captions"
            )

        return filepath

    @staticmethod
    def load_pmids_from_json(json_file: str = "unique_pmids.json") -> List[str]:
        """Load list of PMIDs from a JSON file."""
        with open(json_file, "r") as f:
            pmids = json.load(f)
        print(f"Loaded {len(pmids)} PMIDs from {json_file}")
        return pmids
