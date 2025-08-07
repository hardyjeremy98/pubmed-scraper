#!/usr/bin/env python3
"""
Test script to verify publisher information is being added to metadata.json files
"""

import json
import os
from article_fetcher import DataFetcher
from http_session import HTTPSession
from storage import ArticleManager


def test_publisher_extraction():
    """Test that publisher information is correctly extracted and saved."""

    # Initialize components
    http_session = HTTPSession()
    metadata_fetcher = DataFetcher(email="test@example.com", http_session=http_session)
    article_manager = ArticleManager()

    # Test with a known PMID that should have a DOI
    test_pmid = "33442817"  # Example PMID

    print(f"Testing publisher extraction for PMID: {test_pmid}")

    try:
        # Get article metadata (this should now include publisher)
        article = metadata_fetcher.get_article_metadata(test_pmid)

        print(f"PMID: {article.pmid}")
        print(f"Title: {article.title[:100]}...")
        print(f"DOI: {article.doi}")
        print(f"Journal: {article.journal}")
        print(f"Publisher: {article.publisher}")

        # Test the publisher extraction function directly
        if article.doi:
            publisher_direct = metadata_fetcher.get_publisher_from_doi(article.doi)
            print(f"Direct publisher lookup: {publisher_direct}")

        # Save to JSON and verify the publisher field is included
        filepath = article_manager.save_article_to_json(article, base_dir="test_output")

        # Read back the JSON to verify publisher is saved
        with open(filepath, "r") as f:
            saved_data = json.load(f)

        print("\nSaved JSON contains:")
        for key, value in saved_data.items():
            if key != "html_content":  # Skip HTML content for cleaner output
                print(f"  {key}: {value}")

        # Verify publisher field exists
        if "publisher" in saved_data:
            print(f"\n✓ Publisher field successfully added: {saved_data['publisher']}")
        else:
            print("\n✗ Publisher field missing from saved data")

    except Exception as e:
        print(f"Error during test: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_publisher_extraction()
