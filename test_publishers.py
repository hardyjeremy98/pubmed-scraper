#!/usr/bin/env python3
"""
Quick test of different DOIs to verify publisher detection
"""

from article_fetcher import MetadataFetcher
from http_session import HTTPSession


def test_different_publishers():
    """Test publisher detection for different DOI prefixes."""

    http_session = HTTPSession()
    metadata_fetcher = MetadataFetcher(
        email="test@example.com", http_session=http_session
    )

    test_dois = [
        "10.1038/s41586-021-03819-2",  # Nature
        "10.1016/j.cell.2021.05.015",  # Elsevier
        "10.1021/acs.biochem.1c00123",  # ACS
        "10.1371/journal.pone.0123456",  # PLOS
        "10.1126/science.abm1234",  # Science
        "10.1002/prot.26123",  # Wiley
        "10.1093/nar/gkab123",  # Oxford
    ]

    print("Testing publisher detection for different DOIs:")
    print("-" * 50)

    for doi in test_dois:
        publisher = metadata_fetcher.get_publisher_from_doi(doi)
        print(f"DOI: {doi}")
        print(f"Publisher: {publisher}")
        print()


if __name__ == "__main__":
    test_different_publishers()
