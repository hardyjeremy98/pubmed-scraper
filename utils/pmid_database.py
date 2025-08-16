#!/usr/bin/env python3
"""
Database Manager for PubMed Processing

This module provides SQLite-based storage for tracking processed PMIDs and their status.
It allows for efficient lookup and updating of article processing status.

Author: AI Assistant
Date: August 16, 2025
"""

import sqlite3
import os
import json
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import time


class PMIDDatabase:
    """Manages SQLite database operations for PMID tracking."""

    def __init__(self, db_path: str = "data/pmid_tracker.db"):
        """
        Initialize the database connection and create tables if they don't exist.

        Args:
            db_path: Path to the SQLite database file
        """
        # Make sure the data directory exists
        os.makedirs(os.path.dirname(db_path), exist_ok=True)

        self.db_path = db_path
        self.conn = None
        self.cursor = None
        self._connect()
        self._create_tables()

    def _connect(self) -> None:
        """Establish a connection to the SQLite database."""
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row  # Return rows as dictionaries
        self.cursor = self.conn.cursor()

    def _create_tables(self) -> None:
        """Create the necessary tables if they don't exist."""
        # Table for tracking processed PMIDs
        self.cursor.execute(
            """
        CREATE TABLE IF NOT EXISTS pmids (
            pmid TEXT PRIMARY KEY,
            processed BOOLEAN NOT NULL DEFAULT 0,
            processing_date TEXT,
            publisher TEXT,
            has_metadata BOOLEAN NOT NULL DEFAULT 0,
            has_figures BOOLEAN NOT NULL DEFAULT 0,
            has_pdf BOOLEAN NOT NULL DEFAULT 0,
            error TEXT,
            tags TEXT,
            last_updated TEXT NOT NULL
        )
        """
        )

        # Table for tracking extracted figures
        self.cursor.execute(
            """
        CREATE TABLE IF NOT EXISTS figures (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pmid TEXT NOT NULL,
            figure_id INTEGER NOT NULL,
            figure_name TEXT NOT NULL,
            caption TEXT,
            has_tht_plots BOOLEAN NOT NULL DEFAULT 0,
            tht_plot_count INTEGER DEFAULT 0,
            FOREIGN KEY (pmid) REFERENCES pmids (pmid),
            UNIQUE (pmid, figure_id)
        )
        """
        )

        # Table for tracking ThT plots
        self.cursor.execute(
            """
        CREATE TABLE IF NOT EXISTS tht_plots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pmid TEXT NOT NULL,
            figure_id INTEGER NOT NULL,
            plot_number INTEGER NOT NULL,
            extracted BOOLEAN NOT NULL DEFAULT 0,
            digitized BOOLEAN NOT NULL DEFAULT 0,
            processed BOOLEAN NOT NULL DEFAULT 0,
            FOREIGN KEY (pmid, figure_id) REFERENCES figures (pmid, figure_id),
            UNIQUE (pmid, figure_id, plot_number)
        )
        """
        )

        # Create indexes for faster queries
        self.cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_pmids_processed ON pmids(processed)"
        )
        self.cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_figures_pmid ON figures(pmid)"
        )
        self.cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_tht_plots_pmid ON tht_plots(pmid)"
        )

        self.conn.commit()

    def add_pmid(self, pmid: str, tags: Optional[List[str]] = None) -> None:
        """
        Add a new PMID to the database.

        Args:
            pmid: The PubMed ID to add
            tags: Optional list of tags for categorizing PMIDs
        """
        tags_json = json.dumps(tags) if tags else None
        now = time.strftime("%Y-%m-%d %H:%M:%S")

        self.cursor.execute(
            """
        INSERT OR IGNORE INTO pmids 
        (pmid, processed, last_updated, tags) 
        VALUES (?, 0, ?, ?)
        """,
            (pmid, now, tags_json),
        )

        self.conn.commit()

    def add_multiple_pmids(
        self, pmids: List[str], tags: Optional[List[str]] = None
    ) -> None:
        """
        Add multiple PMIDs to the database.

        Args:
            pmids: List of PMIDs to add
            tags: Optional list of tags for categorizing PMIDs
        """
        tags_json = json.dumps(tags) if tags else None
        now = time.strftime("%Y-%m-%d %H:%M:%S")

        values = [(pmid, 0, now, tags_json) for pmid in pmids]

        self.cursor.executemany(
            """
        INSERT OR IGNORE INTO pmids 
        (pmid, processed, last_updated, tags) 
        VALUES (?, ?, ?, ?)
        """,
            values,
        )

        self.conn.commit()

    def is_pmid_processed(self, pmid: str) -> bool:
        """
        Check if a PMID has already been processed.

        Args:
            pmid: The PubMed ID to check

        Returns:
            True if the PMID has been processed, False otherwise
        """
        self.cursor.execute("SELECT processed FROM pmids WHERE pmid = ?", (pmid,))
        result = self.cursor.fetchone()

        if result:
            return bool(result["processed"])
        return False

    def get_unprocessed_pmids(self, limit: Optional[int] = None) -> List[str]:
        """
        Get a list of unprocessed PMIDs.

        Args:
            limit: Optional maximum number of PMIDs to return

        Returns:
            List of unprocessed PMIDs
        """
        query = "SELECT pmid FROM pmids WHERE processed = 0"
        if limit:
            query += f" LIMIT {limit}"

        self.cursor.execute(query)
        results = self.cursor.fetchall()

        return [row["pmid"] for row in results]

    def mark_pmid_as_processed(
        self,
        pmid: str,
        publisher: Optional[str] = None,
        has_metadata: bool = False,
        has_figures: bool = False,
        has_pdf: bool = False,
        error: Optional[str] = None,
    ) -> None:
        """
        Mark a PMID as processed.

        Args:
            pmid: The PubMed ID to update
            publisher: Optional publisher information
            has_metadata: Whether metadata was successfully extracted
            has_figures: Whether figures were successfully extracted
            has_pdf: Whether PDF was successfully downloaded
            error: Optional error message if processing failed
        """
        now = time.strftime("%Y-%m-%d %H:%M:%S")

        self.cursor.execute(
            """
        INSERT OR REPLACE INTO pmids 
        (pmid, processed, processing_date, publisher, has_metadata, has_figures, has_pdf, error, last_updated) 
        VALUES (?, 1, ?, ?, ?, ?, ?, ?, ?)
        """,
            (pmid, now, publisher, has_metadata, has_figures, has_pdf, error, now),
        )

        self.conn.commit()

    def add_figure(
        self,
        pmid: str,
        figure_id: int,
        figure_name: str,
        caption: Optional[str] = None,
        has_tht_plots: bool = False,
        tht_plot_count: int = 0,
    ) -> None:
        """
        Add figure information to the database.

        Args:
            pmid: The PubMed ID
            figure_id: Figure number/ID
            figure_name: Name of the figure (e.g., "Figure 3")
            caption: Optional figure caption
            has_tht_plots: Whether the figure contains ThT plots
            tht_plot_count: Number of ThT plots in the figure
        """
        self.cursor.execute(
            """
        INSERT OR REPLACE INTO figures 
        (pmid, figure_id, figure_name, caption, has_tht_plots, tht_plot_count) 
        VALUES (?, ?, ?, ?, ?, ?)
        """,
            (pmid, figure_id, figure_name, caption, has_tht_plots, tht_plot_count),
        )

        self.conn.commit()

    def add_tht_plot(
        self,
        pmid: str,
        figure_id: int,
        plot_number: int,
        extracted: bool = False,
        digitized: bool = False,
        processed: bool = False,
    ) -> None:
        """
        Add a ThT plot to the database.

        Args:
            pmid: The PubMed ID
            figure_id: Figure number/ID
            plot_number: Plot number within the figure
            extracted: Whether data has been extracted
            digitized: Whether the plot has been digitized
            processed: Whether the plot data has been fully processed
        """
        self.cursor.execute(
            """
        INSERT OR REPLACE INTO tht_plots 
        (pmid, figure_id, plot_number, extracted, digitized, processed) 
        VALUES (?, ?, ?, ?, ?, ?)
        """,
            (pmid, figure_id, plot_number, extracted, digitized, processed),
        )

        self.conn.commit()

    def get_pmid_status(self, pmid: str) -> Optional[Dict[str, Any]]:
        """
        Get the processing status of a PMID.

        Args:
            pmid: The PubMed ID to check

        Returns:
            Dictionary with PMID status information or None if not found
        """
        self.cursor.execute(
            """
        SELECT * FROM pmids WHERE pmid = ?
        """,
            (pmid,),
        )

        result = self.cursor.fetchone()
        if not result:
            return None

        # Convert to dictionary
        status = dict(result)

        # Get figures
        self.cursor.execute(
            """
        SELECT * FROM figures WHERE pmid = ?
        """,
            (pmid,),
        )

        figures = [dict(row) for row in self.cursor.fetchall()]
        status["figures"] = figures

        # Get ThT plots
        self.cursor.execute(
            """
        SELECT * FROM tht_plots WHERE pmid = ?
        """,
            (pmid,),
        )

        tht_plots = [dict(row) for row in self.cursor.fetchall()]
        status["tht_plots"] = tht_plots

        # Parse tags
        if status["tags"]:
            try:
                status["tags"] = json.loads(status["tags"])
            except:
                status["tags"] = []
        else:
            status["tags"] = []

        return status

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the database.

        Returns:
            Dictionary with database statistics
        """
        stats = {}

        # Total PMIDs
        self.cursor.execute("SELECT COUNT(*) as count FROM pmids")
        stats["total_pmids"] = self.cursor.fetchone()["count"]

        # Processed PMIDs
        self.cursor.execute("SELECT COUNT(*) as count FROM pmids WHERE processed = 1")
        stats["processed_pmids"] = self.cursor.fetchone()["count"]

        # Unprocessed PMIDs
        stats["unprocessed_pmids"] = stats["total_pmids"] - stats["processed_pmids"]

        # PMIDs with metadata
        self.cursor.execute(
            "SELECT COUNT(*) as count FROM pmids WHERE has_metadata = 1"
        )
        stats["pmids_with_metadata"] = self.cursor.fetchone()["count"]

        # PMIDs with figures
        self.cursor.execute("SELECT COUNT(*) as count FROM pmids WHERE has_figures = 1")
        stats["pmids_with_figures"] = self.cursor.fetchone()["count"]

        # PMIDs with PDFs
        self.cursor.execute("SELECT COUNT(*) as count FROM pmids WHERE has_pdf = 1")
        stats["pmids_with_pdf"] = self.cursor.fetchone()["count"]

        # PMIDs with errors
        self.cursor.execute(
            "SELECT COUNT(*) as count FROM pmids WHERE error IS NOT NULL"
        )
        stats["pmids_with_errors"] = self.cursor.fetchone()["count"]

        # Total figures
        self.cursor.execute("SELECT COUNT(*) as count FROM figures")
        stats["total_figures"] = self.cursor.fetchone()["count"]

        # Figures with ThT plots
        self.cursor.execute(
            "SELECT COUNT(*) as count FROM figures WHERE has_tht_plots = 1"
        )
        stats["figures_with_tht_plots"] = self.cursor.fetchone()["count"]

        # Total ThT plots
        self.cursor.execute("SELECT COUNT(*) as count FROM tht_plots")
        stats["total_tht_plots"] = self.cursor.fetchone()["count"]

        # Extracted ThT plots
        self.cursor.execute(
            "SELECT COUNT(*) as count FROM tht_plots WHERE extracted = 1"
        )
        stats["extracted_tht_plots"] = self.cursor.fetchone()["count"]

        # Digitized ThT plots
        self.cursor.execute(
            "SELECT COUNT(*) as count FROM tht_plots WHERE digitized = 1"
        )
        stats["digitized_tht_plots"] = self.cursor.fetchone()["count"]

        # Processed ThT plots
        self.cursor.execute(
            "SELECT COUNT(*) as count FROM tht_plots WHERE processed = 1"
        )
        stats["processed_tht_plots"] = self.cursor.fetchone()["count"]

        # Publishers
        self.cursor.execute(
            """
        SELECT publisher, COUNT(*) as count 
        FROM pmids 
        WHERE publisher IS NOT NULL 
        GROUP BY publisher 
        ORDER BY count DESC
        """
        )

        publishers = [
            (row["publisher"], row["count"]) for row in self.cursor.fetchall()
        ]
        stats["publishers"] = publishers

        return stats

    def export_to_json(
        self, output_file: str = "data/pmid_database_export.json"
    ) -> None:
        """
        Export the database to a JSON file.

        Args:
            output_file: Path to the output JSON file
        """
        # Get all PMIDs
        self.cursor.execute("SELECT pmid FROM pmids")
        pmids = [row["pmid"] for row in self.cursor.fetchall()]

        # Get status for each PMID
        data = {pmid: self.get_pmid_status(pmid) for pmid in pmids}

        # Add statistics
        data["_statistics"] = self.get_statistics()

        # Write to file
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(data, f, indent=2)

    def import_from_json(self, json_file: str = "unique_pmids.json") -> int:
        """
        Import PMIDs from a JSON file.

        Args:
            json_file: Path to the JSON file with PMIDs

        Returns:
            Number of PMIDs imported
        """
        try:
            with open(json_file, "r") as f:
                pmids = json.load(f)

            if isinstance(pmids, list):
                self.add_multiple_pmids(pmids)
                return len(pmids)
            elif isinstance(pmids, dict):
                # Assuming the JSON is a dictionary with PMIDs as keys
                self.add_multiple_pmids(list(pmids.keys()))
                return len(pmids)

            return 0
        except Exception as e:
            print(f"Error importing PMIDs from {json_file}: {e}")
            return 0

    def close(self) -> None:
        """Close the database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None
            self.cursor = None


def migrate_existing_data(
    db_manager: PMIDDatabase,
    articles_dir: str = "data/articles_data",
    unique_pmids_file: str = "data/unique_pmids.json",
) -> Tuple[int, int, int]:
    """
    Migrate existing data to the SQLite database.

    Args:
        db_manager: PMIDDatabase instance
        articles_dir: Directory containing article data
        unique_pmids_file: Path to the JSON file with unique PMIDs

    Returns:
        Tuple of (imported_pmids, processed_articles, imported_figures)
    """
    # First import all PMIDs from the unique_pmids.json file
    imported_pmids = 0
    if os.path.exists(unique_pmids_file):
        imported_pmids = db_manager.import_from_json(unique_pmids_file)
        print(f"Imported {imported_pmids} PMIDs from {unique_pmids_file}")

    # Check existing directories to find processed articles
    processed_articles = 0
    imported_figures = 0

    if os.path.exists(articles_dir):
        pmid_dirs = [
            d
            for d in os.listdir(articles_dir)
            if os.path.isdir(os.path.join(articles_dir, d)) and d.isdigit()
        ]

        print(f"Found {len(pmid_dirs)} article directories to process...")

        for pmid in pmid_dirs:
            pmid_dir = os.path.join(articles_dir, pmid)

            # Check for metadata
            metadata_file = os.path.join(pmid_dir, f"{pmid}_metadata.json")
            has_metadata = os.path.exists(metadata_file)

            # Get publisher from metadata if available
            publisher = None
            if has_metadata:
                try:
                    with open(metadata_file, "r") as f:
                        metadata = json.load(f)
                        publisher = metadata.get("publisher", None)
                except:
                    pass

            # Check for figures
            figures_file = os.path.join(pmid_dir, f"{pmid}_figures.json")
            has_figures = os.path.exists(figures_file)

            # Import figures if available
            if has_figures:
                try:
                    with open(figures_file, "r") as f:
                        figures_data = json.load(f)

                    for figure in figures_data:
                        fig_id = figure.get("id")
                        if fig_id is not None:
                            db_manager.add_figure(
                                pmid=pmid,
                                figure_id=fig_id,
                                figure_name=figure.get("name", f"Figure {fig_id}"),
                                caption=figure.get("caption", ""),
                            )
                            imported_figures += 1
                except Exception as e:
                    print(f"Error importing figures for PMID {pmid}: {e}")

            # Check for PDF
            pdf_dir = os.path.join(pmid_dir, "pdf")
            has_pdf = os.path.exists(pdf_dir) and any(
                f.endswith(".pdf")
                for f in os.listdir(pdf_dir)
                if os.path.isfile(os.path.join(pdf_dir, f))
            )

            # Check for ThT plots
            tht_files = [
                f
                for f in os.listdir(pmid_dir)
                if f.endswith("_tht_identification.json")
            ]

            for tht_file in tht_files:
                try:
                    with open(os.path.join(pmid_dir, tht_file), "r") as f:
                        tht_data = json.load(f)

                    figure_name = tht_data.get("figure_name", "")
                    if figure_name:
                        # Extract figure number
                        import re

                        match = re.search(r"figure_(\d+)", figure_name)
                        if match:
                            figure_id = int(match.group(1))

                            # Get ThT plot numbers
                            tht_plot_numbers = tht_data.get("tht_plot_numbers", [])

                            # Update figure with ThT information
                            db_manager.add_figure(
                                pmid=pmid,
                                figure_id=figure_id,
                                figure_name=figure_name,
                                has_tht_plots=bool(tht_plot_numbers),
                                tht_plot_count=len(tht_plot_numbers),
                            )

                            # Add individual ThT plots
                            for plot_num in tht_plot_numbers:
                                # Check if plot has been extracted
                                plot_extraction_file = os.path.join(
                                    pmid_dir,
                                    f"{figure_name}_plot{plot_num}_extraction.json",
                                )
                                extracted = os.path.exists(plot_extraction_file)

                                db_manager.add_tht_plot(
                                    pmid=pmid,
                                    figure_id=figure_id,
                                    plot_number=plot_num,
                                    extracted=extracted,
                                )
                except Exception as e:
                    print(f"Error importing ThT data for {tht_file}: {e}")

            # Mark the PMID as processed
            db_manager.mark_pmid_as_processed(
                pmid=pmid,
                publisher=publisher,
                has_metadata=has_metadata,
                has_figures=has_figures,
                has_pdf=has_pdf,
            )

            processed_articles += 1

            # Print progress every 10 articles
            if processed_articles % 10 == 0:
                print(f"Processed {processed_articles}/{len(pmid_dirs)} articles...")

    return imported_pmids, processed_articles, imported_figures


if __name__ == "__main__":
    print("PMID Database Initializer")
    print("=" * 40)

    # Initialize database
    db_manager = PMIDDatabase()

    # Migrate existing data
    print("\nMigrating existing data...")
    imported_pmids, processed_articles, imported_figures = migrate_existing_data(
        db_manager
    )

    # Print statistics
    print("\nDatabase Statistics:")
    stats = db_manager.get_statistics()
    print(f"  Total PMIDs: {stats['total_pmids']}")
    print(f"  Processed PMIDs: {stats['processed_pmids']}")
    print(f"  Unprocessed PMIDs: {stats['unprocessed_pmids']}")
    print(f"  PMIDs with metadata: {stats['pmids_with_metadata']}")
    print(f"  PMIDs with figures: {stats['pmids_with_figures']}")
    print(f"  PMIDs with PDFs: {stats['pmids_with_pdf']}")
    print(f"  Total figures: {stats['total_figures']}")
    print(f"  Figures with ThT plots: {stats['figures_with_tht_plots']}")
    print(f"  Total ThT plots: {stats['total_tht_plots']}")

    # Export data
    print("\nExporting database to JSON...")
    db_manager.export_to_json()

    print("\nDone!")
    db_manager.close()
