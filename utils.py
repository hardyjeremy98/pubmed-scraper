import os
import pandas as pd
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from io import StringIO


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


# Table Reformatting Utilities
def is_wrong_format(df: pd.DataFrame) -> bool:
    """
    Check if data needs reformatting - only if it's in long format with good time series coverage.

    Args:
        df: DataFrame to check

    Returns:
        bool: True if data should be pivoted from long to wide format
    """
    df = df.copy()
    df.columns = df.columns.str.strip()

    # Must have Series, X, Y columns
    if not all(col in df.columns for col in ["Series", "X", "Y"]):
        return False

    # Check if all series have the same X values (indicating it's safe to pivot)
    x_values_per_series = {}
    for series in df["Series"].unique():
        x_vals = set(df[df["Series"] == series]["X"].values)
        x_values_per_series[series] = x_vals

    # Get all unique X values
    all_x_values = set(df["X"].unique())

    # Check if most series have most X values (>80% overlap)
    overlap_threshold = 0.8
    series_with_good_overlap = 0

    for series, x_vals in x_values_per_series.items():
        overlap = len(x_vals.intersection(all_x_values)) / len(all_x_values)
        if overlap >= overlap_threshold:
            series_with_good_overlap += 1

    # Only pivot if most series have good X value coverage
    good_coverage = series_with_good_overlap / len(x_values_per_series) >= 0.8

    return good_coverage


def reformat_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reformat table only if it makes sense (minimal data loss).

    Args:
        df: DataFrame to reformat

    Returns:
        pd.DataFrame: Reformatted dataframe (either pivoted or original)
    """
    df = df.copy()  # Don't modify original
    df.columns = df.columns.str.strip()

    if not is_wrong_format(df):
        return df

    # Check for duplicates
    duplicate_keys = df.duplicated(subset=["Series", "X"])
    if duplicate_keys.any():
        print("Warning: Duplicate Series/X pairs found. Aggregating with mean.")
        df = df.groupby(["Series", "X"], as_index=False).mean(numeric_only=True)

    # Pivot
    reformatted = df.pivot(index="X", columns="Series", values="Y").reset_index()
    reformatted.columns.name = None
    reformatted = reformatted.rename(columns={"X": "Time"})

    return reformatted


def load_csv_auto(filepath: str) -> pd.DataFrame:
    """
    Load CSV with automatic separator detection.

    Args:
        filepath: Path to CSV file

    Returns:
        pd.DataFrame: Loaded dataframe
    """
    try:
        df = pd.read_csv(filepath)
    except Exception:
        df = pd.read_csv(filepath, sep=None, engine="python")  # fallback
    return df


def analyze_data_completeness(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze how much data would be lost in pivoting.

    Args:
        df: DataFrame to analyze

    Returns:
        dict: Analysis results including completeness metrics
    """
    if not all(col in df.columns for col in ["Series", "X", "Y"]):
        return {"error": "Cannot analyze - missing required columns"}

    # Calculate what would happen if we pivot
    unique_x = len(df["X"].unique())
    unique_series = len(df["Series"].unique())
    total_possible_combinations = unique_x * unique_series
    actual_data_points = len(df)

    completeness = actual_data_points / total_possible_combinations

    analysis = {
        "unique_time_points": unique_x,
        "unique_series": unique_series,
        "actual_data_points": actual_data_points,
        "possible_combinations": total_possible_combinations,
        "data_completeness": completeness,
        "missing_if_pivoted": total_possible_combinations - actual_data_points,
        "recommendation": "pivot" if completeness > 0.8 else "keep_original",
    }

    return analysis


def process_csv_file(
    filepath: str, output_dir: str = "cleaned_tables"
) -> Dict[str, Any]:
    """
    Process a CSV file with automatic reformatting and save cleaned version.

    Args:
        filepath: Path to input CSV file
        output_dir: Directory to save cleaned file

    Returns:
        dict: Processing results and metadata
    """
    import os
    from pathlib import Path

    try:
        # Load the data
        df = load_csv_auto(filepath)

        # Analyze the data
        analysis = analyze_data_completeness(df)

        # Reformat if needed
        cleaned_df = reformat_table(df)

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Generate output filename
        input_path = Path(filepath)
        base_name = input_path.stem
        cleaned_filename = f"{base_name}_cleaned.csv"
        output_filepath = os.path.join(output_dir, cleaned_filename)

        # Save cleaned data
        cleaned_df.to_csv(output_filepath, index=False)

        result = {
            "success": True,
            "input_file": filepath,
            "output_file": output_filepath,
            "original_shape": df.shape,
            "cleaned_shape": cleaned_df.shape,
            "was_reformatted": cleaned_df.shape != df.shape,
            "analysis": analysis,
            "error": None,
        }

        return result

    except Exception as e:
        return {
            "success": False,
            "input_file": filepath,
            "output_file": None,
            "original_shape": None,
            "cleaned_shape": None,
            "was_reformatted": False,
            "analysis": None,
            "error": str(e),
        }
