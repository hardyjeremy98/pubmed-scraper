import os
import pandas as pd
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple
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
    Check if data needs reformatting - if it's in long format with Series, X, Y columns.

    Args:
        df: DataFrame to check

    Returns:
        bool: True if data should be pivoted from long to wide format
    """
    df = df.copy()
    df.columns = df.columns.str.strip()

    # Must have Series, X, Y columns - if so, reformat it
    return all(col in df.columns for col in ["Series", "X", "Y"])


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


def merge_constants_and_variables(extracted_data: Dict) -> Dict:
    """
    Merge constants and variables JSONs for each plot number into total_conditions.

    For each plot number, the function creates individual condition dictionaries
    by combining the constants (shared across all conditions) with each variable
    entry (specific to each symbol/condition).

    Args:
        extracted_data: Dictionary with plot numbers as keys, each containing
                       "variables" and "constants" dictionaries

    Returns:
        Dictionary with same structure but "total_conditions" key instead of
        "variables" and "constants"

    Example:
        Input:
        {
            "1": {
                "variables": {
                    "•": {"Mutation": ["WT"], "Additives": []},
                    "○": {"Mutation": ["R17C"], "Additives": ["H2O2"]}
                },
                "constants": {
                    "Protein": ["Ure2p"],
                    "Temperature": ["8 °C"]
                }
            }
        }

        Output:
        {
            "1": {
                "total_conditions": {
                    "•": {"Protein": ["Ure2p"], "Temperature": ["8 °C"],
                          "Mutation": ["WT"], "Additives": []},
                    "○": {"Protein": ["Ure2p"], "Temperature": ["8 °C"],
                          "Mutation": ["R17C"], "Additives": ["H2O2"]}
                }
            }
        }
    """
    merged_data = {}

    for plot_number, plot_data in extracted_data.items():
        if not isinstance(plot_data, dict):
            continue

        variables = plot_data.get("variables", {})
        constants = plot_data.get("constants", {})

        # Create total_conditions dictionary
        total_conditions = {}

        # For each variable symbol (•, ○, ▾, ▿, etc.)
        for symbol, variable_data in variables.items():
            # Start with constants as base
            condition = constants.copy()

            # Override/add with variable-specific data
            for key, value in variable_data.items():
                condition[key] = value

            total_conditions[symbol] = condition

        # Store in merged data structure
        merged_data[plot_number] = {"total_conditions": total_conditions}

    return merged_data


def create_clean_merged_file(input_file: str, output_file: str) -> bool:
    """
    Create a clean merged JSON file containing only experimental condition data.

    This utility function takes a ThT data extraction file and creates a clean
    merged version that contains only the experimental conditions, without any
    metadata like PMIDs, file paths, LLM responses, etc.

    Args:
        input_file: Path to the original ThT data extraction JSON file
        output_file: Path where the clean merged data should be saved

    Returns:
        bool: True if successful, False if failed

    Example usage:
        success = create_clean_merged_file(
            "articles_data/19258323/figure_1_tht_data_extraction.json",
            "articles_data/19258323/figure_1_tht_data_merged.json"
        )
    """
    try:
        import json

        # Load the original extraction data
        with open(input_file, "r") as f:
            full_data = json.load(f)

        # Extract only the extracted_data portion
        extracted_data = full_data.get("extracted_data", {})

        if not extracted_data:
            return False

        # Merge constants and variables
        merged_result = merge_constants_and_variables(extracted_data)

        # Save only the merged result (no metadata)
        with open(output_file, "w") as f:
            json.dump(merged_result, f, indent=2, ensure_ascii=False)

        return True

    except Exception:
        return False


def map_csv_data_to_conditions(
    csv_path: str,
    experimental_conditions: Dict,
    match_tuples: List[Tuple[str, str]],
    plot_title: str = "Plot",
) -> Dict[str, Any]:
    """
    Map CSV table data to experimental conditions using tuple pairs.

    Args:
        csv_path: Path to the CSV file containing the plot data
        experimental_conditions: Dictionary containing experimental conditions
                                (should have 'total_conditions' key)
        match_tuples: List of tuples mapping (legend_symbol, column_header)
                     e.g., [("•", "Black Circles"), ("○", "Red Triangles")]
        plot_title: Title for the plot dictionary (e.g., "Plot 1")

    Returns:
        Dictionary with plot title as key, containing experimental conditions
        with Time and Fluorescence data embedded as separate items

    Example output:
        {
            "Plot 1": {
                "•": {
                    "Protein": ["Ure2p"],
                    "Temperature": ["8 °C"],
                    "Mutation": ["WT"],
                    "Additives": [],
                    "Time": [0, 5, 10, 15],
                    "Fluorescence": [1.2, 1.8, 2.5, 3.1]
                },
                "○": {
                    "Protein": ["Ure2p"],
                    "Temperature": ["8 °C"],
                    "Mutation": ["R17C"],
                    "Additives": ["H2O2"],
                    "Time": [0, 5, 10, 15],
                    "Fluorescence": [0.8, 1.1, 1.9, 2.7]
                }
            }
        }
    """
    import json

    try:
        # Load CSV data
        df = pd.read_csv(csv_path)

        if df.empty:
            return {"error": "CSV file is empty"}

        # Get all columns
        all_columns = df.columns.tolist()

        # Assume first column is time
        time_column = all_columns[0] if all_columns else None
        data_columns = all_columns[1:] if len(all_columns) > 1 else []

        if not time_column:
            return {"error": "No columns found in CSV"}

        # Get experimental conditions
        total_conditions = experimental_conditions.get("total_conditions", {})

        # Create the plot dictionary
        plot_data = {}
        matched_columns = set()

        for legend_symbol, column_header in match_tuples:
            # Check if the column exists in the CSV
            if column_header not in data_columns:
                print(f"Warning: Column '{column_header}' not found in CSV")
                continue

            # Check if the legend symbol exists in experimental conditions
            if legend_symbol not in total_conditions:
                print(
                    f"Warning: Legend symbol '{legend_symbol}' not found in experimental conditions"
                )
                continue

            # Start with the experimental conditions for this symbol
            condition_data = total_conditions[legend_symbol].copy()

            # Extract time and fluorescence data for this column
            time_values = []
            fluorescence_values = []

            for _, row in df.iterrows():
                time_value = row[time_column]
                data_value = row[column_header]

                # Skip rows with missing data
                if pd.isna(time_value) or pd.isna(data_value):
                    continue

                time_values.append(float(time_value) if pd.notna(time_value) else None)
                fluorescence_values.append(
                    float(data_value) if pd.notna(data_value) else None
                )

            # Add Time and Fluorescence data to the experimental conditions
            condition_data["Time"] = time_values
            condition_data["Fluorescence"] = fluorescence_values

            # Add this condition to the plot data
            plot_data[legend_symbol] = condition_data
            matched_columns.add(column_header)

        # Calculate metadata for reporting
        unmatched_columns = [col for col in data_columns if col not in matched_columns]
        total_data_points = sum(
            len(condition["Time"]) for condition in plot_data.values()
        )

        # Create the result structure with plot title as key
        result = {
            plot_title: plot_data,
            "_metadata": {
                "csv_file": csv_path,
                "total_data_points": total_data_points,
                "time_column": time_column,
                "matched_columns": len(matched_columns),
                "unmatched_columns": len(unmatched_columns),
                "unmatched_column_names": unmatched_columns,
            },
        }

        return result

    except Exception as e:
        return {"error": f"Error processing CSV data: {str(e)}"}


def save_mapped_data_to_json(mapped_data: Dict[str, Any], output_file: str) -> bool:
    """
    Save the mapped CSV and experimental condition data to a JSON file.

    Args:
        mapped_data: Dictionary containing the mapped data from map_csv_data_to_conditions
        output_file: Path where the JSON file should be saved

    Returns:
        bool: True if successful, False if failed
    """
    try:
        import json
        import os

        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # Save to JSON file
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(mapped_data, f, indent=2, ensure_ascii=False)

        return True

    except Exception as e:
        print(f"Error saving mapped data to JSON: {e}")
        return False


def consolidate_final_data_to_csv(
    final_data_dir: str = "final_data",
    output_file: str = "consolidated_experimental_data.csv",
) -> bool:
    """
    Consolidate all JSON files in final_data directory into a single CSV file.

    Each row represents one experimental condition with Time and Fluorescence as list columns.
    Columns include all experimental parameters plus Time (as list), Fluorescence (as list), PMID, Figure, Plot, and Legend Symbol.

    Args:
        final_data_dir: Directory containing the mapped data JSON files
        output_file: Path where the consolidated CSV should be saved

    Returns:
        bool: True if successful, False if failed
    """
    try:
        import json
        import glob
        import pandas as pd
        import os

        # Get all JSON files in the final_data directory
        json_pattern = os.path.join(final_data_dir, "*_mapped_data.json")
        json_files = glob.glob(json_pattern)

        if not json_files:
            print(f"No mapped data JSON files found in {final_data_dir}")
            return False

        print(f"Found {len(json_files)} JSON files to consolidate")

        # Collect all data rows
        all_rows = []
        all_columns = set()

        for json_file in json_files:
            try:
                # Extract PMID from filename
                filename = os.path.basename(json_file)
                pmid = filename.split("_")[0]

                # Load JSON data
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                # Skip metadata key
                plot_data = {k: v for k, v in data.items() if k != "_metadata"}

                # Process each plot
                for plot_title, conditions in plot_data.items():
                    # Extract figure and plot info from title (e.g., "Figure 1 Plot 1")
                    figure_plot_parts = plot_title.split()
                    figure_num = (
                        figure_plot_parts[1]
                        if len(figure_plot_parts) >= 2
                        else "Unknown"
                    )
                    plot_num = (
                        figure_plot_parts[3]
                        if len(figure_plot_parts) >= 4
                        else "Unknown"
                    )

                    # Process each experimental condition (legend symbol)
                    for legend_symbol, condition_data in conditions.items():
                        # Create one row per experimental condition
                        row = {
                            "PMID": pmid,
                            "Figure": figure_num,
                            "Plot": plot_num,
                            "Legend_Symbol": legend_symbol,
                        }

                        # Add all experimental conditions and data
                        for key, value in condition_data.items():
                            if key == "Time":
                                # Convert Time list to string representation
                                row["Time"] = str(value) if value else "[]"
                            elif key == "Fluorescence":
                                # Convert Fluorescence list to string representation
                                row["Fluorescence"] = str(value) if value else "[]"
                            else:
                                # Handle other experimental conditions
                                if isinstance(value, list):
                                    if len(value) == 0:
                                        row[key] = None
                                    elif len(value) == 1:
                                        row[key] = value[0]
                                    else:
                                        # Multiple values - join with semicolon
                                        row[key] = "; ".join(
                                            str(v) for v in value if v is not None
                                        )
                                else:
                                    row[key] = value

                        all_rows.append(row)
                        all_columns.update(row.keys())

                print(f"  ✓ Processed: {filename}")

            except Exception as e:
                print(f"  ✗ Error processing {json_file}: {e}")
                continue

        if not all_rows:
            print("No data rows collected from JSON files")
            return False

        # Convert to DataFrame
        df = pd.DataFrame(all_rows)

        # Reorder columns for better readability
        priority_columns = [
            "PMID",
            "Figure",
            "Plot",
            "Legend_Symbol",
            "Time",
            "Fluorescence",
            "Protein",
            "Mutation",
            "Temperature",
            "pH",
            "Additives",
        ]

        # Get remaining columns
        remaining_columns = [col for col in df.columns if col not in priority_columns]

        # Final column order
        final_columns = []
        for col in priority_columns:
            if col in df.columns:
                final_columns.append(col)
        final_columns.extend(sorted(remaining_columns))

        # Reorder DataFrame
        df = df[final_columns]

        # Sort by PMID, Figure, Plot, Legend Symbol
        df = df.sort_values(["PMID", "Figure", "Plot", "Legend_Symbol"])

        # Save to CSV
        df.to_csv(output_file, index=False)

        # Calculate some statistics for reporting
        total_data_points = 0
        for _, row in df.iterrows():
            try:
                time_list = (
                    eval(row["Time"]) if row["Time"] and row["Time"] != "[]" else []
                )
                total_data_points += len(time_list)
            except:
                pass

        print(f"\n✓ Consolidated data saved to: {output_file}")
        print(f"  Total rows (experimental conditions): {len(df)}")
        print(f"  Total columns: {len(df.columns)}")
        print(f"  Unique PMIDs: {df['PMID'].nunique()}")
        print(f"  Total data points across all conditions: {total_data_points}")

        return True

    except Exception as e:
        print(f"Error consolidating data to CSV: {e}")
        return False
