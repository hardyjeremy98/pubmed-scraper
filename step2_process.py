import os
import glob
import json
import ast
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple
from config import get_config
from utils import (
    process_csv_file,
    map_csv_data_to_conditions,
    save_mapped_data_to_json,
    consolidate_final_data_to_csv,
)
from llm_input_prep import prepare_csv_and_experimental_data_for_llm
from llm_data_extractor import LLMDataExtractor


def get_csv_files(input_dir: str = "converted_tables") -> List[str]:
    """
    Get all CSV files from the input directory.

    Args:
        input_dir: Directory containing CSV files to process

    Returns:
        List of full paths to CSV files
    """
    csv_pattern = os.path.join(input_dir, "*.csv")
    csv_files = glob.glob(csv_pattern)
    return sorted(csv_files)


def process_all_csv_files(
    input_dir: str = "converted_tables", output_dir: str = "cleaned_tables"
) -> Dict[str, Any]:
    """
    Process all CSV files in the input directory and save cleaned versions.

    Args:
        input_dir: Directory containing CSV files to process
        output_dir: Directory to save cleaned CSV files

    Returns:
        Dictionary containing processing results and statistics
    """
    # Get all CSV files
    csv_files = get_csv_files(input_dir)

    if not csv_files:
        return {
            "success": False,
            "message": f"No CSV files found in {input_dir}",
            "processed_files": [],
            "failed_files": [],
            "statistics": {},
        }

    print(f"Found {len(csv_files)} CSV files to process")

    # Process each file
    processed_files = []
    failed_files = []
    reformatted_count = 0

    for csv_file in csv_files:
        print(f"Processing: {os.path.basename(csv_file)}")

        # Process the file using utils function
        result = process_csv_file(csv_file, output_dir)

        if result["success"]:
            processed_files.append(result)
            if result["was_reformatted"]:
                reformatted_count += 1
            print(f"  ✓ Saved to: {os.path.basename(result['output_file'])}")
            print(f"  Shape: {result['original_shape']} -> {result['cleaned_shape']}")
            if result["was_reformatted"]:
                print(f"  ⚡ Data was reformatted (pivoted)")
        else:
            failed_files.append(result)
            print(f"  ✗ Failed: {result['error']}")

    # Compile statistics
    statistics = {
        "total_files": len(csv_files),
        "successful": len(processed_files),
        "failed": len(failed_files),
        "reformatted": reformatted_count,
        "success_rate": len(processed_files) / len(csv_files) * 100,
    }

    return {
        "success": len(processed_files) > 0,
        "message": f"Processed {len(processed_files)}/{len(csv_files)} files successfully",
        "processed_files": processed_files,
        "failed_files": failed_files,
        "statistics": statistics,
    }


def find_matching_csv_files(
    articles_data_dir: str = "articles_data", cleaned_tables_dir: str = "cleaned_tables"
) -> None:
    """
    Loop through every PMID folder, figure JSON, and plot number to find matching CSVs.
    Process each match individually without storing in a dictionary.

    Args:
        articles_data_dir: Directory containing PMID folders
        cleaned_tables_dir: Directory containing cleaned CSV files
    """
    # Load and validate configuration
    config = get_config()

    # Initialize LLM extractor
    try:
        openai_api_key = config.openai_api_key
        llm_extractor = LLMDataExtractor(openai_api_key)
    except ValueError as e:
        print(f"Warning: {e}. LLM processing will be skipped.")
        llm_extractor = None

    # Get all PMID folders
    pmid_folders = [
        d
        for d in os.listdir(articles_data_dir)
        if os.path.isdir(os.path.join(articles_data_dir, d))
    ]

    print(f"Processing {len(pmid_folders)} PMID folders...")

    for pmid in pmid_folders:
        pmid_path = os.path.join(articles_data_dir, pmid)
        experimental_conditions_path = os.path.join(
            pmid_path, "experimental_conditions"
        )

        # Skip if experimental_conditions folder doesn't exist
        if not os.path.exists(experimental_conditions_path):
            print(f"  Warning: No experimental_conditions folder found for PMID {pmid}")
            continue

        # Get all figure JSON files
        json_files = [
            f for f in os.listdir(experimental_conditions_path) if f.endswith(".json")
        ]

        print(f"  PMID {pmid}: Found {len(json_files)} figure JSONs")

        for json_file in json_files:
            json_path = os.path.join(experimental_conditions_path, json_file)
            figure_name = json_file.replace(".json", "")  # e.g., "figure3"

            try:
                # Load the JSON file
                with open(json_path, "r") as f:
                    figure_data = json.load(f)

                # Loop through each plot number in the JSON
                for plot_number in figure_data.keys():
                    # Construct expected CSV filename pattern
                    csv_pattern = f"{pmid}_{figure_name}_plot{plot_number}_cleaned.csv"
                    csv_path = os.path.join(cleaned_tables_dir, csv_pattern)

                    # Check if the CSV file exists
                    if os.path.exists(csv_path):
                        # Load figure caption from figures.json file
                        figures_json_path = os.path.join(
                            pmid_path, f"{pmid}_figures.json"
                        )
                        figure_caption = ""

                        if os.path.exists(figures_json_path):
                            try:
                                with open(
                                    figures_json_path, "r", encoding="utf-8"
                                ) as f:
                                    figures_data = json.load(f)

                                # Extract figure number from figure_name (e.g., "figure3" -> 3)
                                figure_num_match = re.search(
                                    r"figure(\d+)", figure_name.lower()
                                )
                                if figure_num_match:
                                    figure_num = int(figure_num_match.group(1))

                                    # Find the matching figure in the JSON
                                    for fig in figures_data:
                                        if fig.get("id") == figure_num:
                                            figure_caption = fig.get("caption", "")
                                            break

                            except Exception as e:
                                print(
                                    f"    Warning: Could not load figure caption: {e}"
                                )
                                figure_caption = ""
                        else:
                            print(
                                f"    Warning: No figures.json file found for PMID {pmid}"
                            )

                        # This variable gets overwritten on each iteration
                        current_match = {
                            "pmid": pmid,
                            "figure_name": figure_name,
                            "plot_number": plot_number,
                            "csv_path": csv_path,
                            "experimental_data": figure_data[plot_number],
                            "figure_caption": figure_caption,
                        }

                        # Prepare data for LLM (now including caption)
                        llm_message = prepare_csv_and_experimental_data_for_llm(
                            csv_path,
                            current_match["experimental_data"],
                            figure_caption=None,
                        )

                        # Print the LLM message
                        print("=" * 80)  # Separator between messages
                        print(
                            f"LLM Input Message for {pmid} - {figure_name} - Plot {plot_number}:"
                        )
                        print(llm_message)
                        print("=" * 80)  # Separator between messages

                        # Pass to match_maker LLM and get tuple result
                        if llm_extractor:
                            try:
                                llm_response = llm_extractor.run_model(
                                    text=llm_message,
                                    analysis_type="match_maker",
                                    model="gpt-4o",
                                    temperature=0.1,
                                )

                                # Print the LLM response
                                print("-" * 80)
                                print("LLM Response:")
                                print(llm_response)
                                print("-" * 80)

                                if llm_response["success"]:
                                    # Convert string tuple to actual tuple
                                    response_content = llm_response["content"].strip()
                                    try:
                                        # Look for list/tuple patterns in the response
                                        import re

                                        # Find patterns like [("a", "b")] or [("a", "b"), ("c", "d")]
                                        tuple_pattern = r"\[.*?\]"
                                        matches = re.findall(
                                            tuple_pattern, response_content, re.DOTALL
                                        )

                                        if matches:
                                            # Use the last match (most likely the final answer)
                                            tuple_string = matches[-1]
                                            # Use ast.literal_eval to safely convert string to tuple
                                            match_tuple = ast.literal_eval(tuple_string)
                                            print(f"Match tuple: {match_tuple}")

                                            # Use the match tuple to map CSV data to experimental conditions
                                            if (
                                                match_tuple
                                                and isinstance(match_tuple, list)
                                                and len(match_tuple) > 0
                                            ):
                                                # Create plot title
                                                plot_title = f"{figure_name.replace('figure', 'Figure ')} Plot {plot_number}"

                                                mapped_data = (
                                                    map_csv_data_to_conditions(
                                                        csv_path,
                                                        current_match[
                                                            "experimental_data"
                                                        ],
                                                        match_tuple,
                                                        plot_title,
                                                    )
                                                )

                                                if "error" not in mapped_data:
                                                    # Create output filename for the mapped data JSON
                                                    json_output_filename = f"{pmid}_{figure_name}_plot{plot_number}_mapped_data.json"
                                                    json_output_path = os.path.join(
                                                        "final_data",
                                                        json_output_filename,
                                                    )

                                                    # Save the mapped data to JSON
                                                    success = save_mapped_data_to_json(
                                                        mapped_data, json_output_path
                                                    )

                                                    if success:
                                                        print(
                                                            f"  ✓ Saved mapped data to: {json_output_filename}"
                                                        )

                                                        # Access metadata for reporting
                                                        metadata = mapped_data.get(
                                                            "_metadata", {}
                                                        )
                                                        print(
                                                            f"  Mapped {metadata.get('matched_columns', 0)} columns with {metadata.get('total_data_points', 0)} data points"
                                                        )

                                                        if (
                                                            metadata.get(
                                                                "unmatched_columns", 0
                                                            )
                                                            > 0
                                                        ):
                                                            print(
                                                                f"  ⚠ {metadata['unmatched_columns']} columns remain unmatched: {metadata['unmatched_column_names']}"
                                                            )
                                                    else:
                                                        print(
                                                            f"  ✗ Failed to save mapped data to {json_output_filename}"
                                                        )
                                                else:
                                                    print(
                                                        f"  ✗ Error mapping data: {mapped_data['error']}"
                                                    )
                                            else:
                                                print(
                                                    f"  ⚠ Invalid match tuple format, skipping data mapping"
                                                )

                                        else:
                                            print(f"No tuple pattern found in response")
                                            match_tuple = None
                                    except (ValueError, SyntaxError) as e:
                                        print(
                                            f"Error converting to tuple: {response_content[:100]}..., Error: {e}"
                                        )
                                        match_tuple = None
                                else:
                                    print(f"LLM call failed: {llm_response['error']}")
                                    match_tuple = None

                            except Exception as e:
                                print(f"Error calling LLM: {e}")
                                match_tuple = None
                        else:
                            match_tuple = None

                    else:
                        print(f"    ✗ No CSV found: {csv_pattern}")

            except json.JSONDecodeError as e:
                print(f"    Error reading {json_file}: {e}")
            except Exception as e:
                print(f"    Error processing {json_file}: {e}")


def main():
    """
    Main function to process all CSV files in converted_tables folder.
    """
    print("Table Reformatter - Processing CSV files")
    print("=" * 50)

    # Process all files
    results = process_all_csv_files()

    # Print summary
    print("\n" + "=" * 50)
    print("PROCESSING SUMMARY")
    print("=" * 50)
    print(f"Message: {results['message']}")

    stats = results["statistics"]
    print(f"Total files: {stats['total_files']}")
    print(f"Successful: {stats['successful']}")
    print(f"Failed: {stats['failed']}")
    print(f"Reformatted: {stats['reformatted']}")
    print(f"Success rate: {stats['success_rate']:.1f}%")

    if results["failed_files"]:
        print("\nFAILED FILES:")
        for failed in results["failed_files"]:
            print(f"  - {os.path.basename(failed['input_file'])}: {failed['error']}")

    if results["processed_files"]:
        print(f"\nCleaned files saved to: cleaned_tables/")
        print("Files with reformatting (long->wide format):")
        for processed in results["processed_files"]:
            if processed["was_reformatted"]:
                filename = os.path.basename(processed["input_file"])
                print(f"  - {filename}")

    articles_data_dir = "articles_data"
    cleaned_tables_dir = "cleaned_tables"
    find_matching_csv_files(articles_data_dir, cleaned_tables_dir)

    # Consolidate all mapped data into a single CSV
    print("\n" + "=" * 50)
    print("CONSOLIDATING FINAL DATA")
    print("=" * 50)

    consolidation_success = consolidate_final_data_to_csv()

    if consolidation_success:
        print("Final data consolidation completed successfully!")
    else:
        print("Final data consolidation failed or no data to consolidate.")

    return results


if __name__ == "__main__":
    main()
