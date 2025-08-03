import os
import glob
from pathlib import Path
from typing import List, Dict, Any
from utils import process_csv_file


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

    return results


if __name__ == "__main__":
    main()
