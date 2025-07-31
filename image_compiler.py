#!/usr/bin/env python3
"""
Image Compiler for ThT Plot Digitization

This script collects all segmented images that have been identified as ThT plots
and copies them to a single folder with standardized naming for external digitization tools.

The naming convention is: PMID_figureN_plotM.jpg
Example: 18350169_figure3_plot1.jpg

Author: AI Assistant
Date: July 30, 2025
"""

import json
import shutil
import os
from pathlib import Path
from typing import List, Dict, Tuple


def find_tht_identification_files(base_dir: str) -> List[Path]:
    """
    Find all ThT identification JSON files in the articles directory.

    Args:
        base_dir: Base directory containing article data

    Returns:
        List of paths to ThT identification JSON files
    """
    base_path = Path(base_dir)
    tht_files = []

    # Search through all PMID directories
    for pmid_dir in base_path.iterdir():
        if pmid_dir.is_dir() and pmid_dir.name.isdigit():
            # Look for ThT identification files in this PMID directory
            tht_pattern = pmid_dir.glob("*_tht_identification.json")
            tht_files.extend(tht_pattern)

    return sorted(tht_files)


def load_tht_identification(file_path: Path) -> Dict:
    """
    Load ThT identification data from JSON file.

    Args:
        file_path: Path to the ThT identification JSON file

    Returns:
        Dictionary containing ThT identification data
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return {}


def get_tht_plot_images(
    identification_data: Dict, base_dir: str
) -> List[Tuple[str, str, str, int]]:
    """
    Get list of ThT plot images based on identification data.

    Args:
        identification_data: ThT identification data from JSON
        base_dir: Base directory containing article data

    Returns:
        List of tuples (pmid, figure_name, image_path, plot_number)
    """
    tht_images = []

    if not identification_data or "tht_plot_numbers" not in identification_data:
        return tht_images

    pmid = identification_data["pmid"]
    figure_name = identification_data["figure_name"]
    tht_plot_numbers = identification_data["tht_plot_numbers"]

    # Skip if no ThT plots identified
    if not tht_plot_numbers:
        return tht_images

    # Construct paths to segmented images
    segmented_dir = Path(base_dir) / pmid / "segmented_images"

    for plot_num in tht_plot_numbers:
        # Format: figure_3_plot_01.jpg
        segmented_filename = f"{figure_name}_plot_{plot_num:02d}.jpg"
        segmented_path = segmented_dir / segmented_filename

        if segmented_path.exists():
            tht_images.append((pmid, figure_name, str(segmented_path), plot_num))
        else:
            print(f"Warning: Segmented image not found: {segmented_path}")

    return tht_images


def create_standardized_filename(pmid: str, figure_name: str, plot_number: int) -> str:
    """
    Create standardized filename for ThT plot image.

    Args:
        pmid: PubMed ID
        figure_name: Figure name (e.g., "figure_3")
        plot_number: Plot number

    Returns:
        Standardized filename
    """
    # Extract figure number from figure_name (e.g., "figure_3" -> "3")
    figure_num = figure_name.split("_")[1] if "_" in figure_name else figure_name

    # Format: PMID_figureN_plotM.jpg
    return f"{pmid}_figure{figure_num}_plot{plot_number}.jpg"


def compile_tht_images(
    base_dir: str = "articles_data", output_dir: str = "images_for_digitisation"
) -> Dict:
    """
    Compile all ThT plot images into a single directory with standardized naming.

    Args:
        base_dir: Base directory containing article data
        output_dir: Output directory for compiled images

    Returns:
        Dictionary with compilation results and statistics
    """
    result = {
        "success": False,
        "total_files_found": 0,
        "total_tht_plots": 0,
        "images_copied": 0,
        "errors": [],
        "copied_images": [],
        "pmids_processed": set(),
        "figures_processed": set(),
    }

    try:
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        print(f"Created output directory: {output_path.absolute()}")

        # Find all ThT identification files
        tht_files = find_tht_identification_files(base_dir)
        result["total_files_found"] = len(tht_files)
        print(f"Found {len(tht_files)} ThT identification files")

        if not tht_files:
            result["errors"].append("No ThT identification files found")
            return result

        # Process each ThT identification file
        for tht_file in tht_files:
            print(f"\nProcessing: {tht_file}")

            # Load identification data
            identification_data = load_tht_identification(tht_file)
            if not identification_data:
                result["errors"].append(f"Failed to load {tht_file}")
                continue

            pmid = identification_data.get("pmid", "unknown")
            figure_name = identification_data.get("figure_name", "unknown")
            tht_plot_numbers = identification_data.get("tht_plot_numbers", [])

            result["pmids_processed"].add(pmid)
            result["figures_processed"].add(f"{pmid}_{figure_name}")

            print(f"  PMID: {pmid}, Figure: {figure_name}")
            print(f"  ThT plots identified: {tht_plot_numbers}")

            if not tht_plot_numbers:
                print(f"  No ThT plots found in {figure_name}")
                continue

            result["total_tht_plots"] += len(tht_plot_numbers)

            # Get ThT plot images
            tht_images = get_tht_plot_images(identification_data, base_dir)

            # Copy each ThT plot image
            for pmid, figure_name, source_path, plot_num in tht_images:
                try:
                    # Create standardized filename
                    new_filename = create_standardized_filename(
                        pmid, figure_name, plot_num
                    )
                    dest_path = output_path / new_filename

                    # Copy the image
                    shutil.copy2(source_path, dest_path)

                    result["images_copied"] += 1
                    result["copied_images"].append(
                        {
                            "original": source_path,
                            "compiled": str(dest_path),
                            "pmid": pmid,
                            "figure": figure_name,
                            "plot": plot_num,
                        }
                    )

                    print(f"    ✓ Copied: {Path(source_path).name} → {new_filename}")

                except Exception as e:
                    error_msg = f"Failed to copy {source_path} to {dest_path}: {e}"
                    result["errors"].append(error_msg)
                    print(f"    ✗ Error: {error_msg}")

        # Convert sets to lists for JSON serialization
        result["pmids_processed"] = sorted(list(result["pmids_processed"]))
        result["figures_processed"] = sorted(list(result["figures_processed"]))

        # Mark as successful if we copied any images
        if result["images_copied"] > 0:
            result["success"] = True

        # Create compilation report
        report_path = output_path / "compilation_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        print(f"\n" + "=" * 60)
        print(f"COMPILATION COMPLETE")
        print(f"=" * 60)
        print(f"Total ThT identification files found: {result['total_files_found']}")
        print(f"Total ThT plots identified: {result['total_tht_plots']}")
        print(f"Images successfully copied: {result['images_copied']}")
        print(f"PMIDs processed: {len(result['pmids_processed'])}")
        print(f"Figures processed: {len(result['figures_processed'])}")
        print(f"Output directory: {output_path.absolute()}")
        print(f"Compilation report: {report_path}")

        if result["errors"]:
            print(f"\nErrors encountered: {len(result['errors'])}")
            for error in result["errors"][:5]:  # Show first 5 errors
                print(f"  - {error}")
            if len(result["errors"]) > 5:
                print(f"  ... and {len(result['errors']) - 5} more errors")

        print(f"\nExample compiled filenames:")
        for i, copied_img in enumerate(result["copied_images"][:5]):  # Show first 5
            print(f"  {Path(copied_img['compiled']).name}")
        if len(result["copied_images"]) > 5:
            print(f"  ... and {len(result['copied_images']) - 5} more images")

    except Exception as e:
        result["errors"].append(f"Fatal error: {e}")
        print(f"Fatal error during compilation: {e}")

    return result


if __name__ == "__main__":
    """
    Main execution function for compiling ThT plot images.
    """
    print("ThT Plot Image Compiler")
    print("=" * 50)
    print("Collecting all segmented images identified as ThT plots...")
    print("Standardizing filenames for external digitization tools...")
    print()

    # Run the compilation
    results = compile_tht_images(
        base_dir="articles_data", output_dir="images_for_digitisation"
    )

    # Print final status
    if results["success"]:
        print(f"\n✓ Successfully compiled {results['images_copied']} ThT plot images!")
        print(f"  Output folder: images_for_digitisation/")
        print(f"  Ready for external digitization tools.")
    else:
        print(f"\n✗ Compilation failed or no images found.")
        if results["errors"]:
            print("Errors:")
            for error in results["errors"]:
                print(f"  - {error}")
