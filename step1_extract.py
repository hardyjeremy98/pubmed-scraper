from pubmed_scraper import PubMedClient
from pdf_handler import PageExtractor
from figure_scanner import scan_article_figures_for_keywords
from llm_input_prep import extract_figure_text
from llm_data_extractor import LLMDataExtractor
from image_segmenter import PlotDetector, BoundingBoxLabeler
from utils import merge_constants_and_variables, create_clean_merged_file
from dotenv import load_dotenv
import os
import json
from typing import Dict, List, Optional
from pathlib import Path
import cv2
import numpy as np


def identify_tht_plots(
    pmid: str,
    bbox_image_path: str,
    figure_name: str,
    openai_api_key: str,
    base_dir: str,
) -> Dict:
    """
    Identify ThT fluorescence vs. time plots in a bbox-labeled image using LLM.

    Args:
        pmid: PubMed ID of the article
        bbox_image_path: Path to the bbox-labeled image
        figure_name: Base name of the figure (e.g., "figure_1")
        openai_api_key: OpenAI API key for LLM processing
        base_dir: Base directory where article data is stored

    Returns:
        Dictionary containing ThT plot identification results
    """
    result = {
        "pmid": pmid,
        "figure_name": figure_name,
        "bbox_image_path": bbox_image_path,
        "tht_plot_numbers": [],
        "success": False,
        "error": None,
        "llm_response": None,
    }

    try:
        # Initialize LLM extractor
        extractor = LLMDataExtractor(openai_api_key)

        # Run ThT plot identification
        print(f"    Identifying ThT plots in {figure_name}...")
        llm_result = extractor.run_model(
            text="",  # No text needed for this analysis
            image_path=bbox_image_path,
            analysis_type="ThT_plot_identifier",
            model="gpt-4o",
        )

        result["llm_response"] = llm_result

        if llm_result["success"]:
            try:
                # Parse the response to extract the list of plot numbers
                response_content = llm_result["content"].strip()

                # Handle various response formats
                if response_content.startswith("[") and response_content.endswith("]"):
                    # Try to parse as JSON list
                    import ast

                    tht_plots = ast.literal_eval(response_content)
                    if isinstance(tht_plots, list):
                        result["tht_plot_numbers"] = [
                            int(x) for x in tht_plots if str(x).isdigit()
                        ]
                    else:
                        result["tht_plot_numbers"] = []
                else:
                    # Try to extract numbers from the response
                    import re

                    numbers = re.findall(r"\b\d+\b", response_content)
                    result["tht_plot_numbers"] = [int(x) for x in numbers]

                # Save the ThT identification results
                article_dir = Path(base_dir) / pmid
                tht_results_path = (
                    article_dir / f"{figure_name}_tht_identification.json"
                )

                tht_data = {
                    "pmid": pmid,
                    "figure_name": figure_name,
                    "bbox_image_analyzed": bbox_image_path,
                    "tht_plot_numbers": result["tht_plot_numbers"],
                    "llm_model": llm_result["model"],
                    "llm_usage": llm_result["usage"],
                    "analysis_timestamp": None,  # Could add timestamp if needed
                }

                with open(tht_results_path, "w") as f:
                    json.dump(tht_data, f, indent=2, ensure_ascii=False)

                result["results_file"] = str(tht_results_path)
                result["success"] = True

                print(
                    f"    ✓ ThT identification completed: {len(result['tht_plot_numbers'])} ThT plots found"
                )
                if result["tht_plot_numbers"]:
                    print(f"      ThT plot numbers: {result['tht_plot_numbers']}")
                print(f"    ✓ Saved ThT results: {tht_results_path}")

            except Exception as e:
                result["error"] = f"Error parsing LLM response: {str(e)}"
                print(f"    ✗ Error parsing LLM response: {e}")

        else:
            result["error"] = f"LLM API call failed: {llm_result['error']}"
            print(f"    ✗ LLM API call failed: {llm_result['error']}")

    except Exception as e:
        result["error"] = str(e)
        print(f"    ✗ Error in ThT plot identification: {e}")

    return result


def extract_tht_data(
    pmid: str,
    original_image_path: str,
    figure_name: str,
    tht_plot_numbers: List[int],
    article_content: str,
    figure_data,  # Figure object from utils
    openai_api_key: str,
    base_dir: str,
    segmented_paths: List[str] = None,
) -> Dict:
    """
    Extract experimental data from identified ThT plots using dual LLM extractors.

    Args:
        pmid: PubMed ID of the article
        original_image_path: Path to the original figure image (fallback)
        figure_name: Base name of the figure (e.g., "figure_1")
        tht_plot_numbers: List of ThT plot numbers identified
        article_content: Full article text content
        figure_data: Figure object with metadata (from utils.py)
        openai_api_key: OpenAI API key for LLM processing
        base_dir: Base directory where article data is stored
        segmented_paths: List of paths to segmented plot images

    Returns:
        Dictionary containing data extraction results from both extractors
    """
    result = {
        "pmid": pmid,
        "figure_name": figure_name,
        "original_image_path": original_image_path,
        "segmented_paths": segmented_paths,
        "tht_plot_numbers": tht_plot_numbers,
        "extracted_data": {},
        "success": False,
        "error": None,
        "llm_responses": [],
    }

    if not tht_plot_numbers:
        result["error"] = "No ThT plots to extract data from"
        return result

    def _parse_json_response(response_content: str) -> dict:
        """Helper function to parse JSON from LLM response, handling markdown code blocks."""
        response_content = response_content.strip()

        # Handle markdown code blocks
        if response_content.startswith("```json"):
            lines = response_content.split("\n")
            json_lines = []
            in_json_block = False

            for line in lines:
                if line.strip() == "```json":
                    in_json_block = True
                    continue
                elif line.strip() == "```" and in_json_block:
                    break
                elif in_json_block:
                    json_lines.append(line)

            response_content = "\n".join(json_lines)
        elif response_content.startswith("```"):
            # Handle other code block formats
            lines = response_content.split("\n")
            json_lines = []
            in_code_block = False

            for line in lines:
                if line.strip().startswith("```"):
                    if not in_code_block:
                        in_code_block = True
                        continue
                    else:
                        break
                elif in_code_block:
                    json_lines.append(line)

            response_content = "\n".join(json_lines)

        return json.loads(response_content)

    try:
        # Initialize LLM extractor
        extractor = LLMDataExtractor(openai_api_key)

        # Process each ThT plot individually with dual extractors
        print(
            f"    Extracting data from ThT plots {tht_plot_numbers} in {figure_name}..."
        )

        all_extracted_data = {}
        successful_extractions = 0

        for plot_num in tht_plot_numbers:
            print(f"      Processing plot {plot_num}...")

            # Determine which image to use for this plot
            image_to_use = original_image_path  # Default fallback

            if segmented_paths and len(segmented_paths) >= plot_num:
                # Use segmented image if available
                # Plot numbers are 1-indexed, but list is 0-indexed
                segmented_image_path = segmented_paths[plot_num - 1]
                image_to_use = segmented_image_path
                print(f"        Using segmented image: {segmented_image_path}")
            else:
                print(
                    f"        Using original image (segmented not available): {original_image_path}"
                )

            # Use the existing extract_figure_text function properly
            try:
                # Extract figure number from figure_name (e.g., "figure_3" -> 3)
                fig_num = int(figure_name.split("_")[1])

                # Use extract_figure_text to get caption only (for variables_extractor)
                caption_text, _ = extract_figure_text(
                    article_text=article_content,
                    figure_number=fig_num,
                    figure_data=figure_data,
                    pmid=pmid,
                    base_dir=base_dir,
                    include_surrounding_text=False,  # Caption only
                )

                # Print caption
                print(
                    f"        Caption text: {caption_text[:100]}..."
                )  # Show first 100 chars

                # Use extract_figure_text to get caption + context (for constants_extractor)
                context_text, _ = extract_figure_text(
                    article_text=article_content,
                    figure_number=fig_num,
                    figure_data=figure_data,
                    pmid=pmid,
                    base_dir=base_dir,
                    include_surrounding_text=True,  # Caption + surrounding text
                    context_words=250,
                )

                print(
                    f"        Context text: {context_text[:100]}..."
                )  # Show first 100 chars

                print(f"        Caption text length: {len(caption_text)} chars")
                print(f"        Context text length: {len(context_text)} chars")

            except Exception as e:
                print(f"        ⚠ Text extraction failed: {e}")
                # Simple fallback - use figure caption directly
                caption_text = figure_data.caption if figure_data.caption else ""
                context_text = caption_text

            # Run variables_extractor (image + caption only)
            print(f"        Running variables extractor...")
            variables_result = extractor.run_model(
                text=caption_text,
                analysis_type="variables_extractor",
                model="gpt-4o",
                tht_plot_list=[plot_num],
            )

            # Run constants_extractor (no image, caption + context)
            print(f"        Running constants extractor...")
            constants_result = extractor.run_model(
                text=context_text,
                image_path=None,  # No image for constants extractor
                analysis_type="constants_extractor",
                model="gpt-4o",
                tht_plot_list=[plot_num],
            )

            # Store both results
            result["llm_responses"].append(
                {
                    "plot_number": plot_num,
                    "variables_result": variables_result,
                    "constants_result": constants_result,
                }
            )

            # Process variables extractor results
            variables_data = None
            if variables_result["success"]:
                try:
                    variables_data = _parse_json_response(variables_result["content"])
                    print(
                        f"        ✓ Variables extraction successful for plot {plot_num}"
                    )
                except (json.JSONDecodeError, Exception) as e:
                    print(
                        f"        ✗ Variables extraction parsing error for plot {plot_num}: {e}"
                    )

            # Process constants extractor results
            constants_data = None
            if constants_result["success"]:
                try:
                    constants_data = _parse_json_response(constants_result["content"])
                    print(
                        f"        ✓ Constants extraction successful for plot {plot_num}"
                    )
                except (json.JSONDecodeError, Exception) as e:
                    print(
                        f"        ✗ Constants extraction parsing error for plot {plot_num}: {e}"
                    )

            # Store the extracted data under the plot number
            if variables_data or constants_data:
                all_extracted_data[str(plot_num)] = {
                    "variables": variables_data,
                    "constants": constants_data,
                }
                successful_extractions += 1
                print(f"        ✓ Successfully processed plot {plot_num}")
            else:
                print(f"        ✗ Both extractors failed for plot {plot_num}")

        # Set the final results
        result["extracted_data"] = all_extracted_data

        # Always save the data extraction results, even if all extractions failed
        article_dir = Path(base_dir) / pmid
        data_results_path = article_dir / f"{figure_name}_tht_data_extraction.json"

        extraction_data = {
            "pmid": pmid,
            "figure_name": figure_name,
            "original_image_analyzed": original_image_path,
            "segmented_paths_used": segmented_paths,
            "tht_plot_numbers": tht_plot_numbers,
            "caption_text_used": caption_text if "caption_text" in locals() else "N/A",
            "context_text_used": context_text if "context_text" in locals() else "N/A",
            "extracted_data": result["extracted_data"],
            "successful_extractions": successful_extractions,
            "total_plots_attempted": len(tht_plot_numbers),
            "llm_responses": result["llm_responses"],
            "analysis_timestamp": None,
        }

        with open(data_results_path, "w") as f:
            json.dump(extraction_data, f, indent=2, ensure_ascii=False)

        result["results_file"] = str(data_results_path)

        # Create clean merged conditions file if extraction was successful
        if successful_extractions > 0 and result["extracted_data"]:
            try:
                # Generate merged file path
                merged_file_path = article_dir / f"{figure_name}_tht_data_merged.json"

                # Create clean merged file with only experimental conditions
                merged_result = merge_constants_and_variables(result["extracted_data"])

                with open(merged_file_path, "w") as f:
                    json.dump(merged_result, f, indent=2, ensure_ascii=False)

                result["merged_file"] = str(merged_file_path)

                # Count total conditions
                total_conditions = sum(
                    len(plot_data.get("total_conditions", {}))
                    for plot_data in merged_result.values()
                )

                print(f"    ✓ Created clean merged conditions file: {merged_file_path}")
                print(f"      Total experimental conditions: {total_conditions}")

            except Exception as e:
                print(f"    ⚠ Failed to create merged conditions file: {e}")
                # Don't fail the whole extraction if merger fails
                result["merger_error"] = str(e)

        # Consider successful if we extracted data for at least one plot
        if successful_extractions > 0:
            result["success"] = True
            print(
                f"    ✓ Dual extraction completed: {successful_extractions}/{len(tht_plot_numbers)} plots processed successfully"
            )
            print(f"    ✓ Saved extraction results: {data_results_path}")

            # Report merged file if created
            if "merged_file" in result:
                print(f"    ✓ Saved clean merged conditions: {result['merged_file']}")
            elif "merger_error" in result:
                print(
                    f"    ⚠ Merged conditions file creation failed: {result['merger_error']}"
                )
        else:
            result["error"] = (
                f"Failed to extract data from any of the {len(tht_plot_numbers)} ThT plots"
            )
            print(f"    ✗ Dual extraction failed for all {len(tht_plot_numbers)} plots")
            print(
                f"    ✓ Saved failed extraction results for debugging: {data_results_path}"
            )

    except Exception as e:
        result["error"] = str(e)
        print(f"    ✗ Error in ThT data extraction: {e}")

    return result


def segment_plots_from_image(
    image_path: str, detections: List[Dict], segmented_dir: Path, figure_name: str
) -> List[str]:
    """
    Segment individual plots from the original image using bounding boxes.

    Args:
        image_path: Path to the original image
        detections: List of detection results with bounding boxes
        segmented_dir: Directory to save segmented images
        figure_name: Base name of the figure (e.g., "figure_1")

    Returns:
        List of paths to saved segmented images
    """
    segmented_paths = []

    try:
        # Load the original image
        image = cv2.imread(image_path)
        if image is None:
            print(f"    ✗ Could not load image: {image_path}")
            return segmented_paths

        # Sort detections by confidence (highest first) for consistent numbering
        sorted_detections = sorted(
            detections, key=lambda x: x["confidence"], reverse=True
        )

        # Re-sort by position (same as in BoundingBoxLabeler) for consistent labeling
        # Sort by y-coordinate first (top to bottom), then by x-coordinate (left to right)
        image_height = image.shape[0]
        row_tolerance = int(image_height * 0.10)  # 10% tolerance for row grouping

        # Group boxes by rows based on their center y-coordinates
        rows = []
        for detection in sorted_detections:
            bbox = detection["bbox"]
            center_y = (bbox[1] + bbox[3]) / 2

            # Find if this box belongs to an existing row
            placed = False
            for row in rows:
                row_center_y = sum(
                    (b["bbox"][1] + b["bbox"][3]) / 2 for b in row
                ) / len(row)
                if abs(center_y - row_center_y) <= row_tolerance:
                    row.append(detection)
                    placed = True
                    break

            if not placed:
                rows.append([detection])

        # Sort rows by average y-coordinate (top to bottom)
        rows.sort(
            key=lambda row: sum((b["bbox"][1] + b["bbox"][3]) / 2 for b in row)
            / len(row)
        )

        # Sort boxes within each row by x-coordinate (left to right)
        for row in rows:
            row.sort(key=lambda b: (b["bbox"][0] + b["bbox"][2]) / 2)

        # Flatten the sorted detections
        position_sorted_detections = []
        for row in rows:
            position_sorted_detections.extend(row)

        # Extract each plot
        for i, detection in enumerate(position_sorted_detections, 1):
            bbox = detection["bbox"]
            x1, y1, x2, y2 = bbox

            # Ensure coordinates are within image bounds
            x1 = max(0, int(x1))
            y1 = max(0, int(y1))
            x2 = min(image.shape[1], int(x2))
            y2 = min(image.shape[0], int(y2))

            # Extract the plot region
            plot_image = image[y1:y2, x1:x2]

            # Skip if the cropped region is too small
            if plot_image.shape[0] < 10 or plot_image.shape[1] < 10:
                print(f"    ⚠ Skipped plot {i}: region too small ({plot_image.shape})")
                continue

            # Create filename for the segmented plot
            segment_filename = f"{figure_name}_plot_{i:02d}.jpg"
            segment_path = segmented_dir / segment_filename

            # Save the segmented plot
            success = cv2.imwrite(str(segment_path), plot_image)
            if success:
                segmented_paths.append(str(segment_path))
                print(
                    f"    ✓ Saved plot {i}: {segment_filename} ({plot_image.shape[1]}x{plot_image.shape[0]})"
                )
            else:
                print(f"    ✗ Failed to save plot {i}: {segment_filename}")

    except Exception as e:
        print(f"    ✗ Error segmenting plots from {image_path}: {e}")

    return segmented_paths


def process_figure_bounding_boxes(
    pmid: str,
    base_dir: str,
    openai_api_key: str = None,
    figure_contexts: List[Dict] = None,
    article_content: str = None,
    figure_map: Dict = None,
) -> Dict:
    """
    Process downloaded figure images to detect plots and create bounding box data.

    Args:
        pmid: PubMed ID of the article
        base_dir: Base directory where article data is stored
        openai_api_key: OpenAI API key for LLM processing (optional)
        figure_contexts: List of figure context data with text and metadata (optional)
        article_content: Full article text content (optional, for extract_figure_text)
        figure_map: Mapping of figure numbers to Figure objects (optional, for extract_figure_text)

    Returns:
        Dictionary containing bounding box processing results
    """
    result = {
        "pmid": pmid,
        "processed_figures": {},
        "total_plots_detected": 0,
        "success": False,
        "error": None,
    }

    try:
        # Initialize plot detector with high confidence threshold
        detector = PlotDetector(
            model_path="plot_detector1.pt", confidence_threshold=0.75, device="auto"
        )

        # Path to article directory
        article_dir = Path(base_dir) / pmid
        if not article_dir.exists():
            result["error"] = f"Article directory not found: {article_dir}"
            return result

        # Create bbox subdirectory
        bbox_dir = article_dir / "bbox"
        bbox_dir.mkdir(exist_ok=True)

        # Create segmented_images subdirectory
        segmented_dir = article_dir / "segmented_images"
        segmented_dir.mkdir(exist_ok=True)

        # Initialize labeler for this article
        labeler = BoundingBoxLabeler(str(bbox_dir))

        # Find all figure images in the article directory and images subdirectory
        image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"]
        figure_images = []

        # Check both the article directory and the images subdirectory
        search_paths = [article_dir, article_dir / "images"]

        for search_path in search_paths:
            if search_path.exists():
                for ext in image_extensions:
                    figure_images.extend(search_path.glob(f"figure_*{ext}"))
                    figure_images.extend(search_path.glob(f"figure_*{ext.upper()}"))

        if not figure_images:
            result["error"] = "No figure images found to process"
            return result

        print(f"Found {len(figure_images)} figure images to process for PMID {pmid}")

        # Process each figure image
        for image_path in sorted(figure_images):
            figure_name = image_path.stem  # e.g., "figure_1"
            print(f"  Processing {figure_name}...")

            try:
                # Detect plots in the image
                detections = detector.detect_plots(str(image_path))
                num_plots = len(detections)

                if num_plots == 0:
                    print(f"    No plots detected in {figure_name}")
                    result["processed_figures"][figure_name] = {
                        "image_path": str(image_path),
                        "plots_detected": 0,
                        "bbox_image": None,
                        "bbox_data": None,
                    }
                    continue

                print(f"    Detected {num_plots} plots in {figure_name}")

                # Create labeled bounding box image
                labeled_path, labeled_boxes = labeler.label_and_save(
                    str(image_path),
                    detections,
                    filename=f"{figure_name}_bbox_labeled.jpg",
                    label_position="top_right",
                    thickness=2,
                    font_scale=1.0,
                    show_confidence=False,  # Only show label numbers
                    show_class_names=False,
                    row_tolerance_percent=10.0,  # Reasonable tolerance for row grouping
                )

                # Create reference JSON for this figure
                reference = labeler.create_label_reference(labeled_boxes, figure_name)
                reference_path = bbox_dir / f"{figure_name}_bbox_reference.json"

                with open(reference_path, "w") as f:
                    json.dump(reference, f, indent=2)

                # Segment individual plots from the original image first
                print(f"    Segmenting {num_plots} plots from {figure_name}...")
                segmented_paths = segment_plots_from_image(
                    str(image_path), detections, segmented_dir, figure_name
                )

                # Identify ThT plots in the bbox-labeled image if API key is provided
                tht_result = None
                data_extraction_result = None
                if openai_api_key:
                    tht_result = identify_tht_plots(
                        pmid, labeled_path, figure_name, openai_api_key, base_dir
                    )

                    # Extract data from identified ThT plots if any were found
                    if (
                        tht_result
                        and tht_result["success"]
                        and tht_result["tht_plot_numbers"]
                    ):
                        # Get the figure data for this figure
                        figure_data = None
                        if figure_map:
                            try:
                                fig_num = int(figure_name.split("_")[1])
                                figure_data = figure_map.get(fig_num)
                            except (IndexError, ValueError):
                                print(
                                    f"    ⚠ Could not extract figure number from {figure_name}"
                                )

                        # Use the refactored extract_tht_data function if we have the required data
                        if article_content and figure_data:
                            data_extraction_result = extract_tht_data(
                                pmid,
                                str(image_path),  # Original image for fallback
                                figure_name,
                                tht_result["tht_plot_numbers"],
                                article_content,  # Full article text
                                figure_data,  # Figure object
                                openai_api_key,
                                base_dir,
                                segmented_paths,  # Pass segmented plot paths
                            )
                        else:
                            print(
                                f"    ⚠ Missing article content or figure data for {figure_name}, skipping data extraction"
                            )

                # Store results
                result["processed_figures"][figure_name] = {
                    "image_path": str(image_path),
                    "plots_detected": num_plots,
                    "bbox_image": labeled_path,
                    "bbox_data": str(reference_path),
                    "segmented_plots": segmented_paths,
                    "tht_identification": tht_result,
                    "tht_data_extraction": data_extraction_result,
                    "detections": detections,
                    "labeled_boxes": labeled_boxes,
                }

                result["total_plots_detected"] += num_plots
                print(f"    ✓ Created bbox image: {labeled_path}")
                print(f"    ✓ Created bbox data: {reference_path}")
                print(f"    ✓ Created {len(segmented_paths)} segmented plot images")

            except Exception as e:
                print(f"    ✗ Error processing {figure_name}: {e}")
                result["processed_figures"][figure_name] = {
                    "image_path": str(image_path),
                    "error": str(e),
                }

        # Create summary JSON for all figures in this article
        summary_path = bbox_dir / f"{pmid}_bbox_summary.json"

        # Count total segmented plots
        total_segmented_plots = 0
        for fig_data in result["processed_figures"].values():
            if "segmented_plots" in fig_data:
                total_segmented_plots += len(fig_data["segmented_plots"])

        summary_data = {
            "pmid": pmid,
            "total_figures_processed": len(figure_images),
            "total_plots_detected": result["total_plots_detected"],
            "total_segmented_plots": total_segmented_plots,
            "figures": result["processed_figures"],
        }

        with open(summary_path, "w") as f:
            json.dump(summary_data, f, indent=2)

        result["summary_file"] = str(summary_path)
        result["total_segmented_plots"] = total_segmented_plots
        result["success"] = True

        print(
            f"  ✓ Processed {len(figure_images)} figures, detected {result['total_plots_detected']} total plots"
        )
        print(f"  ✓ Created {total_segmented_plots} segmented plot images")
        print(f"  ✓ Created summary: {summary_path}")

    except Exception as e:
        result["error"] = str(e)
        print(f"  ✗ Error in bounding box processing: {e}")

    return result


def process_article_figures_and_pages(
    pmid: str, pubmed_client: PubMedClient, openai_api_key: str = None
) -> Dict:
    """
    Process an article to extract relevant figures and their corresponding PDF pages.

    Args:
        pmid: PubMed ID of the article
        pubmed_client: Initialized PubMedClient instance
        openai_api_key: OpenAI API key for LLM processing (optional)

    Returns:
        Dictionary containing article data, relevant figures, and their page locations
    """
    result = {
        "pmid": pmid,
        "article": None,
        "relevant_figures": {},
        "success": False,
        "error": None,
    }

    try:
        # Get article metadata and content
        article = pubmed_client.get_full_article(pmid)
        result["article"] = {
            "title": article.title,
            "pmcid": article.pmcid,
            "content": article.content,
        }

        # Save article metadata
        pubmed_client.save_article_to_json(article)

        # Save HTML content if available
        if article.html_content:
            pubmed_client.save_article_html(article.html_content, pmid)

        # Extract figures from the article
        figures = pubmed_client.get_article_figures(pmid)

        if not figures:
            result["error"] = "No figures found for this article"
            return result

        # Save figures metadata
        pubmed_client.save_figures_to_json(figures, pmid)

        # Scan figures for relevant keywords
        scan_result = scan_article_figures_for_keywords(
            figures, pmid=pmid, verbose=False
        )

        if not scan_result.has_relevant_figures:
            result["error"] = "No relevant figures found based on keyword scan"
            return result

        # Download relevant figures
        for i, figure in enumerate(figures):
            figure_number = i + 1
            if figure_number in scan_result.relevant_figure_numbers:
                # Store relevant figure info
                result["relevant_figures"][figure_number] = {
                    "url": figure.url,
                    "alt_text": figure.alt,
                    "caption": figure.caption,
                    "keywords_found": scan_result.keyword_matches[figure_number],
                }

                # Download the figure image
                if figure.url:
                    file_ext = (
                        figure.url.split(".")[-1]
                        if "." in figure.url.split("/")[-1]
                        else "jpg"
                    )
                    filename = f"figure_{figure_number}.{file_ext}"
                    pubmed_client.download_image(figure.url, filename, pmid)

        # Extract text context for LLM input preparation
        if article.content and result["relevant_figures"]:
            try:
                figure_contexts = []

                # Create a mapping of figure numbers to Figure objects
                figure_map = {i + 1: fig for i, fig in enumerate(figures)}

                # Process each relevant figure individually
                for fig_num in result["relevant_figures"].keys():
                    if fig_num in figure_map:
                        figure_data = figure_map[fig_num]

                        # Extract context for this figure with flexible options
                        # You can modify these parameters as needed:
                        # - include_surrounding_text=True: includes surrounding text + caption
                        # - include_surrounding_text=False: only caption
                        # - context_words: amount of surrounding text (ignored if include_surrounding_text=False)
                        combined_text, image_path = extract_figure_text(
                            article_text=article.content,
                            figure_number=fig_num,
                            figure_data=figure_data,
                            pmid=pmid,
                            base_dir=pubmed_client.base_dir,
                            include_surrounding_text=True,  # Set to False for caption only
                            context_words=250,  # Adjust number of surrounding words
                        )

                        figure_contexts.append(
                            {
                                "figure_number": fig_num,
                                "text": combined_text,
                                "image_path": image_path,
                                "keywords": result["relevant_figures"][fig_num][
                                    "keywords_found"
                                ],
                            }
                        )

                result["figure_contexts"] = figure_contexts
                print(f"Extracted contexts for {len(figure_contexts)} figures")

            except Exception as e:
                print(f"Warning: Could not prepare LLM context for {pmid}: {e}")

        # Process figures with image segmentation to detect plots
        if result["relevant_figures"]:
            print(f"Processing bounding boxes for {pmid}...")
            figure_contexts_to_pass = result.get("figure_contexts", None)
            bbox_result = process_figure_bounding_boxes(
                pmid,
                pubmed_client.base_dir,
                openai_api_key,
                figure_contexts_to_pass,
                article.content,  # Pass article content
                figure_map,  # Pass figure mapping
            )
            result["bbox_processing"] = bbox_result

            if bbox_result["success"]:
                print(f"✓ Bounding box processing completed for {pmid}")
                print(f"  Total plots detected: {bbox_result['total_plots_detected']}")
            else:
                print(
                    f"✗ Bounding box processing failed for {pmid}: {bbox_result['error']}"
                )

        result["success"] = True

    except Exception as e:
        result["error"] = str(e)

    return result


def main():
    """Main function to process literature articles."""
    # Load environment variables
    load_dotenv()

    email = os.getenv("EMAIL")
    if not email:
        raise ValueError(
            "EMAIL environment variable is not set. Please set it in your .env file."
        )

    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError(
            "OPENAI_API_KEY environment variable is not set. Please set it in your .env file."
        )

    # Initialize client
    pubmed_client = PubMedClient(email=email, base_dir="articles_data")

    # Load PMIDs from JSON file
    try:
        with open("unique_pmids.json", "r") as f:
            pmids = json.load(f)
        print(f"Loaded {len(pmids)} PMIDs from unique_pmids.json")
    except FileNotFoundError:
        print("Error: unique_pmids.json file not found")
        return []
    except json.JSONDecodeError:
        print("Error: Invalid JSON format in unique_pmids.json")
        return []

    pmids = ["19258323", "18350169"]
    processed_articles = []

    for pmid in pmids:
        print(f"Processing PMID: {pmid}")

        result = process_article_figures_and_pages(pmid, pubmed_client, openai_api_key)
        processed_articles.append(result)

        # Print all keys in the result dictionary
        print(f"Result keys: {list(result.keys())}")

        if result["success"]:
            print(f"✓ Successfully processed {pmid}")
            print(f"  Title: {result['article']['title']}")
            print(f"  Relevant figures: {list(result['relevant_figures'].keys())}")

            # Display figure context info if available
            if "figure_contexts" in result:
                print(f"  Figure contexts extracted: {len(result['figure_contexts'])}")
                for ctx in result["figure_contexts"]:
                    print(
                        f"    Figure {ctx['figure_number']}: {len(ctx['text'])} chars, {ctx['image_path']}"
                    )

            # Display bounding box processing info if available
            if "bbox_processing" in result and result["bbox_processing"]["success"]:
                bbox_data = result["bbox_processing"]
                print(f"  Bounding box processing: ✓ Success")
                print(f"    Total plots detected: {bbox_data['total_plots_detected']}")
                if "total_segmented_plots" in bbox_data:
                    print(
                        f"    Total segmented plots: {bbox_data['total_segmented_plots']}"
                    )
                print(f"    Figures processed: {len(bbox_data['processed_figures'])}")

                # Show details for each processed figure
                for fig_name, fig_data in bbox_data["processed_figures"].items():
                    if "plots_detected" in fig_data and fig_data["plots_detected"] > 0:
                        segmented_count = len(fig_data.get("segmented_plots", []))

                        # Display ThT identification results if available
                        tht_info = ""
                        data_extraction_info = ""

                        if (
                            "tht_identification" in fig_data
                            and fig_data["tht_identification"]
                        ):
                            tht_result = fig_data["tht_identification"]
                            if tht_result["success"] and tht_result["tht_plot_numbers"]:
                                tht_info = (
                                    f" | ThT plots: {tht_result['tht_plot_numbers']}"
                                )

                                # Add data extraction info if available
                                if (
                                    "tht_data_extraction" in fig_data
                                    and fig_data["tht_data_extraction"]
                                ):
                                    data_result = fig_data["tht_data_extraction"]
                                    if (
                                        data_result["success"]
                                        and data_result["extracted_data"]
                                    ):
                                        extracted_plots = list(
                                            data_result["extracted_data"].keys()
                                        )
                                        data_extraction_info = f" | Data extracted for plots: {extracted_plots}"

                                        # Add merged file info if available
                                        if "merged_file" in data_result:
                                            data_extraction_info += (
                                                " | Merged conditions file created"
                                            )
                                        elif "merger_error" in data_result:
                                            data_extraction_info += " | Merger failed"

                                    elif data_result["success"]:
                                        data_extraction_info = (
                                            " | Data extraction: no data found"
                                        )
                                    else:
                                        data_extraction_info = f" | Data extraction failed: {data_result['error']}"

                            elif tht_result["success"]:
                                tht_info = " | ThT plots: none"
                            else:
                                tht_info = (
                                    f" | ThT analysis failed: {tht_result['error']}"
                                )

                        print(
                            f"      {fig_name}: {fig_data['plots_detected']} plots → {segmented_count} segmented{tht_info}{data_extraction_info} → {fig_data['bbox_image']}"
                        )
                    elif "error" in fig_data:
                        print(f"      {fig_name}: Error - {fig_data['error']}")
                    else:
                        print(f"      {fig_name}: No plots detected")
            elif "bbox_processing" in result:
                print(
                    f"  Bounding box processing: ✗ Failed - {result['bbox_processing']['error']}"
                )
        else:
            print(f"✗ Failed to process {pmid}: {result['error']}")

        print("-" * 60)

    return processed_articles


if __name__ == "__main__":
    results = main()
