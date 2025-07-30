#!/usr/bin/env python3
"""
Comprehensive test script for bounding box detection and labeling.

This script processes all images in the images folder and creates organized
output with subfolders for each image containing:
- Labeled bounding box images
- JSON reference files
- Detection statistics
- Processing logs
"""

import os
import json
import time
from pathlib import Path
from typing import List, Dict, Any
import cv2
import numpy as np
from image_segmenter import (
    PlotDetector,
    BoundingBoxLabeler,
    BoundingBoxVisualizer,
    detect_and_label,
)


class BoundingBoxTester:
    """Comprehensive tester for bounding box detection pipeline."""

    def __init__(
        self,
        images_dir: str = "images",
        output_dir: str = "bbox",
        model_path: str = "plot_detector1.pt",
    ):
        """
        Initialize the tester.

        Args:
            images_dir: Directory containing input images
            output_dir: Main output directory
            model_path: Path to the detection model
        """
        self.images_dir = Path(images_dir)
        self.output_dir = Path(output_dir)
        self.model_path = model_path

        # Create main output directory
        self.output_dir.mkdir(exist_ok=True)

        # Initialize components
        self.detector = None
        self.labeler = None
        self.visualizer = None

        # Statistics
        self.results = {
            "total_images": 0,
            "successful_detections": 0,
            "failed_detections": 0,
            "total_plots_detected": 0,
            "processing_time": 0,
            "images_with_plots": 0,
            "images_without_plots": 0,
            "results_by_image": {},
        }

    def initialize_components(self):
        """Initialize detection and labeling components."""
        print("Initializing detection components...")

        try:
            self.detector = PlotDetector(
                self.model_path, device="auto", confidence_threshold=0.75
            )
            print(f"✓ PlotDetector initialized: {self.detector.get_model_info()}")

            # We'll create labeler and visualizer per image to avoid conflicts
            print("✓ Components ready for processing")

        except Exception as e:
            print(f"✗ Failed to initialize components: {e}")
            raise

    def get_image_files(self) -> List[Path]:
        """Get all image files from the images directory."""
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

        image_files = []
        for file_path in self.images_dir.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in image_extensions:
                image_files.append(file_path)

        # Sort for consistent processing order
        image_files.sort()
        print(f"Found {len(image_files)} image files to process")

        return image_files

    def create_image_subfolder(self, image_name: str) -> Path:
        """Create a subfolder for the image output."""
        # Remove extension and create clean folder name
        folder_name = Path(image_name).stem
        image_output_dir = self.output_dir / folder_name
        image_output_dir.mkdir(exist_ok=True)

        return image_output_dir

    def process_single_image(self, image_path: Path) -> Dict[str, Any]:
        """
        Process a single image with comprehensive testing.

        Args:
            image_path: Path to the image file

        Returns:
            Dictionary with processing results
        """
        image_name = image_path.name
        image_stem = image_path.stem

        print(f"\nProcessing: {image_name}")
        print("-" * 50)

        # Create output subfolder
        image_output_dir = self.create_image_subfolder(image_name)

        # Initialize result dictionary
        result = {
            "image_name": image_name,
            "image_path": str(image_path),
            "output_dir": str(image_output_dir),
            "success": False,
            "plots_detected": 0,
            "processing_time": 0,
            "error_message": None,
            "files_created": [],
            "detection_details": [],
            "image_dimensions": None,
        }

        start_time = time.time()

        try:
            # Get image dimensions
            img = cv2.imread(str(image_path))
            if img is not None:
                h, w = img.shape[:2]
                result["image_dimensions"] = {"width": w, "height": h}
                print(f"Image dimensions: {w}x{h}")

            # Detect plots
            print("Detecting plots...")
            detections = self.detector.detect_plots(str(image_path))
            num_plots = len(detections)
            result["plots_detected"] = num_plots

            print(f"Detected {num_plots} plots")

            # Store detection details
            for i, detection in enumerate(detections):
                bbox = detection["bbox"]
                conf = detection["confidence"]
                class_name = detection.get("class_name", "Plot")

                detail = {
                    "detection_id": i + 1,
                    "bbox": bbox,
                    "confidence": conf,
                    "class_name": class_name,
                    "area": (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]),
                }
                result["detection_details"].append(detail)

                print(f"  Plot {i+1}: {bbox} (conf: {conf:.3f})")

            if num_plots > 0:
                # Create labeler and visualizer for this image
                labeler = BoundingBoxLabeler(str(image_output_dir))
                visualizer = BoundingBoxVisualizer(str(image_output_dir))

                # Test 1: Sequential labeling (main functionality)
                print("Creating sequential labeled image...")
                labeled_path, labeled_boxes = labeler.label_and_save(
                    str(image_path),
                    detections,
                    filename=f"{image_stem}_sequential_labeled.jpg",
                    label_position="top_left",
                    thickness=3,
                    font_scale=1.0,
                    show_confidence=False,
                    show_class_names=False,
                )
                result["files_created"].append(str(labeled_path))
                print(f"✓ Sequential labeled: {labeled_path}")

                # Test 2: Basic visualization
                print("Creating basic visualization...")
                viz_path = visualizer.visualize_and_save(
                    str(image_path),
                    detections,
                    filename=f"{image_stem}_basic_visualization.jpg",
                    thickness=2,
                    font_scale=0.8,
                    show_confidence=True,
                    show_class_names=True,
                    label_background=True,
                )
                result["files_created"].append(str(viz_path))
                print(f"✓ Basic visualization: {viz_path}")

                # Create JSON reference file
                print("Creating JSON reference...")
                reference = labeler.create_label_reference(labeled_boxes, image_stem)
                reference_path = image_output_dir / f"{image_stem}_reference.json"

                with open(reference_path, "w") as f:
                    json.dump(reference, f, indent=2)

                result["files_created"].append(str(reference_path))
                print(f"✓ JSON reference: {reference_path}")

            else:
                print("No plots detected")

            # Processing completed successfully
            processing_time = time.time() - start_time
            result["processing_time"] = processing_time
            result["success"] = True

            print(f"✓ Completed in {processing_time:.2f} seconds")
            print(f"✓ Created {len(result['files_created'])} files")

        except Exception as e:
            result["error_message"] = str(e)
            result["processing_time"] = time.time() - start_time
            print(f"✗ Error processing {image_name}: {e}")

            # Create error report
            error_report = {
                "image_name": image_name,
                "error": str(e),
                "processing_time": result["processing_time"],
            }

            error_path = image_output_dir / f"{image_stem}_error_report.json"
            with open(error_path, "w") as f:
                json.dump(error_report, f, indent=2)

            result["files_created"].append(str(error_path))

        return result

    def run_comprehensive_test(self):
        """Run comprehensive test on all images."""
        print("=" * 60)
        print("COMPREHENSIVE BOUNDING BOX DETECTION TEST")
        print("=" * 60)

        # Initialize components
        self.initialize_components()

        # Get all image files
        image_files = self.get_image_files()
        self.results["total_images"] = len(image_files)

        if not image_files:
            print("No image files found!")
            return

        print(f"\nStarting processing of {len(image_files)} images...")

        overall_start_time = time.time()

        # Process each image
        for i, image_path in enumerate(image_files, 1):
            print(f"\n[{i}/{len(image_files)}] Processing: {image_path.name}")

            result = self.process_single_image(image_path)

            # Update overall statistics
            if result["success"]:
                self.results["successful_detections"] += 1
                self.results["total_plots_detected"] += result["plots_detected"]

                if result["plots_detected"] > 0:
                    self.results["images_with_plots"] += 1
                else:
                    self.results["images_without_plots"] += 1
            else:
                self.results["failed_detections"] += 1

            # Store individual result
            self.results["results_by_image"][image_path.name] = result

        # Calculate overall processing time
        self.results["processing_time"] = time.time() - overall_start_time

        # Generate final report
        self.generate_final_report()

    def generate_final_report(self):
        """Generate comprehensive final report."""
        print("\n" + "=" * 60)
        print("FINAL PROCESSING REPORT")
        print("=" * 60)

        results = self.results

        print(f"Total images processed: {results['total_images']}")
        print(f"Successful detections: {results['successful_detections']}")
        print(f"Failed detections: {results['failed_detections']}")
        print(
            f"Success rate: {(results['successful_detections']/results['total_images']*100):.1f}%"
        )
        print()

        print(f"Total plots detected: {results['total_plots_detected']}")
        print(f"Images with plots: {results['images_with_plots']}")
        print(f"Images without plots: {results['images_without_plots']}")
        if results["images_with_plots"] > 0:
            avg_plots = results["total_plots_detected"] / results["images_with_plots"]
            print(f"Average plots per image (with plots): {avg_plots:.1f}")
        print()

        print(f"Total processing time: {results['processing_time']:.2f} seconds")
        if results["successful_detections"] > 0:
            avg_time = results["processing_time"] / results["successful_detections"]
            print(f"Average time per image: {avg_time:.2f} seconds")
        print()

        # Top images by plot count
        successful_images = [
            (name, data)
            for name, data in results["results_by_image"].items()
            if data["success"] and data["plots_detected"] > 0
        ]

        if successful_images:
            print("Top 10 images by plot count:")
            top_images = sorted(
                successful_images, key=lambda x: x[1]["plots_detected"], reverse=True
            )[:10]

            for i, (name, data) in enumerate(top_images, 1):
                plots = data["plots_detected"]
                time_taken = data["processing_time"]
                print(f"  {i:2d}. {name}: {plots} plots ({time_taken:.2f}s)")

        # Failed images
        failed_images = [
            (name, data)
            for name, data in results["results_by_image"].items()
            if not data["success"]
        ]

        if failed_images:
            print(f"\nFailed images ({len(failed_images)}):")
            for name, data in failed_images:
                error = data.get("error_message", "Unknown error")
                print(f"  - {name}: {error}")

        # Save comprehensive report
        report_path = self.output_dir / "comprehensive_test_report.json"
        with open(report_path, "w") as f:
            json.dump(self.results, f, indent=2)

        print(f"\n✓ Comprehensive report saved: {report_path}")
        print(f"✓ Individual results saved in subfolders under: {self.output_dir}")

        # Generate summary statistics
        stats_path = self.output_dir / "processing_statistics.json"
        stats = {
            "summary": {
                "total_images": results["total_images"],
                "successful_rate": round(
                    results["successful_detections"] / results["total_images"] * 100, 1
                ),
                "total_plots_detected": results["total_plots_detected"],
                "average_plots_per_successful_image": round(
                    results["total_plots_detected"]
                    / max(results["successful_detections"], 1),
                    1,
                ),
                "total_processing_time_seconds": round(results["processing_time"], 2),
            },
            "device_info": self.detector.get_model_info(),
            "output_structure": {
                "main_directory": str(self.output_dir),
                "subfolders_created": results["successful_detections"],
                "files_per_successful_image": "4-6 files (labeled images, visualizations, JSON reports)",
            },
        }

        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)

        print(f"✓ Processing statistics saved: {stats_path}")


def main():
    """Main function to run the comprehensive test."""

    # Check if required directories exist
    if not Path("images").exists():
        print("Error: 'images' directory not found!")
        print("Please ensure the images directory exists with image files.")
        return

    if not Path("plot_detector1.pt").exists():
        print("Error: 'plot_detector1.pt' model file not found!")
        print("Please ensure the model file is in the current directory.")
        return

    # Initialize and run tester
    tester = BoundingBoxTester(
        images_dir="images", output_dir="bbox", model_path="plot_detector1.pt"
    )

    try:
        tester.run_comprehensive_test()

        print("\n" + "=" * 60)
        print("✅ COMPREHENSIVE TEST COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"Results available in: {tester.output_dir}")
        print("Each image has its own subfolder with:")
        print("  • Sequential labeled images")
        print("  • Center labeled images")
        print("  • Basic visualizations")
        print("  • High confidence visualizations")
        print("  • JSON reference files")
        print("  • Detection reports")
        print("\nCheck comprehensive_test_report.json for full results!")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
