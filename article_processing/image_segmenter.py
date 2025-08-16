import torch
import cv2
import numpy as np
import time
from typing import List, Tuple, Union, Optional
from pathlib import Path
from ultralytics import YOLO
from PIL import Image


class ImageProcessorBase:
    """
    Base class containing shared functionality for image processing classes.
    """

    def _load_image(self, image: Union[str, np.ndarray, Image.Image]) -> np.ndarray:
        """Load image from various input formats."""
        if isinstance(image, str):
            img = cv2.imread(image)
            if img is None:
                raise ValueError(f"Could not load image from {image}")
        elif isinstance(image, Image.Image):
            img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        else:
            img = image.copy()

        return img

    def _generate_colors(self, num_colors: int) -> List[Tuple[int, int, int]]:
        """Generate a list of purple BGR colors (all same color)."""
        # Purple color in BGR format
        purple = (128, 0, 128)
        return [purple] * num_colors

    def _generate_filename(
        self,
        image: Union[str, np.ndarray, Image.Image],
        num_boxes: int,
        prefix: str = "image",
    ) -> str:
        """Generate a filename for the output image."""
        if isinstance(image, str):
            base_name = Path(image).stem
        else:
            base_name = "image"

        timestamp = int(time.time())
        return f"{prefix}_{base_name}_{num_boxes}boxes_{timestamp}.jpg"


class PlotDetector(ImageProcessorBase):
    """
    A class for detecting and creating bounding boxes around plots/figures in images
    using the plot_detector1.pt model.
    """

    def __init__(
        self,
        model_path: str = "plot_detector1.pt",
        confidence_threshold: float = 0.75,
        device: str = "auto",
    ):
        """
        Initialize the PlotDetector with the specified model.

        Args:
            model_path (str): Path to the plot detector model file
            confidence_threshold (float): Minimum confidence score for detections
            device (str): Device to use ('auto', 'cuda', 'cpu'). 'auto' will use CUDA if available
        """
        self.model_path = Path(model_path)
        self.confidence_threshold = confidence_threshold
        self.device = self._get_device(device)
        self.model = None
        self._load_model()

    def _get_device(self, device: str) -> str:
        """
        Determine the best device to use for inference.

        Args:
            device (str): Device preference ('auto', 'cuda', 'cpu')

        Returns:
            str: The device to use
        """
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
                print(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
            else:
                device = "cpu"
                print("CUDA not available. Using CPU.")
        elif device == "cuda":
            if not torch.cuda.is_available():
                print("Warning: CUDA requested but not available. Falling back to CPU.")
                device = "cpu"
            else:
                print(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("Using CPU.")

        return device

    def _load_model(self) -> None:
        """Load the YOLO model from the specified path and move it to the appropriate device."""
        try:
            if not self.model_path.exists():
                raise FileNotFoundError(f"Model file not found: {self.model_path}")

            self.model = YOLO(str(self.model_path))

            # Move model to the specified device
            if self.device == "cuda" and torch.cuda.is_available():
                self.model.to("cuda")
                print(f"Successfully loaded model from {self.model_path} on GPU")
            else:
                self.model.to("cpu")
                print(f"Successfully loaded model from {self.model_path} on CPU")

        except Exception as e:
            raise RuntimeError(f"Failed to load model: {str(e)}")

    def detect_plots(self, image: Union[str, np.ndarray, Image.Image]) -> List[dict]:
        """
        Detect plots in the given image and return bounding box information.

        Args:
            image: Input image (file path, numpy array, or PIL Image)

        Returns:
            List of dictionaries containing bounding box information:
            [
                {
                    'bbox': (x1, y1, x2, y2),  # Bounding box coordinates
                    'confidence': float,        # Detection confidence score
                    'class_id': int,           # Class ID from model
                    'class_name': str          # Class name if available
                },
                ...
            ]
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call _load_model() first.")

        try:
            # Run inference with device specification
            results = self.model(
                image, conf=self.confidence_threshold, device=self.device
            )

            detections = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Extract bounding box coordinates (xyxy format)
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = float(box.conf[0].cpu().numpy())
                        class_id = int(box.cls[0].cpu().numpy())

                        # Get class name if available
                        class_name = None
                        if hasattr(self.model, "names") and self.model.names:
                            class_name = self.model.names.get(
                                class_id, f"class_{class_id}"
                            )

                        detection = {
                            "bbox": (int(x1), int(y1), int(x2), int(y2)),
                            "confidence": confidence,
                            "class_id": class_id,
                            "class_name": class_name,
                        }
                        detections.append(detection)

            return detections

        except Exception as e:
            raise RuntimeError(f"Error during plot detection: {str(e)}")

    def draw_bounding_boxes(
        self,
        image: Union[str, np.ndarray, Image.Image],
        detections: Optional[List[dict]] = None,
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2,
        show_confidence: bool = True,
    ) -> np.ndarray:
        """
        Draw bounding boxes on the image.

        Args:
            image: Input image
            detections: List of detections (if None, will run detection first)
            color: RGB color for bounding boxes
            thickness: Thickness of bounding box lines
            show_confidence: Whether to show confidence scores

        Returns:
            Image with bounding boxes drawn
        """
        # Load image if it's a path
        if isinstance(image, str):
            img = cv2.imread(image)
            if img is None:
                raise ValueError(f"Could not load image from {image}")
        elif isinstance(image, Image.Image):
            img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        else:
            img = image.copy()

        # Get detections if not provided
        if detections is None:
            detections = self.detect_plots(image)

        # Draw bounding boxes
        for detection in detections:
            x1, y1, x2, y2 = detection["bbox"]
            confidence = detection["confidence"]
            class_name = detection.get("class_name", "plot")

            # Draw rectangle
            cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

            # Add label with confidence
            if show_confidence:
                label = f"{class_name}: {confidence:.2f}"
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                # Position label background inside the bounding box at top-left
                cv2.rectangle(
                    img,
                    (x1 + 2, y1 + 2),
                    (x1 + label_size[0] + 7, y1 + label_size[1] + 12),
                    color,
                    -1,
                )
                cv2.putText(
                    img,
                    label,
                    (x1 + 5, y1 + label_size[1] + 7),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                )

        return img

    def segment_plots(
        self, image: Union[str, np.ndarray, Image.Image]
    ) -> List[np.ndarray]:
        """
        Extract individual plot regions from the image based on detections.

        Args:
            image: Input image

        Returns:
            List of cropped images containing individual plots
        """
        # Load image if it's a path
        if isinstance(image, str):
            img = cv2.imread(image)
            if img is None:
                raise ValueError(f"Could not load image from {image}")
        elif isinstance(image, Image.Image):
            img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        else:
            img = image.copy()

        detections = self.detect_plots(image)

        plot_segments = []
        for detection in detections:
            x1, y1, x2, y2 = detection["bbox"]

            # Ensure coordinates are within image bounds
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(img.shape[1], x2)
            y2 = min(img.shape[0], y2)

            # Extract the plot region
            plot_region = img[y1:y2, x1:x2]
            plot_segments.append(plot_region)

        return plot_segments

    def save_detected_plots(
        self,
        image: Union[str, np.ndarray, Image.Image],
        output_dir: str = "detected_plots",
        prefix: str = "plot",
    ) -> List[str]:
        """
        Save detected plot regions as separate image files.

        Args:
            image: Input image
            output_dir: Directory to save detected plots
            prefix: Prefix for saved file names

        Returns:
            List of saved file paths
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        plot_segments = self.segment_plots(image)
        saved_files = []

        for i, plot_segment in enumerate(plot_segments):
            filename = f"{prefix}_{i+1:03d}.jpg"
            file_path = output_path / filename

            cv2.imwrite(str(file_path), plot_segment)
            saved_files.append(str(file_path))

        return saved_files

    def get_model_info(self) -> dict:
        """
        Get information about the loaded model.

        Returns:
            Dictionary containing model information
        """
        if self.model is None:
            return {"status": "Model not loaded"}

        info = {
            "model_path": str(self.model_path),
            "confidence_threshold": self.confidence_threshold,
            "device": self.device,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": (
                torch.cuda.device_count() if torch.cuda.is_available() else 0
            ),
        }

        if hasattr(self.model, "names") and self.model.names:
            info["classes"] = self.model.names

        # Add GPU memory info if using CUDA
        if self.device == "cuda" and torch.cuda.is_available():
            info["gpu_memory_allocated"] = (
                f"{torch.cuda.memory_allocated() / 1024**3:.2f} GB"
            )
            info["gpu_memory_reserved"] = (
                f"{torch.cuda.memory_reserved() / 1024**3:.2f} GB"
            )

        return info


class BoundingBoxVisualizer(ImageProcessorBase):
    """
    A class for visualizing bounding boxes on images and saving the results.
    """

    def __init__(self, output_dir: str = "bounding_box_images"):
        """
        Initialize the BoundingBoxVisualizer.

        Args:
            output_dir (str): Directory to save images with bounding boxes
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        print(f"BoundingBoxVisualizer initialized. Output directory: {self.output_dir}")

    def visualize_and_save(
        self,
        image: Union[str, np.ndarray, Image.Image],
        bounding_boxes: List[dict],
        filename: Optional[str] = None,
        colors: Optional[List[Tuple[int, int, int]]] = None,
        thickness: int = 3,
        show_confidence: bool = True,
        show_class_names: bool = True,
        font_scale: float = 0.6,
        label_background: bool = True,
    ) -> str:
        """
        Visualize bounding boxes on an image and save the result.

        Args:
            image: Input image (file path, numpy array, or PIL Image)
            bounding_boxes: List of bounding box dictionaries with format:
                [
                    {
                        'bbox': (x1, y1, x2, y2),
                        'confidence': float,
                        'class_name': str (optional),
                        'class_id': int (optional)
                    },
                    ...
                ]
            filename: Output filename (if None, will generate from input)
            colors: List of BGR colors for each box (if None, will use default colors)
            thickness: Thickness of bounding box lines
            show_confidence: Whether to show confidence scores in labels
            show_class_names: Whether to show class names in labels
            font_scale: Scale factor for text
            label_background: Whether to draw background for labels

        Returns:
            str: Path to the saved image
        """
        # Load and prepare the image
        img = self._load_image(image)

        # Generate default colors if not provided
        if colors is None:
            colors = self._generate_colors(len(bounding_boxes))

        # Draw bounding boxes
        for i, bbox_info in enumerate(bounding_boxes):
            color = colors[i % len(colors)] if colors else (0, 255, 0)
            self._draw_single_bbox(
                img,
                bbox_info,
                color,
                thickness,
                show_confidence,
                show_class_names,
                font_scale,
                label_background,
            )

        # Generate output filename if not provided
        if filename is None:
            filename = self._generate_filename(image, len(bounding_boxes), "bbox")

        # Save the image
        output_path = self.output_dir / filename
        cv2.imwrite(str(output_path), img)
        print(
            f"Saved visualization with {len(bounding_boxes)} bounding boxes to: {output_path}"
        )

        return str(output_path)

    def visualize_multiple_images(
        self,
        image_bbox_pairs: List[Tuple[Union[str, np.ndarray, Image.Image], List[dict]]],
        prefix: str = "visualization",
        **kwargs,
    ) -> List[str]:
        """
        Visualize bounding boxes on multiple images.

        Args:
            image_bbox_pairs: List of (image, bounding_boxes) tuples
            prefix: Prefix for output filenames
            **kwargs: Additional arguments passed to visualize_and_save

        Returns:
            List of paths to saved images
        """
        saved_paths = []

        for i, (image, bboxes) in enumerate(image_bbox_pairs):
            filename = f"{prefix}_{i+1:03d}.jpg"
            output_path = self.visualize_and_save(
                image, bboxes, filename=filename, **kwargs
            )
            saved_paths.append(output_path)

        return saved_paths

    def create_comparison_grid(
        self,
        images_with_bboxes: List[
            Tuple[Union[str, np.ndarray, Image.Image], List[dict]]
        ],
        grid_size: Optional[Tuple[int, int]] = None,
        image_size: Tuple[int, int] = (300, 300),
        filename: str = "comparison_grid.jpg",
    ) -> str:
        """
        Create a grid comparison of multiple images with their bounding boxes.

        Args:
            images_with_bboxes: List of (image, bounding_boxes) tuples
            grid_size: (rows, cols) for the grid (auto-calculated if None)
            image_size: Size to resize each image to
            filename: Output filename

        Returns:
            str: Path to the saved grid image
        """
        if not images_with_bboxes:
            raise ValueError("No images provided")

        num_images = len(images_with_bboxes)

        # Auto-calculate grid size if not provided
        if grid_size is None:
            cols = int(np.ceil(np.sqrt(num_images)))
            rows = int(np.ceil(num_images / cols))
            grid_size = (rows, cols)

        rows, cols = grid_size

        # Create the grid
        grid_height = rows * image_size[1]
        grid_width = cols * image_size[0]
        grid_image = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)

        for i, (image, bboxes) in enumerate(images_with_bboxes):
            if i >= rows * cols:
                break

            # Load and process image
            img = self._load_image(image)

            # Draw bounding boxes
            colors = self._generate_colors(len(bboxes))
            for j, bbox_info in enumerate(bboxes):
                color = colors[j % len(colors)]
                self._draw_single_bbox(
                    img,
                    bbox_info,
                    color,
                    thickness=2,
                    show_confidence=True,
                    show_class_names=True,
                    font_scale=0.4,
                    label_background=True,
                )

            # Resize image
            img_resized = cv2.resize(img, image_size)

            # Place in grid
            row = i // cols
            col = i % cols
            y_start = row * image_size[1]
            y_end = y_start + image_size[1]
            x_start = col * image_size[0]
            x_end = x_start + image_size[0]

            grid_image[y_start:y_end, x_start:x_end] = img_resized

        # Save grid
        output_path = self.output_dir / filename
        cv2.imwrite(str(output_path), grid_image)
        print(f"Saved comparison grid ({rows}x{cols}) to: {output_path}")

        return str(output_path)

    def _draw_single_bbox(
        self,
        img: np.ndarray,
        bbox_info: dict,
        color: Tuple[int, int, int],
        thickness: int,
        show_confidence: bool,
        show_class_names: bool,
        font_scale: float,
        label_background: bool,
    ) -> None:
        """Draw a single bounding box on the image."""
        x1, y1, x2, y2 = bbox_info["bbox"]

        # Draw rectangle
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

        # Prepare label
        label_parts = []
        if show_class_names and "class_name" in bbox_info:
            label_parts.append(str(bbox_info["class_name"]))
        if show_confidence and "confidence" in bbox_info:
            label_parts.append(f"{bbox_info['confidence']:.2f}")

        if label_parts:
            label = ": ".join(label_parts)

            # Get text size
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
            )

            # Draw label background if requested
            if label_background:
                # Position background inside the bounding box at top-left
                cv2.rectangle(
                    img,
                    (x1 + 2, y1 + 2),
                    (x1 + text_width + 7, y1 + text_height + baseline + 7),
                    color,
                    -1,
                )

            # Draw text inside the bounding box
            cv2.putText(
                img,
                label,
                (x1 + 5, y1 + text_height + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

    def get_output_stats(self) -> dict:
        """Get statistics about saved images."""
        if not self.output_dir.exists():
            return {"total_images": 0, "output_dir": str(self.output_dir)}

        image_files = list(self.output_dir.glob("*.jpg")) + list(
            self.output_dir.glob("*.png")
        )
        total_size = sum(f.stat().st_size for f in image_files)

        return {
            "total_images": len(image_files),
            "output_dir": str(self.output_dir),
            "total_size_mb": total_size / (1024 * 1024),
            "latest_files": [
                str(f.name)
                for f in sorted(image_files, key=lambda x: x.stat().st_mtime)[-5:]
            ],
        }


class BoundingBoxLabeler(ImageProcessorBase):
    """
    A class for labeling bounding boxes with sequential numbers arranged row by row, left to right.
    """

    def __init__(self, output_dir: str = "bounding_box_images"):
        """
        Initialize the BoundingBoxLabeler.

        Args:
            output_dir (str): Directory to save labeled images
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        print(f"BoundingBoxLabeler initialized. Output directory: {self.output_dir}")

    def sort_bboxes_by_position(
        self,
        bounding_boxes: List[dict],
        image: Union[str, np.ndarray, Image.Image],
        row_tolerance_percent: float = 15.0,
    ) -> List[dict]:
        """
        Sort bounding boxes by position (row by row, left to right) with tolerance for staggered boxes.

        Args:
            bounding_boxes: List of bounding box dictionaries
            image: Input image to calculate tolerance from
            row_tolerance_percent: Vertical tolerance as percentage of image height for grouping boxes into rows

        Returns:
            List of bounding boxes sorted by position with added 'label' field
        """
        if not bounding_boxes:
            return []

        # Load image to get dimensions
        img = self._load_image(image)
        image_height = img.shape[0]

        # Calculate actual tolerance in pixels from percentage
        row_tolerance = int((row_tolerance_percent / 100.0) * image_height)
        print(
            f"Row tolerance set to {row_tolerance} pixels based on image height {image_height}."
        )

        # Make a copy to avoid modifying the original
        boxes = [box.copy() for box in bounding_boxes]

        # Group boxes into rows based on vertical overlap/proximity
        rows = self._group_boxes_into_rows(boxes, row_tolerance)

        # Sort boxes within each row by x-coordinate (left to right)
        for row in rows:
            row.sort(key=lambda box: box["bbox"][0])  # Sort by x1 coordinate

        # Sort rows by their average y-coordinate (top to bottom)
        rows.sort(key=lambda row: sum(box["bbox"][1] for box in row) / len(row))

        # Flatten rows and add sequential labels
        final_sorted_boxes = []
        label_counter = 1

        for row in rows:
            for box in row:
                box["label"] = label_counter
                final_sorted_boxes.append(box)
                label_counter += 1

        return final_sorted_boxes

    def _group_boxes_into_rows(
        self, boxes: List[dict], tolerance: int
    ) -> List[List[dict]]:
        """
        Group bounding boxes into rows based on center y-coordinate proximity with tolerance buffer.

        Args:
            boxes: List of bounding box dictionaries
            tolerance: Vertical tolerance in pixels (calculated from percentage of image height)

        Returns:
            List of rows, where each row is a list of boxes
        """
        if not boxes:
            return []

        # Sort boxes by center y-coordinate to process top to bottom
        boxes_by_center_y = sorted(boxes, key=lambda box: self._get_box_center_y(box))

        rows = []
        current_row = [boxes_by_center_y[0]]
        current_row_center_y = self._get_row_center_y([boxes_by_center_y[0]])

        for box in boxes_by_center_y[1:]:
            box_center_y = self._get_box_center_y(box)

            # Check if this box's center is within tolerance of the current row's center
            if abs(box_center_y - current_row_center_y) <= tolerance:
                current_row.append(box)
                # Update the row's center y-coordinate to include the new box
                current_row_center_y = self._get_row_center_y(current_row)
            else:
                # Start a new row
                rows.append(current_row)
                current_row = [box]
                current_row_center_y = self._get_row_center_y([box])

        # Add the last row
        if current_row:
            rows.append(current_row)

        return rows

    def _get_box_center_y(self, box: dict) -> float:
        """
        Get the center y-coordinate of a bounding box.

        Args:
            box: Bounding box dictionary

        Returns:
            Center y-coordinate of the box
        """
        y1, y2 = box["bbox"][1], box["bbox"][3]
        return (y1 + y2) / 2

    def _get_row_center_y(self, row_boxes: List[dict]) -> float:
        """
        Get the average center y-coordinate of all boxes in a row.

        Args:
            row_boxes: List of boxes in the row

        Returns:
            Average center y-coordinate of the row
        """
        if not row_boxes:
            return 0.0

        centers = [self._get_box_center_y(box) for box in row_boxes]
        return sum(centers) / len(centers)

    def label_and_save(
        self,
        image: Union[str, np.ndarray, Image.Image],
        bounding_boxes: List[dict],
        filename: Optional[str] = None,
        colors: Optional[List[Tuple[int, int, int]]] = None,
        thickness: int = 3,
        show_confidence: bool = True,
        show_class_names: bool = True,
        font_scale: float = 0.8,
        label_background: bool = True,
        label_position: str = "top_left",  # "top_left", "top_right", "center", "top_center"
        row_tolerance_percent: float = 15.0,  # Percentage of image height
    ) -> Tuple[str, List[dict]]:
        """
        Label bounding boxes with sequential numbers and save the result.

        Args:
            image: Input image (file path, numpy array, or PIL Image)
            bounding_boxes: List of bounding box dictionaries
            filename: Output filename (if None, will generate from input)
            colors: List of BGR colors for each box (if None, will use default colors)
            thickness: Thickness of bounding box lines
            show_confidence: Whether to show confidence scores in labels
            show_class_names: Whether to show class names in labels
            font_scale: Scale factor for text
            label_background: Whether to draw background for labels
            label_position: Where to place the label ("top_left", "top_right", "center", "top_center")
            row_tolerance_percent: Vertical tolerance as percentage of image height for grouping boxes into rows

        Returns:
            Tuple of (saved_file_path, labeled_bounding_boxes)
        """
        # Load and prepare the image
        img = self._load_image(image)

        # Sort bounding boxes and add labels
        labeled_boxes = self.sort_bboxes_by_position(
            bounding_boxes, image, row_tolerance_percent
        )

        # Generate default colors if not provided
        if colors is None:
            colors = self._generate_colors(len(labeled_boxes))

        # Draw bounding boxes with labels
        for i, bbox_info in enumerate(labeled_boxes):
            color = colors[i % len(colors)] if colors else (0, 255, 0)
            self._draw_labeled_bbox(
                img,
                bbox_info,
                color,
                thickness,
                show_confidence,
                show_class_names,
                font_scale,
                label_background,
                label_position,
            )

        # Generate output filename if not provided
        if filename is None:
            filename = self._generate_filename(image, len(labeled_boxes), "labeled")

        # Save the image
        output_path = self.output_dir / filename
        cv2.imwrite(str(output_path), img)
        print(
            f"Saved labeled visualization with {len(labeled_boxes)} bounding boxes to: {output_path}"
        )

        return str(output_path), labeled_boxes

    def create_label_reference(
        self, labeled_boxes: List[dict], image_name: str = "image"
    ) -> dict:
        """
        Create a reference dictionary mapping labels to bounding box information.

        Args:
            labeled_boxes: List of labeled bounding boxes
            image_name: Name of the image for reference

        Returns:
            Dictionary with label mapping and statistics
        """
        reference = {
            "image_name": image_name,
            "total_plots": len(labeled_boxes),
            "labels": {},
        }

        for box in labeled_boxes:
            label = box["label"]
            bbox = box["bbox"]
            confidence = box.get("confidence", 0.0)
            class_name = box.get("class_name", "Unknown")

            reference["labels"][label] = {
                "bbox": bbox,
                "confidence": confidence,
                "class_name": class_name,
                "center": ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2),
                "area": (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]),
            }

        return reference

    def batch_label_images(
        self,
        image_bbox_pairs: List[Tuple[Union[str, np.ndarray, Image.Image], List[dict]]],
        prefix: str = "labeled",
        save_references: bool = True,
        **kwargs,
    ) -> Tuple[List[str], List[dict]]:
        """
        Label multiple images with sequential bounding box numbers.

        Args:
            image_bbox_pairs: List of (image, bounding_boxes) tuples
            prefix: Prefix for output filenames
            save_references: Whether to save label reference files
            **kwargs: Additional arguments passed to label_and_save

        Returns:
            Tuple of (saved_paths, label_references)
        """
        saved_paths = []
        label_references = []

        for i, (image, bboxes) in enumerate(image_bbox_pairs):
            # Generate filename
            if isinstance(image, str):
                base_name = Path(image).stem
            else:
                base_name = f"image_{i+1}"

            filename = f"{prefix}_{base_name}.jpg"

            # Label and save
            output_path, labeled_boxes = self.label_and_save(
                image, bboxes, filename=filename, **kwargs
            )
            saved_paths.append(output_path)

            # Create reference
            reference = self.create_label_reference(labeled_boxes, base_name)
            label_references.append(reference)

            # Save reference as JSON if requested
            if save_references:
                import json

                ref_filename = f"{prefix}_{base_name}_reference.json"
                ref_path = self.output_dir / ref_filename
                with open(ref_path, "w") as f:
                    json.dump(reference, f, indent=2)
                print(f"Saved label reference to: {ref_path}")

        return saved_paths, label_references

    def _draw_labeled_bbox(
        self,
        img: np.ndarray,
        bbox_info: dict,
        color: Tuple[int, int, int],
        thickness: int,
        show_confidence: bool,
        show_class_names: bool,
        font_scale: float,
        label_background: bool,
        label_position: str,
    ) -> None:
        """Draw a single labeled bounding box on the image."""
        x1, y1, x2, y2 = bbox_info["bbox"]
        label_num = bbox_info["label"]

        # Draw rectangle
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

        # Prepare label text
        label_parts = [str(label_num)]  # Start with the sequential number

        if show_class_names and "class_name" in bbox_info:
            label_parts.append(str(bbox_info["class_name"]))
        if show_confidence and "confidence" in bbox_info:
            label_parts.append(f"{bbox_info['confidence']:.2f}")

        label_text = ": ".join(label_parts)

        # Calculate label position
        if label_position == "center":
            # Place label in the center of the bounding box
            label_x = (x1 + x2) // 2
            label_y = (y1 + y2) // 2

            # Get text size to center it
            (text_width, text_height), baseline = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2
            )
            label_x -= text_width // 2
            label_y += text_height // 2

        elif label_position == "top_center":
            # Place label at the top center of the bounding box
            label_x = (x1 + x2) // 2
            label_y = y1

            # Get text size to center horizontally
            (text_width, text_height), baseline = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2
            )
            label_x -= text_width // 2

        elif label_position == "top_right":
            # Place label at the top right of the bounding box
            label_x = x2
            label_y = y1

            # Get text size to align right
            (text_width, text_height), baseline = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2
            )
            label_x -= text_width

        else:  # default: "top_left"
            label_x = x1
            label_y = y1

        # Get text size for background
        (text_width, text_height), baseline = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2
        )

        # Draw label background if requested
        if label_background:
            # Adjust background position based on label position
            if label_position == "center":
                bg_x1 = label_x - 5
                bg_y1 = label_y - text_height - 5
                bg_x2 = label_x + text_width + 5
                bg_y2 = label_y + baseline + 5
            elif label_position == "top_center":
                bg_x1 = label_x - 5
                bg_y1 = label_y - text_height - baseline - 10
                bg_x2 = label_x + text_width + 5
                bg_y2 = label_y
            elif label_position == "top_right":
                # Position background inside the bounding box at top-right
                bg_x1 = label_x - 5
                bg_y1 = label_y + 2
                bg_x2 = label_x + text_width + 2
                bg_y2 = label_y + text_height + baseline + 7
            else:  # top_left - position background inside the bounding box
                bg_x1 = label_x + 2
                bg_y1 = label_y + 2
                bg_x2 = label_x + text_width + 7
                bg_y2 = label_y + text_height + baseline + 7

            cv2.rectangle(img, (bg_x1, bg_y1), (bg_x2, bg_y2), color, -1)

        # Draw text
        if label_position == "center":
            text_y = label_y
            text_x = label_x + 2
        elif label_position == "top_left":
            # Position text inside the bounding box at top-left
            text_y = label_y + text_height + 5
            text_x = label_x + 2
        elif label_position == "top_right":
            # Position text inside the bounding box at top-right
            text_y = label_y + text_height + 5
            text_x = label_x - 2
        else:  # top_center
            text_y = label_y - baseline - 2
            text_x = label_x + 2

        cv2.putText(
            img,
            label_text,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    def get_output_stats(self) -> dict:
        """Get statistics about saved images."""
        if not self.output_dir.exists():
            return {"total_images": 0, "output_dir": str(self.output_dir)}

        image_files = list(self.output_dir.glob("*.jpg")) + list(
            self.output_dir.glob("*.png")
        )
        json_files = list(self.output_dir.glob("*.json"))
        total_size = sum(f.stat().st_size for f in image_files + json_files)

        return {
            "total_images": len(image_files),
            "total_references": len(json_files),
            "output_dir": str(self.output_dir),
            "total_size_mb": total_size / (1024 * 1024),
            "latest_files": [
                str(f.name)
                for f in sorted(image_files, key=lambda x: x.stat().st_mtime)[-5:]
            ],
        }


# Example usage and utility functions
def process_image_file(
    image_path: str, model_path: str = "plot_detector1.pt", device: str = "auto"
) -> dict:
    """
    Convenience function to process a single image file.

    Args:
        image_path: Path to the image file
        model_path: Path to the model file
        device: Device to use ('auto', 'cuda', 'cpu')

    Returns:
        Dictionary with detection results and processed image path
    """
    detector = PlotDetector(model_path, device=device)
    detections = detector.detect_plots(image_path)

    # Create output image with bounding boxes
    output_image = detector.draw_bounding_boxes(image_path, detections)

    # Save the output image
    output_path = Path(image_path).parent / f"detected_{Path(image_path).name}"
    cv2.imwrite(str(output_path), output_image)

    return {
        "detections": detections,
        "output_image_path": str(output_path),
        "num_plots_detected": len(detections),
    }


def batch_process_images(
    image_dir: str,
    model_path: str = "plot_detector1.pt",
    output_dir: str = "batch_results",
    device: str = "auto",
) -> dict:
    """
    Process multiple images in a directory.

    Args:
        image_dir: Directory containing images
        model_path: Path to the model file
        output_dir: Directory to save results
        device: Device to use ('auto', 'cuda', 'cpu')

    Returns:
        Dictionary with batch processing results
    """
    image_dir = Path(image_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    detector = PlotDetector(model_path, device=device)

    # Supported image extensions
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
    image_files = [
        f for f in image_dir.iterdir() if f.suffix.lower() in image_extensions
    ]

    results = {}
    total_detections = 0

    for image_file in image_files:
        try:
            detections = detector.detect_plots(str(image_file))

            # Save image with bounding boxes
            output_image = detector.draw_bounding_boxes(str(image_file), detections)
            output_path = output_dir / f"detected_{image_file.name}"
            cv2.imwrite(str(output_path), output_image)

            # Save individual plot segments
            plot_dir = output_dir / f"plots_{image_file.stem}"
            plot_files = detector.save_detected_plots(
                str(image_file), str(plot_dir), f"{image_file.stem}_plot"
            )

            results[str(image_file)] = {
                "detections": detections,
                "output_image": str(output_path),
                "plot_segments": plot_files,
                "num_plots": len(detections),
            }

            total_detections += len(detections)

        except Exception as e:
            results[str(image_file)] = {"error": str(e), "num_plots": 0}

    results["summary"] = {
        "total_images_processed": len(image_files),
        "total_plots_detected": total_detections,
        "average_plots_per_image": (
            total_detections / len(image_files) if image_files else 0
        ),
    }

    return results


def detect_and_visualize(
    image_path: str,
    model_path: str = "plot_detector1.pt",
    device: str = "auto",
    confidence_threshold: float = 0.5,
    output_dir: str = "bounding_box_images",
    filename: Optional[str] = None,
    **viz_kwargs,
) -> dict:
    """
    Convenience function that detects plots and visualizes them in one step.

    Args:
        image_path: Path to the input image
        model_path: Path to the model file
        device: Device to use ('auto', 'cuda', 'cpu')
        confidence_threshold: Minimum confidence for detections
        output_dir: Directory to save the visualization
        filename: Output filename (if None, will be auto-generated)
        **viz_kwargs: Additional arguments for visualization

    Returns:
        Dictionary with detection and visualization results
    """
    # Detect plots
    detector = PlotDetector(
        model_path, confidence_threshold=confidence_threshold, device=device
    )
    detections = detector.detect_plots(image_path)

    # Visualize and save
    visualizer = BoundingBoxVisualizer(output_dir)
    output_path = visualizer.visualize_and_save(
        image_path, detections, filename=filename, **viz_kwargs
    )

    return {
        "detections": detections,
        "num_plots_detected": len(detections),
        "visualization_path": output_path,
        "model_info": detector.get_model_info(),
    }


def batch_detect_and_visualize(
    image_dir: str,
    model_path: str = "plot_detector1.pt",
    device: str = "auto",
    confidence_threshold: float = 0.5,
    output_dir: str = "bounding_box_images",
    create_grid: bool = True,
    **viz_kwargs,
) -> dict:
    """
    Batch process images for detection and visualization.

    Args:
        image_dir: Directory containing input images
        model_path: Path to the model file
        device: Device to use ('auto', 'cuda', 'cpu')
        confidence_threshold: Minimum confidence for detections
        output_dir: Directory to save visualizations
        create_grid: Whether to create a comparison grid
        **viz_kwargs: Additional arguments for visualization

    Returns:
        Dictionary with batch processing results
    """
    image_dir = Path(image_dir)

    # Initialize detector and visualizer
    detector = PlotDetector(
        model_path, confidence_threshold=confidence_threshold, device=device
    )
    visualizer = BoundingBoxVisualizer(output_dir)

    # Find image files
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
    image_files = [
        f for f in image_dir.iterdir() if f.suffix.lower() in image_extensions
    ]

    results = {}
    image_bbox_pairs = []
    total_detections = 0

    for image_file in image_files:
        try:
            # Detect plots
            detections = detector.detect_plots(str(image_file))

            # Visualize and save
            output_path = visualizer.visualize_and_save(
                str(image_file),
                detections,
                filename=f"bbox_{image_file.name}",
                **viz_kwargs,
            )

            results[str(image_file)] = {
                "detections": detections,
                "num_plots": len(detections),
                "visualization_path": output_path,
            }

            total_detections += len(detections)

            # Store for grid creation
            if create_grid and detections:
                image_bbox_pairs.append((str(image_file), detections))

        except Exception as e:
            results[str(image_file)] = {"error": str(e), "num_plots": 0}

    # Create comparison grid if requested
    grid_path = None
    if create_grid and image_bbox_pairs:
        try:
            grid_path = visualizer.create_comparison_grid(
                image_bbox_pairs[:16],  # Limit to 16 images for grid
                filename="batch_comparison_grid.jpg",
            )
        except Exception as e:
            print(f"Failed to create comparison grid: {e}")

    # Summary
    results["summary"] = {
        "total_images_processed": len(image_files),
        "total_plots_detected": total_detections,
        "average_plots_per_image": (
            total_detections / len(image_files) if image_files else 0
        ),
        "comparison_grid": grid_path,
        "visualizer_stats": visualizer.get_output_stats(),
    }

    return results


def detect_and_label(
    image_path: str,
    model_path: str = "plot_detector1.pt",
    device: str = "auto",
    confidence_threshold: float = 0.5,
    output_dir: str = "bounding_box_images",
    filename: Optional[str] = None,
    label_position: str = "top_left",
    save_reference: bool = True,
    **label_kwargs,
) -> dict:
    """
    Convenience function that detects plots and labels them with sequential numbers.

    Args:
        image_path: Path to the input image
        model_path: Path to the model file
        device: Device to use ('auto', 'cuda', 'cpu')
        confidence_threshold: Minimum confidence for detections
        output_dir: Directory to save the labeled visualization
        filename: Output filename (if None, will be auto-generated)
        label_position: Where to place labels ("top_left", "top_right", "center", "top_center")
        save_reference: Whether to save a JSON reference file
        **label_kwargs: Additional arguments for labeling

    Returns:
        Dictionary with detection, labeling, and reference results
    """
    # Detect plots
    detector = PlotDetector(
        model_path, confidence_threshold=confidence_threshold, device=device
    )
    detections = detector.detect_plots(image_path)

    # Label and save
    labeler = BoundingBoxLabeler(output_dir)
    output_path, labeled_boxes = labeler.label_and_save(
        image_path,
        detections,
        filename=filename,
        label_position=label_position,
        **label_kwargs,
    )

    # Create reference
    base_name = Path(image_path).stem if isinstance(image_path, str) else "image"
    reference = labeler.create_label_reference(labeled_boxes, base_name)

    # Save reference if requested
    reference_path = None
    if save_reference:
        import json

        ref_filename = f"labeled_{base_name}_reference.json"
        reference_path = Path(output_dir) / ref_filename
        with open(reference_path, "w") as f:
            json.dump(reference, f, indent=2)
        print(f"Saved label reference to: {reference_path}")

    return {
        "detections": detections,
        "labeled_boxes": labeled_boxes,
        "num_plots_detected": len(detections),
        "labeled_image_path": output_path,
        "reference": reference,
        "reference_path": str(reference_path) if reference_path else None,
        "model_info": detector.get_model_info(),
    }
