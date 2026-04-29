"""YOLOv8 Detection Engine — runs inference on images for defect detection."""

import os
import logging
from typing import List, Dict, Optional, Tuple

import numpy as np
import cv2
from ultralytics import YOLO

logger = logging.getLogger(__name__)

# Defect class configuration
CLASS_NAMES = {0: "crack", 1: "corrosion", 2: "spalling"}
CLASS_COLORS = {
    "crack": (0, 0, 255),       # Red (BGR)
    "corrosion": (0, 140, 255),  # Orange
    "spalling": (0, 255, 255),   # Yellow
}


class DefectDetector:
    """YOLOv8-based infrastructure defect detector."""

    def __init__(self, model_path: str = "models/best.pt", conf_threshold: float = 0.25):
        """
        Initialize the detector.

        Args:
            model_path: Path to YOLOv8 weights (.pt file)
            conf_threshold: Minimum confidence threshold for detections
        """
        self.conf_threshold = conf_threshold

        if os.path.exists(model_path):
            self.model = YOLO(model_path)
            logger.info(f"Loaded custom model: {model_path}")
        else:
            # Use pretrained YOLOv8n as base (will need fine-tuning)
            self.model = YOLO("yolov8n.pt")
            logger.warning(
                f"Custom model not found at {model_path}. "
                "Using base YOLOv8n — train on your dataset first."
            )

    def detect(self, image: np.ndarray, conf: Optional[float] = None) -> List[Dict]:
        """
        Run defect detection on a single image.

        Args:
            image: BGR image as numpy array
            conf: Override confidence threshold

        Returns:
            List of detection dicts with keys:
                class_name, confidence, bbox (x1,y1,x2,y2),
                area, center, class_id
        """
        threshold = conf or self.conf_threshold

        results = self.model.predict(
            source=image,
            conf=threshold,
            verbose=False,
        )

        detections = []

        if results and len(results) > 0:
            result = results[0]

            if result.boxes is not None:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    confidence = float(box.conf[0].cpu().numpy())
                    class_id = int(box.cls[0].cpu().numpy())

                    # Map class ID to name
                    class_name = CLASS_NAMES.get(class_id, f"class_{class_id}")

                    # Calculate area and center
                    width = x2 - x1
                    height = y2 - y1
                    area = width * height
                    center = ((x1 + x2) // 2, (y1 + y2) // 2)

                    detections.append({
                        "class_id": class_id,
                        "class_name": class_name,
                        "confidence": round(confidence, 4),
                        "bbox": (int(x1), int(y1), int(x2), int(y2)),
                        "width": int(width),
                        "height": int(height),
                        "area": int(area),
                        "center": center,
                    })

        logger.info(f"Detected {len(detections)} defects")
        return detections

    def detect_from_file(self, image_path: str, conf: Optional[float] = None) -> Tuple[np.ndarray, List[Dict]]:
        """
        Load an image from file and run detection.

        Returns:
            Tuple of (image, detections)
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")

        detections = self.detect(image, conf)
        return image, detections

    def detect_batch(self, images: List[np.ndarray], conf: Optional[float] = None) -> List[List[Dict]]:
        """
        Run detection on a batch of images.

        Returns:
            List of detection lists (one per image)
        """
        all_detections = []
        for image in images:
            detections = self.detect(image, conf)
            all_detections.append(detections)
        return all_detections

    def get_model_info(self) -> Dict:
        """Get model metadata."""
        return {
            "model_type": "YOLOv8",
            "classes": list(CLASS_NAMES.values()),
            "num_classes": len(CLASS_NAMES),
            "conf_threshold": self.conf_threshold,
        }
