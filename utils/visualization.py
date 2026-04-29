"""Visualization utilities — draw detections, annotations, and overlays."""

import cv2
import numpy as np
from typing import List, Dict

from utils.detector import CLASS_COLORS
from utils.severity import Severity, SEVERITY_COLORS


def draw_detections(
    image: np.ndarray,
    detections: List[Dict],
    show_labels: bool = True,
    show_confidence: bool = True,
    show_severity: bool = True,
    line_thickness: int = 2,
) -> np.ndarray:
    """
    Draw bounding boxes and labels on the image.

    Args:
        image: BGR image
        detections: List of detection dicts
        show_labels: Show class name
        show_confidence: Show confidence percentage
        show_severity: Show severity badge

    Returns:
        Annotated BGR image
    """
    annotated = image.copy()

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        class_name = det["class_name"]
        confidence = det["confidence"]
        severity = det.get("severity")

        # Color based on class
        color = CLASS_COLORS.get(class_name, (255, 255, 255))

        # Draw bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, line_thickness)

        # Build label text
        label_parts = []
        if show_labels:
            label_parts.append(class_name.upper())
        if show_confidence:
            label_parts.append(f"{confidence * 100:.1f}%")
        if show_severity and severity:
            sev_name = severity.value if hasattr(severity, 'value') else str(severity)
            label_parts.append(f"[{sev_name}]")

        label = " ".join(label_parts)

        if label:
            # Calculate text size
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            thickness = 1
            (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)

            # Draw label background
            cv2.rectangle(
                annotated,
                (x1, y1 - th - 10),
                (x1 + tw + 6, y1),
                color,
                -1,
            )

            # Draw label text
            cv2.putText(
                annotated,
                label,
                (x1 + 3, y1 - 5),
                font,
                font_scale,
                (255, 255, 255),
                thickness,
                cv2.LINE_AA,
            )

    return annotated


def draw_summary_bar(
    image: np.ndarray,
    detections: List[Dict],
    bar_height: int = 40,
) -> np.ndarray:
    """Add a summary bar at the bottom of the image showing defect counts."""
    h, w = image.shape[:2]
    bar = np.zeros((bar_height, w, 3), dtype=np.uint8)
    bar[:] = (40, 40, 40)  # Dark gray

    # Count by class
    counts = {}
    for det in detections:
        cls = det["class_name"]
        counts[cls] = counts.get(cls, 0) + 1

    # Draw counts
    x_offset = 10
    font = cv2.FONT_HERSHEY_SIMPLEX

    summary_text = f"Total: {len(detections)} defects"
    cv2.putText(bar, summary_text, (x_offset, 28), font, 0.6, (255, 255, 255), 1)
    x_offset += 200

    for cls, count in counts.items():
        color = CLASS_COLORS.get(cls, (255, 255, 255))
        text = f"{cls}: {count}"
        cv2.putText(bar, text, (x_offset, 28), font, 0.5, color, 1)
        x_offset += 120

    # Stack
    result = np.vstack([image, bar])
    return result


def create_side_by_side(
    original: np.ndarray,
    annotated: np.ndarray,
    label_left: str = "Original",
    label_right: str = "Detected",
) -> np.ndarray:
    """Create a side-by-side comparison of original and annotated images."""
    # Resize to same height
    h1, w1 = original.shape[:2]
    h2, w2 = annotated.shape[:2]
    target_h = min(h1, h2)

    if h1 != target_h:
        scale = target_h / h1
        original = cv2.resize(original, (int(w1 * scale), target_h))
    if h2 != target_h:
        scale = target_h / h2
        annotated = cv2.resize(annotated, (int(w2 * scale), target_h))

    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(original, label_left, (10, 30), font, 0.8, (255, 255, 255), 2)
    cv2.putText(annotated, label_right, (10, 30), font, 0.8, (0, 255, 0), 2)

    # Separator line
    sep = np.ones((target_h, 3, 3), dtype=np.uint8) * 128

    return np.hstack([original, sep, annotated])
