"""Heatmap Generator — creates defect density heatmaps with Gaussian overlays."""

import numpy as np
import cv2
from typing import List, Dict, Tuple


def generate_heatmap(
    image: np.ndarray,
    detections: List[Dict],
    intensity: float = 0.6,
    radius_scale: float = 1.5,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """
    Generate a Gaussian-based defect density heatmap overlaid on the image.

    Args:
        image: Original BGR image
        detections: List of detection dicts with 'center' and 'area' keys
        intensity: Heatmap overlay intensity (0-1)
        radius_scale: Multiplier for Gaussian radius
        colormap: OpenCV colormap to use

    Returns:
        BGR image with heatmap overlay
    """
    h, w = image.shape[:2]

    # Create blank heatmap canvas
    heatmap = np.zeros((h, w), dtype=np.float32)

    for det in detections:
        cx, cy = det["center"]
        area = det["area"]

        # Gaussian radius proportional to defect size
        radius = int(np.sqrt(area) * radius_scale)
        radius = max(radius, 30)  # Minimum radius

        # Confidence-weighted intensity
        weight = det.get("confidence", 0.5)

        # Severity multiplier
        severity_weights = {
            "Low": 0.5,
            "Medium": 0.75,
            "High": 1.0,
            "Critical": 1.5,
        }
        severity = det.get("severity")
        if severity:
            sev_name = severity.value if hasattr(severity, 'value') else str(severity)
            weight *= severity_weights.get(sev_name, 1.0)

        # Create Gaussian kernel
        _add_gaussian(heatmap, cx, cy, radius, weight)

    # Normalize to 0-255
    if heatmap.max() > 0:
        heatmap = (heatmap / heatmap.max() * 255).astype(np.uint8)
    else:
        heatmap = heatmap.astype(np.uint8)

    # Apply colormap
    colored_heatmap = cv2.applyColorMap(heatmap, colormap)

    # Blend with original image
    mask = heatmap > 10  # Only overlay where there's signal
    mask_3ch = np.stack([mask] * 3, axis=-1)

    result = image.copy()
    result[mask_3ch] = cv2.addWeighted(
        image, 1 - intensity, colored_heatmap, intensity, 0
    )[mask_3ch]

    return result


def _add_gaussian(
    heatmap: np.ndarray,
    cx: int, cy: int,
    radius: int,
    weight: float,
) -> None:
    """Add a 2D Gaussian blob to the heatmap at (cx, cy)."""
    h, w = heatmap.shape

    # Create coordinate grids
    y_min = max(0, cy - radius * 2)
    y_max = min(h, cy + radius * 2)
    x_min = max(0, cx - radius * 2)
    x_max = min(w, cx + radius * 2)

    if y_max <= y_min or x_max <= x_min:
        return

    y_grid, x_grid = np.mgrid[y_min:y_max, x_min:x_max]

    # 2D Gaussian
    sigma = radius / 2.0
    gaussian = np.exp(
        -((x_grid - cx) ** 2 + (y_grid - cy) ** 2) / (2 * sigma ** 2)
    )

    heatmap[y_min:y_max, x_min:x_max] += gaussian.astype(np.float32) * weight


def generate_class_heatmaps(
    image: np.ndarray,
    detections: List[Dict],
) -> Dict[str, np.ndarray]:
    """
    Generate separate heatmaps for each defect class.

    Returns:
        Dict mapping class name to heatmap image
    """
    class_maps = {}
    classes = set(d["class_name"] for d in detections)

    colormaps = {
        "crack": cv2.COLORMAP_HOT,
        "corrosion": cv2.COLORMAP_AUTUMN,
        "spalling": cv2.COLORMAP_SUMMER,
    }

    for cls in classes:
        cls_detections = [d for d in detections if d["class_name"] == cls]
        cmap = colormaps.get(cls, cv2.COLORMAP_JET)
        class_maps[cls] = generate_heatmap(image, cls_detections, colormap=cmap)

    return class_maps
