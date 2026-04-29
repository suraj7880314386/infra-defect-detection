"""Severity Classification — categorizes defects by risk level."""

from typing import Dict, List
from enum import Enum


class Severity(str, Enum):
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    CRITICAL = "Critical"


# Severity colors for UI
SEVERITY_COLORS = {
    Severity.LOW: "#4CAF50",       # Green
    Severity.MEDIUM: "#FF9800",    # Orange
    Severity.HIGH: "#F44336",      # Red
    Severity.CRITICAL: "#9C27B0",  # Purple
}

SEVERITY_EMOJI = {
    Severity.LOW: "🟢",
    Severity.MEDIUM: "🟡",
    Severity.HIGH: "🟠",
    Severity.CRITICAL: "🔴",
}


def classify_severity(detection: Dict, image_area: int) -> Severity:
    """
    Classify defect severity based on multiple factors.

    Factors considered:
    1. Defect area relative to image area
    2. Detection confidence (higher confidence = more clear/severe defect)
    3. Defect type (cracks are generally more structurally dangerous)

    Args:
        detection: Detection dict from DefectDetector
        image_area: Total image area in pixels

    Returns:
        Severity enum value
    """
    confidence = detection["confidence"]
    area_ratio = detection["area"] / max(image_area, 1)
    class_name = detection["class_name"]

    # Base score from area ratio (0-40 points)
    if area_ratio > 0.15:
        area_score = 40
    elif area_ratio > 0.08:
        area_score = 30
    elif area_ratio > 0.03:
        area_score = 20
    else:
        area_score = 10

    # Confidence score (0-30 points)
    if confidence > 0.85:
        conf_score = 30
    elif confidence > 0.65:
        conf_score = 20
    elif confidence > 0.45:
        conf_score = 15
    else:
        conf_score = 5

    # Class-based risk modifier (0-30 points)
    class_scores = {
        "crack": 30,      # Structural risk
        "spalling": 20,   # Surface deterioration
        "corrosion": 25,  # Progressive damage
    }
    class_score = class_scores.get(class_name, 15)

    # Total score
    total = area_score + conf_score + class_score

    # Classify
    if total >= 80:
        return Severity.CRITICAL
    elif total >= 60:
        return Severity.HIGH
    elif total >= 40:
        return Severity.MEDIUM
    else:
        return Severity.LOW


def add_severity_to_detections(detections: List[Dict], image_shape: tuple) -> List[Dict]:
    """
    Add severity classification to each detection.

    Args:
        detections: List of detection dicts
        image_shape: (height, width, channels) of the image

    Returns:
        Same detections list with 'severity' key added
    """
    image_area = image_shape[0] * image_shape[1]

    for det in detections:
        det["severity"] = classify_severity(det, image_area)

    return detections


def get_severity_summary(detections: List[Dict]) -> Dict:
    """Generate a summary of severity distribution."""
    summary = {s: 0 for s in Severity}

    for det in detections:
        severity = det.get("severity", Severity.LOW)
        summary[severity] += 1

    return {
        "distribution": {s.value: count for s, count in summary.items()},
        "total_defects": len(detections),
        "highest_severity": max(
            (s for s in Severity if summary[s] > 0),
            default=Severity.LOW,
            key=lambda s: list(Severity).index(s),
        ).value,
        "critical_count": summary[Severity.CRITICAL],
        "high_count": summary[Severity.HIGH],
    }
