"""Reusable Streamlit UI components."""

import streamlit as st
import numpy as np
import cv2
from typing import List, Dict


def display_metrics_row(detections: List[Dict]):
    """Display key metrics in a row of cards."""
    total = len(detections)

    # Count by class
    class_counts = {}
    severity_counts = {"Low": 0, "Medium": 0, "High": 0, "Critical": 0}

    for det in detections:
        cls = det["class_name"]
        class_counts[cls] = class_counts.get(cls, 0) + 1

        severity = det.get("severity")
        if severity:
            sev_name = severity.value if hasattr(severity, 'value') else str(severity)
            severity_counts[sev_name] = severity_counts.get(sev_name, 0) + 1

    # Metrics row
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Defects", total)
    col2.metric("Cracks", class_counts.get("crack", 0))
    col3.metric("Corrosion", class_counts.get("corrosion", 0))
    col4.metric("Spalling", class_counts.get("spalling", 0))
    col5.metric("🔴 Critical", severity_counts["Critical"])


def display_detection_table(detections: List[Dict]):
    """Display detections in a formatted table."""
    if not detections:
        st.info("No defects detected.")
        return

    import pandas as pd

    rows = []
    for i, det in enumerate(detections):
        severity = det.get("severity")
        sev_name = severity.value if hasattr(severity, 'value') else str(severity) if severity else "N/A"

        rows.append({
            "#": i + 1,
            "Defect Type": det["class_name"].capitalize(),
            "Confidence": f"{det['confidence'] * 100:.1f}%",
            "Severity": sev_name,
            "Area (px)": det["area"],
            "Width": det["width"],
            "Height": det["height"],
        })

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)


def numpy_to_streamlit(image: np.ndarray) -> np.ndarray:
    """Convert BGR (OpenCV) image to RGB for Streamlit display."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
