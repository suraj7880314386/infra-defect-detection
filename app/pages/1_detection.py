"""Detection Page — Upload images and run defect detection."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import streamlit as st
import numpy as np
import cv2
from PIL import Image

from utils.detector import DefectDetector
from utils.severity import add_severity_to_detections
from utils.heatmap import generate_heatmap
from utils.visualization import draw_detections, draw_summary_bar, create_side_by_side
from app.components import display_metrics_row, display_detection_table, numpy_to_streamlit

st.set_page_config(page_title="Detection", page_icon="🔍", layout="wide")

st.title("🔍 Defect Detection")
st.caption("Upload building or bridge images to detect structural defects")

# ─── Initialize Detector ─────────────────────────────────

@st.cache_resource
def load_detector():
    return DefectDetector(
        model_path="models/best.pt",
        conf_threshold=0.25,
    )

detector = load_detector()

# ─── Sidebar Controls ────────────────────────────────────

with st.sidebar:
    st.markdown("### 🎛️ Detection Settings")
    conf = st.slider("Confidence", 0.1, 0.95, 0.25, 0.05)
    show_heatmap = st.checkbox("Show Heatmap Overlay", value=True)
    show_side_by_side = st.checkbox("Show Side-by-Side", value=False)

# ─── File Upload ──────────────────────────────────────────

uploaded_files = st.file_uploader(
    "Upload Images",
    type=["jpg", "jpeg", "png", "bmp", "webp"],
    accept_multiple_files=True,
    help="Upload one or more images of buildings, bridges, or infrastructure",
)

if not uploaded_files:
    st.info("👆 Upload one or more images to begin detection")
    st.stop()

# ─── Process Each Image ──────────────────────────────────

# Store results for report generation
if "detection_results" not in st.session_state:
    st.session_state["detection_results"] = []

all_results = []

for idx, uploaded_file in enumerate(uploaded_files):
    st.divider()
    st.markdown(f"### 📷 Image {idx + 1}: `{uploaded_file.name}`")

    # Load image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image is None:
        st.error(f"Could not load: {uploaded_file.name}")
        continue

    # Run detection
    with st.spinner(f"Analyzing {uploaded_file.name}..."):
        detections = detector.detect(image, conf=conf)
        detections = add_severity_to_detections(detections, image.shape)

    # Display metrics
    display_metrics_row(detections)

    # Annotated image
    annotated = draw_detections(image, detections)
    annotated_with_bar = draw_summary_bar(annotated, detections)

    # Display
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Annotated Result**")
        st.image(numpy_to_streamlit(annotated_with_bar), use_container_width=True)

    with col2:
        if show_heatmap and detections:
            st.markdown("**Defect Heatmap**")
            heatmap_img = generate_heatmap(image, detections)
            st.image(numpy_to_streamlit(heatmap_img), use_container_width=True)
        elif show_side_by_side:
            st.markdown("**Original vs Detected**")
            comparison = create_side_by_side(image, annotated)
            st.image(numpy_to_streamlit(comparison), use_container_width=True)
        else:
            st.markdown("**Original Image**")
            st.image(numpy_to_streamlit(image), use_container_width=True)

    # Detection table
    with st.expander(f"📋 Defect Details ({len(detections)} found)", expanded=len(detections) > 0):
        display_detection_table(detections)

    # Store for reports
    all_results.append({
        "name": uploaded_file.name,
        "image": annotated,
        "detections": detections,
        "original": image,
    })

# Save to session state for other pages
st.session_state["detection_results"] = all_results

# ─── Summary ──────────────────────────────────────────────

if len(all_results) > 1:
    st.divider()
    st.markdown("### 📊 Batch Summary")

    total_defects = sum(len(r["detections"]) for r in all_results)
    images_with_defects = sum(1 for r in all_results if r["detections"])

    col1, col2, col3 = st.columns(3)
    col1.metric("Images Processed", len(all_results))
    col2.metric("Images with Defects", images_with_defects)
    col3.metric("Total Defects", total_defects)
