"""Reports Page — Generate and download PDF inspection reports."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import streamlit as st
from datetime import datetime

from utils.report import generate_report
from utils.severity import get_severity_summary

st.set_page_config(page_title="Reports", page_icon="📄", layout="wide")

st.title("📄 Inspection Report Generator")
st.caption("Generate professional PDF reports from detection results")

# ─── Load Results ─────────────────────────────────────────

results = st.session_state.get("detection_results", [])

if not results:
    st.warning("No detection results yet. Go to **Detection** page and upload images first.")
    st.stop()

# ─── Report Preview ───────────────────────────────────────

st.markdown("### 📋 Report Preview")

# Summary stats
total_images = len(results)
total_defects = sum(len(r["detections"]) for r in results)
images_with_defects = sum(1 for r in results if r["detections"])

all_dets = [det for r in results for det in r["detections"]]
severity_info = get_severity_summary(all_dets) if all_dets else {
    "highest_severity": "N/A",
    "critical_count": 0,
    "high_count": 0,
    "distribution": {},
}

col1, col2, col3, col4 = st.columns(4)
col1.metric("Images Analyzed", total_images)
col2.metric("Total Defects", total_defects)
col3.metric("Critical Issues", severity_info["critical_count"])
col4.metric("Highest Severity", severity_info["highest_severity"])

st.divider()

# Per-image summary
st.markdown("### 📷 Image Summary")
for i, result in enumerate(results):
    dets = result["detections"]
    n_defects = len(dets)

    if n_defects == 0:
        status = "✅ No defects"
    else:
        sev_info = get_severity_summary(dets)
        highest = sev_info["highest_severity"]
        if highest in ("Critical", "High"):
            status = f"🔴 {n_defects} defects ({highest} severity)"
        else:
            status = f"🟡 {n_defects} defects ({highest} severity)"

    st.markdown(f"**{i + 1}. {result['name']}** — {status}")

# ─── Generate Report ─────────────────────────────────────

st.divider()
st.markdown("### 📥 Generate PDF Report")

report_title = st.text_input("Report Title (optional)", value="Infrastructure Inspection Report")

col1, col2 = st.columns([1, 3])
with col1:
    generate_btn = st.button("🔄 Generate Report", type="primary", use_container_width=True)

if generate_btn:
    with st.spinner("Generating PDF report..."):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"reports/inspection_report_{timestamp}.pdf"

        try:
            report_path = generate_report(
                image_results=results,
                output_path=output_path,
            )

            st.success(f"Report generated successfully!")

            # Download button
            with open(report_path, "rb") as f:
                pdf_bytes = f.read()

            st.download_button(
                label="📥 Download PDF Report",
                data=pdf_bytes,
                file_name=f"inspection_report_{timestamp}.pdf",
                mime="application/pdf",
                type="primary",
            )

        except Exception as e:
            st.error(f"Report generation failed: {str(e)}")
            st.exception(e)

# ─── Report Info ──────────────────────────────────────────

st.divider()
st.markdown("### ℹ️ Report Contents")
st.markdown("""
The generated PDF report includes:

- **Executive Summary** — Total defects, severity distribution, and defect type breakdown
- **Per-Image Analysis** — Annotated images with bounding boxes and a detailed defect table
- **Defect Table** — Each defect with type, confidence, severity, area, and location
- **Severity Color Coding** — Critical and High severity rows are highlighted
""")
