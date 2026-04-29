"""
Infrastructure Defect Detection — Streamlit Dashboard

Run with: streamlit run app/dashboard.py
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st

# ─── Page Config ──────────────────────────────────────────

st.set_page_config(
    page_title="Infrastructure Defect Detection",
    page_icon="🏗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────

st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #FF4B4B, #FF8C42);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #888;
        margin-top: -10px;
    }
    .stat-card {
        background: #1E2130;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        border: 1px solid #2D3348;
    }
    .stat-value {
        font-size: 2rem;
        font-weight: 700;
        color: #FF4B4B;
    }
    .stat-label {
        font-size: 0.85rem;
        color: #888;
    }
</style>
""", unsafe_allow_html=True)

# ─── Sidebar ─────────────────────────────────────────────

with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/structural-analysis.png", width=60)
    st.title("Defect Detection")
    st.caption("AI-Powered Infrastructure Inspection")

    st.divider()

    st.markdown("### ⚙️ Model Settings")
    conf_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.1,
        max_value=0.95,
        value=0.25,
        step=0.05,
        help="Minimum confidence for a detection to be shown",
    )
    st.session_state["conf_threshold"] = conf_threshold

    st.divider()

    st.markdown("### 📊 Defect Classes")
    st.markdown("""
    - 🔴 **Crack** — Surface/structural cracks
    - 🟠 **Corrosion** — Rust/oxidation
    - 🟡 **Spalling** — Concrete deterioration
    """)

    st.divider()
    st.caption("Built with YOLOv8 + Streamlit")

# ─── Main Content ─────────────────────────────────────────

st.markdown('<p class="main-header">🏗️ Infrastructure Defect Detection</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">AI-powered structural defect detection using YOLOv8 computer vision</p>', unsafe_allow_html=True)

st.divider()

# Feature cards
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div class="stat-card">
        <div class="stat-value">🔍</div>
        <div class="stat-label">Detect Defects</div>
        <p style="font-size:0.8rem; color:#aaa;">Upload images to detect cracks, corrosion, and spalling</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="stat-card">
        <div class="stat-value">🗺️</div>
        <div class="stat-label">Heatmap Overlay</div>
        <p style="font-size:0.8rem; color:#aaa;">Visualize defect density with Gaussian heatmaps</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="stat-card">
        <div class="stat-value">📊</div>
        <div class="stat-label">Analytics</div>
        <p style="font-size:0.8rem; color:#aaa;">Charts for defect distribution and severity breakdown</p>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="stat-card">
        <div class="stat-value">📄</div>
        <div class="stat-label">PDF Reports</div>
        <p style="font-size:0.8rem; color:#aaa;">Automated inspection reports with findings</p>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# Quick start
st.markdown("### 🚀 Quick Start")
st.markdown("""
1. **Navigate to Detection** (sidebar) → Upload building/bridge images
2. **Review results** with annotated images and severity classification
3. **View Analytics** for charts and defect distribution
4. **Generate Report** to download a PDF inspection report
""")

# How it works
st.markdown("### 🧠 How It Works")

col_a, col_b, col_c = st.columns(3)

with col_a:
    st.markdown("**1. Upload & Preprocess**")
    st.caption("Upload images of buildings, bridges, or infrastructure. Images are preprocessed and resized for the detection model.")

with col_b:
    st.markdown("**2. YOLOv8 Detection**")
    st.caption("The trained YOLOv8 model detects and localizes cracks, corrosion, and spalling with bounding boxes and confidence scores.")

with col_c:
    st.markdown("**3. Severity + Report**")
    st.caption("Each defect is classified by severity (Low → Critical). Results are compiled into an automated PDF inspection report.")
