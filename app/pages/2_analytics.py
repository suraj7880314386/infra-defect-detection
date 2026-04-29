"""Analytics Page — Charts and defect distribution analysis."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

st.set_page_config(page_title="Analytics", page_icon="📊", layout="wide")

st.title("📊 Defect Analytics")
st.caption("Visual analysis of detected defects across all images")

# ─── Load Results ─────────────────────────────────────────

results = st.session_state.get("detection_results", [])

if not results:
    st.warning("No detection results yet. Go to **Detection** page and upload images first.")
    st.stop()

# Flatten all detections
all_detections = []
for r in results:
    for det in r["detections"]:
        det_copy = det.copy()
        det_copy["image_name"] = r["name"]
        all_detections.append(det_copy)

if not all_detections:
    st.info("No defects were detected in any of the uploaded images.")
    st.stop()

# ─── Chart 1: Defect Type Distribution ───────────────────

st.markdown("### Defect Type Distribution")

col1, col2 = st.columns(2)

with col1:
    class_counts = {}
    for det in all_detections:
        cls = det["class_name"]
        class_counts[cls] = class_counts.get(cls, 0) + 1

    fig, ax = plt.subplots(figsize=(6, 4))
    colors = {"crack": "#FF4444", "corrosion": "#FF8C42", "spalling": "#FFD700"}
    classes = list(class_counts.keys())
    counts = list(class_counts.values())
    bar_colors = [colors.get(c, "#888") for c in classes]

    bars = ax.bar([c.capitalize() for c in classes], counts, color=bar_colors, edgecolor="white", linewidth=0.5)
    ax.set_ylabel("Count")
    ax.set_title("Defects by Type")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Add value labels on bars
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                str(count), ha="center", fontweight="bold")

    fig.patch.set_alpha(0)
    ax.set_facecolor("none")
    ax.tick_params(colors="white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    st.pyplot(fig)

with col2:
    # Pie chart
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.pie(
        counts,
        labels=[c.capitalize() for c in classes],
        colors=bar_colors,
        autopct="%1.1f%%",
        startangle=90,
        textprops={"color": "white"},
    )
    ax.set_title("Defect Distribution", color="white")
    fig.patch.set_alpha(0)
    st.pyplot(fig)

# ─── Chart 2: Severity Breakdown ─────────────────────────

st.divider()
st.markdown("### Severity Breakdown")

col1, col2 = st.columns(2)

with col1:
    severity_counts = {"Low": 0, "Medium": 0, "High": 0, "Critical": 0}
    for det in all_detections:
        sev = det.get("severity")
        if sev:
            sev_name = sev.value if hasattr(sev, 'value') else str(sev)
            severity_counts[sev_name] = severity_counts.get(sev_name, 0) + 1

    fig, ax = plt.subplots(figsize=(6, 4))
    sev_colors = {"Low": "#4CAF50", "Medium": "#FF9800", "High": "#F44336", "Critical": "#9C27B0"}
    sevs = list(severity_counts.keys())
    sev_vals = list(severity_counts.values())
    s_colors = [sev_colors[s] for s in sevs]

    bars = ax.barh(sevs, sev_vals, color=s_colors, edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Count")
    ax.set_title("Defects by Severity")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.invert_yaxis()

    for bar, val in zip(bars, sev_vals):
        if val > 0:
            ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height() / 2,
                    str(val), va="center", fontweight="bold", color="white")

    fig.patch.set_alpha(0)
    ax.set_facecolor("none")
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.title.set_color("white")
    st.pyplot(fig)

with col2:
    # Severity by defect type (stacked)
    fig, ax = plt.subplots(figsize=(6, 4))

    class_sev_data = {}
    for det in all_detections:
        cls = det["class_name"].capitalize()
        sev = det.get("severity")
        sev_name = sev.value if hasattr(sev, 'value') else str(sev) if sev else "N/A"

        if cls not in class_sev_data:
            class_sev_data[cls] = {"Low": 0, "Medium": 0, "High": 0, "Critical": 0}
        class_sev_data[cls][sev_name] += 1

    if class_sev_data:
        classes_list = list(class_sev_data.keys())
        x = np.arange(len(classes_list))
        width = 0.6
        bottom = np.zeros(len(classes_list))

        for sev_level in ["Low", "Medium", "High", "Critical"]:
            values = [class_sev_data[c].get(sev_level, 0) for c in classes_list]
            ax.bar(x, values, width, bottom=bottom, label=sev_level,
                   color=sev_colors[sev_level], edgecolor="white", linewidth=0.3)
            bottom += np.array(values)

        ax.set_xticks(x)
        ax.set_xticklabels(classes_list)
        ax.set_ylabel("Count")
        ax.set_title("Severity by Defect Type")
        ax.legend(loc="upper right", fontsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        fig.patch.set_alpha(0)
        ax.set_facecolor("none")
        ax.tick_params(colors="white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")

    st.pyplot(fig)

# ─── Chart 3: Confidence Distribution ────────────────────

st.divider()
st.markdown("### Confidence Distribution")

fig, ax = plt.subplots(figsize=(10, 4))
confidences = [d["confidence"] for d in all_detections]

ax.hist(confidences, bins=20, color="#FF4B4B", edgecolor="white", alpha=0.8)
ax.set_xlabel("Confidence Score")
ax.set_ylabel("Count")
ax.set_title("Detection Confidence Distribution")
ax.axvline(np.mean(confidences), color="#FFD700", linestyle="--",
           label=f"Mean: {np.mean(confidences):.2f}")
ax.legend()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.patch.set_alpha(0)
ax.set_facecolor("none")
ax.tick_params(colors="white")
ax.xaxis.label.set_color("white")
ax.yaxis.label.set_color("white")
ax.title.set_color("white")
st.pyplot(fig)

# ─── Chart 4: Defects Per Image ──────────────────────────

if len(results) > 1:
    st.divider()
    st.markdown("### Defects Per Image")

    fig, ax = plt.subplots(figsize=(10, 4))
    img_names = [r["name"][:20] for r in results]
    defect_counts = [len(r["detections"]) for r in results]

    bars = ax.bar(img_names, defect_counts, color="#FF4B4B", edgecolor="white")
    ax.set_ylabel("Defect Count")
    ax.set_title("Defects Per Image")
    plt.xticks(rotation=45, ha="right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.patch.set_alpha(0)
    ax.set_facecolor("none")
    ax.tick_params(colors="white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    fig.tight_layout()
    st.pyplot(fig)
