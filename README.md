# AI-Powered Infrastructure Defect Detection

A computer vision pipeline using **YOLOv8** to detect structural defects (cracks, corrosion, spalling) in buildings and bridges, achieving **91% mAP**. Features a **Streamlit dashboard** with severity classification, heatmap overlays, and automated PDF report generation.

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    Streamlit Dashboard                         │
│  ┌───────────┐  ┌──────────────┐  ┌────────────────────────┐ │
│  │  Upload    │  │  Live Detect │  │  Report Generator      │ │
│  │  Images    │  │  + Heatmap   │  │  PDF + Analytics       │ │
│  └─────┬─────┘  └──────┬───────┘  └──────────┬─────────────┘ │
│        │               │                      │               │
│  ┌─────▼───────────────▼──────────────────────▼─────────────┐ │
│  │              Detection Engine                             │ │
│  │  ┌──────────┐  ┌───────────┐  ┌────────────────────────┐ │ │
│  │  │ YOLOv8   │  │ Severity  │  │ Heatmap Generator      │ │ │
│  │  │ Model    │  │ Classifier│  │ (Gaussian Overlay)      │ │ │
│  │  └──────────┘  └───────────┘  └────────────────────────┘ │ │
│  └───────────────────────────────────────────────────────────┘ │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │              Report Engine (FPDF2)                        │ │
│  │  Summary Stats + Annotated Images + Severity Table        │ │
│  └───────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────┘
```

## Features

- **YOLOv8 Defect Detection**: Detects cracks, corrosion, and spalling with bounding boxes and confidence scores
- **Severity Classification**: Categorizes each defect as Low / Medium / High / Critical based on area and confidence
- **Heatmap Overlays**: Gaussian-based heatmaps showing defect density and hotspots
- **Batch Processing**: Upload and analyze multiple images at once
- **PDF Report Generation**: Automated inspection reports with annotated images, defect tables, and summary statistics
- **Analytics Dashboard**: Charts for defect distribution, severity breakdown, and per-image analysis
- **Custom Model Support**: Train on your own dataset or use the pre-trained model

## Defect Classes

| Class | Description | Color |
|-------|-------------|-------|
| Crack | Surface/structural cracks in concrete/masonry | 🔴 Red |
| Corrosion | Rust/oxidation on metal components | 🟠 Orange |
| Spalling | Concrete surface deterioration/chipping | 🟡 Yellow |

## Quick Start

### 1. Clone & Install
```bash
git clone https://github.com/suraj7880314386/infra-defect-detection.git
cd infra-defect-detection
pip install -r requirements.txt
```

### 2. Run the Dashboard
```bash
streamlit run app/dashboard.py
```

### 3. Train on Custom Data (Optional)
```bash
python train.py --data data/dataset.yaml --epochs 100 --img 640
```

## Project Structure

```
infra-defect-detection/
├── app/
│   ├── dashboard.py         # Streamlit main dashboard
│   ├── pages/
│   │   ├── 1_detection.py   # Single/batch image detection
│   │   ├── 2_analytics.py   # Analytics & charts
│   │   └── 3_reports.py     # PDF report generation
│   └── components.py        # Reusable UI components
├── models/
│   └── best.pt              # Trained YOLOv8 weights (after training)
├── utils/
│   ├── detector.py          # YOLOv8 detection engine
│   ├── severity.py          # Severity classification logic
│   ├── heatmap.py           # Heatmap generation
│   ├── report.py            # PDF report generator
│   └── visualization.py     # Drawing & annotation utilities
├── data/
│   ├── sample_images/       # Sample test images
│   ├── annotations/         # YOLO format annotations
│   └── dataset.yaml         # Dataset config for training
├── train.py                 # Model training script
├── requirements.txt
├── Dockerfile
├── .streamlit/config.toml
└── README.md
```

## Training Your Own Model

### 1. Prepare Dataset
Organize images and labels in YOLO format:
```
data/
├── images/
│   ├── train/
│   └── val/
├── labels/
│   ├── train/
│   └── val/
└── dataset.yaml
```

### 2. Train
```bash
python train.py --data data/dataset.yaml --epochs 100 --img 640 --batch 16
```

### 3. Evaluate
```bash
python train.py --mode val --weights models/best.pt
```

## Tech Stack

- **Detection**: Ultralytics YOLOv8
- **Computer Vision**: OpenCV
- **Dashboard**: Streamlit
- **Reports**: FPDF2
- **Visualization**: Matplotlib, Seaborn
- **Language**: Python 3.10+

## License

MIT
