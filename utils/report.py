"""PDF Report Generator — creates professional inspection reports."""

import os
import tempfile
from datetime import datetime
from typing import List, Dict, Optional

import cv2
import numpy as np
from fpdf import FPDF

from utils.severity import Severity, get_severity_summary


class InspectionReport(FPDF):
    """Custom PDF report for infrastructure defect inspection."""

    def __init__(self):
        super().__init__()
        self.set_auto_page_break(auto=True, margin=20)

    def header(self):
        self.set_font("Helvetica", "B", 14)
        self.cell(0, 10, "Infrastructure Defect Inspection Report", align="C", new_x="LMARGIN", new_y="NEXT")
        self.set_font("Helvetica", "", 9)
        self.set_text_color(128, 128, 128)
        self.cell(0, 5, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", align="C", new_x="LMARGIN", new_y="NEXT")
        self.set_text_color(0, 0, 0)
        self.line(10, self.get_y() + 2, 200, self.get_y() + 2)
        self.ln(8)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")

    def add_section_title(self, title: str):
        self.set_font("Helvetica", "B", 12)
        self.set_fill_color(240, 240, 240)
        self.cell(0, 8, f"  {title}", fill=True, new_x="LMARGIN", new_y="NEXT")
        self.ln(4)

    def add_summary_section(self, all_detections: List[List[Dict]], image_names: List[str]):
        """Add executive summary section."""
        self.add_section_title("Executive Summary")

        total_defects = sum(len(d) for d in all_detections)
        total_images = len(all_detections)
        images_with_defects = sum(1 for d in all_detections if len(d) > 0)

        # Aggregate severity
        all_dets = [det for dets in all_detections for det in dets]
        severity_info = get_severity_summary(all_dets)

        self.set_font("Helvetica", "", 10)

        # Summary stats
        stats = [
            ("Images Analyzed", str(total_images)),
            ("Images with Defects", f"{images_with_defects}/{total_images}"),
            ("Total Defects Found", str(total_defects)),
            ("Highest Severity", severity_info["highest_severity"]),
            ("Critical Defects", str(severity_info["critical_count"])),
            ("High Severity", str(severity_info["high_count"])),
        ]

        for label, value in stats:
            self.set_font("Helvetica", "B", 10)
            self.cell(60, 7, f"{label}:", new_x="RIGHT")
            self.set_font("Helvetica", "", 10)
            self.cell(0, 7, value, new_x="LMARGIN", new_y="NEXT")

        self.ln(4)

        # Defect distribution
        class_counts = {}
        for det in all_dets:
            cls = det["class_name"]
            class_counts[cls] = class_counts.get(cls, 0) + 1

        if class_counts:
            self.set_font("Helvetica", "B", 10)
            self.cell(0, 7, "Defect Distribution:", new_x="LMARGIN", new_y="NEXT")
            self.set_font("Helvetica", "", 10)
            for cls, count in sorted(class_counts.items(), key=lambda x: -x[1]):
                pct = (count / total_defects * 100) if total_defects > 0 else 0
                self.cell(20, 6, "", new_x="RIGHT")
                self.cell(0, 6, f"{cls.capitalize()}: {count} ({pct:.1f}%)", new_x="LMARGIN", new_y="NEXT")

        self.ln(6)

    def add_image_result(
        self,
        image_name: str,
        annotated_image: np.ndarray,
        detections: List[Dict],
        image_index: int,
    ):
        """Add a single image analysis result to the report."""
        # Check if we need a new page
        if self.get_y() > 180:
            self.add_page()

        self.add_section_title(f"Image {image_index + 1}: {image_name}")

        # Save annotated image to temp file
        temp_path = os.path.join(tempfile.gettempdir(), f"report_img_{image_index}.jpg")
        cv2.imwrite(temp_path, annotated_image)

        # Add image
        try:
            img_width = 180
            self.image(temp_path, x=15, w=img_width)
            self.ln(4)
        except Exception as e:
            self.set_font("Helvetica", "I", 9)
            self.cell(0, 6, f"[Image could not be embedded: {str(e)}]", new_x="LMARGIN", new_y="NEXT")

        # Defect table
        if detections:
            self.set_font("Helvetica", "B", 9)

            # Table header
            col_widths = [15, 35, 30, 25, 35, 50]
            headers = ["#", "Defect Type", "Confidence", "Severity", "Area (px)", "Location (x,y,w,h)"]

            self.set_fill_color(60, 60, 60)
            self.set_text_color(255, 255, 255)
            for i, header in enumerate(headers):
                self.cell(col_widths[i], 7, header, border=1, fill=True, align="C")
            self.ln()
            self.set_text_color(0, 0, 0)

            # Table rows
            self.set_font("Helvetica", "", 8)
            for j, det in enumerate(detections):
                severity = det.get("severity")
                sev_name = severity.value if hasattr(severity, 'value') else str(severity) if severity else "N/A"
                x1, y1, x2, y2 = det["bbox"]
                location = f"({x1},{y1},{det['width']},{det['height']})"

                # Severity-based row color
                if sev_name == "Critical":
                    self.set_fill_color(255, 220, 220)
                elif sev_name == "High":
                    self.set_fill_color(255, 240, 220)
                else:
                    self.set_fill_color(255, 255, 255)

                row = [
                    str(j + 1),
                    det["class_name"].capitalize(),
                    f"{det['confidence'] * 100:.1f}%",
                    sev_name,
                    str(det["area"]),
                    location,
                ]

                for i, cell_text in enumerate(row):
                    self.cell(col_widths[i], 6, cell_text, border=1, fill=True, align="C")
                self.ln()
        else:
            self.set_font("Helvetica", "I", 10)
            self.set_text_color(0, 150, 0)
            self.cell(0, 7, "No defects detected in this image.", new_x="LMARGIN", new_y="NEXT")
            self.set_text_color(0, 0, 0)

        self.ln(6)

        # Cleanup temp file
        try:
            os.remove(temp_path)
        except OSError:
            pass


def generate_report(
    image_results: List[Dict],
    output_path: str = "reports/inspection_report.pdf",
) -> str:
    """
    Generate a complete PDF inspection report.

    Args:
        image_results: List of dicts with keys:
            - name: filename
            - image: annotated image (numpy array)
            - detections: list of detection dicts
        output_path: Where to save the PDF

    Returns:
        Path to generated PDF
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    pdf = InspectionReport()
    pdf.alias_nb_pages()
    pdf.add_page()

    # Summary
    all_detections = [r["detections"] for r in image_results]
    image_names = [r["name"] for r in image_results]
    pdf.add_summary_section(all_detections, image_names)

    # Individual results
    for i, result in enumerate(image_results):
        pdf.add_image_result(
            image_name=result["name"],
            annotated_image=result["image"],
            detections=result["detections"],
            image_index=i,
        )

    # Save
    pdf.output(output_path)
    return output_path
