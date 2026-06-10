"""
PDF Report Generator using ReportLab.
Produces clinical-grade formatted PDF reports.
"""

from __future__ import annotations

import base64
import io
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def generate_pdf_report(
    prediction: Dict[str, Any],
    patient_info: Optional[Dict] = None,
    study_info: Optional[Dict] = None,
    output_path: Optional[str] = None,
) -> bytes:
    """
    Generate a complete PDF clinical report.

    Args:
        prediction: Full prediction dict from predictor
        patient_info: Patient demographics dict
        study_info: Study metadata dict
        output_path: Optional path to save PDF file

    Returns:
        PDF bytes
    """
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm, mm
        from reportlab.lib import colors
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
            Image, HRFlowable, PageBreak
        )
        from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
        return _generate_with_reportlab(prediction, patient_info, study_info, output_path)
    except ImportError:
        logger.warning("reportlab not installed. Generating plain-text PDF fallback.")
        return _generate_plain_text_pdf(prediction, patient_info, output_path)


def _generate_with_reportlab(
    prediction: Dict,
    patient_info: Optional[Dict],
    study_info: Optional[Dict],
    output_path: Optional[str],
) -> bytes:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm, mm
    from reportlab.lib import colors
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        HRFlowable
    )
    from reportlab.lib.enums import TA_CENTER, TA_LEFT

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=2*cm, leftMargin=2*cm,
        topMargin=2*cm, bottomMargin=2*cm,
        title="Lumbar Spine MRI Report - ATM-Net++",
    )

    styles = getSampleStyleSheet()
    # Custom styles
    title_style = ParagraphStyle(
        "Title", parent=styles["Heading1"],
        fontSize=18, textColor=colors.HexColor("#1a3a5c"),
        spaceAfter=6, alignment=TA_CENTER,
    )
    subtitle_style = ParagraphStyle(
        "Subtitle", parent=styles["Normal"],
        fontSize=10, textColor=colors.HexColor("#555555"),
        spaceAfter=4, alignment=TA_CENTER,
    )
    section_style = ParagraphStyle(
        "Section", parent=styles["Heading2"],
        fontSize=12, textColor=colors.HexColor("#1a3a5c"),
        spaceBefore=12, spaceAfter=4,
        borderPad=4,
    )
    body_style = ParagraphStyle(
        "Body", parent=styles["Normal"],
        fontSize=10, leading=14, spaceAfter=6,
    )
    label_style = ParagraphStyle(
        "Label", parent=styles["Normal"],
        fontSize=9, textColor=colors.HexColor("#666666"),
    )
    value_style = ParagraphStyle(
        "Value", parent=styles["Normal"],
        fontSize=10, fontName="Helvetica-Bold",
    )
    ai_style = ParagraphStyle(
        "AI", parent=styles["Normal"],
        fontSize=8, textColor=colors.HexColor("#888888"),
        alignment=TA_CENTER,
    )

    story = []
    report_data = prediction.get("report", {})
    cls_data = prediction.get("classification", {})
    sev_data = prediction.get("severity", {})
    lvl_data = prediction.get("levels", {})

    # ── Header ────────────────────────────────────────────────────────
    story.append(Paragraph("ATM-Net++", title_style))
    story.append(Paragraph("Anatomy-Aware Multimodal Lumbar Spine MRI Diagnostic Report", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor("#1a3a5c")))
    story.append(Spacer(1, 8))

    # ── Report metadata ───────────────────────────────────────────────
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    meta_data = [
        ["Report Date:", now, "Model Version:", "ATM-Net++ v1.0.0"],
        ["Modality:", study_info.get("modality", "T2") if study_info else "T2",
         "AI Confidence:", f"{cls_data.get('confidence', 0)*100:.1f}%"],
    ]
    meta_table = Table(meta_data, colWidths=[3*cm, 5*cm, 4*cm, 5*cm])
    meta_table.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#666666")),
        ("TEXTCOLOR", (2, 0), (2, -1), colors.HexColor("#666666")),
        ("FONTNAME", (1, 0), (1, -1), "Helvetica-Bold"),
        ("FONTNAME", (3, 0), (3, -1), "Helvetica-Bold"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 8))

    # ── Patient Information ───────────────────────────────────────────
    if patient_info:
        story.append(Paragraph("PATIENT INFORMATION", section_style))
        story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#cccccc")))
        story.append(Spacer(1, 4))

        pat_data = [
            ["Patient ID:", patient_info.get("patient_code", "N/A"),
             "Sex:", patient_info.get("sex", "N/A")],
            ["Age:", f"{patient_info.get('age', 'N/A')} years",
             "BMI:", f"{patient_info.get('bmi', 'N/A')} kg/m²"],
            ["Height:", f"{patient_info.get('height_cm', 'N/A')} cm",
             "Weight:", f"{patient_info.get('weight_kg', 'N/A')} kg"],
        ]
        if patient_info.get("clinical_symptoms"):
            pat_data.append(["Symptoms:", patient_info["clinical_symptoms"], "", ""])

        pat_table = Table(pat_data, colWidths=[3*cm, 5*cm, 3*cm, 6*cm])
        pat_table.setStyle(TableStyle([
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#666666")),
            ("TEXTCOLOR", (2, 0), (2, -1), colors.HexColor("#666666")),
            ("FONTNAME", (1, 0), (1, -1), "Helvetica-Bold"),
            ("FONTNAME", (3, 0), (3, -1), "Helvetica-Bold"),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ]))
        story.append(pat_table)
        story.append(Spacer(1, 12))

    # ── Primary Diagnosis ─────────────────────────────────────────────
    story.append(Paragraph("PRIMARY DIAGNOSIS", section_style))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#cccccc")))
    story.append(Spacer(1, 4))

    disease_name = cls_data.get("disease_name", "Unknown").replace("_", " ")
    severity_name = sev_data.get("name", "Unknown")
    confidence = cls_data.get("confidence", 0)
    pfirrmann = prediction.get("pfirrmann_grade", 0)
    affected = lvl_data.get("affected", [])

    # Confidence color
    conf_color = (
        colors.HexColor("#22c55e") if confidence >= 0.8
        else colors.HexColor("#f59e0b") if confidence >= 0.6
        else colors.HexColor("#ef4444")
    )

    diag_data = [
        ["Diagnosis:", disease_name, "Severity:", severity_name],
        ["Confidence:", f"{confidence*100:.1f}%", "Pfirrmann Grade:", f"{pfirrmann:.1f}/5"],
        ["Affected Levels:", ", ".join(affected) if affected else "None", "", ""],
    ]
    diag_table = Table(diag_data, colWidths=[3.5*cm, 6.5*cm, 3.5*cm, 4.5*cm])
    diag_table.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#666666")),
        ("TEXTCOLOR", (2, 0), (2, -1), colors.HexColor("#666666")),
        ("FONTNAME", (1, 0), (1, -1), "Helvetica-Bold"),
        ("FONTNAME", (3, 0), (3, -1), "Helvetica-Bold"),
        ("TEXTCOLOR", (1, 0), (1, 0), colors.HexColor("#1a3a5c")),
        ("TEXTCOLOR", (1, 1), (1, 1), conf_color),
        ("FONTSIZE", (1, 0), (1, 0), 12),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(diag_table)
    story.append(Spacer(1, 12))

    # ── Probability Distribution ──────────────────────────────────────
    story.append(Paragraph("DISEASE PROBABILITY DISTRIBUTION", section_style))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#cccccc")))
    story.append(Spacer(1, 4))

    prob_data = [["Disease", "Probability", "Bar"]]
    for disease, prob in cls_data.get("disease_probabilities", {}).items():
        bar = "█" * int(prob * 30)
        prob_data.append([
            disease.replace("_", " "),
            f"{prob*100:.1f}%",
            bar,
        ])
    prob_table = Table(prob_data, colWidths=[5*cm, 3*cm, 10*cm])
    prob_table.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a3a5c")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8f9fa")]),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#dddddd")),
        ("FONTNAME", (2, 1), (2, -1), "Courier"),
        ("TEXTCOLOR", (2, 1), (2, -1), colors.HexColor("#1a3a5c")),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(prob_table)
    story.append(Spacer(1, 12))

    # ── Findings ─────────────────────────────────────────────────────
    story.append(Paragraph("RADIOLOGICAL FINDINGS", section_style))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#cccccc")))
    story.append(Spacer(1, 4))
    story.append(Paragraph(report_data.get("findings", "No findings available."), body_style))
    story.append(Spacer(1, 8))

    # ── Impression ───────────────────────────────────────────────────
    story.append(Paragraph("IMPRESSION", section_style))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#cccccc")))
    story.append(Spacer(1, 4))
    story.append(Paragraph(report_data.get("impression", ""), body_style))
    story.append(Spacer(1, 8))

    # ── Recommendation ───────────────────────────────────────────────
    story.append(Paragraph("RECOMMENDATION", section_style))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#cccccc")))
    story.append(Spacer(1, 4))
    story.append(Paragraph(report_data.get("recommendation", ""), body_style))
    story.append(Spacer(1, 20))

    # ── Footer ────────────────────────────────────────────────────────
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#1a3a5c")))
    story.append(Spacer(1, 4))
    story.append(Paragraph(
        "⚠ This report is generated by ATM-Net++ AI system and must be reviewed and validated by a qualified radiologist before clinical use. "
        "AI-generated reports do not replace professional medical judgment.",
        ai_style,
    ))

    doc.build(story)
    pdf_bytes = buffer.getvalue()
    buffer.close()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(pdf_bytes)

    return pdf_bytes


def _generate_plain_text_pdf(prediction: Dict, patient_info: Optional[Dict], output_path: Optional[str]) -> bytes:
    """Minimal PDF fallback using only stdlib."""
    report = prediction.get("report", {})
    text = report.get("report_text", "No report available.")
    # Return UTF-8 encoded text as bytes (not a real PDF but functional fallback)
    content = f"%PDF-1.4\n% ATM-Net++ Report\n{text}\n%%EOF".encode("utf-8")
    if output_path:
        with open(output_path, "wb") as f:
            f.write(content)
    return content
