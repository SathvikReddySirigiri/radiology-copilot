"""Generate project summary Word document."""

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt
from docx.oxml.ns import qn
from docx.oxml import OxmlElement


def set_cell_shading(cell, fill_hex: str) -> None:
    shading = OxmlElement("w:shd")
    shading.set(qn("w:fill"), fill_hex)
    cell._tc.get_or_add_tcPr().append(shading)


def add_table(doc, headers, rows):
    table = doc.add_table(rows=1, cols=len(headers))
    table.style = "Table Grid"
    hdr = table.rows[0].cells
    for i, text in enumerate(headers):
        hdr[i].text = text
        set_cell_shading(hdr[i], "D9E2F3")
        for p in hdr[i].paragraphs:
            for run in p.runs:
                run.bold = True
    for row in rows:
        cells = table.add_row().cells
        for i, text in enumerate(row):
            cells[i].text = text
    doc.add_paragraph()


def main():
    doc = Document()

    # Title page
    title = doc.add_heading("Radiology Report Copilot", 0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    sub = doc.add_paragraph("Project Summary Document")
    sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    sub.runs[0].bold = True
    sub.runs[0].font.size = Pt(14)
    doc.add_paragraph("June 2026").alignment = WD_ALIGN_PARAGRAPH.CENTER
    doc.add_page_break()

    # 1. Executive Summary
    doc.add_heading("1. Executive Summary", 1)
    doc.add_paragraph(
        "Radiology Report Copilot is a local-first AI application for chest X-ray analysis. "
        "A radiologist or researcher uploads a chest X-ray image; the system automatically scores "
        "pathologies, generates explainability visualizations, assigns a Lung-RADS risk category, "
        "drafts a structured clinical report, and validates the output before export."
    )
    doc.add_paragraph(
        "All processing runs entirely on the local machine — no patient data is sent to external "
        "cloud APIs. Vision models, language models, and the web interface are orchestrated through "
        "a single Streamlit application."
    )
    p = doc.add_paragraph()
    p.add_run("Primary goal: ").bold = True
    p.add_run(
        "Demonstrate an end-to-end radiology AI workflow that combines computer vision, "
        "explainability, clinical risk scoring, and natural language report generation "
        "in a research and educational setting."
    )

    # 2. Problem Statement
    doc.add_heading("2. Problem Statement", 1)
    doc.add_paragraph("Chest X-ray interpretation is time-consuming and requires specialized expertise. AI can assist by:")
    for item in [
        "Flagging likely pathologies with confidence scores",
        "Showing where the model is looking (explainability)",
        "Standardizing report structure (FINDINGS / IMPRESSION / RECOMMENDATIONS)",
        "Applying clinical guidelines (e.g., ACR Lung-RADS 1.1)",
        "Reducing common LLM errors through rule-based quality checks",
    ]:
        doc.add_paragraph(item, style="List Bullet")
    doc.add_paragraph("This project implements that workflow as a cohesive, locally runnable system.")

    # 3. System Architecture
    doc.add_heading("3. System Architecture", 1)
    doc.add_paragraph(
        "The application uses a 7-stage LangGraph pipeline orchestrated in app/pipeline/graph.py:"
    )
    add_table(
        doc,
        ["Step", "Module", "Function"],
        [
            ("1", "app/vision/torchxray.py", "Score 14 chest pathologies (DenseNet121)"),
            ("2", "app/vision/torchxray.py", "Generate Grad-CAM heatmaps for explainability"),
            ("3", "app/pipeline/lung_rads.py", "Assign Lung-RADS 1.1 category (1, 2, 3, 4A, 4B, 4X)"),
            ("4", "app/vision/llava_client.py", "Produce visual description (LLaVA via Ollama)"),
            ("5", "app/utils/ner.py", "Extract clinical entities with scispaCy"),
            ("6", "app/pipeline/drafter.py", "Draft structured report with Llama 3.1 8B"),
            ("7", "app/pipeline/qa_agent.py", "Rule-based validation before final export"),
        ],
    )
    doc.add_paragraph(
        "Flow: Upload X-ray → Vision Model → Grad-CAM + Lung-RADS → LLaVA Description → NER "
        "→ Report Drafter → QA Agent → Final Report"
    )
    doc.add_paragraph("The Streamlit UI (app/main.py) is the single entry point for upload, analysis, visualization, and download.")

    # 4. Components Built
    doc.add_heading("4. Components Built", 1)

    doc.add_heading("4.1 Vision & Pathology Detection (app/vision/torchxray.py)", 2)
    for item in [
        "Uses TorchXRayVision with a DenseNet121 backbone trained on multiple chest X-ray datasets",
        "Returns confidence scores for 18 pathologies (e.g., Mass, Nodule, Effusion, Pneumothorax, Cardiomegaly)",
        "Implements Grad-CAM via pytorch-grad-cam targeting model.features.denseblock4",
        "Exposes get_pathology_labels(), get_heatmap(), and get_all_heatmaps() for multi-label views",
    ]:
        doc.add_paragraph(item, style="List Bullet")

    doc.add_heading("4.2 Bounding Box & Anatomical Labeling (app/vision/bbox.py)", 2)
    for item in [
        "Extracts bounding boxes from Grad-CAM activation maps using OpenCV contour detection",
        "Maps box centers to a 3×3 anatomical grid (e.g., Right Upper Lobe, Cardiac Silhouette, Left Lower Lobe)",
        "Assigns risk levels (high / medium / low) from model confidence",
        "Draws color-coded boxes on the X-ray with pathology label, confidence, and region name",
        "Main entry point: annotate_xray() — returns annotated image, box list, and text summary",
    ]:
        doc.add_paragraph(item, style="List Bullet")

    doc.add_heading("4.3 LLaVA Visual Description (app/vision/llava_client.py)", 2)
    for item in [
        "Calls LLaVA through local Ollama (/api/generate)",
        "Resizes images to 512px RGB for consistent inference",
        "Returns a free-text clinical description of visible findings",
        "Includes graceful fallbacks when Ollama is unavailable",
    ]:
        doc.add_paragraph(item, style="List Bullet")

    doc.add_heading("4.4 Lung-RADS Scoring (app/pipeline/lung_rads.py)", 2)
    for item in [
        "Rule-based ACR Lung-RADS 1.1 implementation — no LLM required",
        "Categories: 1, 2, 3, 4A, 4B, 4X",
        "Outputs malignancy risk range, follow-up interval, urgency flag, and recommended actions",
        "Supports Mass and Nodule findings with score-based thresholds",
    ]:
        doc.add_paragraph(item, style="List Bullet")

    doc.add_heading("4.5 Clinical Entity Extraction (app/utils/ner.py)", 2)
    doc.add_paragraph("Uses scispaCy model en_core_sci_sm to extract structured entities from LLaVA text for the report drafter.")

    doc.add_heading("4.6 Report Drafter (app/pipeline/drafter.py)", 2)
    for item in [
        "Generates structured reports in FINDINGS / IMPRESSION / RECOMMENDATIONS format using Llama 3.1 8B via Ollama",
        "detect_conflict() — flags when LLaVA says normal but pathology scores are high (>=0.70); drafter prioritizes quantitative scores",
        "generate_patient_summary() — converts clinical reports into plain-English patient-facing summaries with sections: WHAT WE FOUND, WHAT THIS MEANS FOR YOU, WHAT HAPPENS NEXT, WHEN TO SEEK IMMEDIATE HELP",
    ]:
        doc.add_paragraph(item, style="List Bullet")

    doc.add_heading("4.7 QA Agent (app/pipeline/qa_agent.py)", 2)
    for item in [
        "Rule-based validation — no additional LLM call",
        "Checks for required sections (FINDINGS, IMPRESSION, RECOMMENDATIONS)",
        "Flags sparse findings, score/report mismatches, and common hallucinations",
        "Returns pass/fail status with specific flags",
    ]:
        doc.add_paragraph(item, style="List Bullet")

    doc.add_heading("4.8 Evaluation Utilities (app/utils/evaluator.py)", 2)
    doc.add_paragraph("CheXpert label mapping, threshold-based predictions, and metrics computation for benchmarking.")

    doc.add_heading("4.9 Pipeline Orchestration (app/pipeline/graph.py)", 2)
    doc.add_paragraph(
        "LangGraph state machine with typed ReportState. Nodes: run_vision → run_heatmap → "
        "run_lung_rads → run_llava → run_ner → run_drafter → run_qa → finalize. "
        "Final report includes timestamp, top-5 pathology score bars, and clinical disclaimer."
    )

    # 5. UI Features
    doc.add_heading("5. User Interface Features (app/main.py)", 1)
    add_table(
        doc,
        ["Feature", "Description"],
        [
            ("File upload", "PNG, JPG, or DICOM chest X-rays"),
            ("Progress tracking", "7-step pipeline progress bar with status toasts"),
            ("Pathology scores", "Expandable table of all detected pathologies"),
            ("Explainability Explorer", "Switch between 8 pathology labels to view Grad-CAM heatmaps live"),
            ("Lesion Localization", "Grad-CAM / Bounding Box / Side-by-Side views; detection table; threshold slider; annotated image download"),
            ("Lung-RADS Assessment", "Category, risk level, follow-up timeline, and action items"),
            ("LLaVA description", "Expandable visual findings text"),
            ("Final report", "Downloadable timestamped .txt clinical report"),
            ("QA results", "Pass/fail validation with specific flags"),
            ("Sidebar summary", "Top findings, highest confidence, Lung-RADS category"),
        ],
    )

    # 6. Technology Stack
    doc.add_heading("6. Technology Stack", 1)
    add_table(
        doc,
        ["Layer", "Technology"],
        [
            ("Vision classification", "TorchXRayVision (DenseNet121)"),
            ("Explainability", "pytorch-grad-cam, OpenCV"),
            ("Vision-language", "LLaVA (Ollama)"),
            ("Report generation", "Llama 3.1 8B (Ollama)"),
            ("Orchestration", "LangGraph"),
            ("Clinical NER", "scispaCy en_core_sci_sm"),
            ("Risk scoring", "Custom Lung-RADS 1.1 engine"),
            ("Web UI", "Streamlit"),
            ("HTTP client", "httpx"),
            ("Testing", "pytest"),
        ],
    )
    doc.add_paragraph("Environment: Python 3.10+, Windows 10/11, ~16 GB RAM recommended, ~8 GB disk for models.")

    # 7. Development Work
    doc.add_heading("7. Development Work Completed", 1)
    for item in [
        "Core pipeline — Full LangGraph workflow from image upload to final report",
        "Grad-CAM explainability — Single and multi-label heatmap generation",
        "Bounding box localization — Anatomical region labeling from CAM arrays (bbox.py)",
        "Lung-RADS integration — Automated risk categorization in pipeline and UI",
        "LLaVA integration — Switched to Ollama /api/generate for reliable local inference",
        "Conflict detection — High-confidence pathology scores override normal LLaVA descriptions in drafting",
        "QA validation — Rule-based checks before report export",
        "Patient summary function — generate_patient_summary() for plain-English output",
        "Windows compatibility — UTF-8 encoding fixes, numpy 1.24.4 pin for scispaCy",
    ]:
        doc.add_paragraph(item, style="List Number")

    doc.add_heading("Testing Performed", 2)
    add_table(
        doc,
        ["Test", "Result"],
        [
            ("bbox.py anatomical region mapping", "Passed (Right Upper Lobe, Cardiac Silhouette, Left Lower Lobe)"),
            ("bbox.py bounding box extraction", "Passed (2 boxes from synthetic CAM)"),
            ("generate_patient_summary import", "Passed"),
            ("Patient summary generation (Ollama)", "Fallback on timeout when Ollama slow; succeeds when Ollama is responsive"),
            ("Streamlit UI", "Running on http://localhost:8509"),
        ],
    )

    # 8. How to Run
    doc.add_heading("8. How to Run", 1)
    doc.add_heading("Prerequisites", 2)
    doc.add_paragraph("ollama pull llava")
    doc.add_paragraph("ollama pull llama3.1:8b")
    doc.add_heading("Install", 2)
    doc.add_paragraph("pip install -r requirements.txt")
    doc.add_paragraph(
        "pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.3/en_core_sci_sm-0.5.3.tar.gz"
    )
    doc.add_heading("Start (Windows)", 2)
    doc.add_paragraph('$env:PYTHONIOENCODING = "utf-8"')
    doc.add_paragraph("python -m streamlit run app/main.py")
    doc.add_paragraph(
        "Usage: Open the local URL → upload chest X-ray → click Run Analysis → "
        "review scores, heatmaps, bounding boxes, Lung-RADS, and download the report."
    )

    # 9. Limitations
    doc.add_heading("9. Limitations & Disclaimer", 1)
    for item in [
        "Research and educational use only — not FDA-cleared, not a medical device",
        "AI findings must be verified by a licensed radiologist before any clinical use",
        "LLM outputs may contain errors; QA agent reduces but does not eliminate risk",
        "Ollama model load times can cause timeouts on first run (~2 minutes warm-up observed)",
        "Bounding boxes are derived from Grad-CAM activations, not a dedicated object detector — approximate localization only",
        "Lung-RADS scoring uses model confidence as a proxy; not a substitute for radiologist measurement",
    ]:
        doc.add_paragraph(item, style="List Bullet")

    # 10. File Structure
    doc.add_heading("10. Project File Structure", 1)
    structure = """radiology-copilot/
├── app/
│   ├── main.py                 # Streamlit UI
│   ├── vision/
│   │   ├── torchxray.py        # Pathology scores + Grad-CAM
│   │   ├── llava_client.py     # LLaVA via Ollama
│   │   └── bbox.py             # Bounding boxes + anatomical labels
│   ├── pipeline/
│   │   ├── graph.py            # LangGraph orchestration
│   │   ├── drafter.py          # Report + patient summary generation
│   │   ├── qa_agent.py         # Rule-based validation
│   │   └── lung_rads.py        # Lung-RADS 1.1 scorer
│   └── utils/
│       ├── ner.py              # scispaCy entity extraction
│       └── evaluator.py        # CheXpert benchmark metrics
├── tests/
├── requirements.txt
└── README.md"""
    p = doc.add_paragraph()
    run = p.add_run(structure)
    run.font.name = "Consolas"
    run.font.size = Pt(9)

    # 11. Conclusion
    doc.add_heading("11. Conclusion", 1)
    doc.add_paragraph(
        "Radiology Report Copilot delivers a complete, locally runnable chest X-ray analysis workflow. "
        "It combines quantitative pathology scoring, visual explainability (Grad-CAM and bounding boxes), "
        "clinical risk stratification (Lung-RADS), natural language reporting, and automated quality checks "
        "into a single Streamlit application suitable for research demonstrations and educational use."
    )

    doc.add_paragraph()
    footer = doc.add_paragraph("Document prepared: June 2026  |  Project: Radiology Report Copilot  |  License: MIT")
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    footer.runs[0].italic = True

    out_path = "Radiology_Report_Copilot_Summary.docx"
    doc.save(out_path)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
