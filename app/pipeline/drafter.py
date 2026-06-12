"""Radiology report drafter using a local LLM via Ollama.

Takes vision model outputs and generates structured radiology report text
following standard reporting conventions.
"""

import os
from typing import Optional

import httpx
from dotenv import load_dotenv

from app.pipeline.lung_rads import LungRADSResult, get_lung_rads_display

load_dotenv()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
MODEL_LLM = os.getenv("MODEL_LLM", "llama3.1:8b")
REQUEST_TIMEOUT = 120.0

_NORMAL_PHRASES = (
    "appears to be normal",
    "no significant abnormalities",
    "no obvious acute process",
    "lungs are clear",
    "no masses",
    "no nodules",
    "does not show any abnormalities",
)


def detect_conflict(llava_description: str, pathology_labels: dict) -> bool:
    """Return True if LLaVA contradicts high-confidence pathology scores."""
    high_confidence = {
        label: score
        for label, score in pathology_labels.items()
        if score > 0.65
    }

    if not high_confidence:
        return False

    llava_lower = llava_description.lower()
    conflict_detected = any(phrase in llava_lower for phrase in _NORMAL_PHRASES)

    return conflict_detected and len(high_confidence) > 0


def _build_prompt(
    pathology_labels: dict,
    llava_description: str,
    ner_entities: dict,
    lung_rads: Optional[LungRADSResult] = None,
) -> str:
    prompt = (
        "You are an expert radiologist. Generate a structured chest "
        "X-ray report using ONLY the information provided below. "
        "Do not invent findings not supported by the inputs.\n\n"
        f"Model pathology scores: {pathology_labels}\n"
        f"Visual description: {llava_description}\n"
        f"Extracted entities: {ner_entities}\n"
    )

    if lung_rads is not None:
        display = get_lung_rads_display(lung_rads)
        action_items = "\n".join(display["action_items"])
        urgent_label = (
            "YES — IMMEDIATE ACTION" if lung_rads.urgent else "No"
        )
        follow_up = (
            f"{lung_rads.follow_up_months} months"
            if lung_rads.follow_up_months > 0
            else "IMMEDIATE"
        )

        lung_rads_block = f"""RECOMMENDATIONS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LUNG-RADS ASSESSMENT (ACR v1.1)
Category    : {lung_rads.category_str}
Risk        : {lung_rads.malignancy_risk} malignancy risk
Follow-up   : {follow_up}
Urgent      : {urgent_label}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Clinical Actions:
{action_items}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DISCLAIMER: This AI-assisted report is intended to support,
not replace, qualified radiologist review. All findings must
be verified by a licensed radiologist before clinical use.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"""

        prompt += (
            f"\nLung-RADS Assessment: Category {lung_rads.category_str}\n"
            f"Malignancy Risk: {lung_rads.malignancy_risk}\n"
            f"Clinical Recommendation: {lung_rads.recommendation}\n\n"
            "The RECOMMENDATIONS section MUST end with the exact "
            "Lung-RADS block below. Do not paraphrase it.\n\n"
            f"{lung_rads_block}\n"
        )

    prompt += (
        "\nWrite the report in FINDINGS / IMPRESSION / RECOMMENDATIONS "
        "format."
    )
    return prompt


def _raise_for_ollama_error(response: httpx.Response) -> None:
    if response.is_success:
        return

    error_text = response.text.lower()
    if response.status_code == 404 or "not found" in error_text:
        raise RuntimeError(
            f"LLM model not pulled. Run: ollama pull {MODEL_LLM}"
        )

    response.raise_for_status()


def _unload_model(client: httpx.Client, model: str) -> None:
    client.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json={"model": model, "prompt": "", "keep_alive": 0},
        timeout=10.0,
    )


def draft_report(
    pathology_labels: dict,
    llava_description: str,
    ner_entities: dict,
    lung_rads: Optional[LungRADSResult] = None,
) -> str:
    """Generate a structured radiology report from pipeline inputs."""
    high_confidence = {
        label: score
        for label, score in pathology_labels.items()
        if score > 0.5
    }
    high_confidence_findings = {
        label: score
        for label, score in pathology_labels.items()
        if score > 0.65
    }
    conflict = detect_conflict(llava_description, pathology_labels)

    prompt = _build_prompt(
        pathology_labels,
        llava_description,
        ner_entities,
        lung_rads=lung_rads,
    )

    if conflict:
        prompt = (
            "IMPORTANT CONFLICT NOTICE: The visual description says "
            "the image appears normal, BUT the AI vision model detected "
            "the following with HIGH confidence:\n"
            f"{high_confidence_findings}\n\n"
            "In this case:\n"
            "- TRUST the quantitative vision model scores over the "
            "visual description\n"
            "- DO NOT use phrases like 'in addition to normal findings'\n"
            "- The FINDINGS section must clearly state the high-confidence "
            "detections\n"
            "- The IMPRESSION must reflect the abnormal findings\n"
            "- Add this note in FINDINGS:\n"
            "'Note: Visual description suggested normal appearance, "
            "however quantitative analysis detected significant "
            "findings. Radiologist review essential.'\n\n"
            + prompt
        )

    if high_confidence:
        prompt = (
            "CRITICAL INSTRUCTION: The following pathologies were "
            "detected with HIGH confidence and MUST appear in the "
            "FINDINGS section. Do not write 'no abnormalities' when "
            "these are present:\n"
            f"{high_confidence}\n"
            "Failure to include these is a critical clinical error.\n\n"
            + prompt
        )

    prompt += (
        "\nIf any finding has confidence > 0.7, start the FINDINGS "
        "section with: ⚠️ HIGH CONFIDENCE FINDING: [pathology name]"
    )
    payload = {
        "model": MODEL_LLM,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
    }

    try:
        with httpx.Client(timeout=REQUEST_TIMEOUT) as client:
            response = client.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload)
            _raise_for_ollama_error(response)

            data = response.json()
            if "error" in data:
                error_text = str(data["error"]).lower()
                if "not found" in error_text:
                    raise RuntimeError(
                        f"LLM model not pulled. Run: ollama pull {MODEL_LLM}"
                    )
                raise RuntimeError(data["error"])

            message = data.get("message", {})
            report = message.get("content", "").strip()
            if not report:
                raise RuntimeError("LLM returned an empty report.")

            _unload_model(client, MODEL_LLM)
            return report

    except httpx.ConnectError as exc:
        raise RuntimeError(
            "Ollama not running. Start with: ollama serve"
        ) from exc
    except httpx.TimeoutException as exc:
        raise TimeoutError(
            f"Report drafting timed out after {int(REQUEST_TIMEOUT)} seconds."
        ) from exc


def generate_patient_summary(
    clinical_report: str,
    lung_rads_category: str,
    lung_rads_risk: str,
    follow_up_months: int,
    urgent: bool
) -> str:
    """
    Convert clinical radiology report into plain-English
    patient-friendly summary. No medical jargon.
    """
    import httpx, os
    from dotenv import load_dotenv
    load_dotenv()

    OLLAMA_BASE_URL = os.getenv(
        "OLLAMA_BASE_URL", "http://localhost:11434"
    )
    MODEL_LLM = os.getenv("MODEL_LLM", "llama3.1:8b")

    if urgent:
        urgency_note = (
            "The findings are concerning and require "
            "immediate medical attention."
        )
    elif follow_up_months <= 3:
        urgency_note = (
            f"A follow-up appointment is needed within "
            f"{follow_up_months} months."
        )
    elif follow_up_months <= 6:
        urgency_note = (
            "A follow-up appointment is recommended "
            "in the next few months."
        )
    else:
        urgency_note = (
            "Routine annual follow-up is recommended."
        )

    prompt = f"""
You are a compassionate doctor explaining a chest X-ray
result to a patient with no medical background.

Here is the clinical radiology report:
{clinical_report}

Risk assessment: {lung_rads_category}
({lung_rads_risk} malignancy risk)
{urgency_note}

Write a patient-friendly summary following these rules:
1. NO medical jargon — if you must use a term, explain it
2. Be warm, calm, and reassuring but honest
3. Do NOT minimize serious findings
4. Use short sentences and simple words
5. Structure it exactly like this:

WHAT WE FOUND:
[2-3 sentences in plain English]

WHAT THIS MEANS FOR YOU:
[2-3 sentences explaining significance]

WHAT HAPPENS NEXT:
[Clear action steps the patient needs to take]

WHEN TO SEEK IMMEDIATE HELP:
[Warning signs — or "No immediate emergency signs 
 detected" if low risk]
"""

    try:
        with httpx.Client(timeout=120.0) as client:
            response = client.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json={
                    "model": MODEL_LLM,
                    "prompt": prompt,
                    "stream": False,
                    "keep_alive": 0
                }
            )
            response.raise_for_status()
            result = response.json()
            summary = result.get("response", "").strip()

            if len(summary) < 50:
                return (
                    "Patient summary could not be generated. "
                    "Please ask your doctor to explain "
                    "the findings."
                )
            return summary

    except Exception as e:
        print(f"Patient summary error: {e}")
        return (
            "Patient summary temporarily unavailable. "
            "Please discuss results with your physician."
        )


if __name__ == "__main__":
    print(
        detect_conflict(
            "The chest X-ray appears normal, no masses detected",
            {"Mass": 0.80, "Nodule": 0.55},
        )
    )
    print(
        detect_conflict(
            "Dense opacity in right upper lobe, mass present",
            {"Mass": 0.80},
        )
    )
