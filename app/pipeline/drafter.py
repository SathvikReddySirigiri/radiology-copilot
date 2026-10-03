"""Radiology report drafter using a local LLM via Ollama.

Takes vision model outputs and generates structured radiology report text
following standard reporting conventions.
"""

import os
import re
from datetime import datetime
from typing import Optional

import httpx
from dotenv import load_dotenv

from app.pipeline.lung_rads import LungRADSResult, get_lung_rads_display

load_dotenv()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
MODEL_LLM = os.getenv("MODEL_LLM", "llama3.1:8b")
REQUEST_TIMEOUT = 120.0

_NORMAL_PATTERN = re.compile(
    r"\b(normal|clear|unremarkable|no (acute|significant|visible)|without any visible|no evidence of)\b",
    re.IGNORECASE,
)
_VIEW_MISMATCH = "LLaVA view mismatch — description may be unreliable"
_FALLBACK_IMPRESSION = (
    "Findings of moderate confidence are present. Radiologist review is recommended."
)


def confidence_label(score: float) -> str:
    """Map a pathology score to high, moderate, or low confidence."""
    if score >= 0.70:
        return "high"
    if score >= 0.50:
        return "moderate"
    return "low"


def build_findings_section(scores: dict, urgent_flags: Optional[list[str]] = None) -> str:
    """Build FINDINGS from scores only. LLaVA text is never included."""
    ordered = sorted(scores.items(), key=lambda item: float(item[1]), reverse=True)
    lines = [flag for flag in (urgent_flags or []) if flag]
    reportable = [(label, float(score)) for label, score in ordered if float(score) >= 0.50]

    if not reportable:
        lines.append("No significant findings above threshold.")
        lines.append("Scores below 0.50 are not reported as findings.")
        return "\n".join(lines)

    for label, score in reportable:
        if score >= 0.70:
            lines.append(f"⚠️ HIGH CONFIDENCE FINDING: {label}")
        lines.append(
            f"- {label} — {confidence_label(score)} confidence ({score:.0%})"
        )
    return "\n".join(lines)


def detect_conflict(llava_text: str, scores: dict) -> dict:
    """Return whether a normal-type description conflicts with scores >= 0.50."""
    text = llava_text or ""
    matches: list[str] = []
    for match in _NORMAL_PATTERN.finditer(text):
        token = match.group(0).lower()
        if token not in matches:
            matches.append(token)

    reportable = [
        (label, float(score))
        for label, score in scores.items()
        if float(score) >= 0.50
    ]
    reasons: list[str] = []
    conflict = bool(matches) and bool(reportable)
    if conflict:
        listed = ", ".join(
            f"{label} {score:.2f}"
            for label, score in sorted(reportable, key=lambda item: item[1], reverse=True)
        )
        reasons.append(
            "Visual description used normal-type language "
            f"({', '.join(matches)}) while scores are >= 0.50 ({listed})."
        )
    if re.search(r"\blateral\b", text, re.IGNORECASE):
        reasons.append(_VIEW_MISMATCH)
    return {"conflict": conflict, "reasons": reasons}


def _conflict_notice(conflict: Optional[dict]) -> str:
    if not conflict:
        return ""
    lines = []
    if conflict.get("conflict"):
        lines.append(
            "The visual description disagreed with the quantitative scores. "
            "Radiologist review is essential."
        )
    for reason in conflict.get("reasons") or []:
        if reason == _VIEW_MISMATCH:
            lines.append(reason)
    return "\n".join(lines)


def _build_prompt(
    pathology_labels: dict,
    llava_description: str,
    ner_entities: dict,
    lung_rads: Optional[LungRADSResult] = None,
    urgent_findings: Optional[list[str]] = None,
    conflict: Optional[dict] = None,
) -> str:
    """Prompt Llama for the IMPRESSION only. FINDINGS stay in code."""
    del ner_entities  # Kept on the pipeline; not copied into FINDINGS.

    findings = build_findings_section(pathology_labels, urgent_findings)
    category = lung_rads.category_str if lung_rads is not None else "not scored"
    notice = _conflict_notice(conflict)
    prompt = (
        "You are an expert radiologist. Write ONLY the IMPRESSION in 2–3 sentences. "
        "Do not write FINDINGS or RECOMMENDATIONS.\n\n"
        "Authoritative findings:\n"
        f"{findings}\n\n"
        f"Lung-RADS category: {category}\n\n"
        "Secondary, lower-reliability observation:\n"
        f"{llava_description}\n\n"
    )
    if notice:
        prompt += f"Conflict notice:\n{notice}\n\n"
    prompt += (
        "Rules:\n"
        "Use the confidence labels exactly as given. Never call a moderate finding high confidence.\n"
        "Do not say the lungs are clear or normal if any finding is >= 0.50.\n"
        "Do not suggest cancer or malignancy unless a related finding is >= 0.70. "
        "Otherwise write 'findings warrant radiologist review'.\n"
        "If a conflict notice is present, state that the visual description disagreed "
        "with the quantitative scores and radiologist review is essential.\n"
    )
    return prompt


def _recommendations_section(lung_rads: Optional[LungRADSResult]) -> str:
    if lung_rads is None:
        return "RECOMMENDATIONS:\nRadiologist review is recommended."

    display = get_lung_rads_display(lung_rads)
    action_items = "\n".join(f"- {item}" for item in display["action_items"])
    urgent_label = "YES — IMMEDIATE ACTION" if lung_rads.urgent else "No"
    follow_up = (
        f"{lung_rads.follow_up_months} months"
        if lung_rads.follow_up_months > 0
        else "IMMEDIATE"
    )
    return f"""RECOMMENDATIONS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LUNG-RADS ASSESSMENT (reference scale; Lung-RADS is designed for CT)
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
Lung-RADS-inspired risk category (Lung-RADS is designed for CT; used here as a reference scale).
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"""


def assemble_report(
    scores: dict,
    findings: str,
    impression: str,
    lung_rads: Optional[LungRADSResult],
    conflict: Optional[dict],
) -> str:
    """Assemble header, scores, code FINDINGS, LLM IMPRESSION, and code RECOMMENDATIONS."""
    header = (
        "RADIOLOGY REPORT — AI ASSISTED\n"
        f"Generated : {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
        "System    : Radiology Copilot v1.0\n"
        "Models    : TorchXRayVision + LLaVA + Llama 3.1 8B\n"
        "Study     : Frontal (PA/AP) chest X-ray\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    )
    score_lines = ["PATHOLOGY SCORES (TorchXRayVision):"]
    for label, score in sorted(scores.items(), key=lambda item: float(item[1]), reverse=True):
        value = float(score)
        filled = min(10, max(0, int(value * 10)))
        bar = "█" * filled + "░" * (10 - filled)
        score_lines.append(f"  {label:<28} {bar} {value:.0%}")

    parts = [header, "\n".join(score_lines)]
    notice = _conflict_notice(conflict)
    if notice:
        parts.append("CONFLICT NOTICE:\n" + notice)
    parts.append("FINDINGS:\n" + findings)
    parts.append("IMPRESSION:\n" + impression.strip())
    parts.append(_recommendations_section(lung_rads))
    return "\n\n".join(parts)


def _impression_is_invalid(impression: str, scores: dict) -> bool:
    text = impression.lower()
    has_high = any(float(score) >= 0.70 for score in scores.values())
    has_reportable = any(float(score) >= 0.50 for score in scores.values())
    if "high confidence" in text and not has_high:
        return True
    if has_reportable and re.search(r"\b(clear|normal)\b", text):
        return True
    return False


def _clean_impression(text: str) -> str:
    match = re.search(
        r"(?is)\bIMPRESSION\b\s*:?\s*(.*?)(?=\b(?:FINDINGS|RECOMMENDATIONS)\b\s*:|\Z)",
        text,
    )
    body = match.group(1).strip() if match else text.strip()
    return re.sub(r"(?is)^\s*IMPRESSION\s*:?\s*", "", body).strip()


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


def _generate_text(prompt: str) -> str:
    payload = {
        "model": MODEL_LLM,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
    }
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
        text = message.get("content", "").strip()
        if not text:
            raise RuntimeError("LLM returned an empty report.")

        _unload_model(client, MODEL_LLM)
        return text


def draft_report(
    pathology_labels: dict,
    llava_description: str,
    ner_entities: dict,
    lung_rads: Optional[LungRADSResult] = None,
    urgent_findings: Optional[list[str]] = None,
    conflict: Optional[dict] = None,
) -> str:
    """Assemble a report whose FINDINGS and RECOMMENDATIONS are written in code."""
    if conflict is None:
        conflict = detect_conflict(llava_description, pathology_labels)

    findings = build_findings_section(pathology_labels, urgent_findings)
    prompt = _build_prompt(
        pathology_labels,
        llava_description,
        ner_entities,
        lung_rads=lung_rads,
        urgent_findings=urgent_findings,
        conflict=conflict,
    )

    try:
        impression = _clean_impression(_generate_text(prompt))
        if _impression_is_invalid(impression, pathology_labels):
            retry_prompt = (
                prompt
                + "\nThe previous impression was rejected:\n"
                + impression
                + "\nRewrite the IMPRESSION so it follows the rules exactly.\n"
            )
            impression = _clean_impression(_generate_text(retry_prompt))
            if _impression_is_invalid(impression, pathology_labels):
                impression = _FALLBACK_IMPRESSION
        if not impression:
            impression = _FALLBACK_IMPRESSION
    except httpx.ConnectError as exc:
        raise RuntimeError(
            "Ollama not running. Start with: ollama serve"
        ) from exc
    except httpx.TimeoutException as exc:
        raise TimeoutError(
            f"Report drafting timed out after {int(REQUEST_TIMEOUT)} seconds."
        ) from exc

    return assemble_report(
        pathology_labels,
        findings,
        impression,
        lung_rads,
        conflict,
    )


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
