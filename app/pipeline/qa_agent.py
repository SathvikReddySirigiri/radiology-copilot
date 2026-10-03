"""QA agent for validating drafted radiology reports.

Checks report consistency, completeness, and clinical plausibility using
rule-based validation before final output.
"""

import re
from typing import Optional

from app.pipeline.lung_rads import check_urgent_findings

_REQUIRED_SECTIONS = ("FINDINGS", "IMPRESSION", "RECOMMENDATIONS")
_LUNG_RADS_SEPARATOR = re.compile(r"━{8,}")
_HALLUCINATION_PHRASES = (
    "biopsy recommended",
    "ct scan required",
    "mri suggested",
    "clinical correlation required",
    "cannot rule out malignancy",
)


def _extract_section(report: str, section: str) -> str:
    pattern = (
        rf"(?is)\b{section}\b\s*:?\s*(.*?)"
        rf"(?=\b(?:FINDINGS|IMPRESSION|RECOMMENDATIONS)\b\s*:|\Z)"
    )
    match = re.search(pattern, report)
    return match.group(1).strip() if match else ""


def _count_findings_items(findings_text: str) -> int:
    bullet_lines = re.findall(r"(?m)^\s*[-*•]\s+\S+", findings_text)
    if bullet_lines:
        return len(bullet_lines)

    sentences = [s.strip() for s in re.split(r"[.!?]+", findings_text) if s.strip()]
    return len(sentences)


def _check_sections(report: str) -> list[str]:
    flags = []
    for section in _REQUIRED_SECTIONS:
        if not re.search(rf"(?i)\b{section}\b", report):
            flags.append(f"Missing section: {section}")
    return flags


def _check_findings_content(report: str) -> list[str]:
    findings_text = _extract_section(report, "FINDINGS")
    if _count_findings_items(findings_text) < 2:
        return ["FINDINGS section too sparse — add more detail"]
    return []


def _text_outside_lung_rads_block(report: str) -> str:
    """Drop text that sits between Lung-RADS separator lines."""
    parts = _LUNG_RADS_SEPARATOR.split(report)
    if len(parts) <= 1:
        return report
    return parts[0] + parts[-1]


def _check_pathology_coverage(report: str, pathology_labels: dict) -> list[str]:
    flags = []
    report_lower = report.lower()

    for label, score in pathology_labels.items():
        if score < 0.70:
            continue
        if label.lower() not in report_lower:
            flags.append(f"High-confidence finding not reported: {label}")

    return flags


def _check_overstated_confidence(report: str, pathology_labels: dict) -> list[str]:
    flags = []
    impression = _extract_section(report, "IMPRESSION").lower()
    finding_lines = _extract_section(report, "FINDINGS").splitlines()
    any_high = any(float(score) >= 0.70 for score in pathology_labels.values())

    for label, score in pathology_labels.items():
        if float(score) >= 0.70:
            continue
        label_pattern = re.compile(rf"\b{re.escape(label.lower())}\b")
        flagged = False
        for line in finding_lines:
            line_lower = line.lower()
            if label_pattern.search(line_lower) and "high confidence" in line_lower:
                flags.append(f"Overstated confidence: {label}")
                flagged = True
                break
        if flagged:
            continue
        if (
            not any_high
            and "high confidence" in impression
            and label_pattern.search(impression)
        ):
            flags.append(f"Overstated confidence: {label}")

    return flags


def _check_unsupported_diagnosis(report: str, pathology_labels: dict) -> list[str]:
    if any(float(score) >= 0.70 for score in pathology_labels.values()):
        return []

    text = _text_outside_lung_rads_block(report).lower()
    text = text.replace("malignancy risk", "")
    if re.search(r"\bcancer\b|\bmalignancy\b", text):
        return ["Unsupported diagnosis"]
    return []


_CONTRADICTION_PATTERN = re.compile(
    r"\b(clear|normal)\b|no abnormalities",
    re.IGNORECASE,
)


def _check_contradiction(report: str, pathology_labels: dict) -> list[str]:
    if not any(float(score) >= 0.50 for score in pathology_labels.values()):
        return []
    text = _text_outside_lung_rads_block(report)
    if _CONTRADICTION_PATTERN.search(text):
        return [
            "Contradiction: report describes the study as clear or normal while a score is >= 0.50"
        ]
    return []


def _check_urgent_in_report(report: str, urgent_flags: list[str]) -> list[str]:
    flags = []
    report_lower = report.lower()
    for finding in urgent_flags or []:
        if finding.lower() not in report_lower:
            flags.append(f"Missing urgent finding: {finding}")
    return flags


def _check_hallucinations(report: str) -> list[str]:
    flags = []
    report_lower = report.lower()
    outside_lower = _text_outside_lung_rads_block(report).lower()

    for phrase in _HALLUCINATION_PHRASES:
        in_template = "biopsy" in phrase or "ct scan" in phrase
        haystack = outside_lower if in_template else report_lower
        if phrase in haystack:
            flags.append(f"Possible hallucination detected: {phrase}")

    return flags


def _score_confidence(flag_count: int) -> str:
    if flag_count == 0:
        return "high"
    if flag_count <= 2:
        return "medium"
    return "low"


def run_qa(
    report: str,
    scores: dict,
    urgent_flags: Optional[list[str]] = None,
) -> dict:
    """Validate a drafted report against the full scores and urgent flags."""
    if urgent_flags is None:
        urgent_flags = check_urgent_findings(scores)

    flags: list[str] = []
    flags.extend(_check_sections(report))
    flags.extend(_check_findings_content(report))
    flags.extend(_check_pathology_coverage(report, scores))
    flags.extend(_check_overstated_confidence(report, scores))
    flags.extend(_check_contradiction(report, scores))
    flags.extend(_check_unsupported_diagnosis(report, scores))
    flags.extend(_check_urgent_in_report(report, urgent_flags))
    hallucination_flags = _check_hallucinations(report)
    flags.extend(hallucination_flags)

    confidence = _score_confidence(len(flags))
    passed = confidence in {"high", "medium"} and not hallucination_flags
    cleaned_report = report.strip()

    return {
        "passed": passed,
        "flags": flags,
        "confidence": confidence,
        "approved_report": cleaned_report if passed else None,
    }
