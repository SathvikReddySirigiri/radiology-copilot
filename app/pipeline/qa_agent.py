"""QA agent for validating drafted radiology reports.

Checks report consistency, completeness, and clinical plausibility using
rule-based validation before final output.
"""

import re

_REQUIRED_SECTIONS = ("FINDINGS", "IMPRESSION", "RECOMMENDATIONS")
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


def _check_pathology_coverage(report: str, pathology_labels: dict) -> list[str]:
    flags = []
    report_lower = report.lower()

    for label, score in pathology_labels.items():
        if score <= 0.7:
            continue
        if label.lower() not in report_lower:
            flags.append(f"High-confidence finding not reported: {label}")

    return flags


def _check_hallucinations(report: str) -> list[str]:
    flags = []
    report_lower = report.lower()

    for phrase in _HALLUCINATION_PHRASES:
        if phrase in report_lower:
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
    pathology_labels: dict,
    ner_entities: dict,
) -> dict:
    """Validate a drafted report and return QA results."""
    del ner_entities  # Reserved for future cross-checks.

    flags: list[str] = []
    flags.extend(_check_sections(report))
    flags.extend(_check_findings_content(report))
    flags.extend(_check_pathology_coverage(report, pathology_labels))
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
