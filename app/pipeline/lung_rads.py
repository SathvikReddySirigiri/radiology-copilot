"""ACR Lung-RADS 1.1 risk scoring for chest imaging findings.

Pure rule-based malignancy risk categorization using pathology model
scores — no LLM required.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class LungRADSResult:
    category: str
    category_str: str
    malignancy_risk: str
    risk_level: str
    finding_description: str
    recommendation: str
    follow_up_months: int
    urgent: bool


_CATEGORY_META = {
    "1": {
        "malignancy_risk": "<1%",
        "risk_level": "low",
        "finding_description": "No significant lung nodule or suspicious opacity detected.",
        "recommendation": "Continue annual low-dose CT screening",
        "follow_up_months": 12,
        "urgent": False,
    },
    "2": {
        "malignancy_risk": "<1%",
        "risk_level": "low",
        "finding_description": "Benign-appearing pulmonary finding with very low suspicion.",
        "recommendation": "Continue annual low-dose CT in 12 months",
        "follow_up_months": 12,
        "urgent": False,
    },
    "3": {
        "malignancy_risk": "1–2%",
        "risk_level": "moderate",
        "finding_description": "Probably benign finding warranting short-interval follow-up.",
        "recommendation": "6-month low-dose CT follow-up",
        "follow_up_months": 6,
        "urgent": False,
    },
    "4A": {
        "malignancy_risk": "5–15%",
        "risk_level": "high",
        "finding_description": "Suspicious pulmonary finding requiring closer surveillance.",
        "recommendation": "3-month low-dose CT or PET/CT recommended",
        "follow_up_months": 3,
        "urgent": False,
    },
    "4B": {
        "malignancy_risk": "15–30%",
        "risk_level": "very_high",
        "finding_description": "Very suspicious pulmonary finding with high malignancy concern.",
        "recommendation": (
            "Chest CT with/without contrast + Pulmonologist referral"
        ),
        "follow_up_months": 1,
        "urgent": True,
    },
    "4X": {
        "malignancy_risk": ">30%",
        "risk_level": "very_high",
        "finding_description": (
            "Very suspicious finding with additional features concerning for malignancy."
        ),
        "recommendation": (
            "URGENT: Tissue sampling / biopsy required. "
            "Oncology referral immediately."
        ),
        "follow_up_months": 0,
        "urgent": True,
    },
}

_DISPLAY_MAP = {
    "1": {
        "color": "#2ecc71",
        "emoji": "🟢",
        "action_items": [
            "Schedule routine annual screening",
            "Maintain healthy lifestyle",
            "No additional action required",
        ],
    },
    "2": {
        "color": "#27ae60",
        "emoji": "🟢",
        "action_items": [
            "Schedule routine annual screening",
            "Maintain healthy lifestyle",
            "No additional action required",
        ],
    },
    "3": {
        "color": "#f39c12",
        "emoji": "🟡",
        "action_items": [
            "Schedule 6-month follow-up CT",
            "Discuss findings with primary care physician",
            "Monitor for new symptoms: cough, weight loss",
        ],
    },
    "4A": {
        "color": "#e67e22",
        "emoji": "🟠",
        "action_items": [
            "Schedule 3-month CT or PET/CT",
            "Consult pulmonologist",
            "Document nodule characteristics",
        ],
    },
    "4B": {
        "color": "#e74c3c",
        "emoji": "🔴",
        "action_items": [
            "Immediate pulmonologist referral",
            "CT with contrast within 1 month",
            "Prepare for possible biopsy discussion",
        ],
    },
    "4X": {
        "color": "#8e44ad",
        "emoji": "🟣",
        "action_items": [
            "URGENT: Contact oncologist today",
            "Biopsy scheduling required",
            "Do not delay — same week action needed",
        ],
    },
}


def _label_score(pathology_labels: dict, key: str) -> float:
    if key in pathology_labels:
        return float(pathology_labels[key])

    key_norm = key.lower().replace("_", " ").strip()
    for label, value in pathology_labels.items():
        if label.lower().replace("_", " ").strip() == key_norm:
            return float(value)
    return 0.0


def _build_result(category: str, finding_description: Optional[str] = None) -> LungRADSResult:
    meta = _CATEGORY_META[category]
    return LungRADSResult(
        category=category,
        category_str=f"Lung-RADS {category}",
        malignancy_risk=meta["malignancy_risk"],
        risk_level=meta["risk_level"],
        finding_description=finding_description or meta["finding_description"],
        recommendation=meta["recommendation"],
        follow_up_months=meta["follow_up_months"],
        urgent=meta["urgent"],
    )


def check_urgent_findings(scores: dict) -> list[str]:
    """Flag findings that need urgent review, separate from the Lung-RADS category."""
    flags = []
    if _label_score(scores, "Pneumothorax") >= 0.60:
        flags.append("Possible pneumothorax — urgent clinical review")
    return flags


def score_lung_rads(
    pathology_labels: dict,
    nodule_size_mm: float = None,
    nodule_density: str = None,
) -> LungRADSResult:
    """Score chest findings using ACR Lung-RADS 1.1 logic."""
    del nodule_density  # Reserved for future nodule characterization rules.

    mass_score = _label_score(pathology_labels, "Mass")
    nodule_score = _label_score(pathology_labels, "Nodule")
    opacity_score = _label_score(pathology_labels, "Lung Opacity")
    consolidation_score = _label_score(pathology_labels, "Consolidation")
    effusion_score = _label_score(pathology_labels, "Effusion")
    pneumothorax_score = _label_score(pathology_labels, "Pneumothorax")

    if (mass_score > 0.75 or opacity_score > 0.80) and (
        effusion_score > 0.75 or pneumothorax_score > 0.6
    ):
        triggers = []
        if mass_score > 0.75:
            triggers.append(
                f"Mass detected with {mass_score:.0%} confidence with "
                "additional critical complications."
            )
        if opacity_score > 0.80:
            triggers.append(
                f"Lung Opacity detected with {opacity_score:.0%} confidence with "
                "additional critical complications."
            )
        if effusion_score > 0.75:
            triggers.append(
                f"Pleural Effusion detected with {effusion_score:.0%} confidence."
            )
        if pneumothorax_score > 0.6:
            triggers.append(
                f"Pneumothorax detected with {pneumothorax_score:.0%} confidence."
            )
        return _build_result("4X", " ".join(triggers))

    if mass_score > 0.70:
        return _build_result(
            "4B",
            f"Mass detected with {mass_score:.0%} confidence. "
            "High suspicion for malignancy.",
        )

    if opacity_score > 0.80 and consolidation_score > 0.50:
        return _build_result(
            "4B",
            f"Lung Opacity ({opacity_score:.0%}) and Consolidation "
            f"({consolidation_score:.0%}) indicate very suspicious findings.",
        )

    if nodule_size_mm is not None and nodule_size_mm > 15:
        return _build_result(
            "4B",
            f"Large nodule measuring {nodule_size_mm:.0f} mm. "
            "High suspicion for malignancy.",
        )

    if 0.45 <= mass_score <= 0.70:
        return _build_result(
            "4A",
            f"Mass detected with {mass_score:.0%} confidence. "
            "Suspicious finding requiring closer surveillance.",
        )

    if nodule_score > 0.60:
        return _build_result(
            "4A",
            f"Nodule detected with {nodule_score:.0%} confidence. "
            "Suspicious finding requiring closer surveillance.",
        )

    if 0.65 <= opacity_score <= 0.80:
        return _build_result(
            "4A",
            f"Lung Opacity detected with {opacity_score:.0%} confidence. "
            "Suspicious finding requiring closer surveillance.",
        )

    if nodule_size_mm is not None and 8 <= nodule_size_mm <= 15:
        return _build_result(
            "4A",
            f"Nodule measuring {nodule_size_mm:.0f} mm requires closer surveillance.",
        )

    if 0.35 <= nodule_score <= 0.60:
        return _build_result(
            "3",
            f"Nodule detected with {nodule_score:.0%} confidence. "
            "Probably benign; short-interval follow-up advised.",
        )

    if 0.35 <= opacity_score <= 0.65:
        return _build_result(
            "3",
            f"Lung Opacity detected with {opacity_score:.0%} confidence. "
            "Probably benign; short-interval follow-up advised.",
        )

    if nodule_size_mm is not None and 6 <= nodule_size_mm <= 8:
        return _build_result(
            "3",
            f"Nodule measuring {nodule_size_mm:.0f} mm warrants 6-month follow-up.",
        )

    opacity_or_nodule_low = (
        0.15 <= opacity_score <= 0.35 or 0.15 <= nodule_score <= 0.35
    )
    if opacity_or_nodule_low and (
        nodule_size_mm is None or nodule_size_mm < 6
    ):
        if 0.15 <= nodule_score <= 0.35:
            return _build_result(
                "2",
                f"Nodule detected with {nodule_score:.0%} confidence. "
                "Benign appearance with very low suspicion.",
            )
        return _build_result(
            "2",
            f"Lung Opacity detected with {opacity_score:.0%} confidence. "
            "Benign appearance with very low suspicion.",
        )

    return _build_result("1")


def get_lung_rads_display(result: LungRADSResult) -> dict:
    """Return UI-friendly display metadata for a Lung-RADS result."""
    display = _DISPLAY_MAP[result.category]
    return {
        "color": display["color"],
        "emoji": display["emoji"],
        "action_items": display["action_items"],
    }


if __name__ == "__main__":
    labels1 = {
        "Lung Opacity": 0.10,
        "Consolidation": 0.08,
        "Effusion": 0.05,
    }
    r1 = score_lung_rads(labels1)
    print(
        f"Scenario 1: {r1.category_str} | {r1.malignancy_risk} | Urgent: {r1.urgent}"
    )

    labels2 = {
        "Lung Opacity": 0.78,
        "Consolidation": 0.61,
        "Effusion": 0.20,
    }
    r2 = score_lung_rads(labels2)
    print(
        f"Scenario 2: {r2.category_str} | {r2.malignancy_risk} | Urgent: {r2.urgent}"
    )

    labels3 = {
        "Lung Opacity": 0.85,
        "Consolidation": 0.55,
        "Effusion": 0.80,
    }
    r3 = score_lung_rads(labels3)
    print(
        f"Scenario 3: {r3.category_str} | {r3.malignancy_risk} | Urgent: {r3.urgent}"
    )

    labels = {
        "Mass": 0.80,
        "Nodule": 0.55,
        "Infiltration": 0.53,
        "Lung Opacity": 0.20,
    }
    result = score_lung_rads(labels)
    print(result.category_str)
    print(result.malignancy_risk)
    print(result.urgent)
    print(result.recommendation)
