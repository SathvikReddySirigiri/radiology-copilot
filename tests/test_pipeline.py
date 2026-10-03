"""Tests for findings text, conflict detection, and Lung-RADS scoring."""

from app.pipeline.drafter import build_findings_section, confidence_label, detect_conflict
from app.pipeline.lung_rads import (
    _CATEGORY_META,
    _DISPLAY_MAP,
    check_urgent_findings,
    score_lung_rads,
)


def test_confidence_label():
    assert confidence_label(0.75) == "high"
    assert confidence_label(0.63) == "moderate"
    assert confidence_label(0.30) == "low"


def test_build_findings():
    text = build_findings_section({"Lung Opacity": 0.63}, [])
    assert "Lung Opacity — moderate confidence (63%)" in text
    assert "HIGH CONFIDENCE" not in text

    high = build_findings_section({"Mass": 0.80}, [])
    assert "⚠️ HIGH CONFIDENCE FINDING: Mass" in high
    assert "high confidence (80%)" in high


def test_detect_conflict():
    conflict = detect_conflict(
        "The lungs appear to be clear.",
        {"Mass": 0.55},
    )
    assert conflict["conflict"] is True
    assert conflict["reasons"]

    quiet = detect_conflict(
        "The lungs appear to be clear.",
        {"Mass": 0.30, "Nodule": 0.20},
    )
    assert quiet["conflict"] is False

    lateral = detect_conflict("This lateral view is uncertain.", {"Mass": 0.20})
    assert any("view mismatch" in reason.lower() for reason in lateral["reasons"])


def test_lung_rads():
    assert score_lung_rads({"Mass": 0.61}).category == "4A"
    assert score_lung_rads({"Mass": 0.80}).category == "4B"
    low = score_lung_rads(
        {
            "Mass": 0.05,
            "Nodule": 0.05,
            "Lung Opacity": 0.05,
            "Consolidation": 0.05,
            "Effusion": 0.05,
            "Pneumothorax": 0.05,
        }
    )
    assert low.category in {"1", "2"}


def test_lung_rads_wording():
    for category, meta in _CATEGORY_META.items():
        if meta["urgent"]:
            continue
        actions = " ".join(_DISPLAY_MAP[category]["action_items"]).lower()
        assert "urgent" not in actions
        assert "immediately" not in actions
        assert "immediate" not in actions
    assert "Schedule 3-month CT or PET/CT" in _DISPLAY_MAP["4A"]["action_items"]
    assert "urgently" not in " ".join(_DISPLAY_MAP["4A"]["action_items"]).lower()


def test_urgent():
    assert check_urgent_findings({"Pneumothorax": 0.65}) == [
        "Possible pneumothorax — urgent clinical review"
    ]
    assert check_urgent_findings({"Pneumothorax": 0.40}) == []
