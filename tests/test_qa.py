"""Tests for the QA agent report validation logic."""

from app.pipeline.qa_agent import run_qa

_COMPLETE_REPORT = """FINDINGS
- Opacity is present in the right lung.
- The remaining lung is aerated.
IMPRESSION
Findings warrant radiologist review.
RECOMMENDATIONS
Clinical follow-up.
"""


def test_qa():
    overstated = """FINDINGS
Mass is present with high confidence. A second opacity is also seen.
IMPRESSION
Findings warrant radiologist review.
RECOMMENDATIONS
Clinical follow-up.
"""
    overstated_flags = run_qa(overstated, {"Mass": 0.63}, [])["flags"]
    assert "Overstated confidence: Mass" in overstated_flags

    contradiction = """FINDINGS
- A mass is present.
- Additional opacity is noted.
IMPRESSION
The lungs are clear and normal.
RECOMMENDATIONS
Clinical follow-up.
"""
    contradiction_flags = run_qa(contradiction, {"Mass": 0.55}, [])["flags"]
    assert any(flag.startswith("Contradiction") for flag in contradiction_flags)

    cancer = """FINDINGS
- A mass is present.
- Additional opacity is noted.
IMPRESSION
Possible lung cancer.
RECOMMENDATIONS
Findings warrant radiologist review.
"""
    assert "Unsupported diagnosis" in run_qa(cancer, {"Mass": 0.63}, [])["flags"]
    assert "Unsupported diagnosis" not in run_qa(cancer, {"Mass": 0.80}, [])["flags"]

    lung_rads_report = (
        _COMPLETE_REPORT
        + "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        + "LUNG-RADS ASSESSMENT\n"
        + "Risk        : 5–15% malignancy risk\n"
        + "Clinical Actions:\n"
        + "biopsy recommended\n"
        + "ct scan required\n"
        + "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    )
    block_flags = run_qa(lung_rads_report, {"Mass": 0.63}, [])["flags"]
    assert not any("biopsy" in flag.lower() for flag in block_flags)
    assert not any("ct scan" in flag.lower() for flag in block_flags)
    assert "Unsupported diagnosis" not in block_flags

    outside = """FINDINGS
Opacity is present. Biopsy recommended for this mass.
IMPRESSION
Findings warrant radiologist review.
RECOMMENDATIONS
Clinical follow-up.
"""
    outside_flags = run_qa(outside, {"Mass": 0.63}, [])["flags"]
    assert any("biopsy" in flag.lower() for flag in outside_flags)

    urgent_flags = run_qa(
        _COMPLETE_REPORT,
        {"Pneumothorax": 0.65},
        ["Possible pneumothorax — urgent clinical review"],
    )["flags"]
    assert any(flag.startswith("Missing urgent finding:") for flag in urgent_flags)
