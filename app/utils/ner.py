"""Named entity recognition for radiology reports using scispaCy.

Extracts clinical entities (findings, anatomy, measurements) from report
text to support structured output and validation.
"""

import re

import scispacy
import spacy

_MODEL_INSTALL_CMD = (
    "scispaCy model missing. Run: pip install "
    "https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.3/"
    "en_core_sci_sm-0.5.3.tar.gz"
)

try:
    _NLP = spacy.load("en_core_sci_sm")
except OSError:
    _NLP = None

_MODIFIERS = {"bilateral", "mild", "severe", "acute", "chronic"}
_SKIP_ENTITIES = {"present", "consistent with", "noted", "no"}
_ANATOMY_KEYWORDS = (
    "lobe",
    "lung",
    "pleural",
    "heart",
    "cardiac",
    "mediastin",
    "hemithorax",
    "chest wall",
    "diaphragm",
    "hilum",
    "hilar",
    "apex",
    "costophrenic",
    "trachea",
    "bronch",
    "rib",
    "clavicle",
)
_CONDITION_KEYWORDS = (
    "pneumonia",
    "pneumothorax",
    "tuberculosis",
    "copd",
    "infection",
    "malignancy",
    "cancer",
    "carcinoma",
    "edema",
    "emphysema",
    "fibrosis",
    "hernia",
    "asthma",
)
_FINDING_KEYWORDS = (
    "effusion",
    "consolidation",
    "cardiomegaly",
    "atelectasis",
    "opacity",
    "infiltrate",
    "infiltration",
    "mass",
    "nodule",
    "fracture",
    "thickening",
    "enlargement",
    "opacification",
)


def _contains_keyword(text: str, keywords: tuple[str, ...]) -> bool:
    return any(keyword in text for keyword in keywords)


def _dedupe(values: list[str]) -> list[str]:
    seen = set()
    unique = []
    for value in values:
        key = value.casefold()
        if key in seen:
            continue
        seen.add(key)
        unique.append(value)
    return unique


def _extract_modifiers_from_text(text: str) -> list[str]:
    found = []
    for modifier in _MODIFIERS:
        if re.search(rf"\b{re.escape(modifier)}\b", text, flags=re.IGNORECASE):
            found.append(modifier)
    return found


def _classify_entity(entity_text: str) -> tuple[list[str], list[str], list[str], list[str]]:
    """Return entity buckets: findings, conditions, anatomy, modifiers."""
    findings: list[str] = []
    conditions: list[str] = []
    anatomy: list[str] = []
    modifiers: list[str] = []

    normalized = entity_text.strip().casefold()
    if len(normalized) < 3 or normalized in _SKIP_ENTITIES:
        return findings, conditions, anatomy, modifiers

    for modifier in _MODIFIERS:
        if re.search(rf"\b{re.escape(modifier)}\b", entity_text, flags=re.IGNORECASE):
            modifiers.append(modifier)

    if normalized in _MODIFIERS and len(normalized.split()) == 1:
        return findings, conditions, anatomy, modifiers

    if _contains_keyword(normalized, _FINDING_KEYWORDS):
        findings.append(entity_text.strip())
    elif _contains_keyword(normalized, _CONDITION_KEYWORDS):
        conditions.append(entity_text.strip())
    elif _contains_keyword(normalized, _ANATOMY_KEYWORDS):
        anatomy.append(entity_text.strip())
    elif not modifiers:
        findings.append(entity_text.strip())

    return findings, conditions, anatomy, modifiers


def extract_entities(text: str) -> dict:
    """Extract categorized clinical entities from radiology report text."""
    if _NLP is None:
        raise RuntimeError(_MODEL_INSTALL_CMD)

    findings: list[str] = []
    conditions: list[str] = []
    anatomy: list[str] = []
    modifiers: list[str] = _extract_modifiers_from_text(text)

    for ent in _NLP(text).ents:
        entity_findings, entity_conditions, entity_anatomy, entity_modifiers = _classify_entity(
            ent.text
        )
        findings.extend(entity_findings)
        conditions.extend(entity_conditions)
        anatomy.extend(entity_anatomy)
        modifiers.extend(entity_modifiers)

    return {
        "findings": _dedupe(findings),
        "conditions": _dedupe(conditions),
        "anatomy": _dedupe(anatomy),
        "modifiers": _dedupe(modifiers),
    }
