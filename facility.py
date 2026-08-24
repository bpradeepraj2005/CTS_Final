"""Shared facility-level hierarchy used by rule extraction, fact derivation,
and the eligibility gate's `facility_level_authorized_for_service` check."""

from __future__ import annotations

RANK_WORDS: list[tuple[str, int]] = [
    ("tertiary", 2), ("comprehensive stroke centre", 2),
    ("district", 1), ("secondary", 1), ("pci capable", 1),
    ("primary", 0), ("phc", 0), ("chc", 0),
]

LEVEL_TO_RANK = {"phc": 0, "chc": 0, "primary": 0, "secondary": 1, "district": 1, "tertiary": 2}


def rank_from_text(level_text: str | None) -> int | None:
    """Lowest rank among all matched tier words -- a phrase like "District
    hospital or tertiary" is disjunctive ("either is fine"), so the
    controlling requirement is the lower tier explicitly mentioned, not
    whichever word happens to appear first in the phrase."""
    text = (level_text or "").lower()
    matched = [rank for word, rank in RANK_WORDS if word in text]
    return min(matched) if matched else None


def rank_from_level(facility_level: str | None) -> int | None:
    if not facility_level:
        return None
    return LEVEL_TO_RANK.get(str(facility_level).strip().lower())
