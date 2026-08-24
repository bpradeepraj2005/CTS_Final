"""Pillar 1 -- Policy Eligibility (scoring model section 3).

Boolean-with-reasons, runs first, short-circuits everything downstream.
There is no point computing medical necessity for a member whose coverage
lapsed, and a plan exclusion must never be expressed as a low necessity
score -- they are different rejections with different appeal paths.
"""

from __future__ import annotations

from typing import Any

HARD_CHECKS = (
    "enrollment_active_on_service_date",
    "benefit_covers_service",
    "service_not_in_plan_exclusions",
    "waiting_period_satisfied",
    "annual_or_lifetime_limit_available",
)
SOFT_CHECKS = ("provider_empanelled", "no_duplicate_active_authorization")
GATE_CHECKS = HARD_CHECKS + SOFT_CHECKS  # facility_level and prior_auth_required are handled separately


def evaluate_eligibility(eligibility: dict[str, Any], facility_level_rank: int | None,
                          required_facility_rank: int | None) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    indeterminate: list[str] = []

    for name in GATE_CHECKS:
        result = eligibility.get(name)
        checks.append({"check": name, "result": result})
        if result is None:
            indeterminate.append(name)

    facility_ok: bool | None
    if required_facility_rank is None:
        facility_ok = True
    elif facility_level_rank is None:
        facility_ok = None
        indeterminate.append("facility_level_authorized_for_service")
    else:
        facility_ok = facility_level_rank >= required_facility_rank
    checks.append({"check": "facility_level_authorized_for_service", "result": facility_ok})

    prior_auth_required = eligibility.get("prior_auth_actually_required", True)
    checks.append({"check": "prior_auth_actually_required", "result": prior_auth_required})

    hard_failed = [c["check"] for c in checks
                   if c["check"] in HARD_CHECKS and c["result"] is False]
    if facility_ok is False:
        hard_failed.append("facility_level_authorized_for_service")

    if prior_auth_required is False:
        return {"status": "not_required", "gate": 0.0, "checks": checks,
                "reason": "Prior authorization is not required for this service; no clinical evaluation performed."}

    if hard_failed:
        return {"status": "not_covered", "gate": 0.0, "checks": checks,
                "reason": f"Eligibility failed on: {', '.join(hard_failed)}."}

    soft_failed = [c["check"] for c in checks if c["check"] in SOFT_CHECKS and c["result"] is False]
    if soft_failed:
        return {"status": "pass_with_flag", "gate": 0.85, "checks": checks,
                "reason": f"Administrative flag: {', '.join(soft_failed)}.", "indeterminate": indeterminate}

    if indeterminate:
        return {"status": "pass_indeterminate", "gate": 1.0, "checks": checks,
                "reason": f"Eligibility indeterminate on: {', '.join(indeterminate)}.", "indeterminate": indeterminate}

    return {"status": "pass", "gate": 1.0, "checks": checks, "reason": "All eligibility checks passed.", "indeterminate": []}
