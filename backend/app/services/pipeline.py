"""Runs a request end to end: score, decide, explain, route, audit.

The decision flow:

    Model 1 (guideline service)
        |
    Decision Router (necessity_engine)
        |-- approve --> Auto Approval ---------------> Track history
        |-- review  --> Reviewer assignment ---------> Track history
        '-- denial  --> Model 2 (supporting material)
                            |-- fixable gaps -------> Human Review
                            '-- no fixable gaps ----> Auto Denial

Model 2 runs only on the denial branch. Nothing else calls it, so an approved
request carries no appeal prediction.
"""
import time
from datetime import datetime, timezone

from sqlalchemy.orm import Session

from ..config import (
    AUTO_APPROVE_MIN_POLICY_FIT,
    AUTO_DENY_MAX_POLICY_FIT,
    INCLUDE_GUIDELINE_CRITERIA,
)
from ..models import AuditEvent, AuthRequest, Document
from . import explain, ml, model2, necessity_engine, prior_auth_client, routing
from .hospital_predictor import predict_hospital_pa
from .prior_auth_client import ModelUnavailable

from ml.feature_schema import derive_features


def log(db: Session, action: str, *, request_id=None, actor=None, detail=None):
    db.add(AuditEvent(
        request_id=request_id,
        actor_id=getattr(actor, "id", None),
        actor_email=getattr(actor, "email", None),
        action=action,
        detail=detail or {},
    ))


def _document_text(db: Session, req: AuthRequest) -> str:
    """Extracted text from every document attached to this request.

    The guideline service reads prose, and the PDF usually carries clinical
    detail the structured fields dropped.
    """
    docs = (
        db.query(Document)
        .filter(Document.request_id == req.id)
        .all()
    )
    return "\n\n".join(d.raw_text for d in docs if d.raw_text)


def _apply_guideline_gate(verdict: dict, guideline_fit: float) -> dict:
    """Fold the guideline verdict into a hospital packet's completeness score.

    predict_hospital_pa answers "is this packet complete?" -- diagnosis coded,
    findings attached, imaging present, rationale written. It does not answer
    "is this procedure indicated?", because nothing in that path reads a
    guideline. A thoroughly documented request for a procedure the guideline
    does not support scored well and auto-approved.

    Both legs now have to hold. The packet score still gates on its own, so an
    incomplete submission is caught as before; the guideline can only downgrade
    an approval, never manufacture one. A case is no stronger than its weaker leg,
    so that is what the reviewer sees as the score.
    """
    combined = min(verdict["policy_fit_score"], guideline_fit)
    out = dict(verdict, policy_fit_score=round(combined, 4))

    if verdict["decision"] != "APPROVED":
        return out  # incomplete or already contested -- unchanged

    if guideline_fit >= AUTO_APPROVE_MIN_POLICY_FIT:
        return out  # both legs hold

    if guideline_fit <= AUTO_DENY_MAX_POLICY_FIT:
        out.update(
            decision="DENIED",
            status="AUTO_DENIED",
            rationale=(
                f"{verdict['rationale']} The packet is complete, but the matched "
                f"guideline record does not support the requested procedure "
                f"(guideline alignment {guideline_fit:.2f})."
            ),
        )
    else:
        out.update(
            decision=None,
            status="PENDING_REVIEW",
            rationale=(
                f"{verdict['rationale']} Routed to a clinical reviewer: the packet "
                f"is complete, but guideline alignment is borderline at "
                f"{guideline_fit:.2f}."
            ),
        )
    return out


def _model2_context(req: AuthRequest, features: dict) -> dict:
    """Case context for Model 2. Anything absent imputes to the population
    median rather than zero -- unknown is not never."""
    prior_denials = None
    if req.patient_id:
        prior_denials = (
            None  # left unimputed; requires a patient-history query to be meaningful
        )
    return {
        "patient_age": features.get("age"),
        "comorbidity_count": (
            len([c for c in str(features.get("comorbidities", "")).split(",") if c.strip()])
            if features.get("comorbidities")
            and str(features["comorbidities"]).lower() != "unknown"
            else None
        ),
        "prior_denials_same_patient": prior_denials,
    }


def adjudicate(db: Session, req: AuthRequest, actor) -> AuthRequest:
    started = time.perf_counter()
    features = req.features
    report = None
    complexity = None

    # -----------------------------------------------------------------
    # 1. Score + evaluate, branching on document type
    # -----------------------------------------------------------------
    if features.get("document_type") == "HOSPITAL_PA":
        # Surgical packets keep their own completeness scorer, then go through
        # the guideline service like everything else. Both legs have to hold.
        verdict = predict_hospital_pa(features)

        report = prior_auth_client.analyse(features, _document_text(db, req))
        guideline_fit = ml.approval_likelihood(report)
        complexity = ml.complexity_score(report)

        verdict = _apply_guideline_gate(verdict, guideline_fit)
        policy_fit = verdict["policy_fit_score"]

        attribution = {
            "base_score": policy_fit,
            "method": (
                "hospital packet completeness, gated on the matched guideline record"
            ),
            "caveat": (
                f"Packet completeness and guideline alignment are scored "
                f"separately; the lower of the two is shown. Guideline alignment "
                f"for this case is {guideline_fit:.2f}."
            ),
            "contributions": [
                {
                    "feature": c["code"],
                    "label": c["label"],
                    # `reference` and `actionable` are rendered by the attribution
                    # rail; without them the row prints "undefined".
                    "value": c["observed"] if c["observed"] not in (None, "") else "not provided",
                    "reference": "packet evidence",
                    "contribution": c["weight"] if c["passed"] else -c["weight"],
                    "direction": "supports" if c["passed"] else "weakens",
                    "actionable": not c["passed"],
                }
                for c in verdict["criteria"]
            ]
            + explain.explain_policy_fit(report)["contributions"],
        }

        appeal = ml.empty_appeal()

    else:
        # Model 1. One call; the Report feeds scoring, explanation, complexity
        # and, on the denial branch, Model 2.
        report = prior_auth_client.analyse(features, _document_text(db, req))

        policy_fit = ml.approval_likelihood(report)
        complexity = ml.complexity_score(report)
        attribution = explain.explain_policy_fit(report)

        # Decision Router.
        verdict = necessity_engine.evaluate(features, policy_fit)

        appeal = ml.empty_appeal()

    # -----------------------------------------------------------------
    # 2. Persist scores/decision to the request (common to both paths)
    # -----------------------------------------------------------------
    derived = derive_features(features)

    req.policy_fit_score = round(policy_fit, 4)
    req.documentation_score = derived["documentation_score"]
    req.necessity_score = verdict["necessity_score"]
    req.confidence = verdict["confidence"]

    criteria = list(verdict["criteria"])
    if report is not None and INCLUDE_GUIDELINE_CRITERIA:
        # Appended after the necessity score is computed, so these rows change
        # what a reviewer sees and not what the engine decided.
        criteria += explain.guideline_criteria(report)

    req.criteria = {
        "criteria": criteria,
        "rationale": verdict["rationale"],
    }
    req.explanation = attribution
    req.status = verdict["status"]
    req.decision = verdict["decision"]
    req.decision_source = "ENGINE" if verdict["decision"] else None
    req.urgency_score = necessity_engine.urgency_score(
        features, verdict["necessity_score"]
    )

    # Complexity has no column on AuthRequest; stash it alongside the other
    # per-request metadata already kept in features.
    if complexity is not None:
        req.features = {**(req.features or {}), "complexity_score": complexity}

    # -----------------------------------------------------------------
    # 3. Denial branch -- Model 2, supporting-material assessment
    # -----------------------------------------------------------------
    assessment = None
    if verdict["decision"] == "DENIED" and report is not None:
        try:
            assessment = model2.assess(report, _model2_context(req, features))
            appeal = assessment["appeal_prediction"]

            # Surface the assessment to the reviewer, not just the audit log --
            # the checklist is the whole point of asking before denying.
            req.features = {
                **(req.features or {}),
                "model2_assessment": {
                    k: assessment[k]
                    for k in (
                        "route",
                        "reason",
                        "reappeal_percent",
                        "reappeal_percentile",
                        "reappeal_lift",
                        "criteria_satisfaction",
                        "fixable_gaps",
                        "hard_gaps",
                        "resubmission_checklist",
                        "raises_risk",
                        "model_version",
                        "trained_on",
                    )
                },
            }

            if assessment["route"] == "HUMAN_REVIEW":
                # The denial is not safe to automate: either the provider can
                # close the gaps, or this is the kind of denial that gets
                # overturned. Pull it back for a human.
                req.status = "PENDING_REVIEW"
                req.decision = None
                req.decision_source = None
                req.criteria = {
                    **req.criteria,
                    "rationale": (
                        f"{verdict['rationale']} {assessment['reason']}"
                    ),
                }

            log(db, "MODEL2_ASSESSED", request_id=req.id, actor=actor, detail={
                "route": assessment["route"],
                "reason": assessment["reason"],
                "reappeal_percent": assessment["reappeal_percent"],
                "reappeal_percentile": assessment["reappeal_percentile"],
                "criteria_satisfaction": assessment["criteria_satisfaction"],
                "fixable_gaps": assessment["fixable_gaps"],
                "hard_gaps": assessment["hard_gaps"],
                "model_version": assessment["model_version"],
            })

        except ModelUnavailable as exc:
            # Model 2 is an escalation check, not the decision-maker. If it
            # cannot run, the engine's denial stands and the reason is recorded.
            log(db, "MODEL2_UNAVAILABLE", request_id=req.id, actor=actor,
                detail={"error": str(exc)})
            appeal = ml.empty_appeal(f"Supporting-material assessment unavailable")

    req.appeal_prediction = appeal

    # -----------------------------------------------------------------
    # 4. Auto-decide or route to a human reviewer
    # -----------------------------------------------------------------
    if req.decision:
        req.decision_at = datetime.now(timezone.utc)
    else:
        result = routing.assign(db, features)
        req.assigned_reviewer_id = result["reviewer_id"]
        req.assignment_reason = result["reason"]
        req.assignment_was_reassigned = result["reassigned"]
        log(db, "REVIEWER_ASSIGNED", request_id=req.id, actor=actor,
            detail={
                "reason": result["reason"],
                "reassigned": result["reassigned"],
                "candidates": result["candidates"],
            })

    req.processing_ms = round((time.perf_counter() - started) * 1000, 2)

    # -----------------------------------------------------------------
    # 5. Audit log (always written, both paths)
    # -----------------------------------------------------------------
    log(db, "DECISION_COMPUTED", request_id=req.id, actor=actor, detail={
        "status": req.status,
        "decision": req.decision,
        "policy_fit_score": req.policy_fit_score,
        "necessity_score": req.necessity_score,
        "complexity_score": complexity,
        "confidence": req.confidence,
        "processing_ms": req.processing_ms,
        "rationale": req.criteria.get("rationale"),
        "failed_criteria": [
            c["code"] for c in verdict["criteria"] if not c["passed"]
        ],
        "guideline_version": (report or {}).get("guideline_version"),
        "model2_route": (assessment or {}).get("route"),
    })
    return req
