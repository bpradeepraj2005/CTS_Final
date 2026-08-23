"""
Gap classification -- can the provider fix this with paperwork?

This is the half of the supporting-material assessment that decides routing, and
it is deliberately rule-based rather than learned. It reads the unmet criteria in
Model 1's Report and sorts them by what it would actually take to close them:

    CLERICAL          a form, a signature, a note -- hours
    PROCURABLE_FAST   a test, a scan, a lab panel -- days
    PROCURABLE_SLOW   a treatment trial or observation period -- weeks
    BEHAVIOURAL       the patient declined; not the provider's to supply
    CLINICAL_FACT     a measurement against a threshold; no document changes it
    CATEGORICAL       an explicit exclusion or contraindication

Only the first two count as fixable, because only those can be closed inside the
window a payer decision lives in.

Keeping this separate from the model matters. The appeal regressor scores how
likely a denial is to be challenged; this decides whether denying is defensible
at all. The second question is the one that stops a case reaching a human, so it
should not depend on a fitted model -- and after the retraining on
appeals_prediction_transformed.csv it visibly should not: that model carries
almost no signal, while this logic is unchanged and auditable line by line.

Ported from the PriorAuthTriage artifact so the behaviour survives that file
being retired.
"""
import re

CURABILITY_WEIGHT = {
    "CLERICAL": 1.00,
    "PROCURABLE_FAST": 0.80,
    "PROCURABLE_SLOW": 0.50,
    "BEHAVIOURAL": 0.50,
    "CLINICAL_FACT": 0.10,
    "CATEGORICAL": 0.00,
}

CURABILITY_ORDER = [
    "CATEGORICAL",
    "CLINICAL_FACT",
    "BEHAVIOURAL",
    "PROCURABLE_SLOW",
    "PROCURABLE_FAST",
    "CLERICAL",
]

FIXABLE = ("CLERICAL", "PROCURABLE_FAST")

# Why a gap in each hard class cannot simply be sent in.
WHY_HARD = {
    "PROCURABLE_SLOW": "need a treatment trial or observation period first",
    "BEHAVIOURAL": "depend on the patient accepting a treatment they declined",
    "CLINICAL_FACT": "rest on a measurement no paperwork changes",
    "CATEGORICAL": "fall under an explicit guideline exclusion",
}

_PATTERNS = [
    (
        "CATEGORICAL",
        [
            r"should not be (done|performed)",
            r"not to be (done|performed)",
            r"contraindicat",
            r"explicitly excluded",
            r"non-indication",
            r"never",
            r"not indicated for",
            r"do not use",
        ],
    ),
    (
        "BEHAVIOURAL",
        [
            r"patient declin",
            r"patient refus",
            r"declines a trial",
            r"non[- ]complian",
            r"non[- ]adheren",
            r"refused (treatment|therapy)",
            r"against medical advice",
        ],
    ),
    (
        "PROCURABLE_SLOW",
        [
            r"\b\d+[- ]week trial",
            r"trial of \w+ therapy",
            r"medical expulsive therapy",
            r"failed medical (management|treatment|therapy)",
            r"conservative management",
            r"\bmonths? of (optimal|medical)",
            r"observation period",
            r"documented .{0,30}trial",
            r"prior therap",
            r"first[- ]line therap",
        ],
    ),
    (
        "PROCURABLE_FAST",
        [
            r"not (performed|done|available|submitted|attached)",
            r"was not (performed|done)",
            r"not been (performed|done)",
            r"missing",
            r"no .{0,25}(report|result|scan|test|level|panel)",
            r"x-?ray",
            r"ultrasound",
            r"\bct\b",
            r"\bmri\b",
            r"echocardiograph",
            r"serum \w+",
            r"electrolyte",
            r"culture",
            r"biopsy",
            r"audiometr",
            r"spirometr",
            r"\becg\b",
            r"investigation",
            r"laborator",
        ],
    ),
    (
        "CLERICAL",
        [
            r"not (recorded|documented|stated|specified|noted)",
            r"documentation (absent|missing|not)",
            r"consent",
            r"second opinion",
            r"form",
            r"attestation",
            r"signature",
            r"referral letter",
        ],
    ),
]

_CLINICAL_FACT = [
    r"(less|more|greater|lower|higher) than \d",
    r"\b(under|over|below|above|at least|at most)\s+\d",
    r"[<>]=?\s*\d",
    r"\d+\s*(mm|cm|kg|mg|ml|weeks?|years?|months?|bpm|mmhg|mEq|g/dl|ml/min)",
    r"score (of )?\d",
    r"threshold",
    r"cut[- ]?off",
]

_UNMET = ("FAIL", "FAILED", "NOT_MET", "MISSING", "INSUFFICIENT", "UNKNOWN", "NOT_EVALUABLE")


def _match(blob: str):
    for cls, patterns in _PATTERNS:
        for p in patterns:
            if re.search(p, blob):
                return cls
    return None


def classify(rule_text: str, evidence: str = "", explanation: str = "") -> str:
    """Classify on the rule text first, consulting the evidence only if the rule
    text is uninformative. Blending all three lets incidental phrasing in the
    evidence hijack the label."""
    rt = (rule_text or "").lower()

    cls = _match(rt)
    if cls is not None:
        # A rule that names a test but states a threshold is a clinical fact, not
        # a missing document -- unless it literally says the test was not done.
        if cls in ("PROCURABLE_FAST", "CLERICAL"):
            if any(re.search(q, rt) for q in _CLINICAL_FACT):
                if not re.search(r"not (performed|done|available|submitted)", rt):
                    return "CLINICAL_FACT"
        return cls

    if any(re.search(q, rt) for q in _CLINICAL_FACT):
        return "CLINICAL_FACT"

    tail = " ".join(x for x in (evidence, explanation) if x).lower()
    cls = _match(tail)
    if cls is not None:
        return cls
    if any(re.search(q, tail) for q in _CLINICAL_FACT):
        return "CLINICAL_FACT"

    return "PROCURABLE_SLOW"


def is_threshold_bound(rule_text: str) -> bool:
    return any(re.search(q, (rule_text or "").lower()) for q in _CLINICAL_FACT)


def gaps(report: dict) -> tuple[list, list]:
    """Split the Report's unmet criteria into (fixable, hard)."""
    fixable, hard = [], []
    for rule in report.get("rules") or []:
        if str(rule.get("status", "")).upper() not in _UNMET:
            continue
        cls = classify(
            rule.get("rule", ""),
            rule.get("patient_evidence", "") or "",
            rule.get("explanation", "") or "",
        )
        entry = {
            "rule": rule.get("rule", ""),
            "class": cls,
            "page": rule.get("page"),
            "threshold_bound": is_threshold_bound(rule.get("rule", "")),
        }
        (fixable if cls in FIXABLE else hard).append(entry)
    return fixable, hard


def index(classes: list) -> float:
    """Mean curability of a set of gaps, 0..1. Higher means more closeable."""
    if not classes:
        return 1.0
    return sum(CURABILITY_WEIGHT[c] for c in classes) / len(classes)
