"""
Model 1 -- the guideline reasoning service hosted on Render.

Replaces the local policy-fit regressor (ml/models/policy_fit.joblib). The
service reasons over a free-text case against the ICMR STW guideline corpus and
returns a Report: per-rule verdicts with page citations, a tally, flagged items,
a routing block and a confidence block.

The `/analyze` endpoint is the one we want. The service also exposes
`/adjudicate` and `/adjudicate/full`, which are faster but return a different
shape that the gap classifier cannot consume -- it reads `rules[].status`,
`confidence.score`, `routing.matched` and `flagged[].severity`, which only
`/analyze` produces.

Timing measured against the live free-tier instance: cold start ~50-90s, warm
`/analyze` ~54s (2.3s routing + 50.7s reasoning). The 120s read timeout and the
single automatic retry below are set from those numbers.
"""
import atexit
import hashlib
import json
import threading
from collections import OrderedDict

import httpx

from ..config import (
    PRIOR_AUTH_CONNECT_TIMEOUT,
    PRIOR_AUTH_READ_TIMEOUT,
    PRIOR_AUTH_TOKEN,
    PRIOR_AUTH_URL,
)


class ModelUnavailable(RuntimeError):
    """Raised when a model cannot produce a result.

    Kept here rather than in ml.py so this module has no import cycle with it.
    ml.py re-exports it, so `ml.ModelUnavailable` still works for the 503 handler
    in routers/requests.py.
    """


_TIMEOUT = httpx.Timeout(PRIOR_AUTH_READ_TIMEOUT, connect=PRIOR_AUTH_CONNECT_TIMEOUT)
_LIMITS = httpx.Limits(max_keepalive_connections=5, max_connections=10)

_client: httpx.Client | None = None
_client_lock = threading.Lock()

# One Report per distinct case text. predict_policy_fit, explain_policy_fit and
# complexity_score all need the same Report; the pipeline fetches it once and
# threads it through, so this is only a guard against an accidental second call.
_CACHE: OrderedDict[str, dict] = OrderedDict()
_CACHE_MAX = 32
_cache_lock = threading.Lock()


def _headers() -> dict:
    return {"Authorization": f"Bearer {PRIOR_AUTH_TOKEN}"} if PRIOR_AUTH_TOKEN else {}


def client() -> httpx.Client:
    """One client for the process. A client per request throws away the
    connection pool and pays a TLS handshake every time."""
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:
                _client = httpx.Client(
                    base_url=PRIOR_AUTH_URL,
                    headers=_headers(),
                    timeout=_TIMEOUT,
                    limits=_LIMITS,
                )
    return _client


@atexit.register
def _close() -> None:
    global _client
    if _client is not None:
        try:
            _client.close()
        finally:
            _client = None


# ---------------------------------------------------------------------------
# Case text
# ---------------------------------------------------------------------------

_YES_NO = {1: "yes", 0: "no"}


def _flag(features: dict, key: str) -> str | None:
    v = features.get(key)
    if v in (None, ""):
        return None
    try:
        return _YES_NO.get(int(v))
    except (TypeError, ValueError):
        return None


def build_case_text(features: dict, document_text: str | None = None) -> str:
    """Turn the structured feature dict into the free-text case the service reads.

    `/analyze` takes prose, not fields -- it says no particular structure is
    required. We write a clinical summary from the features and append the
    extracted PDF text, which usually carries detail the fields dropped.
    """
    f = features
    lines: list[str] = []

    who = []
    if f.get("age") not in (None, ""):
        who.append(f"{f['age']}-year-old")
    sex = str(f.get("sex") or "").strip().lower()
    if sex:
        who.append({"m": "man", "male": "man", "f": "woman", "female": "woman"}.get(sex, sex))
    if who:
        lines.append(f"Patient: {' '.join(who)}.")
    if f.get("bmi") not in (None, ""):
        lines.append(f"BMI {f['bmi']}.")

    dx = f.get("diagnosis")
    code = f.get("diagnosis_code")
    if dx:
        lines.append(f"Diagnosis: {dx}" + (f" (ICD-10 {code})." if code else "."))

    sev = []
    if f.get("disease_severity"):
        sev.append(f"disease severity {f['disease_severity']}")
    if f.get("symptom_burden_0_10") not in (None, ""):
        sev.append(f"symptom burden {f['symptom_burden_0_10']}/10")
    if f.get("symptom_duration_months") not in (None, ""):
        sev.append(f"symptoms for {f['symptom_duration_months']} months")
    if sev:
        lines.append("Presentation: " + ", ".join(sev) + ".")

    if f.get("comorbidities") and str(f["comorbidities"]).lower() != "unknown":
        lines.append(f"Comorbidities: {f['comorbidities']}.")

    prior = []
    for key, label in (
        ("previous_treatment_count", "prior therapies tried"),
        ("previous_failed_count", "failed"),
        ("previous_partial_response_count", "partial response"),
        ("previous_adverse_effect_count", "stopped for adverse effects"),
    ):
        if f.get(key) not in (None, ""):
            prior.append(f"{f[key]} {label}")
    if f.get("longest_previous_treatment_weeks") not in (None, ""):
        prior.append(f"longest trial {f['longest_previous_treatment_weeks']} weeks")
    if prior:
        lines.append("Prior treatment: " + ", ".join(prior) + ".")

    # Hospital surgical packets carry free-text clinical narrative under their
    # own keys rather than the coded fields above. Include whatever is present.
    if f.get("clinical_complaint"):
        lines.append(f"Presenting complaint: {f['clinical_complaint']}.")
    if f.get("clinical_findings"):
        lines.append(f"Clinical findings: {f['clinical_findings']}.")

    req = []
    if f.get("requested_treatment"):
        req.append(str(f["requested_treatment"]))
    if f.get("procedure_code"):
        req.append(f"procedure code {f['procedure_code']}")
    for key, label in (
        ("dose_category", "dose"),
        ("frequency", "frequency"),
        ("route", "route"),
    ):
        if f.get(key):
            req.append(f"{label} {f[key]}")
    if f.get("requested_duration_months") not in (None, ""):
        req.append(f"for {f['requested_duration_months']} months")
    if req:
        lines.append("Requested service: " + ", ".join(req) + ".")
    if f.get("request_reason"):
        lines.append(f"Stated reason: {f['request_reason']}.")
    if f.get("clinical_rationale"):
        lines.append(f"Clinical rationale for the request: {f['clinical_rationale']}.")
    if f.get("facility_status"):
        lines.append(f"Facility network status: {f['facility_status']}.")

    docs_present, docs_absent = [], []
    for key, label in (
        ("doctor_note_present", "clinical note"),
        ("lab_results_present", "lab results"),
        ("imaging_present", "imaging"),
        ("medication_history_present", "medication history"),
    ):
        state = _flag(f, key)
        if state == "yes":
            docs_present.append(label)
        elif state == "no":
            docs_absent.append(label)
    if docs_present:
        lines.append("Documentation on file: " + ", ".join(docs_present) + ".")
    if docs_absent:
        lines.append("Documentation NOT submitted: " + ", ".join(docs_absent) + ".")

    prov = []
    for key in ("provider_specialty", "provider_type", "provider_state"):
        if f.get(key):
            prov.append(str(f[key]))
    if prov:
        lines.append("Provider: " + ", ".join(prov) + ".")
    if f.get("payer"):
        lines.append(f"Payer: {f['payer']}.")

    elig = _flag(f, "member_eligible")
    cov = _flag(f, "treatment_covered")
    if elig is not None:
        lines.append(f"Member eligible on date of service: {elig}.")
    if cov is not None:
        lines.append(f"Requested therapy is a covered benefit: {cov}.")

    text = "\n".join(lines)

    if document_text and document_text.strip():
        text += "\n\nSubmitted documentation (extracted):\n" + document_text.strip()

    text = text.strip()
    if len(text) < 20:
        raise ModelUnavailable(
            "Not enough information to build a case for the guideline service. "
            "Fill in diagnosis and requested treatment, or attach a document."
        )
    return text[:20000]  # service caps `case` at 20000 characters


# ---------------------------------------------------------------------------
# Calls
# ---------------------------------------------------------------------------


def _detail(response: httpx.Response) -> str:
    if response.headers.get("content-type", "").startswith("application/json"):
        try:
            return str(response.json().get("detail", response.text))
        except (ValueError, json.JSONDecodeError):
            pass
    return response.text[:400]


def analyse(features: dict, document_text: str | None = None) -> dict:
    """Send one case, get the Report back.

    Retries once on timeout: the free instance spins down with inactivity and
    the first request after that pays the cold start, but a retry lands warm.
    """
    case = build_case_text(features, document_text)
    key = hashlib.sha256(case.encode("utf-8")).hexdigest()

    with _cache_lock:
        if key in _CACHE:
            _CACHE.move_to_end(key)
            return _CACHE[key]

    last_timeout: Exception | None = None
    for attempt in (1, 2):
        try:
            response = client().post("/analyze", json={"case": case})
            break
        except httpx.TimeoutException as exc:
            last_timeout = exc
            if attempt == 2:
                raise ModelUnavailable(
                    "Guideline service timed out twice. It is likely waking from "
                    "idle on the free tier; try again in a minute."
                ) from exc
        except httpx.RequestError as exc:
            raise ModelUnavailable(f"Guideline service unreachable: {exc}") from exc
    else:  # pragma: no cover - loop always breaks or raises
        raise ModelUnavailable(str(last_timeout))

    if response.status_code != 200:
        detail = _detail(response)
        if response.status_code == 422:
            raise ModelUnavailable(f"Guideline service rejected the case: {detail}")
        if response.status_code == 503:
            raise ModelUnavailable(
                f"Guideline reasoning unavailable (missing API key or model quota "
                f"spent): {detail}"
            )
        raise ModelUnavailable(
            f"Guideline service error {response.status_code}: {detail}"
        )

    report = response.json()

    with _cache_lock:
        _CACHE[key] = report
        _CACHE.move_to_end(key)
        while len(_CACHE) > _CACHE_MAX:
            _CACHE.popitem(last=False)

    return report


def health(timeout: float = 10.0) -> dict:
    """Cheap reachability probe. Never raises."""
    try:
        r = client().get("/health", timeout=httpx.Timeout(timeout, connect=5.0))
        if r.status_code == 200:
            return {"reachable": True, **r.json()}
        return {"reachable": False, "status_code": r.status_code}
    except Exception as exc:
        return {"reachable": False, "error": str(exc)}


def warm() -> None:
    """Fire-and-forget wake-up so the first real user does not eat the cold start."""
    threading.Thread(target=health, kwargs={"timeout": 120.0}, daemon=True).start()
