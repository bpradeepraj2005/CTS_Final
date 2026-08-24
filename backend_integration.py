"""Drop this into your other FastAPI project and import `prior_auth`.

Sized against the deployed service's real behaviour: a cold Render instance
took 89s to answer, a warm one ~31s. The timeouts and the keep-warm ping below
are set from those numbers, not from guesses.

    pip install httpx
"""

from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException

PRIOR_AUTH_URL = os.getenv(
    "PRIOR_AUTH_URL", "https://prior-auth-api-bmju.onrender.com"
).rstrip("/")
PRIOR_AUTH_TOKEN = os.getenv("PRIOR_AUTH_TOKEN", "").strip()

# read=120s: a cold instance needs ~90s before it answers at all, and the two
# Gemini calls take ~30s on top once it is awake.
_TIMEOUT = httpx.Timeout(120.0, connect=10.0)
_LIMITS = httpx.Limits(max_keepalive_connections=5, max_connections=10)

_client: httpx.AsyncClient | None = None


def _headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {PRIOR_AUTH_TOKEN}"} if PRIOR_AUTH_TOKEN else {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Wire this into your app: FastAPI(lifespan=lifespan).

    One client for the process -- a client per request throws away the
    connection pool and pays a TLS handshake every time.
    """
    global _client
    _client = httpx.AsyncClient(headers=_headers(), timeout=_TIMEOUT, limits=_LIMITS)
    # Wake the instance now rather than making the first real user wait 90s.
    asyncio.create_task(_wake())
    try:
        yield
    finally:
        await _client.aclose()
        _client = None


async def _wake() -> None:
    try:
        await _client.get(f"{PRIOR_AUTH_URL}/health", timeout=httpx.Timeout(120.0, connect=10.0))
    except Exception:
        pass  # best effort; a failed warm-up must not block start-up


async def analyse(case: str) -> dict[str, Any]:
    """Send one free-text application, get the Report back.

    Raises HTTPException so the failure surfaces in your API rather than
    becoming a silent empty result.
    """
    if _client is None:
        raise RuntimeError("client not initialised -- is lifespan wired into your app?")

    try:
        response = await _client.post(f"{PRIOR_AUTH_URL}/analyze", json={"case": case})
    except httpx.TimeoutException as exc:
        # Almost always a cold instance. A retry usually lands warm.
        raise HTTPException(
            status_code=504,
            detail="Prior-auth service timed out (likely waking from idle). Retry once.",
        ) from exc
    except httpx.RequestError as exc:
        raise HTTPException(status_code=502, detail=f"Prior-auth service unreachable: {exc}") from exc

    if response.status_code == 200:
        return response.json()

    detail = response.json().get("detail", response.text) if response.headers.get(
        "content-type", ""
    ).startswith("application/json") else response.text

    if response.status_code == 422:
        raise HTTPException(status_code=422, detail=f"Rejected by prior-auth service: {detail}")
    if response.status_code == 503:
        # Missing/invalid GEMINI_API_KEY, or the daily model quota is spent.
        raise HTTPException(status_code=503, detail=f"Prior-auth reasoning unavailable: {detail}")
    raise HTTPException(status_code=502, detail=f"Prior-auth error {response.status_code}: {detail}")


def decide(report: dict[str, Any]) -> str:
    """Turn the Report into one routing decision.

    Reads `tally`, which counts only the rules the model marked as gating --
    background material such as dietary advice is already excluded, so a
    prevention target the patient missed cannot push a case to REJECT.
    """
    tally = report.get("tally") or {}
    if tally.get("FAIL"):
        return "REJECT"
    if tally.get("MISSING") or report.get("missing_mandatory"):
        return "MORE_INFO_NEEDED"
    if not report.get("routing", {}).get("matched"):
        return "NO_GUIDELINE_COVERAGE"
    return "APPROVE"


def resubmission_checklist(report: dict[str, Any]) -> list[str]:
    """What to ask the provider for, from the flagged items the model ranked."""
    return [f["action_required"] for f in report.get("flagged", []) if f.get("action_required")]


# ---------------------------------------------------------------------------
# Example route
# ---------------------------------------------------------------------------
app = FastAPI(lifespan=lifespan)


@app.post("/check-authorization")
async def check_authorization(payload: dict[str, Any]) -> dict[str, Any]:
    case = (payload.get("case") or "").strip()
    if len(case) < 20:
        raise HTTPException(status_code=422, detail="`case` must be at least 20 characters.")

    report = await analyse(case)
    return {
        "decision": decide(report),
        "confidence": report["confidence"]["score"],
        "confidence_band": report["confidence"]["band"],
        "explanation": report["overall_explanation"],
        "flagged": report["flagged"],
        "checklist": resubmission_checklist(report),
        "report": report,          # keep the full Report for audit
    }
