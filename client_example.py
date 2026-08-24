"""How another backend calls this API.

Two things worth copying into your own service:

1. Reuse one `httpx.AsyncClient` for the process. A new client per request
   throws away the connection pool and adds a TLS handshake every time.
2. A free-tier instance sleeps after ~15 minutes idle and takes about a
   minute to wake, so the first call after a quiet period needs a
   generous timeout.

    pip install httpx
    API_URL=https://<your-service>.onrender.com python client_example.py
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any

import httpx

API_URL = os.getenv("API_URL", "http://localhost:7860").rstrip("/")
API_TOKEN = os.getenv("API_TOKEN", "").strip()

_headers = {"Authorization": f"Bearer {API_TOKEN}"} if API_TOKEN else {}
# connect=10s, read=90s: covers a cold instance waking from sleep.
_timeout = httpx.Timeout(90.0, connect=10.0)


async def adjudicate(request: dict[str, Any], client: httpx.AsyncClient) -> dict[str, Any]:
    response = await client.post(f"{API_URL}/adjudicate", json=request)
    response.raise_for_status()
    return response.json()


async def main() -> None:
    sample = {
        "request_id": "PA-2026-90152",
        "patient": {"age": 54, "sex": "M"},
        "diagnosis": {"icd10": ["N17.9"], "text": "Acute Kidney Injury following viral gastroenteritis"},
        "requested_service": {
            "text": "Outpatient Hemodialysis Evaluation & Procedure",
            "setting": "outpatient",
            "facility_level": "tertiary",
        },
        "clinical_findings": [
            {"parameter": "hyperkalemia", "value": False, "provenance": "structured_report"},
            {"parameter": "creatinine_mg_dl", "value": 2.4, "unit": "mg/dl"},
            {"parameter": "urine_output_ml_per_day", "value": 700, "unit": "ml/day"},
        ],
        "documentation_present": ["renal_function_test", "ecg"],
        "eligibility": {
            "enrollment_active_on_service_date": True,
            "benefit_covers_service": True,
            "service_not_in_plan_exclusions": True,
            "waiting_period_satisfied": True,
            "annual_or_lifetime_limit_available": True,
            "provider_empanelled": True,
            "prior_auth_actually_required": True,
            "no_duplicate_active_authorization": True,
        },
    }

    async with httpx.AsyncClient(headers=_headers, timeout=_timeout) as client:
        health = (await client.get(f"{API_URL}/health")).json()
        print("health:", json.dumps(health, indent=2))

        report = await adjudicate(sample, client)
        print("\nreport:", json.dumps(report, indent=2))
        print("\nfinal_decision:", report["final_decision"])


if __name__ == "__main__":
    asyncio.run(main())
