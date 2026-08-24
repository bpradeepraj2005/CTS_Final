"""Thin Gemini wrapper: config from the environment, one retry on transient
failures, and errors that say what actually went wrong.

Deliberately not a fallback-to-canned-text layer. If the model cannot be
reached the caller gets an exception and the API returns 503 -- silently
degrading to a stub would hand back an "analysis" that no model produced.
"""

from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Local convenience only; on a deployed host the vars come from the environment.
try:
    from dotenv import load_dotenv

    for _candidate in (Path(__file__).parent.parent / ".env",
                       Path(__file__).parent.parent / "rag" / ".env"):
        if _candidate.exists():
            load_dotenv(_candidate)
            break
except ImportError:  # dotenv is optional
    pass

DEFAULT_MODEL = "models/gemini-3.6-flash"


class LLMUnavailable(RuntimeError):
    """Raised when the model cannot be reached or returns nothing usable."""


@dataclass
class Usage:
    model: str
    prompt_tokens: int | None
    output_tokens: int | None
    latency_s: float


def api_key() -> str:
    key = (os.getenv("GEMINI_API_KEY") or "").strip()
    if not key:
        raise LLMUnavailable(
            "GEMINI_API_KEY is not set. Add it to the environment "
            "(Render: Settings -> Environment) or to a local .env file."
        )
    return key


def chat_model() -> str:
    return (os.getenv("GEMINI_CHAT_MODEL") or DEFAULT_MODEL).strip()


def configured() -> bool:
    """True when a call could be attempted. Used by /health."""
    return bool((os.getenv("GEMINI_API_KEY") or "").strip())


_client = None


def _get_client():
    global _client
    if _client is None:
        try:
            from google import genai
        except ImportError as exc:  # pragma: no cover
            raise LLMUnavailable("google-genai is not installed") from exc
        _client = genai.Client(api_key=api_key())
    return _client


def generate(
    prompt: str,
    system_instruction: str,
    *,
    temperature: float = 0.4,
    max_output_tokens: int | None = None,
    response_mime_type: str | None = None,
    attempts: int = 3,
) -> tuple[str, Usage]:
    """One completion. Retries transient failures with jittered backoff; does
    not retry a bad request, which would just fail the same way again."""
    model = chat_model()
    config: dict[str, Any] = {
        "system_instruction": system_instruction,
        "temperature": temperature,
    }
    if max_output_tokens:
        config["max_output_tokens"] = max_output_tokens
    if response_mime_type:
        config["response_mime_type"] = response_mime_type

    client = _get_client()
    last: Exception | None = None
    started = time.time()

    for attempt in range(attempts):
        try:
            response = client.models.generate_content(
                model=model, contents=prompt, config=config
            )
            text = (response.text or "").strip()
            if not text:
                raise LLMUnavailable(f"{model} returned an empty response")

            meta = getattr(response, "usage_metadata", None)
            return text, Usage(
                model=model,
                prompt_tokens=getattr(meta, "prompt_token_count", None),
                output_tokens=getattr(meta, "candidates_token_count", None),
                latency_s=round(time.time() - started, 2),
            )
        except Exception as exc:  # noqa: BLE001 - classified just below
            last = exc
            message = str(exc).lower()
            # A per-DAY quota will not clear within any backoff we could sit
            # through, so retrying just burns latency (and the caller's
            # patience) before failing the same way. Fail fast and say so.
            if "perday" in message.replace("_", "").replace("-", "") or (
                "quota" in message and "per day" in message
            ):
                raise LLMUnavailable(
                    f"{model} daily free-tier quota is exhausted. Raise the quota, "
                    f"enable billing, or switch GEMINI_CHAT_MODEL to a model with "
                    f"remaining allowance. Original error: {exc}"
                ) from exc
            transient = any(
                s in message
                for s in ("429", "rate", "503", "500", "unavailable",
                          "timeout", "deadline", "overloaded")
            )
            if not transient or attempt == attempts - 1:
                break
            time.sleep((2 ** attempt) + random.random())

    raise LLMUnavailable(f"{model} call failed: {last}") from last
