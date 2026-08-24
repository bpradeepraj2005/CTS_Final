"""
Core library for the Standard Treatment Workflows RAG system.

Pipeline:
  parse_document()   -> semantic chunks (one per medical condition)
  build_index()       -> embeds chunks via Gemini, persists to disk (with backup rotation)
  load_index()         -> loads a persisted index, verifying it matches the source file
  retrieve()            -> cosine-similarity top-k search over the persisted index
  generate_answer()   -> grounded Gemini answer over retrieved chunks, with retrieval-only fallback
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
BACKUP_DIR = DATA_DIR / "backups"
EMBEDDINGS_PATH = DATA_DIR / "embeddings.npy"
METADATA_PATH = DATA_DIR / "metadata.json"
MAX_BACKUPS = 3

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
EMBED_MODEL = os.getenv("GEMINI_EMBED_MODEL", "models/gemini-embedding-001")
CHAT_MODEL = os.getenv("GEMINI_CHAT_MODEL", "models/gemini-3.6-flash")
MAX_CHUNK_CHARS = 1400
CHUNK_OVERLAP_CHARS = 200


def _client():
    if not GEMINI_API_KEY:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Copy .env.example to .env and fill it in."
        )
    from google import genai
    return genai.Client(api_key=GEMINI_API_KEY)


# ---------------------------------------------------------------------------
# Chunk data model
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    id: int
    specialty: str
    condition: str
    text: str
    start_line: int
    end_line: int

    def citation(self) -> str:
        return f"{self.specialty} > {self.condition} (lines {self.start_line}-{self.end_line})"


# ---------------------------------------------------------------------------
# Parsing: split the source file into one chunk per medical condition
# ---------------------------------------------------------------------------

_DIVIDER_RE = re.compile(r"^=+$")
_SPECIALTY_RE = re.compile(r"^\d+\.\s+(.+)$")


def _is_all_caps_header(line: str) -> bool:
    s = line.strip()
    if not s or len(s) < 3:
        return False
    if not any(c.isalpha() for c in s):
        return False
    if _SPECIALTY_RE.match(s):
        return False
    return s == s.upper() and s.lower() != s.upper()


def parse_document(path: Path) -> list[Chunk]:
    lines = path.read_text(encoding="utf-8").splitlines()

    chunks: list[Chunk] = []
    chunk_id = 0

    # Front matter: everything before the first "====" divider becomes its own chunk.
    first_divider = next((i for i, l in enumerate(lines) if _DIVIDER_RE.match(l.strip())), len(lines))
    if first_divider > 0:
        front = "\n".join(lines[:first_divider]).strip()
        if front:
            chunks.append(Chunk(chunk_id, "Document Info", "Source & Purpose", front, 1, first_divider))
            chunk_id += 1

    specialty = "General"
    i = first_divider
    n = len(lines)
    condition = None
    cond_start = None
    cond_lines: list[str] = []

    def flush_condition(end_line: int):
        nonlocal chunk_id
        if condition is not None and cond_lines:
            text = "\n".join(cond_lines).strip()
            if text:
                chunks.append(Chunk(chunk_id, specialty, condition, text, cond_start, end_line))
                chunk_id += 1

    while i < n:
        raw = lines[i]
        s = raw.strip()

        if _DIVIDER_RE.match(s):
            # Specialty header block: "====" / "N. NAME" / "====" / blank
            flush_condition(i)
            condition, cond_lines = None, []
            if i + 1 < n:
                m = _SPECIALTY_RE.match(lines[i + 1].strip())
                specialty = m.group(1).strip() if m else lines[i + 1].strip()
            # skip past the header block (divider, title, divider)
            j = i + 1
            while j < n and not _DIVIDER_RE.match(lines[j].strip()):
                j += 1
            i = j + 1
            continue

        if _is_all_caps_header(s):
            flush_condition(i)
            condition = s
            cond_start = i + 1
            cond_lines = []
            i += 1
            continue

        if condition is not None:
            cond_lines.append(raw)
        i += 1

    flush_condition(n)
    return chunks


def _split_large_chunk(chunk: Chunk) -> list[Chunk]:
    """Paragraph-based sub-split with overlap, only for chunks that exceed MAX_CHUNK_CHARS."""
    if len(chunk.text) <= MAX_CHUNK_CHARS:
        return [chunk]

    paragraphs = re.split(r"\n\s*\n", chunk.text)
    parts: list[str] = []
    current = ""
    for para in paragraphs:
        candidate = f"{current}\n\n{para}".strip() if current else para
        if len(candidate) > MAX_CHUNK_CHARS and current:
            parts.append(current)
            overlap = current[-CHUNK_OVERLAP_CHARS:]
            current = f"{overlap}\n\n{para}".strip()
        else:
            current = candidate
    if current:
        parts.append(current)

    return [
        Chunk(-1, chunk.specialty, f"{chunk.condition} (part {idx + 1}/{len(parts)})",
              part, chunk.start_line, chunk.end_line)
        for idx, part in enumerate(parts)
    ]


def build_chunks(path: Path) -> list[Chunk]:
    raw_chunks = parse_document(path)
    expanded: list[Chunk] = []
    for c in raw_chunks:
        expanded.extend(_split_large_chunk(c))
    for idx, c in enumerate(expanded):
        c.id = idx
    return expanded


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def _embed_batch(texts: list[str], task_type: str) -> np.ndarray:
    from google.genai import types
    client = _client()
    vectors = []
    # Gemini embedding API accepts batches; keep batches modest to stay under request limits.
    batch_size = 32
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        result = client.models.embed_content(
            model=EMBED_MODEL,
            contents=batch,
            config=types.EmbedContentConfig(task_type=task_type),
        )
        vectors.extend([e.values for e in result.embeddings])
    return np.array(vectors, dtype=np.float32)


def embed_documents(texts: list[str]) -> np.ndarray:
    return _embed_batch(texts, task_type="RETRIEVAL_DOCUMENT")


def embed_query(text: str) -> np.ndarray:
    return _embed_batch([text], task_type="RETRIEVAL_QUERY")[0]


# ---------------------------------------------------------------------------
# Index: build / persist / load (with backup rotation = the "backup feature")
# ---------------------------------------------------------------------------

def _source_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rotate_backup():
    """Move the current index into data/backups/<timestamp>/ before it gets overwritten,
    keeping only the most recent MAX_BACKUPS snapshots."""
    if not (EMBEDDINGS_PATH.exists() and METADATA_PATH.exists()):
        return
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    snap_dir = BACKUP_DIR / time.strftime("%Y%m%d-%H%M%S")
    snap_dir.mkdir(exist_ok=True)
    shutil.copy2(EMBEDDINGS_PATH, snap_dir / EMBEDDINGS_PATH.name)
    shutil.copy2(METADATA_PATH, snap_dir / METADATA_PATH.name)

    snapshots = sorted(p for p in BACKUP_DIR.iterdir() if p.is_dir())
    for old in snapshots[:-MAX_BACKUPS]:
        shutil.rmtree(old, ignore_errors=True)


def index_is_fresh(source_path: Path) -> bool:
    if not (EMBEDDINGS_PATH.exists() and METADATA_PATH.exists()):
        return False
    try:
        meta = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    return (
        meta.get("source_hash") == _source_hash(source_path)
        and meta.get("embed_model") == EMBED_MODEL
    )


def build_index(source_path: Path, force: bool = False) -> tuple[np.ndarray, list[Chunk]]:
    if not force and index_is_fresh(source_path):
        return load_index()

    chunks = build_chunks(source_path)
    vectors = embed_documents([c.text for c in chunks])

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    _rotate_backup()

    np.save(EMBEDDINGS_PATH, vectors)
    METADATA_PATH.write_text(
        json.dumps({
            "source_hash": _source_hash(source_path),
            "embed_model": EMBED_MODEL,
            "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "chunks": [asdict(c) for c in chunks],
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return vectors, chunks


def load_index() -> tuple[np.ndarray, list[Chunk]]:
    vectors = np.load(EMBEDDINGS_PATH)
    meta = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    chunks = [Chunk(**c) for c in meta["chunks"]]
    return vectors, chunks


def ensure_index(source_path: Path, force: bool = False) -> tuple[np.ndarray, list[Chunk]]:
    """Load the cached index if it matches the source file; otherwise (re)build it.
    This is the persisted-vector-index backup feature: normal runs never re-embed."""
    if force or not index_is_fresh(source_path):
        return build_index(source_path, force=True)
    return load_index()


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def retrieve(
    query: str,
    vectors: np.ndarray,
    chunks: list[Chunk],
    k: int = 5,
    min_score: float = 0.0,
    specialty_filter: Optional[str] = None,
) -> list[tuple[Chunk, float]]:
    q = embed_query(query)
    q = q / (np.linalg.norm(q) + 1e-8)
    v = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-8)
    scores = v @ q

    order = np.argsort(-scores)
    results: list[tuple[Chunk, float]] = []
    for idx in order:
        chunk = chunks[idx]
        score = float(scores[idx])
        if score < min_score:
            break
        if specialty_filter and specialty_filter.lower() not in chunk.specialty.lower():
            continue
        results.append((chunk, score))
        if len(results) >= k:
            break
    return results


# ---------------------------------------------------------------------------
# Generation (grounded, with retrieval-only fallback if the chat model call fails)
# ---------------------------------------------------------------------------

SYSTEM_INSTRUCTION = (
    "You are a retrieval-grounded assistant answering questions about the Standard "
    "Treatment Workflows of India. Answer ONLY using the provided context passages. "
    "Cite the specialty/condition for every claim you make, in the form "
    "(Specialty > Condition). If the context does not contain the answer, say so "
    "explicitly instead of guessing. Do not add outside medical knowledge."
)


def generate_answer(query: str, results: list[tuple[Chunk, float]]) -> tuple[str, bool]:
    """Returns (answer_text, used_llm). Falls back to a raw-context dump if the
    Gemini call fails for any reason, so a broken/misconfigured API key or an
    invalid model name degrades gracefully instead of crashing."""
    context = "\n\n---\n\n".join(
        f"[{c.citation()}]\n{c.text}" for c, _ in results
    )
    prompt = f"Context passages:\n\n{context}\n\nQuestion: {query}\n\nAnswer:"

    try:
        client = _client()
        response = client.models.generate_content(
            model=CHAT_MODEL,
            contents=prompt,
            config={"system_instruction": SYSTEM_INSTRUCTION},
        )
        text = response.text
        if not text:
            raise RuntimeError("Empty response from model")
        return text, True
    except Exception as e:  # noqa: BLE001 - deliberate broad catch for graceful fallback
        fallback = (
            f"[Generation unavailable: {e}]\n"
            f"Falling back to raw retrieved passages for model '{CHAT_MODEL}'.\n"
            f"If this is a 'model not found' error, run with --list-models to see "
            f"available Gemini models for your API key and update GEMINI_CHAT_MODEL in .env.\n\n"
            + "\n\n".join(f"[{c.citation()}]\n{c.text}" for c, _ in results)
        )
        return fallback, False


def list_available_models() -> list[str]:
    client = _client()
    return [m.name for m in client.models.list()]
