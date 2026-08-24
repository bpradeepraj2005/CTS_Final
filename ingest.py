"""
Build (or rebuild) the persisted vector index for the RAG system.

Usage:
    python ingest.py                 # build only if the source changed since last run
    python ingest.py --rebuild       # force a full re-embed
    python ingest.py --source PATH   # use a different source .txt file
"""

import argparse
from pathlib import Path

import rag_core as core

DEFAULT_SOURCE = Path(__file__).parent.parent / "Vol1_meaning_based_extraction.txt"


def main():
    parser = argparse.ArgumentParser(description="Build the persisted vector index.")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE, help="Path to the source .txt file")
    parser.add_argument("--rebuild", action="store_true", help="Force re-embedding even if a fresh index exists")
    args = parser.parse_args()

    if not args.source.exists():
        raise SystemExit(f"Source file not found: {args.source}")

    if not args.rebuild and core.index_is_fresh(args.source):
        print(f"Index is already up to date for {args.source.name}. Use --rebuild to force.")
        return

    print(f"Parsing {args.source.name} ...")
    chunks = core.build_chunks(args.source)
    print(f"Found {len(chunks)} chunks across "
          f"{len(set(c.specialty for c in chunks))} specialties.")

    print(f"Embedding with {core.EMBED_MODEL} ...")
    core.build_index(args.source, force=True)
    print(f"Index saved to {core.DATA_DIR}")
    print("(previous index, if any, was backed up to data/backups/)")


if __name__ == "__main__":
    main()
