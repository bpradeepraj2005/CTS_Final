"""
Query the Standard Treatment Workflows RAG system.

Usage:
    python query.py "when should dialysis be started in AKI?"
    python query.py                          # interactive REPL mode
    python query.py "..." --no-generate       # retrieval only, no LLM call
    python query.py "..." --k 8 --threshold 0.5
    python query.py "..." --specialty cardiology
    python query.py --rebuild "..."           # force re-index before answering
    python query.py --list-models             # show Gemini models available to your key
"""

import argparse
from pathlib import Path

import rag_core as core

DEFAULT_SOURCE = Path(__file__).parent.parent / "Vol1_meaning_based_extraction.txt"


def show_result(query: str, results, no_generate: bool):
    if not results:
        print("\nNo passages matched above the similarity threshold. Try lowering --threshold "
              "or rephrasing the question.\n")
        return

    print(f"\nTop {len(results)} retrieved passages:")
    for rank, (chunk, score) in enumerate(results, start=1):
        print(f"  [{rank}] score={score:.3f}  {chunk.citation()}")

    if no_generate:
        print()
        for chunk, score in results:
            print(f"--- [{chunk.citation()}]  (score={score:.3f}) ---")
            print(chunk.text)
            print()
        return

    answer, used_llm = core.generate_answer(query, results)
    label = "Answer" if used_llm else "Answer (retrieval-only fallback — see warning above)"
    print(f"\n{label}:\n{answer}\n")


def run_query(args, vectors, chunks, query: str):
    results = core.retrieve(
        query, vectors, chunks,
        k=args.k, min_score=args.threshold, specialty_filter=args.specialty,
    )
    show_result(query, results, args.no_generate)


def main():
    parser = argparse.ArgumentParser(description="Query the Standard Treatment Workflows RAG system.")
    parser.add_argument("question", nargs="?", help="Question to ask. Omit for interactive mode.")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE, help="Path to the source .txt file")
    parser.add_argument("--k", type=int, default=5, help="Number of passages to retrieve (default 5)")
    parser.add_argument("--threshold", type=float, default=0.3, help="Minimum cosine similarity (default 0.3)")
    parser.add_argument("--specialty", type=str, default=None, help="Restrict retrieval to one specialty, e.g. cardiology")
    parser.add_argument("--no-generate", action="store_true", help="Skip LLM generation; print raw retrieved passages only")
    parser.add_argument("--rebuild", action="store_true", help="Force rebuild the index before querying")
    parser.add_argument("--list-models", action="store_true", help="List Gemini models available to your API key and exit")
    args = parser.parse_args()

    if args.list_models:
        for name in core.list_available_models():
            print(name)
        return

    if not args.source.exists():
        raise SystemExit(f"Source file not found: {args.source}")

    stale = not core.index_is_fresh(args.source)
    if stale and not args.rebuild:
        print("No fresh index found for this source file — building it now "
              "(this happens once; future runs load the cached index instantly).")
    vectors, chunks = core.ensure_index(args.source, force=args.rebuild)

    if args.question:
        run_query(args, vectors, chunks, args.question)
        return

    print(f"Loaded index: {len(chunks)} chunks. Type a question, or 'exit' to quit.")
    while True:
        try:
            q = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q:
            continue
        if q.lower() in ("exit", "quit"):
            break
        run_query(args, vectors, chunks, q)


if __name__ == "__main__":
    main()
