"""RAG evaluation runner (assignment §16).

Ingests the fixture corpus into a throwaway database, runs every question in
eval/userdocs_eval.json through the real retrieval path, and reports whether
the expected source document was actually retrieved.

This is a reporting script, not a pytest test — same role as the existing
src/benchmark.py. It exits non-zero if any question misses, so it also works as
a CI gate and as a live demo (demo checklist item 11).

Usage:
    python scripts/run_userdocs_eval.py
"""

import json
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tests.fixtures import CORPUS_DIR  # noqa: E402
from userdocs.pipeline import ingest_document  # noqa: E402
from userdocs.retrieve import retrieve  # noqa: E402
from userdocs.store import UserDocsStore  # noqa: E402

EVAL_PATH = Path(__file__).parent.parent / "eval" / "userdocs_eval.json"
EVAL_USER_ID = 1


@dataclass
class EvalResult:
    question: str
    expected_source: str
    retrieved_sources: list
    hit: bool


def evaluate(store, user_id: int, dataset: list) -> list:
    results = []
    for entry in dataset:
        chunks = retrieve(store, user_id, entry["question"])

        seen = []
        for chunk in chunks:
            if chunk.filename not in seen:
                seen.append(chunk.filename)

        results.append(
            EvalResult(
                question=entry["question"],
                expected_source=entry["expected_source"],
                retrieved_sources=seen,
                hit=entry["expected_source"] in seen,
            )
        )
    return results


def format_report(results: list) -> str:
    if not results:
        return "No evaluation questions were run."

    lines = ["RAG evaluation", "=" * 60, ""]
    for result in results:
        marker = "✅" if result.hit else "❌"
        lines.append(f"{marker} {result.question}")
        lines.append(f"     expected: {result.expected_source}")
        lines.append(f"     retrieved: {', '.join(result.retrieved_sources) or '(nothing)'}")
        lines.append("")

    hits = sum(1 for result in results if result.hit)
    lines.append("=" * 60)
    lines.append(f"Retrieved the expected source for {hits}/{len(results)} questions.")
    return "\n".join(lines)


def main() -> int:
    dataset = json.loads(EVAL_PATH.read_text(encoding="utf-8"))

    with tempfile.TemporaryDirectory() as tmpdir:
        store = UserDocsStore(str(Path(tmpdir) / "eval.db"))
        try:
            for path in sorted(CORPUS_DIR.iterdir()):
                if path.is_file() and path.suffix.lower() in {".txt", ".md", ".docx", ".pdf"}:
                    ingest_document(
                        store, EVAL_USER_ID, filename=path.name, raw_bytes=path.read_bytes()
                    )
            results = evaluate(store, EVAL_USER_ID, dataset)
        finally:
            store.close()

    print(format_report(results))
    return 0 if all(result.hit for result in results) else 1


if __name__ == "__main__":
    sys.exit(main())
