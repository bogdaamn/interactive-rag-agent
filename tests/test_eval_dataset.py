"""Guards the evaluation dataset itself (assignment §16).

A dataset whose expected_source names a file that isn't in the corpus, or whose
questions duplicate each other, would let the eval runner report a
meaningless score.
"""

import json
from pathlib import Path

from tests.fixtures import CORPUS_DIR

EVAL_PATH = Path(__file__).parent.parent / "eval" / "userdocs_eval.json"


def _load():
    return json.loads(EVAL_PATH.read_text(encoding="utf-8"))


def test_dataset_has_at_least_five_questions():
    assert len(_load()) >= 5


def test_every_entry_has_a_question_and_an_expected_source():
    for entry in _load():
        assert entry["question"].strip()
        assert entry["expected_source"].strip()


def test_every_expected_source_exists_in_the_fixture_corpus():
    for entry in _load():
        path = CORPUS_DIR / entry["expected_source"]
        assert path.exists(), f"expected_source not in corpus: {entry['expected_source']}"


def test_questions_are_unique():
    questions = [entry["question"] for entry in _load()]
    assert len(set(questions)) == len(questions)


def test_dataset_covers_more_than_one_source_document():
    sources = {entry["expected_source"] for entry in _load()}
    assert len(sources) >= 3, "an eval that only probes one document tests very little"
