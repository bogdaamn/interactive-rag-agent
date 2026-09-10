# Fixture corpus

Four small documents, one per supported format, used by
`tests/userdocs/test_pipeline_e2e.py` and `scripts/run_userdocs_eval.py`.

Regenerate with:

```bash
pip install reportlab python-docx
python tests/fixtures/generate_corpus.py
```

`reportlab` is only needed to generate the `.pdf` — it is not a runtime or test
dependency and must not be added to `requirements.txt`. The generated files are
committed so the test suite needs neither package nor a network connection.

`vacation_policy.pdf` is deliberately two pages, with the carry-over rule on
page 2, so PDF page attribution (bonus +1) is verifiable rather than trivially
always "page 1".
