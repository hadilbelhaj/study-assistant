# Evaluation sets (Phase 4)

One JSONL file per course: `questions_<course>.jsonl`, one line per question:

```json
{"question": "...", "expected_source_file": "reseaux_ch3.pdf", "expected_page": 12, "expected_answer_gist": "..."}
```

~20 pairs per course, written from material you already know cold.
Check two things periodically as you add more PDFs:
- **recall** — was the right chunk actually retrieved?
- **faithfulness** — did the generated answer stick to the retrieved source?

Not needed for Phase 1 — this folder just exists now so the habit
starts once you're there.
