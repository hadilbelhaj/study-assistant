"""Minimal smoke test — chunking logic only, no PDF/docling needed.
Run with: pytest
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingest import chunk_pages


def test_chunk_pages_attaches_metadata():
    pages = [{"page": 1, "text": "Le WiMAX est un standard de reseau sans fil. " * 20}]
    chunks = chunk_pages(
        pages, course="Reseaux", lecture="Chapitre 3", doc_type="lecture",
        source_file="reseaux_ch3.pdf", language="fr",
    )

    assert len(chunks) > 0
    for chunk in chunks:
        assert chunk["course"] == "Reseaux"
        assert chunk["lecture"] == "Chapitre 3"
        assert chunk["page"] == 1
        assert chunk["text"]
