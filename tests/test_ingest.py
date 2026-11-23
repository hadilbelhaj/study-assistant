
"""Unit tests for the current Docling-based ingestion pipeline."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from rag_study_assistant import ingest
from rag_study_assistant.schemas import Chunk


def make_doc_chunk(text: str, page_numbers=(2,), headings=None):
    """Create a lightweight Docling-like chunk for unit tests."""
    doc_items = [
        SimpleNamespace(prov=[SimpleNamespace(page_no=number)])
        for number in page_numbers
    ]

    return SimpleNamespace(
        text=text,
        meta=SimpleNamespace(doc_items=doc_items, headings=headings),
    )


def make_metadata() -> dict:
    return {
        "course": "reseaux",
        "study_year": "1dni",
        "semester": "S2",
        "lecture": "Chapitre 3 - WiMAX",
        "doc_type": "lecture",
        "source_file": "lecture3-wimax.pdf",
    }


class FakeChunker:
    """Minimal replacement for Docling HybridChunker."""

    def __init__(self, chunks):
        self.chunks = chunks

    def chunk(self, dl_doc):
        return self.chunks

    def contextualize(self, doc_chunk):
        return doc_chunk.text


# --------------------------------------------------
# Metadata resolution
# --------------------------------------------------

def test_resolve_metadata_from_filename(tmp_path):
    raw_dir = tmp_path / "raw"
    pdf_path = (
        raw_dir / "1DNI" / "s2" / "JAVA"
        / "lecture6-les-collections.pdf"
    )

    metadata = ingest.resolve_metadata(pdf_path, raw_dir)

    assert metadata["study_year"] == "1dni"
    assert metadata["semester"] == "S2"
    assert metadata["course"] == "java"
    assert metadata["doc_type"] == "lecture"
    assert metadata["lecture"] == "Chapitre 6 - Les collections"
    assert metadata["source_file"] == "lecture6-les-collections.pdf"


@pytest.mark.parametrize(
    ("filename", "expected_type", "expected_title"),
    [
        ("tp2-java-collections.pdf", "exercise", "TP 2 - Java collections"),
        ("td3-java-collections.pdf", "exercise", "TD 3 - Java collections"),
        ("examen1-java.pdf", "exam", "Examen 1 - Java"),
    ],
)
def test_resolve_metadata_for_document_types(
    tmp_path, filename, expected_type, expected_title
):
    raw_dir = tmp_path / "raw"
    pdf_path = raw_dir / "1dni" / "S2" / "java" / filename

    metadata = ingest.resolve_metadata(pdf_path, raw_dir)

    assert metadata["doc_type"] == expected_type
    assert metadata["lecture"] == expected_title


def test_resolve_metadata_rejects_invalid_directory_structure(tmp_path):
    raw_dir = tmp_path / "raw"
    pdf_path = raw_dir / "1dni" / "S2" / "java" / "extra" / "lecture1-java.pdf"

    with pytest.raises(ValueError, match="Invalid PDF path"):
        ingest.resolve_metadata(pdf_path, raw_dir)


def test_resolve_metadata_rejects_invalid_semester(tmp_path):
    raw_dir = tmp_path / "raw"
    pdf_path = raw_dir / "1dni" / "S3" / "java" / "lecture1-java.pdf"

    with pytest.raises(ValueError, match="Invalid semester"):
        ingest.resolve_metadata(pdf_path, raw_dir)


# --------------------------------------------------
# Document identity and processing state
# --------------------------------------------------

def test_get_document_id_returns_sha256(tmp_path):
    pdf_path = tmp_path / "sample.pdf"
    content = b"Example PDF content for testing."
    pdf_path.write_bytes(content)

    expected = hashlib.sha256(content).hexdigest()

    assert ingest.get_document_id(pdf_path) == expected


def test_processed_ids_can_be_saved_and_loaded(tmp_path):
    state_path = tmp_path / "processed" / "already_processed.json"
    document_ids = {"hash-a", "hash-b", "hash-c"}

    assert ingest.load_processed_ids(state_path) == set()

    ingest.save_processed_ids(state_path, document_ids)

    assert ingest.load_processed_ids(state_path) == document_ids


@pytest.mark.parametrize(
    "content",
    [
        "{invalid json",
        '{"document": "not a list"}',
    ],
)
def test_load_processed_ids_handles_invalid_state(tmp_path, content):
    state_path = tmp_path / "already_processed.json"
    state_path.write_text(content, encoding="utf-8")

    assert ingest.load_processed_ids(state_path) == set()


# --------------------------------------------------
# Language, page and section metadata
# --------------------------------------------------

def test_detect_document_language(monkeypatch):
    document = SimpleNamespace(
        export_to_markdown=lambda: "Le réseau WiMAX utilise plusieurs techniques."
    )

    monkeypatch.setattr(ingest, "detect_language", lambda text: "fr")

    assert ingest.detect_document_language(document) == "fr"


def test_detect_document_language_rejects_empty_document():
    document = SimpleNamespace(export_to_markdown=lambda: "   ")

    with pytest.raises(ValueError, match="Cannot detect language"):
        ingest.detect_document_language(document)


def test_get_page_range_returns_first_and_last_pages():
    doc_chunk = make_doc_chunk(
        "Example content",
        page_numbers=(3, 4, 5),
    )

    assert ingest.get_page_range(doc_chunk) == (3, 5)


def test_get_page_range_handles_missing_pages():
    doc_chunk = make_doc_chunk("Example content", page_numbers=())

    assert ingest.get_page_range(doc_chunk) == (None, None)


def test_get_section_returns_hierarchical_headings():
    doc_chunk = make_doc_chunk(
        "Example content",
        headings=["Chapitre 3", "OFDM", "Principe"],
    )

    assert ingest.get_section(doc_chunk) == "Chapitre 3 > OFDM > Principe"


def test_get_section_handles_missing_headings():
    doc_chunk = make_doc_chunk("Example content", headings=[])

    assert ingest.get_section(doc_chunk) is None


# --------------------------------------------------
# Chunk identity and generation
# --------------------------------------------------

def test_make_chunk_id_is_deterministic():
    first = ingest.make_chunk_id("document-123", 0, "Example content")
    second = ingest.make_chunk_id("document-123", 0, "Example content")
    different = ingest.make_chunk_id("document-123", 1, "Example content")

    assert first == second
    assert first != different


def test_chunk_document_creates_valid_chunks_and_skips_short_text():
    short_chunk = make_doc_chunk("Too short")

    valid_text = (
        "Le WiMAX est un standard de réseau sans fil qui utilise "
        "plusieurs techniques de transmission pour améliorer "
        "les performances et la couverture du réseau."
    )
    valid_chunk = make_doc_chunk(
        valid_text,
        page_numbers=(4, 5),
        headings=["Chapitre 3", "WiMAX"],
    )

    chunker = FakeChunker([short_chunk, valid_chunk])

    chunks = ingest.chunk_document(
        document=object(),
        metadata=make_metadata(),
        document_id="document-123",
        language="fr",
        chunker=chunker,
    )

    assert len(chunks) == 1

    chunk = chunks[0]

    assert isinstance(chunk, Chunk)
    assert chunk.text == valid_text
    assert chunk.course == "reseaux"
    assert chunk.study_year == "1dni"
    assert chunk.semester == "S2"
    assert chunk.doc_type == "lecture"
    assert chunk.language == "fr"
    assert chunk.source_file == "lecture3-wimax.pdf"
    assert chunk.page_start == 4
    assert chunk.page_end == 5
    assert chunk.section == "Chapitre 3 > WiMAX"
    assert chunk.chunk_index == 1


# --------------------------------------------------
# JSONL persistence
# --------------------------------------------------

def test_save_chunks_writes_valid_jsonl(tmp_path):
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"

    pdf_path = (
        raw_dir / "1dni" / "S2" / "reseaux"
        / "lecture3-wimax.pdf"
    )

    chunk = Chunk(
        id=ingest.make_chunk_id("document-123", 0, "Example content"),
        document_id="document-123",
        text="Example content for JSONL persistence.",
        course="reseaux",
        study_year="1dni",
        semester="S2",
        lecture="Chapitre 3 - WiMAX",
        doc_type="lecture",
        language="fr",
        source_file="lecture3-wimax.pdf",
        page_start=2,
        page_end=2,
        section="Chapitre 3",
        chunk_index=0,
    )

    output_path = ingest.save_chunks(
        chunks=[chunk],
        pdf_path=pdf_path,
        raw_dir=raw_dir,
        processed_dir=processed_dir,
    )

    expected_path = (
        processed_dir / "1dni" / "S2" / "reseaux"
        / "document-123.jsonl"
    )

    assert output_path == expected_path
    assert output_path.exists()

    lines = output_path.read_text(encoding="utf-8").splitlines()

    assert len(lines) == 1

    saved_chunk = json.loads(lines[0])

    assert saved_chunk["document_id"] == "document-123"
    assert saved_chunk["text"] == "Example content for JSONL persistence."
    assert saved_chunk["page_start"] == 2
    assert saved_chunk["page_end"] == 2
    assert saved_chunk["language"] == "fr"
