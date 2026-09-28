"""PDF ingestion pipeline.

Flow:
    PDF
    -> DoclingDocument
    -> language detection
    -> HybridChunker
    -> contextualized chunks
    -> Pydantic validation
    -> JSONL
"""

import hashlib
import json
import logging
import re
import uuid
from pathlib import Path

import yaml
from docling.chunking import HybridChunker
from docling.document_converter import DocumentConverter
from docling_core.transforms.chunker.tokenizer.huggingface import (
    HuggingFaceTokenizer,
)
from transformers import AutoTokenizer

from src.language import detect_language
from src.schemas import Chunk

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Filename / metadata rules
# ---------------------------------------------------------------------------

PREFIX_INFO = {
    "lecture": ("lecture", "Chapitre"),
    "tp": ("exercise", "TP"),
    "td": ("exercise", "TD"),
    "examen": ("exam", "Examen"),
}

PROPER_NOUNS = {
    "java": "Java",
    "jdbc": "JDBC",
    "bd": "BD",
}

FILENAME_PATTERN = re.compile(r"([a-z]+)(\d+)(?:-part(\d+))?-(.+)")


# ---------------------------------------------------------------------------
# Processing state
# ---------------------------------------------------------------------------

def get_document_id(pdf_path: Path) -> str:
    """Return a SHA-256 hash of the PDF content."""
    sha256 = hashlib.sha256()
    with pdf_path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            sha256.update(chunk)
    return sha256.hexdigest()


def load_processed_ids(path: Path) -> set[str]:
    """Load IDs of documents that were already processed."""
    if not path.exists() or path.stat().st_size == 0:
        return set()

    try:
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)
        if not isinstance(data, list):
            raise ValueError(
                "already_processed.json must contain a JSON list."
            )
        return set(data)
    except (json.JSONDecodeError, ValueError) as exc:
        logger.warning(
            "Invalid processing state in %s: %s. "
            "Starting with an empty processed set.",
            path,
            exc,
        )
        return set()


def save_processed_ids(path: Path, processed_ids: set[str]) -> None:
    """Persist processed document IDs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(sorted(processed_ids), file, indent=2)


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

def resolve_metadata(pdf_path: Path, raw_dir: Path) -> dict:
    """Resolve metadata from directory structure and filename.

    Expected structure:
        raw/<study_year>/<semester>/<course>/<file>.pdf

    Example:
        raw/1dni/s1/java/lecture5-les-exceptions.pdf
    """
    relative_path = pdf_path.relative_to(raw_dir)
    if len(relative_path.parts) != 4:
        raise ValueError(
            f"Invalid PDF path: {pdf_path}. "
            "Expected: raw/<study_year>/<semester>/<course>/<file>.pdf"
        )

    study_year, semester, course, filename = relative_path.parts
    study_year = study_year.lower().strip()
    semester = semester.upper().strip()
    course = course.lower().strip()

    if semester not in {"S1", "S2"}:
        raise ValueError(
            f"Invalid semester '{semester}' in {pdf_path}. Expected S1 or S2."
        )

    stem = pdf_path.stem.lower()
    match = FILENAME_PATTERN.fullmatch(stem)

    if match:
        prefix, number, part, title = match.groups()
        doc_type, label = PREFIX_INFO.get(
            prefix,
            ("lecture", prefix.capitalize()),
        )
        words = [
            PROPER_NOUNS.get(word, word)
            for word in title.split("-")
        ]
        title_body = " ".join(words)
        if title_body:
            title_body = title_body[0].upper() + title_body[1:]

        part_suffix = f" (Partie {part})" if part else ""
        lecture = f"{label} {number}{part_suffix} - {title_body}"
    else:
        logger.warning(
            "Filename does not match naming convention: %s",
            pdf_path.name,
        )
        doc_type = "lecture"
        lecture = pdf_path.stem

    return {
        "study_year": study_year,
        "semester": semester,
        "course": course,
        "doc_type": doc_type,
        "lecture": lecture,
        "source_file": filename,
    }


# ---------------------------------------------------------------------------
# Docling
# ---------------------------------------------------------------------------

def build_chunker(cfg: dict) -> HybridChunker:
    """Create the HybridChunker using the embedding tokenizer."""
    embedding_model = cfg["embedding"]["model"]
    max_tokens = cfg["chunking"]["chunk_size_tokens"]

    tokenizer = HuggingFaceTokenizer(
        tokenizer=AutoTokenizer.from_pretrained(embedding_model),
        max_tokens=max_tokens,
    )
    return HybridChunker(
        tokenizer=tokenizer,
        merge_peers=True,
        repeat_table_header=True,
    )


def extract_document(pdf_path: Path, converter: DocumentConverter):
    """Convert a PDF into a DoclingDocument."""
    try:
        result = converter.convert(source=str(pdf_path))
    except Exception:
        logger.exception("Failed to convert %s", pdf_path.name)
        return None

    return result.document


# ---------------------------------------------------------------------------
# Language detection
# ---------------------------------------------------------------------------

def detect_document_language(document) -> str:
    """Detect the primary language of a DoclingDocument."""
    text = document.export_to_markdown()
    if not text.strip():
        raise ValueError(
            "Cannot detect language: document contains no text."
        )
    return detect_language(text)


# ---------------------------------------------------------------------------
# Chunk metadata extraction
# ---------------------------------------------------------------------------

def get_page_range(doc_chunk) -> tuple[int | None, int | None]:
    """Extract the page range covered by a Docling chunk."""
    page_numbers = []
    if not doc_chunk.meta:
        return None, None

    for doc_item in doc_chunk.meta.doc_items:
        for provenance in doc_item.prov:
            page_numbers.append(provenance.page_no)

    if not page_numbers:
        return None, None

    return min(page_numbers), max(page_numbers)


def get_section(doc_chunk) -> str | None:
    """Return the hierarchical heading context of a chunk."""
    if not doc_chunk.meta:
        return None
    headings = doc_chunk.meta.headings
    if not headings:
        return None
    return " > ".join(headings)


# ---------------------------------------------------------------------------
# Chunk identity
# ---------------------------------------------------------------------------

def make_chunk_id(
    document_id: str,
    chunk_index: int,
    text: str,
) -> uuid.UUID:
    """Create a deterministic UUID for a chunk."""
    identity = f"{document_id}|{chunk_index}|{text}"
    return uuid.uuid5(uuid.NAMESPACE_URL, identity)


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def chunk_document(
    document,
    metadata: dict,
    document_id: str,
    language: str,
    chunker: HybridChunker,
) -> list[Chunk]:
    """Chunk a DoclingDocument and convert the chunks to our schema."""
    chunks = []

    for chunk_index, doc_chunk in enumerate(chunker.chunk(dl_doc=document)):
        text = chunker.contextualize(doc_chunk)
        if not text.strip():
            continue

        page_start, page_end = get_page_range(doc_chunk)
        section = get_section(doc_chunk)

        chunk = Chunk(
            id=make_chunk_id(
                document_id=document_id,
                chunk_index=chunk_index,
                text=text,
            ),
            document_id=document_id,
            text=text,
            course=metadata["course"],
            study_year=metadata["study_year"],
            semester=metadata["semester"],
            lecture=metadata["lecture"],
            doc_type=metadata["doc_type"],
            language=language,
            source_file=metadata["source_file"],
            page_start=page_start,
            page_end=page_end,
            section=section,
            chunk_index=chunk_index,
        )
        chunks.append(chunk)

    return chunks


# ---------------------------------------------------------------------------
# Per-document pipeline
# ---------------------------------------------------------------------------

def ingest_pdf(
    pdf_path: Path,
    metadata: dict,
    document_id: str,
    converter: DocumentConverter,
    chunker: HybridChunker,
) -> list[Chunk]:
    """Run extraction -> language detection -> chunking."""
    document = extract_document(pdf_path, converter)
    if document is None:
        return []

    language = detect_document_language(document)
    logger.info(
        "Detected language '%s' for %s",
        language,
        pdf_path.name,
    )

    return chunk_document(
        document=document,
        metadata=metadata,
        document_id=document_id,
        language=language,
        chunker=chunker,
    )


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_chunks(
    chunks: list[Chunk],
    pdf_path: Path,
    raw_dir: Path,
    processed_dir: Path,
) -> Path:
    """Write validated chunks to JSONL."""
    relative_path = pdf_path.relative_to(raw_dir)
    study_year, semester, course = relative_path.parts[:3]

    output_dir = Path(processed_dir) / study_year / semester / course
    output_dir.mkdir(parents=True, exist_ok=True)

    document_id = chunks[0].document_id
    output_path = output_dir / f"{document_id}.jsonl"

    with output_path.open("w", encoding="utf-8") as file:
        for chunk in chunks:
            file.write(
                json.dumps(
                    chunk.model_dump(mode="json"),
                    ensure_ascii=False,
                )
                + "\n"
            )

    return output_path


# ---------------------------------------------------------------------------
# Per-file orchestration
# ---------------------------------------------------------------------------

def process_pdf(
    pdf_path: Path,
    raw_dir: Path,
    processed_dir: Path,
    converter: DocumentConverter,
    chunker: HybridChunker,
) -> tuple[str, Path] | None:
    """Process one PDF and return its document ID and output path."""
    document_id = get_document_id(pdf_path)
    metadata = resolve_metadata(pdf_path, raw_dir)

    chunks = ingest_pdf(
        pdf_path=pdf_path,
        metadata=metadata,
        document_id=document_id,
        converter=converter,
        chunker=chunker,
    )
    if not chunks:
        return None

    output_path = save_chunks(
        chunks=chunks,
        pdf_path=pdf_path,
        raw_dir=raw_dir,
        processed_dir=processed_dir,
    )

    return document_id, output_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Run batch ingestion."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s | %(message)s",
    )

    config_path = Path("config.yaml")
    with config_path.open(encoding="utf-8") as file:
        cfg = yaml.safe_load(file)

    raw_dir = Path(cfg["paths"]["raw_dir"])
    processed_dir = Path(cfg["paths"]["processed_dir"])
    processed_ids_path = processed_dir / "already_processed.json"

    processed_ids = load_processed_ids(processed_ids_path)
    chunker = build_chunker(cfg)
    converter = DocumentConverter()

    pdf_paths = sorted(raw_dir.rglob("*.pdf"))
    if not pdf_paths:
        logger.warning("No PDFs found under %s", raw_dir)
        return

    for pdf_path in pdf_paths:
        document_id = get_document_id(pdf_path)

        if document_id in processed_ids:
            logger.info("Skipping already processed: %s", pdf_path.name)
            continue

        logger.info("Processing document %s", pdf_path.name)

        try:
            result = process_pdf(
                pdf_path=pdf_path,
                raw_dir=raw_dir,
                processed_dir=processed_dir,
                converter=converter,
                chunker=chunker,
            )

            if result is None:
                logger.warning(
                    "No chunks generated for %s",
                    pdf_path.name,
                )
                continue

            document_id, output_path = result

            processed_ids.add(document_id)
            save_processed_ids(processed_ids_path, processed_ids)

            logger.info("Wrote chunks -> %s", output_path)
            logger.info("Marked as processed: %s", pdf_path.name)

        except Exception:
            logger.exception("Failed to process %s", pdf_path)


if __name__ == "__main__":
    main()