"""PDF ingestion pipeline.

Flow:
    PDF
    -> resolve filesystem/filename metadata-> extract pages -> detect document language-> chunk-> validate chunks with Pydantic-> write JSONL
"""
import hashlib
import json
import logging
import re
import uuid
from pathlib import Path
import yaml
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer
from src.language import detect_language
from src.schemas import Chunk
logger = logging.getLogger(__name__)
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
FILENAME_PATTERN = re.compile(
    r"([a-z]+)(\d+)(?:-part(\d+))?-(.+)"
)
def load_processed_ids(path: Path) -> set[str]:
    """Load already processed document IDs."""

    if not path.exists():
        return set()

    if path.stat().st_size == 0:
        return set()

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return set(json.load(file))

def save_processed_ids(path: Path,processed_ids: set[str],) -> None:
    path.parent.mkdir(parents=True,exist_ok=True,)
    with path.open("w",encoding="utf-8",) as file:
        json.dump(sorted(processed_ids),file,indent=2,)
 

def get_document_id(pdf_path: Path) -> str:
    sha256 = hashlib.sha256()
    with pdf_path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            sha256.update(chunk)
    return sha256.hexdigest()

def resolve_metadata(pdf_path: Path,raw_dir: Path,) -> dict:
    relative_path = pdf_path.relative_to(raw_dir)
    if len(relative_path.parts) != 4:
        raise ValueError(
            f"Invalid PDF path: {pdf_path}. "
            "Expected: raw/<study_year>/<semester>/<course>/<file>.pdf"
        )
    study_year, semester, course, filename = relative_path.parts
    semester = semester.upper()
    course = course.lower()
    if semester not in {"S1", "S2"}:
        raise ValueError(
            f"Invalid semester '{semester}' in {pdf_path}"
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
            title_body = (
                title_body[0].upper()
                + title_body[1:]
            )
        part_suffix = (
            f" (Partie {part})"
            if part
            else ""
        )
        lecture = (
            f"{label} {number}"
            f"{part_suffix} - "
            f"{title_body}"
        )
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

def build_splitter(
    cfg: dict,
    tokenizer,
) -> RecursiveCharacterTextSplitter:
    """Create the text splitter once."""
    chunk_cfg = cfg["chunking"]
    return RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer,
        chunk_size=chunk_cfg["chunk_size_tokens"],
        chunk_overlap=int(
            chunk_cfg["chunk_size_tokens"]
            * chunk_cfg["chunk_overlap_pct"]
        ),
        separators=chunk_cfg["separators"],
    )

def extract_pages(
    pdf_path: Path,
    converter,
) -> list[dict]:
    """Extract PDF pages as markdown text."""
    try:
        result = converter.convert(str(pdf_path))
    except Exception:
        logger.exception(
            "Failed to convert PDF: %s",
            pdf_path.name,
        )
        return []
    pages = []
    for page_no in result.document.pages:
        text = result.document.export_to_markdown(
            page_no=page_no
        )
        if not text or not text.strip():
            continue
        pages.append({
            "page": page_no,
            "text": text,
        })
    return pages
def detect_document_language(
    pages: list[dict],
) -> str:
    """Detect language once for the whole document."""
    document_text = "\n".join(
        page["text"]
        for page in pages
        if page["text"].strip()
    )
    if not document_text.strip():
        raise ValueError(
            "Cannot detect language: document contains no text."
        )
    return detect_language(document_text)

def make_chunk_id(document_id: str,page: int,chunk_index: int,text: str,) -> uuid.UUID:
    identity = (
        f"{document_id}|"
        f"{page}|"
        f"{chunk_index}|"
        f"{text}"
    )

    return uuid.uuid5(
        uuid.NAMESPACE_URL,
        identity,
    )
def chunk_pages(
    pages: list[dict],
    metadata: dict,
    language: str,
    document_id: str,
    splitter: RecursiveCharacterTextSplitter,
) -> list[Chunk]:

    chunks = []
    chunk_index = 0

    for page in pages:
        pieces = splitter.split_text(page["text"])

        for piece in pieces:

            if not piece.strip():
                continue

            chunk = Chunk(
                id=make_chunk_id(
                    document_id=document_id,
                    page=page["page"],
                    chunk_index=chunk_index,
                    text=piece,
                ),
                document_id=document_id,
                text=piece,
                course=metadata["course"],
                study_year=metadata["study_year"],
                semester=metadata["semester"],
                lecture=metadata["lecture"],
                doc_type=metadata["doc_type"],
                language=language,
                source_file=metadata["source_file"],
                page=page["page"],
                chunk_index=chunk_index,
            )

            chunks.append(chunk)
            chunk_index += 1

    return chunks
def ingest_pdf(
    pdf_path: Path,
    metadata: dict,
    document_id: str,
    converter,
    splitter,
) -> list[Chunk]:

    pages = extract_pages(
        pdf_path,
        converter,
    )

    if not pages:
        logger.warning(
            "No text extracted from %s",
            pdf_path.name,
        )
        return []

    language = detect_document_language(pages)

    logger.info(
        "Detected language '%s' for %s",
        language,
        pdf_path.name,
    )

    return chunk_pages(
        pages=pages,
        metadata=metadata,
        language=language,
        document_id=document_id,
        splitter=splitter,
    )
# Persistence
def save_chunks(chunks: list[Chunk],pdf_path: Path,raw_dir: Path,processed_dir: Path,) -> Path:
    """Write validated chunks to a deterministic JSONL file."""
    relative_path = pdf_path.relative_to(raw_dir)
    study_year = relative_path.parts[0]
    semester = relative_path.parts[1]
    course = relative_path.parts[2]
    output_dir = (Path(processed_dir)/study_year/ semester/ course)
    output_dir.mkdir( parents=True,exist_ok=True,)
    source_identity = relative_path.as_posix()
    file_id = hashlib.md5(source_identity.encode("utf-8")).hexdigest()
    output_path = (output_dir/ f"{file_id}.jsonl")
    with output_path.open("w",encoding="utf-8",) as file:
        for chunk in chunks:
            file.write(
                json.dumps(
                    chunk.model_dump(mode="json"),
                    ensure_ascii=False,
                )
                + "\n")
    return output_path
# Per-file orchestration
def process_pdf(pdf_path: Path,raw_dir: Path,processed_dir: Path,converter,splitter,) -> Path | None:
    """Process one PDF from extraction to JSONL."""
    metadata = resolve_metadata( pdf_path,raw_dir,)

    chunks = ingest_pdf(pdf_path=pdf_path,metadata=metadata,converter=converter,splitter=splitter,)
    if not chunks:
        return None
    output_path = save_chunks(
        chunks=chunks,
        pdf_path=pdf_path,
        raw_dir=raw_dir,
        processed_dir=processed_dir,
    )
    logger.info("Wrote %d chunks -> %s",len(chunks),output_path,)
    return output_path
def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s | %(message)s",
    )

    # Load configuration the current way.
    config_path = Path("config.yaml")

    with config_path.open(
        encoding="utf-8"
    ) as file:
        cfg = yaml.safe_load(file)

    raw_dir = Path(
        cfg["paths"]["raw_dir"]
    )

    processed_dir = Path(
        cfg["paths"]["processed_dir"]
    )

    # Load IDs of documents that were already processed.
    processed_ids_path = (
        processed_dir / "already_processed.json"
    )
    processed_ids = load_processed_ids(
        processed_ids_path
    )

    # Create expensive resources once.
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["embedding"]["model"]
    )

    splitter = build_splitter(
        cfg,
        tokenizer,
    )

    from docling.document_converter import DocumentConverter

    converter = DocumentConverter()

    pdf_paths = sorted(
        raw_dir.rglob("*.pdf")
    )

    if not pdf_paths:
        logger.warning(
            "No PDFs found under %s",
            raw_dir,
        )
        return

    for pdf_path in pdf_paths:

        # SHA-256 of the PDF content.
        document_id = get_document_id(
            pdf_path
        )
        # Skip documents that were already ingested.
        if document_id in processed_ids:
            logger.info(
                "Skipping already processed: %s",
                pdf_path.name,
            )
            continue

        try:
            logger.info(
                "Processing document %s",
                pdf_path.name,
            )

            metadata = resolve_metadata(
                pdf_path,
                raw_dir,
            )

            chunks = ingest_pdf(
                pdf_path=pdf_path,
                metadata=metadata,
                document_id=document_id,
                converter=converter,
                splitter=splitter,
            )

            if not chunks:
                logger.warning(
                    "No chunks generated for %s",
                    pdf_path.name,
                )
                continue

            output_path = save_chunks(
                chunks=chunks,
                pdf_path=pdf_path,
                raw_dir=raw_dir,
                processed_dir=processed_dir,
            )

            # Only mark the document as processed
            # after successful chunk generation + saving.
            processed_ids.add(document_id)
            save_processed_ids(
                processed_ids_path,
                processed_ids,
            )
            logger.info(
                "Wrote %d chunks -> %s",
                len(chunks),
                output_path,
            )

        except Exception:
            logger.exception(
                "Failed to process %s",
                pdf_path,
            )
if __name__ == "__main__":
    main()