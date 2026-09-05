"""Extract PDFs -> chunk -> attach metadata -> write JSONL.

Edit the constants below, then run for a smoke test:
    python -m src.ingest
"""
import hashlib
import json
import uuid
from pathlib import Path

import yaml
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- Run parameters (edit before running) ---
PDF_PATH = Path("data/raw/reseaux/ch3.pdf")
COURSE = "Reseaux"
LECTURE = ""
DOC_TYPE = "lecture"
LANGUAGE = "fr"
CONFIG_PATH = Path("config.yaml")


def load_config(config_path: Path = CONFIG_PATH) -> dict:
    """Loaded on demand (inside main()) instead of at import time, so
    importing this module never touches the filesystem."""
    return yaml.safe_load(config_path.read_text())


def extract_pages(pdf_path: Path) -> list[dict]:
    """Docling handles digital + scanned pages (built-in OCR) and tables
    in one pass. Returns one dict per page: {page, text}.
    Formula-heavy pages still need a manual spot-check early on — see
    the roadmap's PDF extraction watch-outs.

    Imported lazily so chunk_pages() (and its tests) don't need
    docling — a much heavier install — just to run.
    """
    from docling.document_converter import DocumentConverter

    try:
        result = DocumentConverter().convert(str(pdf_path))
    except Exception as e:
        print(f"[extract_pages] failed to convert {pdf_path.name}: {e}")
        return []

    pages = []
    for page_no in result.document.pages:  # dict keyed by page number, 1-based
        text = result.document.export_to_markdown(page_no=page_no)
        pages.append({"page": page_no, "text": text})
    return pages


def chunk_pages(pages: list[dict], course: str, lecture: str, doc_type: str,
                 source_file: str, cfg: dict, language: str = "fr") -> list[dict]:
    chunk_cfg = cfg["chunking"]
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_cfg["chunk_size_tokens"],
        chunk_overlap=int(chunk_cfg["chunk_size_tokens"] * chunk_cfg["chunk_overlap_pct"]),
        separators=chunk_cfg["separators"],
    )
    chunks = []
    for page in pages:
        for piece in splitter.split_text(page["text"]):
            chunks.append({
                "id": str(uuid.uuid4()),
                "text": piece,
                "course": course,
                "lecture": lecture,
                "doc_type": doc_type,
                "language": language,
                "source_file": source_file,
                "page": page["page"],
            })
    return chunks


def ingest_pdf(pdf_path: Path, course: str, lecture: str, doc_type: str,
                cfg: dict, language: str = "fr") -> list[dict]:
    pages = extract_pages(pdf_path)
    return chunk_pages(pages, course, lecture, doc_type, pdf_path.name, cfg, language)


def save_chunks(chunks: list[dict], course: str, pdf_path: Path, processed_dir: Path) -> Path:
    """File name is the md5 of the source PDF's name, so re-running on
    the same PDF overwrites the same file instead of piling up random
    duplicates."""
    out_dir = Path(processed_dir) / course
    out_dir.mkdir(parents=True, exist_ok=True)
    file_id = hashlib.md5(pdf_path.name.encode()).hexdigest()
    out_path = out_dir / f"{file_id}.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    return out_path


def main():
    cfg = load_config()
    chunks = ingest_pdf(PDF_PATH, COURSE, LECTURE, DOC_TYPE, cfg, LANGUAGE)
    out_path = save_chunks(chunks, COURSE, PDF_PATH, cfg["paths"]["processed_dir"])
    print(f"Wrote {len(chunks)} chunks -> {out_path}")


if __name__ == "__main__":
    main()