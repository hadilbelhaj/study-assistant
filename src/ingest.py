"""Extract PDFs -> chunk -> attach metadata -> write JSONL.

Run directly for a first smoke test:
    python -m src.ingest --course "Reseaux" --doc-type lecture data/raw/reseaux/ch3.pdf
"""
import argparse
import json
import uuid
from pathlib import Path

import yaml
from langchain_text_splitters import RecursiveCharacterTextSplitter

CONFIG = yaml.safe_load(Path("config.yaml").read_text())


def extract_pages(pdf_path: Path) -> list[dict]:
    """Docling handles digital + scanned pages (built-in OCR) and tables
    in one pass. Returns one dict per page: {page, text}.
    Formula-heavy pages still need a manual spot-check early on — see
    the roadmap's PDF extraction watch-outs.

    Imported lazily so chunk_pages() (and its tests) don't need
    docling — a much heavier install — just to run.
    """
    from docling.document_converter import DocumentConverter

    converter = DocumentConverter()
    result = converter.convert(str(pdf_path))
    pages = []
    for page in result.document.pages:
        text = result.document.export_to_markdown(page_no=page.page_no)
        pages.append({"page": page.page_no, "text": text})
    return pages


def chunk_pages(pages: list[dict], course: str, lecture: str, doc_type: str,
                 source_file: str, language: str = "fr") -> list[dict]:
    cfg = CONFIG["chunking"]
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=cfg["chunk_size_tokens"],
        chunk_overlap=int(cfg["chunk_size_tokens"] * cfg["chunk_overlap_pct"]),
        separators=cfg["separators"],
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
                language: str = "fr") -> list[dict]:
    pages = extract_pages(pdf_path)
    return chunk_pages(pages, course, lecture, doc_type, pdf_path.name, language)


def save_chunks(chunks: list[dict], course: str) -> Path:
    out_dir = Path(CONFIG["paths"]["processed_dir"]) / course
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{uuid.uuid4().hex[:8]}.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--course", required=True)
    parser.add_argument("--lecture", default="")
    parser.add_argument("--doc-type", default="lecture")
    parser.add_argument("--language", default="fr")
    args = parser.parse_args()

    chunks = ingest_pdf(args.pdf, args.course, args.lecture, args.doc_type, args.language)
    out_path = save_chunks(chunks, args.course)
    print(f"Wrote {len(chunks)} chunks -> {out_path}")
