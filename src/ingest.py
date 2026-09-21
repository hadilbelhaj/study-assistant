"""Extract PDFs -> chunk -> attach metadata -> write JSONL.

Batch mode: walks data/raw/<course>/*.pdf. Everything is derived from
the path and filename — no manifest, no per-file config:

  - course:   parent folder name (data/raw/java/... -> "java")
  - doc_type: filename prefix (lecture/tp/td/examen -> ...)
  - lecture:  built from the filename, e.g.
              lecture7-interface-graphique-en-java.pdf
              -> "Chapitre 7 - Interface graphique en Java"
  - language: a single project-wide constant (LANGUAGE below) — not
              per-file, since every course note is French right now.

This only works because the naming convention is fixed:
    <prefix><N>[-part<M>]-<title-words-separated-by-hyphens>.pdf
A file that doesn't match it still gets ingested (fallback: filename
as title, doc_type="lecture") but prints a warning — there's no
manifest left to override it, so a genuine exception has to be fixed
by renaming the file to fit the convention.

Run:
    python -m src.ingest
"""
import hashlib
import json
import re
import uuid
from pathlib import Path
from transformers import AutoTokenizer
import yaml
from langchain_text_splitters import RecursiveCharacterTextSplitter

CONFIG_PATH = Path("config.yaml")
LANGUAGE = "fr"

# Filename prefix -> (doc_type, display label used in the auto-title)
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
def resolve_metadata(pdf_path: Path) -> dict:
    """Derive course/doc_type/lecture/language purely from the path,
    per the naming convention documented at the top of this file."""
    course = pdf_path.parent.name
    stem = pdf_path.stem.lower()
    match = FILENAME_PATTERN.match(stem)

    if match:
        prefix, number, part, rest = match.groups()
        doc_type, label = PREFIX_INFO.get(prefix, ("lecture", prefix.capitalize()))
        words = [PROPER_NOUNS.get(w, w) for w in rest.split("-")]
        title_body = " ".join(words)
        title_body = title_body[0].upper() + title_body[1:] if title_body else title_body
        part_suffix = f" (Partie {part})" if part else ""
        lecture = f"{label} {number}{part_suffix} - {title_body}"
    else:
        print(f"[resolve_metadata] WARNING: {pdf_path.name} doesn't match the naming convention "
              f"— using filename as title and doc_type='lecture'")
        doc_type = "lecture"
        lecture = pdf_path.stem

    return {"course": course, "doc_type": doc_type, "lecture": lecture, "language": LANGUAGE}


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
                 source_file: str, cfg: dict, tokenizer, language: str = "fr") -> list[dict]:
    chunk_cfg = cfg["chunking"]
    splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer,
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
                cfg: dict, tokenizer: AutoTokenizer, language: str = "fr") -> list[dict]:
    pages = extract_pages(pdf_path)
    return chunk_pages(pages, course, lecture, doc_type, pdf_path.name, cfg, tokenizer, language)


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
    cfg = yaml.safe_load(config_path.read_text())
    tokenizer = AutoTokenizer.from_pretrained(cfg["embedding"]["model"])
    raw_dir = Path(cfg["paths"]["raw_dir"])
    pdf_paths = sorted(raw_dir.glob("**/*.pdf"))
    if not pdf_paths:
        print(f"[main] no PDFs found under {raw_dir}")
        return

    for pdf_path in pdf_paths:
        meta = resolve_metadata(pdf_path)
        chunks = ingest_pdf(pdf_path, meta["course"], meta["lecture"], meta["doc_type"],
                             cfg, tokenizer, meta["language"])
        out_path = save_chunks(chunks, meta["course"], pdf_path, cfg["paths"]["processed_dir"])
        print(f"Wrote {len(chunks)} chunks -> {out_path} ({meta['course']}/{meta['lecture']})")


if __name__ == "__main__":
    main()