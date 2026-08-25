# RAG Study Assistant

A grounded RAG chatbot for revising course material: French-heavy PDFs
in, cited answers out, refusing to guess when the material doesn't
cover the question. This repo currently implements **Phase 1 (MVP)**
of the full roadmap — one course, local, single script — with a
folder layout that won't need restructuring as later phases land.

## Why this structure

- `src/` holds one module per pipeline stage (`ingest → embed → index
  → retrieve → generate`), tied together by `rag.py`. No repository/
  service/interface layers — each file does one job directly, so the
  diff for "swap FAISS for Qdrant" or "swap Anthropic for Ollama"
  stays inside one file.
- `config.yaml` centralizes the things you'll actually tune (model
  names, chunk size, thresholds) so tuning never means hunting through
  code.
- `data/raw|processed|vectorstore` are gitignored — course PDFs are
  almost certainly your professors' copyrighted material, and neither
  they nor the derived index belong in a public repo. Only the folder
  structure (`.gitkeep`) is tracked.
- `eval/` and the metadata fields in every chunk (`course`, `lecture`,
  `doc_type`, `language`, `page`) exist from day one even though
  Phases 2-4 are what actually use them — retrofitting metadata onto
  chunks you've already embedded is much more painful than including
  it from the first script.

```
rag-study-assistant/
├── config.yaml            # model names, chunk size, paths, thresholds
├── requirements.txt        # Phase 1 — pinned, installable now
├── requirements-later.txt  # Phase 2-5 — reference only, not installed
├── data/
│   ├── raw/<course>/       # your PDFs, gitignored
│   ├── processed/<course>/ # chunked JSONL, gitignored
│   └── vectorstore/        # FAISS index + metadata, gitignored
├── src/
│   ├── ingest.py           # extract (Docling) + chunk + tag metadata
│   ├── embed.py            # embedding model wrapper
│   ├── index.py            # FAISS build/save/load
│   ├── retrieve.py         # query -> top-k chunks (+ metadata filter)
│   ├── generate.py         # grounded prompt template + LLM call
│   └── rag.py              # rag_query() — what everything else calls
├── scripts/run_cli.py       # Phase 1 interface: plain CLI loop
├── notebooks/00_explore.ipynb
├── tests/                  # pytest — chunking logic tested without docling
└── eval/                   # Phase 4 question/answer sets, per course
```

## Library choices (checked against Aug 2026 benchmarks)

| Stage | Pick | Why |
|---|---|---|
| PDF extraction | **Docling** | Handles born-digital text, reading order, tables, *and* scanned pages (built-in OCR) in one call — replaces the old pdfplumber+PyMuPDF+separate-OCR combo. MIT licensed. Good current fit for slide decks with tables/diagrams; still spot-check formula-heavy chunks by hand. |
| Chunking | **`langchain-text-splitters`** | Just the splitter, not the full `langchain` framework — same `RecursiveCharacterTextSplitter` the roadmap specifies, without the extra dependency weight. |
| Embedding | **`Qwen/Qwen3-Embedding-0.6B`** (or `BAAI/bge-m3`) | Qwen3-Embedding currently tops the open-weight multilingual leaderboards at a size that runs comfortably on a laptop CPU, Apache-2.0. BGE-M3 is the alternative if you want native dense+sparse+multi-vector output in one model for Phase 2's hybrid search — worth a head-to-head on your own course corpus, similar to how you approached [[french-embeddings-benchmark]]. Set your pick in `config.yaml`. |
| Vector store | **FAISS** (`IndexFlatL2`) → Chroma (Phase 3) → Qdrant (Phase 5) | Matches the roadmap's own progression; still the current-consensus path for local-prototype-to-light-production scale in 2026. |
| Generation | **Anthropic API** | Swappable in `generate.py` alone; nothing else in the pipeline depends on the provider. |

**One thing the dry-run install caught:** `sentence-transformers` and
`docling` both pull in full GPU-enabled PyTorch by default, which
means several GB of CUDA packages even on a CPU-only laptop. If you
don't have a GPU, install the CPU wheel first:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

**License note for later:** if you ever add PyMuPDF as a faster
fallback for clean digital PDFs, know that it's AGPL-3.0 — fine for
local personal use, but AGPL's network-use clause applies once Phase
5's classmate-facing app is a running service. Docling alone avoids
this question entirely.

## Setup

```bash
git clone <your-repo-url>
cd rag-study-assistant

python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# CPU-only machine? Run this first (see note above):
pip install torch --index-url https://download.pytorch.org/whl/cpu

pip install -r requirements.txt

cp .env.example .env             # then paste your real ANTHROPIC_API_KEY in
```

## Usage

```bash
# 1. Ingest one course's PDFs (repeat per lecture)
python -m src.ingest data/raw/reseaux/ch3.pdf --course "Reseaux" --lecture "Chapitre 3" --doc-type lecture

# 2. Build the FAISS index from everything ingested so far
python -c "from src.index import build_index, load_chunks_from_jsonl; build_index(load_chunks_from_jsonl('data/processed'))"

# 3. Ask questions
python scripts/run_cli.py
```
Or open `notebooks/00_explore.ipynb` for the same loop interactively.

## Dev

```bash
pytest              # tests/test_ingest.py runs without needing docling installed
ruff check .         # lint
ruff format .        # format
```

## Roadmap

This README covers Phase 1 only. See the full plan for Phases 2-5
(hybrid search + reranking, multi-course scale, exam/key-point/
synthesis modes, and the classmate-facing deployment) — 
`requirements-later.txt` tracks the library choices for those phases
as they get checked against benchmarks, but nothing there installs
until you're actually on that phase.
