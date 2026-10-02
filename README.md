# RAG Study Assistant

A grounded **Retrieval-Augmented Generation (RAG)** study assistant for university course material.

The system ingests course PDFs, extracts and structures their content, retrieves relevant chunks using dense and lexical search, reranks candidates, and generates answers grounded in the available course material.

The knowledge base currently contains primarily **French** course material, with support for **English** documents. It is being expanded across multiple courses, semesters, and study years.

---

## Architecture

```text
Course PDFs
    │
    ▼
Docling extraction
    │
    ▼
DoclingDocument
    │
    ▼
HybridChunker + metadata validation
    │
    ├── Dense embeddings ──► FAISS
    │
    └── Sparse tokens ─────► BM25
              │
              ▼
        Reciprocal Rank Fusion
              │
              ▼
      Qwen3-Reranker-0.6B
              │
              ▼
        Top relevant chunks
              │
              ▼
       Qwen2.5-7B generation
        through Ollama
              │
              ▼
      Grounded answer + sources
```

The retrieval pipeline can also be evaluated at each stage independently:

```text
Dense
BM25
RRF
RRF + Reranker
```

---

## Current Status

The project has progressed beyond the initial MVP and currently includes:

* PDF extraction with **Docling**
* Structure-aware, token-aware chunking with **Docling HybridChunker**
* Pydantic-based chunk validation
* Automatic **French/English language detection**
* Metadata for course, study year, semester, lecture, document type, source file, page range, and section
* Incremental ingestion using **SHA-256 document IDs**
* Deterministic chunk IDs
* Dense retrieval with **Qwen3-Embedding-0.6B + FAISS**
* Sparse retrieval with **BM25**
* **Reciprocal Rank Fusion (RRF)**
* **Qwen3-Reranker-0.6B** cross-encoder reranking
* Centralized configuration through `config.yaml`
* Separation between retrieval logic and local storage
* Persistent loading of embedding and reranking models within the application process
* Local LLM generation through **Ollama + Qwen2.5-7B**
* Basic unit tests for retrieval logic
* A retrieval evaluation dataset with manually verified relevant chunk IDs

The current chunking pipeline was changed from page-based text splitting to **structure-aware HybridChunker** in order to preserve coherent topics and section context across document boundaries.

---

## Repository Structure

```text
rag-study-assistant/
├── config.yaml
├── data/
│   ├── raw/                 # Source PDFs (gitignored)
│   ├── processed/           # Validated chunks (gitignored)
│   └── vectorstore/         # Local FAISS/BM25 indexes (gitignored)
├── evaluation/
│   ├── questions.json       # Retrieval evaluation dataset
│   └── evaluate.py          # Evaluation and retrieval-stage comparison
├── src/
│   ├── config.py            # Central configuration loading
│   ├── schemas.py           # Pydantic data models
│   ├── ingest.py            # PDF ingestion and chunking
│   ├── language.py          # Language detection
│   ├── embed.py             # Embedding model
│   ├── sparse.py            # BM25 indexing/search
│   ├── storage.py           # Local FAISS/BM25 storage
│   ├── retrieve.py          # Retrieval and RRF
│   ├── reranker.py          # Cross-encoder reranking
│   └── generate.py          # LLM generation
├── tests/
│   └── test_retrieval.py    # Basic retrieval unit tests
└── notebooks/
```

Each module has a focused responsibility while keeping the architecture simple enough for the current project scale.

---

## Ingestion and Chunking

The current ingestion pipeline is:

```text
PDF
 ↓
DoclingDocument
 ↓
Language detection
 ↓
HybridChunker
 ↓
Metadata enrichment
 ↓
Pydantic validation
 ↓
JSONL
```

`HybridChunker` is used instead of the previous page-based `RecursiveCharacterTextSplitter`.

This allows chunk boundaries to follow the document's structural organization while still respecting the embedding model's token budget.

The current chunk metadata includes:

```json
{
  "id": "...",
  "document_id": "...",
  "text": "...",
  "course": "java",
  "study_year": "1dni",
  "semester": "S2",
  "lecture": "Chapitre 5 - Les exceptions",
  "doc_type": "lecture",
  "language": "fr",
  "source_file": "lecture5-les-exceptions.pdf",
  "page_start": 27,
  "page_end": 28,
  "section": "Exceptions > Gestion des erreurs",
  "chunk_index": 8
}
```

Language is detected from the extracted document content rather than being encoded in the directory structure.

Document identity is based on the SHA-256 hash of the PDF content. This allows unchanged documents to be skipped during subsequent ingestion while modified documents receive a new document ID.

---

## Configuration

Models and tunable parameters are centralized in `config.yaml`.

```yaml
embedding:
  model: "Qwen/Qwen3-Embedding-0.6B"

generation:
  provider: "ollama"
  model: "qwen2.5:7b"

chunking:
  chunk_size_tokens: 500

paths:
  raw_dir: "data/raw"
  processed_dir: "data/processed"
  vectorstore_dir: "data/vectorstore"

retrieval:
  top_k: 5
  candidate_k: 20
  rrf_k: 60
  min_similarity_threshold: 0.35

reranker:
  model: "Qwen/Qwen3-Reranker-0.6B"
```

This allows model and retrieval experiments without modifying the core pipeline.

---

## Retrieval

The retrieval system currently supports four measurable stages:

```text
1. Dense retrieval
       ↓
2. BM25 retrieval
       ↓
3. Reciprocal Rank Fusion (RRF)
       ↓
4. Qwen3-Reranker-0.6B
```

Dense and sparse retrieval each produce candidate results. RRF combines their rankings, after which the cross-encoder reranks the fused candidate set.

The production API remains simple:

```python
results = retrieve(query)
```

while the evaluation pipeline can inspect each retrieval stage independently.

---

## Evaluation

A manually curated retrieval evaluation dataset is used to measure retrieval quality objectively.

The current dataset contains **29 questions** covering multiple course topics. Each question contains one or more manually verified relevant chunk IDs.

Example:

```json
{
  "id": "q001",
  "question": "Quelle est la différence entre une transmission synchrone et asynchrone ?",
  "course": "communication-numerique",
  "study_year": "1dni",
  "semester": "S1",
  "relevant_chunk_ids": [
    "..."
  ]
}
```

The evaluation compares:

```text
Dense
BM25
RRF
RRF + Reranker
```

using:

* Hit@K
* Recall@K
* MRR

#Current retrieval benchmark (29 questions)
- Qdrant hybrid: Hit@1 65.5%, MRR 69.0%
- Qdrant hybrid + reranker: Hit@1 72.4%, MRR 72.4%
- Previous FAISS + BM25 + RRF + reranker: Hit@1 72.4%, MRR 72.4%

The Qdrant migration preserves final retrieval performance on the current evaluation set.

## Testing

Basic retrieval unit tests are implemented with `pytest`.

Current tests cover:

* Reciprocal Rank Fusion ranking behavior
* Preservation of documents during RRF
* The retrieval pipeline's use of candidate retrieval and reranking

Run the tests with:

```bash
pytest tests/test_retrieval.py -v
```

Current status:

```text
3 passed
```

The evaluation dataset is kept separate from the unit tests because it measures retrieval quality on the actual course corpus, while the unit tests validate individual retrieval components and their interactions.

---

## Design Principles

* **Grounded generation:** answers should rely on retrieved course material.
* **Structure-aware chunking:** preserve meaningful document sections instead of relying only on page boundaries.
* **Hybrid retrieval:** combine semantic and lexical signals where useful.
* **Reranking:** use a stronger cross-encoder to improve candidate ordering.
* **Reproducibility:** use deterministic document and chunk identities where appropriate.
* **Configuration over hardcoding:** models and tunable parameters live in `config.yaml`.
* **Incremental complexity:** introduce infrastructure only when the project requires it.
* **Separation of concerns:** ingestion, storage, retrieval, reranking, and generation remain independent components.
* **Evaluation-driven development:** retrieval changes should be validated using a fixed ground-truth dataset rather than manual inspection alone.

Course PDFs and generated indexes are kept out of version control because they are local/derived data and may contain copyrighted teaching material.
