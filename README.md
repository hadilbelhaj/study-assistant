# RAG Study Assistant

A grounded **Retrieval-Augmented Generation (RAG)** study assistant for university course material.

The system ingests course PDFs, retrieves relevant content, reranks the candidates, and uses an LLM to generate answers grounded in the available material.

Current documents are primarily **French**, with support for **English** content. The knowledge base is being expanded to support multiple courses, semesters, and study years.

---

## Architecture

```text
Course PDFs
    │
    ▼
Docling extraction
    │
    ▼
Chunking + metadata validation
    │
    ▼
Embeddings + BM25
    │
    ├── Dense retrieval (FAISS)
    └── Sparse retrieval (BM25)
              │
              ▼
        RRF fusion
              │
              ▼
   Qwen3-Reranker-0.6B
              │
              ▼
        Top relevant chunks
              │
              ▼
        LLM generation
              │
              ▼
     Grounded answer + sources
```

---

## Current Status

The project has progressed beyond the initial MVP and currently includes:

* PDF extraction with **Docling**
* Token-aware chunking with `RecursiveCharacterTextSplitter`
* Pydantic-based chunk validation
* Automatic **French/English language detection**
* Metadata for course, study year, semester, lecture, document type, source, and page
* Incremental ingestion using **SHA-256 document IDs**
* Dense retrieval with **Qwen3-Embedding-0.6B + FAISS**
* Sparse retrieval with **BM25**
* **Reciprocal Rank Fusion (RRF)**
* **Qwen3-Reranker-0.6B** cross-encoder reranking
* Centralized configuration through `config.yaml`
* Separation between retrieval logic and the current local storage implementation
* Persistent loading of embedding and reranking models within the application process
* Local LLM generation through **Ollama + Qwen2.5-7B**

The reranker has been tested on the course corpus and improves the relevance of the retrieved context compared with the pre-reranking results.

---

## Repository Structure

```text
rag-study-assistant/
├── config.yaml
├── data/
│   ├── raw/              # Source PDFs (gitignored)
│   └── processed/        # Validated chunks (gitignored)
├── evaluation/           # Retrieval evaluation dataset
├── src/
│   ├── config.py
│   ├── schemas.py
│   ├── ingest.py
│   ├── language.py
│   ├── embed.py
│   ├── sparse.py
│   ├── storage.py
│   ├── retrieve.py
│   ├── reranker.py
│   └── generate.py
├── tests/
└── notebooks/
```

Each module has a focused responsibility, while avoiding unnecessary abstraction for the current project scale.

---

## Chunk Schema

Chunks contain the information required for retrieval, filtering, and source tracing:

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
  "page": 27,
  "chunk_index": 8
}
```

Language is detected from the extracted document content rather than being encoded in the directory structure.

---

## Configuration

Models and tunable parameters are centralized in `config.yaml`.

```yaml
embedding:
  model: "Qwen/Qwen3-Embedding-0.6B"

generation:
  provider: "ollama"
  model: "qwen2.5:7b"

retrieval:
  top_k: 5
  candidate_k: 20
  rrf_k: 60

reranker:
  model: "Qwen/Qwen3-Reranker-0.6B"
```

This allows model and retrieval experiments without modifying the core pipeline.

---

## Evaluation

A retrieval evaluation dataset is being introduced to measure improvements objectively rather than relying only on manual inspection.

Planned comparisons:

```text
Dense
Dense + BM25
Dense + BM25 + RRF
Dense + BM25 + RRF + Reranking
```

Initial metrics:

* Recall@K
* MRR

---
## Design Principles

* **Grounded generation:** answers should rely on retrieved course material.
* **Hybrid retrieval:** combine semantic and lexical search.
* **Reranking:** use a stronger cross-encoder before generation.
* **Reproducibility:** deterministic document/chunk identities where appropriate.
* **Configuration over hardcoding:** models and tunable parameters live in `config.yaml`.
* **Incremental complexity:** introduce infrastructure only when the project requires it.

Course PDFs and generated indexes are kept out of version control because they are derived/local data and may contain copyrighted teaching material.
