# Changelog

All notable changes to Q_and_A_bot are documented here.

## 07-10-2026

**commit:** `docs: updated the doc files`

### Added

- Updated the docs files

## 07-10-2026

**commit:** `feat: Implemented tests`

### Added

- Added unit tests for text cleaning and chunking.
- Added ingestion validation tests.
- Added chunk metadata tests.
- Added RRF ranking tests.
- Added retrieval result-limit tests.
- Added prompt construction tests.
- Added invalid citation refusal tests.
- Added Streamlit refusal-text validation tests.
- Added `pytest` to `requirements.txt`.

## 07-10-2026

**commit:** `feat: Implemented evaluation layer`

### Added

- Implemented evaluation metrics in `eval/metrics.py`:
  - Token F1
  - Answer accuracy
  - Refusal rate
  - Citation validity
- Implemented evaluation runner in `eval/evaluate.py`.
- Added support for evaluating:
  - Dense retrieval
  - Hybrid retrieval
  - Development split
  - Test split
- Implemented result comparison in `eval/compare.py`.
- Added JSON result output under `eval/results/`.

## 07-10-2026

**commmit:** `feat: Implemented streamlit`

### Added

- Added configurable embedding model support through `OPENAI_EMBEDDING_MODEL`.
- Added `OPENAI_EMBEDDING_MODEL=text-embedding-3-small` to `.env.example`.
- Added `get_embedding_model()` for loading the configured embedding model.
- Updated document ingestion to use the embedding model from `.env`.
- Updated query embedding generation to use the same configured model.
- Updated `HANDOFF.md` with embedding model configuration details.

### Changed

- Replaced the hardcoded embedding model usage with environment-based configuration.
- Kept `text-embedding-3-small` as the default model when no environment variable is provided.

## 07-10-2026

**commit:** `feat: Implemented answer generation layer`

### Added

- Implemented grounded answer generation using the OpenAI chat API.
- Added configurable chat model support through `OPENAI_CHAT_MODEL`.
- Set the default chat model to `gpt-4o-mini`.
- Added an evidence-only system prompt.
- Added numbered evidence chunks for answer generation.
- Added citation requirements using the format `[Chunk N]`.
- Added citation extraction and validation.
- Added exact refusal handling:


## 07-10-2026

**commit:** `feat: Implemeted retrieval part`

### Added

- Implemented dense retrieval using ChromaDB.
- Added cosine-distance search with selected-document filtering.
- Implemented BM25 search for lexical retrieval.
- Added per-document BM25 index loading and querying.
- Implemented hybrid retrieval using:
  - Dense top-10 results
  - BM25 top-10 results
  - Reciprocal Rank Fusion with `k=60`
  - Final top-5 fused chunks
- Added a public `retrieve()` function supporting:
  - `dense`
  - `hybrid`

## 07-10-2026

**commit:** `feat: Implemented Ingestion part`

### Added

- Implemented the document ingestion pipeline.
- Added CSV loading and validation for `documents.csv`.
- Added text cleaning while preserving paragraph boundaries.
- Added token-aware chunking.
- Added OpenAI embedding generation using `text-embedding-3-small`.
- Added `.env` support for `OPENAI_API_KEY`.
- Added persistent ChromaDB storage for dense embeddings.
- Added per-document BM25 index generation.
- Added BM25 index persistence under `data/processed/bm25/`.
- Added dense vector storage under `data/processed/dense/`.

## 07-10-2026

**commit:** `Remove proposal and design files from root`

### Removed

- Removed additional copy of proposal and design files from root.

## 07-10-2026

**commit:** `chore: Initialize project structure`

### Added

- Created Initial project structure.
- Created Initial required files.

## 07-10-2026

**commit:** `docs : add design and proposal for project`

### Added

- Architecture documentation of this project