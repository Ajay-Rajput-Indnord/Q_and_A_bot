# Changelog

All notable changes to Q_and_A_bot are documented here.

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