# Ask the Document Handoff

## Current stage

Date: 2026-10-07

The project currently has a working ingestion and retrieval foundation for a Streamlit-based document Q&A bot. The source datasets are in `data/raw/`. The implementation is designed to answer questions only from one selected document at a time.

## Confirmed technical decisions

- User interface: Streamlit, started with `streamlit run app.py`.
- Vector database: local ChromaDB.
- Dense embedding model: OpenAI `text-embedding-3-small`.
- Dense similarity: cosine distance in ChromaDB.
- Chunk size: 512 tokens.
- Chunk overlap: 64 tokens.
- Lexical retrieval: BM25 per document.
- Hybrid retrieval: dense top-10 plus BM25 top-10, merged with Reciprocal Rank Fusion.
- RRF constant: 60.
- Final context: top 5 fused chunks.
- API key source: `OPENAI_API_KEY` in `.env`.

## Completed

### Data and structure

- Added the four supplied CSV files to `data/raw/`:
  - `documents.csv`
  - `single_passage_answer_questions.csv`
  - `multi_passage_answer_questions.csv`
  - `no_answer_questions.csv`
- Added `requirements.txt`, `.env.example`, and `.gitignore` configuration.
- Added project documentation and evaluation folders.

### Ingestion

Implemented in `src/ask_document/`:

- `chunking.py` cleans text and creates overlapping token chunks.
- `embeddings.py` loads the OpenAI key and creates embeddings.
- `vector_store.py` persists chunks and vectors in ChromaDB.
- `bm25.py` builds and saves the BM25 index.
- `ingestion.py` orchestrates loading, validation, chunking, embedding, ChromaDB storage, and BM25 indexing.

Run ingestion from the repository root:

```powershell
pip install -r requirements.txt
python -m src.ask_document.ingestion --documents data/raw/documents.csv
```

Generated data is stored under:

```text
data/processed/dense/
data/processed/bm25/bm25_by_document.pkl
```

### Retrieval

Implemented in:

- `vector_store.py` for filtered ChromaDB dense search.
- `bm25.py` for selected-document BM25 search.
- `retrieval.py` for dense retrieval, hybrid retrieval, and RRF.

Example:

```python
from src.ask_document.retrieval import retrieve

results = retrieve(
    question="What is the maximum stack size?",
    document_index=3,
    method="hybrid",
)
```

The retrieval result contains `chunk_id`, `text`, `metadata`, `score`, and `retriever` fields.

## Not completed yet

- `answer_chain.py` still needs the grounded answer-generation flow.
- `prompts.py` still needs the final system prompt and refusal rule.
- `streamlit_ui.py` and `app.py` still need the user interface.
- `eval/evaluate.py`, `eval/metrics.py`, and `eval/compare.py` still need implementation.
- Tests are currently placeholders.
- `README.md` still needs setup and run instructions.
- `REPORT.md` still needs final metrics, comparison results, and failure analysis.

## Recommended next steps

1. Implement the grounded prompt and answer chain.
2. Add citation validation and the exact refusal response:

   ```text
   I don't know from this document.
   ```

3. Implement `streamlit_ui.py` with document selection, question input, retrieval-method selection, answer display, and citation display.
4. Connect `app.py` to the Streamlit UI.
5. Implement evaluation for dense versus hybrid retrieval using development questions for document indices 0–9.
6. Add tests for chunking, document isolation, RRF ordering, refusal behavior, and citation validation.
7. Run the held-out test questions for document indices 10–19 only after the configuration is frozen.

## Important safeguards

- Never commit `.env` or expose `OPENAI_API_KEY`.
- Always filter ChromaDB and BM25 retrieval by the selected `document_index`.
- Do not use test questions for tuning chunk size, thresholds, prompts, or retrieval settings.
- Do not answer from general model knowledge when the retrieved evidence is insufficient.
- Do not run ingestion until dependencies are installed and `.env` contains a valid key.
