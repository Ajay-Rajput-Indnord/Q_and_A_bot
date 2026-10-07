# Ask the Document Handoff

## Current stage

Date: 2026-10-07

The project currently has a working ingestion, retrieval, and grounded answer-generation foundation for a Streamlit-based document Q&A bot. The source datasets are in `data/raw/`. The implementation is designed to answer questions only from one selected document at a time.

## Confirmed technical decisions

- User interface: Streamlit, started with `streamlit run app.py`.
- Vector database: local ChromaDB.
- Dense embedding model: OpenAI `text-embedding-3-small`.
- Embedding model configuration: `OPENAI_EMBEDDING_MODEL` in `.env`.
- Dense similarity: cosine distance in ChromaDB.
- Chunk size: 512 tokens.
- Chunk overlap: 64 tokens.
- Lexical retrieval: BM25 per document.
- Hybrid retrieval: dense top-10 plus BM25 top-10, merged with Reciprocal Rank Fusion.
- RRF constant: 60.
- Final context: top 5 fused chunks.
- API key source: `OPENAI_API_KEY` in `.env`.
- Chat model: configurable through `OPENAI_CHAT_MODEL`, defaulting to `gpt-4o-mini`.

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

### Answer generation

Implemented in:

- `prompts.py` for the evidence-only system prompt and user prompt formatting.
- `answer_chain.py` for retrieval, OpenAI chat completion, refusal handling, and citation validation.

The answer chain:

1. Retrieves evidence using dense or hybrid retrieval.
2. Sends only the retrieved chunks to the chat model.
3. Requires citations such as `[Chunk 1]`.
4. Rejects empty or invalidly cited responses.
5. Returns the exact refusal when evidence is missing or citations are invalid:

```text
I don't know from this document.
```

Example:

```python
from src.ask_document.answer_chain import answer_question

result = answer_question(
    question="What is the maximum stack size?",
    document_index=3,
    method="hybrid",
)
```

### Streamlit interface

Implemented in:

- `app.py` as the Streamlit entry point.
- `streamlit_ui.py` for document selection, retrieval-method selection, question input, answer display, citations, and error handling.

Run the application from the repository root:

```powershell
streamlit run app.py
```

The interface supports both dense ChromaDB retrieval and hybrid dense-plus-BM25 retrieval. It displays the cited chunks returned by the answer chain and shows the exact refusal when the evidence is insufficient.

### Evaluation

Implemented in:

- `eval/metrics.py` for token F1, answer accuracy, refusal rate, and citation validity.
- `eval/evaluate.py` for running dense or hybrid evaluation on development or test questions.
- `eval/compare.py` for comparing dense and hybrid result summaries.

Run development evaluation from the repository root:

```powershell
python eval/evaluate.py --method dense --split dev
python eval/evaluate.py --method hybrid --split dev
python eval/compare.py --split dev
```

Results are saved under `eval/results/`. Evaluation makes OpenAI API calls through the answer chain, so ingestion must be complete and `.env` must contain valid credentials first.

## Not completed yet

- Tests are currently placeholders.
- `README.md` still needs setup and run instructions.
- `REPORT.md` still needs final metrics, comparison results, and failure analysis.

## Recommended next steps

1. Add tests for chunking, document isolation, RRF ordering, refusal behavior, citation validation, and the Streamlit interface.
2. Add README setup and run instructions.
3. Run development evaluation and inspect dense versus hybrid results.
4. Run the held-out test questions for document indices 10–19 only after the configuration is frozen.

## Important safeguards

- Never commit `.env` or expose `OPENAI_API_KEY`.
- Always filter ChromaDB and BM25 retrieval by the selected `document_index`.
- Do not use test questions for tuning chunk size, thresholds, prompts, or retrieval settings.
- Do not answer from general model knowledge when the retrieved evidence is insufficient.
- Do not run ingestion until dependencies are installed and `.env` contains a valid key.
