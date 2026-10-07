# Q_and_A_bot

Q_and_A_bot is a Streamlit question-and-answer bot for the Week 7 RAG project. The user selects one of the supplied documents, asks a question, and receives an answer grounded only in that selected document. The application displays the retrieved passages used for the answer and refuses unsupported questions with the exact text `I don't know from this document.`

## Requirements

- Python 3.10 or newer
- An OpenAI API key for embeddings and answer generation

Install dependencies from the repository root:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and set `OPENAI_API_KEY`. The default models are `text-embedding-3-small` for embeddings and `gpt-4o-mini` for answers; both can be changed in `.env`.

## Build the indexes

Run ingestion once after installing dependencies and configuring the API key:

```powershell
python -m src.ask_document.ingestion --documents data/raw/documents.csv
```

This creates the persistent ChromaDB dense index under `data/processed/dense/` and the per-document BM25 index under `data/processed/bm25/`.

## Run the application

```powershell
streamlit run app.py
```

Choose a document, select either dense or hybrid retrieval, enter a question, and press **Ask**. Retrieval is always filtered by the selected `document_index`; the system does not search across documents for one answer.

## Retrieval methods

- `dense`: ChromaDB embedding search. This is the baseline and handles semantic paraphrases.
- `hybrid`: dense search combined with BM25 keyword search through Reciprocal Rank Fusion. This is the improved method and helps with exact names, versions, and technical terms.

## Evaluation

Development evaluation uses document indices 0–9. The held-out test split uses indices 10–19 and should only be run after the configuration is frozen.

```powershell
python eval/evaluate.py --method dense --split dev
python eval/evaluate.py --method hybrid --split dev
python eval/compare.py --split dev
python eval/evaluate.py --method hybrid --split test
```

Evaluation JSON files are written to `eval/results/`. The final report should include the generated metrics, comparison, and representative failure analysis.

## Repository map

- `architecture/PROPOSAL.md`: problem, scope, data, and success measures
- `architecture/DESIGN.md`: architecture and implementation decisions
- `src/ask_document/`: ingestion, retrieval, answer generation, and UI code
- `tests/`: unit tests
- `eval/`: evaluation runner, metrics, and comparison script
- `HANDOFF.md`: current implementation status and safeguards
- `REPORT.md`: evaluation report and known limitations
