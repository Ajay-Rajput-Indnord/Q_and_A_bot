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
- `hybrid`: 20 dense candidates and two lexical BM25 views (full query plus content-focused query) combined with Reciprocal Rank Fusion. Fusion keeps a larger reranking pool so exact technical terms can promote candidates that narrowly miss the initial cutoff. The final context uses eight fused chunks plus nearby same-document chunks.

BM25 preserves technical compounds such as `text-embedding-3-small`, `mo.ui.text`, and version numbers. Hybrid results receive deterministic exact-term boosting and singular/plural normalization. Answer generation uses a second grounded review pass to catch incomplete answers and validate citations.

For broad list, options, steps, rules, and multi-item questions, the application adaptively retrieves up to 16 context chunks to improve multi-passage coverage. Exact-term reranking ignores conversational filler words when measuring lexical coverage, while BM25 itself retains its full tokenization.

## Evaluation

Development evaluation uses document indices 0–9. The held-out test split uses indices 10–19 and should only be run after the configuration is frozen.

```powershell
python eval/evaluate.py --method dense --split dev
python eval/evaluate.py --method hybrid --split dev
python eval/compare.py --split dev
python eval/evaluate.py --method hybrid --split test
```

Evaluation JSON files are written to `eval/results/`. The final report should include the generated metrics, comparison, and representative failure analysis.

Token F1 is retained as a strict diagnostic; manually review semantically correct answers that receive a low token score before drawing conclusions.

Answer accuracy uses required-answer token recall with a 50% coverage threshold, so an answer that includes the expected facts plus additional grounded detail is not rejected solely for being longer. Token F1 remains available as a strict diagnostic.

## Example questions by document

Select the document number in the sidebar before asking. These questions come from the answerable evaluation set and are good smoke tests for the indexed corpus:

- **Document 0:** What do keybullet kin drop? · Which enemy types wield an AK-47?
- **Document 1:** What do the giants look like? · What monsters are encountered in this journey?
- **Document 2:** What were the requirements for the project? · What framework was chosen to execute the RAG process and what alternatives were considered?
- **Document 3:** How do the data storage options compare? · What kind of model is the bling-phi-3 model?
- **Document 4:** How do I make a button? · What are the parameters for the mo.ui.text function?
- **Document 5:** What are the key topics of this article? · What are the four steps to become more impact-focused?
- **Document 6:** What kinds of models would need to be officially registered? · What are the risk classifications for AI?
- **Document 7:** Which quest does the emperor give the player? · What are the emperor's aliases?
- **Document 8:** Which part of the trip did I like the most? · What forms of exercise did we do?
- **Document 9:** What is the meaning behind the infrared scenes? · Who is in the family that this film is about?
- **Document 10:** What is the policy on Tai Chi? · What are the proposed changes that affect healthcare?
- **Document 11:** How much faster is the Tesla A100 compared to the Tesla V100? · In what contexts is BERT mentioned?
- **Document 12:** How does function exporting differ between Gleam and Python? · What data structures does Gleam natively support?
- **Document 13:** How can I freeze a variable during training in MLX? · Where is broadcasting used?
- **Document 14:** How long has the narrator been sober for? · What references to alcohol are there?
- **Document 15:** What is the name of the space station? · Who are the characters in this script?
- **Document 16:** In which patch was pet adoption added? · What can moss be used for?
- **Document 17:** Where was Alan Wake 2 officially announced? · What things does Scratch do?
- **Document 18:** Who wrote 'Divine Rivals'? · Which books have dragons in them?
- **Document 19:** What categories does this paper split the RAG technique paradigm into? · What are the advantages and disadvantages of the BM25 algorithm?

Questions from `no_answer_questions.csv` are intentionally unsupported and should return `I don't know from this document.`.

## Repository map

- `architecture/PROPOSAL.md`: problem, scope, data, and success measures
- `architecture/DESIGN.md`: architecture and implementation decisions
- `src/ask_document/`: ingestion, retrieval, answer generation, and UI code
- `tests/`: unit tests
- `eval/`: evaluation runner, metrics, and comparison script
- `HANDOFF.md`: current implementation status and safeguards
- `REPORT.md`: evaluation report and known limitations
