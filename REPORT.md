# Q_and_A_bot Evaluation Report

## 1. Project summary

Q_and_A_bot is a document-grounded Q&A bot built for the Week 7 project brief. The user selects one of 20 documents and asks a question. The system retrieves evidence only from that selected document, generates a cited answer, and returns `I don't know from this document.` when the evidence is insufficient.

## 2. Implemented system

- Streamlit interface for document selection, question input, retrieval-method selection, answers, and cited passages.
- Paragraph-aware 512-token chunks with 64-token overlap.
- Persistent ChromaDB dense retrieval using `text-embedding-3-small`.
- Per-document BM25 lexical retrieval.
- Hybrid retrieval using dense top-10 and BM25 top-10 results merged with RRF using `k=60`, then reduced to the top five evidence chunks.
- Evidence-only answer prompting with citation validation.
- Exact refusal text for empty retrieval, empty responses, refusal responses, or invalid citations.
- Development/test evaluation split by document index: 0–9 for development and 10–19 for held-out testing.

## 3. Requirement coverage

| Brief requirement | Status | Evidence |
|---|---|---|
| User picks a document | Implemented | `src/ask_document/streamlit_ui.py` |
| Answer only from that document | Implemented | `src/ask_document/retrieval.py`, answer prompt, document filtering |
| Show passages used | Implemented | Streamlit source display and answer result chunks |
| Say when the answer is not present | Implemented | Exact refusal handling in `answer_chain.py` |
| Proposal and design before build | Documented | `architecture/PROPOSAL.md`, `architecture/DESIGN.md` |
| Compare baseline and improved retrieval | Code ready; results pending | `eval/evaluate.py`, `eval/compare.py` |

## 4. Evaluation status

The evaluation runner and metrics are implemented, but no evaluation JSON files are currently present in `eval/results/`. Therefore, final accuracy, refusal rate, citation validity, and dense-versus-hybrid improvement cannot be reported honestly yet.

The intended commands are:

```powershell
python eval/evaluate.py --method dense --split dev
python eval/evaluate.py --method hybrid --split dev
python eval/compare.py --split dev
python eval/evaluate.py --method hybrid --split test
```

The test split must only be run after retrieval settings and prompts are frozen.

## 5. Verification status

Unit tests are present for chunking, ingestion validation, RRF ranking, result limits, prompt construction, citation refusal, and Streamlit refusal text. The test suite could not be executed in the current environment because the configured Python launcher could not start. A clean Python environment should run:

```powershell
python -m pytest -q
```

## 6. Known limitations

- Final API-backed evaluation metrics and failure analysis are still required.
- The current report metrics include answer token F1, answer accuracy, refusal rate, and citation validity; manual citation entailment review should supplement these automated measures.
- `DESIGN.md` describes additional safeguards such as an index manifest and an evidence threshold that are not yet represented as separate implemented components.
- The current answer-generation path uses the OpenAI API and does not implement the optional local-model fallback described in the original design.

## 7. Conclusion

The required single-document RAG workflow is implemented and documented. The project is not fully submission-ready until dependencies can run successfully, the unit tests pass, and the development and held-out evaluations are generated and summarized here.
