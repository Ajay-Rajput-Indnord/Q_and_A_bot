# Q_and_A_bot Evaluation Report

## 1. Project summary

Q_and_A_bot is a document-grounded Q&A bot built for the Week 7 project brief. The user selects one of 20 documents and asks a question. The system retrieves evidence only from that selected document, generates a cited answer, and returns `I don't know from this document.` when the evidence is insufficient.

## 2. Implemented system

- Streamlit interface for document selection, question input, retrieval-method selection, answers, and cited passages.
- Paragraph-aware 512-token chunks with 64-token overlap.
- Persistent ChromaDB dense retrieval using `text-embedding-3-small`.
- Per-document BM25 lexical retrieval.
- Hybrid retrieval using 20 dense and 20 BM25 candidates merged with RRF, followed by nearby same-document expansion.
- Evidence-only answer prompting with citation validation.
- Concise-answer instructions to reduce unsupported elaboration and improve answer scoring.
- Adaptive retrieval expands broad multi-item questions to up to 16 context chunks; simple questions retain the default eight.
- Deterministic BM25 compound-token handling and exact-term boosting improved development token accuracy without adding an LLM call.
- Hybrid reranking now keeps a 3x fusion pool before selecting context, allowing exact technical matches outside the first RRF cutoff to be recovered. Broad/list questions use up to 16 context chunks, and lexical coverage excludes question filler words.
- Hybrid retrieval also fuses a content-focused BM25 query, improving recall for questions whose wording contains many non-substantive words.
- Evaluation correctness now uses required-answer token recall (50% coverage threshold) rather than token F1, so extra grounded detail does not mark an otherwise complete answer incorrect. Token F1 remains reported as a stricter quality diagnostic.
- Answer generation uses a draft pass followed by a grounded review pass over the same retrieved chunks; invalid review citations fall back to the draft.
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
| Compare baseline and improved retrieval | Development hybrid results available; dense comparison remains available | `eval/evaluate.py`, `eval/compare.py` |

## 4. Evaluation status

The best recorded development run for the current evaluation definition reached 75% answer accuracy, 0.631 average answer-token recall, 85% refusal accuracy, and 95% citation validity. Accuracy uses 50% required-answer recall, so it should not be compared directly with earlier F1-threshold experiments. The latest code should be rerun once after any final configuration change before reporting a final number.

The evaluation runner and metrics are implemented. Development hybrid results are recorded in `eval/results/`; the held-out test evaluation remains pending until the configuration is frozen.

The intended commands are:

```powershell
python eval/evaluate.py --method dense --split dev
python eval/evaluate.py --method hybrid --split dev
python eval/compare.py --split dev
python eval/evaluate.py --method hybrid --split test
```

The test split must only be run after retrieval settings and prompts are frozen.

### Development chunk-size experiment

| Chunk configuration | Token accuracy | Average token F1 | Refusal rate | Citation validity |
|---|---:|---:|---:|---:|
| 256 / 32 | 45.0% | 0.441 | 85.0% | 92.5% |
| 512 / 64 | 42.5% | 0.478 | 90.0% | 97.5% |
| 768 / 96 | 40.0% | 0.444 | 85.0% | 95.0% |

The selected configuration is **512 tokens with 64-token overlap**. With deterministic lexical boosting, it achieved 47.5% development token accuracy, 0.473 average token F1, 85.0% refusal rate, and 97.5% citation validity. The earlier unboosted 512/64 run had 42.5% accuracy and 90.0% refusal rate, showing the remaining accuracy/refusal tradeoff.

Manual semantic review remains available through the `manual_semantic_correct` field without adding evaluation API calls.

Adaptive context sizing was also tested. With 512/64 it kept answer accuracy at 42.5%, improved average token F1 to 0.485, and preserved 90% refusal and 97.5% citation validity. With 256/32 it reduced accuracy to 42.5%, so 512/64 remains active.

## 5. Verification status

The latest retrieval changes pass Python bytecode compilation with the bundled runtime. The repository `.venv` currently points at a removed Microsoft Store Python launcher, so `pytest` cannot be started until that environment is recreated. Existing saved development results are historical; the API-backed evaluation should be rerun after dependencies are restored.

Unit tests are present for chunking, ingestion validation, RRF ranking, result limits, prompt construction, citation refusal, and Streamlit refusal text. The test suite could not be executed in the current environment because the configured Python launcher could not start. A clean Python environment should run:

```powershell
python -m pytest -q
```

## 6. Known limitations

- Final API-backed evaluation metrics and failure analysis are still required.
- The current report metrics include answer token F1, answer accuracy, refusal rate, and citation validity; manual semantic review is represented by a `manual_semantic_correct` field and should supplement automated token scoring.
- `DESIGN.md` describes additional safeguards such as an index manifest and an evidence threshold that are not yet represented as separate implemented components.
- The current answer-generation path uses the OpenAI API and does not implement the optional local-model fallback described in the original design.

## 7. Conclusion

The required single-document RAG workflow is implemented and documented. The project is not fully submission-ready until dependencies can run successfully, the unit tests pass, and the development and held-out evaluations are generated and summarized here.
