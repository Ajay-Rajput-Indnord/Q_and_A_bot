# Q_and_A_bot Proposal

## 1. Problem and user

The user selects one of 20 documents and asks a question about it. The bot must answer from that document, show the passages that support the answer, and explicitly say when the selected document does not contain enough information.

A plain large language model is not sufficient because the documents are niche, heterogeneous, and likely outside the model's reliable knowledge. The system also needs document-specific grounding, transparent citations, and controlled refusal behavior. Retrieval-augmented generation (RAG) is appropriate because it can locate relevant passages at question time and pass only those passages to the answer model.

## 2. Scope

### In scope

- Load all 20 records from `data/raw/documents.csv`.
- Preserve each document's `index` and `source_url` as metadata on every stored chunk.
- Let the user choose a document before asking a question.
- Retrieve relevant chunks only from the chosen document.
- Answer only from retrieved evidence and cite the chunks used.
- Refuse unsupported questions with a clear “I don't know from this document” response.
- Compare a dense-embedding baseline with one improved retrieval method using the provided development questions.
- Provide a Streamlit interface for document selection, question input, grounded answers, and citations.

### Out of scope

- Searching across multiple documents for a single answer.
- Answering from general model knowledge when the document is silent.
- Fine-tuning a language model.
- Production hosting, authentication, multi-user state, or document editing.
- The stretch MultiHop-RAG dataset, unless the required project is complete and time remains.

## 3. Data

The supplied dataset is stored under `data/raw/` and contains:

- `documents.csv` with 20 text documents and the fields `index`, `source_url`, and `text`.
- `single_passage_answer_questions.csv` with 40 single-passage answerable questions.
- `multi_passage_answer_questions.csv` with 40 multi-passage answerable questions.
- `no_answer_questions.csv` with 40 questions whose answers are absent from the selected document.
- Two questions of each type for every document index from 0 through 19.
- Development questions for document indices 0–9 and a held-out test set for indices 10–19. The development set will be used for chunking and retrieval decisions. The test set will remain untouched until final evaluation.

The documents are heterogeneous: they include game-wiki pages, software documentation, articles, policy or technical writing, a story, and a research paper. Document length ranges from approximately 2,750 to 211,782 characters, with a median of approximately 20,251 characters. The longest document is large enough that sending it directly to the model is impractical. The data also contains noisy web-style text, repeated headings, markup-like fragments, tables or lists, and source URLs that may be inaccessible at run time. Ingestion will therefore normalize whitespace without destroying paragraph boundaries, retain the original text for traceability, and log empty or unusually short chunks.

## 4. Approach

The system will be implemented in Python with a small, explicit pipeline rather than hiding the core behavior behind a large orchestration layer. Ingestion will clean the source text and split it into 512-token chunks with 64-token overlap. Chunks will be embedded with OpenAI's `text-embedding-3-small` model and stored in a local ChromaDB collection. A parallel BM25 index will support exact-term retrieval. The baseline will use dense cosine-similarity search; the improvement will combine dense and BM25 rankings with Reciprocal Rank Fusion (RRF). Each chunk will include `document_index`, `source_url`, `chunk_id`, and character or paragraph offsets. The answer step will use a configurable grounded language-model adapter, with the exact model recorded in the final report. A Streamlit application will provide the user interface, while ingestion and evaluation remain separate Python modules.

ChromaDB is the best fit for this project because it is local, persistent, easy to install, supports metadata filtering by `document_index`, and makes the baseline easy to inspect during a short assignment. A managed database would add unnecessary setup and network dependencies. FAISS was considered as an alternative, but it would require more application code for persistence and metadata filtering. Qdrant is a strong production alternative, but its service or container setup is more than this small assignment needs.

## 5. Planned improvement

The improvement will be hybrid retrieval: combine ChromaDB dense search with lexical BM25 search over the same chunks, then fuse the two ranked lists using reciprocal rank fusion. Dense search should handle paraphrased questions and multi-sentence concepts. BM25 should help with exact names, version numbers, uncommon entities, and terminology-heavy questions in the game, software, and technical documents. Both retrievers will apply the same `document_index` filter before ranking, so the improvement cannot leak evidence from another document.

The experiment will compare at least these configurations on the development set:

1. Dense ChromaDB search, the required baseline.
2. Dense plus BM25 retrieval merged with RRF, the proposed improvement.

The development script will report answer accuracy, no-answer refusal rate, and citation correctness for both configurations. The final test set will be run only after the configuration is frozen.

## 6. Success measures

The initial targets are deliberately measurable and apply to the held-out test set as well as the development comparison:

| Measure | Definition | Target |
|---|---|---:|
| Answer accuracy | For answerable questions, the generated answer is judged against the supplied reference answer using normalized exact-match or token F1, with manual review for semantically equivalent answers. | At least 85% overall after the improvement |
| Multi-passage accuracy | Accuracy on the 40 questions whose answers require combining multiple passages. | At least 75% |
| Refusal rate | Percentage of the 40 no-answer questions for which the bot clearly refuses instead of inventing an answer. | At least 95% |
| Citation correctness | Percentage of cited chunks that actually support the answer, checked by an evaluator using chunk text and metadata. | At least 90% |
| Document isolation | Percentage of responses whose cited chunks all belong to the selected document. | 100% |
| Improvement | Improved retrieval must beat the dense baseline on answer accuracy or citation correctness without reducing refusal rate below its target. | Positive and reported with before/after numbers |

The evaluation output will separate single-passage, multi-passage, and no-answer results so a strong overall score cannot hide a failure mode.

## 7. Risks and mitigations

1. **Chunk boundaries may split facts or make multi-passage answers difficult.** Start with paragraph-aware 512-token chunks and 64-token overlap. Compare nearby chunk sizes on documents 0–9 and retain the setting that improves development results without creating excessive context.

2. **Retrieval may return plausible but irrelevant text, especially for short or ambiguous questions.** Enforce the selected-document metadata filter, use a conservative top-k, inspect retrieval recall during evaluation, and add BM25 fusion for exact terminology. If the evidence score is too weak, the answer chain will refuse rather than pass weak context to the model.

3. **A local answer model may be unavailable or inconsistent across machines.** Keep the LLM behind a provider interface, document the Ollama model and setup command, log the model name and prompt version, and retain a deterministic evidence-only fallback that returns the best passages and a refusal when generation cannot be performed.

