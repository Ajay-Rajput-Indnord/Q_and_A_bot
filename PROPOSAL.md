# PROPOSAL.md — Ask the Document: A Q&A Bot That Cites Its Sources

**Course:** Month 2 · LLMs, Prompt Engineering and RAG — Week 7 Project
**Author:** [Your Name]
**Date:** 2026-10-06

---

## 1. Problem and User

**Who asks the questions?**
A researcher, student, or knowledge worker who has a specific document in front of them and wants precise answers from *only that document*, with traceable evidence — not a generic LLM response that mixes knowledge from training data.

**Why is a plain LLM not enough?**
The 20 documents in this dataset are niche and recent: game wikis, software documentation, blog posts, and a research survey. A general-purpose LLM:
- Has likely never seen most of this content in training.
- Cannot cite the exact passage it is using.
- Will confidently hallucinate when the answer is absent.

A retrieval-augmented generation (RAG) system grounds every answer in retrieved passages and can explicitly refuse to answer when the document does not contain the information.

---

## 2. Scope

### The bot WILL:
- Accept a document index (0–19) and a natural-language question.
- Search only within that chosen document.
- Return an answer composed exclusively from retrieved passages.
- List every passage (chunk) it used as a citation.
- Refuse with a clear "I don't know" message when no relevant passage is found.

### The bot will NOT:
- Combine information across multiple documents.
- Use the LLM's parametric knowledge to supplement an answer.
- Handle images, tables, or non-text content.
- Perform web search or access external URLs.

---

## 3. Data

**Dataset:** Single-Topic RAG Evaluation Dataset (Kaggle, MIT licence)

### What I found when I opened it

| File | Rows | Notes |
|------|------|-------|
| `documents.csv` | 20 | Columns: `index`, `source_url`, `text` |
| `qa_single.csv` | 40 | Each Q answered from one passage; includes `document_index` and gold answer |
| `qa_multi.csv` | 40 | Each Q requires multiple passages from the same document |
| `qa_none.csv` | 40 | No answer exists in the document; refusal is the correct response |

### Document characteristics
- **Types:** game wikis (short, term-dense), software docs (structured), blog posts (narrative), one research survey (long, technical).
- **Lengths:** estimated 500–4,000 tokens per document after cleaning.
- **Vocabulary:** highly specific proper nouns (item names, API function names, author names) — this means keyword matching matters alongside semantic search.
- **Potential messiness:** URLs, markdown artefacts, and repeated headers may be present and will need stripping during ingestion.

### Why this dataset is hard
40 of 120 questions (33%) have *no answer* in their document. The bot must reliably refuse rather than guess, making refusal rate a first-class metric.

---

## 4. Approach

Documents are loaded from `documents.csv`, cleaned, and split into overlapping chunks (512 tokens, 64-token overlap) using LangChain's `RecursiveCharacterTextSplitter`. Each chunk is tagged with `doc_index`, `source_url`, and a `chunk_id`. Chunks are embedded with **OpenAI `text-embedding-3-large`** (3072 dims, best recall on niche technical text) and stored in **ChromaDB** (local, file-based, no Docker required). A parallel **BM25 index** (`rank_bm25`) is built per-document for keyword retrieval. At query time, both indices are searched (filtered to the chosen document) and results are merged with **Reciprocal Rank Fusion (RRF)**. The merged top-5 chunks are passed to **GPT-4o** with a strict system prompt that forbids using any knowledge outside the provided passages.

---

## 5. Planned Improvement

**Chosen improvement: Hybrid Search (Dense + BM25 with RRF)**

**Why this improvement on this data:**
The documents contain two very different text styles:
- *Game wikis and software docs* use precise, rare terminology (item names, function names). Dense embeddings alone may not rank exact keyword matches first. BM25 excels here.
- *Research surveys and blog posts* use paraphrased, conceptual language. BM25 alone misses semantic similarity. Dense embeddings excel here.

RRF (Reciprocal Rank Fusion) combines both ranked lists without requiring a hand-tuned weight:

```
RRF_Score(chunk) = Σ  1 / (60 + rank_i)
```

**Hypothesis:** Hybrid search will raise multi-passage answer accuracy by at least 8 percentage points over the dense-only baseline, because multi-passage questions often use the exact terminology of the document.

---

## 6. Success Measures

| Metric | How measured | Baseline target | Hybrid target |
|--------|-------------|-----------------|---------------|
| Answer accuracy | GPT-4o-mini LLM judge vs gold answer | ≥ 65% | ≥ 75% |
| Refusal rate | % of `qa_none` Qs answered with "I don't know" | ≥ 80% | ≥ 85% |
| Citation correctness | Gold passage text found in cited chunk (string overlap ≥ 0.5) | ≥ 70% | ≥ 80% |

All numbers reported on the **dev set only** (documents 0–9, 60 questions). The test set (documents 10–19) is used only for final grading.

---

## 7. Risks

| # | Risk | Likelihood | Impact | Mitigation |
|---|------|-----------|--------|-----------|
| 1 | GPT-4o ignores the system prompt and uses parametric knowledge | Medium | High | Set `temperature=0`, add explicit "ONLY use the passages below" instruction, test heavily on `qa_none.csv` |
| 2 | Chunk size splits a critical passage across two chunks, harming single-passage accuracy | Medium | Medium | Test chunk sizes 256 / 512 / 1024 on dev set; keep 64-token overlap to reduce boundary cuts |
| 3 | BM25 index is per-document but documents vary greatly in length (< 500 to > 4000 tokens) | Low | Medium | Normalise BM25 scores with `BM25Okapi` default parameters; fall back to dense-only for documents with fewer than 10 chunks |

---

## 8. Plan

| Day | Deliverable |
|-----|------------|
| **Day 1** | `PROPOSAL.md` (this document) — finalised and reviewed |
| **Day 2** | `DESIGN.md` — architecture diagram, component decisions, prompt design |
| **Day 3** | `src/ingest.py` — load CSV, clean, chunk, embed, store in ChromaDB |
| **Day 4** | `src/retrieval.py` + `src/chain.py` — baseline dense search + GPT-4o answer |
| **Day 5** | `eval/evaluate.py` — run 60 dev questions, save baseline results |
| **Day 6** | Add BM25 index builder + RRF hybrid retrieval; re-run evaluation |
| **Day 7** | `REPORT.md` — results table, 10 failure cases, before/after numbers |
| **Day 8** | Polish README, demo prep (3 live questions + 1 failure explanation) |
