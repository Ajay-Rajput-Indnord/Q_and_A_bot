"""Dense, BM25, and hybrid RRF retrieval."""

from __future__ import annotations

from typing import Any
from pathlib import Path

from .bm25 import search_bm25
from .bm25 import load_bm25_payload
from .embeddings import embed_texts
from .vector_store import search_dense
from .bm25 import content_tokens, tokenize


DEFAULT_CANDIDATE_K = 20
DEFAULT_CONTEXT_K = 8
DEFAULT_RRF_K = 60
DEFAULT_NEIGHBOR_K = 4


def retrieve_dense(
    question: str,
    document_index: int,
    *,
    top_k: int = DEFAULT_CANDIDATE_K,
) -> list[dict[str, Any]]:
    query_embedding = embed_texts([question])[0]
    return search_dense(query_embedding, document_index, top_k=top_k)


def reciprocal_rank_fusion(
    ranked_lists: list[list[dict[str, Any]]],
    *,
    rrf_k: int = DEFAULT_RRF_K,
    top_k: int = DEFAULT_CONTEXT_K,
) -> list[dict[str, Any]]:
    """Merge ranked results by chunk_id using reciprocal rank fusion."""
    scores: dict[str, float] = {}
    records: dict[str, dict[str, Any]] = {}
    for ranked_list in ranked_lists:
        for rank, result in enumerate(ranked_list):
            chunk_id = result["chunk_id"]
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (rrf_k + rank + 1)
            records.setdefault(chunk_id, dict(result))

    fused: list[dict[str, Any]] = []
    for chunk_id, score in sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k]:
        result = records[chunk_id]
        result["score"] = score
        result["retriever"] = "hybrid_rrf"
        fused.append(result)
    return fused


def retrieve_hybrid(
    question: str,
    document_index: int,
    *,
    candidate_k: int = DEFAULT_CANDIDATE_K,
    context_k: int = DEFAULT_CONTEXT_K,
    rrf_k: int = DEFAULT_RRF_K,
) -> list[dict[str, Any]]:
    # Keep a larger fusion pool than the final context. Exact-term reranking
    # must be able to promote a technically precise chunk that narrowly missed
    # the initial RRF cutoff.
    fusion_k = max(context_k * 3, candidate_k)
    query_embedding = embed_texts([question])[0]
    dense_results = search_dense(query_embedding, document_index, top_k=candidate_k)
    bm25_results = search_bm25(question, document_index, top_k=candidate_k)
    # A second lexical view removes conversational filler while retaining
    # entities, exact names, versions, and technical compounds. This recovers
    # chunks that are relevant to the subject but score poorly for the full
    # natural-language question.
    content_query = " ".join(content_tokens(question))
    focused_bm25_results = (
        search_bm25(content_query, document_index, top_k=candidate_k)
        if content_query and content_query != question
        else []
    )
    fused = reciprocal_rank_fusion(
        [dense_results, bm25_results, focused_bm25_results],
        rrf_k=rrf_k,
        top_k=fusion_k,
    )
    return boost_exact_matches(question, fused)[:context_k]


def boost_exact_matches(
    question: str,
    results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Prefer fused chunks containing exact technical terms or phrases."""
    query_tokens = set(content_tokens(question))
    query_text = " ".join(content_tokens(question))
    if not query_tokens:
        return results

    scored: list[tuple[float, dict[str, Any]]] = []
    for result in results:
        text_tokens = set(content_tokens(result.get("text", "")))
        coverage = len(query_tokens & text_tokens) / len(query_tokens)
        phrase_bonus = 0.008 if query_text in " ".join(content_tokens(result.get("text", ""))) else 0.0
        adjusted = float(result.get("score", 0.0)) + (0.01 * coverage) + phrase_bonus
        updated = dict(result)
        updated["score"] = adjusted
        updated["lexical_coverage"] = coverage
        scored.append((adjusted, updated))
    return [result for _, result in sorted(scored, key=lambda item: item[0], reverse=True)]


def add_neighbor_chunks(
    results: list[dict[str, Any]],
    document_index: int,
    *,
    max_neighbors: int = DEFAULT_NEIGHBOR_K,
    path: str | Path = "data/processed/bm25/bm25_by_document.pkl",
) -> list[dict[str, Any]]:
    """Add a small bounded set of nearby chunks from the selected document."""
    if not results or max_neighbors <= 0:
        return results

    payload = load_bm25_payload(path).get(int(document_index))
    if payload is None:
        return results

    by_sequence = {
        int(metadata["chunk_sequence"]): {
            "chunk_id": chunk_id,
            "text": text,
            "metadata": metadata,
            "score": 0.0,
            "retriever": "neighbor",
        }
        for chunk_id, text, metadata in zip(
            payload["chunk_ids"], payload["documents"], payload["metadata"]
        )
    }
    selected = {result["chunk_id"] for result in results}
    expanded = list(results)
    added = 0
    for result in results:
        sequence = result.get("metadata", {}).get("chunk_sequence")
        if sequence is None:
            continue
        for offset in range(1, max_neighbors + 1):
            for candidate_sequence in (int(sequence) - offset, int(sequence) + offset):
                candidate = by_sequence.get(candidate_sequence)
                if candidate and candidate["chunk_id"] not in selected:
                    selected.add(candidate["chunk_id"])
                    expanded.append(candidate)
                    added += 1
                    if added >= max_neighbors:
                        return expanded
    return expanded


def retrieve(
    question: str,
    document_index: int,
    *,
    method: str = "hybrid",
    top_k: int = DEFAULT_CONTEXT_K,
) -> list[dict[str, Any]]:
    """Public retrieval entry point for the Streamlit layer."""
    if not question.strip():
        raise ValueError("Question cannot be blank.")
    if document_index < 0:
        raise ValueError("document_index must be non-negative.")
    if method == "dense":
        results = retrieve_dense(question, document_index, top_k=top_k)
    elif method == "hybrid":
        results = retrieve_hybrid(question, document_index, context_k=top_k)
    else:
        raise ValueError("method must be 'dense' or 'hybrid'.")
    return add_neighbor_chunks(results, document_index)
