"""Dense, BM25, and hybrid RRF retrieval."""

from __future__ import annotations

from typing import Any

from .bm25 import search_bm25
from .embeddings import embed_texts
from .vector_store import search_dense


DEFAULT_CANDIDATE_K = 10
DEFAULT_CONTEXT_K = 5
DEFAULT_RRF_K = 60


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
    query_embedding = embed_texts([question])[0]
    dense_results = search_dense(query_embedding, document_index, top_k=candidate_k)
    bm25_results = search_bm25(question, document_index, top_k=candidate_k)
    return reciprocal_rank_fusion(
        [dense_results, bm25_results],
        rrf_k=rrf_k,
        top_k=context_k,
    )


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
        return retrieve_dense(question, document_index, top_k=top_k)
    if method == "hybrid":
        return retrieve_hybrid(question, document_index, context_k=top_k)
    raise ValueError("method must be 'dense' or 'hybrid'.")
