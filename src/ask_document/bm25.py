"""Build and persist per-document BM25 indexes."""

from __future__ import annotations

import pickle
import re
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi


def tokenize(text: str) -> list[str]:
    """Tokenize words while preserving technical compounds and their parts."""
    compounds = re.findall(r"[a-z0-9_]+(?:[-.][a-z0-9_]+)+", text.lower())
    simple = re.findall(r"[a-z0-9_]+", text.lower())
    tokens = list(simple)
    for compound in compounds:
        tokens.append(compound)
        tokens.extend(part for part in re.split(r"[-.]", compound) if part)
    return tokens


# These words are useful to BM25 (they help its document-frequency weighting),
# but they are noise when measuring whether a retrieved chunk contains the
# subject of a question.
CONTENT_STOPWORDS = {
    "a", "an", "and", "are", "be", "does", "for", "from", "how", "in",
    "is", "it", "of", "on", "or", "the", "to", "was", "were", "what",
    "when", "where", "which", "who", "why", "with", "do", "did", "can",
    "could", "would", "should", "i", "we", "you", "they", "their", "this",
    "that", "these", "those", "me", "my", "your", "our", "does", "have",
    "has", "had", "about", "all", "any", "also", "than", "then", "into",
}


def content_tokens(text: str) -> list[str]:
    """Return query/content tokens with conversational filler removed."""
    tokens: list[str] = []
    for token in tokenize(text):
        if token in CONTENT_STOPWORDS:
            continue
        # Lightweight normalization is used only for reranking, not for the
        # authoritative BM25 index. It handles common singular/plural pairs
        # such as giant/giants and version/versions.
        if token.endswith("ies") and len(token) > 4:
            token = token[:-3] + "y"
        elif token.endswith("s") and not token.endswith(("ss", "us", "is")) and len(token) > 3:
            token = token[:-1]
        tokens.append(token)
    return tokens


def build_and_save_bm25(
    chunks: list[dict[str, Any]],
    output_directory: str | Path = "data/processed/bm25",
) -> Path:
    """Save a compact BM25 payload grouped by document index."""
    grouped: dict[int, list[dict[str, Any]]] = {}
    for chunk in chunks:
        grouped.setdefault(int(chunk["document_index"]), []).append(chunk)

    payload: dict[int, dict[str, Any]] = {}
    for document_index, document_chunks in grouped.items():
        tokenized = [tokenize(chunk["text"]) for chunk in document_chunks]
        # Construct once here to validate the corpus; the serializable payload
        # stores tokenized text and chunk metadata for query-time loading.
        BM25Okapi(tokenized)
        payload[document_index] = {
            "chunk_ids": [chunk["chunk_id"] for chunk in document_chunks],
            "documents": [chunk["text"] for chunk in document_chunks],
            "tokenized": tokenized,
            "metadata": [
                {key: value for key, value in chunk.items() if key != "text"}
                for chunk in document_chunks
            ],
        }

    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)
    index_path = output_path / "bm25_by_document.pkl"
    with index_path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return index_path


def load_bm25_index(path: str | Path = "data/processed/bm25/bm25_by_document.pkl") -> dict[int, BM25Okapi]:
    payload = load_bm25_payload(path)
    return {
        int(document_index): BM25Okapi(value["tokenized"])
        for document_index, value in payload.items()
    }


def load_bm25_payload(
    path: str | Path = "data/processed/bm25/bm25_by_document.pkl",
) -> dict[int, dict[str, Any]]:
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def search_bm25(
    query: str,
    document_index: int,
    *,
    top_k: int = 10,
    path: str | Path = "data/processed/bm25/bm25_by_document.pkl",
) -> list[dict[str, Any]]:
    """Search the BM25 index for one selected document only."""
    if top_k <= 0:
        return []

    payload = load_bm25_payload(path)
    document = payload.get(int(document_index))
    if document is None:
        return []

    index = BM25Okapi(document["tokenized"])
    scores = index.get_scores(tokenize(query))
    ranked = sorted(range(len(scores)), key=lambda position: scores[position], reverse=True)
    results: list[dict[str, Any]] = []
    for position in ranked[:top_k]:
        results.append(
            {
                "chunk_id": document["chunk_ids"][position],
                "text": document["documents"][position],
                "metadata": document["metadata"][position],
                "score": float(scores[position]),
                "retriever": "bm25",
            }
        )
    return results
