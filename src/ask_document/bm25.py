"""Build and persist per-document BM25 indexes."""

from __future__ import annotations

import pickle
import re
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi


def tokenize(text: str) -> list[str]:
    return re.findall(r"[\w]+", text.lower())


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
    with Path(path).open("rb") as handle:
        payload = pickle.load(handle)
    return {
        int(document_index): BM25Okapi(value["tokenized"])
        for document_index, value in payload.items()
    }
