"""Persistent ChromaDB storage for dense document chunks."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import chromadb


COLLECTION_NAME = "ask_document_chunks"


def get_collection(
    persist_directory: str | Path = "data/processed/dense",
    collection_name: str = COLLECTION_NAME,
):
    client = chromadb.PersistentClient(path=str(persist_directory))
    return client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )


def upsert_chunks(
    chunks: Sequence[dict[str, Any]],
    embeddings: Sequence[Sequence[float]],
    *,
    persist_directory: str | Path = "data/processed/dense",
) -> None:
    if len(chunks) != len(embeddings):
        raise ValueError("chunks and embeddings must have the same length")
    if not chunks:
        return

    collection = get_collection(persist_directory)
    collection.upsert(
        ids=[chunk["chunk_id"] for chunk in chunks],
        documents=[chunk["text"] for chunk in chunks],
        metadatas=[
            {key: value for key, value in chunk.items() if key != "text"}
            for chunk in chunks
        ],
        embeddings=[list(vector) for vector in embeddings],
    )
