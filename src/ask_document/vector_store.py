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


def search_dense(
    query_embedding: Sequence[float],
    document_index: int,
    *,
    top_k: int = 10,
    persist_directory: str | Path = "data/processed/dense",
) -> list[dict[str, Any]]:
    """Search ChromaDB while enforcing selected-document isolation."""
    if top_k <= 0:
        return []

    collection = get_collection(persist_directory)
    if collection.count() == 0:
        return []

    result = collection.query(
        query_embeddings=[list(query_embedding)],
        n_results=top_k,
        where={"document_index": int(document_index)},
        include=["documents", "metadatas", "distances"],
    )
    ids = (result.get("ids") or [[]])[0]
    documents = (result.get("documents") or [[]])[0]
    metadatas = (result.get("metadatas") or [[]])[0]
    distances = (result.get("distances") or [[]])[0]

    return [
        {
            "chunk_id": ids[position],
            "text": documents[position],
            "metadata": metadatas[position],
            "score": 1.0 - float(distances[position]),
            "retriever": "dense",
        }
        for position in range(len(ids))
    ]
