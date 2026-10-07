"""End-to-end ingestion: load, clean, chunk, embed, store, and index."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from .bm25 import build_and_save_bm25
from .chunking import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE, chunk_text
from .embeddings import embed_texts, get_embedding_model
from .vector_store import upsert_chunks


REQUIRED_COLUMNS = {"index", "source_url", "text"}
DEFAULT_DOCUMENTS_PATH = Path("data/raw/documents.csv")


def load_documents(path: str | Path = DEFAULT_DOCUMENTS_PATH) -> pd.DataFrame:
    """Load and validate the authoritative documents CSV."""
    frame = pd.read_csv(path, encoding="utf-8")
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    frame = frame[["index", "source_url", "text"]].copy()
    frame["index"] = pd.to_numeric(frame["index"], errors="raise").astype(int)
    if frame["index"].duplicated().any():
        raise ValueError("The documents file contains duplicate index values.")
    if frame["source_url"].fillna("").astype(str).str.strip().eq("").any():
        raise ValueError("Every document must have a non-empty source_url.")
    frame["text"] = frame["text"].fillna("").astype(str)
    return frame


def create_chunks(frame: pd.DataFrame) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    for row in frame.to_dict("records"):
        document_index = int(row["index"])
        document_chunks = chunk_text(
            row["text"],
            chunk_size=DEFAULT_CHUNK_SIZE,
            overlap=DEFAULT_CHUNK_OVERLAP,
        )
        for sequence, text in enumerate(document_chunks):
            chunks.append(
                {
                    "document_index": document_index,
                    "source_url": str(row["source_url"]),
                    "chunk_id": f"doc-{document_index:02d}-chunk-{sequence:04d}",
                    "chunk_sequence": sequence,
                    "text": text,
                }
            )
    return chunks


def ingest(
    documents_path: str | Path = DEFAULT_DOCUMENTS_PATH,
) -> dict[str, Any]:
    frame = load_documents(documents_path)
    chunks = create_chunks(frame)
    embeddings = embed_texts([chunk["text"] for chunk in chunks])
    upsert_chunks(chunks, embeddings)
    bm25_path = build_and_save_bm25(chunks)
    return {
        "documents": len(frame),
        "chunks": len(chunks),
        "embedding_model": get_embedding_model(),
        "chunk_size": DEFAULT_CHUNK_SIZE,
        "chunk_overlap": DEFAULT_CHUNK_OVERLAP,
        "bm25_index": str(bm25_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest Q_and_A_bot source data")
    parser.add_argument("--documents", default=str(DEFAULT_DOCUMENTS_PATH))
    args = parser.parse_args()
    print(ingest(args.documents))


if __name__ == "__main__":
    main()
