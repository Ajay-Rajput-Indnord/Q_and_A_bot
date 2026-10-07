"""OpenAI embedding helpers."""

from __future__ import annotations

import os
from collections.abc import Sequence

from dotenv import load_dotenv
from openai import OpenAI


DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"


def get_openai_client() -> OpenAI:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is missing. Add it to the .env file.")
    return OpenAI(api_key=api_key)


def get_embedding_model() -> str:
    load_dotenv()
    return os.getenv("OPENAI_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)


def embed_texts(
    texts: Sequence[str],
    *,
    client: OpenAI | None = None,
    model: str | None = None,
    batch_size: int = 100,
) -> list[list[float]]:
    """Embed texts in batches and return vectors in input order."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not texts:
        return []

    client = client or get_openai_client()
    model = model or get_embedding_model()
    vectors: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        batch = list(texts[start : start + batch_size])
        response = client.embeddings.create(model=model, input=batch)
        vectors.extend(item.embedding for item in sorted(response.data, key=lambda item: item.index))
    return vectors
