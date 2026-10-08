"""Text normalization and token-aware chunking utilities."""

from __future__ import annotations

import re
from typing import Iterable

import tiktoken

DEFAULT_CHUNK_SIZE = 512
DEFAULT_CHUNK_OVERLAP = 64

def clean_text(text: str) -> str:
    """Normalize whitespace while preserving paragraph boundaries."""
    text = str(text).replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _paragraphs(text: str) -> Iterable[str]:
    for paragraph in re.split(r"\n\s*\n", text):
        paragraph = paragraph.strip()
        if paragraph:
            yield paragraph


def chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
    encoding_name: str = "cl100k_base",
) -> list[str]:
    """Create overlapping chunks, preferring paragraph boundaries."""
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size")

    encoding = tiktoken.get_encoding(encoding_name)
    paragraphs = list(_paragraphs(clean_text(text)))
    if not paragraphs:
        return []

    tokens: list[int] = []
    for index, paragraph in enumerate(paragraphs):
        if index:
            tokens.extend(encoding.encode("\n\n"))
        tokens.extend(encoding.encode(paragraph))

    chunks: list[str] = []
    step = chunk_size - overlap
    for start in range(0, len(tokens), step):
        chunk_tokens = tokens[start : start + chunk_size]
        if not chunk_tokens:
            break
        chunk = encoding.decode(chunk_tokens).strip()
        if chunk:
            chunks.append(chunk)
        if start + chunk_size >= len(tokens):
            break
    return chunks
