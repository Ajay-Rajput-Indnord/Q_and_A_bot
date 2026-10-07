"""Grounded answer generation with citation validation."""

from __future__ import annotations

import os
import re
from typing import Any

from openai import OpenAI

from .embeddings import get_openai_client
from .prompts import REFUSAL_TEXT, SYSTEM_PROMPT, build_user_prompt
from .retrieval import retrieve


DEFAULT_CHAT_MODEL = "gpt-4o-mini"
_CITATION_PATTERN = re.compile(r"\[Chunk\s+(\d+)\]")


def _citation_numbers(answer: str) -> list[int]:
    return sorted({int(value) for value in _CITATION_PATTERN.findall(answer)})


def _validate_citations(answer: str, result_count: int) -> bool:
    citations = _citation_numbers(answer)
    return bool(citations) and all(1 <= number <= result_count for number in citations)


def answer_question(
    question: str,
    document_index: int,
    *,
    method: str = "hybrid",
    retrieval_k: int = 5,
    client: OpenAI | None = None,
    model: str | None = None,
) -> dict[str, Any]:
    """Retrieve evidence and generate an answer grounded in that evidence."""
    results = retrieve(
        question,
        document_index,
        method=method,
        top_k=retrieval_k,
    )
    if not results:
        return {"answer": REFUSAL_TEXT, "citations": [], "chunks": []}

    client = client or get_openai_client()
    chat_model = model or os.getenv("OPENAI_CHAT_MODEL", DEFAULT_CHAT_MODEL)
    response = client.chat.completions.create(
        model=chat_model,
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(question, results)},
        ],
    )
    answer = (response.choices[0].message.content or "").strip()

    if not answer or REFUSAL_TEXT in answer:
        return {"answer": REFUSAL_TEXT, "citations": [], "chunks": results}
    if not _validate_citations(answer, len(results)):
        return {"answer": REFUSAL_TEXT, "citations": [], "chunks": results}

    citation_numbers = _citation_numbers(answer)
    cited_chunks = [results[number - 1] for number in citation_numbers]
    return {
        "answer": answer,
        "citations": citation_numbers,
        "chunks": cited_chunks,
        "retrieved_chunks": results,
    }
