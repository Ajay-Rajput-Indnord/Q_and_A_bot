"""Grounded answer generation with citation validation."""

from __future__ import annotations

import os
import re
from typing import Any

from openai import OpenAI

from .embeddings import get_openai_client
from .prompts import REFUSAL_TEXT, SYSTEM_PROMPT, build_review_prompt, build_user_prompt
from .retrieval import retrieve


DEFAULT_CHAT_MODEL = "gpt-4o-mini"
DEFAULT_BROAD_QUESTION_K = 16
_CITATION_PATTERN = re.compile(r"\[Chunk\s+(\d+)\]")


def _retrieval_k_for_question(
    question: str,
    default_k: int,
    question_type: str | None = None,
) -> int:
    """Use a wider evidence set for questions likely to require several items."""
    if question_type == "multi_passage":
        return max(default_k, DEFAULT_BROAD_QUESTION_K)
    normalized = question.lower()
    broad_markers = (
        "what are",
        "what things",
        "what monsters",
        "what enemies",
        "who are",
        "which",
        "list",
        "all of",
        "how many",
        "requirements",
        "steps",
        "options",
        "alternatives",
        "parameters",
        "versions",
        "rules",
        "types",
        "forms",
        "aliases",
    )
    if any(marker in normalized for marker in broad_markers):
        return max(default_k, DEFAULT_BROAD_QUESTION_K)
    return default_k


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
    retrieval_k: int = 8,
    question_type: str | None = None,
    client: OpenAI | None = None,
    model: str | None = None,
) -> dict[str, Any]:
    """Retrieve evidence and generate an answer grounded in that evidence."""
    results = retrieve(
        question,
        document_index,
        method=method,
        top_k=_retrieval_k_for_question(question, retrieval_k, question_type),
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
            {"role": "user", "content": build_user_prompt(question, results, question_type)},
        ],
    )
    answer = (response.choices[0].message.content or "").strip()

    if not answer or REFUSAL_TEXT in answer:
        return {"answer": REFUSAL_TEXT, "citations": [], "chunks": results}

    review_response = client.chat.completions.create(
        model=chat_model,
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": build_review_prompt(question, results, answer, question_type),
            },
        ],
    )
    reviewed_answer = (review_response.choices[0].message.content or "").strip()
    if reviewed_answer and REFUSAL_TEXT not in reviewed_answer and _validate_citations(reviewed_answer, len(results)):
        answer = reviewed_answer

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
