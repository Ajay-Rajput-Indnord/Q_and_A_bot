"""Evaluation metrics for answer quality and grounding."""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

from src.ask_document.prompts import REFUSAL_TEXT


ANSWER_COVERAGE_THRESHOLD = 0.5


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).lower()).strip()


def token_f1(prediction: str, reference: str) -> float:
    predicted_tokens = normalize(prediction).split()
    reference_tokens = normalize(reference).split()
    if not predicted_tokens or not reference_tokens:
        return float(predicted_tokens == reference_tokens)

    overlap = sum((Counter(predicted_tokens) & Counter(reference_tokens)).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(predicted_tokens)
    recall = overlap / len(reference_tokens)
    return 2 * precision * recall / (precision + recall)


def token_recall(prediction: str, reference: str) -> float:
    """Measure how much of the reference answer is covered by the prediction.

    Recall is separate from token F1 because a grounded answer may add useful,
    cited detail and should not be marked wrong merely for being longer than
    the compact gold answer.
    """
    predicted_tokens = normalize(prediction).split()
    reference_tokens = normalize(reference).split()
    if not reference_tokens:
        return float(not predicted_tokens)
    overlap = sum((Counter(predicted_tokens) & Counter(reference_tokens)).values())
    return overlap / len(reference_tokens)


def citation_numbers(answer: str) -> list[int]:
    return sorted({int(value) for value in re.findall(r"\[Chunk\s+(\d+)\]", answer)})


def evaluate_record(
    question_type: str,
    gold_answer: str | None,
    result: dict[str, Any],
) -> dict[str, Any]:
    answer = result.get("answer", "")
    citations = result.get("citations", [])
    chunks = result.get("chunks", [])
    is_refusal = answer == REFUSAL_TEXT

    record: dict[str, Any] = {
        "answer": answer,
        "citations": citations,
        "is_refusal": is_refusal,
        "citation_count": len(citations),
    }
    if question_type == "no_answer":
        record["refusal_correct"] = is_refusal
    else:
        score = token_f1(answer, gold_answer or "")
        coverage = token_recall(answer, gold_answer or "")
        record["answer_token_f1"] = score
        record["answer_token_recall"] = coverage
        # F1 penalizes extra words. Correctness is based on required-answer
        # coverage, while citation validity remains tracked separately.
        record["answer_correct"] = (
            coverage >= ANSWER_COVERAGE_THRESHOLD and not is_refusal
        )
        # Filled during manual review; automated token F1 is intentionally retained.
        record["manual_semantic_correct"] = None

    cited_text = " ".join(chunk.get("text", "") for chunk in chunks)
    record["citation_supported_overlap"] = token_f1(answer, cited_text) if citations else 0.0
    record["citation_valid"] = bool(citations) and len(citations) == len(chunks)
    return record


def summarize(records: list[dict[str, Any]]) -> dict[str, float | int]:
    total = len(records)
    answerable = [r for r in records if r["question_type"] != "no_answer"]
    no_answer = [r for r in records if r["question_type"] == "no_answer"]
    return {
        "questions": total,
        "answer_accuracy": (
            sum(bool(r.get("answer_correct")) for r in answerable) / len(answerable)
            if answerable else 0.0
        ),
        "average_answer_token_f1": (
            sum(float(r.get("answer_token_f1", 0.0)) for r in answerable) / len(answerable)
            if answerable else 0.0
        ),
        "average_answer_token_recall": (
            sum(float(r.get("answer_token_recall", 0.0)) for r in answerable) / len(answerable)
            if answerable else 0.0
        ),
        "refusal_rate": (
            sum(bool(r.get("refusal_correct")) for r in no_answer) / len(no_answer)
            if no_answer else 0.0
        ),
        "citation_validity": (
            sum(bool(r.get("citation_valid")) for r in answerable) / len(answerable)
            if answerable else 0.0
        ),
        "manual_semantic_accuracy": None,
    }
