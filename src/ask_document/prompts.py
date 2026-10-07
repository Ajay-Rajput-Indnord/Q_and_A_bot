"""Prompt construction for grounded answers."""

from __future__ import annotations

from typing import Any


REFUSAL_TEXT = "I don't know from this document."

SYSTEM_PROMPT = f"""You are a precise document question-answering assistant.

Follow these rules exactly:
1. Use only the numbered evidence chunks supplied in the user message.
2. Do not use general knowledge, hidden memory, web pages, or source URLs.
3. Answer only what the evidence supports.
4. Cite every factual claim with one or more citations such as [Chunk 1].
5. If the evidence does not answer the question, reply exactly:
   {REFUSAL_TEXT}
6. Do not guess, speculate, or add unsupported details.
"""


def build_user_prompt(question: str, results: list[dict[str, Any]]) -> str:
    evidence_blocks = []
    for position, result in enumerate(results, start=1):
        evidence_blocks.append(f"[Chunk {position}]\n{result['text']}")

    evidence = "\n\n".join(evidence_blocks) or "[No evidence was retrieved.]"
    return f"""Evidence:
{evidence}

Question: {question}

Answer using only the evidence above. Include citations for supported claims."""
