"""Prompt construction for grounded answers."""

from __future__ import annotations

from typing import Any


REFUSAL_TEXT = "I don't know from this document."

SYSTEM_PROMPT = f"""You are a precise document question-answering assistant.

Follow these rules exactly:
1. Use only the numbered evidence chunks supplied in the user message.
2. Do not use general knowledge, hidden memory, web pages, or source URLs.
3. Answer only what the evidence supports.
4. For simple factual questions, answer in one concise sentence.
5. For a simple factual question, answer only the direct fact requested. Do not append related variants, exceptions, examples, or neighboring facts unless the question asks for them.
6. For questions asking what happened, what something looks like, or what the requirements/details are, include the complete directly relevant description from the evidence, even if it takes multiple sentences.
7. For list, comparison, code, version, or "all" questions, first silently make a checklist of every requested item found in every evidence chunk, then include every checklist item in the final answer.
8. Preserve exact names, aliases, numbers, version strings, technical terms, and code from the evidence. Do not replace several exact items with a vague summary.
9. For questions asking for steps, options, requirements, or multiple items, inspect all evidence chunks and include every supported requested item.
10. Cite every factual claim with one or more citations such as [Chunk 1].
11. If the evidence does not answer the question, reply exactly:
   {REFUSAL_TEXT}
12. Do not guess, speculate, or add unsupported details. If only part of a multi-part question is supported, clearly provide only that supported part.
"""


def build_user_prompt(
    question: str,
    results: list[dict[str, Any]],
    question_type: str | None = None,
) -> str:
    evidence_blocks = []
    for position, result in enumerate(results, start=1):
        evidence_blocks.append(f"[Chunk {position}]\n{result['text']}")

    evidence = "\n\n".join(evidence_blocks) or "[No evidence was retrieved.]"
    multi_instruction = ""
    if question_type == "multi_passage":
        multi_instruction = (
            "This question requires combining multiple passages. Inspect every evidence "
            "block, collect every requested fact, and do not stop after finding only one part. "
            "Return a complete, exact list when the question asks for multiple items.\n\n"
        )
    return f"""Evidence:
{evidence}

Question: {question}

{multi_instruction}Answer using only the evidence above. Be concise and answer exactly what was asked. Include citations for supported claims."""


def build_review_prompt(
    question: str,
    results: list[dict[str, Any]],
    draft: str,
    question_type: str | None = None,
) -> str:
    """Build a second-pass completeness check grounded in the same evidence."""
    evidence_blocks = []
    for position, result in enumerate(results, start=1):
        evidence_blocks.append(f"[Chunk {position}]\n{result['text']}")
    evidence = "\n\n".join(evidence_blocks) or "[No evidence was retrieved.]"
    scope = (
        "Include every requested item found across all chunks."
        if question_type == "multi_passage"
        else "Include all directly relevant details needed to answer the question, not just the first matching sentence."
    )
    return f"""Review this draft answer against the evidence.

Evidence:
{evidence}

Question: {question}

Draft answer:
{draft}

{scope} Rewrite the draft using only the evidence. Add missing supported facts, remove unsupported claims, preserve exact names, numbers, and code, and keep citations such as [Chunk 1]. If the evidence does not answer the question, reply exactly: {REFUSAL_TEXT}"""
