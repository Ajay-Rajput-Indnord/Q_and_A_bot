"""Run dense or hybrid evaluation over the development or test split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ask_document.answer_chain import answer_question  # noqa: E402
try:
    from .metrics import evaluate_record, summarize  # noqa: E402
except ImportError:
    from metrics import evaluate_record, summarize  # noqa: E402


QUESTION_FILES = {
    "single_passage": ROOT / "data/raw/single_passage_answer_questions.csv",
    "multi_passage": ROOT / "data/raw/multi_passage_answer_questions.csv",
    "no_answer": ROOT / "data/raw/no_answer_questions.csv",
}


def load_records(split: str) -> list[dict[str, Any]]:
    if split == "dev":
        valid_indices = set(range(10))
    elif split == "test":
        valid_indices = set(range(10, 20))
    else:
        raise ValueError("split must be 'dev' or 'test'")

    records: list[dict[str, Any]] = []
    for question_type, path in QUESTION_FILES.items():
        frame = pd.read_csv(path)
        for row in frame.to_dict("records"):
            if int(row["document_index"]) not in valid_indices:
                continue
            records.append(
                {
                    "question_type": question_type,
                    "document_index": int(row["document_index"]),
                    "question": str(row["question"]),
                    "gold_answer": row.get("answer"),
                }
            )
    return records


def run_evaluation(method: str, split: str, limit: int | None = None) -> dict[str, Any]:
    records = load_records(split)
    if limit is not None:
        records = records[:limit]

    evaluated: list[dict[str, Any]] = []
    for record in records:
        result = answer_question(
            question=record["question"],
            document_index=record["document_index"],
            method=method,
        )
        evaluated_record = {
            **record,
            **evaluate_record(record["question_type"], record["gold_answer"], result),
        }
        evaluated.append(evaluated_record)

    return {
        "method": method,
        "split": split,
        "summary": summarize(evaluated),
        "records": evaluated,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["dense", "hybrid"], default="hybrid")
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    output = run_evaluation(args.method, args.split, args.limit)
    output_path = ROOT / "eval/results" / f"{args.method}_{args.split}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output["summary"], indent=2))
    print(f"Saved results to {output_path}")


if __name__ == "__main__":
    main()
