"""Compare dense and hybrid evaluation result files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    args = parser.parse_args()

    result_directory = Path("eval/results")
    dense_path = result_directory / f"dense_{args.split}.json"
    hybrid_path = result_directory / f"hybrid_{args.split}.json"
    if not dense_path.exists() or not hybrid_path.exists():
        raise SystemExit("Run both dense and hybrid evaluations before comparing them.")

    dense = json.loads(dense_path.read_text(encoding="utf-8"))["summary"]
    hybrid = json.loads(hybrid_path.read_text(encoding="utf-8"))["summary"]
    metrics = ["answer_accuracy", "average_answer_token_f1", "refusal_rate", "citation_validity"]
    print("metric                         dense       hybrid")
    print("-" * 55)
    for metric in metrics:
        print(f"{metric:30} {dense[metric]:.3f}       {hybrid[metric]:.3f}")


if __name__ == "__main__":
    main()
