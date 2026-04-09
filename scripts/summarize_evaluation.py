#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from src.evaluation.io import read_jsonl
from src.evaluation.reporting import build_summary_report, write_summary_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize before/after benchmark results")
    parser.add_argument("--tifa_results_before", type=str, required=True)
    parser.add_argument("--tifa_results_after", type=str, required=True)
    parser.add_argument("--genai_results_before", type=str, required=True)
    parser.add_argument("--genai_results_after", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_summary_report(
        read_jsonl(Path(args.tifa_results_before)),
        read_jsonl(Path(args.tifa_results_after)),
        read_jsonl(Path(args.genai_results_before)),
        read_jsonl(Path(args.genai_results_after)),
    )
    write_summary_outputs(Path(args.output_dir), summary)


if __name__ == "__main__":
    main()

