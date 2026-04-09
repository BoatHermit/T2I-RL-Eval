#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.tifa_runner import build_openai_client, evaluate_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run TIFA evaluation on generated images")
    parser.add_argument("--manifest_path", type=str, required=True)
    parser.add_argument("--images_root", type=str, required=True)
    parser.add_argument("--variant", choices=["before", "after"], required=True)
    parser.add_argument("--judge_model", type=str, default="gpt-4.1-mini")
    parser.add_argument("--api_provider", type=str, default="openai")
    parser.add_argument("--base_url", type=str, default=None)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("SILICONFLOW_API_KEY")
    client = build_openai_client(api_key=api_key, base_url=args.base_url)
    evaluate_manifest(
        manifest_path=Path(args.manifest_path),
        images_root=Path(args.images_root),
        variant=args.variant,
        judge_client=client,
        judge_model=args.judge_model,
        output_path=Path(args.output_path),
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
