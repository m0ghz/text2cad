from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from .evaluator import Evaluator


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run CAD evaluation using compilation success and LLM judging."
    )
    parser.add_argument(
        "--input-file",
        type=Path,
        required=True,
        help="Text file containing one prompt per line.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/evaluation"),
        help="Directory where evaluation artefacts will be written.",
    )
    parser.add_argument(
        "--generator-backend",
        type=str,
        choices=("openai", "gemini", "huggingface"),
        default="openai",
        help="Backend powering the CAD generator.",
    )
    parser.add_argument(
        "--generator-kwargs",
        type=str,
        default=None,
        help="JSON object of keyword arguments passed to the generator constructor.",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default="gpt-5",
        help=(
            "Model identifier passed to the judge backend (e.g. gpt-5, gemini2.5-pro, Qwen/Qwen3-32B:groq)."
        ),
    )
    parser.add_argument(
        "--judge-backend",
        type=str,
        choices=("openai", "gemini", "huggingface"),
        default="openai",
        help="Backend powering the LLM judge.",
    )
    parser.add_argument(
        "--judge-weight",
        type=float,
        default=0.7,
        help="Blend factor between judge score and compilation score (0-1).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional limit on number of samples evaluated.",
    )
    parser.add_argument(
        "--views",
        type=str,
        default="front,side,top,iso",
        help="Comma separated camera views to render.",
    )
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    )

    views = tuple(view.strip() for view in args.views.split(",") if view.strip())
    if args.generator_kwargs:
        try:
            generator_kwargs = json.loads(args.generator_kwargs)
            if not isinstance(generator_kwargs, dict):
                raise ValueError("generator kwargs must decode to a JSON object.")
        except (json.JSONDecodeError, ValueError) as exc:
            parser.error(f"Invalid --generator-kwargs value: {exc}")
    else:
        generator_kwargs = {}

    evaluator = Evaluator.from_generator_class(
        input_file=args.input_file,
        output_dir=args.output_dir,
        generator_backend=args.generator_backend,
        generator_kwargs=generator_kwargs,
        judge_model_name=args.judge_model,
        judge_backend=args.judge_backend,
        judge_weight=args.judge_weight,
        max_samples=args.max_samples,
        views=views,
    )
    result = evaluator.run()
    summary = {
        "average_score": result.average_score,
        "num_samples": len(result.samples),
        "output_dir": str(Path(args.output_dir).resolve()),
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())
