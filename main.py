#!/usr/bin/env python3
"""
PrismEval CLI: batch generation, batch evaluation, evaluator prompt generation, and Streamlit app.

Usage:
  python main.py generate    <input.csv> <output.csv>   [--workers N]
  python main.py evaluate     <input.csv> <output.csv>   [--workers N]
  python main.py gen-prompt   <scenario> <north_star>     [--output FILE]
  python main.py app                                         # Launch Streamlit UI
"""

import argparse
import sys
from pathlib import Path

# Ensure project root is on path
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def cmd_generate(args: argparse.Namespace) -> int:
    """Run batch generation."""
    from prism_eval.core.batch_generate import batch_generate
    try:
        batch_generate(
            args.input_csv,
            args.output_csv,
            max_workers=args.workers,
        )
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def cmd_evaluate(args: argparse.Namespace) -> int:
    """Run batch evaluation."""
    from prism_eval.core.batch_evaluate import batch_evaluate
    try:
        batch_evaluate(
            args.input_csv,
            args.output_csv,
            max_workers=args.workers,
        )
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def cmd_gen_prompt(args: argparse.Namespace) -> int:
    """Generate evaluator prompt from scenario and north-star metric."""
    from prism_eval.core.prompt_generator import generate_evaluator_prompt
    try:
        prompt = generate_evaluator_prompt(args.scenario, args.north_star)
        print("\n" + "=" * 60)
        print("Generated Evaluator Prompt")
        print("=" * 60)
        print(prompt)
        print("=" * 60)
        if args.output:
            Path(args.output).write_text(prompt, encoding="utf-8")
            print(f"\nSaved to: {args.output}")
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def cmd_app(_: argparse.Namespace) -> int:
    """Launch Streamlit app."""
    import subprocess
    app_path = _PROJECT_ROOT / "app.py"
    if not app_path.exists():
        print("app.py not found.", file=sys.stderr)
        return 1
    return subprocess.run([sys.executable, "-m", "streamlit", "run", str(app_path), "--", *sys.argv[2:]]).returncode


def main() -> int:
    parser = argparse.ArgumentParser(
        description="PrismEval: LLM evaluation pipeline (generate, evaluate, gen-prompt, app).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", help="Command")

    # generate
    p_gen = sub.add_parser("generate", help="Batch generate model responses from input CSV")
    p_gen.add_argument("input_csv", help="Input CSV path (must have input_text column)")
    p_gen.add_argument("output_csv", help="Output CSV path")
    p_gen.add_argument("--workers", type=int, default=None, help="Concurrency (default from config)")
    p_gen.set_defaults(func=cmd_generate)

    # evaluate
    p_eval = sub.add_parser("evaluate", help="Batch evaluate model outputs and write scores to CSV")
    p_eval.add_argument("input_csv", help="Input CSV path (must have input_text and final_content)")
    p_eval.add_argument("output_csv", help="Output CSV path")
    p_eval.add_argument("--workers", type=int, default=None, help="Concurrency (default from config)")
    p_eval.set_defaults(func=cmd_evaluate)

    # gen-prompt
    p_prompt = sub.add_parser("gen-prompt", help="Generate evaluator prompt from scenario and north-star metric")
    p_prompt.add_argument("scenario", help="Application scenario description")
    p_prompt.add_argument("north_star", help="North star metric description")
    p_prompt.add_argument("--output", "-o", help="Write prompt to this file")
    p_prompt.set_defaults(func=cmd_gen_prompt)

    # app
    p_app = sub.add_parser("app", help="Launch Streamlit evaluation UI")
    p_app.set_defaults(func=cmd_app)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return 0
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
