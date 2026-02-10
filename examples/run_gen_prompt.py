#!/usr/bin/env python3
"""
Minimal example: generate an evaluator prompt from scenario and north-star metric.

Usage (from project root):
  python examples/run_gen_prompt.py
  python examples/run_gen_prompt.py "Customer support replies" "Empathy and clarity"
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main() -> None:
    scenario = sys.argv[1] if len(sys.argv) > 1 else "Social media copy generation"
    north_star = sys.argv[2] if len(sys.argv) > 2 else "Humor and shareability"
    from prism_eval.core.prompt_generator import generate_evaluator_prompt
    prompt = generate_evaluator_prompt(scenario, north_star)
    print("\n--- Generated evaluator prompt (first 800 chars) ---\n")
    print(prompt[:800] + ("..." if len(prompt) > 800 else ""))
    print("\n--- End ---")


if __name__ == "__main__":
    main()
