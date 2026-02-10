"""
Legacy entry point: generate evaluator prompt from scenario and north-star metric.
Prefer `python main.py gen-prompt <scenario> <north_star> [--output FILE]`.
"""

import sys

from pathlib import Path
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def main() -> None:
    if len(sys.argv) < 3:
        print("Usage: python generate_evaluator_prompt.py <scenario> <north_star> [output_file]")
        print('Example: python generate_evaluator_prompt.py "Social copy generation" "Humor and shareability"')
        print("Recommended: python main.py gen-prompt <scenario> <north_star> [--output FILE]")
        sys.exit(1)
    from prism_eval.core.prompt_generator import generate_evaluator_prompt
    scenario = sys.argv[1]
    north_star = sys.argv[2]
    output_file = sys.argv[3] if len(sys.argv) > 3 else "generated_evaluator_prompt.txt"
    prompt = generate_evaluator_prompt(scenario, north_star)
    print("\n" + "=" * 60)
    print("Generated Evaluator Prompt")
    print("=" * 60)
    print(prompt)
    print("=" * 60)
    Path(output_file).write_text(prompt, encoding="utf-8")
    print(f"\nSaved to: {output_file}")


if __name__ == "__main__":
    main()
