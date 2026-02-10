"""
Legacy entry point: batch evaluation. Prefer `python main.py evaluate <input.csv> <output.csv>`.
"""

import sys

from pathlib import Path
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def main() -> None:
    if len(sys.argv) < 3:
        print("Usage: python batch_evaluator.py <input.csv> <output.csv>")
        print("Example: python batch_evaluator.py output.csv evaluated.csv")
        print("Recommended: python main.py evaluate output.csv evaluated.csv")
        sys.exit(1)
    from prism_eval.core.batch_evaluate import batch_evaluate
    batch_evaluate(sys.argv[1], sys.argv[2])


if __name__ == "__main__":
    main()
