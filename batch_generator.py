"""
Legacy entry point: batch generation. Prefer `python main.py generate <input.csv> <output.csv>`.
"""

import sys

# Ensure project root on path
from pathlib import Path
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def main() -> None:
    if len(sys.argv) < 3:
        print("Usage: python batch_generator.py <input.csv> <output.csv>")
        print("Example: python batch_generator.py input.csv output.csv")
        print("Recommended: python main.py generate input.csv output.csv")
        sys.exit(1)
    from prism_eval.core.batch_generate import batch_generate
    batch_generate(sys.argv[1], sys.argv[2])


if __name__ == "__main__":
    main()
