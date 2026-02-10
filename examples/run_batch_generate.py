#!/usr/bin/env python3
"""
Minimal example: run batch generation using the PrismEval pipeline.

Usage (from project root):
  python examples/run_batch_generate.py

Expects examples/input_example.csv (or set INPUT_CSV / OUTPUT_CSV).
"""

import os
import sys
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

def main() -> None:
    input_csv = os.environ.get("INPUT_CSV", str(ROOT / "examples" / "input_example.csv"))
    output_csv = os.environ.get("OUTPUT_CSV", str(ROOT / "examples" / "output_generated.csv"))
    if not Path(input_csv).exists():
        print(f"Input file not found: {input_csv}")
        print("Set INPUT_CSV and OUTPUT_CSV or use examples/input_example.csv")
        sys.exit(1)
    from prism_eval.core.batch_generate import batch_generate
    batch_generate(input_csv, output_csv)
    print(f"Output written to: {output_csv}")


if __name__ == "__main__":
    main()
