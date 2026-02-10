"""
Legacy shim: re-exports LLMClient from prism_eval.providers for backward compatibility.
Prefer: from prism_eval.providers import LLMClient
"""

from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from prism_eval.providers import LLMClient

__all__ = ["LLMClient"]
