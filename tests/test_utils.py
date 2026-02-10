"""
Tests for utils: config loader, CSV validation, logging helpers.
"""

import pytest

from prism_eval.utils.config_loader import get_config, get, get_system_prompt, reset_cache
from prism_eval.utils.io import validate_csv_columns
from prism_eval.utils.logging import format_section_header, format_metric_line


def test_get_config_returns_dict() -> None:
    """get_config() returns a dict-like structure."""
    reset_cache()
    cfg = get_config()
    assert isinstance(cfg, dict)


def test_get_nested_key() -> None:
    """get() resolves dot-separated keys."""
    reset_cache()
    # May be empty if YAML not present
    val = get("batch.max_workers")
    assert val is None or isinstance(val, int)


def test_validate_csv_columns_ok() -> None:
    """validate_csv_columns passes when columns exist."""
    import pandas as pd
    df = pd.DataFrame({"question": ["q1"], "expected_answer": ["a1"]})
    ok, err = validate_csv_columns(df, ["question"])
    assert ok is True
    assert err == ""


def test_validate_csv_columns_missing() -> None:
    """validate_csv_columns fails when a column is missing."""
    import pandas as pd
    df = pd.DataFrame({"question": ["q1"]})
    ok, err = validate_csv_columns(df, ["question", "missing_col"])
    assert ok is False
    assert "missing" in err.lower()


def test_format_section_header() -> None:
    """Section header contains title and line."""
    s = format_section_header("Title", width=10)
    assert "Title" in s
    assert "=" in s


def test_format_metric_line() -> None:
    """Metric line formats label and value."""
    s = format_metric_line("Score", "82.5")
    assert "Score" in s
    assert "82.5" in s
