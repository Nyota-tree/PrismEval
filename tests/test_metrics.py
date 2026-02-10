"""
Tests for evaluation metric extraction (extract_evaluation, extract_json_from_text).
"""

import pytest

from prism_eval.metrics.extractor import extract_evaluation, extract_json_from_text


def test_extract_json_from_text_direct() -> None:
    """Direct JSON string is parsed."""
    text = '{"scores": {"factuality_score": 8}, "reasoning": "Good"}'
    out = extract_json_from_text(text)
    assert out is not None
    assert out.get("scores", {}).get("factuality_score") == 8


def test_extract_json_from_text_code_block() -> None:
    """JSON inside ```json ... ``` is extracted."""
    text = 'Some text\n```json\n{"a": 1}\n```'
    out = extract_json_from_text(text)
    assert out is not None
    assert out.get("a") == 1


def test_extract_json_from_text_invalid() -> None:
    """Invalid or missing JSON returns None."""
    assert extract_json_from_text("no json here") is None
    assert extract_json_from_text("") is None


def test_extract_evaluation_flat_scores() -> None:
    """Flat score format is normalized."""
    raw = {
        "determined_priority": "P1",
        "scores": {
            "factuality_score": 8,
            "completeness_score": 7,
            "adherence_score": 8,
            "attractiveness_score": 7,
        },
        "weighted_total_score": 76,
        "reasoning": "Solid answer.",
        "pass": True,
    }
    result = extract_evaluation(raw)
    assert result["priority"] == "P1"
    assert result["factuality_score"] == 8
    assert result["weighted_total_score"] == 76
    assert result["decision"] == "PUBLISH"
    assert "reasoning" in result
