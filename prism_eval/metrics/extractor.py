"""
Evaluation result extraction: parse LLM evaluation JSON and compute scores.

Supports both flat and nested score formats; normalizes to a canonical
structure with factuality, north_star, completeness, weighted_total_score,
decision, and reasoning.
"""

import json
import re
from typing import Any, Dict, Optional


def _extract_balanced_brace_block(text: str, start: int) -> Optional[str]:
    """From text[start:] find the substring that is one balanced { ... } block."""
    i = text.find("{", start)
    if i == -1:
        return None
    depth = 0
    in_string = False
    escape = False
    quote = None
    j = i
    while j < len(text):
        c = text[j]
        if escape:
            escape = False
            j += 1
            continue
        if c == "\\" and in_string:
            escape = True
            j += 1
            continue
        if not in_string:
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return text[i : j + 1]
            elif c in ("'", '"'):
                in_string = True
                quote = c
        elif c == quote:
            in_string = False
        j += 1
    return None


def extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    """
    Extract a JSON object from raw text (e.g. LLM response with markdown or prose).

    Tries: direct parse, ```json ... ``` block, balanced {...} block, then regex brace blocks.

    Args:
        text: Raw response string.

    Returns:
        Parsed dict or None if no valid JSON found.
    """
    raw = (text or "").strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Markdown code block: use greedy match so nested JSON is captured fully
    json_block_pattern = r"```(?:json)?\s*(\{.*\})\s*```"
    match = re.search(json_block_pattern, raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Balanced brace block (handles raw JSON with leading/trailing text or nesting)
    balanced = _extract_balanced_brace_block(raw, 0)
    if balanced:
        try:
            return json.loads(balanced)
        except json.JSONDecodeError:
            pass

    # Fallback: regex for one/two level nesting
    brace_pattern = r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}"
    for m in re.findall(brace_pattern, raw, re.DOTALL):
        try:
            return json.loads(m)
        except json.JSONDecodeError:
            continue

    return None


def _get_numeric_score(scores: Dict[str, Any], *keys: str) -> float:
    """Return first matching numeric value from scores for the given keys."""
    for k in keys:
        v = scores.get(k)
        if v is None:
            continue
        try:
            return float(v)
        except (ValueError, TypeError):
            continue
    return 0.0


def _detect_north_star_from_scores(scores: Dict[str, Any]) -> float:
    """Detect north_star_score from various key names (north_star, northstar, etc.)."""
    val = scores.get("north_star_score", 0)
    try:
        v = float(val) if val not in ("", None) else 0
        if v > 0:
            return v
    except (ValueError, TypeError):
        pass
    for key, value in scores.items():
        if not isinstance(value, (int, float, str)):
            continue
        k = (key or "").lower().replace(" ", "").replace("_", "")
        if "northstar" in k or "north_star" in (key or "").lower():
            try:
                v = float(value) if value not in ("", None) else 0
                if v > 0:
                    return v
            except (ValueError, TypeError):
                pass
    return 0.0


def extract_evaluation(response_json: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract and normalize evaluation result from evaluator JSON.

    Supports:
    - Flat format: scores.factuality_score, north_star_score, etc.
    - Nested format: scores.dimension_name = { "score": 9, "weight": 0.35 }

    Ensures numeric types, computes weighted_total_score if missing,
    and applies decision rules (REJECT / REVIEW / PUBLISH).

    Args:
        response_json: Raw JSON returned by the evaluator LLM.

    Returns:
        Dict with keys: priority, factuality_score, completeness_score,
        north_star_score, weighted_total_score, decision, reason, reasoning, pass.
    """
    priority = response_json.get("determined_priority", "P3")
    scores = response_json.get("scores", {})
    north_star_score_val = 0.0

    # Detect nested vs flat format
    is_nested = False
    if scores:
        first_val = next(iter(scores.values()), None)
        if isinstance(first_val, dict) and ("score" in first_val or "weight" in first_val):
            is_nested = True

    if is_nested:
        factuality_score = completeness_score = adherence_score = attractiveness_score = 0.0
        for key, value in scores.items():
            if not isinstance(value, dict):
                continue
            score_val = value.get("score", 0)
            key_lower = (key or "").lower()
            if "factuality" in key_lower or "safety" in key_lower:
                factuality_score = score_val
            elif "completeness" in key_lower or "coverage" in key_lower:
                completeness_score = score_val
            elif "adherence" in key_lower or "instruction" in key_lower or "compliance" in key_lower:
                adherence_score = score_val
            elif "attractiveness" in key_lower or "quality" in key_lower or "appeal" in key_lower:
                attractiveness_score = score_val
            if "north_star" in key_lower or "northstar" in key_lower.replace("_", ""):
                north_star_score_val = score_val
    else:
        factuality_score = _get_numeric_score(
            scores, "factuality_score", "factuality_safety_score", "factuality", "safety_score"
        )
        completeness_score = _get_numeric_score(
            scores, "completeness_score", "completeness_coherence_score", "completeness", "coverage_score"
        )
        adherence_score = _get_numeric_score(
            scores, "adherence_score", "adherence", "instruction_score"
        )
        attractiveness_score = _get_numeric_score(
            scores, "attractiveness_score", "attractiveness", "quality_score"
        )
        north_star_score_val = _detect_north_star_from_scores(scores)
        if attractiveness_score == 0 and north_star_score_val:
            attractiveness_score = north_star_score_val

        # Fallback: fuzzy key match
        for key, value in scores.items():
            if not isinstance(value, (int, float)):
                continue
            try:
                v = float(value)
            except (ValueError, TypeError):
                continue
            key_lower = key.lower()
            if (("factuality" in key_lower or "safety" in key_lower) and factuality_score == 0):
                factuality_score = v
            elif (("completeness" in key_lower or "coverage" in key_lower) and completeness_score == 0):
                completeness_score = v
            elif (("adherence" in key_lower or "instruction" in key_lower) and adherence_score == 0):
                adherence_score = v
            elif (("attractiveness" in key_lower or "quality" in key_lower or "north_star" in key_lower) and attractiveness_score == 0):
                attractiveness_score = v
            if "north_star" in key_lower and north_star_score_val == 0:
                north_star_score_val = v

    # Top-level fallback
    if factuality_score == 0 and completeness_score == 0 and adherence_score == 0 and attractiveness_score == 0:
        factuality_score = response_json.get("factuality_score", response_json.get("factuality", 0)) or 0
        completeness_score = response_json.get("completeness_score", response_json.get("completeness", 0)) or 0
        adherence_score = response_json.get("adherence_score", response_json.get("adherence", 0)) or 0
        attractiveness_score = response_json.get("attractiveness_score", response_json.get("north_star_score", response_json.get("attractiveness", 0))) or 0
    if north_star_score_val == 0:
        north_star_score_val = response_json.get("north_star_score", 0) or scores.get("north_star_score", 0)

    # Ensure numeric
    try:
        factuality_score = float(factuality_score) if factuality_score else 0.0
        completeness_score = float(completeness_score) if completeness_score else 0.0
        adherence_score = float(adherence_score) if adherence_score else 0.0
        attractiveness_score = float(attractiveness_score) if attractiveness_score else 0.0
        north_star_score_val = float(north_star_score_val) if north_star_score_val else 0.0
    except (ValueError, TypeError):
        factuality_score = completeness_score = adherence_score = attractiveness_score = north_star_score_val = 0.0

    # Scale: 0–10 vs 0–100
    # Read the model-reported weighted total first to use as a tiebreaker.
    # If any individual score > 10 → definitely 0–100 scale.
    # If all individual scores ≤ 10 AND model_weighted > 10 → 0–10 scale.
    # If all individual scores ≤ 10 AND model_weighted ≤ 10 → 0–100 scale with low scores.
    # If model_weighted is absent and all scores ≤ 10 → assume 0–10 (most eval prompts use it).
    max_score = max(factuality_score, completeness_score, adherence_score, attractiveness_score)
    model_weighted_raw = response_json.get("weighted_total_score")
    try:
        _mw_hint = float(model_weighted_raw) if model_weighted_raw is not None else None
    except (ValueError, TypeError):
        _mw_hint = None

    if max_score > 10:
        is_0_10_scale = False
    elif max_score > 0:
        # Use weighted total as tiebreaker: >10 means individual scores are on 0–10 scale
        is_0_10_scale = _mw_hint is None or _mw_hint > 10
    else:
        is_0_10_scale = False

    model_weighted = model_weighted_raw
    if model_weighted is not None:
        try:
            model_weighted = float(model_weighted)
            if 0 <= model_weighted <= 10:
                model_weighted = model_weighted * 10
            if 0 <= model_weighted <= 100:
                weighted_score = model_weighted
            else:
                weighted_score = _compute_weighted(
                    factuality_score, completeness_score, adherence_score, attractiveness_score, is_0_10_scale
                )
        except (ValueError, TypeError):
            weighted_score = _compute_weighted(
                factuality_score, completeness_score, adherence_score, attractiveness_score, is_0_10_scale
            )
    else:
        weighted_score = _compute_weighted(
            factuality_score, completeness_score, adherence_score, attractiveness_score, is_0_10_scale
        )

    # Decision rules (from configs/config.yaml evaluation section)
    try:
        from prism_eval.utils.config_loader import get_evaluation_thresholds
        thresholds = get_evaluation_thresholds()
        factuality_threshold = (
            thresholds["factuality_threshold_0_10"]
            if is_0_10_scale
            else thresholds["factuality_threshold_0_100"]
        )
        publish_threshold = thresholds["publish_threshold"]
    except Exception:
        factuality_threshold = 5 if is_0_10_scale else 50
        publish_threshold = 75
    if factuality_score < factuality_threshold:
        decision = "REJECT"
        reason = "Hallucination Detected"
    elif weighted_score >= publish_threshold:
        decision = "PUBLISH"
        reason = "High Quality Score"
    else:
        decision = "REVIEW"
        reason = "Low Quality Score"

    return {
        "priority": priority,
        "factuality_score": factuality_score,
        "completeness_score": completeness_score,
        "adherence_score": adherence_score,
        "attractiveness_score": attractiveness_score,
        "north_star_score": north_star_score_val,
        "weighted_total_score": weighted_score,
        "decision": decision,
        "reason": reason,
        "reasoning": response_json.get("reasoning", ""),
        "pass": decision == "PUBLISH",
    }


def _compute_weighted(
    fact: float,
    complete: float,
    adherence: float,
    attract: float,
    is_0_10: bool,
) -> float:
    """Compute weighted total (weights: 3, 2, 2.5, 2.5 for 0–10 scale)."""
    if is_0_10:
        return fact * 3 + complete * 2 + adherence * 2.5 + attract * 2.5
    return fact * 0.3 + complete * 0.2 + adherence * 0.25 + attract * 0.25


def flatten_eval_json(
    json_obj: Dict[str, Any],
    prefix: str = "",
    score_prefix: str = "score_",
    reasoning_column: str = "eval_reasoning",
) -> Dict[str, Any]:
    """
    Flatten nested evaluation JSON to one-level dict for CSV columns.

    Example: {"scores": {"clarity": 5}, "reasoning": "good"}
    -> {"score_clarity": 5, "eval_reasoning": "good"}.

    Args:
        json_obj: Evaluation result dict.
        prefix: Unused (kept for API compatibility).
        score_prefix: Prefix for score keys.
        reasoning_column: Key name for the reasoning field.

    Returns:
        Flattened dict.
    """
    flattened: Dict[str, Any] = {}
    for key, value in json_obj.items():
        if key.lower() == "reasoning":
            new_key = reasoning_column
        else:
            new_key = f"{score_prefix}{key}"
        if isinstance(value, dict):
            nested = flatten_eval_json(value, new_key + "_", score_prefix, reasoning_column)
            flattened.update(nested)
        else:
            flattened[new_key] = value
    return flattened
