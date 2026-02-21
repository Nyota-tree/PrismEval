"""
Batch evaluation: read CSV with model outputs, run LLM evaluator per row, write scores to CSV.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, Optional

import pandas as pd
from tqdm import tqdm

from prism_eval.metrics.extractor import extract_evaluation, extract_json_from_text
from prism_eval.providers import LLMClient
from prism_eval.utils.config_loader import get_config, get_evaluation_prompt_template
from prism_eval.utils.io import load_csv_with_encoding, validate_csv_columns
from prism_eval.utils.logging import format_section_header, format_metric_line


EVAL_COLUMNS = [
    "eval_priority",
    "factuality_score",
    "north_star_score",
    "completeness_score",
    "adherence_score",
    "attractiveness_score",
    "weighted_total_score",
    "decision",
    "reason",
    "reasoning",
    "pass",
]


def _process_single_row(
    idx: int,
    row: pd.Series,
    llm_client: LLMClient,
    input_column: str,
    content_column: str,
    evaluation_prompt: str,
) -> tuple:
    """Run evaluation for one row. Returns (idx, evaluation_result dict or None, error str or None)."""
    try:
        original_text = str(row[input_column])
        model_output = str(row[content_column])

        if model_output.startswith("error:") or not model_output or model_output == "nan":
            return (idx, None, "Skipped: error or empty content")

        filled = evaluation_prompt.replace("{original_text}", original_text).replace("{model_output}", model_output)
        response = llm_client.generate(system_prompt="", user_prompt=filled)
        json_obj = extract_json_from_text(response)
        if json_obj is None:
            return (idx, None, f"Could not extract JSON: {response[:100]}")
        evaluation_result = extract_evaluation(json_obj)
        return (idx, evaluation_result, None)
    except Exception as e:
        return (idx, None, str(e))


def batch_evaluate(
    input_csv: str,
    output_csv: str,
    *,
    input_column: Optional[str] = None,
    content_column: Optional[str] = None,
    evaluation_prompt: Optional[str] = None,
    max_workers: Optional[int] = None,
) -> None:
    """
    Load CSV, evaluate each row's model output with the evaluator LLM, save results.

    Expects columns for input text and model output (e.g. input_text, final_content).
    Adds columns: eval_priority, factuality_score, north_star_score, completeness_score,
    weighted_total_score, decision, reason, reasoning, pass.

    Args:
        input_csv: Path to input CSV.
        output_csv: Path to output CSV.
        input_column: Column name for original input (default from config).
        content_column: Column name for model output to evaluate (default: final_content).
        evaluation_prompt: Override default evaluation prompt (must contain {original_text}, {model_output}).
        max_workers: Concurrency (default from config).
    """
    cfg = get_config()
    cols = cfg.get("columns") or {}
    batch_cfg = cfg.get("batch") or {}
    eval_cfg = cfg.get("evaluator") or {}
    input_col = input_column or cols.get("input_text", "input_text")
    content_col = content_column or cols.get("final_content", "final_content")
    workers = max_workers or batch_cfg.get("max_workers", 5)
    eval_prompt = evaluation_prompt or get_evaluation_prompt_template()

    if not eval_prompt.strip():
        raise ValueError(
            "Evaluation prompt is empty. Set it in configs/prompts.yaml or pass evaluation_prompt."
        )

    print(f"  Loading CSV: {input_csv}")
    df = load_csv_with_encoding(input_csv)
    ok, err = validate_csv_columns(df, [input_col, content_col])
    if not ok:
        raise ValueError(err)

    provider = eval_cfg.get("api_provider", cfg.get("api", {}).get("default_provider", "deepseek"))
    print(f"  Initializing {str(provider).upper()} evaluator client...")
    try:
        llm_client = LLMClient(provider=provider)
    except Exception as e:
        raise RuntimeError(f"Failed to initialize evaluator client: {e}") from e

    for col in EVAL_COLUMNS:
        if col not in df.columns:
            df[col] = None

    total = len(df)
    print(f"  Total rows: {total}")
    results: Dict[int, Optional[Dict[str, Any]]] = {}
    errors: Dict[int, str] = {}

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _process_single_row,
                idx,
                row,
                llm_client,
                input_col,
                content_col,
                eval_prompt,
            ): idx
            for idx, row in df.iterrows()
        }
        with tqdm(total=total, desc="  Evaluating") as pbar:
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    row_idx, eval_result, error = future.result()
                    if error:
                        errors[row_idx] = error
                        results[row_idx] = None
                    else:
                        results[row_idx] = eval_result
                except Exception as e:
                    errors[idx] = str(e)
                    results[idx] = None
                finally:
                    pbar.update(1)

    for idx, eval_result in results.items():
        if eval_result:
            df.at[idx, "eval_priority"] = eval_result.get("priority")
            df.at[idx, "factuality_score"] = eval_result.get("factuality_score")
            df.at[idx, "north_star_score"] = eval_result.get("north_star_score")
            df.at[idx, "completeness_score"] = eval_result.get("completeness_score")
            df.at[idx, "adherence_score"] = eval_result.get("adherence_score")
            df.at[idx, "attractiveness_score"] = eval_result.get("attractiveness_score")
            df.at[idx, "weighted_total_score"] = eval_result.get("weighted_total_score")
            df.at[idx, "decision"] = eval_result.get("decision")
            df.at[idx, "reason"] = eval_result.get("reason")
            df.at[idx, "reasoning"] = eval_result.get("reasoning")
            df.at[idx, "pass"] = eval_result.get("pass")
        else:
            df.at[idx, "decision"] = "ERROR"
            df.at[idx, "reason"] = f"error: {errors.get(idx, 'Unknown error')}"

    df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    success = sum(1 for v in results.values() if v is not None)
    print(format_section_header("Evaluation summary", width=50))
    print(f"  Success: {success}  |  Errors: {total - success}")
    if errors:
        print("  Error details:")
        for i, msg in list(errors.items())[:10]:
            print(f"    Row {i}: {msg}")
        if len(errors) > 10:
            print(f"    ... and {len(errors) - 10} more.")

    if "decision" in df.columns:
        decision_counts = df["decision"].value_counts()
        print("\n  Decision distribution:")
        for decision, count in decision_counts.items():
            pct = (count / total) * 100
            print(f"    {decision:12s}: {count:4d}  ({pct:5.1f}%)")

    if "weighted_total_score" in df.columns:
        valid = df[df["weighted_total_score"].notna()]["weighted_total_score"]
        if len(valid) > 0:
            try:
                from prism_eval.utils.config_loader import get_evaluation_thresholds
                th = get_evaluation_thresholds()
                publish_th = th["publish_threshold"]
                medium_min = th["medium_min"]
            except Exception:
                publish_th = 75
                medium_min = 60
            print("\n  Weighted total score (0–100):")
            print(format_metric_line("Mean", f"{valid.mean():.2f}"))
            print(format_metric_line("Max", f"{valid.max():.2f}"))
            print(format_metric_line("Min", f"{valid.min():.2f}"))
            high = len(valid[valid >= publish_th])
            mid = len(valid[(valid >= medium_min) & (valid < publish_th)])
            low = len(valid[valid < medium_min])
            n = len(valid)
            print(f"    High (≥{publish_th}): {high} ({high/n*100:.1f}%)  |  Medium ({medium_min}–{publish_th-1}): {mid}  |  Low (<{medium_min}): {low}")

    print(f"\n  Output columns: {', '.join(EVAL_COLUMNS)}")
    print("  " + "=" * 48)
