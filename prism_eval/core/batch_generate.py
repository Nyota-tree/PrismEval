"""
Batch generation: read CSV, call LLM for each row, write results to a new CSV.
"""

import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Optional, Tuple

import pandas as pd
from tqdm import tqdm

from prism_eval.providers import LLMClient
from prism_eval.utils.config_loader import (
    get_config,
    get_system_prompt,
    get_user_prompt_template,
)
from prism_eval.utils.io import load_csv_with_encoding, validate_csv_columns
from prism_eval.utils.logging import format_section_header


def extract_rewrite_result(llm_output: str) -> Tuple[str, str]:
    """
    Extract priority and final content from LLM output (XML-style tags).

    Expects <priority>...</priority> and <content>...</content> or <final_result>...</final_result>.

    Args:
        llm_output: Raw model response.

    Returns:
        (priority, final_content). Defaults to ("P4", "") if parsing fails.
    """
    if not llm_output or not isinstance(llm_output, str):
        return "P4", ""

    priority_regex = r"<priority>(.*?)</priority>"
    priority_match = re.search(priority_regex, llm_output, re.DOTALL)
    priority = "P4"
    if priority_match and priority_match.group(1):
        priority = priority_match.group(1).strip()

    content_regex = r"<content>([\s\S]*?)</content>"
    content_match = re.search(content_regex, llm_output, re.DOTALL)
    if content_match and content_match.group(1):
        final_content = content_match.group(1).strip()
        final_content = re.sub(r"<Title>|</Title>|<Brief>|</Brief>", "", final_content)
    else:
        final_content = re.sub(r"<thinking>[\s\S]*?</thinking>", "", llm_output)
        final_content = re.sub(r"<final_result>|</final_result>", "", final_content)
        final_content = final_content.strip()

    return priority, final_content


def _process_single_row(
    idx: int,
    row: pd.Series,
    llm_client: LLMClient,
    input_column: str,
    user_template: str,
    system_prompt: str,
) -> Tuple[int, Optional[str], str, str, Optional[str]]:
    """Process one row: call LLM, extract priority and content. Returns (idx, full_response, priority, final_content, error)."""
    try:
        input_text = str(row[input_column])
        user_prompt = user_template.format(input_text=input_text)
        full_response = llm_client.generate(system_prompt=system_prompt, user_prompt=user_prompt)
        priority, final_content = extract_rewrite_result(full_response)
        return (idx, full_response, priority, final_content, None)
    except Exception as e:
        return (idx, None, "P4", "", str(e))


def batch_generate(
    input_csv: str,
    output_csv: str,
    *,
    input_column: Optional[str] = None,
    output_column: Optional[str] = None,
    system_prompt: Optional[str] = None,
    user_prompt_template: Optional[str] = None,
    max_workers: Optional[int] = None,
) -> None:
    """
    Load CSV, generate model responses for each row, and save to output CSV.

    Uses config for column names and prompts if not overridden. Adds columns:
    model_response (or output_column), eval_priority, final_content.

    Args:
        input_csv: Path to input CSV.
        output_csv: Path to output CSV.
        input_column: Column name for input text (default from config).
        output_column: Column name for full model response (default from config).
        system_prompt: Override default system prompt.
        user_prompt_template: Override user prompt template (use {input_text}).
        max_workers: Concurrency (default from config).
    """
    cfg = get_config()
    cols = cfg.get("columns") or {}
    batch_cfg = cfg.get("batch") or {}
    input_col = input_column or cols.get("input_text", "input_text")
    out_col = output_column or cols.get("output_column", "model_response")
    workers = max_workers or batch_cfg.get("max_workers", 5)
    system_p = system_prompt or get_system_prompt()
    user_tpl = user_prompt_template or get_user_prompt_template()

    print(f"  Loading CSV: {input_csv}")
    df = load_csv_with_encoding(input_csv)
    ok, err = validate_csv_columns(df, [input_col])
    if not ok:
        raise ValueError(err)

    print(f"  Initializing {cfg.get('api', {}).get('default_provider', 'deepseek').upper()} client...")
    try:
        llm_client = LLMClient()
    except Exception as e:
        raise RuntimeError(f"Failed to initialize LLM client: {e}") from e

    if out_col not in df.columns:
        df[out_col] = None
    if "eval_priority" not in df.columns:
        df["eval_priority"] = None
    if "final_content" not in df.columns:
        df["final_content"] = None

    total = len(df)
    print(f"  Total rows: {total}")
    results: dict = {}
    errors: dict = {}

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _process_single_row,
                idx,
                row,
                llm_client,
                input_col,
                user_tpl,
                system_p,
            ): idx
            for idx, row in df.iterrows()
        }
        with tqdm(total=total, desc="  Generating") as pbar:
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    row_idx, full_resp, priority, final_content, error = future.result()
                    if error:
                        errors[row_idx] = error
                        results[row_idx] = ("error", "P4", "")
                    else:
                        results[row_idx] = (full_resp, priority, final_content)
                except Exception as e:
                    errors[idx] = str(e)
                    results[idx] = ("error", "P4", "")
                finally:
                    pbar.update(1)

    for idx, (full_response, priority, final_content) in results.items():
        if full_response == "error":
            df.at[idx, out_col] = f"error: {errors.get(idx, 'Unknown error')}"
            df.at[idx, "eval_priority"] = "P4"
            df.at[idx, "final_content"] = ""
        else:
            df.at[idx, out_col] = full_response
            df.at[idx, "eval_priority"] = priority
            df.at[idx, "final_content"] = final_content

    df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    success = sum(1 for v in results.values() if v[0] != "error")
    print(format_section_header("Batch generation complete", width=50))
    print(f"  Success: {success}  |  Errors: {total - success}")
    if errors:
        print("  Error details:")
        for i, msg in list(errors.items())[:10]:
            print(f"    Row {i}: {msg}")
        if len(errors) > 10:
            print(f"    ... and {len(errors) - 10} more.")
