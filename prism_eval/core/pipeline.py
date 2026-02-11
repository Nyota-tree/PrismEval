"""
Shared pipeline helpers: single-item generation, single-item evaluation,
and evaluation prompt placeholder substitution.
"""

from typing import Any, Dict, Optional, Tuple

import pandas as pd

from prism_eval.metrics.extractor import extract_evaluation, extract_json_from_text
from prism_eval.providers import LLMClient
from prism_eval.providers.llm_client import PROVIDER_ENV_KEYS


def fill_evaluation_prompt(
    prompt_template: str,
    original_text: str,
    model_output: str,
) -> str:
    """
    Replace {original_text} and {model_output} placeholders in the template.

    Also replaces {original_input} (common in LLM-generated prompts) with original_text.
    Uses str.replace to avoid JSON/markdown curly braces being interpreted by .format().

    Args:
        prompt_template: Template containing {original_text} and {model_output}.
        original_text: Input/question text.
        model_output: Model-generated answer to evaluate.

    Returns:
        Filled prompt string.
    """
    out = prompt_template.replace("{original_text}", original_text).replace("{model_output}", model_output)
    # Support {original_input} as synonym (some generated prompts use this)
    out = out.replace("{original_input}", original_text)
    return out


def run_single_generation(
    question: str,
    system_prompt: str,
    api_key: str,
    model: str,
    provider: str = "deepseek",
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Generate a single answer for one question using the given system prompt.

    Args:
        question: User question or input text.
        system_prompt: System prompt for the generator.
        api_key: API key (will be set in env for the call).
        model: Model name.
        provider: LLM provider (e.g. deepseek).
        temperature: Override sampling temperature.
        max_tokens: Override max tokens.

    Returns:
        (answer_text, None) on success, or (None, error_message) on failure.
    """
    if not (question or "").strip():
        return None, "Question is empty."
    if not (system_prompt or "").strip():
        return None, "Generation prompt is empty."

    import os
    env_key = PROVIDER_ENV_KEYS.get((provider or "deepseek").lower(), "DEEPSEEK_API_KEY")
    prev = os.environ.get(env_key)
    try:
        os.environ[env_key] = api_key
        client = LLMClient(
            provider=provider,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        answer = client.generate(
            system_prompt=system_prompt.strip(),
            user_prompt=(question or "").strip(),
        )
        return (answer or "").strip(), None
    except Exception as e:
        return None, str(e)
    finally:
        if prev is not None:
            os.environ[env_key] = prev
        else:
            os.environ.pop(env_key, None)


def run_single_evaluation(
    row: pd.Series,
    evaluation_prompt: str,
    api_key: str,
    model: str,
    question_column: str = "question",
    generated_answer_column: str = "generated_answer",
    expected_answer_column: str = "expected_answer",
    provider: str = "deepseek",
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Run evaluation for a single row: fill prompt, call LLM, parse JSON, extract scores.

    Args:
        row: DataFrame row with question and answer columns.
        evaluation_prompt: Template with {original_text} and {model_output}.
        api_key: API key for the evaluator.
        model: Model name.
        question_column: Column name for the original question.
        generated_answer_column: Column for model-generated answer.
        expected_answer_column: Fallback column for reference answer.
        provider: LLM provider.

    Returns:
        (evaluation_result dict, None) on success, or (None, error_message) on failure.
    """
    original_text = str(row.get(question_column, ""))
    model_output = str(
        row.get(generated_answer_column) or row.get(expected_answer_column) or ""
    ).strip()

    if not original_text.strip():
        return None, "Question is empty; row skipped."
    if not model_output or model_output.lower() in ("nan", ""):
        return None, "No answer content (add generated_answer or expected_answer); row skipped."

    prompt_filled = fill_evaluation_prompt(evaluation_prompt, original_text, model_output)
    if "{original_text}" in prompt_filled or "{model_output}" in prompt_filled or "{original_input}" in prompt_filled:
        return None, (
            "Evaluation prompt template is missing placeholder substitution: "
            "ensure it contains {original_text} and {model_output} (or {original_input}) so the system can fill question and answer."
        )

    import os
    try:
        from prism_eval.utils.config_loader import get_evaluator_model_params
        eval_params = get_evaluator_model_params()
        eval_temperature = eval_params.get("temperature", 0.0)
        eval_max_tokens = eval_params.get("max_tokens", 2000)
    except Exception:
        eval_temperature = 0.0
        eval_max_tokens = 2000
    env_key = PROVIDER_ENV_KEYS.get((provider or "deepseek").lower(), "DEEPSEEK_API_KEY")
    prev_key = os.environ.get(env_key)
    try:
        os.environ[env_key] = api_key
        client = LLMClient(
            provider=provider,
            model=model,
            temperature=eval_temperature,
            max_tokens=eval_max_tokens,
        )
        response = client.generate(system_prompt="", user_prompt=prompt_filled)
    except Exception as e:
        return None, str(e)
    finally:
        if prev_key is not None:
            os.environ[env_key] = prev_key
        else:
            os.environ.pop(env_key, None)

    json_obj = extract_json_from_text(response)
    if json_obj is None:
        return None, f"Could not extract JSON from response: {response[:200]}…"

    try:
        evaluation_result = extract_evaluation(json_obj)
        return evaluation_result, None
    except Exception as e:
        return None, str(e)
