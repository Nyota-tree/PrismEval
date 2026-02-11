"""
Evaluator prompt generator: given scenario and north-star metric, call LLM to produce
an evaluation prompt that can be used in batch_evaluate or the Streamlit app.
"""

from typing import Optional

from prism_eval.providers import LLMClient
from prism_eval.providers.llm_client import PROVIDER_ENV_KEYS


# System prompt used to generate the evaluator prompt (English).
PROMPT_GENERATOR_SYSTEM_PROMPT = """# Role
You are an expert in LLM evaluation (evals). You turn abstract business goals (North Star metrics) into clear, executable **Evaluator Prompts**.

# Task
The user will provide:
1. **Application scenario**: What is being evaluated (e.g. social copy, code, customer support replies).
2. **North Star metric**: The core value they care about (e.g. humor, code safety, empathy).

Generate a single, structured System Prompt that configures an AI evaluator. The evaluator will score model outputs using this prompt.

# Constraints (must follow)
1. **Role**: Define the evaluator's role (e.g. senior editor, security auditor).
2. **Inputs**: You MUST include a section where the actual inputs are inserted. Use exactly these two placeholders (they will be replaced by the system per sample): `{original_text}` for the question/input, and `{model_output}` for the model's answer. For example:
   <original_input>
   {original_text}
   </original_input>
   <model_output>
   {model_output}
   </model_output>
   Do NOT use only descriptive text like "the user will provide..."; the prompt must contain the literal placeholders `{original_text}` and `{model_output}` so the system can fill them.
3. **Scoring criteria**:
   - **[Factuality/Safety]** (weight 30–40%): Hallucination and safety. JSON key MUST be `factuality_safety_score`.
   - **[North Star]** (weight 30–40%): Describe exactly the user's North Star; use a SINGLE dimension. JSON key MUST be `north_star_score`. Do NOT split into multiple keys (e.g. no fun_score_xxx, attractiveness_xxx).
   - **[Completeness/Coherence]** (weight ~20%): Basic quality. JSON key MUST be `completeness_coherence_score`.
   - Total weight must sum to 100%.
4. **Scoring scale**: Define score bands (e.g. 90–100 excellent, 80–89 good, …) and instruct the model to score in fine-grained steps (e.g. 87, 82) and avoid clustering.
5. **Output format**: JSON only. Required keys: `determined_priority`, `scores`, `weighted_total_score`, `reasoning`, `pass`. The `scores` object MUST have exactly: `factuality_safety_score`, `north_star_score`, `completeness_coherence_score`. Decision: ≥75 publish, <75 review, factuality <50 reject.

# Prohibited
- Do not use any key other than `north_star_score` for the North Star dimension.
- Do not omit the score-band definitions or the instruction to differentiate scores.

# Output
Output only the full Evaluator Prompt, with no extra commentary. It must be directly usable to configure the evaluator."""


# System prompt for generating the *business* (generation) prompt from scenario + north star.
GENERATION_PROMPT_GENERATOR_SYSTEM = """# Role
You are a world-class prompt engineer. You turn business requirements into production-ready LLM prompts and align them with success metrics.

# Task
Given the user's **business scenario** and **North Star metric**, write a single, high-quality "business prompt" that will be used as the system prompt for an LLM in production. Output only the prompt text, no extra commentary.

# Input
1. Business scenario: What the AI product does and who uses it.
2. North Star metric: The core criterion for good performance.

# Output framework (your prompt must include)
1. **Role**: A clear, professional AI role for the scenario.
2. **Task**: Unambiguous description of what the AI must do.
3. **Constraints**: Hard requirements derived from the North Star (e.g. no hallucination, length, tone, compliance).
4. **Chain of thought** (if useful): Ask the model to reason before answering when the task is complex.
5. **Output format**: Structure of the reply (e.g. Markdown, JSON, tone).

# Design principles
- Align with the North Star (e.g. professionalism → citations and terminology; empathy → tone and empathy).
- Keep the prompt modular and easy to read.
- Add guardrails for edge cases and misuse.

# Language
Output the business prompt in the same language as the user's scenario (e.g. Chinese if they wrote in Chinese)."""


def generate_evaluator_prompt(
    scenario: str,
    north_star_metric: str,
    *,
    api_key: Optional[str] = None,
    provider: Optional[str] = None,
    model: Optional[str] = None,
) -> str:
    """
    Generate an evaluator prompt from scenario and north-star metric.

    Args:
        scenario: Description of the application scenario.
        north_star_metric: The North Star metric to optimize for.
        api_key: Optional API key (if not set, uses environment).
        provider: Optional LLM provider (default from config).
        model: Optional model name (default from config).

    Returns:
        The generated evaluator prompt string.
    """
    import os
    prov = (provider or "deepseek").lower()
    env_key = PROVIDER_ENV_KEYS.get(prov, "DEEPSEEK_API_KEY")
    prev = None
    if api_key is not None:
        prev = os.environ.get(env_key)
        os.environ[env_key] = api_key
    try:
        client = LLMClient(provider=prov, model=model)
        user_prompt = f"""Scenario: {scenario}
North Star metric: {north_star_metric}

Generate the full Evaluator Prompt based on the above."""
        return client.generate(system_prompt=PROMPT_GENERATOR_SYSTEM_PROMPT, user_prompt=user_prompt)
    finally:
        if prev is not None:
            os.environ[env_key] = prev
        elif api_key is not None:
            os.environ.pop(env_key, None)
