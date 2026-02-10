"""
Unified LLM client supporting OpenAI, Anthropic, DeepSeek, and Gemini.

Uses configuration from configs/default.yaml (or legacy config.py) and
environment variables for API keys.
"""

import os
import time
from typing import Any, Optional

from dotenv import load_dotenv

load_dotenv()

# Env var name per provider (for callers to set before creating client)
PROVIDER_ENV_KEYS = {
    "openai": "OPENAI_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "gemini": "GOOGLE_API_KEY",
}

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    from anthropic import Anthropic
except ImportError:
    Anthropic = None

try:
    import google.generativeai as genai
    _HAS_GEMINI = True
except ImportError:
    genai = None
    _HAS_GEMINI = False

# Prefer new config; fallback to legacy config module
try:
    from prism_eval.utils.config_loader import get_config, get
    _USE_YAML_CONFIG = True
except Exception:
    _USE_YAML_CONFIG = False


def _get_api_config(provider: str) -> dict:
    """Resolve API settings from YAML or legacy config."""
    if _USE_YAML_CONFIG:
        cfg = get_config()
        api = cfg.get("api") or {}
        prov = (api.get(provider) or {}).copy()
        if provider == "deepseek" and prov.get("reasoner"):
            # For evaluator, caller can pass model; here we just have defaults
            pass
        return prov
    # Legacy config
    import config as legacy
    if provider == "openai":
        return {
            "model": getattr(legacy, "OPENAI_MODEL", "gpt-4o-mini"),
            "temperature": getattr(legacy, "OPENAI_TEMPERATURE", 0.7),
            "max_tokens": getattr(legacy, "OPENAI_MAX_TOKENS", 2000),
        }
    if provider == "anthropic":
        return {
            "model": getattr(legacy, "ANTHROPIC_MODEL", ""),
            "temperature": getattr(legacy, "ANTHROPIC_TEMPERATURE", 0.7),
            "max_tokens": getattr(legacy, "ANTHROPIC_MAX_TOKENS", 2000),
        }
    if provider == "deepseek":
        return {
            "base_url": getattr(legacy, "DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
            "model": getattr(legacy, "DEEPSEEK_MODEL", "deepseek-chat"),
            "temperature": getattr(legacy, "DEEPSEEK_TEMPERATURE", 0.7),
            "max_tokens": getattr(legacy, "DEEPSEEK_MAX_TOKENS", 4000),
            "reasoner": {
                "model": getattr(legacy, "DEEPSEEK_REASONER_MODEL", "deepseek-reasoner"),
                "temperature": getattr(legacy, "DEEPSEEK_REASONER_TEMPERATURE", 0.0),
                "top_p": getattr(legacy, "DEEPSEEK_REASONER_TOP_P", 1.0),
                "max_tokens": getattr(legacy, "DEEPSEEK_REASONER_MAX_TOKENS", 2000),
            },
        }
    if provider == "gemini":
        return {
            "model": getattr(legacy, "GEMINI_MODEL", "gemini-1.5-flash"),
            "temperature": getattr(legacy, "GEMINI_TEMPERATURE", 0.7),
            "max_tokens": getattr(legacy, "GEMINI_MAX_TOKENS", 2000),
        }
    return {}


def _get_batch_config() -> dict:
    """Resolve batch settings (max_retries, retry_delay)."""
    if _USE_YAML_CONFIG:
        batch = get_config().get("batch") or {}
        return {
            "max_retries": batch.get("max_retries", 3),
            "retry_delay": batch.get("retry_delay_seconds", 1),
        }
    import config as legacy
    return {
        "max_retries": getattr(legacy, "MAX_RETRIES", 3),
        "retry_delay": getattr(legacy, "RETRY_DELAY", 1),
    }


class LLMClient:
    """
    Unified LLM client for OpenAI, Anthropic, and DeepSeek.

    API keys are read from environment variables (e.g. OPENAI_API_KEY, DEEPSEEK_API_KEY).
    """

    def __init__(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> None:
        """
        Initialize the LLM client.

        Args:
            provider: One of "openai", "anthropic", "deepseek", "gemini". If None, uses
                evaluator provider from config, then default provider.
            model: Model name. If None, uses config default for the provider.
            temperature: Sampling temperature. If None, uses config default.
            max_tokens: Maximum tokens to generate. If None, uses config default.
            top_p: Top-p sampling (OpenAI/DeepSeek only). If None, uses config default.
        """
        if provider is not None:
            self._provider = provider.lower()
        else:
            if _USE_YAML_CONFIG:
                eval_cfg = get_config().get("evaluator") or {}
                self._provider = (eval_cfg.get("api_provider") or get("api.default_provider") or "deepseek").lower()
            else:
                import config as legacy
                self._provider = (
                    getattr(legacy, "EVALUATOR_API_PROVIDER", None) or getattr(legacy, "API_PROVIDER", "deepseek")
                ).lower()

        if self._provider == "openai":
            if OpenAI is None:
                raise ImportError("Package 'openai' is not installed. Run: pip install openai")
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not set. Set it in .env or environment.")
            self._client: Any = OpenAI(api_key=api_key)
            cfg = _get_api_config("openai")
            self._model = model or cfg.get("model", "gpt-4o-mini")
            self._temperature = temperature if temperature is not None else cfg.get("temperature", 0.7)
            self._max_tokens = max_tokens or cfg.get("max_tokens", 2000)
            self._top_p = top_p

        elif self._provider == "anthropic":
            if Anthropic is None:
                raise ImportError("Package 'anthropic' is not installed. Run: pip install anthropic")
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY not set. Set it in .env or environment.")
            self._client = Anthropic(api_key=api_key)
            cfg = _get_api_config("anthropic")
            self._model = model or cfg.get("model", "")
            self._temperature = temperature if temperature is not None else cfg.get("temperature", 0.7)
            self._max_tokens = max_tokens or cfg.get("max_tokens", 2000)
            self._top_p = None

        elif self._provider == "deepseek":
            if OpenAI is None:
                raise ImportError("Package 'openai' is not installed. Run: pip install openai")
            api_key = os.getenv("DEEPSEEK_API_KEY")
            if not api_key:
                raise ValueError("DEEPSEEK_API_KEY not set. Set it in .env or environment.")
            cfg = _get_api_config("deepseek")
            base_url = cfg.get("base_url", "https://api.deepseek.com/v1")
            self._client = OpenAI(api_key=api_key, base_url=base_url)
            if model is not None:
                self._model = model
            else:
                self._model = cfg.get("model", "deepseek-chat")
            if temperature is not None:
                self._temperature = temperature
            else:
                self._temperature = cfg.get("temperature", 0.7)
            if max_tokens is not None:
                self._max_tokens = max_tokens
            else:
                self._max_tokens = cfg.get("max_tokens", 4000)
            self._top_p = top_p if top_p is not None else cfg.get("reasoner", {}).get("top_p")

        elif self._provider == "gemini":
            if not _HAS_GEMINI:
                raise ImportError(
                    "Package 'google-generativeai' is not installed. Run: pip install google-generativeai"
                )
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("GOOGLE_API_KEY not set. Set it in .env or environment.")
            genai.configure(api_key=api_key)
            cfg = _get_api_config("gemini")
            self._model = model or cfg.get("model", "gemini-1.5-flash")
            self._temperature = temperature if temperature is not None else cfg.get("temperature", 0.7)
            self._max_tokens = max_tokens or cfg.get("max_tokens", 2000)
            self._top_p = None
            self._client = genai.GenerativeModel(self._model)

        else:
            raise ValueError(
                f"Unsupported API provider: {self._provider}. "
                "Use 'openai', 'anthropic', 'deepseek', or 'gemini'."
            )

    def _call_api(self, system_prompt: str, user_prompt: str) -> str:
        """Perform a single API call (no retry)."""
        if self._provider in ("openai", "deepseek"):
            params: dict = {
                "model": self._model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": self._temperature,
                "max_tokens": self._max_tokens,
            }
            if self._top_p is not None:
                params["top_p"] = self._top_p
            response = self._client.chat.completions.create(**params)
            return response.choices[0].message.content.strip()
        elif self._provider == "anthropic":
            response = self._client.messages.create(
                model=self._model,
                max_tokens=self._max_tokens,
                temperature=self._temperature,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )
            return response.content[0].text.strip()
        elif self._provider == "gemini":
            combined = f"{system_prompt}\n\n{user_prompt}" if system_prompt else user_prompt
            response = self._client.generate_content(
                combined,
                generation_config=genai.GenerationConfig(
                    temperature=self._temperature,
                    max_output_tokens=self._max_tokens,
                ),
            )
            if not response.text:
                raise RuntimeError("Gemini returned empty response")
            return response.text.strip()
        else:
            raise RuntimeError(f"Unknown provider: {self._provider}")

    def _call_api_with_retry(self, system_prompt: str, user_prompt: str) -> str:
        """Call the API with exponential backoff retry."""
        batch = _get_batch_config()
        max_retries = batch.get("max_retries", 3)
        retry_delay = batch.get("retry_delay", 1)
        last_error: Optional[Exception] = None
        for attempt in range(max_retries):
            try:
                return self._call_api(system_prompt, user_prompt)
            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    wait = min(retry_delay * (2 ** attempt), 10)
                    time.sleep(wait)
        raise RuntimeError(
            f"API call failed after {max_retries} retries: {last_error}"
        ) from last_error

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """
        Generate a completion (with retries).

        Args:
            system_prompt: System message content.
            user_prompt: User message content.

        Returns:
            The model's response text.
        """
        return self._call_api_with_retry(system_prompt, user_prompt)
