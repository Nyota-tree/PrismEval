"""
Load configuration from YAML files and environment variables.

Provides a single source of truth for API, batch, column, and prompt settings.
"""

from pathlib import Path
from typing import Any, Optional

# Optional YAML support
try:
    import yaml
    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False

# Default paths relative to project root (parent of prism_eval)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_CONFIG_PATH = _PROJECT_ROOT / "configs" / "config.yaml"
_DEFAULT_CONFIG_PATH = _PROJECT_ROOT / "configs" / "default.yaml"
_PROMPTS_CONFIG_PATH = _PROJECT_ROOT / "configs" / "prompts.yaml"

_cached_config: Optional[dict] = None
_cached_prompts: Optional[dict] = None

# Default evaluation thresholds (used when config has no evaluation section)
_DEFAULT_EVALUATION = {
    "factuality_threshold_0_10": 5,
    "factuality_threshold_0_100": 50,
    "publish_threshold": 75,
    "medium_min": 60,
}


def _load_yaml(path: Path) -> dict:
    """Load a YAML file and return its contents as a dict."""
    if not path.exists():
        return {}
    if not _HAS_YAML:
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def get_config() -> dict:
    """
    Load and return the main configuration.
    Prefers configs/config.yaml; falls back to configs/default.yaml.

    Returns:
        A nested dict with keys: api, batch, columns, evaluator, eval_output, evaluation.
    """
    global _cached_config
    if _cached_config is not None:
        return _cached_config
    _cached_config = _load_yaml(_CONFIG_PATH)
    if not _cached_config:
        _cached_config = _load_yaml(_DEFAULT_CONFIG_PATH)
    return _cached_config


def get_prompts_config() -> dict:
    """Load and return the prompts configuration (generation and evaluation templates)."""
    global _cached_prompts
    if _cached_prompts is not None:
        return _cached_prompts
    _cached_prompts = _load_yaml(_PROMPTS_CONFIG_PATH)
    return _cached_prompts


def _nested_get(data: dict, key_path: str, default: Any = None) -> Any:
    """Get a value by dot-separated key path (e.g. 'api.deepseek.model')."""
    keys = key_path.split(".")
    current = data
    for k in keys:
        current = current.get(k) if isinstance(current, dict) else None
        if current is None:
            return default
    return current


def get(key_path: str, default: Any = None) -> Any:
    """
    Get a configuration value by dot-separated path.

    Args:
        key_path: e.g. "api.default_provider", "batch.max_workers"
        default: Value to return if key is missing.

    Returns:
        The configuration value.
    """
    cfg = get_config()
    return _nested_get(cfg, key_path, default)


def get_system_prompt() -> str:
    """Return the default generation system prompt."""
    prompts = get_prompts_config()
    gen = prompts.get("generation") or {}
    return gen.get("system", "").strip() or "{input_text}"


def get_user_prompt_template() -> str:
    """Return the default user prompt template for generation (uses {input_text})."""
    prompts = get_prompts_config()
    gen = prompts.get("generation") or {}
    return gen.get("user_template", "{input_text}").strip()


def get_evaluation_prompt_template() -> str:
    """Return the default evaluation prompt template (uses {original_text}, {model_output})."""
    prompts = get_prompts_config()
    return (prompts.get("evaluation") or "").strip()


def get_evaluation_thresholds() -> dict:
    """
    Return evaluation decision thresholds from config.

    Returns:
        Dict with factuality_threshold_0_10, factuality_threshold_0_100,
        publish_threshold, medium_min. Missing keys use defaults.
    """
    cfg = get_config()
    ev = cfg.get("evaluation") or {}
    return {
        "factuality_threshold_0_10": ev.get("factuality_threshold_0_10", _DEFAULT_EVALUATION["factuality_threshold_0_10"]),
        "factuality_threshold_0_100": ev.get("factuality_threshold_0_100", _DEFAULT_EVALUATION["factuality_threshold_0_100"]),
        "publish_threshold": ev.get("publish_threshold", _DEFAULT_EVALUATION["publish_threshold"]),
        "medium_min": ev.get("medium_min", _DEFAULT_EVALUATION["medium_min"]),
    }


def get_evaluator_model_params() -> dict:
    """
    Return evaluator model parameters (temperature, max_tokens) from config.

    Returns:
        Dict with temperature, max_tokens. Evaluator section overrides api.reasoner defaults.
    """
    cfg = get_config()
    ev = cfg.get("evaluator") or {}
    api = cfg.get("api") or {}
    deepseek = api.get("deepseek") or {}
    reasoner = deepseek.get("reasoner") or {}
    return {
        "temperature": ev.get("temperature", reasoner.get("temperature", 0.0)),
        "max_tokens": ev.get("max_tokens", reasoner.get("max_tokens", 2000)),
    }


def reset_cache() -> None:
    """Clear cached config (useful for tests)."""
    global _cached_config, _cached_prompts
    _cached_config = None
    _cached_prompts = None
