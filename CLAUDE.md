# CLAUDE.md — PrismEval

## Project Overview

**PrismEval** is a lightweight LLM evaluation pipeline that uses LLM-as-a-Judge to score AI-generated outputs on multiple dimensions and make automated PUBLISH / REVIEW / REJECT decisions.

Three AI roles in the pipeline:
1. **Prompt Architect** — converts a scenario + north-star metric into structured business and evaluation prompts
2. **Content Generator** — batch-generates LLM responses from a CSV input (concurrent, with auto-retry)
3. **Quality Judge** — scores each output on factuality/safety, north-star, and completeness; returns structured JSON

## Tech Stack

- **Python 3.10+**
- **Streamlit** — web UI (`app.py`)
- **OpenAI SDK** — unified LLM client for all providers (OpenAI, DeepSeek, Anthropic, Gemini)
- **pandas** — CSV data processing
- **PyYAML** — configuration
- **Plotly** — evaluation result charts
- **tqdm** — progress bars
- **pytest** — test suite
- **ThreadPoolExecutor** — concurrent batch processing (default 5 workers)

## Project Structure

```
PrismEval/
├── app.py                        # Streamlit web UI (6-step wizard)
├── main.py                       # CLI entry point
├── batch_generator.py            # CLI wrapper for batch generation
├── batch_evaluator.py            # CLI wrapper for batch evaluation
├── generate_evaluator_prompt.py  # Standalone prompt generator
│
├── prism_eval/                   # Core package
│   ├── core/
│   │   ├── batch_generate.py     # Batch generation from CSV
│   │   ├── batch_evaluate.py     # Batch evaluation from CSV
│   │   ├── prompt_generator.py   # Generate eval/business prompts
│   │   └── pipeline.py           # Single-item generation/evaluation
│   ├── metrics/
│   │   └── extractor.py          # JSON extraction and score normalization
│   ├── providers/
│   │   └── llm_client.py         # Unified LLM client abstraction
│   └── utils/
│       ├── config_loader.py      # YAML config loading (cached)
│       ├── io.py                 # CSV loading with encoding detection
│       ├── logging.py            # CLI formatting helpers (NOT a logger)
│       └── i18n.py               # Bilingual UI copy (EN/ZH)
│
├── configs/
│   ├── default.yaml              # API, batch, column, evaluator defaults
│   └── prompts.yaml              # Prompt templates
│
├── examples/
│   ├── input_example.csv
│   ├── run_batch_generate.py
│   └── run_gen_prompt.py
│
└── tests/
    ├── test_utils.py
    └── test_metrics.py
```

## Common Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Install with dev dependencies (for testing)
pip install ".[dev]"

# Run the web UI
streamlit run app.py
# or
python main.py app

# CLI: batch generate responses from CSV
python main.py generate examples/input_example.csv output.csv

# CLI: batch evaluate generated responses
python main.py evaluate output.csv results.csv

# CLI: generate evaluator prompt from a scenario description
python main.py gen-prompt "Customer support replies" "Empathy and clarity" --output eval_prompt.txt

# Run tests
pytest tests/ -v
```

## Configuration

Copy `.env.example` to `.env` and add your API keys:

```
DEEPSEEK_API_KEY=...
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
GOOGLE_API_KEY=...
```

Key settings in `configs/default.yaml`:
- `api.default_provider` — default LLM provider (`deepseek`, `openai`, `anthropic`, `gemini`)
- `batch.max_workers` — thread pool concurrency (default: 5)
- `batch.max_retries` — API retry attempts (default: 3)
- `evaluator.model` — model used for judging (default: `deepseek-reasoner`)

## Evaluation Output Schema

Every evaluation returns a JSON object:

```json
{
  "determined_priority": "P0",
  "scores": {
    "factuality_safety_score": 9,
    "north_star_score": 8,
    "completeness_coherence_score": 9
  },
  "weighted_total_score": 87,
  "decision": "PUBLISH",
  "reasoning": "...",
  "pass": true
}
```

Decision rules:
- **PUBLISH** — weighted score ≥ 75
- **REVIEW** — weighted score < 75
- **REJECT** — factuality score < 50 (0–100 scale) or < 5 (0–10 scale)

## Coding Conventions

- **Type hints** on all function signatures (`Optional[str]`, `Dict[str, Any]`, etc.)
- **Config via YAML**, not hardcoded values; environment variables override YAML
- **Error handling**: try/catch with informative messages; row-level error tracking in batch ops
- **JSON extraction** uses multiple fallback strategies: direct parse → markdown block → balanced braces → regex
- **Score normalization**: extractor handles both 0–10 and 0–100 scales automatically
- `prism_eval/utils/logging.py` contains **CLI formatting helpers** (`format_section_header`, `format_metric_line`), not a standard logger — use Python's `logging` module for actual error logging
- Tests live in `tests/` and are run via GitHub Actions CI on Python 3.10 / 3.11 / 3.12

## Architecture Notes

- `providers/llm_client.py` is the **single abstraction** for all LLM calls; each pipeline role (generation, evaluation, prompt-gen) can use a different provider and model
- Config is **cached** in `config_loader.py` via module-level globals (`_cached_config`, `_cached_prompts`)
- CSV encoding is auto-detected (UTF-8, GBK, GB2312, Latin-1)
- The Streamlit UI (`app.py`) shares core logic with the CLI via the same `prism_eval` package — do not duplicate business logic in the UI layer
