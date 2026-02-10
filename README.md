# 💎 PrismEval

**Refracting complex LLM outputs into a spectrum of actionable, high-precision metrics for production-grade AI agents.**

Stop the "vibe checks." Start engineering trust.

---

## 🌪 The Problem: The "Black Box" of LLM Reliability

In the journey from prompt engineering to production, teams often hit an **evaluation gap**:

- **Vibe-based iteration** — "It feels better" is not a metric.
- **Opaque failures** — Knowing a model failed is easy; knowing *why* (faithfulness vs. relevance vs. logic) is hard.
- **The cost of hallucinations** — Manual review doesn't scale; un-gated deployments are a liability.

---

## 🌈 The Solution: PrismEval

PrismEval acts as a **prism** for your LLM outputs: it splits a single response into a **multi-dimensional spectrum** of quantitative metrics, so you can automate decisions and optimize prompts with precision.

### Core "Wavelengths" (Metrics)

| Metric | What it measures |
|--------|-------------------|
| **Faithfulness** | Detects hallucinations by cross-referencing source context. |
| **Instruction Adherence** | How strictly the model followed system prompts. |
| **Contextual Relevance** | Whether the output directly addresses user intent. |
| **Interaction Fluidity** | Naturalness of the agent's persona (UX-inspired). |

---

## 🛠 Features

### 1. Automated Decision Gating

PrismEval doesn’t just score — it **decides**. Based on your thresholds, it drives a tri-state flow:

- ✅ **PUBLISH** — Meets high-quality benchmarks.
- ⚠️ **REVIEW** — Potential issues; flagged for human-in-the-loop (HITL) audit.
- ❌ **REJECT** — Critical failure (e.g. factuality below threshold).

### 2. North-Star Metric Customization

Using `generate_evaluator_prompt` (or the Streamlit app), you define a **North Star** (e.g. "Extreme Empathy", "Strict JSON Schema"). The system then recalibrates evaluation weights automatically.

### 3. Developer-Centric DX

- **Modular architecture** — Decoupled metrics, providers, and orchestration.
- **Self-documenting CLI** — Clear, structured terminal output for debugging.
- **Production-ready** — Fits into CI/CD and batch pipelines.

---

## 🚀 Quick Start

```bash
# Clone the repo
git clone https://github.com/Nyota-tree/PrismEval.git
cd PrismEval

# Install dependencies
pip install -r requirements.txt

# Configure API keys (copy template and fill in)
cp .env.example .env

# Option A: Streamlit UI (recommended)
streamlit run app.py

# Option B: CLI — batch generate then evaluate
python main.py generate examples/input_example.csv output.csv
python main.py evaluate output.csv results.csv

# Option C: Generate an evaluator prompt from scenario + North Star
python main.py gen-prompt "Customer support replies" "Empathy and clarity" --output eval_prompt.txt
```

---

## 🏗 Architecture

```
prism_eval/
├── core/         # Orchestration, batch generate/evaluate, prompt generator
├── metrics/      # Score extraction (faithfulness, north_star, completeness, etc.)
├── providers/    # LLM connectors (OpenAI, Anthropic, DeepSeek)
└── utils/        # Config loader, I/O, logging
```

Configuration: **`configs/config.yaml`** (model params, evaluation thresholds).  
Prompts: **`configs/prompts.yaml`** or **`config.py`** (legacy).

---

## 🔮 Roadmap

- [ ] **Visual spectrum dashboard** — UI to track evaluation drift over time.
- [ ] **Cross-model benchmarking** — Compare GPT-4o, Claude, DeepSeek side-by-side.
- [ ] **Real-time feedback loop** — Auto-refine prompts from failed evaluation scores.

---

## 📄 License

MIT. See [LICENSE](LICENSE).

---

## 📝 Author

**nyota佳树** — AI Product Manager, interaction design background.  
Bridging human-centric design and machine-driven infrastructure.
