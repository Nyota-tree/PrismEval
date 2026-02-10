"""
LLM Evaluation Pipeline – Streamlit app.

Six phases: Config → Business Prompt → Eval Prompt → Generate Answers → Evaluate → Results.
"""

import io
import os
import time
from typing import Any, Dict, Optional

import pandas as pd
import streamlit as st
import plotly.express as px

from prism_eval.providers import LLMClient
from prism_eval.metrics.extractor import extract_json_from_text, extract_evaluation
from prism_eval.core.pipeline import run_single_generation, run_single_evaluation, fill_evaluation_prompt
from prism_eval.core.prompt_generator import (
    PROMPT_GENERATOR_SYSTEM_PROMPT,
    GENERATION_PROMPT_GENERATOR_SYSTEM,
    generate_evaluator_prompt,
)
from prism_eval.utils.config_loader import get_config
from i18n import t


def generate_evaluator_prompt_in_app(scenario: str, north_star_metric: str, api_key: str) -> str:
    """Generate evaluator prompt in-app (no sys.exit; errors surface in UI)."""
    prev = os.environ.get("DEEPSEEK_API_KEY")
    try:
        os.environ["DEEPSEEK_API_KEY"] = api_key
        return generate_evaluator_prompt(scenario, north_star_metric, api_key=api_key)
    finally:
        if prev is not None:
            os.environ["DEEPSEEK_API_KEY"] = prev
        else:
            os.environ.pop("DEEPSEEK_API_KEY", None)


def generate_generation_prompt_in_app(scenario: str, north_star: str, api_key: str) -> str:
    """Generate business (generation) prompt from scenario and north-star metric."""
    user_prompt = f"""Business scenario:\n{scenario}\n\nNorth star metric:\n{north_star}\n\nGenerate a production-ready business prompt (output only the prompt text)."""
    prev = os.environ.get("DEEPSEEK_API_KEY")
    try:
        os.environ["DEEPSEEK_API_KEY"] = api_key
        llm_client = LLMClient(provider="deepseek", model=st.session_state.get("model", DEFAULT_MODEL))
        return llm_client.generate(
            system_prompt=GENERATION_PROMPT_GENERATOR_SYSTEM,
            user_prompt=user_prompt,
        )
    finally:
        if prev is not None:
            os.environ["DEEPSEEK_API_KEY"] = prev
        else:
            os.environ.pop("DEEPSEEK_API_KEY", None)


# Page config
st.set_page_config(
    page_title="LLM Eval Pipeline",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Constants
REQUIRED_CSV_COLUMNS = ["question"]
OPTIONAL_ANSWER_COLUMN = "expected_answer"
GENERATED_ANSWER_COLUMN = "generated_answer"
DEFAULT_MODEL = "deepseek-chat"
MODEL_OPTIONS = ["deepseek-chat", "deepseek-reasoner"]
PHASES = ["CONFIG", "GENERATION_PROMPT_EDIT", "GENERATING", "PROMPT_EDIT", "EVALUATING", "RESULT"]


def init_session_state() -> None:
    """Initialize Streamlit session state for pipeline phases and data."""
    if "phase" not in st.session_state:
        st.session_state.phase = "CONFIG"
    if "lang" not in st.session_state:
        st.session_state.lang = "en"
    if "api_key" not in st.session_state:
        st.session_state.api_key = ""
    if "model" not in st.session_state:
        st.session_state.model = DEFAULT_MODEL
    if "scenario" not in st.session_state:
        st.session_state.scenario = ""
    if "north_star" not in st.session_state:
        st.session_state.north_star = ""
    if "uploaded_df" not in st.session_state:
        st.session_state.uploaded_df = None
    if "generation_prompt" not in st.session_state:
        st.session_state.generation_prompt = ""
    if "generated_prompt" not in st.session_state:
        st.session_state.generated_prompt = ""
    if "evaluation_prompt" not in st.session_state:
        st.session_state.evaluation_prompt = ""
    if "results_df" not in st.session_state:
        st.session_state.results_df = None
    if "eval_elapsed" not in st.session_state:
        st.session_state.eval_elapsed = None


def get_csv_template_bytes() -> bytes:
    """Return a minimal CSV template (question column only) for the generate-answers flow."""
    template_df = pd.DataFrame({
        "question": [
            "Example question 1: Summarize the key compliance points.",
            "Example question 2: How should we reply to the customer in this scenario?",
        ],
    })
    buf = io.BytesIO()
    template_df.to_csv(buf, index=False, encoding="utf-8-sig")
    return buf.getvalue()


def validate_csv(df: pd.DataFrame) -> tuple[bool, str]:
    """Validate that the DataFrame has required columns and at least one row. Returns (ok, error_message)."""
    missing = [c for c in REQUIRED_CSV_COLUMNS if c not in df.columns]
    if missing:
        return False, f"CSV missing required columns: {', '.join(missing)}. Required: {', '.join(REQUIRED_CSV_COLUMNS)}"
    if df.empty:
        return False, "CSV is empty; please upload at least one row."
    return True, ""


# Sidebar
def render_sidebar():
    with st.sidebar:
        st.header("⚙️ " + t("sidebar_config"))
        st.divider()

        lang = st.selectbox(
            t("language"),
            options=["en", "zh"],
            format_func=lambda x: "English" if x == "en" else "中文",
            index=0 if st.session_state.get("lang", "en") == "en" else 1,
        )
        st.session_state.lang = lang

        api_key = st.text_input(
            t("api_key"),
            value=st.session_state.get("api_key", ""),
            type="password",
            placeholder="sk-…",
            help=t("api_key_help"),
        )
        st.session_state.api_key = api_key

        model = st.selectbox(
            t("model"),
            options=MODEL_OPTIONS,
            index=MODEL_OPTIONS.index(st.session_state.get("model", DEFAULT_MODEL)),
            help=t("model_help"),
        )
        st.session_state.model = model

        st.divider()
        st.caption(t("data_template"))
        template_bytes = get_csv_template_bytes()
        st.download_button(
            label=t("download_csv_template"),
            data=template_bytes,
            file_name="eval_template.csv",
            mime="text/csv",
        )
        st.divider()

        if st.button("🔄 " + t("restart"), width="stretch"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            init_session_state()
            st.rerun()


# Phase 1: Config and upload
def render_phase_config():
    st.subheader(t("phase1_title"))
    st.divider()

    st.caption(t("scenario_caption"))
    scenario = st.text_area(
        t("scenario_label"),
        value=st.session_state.scenario,
        height=220,
        placeholder=t("scenario_placeholder"),
        help=t("scenario_help"),
    )
    st.session_state.scenario = scenario

    st.divider()
    north_star = st.text_input(
        t("north_star_label"),
        value=st.session_state.north_star,
        placeholder=t("north_star_placeholder"),
        help=t("north_star_help"),
    )
    st.session_state.north_star = north_star

    st.divider()
    uploaded = st.file_uploader(t("upload_csv"), type=["csv"], help=t("upload_help"))

    if uploaded is not None:
        df = None
        last_err = None
        for enc in ("utf-8", "utf-8-sig", "gbk", "gb2312", "latin-1"):
            try:
                uploaded.seek(0)
                df = pd.read_csv(uploaded, encoding=enc)
                break
            except UnicodeDecodeError:
                continue
            except Exception as e:
                last_err = e
                break
        if df is None:
            st.error(t("err_file_decode", msg=last_err or "Unrecognized encoding"))
            return
        ok, err = validate_csv(df)
        if not ok:
            st.error(err)
            return
        st.session_state.uploaded_df = df
        st.caption(t("preview"))
        st.dataframe(df.head(3), width="stretch", hide_index=True)

    st.divider()
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button(t("next_generate_prompt"), type="primary", width="stretch"):
            if not st.session_state.api_key.strip():
                st.error(t("err_fill_api_key"))
            elif not st.session_state.scenario.strip() or not st.session_state.north_star.strip():
                st.error(t("err_fill_scenario"))
            elif st.session_state.uploaded_df is None or st.session_state.uploaded_df.empty:
                st.error(t("err_upload_csv"))
            else:
                with st.spinner("Generating business prompt…"):
                    try:
                        st.session_state.generation_prompt = generate_generation_prompt_in_app(
                            st.session_state.scenario,
                            st.session_state.north_star,
                            st.session_state.api_key,
                        )
                        st.session_state.phase = "GENERATION_PROMPT_EDIT"
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to generate prompt (check API key and network): {e}")
    with col_btn2:
        has_answer = (
            st.session_state.uploaded_df is not None
            and not st.session_state.uploaded_df.empty
            and (
                OPTIONAL_ANSWER_COLUMN in st.session_state.uploaded_df.columns
                or GENERATED_ANSWER_COLUMN in st.session_state.uploaded_df.columns
            )
        )
        if st.button(t("has_answer_btn"), width="stretch", disabled=not has_answer):
            if not st.session_state.api_key.strip():
                st.error(t("err_fill_api_key"))
            elif not st.session_state.scenario.strip() or not st.session_state.north_star.strip():
                st.error(t("err_fill_scenario"))
            else:
                with st.spinner("Generating evaluation plan…"):
                    try:
                        prompt = generate_evaluator_prompt_in_app(
                            st.session_state.scenario,
                            st.session_state.north_star,
                            st.session_state.api_key,
                        )
                        st.session_state.generated_prompt = prompt
                        st.session_state.evaluation_prompt = prompt
                        st.session_state.phase = "PROMPT_EDIT"
                        st.success(t("success_eval_generated"))
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to generate evaluation plan (check API key and network): {e}")


# Phase 2: Business prompt (editable)
def render_phase_generation_prompt_edit():
    st.subheader(t("phase2_title"))
    st.divider()
    st.caption(t("phase2_caption"))

    generation_prompt = st.text_area(
        t("business_prompt_label"),
        value=st.session_state.generation_prompt,
        height=280,
        help=t("business_prompt_help"),
    )
    st.session_state.generation_prompt = generation_prompt

    st.divider()
    if st.button(t("next_generate_eval"), type="primary", width="content"):
        if not (st.session_state.generation_prompt or "").strip():
            st.error(t("err_fill_business_prompt"))
            return
        with st.spinner("Generating evaluator prompt…"):
            try:
                prompt = generate_evaluator_prompt_in_app(
                    st.session_state.scenario,
                    st.session_state.north_star,
                    st.session_state.api_key,
                )
                st.session_state.generated_prompt = prompt
                st.session_state.evaluation_prompt = prompt
                st.session_state.phase = "PROMPT_EDIT"
                st.success(t("success_gen_prompt"))
                st.rerun()
            except Exception as e:
                st.error(f"Failed to generate evaluator prompt (check API key and network): {e}")

    if st.button(t("back_config"), width="content"):
        st.session_state.phase = "CONFIG"
        st.rerun()


# Phase 4: Generate answers
def render_phase_generating():
    st.subheader(t("phase4_title"))
    st.divider()

    df = st.session_state.uploaded_df
    n = len(df) if df is not None else 0
    api_key = st.session_state.api_key
    model = st.session_state.model
    generation_prompt = (st.session_state.generation_prompt or "").strip()

    if not api_key or df is None or n == 0:
        st.error(t("err_config_incomplete"))
        if st.button(t("return_eval_prompt")):
            st.session_state.phase = "PROMPT_EDIT"
            st.rerun()
        return

    if GENERATED_ANSWER_COLUMN not in df.columns:
        df[GENERATED_ANSWER_COLUMN] = None

    # Rows with an answer: either generated_answer or expected_answer is non-empty
    def _has_answer(row):
        gen = str(row.get(GENERATED_ANSWER_COLUMN) or "").strip()
        exp = str(row.get(OPTIONAL_ANSWER_COLUMN) or "").strip() if OPTIONAL_ANSWER_COLUMN in df.columns else ""
        return bool(gen or exp)

    has_question = df["question"].notna() & (df["question"].astype(str).str.strip() != "")
    has_answer = df.apply(_has_answer, axis=1)
    rows_with_question = has_question.sum()
    rows_need_generation = (has_question & ~has_answer).sum()

    # Require business prompt only when we need to generate answers
    if rows_need_generation > 0 and not generation_prompt:
        st.error(t("err_config_incomplete_no_prompt"))
        if st.button(t("return_eval_prompt")):
            st.session_state.phase = "PROMPT_EDIT"
            st.rerun()
        return

    # If all rows already have answers, skip generation and show next step
    if rows_with_question > 0 and rows_need_generation == 0:
        st.caption(t("caption_has_answers"))
        st.session_state.uploaded_df = df
        _render_generation_result_and_next(df, api_key)
        return

    progress_bar = st.progress(0.0, text=t("progress_preparing"))
    status = st.status(t("status_generating"), expanded=True)

    api_cfg = (get_config().get("api") or {}).get("deepseek") or {}
    with status:
        for i, (idx, row) in enumerate(df.iterrows()):
            progress_bar.progress((i + 1) / n, text=f"{i+1}/{n}")
            q = str(row.get("question", "") or "").strip()
            st.write(f"[{i+1}/{n}] {q[:60]}…" if len(q) > 60 else f"[{i+1}/{n}] {q}")
            if not q:
                df.at[idx, GENERATED_ANSWER_COLUMN] = ""
                st.write("  ⏭ " + t("skip_empty_question"))
                continue
            answer, err = run_single_generation(
                q,
                generation_prompt,
                api_key,
                model,
                temperature=api_cfg.get("temperature", 0.7),
                max_tokens=api_cfg.get("max_tokens", 4000),
            )
            if err:
                df.at[idx, GENERATED_ANSWER_COLUMN] = ""
                st.write(f"  ❌ {err}")
            else:
                df.at[idx, GENERATED_ANSWER_COLUMN] = answer or ""
                st.write("  ✅ " + t("gen_ok"))

    progress_bar.progress(1.0, text="Done")
    status.update(label="Generation complete", state="complete")
    st.session_state.uploaded_df = df

    st.divider()
    _render_generation_result_and_next(df, api_key)


def _render_generation_result_and_next(df: pd.DataFrame, api_key: str) -> None:
    """Show generated results (read-only) and a button to proceed to evaluation."""
    st.subheader(t("gen_result_title"))
    st.caption(t("gen_result_caption"))
    answer_series = df.apply(
        lambda r: str(r.get(GENERATED_ANSWER_COLUMN) or r.get(OPTIONAL_ANSWER_COLUMN) or ""),
        axis=1,
    )
    display_df = df[["question"]].copy()
    display_df.columns = [t("col_question")]
    display_df[t("col_answer")] = answer_series
    st.dataframe(display_df, width="stretch", hide_index=True)
    st.divider()
    if st.button(t("next_start_eval"), type="primary", width="content"):
        st.session_state.phase = "EVALUATING"
        st.rerun()


# Phase 3: Eval prompt (editable)
def render_phase_prompt_edit():
    st.subheader(t("phase3_title"))
    st.divider()
    st.caption(t("phase3_caption"))

    evaluation_prompt = st.text_area(
        t("eval_prompt_label"),
        value=st.session_state.evaluation_prompt,
        height=320,
        help=t("eval_prompt_help"),
    )
    st.session_state.evaluation_prompt = evaluation_prompt

    if "{original_text}" not in evaluation_prompt or "{model_output}" not in evaluation_prompt:
        st.warning(t("placeholder_warning"))

    st.divider()
    if st.button(t("confirm_start"), type="primary", width="content"):
        if not st.session_state.evaluation_prompt.strip():
            st.error(t("err_fill_eval_prompt"))
            return
        st.session_state.phase = "GENERATING"
        st.rerun()

    if st.button(t("back_business_prompt"), width="content"):
        st.session_state.phase = "GENERATION_PROMPT_EDIT"
        st.rerun()


# Phase 5: Run evaluation
def render_phase_evaluating():
    st.subheader(t("phase5_title"))
    st.divider()

    df = st.session_state.uploaded_df
    n = len(df) if df is not None else 0
    api_key = st.session_state.api_key
    model = st.session_state.model
    evaluation_prompt = st.session_state.evaluation_prompt

    if not api_key:
        st.error(t("err_api_key"))
        st.session_state.phase = "PROMPT_EDIT"
        return
    if df is None or n == 0:
        st.error(t("err_no_data"))
        st.session_state.phase = "CONFIG"
        return

    progress_bar = st.progress(0.0, text=t("progress_preparing"))
    status = st.status(t("status_evaluating"), expanded=True)

    eval_columns = [
        "eval_priority", "factuality_score", "north_star_score", "completeness_score",
        "weighted_total_score", "decision", "reason", "reasoning", "pass",
    ]
    for col in eval_columns:
        if col not in df.columns:
            df[col] = None

    start_time = time.time()
    with status:
        for i, (idx, row) in enumerate(df.iterrows()):
            progress_bar.progress((i + 1) / n, text=f"Evaluating {i+1}/{n}…")
            st.write(f"[{i+1}/{n}] {str(row.get('question', ''))[:50]}…")

            result, err = run_single_evaluation(row, evaluation_prompt, api_key, model)
            if err:
                df.at[idx, "decision"] = "ERROR"
                df.at[idx, "reason"] = f"error: {err}"
                st.write(f"  ❌ {err}")
            else:
                df.at[idx, "eval_priority"] = result.get("priority")
                df.at[idx, "factuality_score"] = result.get("factuality_score")
                df.at[idx, "north_star_score"] = result.get("north_star_score")
                df.at[idx, "completeness_score"] = result.get("completeness_score")
                df.at[idx, "weighted_total_score"] = result.get("weighted_total_score")
                df.at[idx, "decision"] = result.get("decision")
                df.at[idx, "reason"] = result.get("reason")
                df.at[idx, "reasoning"] = result.get("reasoning")
                df.at[idx, "pass"] = result.get("pass")
                st.write(f"  ✅ {t('eval_ok')}: {result.get('weighted_total_score', 0):.1f} | {result.get('decision', '')}")

    elapsed = time.time() - start_time
    progress_bar.progress(1.0, text="Evaluation complete")
    status.update(label="Evaluation complete", state="complete")

    st.session_state.results_df = df
    st.session_state.eval_elapsed = elapsed
    st.session_state.phase = "RESULT"
    st.success(t("success_eval_done", n=n, elapsed=elapsed))
    st.rerun()


# Phase 6: Results
def render_phase_result():
    st.subheader(t("phase6_title"))
    st.divider()

    df = st.session_state.results_df
    if df is None:
        st.warning(t("no_results"))
        return

    score_numeric = pd.to_numeric(df["weighted_total_score"], errors="coerce") if "weighted_total_score" in df.columns else pd.Series(dtype=float)
    valid = df[score_numeric.notna()]
    n_valid = len(valid)
    n_total = len(df)

    st.caption(t("core_metrics"))
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        avg_score = score_numeric.mean() if score_numeric.notna().any() else 0
        st.metric(t("avg_score"), f"{avg_score:.1f}" if score_numeric.notna().any() else "—")
    with col2:
        pass_count = valid["pass"].sum() if "pass" in valid.columns else (valid["decision"] == "PUBLISH").sum()
        pass_rate = (pass_count / n_valid * 100) if n_valid else 0
        st.metric(t("pass_rate"), f"{pass_rate:.1f}%" if n_valid else "—")
    with col3:
        err_count = (df["decision"] == "ERROR").sum()
        st.metric(t("error_count"), int(err_count))
    with col4:
        st.metric(t("total_count"), n_total)
    with col5:
        elapsed = st.session_state.get("eval_elapsed")
        st.metric(t("elapsed"), f"{elapsed:.1f} s" if elapsed is not None else "—")

    st.divider()

    if n_valid > 0 and "weighted_total_score" in df.columns:
        st.caption(t("score_dist"))
        score_series = pd.to_numeric(valid["weighted_total_score"], errors="coerce").dropna()
        if len(score_series) > 0:
            score_counts = score_series.round(0).value_counts().sort_index()
            fig = px.bar(
                x=score_counts.index.astype(int),
                y=score_counts.values,
                labels={"x": t("score_x"), "y": t("score_y")},
                title=t("score_dist_title"),
            )
            fig.update_layout(showlegend=False, margin=dict(t=40))
            st.plotly_chart(fig, width="stretch")
            if len(score_counts) <= 2 and score_counts.sum() > 5:
                with st.expander("💡 " + t("expander_diff_title"), expanded=False):
                    st.caption(t("expander_diff_caption"))
        else:
            st.caption(t("no_valid_scores"))
    st.divider()

    st.caption(t("prompts_section"))
    with st.expander(t("expand_business_prompt"), expanded=False):
        gen_prompt = st.session_state.get("generation_prompt") or ""
        st.text_area(t("expand_business_prompt"), value=gen_prompt, height=200, disabled=True, label_visibility="collapsed")
    with st.expander(t("expand_eval_prompt"), expanded=False):
        eval_prompt = st.session_state.get("evaluation_prompt") or ""
        st.text_area(t("expand_eval_prompt"), value=eval_prompt, height=280, disabled=True, label_visibility="collapsed")
    st.divider()

    st.caption(t("full_results_caption"))
    st.caption(t("reject_note"))
    display_cols = [
        "question", GENERATED_ANSWER_COLUMN, OPTIONAL_ANSWER_COLUMN,
        "factuality_score", "north_star_score", "completeness_score",
        "weighted_total_score", "decision", "reason", "reasoning",
    ]
    display_cols = [c for c in display_cols if c in df.columns]
    col_config = {}
    for col, label in [
        ("factuality_score", t("col_factuality")),
        ("north_star_score", t("col_north_star")),
        ("completeness_score", t("col_completeness")),
        ("weighted_total_score", t("col_weighted_total")),
    ]:
        if col in display_cols:
            col_config[col] = st.column_config.NumberColumn(label, format="%.1f")
    st.dataframe(
        df[display_cols] if display_cols else df,
        width="stretch",
        hide_index=True,
        column_config=col_config if col_config else None,
    )

    st.divider()
    buf = io.BytesIO()
    df.to_csv(buf, index=False, encoding="utf-8-sig")
    st.download_button(
        label=t("download_csv"),
        data=buf.getvalue(),
        file_name="eval_results.csv",
        mime="text/csv",
    )


# Main
def main():
    init_session_state()
    render_sidebar()

    st.title("📊 " + t("app_title"))
    st.caption(t("breadcrumb"))
    st.divider()

    phase = st.session_state.phase
    if phase == "CONFIG":
        render_phase_config()
    elif phase == "GENERATION_PROMPT_EDIT":
        render_phase_generation_prompt_edit()
    elif phase == "GENERATING":
        render_phase_generating()
    elif phase == "PROMPT_EDIT":
        render_phase_prompt_edit()
    elif phase == "EVALUATING":
        render_phase_evaluating()
    elif phase == "RESULT":
        render_phase_result()
    else:
        st.session_state.phase = "CONFIG"
        st.rerun()


if __name__ == "__main__":
    main()
