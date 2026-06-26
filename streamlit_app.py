"""Streamlit UI for AutoResme — tailor LaTeX resumes to job descriptions."""

from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from main import SectionError, run_pipeline
from llm import AVAILABLE_MODELS, default_model

st.set_page_config(
    page_title="AutoResme",
    page_icon="📄",
    layout="wide",
)

MODELS = AVAILABLE_MODELS
KEYWORD_METHODS = ["auto", "spacy", "regex"]


def _render_sidebar() -> tuple[str, str]:
    st.sidebar.header("Settings")
    default = default_model()
    model = st.sidebar.selectbox(
        "Model",
        MODELS,
        index=MODELS.index(default) if default in MODELS else 0,
    )
    keyword_method = st.sidebar.selectbox("Keyword extraction", KEYWORD_METHODS, index=0)
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "**Auth:** Set `DATABRICKS_TOKEN` (or `OPENAI_API_KEY`) in `.env`."
    )
    return model, keyword_method


def main() -> None:
    st.title("AutoResme")
    st.caption(
        "Tailor your LaTeX resume to a job description. "
        "Only **Experience** and **Technical Skills** are edited. "
        "Copy the output into [Overleaf](https://www.overleaf.com) to compile a PDF."
    )

    model, keyword_method = _render_sidebar()

    col_jd, col_resume = st.columns(2)
    with col_jd:
        job_description = st.text_area(
            "Job description",
            height=320,
            placeholder="Paste the full job description here...",
        )
    with col_resume:
        resume_file = st.file_uploader("Resume (.tex)", type=["tex"])

    tailor = st.button("Tailor resume", type="primary", use_container_width=True)

    if not tailor:
        return

    if not job_description.strip():
        st.error("Job description is required.")
        return
    if resume_file is None:
        st.error("Upload a .tex resume file.")
        return

    with st.spinner("Extracting keywords and tailoring resume..."):
        with tempfile.NamedTemporaryFile(suffix=".tex", delete=False) as tmp:
            tmp.write(resume_file.getvalue())
            tmp_path = Path(tmp.name)

        try:
            result = run_pipeline(
                job_description,
                tmp_path,
                model=model,
                keyword_method=keyword_method,
                compile_pdf=False,
            )
        except (ValueError, RuntimeError, SectionError) as exc:
            st.error(str(exc))
            return
        finally:
            tmp_path.unlink(missing_ok=True)

    st.success("Resume tailored successfully.")

    st.subheader("Keywords")
    st.write(", ".join(f"`{kw}`" for kw in result["keywords"]))

    if result["warnings"]:
        for warning in result["warnings"]:
            st.warning(warning)

    tab_diff, tab_latex = st.tabs(["Changes (diff)", "Full LaTeX"])

    with tab_diff:
        st.markdown("#### Experience")
        st.code(result["experience_diff"], language="diff")
        st.markdown("#### Technical Skills")
        st.code(result["skills_diff"], language="diff")

    with tab_latex:
        st.code(result["tailored_tex"], language="latex", line_numbers=True)
        st.download_button(
            label="Download tailored .tex",
            data=result["tailored_tex"],
            file_name=f"{Path(resume_file.name).stem}_tailored.tex",
            mime="application/x-tex",
            type="primary",
        )
        st.caption(f"Also saved to: `{result['tex_path']}`")


if __name__ == "__main__":
    main()
