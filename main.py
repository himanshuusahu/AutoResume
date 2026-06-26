"""Orchestrate resume tailoring: keywords → LLM edit → sanitize → compile."""

from __future__ import annotations

import argparse
import difflib
import re
import subprocess
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from keywords import extract_keywords
from llm import AVAILABLE_MODELS, DEFAULT_MODEL, call_llm, default_model
from sanitize import sanitize_latex, validate_section

ROOT = Path(__file__).resolve().parent
DEFAULT_PROMPT = ROOT / "prompts" / "edit_prompt.txt"
DEFAULT_RESUME = ROOT / "templates" / "resume.tex"

EXPERIENCE_BEGIN = "% BEGIN_EXPERIENCE"
EXPERIENCE_END = "% END_EXPERIENCE"
SKILLS_BEGIN = "% BEGIN_SKILLS"
SKILLS_END = "% END_SKILLS"

TAG_EXPERIENCE = re.compile(
    r"<experience>\s*(.*?)\s*</experience>",
    re.DOTALL | re.IGNORECASE,
)
TAG_SKILLS = re.compile(
    r"<skills>\s*(.*?)\s*</skills>",
    re.DOTALL | re.IGNORECASE,
)


class SectionError(ValueError):
    """Raised when Experience/Skills sections cannot be located or patched."""


def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def save_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def extract_between_markers(text: str, begin: str, end: str) -> str | None:
    pattern = re.compile(
        re.escape(begin) + r"\s*(.*?)\s*" + re.escape(end),
        re.DOTALL,
    )
    match = pattern.search(text)
    return match.group(1).strip() if match else None


def replace_between_markers(
    text: str,
    begin: str,
    end: str,
    new_content: str,
) -> str:
    pattern = re.compile(
        re.escape(begin) + r"\s*.*?\s*" + re.escape(end),
        re.DOTALL,
    )
    if not pattern.search(text):
        raise SectionError(f"Markers not found: {begin} / {end}")
    replacement = f"{begin}\n{new_content.strip()}\n{end}"
    return pattern.sub(lambda _match: replacement, text, count=1)


def extract_section_by_heading(text: str, heading: str) -> str | None:
    """Fallback: extract content from \\section{Heading} to next \\section."""
    pattern = re.compile(
        rf"(\\section\{{{re.escape(heading)}\}}.*?)(?=\\section\{{|\Z)",
        re.DOTALL | re.IGNORECASE,
    )
    match = pattern.search(text)
    return match.group(1).strip() if match else None


SKILLS_HEADINGS = ("Technical Skills", "Skills")


def _find_skills_heading(resume_tex: str) -> str | None:
    for heading in SKILLS_HEADINGS:
        if extract_section_by_heading(resume_tex, heading):
            return heading
    return None


def get_experience_section(resume_tex: str) -> str:
    section = extract_between_markers(resume_tex, EXPERIENCE_BEGIN, EXPERIENCE_END)
    if section:
        return section
    section = extract_section_by_heading(resume_tex, "Experience")
    if section:
        return section
    raise SectionError(
        "Could not find Experience section. Add % BEGIN_EXPERIENCE / % END_EXPERIENCE markers "
        "or a \\section{Experience} heading."
    )


def get_skills_section(resume_tex: str) -> str:
    section = extract_between_markers(resume_tex, SKILLS_BEGIN, SKILLS_END)
    if section:
        return section
    for heading in SKILLS_HEADINGS:
        section = extract_section_by_heading(resume_tex, heading)
        if section:
            return section
    raise SectionError(
        "Could not find Skills section. Add % BEGIN_SKILLS / % END_SKILLS markers "
        "or a \\section{Technical Skills} or \\section{Skills} heading."
    )


def _patch_skills_section(resume_tex: str, new_skills: str) -> str:
    if extract_between_markers(resume_tex, SKILLS_BEGIN, SKILLS_END) is not None:
        return replace_between_markers(resume_tex, SKILLS_BEGIN, SKILLS_END, new_skills)
    heading = _find_skills_heading(resume_tex)
    if heading:
        return _replace_section_heading(resume_tex, heading, new_skills)
    raise SectionError("Could not patch Skills / Technical Skills section.")


def patch_resume(
    resume_tex: str,
    new_experience: str,
    new_skills: str,
) -> str:
    """Insert sanitized Experience and Skills sections back into the resume."""
    updated = resume_tex
    if extract_between_markers(resume_tex, EXPERIENCE_BEGIN, EXPERIENCE_END) is not None:
        updated = replace_between_markers(updated, EXPERIENCE_BEGIN, EXPERIENCE_END, new_experience)
    else:
        updated = _replace_section_heading(updated, "Experience", new_experience)

    if extract_between_markers(resume_tex, SKILLS_BEGIN, SKILLS_END) is not None:
        updated = replace_between_markers(updated, SKILLS_BEGIN, SKILLS_END, new_skills)
    else:
        updated = _patch_skills_section(updated, new_skills)
    return updated


def _replace_section_heading(text: str, heading: str, new_content: str) -> str:
    pattern = re.compile(
        rf"\\section\{{{re.escape(heading)}\}}.*?(?=\\section\{{|\Z)",
        re.DOTALL | re.IGNORECASE,
    )
    if not pattern.search(text):
        raise SectionError(f"Could not patch \\section{{{heading}}}")
    new_block = new_content.strip() + "\n\n"
    return pattern.sub(lambda _match: new_block, text, count=1)


def build_prompt(
    template_path: Path,
    *,
    job_description: str,
    keywords: list[str],
    experience_section: str,
    skills_section: str,
) -> str:
    template = load_text(template_path)

    # Some users may edit the prompt template and accidentally introduce
    # placeholders for Python str.format, e.g. {Experience}. Replace only
    # the known placeholders and fail fast with a clear message otherwise.
    try:
        # Replace only the known placeholders, and avoid Python's .format
        # trying to treat LaTeX braces (e.g. {\itemize}) as format placeholders.
        # We do this via simple token substitution.
        return (
            template.replace("{job_description}", job_description.strip())
            .replace("{keywords}", ", ".join(keywords))
            .replace("{experience_section}", experience_section)
            .replace("{skills_section}", skills_section)
        )
    except KeyError as exc:
        missing = str(exc).strip("'")
        raise KeyError(
            f"Prompt template is missing placeholder value for '{missing}'. "
            "Ensure the template only contains {job_description}, {keywords}, "
            "{experience_section}, {skills_section}."
        ) from exc


def call_openai(prompt: str, *, model: str | None = None) -> str:
    """Call the configured LLM (Databricks AI Gateway or OpenAI)."""
    return call_llm(prompt, model=model)


def parse_llm_sections(llm_output: str) -> tuple[str, str]:
    exp_match = TAG_EXPERIENCE.search(llm_output)
    skills_match = TAG_SKILLS.search(llm_output)
    if not exp_match or not skills_match:
        raise ValueError(
            "LLM response missing <experience> or <skills> tags. "
            f"Raw output:\n{llm_output[:500]}..."
        )
    return exp_match.group(1).strip(), skills_match.group(1).strip()


def make_diff(original: str, updated: str, label: str) -> str:
    lines = difflib.unified_diff(
        original.splitlines(),
        updated.splitlines(),
        fromfile=f"{label} (original)",
        tofile=f"{label} (tailored)",
        lineterm="",
    )
    return "\n".join(lines) or f"No changes in {label}."


def run_pipeline(
    job_description: str,
    resume_path: Path,
    *,
    output_dir: Path | None = None,
    prompt_path: Path = DEFAULT_PROMPT,
    model: str | None = None,
    keyword_method: str = "auto",
    compile_pdf: bool = False,
) -> dict:
    """
    Full tailoring pipeline.

    Returns a dict with paths, diff text, keywords, and optional warnings.
    """
    resume_tex = load_text(resume_path)
    experience = get_experience_section(resume_tex)
    skills = get_skills_section(resume_tex)

    keywords = extract_keywords(job_description, method=keyword_method)
    prompt = build_prompt(
        prompt_path,
        job_description=job_description,
        keywords=keywords,
        experience_section=experience,
        skills_section=skills,
    )

    llm_raw = call_openai(prompt, model=model or default_model())
    new_experience_raw, new_skills_raw = parse_llm_sections(llm_raw)

    exp_result = sanitize_latex(new_experience_raw)
    skills_result = sanitize_latex(new_skills_raw)

    for label, errors in (
        ("Experience", validate_section(exp_result.text)),
        ("Skills", validate_section(skills_result.text)),
    ):
        if errors:
            raise ValueError(f"{label} validation failed: {'; '.join(errors)}")

    tailored_tex = patch_resume(resume_tex, exp_result.text, skills_result.text)

    out_dir = (output_dir or resume_path.parent / "output").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    tex_out = out_dir / f"{resume_path.stem}_tailored.tex"
    save_text(tex_out, tailored_tex)

    result: dict = {
        "keywords": keywords,
        "tex_path": tex_out,
        "pdf_path": None,
        "experience_diff": make_diff(experience, exp_result.text, "Experience"),
        "skills_diff": make_diff(skills, skills_result.text, "Skills"),
        "warnings": exp_result.warnings + skills_result.warnings,
        "tailored_tex": tailored_tex,
    }

    if compile_pdf:
        from compile import compile_tex

        pdf_path = compile_tex(tex_out, output_dir=out_dir)
        result["pdf_path"] = pdf_path

    return result


def _cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tailor a LaTeX resume to a job description using an LLM.",
    )
    parser.add_argument(
        "--job-description",
        "-j",
        help="Job description text (or use --job-file).",
    )
    parser.add_argument(
        "--job-file",
        "-f",
        type=Path,
        help="Path to a text file containing the job description.",
    )
    parser.add_argument(
        "--resume",
        "-r",
        type=Path,
        default=DEFAULT_RESUME,
        help=f"Path to resume .tex file (default: {DEFAULT_RESUME})",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        help="Directory for tailored .tex (default: <resume_dir>/output).",
    )
    parser.add_argument(
        "--prompt",
        type=Path,
        default=DEFAULT_PROMPT,
        help="Path to LLM prompt template.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help=f"Model id (default: {DEFAULT_MODEL} or LLM_MODEL env var).",
    )
    parser.add_argument(
        "--keyword-method",
        choices=["auto", "spacy", "regex"],
        default="auto",
        help="Keyword extraction method (default: auto).",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Also compile PDF locally (requires pdflatex/latexmk on PATH).",
    )
    parser.add_argument(
        "--ui",
        action="store_true",
        help="Launch Gradio web UI.",
    )
    parser.add_argument(
        "--streamlit",
        action="store_true",
        help="Launch Streamlit web UI.",
    )
    return parser.parse_args()


def _resolve_job_description(args: argparse.Namespace) -> str:
    if args.job_file:
        return load_text(args.job_file)
    if args.job_description:
        return args.job_description
    raise SystemExit("Provide --job-description or --job-file.")


def launch_gradio() -> None:
    """Gradio UI: upload JD + resume, preview diff and tailored LaTeX."""
    import gradio as gr

    def tailor_ui(job_text: str, resume_file, model: str, keyword_method: str):
        if not job_text or not job_text.strip():
            return "Job description is required.", "", "", None
        if resume_file is None:
            return "Upload a .tex resume file.", "", "", None

        resume_path = Path(resume_file.name)
        try:
            result = run_pipeline(
                job_text,
                resume_path,
                model=model,
                keyword_method=keyword_method,
                compile_pdf=False,
            )
        except (ValueError, RuntimeError, SectionError) as exc:
            return f"Error: {exc}", "", "", None

        keywords_line = "Keywords: " + ", ".join(result["keywords"])
        diff = result["experience_diff"] + "\n\n" + result["skills_diff"]
        warnings = "\n".join(result["warnings"]) if result["warnings"] else "None"
        status = (
            f"Done.\n{keywords_line}\n"
            f"Saved to: {result['tex_path']}\n"
            f"Warnings: {warnings}\n\n"
            "Copy the LaTeX below into Overleaf to compile a PDF."
        )
        return status, diff, result["tailored_tex"], str(result["tex_path"])

    with gr.Blocks(title="AutoResme — LaTeX Resume Tailor") as demo:
        gr.Markdown(
            "# AutoResme\n"
            "Upload a job description and LaTeX resume. "
            "Only **Experience** and **Technical Skills** are edited. "
            "Output is tailored `.tex` for Overleaf."
        )
        with gr.Row():
            job_input = gr.Textbox(
                label="Job Description",
                lines=12,
                placeholder="Paste the full job description here...",
            )
            resume_input = gr.File(label="Resume (.tex)", file_types=[".tex"])
        with gr.Row():
            model_input = gr.Dropdown(
                choices=AVAILABLE_MODELS,
                value=default_model(),
                label="Model",
            )
            method_input = gr.Dropdown(
                choices=["auto", "spacy", "regex"],
                value="auto",
                label="Keyword Method",
            )
        run_btn = gr.Button("Tailor Resume", variant="primary")
        status_output = gr.Textbox(label="Status", lines=6)
        diff_output = gr.Code(label="Diff (Experience + Technical Skills)", language=None)
        latex_output = gr.Code(label="Tailored LaTeX (full document)", language="latex")
        tex_download = gr.File(label="Download tailored .tex")

        run_btn.click(
            tailor_ui,
            inputs=[job_input, resume_input, model_input, method_input],
            outputs=[status_output, diff_output, latex_output, tex_download],
        )

    demo.launch()


def launch_streamlit() -> None:
    """Launch the Streamlit app."""
    app = ROOT / "streamlit_app.py"
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(app), "--server.headless", "true"],
        check=True,
    )


def main() -> None:
    args = _cli()

    if args.ui:
        launch_gradio()
        return

    if args.streamlit:
        launch_streamlit()
        return

    job_description = _resolve_job_description(args)

    try:
        result = run_pipeline(
            job_description,
            args.resume.resolve(),
            output_dir=args.output_dir,
            prompt_path=args.prompt.resolve(),
            model=args.model or default_model(),
            keyword_method=args.keyword_method,
            compile_pdf=args.compile,
        )
    except Exception as exc:
        from compile import CompileError

        if isinstance(exc, (ValueError, RuntimeError, SectionError, CompileError)):
            print(f"Error: {exc}", file=sys.stderr)
            sys.exit(1)
        raise

    print("Keywords:", ", ".join(result["keywords"]))
    print(f"Tailored LaTeX: {result['tex_path']}")
    if result["pdf_path"]:
        print(f"PDF:            {result['pdf_path']}")
    else:
        print("Open the .tex file in Overleaf (or any LaTeX editor) to compile a PDF.")
    if result["warnings"]:
        print("Warnings:")
        for warning in result["warnings"]:
            print(f"  - {warning}")
    print("\n--- Experience diff ---")
    print(result["experience_diff"])
    print("\n--- Skills diff ---")
    print(result["skills_diff"])


if __name__ == "__main__":
    main()
