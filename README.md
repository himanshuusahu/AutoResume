# AutoResme

Tailor a **LaTeX resume** to a job description using keyword extraction + an OpenAI-compatible LLM (OpenAI or Databricks AI Gateway).

- Only the **Experience** and **Technical Skills** sections are edited.
- Output is a tailored **`.tex`** file (paste into **Overleaf** to compile).

---

## Architecture

```mermaid
flowchart TD
  A[Input: Job Description + Resume .tex] --> B[Keyword Extraction (keywords.py)]
  B --> C[Prompt Builder (main.py)]
  C --> D[LLM Call (llm.py)]
  D --> E[Parse XML Tags + Patch Resume Sections (main.py)]
  E --> F[Sanitize & Validate LaTeX Fragments (sanitize.py)]
  F --> G[Write tailored .tex to output/]

  subgraph Optional
    H[Compile PDF locally (compile.py)]
  end

  G -->|--compile| H


```

### Key modules

- **`main.py`**: Orchestrates the pipeline, finds/patches the target sections, produces diffs, and writes output.
- **`keywords.py`**: Extracts top job keywords (spaCy if available, otherwise regex fallback).
- **`llm.py`**: OpenAI-compatible client wrapper. Supports Databricks AI Gateway via `base_url`.
- **`sanitize.py`**: Removes markdown fences, strips forbidden LaTeX commands, validates fragments.
- **`compile.py`** (optional): Compiles `.tex → .pdf` using `latexmk` or `pdflatex` if installed.
- **`streamlit_app.py`**: Streamlit UI.

---

## Repo layout

```
AutoResme/
├── main.py
├── llm.py
├── keywords.py
├── sanitize.py
├── compile.py
├── streamlit_app.py
├── prompts/
│   └── edit_prompt.txt
├── templates/
│   └── resume.tex
└── requirements.txt
```

---

## Setup

### 1) Create venv + install deps

```powershell
cd C:\Users\Himanshu\Desktop\AutoResme
python -m venv myenv
.\myenv\Scripts\Activate.ps1
pip install -r requirements.txt

# Optional: only needed if you want spaCy keyword extraction
python -m spacy download en_core_web_sm
```

### 2) Configure API credentials

Set credentials in environment variables (or via `.env` + `python-dotenv`).

- Use **Databricks AI Gateway** (default in this repo):

```powershell
set DATABRICKS_TOKEN=your-token
set LLM_MODEL=databricks-llama-4-maverick

# optional overrides
set DATABRICKS_BASE_URL=https://dbc-.../ai-gateway/mlflow/v1
```

- Or use **OpenAI** directly (if you have `OPENAI_API_KEY`):

```powershell
set OPENAI_API_KEY=your-openai-key
set OPENAI_BASE_URL=...   # optional
```

---

## How to use

### CLI (tailor + save tailored `.tex`)

```powershell
python main.py `
  --job-description "Python engineer with Flask, PostgreSQL, Docker, CI/CD." `
  --resume templates\resume.tex
```

The tailored file is written to:

- `<resume_dir>/output/<resume-name>_tailored.tex` (default)

In the current repo, examples can be found under `templates/output_test/`.

Open the output in **Overleaf** to generate the PDF.

### Choose keyword extraction method

```powershell
python main.py -j "..." -r templates\resume.tex --keyword-method spacy
python main.py -j "..." -r templates\resume.tex --keyword-method regex
```

### Optional: compile PDF locally

Requires `latexmk` or `pdflatex` on your PATH.

```powershell
python main.py -j "..." -r templates\resume.tex --compile
```

---

## UI modes

### Gradio

```powershell
python main.py --ui
```

### Streamlit

```powershell
python main.py --streamlit
# or directly
streamlit run streamlit_app.py
```

---

## Resume template requirements

The LLM is instructed to edit only the following blocks inside `templates/resume.tex`:

- Marked by comments:

```latex
% BEGIN_EXPERIENCE
... % Experience block ...
% END_EXPERIENCE

% BEGIN_SKILLS
... % Technical Skills block ...
% END_SKILLS
```

If those markers aren’t present, `main.py` can fall back to locating `\section{Experience}` and `\section{Technical Skills}` headings.

---

## Output behavior

The pipeline:

1. Extracts keywords from the job description.
2. Builds a prompt using `prompts/edit_prompt.txt`.
3. Calls the LLM and expects output wrapped as XML tags:
   - `<experience> ... </experience>`
   - `<skills> ... </skills>`
4. Sanitizes and validates the LaTeX fragments.
5. Patches only **Experience** and **Technical Skills** back into the template.
6. Writes the full tailored document to `output/`.

---

## Prompt & safety

- The prompt explicitly forbids editing other resume sections.
- `sanitize.py` strips markdown code fences and removes forbidden LaTeX commands (e.g., `\write18`, `\include`, `\usepackage`, etc.).

---

## Development notes

If you change how sections are located/patched:

- Update `main.py` section extraction + patching logic.
- Keep the markers or ensure the fallback headings stay consistent.

---

## License

MIT

