"""Optional local PDF compilation (not required — default output is .tex only)."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


class CompileError(RuntimeError):
    """Raised when LaTeX compilation fails."""


def _run(cmd: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        check=False,
    )


def compile_tex(
    tex_path: Path,
    output_dir: Path | None = None,
    *,
    clean_aux: bool = True,
) -> Path:
    """
    Compile *tex_path* to PDF using latexmk or pdflatex on PATH.

    This is optional. The main pipeline outputs tailored .tex for Overleaf.
    """
    tex_path = tex_path.resolve()
    if not tex_path.exists():
        raise FileNotFoundError(f"TeX file not found: {tex_path}")

    work_dir = (output_dir or tex_path.parent).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = work_dir / f"{tex_path.stem}.pdf"

    latexmk = shutil.which("latexmk")
    if latexmk:
        cmd = [
            latexmk,
            "-pdf",
            "-interaction=nonstopmode",
            "-halt-on-error",
            f"-output-directory={work_dir}",
            str(tex_path),
        ]
        result = _run(cmd, work_dir)
        if result.returncode == 0 and pdf_path.exists():
            if clean_aux:
                _run([latexmk, "-c", f"-output-directory={work_dir}", str(tex_path)], work_dir)
            return pdf_path
        raise CompileError(
            "latexmk failed.\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )

    pdflatex = shutil.which("pdflatex")
    if pdflatex:
        cmd = [
            pdflatex,
            "-interaction=nonstopmode",
            "-halt-on-error",
            f"-output-directory={work_dir}",
            str(tex_path),
        ]
        for _ in range(2):
            result = _run(cmd, work_dir)
            if result.returncode != 0:
                raise CompileError(
                    "pdflatex failed.\n"
                    f"stdout:\n{result.stdout}\n"
                    f"stderr:\n{result.stderr}"
                )
        if pdf_path.exists():
            return pdf_path
        raise CompileError(f"pdflatex completed but PDF not found: {pdf_path}")

    raise CompileError(
        "No LaTeX compiler on PATH. Output is still saved as .tex — "
        "open it in Overleaf to compile, or install pdflatex locally."
    )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python compile.py <file.tex>")
        sys.exit(1)
    pdf = compile_tex(Path(sys.argv[1]))
    print(f"PDF written to: {pdf}")
