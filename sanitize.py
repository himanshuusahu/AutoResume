"""Validate and sanitize LaTeX output from the LLM."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Commands that must never appear in LLM-generated content.
FORBIDDEN_COMMANDS: tuple[str, ...] = (
    r"\\write18",
    r"\\input",
    r"\\include",
    r"\\includeonly",
    r"\\openin",
    r"\\openout",
    r"\\immediate",
    r"\\write",
    r"\\read",
    r"\\csname",
    r"\\expandafter",
    r"\\def",
    r"\\let",
    r"\\newcommand",
    r"\\renewcommand",
    r"\\providecommand",
    r"\\usepackage",
    r"\\RequirePackage",
    r"\\documentclass",
    r"\\begin\{document\}",
    r"\\end\{document\}",
)

FORBIDDEN_PATTERN = re.compile(
    "|".join(FORBIDDEN_COMMANDS),
    re.IGNORECASE,
)

# Strip markdown code fences the model may wrap around LaTeX.
CODE_FENCE_PATTERN = re.compile(
    r"^```(?:latex|tex)?\s*\n?|```\s*$",
    re.MULTILINE,
)


@dataclass
class SanitizeResult:
    text: str
    removed: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def strip_code_fences(text: str) -> str:
    """Remove markdown code fences from model output."""
    return CODE_FENCE_PATTERN.sub("", text).strip()


def find_forbidden_commands(text: str) -> list[str]:
    """Return a list of forbidden LaTeX commands found in *text*."""
    return sorted({match.group(0) for match in FORBIDDEN_PATTERN.finditer(text)})


def remove_forbidden_commands(text: str) -> SanitizeResult:
    """Remove forbidden commands and report what was stripped."""
    removed = find_forbidden_commands(text)
    cleaned = FORBIDDEN_PATTERN.sub("", text)
    warnings: list[str] = []
    if removed:
        warnings.append(f"Removed forbidden LaTeX commands: {', '.join(removed)}")
    return SanitizeResult(text=cleaned, removed=removed, warnings=warnings)


def sanitize_latex(text: str) -> SanitizeResult:
    """
    Full sanitization pipeline for LLM-produced LaTeX fragments.

    1. Strip markdown fences
    2. Remove forbidden commands
    3. Normalize line endings
    """
    result = remove_forbidden_commands(strip_code_fences(text))
    result.text = result.text.replace("\r\n", "\n").strip()
    return result


def validate_section(text: str) -> list[str]:
    """Return validation errors for a LaTeX section fragment."""
    errors: list[str] = []
    forbidden = find_forbidden_commands(text)
    if forbidden:
        errors.append(f"Forbidden commands present: {', '.join(forbidden)}")
    if "\\begin{document}" in text or "\\end{document}" in text:
        errors.append("Section must not contain \\begin{document} or \\end{document}")
    return errors
