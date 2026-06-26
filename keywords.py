"""Extract keywords from a job description using spaCy or regex fallback."""

from __future__ import annotations

import re
from collections import Counter

# Common English stop words plus resume/JD boilerplate.
STOP_WORDS: set[str] = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for", "of",
    "with", "by", "from", "as", "is", "was", "are", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could", "should",
    "may", "might", "must", "shall", "can", "need", "our", "your", "you", "we",
    "they", "them", "their", "this", "that", "these", "those", "it", "its",
    "who", "whom", "which", "what", "when", "where", "why", "how", "all", "each",
    "every", "both", "few", "more", "most", "other", "some", "such", "no", "nor",
    "not", "only", "own", "same", "so", "than", "too", "very", "just", "about",
    "into", "through", "during", "before", "after", "above", "below", "between",
    "out", "off", "over", "under", "again", "further", "then", "once", "here",
    "there", "any", "work", "working", "role", "position", "job", "team", "company",
    "experience", "years", "year", "ability", "able", "including", "etc", "us",
    "applicant", "candidate", "responsibilities", "requirements", "qualifications",
    "preferred", "required", "looking", "join", "opportunity", "apply",
}

# Technical tokens often split across punctuation in JDs.
TECH_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\b(?:c\+\+|c#|\.net|node\.js|react\.js|vue\.js|next\.js)\b", re.I),
    re.compile(r"\b(?:aws|gcp|azure|kubernetes|docker|terraform|ansible)\b", re.I),
    re.compile(r"\b(?:python|java|javascript|typescript|golang|rust|scala|kotlin)\b", re.I),
    re.compile(r"\b(?:sql|nosql|postgresql|mongodb|redis|kafka|spark)\b", re.I),
    re.compile(r"\b(?:machine learning|deep learning|nlp|computer vision)\b", re.I),
    re.compile(r"\b(?:selenium|playwright|cypress|appium|restassured|testng|junit|cucumber)\b", re.I),
    re.compile(r"\b(?:qa|quality assurance|automation|ci/cd|jenkins|gitlab)\b", re.I),
]

WORD_PATTERN = re.compile(r"\b[A-Za-z][A-Za-z0-9+#./-]{1,}\b")
CAMEL_PATTERN = re.compile(r"\b[A-Z][a-z]+(?:[A-Z][a-z]+)+\b")


def _normalize_keyword(word: str) -> str:
    return word.strip().lower()


def extract_keywords_regex(job_description: str, top_n: int = 25) -> list[str]:
    """Regex-based keyword extraction (no external model required)."""
    text = job_description
    counts: Counter[str] = Counter()

    for pattern in TECH_PATTERNS:
        for match in pattern.finditer(text):
            counts[_normalize_keyword(match.group(0))] += 3

    for match in WORD_PATTERN.finditer(text):
        word = match.group(0)
        key = _normalize_keyword(word)
        if len(key) < 2 or key in STOP_WORDS:
            continue
        weight = 2 if word[0].isupper() and word.lower() != word else 1
        counts[key] += weight

    for match in CAMEL_PATTERN.finditer(text):
        counts[_normalize_keyword(match.group(0))] += 2

    ranked = [kw for kw, _ in counts.most_common(top_n)]
    return ranked


def extract_keywords_spacy(job_description: str, top_n: int = 25) -> list[str]:
    """spaCy-based extraction using nouns, proper nouns, and noun chunks."""
    import spacy

    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError as exc:
        raise RuntimeError(
            "spaCy model 'en_core_web_sm' not found. "
            "Run: python -m spacy download en_core_web_sm"
        ) from exc

    doc = nlp(job_description)
    counts: Counter[str] = Counter()

    for token in doc:
        if token.is_stop or token.is_punct or len(token.text) < 2:
            continue
        if token.pos_ in {"NOUN", "PROPN"} and token.lemma_.lower() not in STOP_WORDS:
            counts[token.lemma_.lower()] += 2 if token.pos_ == "PROPN" else 1

    for chunk in doc.noun_chunks:
        phrase = chunk.text.strip().lower()
        if len(phrase) > 2 and phrase not in STOP_WORDS:
            counts[phrase] += 1

    ranked = [kw for kw, _ in counts.most_common(top_n)]
    return ranked


def extract_keywords(
    job_description: str,
    *,
    method: str = "auto",
    top_n: int = 25,
) -> list[str]:
    """
    Extract keywords from a job description.

    *method* may be ``auto`` (try spaCy, fall back to regex), ``spacy``, or ``regex``.
    """
    if method == "regex":
        return extract_keywords_regex(job_description, top_n=top_n)

    if method == "spacy":
        return extract_keywords_spacy(job_description, top_n=top_n)

    # auto
    try:
        return extract_keywords_spacy(job_description, top_n=top_n)
    except (ImportError, OSError, RuntimeError):
        return extract_keywords_regex(job_description, top_n=top_n)
