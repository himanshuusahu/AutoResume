"""LLM client — OpenAI-compatible API (OpenAI or Databricks AI Gateway)."""

from __future__ import annotations

import os

DEFAULT_DATABRICKS_BASE_URL = (
    "https://dbc-948693ae-8673.cloud.databricks.com/ai-gateway/mlflow/v1"
)
DEFAULT_MODEL = "databricks-llama-4-maverick"

AVAILABLE_MODELS = [
    "databricks-llama-4-maverick",
    
]


def _api_key() -> str:
    key = os.environ.get("DATABRICKS_TOKEN") or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError(
            "No API key found. Set DATABRICKS_TOKEN or OPENAI_API_KEY in your "
            "environment or .env file."
        )
    return key


def _base_url() -> str | None:
    explicit = os.environ.get("OPENAI_BASE_URL") or os.environ.get("DATABRICKS_BASE_URL")
    if explicit:
        return explicit
    if os.environ.get("DATABRICKS_TOKEN"):
        return DEFAULT_DATABRICKS_BASE_URL
    return None


def default_model() -> str:
    if os.environ.get("LLM_MODEL"):
        return os.environ["LLM_MODEL"]
    if os.environ.get("DATABRICKS_TOKEN") or os.environ.get("DATABRICKS_BASE_URL"):
        return DEFAULT_MODEL
    return "gpt-4o-mini"


def get_client():
    """Return an OpenAI SDK client (works with Databricks AI Gateway)."""
    from openai import OpenAI

    kwargs: dict = {"api_key": _api_key()}
    base_url = _base_url()
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs)


def call_llm(prompt: str, *, model: str | None = None) -> str:
    """Send a tailoring prompt and return the model response text."""
    model = model or default_model()
    max_tokens = int(os.environ.get("LLM_MAX_TOKENS", "4096"))

    client = get_client()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You tailor LaTeX resume sections to job descriptions. "
                    "Follow instructions exactly and output only the requested XML tags."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=float(os.environ.get("LLM_TEMPERATURE", "0.3")),
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content or ""
