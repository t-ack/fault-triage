"""OpenRouter client for the LLM classification stage.

Reads OPENROUTER_API_KEY and OPENROUTER_MODEL from the environment, loading a
.env file at the project root first if one exists. The key is never printed,
logged or included in error messages.
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PLACEHOLDER_KEY = "sk-or-your-key-here"
PLACEHOLDER_MODEL = "model-slug-from-the-course"

_SETUP_HINT = "Copy .env.example to .env and fill in your OpenRouter values."


class OpenRouterConfigError(RuntimeError):
    """Raised when the OpenRouter key or model is missing or still a placeholder."""


def load_env() -> None:
    """Load .env from the project root. Values already in the environment win."""
    load_dotenv(PROJECT_ROOT / ".env", override=False)


def _required(name: str, placeholder: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise OpenRouterConfigError(f"{name} is not set. {_SETUP_HINT}")
    if value == placeholder:
        raise OpenRouterConfigError(f"{name} still has the placeholder value. {_SETUP_HINT}")
    return value


def get_model() -> str:
    """Model slug from OPENROUTER_MODEL."""
    load_env()
    return _required("OPENROUTER_MODEL", PLACEHOLDER_MODEL)


def get_client() -> OpenAI:
    """OpenAI client pointed at OpenRouter, using OPENROUTER_API_KEY."""
    load_env()
    api_key = _required("OPENROUTER_API_KEY", PLACEHOLDER_KEY)
    return OpenAI(base_url=OPENROUTER_BASE_URL, api_key=api_key)
