"""Live smoke test for the OpenRouter connection.

Skipped automatically when OPENROUTER_API_KEY is not set (in the environment or .env),
so the rest of the suite passes without a key.
"""

import os

import pytest

from fault_triage.llm_client import get_client, get_model, load_env

load_env()

pytestmark = pytest.mark.skipif(
    not os.getenv("OPENROUTER_API_KEY", "").strip(),
    reason="OPENROUTER_API_KEY not set; copy .env.example to .env to run the live test",
)


def test_openrouter_replies():
    client = get_client()
    response = client.chat.completions.create(
        model=get_model(),
        messages=[{"role": "user", "content": "Reply with the single word: ok"}],
        # Reasoning models spend tokens thinking before they answer, so leave room.
        max_tokens=500,
    )
    choice = response.choices[0]
    reply = choice.message.content
    assert reply and reply.strip(), f"empty reply (finish_reason={choice.finish_reason})"
