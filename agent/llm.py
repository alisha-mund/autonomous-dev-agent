"""LLM client setup and a single helper for structured (Pydantic) calls.

Every agent talks to the model through `structured_call`, so switching between
Azure OpenAI and the OpenAI API is one env var (LLM_PROVIDER), not a code change.
"""

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, TypeVar

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

load_dotenv()

T = TypeVar("T", bound=BaseModel)


class LLMError(RuntimeError):
    """Raised when the model refuses or returns output that can't be parsed."""


@dataclass
class Usage:
    prompt_tokens: int
    completion_tokens: int

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


def _require(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def _provider() -> str:
    return os.getenv("LLM_PROVIDER", "azure").lower()


def _temperature() -> float | None:
    """Return LLM_TEMPERATURE as a float, or None to use the model's default.

    gpt-5 family models reject any temperature except the default, so leave it empty for them.
    """
    raw = os.getenv("LLM_TEMPERATURE", "").strip()
    return float(raw) if raw else None

def _reasoning_effort() -> str | None:
    """Return LLM_REASONING_EFFORT (e.g. 'minimal', 'low'), or None to not send it.

    Only reasoning models (gpt-5 family) accept this; leave it empty for gpt-4o-mini.
    Lower effort = fewer hidden reasoning tokens = lower cost.
    """
    raw = os.getenv("LLM_REASONING_EFFORT", "").strip().lower()
    return raw or None

@lru_cache(maxsize=1)
def get_client() -> OpenAI:
    """Create the client once and reuse it (it holds a connection pool).

    Azure exposes an OpenAI-compatible endpoint at <resource>/openai/v1/, so both
    providers use the same OpenAI client; only base_url and key differ.
    """
    if _provider() == "azure":
        endpoint = _require("AZURE_OPENAI_ENDPOINT").rstrip("/")
        return OpenAI(
            base_url=f"{endpoint}/openai/v1/",
            api_key=_require("AZURE_OPENAI_API_KEY"),
            max_retries=3,
            timeout=60,
        )
    if _provider() == "openai":
        return OpenAI(api_key=_require("OPENAI_API_KEY"), max_retries=3, timeout=60)
    raise RuntimeError(f"Unknown LLM_PROVIDER: {_provider()!r} (use 'azure' or 'openai')")


def get_model_name() -> str:
    """On Azure the 'model' argument is your deployment name, not the model name."""
    if _provider() == "azure":
        return _require("AZURE_OPENAI_DEPLOYMENT")
    return os.getenv("OPENAI_MODEL", "gpt-4o-mini")


def structured_call(system: str, user: str, schema: type[T]) -> tuple[T, Usage]:
    """Send one system+user prompt and get back a validated instance of `schema`."""
    extra: dict[str, Any] = {}
    temperature = _temperature()
    if temperature is not None:
        extra["temperature"] = temperature

    reasoning_effort = _reasoning_effort()
    if reasoning_effort is not None:
        extra["reasoning_effort"] = reasoning_effort

    completion = get_client().chat.completions.parse(
        model=get_model_name(),
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        response_format=schema,
        **extra,
    )
    message = completion.choices[0].message
    if message.refusal:
        raise LLMError(f"Model refused: {message.refusal}")
    if message.parsed is None:
        raise LLMError("Model returned no parseable output")

    usage = Usage(
        prompt_tokens=completion.usage.prompt_tokens if completion.usage else 0,
        completion_tokens=completion.usage.completion_tokens if completion.usage else 0,
    )
    return message.parsed, usage


if __name__ == "__main__":
    # Smoke test: python -m agent.llm
    class Ping(BaseModel):
        answer: str
        number: int

    result, usage = structured_call(
        system="You are a test endpoint. Reply in the requested format.",
        user="Say 'pong' and pick the number 42.",
        schema=Ping,
    )
    print(f"provider={_provider()} model={get_model_name()} temperature={_temperature()}")
    print(result)
    print(f"tokens: {usage.prompt_tokens} in / {usage.completion_tokens} out")