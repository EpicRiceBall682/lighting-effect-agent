"""Minimal DeepSeek chat-completions client using the Python standard library."""

from __future__ import annotations

import json
import os
import time
from typing import Any, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DEFAULT_ENDPOINT = "https://api.deepseek.com/chat/completions"
DEFAULT_MODEL = "deepseek-v4-flash"


class ModelClientError(RuntimeError):
    """Base exception for language-model request or response failures."""


class ModelConfigurationError(ModelClientError):
    """Raised when required local model configuration is missing."""


class ModelResponseError(ModelClientError):
    """Raised when the model API returns a malformed or unsuccessful response."""


def decode_json_content(content: str) -> dict[str, Any]:
    """Decode JSON mode output, tolerating accidental Markdown fences."""

    cleaned = content.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()

    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ModelResponseError("model response was not valid JSON") from exc
    if not isinstance(value, dict):
        raise ModelResponseError("model response JSON must be an object")
    return value


def _assistant_content(response_data: Any) -> str:
    try:
        content = response_data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ModelResponseError("DeepSeek response has no assistant content") from exc
    if not isinstance(content, str):
        raise ModelResponseError("DeepSeek assistant content was not text")
    if not content.strip():
        raise ModelResponseError("DeepSeek returned empty assistant content")
    return content


class DeepSeekClient:
    """Call the official DeepSeek OpenAI-compatible Chat Completions API."""

    def __init__(
        self,
        *,
        token: str | None = None,
        model: str | None = None,
        endpoint: str | None = None,
        timeout_seconds: float = 45.0,
        max_retries: int = 2,
    ) -> None:
        self.token = token or os.getenv("DEEPSEEK_API_KEY")
        self.model = model or os.getenv("DEEPSEEK_MODEL", DEFAULT_MODEL)
        self.endpoint = endpoint or os.getenv("DEEPSEEK_ENDPOINT", DEFAULT_ENDPOINT)
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries

        if not self.token:
            raise ModelConfigurationError(
                "missing DeepSeek API key; set DEEPSEEK_API_KEY before running module 1"
            )
        if not self.endpoint.startswith("https://"):
            raise ModelConfigurationError("DeepSeek endpoint must use HTTPS")
        if self.max_retries < 0:
            raise ModelConfigurationError("max_retries must be zero or greater")

    def complete_json(self, messages: Sequence[Mapping[str, str]]) -> dict[str, Any]:
        payload = {
            "model": self.model,
            "messages": list(messages),
            "response_format": {"type": "json_object"},
            # V4 enables thinking by default. This task is a short deterministic
            # transformation, so disabling it reduces latency and token usage.
            "thinking": {"type": "disabled"},
            "temperature": 0.2,
            "max_tokens": 600,
        }
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = Request(
            self.endpoint,
            data=body,
            method="POST",
            headers={
                "Accept": "application/json",
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json",
                "User-Agent": "lighting-effect-agent-module-1",
            },
        )

        for attempt in range(self.max_retries + 1):
            try:
                with urlopen(request, timeout=self.timeout_seconds) as response:
                    response_data = json.loads(response.read().decode("utf-8"))
                return decode_json_content(_assistant_content(response_data))
            except HTTPError as exc:
                error_body = exc.read().decode("utf-8", errors="replace")[:800]
                retryable = exc.code == 429 or 500 <= exc.code < 600
                if not retryable or attempt >= self.max_retries:
                    raise ModelResponseError(
                        f"DeepSeek request failed with HTTP {exc.code}: {error_body}"
                    ) from exc
            except (URLError, TimeoutError) as exc:
                if attempt >= self.max_retries:
                    reason = exc.reason if isinstance(exc, URLError) else exc
                    raise ModelResponseError(
                        f"DeepSeek request failed after retries: {reason}"
                    ) from exc
            except json.JSONDecodeError as exc:
                if attempt >= self.max_retries:
                    raise ModelResponseError("DeepSeek returned invalid response JSON") from exc
            except ModelResponseError:
                if attempt >= self.max_retries:
                    raise

            time.sleep(2**attempt)

        raise ModelResponseError("DeepSeek returned no usable response")
