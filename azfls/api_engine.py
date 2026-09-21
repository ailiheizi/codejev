"""OpenAI-compatible chat-completions executor for candidate selection."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable

from azfls.model import Stats


@dataclass(frozen=True)
class APIConfig:
    """Connection settings for an OpenAI-compatible chat-completions endpoint."""

    base_url: str
    api_key: str = field(repr=False)
    model: str
    timeout_seconds: float = 90.0
    reasoning_effort: str | None = "none"


class APIEngineError(RuntimeError):
    """A safe, non-retryable API executor failure."""


class OpenAICompatibleEngine:
    """Execute chat-completions requests without a provider SDK."""

    def __init__(
        self,
        config: APIConfig,
        *,
        opener: Callable[..., Any] | None = None,
    ) -> None:
        self.config = config
        self._opener = opener if opener is not None else urllib.request.urlopen

    def generate(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 512,
    ) -> tuple[str, Stats]:
        """Send one request and return the assistant content and normalized stats."""

        started = time.perf_counter()
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": max_tokens,
        }
        if self.config.reasoning_effort is not None:
            payload["reasoning_effort"] = self.config.reasoning_effort

        try:
            encoded_payload = json.dumps(payload).encode("utf-8")
        except (TypeError, ValueError):
            raise APIEngineError("API request payload is not JSON-serializable") from None

        request = urllib.request.Request(
            self._completion_url(),
            data=encoded_payload,
            headers={
                "Accept": "application/json",
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        response: Any
        try:
            response = self._opener(request, timeout=self.config.timeout_seconds)
        except urllib.error.HTTPError as exc:
            status = exc.code if isinstance(exc.code, int) else None
            if status is None:
                raise APIEngineError("API HTTP error") from None
            raise APIEngineError(f"API HTTP error {status}") from None
        except urllib.error.URLError:
            raise APIEngineError("API network error") from None
        except (OSError, TimeoutError):
            raise APIEngineError("API network error") from None
        except Exception:
            raise APIEngineError("API request failed") from None

        try:
            status = getattr(response, "status", None)
            if isinstance(status, int) and status >= 400:
                raise APIEngineError(f"API HTTP error {status}")
            raw_body = response.read()
        except APIEngineError:
            raise
        except urllib.error.HTTPError as exc:
            status = exc.code if isinstance(exc.code, int) else None
            if status is None:
                raise APIEngineError("API HTTP error") from None
            raise APIEngineError(f"API HTTP error {status}") from None
        except urllib.error.URLError:
            raise APIEngineError("API network error") from None
        except (OSError, TimeoutError):
            raise APIEngineError("API network error") from None
        except Exception:
            raise APIEngineError("API response could not be read") from None
        finally:
            close = getattr(response, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass

        body = self._parse_json(raw_body)
        content, prompt_tokens, completion_tokens = self._parse_response(body)
        if self.config.api_key:
            content = content.replace(self.config.api_key, "[REDACTED]")
        stats = Stats(
            load_seconds=0.0,
            prompt_tokens=prompt_tokens,
            generated_tokens=completion_tokens,
            generate_seconds=time.perf_counter() - started,
        )
        return content, stats

    def _completion_url(self) -> str:
        base_url = self.config.base_url.rstrip("/")
        suffix = "/chat/completions"
        if base_url.endswith(suffix):
            return base_url
        return f"{base_url}{suffix}"

    @staticmethod
    def _parse_json(raw_body: Any) -> Any:
        try:
            if isinstance(raw_body, bytes):
                raw_body = raw_body.decode("utf-8")
            return json.loads(raw_body)
        except (UnicodeError, TypeError, ValueError):
            raise APIEngineError("API response is not valid JSON") from None

    @staticmethod
    def _parse_response(body: Any) -> tuple[str, int, int]:
        if not isinstance(body, dict):
            raise APIEngineError("API response has invalid structure")

        choices = body.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            raise APIEngineError("API response is missing choices[0].message.content")
        message = choices[0].get("message")
        if not isinstance(message, dict):
            raise APIEngineError("API response is missing choices[0].message.content")
        content = message.get("content")
        if not isinstance(content, str):
            raise APIEngineError("API response is missing choices[0].message.content")
        if not content:
            raise APIEngineError("API response content is empty")

        usage = body.get("usage")
        if usage is None:
            return content, 0, 0
        if not isinstance(usage, dict):
            raise APIEngineError("API response has invalid usage")
        prompt_tokens = OpenAICompatibleEngine._token_count(usage, "prompt_tokens")
        completion_tokens = OpenAICompatibleEngine._token_count(usage, "completion_tokens")
        return content, prompt_tokens, completion_tokens

    @staticmethod
    def _token_count(usage: dict[str, Any], key: str) -> int:
        value = usage.get(key, 0)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise APIEngineError("API response has invalid usage")
        return value
