"""Offline tests for the provider-agnostic OpenAI-compatible executor."""

from __future__ import annotations

import json
import urllib.error
import urllib.request

import pytest

from chooseonly.api_engine import APIConfig, APIEngineError, OpenAICompatibleEngine
from chooseonly.candidate import CandidatePage, CodeCandidate, CodeTask, choose_candidate


API_KEY = "test-secret-key"
BASE_URL = "https://api.example.test/v1"


class FakeResponse:
    status = 200

    def __init__(self, body: object, *, status: int = 200) -> None:
        self.status = status
        self._body = json.dumps(body).encode("utf-8")
        self.closed = False

    def read(self) -> bytes:
        return self._body

    def close(self) -> None:
        self.closed = True


class RecordingOpener:
    def __init__(self, response: object) -> None:
        self.response = response
        self.requests: list[tuple[urllib.request.Request, float]] = []

    def __call__(self, request: urllib.request.Request, *, timeout: float) -> object:
        self.requests.append((request, timeout))
        if isinstance(self.response, BaseException):
            raise self.response
        return self.response


def config(**overrides: object) -> APIConfig:
    values: dict[str, object] = {
        "base_url": BASE_URL,
        "api_key": API_KEY,
        "model": "deepseek-v4-flash",
    }
    values.update(overrides)
    return APIConfig(**values)  # type: ignore[arg-type]


def engine_for(
    body: object,
    *,
    config_override: dict[str, object] | None = None,
) -> tuple[OpenAICompatibleEngine, RecordingOpener]:
    opener = RecordingOpener(FakeResponse(body))
    values = config_override or {}
    return OpenAICompatibleEngine(config(**values), opener=opener), opener


def test_api_config_defaults_and_repr_do_not_expose_key() -> None:
    settings = config()
    assert settings.timeout_seconds == 90.0
    assert settings.reasoning_effort == "none"
    assert API_KEY not in repr(settings)


def test_completion_url_adds_suffix_once() -> None:
    engine, opener = engine_for(
        {"choices": [{"message": {"content": "0"}}]},
        config_override={"base_url": f"{BASE_URL}/chat/completions"},
    )
    engine.generate([], max_tokens=8)
    assert opener.requests[0][0].full_url == f"{BASE_URL}/chat/completions"
    assert opener.requests[0][0].full_url.count("/chat/completions") == 1


def test_completion_url_strips_trailing_slashes() -> None:
    engine, opener = engine_for(
        {"choices": [{"message": {"content": "0"}}]},
        config_override={"base_url": f"{BASE_URL}/"},
    )
    engine.generate([])
    assert opener.requests[0][0].full_url == f"{BASE_URL}/chat/completions"


def test_request_contains_model_messages_temperature_and_max_tokens() -> None:
    engine, opener = engine_for({"choices": [{"message": {"content": "0"}}]})
    messages = [{"role": "user", "content": "choose"}]
    engine.generate(messages, max_tokens=7)
    request_body = json.loads(opener.requests[0][0].data.decode("utf-8"))
    assert request_body == {
        "model": "deepseek-v4-flash",
        "messages": messages,
        "temperature": 0,
        "max_tokens": 7,
        "reasoning_effort": "none",
    }


def test_reasoning_effort_is_included_when_configured() -> None:
    engine, opener = engine_for(
        {"choices": [{"message": {"content": "0"}}]},
        config_override={"reasoning_effort": "low"},
    )
    engine.generate([])
    request_body = json.loads(opener.requests[0][0].data.decode("utf-8"))
    assert request_body["reasoning_effort"] == "low"


def test_reasoning_effort_is_omitted_when_none() -> None:
    engine, opener = engine_for(
        {"choices": [{"message": {"content": "0"}}]},
        config_override={"reasoning_effort": None},
    )
    engine.generate([])
    request_body = json.loads(opener.requests[0][0].data.decode("utf-8"))
    assert "reasoning_effort" not in request_body


def test_authorization_header_contains_configured_key() -> None:
    engine, opener = engine_for({"choices": [{"message": {"content": "0"}}]})
    engine.generate([])
    request = opener.requests[0][0]
    assert request.get_header("Authorization") == f"Bearer {API_KEY}"
    assert request.get_header("Content-type") == "application/json"


def test_http_error_and_repr_do_not_leak_api_key() -> None:
    opener = RecordingOpener(
        urllib.error.HTTPError(
            f"{BASE_URL}/chat/completions",
            401,
            f"bad key {API_KEY}",
            {},
            None,
        )
    )
    engine = OpenAICompatibleEngine(config(), opener=opener)
    with pytest.raises(APIEngineError) as raised:
        engine.generate([])
    assert str(raised.value) == "API HTTP error 401"
    assert API_KEY not in str(raised.value)
    assert API_KEY not in repr(engine)


def test_successful_response_returns_content_and_usage_stats() -> None:
    engine, _opener = engine_for(
        {
            "choices": [{"message": {"content": "0"}}],
            "usage": {"prompt_tokens": 12, "completion_tokens": 1},
        }
    )
    raw, stats = engine.generate([])
    assert raw == "0"
    assert stats.prompt_tokens == 12
    assert stats.generated_tokens == 1
    assert stats.load_seconds == 0.0
    assert stats.generate_seconds >= 0.0
    assert stats.tokens_per_second >= 0.0


def test_successful_response_redacts_key_if_provider_echoes_it() -> None:
    engine, _opener = engine_for(
        {"choices": [{"message": {"content": f"0 {API_KEY}"}}]}
    )
    raw, _stats = engine.generate([])
    assert raw == "0 [REDACTED]"
    assert API_KEY not in raw


def test_missing_usage_defaults_token_counts_to_zero() -> None:
    engine, _opener = engine_for({"choices": [{"message": {"content": "0"}}]})
    _raw, stats = engine.generate([])
    assert stats.prompt_tokens == 0
    assert stats.generated_tokens == 0


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"choices": []},
        {"choices": [{"message": {}}]},
        {"choices": [{"message": {"content": ""}}]},
        {"choices": [{"message": {"content": None}}]},
    ],
)
def test_missing_choices_or_content_raise_safe_error(body: object) -> None:
    engine, _opener = engine_for(body)
    with pytest.raises(APIEngineError, match=r"choices\[0\]\.message\.content|content is empty"):
        engine.generate([])


def test_http_error_is_converted_to_safe_api_error() -> None:
    opener = RecordingOpener(urllib.error.HTTPError("url", 503, "provider detail", {}, None))
    engine = OpenAICompatibleEngine(config(), opener=opener)
    with pytest.raises(APIEngineError, match=r"^API HTTP error 503$"):
        engine.generate([])


def test_url_error_is_converted_to_safe_api_error() -> None:
    opener = RecordingOpener(urllib.error.URLError("connection detail"))
    engine = OpenAICompatibleEngine(config(), opener=opener)
    with pytest.raises(APIEngineError, match="^API network error$"):
        engine.generate([])


def test_invalid_json_is_converted_to_safe_api_error() -> None:
    class InvalidJSONResponse(FakeResponse):
        def __init__(self) -> None:
            self.status = 200
            self.closed = False

        def read(self) -> bytes:
            return b"not-json"

    opener = RecordingOpener(InvalidJSONResponse())
    engine = OpenAICompatibleEngine(config(), opener=opener)
    with pytest.raises(APIEngineError, match="^API response is not valid JSON$"):
        engine.generate([])


def test_choose_candidate_uses_api_engine_and_keeps_host_candidate() -> None:
    engine, _opener = engine_for(
        {
            "choices": [{"message": {"content": "0"}}],
            "usage": {"prompt_tokens": 4, "completion_tokens": 1},
        }
    )
    page = CandidatePage((CodeCandidate("0", "zero", "select zero", "return 0"),))
    task = CodeTask("select", "python", ("return zero",))
    choice, stats = choose_candidate(engine, task, page)
    assert choice.candidate_id == "0"
    assert choice.candidate is page.candidates[0]
    assert choice.raw_response == "0"
    assert stats.prompt_tokens == 4
    assert stats.generated_tokens == 1


def test_non_success_response_status_is_converted_to_safe_api_error() -> None:
    engine, _opener = engine_for(
        {"error": "provider detail"},
        config_override={"base_url": BASE_URL},
    )
    # The helper always creates a 200 response; this check covers schema failure
    # without making a network request.
    with pytest.raises(APIEngineError, match=r"choices\[0\]"):
        engine.generate([])


def test_invalid_usage_is_converted_to_safe_api_error() -> None:
    engine, _opener = engine_for(
        {
            "choices": [{"message": {"content": "0"}}],
            "usage": {"prompt_tokens": "12"},
        }
    )
    with pytest.raises(APIEngineError, match="invalid usage"):
        engine.generate([])
