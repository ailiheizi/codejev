"""TypesafeEngine（Jev）的离线测试：不联网、不打真实 API。"""

from __future__ import annotations

import io
import json
import urllib.error

import pytest

from codejev.candidate import CandidatePage, CodeCandidate, CodeTask
from codejev.jev_engine import TypesafeConfig, TypesafeEngine, TypesafeError

TASK = CodeTask(
    operation="filter_and_project",
    target_language="python",
    requirements=("keep rows where active is true", "return id and name"),
    constraints=("preserve input order",),
)

CANDIDATES = (
    CodeCandidate("0", "filter_and_project", "过滤 active 并投影 id/name", "result = []\nreturn result"),
    CodeCandidate("1", "sort_rows", "按 score 降序排序", "return sorted(rows)"),
    CodeCandidate("2", "group_and_count", "按 category 分组计数", "counts = {}\nreturn counts"),
)
PAGE = CandidatePage(CANDIDATES)

KEY = "sk-test-key-must-not-leak"


def engine_with(payload: dict | None, *, status: int = 200):
    """构造一个用假 opener 的执行器；记录请求以便断言。"""
    seen: list[dict] = []

    def opener(request, timeout=None):  # noqa: ANN001
        seen.append(
            {
                "url": request.full_url,
                "headers": dict(request.headers),
                "body": json.loads(request.data.decode("utf-8")),
                "timeout": timeout,
            }
        )
        if status != 200:
            raise urllib.error.HTTPError(request.full_url, status, "err", {}, io.BytesIO(b""))
        raw = json.dumps(payload or {}).encode("utf-8")

        class Response(io.BytesIO):
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        return Response(raw)

    engine = TypesafeEngine(
        TypesafeConfig(api_key=KEY, model="jev-latest"), opener=opener
    )
    return engine, seen


def answer(choice: str, confidence: float = 0.9, keys=("0", "1", "2", "NONE")) -> dict:
    probs = {k: (confidence if k == choice else (1 - confidence) / max(len(keys) - 1, 1)) for k in keys}
    return {"answers": {"candidate": {"type": "choice", "choice": choice, "confidence": confidence, "probabilities": probs}},
            "usage": {"input_tokens": 419, "output_tokens": 40}}


# -- build_request ---------------------------------------------------------


def test_build_request_includes_every_candidate_and_none_option() -> None:
    engine, _ = engine_with(answer("0"))
    body = engine.build_request(TASK, PAGE)
    criteria = body["questions"]["candidate"]["criteria"]
    assert set(criteria) == {"0", "1", "2", "NONE"}
    assert criteria["1"]["purpose"] == "按 score 降序排序"
    assert criteria["1"]["code"] == "return sorted(rows)"
    assert criteria["NONE"]["code"] == ""


def test_build_request_carries_requirements_and_constraints() -> None:
    engine, _ = engine_with(answer("0"))
    instructions = engine.build_request(TASK, PAGE)["questions"]["candidate"]["instructions"]
    assert "keep rows where active is true" in instructions["goal"]
    assert any("preserve input order" in rule for rule in instructions["rules"])


def test_build_request_rejects_invalid_page() -> None:
    engine, _ = engine_with(answer("0"))
    bad = CandidatePage((CodeCandidate("0", "a", "p", "  "),))
    with pytest.raises(Exception):
        engine.build_request(TASK, bad)


# -- parse_answer ----------------------------------------------------------


def test_parse_answer_returns_host_owned_candidate() -> None:
    engine, _ = engine_with(answer("1"))
    choice = engine.parse_answer(answer("1"), PAGE)
    assert choice.candidate_id == "1"
    assert choice.candidate is PAGE.candidates[1]


def test_parse_answer_accepts_none() -> None:
    engine, _ = engine_with(answer("NONE"))
    choice = engine.parse_answer(answer("NONE"), PAGE)
    assert choice.candidate_id is None
    assert choice.candidate is None


def test_parse_answer_rejects_choice_outside_the_page() -> None:
    engine, _ = engine_with(answer("0"))
    with pytest.raises(TypesafeError, match="候选页外"):
        engine.parse_answer({"answers": {"candidate": {"choice": "99"}}}, PAGE)


def test_parse_answer_rejects_missing_answer() -> None:
    engine, _ = engine_with(answer("0"))
    with pytest.raises(TypesafeError, match="answers"):
        engine.parse_answer({}, PAGE)
    with pytest.raises(TypesafeError, match="answers.candidate"):
        engine.parse_answer({"answers": {}}, PAGE)


def test_parse_answer_rejects_bad_confidence() -> None:
    engine, _ = engine_with(answer("0"))
    with pytest.raises(TypesafeError, match="confidence"):
        engine.parse_answer({"answers": {"candidate": {"choice": "0", "confidence": 1.5}}}, PAGE)
    with pytest.raises(TypesafeError, match="confidence"):
        engine.parse_answer({"answers": {"candidate": {"choice": "0", "confidence": True}}}, PAGE)


def test_parse_answer_rejects_probability_mismatch() -> None:
    engine, _ = engine_with(answer("0"))
    with pytest.raises(TypesafeError, match="键与候选页不一致"):
        engine.parse_answer(
            {"answers": {"candidate": {"choice": "0", "probabilities": {"0": 1.0}}}}, PAGE
        )
    with pytest.raises(TypesafeError, match="之和不为 1"):
        engine.parse_answer(
            {
                "answers": {
                    "candidate": {
                        "choice": "0",
                        "probabilities": {"0": 0.5, "1": 0.5, "2": 0.5, "NONE": 0.5},
                    }
                }
            },
            PAGE,
        )


# -- choose ----------------------------------------------------------------


def test_choose_end_to_end_and_stats() -> None:
    engine, seen = engine_with(answer("2"))
    choice, stats = engine.choose(TASK, PAGE)
    assert choice.candidate_id == "2"
    assert stats.prompt_tokens == 419
    assert stats.generated_tokens == 40
    assert stats.generate_seconds >= 0
    assert len(seen) == 1
    assert seen[0]["url"].endswith("/v1/systemone")
    assert seen[0]["body"]["model"] == "jev-latest"


def test_choose_sends_bearer_key_but_never_returns_it() -> None:
    engine, seen = engine_with(answer("0"))
    choice, stats = engine.choose(TASK, PAGE)
    auth = {k.lower(): v for k, v in seen[0]["headers"].items()}.get("authorization", "")
    assert KEY in auth
    for value in (repr(choice), repr(stats), str(choice.raw_response), repr(engine.config)):
        assert KEY not in value


def test_choose_does_not_retry_and_raises_safely_on_http_error() -> None:
    engine, seen = engine_with({}, status=429)
    with pytest.raises(TypesafeError, match="429"):
        engine.choose(TASK, PAGE)
    assert len(seen) == 1


def test_choose_raises_safely_on_network_error() -> None:
    def opener(request, timeout=None):  # noqa: ANN001
        raise urllib.error.URLError("boom")

    engine = TypesafeEngine(TypesafeConfig(api_key=KEY), opener=opener)
    with pytest.raises(TypesafeError, match="网络错误") as excinfo:
        engine.choose(TASK, PAGE)
    assert KEY not in str(excinfo.value)


def test_choose_raises_on_non_json_response() -> None:
    class Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    engine = TypesafeEngine(
        TypesafeConfig(api_key=KEY), opener=lambda request, timeout=None: Response(b"<html>")
    )
    with pytest.raises(TypesafeError, match="合法 JSON"):
        engine.choose(TASK, PAGE)


def test_config_repr_hides_api_key() -> None:
    assert KEY not in repr(TypesafeConfig(api_key=KEY))
