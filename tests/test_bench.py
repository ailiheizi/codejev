"""bench/compare.py 的测试：生成路线与选择路线的判断标准。

全部离线运行：用 StubEngine 直接给出预设回复，不加载权重、不联网、不写盘。
重点验证“通过”是真的按运行结果判定，而不是只看 id 是否合法——
选择路线的错字段（id 合法但语义错）必须被判为失败。
"""

from __future__ import annotations

import ast

import pytest

from azfls.model import Stats
from bench.compare import (
    BENCH_SOURCE,
    EXPECTED_FIELDS,
    EXPECTED_IDS,
    TASK_FUNCTION,
    Outcome,
    Report,
    check_runtime_behaviour,
    looks_like_explanation,
    looks_truncated,
    run_generate,
    run_select,
)

# 固定任务用的样本：只留 active 为真的项、只返回 id 和 name。
SAMPLE = [
    {"id": 1, "name": "A", "active": True, "extra": "x"},
    {"id": 2, "name": "B", "active": False, "extra": "y"},
    {"id": 3, "name": "C", "active": True, "extra": "z"},
]
EXPECTED_OUTPUT = [{"id": 1, "name": "A"}, {"id": 3, "name": "C"}]

# 一份“做对了”的整份文件正文：生成路线的正确产物。
GOOD_BODY = '''"""用户列表：只留 active 的项。"""


def active_users(users):
    """返回用户，保持原顺序。"""
    result = []
    for user in users:
        if user["active"]:
            result.append({"id": user["id"], "name": user["name"]})
    return result
'''

# 选择路线的固定候选：BENCH_SOURCE 里 f0=id、f1=name、f2=active；条件同理。
VALID_DECISION = '{"function": "fn0", "filter_field": "c2", "return_fields": ["f0", "f1"]}'


class StubEngine:
    """离线替身：按预设回复返回，实现 azfls.model.Engine 协议，不加载模型。"""

    def __init__(self, *replies: str) -> None:
        self.replies = list(replies)
        self.calls: list[tuple[list[dict[str, str]], int]] = []

    def generate(
        self, messages: list[dict[str, str]], max_tokens: int = 512
    ) -> tuple[str, Stats]:
        self.calls.append((messages, max_tokens))
        return self.replies.pop(0), Stats(prompt_tokens=10, generated_tokens=8, generate_seconds=0.5)


def _call(source: str, argument: object) -> object:
    """把源码 exec 起来并调用目标函数，拿到真实运行结果。"""
    namespace: dict[str, object] = {}
    exec(compile(source, "<bench-candidate>", "exec"), namespace)  # noqa: S102 - 测试专用
    return namespace[TASK_FUNCTION](argument)  # type: ignore[operator]


def _outcome(path: str, ok: bool, reasons: tuple[str, ...], seconds: float = 0.0) -> Outcome:
    """手工造一个 Outcome，用于 Report 的展示逻辑测试。"""
    return Outcome(path, ok, reasons, None, Stats(generate_seconds=seconds), seconds)


# --------------------------------------------------------------------------
# check_runtime_behaviour：唯一可信的判断是“真的跑一遍”
# --------------------------------------------------------------------------


def test_runtime_behaviour_accepts_correct_function() -> None:
    ok, reasons = check_runtime_behaviour(GOOD_BODY, TASK_FUNCTION)
    assert ok is True
    assert reasons == ()


@pytest.mark.parametrize(
    ("label", "source"),
    [
        # 不过滤：三个 id 全返回。
        (
            "no_filter",
            "def active_users(users):\n"
            "    return [{'id': u['id'], 'name': u['name']} for u in users]\n",
        ),
        # 过滤对了，但多返回一个字段。
        (
            "extra_field",
            "def active_users(users):\n"
            "    return [\n"
            "        {'id': u['id'], 'name': u['name'], 'active': u['active']}\n"
            "        for u in users\n"
            "        if u['active']\n"
            "    ]\n",
        ),
        # 语法不合法：def 行少冒号。
        ("syntax_error", "def active_users(users)\n    return []\n"),
        # 没有这个函数名。
        ("missing_function", "def other(users):\n    return []\n"),
        # 一调用就抛异常。
        ("raises", "def active_users(users):\n    raise ValueError('boom')\n"),
    ],
)
def test_runtime_behaviour_rejects_bad_function(label: str, source: str) -> None:
    ok, reasons = check_runtime_behaviour(source, TASK_FUNCTION)
    assert ok is False, label
    assert reasons, label


def test_runtime_behaviour_reports_syntax_error() -> None:
    ok, reasons = check_runtime_behaviour("def active_users(users)\n    return []\n", TASK_FUNCTION)
    assert ok is False
    assert any("语法错误" in reason for reason in reasons)


def test_runtime_behaviour_reports_call_failure() -> None:
    ok, reasons = check_runtime_behaviour(
        "def active_users(users):\n    raise ValueError('boom')\n", TASK_FUNCTION
    )
    assert ok is False
    assert any("调用失败" in reason for reason in reasons)


def test_runtime_behaviour_separates_filter_and_field_checks() -> None:
    """过滤对、字段错时只报字段问题，两条判断彼此独立。"""
    source = (
        "def active_users(users):\n"
        "    return [\n"
        "        {'id': u['id'], 'name': u['name'], 'active': u['active']}\n"
        "        for u in users\n"
        "        if u['active']\n"
        "    ]\n"
    )
    ok, reasons = check_runtime_behaviour(source, TASK_FUNCTION)
    assert ok is False
    assert any("字段不正确" in reason for reason in reasons)
    assert not any("过滤不正确" in reason for reason in reasons)


# --------------------------------------------------------------------------
# looks_truncated / looks_like_explanation：原始输出的形状检查
# --------------------------------------------------------------------------


def test_looks_truncated_detects_unclosed_fence() -> None:
    assert looks_truncated("```python\ndef f():\n    return 1\n") is True


def test_looks_truncated_accepts_balanced_fence() -> None:
    assert looks_truncated("```python\ndef f():\n    return 1\n```") is False
    assert looks_truncated("def f():\n    return 1\n") is False  # 没有围栏也算闭合


def test_looks_like_explanation_detects_chinese_prose() -> None:
    assert looks_like_explanation("好的，以下是修改后的代码：\n```python\ndef f(): ...\n```") is True
    assert looks_like_explanation("以下是修改结果：") is True


def test_looks_like_explanation_accepts_plain_body() -> None:
    assert looks_like_explanation(GOOD_BODY) is False


# --------------------------------------------------------------------------
# run_select：宿主组装，语义由宿主核对
# --------------------------------------------------------------------------


def test_run_select_accepts_valid_decision_and_assembles_working_code() -> None:
    engine = StubEngine(VALID_DECISION)
    report = run_select(engine, BENCH_SOURCE, 1)
    outcome = report.results[0]

    assert outcome.path == "select"
    assert outcome.ok is True, outcome.reasons
    assert outcome.reasons == ()
    assert outcome.artifact is not None

    body = outcome.artifact.body
    ast.parse(body)  # 组装出来的必须是合法 Python
    assert _call(body, SAMPLE) == EXPECTED_OUTPUT  # 并且真的按指令工作
    assert len(engine.calls) == 1  # 只问一次，不重试
    assert engine.calls[0][1] <= 128  # 只要一个极短的决策


def test_run_select_rejects_bogus_candidate_id_without_raising() -> None:
    """未知 id 不合法：记为失败并给出原因，不把异常抛给调用方。"""
    engine = StubEngine('{"function": "fn0", "filter_field": "c9", "return_fields": ["f0"]}')
    report = run_select(engine, BENCH_SOURCE, 1)
    outcome = report.results[0]

    assert outcome.ok is False
    assert any("决策不合法" in reason for reason in outcome.reasons)


def test_run_select_rejects_wrong_fields_even_with_valid_ids() -> None:
    """id 合法不代表选得对：f0/f2 = id/active，宿主必须按真实语义判失败。"""
    engine = StubEngine('{"function": "fn0", "filter_field": "c2", "return_fields": ["f0", "f2"]}')
    report = run_select(engine, BENCH_SOURCE, 1)
    outcome = report.results[0]

    assert outcome.ok is False
    assert any("选的字段不是期望值" in reason for reason in outcome.reasons)
    assert outcome.artifact is not None  # 组装仍成功了，失败来自语义核对


def test_run_select_rejects_wrong_filter_field_even_with_valid_id() -> None:
    """过滤条件同样按真实名字核对：c0 是 id，不是 active。"""
    engine = StubEngine('{"function": "fn0", "filter_field": "c0", "return_fields": ["f0", "f1"]}')
    report = run_select(engine, BENCH_SOURCE, 1)
    outcome = report.results[0]

    assert outcome.ok is False
    assert any("过滤字段不是期望值" in reason for reason in outcome.reasons)


# --------------------------------------------------------------------------
# run_generate：自由生成整份正文
# --------------------------------------------------------------------------


def test_run_generate_accepts_correct_body() -> None:
    engine = StubEngine(GOOD_BODY)
    report = run_generate(engine, BENCH_SOURCE, 1)
    outcome = report.results[0]

    assert outcome.path == "generate"
    assert outcome.ok is True, outcome.reasons
    assert outcome.artifact is not None
    assert _call(outcome.artifact.body, SAMPLE) == EXPECTED_OUTPUT


def test_run_generate_rejects_unfiltered_body() -> None:
    """小模型把原文照抄回来：形状没变，运行结果不对，判失败。"""
    engine = StubEngine(BENCH_SOURCE)
    report = run_generate(engine, BENCH_SOURCE, 1)
    outcome = report.results[0]

    assert outcome.ok is False
    assert any("过滤不正确" in reason for reason in outcome.reasons)


def test_run_generate_flags_explanation_and_truncation() -> None:
    """带说明、围栏没闭合：即使正文能用，也按原始输出的问题记失败。"""
    raw = "好的，以下是修改后的代码：\n```python\n" + GOOD_BODY
    engine = StubEngine(raw)
    report = run_generate(engine, BENCH_SOURCE, 1)
    outcome = report.results[0]

    assert outcome.ok is False
    assert any("说明文字" in reason for reason in outcome.reasons)
    assert any("截断" in reason for reason in outcome.reasons)


# --------------------------------------------------------------------------
# Report：一行一路径；明细只列失败
# --------------------------------------------------------------------------


def test_report_summary_is_one_line_per_path() -> None:
    report = Report(trials=2)
    report.add(_outcome("generate", True, (), seconds=0.5))
    report.add(_outcome("select", True, (), seconds=0.02))

    lines = report.summary().splitlines()
    assert len(lines) == 2
    assert "generate" in lines[0] and "通过 1/1" in lines[0]
    assert "select" in lines[1] and "通过 1/1" in lines[1]


def test_report_summary_omits_paths_without_results() -> None:
    report = Report(trials=1)
    report.add(_outcome("select", True, ()))
    assert len(report.summary().splitlines()) == 1


def test_report_failure_detail_lists_only_failures() -> None:
    report = Report(trials=2)
    report.add(_outcome("generate", True, ()))
    report.add(_outcome("select", False, ("甲", "乙")))

    assert report.failure_detail() == "[select] 甲, 乙"


def test_report_failure_detail_is_empty_when_all_pass() -> None:
    report = Report(trials=1)
    report.add(_outcome("generate", True, ()))
    assert report.failure_detail() == ""


# --------------------------------------------------------------------------
# BENCH_SOURCE：这必须是一次真实改动，而不是把已有行为抄一遍
# --------------------------------------------------------------------------


def test_bench_source_returns_every_field_and_does_not_filter() -> None:
    """基准函数不做任何过滤，返回全部字段：所以“加过滤 + 裁字段”是真任务。"""
    tree = ast.parse(BENCH_SOURCE)
    function = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == TASK_FUNCTION
    )
    # 函数体里没有任何 If：原文根本没有可抄的过滤。
    assert not any(isinstance(node, ast.If) for node in ast.walk(function))

    result = _call(BENCH_SOURCE, SAMPLE)
    assert [item["id"] for item in result] == [1, 2, 3]  # 三个都在，没有过滤
    assert all(set(item) == {"id", "name", "active"} for item in result)
    assert set(EXPECTED_FIELDS) == {"id", "name"}
    assert EXPECTED_IDS == [1, 3]
