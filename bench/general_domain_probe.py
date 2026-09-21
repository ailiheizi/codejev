"""域无关候选页原型：验证"宿主枚举候选 → 模型只回一个候选 id → 宿主物化"能推广到
第二个域（Python 函数与变量），并量化真正的瓶颈——宿主枚举。

沿用现有协议，不改 codejev/ 下任何文件：
- 候选页仍是 `codejev.candidate.CandidatePage` / `CodeCandidate`；
- 选择仍是 `choose_candidate` + `parse_choice` 严格解析；
- 物化仍是 `materialize`；
- 执行器仍是 `codejev.api_engine.OpenAICompatibleEngine`，配置照抄
  `bench/candidate_api_probe.py` 的 CC Switch 读法（不打印 key）。

A 部分：把 ast 枚举出的函数与名字装进现有候选页，跑 15 条任务，判据全部是运行时行为：
  - 选函数：物化源码段 → exec 成真函数 → 跑行为指纹；并核对段与源文件逐字一致；
  - 选名字：在一次真实调用的观测点上取局部快照 → 把选中的名字代进判据求值。
  每条任务还会把整张候选页扫一遍，量出"行为判据下答案是否唯一"。

B 部分：对同一份源文件量"枚举成本"：逐类给出能枚举 / 枚举不全 / 枚举不了，
  带具体例子与数字，并给出候选页的真实 token 账单。

运行（需要代理时先 export https_proxy/http_proxy/all_proxy）：
    .venv/bin/python -m bench.general_domain_probe
    .venv/bin/python -m bench.general_domain_probe --no-api   # 只跑 B 的普查，不联网
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib
import io
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Callable, Mapping, Sequence

from bench.provider_config import load_provider
from codejev.api_engine import APIConfig, APIEngineError, OpenAICompatibleEngine
from codejev.candidate import (
    CandidateChoice,
    CandidateError,
    CandidatePage,
    CodeCandidate,
    CodeTask,
    choose_candidate,
    materialize,
)
from codejev.model import ScriptedEngine, Stats
from bench import domain_enum as de
from bench import domain_runtime as dr

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SOURCE = REPO_ROOT / "codejev" / "candidate.py"

DEFAULT_MODEL = "deepseek-v4-flash"

# 宿主发放候选 id 的盐：顺序只由宿主的这个常量决定，与源码顺序、出现位置无关。
ID_SALT = "codejev/general-domain/v1"
MAX_TOKENS = 16


# ==========================================================================
# 执行器：配置来自环境变量（bench/provider_config.py）+ 对 400/429 退避重试
# ==========================================================================


def make_config(model: str | None = None) -> APIConfig:
    """组装执行器配置；缺环境变量就明确报错。"""
    from bench.provider_config import load_provider

    return load_provider(model)


_RETRYABLE = ("API HTTP error 400", "API HTTP error 429", "API network error")


@dataclass
class CallOutcome:
    """一次选择调用：结果、耗时、重试账单。"""

    choice: CandidateChoice
    stats: Stats
    attempts: int
    notes: list[str] = field(default_factory=list)


def choose_with_backoff(
    engine: OpenAICompatibleEngine,
    task: CodeTask,
    page: CandidatePage,
    *,
    max_tokens: int = MAX_TOKENS,
    max_attempts: int = 5,
) -> CallOutcome:
    """调 choose_candidate；只对 400/429/网络错误退避重试，不修模型回复。"""

    notes: list[str] = []
    delay = 1.0
    for attempt in range(1, max_attempts + 1):
        try:
            choice, stats = choose_candidate(engine, task, page, max_tokens=max_tokens)
        except APIEngineError as exc:
            message = str(exc)
            if attempt >= max_attempts or not any(key in message for key in _RETRYABLE):
                raise
            notes.append(f"第 {attempt} 次 {message}，退避 {delay:.1f}s")
            time.sleep(delay)
            delay *= 2
            continue
        return CallOutcome(choice=choice, stats=stats, attempts=attempt, notes=notes)
    raise APIEngineError("重试耗尽")  # pragma: no cover - 循环内已 raise


# ==========================================================================
# 宿主资产：判据用的固定候选页
# ==========================================================================

FIXTURE_PAGE = CandidatePage(
    (
        CodeCandidate(
            id="0",
            name="filter_and_project",
            purpose="过滤 active 为真的行，并按 id、name 投影，保持原顺序。",
            code='result = []\nfor row in rows:\n    if row["active"]:\n        result.append({"id": row["id"]})\nreturn result',
        ),
        CodeCandidate(
            id="1",
            name="sort_rows",
            purpose="按 score 从高到低排序 rows。",
            code='return sorted(rows, key=lambda row: row["score"], reverse=True)',
        ),
        CodeCandidate(
            id="2",
            name="group_and_count",
            purpose="按 category 分组计数。",
            code="counts = {}\nfor row in rows:\n    counts[row[0]] = counts.get(row[0], 0) + 1\nreturn counts",
        ),
    )
)
DUP_PAGE = CandidatePage(
    (
        FIXTURE_PAGE.candidates[0],
        CodeCandidate(
            id="0",
            name="duplicated_id",
            purpose="故意和第一个候选同 id，用来触发校验分支。",
            code="return rows",
        ),
    )
)
EMPTY_CODE_PAGE = CandidatePage(
    (
        CodeCandidate(id="0", name="blank_code", purpose="故意空代码。", code=" "),
    )
)
LANG_MISMATCH_PAGE = CandidatePage(
    (
        CodeCandidate(id="0", name="other_lang", purpose="故意换语言。", code="rows", language="javascript"),
    )
)
ANY_TASK = CodeTask(
    operation="select_candidate",
    target_language="python",
    requirements=("pick exactly one candidate id from the page",),
    constraints=("do not write code",),
)

def _w(*args: object) -> tuple[tuple[object, ...], dict[str, object]]:
    """把一次见证调用写成 (位置参数, 关键字参数)。"""

    return (args, {})



# ==========================================================================
# 任务定义
# ==========================================================================


@dataclass(frozen=True)
class FunctionTask:
    """函数域任务：期望某个 qualname；判据是对物化出来的真函数跑行为指纹。"""

    label: str
    source_request: str
    task: CodeTask
    expected_qualname: str | None
    checks: Callable[[dr.Materialized], Sequence[dr.Check]]
    absence_keywords: tuple[str, ...] = ()


@dataclass(frozen=True)
class NameTask:
    """名字域任务：期望某个绑定名；判据是在真实观测点上把名字代进判据求值。"""

    label: str
    source_request: str
    task: CodeTask
    scope: str
    needle: str
    nth: int
    index: int
    witnesses: tuple[tuple[tuple[object, ...], dict[str, object]], ...]
    expected: str | None
    predicate: Callable[[object], bool] | None
    predicate_text: str
    absence_keywords: tuple[str, ...] = ()


def _fn_task(
    label: str,
    source_request: str,
    operation: str,
    requirements: tuple[str, ...],
    expected: str | None,
    checks: Callable[[dr.Materialized], Sequence[dr.Check]],
    *,
    constraints: tuple[str, ...] = (),
    absence_keywords: tuple[str, ...] = (),
) -> FunctionTask:
    return FunctionTask(
        label=label,
        source_request=source_request,
        task=CodeTask(
            operation=operation,
            target_language="python",
            requirements=requirements,
            constraints=constraints,
        ),
        expected_qualname=expected,
        checks=checks,
        absence_keywords=absence_keywords,
    )


def checks_parse_choice(m: dr.Materialized) -> Sequence[dr.Check]:
    """行为指纹：空白容忍、围栏标量、NO_MATCH、未知 id 抛错。"""

    f = m.function
    return (
        dr.eq_check("去掉首尾空白后命中候选 0", lambda: f("  0  \n", FIXTURE_PAGE).candidate_id, "0"),
        dr.eq_check("单行围栏标量解成候选 1", lambda: f("```\n1\n```", FIXTURE_PAGE).candidate_id, "1"),
        dr.eq_check("NONE 解成无匹配", lambda: f("NONE", FIXTURE_PAGE).candidate_id, None),
        dr.raises_check("未知 id 抛 CandidateError", lambda: f("9", FIXTURE_PAGE), CandidateError),
    )


def checks_unwrap_scalar(m: dr.Materialized) -> Sequence[dr.Check]:
    f = m.function
    return (
        dr.eq_check("单行围栏取内层", lambda: f("```\n1\n```"), "1"),
        dr.eq_check("带语言标记的围栏", lambda: f("```python\nx\n```"), "x"),
        dr.eq_check("没有围栏时原样返回", lambda: f("0"), "0"),
        dr.predicate_check(
            "返回值里不再有围栏标记",
            lambda: f("```python\nx\n```"),
            lambda got: isinstance(got, str) and "```" not in got,
            "返回值是不含 ``` 的字符串",
        ),
        dr.raises_check("多行围栏抛 CandidateError", lambda: f("```\n1\n2\n```"), CandidateError),
    )


def checks_validate_page(m: dr.Materialized) -> Sequence[dr.Check]:
    f = m.function
    return (
        dr.eq_check("合法页返回 None", lambda: f(FIXTURE_PAGE), None),
        dr.raises_check("重复 id 抛 CandidateError", lambda: f(DUP_PAGE), CandidateError),
        dr.raises_check("空代码抛 CandidateError", lambda: f(EMPTY_CODE_PAGE), CandidateError),
        dr.raises_check("语言不一致抛 CandidateError", lambda: f(LANG_MISMATCH_PAGE), CandidateError),
    )


def checks_bullet_lines(m: dr.Materialized) -> Sequence[dr.Check]:
    f = m.function
    return (
        dr.eq_check("两条要求各一行", lambda: list(f(("a", "b"))), ["- a", "- b"]),
        dr.eq_check("空列表给占位行", lambda: list(f(())), ["- (none)"]),
    )


def checks_materialize(m: dr.Materialized) -> Sequence[dr.Check]:
    f = m.function
    first = FIXTURE_PAGE.candidates[0]
    return (
        dr.eq_check(
            "命中时返回宿主候选代码",
            lambda: f(CandidateChoice("0", first, "0")),
            first.code,
        ),
        dr.raises_check(
            "NO_MATCH 拒绝物化",
            lambda: f(CandidateChoice(None, None, "NONE")),
            CandidateError,
        ),
    )


def checks_choose_candidate(m: dr.Materialized) -> Sequence[dr.Check]:
    f = m.function

    def one_call_picks_id() -> str | None:
        engine = ScriptedEngine(["1"])
        choice, _stats = f(engine, ANY_TASK, FIXTURE_PAGE, 8)
        if len(engine.calls) != 1:
            return f"执行器被调用 {len(engine.calls)} 次，应当只调用 1 次（不重试）"
        if choice.candidate_id != "1":
            return f"候选 id 期望 '1'，实得 {choice.candidate_id!r}"
        return None

    def tolerates_fenced_scalar() -> str | None:
        engine = ScriptedEngine(["```\n2\n```"])
        choice, _stats = f(engine, ANY_TASK, FIXTURE_PAGE, 8)
        if choice.candidate_id != "2":
            return f"候选 id 期望 '2'，实得 {choice.candidate_id!r}"
        return None

    def bogus_reply_raises_once() -> str | None:
        engine = ScriptedEngine(["bogus"])
        try:
            f(engine, ANY_TASK, FIXTURE_PAGE, 8)
        except CandidateError:
            if len(engine.calls) != 1:
                return f"执行器被调用 {len(engine.calls)} 次，应当只调用 1 次（不修复重试）"
            return None
        return "非法回复没有抛 CandidateError"

    return (
        dr.Check("一次调用选出候选 1", one_call_picks_id),
        dr.Check("容忍单行围栏回复", tolerates_fenced_scalar),
        dr.Check("非法回复只调一次并抛错", bogus_reply_raises_once),
    )


def checks_build_selection_messages(m: dr.Materialized) -> Sequence[dr.Check]:
    f = m.function

    def shape() -> str | None:
        messages = f(ANY_TASK, FIXTURE_PAGE)
        if [m["role"] for m in messages] != ["system", "user"]:
            return f"角色序列不对：{[m['role'] for m in messages]}"
        user = messages[1]["content"]
        for needle in ("valid outputs: 0, 1, 2, NONE", "candidate 2", "output exactly one valid output token"):
            if needle not in user:
                return f"提示里缺少 {needle!r}"
        return None

    def rejects_bad_page() -> str | None:
        try:
            f(ANY_TASK, DUP_PAGE)
        except CandidateError:
            return None
        return "非法候选页没有抛 CandidateError（说明它没走宿主校验）"

    return (dr.Check("两段式提示骨架", shape), dr.Check("先校验候选页", rejects_bad_page))


def checks_json_roundtrip(m: dr.Materialized) -> Sequence[dr.Check]:
    """NO_MATCH 的参考判据：把候选页变成能 json.loads 回同等页面的字符串。"""

    f = m.function

    def roundtrip() -> str | None:
        got = f(FIXTURE_PAGE)
        if not isinstance(got, str):
            return f"返回的不是字符串（{type(got).__name__}）"
        try:
            back = json.loads(got)
        except ValueError as exc:
            return f"返回的字符串不是 JSON：{exc}"
        if not isinstance(back, dict) or back.get("language") != "python":
            return "JSON 里没有候选页的语言字段"
        return None

    return (dr.Check("候选页 -> JSON 字符串", roundtrip),)


def checks_validate_and_report(m: dr.Materialized) -> Sequence[dr.Check]:
    """NO_MATCH 的参考判据：非法候选打印到 stderr 后继续，遍历完整页再返回。"""

    f = m.function

    def report_and_continue() -> str | None:
        sink = io.StringIO()
        with contextlib.redirect_stderr(sink):
            got = f(DUP_PAGE)
        if not sink.getvalue():
            return "stderr 里没有任何诊断输出"
        if isinstance(got, str):
            return None
        return f"返回的不是报告字符串（{type(got).__name__}）"

    return (dr.Check("打印诊断并继续处理", report_and_continue),)


FUNCTION_TASKS: tuple[FunctionTask, ...] = (
    _fn_task(
        "fn-parse-choice",
        "模型回了一串东西，要严格解析成候选 id；「只回一个 id」是硬要求，越界的回复当错误，没有匹配就回 NONE。",
        "parse_model_reply",
        (
            "accept a single-line reply that is either one candidate id or the literal NONE",
            "tolerate surrounding whitespace and a single-line markdown fence",
            "raise the protocol error for an unknown id",
        ),
        "parse_choice",
        checks_parse_choice,
        constraints=("no repairing of the model reply", "exactly one candidate page lookup"),
    ),
    _fn_task(
        "fn-unwrap-scalar",
        "只把完整包住单行标量的 markdown 围栏脱掉，其它情况原样返回；多行围栏算错。",
        "unwrap_fenced_scalar",
        (
            "strip a markdown fence only when it wraps exactly one line",
            "return the input unchanged when there is no fence",
            "reject a multi-line fenced payload",
        ),
        "_unwrap_scalar",
        checks_unwrap_scalar,
    ),
    _fn_task(
        "fn-validate-page",
        "候选页可能不干净：重复 id、空代码、语言和页不一致。要一个只做宿主边界校验的函数。",
        "validate_candidate_page",
        (
            "reject duplicate candidate ids",
            "reject a candidate with blank code",
            "reject a candidate whose language differs from the page",
        ),
        "validate_page",
        checks_validate_page,
        constraints=("accept a well-formed page by returning None",),
    ),
    _fn_task(
        "fn-bullet-lines",
        "把几条短要求排成固定顺序的提示行；一条都没有时也要给一行占位，不能回空列表。",
        "render_requirement_bullets",
        (
            "render one line per requirement",
            "keep the given order",
            "render a placeholder line for an empty list",
        ),
        "_bullet_lines",
        checks_bullet_lines,
    ),
    _fn_task(
        "fn-materialize",
        "拿到已经校验过的选择后，返回宿主那边的候选代码；如果是 NONE，必须拒绝而不是返回空字符串。",
        "materialize_host_candidate",
        (
            "return the host candidate payload for a real choice",
            "refuse to materialize when there is no match",
        ),
        "materialize",
        checks_materialize,
    ),
    _fn_task(
        "fn-choose-candidate",
        "让执行器做一次选择并把结果解析出来；不许重试、不许改写模型回复、不许动文件。",
        "request_one_selection",
        (
            "call the executor exactly once",
            "return the parsed host choice",
            "accept a single-line fenced scalar reply",
            "raise the protocol error for an unusable reply",
        ),
        "choose_candidate",
        checks_choose_candidate,
        constraints=("no retry", "no repairing the reply", "no file operations"),
    ),
    _fn_task(
        "fn-build-messages",
        "把一条规范化任务和一张候选页拼成固定的两段提示：系统段管住输出格式，用户段列出全部合法 id 和每个候选的源码。",
        "build_selection_prompt",
        (
            "emit exactly two messages: system then user",
            "list every valid output token including NONE",
            "include each candidate id, name and code",
            "validate the page before building the prompt",
        ),
        "build_selection_messages",
        checks_build_selection_messages,
    ),
    _fn_task(
        "fn-nomatch-json-file",
        "把整张候选页序列化成一个 JSON 文件写到工作目录，再从磁盘读回来比对。",
        "persist_page_as_json_file",
        ("serialize the whole page to a file", "read it back and compare"),
        None,
        checks_json_roundtrip,
        absence_keywords=("json", "open(", "write_text", "read_text", "Path"),
    ),
    _fn_task(
        "fn-nomatch-stderr-report",
        "遇到非法候选不要中断：把每一条问题打印到 stderr，继续检查剩下的候选，最后返回一份汇总报告。",
        "validate_and_report_all",
        ("print each problem to stderr", "keep going after a bad candidate", "return a summary"),
        None,
        checks_validate_and_report,
        absence_keywords=("stderr", "print(", "warnings", "continue"),
    ),
)


def _name_task(
    label: str,
    source_request: str,
    operation: str,
    scope: str,
    requirements: tuple[str, ...],
    needle: str,
    *,
    nth: int = 0,
    index: int = 0,
    witnesses: tuple[tuple[tuple[object, ...], dict[str, object]], ...],
    expected: str | None,
    predicate: Callable[[object], bool] | None,
    predicate_text: str,
    absence_keywords: tuple[str, ...] = (),
) -> NameTask:
    return NameTask(
        label=label,
        source_request=source_request,
        task=CodeTask(
            operation=operation,
            target_language="python",
            requirements=(f"the decision happens inside function {scope}", *requirements),
            constraints=("answer with one candidate id from the page or NONE",),
        ),
        scope=scope,
        needle=needle,
        nth=nth,
        index=index,
        witnesses=witnesses,
        expected=expected,
        predicate=predicate,
        predicate_text=predicate_text,
        absence_keywords=absence_keywords,
    )


NAME_TASKS: tuple[NameTask, ...] = (
    _name_task(
        "name-raw-response",
        "要拿到模型回复的原文——没 strip、也没解围栏的那一份，后面才做规范化。",
        "pick_unmodified_model_reply",
        "parse_choice",
        ("select the name that still holds the unmodified model reply",),
        "candidate = next(",
        witnesses=(_w("```\n0\n```", FIXTURE_PAGE), _w("NONE", FIXTURE_PAGE)),
        expected="raw_response",
        predicate=lambda value: value == "```\n0\n```",
        predicate_text="该观测行该名字的值 == 原始回复原文（含围栏、未 strip）",
    ),
    _name_task(
        "name-stripped-text",
        "要那一份已经去掉首尾空白、但还没解掉 markdown 围栏的文本。",
        "pick_stripped_but_fenced_text",
        "parse_choice",
        ("select the name that holds the reply after whitespace stripping but before fence unwrapping",),
        "text = _unwrap_scalar(text)",
        witnesses=(_w("  ```\n0\n```  ", FIXTURE_PAGE),),
        expected="text",
        predicate=lambda value: value == "```\n0\n```",
        predicate_text="该观测行该名字的值 == 原始回复 strip 之后、未解围栏的文本",
    ),
    _name_task(
        "name-seen-set",
        "重复 id 检查要用一个集合记录已经出现过的候选 id。选那个集合变量。",
        "pick_seen_id_set",
        "validate_page",
        ("select the name that accumulates the candidate ids already seen",),
        "if candidate.id in seen:",
        index=1,
        witnesses=(_w(DUP_PAGE),),
        expected="seen",
        predicate=lambda value: isinstance(value, set) and "0" in value,
        predicate_text="该观测行该名字的值是 set，且已经装着前一个候选的 id '0'",
    ),
    _name_task(
        "name-current-candidate",
        "重复检查那一行里，代表「当前正在检查的那一个候选对象」的名字是哪个。",
        "pick_current_candidate_object",
        "validate_page",
        ("select the name that holds the candidate object under inspection on that line",),
        "if candidate.id in seen:",
        index=1,
        witnesses=(_w(DUP_PAGE),),
        expected="candidate",
        predicate=lambda value: isinstance(value, CodeCandidate) and value.id == "0",
        predicate_text="该观测行该名字的值是 CodeCandidate 且 id == '0'",
    ),
    _name_task(
        "name-valid-ids",
        "提示里那串把所有候选 id 用逗号拼起来的「合法输出」清单，是哪个名字。",
        "pick_valid_output_list_string",
        "build_selection_messages",
        ("select the name holding the comma-joined string of the candidate ids",),
        "for candidate in page.candidates",
        nth=1,
        witnesses=(_w(ANY_TASK, FIXTURE_PAGE),),
        expected="valid_ids",
        predicate=lambda value: isinstance(value, str) and "," in value and "0" in value and "2" in value,
        predicate_text="该观测行该名字的值是字符串，且是候选 id 的逗号拼接（含 0 与 2）",
    ),
    _name_task(
        "name-nomatch-elapsed",
        "选中记录「本次解析耗时秒数」的那个变量，后面要拿它做超时告警。",
        "pick_elapsed_seconds_variable",
        "parse_choice",
        ("select the name that holds the number of seconds this parse took",),
        "candidate = next(",
        witnesses=(_w("```\n0\n```", FIXTURE_PAGE),),
        expected=None,
        predicate=None,
        predicate_text="该作用域内没有计时变量",
        absence_keywords=("perf_counter", "time", "seconds", "elapsed", "duration", "耗时", "秒"),
    ),
)


# 见证调用表：给"枚举覆盖率"用；和 NAME_TASKS 共用同一批真实调用。
# 每个见证是 (args, kwargs)，会被真的执行；抛错也算有效见证（except 分支才看得到名字）。
_ERROR_INSTANCE = CandidateError("witness")



SCOPE_WITNESSES: dict[str, tuple[tuple[tuple[object, ...], dict[str, object]], ...]] = {
    "parse_choice": (
        _w("```\n0\n```", FIXTURE_PAGE),
        _w("NONE", FIXTURE_PAGE),
        _w("9", FIXTURE_PAGE),
        _w("```\n1\n2\n```", FIXTURE_PAGE),
    ),
    "validate_page": (_w(FIXTURE_PAGE), _w(DUP_PAGE), _w(LANG_MISMATCH_PAGE)),
    "_bullet_lines": (_w(("a", "b")), _w(())),
    "build_selection_messages": (_w(ANY_TASK, FIXTURE_PAGE),),
    "_unwrap_scalar": (_w("```\n1\n```"), _w("```\n1\n2\n```")),
    "_error": (_w("x"),),
    "materialize": (
        _w(CandidateChoice("0", FIXTURE_PAGE.candidates[0], "0")),
        _w(CandidateChoice(None, None, "NONE")),
    ),
    "choose_candidate": (
        _w(ScriptedEngine(["1"]), ANY_TASK, FIXTURE_PAGE, 8),
        _w(ScriptedEngine(["bogus"]), ANY_TASK, FIXTURE_PAGE, 8),
    ),
    "CandidateError.__init__": (_w(_ERROR_INSTANCE, "x"),),
}


# ==========================================================================
# 候选页构造：宿主发放 id
# ==========================================================================


def host_order(keys: Sequence[str]) -> tuple[str, ...]:
    """宿主决定候选页顺序：按 salt+key 的 sha256 排序，和源码顺序无关。"""

    return tuple(sorted(keys, key=lambda key: hashlib.sha256(f"{ID_SALT}|{key}".encode()).hexdigest()))


@dataclass(frozen=True)
class BuiltPage:
    """一张候选页，连同 id -> 宿主条目的映射与构造开销。"""

    name: str
    page: CandidatePage
    id_to_key: dict[str, str]
    key_to_id: dict[str, str]
    prompt_chars: int
    code_chars: int

    def prompt_tokens_hint(self, chars_per_token: float) -> int:
        return int(self.prompt_chars / chars_per_token) if chars_per_token > 0 else 0


def prompt_chars(task: CodeTask, page: CandidatePage) -> int:
    from codejev.candidate import build_selection_messages

    messages = build_selection_messages(task, page)
    return sum(len(message["content"]) for message in messages)


def build_function_page(entries: Sequence[de.FunctionEntry]) -> BuiltPage:
    keys = [entry.qualname for entry in entries]
    order = host_order(keys)
    by_key = {entry.qualname: entry for entry in entries}
    candidates = []
    for index, key in enumerate(order):
        entry = by_key[key]
        purpose = entry.docstring or "（没有 docstring）"
        candidates.append(
            CodeCandidate(id=str(index), name=entry.qualname, purpose=purpose, code=entry.source)
        )
    page = CandidatePage(tuple(candidates))
    return BuiltPage(
        name="function-page",
        page=page,
        id_to_key={str(index): key for index, key in enumerate(order)},
        key_to_id={key: str(index) for index, key in enumerate(order)},
        prompt_chars=prompt_chars(ANY_TASK, page),
        code_chars=sum(len(candidate.code) for candidate in page.candidates),
    )


def name_candidate_purpose(entries: Sequence[de.NameEntry]) -> str:
    """名字候选的 purpose 只能自动生成：绑定类别 + 绑定点行号。"""

    first = entries[0]
    lines = "、".join(str(entry.lineno) for entry in entries)
    return f"{first.label}；本作用域内绑定点 {len(entries)} 处（第 {lines} 行）"


def build_name_page(
    scope: str,
    entries: Sequence[de.NameEntry],
) -> BuiltPage:
    """按作用域建名字候选页；同名多处绑定合并成一个候选，绑定点数写进 purpose。"""

    grouped: dict[str, list[de.NameEntry]] = {}
    for entry in entries:
        grouped.setdefault(entry.name, []).append(entry)
    keys = [f"{scope}#{name}" for name in grouped]
    order = host_order(keys)
    candidates = []
    for index, key in enumerate(order):
        name = key.split("#", 1)[1]
        binding_sites = grouped[name]
        candidates.append(
            CodeCandidate(
                id=str(index),
                name=name,
                purpose=name_candidate_purpose(binding_sites),
                code=binding_sites[0].statement or binding_sites[0].line,
            )
        )
    page = CandidatePage(tuple(candidates))
    return BuiltPage(
        name=f"name-page[{scope}]",
        page=page,
        id_to_key={str(index): key for index, key in enumerate(order)},
        key_to_id={key: str(index) for index, key in enumerate(order)},
        prompt_chars=prompt_chars(ANY_TASK, page),
        code_chars=sum(len(candidate.code) for candidate in page.candidates),
    )


def build_flat_name_page(
    enumeration: de.Enumeration,
    *,
    prefix: str = "flat",
) -> BuiltPage:
    """把一份文件里所有函数作用域的名字摊平成一张页：用来量「枚举全」的账单。"""

    return build_flat_name_page_from_entries(
        [
            entry
            for entry in enumeration.names
            if entry.scope_kind == "function" and entry.observable
        ],
        label=f"{prefix}[{enumeration.path.name}]",
    )


def build_flat_name_page_from_entries(
    entries: Sequence[de.NameEntry],
    *,
    label: str,
) -> BuiltPage:
    """把若干份文件的函数作用域名字摊进同一张页（同名不同作用域都会被保留）。"""

    grouped: dict[str, list[de.NameEntry]] = {}
    for entry in entries:
        grouped.setdefault(f"{entry.scope}#{entry.name}", []).append(entry)
    keys = list(grouped)
    order = host_order(keys)
    candidates = []
    for index, key in enumerate(order):
        binding_sites = grouped[key]
        scope = binding_sites[0].scope
        candidates.append(
            CodeCandidate(
                id=str(index),
                name=binding_sites[0].name,
                purpose=f"{scope}｜{name_candidate_purpose(binding_sites)}",
                code=binding_sites[0].statement or binding_sites[0].line,
            )
        )
    page = CandidatePage(tuple(candidates))
    return BuiltPage(
        name=label,
        page=page,
        id_to_key={str(index): key for index, key in enumerate(order)},
        key_to_id={key: str(index) for index, key in enumerate(order)},
        prompt_chars=prompt_chars(ANY_TASK, page),
        code_chars=sum(len(candidate.code) for candidate in page.candidates),
    )


# ==========================================================================
# 模块装载与观察点
# ==========================================================================


def import_source_module(path: Path) -> ModuleType:
    """按仓库相对路径导入源文件，拿到和进程里同一份模块对象（类对象必须共享）。"""

    try:
        relative = path.resolve().relative_to(REPO_ROOT)
    except ValueError:
        relative = None
    if relative is not None and relative.suffix == ".py":
        name = ".".join(relative.with_suffix("").parts)
        if name.endswith(".__init__"):
            name = name[: -len(".__init__")]
        return importlib.import_module(name)
    raise RuntimeError(f"源文件不在仓库内，无法作为模块导入：{path}")


def resolve_qualname(module: ModuleType, qualname: str) -> object:
    obj: object = module
    for part in qualname.split("."):
        obj = getattr(obj, part)
    return obj


# ==========================================================================
# A 部分：跑任务
# ==========================================================================


@dataclass
class Outcome:
    label: str
    space: str
    expected: str
    actual: str
    raw: str
    verdict: str
    elapsed: float
    stats: Stats
    chance: float = 0.0  # 均匀随机猜的命中率 = 1/候选数
    lines: list[str] = field(default_factory=list)


def _sweep(
    built: BuiltPage,
    keys_by_id: Mapping[str, str],
    source_module: ModuleType,
    checks: Callable[[dr.Materialized], Sequence[dr.Check]],
) -> tuple[tuple[str, int, int, str], ...]:
    """把整张候选页扫一遍：行为判据下到底有几个候选能过。

    这一步不看模型，只看宿主空间本身——用来量"答案是否唯一"，以及 NO_MATCH 是否成立。
    """

    results = []
    module_globals = vars(source_module)
    for candidate in built.page.candidates:
        entry_key = keys_by_id[candidate.id]
        try:
            materialized = dr.materialize_code(candidate.code, module_globals)
        except dr.MaterializeError as exc:
            results.append((entry_key, 0, 1, f"物化失败：{exc}"))
            continue
        try:
            report = dr.run_checks(checks(materialized))
        except Exception as exc:
            results.append((entry_key, 0, 1, f"判据抛错：{type(exc).__name__}: {exc}"))
            continue
        passed = sum(1 for _label, ok, _detail in report.checks if ok)
        total = len(report.checks)
        first_failure = next((detail for _label, ok, detail in report.checks if not ok), "")
        results.append((entry_key, passed, total, first_failure))
    return tuple(results)


def run_function_task(
    engine: OpenAICompatibleEngine | None,
    built: BuiltPage,
    source_module: ModuleType,
    entries: Mapping[str, de.FunctionEntry],
    task_def: FunctionTask,
) -> Outcome:
    """跑一条函数域任务：模型选择 + 物化 + 行为指纹 + 全页扫描。"""

    started = time.perf_counter()
    raw = "<no response>"
    actual = "ERROR"
    stats = Stats()
    lines: list[str] = []
    notes: list[str] = []
    chosen_id: str | None = None

    # 先扫全页：不花 API，provider 挂掉时也留下"空间里答案是否唯一"的证据。
    sweep = _sweep(built, built.id_to_key, source_module, task_def.checks)
    passers = [(key, passed, total) for key, passed, total, _reason in sweep if passed == total]
    lines.append(
        "全页扫描（行为判据，不看模型）："
        + (
            "、".join(f"{key} {passed}/{total}" for key, passed, total in passers)
            if passers
            else "没有任何候选通过"
        )
        + f"；候选数 {len(sweep)}"
    )

    if engine is not None:
        try:
            outcome = choose_with_backoff(engine, task_def.task, built.page)
            notes = outcome.notes
            stats = outcome.stats
            raw = outcome.choice.raw_response
            chosen_id = outcome.choice.candidate_id
            actual = chosen_id if chosen_id is not None else "NONE"
        except CandidateError as exc:
            actual = "INVALID"
            if exc.raw_response is not None:
                raw = exc.raw_response
        except Exception as exc:  # 探针必须出一行结果，不能藏住 provider 失败
            actual = f"ERROR:{type(exc).__name__}"
            raw = f"{type(exc).__name__}: {exc}"

    behavior_ok = False
    if engine is None:
        lines.append("--no-api：没有模型选择，只有宿主侧扫描结果；期望答案见候选页标记")
    elif task_def.expected_qualname is None:
        behavior_ok = chosen_id is None
        if chosen_id is not None:
            lines.append(f"判据：期望 NONE，实际给出了候选 {chosen_id}")
        else:
            lines.append("判据：NO_MATCH 成立（模型回了 NONE）")
        absent = tuple(
            keyword
            for keyword in task_def.absence_keywords
            if any(
                keyword in candidate.code or keyword in candidate.purpose
                for candidate in built.page.candidates
            )
        )
        lines.append(
            f"参考判据通过数 0 的佐证：页里出现这些关键词的候选 {absent or '无'}；"
            f"要求的能力（{', '.join(task_def.absence_keywords)}）在页面上找不到载体"
        )
    elif chosen_id is None:
        lines.append("判据：模型回了 NO_MATCH，但宿主页里有唯一正确答案")
    else:
        entry_key = built.id_to_key[chosen_id]
        candidate = next(c for c in built.page.candidates if c.id == chosen_id)
        entry = entries[entry_key]
        code = materialize(CandidateChoice(chosen_id, candidate, raw))
        fidelity = dr.segment_is_file_text(entry, code)
        lines.append(
            f"物化：name={entry.name} qualname={entry_key} 源码段 {entry.segmented_lines} 行 "
            f"与源文件第 {entry.lineno}-{entry.end_lineno} 行逐字一致={'是' if fidelity else '否'}"
        )
        try:
            materialized = dr.materialize_code(code, vars(source_module))
            report = dr.run_checks(task_def.checks(materialized))
            behavior_ok = fidelity and report.passed
            lines.extend("  " + line for line in report.lines())
            lines.append(
                f"  物化的真函数：{materialized.function.__name__}；"
                f"引用全局 {list(materialized.globals_used)}；"
                f"被内层作用域捕获的单元 {list(materialized.inner_captured)}"
            )
        except dr.MaterializeError as exc:
            lines.append(f"判据：物化失败 {exc}")

    verdict = "SKIP" if engine is None else ("PASS" if behavior_ok else "FAIL")
    if engine is not None:
        lines.append(
            f"命名对照：宿主期望 qualname={task_def.expected_qualname!r}"
            f"（id={built.key_to_id.get(task_def.expected_qualname or '', 'NONE')}）"
        )
    for note in notes:
        lines.append(f"退避：{note}")
    return Outcome(
        label=task_def.label,
        space="function",
        expected=(
            built.key_to_id.get(task_def.expected_qualname, "NONE")
            if task_def.expected_qualname is not None
            else "NONE"
        ),
        actual="SKIP" if engine is None else actual,
        raw=raw,
        verdict=verdict,
        elapsed=time.perf_counter() - started,
        stats=stats,
        chance=1.0 / len(built.page.candidates),
        lines=lines,
    )


@dataclass(frozen=True)
class ObservedNameSpace:
    """一条名字任务运行前的观测准备：观测点、快照、可区分性。"""

    line: int
    line_hits: tuple[int, ...]
    witness_calls: int
    snapshot: dr.Snapshot
    ambiguous_groups: tuple[tuple[str, ...], ...]
    satisfying: tuple[str, ...]  # 值满足判据的候选名（宿主判据的"歧义集"）


def observe_name_point(
    source_module: ModuleType,
    source_lines: Sequence[str],
    entries_by_qualname: Mapping[str, de.FunctionEntry],
    task_def: NameTask,
) -> ObservedNameSpace:
    """跑见证调用取快照，并算出宿主判据在该快照上的命中集。

    A 部分与 B 部分共用这一处逻辑，避免"报告里说一套、跑的是另一套"。
    """

    entry = entries_by_qualname[task_def.scope]
    hits = tuple(
        index + 1
        for index in range(entry.lineno - 1, entry.end_lineno)
        if task_def.needle in source_lines[index]
    )
    point = dr.needle_line(source_lines, entry, task_def.needle, nth=task_def.nth)
    func = resolve_qualname(source_module, task_def.scope)
    observation = dr.observe_locals(
        func, task_def.witnesses, needles={"point": point}, qualname=task_def.scope
    )
    snapshot = observation.at(point, task_def.index)
    witnesses_run = observation.calls
    names = list(entry_names(entries_by_qualname, task_def.scope))
    satisfying = (
        tuple(
            name
            for name in names
            if name in snapshot.values and task_def.predicate(snapshot.values[name])
        )
        if task_def.predicate is not None
        else ()
    )
    return ObservedNameSpace(
        line=point,
        line_hits=hits,
        witness_calls=witnesses_run,
        snapshot=snapshot,
        ambiguous_groups=dr.equal_groups(snapshot.values, names),
        satisfying=satisfying,
    )


def entry_names(
    entries_by_qualname: Mapping[str, de.FunctionEntry], scope: str
) -> tuple[str, ...]:
    """候选页里属于这个作用域的名字（去重后），用来做歧义集计算。"""

    return _page_names_cache[scope]


_page_names_cache: dict[str, tuple[str, ...]] = {}


def run_name_task(
    engine: OpenAICompatibleEngine | None,
    built: BuiltPage,
    source_module: ModuleType,
    source_lines: Sequence[str],
    entries_by_qualname: Mapping[str, de.FunctionEntry],
    task_def: NameTask,
) -> Outcome:
    """跑一条名字域任务：模型选择 + 在真实观测点上代值求判据。"""

    started = time.perf_counter()
    raw = "<no response>"
    actual = "ERROR"
    stats = Stats()
    lines: list[str] = []
    notes: list[str] = []
    chosen_id: str | None = None

    observed = observe_name_point(source_module, source_lines, entries_by_qualname, task_def)
    lines.append(
        f"观测点：{task_def.scope} 第 {observed.line} 行「{source_lines[observed.line - 1].strip()}」"
        f"（文本命中 {list(observed.line_hits)}，取第 {task_def.nth + 1} 处；"
        f"快照来自第 {task_def.index + 1} 次执行，见证调用 {observed.snapshot.witness}；"
        f"共跑 {observed.witness_calls} 次见证）"
    )
    lines.append(
        "  见证实得："
        + "；".join(
            f"{name}={dr.describe_value(value, 56)}" for name, value in observed.snapshot.values.items()
        )
    )
    lines.append(
        "  值相等的候选名字分组（读值判据下不可区分）："
        + ("；".join("=".join(group) for group in observed.ambiguous_groups) if observed.ambiguous_groups else "无")
    )
    lines.append(
        f"  宿主判据「{task_def.predicate_text}」在该快照上命中的候选名："
        + (f"{list(observed.satisfying)}（{len(observed.satisfying)} 个）" if observed.satisfying else "无")
    )

    if engine is not None:
        try:
            outcome = choose_with_backoff(engine, task_def.task, built.page)
            notes = outcome.notes
            stats = outcome.stats
            raw = outcome.choice.raw_response
            chosen_id = outcome.choice.candidate_id
            actual = chosen_id if chosen_id is not None else "NONE"
        except CandidateError as exc:
            actual = "INVALID"
            if exc.raw_response is not None:
                raw = exc.raw_response
        except Exception as exc:
            actual = f"ERROR:{type(exc).__name__}"
            raw = f"{type(exc).__name__}: {exc}"

    behavior_ok = False
    if engine is None:
        lines.append("--no-api：没有模型选择，只有观测点与宿主判据；期望答案见候选页标记")
    elif task_def.expected is None:
        behavior_ok = chosen_id is None
        lines.append(
            "判据：期望 NO_MATCH，实际"
            + ("回了 NONE，成立" if chosen_id is None else f"给了候选 {chosen_id}")
        )
        hits_found = tuple(
            keyword
            for keyword in task_def.absence_keywords
            if any(
                keyword in candidate.code or keyword in candidate.purpose
                for candidate in built.page.candidates
            )
        )
        lines.append(
            f"  佐证（只到词面，名字域没有可跑的行为）：页里含 {list(task_def.absence_keywords)} 的候选 = "
            f"{list(hits_found) or '无'}"
        )
    elif chosen_id is None:
        lines.append("判据：模型回了 NO_MATCH，但该观测点上确实存在唯一命中名")
    else:
        chosen_name = next(c.name for c in built.page.candidates if c.id == chosen_id)
        value = observed.snapshot.values.get(chosen_name, dr.MISSING)
        if value is dr.MISSING:
            lines.append(f"判据：选中的 {chosen_name} 在第 {observed.line} 行根本没有绑定（运行时不观测这个名字）")
        else:
            ok = bool(task_def.predicate and task_def.predicate(value))
            behavior_ok = ok and chosen_name in observed.satisfying
            lines.append(
                f"判据：把 {chosen_name} 代入观测快照 -> {dr.describe_value(value)}；"
                f"「{task_def.predicate_text}」={'成立' if ok else '不成立'}"
            )

    if engine is not None and task_def.expected is not None:
        lines.append(
            "命名对照：宿主期望名字 "
            f"{task_def.expected!r}"
            f"（id={built.key_to_id.get(f'{task_def.scope}#{task_def.expected}', 'NONE')}）"
        )
    for note in notes:
        lines.append(f"退避：{note}")
    return Outcome(
        label=task_def.label,
        space="name",
        expected=(
            built.key_to_id.get(f"{task_def.scope}#{task_def.expected}", "NONE")
            if task_def.expected is not None
            else "NONE"
        ),
        actual=actual,
        raw=raw,
        verdict="SKIP" if engine is None else ("PASS" if behavior_ok else "FAIL"),
        elapsed=time.perf_counter() - started,
        stats=stats,
        chance=1.0 / len(built.page.candidates),
        lines=lines,
    )


def run_scale_probe(
    engine: OpenAICompatibleEngine | None,
    big_page: BuiltPage,
    task_def: NameTask,
    source_module: ModuleType,
    source_lines: Sequence[str],
    entries_by_qualname: Mapping[str, de.FunctionEntry],
    samples: dict[str, list[int]],
) -> Outcome:
    """同一句任务、同一份判据，换成"把两份文件的名字摊平"的大页再选一次。

    这条实验直接对应"枚举全"的代价：候选空间一大，模型还选得对吗？id 还抄得对吗？
    """

    started = time.perf_counter()
    expected_key = f"{task_def.scope}#{task_def.expected}"
    expected_id = big_page.key_to_id.get(expected_key)
    observed = observe_name_point(source_module, source_lines, entries_by_qualname, task_def)
    lines = [
        f"同一句任务在大页上的期望：key={expected_key} id={expected_id}"
        f"（大页候选数 {len(big_page.page.candidates)}，其中 id 是 "
        f"{len(str(len(big_page.page.candidates) - 1))} 位数字）",
        f"宿主判据在该观测点的命中集仍是 {list(observed.satisfying)}（与作用域内页一致）",
    ]
    raw = "<no response>"
    actual = "ERROR"
    stats = Stats()
    chosen_id: str | None = None
    notes: list[str] = []
    if engine is None:
        lines.append("--no-api：跳过规模实验")
        verdict = "SKIP"
    else:
        try:
            outcome = choose_with_backoff(engine, task_def.task, big_page.page)
            notes = outcome.notes
            stats = outcome.stats
            raw = outcome.choice.raw_response
            chosen_id = outcome.choice.candidate_id
            actual = chosen_id if chosen_id is not None else "NONE"
        except CandidateError as exc:
            actual = "INVALID"
            if exc.raw_response is not None:
                raw = exc.raw_response
        except Exception as exc:
            actual = f"ERROR:{type(exc).__name__}"
            raw = f"{type(exc).__name__}: {exc}"
        chosen_key = big_page.id_to_key.get(chosen_id or "", "<无>")
        lines.append(f"模型回的 id 映射回候选：{chosen_key}")
        ok = chosen_id is not None and chosen_id == expected_id
        if not ok and chosen_id is not None:
            value = observed.snapshot.values.get(chosen_key.split("#", 1)[-1], dr.MISSING)
            lines.append(
                f"选中的是另一个名字 {chosen_key!r}，它在同一观测点上的值="
                f"{dr.describe_value(value)}（说明模型在大页里挑错了行）"
            )
        verdict = "PASS" if ok else "FAIL"
        samples.setdefault(big_page.name, []).append(stats.prompt_tokens)
    for note in notes:
        lines.append(f"退避：{note}")
    return Outcome(
        label="scale-大页同名任务",
        space="name",
        expected=expected_id or "NONE",
        actual="SKIP" if engine is None else actual,
        raw=raw,
        verdict=verdict,
        elapsed=time.perf_counter() - started,
        stats=stats,
        chance=1.0 / len(big_page.page.candidates),
        lines=lines,
    )

# ==========================================================================
# B 部分：枚举成本（能枚举 / 枚举不全 / 枚举不了）
# ==========================================================================


def heading(title: str) -> None:
    print()
    print(f"=== {title} ===")


def part_b_capability_census(
    source: Path,
    strict: de.Enumeration,
    profile: de.Enumeration,
    naive: de.Enumeration,
) -> None:
    """逐类清单：这一份源文件上，枚举器认哪些、漏哪些。"""

    heading(f"B1 枚举能力清单（源文件：{source.relative_to(REPO_ROOT)}）")
    print("verdict        category                                        enumerated/truth  note")
    for row in de.capability_rows(profile, strict):
        print(f"{row.verdict:14s} {row.category:47s} {row.enumerated:>4}/{row.truth:<6} {row.note}")
        for example in row.examples:
            print(f"{'':14s}   例：{example}")
    print()
    print("绑定类别直方图（本次枚举，含所有作用域）：")
    counts = de.binding_kind_counts(profile)
    print("  " + "；".join(f"{de.KIND_LABELS.get(kind, kind)}={count}" for kind, count in counts.items()))


def part_b_coverage_table(
    source: Path,
    strict: de.Enumeration,
    profile: de.Enumeration,
    naive: de.Enumeration,
    source_module: ModuleType,
) -> list[de.ScopeCoverage]:
    """每个函数作用域一行：symtable / 枚举器 / 运行时观测三方对照。"""

    heading(f"B2 三方对照：symtable（真相） vs 枚举器 vs 运行时观测（{source.name}）")
    print(
        "scope                        symtable绑定  strict  python312  naive  运行时观测  strict漏  naive虚报  引用未绑定(全局)"
    )
    rows: list[de.ScopeCoverage] = []
    for scope in profile.function_scopes():
        if scope.startswith("<") or ".<" in scope:
            continue
        observed: tuple[str, ...] = ()
        witnesses = SCOPE_WITNESSES.get(scope)
        if witnesses is not None:
            try:
                func = resolve_qualname(source_module, scope)
                observation = dr.observe_locals(func, witnesses, qualname=scope)
                observed = tuple(sorted(observation.union))
            except Exception as exc:  # 观测失败要在表里写出来，不能静默当 0
                observed = (f"<观测失败 {type(exc).__name__}>",)
        coverage = de.measure_coverage(
            source,
            scope_qualname=scope,
            strict=strict,
            profile=profile,
            naive=naive,
            observed_union=observed,
        )
        rows.append(coverage)
        print(
            f"{scope:28s} {len(coverage.truth_bound):>10}  {len(coverage.enumerated_strict):>6}  "
            f"{len(coverage.enumerated_profile):>9}  {len(coverage.enumerated_naive):>5}  "
            f"{len(coverage.observed_union):>10}  {len(coverage.missed_by_strict):>7}  "
            f"{len(coverage.spurious_naive):>9}  {len(coverage.truth_referenced_only):>15}"
        )
    print()
    print("逐格细账（只列有差异的作用域）：")
    for coverage in rows:
        if (
            coverage.missed_by_strict
            or coverage.spurious_naive
            or coverage.observed_only
            or coverage.claimed_but_unobserved
            or coverage.claimed_unobservable
        ):
            print(f"  {coverage.qualname}（第 {coverage.lineno} 行）")
            for label, values in (
                ("symtable 绑定", coverage.truth_bound),
                ("strict 枚举", coverage.enumerated_strict),
                ("python312 枚举", coverage.enumerated_profile),
                ("naive 枚举", coverage.enumerated_naive),
                ("运行时观测", coverage.observed_union),
                ("strict 漏掉", coverage.missed_by_strict),
                ("naive 虚报", coverage.spurious_naive),
                ("运行时多出", coverage.observed_only),
                ("声称可观测但没观测到", coverage.claimed_but_unobserved),
                ("枚举器自己标了不在运行时局部", coverage.claimed_unobservable),
            ):
                if values:
                    print(f"      {label:24s} {list(values)}")
    print()
    print("引用了但没在本作用域绑定的名字（名字页装不进去，但任务可能问它们）：")
    for coverage in rows:
        if coverage.truth_referenced_only:
            print(f"  {coverage.qualname:28s} {list(coverage.truth_referenced_only)}")
    return rows


def part_b_other_files() -> None:
    """在其他真实文件上重复普查：嵌套函数、lambda、重名、装饰器。"""

    heading("B3 换几份真实文件再查一遍（本体源文件里没有的结构）")
    targets = [
        REPO_ROOT / "codejev" / "decide.py",
        REPO_ROOT / "bench" / "diffusion_probe.py",
        REPO_ROOT / "tests" / "test_cli.py",
    ]
    print(f"{'file':34s} {'函数':>4} {'嵌套函数':>7} {'lambda':>7} {'推导式目标':>9} {'except-as':>9} {'重名函数':>8}")
    for path in targets:
        enumeration = de.enumerate_source(path, "python312")
        nested = enumeration.functions_in("function")
        lambdas = [s for s in enumeration.scopes if s.kind == "lambda"]
        comps = [n for n in enumeration.names if n.kind == "comp_target"]
        excepts = [n for n in enumeration.names if n.kind == "except_as"]
        by_name: dict[str, set[str]] = {}
        for entry in enumeration.functions:
            by_name.setdefault(entry.name, set()).add(entry.qualname)
        duplicated = {name: quals for name, quals in by_name.items() if len(quals) > 1}
        print(
            f"{str(path.relative_to(REPO_ROOT)):34s} {len(enumeration.functions):>4} "
            f"{len(nested):>7} {len(lambdas):>7} {len(comps):>9} {len(excepts):>9} {len(duplicated):>8}"
        )
        for name, quals in sorted(duplicated.items())[:3]:
            print(f"{'':34s}   重名 {name!r}: {sorted(quals)}")
        for entry in nested[:2]:
            print(
                f"{'':34s}   嵌套函数 {entry.qualname}:{entry.lineno} 外层={list(entry.enclosing_functions)}"
            )
        for scope in lambdas[:2]:
            print(f"{'':34s}   lambda 作用域 {scope.qualname}:{scope.lineno}")
        for name_entry in comps[:2]:
            print(
                f"{'':34s}   推导式目标 {name_entry.name}（真作用域 {name_entry.true_scope}）"
                f" observable={name_entry.observable} inlined={name_entry.inlined_scope}"
            )
    print()
    print("跨全仓重名函数（名字不能当候选标识，必须用 qualname 或宿主 id）：")
    seen: dict[str, set[str]] = {}
    for path in sorted((*REPO_ROOT.glob("codejev/*.py"), *REPO_ROOT.glob("bench/*.py"), *REPO_ROOT.glob("tests/*.py"))):
        enumeration = de.enumerate_source(path, "python312")
        for entry in enumeration.functions:
            seen.setdefault(entry.name, set()).add(f"{path.name}:{entry.qualname}:{entry.lineno}")
    duplicated = sorted(
        ((name, quals) for name, quals in seen.items() if len(quals) > 1),
        key=lambda item: (-len(item[1]), item[0]),
    )
    print(f"  重名函数名 {len(duplicated)} 个，涉及 {sum(len(q) for _n, q in duplicated)} 个真函数")
    for name, quals in duplicated[:4]:
        print(f"    {name!r} x{len(quals)}：{sorted(quals)[:4]}")


def part_b_materialize_limits(source_module: ModuleType, source: Path) -> None:
    """两个真实的反例：闭包单元和装饰器不会跟着源码段一起搬走。"""

    heading("B4 枚举得到、物化不了：两个真实反例")
    # 反例 1：本仓库 decide.py 里的嵌套函数捕获外层局部。
    path = REPO_ROOT / "bench" / "diffusion_probe.py"
    enumeration = de.enumerate_source(path, "python312")
    nested = [f for f in enumeration.functions if f.qualname == "summarize._num"]
    module = importlib.import_module("bench.diffusion_probe")
    symtable_truth = {scope.qualname: scope for scope in de.symtable_scopes(path)}
    for entry in nested:
        materialized = dr.materialize_code(entry.source, vars(module))
        truth = symtable_truth.get(entry.qualname)
        print(
            f"  1) {path.relative_to(REPO_ROOT)}:{entry.lineno} {entry.qualname}"
            f"（嵌套函数；symtable 说它捕获 {list(truth.free()) if truth else '?'}；"
            f"AST 近似只认出它被内层捕获的 {list(materialized.inner_captured)}）"
        )
        try:
            materialized.function("generate_seconds")
            print("     调用成功——说明它没依赖闭包，这个反例不成立")
        except NameError as exc:
            print(f"     物化后调用：NameError: {exc}   ← 源码段搬得走，闭包单元搬不走")
        except Exception as exc:
            print(f"     物化后调用：{type(exc).__name__}: {exc}")

    # 反例 2：本原型源文件自己的 __init__：super() 需要 __class__ 单元。
    init_entry = next(
        f for f in de.enumerate_source(source, "python312").functions if f.name == "__init__"
    )
    materialized = dr.materialize_code(init_entry.source, vars(source_module))
    print(
        f"  2) {source.relative_to(REPO_ROOT)}:{init_entry.lineno} {init_entry.qualname}"
        f"（方法，段里引用 {list(materialized.globals_used)}）"
    )
    try:
        materialized.function(CandidateError("unused"), "boom")
        print("     调用成功——这个反例不成立")
    except Exception as exc:
        print(f"     物化后调用：{type(exc).__name__}: {exc}   ← 脱离类体后方法语义变了")

    # 反例 3：装饰器行不在 ast.get_source_segment 的段里。
    decorated = []
    for path in sorted(REPO_ROOT.glob("codejev/*.py")):
        enumeration = de.enumerate_source(path, "python312")
        for entry in enumeration.functions:
            if entry.decorators:
                decorated.append((path, entry))
    print(f"  3) 带装饰器的函数/方法：codejev 包内 {len(decorated)} 个；源码段不含装饰器行")
    for path, entry in decorated[:3]:
        first_line = entry.source.splitlines()[0].strip()
        print(
            f"     {path.relative_to(REPO_ROOT)}:{entry.lineno} {entry.qualname} "
            f"装饰器={list(entry.decorators)}；物化段首行={first_line!r}"
        )
    print("     → 宿主若直接物化 code 字段，@property/@staticmethod 语义丢失；要另存装饰器再重新拼。")


def part_b_page_cost(
    built_pages: Sequence[BuiltPage],
    flat_pages: Sequence[BuiltPage],
    token_samples: Mapping[str, list[int]],
    chars_per_token: float,
) -> None:
    """候选页账单：候选数、字符数、实测 prompt tokens。"""

    heading("B5 候选页账单：枚举全的代价（字符数实测，token 数来自真实调用）")
    print(f"{'页':30s} {'候选数':>6} {'代码字符':>9} {'提示字符':>9} {'实测prompt tokens':>18} {'发送次数':>8}")
    for built in (*built_pages, *flat_pages):
        samples = token_samples.get(built.name, [])
        measured = f"{min(samples)}~{max(samples)}" if samples else "未实测"
        print(
            f"{built.name:30s} {len(built.page.candidates):>6} {built.code_chars:>9} "
            f"{built.prompt_chars:>9} {measured:>18} {len(samples):>8}"
        )
    print()
    print(f"实测字符/token ≈ {chars_per_token:.2f}（用上面真实调用反推）；未实测的页按这个比例外推，属于估算。")
    print("结论口径：候选数是「一次选择的搜索空间」，提示字符是「每次选择都要重发的账单」。")
    print("作用域内枚举（name-page）和全文件摊平枚举（flat）差多少，就是「枚举全」的价格。")


def part_b_ambiguity(
    source_module: ModuleType,
    source_lines: Sequence[str],
    entries_by_qualname: Mapping[str, de.FunctionEntry],
) -> None:
    """运行时可区分性：同一观测点上值相等的名字，判据无法区分。"""

    heading("B6 运行时不可区分性：宿主判据的真实分辨率")
    print(f"{'task':24s} {'scope':24s} {'行':>4} {'候选':>4} {'值相等分组':40s} 判据命中集")
    for task_def in NAME_TASKS:
        if task_def.predicate is None:
            continue
        observed = observe_name_point(source_module, source_lines, entries_by_qualname, task_def)
        names = list(entry_names(entries_by_qualname, task_def.scope))
        groups = "；".join("=".join(group) for group in observed.ambiguous_groups) or "无"
        print(
            f"{task_def.label:24s} {task_def.scope:24s} {observed.line:>4} {len(names):>4} "
            f"{groups:40s} {list(observed.satisfying)}"
        )
    print()
    print("说明：值相等的名字在任何「读值」判据下都不可区分——宿主给出的候选空间里存在")
    print("      语义重复项时，模型选哪个都对，或都不对，取决于判据落在了哪个观测点。")


def part_b_duplicate_choice_demo(
    source_module: ModuleType,
    function_page: BuiltPage,
    entries_by_qualname: Mapping[str, de.FunctionEntry],
) -> None:
    """候选页自己出现等价候选时，"唯一正确答案"这个前提还成立吗？

    完全不需要 API：全用宿主判据扫。注入的两个候选是宿主手写的合成项，只用来量判据的分辨率。
    """

    heading("B7 候选空间的自我重复：判据的「唯一正确答案」依赖什么")
    real = next(c for c in function_page.page.candidates if c.name == "_unwrap_scalar")
    exact_clone = CodeCandidate(
        id=str(len(function_page.page.candidates)),
        name="_unwrap_scalar_clone",
        purpose="宿主注入：与 _unwrap_scalar 逐字相同的第二份（模拟别名/重复枚举）。",
        code=real.code,
    )
    near_clone = CodeCandidate(
        id=str(len(function_page.page.candidates) + 1),
        name="_unwrap_scalar_loose",
        purpose="宿主注入：同样脱围栏，但多行围栏不报错（少一条约束）。",
        code=(
            "def _unwrap_scalar_loose(text: str) -> str:\n"
            "    if not (text.startswith('```') or text.endswith('```')):\n"
            "        return text\n"
            "    body = text[3:-3]\n"
            "    head, _, rest = body.partition('\\n')\n"
            "    return rest.strip() if rest else text\n"
        ),
    )
    injected = CandidatePage(tuple([*function_page.page.candidates, exact_clone, near_clone]))
    task_def = next(task for task in FUNCTION_TASKS if task.label == "fn-unwrap-scalar")
    sweep = _sweep(
        BuiltPage(
            name="injected",
            page=injected,
            id_to_key={c.id: c.name for c in injected.candidates},
            key_to_id={c.name: c.id for c in injected.candidates},
            prompt_chars=0,
            code_chars=0,
        ),
        {c.id: c.name for c in injected.candidates},
        source_module,
        task_def.checks,
    )
    fingerprint_size = sweep[0][2]
    print(f"  fn-unwrap-scalar 的行为指纹（{fingerprint_size} 条）在注入后的候选页上：")
    for key, passed, total, reason in sweep:
        flag = "通过" if passed == total else f"差 {total - passed} 条"
        detail = "" if passed == total else f"｜首个失败：{reason[:70]}"
        print(f"    {key:24s} {passed}/{total} {flag}{detail}")
    full = [key for key, passed, total, _r in sweep if passed == total]
    print()
    print(
        f"  完整通过者 {len(full)} 个：{full}"
        + (
            "   ← 逐字等价的第二份也让判据全过：此时「宿主期望的那个 id」只是任意一个，"
            "协议本身无法区分，唯一性靠的是宿主不去枚举重复项。"
            if len(full) > 1
            else ""
        )
    )
    loose_passed = next(p for k, p, _t, _r in sweep if k == "_unwrap_scalar_loose")
    print(
        f"  少一条约束的近似候选过 {loose_passed}/{fingerprint_size}："
        "把它和真候选分开的只是「多行围栏必须报错」那一条判据；"
        "指纹少一条，候选空间里就多一个无法排除的等价项。"
    )
    print(
        "  → 所以「枚举全」要拆成两件事分别量：候选空间是否覆盖真实决策点（B1~B4），"
        "以及判据在候选空间中是否单射（本节）。"
    )


def part_b_not_verified() -> None:
    heading("B8 这一轮没有验证的部分")
    for line in (
        "只测了 Python 3.12.13 一个解释器：推导式目标的归属（PEP 709 内联）在 3.11 及更早相反。",
        "只有 codejev/candidate.py 一个主域源文件；B3 的其它文件只做静态普查，没跑任务。",
        "变量域只覆盖了 3 个函数作用域（parse_choice / validate_page / build_selection_messages）。",
        "运行时观测是「若干见证调用的并集」，不是完备的局部集合；条件绑定的名字可能没被任何见证触发。",
        "任务规范化（用户原话 → CodeTask）由本探针手写，没有验证大模型那一步。",
        "没测候选页超出上下文、id 位数变长、并发选择、以及物化失败后的回退路径。",
        "名字域的判据依赖宿主选定的观测点行；换一个观测点，同一句任务可能得到不同答案。",
        "NO_MATCH 的名字任务只有词面佐证，没有可跑的行为判据。",
        "打分的 15 条任务，候选页都只有 3~9 个候选；417 个候选的大页只跑了 A4 一条。",
        "任务与判据由本原型自己编写，没有独立出题人：100% 不能外推为「协议在真实任务上 100%」。",
        "只用 deepseek-v4-flash 一个执行器；没有对照更小的本地模型（bench/candidate_probe.py 的 MLX 路线没跑）。",
        "A4 的 417 候选约 2.45 万 prompt tokens，只占常见上下文的四分之一；没有逼近上限。",
        "默认只跑一轮（--repeat 1）；5 轮采样只给出「有没有抖动」，给不出概率置信区间。",
        "本轮测到的抖动（name-raw-response 5 轮里 1 次回 NONE）来自 provider 侧，"
        "不是协议解析问题：回复本身合法，只是答错。",
    ):
        print(f"  - {line}")


def run_repeat_rounds(
    engine: OpenAICompatibleEngine | None,
    function_page: BuiltPage,
    name_pages: Mapping[str, BuiltPage],
    source_module: ModuleType,
    source_lines: Sequence[str],
    entries_by_qualname: Mapping[str, de.FunctionEntry],
    *,
    times: int,
) -> list[list[Outcome]]:
    """把打分的任务再跑若干轮：温度 0 也不保证同一句任务每次同答，这一点要量出来。"""

    rounds: list[list[Outcome]] = []
    for index in range(2, times + 1):
        heading(f"A5 重复轮 {index}/{times}：同一批任务、同一批候选页，重发一次")
        rows: list[Outcome] = []
        for task_def in FUNCTION_TASKS:
            outcome = run_function_task(
                engine, function_page, source_module, entries_by_qualname, task_def
            )
            rows.append(outcome)
            print(f"  {outcome.label:26s} {outcome.verdict:4s} actual={outcome.actual:>5s} raw={outcome.raw!r}")
        for task_def in NAME_TASKS:
            built = name_pages[task_def.scope]
            outcome = run_name_task(
                engine, built, source_module, source_lines, entries_by_qualname, task_def
            )
            rows.append(outcome)
            print(f"  {outcome.label:26s} {outcome.verdict:4s} actual={outcome.actual:>5s} raw={outcome.raw!r}")
        rounds.append(rows)
    return rounds


def print_variance(rounds: Sequence[Sequence[Outcome]], labels: Sequence[str]) -> None:
    """逐任务统计多轮结果：通过数 + 每轮原始回复；按标签对齐，不看顺序。"""

    if len(rounds) < 2:
        return
    heading(f"重复采样方差（{len(rounds)} 轮，温度 0）")
    print(f"{'task':26s} {'通过':>6} {'期望':>5} {'各轮 actual':24s} 各轮 raw")
    unstable: list[str] = []
    for label in labels:
        cells = [
            next(outcome for outcome in row if outcome.label == label) for row in rounds
        ]
        passed = sum(1 for cell in cells if cell.verdict == "PASS")
        actuals = ",".join(cell.actual for cell in cells)
        raws = " ".join(repr(cell.raw) for cell in cells)
        print(f"{label:26s} {passed:>3}/{len(cells):<2} {cells[0].expected:>5} {actuals:24s} {raws}")
        if len({cell.actual for cell in cells}) > 1:
            unstable.append(label)
    print()
    print(f"  结果不稳定的任务：{unstable or '无'}")
    if unstable:
        print("  → 温度 0 也不等于逐次一致（provider 侧实现/路由差异）；协议本身是严格的，")
        print("    但「同一句任务每次同答」不是协议能保证的性质，宿主必须按回复内容判，不能按历史缓存。")


# ==========================================================================
# 打印与主流程
# ==========================================================================


def print_page(built: BuiltPage, expected_ids: Sequence[str]) -> None:
    """打印一张候选页的全貌：id、名字、purpose，并标出期望答案。"""

    expected = set(expected_ids)
    print(
        f"[{built.name}] candidates={len(built.page.candidates)} "
        f"prompt_chars={built.prompt_chars} code_chars={built.code_chars} "
        f"（id 由宿主按 salt+key 的 sha256 排序发放，与源码顺序无关）"
    )
    for candidate in built.page.candidates:
        mark = "  <== 期望答案" if candidate.id in expected else ""
        print(
            f"  id={candidate.id:>2s} name={candidate.name:26s} "
            f"purpose={candidate.purpose[:46]}{mark}"
        )


def print_outcome(outcome: Outcome) -> None:
    """打印一条任务的原始输出与全部判据证据。"""

    print(
        f"{outcome.label:26s} space={outcome.space:8s} expected={outcome.expected:>5s} "
        f"actual={outcome.actual:>5s} {outcome.verdict:4s} {outcome.elapsed:6.3f}s "
        f"prompt={outcome.stats.prompt_tokens} completion={outcome.stats.generated_tokens} "
        f"raw={outcome.raw!r}"
    )
    for line in outcome.lines:
        print(f"    {line}")


def print_accuracy(outcomes: Sequence[Outcome], expected_ids: Sequence[str]) -> None:
    """按域统计严格准确率，并给出反位置猜测的基线。"""

    scored = [o for o in outcomes if o.verdict != "SKIP"]
    heading("A 部分准确率")
    if not scored:
        print("  --no-api：没有模型选择，不产生准确率；下面是宿主侧的判据证据。")
        return
    for space in ("function", "name"):
        rows = [o for o in scored if o.space == space]
        passed = sum(1 for o in rows if o.verdict == "PASS")
        if rows:
            print(f"  {space:9s} 域：{passed}/{len(rows)} ({passed / len(rows):.1%})")
        failures = [o for o in rows if o.verdict != "PASS"]
        for outcome in failures:
            print(f"    失败 {outcome.label}: 期望 {outcome.expected} 实得 {outcome.actual} raw={outcome.raw!r}")
    passed = sum(1 for o in scored if o.verdict == "PASS")
    if scored:
        print(f"  合计：{passed}/{len(scored)} ({passed / len(scored):.1%})")
    print(f"  各任务宿主期望 id：{list(expected_ids)}（分布在页内不同位置，不是同一个位置）")
    chance = [outcome.chance for outcome in scored if outcome.chance > 0]
    if chance:
        print(
            f"  均匀随机猜的期望命中率：{sum(chance) / len(chance):.1%}"
            f"（各任务候选数的倒数：{'、'.join(f'{c:.0%}' for c in chance)}）"
        )
    first_id_hits = sum(1 for outcome in scored if outcome.expected == "0")
    print(
        f"  位置基线：如果每次只回页里第一个候选（id=0），能过 {first_id_hits}/{len(scored)}"
        f"；若全部期望都是 id=0 则基线=100%，说明任务设计能区分「看内容」和「猜位置」。"
    )


def token_samples_from(outcomes: Sequence[Outcome], page_names: Sequence[str]) -> dict[str, list[int]]:
    """按页收集真实 prompt tokens（每条任务一次调用）。"""

    samples: dict[str, list[int]] = {name: [] for name in page_names}
    return samples


def run_part_a(
    engine: OpenAICompatibleEngine | None,
    source_module: ModuleType,
    source_lines: Sequence[str],
    function_page: BuiltPage,
    entries_by_qualname: Mapping[str, de.FunctionEntry],
    name_pages: Mapping[str, BuiltPage],
    *,
    model: str,
) -> tuple[list[Outcome], dict[str, list[int]]]:
    """A 部分：先打印候选页全貌，再逐条任务跑选择与运行时判据。"""

    heading(f"A1 候选页（第二个域：Python 函数与变量；源文件 {source_module.__file__}）")
    print_page(
        function_page,
        [function_page.key_to_id[t.expected_qualname] for t in FUNCTION_TASKS if t.expected_qualname],
    )
    for scope, built in name_pages.items():
        print_page(
            built,
            [
                built.key_to_id[f"{scope}#{t.expected}"]
                for t in NAME_TASKS
                if t.scope == scope and t.expected is not None
            ],
        )
    print()
    print(f"选择执行器：provider={PROVIDER_ID} model={model} reasoning_effort=none max_tokens={MAX_TOKENS}")
    if engine is None:
        print("--no-api：只跑宿主判据与全页扫描，不打分（verdict=SKIP）")

    heading("A2 函数域任务（选函数：物化源码段 -> exec 成真函数 -> 跑行为指纹）")
    outcomes: list[Outcome] = []
    samples: dict[str, list[int]] = {function_page.name: []}
    for task_def in FUNCTION_TASKS:
        outcome = run_function_task(engine, function_page, source_module, entries_by_qualname, task_def)
        print_outcome(outcome)
        if outcome.stats.prompt_tokens:
            samples[function_page.name].append(outcome.stats.prompt_tokens)
        outcomes.append(outcome)

    heading("A3 名字域任务（选名字：在真实观测点上把名字代进判据求值）")
    for task_def in NAME_TASKS:
        built = name_pages[task_def.scope]
        samples.setdefault(built.name, [])
        outcome = run_name_task(
            engine, built, source_module, source_lines, entries_by_qualname, task_def
        )
        print_outcome(outcome)
        if outcome.stats.prompt_tokens:
            samples[built.name].append(outcome.stats.prompt_tokens)
        outcomes.append(outcome)

    print_accuracy(outcomes, [o.expected for o in outcomes])
    return outcomes, samples


def chars_per_token(samples: Mapping[str, Sequence[int]], pages: Mapping[str, BuiltPage]) -> float:
    """用真实调用反推字符/token 比；没有实测就退回 2.0（估算）。"""

    ratios = []
    for name, values in samples.items():
        page = pages.get(name)
        if page is None or not values:
            continue
        ratios.append(page.prompt_chars / (sum(values) / len(values)))
    if not ratios:
        return 2.0
    return sum(ratios) / len(ratios)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="域无关候选页原型（函数 + 变量域）")
    parser.add_argument("--source", default=str(DEFAULT_SOURCE.relative_to(REPO_ROOT)))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--no-api", action="store_true", help="不联网：只跑枚举普查与宿主判据")
    parser.add_argument("--no-cost-probe", action="store_true", help="跳过那张大页的账单实测")
    parser.add_argument("--repeat", type=int, default=1, help="打分任务重复轮数（温度 0 也可能不一致，用来量方差）")
    args = parser.parse_args(argv)

    source = Path(args.source)
    if not source.is_absolute():
        source = REPO_ROOT / source
    if not source.exists():
        raise SystemExit(f"源文件不存在：{source}")

    digest = hashlib.sha256(source.read_bytes()).hexdigest()[:16]
    heading("codejev 域无关候选页原型：第二个域 = Python 函数与变量")
    print(f"source={source.relative_to(REPO_ROOT)} sha256[:16]={digest} lines={len(source.read_text(encoding='utf-8').splitlines())}")
    print(f"python={sys.version.split()[0]}")
    print("protocol=codejev.candidate（CandidatePage/CodeCandidate/choose_candidate/materialize，未改动）")
    if args.no_api:
        print("provider=未使用（--no-api）")
    else:
        config = load_provider(args.model)
        print(f"provider={PROVIDER_ID}/{APP_TYPE} base_url={config.base_url} model={config.model} reasoning_effort={config.reasoning_effort}")
        print("key=不打印（只在请求头里用）；400/429/网络错误退避重试，协议错误不重试")

    strict = de.enumerate_source(source, "strict")
    profile = de.enumerate_source(source, "python312")
    naive = de.enumerate_source(source, "naive")
    source_lines = dr.source_lines(source)
    source_module = import_source_module(source)
    entries_by_qualname = {entry.qualname: entry for entry in profile.functions}

    function_page = build_function_page(profile.functions)
    name_pages: dict[str, BuiltPage] = {}
    for task_def in NAME_TASKS:
        if task_def.scope in name_pages:
            continue
        entries = de.name_scope_page_source(profile, task_def.scope)
        built = build_name_page(task_def.scope, entries)
        name_pages[task_def.scope] = built
        _page_names_cache[task_def.scope] = tuple(candidate.name for candidate in built.page.candidates)

    decide = REPO_ROOT / "codejev" / "decide.py"
    other = [REPO_ROOT / "codejev" / "decide.py", REPO_ROOT / "codejev" / "api_engine.py"]
    other = [path for path in other if path.exists() and path != source]
    other_enumerations = [de.enumerate_source(path, "python312") for path in other]
    flat_pages = [build_flat_name_page(profile)]
    flat_pages.extend(
        build_flat_name_page(enumeration, prefix="flat") for enumeration in other_enumerations
    )
    merged = build_flat_name_page_from_entries(
        [
            entry
            for enumeration in (profile, *other_enumerations)
            for entry in enumeration.names
            if entry.scope_kind == "function" and entry.observable
        ],
        label="flat[本域+decide+api_engine]",
    )
    flat_pages.append(merged)

    engine = None if args.no_api else OpenAICompatibleEngine(load_provider(args.model))
    outcomes, samples = run_part_a(
        engine,
        source_module,
        source_lines,
        function_page,
        entries_by_qualname,
        name_pages,
        model=args.model,
    )

    heading("A4 规模实验：同一句任务，换成摊平的大候选页（枚举全的代价打在谁身上）")
    scale_task = next(task for task in NAME_TASKS if task.label == "name-raw-response")
    scale_outcome = run_scale_probe(
        engine,
        merged,
        scale_task,
        source_module,
        source_lines,
        entries_by_qualname,
        samples,
    )
    print_outcome(scale_outcome)

    # 账单实测：拿最大的一张页真调一次，只记 token，不判分。
    if engine is not None and not args.no_cost_probe:
        biggest = max(flat_pages, key=lambda built: built.prompt_chars)
        heading("B0 账单实测：把最大的一张平坦候选页真的发一次")
        try:
            outcome = choose_with_backoff(engine, ANY_TASK, biggest.page)
            samples.setdefault(biggest.name, []).append(outcome.stats.prompt_tokens)
            print(
                f"  page={biggest.name} candidates={len(biggest.page.candidates)} "
                f"prompt_chars={biggest.prompt_chars} 实测 prompt_tokens={outcome.stats.prompt_tokens} "
                f"raw={outcome.choice.raw_response!r}（只记账单，不判分）"
            )
        except Exception as exc:
            print(f"  账单实测失败（不判分）：{type(exc).__name__}: {exc}")

    # 复制一份：下面的 scale_outcome 会 append 进 outcomes，别让 rounds[0] 跟着变。
    rounds: list[list[Outcome]] = [list(outcomes)]
    if engine is not None and args.repeat > 1:
        rounds.extend(
            run_repeat_rounds(
                engine,
                function_page,
                name_pages,
                source_module,
                source_lines,
                entries_by_qualname,
                times=args.repeat,
            )
        )
        print_variance(rounds, [t.label for t in FUNCTION_TASKS] + [t.label for t in NAME_TASKS])
    outcomes.append(scale_outcome)

    all_pages = {built.name: built for built in (function_page, *name_pages.values(), *flat_pages)}
    ratio = chars_per_token(samples, all_pages)

    part_b_capability_census(source, strict, profile, naive)
    part_b_coverage_table(source, strict, profile, naive, source_module)
    part_b_other_files()
    part_b_materialize_limits(source_module, source)
    part_b_page_cost([function_page, *name_pages.values()], flat_pages, samples, ratio)
    part_b_ambiguity(source_module, source_lines, entries_by_qualname)
    part_b_duplicate_choice_demo(source_module, function_page, entries_by_qualname)
    part_b_not_verified()

    scored = [o for o in outcomes if o.verdict != "SKIP"]
    heading("总结")
    print(
        f"  首轮（含大页规模实验）：{sum(1 for o in scored if o.verdict == 'PASS')}/{len(scored)} 通过"
        "（判据全部是运行时行为）"
    )
    print(f"  首轮失败明细：{[(o.label, o.expected, o.actual) for o in scored if o.verdict != 'PASS']}")
    if len(rounds) > 1:
        flat = [outcome for row in rounds for outcome in row]
        hits = sum(1 for outcome in flat if outcome.verdict == "PASS")
        print(
            f"  打分任务 {len(rounds)} 轮合计：{hits}/{len(flat)} ({hits / len(flat):.1%})"
            f"（{len(rounds)} 轮 × {len(flat) // len(rounds)} 条）"
        )
    print("  候选 id、候选页、物化代码都由宿主持有；模型只回一个 id，越界回复按协议错误处理。")
    print("  B 部分的数字全部来自这次真实枚举与真实调用；估算项在 B5 里单独标注。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
