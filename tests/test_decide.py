"""decide.py 的测试：候选提取、决策校验、确定性组装、端到端。

全部离线运行：用小模型的 ScriptedEngine 给出决策，不加载权重、不联网、不写盘。
最强的证据是最后把重写后的函数 exec 起来，直接验证它真的过滤、真的只返回选定的键。
"""

from __future__ import annotations

import ast
import json
import re
from dataclasses import replace

import pytest

from codejev.contracts import Action, Kind, content_hash, normalize_body
from codejev.decide import (
    DECISION_SYSTEM_PROMPT,
    Candidate,
    Candidates,
    Decision,
    DecisionError,
    assemble,
    build_decision_prompt,
    describe_decision,
    extract,
    parse_decision,
    run_decision,
)
from codejev.fixtures import (
    BROKEN_MODULE,
    NO_FUNCTION_MODULE,
    ORDERS_MODULE,
    PRODUCTS_MODULE,
    USERS_MODULE,
)
from codejev.model import ScriptedEngine

TARGET = "app/users.py"

# 固定任务的那句示例指令。
FILTER_INSTRUCTION = "只保留 active 为真的项，返回 id 和 name，保持原顺序。"

# 原文完全不过滤：用来验证“新加一个过滤条件”也能用选择式表达。
BENCH_SOURCE_FOR_FILTER = '''"""用户列表：当前返回全部字段。"""


def active_users(users):
    """返回用户，保持原顺序。"""
    result = []
    for user in users:
        result.append(
            {"id": user["id"], "name": user["name"], "active": user["active"]}
        )
    return result
'''


def _lines(text: str) -> list[str]:
    """按 \\n 切行，保留空行，便于比较函数外内容。"""
    return text.split("\n")


def _outside(text: str, start_line: int, end_line: int) -> tuple[str, str]:
    """函数 span 之前、之后的原文（1 基行号）。"""
    lines = _lines(text)
    return "\n".join(lines[: start_line - 1]), "\n".join(lines[end_line:])


def _run(source: str, namespace: dict[str, object], name: str, arg: object) -> object:
    """把重写后的源码 exec 起来，调用其中的函数，拿到真实运行结果。"""
    exec(compile(source, "<assemble>", "exec"), namespace)  # noqa: S102 - 测试专用
    return namespace[name](arg)  # type: ignore[operator]


# --------------------------------------------------------------------------
# extract：确定性提取
# --------------------------------------------------------------------------


def test_extract_finds_function_by_name_and_span() -> None:
    candidates = extract(USERS_MODULE, "active_users")
    assert candidates.function_name == "active_users"
    assert candidates.function_id == "fn0"
    assert (candidates.start_line, candidates.end_line) == (9, 15)
    # 行号确实对着原文的函数定义与最后一行
    lines = _lines(USERS_MODULE)
    assert lines[candidates.start_line - 1] == "def active_users(users):"
    assert lines[candidates.end_line - 1] == "    return result"


def test_extract_field_names_and_ids_are_stable() -> None:
    """字段候选按源码出现顺序编号：f0,f1,…；重复出现不重复占号。

    条件候选同样编号，且覆盖所有观察到的名字（判断里出现过的排最前）。
    """
    candidates = extract(USERS_MODULE, "active_users")
    assert candidates.fields == (
        Candidate("f0", "active", "field"),
        Candidate("f1", "deleted", "field"),
        Candidate("f2", "id", "field"),
        Candidate("f3", "name", "field"),
    )
    assert candidates.conditions == (
        Candidate("c0", "active", "condition"),
        Candidate("c1", "deleted", "condition"),
        Candidate("c2", "id", "condition"),
        Candidate("c3", "name", "condition"),
    )


def test_extract_is_deterministic() -> None:
    """同一份源码每次提取出完全一样的候选（id 是宿主生成的稳定身份）。"""
    assert extract(USERS_MODULE, "active_users") == extract(USERS_MODULE, "active_users")


def test_extract_same_name_can_be_both_field_and_condition() -> None:
    """active 既能返回也能过滤：两张表都出现，id 不同（f0 与 c0）。"""
    candidates = extract(USERS_MODULE, "active_users")
    field_names = {item.name for item in candidates.fields}
    condition_names = {item.name for item in candidates.conditions}
    assert "active" in field_names and "active" in condition_names
    ids = [item.id for item in candidates.fields] + [item.id for item in candidates.conditions]
    assert len(ids) == len(set(ids))


def test_extract_without_name_uses_first_public_function() -> None:
    candidates = extract(USERS_MODULE)
    assert candidates.function_name == "active_users"


def test_extract_handles_attribute_style() -> None:
    """属性读取也是字段；方法名（rows.append）不算字段。

    条件候选要覆盖所有观察到的名字（这样才能表达“新加一个过滤条件”），
    已经当过判断的（paid / shipped）排在最前面。
    """
    candidates = extract(ORDERS_MODULE, "paid_orders")
    names = [item.name for item in candidates.fields]
    assert names == ["paid", "shipped", "order_id", "total"]
    assert "append" not in names
    condition_names = [item.name for item in candidates.conditions]
    assert condition_names[:2] == ["paid", "shipped"]  # 原文判断里的排最前
    assert set(condition_names) == {"paid", "shipped", "order_id", "total"}


def test_extract_handles_get_style() -> None:
    """`item.get("k")` 的键算字段，键与下标混用也各记一次。"""
    candidates = extract(PRODUCTS_MODULE, "visible_products")
    assert [item.name for item in candidates.fields] == ["visible", "sku", "price"]
    assert [item.name for item in candidates.conditions][0] == "visible"


def test_extract_class_method_keeps_indent_info() -> None:
    """类方法只取方法自身的行范围，类头行不算进去。"""
    candidates = extract(ORDERS_MODULE, "paid_orders")
    lines = _lines(ORDERS_MODULE)
    assert lines[candidates.start_line - 1] == "def paid_orders(orders):"
    assert candidates.start_line > 1


def test_extract_missing_function_raises() -> None:
    with pytest.raises(DecisionError, match="没有函数"):
        extract(USERS_MODULE, "no_such_function")


def test_extract_source_without_function_raises() -> None:
    with pytest.raises(DecisionError):
        extract(NO_FUNCTION_MODULE)


def test_extract_invalid_source_raises() -> None:
    with pytest.raises(DecisionError, match="源码无法解析"):
        extract(BROKEN_MODULE)


# --------------------------------------------------------------------------
# build_decision_prompt：短，且只给候选
# --------------------------------------------------------------------------


def test_prompt_is_short_and_lists_candidates_only() -> None:
    candidates = extract(USERS_MODULE, "active_users")
    messages = build_decision_prompt(FILTER_INSTRUCTION, candidates)
    assert [m["role"] for m in messages] == ["system", "user"]
    assert messages[0]["content"] == DECISION_SYSTEM_PROMPT
    user = messages[1]["content"]
    assert FILTER_INSTRUCTION in user
    for item in candidates.fields + candidates.conditions:
        assert f"{item.id}={item.name}" in user
    # 只回 JSON，不要解释、不要例子
    assert "只回 JSON" in messages[0]["content"] + user
    for word in ("例如", "思考", "分析"):
        assert word not in user
    assert len(user) < 400  # 短是这条路径的意义


# --------------------------------------------------------------------------
# parse_decision：严格校验
# --------------------------------------------------------------------------


def _candidates() -> Candidates:
    return extract(USERS_MODULE, "active_users")


def test_parse_accepts_valid_json() -> None:
    decision = parse_decision(
        '{"function": "fn0", "filter_field": "c0", "return_fields": ["f2", "f3"]}',
        _candidates(),
    )
    assert decision == Decision(function_id="fn0", filter_field="c0", return_fields=("f2", "f3"))


def test_parse_accepts_null_filter() -> None:
    decision = parse_decision(
        '{"function": "fn0", "filter_field": null, "return_fields": ["f2"]}', _candidates()
    )
    assert decision.filter_field is None


def test_parse_accepts_fenced_json() -> None:
    """小模型偶尔仍带围栏：围栏只是展示，去掉后照常解析。"""
    text = '```json\n{"function": "fn0", "filter_field": "c0", "return_fields": ["f2"]}\n```'
    decision = parse_decision(text, _candidates())
    assert decision.return_fields == ("f2",)


def test_parse_accepts_json_with_surrounding_prose() -> None:
    """外面有少量说明文字时取第一个 JSON 对象。"""
    text = '好的：{"function": "fn0", "filter_field": null, "return_fields": ["f3"]} 以上。'
    assert parse_decision(text, _candidates()).return_fields == ("f3",)


def test_parse_accepts_json_with_braces_in_prose() -> None:
    """说明文字里先出现不配平的 { 也不影响取出真正的决策对象。"""
    text = '按 {要求} 给出：{"function": "fn0", "filter_field": "c1", "return_fields": ["f2"]}'
    assert parse_decision(text, _candidates()).filter_field == "c1"


def test_parse_rejects_unknown_field_id() -> None:
    with pytest.raises(DecisionError, match="未知的字段 id"):
        parse_decision(
            '{"function": "fn0", "filter_field": null, "return_fields": ["f9"]}', _candidates()
        )


def test_parse_rejects_unknown_condition_id() -> None:
    with pytest.raises(DecisionError, match="未知的条件 id"):
        parse_decision(
            '{"function": "fn0", "filter_field": "c7", "return_fields": ["f2"]}', _candidates()
        )


def test_parse_rejects_model_invented_field_name() -> None:
    """模型自造标识一律拒绝：只能选候选 id，不能写真实字段名。"""
    with pytest.raises(DecisionError):
        parse_decision(
            '{"function": "fn0", "filter_field": null, "return_fields": ["email"]}',
            _candidates(),
        )


def test_parse_rejects_unknown_function_id() -> None:
    with pytest.raises(DecisionError, match="未知的函数 id"):
        parse_decision(
            '{"function": "fn1", "filter_field": null, "return_fields": ["f2"]}', _candidates()
        )


def test_parse_rejects_non_json() -> None:
    with pytest.raises(DecisionError, match="不是 JSON"):
        parse_decision("好的，我建议只返回 id 和 name。", _candidates())


def test_parse_rejects_empty_text() -> None:
    with pytest.raises(DecisionError):
        parse_decision("", _candidates())


def test_parse_rejects_malformed_json() -> None:
    """残缺 JSON 不修补、不猜测。"""
    with pytest.raises(DecisionError):
        parse_decision('{"function": "fn0", "return_fields": ["f2",]}', _candidates())


def test_parse_rejects_missing_keys() -> None:
    for text in (
        '{"function": "fn0", "return_fields": ["f2"]}',
        '{"function": "fn0", "filter_field": null}',
        '{"filter_field": null, "return_fields": ["f2"]}',
    ):
        with pytest.raises(DecisionError, match="缺少字段"):
            parse_decision(text, _candidates())


def test_parse_rejects_extra_keys() -> None:
    """决策形状固定，多余的键（比如模型想塞代码或路径）直接拒绝。"""
    with pytest.raises(DecisionError, match="多余"):
        parse_decision(
            '{"function": "fn0", "filter_field": null, "return_fields": ["f2"],'
            ' "code": "pass", "target": "evil.py"}',
            _candidates(),
        )


def test_parse_rejects_wrong_types() -> None:
    for text in (
        '{"function": 0, "filter_field": null, "return_fields": ["f2"]}',
        '{"function": "fn0", "filter_field": 0, "return_fields": ["f2"]}',
        '{"function": "fn0", "filter_field": null, "return_fields": "f2"}',
        '{"function": "fn0", "filter_field": null, "return_fields": [2]}',
        '{"function": "fn0", "filter_field": null, "return_fields": {}}',
        # 顶层不是对象
        '["f2"]',
    ):
        with pytest.raises(DecisionError):
            parse_decision(text, _candidates())


def test_parse_rejects_empty_return_fields() -> None:
    with pytest.raises(DecisionError, match="不能为空"):
        parse_decision(
            '{"function": "fn0", "filter_field": "c0", "return_fields": []}', _candidates()
        )


def test_parse_rejects_duplicate_return_fields() -> None:
    with pytest.raises(DecisionError, match="重复"):
        parse_decision(
            '{"function": "fn0", "filter_field": null, "return_fields": ["f2", "f2"]}',
            _candidates(),
        )


# --------------------------------------------------------------------------
# 精简格式（f / r）：现在的输出契约，同时保留旧长键的向后兼容
# --------------------------------------------------------------------------


def test_prompt_asks_for_terse_shape_and_disables_sort_by_default() -> None:
    """默认只开放 f / r；排序槽位必须由调用方显式启用。

    禁用侧的说明放在**用户消息**的 sort_note 里，不放在系统提示里：把禁令写进
    系统提示会把那行 `正确形状示例` 挤掉，而实测（2026-09-20）表明删掉示例会让
    模型整个省掉 f 键，过滤被静默丢弃。
    """
    candidates = extract(USERS_MODULE, "active_users")
    messages = build_decision_prompt(FILTER_INSTRUCTION, candidates)
    system = messages[0]["content"]
    user = messages[1]["content"]
    assert "只回这两个键：f（条件 id" in system
    assert "正确形状示例" in system
    assert "排序槽位必须由调用方明确启用" not in system
    assert "排序槽位：未启用。严禁输出 s 或 d" in user
    assert "（r 只能选这里）" in user
    assert "（f 只能选这里，不过滤时用 null）" in user
    # 示例只出现在系统提示里，用户消息保持无例子（短是这条路径的意义）。
    assert "{" not in user and "}" not in user
    # 旧的长键不出现在提示里（模型不用回抄函数 id）。
    assert "function" not in system + user
    assert "filter_field" not in system + user
    assert "return_fields" not in system + user


def test_prompt_explicitly_enables_sort_slot_without_a_fixed_sort_example() -> None:
    """启用态在基础提示后面追加启用条款；不放任何会被模型抄走的排序示例。"""
    candidates = extract(SORT_INSERT_SOURCE, "active_users")
    messages = build_decision_prompt(
        "只保留 active 为真的项，返回 id 和 name，按名字降序。",
        candidates,
        sort_enabled=True,
    )
    system = messages[0]["content"]
    user = messages[1]["content"]
    assert "本次调用方已明确启用排序槽位" in system
    assert "排序槽位：已启用" in user
    assert "s（排序字段 id" in system
    # 基础 f / r 示例仍在（实测删掉会让模型丢掉 f 键），但排序键的示例一个都不给。
    assert "正确形状示例" in system
    assert '"s"' not in system and '"d"' not in system
    assert '"s": "f1"' not in system
    assert '"d": "desc"' not in system


# --------------------------------------------------------------------------
# 提示里的示例：2026-09-20 “过滤被静默丢掉”那条回归的防线
# --------------------------------------------------------------------------

# 提示里的 JSON 示例：只匹配单层对象（决策契约只有一层，嵌套的不算示例）。
_JSON_EXAMPLE = re.compile(r"\{[^{}]*\}")
# 会被小模型逐字照抄的“排序键”写法：JSON 键（"s":），或历史上出现过的 s=f1 / d=desc。
_SORT_KEY_WRITING = re.compile(
    r"""["'][sd]["']\s*:|(?<![A-Za-z0-9_])[sd]\s*=\s*(?:f\d+|c\d+|asc|desc)"""
)
# 候选 id 的真实形状：宿主生成的 f0 / c1（示例必须是这个形状，不是占位措辞）。
_CANDIDATE_ID_SHAPE = re.compile(r"[fc]\d+")


def _examples(text: str) -> list[str]:
    """文本里出现的所有 JSON 示例原文。"""
    return _JSON_EXAMPLE.findall(text)


def _prompt_messages(
    source: str, function: str, instruction: str, sort_enabled: bool = False
) -> tuple[str, str]:
    """取 (系统提示, 用户消息)；省得每个用例都重复一遍 build_decision_prompt。"""
    messages = build_decision_prompt(
        instruction, extract(source, function), sort_enabled=sort_enabled
    )
    assert [m["role"] for m in messages] == ["system", "user"]
    return messages[0]["content"], messages[1]["content"]


def test_default_prompt_carries_one_real_filter_example() -> None:
    """未启用排序时，系统提示必须带一个带非空 f 的真实候选形状示例。

    只写形状描述、不给示例时，1.5B 会整个省掉 f 键（回 `{"r": [...]}`），过滤被
    静默丢掉而产物仍然合法——这就是 2026-09-20 那条回归的本体。
    """
    system, _user = _prompt_messages(USERS_MODULE, "active_users", FILTER_INSTRUCTION)

    examples = _examples(system)
    assert len(examples) == 1, system
    example = json.loads(examples[0])  # 必须是能解析的 JSON，不是“c0 或 null”这类措辞
    assert example.get("f"), example  # f 非空：这条断言就是回归的靶心
    assert example["r"], example
    assert set(example) == {"f", "r"}  # 示例里只有基础键，一个排序键都没有
    assert _CANDIDATE_ID_SHAPE.fullmatch(example["f"]), example
    assert all(_CANDIDATE_ID_SHAPE.fullmatch(item) for item in example["r"]), example
    assert "或" not in examples[0]  # 占位措辞会被逐字照抄，不许出现
    assert "null" not in examples[0]


@pytest.mark.parametrize(
    ("source", "function"),
    [
        pytest.param(USERS_MODULE, "active_users", id="dict-style"),
        pytest.param(ORDERS_MODULE, "paid_orders", id="attribute-style"),
        pytest.param(PRODUCTS_MODULE, "visible_products", id="get-style"),
        pytest.param(BENCH_SOURCE_FOR_FILTER, "active_users", id="no-filter-in-source"),
    ],
)
@pytest.mark.parametrize(
    "instruction",
    [
        pytest.param(FILTER_INSTRUCTION, id="no-sort-asked"),
        # 连“任务里明说按 id 降序”的指令也不给排序写法：能不能排序只由 sort_enabled 决定。
        pytest.param(
            "只保留 active 为真的项，返回 id 和 name，按 id 降序。", id="sort-asked-out-of-band"
        ),
    ],
)
def test_default_prompt_never_shows_a_sort_key(
    source: str, function: str, instruction: str
) -> None:
    """未启用排序时，系统提示与用户消息里都不许出现 s / d 键的写法。

    历史回归：系统提示里出现 `s=f1,d=desc` 示例后，没要求排序的 5 条指令被模型
    全部错误地加上排序。这里把那种写法整个挡在提示外面（含用户消息），并确认
    禁令确实写在用户消息的 sort_note 里，而不是靠“系统提示里没有示例”这条默契。
    """
    system, user = _prompt_messages(source, function, instruction)

    for text in (system, user):
        assert _SORT_KEY_WRITING.search(text) is None, text
        for example in _examples(text):
            assert not set(json.loads(example)) & {"s", "d"}, example
    # 基础提示里唯一那个示例仍然只讲 f / r。
    assert len(_examples(system)) == 1, system
    assert set(json.loads(_examples(system)[0])) == {"f", "r"}
    # 禁用侧的禁令在用户消息里；启用侧的说明一个字都不出现。
    assert "严禁输出 s 或 d" in user
    assert "已启用" not in system + user


def test_sort_explanation_appears_only_when_the_slot_is_enabled() -> None:
    """启用排序只是**追加**一段说明：基础提示（含 f 示例）原样保留，且它不带示例。"""
    default_system, default_user = _prompt_messages(
        USERS_MODULE, "active_users", FILTER_INSTRUCTION
    )
    sort_system, sort_user = _prompt_messages(
        USERS_MODULE, "active_users", FILTER_INSTRUCTION, sort_enabled=True
    )

    assert sort_system.startswith(default_system + "\n")
    extra = sort_system[len(default_system) + 1 :]
    assert "启用排序槽位" in extra
    assert "s" in extra and "d" in extra
    # 排序说明只讲怎么写、不给可抄的 JSON 示例（历史回归正是抄示例抄出来的）。
    assert _examples(extra) == []
    assert _examples(sort_system) == _examples(default_system)
    assert "启用排序槽位" not in default_system + default_user
    assert "排序槽位：已启用" in sort_user


def test_parse_accepts_terse_json() -> None:
    decision = parse_decision('{"f": "c0", "r": ["f2", "f3"]}', _candidates())
    assert decision == Decision(function_id="fn0", filter_field="c0", return_fields=("f2", "f3"))


def test_parse_terse_null_and_absent_filter() -> None:
    """f 写成 null、或整个键省掉，都按不过滤处理；函数身份来自候选表。"""
    for text in ('{"f": null, "r": ["f2"]}', '{"r": ["f2"]}'):
        decision = parse_decision(text, _candidates())
        assert decision.filter_field is None
        assert decision.function_id == "fn0"


def test_parse_terse_preserves_return_field_order() -> None:
    decision = parse_decision('{"f": "c1", "r": ["f3", "f2"]}', _candidates())
    assert decision.return_fields == ("f3", "f2")


def test_parse_terse_accepts_quoted_null() -> None:
    """模型把 f 写成字符串 "null"：写法差异，仍按不过滤处理。"""
    assert parse_decision('{"f": "null", "r": ["f2"]}', _candidates()).filter_field is None


def test_parse_terse_rejects_mixed_keys() -> None:
    """两套键混在一个对象里：格式说不清，一律拒绝。"""
    for text in (
        '{"function": "fn0", "f": "c0", "r": ["f2"]}',
        '{"f": "c0", "r": ["f2"], "return_fields": ["f2"]}',
    ):
        with pytest.raises(DecisionError, match="混用"):
            parse_decision(text, _candidates())


def test_parse_legacy_long_keys_still_parse() -> None:
    """向后兼容：改动前生成的长键决策与夹具仍然解析，语义完全一样。"""
    decision = parse_decision(
        '{"function": "fn0", "filter_field": "c0", "return_fields": ["f2", "f3"]}',
        _candidates(),
    )
    assert decision == Decision(function_id="fn0", filter_field="c0", return_fields=("f2", "f3"))


def test_parse_terse_rejects_unknown_ids() -> None:
    with pytest.raises(DecisionError, match="未知的条件 id"):
        parse_decision('{"f": "c9", "r": ["f2"]}', _candidates())
    with pytest.raises(DecisionError, match="未知的字段 id"):
        parse_decision('{"f": "c0", "r": ["f9"]}', _candidates())


def test_parse_terse_rejects_empty_and_duplicate_return_fields() -> None:
    with pytest.raises(DecisionError, match="不能为空"):
        parse_decision('{"f": "c0", "r": []}', _candidates())
    with pytest.raises(DecisionError, match="重复"):
        parse_decision('{"f": null, "r": ["f2", "f2"]}', _candidates())


def test_parse_terse_rejects_wrong_types_extra_keys_and_missing_r() -> None:
    for text in (
        '{"f": 0, "r": ["f2"]}',
        '{"r": "f2"}',
        '{"r": [2]}',
        '{"f": "c0", "r": ["f2"], "code": "pass"}',
        '{"f": "c0"}',  # 只认 f 不认 r：缺少要返回的字段
        '["f2"]',  # 顶层不是对象
    ):
        with pytest.raises(DecisionError):
            parse_decision(text, _candidates())


def test_assemble_still_verifies_function_identity_from_source() -> None:
    """精简格式不回抄函数 id，身份由宿主保证：候选过期时 assemble 仍然拒绝。"""
    candidates = extract(USERS_MODULE, "active_users")
    stale = replace(candidates, start_line=candidates.start_line + 1)
    decision = parse_decision('{"f": "c0", "r": ["f2"]}', candidates)
    with pytest.raises(DecisionError, match="函数位置已改变"):
        assemble(USERS_MODULE, stale, decision)


def test_run_decision_accepts_terse_engine_reply() -> None:
    """端到端：模型回精简 JSON，宿主照样组装出同一份正文。"""
    engine = ScriptedEngine(['{"f": "c0", "r": ["f2", "f3"]}'])
    artifact, decision, candidates = run_decision(
        engine, FILTER_INSTRUCTION, USERS_MODULE, TARGET, "active_users"
    )
    assert decision == Decision(function_id="fn0", filter_field="c0", return_fields=("f2", "f3"))
    assert artifact.body == normalize_body(assemble(USERS_MODULE, candidates, decision))
    assert 'if user["active"]:' in artifact.body


# --------------------------------------------------------------------------
# assemble：确定性重写
# --------------------------------------------------------------------------


def _decision(candidate_set: Candidates, filter_id: str | None, fields: list[str]) -> Decision:
    return parse_decision(
        json.dumps(
            {"function": candidate_set.function_id, "filter_field": filter_id, "return_fields": fields}
        ),
        candidate_set,
    )


def test_assemble_filters_and_returns_only_chosen_fields() -> None:
    candidates = extract(USERS_MODULE, "active_users")
    out = assemble(USERS_MODULE, candidates, _decision(candidates, "c0", ["f2", "f3"]))
    ast.parse(out)  # 仍是合法 Python

    body = _lines(out)[candidates.start_line - 1 :]
    text = "\n".join(body)
    assert 'if user["active"]:' in text  # 过滤条件来自决策
    assert '{"id": user["id"], "name": user["name"]}' in text  # 只返回选定字段
    assert '"deleted"' not in text  # 未选定的字段不再出现
    assert "    return result" in out  # 累加与返回保持原风格


def test_assemble_preserves_everything_outside_the_function() -> None:
    candidates = extract(USERS_MODULE, "active_users")
    out = assemble(USERS_MODULE, candidates, _decision(candidates, "c0", ["f2"]))
    head_src, tail_src = _outside(USERS_MODULE, candidates.start_line, candidates.end_line)
    head_out, tail_out = _outside(out, candidates.start_line, len(_lines(out)))
    assert head_out == head_src
    assert tail_out == "" and tail_src == ""
    # 函数外的行数没有变化（这里函数是文件最后一段）
    assert len(_lines(USERS_MODULE)[: candidates.start_line - 1]) == len(
        _lines(out)[: candidates.start_line - 1]
    )


def test_assemble_keeps_file_edges_when_function_is_in_the_middle() -> None:
    """函数在中间时：前后内容逐字保留，行数不变。"""
    source = (
        '"""模块说明。"""\n'
        "import json\n"
        "\n"
        "CONST = 1\n"
        "\n"
        "\n"
        "def users(rows):\n"
        '    """筛选。"""\n'
        "    out = []\n"
        "    for row in rows:\n"
        '        if row["active"]:\n'
        '            out.append({"id": row["id"], "name": row["name"]})\n'
        "    return out\n"
        "\n"
        "\n"
        "def tail():\n"
        "    return CONST\n"
    )
    candidates = extract(source, "users")
    # 这段源码里 active 先出现：f0=active、f1=id、f2=name；原文已带同样的 if。
    out = assemble(source, candidates, _decision(candidates, "c0", ["f1", "f2"]))
    ast.parse(out)

    src_lines, out_lines = _lines(source), _lines(out)
    assert out_lines[: candidates.start_line - 1] == src_lines[: candidates.start_line - 1]
    assert out_lines[-(len(src_lines) - candidates.end_line) :] == src_lines[candidates.end_line :]
    assert out.startswith('"""模块说明。"""\nimport json\n')
    assert out.endswith("def tail():\n    return CONST\n")
    # 原文本来就是“过滤 + 缩进一行”的形状，所以函数内行数不变。
    assert len(out_lines) == len(src_lines)


def test_assemble_preserves_docstring() -> None:
    candidates = extract(ORDERS_MODULE, "paid_orders")
    out = assemble(ORDERS_MODULE, candidates, _decision(candidates, None, ["f2", "f3"]))
    ast.parse(out)
    assert '"""返回已付款订单的编号与金额。"""' in out


def test_assemble_keeps_accumulator_initialization() -> None:
    candidates = extract(ORDERS_MODULE, "paid_orders")
    out = assemble(ORDERS_MODULE, candidates, _decision(candidates, None, ["f2"]))
    assert "    rows = []" in out
    assert "        rows.append(" in out
    assert "    return rows" in out


def test_assemble_without_filter_adds_no_condition() -> None:
    candidates = extract(USERS_MODULE, "active_users")
    out = assemble(USERS_MODULE, candidates, _decision(candidates, None, ["f2", "f3"]))
    ast.parse(out)
    body = "\n".join(_lines(out)[candidates.start_line - 1 :])
    # 注意：函数名本身叫 active_users，所以只能查“是否读取了这个字段”，
    # 不能对 "active" 做子串匹配。
    assert 'user["active"]' not in body  # 过滤字段没有被读取
    assert not any(
        isinstance(node, ast.If) for stmt in ast.parse(out).body if isinstance(stmt, ast.FunctionDef)
        for node in ast.walk(stmt)
    )
    assert '{"id": user["id"], "name": user["name"]}' in body


def test_assemble_keeps_requested_output_order() -> None:
    candidates = extract(USERS_MODULE, "active_users")
    out = assemble(USERS_MODULE, candidates, _decision(candidates, "c0", ["f3", "f2"]))
    body = "\n".join(_lines(out)[candidates.start_line - 1 :])
    assert body.index('"name"') < body.index('"id"')


def test_assemble_keeps_attribute_style_for_objects() -> None:
    candidates = extract(ORDERS_MODULE, "paid_orders")
    out = assemble(ORDERS_MODULE, candidates, _decision(candidates, "c0", ["f2", "f3"]))
    ast.parse(out)
    assert 'if order.paid:' in out
    assert '{"order_id": order.order_id, "total": order.total}' in out


def test_assemble_keeps_get_style() -> None:
    candidates = extract(PRODUCTS_MODULE, "visible_products")
    out = assemble(PRODUCTS_MODULE, candidates, _decision(candidates, "c0", ["f1", "f2"]))
    ast.parse(out)
    assert 'if product.get("visible"):' in out
    assert '{"sku": product["sku"], "price": product.get("price")}' in out


def test_assemble_rejects_stale_candidates() -> None:
    """源码变了（或候选来自别的文件）时拒绝重写，不猜。"""
    candidates = extract(USERS_MODULE, "active_users")
    with pytest.raises(DecisionError):
        assemble(ORDERS_MODULE, candidates, _decision(candidates, "c0", ["f0"]))


def test_assemble_unsupported_shape_is_rejected_not_guessed() -> None:
    source = 'def f(rows):\n    for row in rows:\n        yield {"a": row["a"]}\n'
    candidates = extract(source)
    with pytest.raises(DecisionError, match="不支持"):
        assemble(source, candidates, _decision(candidates, None, ["f0"]))


def test_assemble_indented_method_keeps_indentation() -> None:
    """类方法要保住自己的缩进，函数外内容一字不动。"""
    source = (
        "class Repo:\n"
        '    """仓库。"""\n'
        "\n"
        "    def items(self, rows):\n"
        '        """返回条目。"""\n'
        "        result = []\n"
        "        for row in rows:\n"
        '            result.append({"code": row["code"]})\n'
        "        return result\n"
        "\n"
        "    def other(self):\n"
        "        return 1\n"
    )
    candidates = extract(source, "items")
    out = assemble(source, candidates, _decision(candidates, None, ["f0"]))
    ast.parse(out)
    assert out.startswith('class Repo:\n    """仓库。"""\n\n    def items(self, rows):')
    assert out.endswith("    def other(self):\n        return 1\n")
    assert "\n        result = []\n" in out
    assert "\n            result.append(" in out


# --------------------------------------------------------------------------
# 运行时验证：重写出来的函数真的按决策工作
# --------------------------------------------------------------------------


def test_assembled_function_filters_and_trims_at_runtime() -> None:
    """最强证据：把重写后的模块 exec 起来，验证真实行为。"""
    candidates = extract(USERS_MODULE, "active_users")
    out = assemble(USERS_MODULE, candidates, _decision(candidates, "c0", ["f2", "f3"]))
    rows = [
        {"id": 1, "name": "Ada", "active": True, "deleted": False},
        {"id": 2, "name": "Bob", "active": False, "deleted": False},
        {"id": 3, "name": "Cid", "active": True, "deleted": True},
    ]
    # 决策只选了 active 这一个条件，所以 Cid（active=True）按指令就该留下。
    assert _run(out, {}, "active_users", rows) == [
        {"id": 1, "name": "Ada"},
        {"id": 3, "name": "Cid"},
    ]


def test_assembled_single_condition_cannot_keep_compound_original() -> None:
    """已知边界：选择式产出一次只表达一个条件。

    users 样本原来的过滤是 `active and not deleted`（复合条件），而决策里
    filter_field 只能选一个候选，因此重写后会放宽成只看 active。这是这条
    路径的真实限制：大模型要么接受这个简化，要么改用生成路线。
    """
    candidates = extract(USERS_MODULE, "active_users")
    out = assemble(USERS_MODULE, candidates, _decision(candidates, "c0", ["f2"]))
    body = "\n".join(_lines(out)[candidates.start_line - 1 :])
    assert 'user["deleted"]' not in body  # 原来的第二个条件没有保留
    assert 'if user["active"]:' in body


def test_assembled_function_without_filter_keeps_all_items_in_order() -> None:
    candidates = extract(USERS_MODULE, "active_users")
    out = assemble(USERS_MODULE, candidates, _decision(candidates, None, ["f2"]))
    rows = [{"id": 2, "name": "Bob"}, {"id": 1, "name": "Ada"}]
    assert _run(out, {}, "active_users", rows) == [{"id": 2}, {"id": 1}]


def test_assembled_function_works_on_attribute_objects() -> None:
    candidates = extract(ORDERS_MODULE, "paid_orders")
    out = assemble(ORDERS_MODULE, candidates, _decision(candidates, "c0", ["f3"]))
    orders = [
        type("Order", (), {"paid": True, "shipped": False, "order_id": "A1", "total": 10.0})(),
        type("Order", (), {"paid": False, "shipped": False, "order_id": "A2", "total": 20.0})(),
    ]
    assert _run(out, {}, "paid_orders", orders) == [{"total": 10.0}]


def test_assembled_function_works_on_mixed_styles() -> None:
    candidates = extract(PRODUCTS_MODULE, "visible_products")
    out = assemble(PRODUCTS_MODULE, candidates, _decision(candidates, "c0", ["f1", "f2"]))
    products = [{"sku": "S1", "price": 3, "visible": True}, {"sku": "S2", "price": 4}]
    assert _run(out, {}, "visible_products", products) == [{"sku": "S1", "price": 3}]


# --------------------------------------------------------------------------
# run_decision：完整流程
# --------------------------------------------------------------------------


def test_run_decision_end_to_end_returns_host_artifact() -> None:
    engine = ScriptedEngine(
        ['{"function": "fn0", "filter_field": "c0", "return_fields": ["f2", "f3"]}']
    )
    artifact, decision, candidates = run_decision(
        engine, FILTER_INSTRUCTION, USERS_MODULE, TARGET, "active_users"
    )

    expected = normalize_body(assemble(USERS_MODULE, candidates, decision))
    assert artifact.body == expected
    assert artifact.target == TARGET  # 目标来自调用方，不来自模型
    assert artifact.kind is Kind.CODE and artifact.action is Action.REPLACE
    assert artifact.content_hash  # 宿主计算，非空
    assert artifact.content_hash == content_hash(TARGET, artifact.body)
    assert decision == Decision(function_id="fn0", filter_field="c0", return_fields=("f2", "f3"))
    assert candidates.function_name == "active_users"
    ast.parse(artifact.body)
    assert artifact.raw_response == artifact.body
    # 提示里说明了这是选择式产出，正文由宿主组装
    assert any("宿主组装" in note for note in artifact.notes)
    assert any("模型决策原文" in note for note in artifact.notes)


def test_run_decision_prompt_stays_short() -> None:
    engine = ScriptedEngine(['{"function": "fn0", "filter_field": null, "return_fields": ["f2"]}'])
    run_decision(engine, FILTER_INSTRUCTION, USERS_MODULE, TARGET, "active_users")
    messages = engine.calls[0]
    assert len(messages) == 2
    assert len(messages[1]["content"]) < 400
    # 模型看到的是候选 id，不是整份源码
    assert "def active_users" not in messages[1]["content"]


def test_run_decision_propagates_decision_error_for_bogus_id() -> None:
    engine = ScriptedEngine(
        ['{"function": "fn0", "filter_field": "c0", "return_fields": ["f99"]}']
    )
    with pytest.raises(DecisionError, match="未知的字段 id"):
        run_decision(engine, FILTER_INSTRUCTION, USERS_MODULE, TARGET, "active_users")
    assert len(engine.calls) == 1  # 只问一次，不重试、不反复要求模型


def test_run_decision_propagates_decision_error_for_non_json() -> None:
    engine = ScriptedEngine(["我建议返回 id 和 name。"])
    with pytest.raises(DecisionError):
        run_decision(engine, FILTER_INSTRUCTION, USERS_MODULE, TARGET, "active_users")


def test_run_decision_accepts_fenced_model_reply() -> None:
    engine = ScriptedEngine(
        ['```json\n{"function": "fn0", "filter_field": "c0", "return_fields": ["f2"]}\n```']
    )
    artifact, decision, _ = run_decision(
        engine, FILTER_INSTRUCTION, USERS_MODULE, TARGET, "active_users"
    )
    assert decision.return_fields == ("f2",)
    assert artifact.body.count("user[") == 2  # 条件 + 返回各取一次


def test_run_decision_hash_changes_with_decision() -> None:
    """决策不同 → 正文不同 → 哈希不同，旧确认自然失效。"""
    first = run_decision(
        ScriptedEngine(['{"function":"fn0","filter_field":"c0","return_fields":["f2"]}']),
        FILTER_INSTRUCTION,
        USERS_MODULE,
        TARGET,
        "active_users",
    )[0]
    second = run_decision(
        ScriptedEngine(['{"function":"fn0","filter_field":null,"return_fields":["f2"]}']),
        FILTER_INSTRUCTION,
        USERS_MODULE,
        TARGET,
        "active_users",
    )[0]
    assert first.content_hash != second.content_hash


def test_run_decision_missing_function_raises_before_calling_model() -> None:
    engine = ScriptedEngine([])
    with pytest.raises(DecisionError):
        run_decision(engine, FILTER_INSTRUCTION, USERS_MODULE, TARGET, "nope")
    assert engine.calls == []  # 提取失败就不问模型


def test_describe_decision_is_one_line_with_names() -> None:
    candidates = extract(USERS_MODULE, "active_users")
    text = describe_decision(candidates, _decision(candidates, "c0", ["f2", "f3"]))
    assert "\n" not in text
    assert "active" in text and "id" in text and "name" in text


def test_run_decision_warns_when_original_condition_was_compound() -> None:
    """原函数是复合条件时，如实给出提示，不静默放宽。"""
    engine = ScriptedEngine(
        ['{"function": "fn0", "filter_field": "c0", "return_fields": ["f2", "f3"]}']
    )
    artifact, _, _ = run_decision(
        engine, FILTER_INSTRUCTION, USERS_MODULE, TARGET, "active_users"
    )
    assert any("复合条件" in note for note in artifact.notes)


def test_run_decision_has_no_compound_warning_for_single_condition() -> None:
    """products 样本的判断是单一条件（products 的可见性），不该出现复合条件提示。"""
    engine = ScriptedEngine(
        ['{"function": "fn0", "filter_field": "c0", "return_fields": ["f1", "f2"]}']
    )
    artifact, _, _ = run_decision(
        engine, FILTER_INSTRUCTION, PRODUCTS_MODULE, TARGET, "visible_products"
    )
    assert not any("复合条件" in note for note in artifact.notes)


def test_run_decision_body_matches_normalized_assemble() -> None:
    """raw_response 与 body 一致，落盘内容不会多出结尾空行差异。"""
    engine = ScriptedEngine(
        ['{"function": "fn0", "filter_field": "c0", "return_fields": ["f2", "f3"]}']
    )
    artifact, decision, candidates = run_decision(
        engine, FILTER_INSTRUCTION, USERS_MODULE, TARGET, "active_users"
    )
    assert artifact.raw_response == artifact.body
    assert artifact.body == normalize_body(assemble(USERS_MODULE, candidates, decision))


def test_parse_accepts_quoted_null_as_no_filter() -> None:
    """模型把 null 写成字符串 "null" 只是一种写法差异，按不过滤处理。"""
    candidates = extract(USERS_MODULE, "active_users")
    decision = parse_decision(
        '{"function": "fn0", "filter_field": "null", "return_fields": ["f2", "f3"]}',
        candidates,
    )
    assert decision.filter_field is None


def test_parse_rejects_unknown_string_that_is_not_null() -> None:
    """除了 null 写法之外，任何不存在的 id 仍然必须拒绝。"""
    candidates = extract(USERS_MODULE, "active_users")
    with pytest.raises(DecisionError, match="未知的条件 id"):
        parse_decision(
            '{"function": "fn0", "filter_field": "c99", "return_fields": ["f2"]}',
            candidates,
        )


def test_extract_offers_filter_candidates_not_yet_used_as_conditions() -> None:
    """原文完全不过滤时，也必须能把某个字段作为新加的条件候选提供给模型。

    否则“给这个函数加一个过滤条件”这条最常见的指令就没法用选择式表达。
    """
    candidates = extract(BENCH_SOURCE_FOR_FILTER, "active_users")
    names = [item.name for item in candidates.conditions]
    assert "active" in names


# --------------------------------------------------------------------------
# 守卫子句与循环后语句：逐字保留，且运行时真的生效
# --------------------------------------------------------------------------


def _assemble_decision(source: str, function: str, fields: list[str], condition: str | None):
    candidates = extract(source, function)
    return candidates, _decision(candidates, condition, fields)


# 前置守卫：空输入直接返回 []，手术必须逐字留住它。
GUARDED_SOURCE = '''"""订单。"""


def paid_orders(orders):
    """返回已付款订单。"""
    if not orders:
        return []
    rows = []
    for order in orders:
        if order.paid:
            rows.append({"id": order.order_id, "total": order.total})
    return rows
'''

# 循环之后还有排序：手术必须逐字留住 sort，输出顺序才不会变。
POST_LOOP_SOURCE = '''"""订单。"""


def paid_orders(orders):
    """返回已付款订单。"""
    rows = []
    for order in orders:
        if order.paid:
            rows.append({"id": order.order_id, "total": order.total})
    rows.sort(key=lambda r: -r["total"])
    return rows
'''

# 守卫与排序同时出现：run_decision 端到端用的样本。
GUARD_AND_SORT_SOURCE = '''"""订单。"""


def paid_orders(orders):
    """返回已付款订单，按金额从高到低。"""
    if not orders:
        return []
    rows = []
    for order in orders:
        if order.paid:
            rows.append({"id": order.order_id, "total": order.total})
    rows.sort(key=lambda r: -r["total"])
    return rows
'''


def test_assemble_preserves_leading_guard_clause_and_it_still_works() -> None:
    """前置守卫逐字留在原位；运行时空输入仍由守卫直接返回 []。"""
    candidates, decision = _assemble_decision(GUARDED_SOURCE, "paid_orders", ["f2", "f3"], "c0")
    out = assemble(GUARDED_SOURCE, candidates, decision)
    ast.parse(out)

    # 逐字保留，且行号未动：守卫与它的 return 还在函数开头那两行。
    index = _lines(GUARDED_SOURCE).index("    if not orders:")
    assert _lines(out)[index : index + 2] == ["    if not orders:", "        return []"]

    orders = [
        type("Order", (), {"paid": True, "order_id": "A1", "total": 10.0})(),
        type("Order", (), {"paid": False, "order_id": "A2", "total": 20.0})(),
    ]
    assert _run(out, {}, "paid_orders", orders) == [{"order_id": "A1", "total": 10.0}]
    assert _run(out, {}, "paid_orders", []) == []
    # None 只有守卫拦得住：守卫若被删掉，这里会抛 TypeError。
    assert _run(out, {}, "paid_orders", None) == []


def test_assemble_preserves_post_loop_statement_and_it_still_runs() -> None:
    """循环后的 sort 逐字留在原位；运行时结果真的按金额从高到低。"""
    candidates, decision = _assemble_decision(POST_LOOP_SOURCE, "paid_orders", ["f2", "f3"], "c0")
    out = assemble(POST_LOOP_SOURCE, candidates, decision)
    ast.parse(out)

    sort_line = '    rows.sort(key=lambda r: -r["total"])'
    index = _lines(POST_LOOP_SOURCE).index(sort_line)
    assert _lines(out)[index] == sort_line  # 同一行号上逐字不变
    assert _lines(out)[-2] == "    return rows"

    orders = [
        type("Order", (), {"paid": True, "order_id": "A1", "total": 10.0})(),
        type("Order", (), {"paid": True, "order_id": "A2", "total": 30.0})(),
        type("Order", (), {"paid": False, "order_id": "A3", "total": 99.0})(),
    ]
    # 没有排序时结果会是 [A1, A2]（10 在前）；这里必须是排过序的 30、10。
    assert _run(out, {}, "paid_orders", orders) == [
        {"order_id": "A2", "total": 30.0},
        {"order_id": "A1", "total": 10.0},
    ]


def test_assemble_still_accepts_the_supported_shape() -> None:
    """支持的标准形状不受影响：docstring + 初始化 + 循环 + return。"""
    candidates, decision = _assemble_decision(
        BENCH_SOURCE_FOR_FILTER, "active_users", ["f0", "f1"], "c2"
    )
    out = assemble(BENCH_SOURCE_FOR_FILTER, candidates, decision)
    ast.parse(out)
    assert 'if user["active"]:' in out


def test_run_decision_end_to_end_keeps_guard_and_sort() -> None:
    """完整流程跑带守卫与循环后排序的源码：正文两段都在，exec 起来真的生效。"""
    engine = ScriptedEngine(
        ['{"function": "fn0", "filter_field": "c0", "return_fields": ["f2", "f3"]}']
    )
    artifact, decision, candidates = run_decision(
        engine,
        "只保留已付款订单，返回编号和金额，按金额从高到低。",
        GUARD_AND_SORT_SOURCE,
        TARGET,
        "paid_orders",
    )

    assert artifact.body == normalize_body(assemble(GUARD_AND_SORT_SOURCE, candidates, decision))
    assert artifact.target == TARGET  # 目标来自调用方，不来自模型
    assert artifact.kind is Kind.CODE and artifact.action is Action.REPLACE
    assert artifact.raw_response == artifact.body
    ast.parse(artifact.body)
    assert decision == Decision(function_id="fn0", filter_field="c0", return_fields=("f2", "f3"))

    # 守卫与排序逐字留在正文里，行号与原文一致。
    src_lines, out_lines = _lines(GUARD_AND_SORT_SOURCE), _lines(artifact.body)
    kept = ("    if not orders:", "        return []", '    rows.sort(key=lambda r: -r["total"])')
    for line in kept:
        assert out_lines[src_lines.index(line)] == line

    # 最强证据：把产物 exec 起来，守卫与排序都真的生效。
    orders = [
        type("Order", (), {"paid": True, "order_id": "A1", "total": 10.0})(),
        type("Order", (), {"paid": True, "order_id": "A2", "total": 30.0})(),
        type("Order", (), {"paid": False, "order_id": "A3", "total": 99.0})(),
        type("Order", (), {"paid": True, "order_id": "A4", "total": 20.0})(),
    ]
    assert _run(artifact.body, {}, "paid_orders", orders) == [
        {"order_id": "A2", "total": 30.0},
        {"order_id": "A4", "total": 20.0},
        {"order_id": "A1", "total": 10.0},
    ]
    assert _run(artifact.body, {}, "paid_orders", []) == []
    assert _run(artifact.body, {}, "paid_orders", None) == []


# --------------------------------------------------------------------------
# libcst 无损手术：只有循环里那两处变，其余每个字节都不许动
# --------------------------------------------------------------------------


# 守卫 + 复合条件 + 循环后排序：三样都必须活下来。
COMPOUND_GUARD_SOURCE = '''"""订单。"""


def paid_orders(orders):
    """返回已付款订单的编号与金额，按金额从高到低。"""
    if not orders:
        return []
    rows = []
    for order in orders:
        if order.paid and not order.shipped:
            rows.append({"order_id": order.order_id, "total": order.total})
    rows.sort(key=lambda r: -r["total"])
    return rows
'''

# 同一份源码，函数身后还有别的代码：函数之外的字节也要逐字保留。
COMPOUND_GUARD_WITH_TAIL = COMPOUND_GUARD_SOURCE + '''

def tail_helper():
    """函数之外的代码必须逐字保留。"""
    return {"x": 1}
'''

# 循环里直接 append，没有任何 if：用来验证“新加一个过滤条件”。
NOT_FILTERED_SOURCE = '''"""用户列表：当前返回全部字段。"""


def active_users(users):
    """返回用户，保持原顺序。"""
    result = []
    for user in users:
        result.append({"id": user["id"], "name": user["name"], "active": user["active"]})
    return result
'''

# 带 if 的循环：用来验证“去掉过滤条件”。
FILTERED_SOURCE = '''def users(rows):
    """返回有效行。"""
    out = []
    for row in rows:
        if row["active"]:
            out.append({"id": row["id"]})
    return out
'''

# 三个字段的循环：用来验证 dict 只留选中的字段、并按决策要求的顺序。
THREE_FIELDS_SOURCE = '''"""订单。"""


def paid_orders(orders):
    """返回订单摘要。"""
    rows = []
    for order in orders:
        if order.paid:
            rows.append({"order_id": order.order_id, "total": order.total, "note": order.note})
    return rows
'''

# 不支持手术的累加形状：+=、循环里两处 append、append 的不是 dict 字面量。
UNSUPPORTED_ASSEMBLE_SOURCES = [
    pytest.param(
        "def f(rows):\n"
        "    out = []\n"
        "    for row in rows:\n"
        '        out += [{"id": row["id"]}]\n'
        "    return out\n",
        id="augmented-assign",
    ),
    pytest.param(
        "def f(rows):\n"
        "    out = []\n"
        "    for row in rows:\n"
        '        out.append({"id": row["id"]})\n'
        '        out.append({"id": row["id"], "name": row["name"]})\n'
        "    return out\n",
        id="two-appends",
    ),
    pytest.param(
        "def f(rows):\n"
        "    out = []\n"
        "    for row in rows:\n"
        '        out.append([row["id"], row["name"]])\n'
        "    return out\n",
        id="append-is-not-a-dict",
    ),
]


def _split(text: str) -> list[str]:
    """按行切并保留行尾，便于逐字节比较。"""
    return text.splitlines(keepends=True)


def _function_node(source: str, name: str) -> ast.FunctionDef:
    """源码里名叫 name 的函数节点。"""
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"源码里没有函数 {name}")


def _for_loop(source: str, name: str) -> ast.For:
    """函数体里的第一个 for 循环。"""
    for statement in _function_node(source, name).body:
        if isinstance(statement, ast.For):
            return statement
    raise AssertionError("函数体里没有 for 循环")


def _loop_span(source: str, name: str) -> tuple[int, int]:
    """函数体里 for 循环的起止行号（1 基，含），用来比较循环以外的字节。"""
    loop = _for_loop(source, name)
    return loop.lineno, loop.end_lineno or loop.lineno


def _assert_outside_loop_identical(source: str, out: str, name: str) -> None:
    """循环之前与之后的每一个字节都和原文相同（守卫、docstring、sort、函数之外都算）。"""
    start, end = _loop_span(source, name)
    source_lines, out_lines = _split(source), _split(out)
    assert out_lines[: start - 1] == source_lines[: start - 1], "循环之前的字节被改动"
    assert out_lines[end:] == source_lines[end:], "循环之后的字节被改动"


def _order(order_id: str, total: float, *, paid: bool, shipped: bool) -> object:
    """测试用订单对象：属性取值风格，shipped 也带上，便于验证复合条件被换掉。"""
    return type(
        "Order", (), {"order_id": order_id, "total": total, "paid": paid, "shipped": shipped}
    )()


def test_assemble_keeps_guard_and_sort_byte_for_byte() -> None:
    """守卫子句与循环后的 sort 逐字节保留：手术只动循环里那两处。"""
    candidates, decision = _assemble_decision(
        COMPOUND_GUARD_SOURCE, "paid_orders", ["f2", "f3"], "c0"
    )
    out = assemble(COMPOUND_GUARD_SOURCE, candidates, decision)
    ast.parse(out)

    _assert_outside_loop_identical(COMPOUND_GUARD_SOURCE, out, "paid_orders")
    assert "    if not orders:\n        return []\n" in out
    assert '    rows.sort(key=lambda r: -r["total"])\n' in out
    # 复合条件换成决策选中的单个条件：shipped 整个不再被读取。
    assert "if order.paid:" in out
    assert "shipped" not in out
    assert len(_split(out)) == len(_split(COMPOUND_GUARD_SOURCE))  # 行数不变


def test_assembled_guard_and_sort_really_work_at_runtime() -> None:
    """最强证据：exec 组装结果，守卫拦得住空输入，循环后的排序真的在排。"""
    candidates, decision = _assemble_decision(
        COMPOUND_GUARD_SOURCE, "paid_orders", ["f2", "f3"], "c0"
    )
    out = assemble(COMPOUND_GUARD_SOURCE, candidates, decision)
    orders = [
        _order("A1", 10.0, paid=True, shipped=True),  # 已付款就留下，shipped 不再参与判断
        _order("A2", 30.0, paid=True, shipped=False),
        _order("A3", 20.0, paid=False, shipped=False),  # 没付款，过滤掉
    ]
    # 30 在 10 前面：只有循环后的 sort 真的跑了才会是这个顺序。
    assert _run(out, {}, "paid_orders", orders) == [
        {"order_id": "A2", "total": 30.0},
        {"order_id": "A1", "total": 10.0},
    ]
    # 守卫仍然生效：None 只有 `if not orders: return []` 拦得住，守卫没了这里会抛 TypeError。
    assert _run(out, {}, "paid_orders", None) == []
    assert _run(out, {}, "paid_orders", []) == []


def test_assemble_replaces_compound_condition_with_the_chosen_one() -> None:
    """复合条件 `paid and not shipped` 换成单个 `paid`，循环后的排序照旧运行。"""
    candidates, decision = _assemble_decision(COMPOUND_GUARD_SOURCE, "paid_orders", ["f3"], "c0")
    out = assemble(COMPOUND_GUARD_SOURCE, candidates, decision)
    assert "if order.paid:" in out
    assert "not order.shipped" not in out

    orders = [
        _order("A1", 10.0, paid=True, shipped=True),  # 原来会被 and not shipped 挡掉，现在该留下
        _order("A2", 30.0, paid=True, shipped=False),
    ]
    assert _run(out, {}, "paid_orders", orders) == [{"total": 30.0}, {"total": 10.0}]


def test_assemble_inserts_filter_when_the_loop_has_no_if() -> None:
    """循环里本来没有判断：插入 `if <条件>:`，append 缩进一级，其余行原样。"""
    candidates, decision = _assemble_decision(NOT_FILTERED_SOURCE, "active_users", ["f0", "f1"], "c2")
    out = assemble(NOT_FILTERED_SOURCE, candidates, decision)
    ast.parse(out)

    assert (
        "    for user in users:\n"
        '        if user["active"]:\n'
        '            result.append({"id": user["id"], "name": user["name"]})\n'
    ) in out
    users = [
        {"id": 1, "name": "Ada", "active": True},
        {"id": 2, "name": "Bob", "active": False},
        {"id": 3, "name": "Cid", "active": True},
    ]
    assert _run(out, {}, "active_users", users) == [
        {"id": 1, "name": "Ada"},
        {"id": 3, "name": "Cid"},
    ]


def test_assemble_removes_filter_and_runs_unfiltered() -> None:
    """filter_field=None：循环里的判断整个去掉，append 回到循环体一层，结果不过滤。"""
    candidates, decision = _assemble_decision(FILTERED_SOURCE, "users", ["f1"], None)
    out = assemble(FILTERED_SOURCE, candidates, decision)
    ast.parse(out)

    loop = _for_loop(out, "users")
    assert not any(isinstance(node, ast.If) for node in ast.walk(loop)), "循环里不该再留下 if"
    assert '        out.append({"id": row["id"]})\n' in out  # 缩进回到循环体一层
    rows = [{"id": 1, "active": True}, {"id": 2, "active": False}]
    assert _run(out, {}, "users", rows) == [{"id": 1}, {"id": 2}]


def test_assemble_replaces_dict_with_chosen_fields_in_requested_order() -> None:
    """dict 只留决策选中的字段，顺序按决策来；未选中的字段不再出现。"""
    candidates, decision = _assemble_decision(THREE_FIELDS_SOURCE, "paid_orders", ["f2", "f1"], "c0")
    out = assemble(THREE_FIELDS_SOURCE, candidates, decision)
    ast.parse(out)

    assert 'rows.append({"total": order.total, "order_id": order.order_id})' in out
    assert "note" not in out, "未选中的字段名不该再出现"
    records = _run(out, {}, "paid_orders", [_order("A1", 10.0, paid=True, shipped=False)])
    assert records == [{"total": 10.0, "order_id": "A1"}]
    assert list(records[0]) == ["total", "order_id"], "运行时键顺序也要按决策来"


def test_assemble_keeps_everything_outside_the_function_identical() -> None:
    """函数之外（docstring、import、身后的函数）逐字节不变，行数也不变。"""
    source = COMPOUND_GUARD_WITH_TAIL
    candidates, decision = _assemble_decision(source, "paid_orders", ["f2", "f3"], "c0")
    out = assemble(source, candidates, decision)

    assert _split(out)[: candidates.start_line - 1] == _split(source)[: candidates.start_line - 1]
    assert out.endswith(
        '\n\ndef tail_helper():\n    """函数之外的代码必须逐字保留。"""\n    return {"x": 1}\n'
    )
    assert len(_split(out)) == len(_split(source))


@pytest.mark.parametrize("source", UNSUPPORTED_ASSEMBLE_SOURCES)
def test_assemble_unsupported_shapes_still_raise_decision_error(source: str) -> None:
    """+=、循环里两处 append、append 的不是 dict 字面量：一律 DecisionError，不猜。"""
    candidates, decision = _assemble_decision(source, "f", ["f0"], None)
    with pytest.raises(DecisionError, match="不支持"):
        assemble(source, candidates, decision)


# --------------------------------------------------------------------------
# 两个循环：只改产出返回值的那个，另一个逐字不动
# --------------------------------------------------------------------------


# 先去重、再产出行：只有第二个循环产出返回值。
TWO_LOOPS_SOURCE = '''def summarize(rows):
    seen = []
    for row in rows:
        seen.append({"name": row["name"]})
    out = []
    for row in rows:
        out.append({"id": row["id"], "name": row["name"]})
    return out
'''

# 两个循环都在向返回的那个累加列表 append：说不清哪个产出返回值。
TWO_LOOPS_TO_RETURNED = '''def f(rows):
    out = []
    for row in rows:
        out.append({"id": row["id"]})
    for row in rows:
        out.append({"id": row["id"], "name": row["name"]})
    return out
'''

# 没有循环向返回的那个累加列表 append。
NO_LOOP_TO_RETURNED = '''def f(rows):
    out = []
    ids = []
    for row in rows:
        ids.append({"id": row["id"]})
    return out
'''


def test_assemble_edits_the_loop_that_produces_the_returned_list() -> None:
    """两个循环时只改产出返回值的那个：seen 循环逐字不动，out 循环才被改。"""
    candidates, decision = _assemble_decision(TWO_LOOPS_SOURCE, "summarize", ["f1"], None)
    out = assemble(TWO_LOOPS_SOURCE, candidates, decision)
    ast.parse(out)

    # seen 那个循环整段逐字保留，一行都没被碰过。
    assert (
        "    seen = []\n"
        "    for row in rows:\n"
        '        seen.append({"name": row["name"]})\n'
    ) in out
    # 被改的是 out：只留决策选的 id，name 只剩 seen 那一处。
    assert '        out.append({"id": row["id"]})\n' in out
    assert out.count('row["name"]') == 1


def test_two_loop_result_returns_only_the_chosen_fields_at_runtime() -> None:
    """最强证据：exec 组装结果，返回值里只有决策选中的字段，别的都没混进来。"""
    candidates, decision = _assemble_decision(TWO_LOOPS_SOURCE, "summarize", ["f1"], None)
    out = assemble(TWO_LOOPS_SOURCE, candidates, decision)
    rows = [{"id": 1, "name": "Ada"}, {"id": 2, "name": "Bob"}]
    assert _run(out, {}, "summarize", rows) == [{"id": 1}, {"id": 2}]
    # 如果手术落在 seen 那个循环上，返回值仍会带上 name（决策等于没生效）。
    assert list(_run(out, {}, "summarize", rows)[0]) == ["id"]


def test_assemble_single_returned_loop_still_works() -> None:
    """常规形状：函数里只有产出返回值的那个循环，照常手术。"""
    candidates, decision = _assemble_decision(FILTERED_SOURCE, "users", ["f1"], "c0")
    out = assemble(FILTERED_SOURCE, candidates, decision)
    ast.parse(out)
    assert 'if row["active"]:' in out
    assert 'out.append({"id": row["id"]})' in out
    rows = [{"id": 1, "active": True}, {"id": 2, "active": False}]
    assert _run(out, {}, "users", rows) == [{"id": 1}]


def test_assemble_rejects_two_loops_both_feeding_the_returned_list() -> None:
    """两个循环都在向返回的累加列表 append：说不清哪个产出返回值，拒绝。"""
    candidates, decision = _assemble_decision(TWO_LOOPS_TO_RETURNED, "f", ["f0"], None)
    with pytest.raises(DecisionError, match="2 个循环都在向 out append"):
        assemble(TWO_LOOPS_TO_RETURNED, candidates, decision)


def test_assemble_rejects_when_no_loop_feeds_the_returned_list() -> None:
    """没有循环向返回的累加列表 append：消息要说清是“没有循环向 out append”。"""
    candidates, decision = _assemble_decision(NO_LOOP_TO_RETURNED, "f", ["f0"], None)
    with pytest.raises(DecisionError, match="没有 for 循环在向 out append"):
        assemble(NO_LOOP_TO_RETURNED, candidates, decision)


def test_run_decision_notes_the_loops_it_left_alone() -> None:
    """完整流程里如实说明：还有别的循环在向别的累加列表 append，本次没有动它。"""
    engine = ScriptedEngine(['{"function": "fn0", "filter_field": null, "return_fields": ["f1"]}'])
    artifact, _, _ = run_decision(
        engine, "只返回编号。", TWO_LOOPS_SOURCE, TARGET, "summarize"
    )
    assert any("别的累加列表" in note for note in artifact.notes)
    candidates, decision = _assemble_decision(TWO_LOOPS_SOURCE, "summarize", ["f1"], None)
    assert artifact.body == normalize_body(assemble(TWO_LOOPS_SOURCE, candidates, decision))

    # 只有一个循环的常规形状不加这条提示，避免噪音。
    single = ScriptedEngine(
        ['{"function": "fn0", "filter_field": "c0", "return_fields": ["f2", "f3"]}']
    )
    other, _, _ = run_decision(single, FILTER_INSTRUCTION, USERS_MODULE, TARGET, "active_users")
    assert not any("别的累加列表" in note for note in other.notes)


# --------------------------------------------------------------------------
# 排序槽位：决策里给了就替换 / 插入，没给就一个字节都不动
# --------------------------------------------------------------------------


# 循环之后本来就有排序：决策不带排序时必须逐字留住，带了就整条替换。
SORT_EXISTING_SOURCE = '''"""订单。"""


def paid_orders(orders):
    """返回已付款订单的编号与金额，按金额从高到低。"""
    if not orders:
        return []
    rows = []
    for order in orders:
        if order.paid:
            rows.append({"order_id": order.order_id, "total": order.total})
    # 按金额排：这条注释也要活下来。
    rows.sort(key=lambda r: -r["total"])
    return rows
'''

# 循环之前没有排序：决策带了排序就插在循环之后、return 之前。
SORT_INSERT_SOURCE = '''"""用户列表。"""


def active_users(users):
    """返回有效用户。"""
    result = []
    for user in users:
        if user["active"]:
            result.append({"id": user["id"], "name": user["name"]})
    return result
'''

# 排序写成 `out = sorted(out, ...)`：整条替换，不留下第二处排序。
SORTED_ASSIGN_SOURCE = '''"""价格表。"""


def priced(rows):
    """返回名称与价格。"""
    out = []
    for row in rows:
        out.append({'name': row['name'], 'price': row['price']})
    out = sorted(out, key=lambda r: r['price'])
    return out
'''

# 取值风格是 .get()：排序键跟着用 .get()。
SORT_GET_SOURCE = '''"""商品。"""


def visible_products(products):
    """返回上架商品。"""
    out = []
    for product in products:
        if product.get("visible"):
            out.append({"sku": product["sku"], "price": product.get("price")})
    return out
'''


def _with_sort(source: str, function: str, decision_json: str) -> tuple[Candidates, Decision]:
    """解析一个显式启用排序槽位的决策，返回候选表与决策。"""
    candidates = extract(source, function)
    return candidates, parse_decision(decision_json, candidates, sort_enabled=True)


def _run_orders(source: str, orders: list[object]) -> object:
    """把重写后的 paid_orders exec 起来真跑一次。"""
    return _run(source, {}, "paid_orders", orders)


def test_parse_rejects_sort_slot_when_host_did_not_enable_it() -> None:
    """无排序请求走默认协议时，宿主不能接受模型误回的 s / d。"""
    candidates = extract(SORT_INSERT_SOURCE, "active_users")
    with pytest.raises(DecisionError, match="排序槽位未启用"):
        parse_decision('{"f": "c0", "r": ["f1", "f2"], "s": "f2", "d": "desc"}', candidates)


def test_parse_accepts_sort_slot_only_when_host_enabled() -> None:
    """s / d 是显式启用后的决策键：升序 asc、降序 desc。"""
    candidates = extract(SORT_INSERT_SOURCE, "active_users")
    desc = parse_decision(
        '{"f": "c0", "r": ["f1", "f2"], "s": "f2", "d": "desc"}',
        candidates,
        sort_enabled=True,
    )
    assert desc.sort_field == "f2" and desc.sort_desc is True
    asc = parse_decision(
        '{"f": "c0", "r": ["f1", "f2"], "s": "f2", "d": "asc"}',
        candidates,
        sort_enabled=True,
    )
    assert asc.sort_field == "f2" and asc.sort_desc is False
    # 没给排序时两个新字段都停在默认值上，老决策与老夹具照旧
    plain = parse_decision('{"f": "c0", "r": ["f1", "f2"]}', candidates)
    assert plain.sort_field is None and plain.sort_desc is False


def test_parse_absent_or_quoted_null_sort_means_no_sort() -> None:
    """s 缺席、写成 null、或写成字符串 "null"，都按“不动原文排序”处理。"""
    candidates = extract(SORT_INSERT_SOURCE, "active_users")
    for text in (
        '{"f": "c0", "r": ["f1", "f2"]}',
        '{"f": "c0", "r": ["f1", "f2"], "s": null, "d": null}',
        '{"f": "c0", "r": ["f1", "f2"], "s": "null", "d": "none"}',
    ):
        decision = parse_decision(text, candidates, sort_enabled=True)
        assert decision.sort_field is None and decision.sort_desc is False


def test_parse_rejects_half_a_sort() -> None:
    """只给一半：有 s 没 d、有 d 没 s，都拒绝（不知道方向就不替模型猜）。"""
    candidates = extract(SORT_INSERT_SOURCE, "active_users")
    with pytest.raises(DecisionError, match="没给方向"):
        parse_decision(
            '{"f": "c0", "r": ["f1", "f2"], "s": "f2"}',
            candidates,
            sort_enabled=True,
        )
    with pytest.raises(DecisionError, match="没给排序字段"):
        parse_decision(
            '{"f": "c0", "r": ["f1", "f2"], "d": "desc"}',
            candidates,
            sort_enabled=True,
        )


def test_parse_rejects_unknown_sort_field_ids() -> None:
    """排序字段只能用字段候选：不存在的 id、以及条件候选 id 都拒绝。"""
    candidates = extract(SORT_INSERT_SOURCE, "active_users")
    for bad in ("f9", "c0", "email"):
        with pytest.raises(DecisionError, match="未知的排序字段 id"):
            parse_decision(
                f'{{"f": "c0", "r": ["f1"], "s": "{bad}", "d": "asc"}}',
                candidates,
                sort_enabled=True,
            )


def test_parse_rejects_sort_field_not_returned() -> None:
    """排序字段必须同时被返回：累加列表里只有 r 选中的键，否则运行时读不到。"""
    candidates = extract(SORT_INSERT_SOURCE, "active_users")
    with pytest.raises(DecisionError, match="不在返回字段"):
        parse_decision(
            '{"f": "c0", "r": ["f1"], "s": "f2", "d": "asc"}',
            candidates,
            sort_enabled=True,
        )


def test_parse_rejects_unknown_direction() -> None:
    candidates = extract(SORT_INSERT_SOURCE, "active_users")
    with pytest.raises(DecisionError, match="未知的排序方向"):
        parse_decision(
            '{"f": "c0", "r": ["f1", "f2"], "s": "f2", "d": "横向"}',
            candidates,
            sort_enabled=True,
        )


def test_parse_legacy_long_keys_still_parse_and_cannot_sort() -> None:
    """旧的长键格式照旧解析（没有排序槽位）；想排序只能换精简格式。"""
    candidates = extract(SORT_INSERT_SOURCE, "active_users")
    legacy = parse_decision(
        '{"function": "fn0", "filter_field": "c0", "return_fields": ["f1", "f2"]}', candidates
    )
    assert legacy.sort_field is None
    with pytest.raises(DecisionError, match="混用"):
        parse_decision(
            '{"function": "fn0", "filter_field": "c0", "return_fields": ["f1", "f2"],'
            ' "s": "f2", "d": "desc"}',
            candidates,
        )


def test_describe_decision_mentions_the_sort() -> None:
    candidates, decision = _with_sort(
        SORT_INSERT_SOURCE, "active_users", '{"r": ["f1", "f2"], "s": "f2", "d": "desc"}'
    )
    text = describe_decision(candidates, decision)
    assert "\n" not in text and "排序=f2=name 降序" in text
    assert "不动原文" in describe_decision(
        candidates, parse_decision('{"r": ["f1"]}', candidates)
    )


def test_assemble_without_sort_keeps_existing_sort_byte_for_byte() -> None:
    """决策没带排序：原文那条 sort（连它上方的注释）逐字节留在同一行上。"""
    candidates = extract(SORT_EXISTING_SOURCE, "paid_orders")
    decision = parse_decision('{"f": "c0", "r": ["f1", "f2"]}', candidates)
    out = assemble(SORT_EXISTING_SOURCE, candidates, decision)
    ast.parse(out)
    src_lines, out_lines = _split(SORT_EXISTING_SOURCE), _split(out)
    kept = '    rows.sort(key=lambda r: -r["total"])\n'
    index = src_lines.index(kept)
    assert out_lines[index] == kept
    assert out_lines[index - 1] == "    # 按金额排：这条注释也要活下来。\n"
    # 排序一行不多一行不少，函数里没有第二处排序
    assert out.count(".sort(") == 1
    assert len(out_lines) == len(src_lines)


def test_assemble_without_sort_keeps_sorted_assign_byte_for_byte() -> None:
    """`out = sorted(out, ...)` 同样逐字节不动：没要求排序就不碰。"""
    candidates = extract(SORTED_ASSIGN_SOURCE, "priced")
    decision = parse_decision('{"r": ["f0", "f1"]}', candidates)
    out = assemble(SORTED_ASSIGN_SOURCE, candidates, decision)
    assert "    out = sorted(out, key=lambda r: r['price'])\n" in out
    assert out == SORTED_ASSIGN_SOURCE  # 原文本来就是这个决策要的形状


def test_assemble_with_sort_replaces_the_existing_sort() -> None:
    """决策带了排序：旧那条 sort 整条消失，换成决策指定的字段与方向。"""
    candidates, decision = _with_sort(
        SORT_EXISTING_SOURCE, "paid_orders", '{"f": "c0", "r": ["f1", "f2"], "s": "f2", "d": "asc"}'
    )
    out = assemble(SORT_EXISTING_SOURCE, candidates, decision)
    ast.parse(out)
    assert "rows.sort(key=lambda r: -r[\"total\"])" not in out  # 旧排序一行不剩
    assert "    rows.sort(key=lambda r: r[\"total\"])\n" in out  # 升序：不带 reverse
    assert out.count(".sort(") == 1  # 只剩一处排序
    # 守卫子句与注释照旧
    assert "    if not orders:\n        return []\n" in out
    assert "    # 按金额排：这条注释也要活下来。\n" in out


def test_assemble_with_sort_replaces_the_sorted_assignment() -> None:
    """`out = sorted(out, ...)` 也是“已有的排序”：整条替换，不留第二处。"""
    candidates, decision = _with_sort(
        SORTED_ASSIGN_SOURCE, "priced", '{"r": ["f0", "f1"], "s": "f1", "d": "desc"}'
    )
    out = assemble(SORTED_ASSIGN_SOURCE, candidates, decision)
    ast.parse(out)
    assert "sorted(" not in out
    assert "    out.sort(key=lambda r: r['price'], reverse=True)\n" in out
    assert out.count(".sort(") == 1


def test_assemble_with_sort_inserts_after_the_loop_before_the_return() -> None:
    """本来没有排序：插在产出返回值的循环之后、`return 累加列表` 之前。"""
    candidates, decision = _with_sort(
        SORT_INSERT_SOURCE, "active_users", '{"f": "c0", "r": ["f1", "f2"], "s": "f2", "d": "desc"}'
    )
    out = assemble(SORT_INSERT_SOURCE, candidates, decision)
    ast.parse(out)
    assert (
        '            result.append({"id": user["id"], "name": user["name"]})\n'
        '    result.sort(key=lambda r: r["name"], reverse=True)\n'
        "    return result\n"
    ) in out


def test_assemble_sort_keeps_observed_access_and_quote_style() -> None:
    """排序键的取值风格与引号跟着源码走：.get() 与单引号都要保住。"""
    candidates, decision = _with_sort(
        SORT_GET_SOURCE, "visible_products", '{"f": "c0", "r": ["f1", "f2"], "s": "f2", "d": "desc"}'
    )
    out = assemble(SORT_GET_SOURCE, candidates, decision)
    assert '    out.sort(key=lambda r: r.get("price"), reverse=True)\n' in out

    single = extract(SORTED_ASSIGN_SOURCE, "priced")
    decision = parse_decision(
        '{"r": ["f0", "f1"], "s": "f1", "d": "asc"}',
        single,
        sort_enabled=True,
    )
    assert "    out.sort(key=lambda r: r['price'])\n" in assemble(
        SORTED_ASSIGN_SOURCE, single, decision
    )


def test_assemble_sort_works_on_attribute_style_source() -> None:
    """属性取值风格的源码：累加的是 dict，排序键照样按下标读同一个键名。"""
    candidates, decision = _with_sort(
        ORDERS_MODULE, "paid_orders", '{"f": "c0", "r": ["f2", "f3"], "s": "f3", "d": "desc"}'
    )
    out = assemble(ORDERS_MODULE, candidates, decision)
    ast.parse(out)
    assert '    rows.sort(key=lambda r: r["total"], reverse=True)\n' in out


def test_assembled_sort_really_orders_at_runtime() -> None:
    """最强证据：exec 组装结果真调用，过滤对了、字段对了、顺序也真的对。"""
    candidates, decision = _with_sort(
        SORT_EXISTING_SOURCE, "paid_orders", '{"f": "c0", "r": ["f1", "f2"], "s": "f2", "d": "asc"}'
    )
    out = assemble(SORT_EXISTING_SOURCE, candidates, decision)
    orders = [
        _order("A3", 30.0, paid=True, shipped=False),
        _order("A1", 10.0, paid=True, shipped=False),
        _order("A2", 20.0, paid=False, shipped=False),  # 没付款，过滤掉
        _order("A4", 20.0, paid=True, shipped=False),
    ]
    # 原文那条是降序（-total），这里换成升序：10、20、30 才说明替换真的生效。
    assert _run_orders(out, orders) == [
        {"order_id": "A1", "total": 10.0},
        {"order_id": "A4", "total": 20.0},
        {"order_id": "A3", "total": 30.0},
    ]
    assert _run_orders(out, None) == []  # 守卫子句仍在
    assert _run_orders(out, []) == []


def test_assembled_inserted_sort_really_orders_at_runtime() -> None:
    """插入的排序也要在运行时真的生效（降序、且只留下过滤后的项）。"""
    candidates, decision = _with_sort(
        SORT_INSERT_SOURCE, "active_users", '{"f": "c0", "r": ["f1", "f2"], "s": "f2", "d": "desc"}'
    )
    out = assemble(SORT_INSERT_SOURCE, candidates, decision)
    users = [
        {"id": 1, "name": "ada", "active": True},
        {"id": 2, "name": "bob", "active": False},
        {"id": 3, "name": "cid", "active": True},
    ]
    assert _run(out, {}, "active_users", users) == [
        {"id": 3, "name": "cid"},
        {"id": 1, "name": "ada"},
    ]


def test_assemble_sort_keeps_everything_outside_the_function_identical() -> None:
    """带排序的手术同样只动该动的那几行：函数之外与守卫逐字节不变。"""
    source = SORT_EXISTING_SOURCE + '''

def tail_helper():
    """函数之外的代码必须逐字保留。"""
    return {"x": 1}
'''
    candidates, decision = _with_sort(
        source, "paid_orders", '{"f": "c0", "r": ["f1", "f2"], "s": "f2", "d": "desc"}'
    )
    out = assemble(source, candidates, decision)
    assert _split(out)[: candidates.start_line - 1] == _split(source)[: candidates.start_line - 1]
    assert out.endswith(
        '\n\ndef tail_helper():\n    """函数之外的代码必须逐字保留。"""\n    return {"x": 1}\n'
    )
    assert len(_split(out)) == len(_split(source))  # 替换不改行数
    assert "    if not orders:\n        return []\n" in out


# 说不清怎么替换的排序形状：多处、嵌在分支里、排在循环之前。
UNSAFE_SORT_SOURCES = [
    pytest.param(
        "def f(rows):\n"
        "    out = []\n"
        "    for row in rows:\n"
        '        out.append({"id": row["id"], "total": row["total"]})\n'
        '    out.sort(key=lambda r: r["id"])\n'
        '    out.sort(key=lambda r: r["total"])\n'
        "    return out\n",
        "2 处对 out 的排序",
        id="two-sorts",
    ),
    pytest.param(
        "def f(rows):\n"
        "    out = []\n"
        "    for row in rows:\n"
        '        out.append({"id": row["id"], "total": row["total"]})\n'
        "    if len(out) > 1:\n"
        '        out.sort(key=lambda r: r["id"])\n'
        "    return out\n",
        "无法安全替换",
        id="nested-sort",
    ),
    pytest.param(
        "def f(rows):\n"
        "    out = []\n"
        "    out.sort()\n"
        "    for row in rows:\n"
        '        out.append({"id": row["id"], "total": row["total"]})\n'
        "    return out\n",
        "在产出返回值的循环之前",
        id="sort-before-loop",
    ),
    pytest.param(
        "def f(rows):\n"
        "    out = []\n"
        "    for row in rows:\n"
        '        out.append({"id": row["id"], "total": row["total"]})\n'
        "    count = out.sort()\n"
        "    return out\n",
        "无法安全替换",
        id="sort-as-value",
    ),
]


@pytest.mark.parametrize(("source", "message"), UNSAFE_SORT_SOURCES)
def test_assemble_rejects_unsafe_sorts_instead_of_guessing(source: str, message: str) -> None:
    """替换不了就拒绝：多处排序、嵌在分支里、排在循环之前都不猜。"""
    candidates, decision = _with_sort(source, "f", '{"r": ["f0", "f1"], "s": "f1", "d": "desc"}')
    with pytest.raises(DecisionError, match=message):
        assemble(source, candidates, decision)


def test_assemble_rejects_sort_when_the_return_is_not_after_the_loop() -> None:
    """`return 累加列表` 不在循环之后：说不出排序插在哪里，拒绝。"""
    source = (
        "def f(rows):\n"
        "    out = []\n"
        "    return out\n"
        "    for row in rows:\n"
        '        out.append({"id": row["id"]})\n'
    )
    candidates, decision = _with_sort(source, "f", '{"r": ["f0"], "s": "f0", "d": "asc"}')
    with pytest.raises(DecisionError, match="不在产出返回值的循环之后"):
        assemble(source, candidates, decision)


def test_assemble_rejects_sort_by_a_field_that_is_not_returned() -> None:
    """直接构造的 Decision 也要在 assemble 里被复核：排序字段必须在 r 里。"""
    candidates = extract(SORT_INSERT_SOURCE, "active_users")
    decision = Decision(
        function_id="fn0", filter_field=None, return_fields=("f1",), sort_field="f2"
    )
    with pytest.raises(DecisionError, match="不在返回字段"):
        assemble(SORT_INSERT_SOURCE, candidates, decision)


def test_run_decision_rejects_sort_reply_when_slot_is_disabled() -> None:
    """端到端宿主闸门：未启用排序时即使模型回 s/d 也不能执行。"""
    engine = ScriptedEngine(
        ['{"f": "c0", "r": ["f1", "f2"], "s": "f2", "d": "desc"}']
    )
    with pytest.raises(DecisionError, match="排序槽位未启用"):
        run_decision(
            engine,
            "只保留已付款订单，返回编号和金额，其他不变。",
            SORT_EXISTING_SOURCE,
            TARGET,
            "paid_orders",
        )
    # 禁用侧的禁令在用户消息里（系统提示保留那行实测必须存在的形状示例）。
    assert "严禁输出 s 或 d" in engine.calls[0][1]["content"]
    assert "排序槽位：未启用" in engine.calls[0][1]["content"]
    assert "正确形状示例" in engine.calls[0][0]["content"]


def test_run_decision_end_to_end_with_sort() -> None:
    """完整流程：显式启用排序槽位后，宿主组装出的正文运行时真的按该顺序。"""
    engine = ScriptedEngine(
        ['{"f": "c0", "r": ["f1", "f2"], "s": "f2", "d": "desc"}']
    )
    artifact, decision, candidates = run_decision(
        engine, "只保留已付款订单，返回编号和金额，按金额从高到低。",
        SORT_EXISTING_SOURCE, TARGET, "paid_orders", sort_enabled=True,
    )
    assert decision.sort_field == "f2" and decision.sort_desc is True
    assert artifact.body == normalize_body(assemble(SORT_EXISTING_SOURCE, candidates, decision))
    assert artifact.raw_response == artifact.body
    assert '    rows.sort(key=lambda r: r["total"], reverse=True)\n' in artifact.body
    assert "-r[\"total\"]" not in artifact.body  # 旧排序已替换
    assert any("排序=f2=total 降序" in note for note in artifact.notes)  # 大模型看得见这次排序
    orders = [_order("A1", 10.0, paid=True, shipped=False), _order("A2", 30.0, paid=True, shipped=False)]
    assert _run_orders(artifact.body, orders) == [
        {"order_id": "A2", "total": 30.0},
        {"order_id": "A1", "total": 10.0},
    ]


def test_run_decision_without_sort_still_leaves_the_source_sort_alone() -> None:
    """端到端对照：决策不带排序时，原文那条 sort 与旧行为一字不差。"""
    engine = ScriptedEngine(['{"f": "c0", "r": ["f1", "f2"]}'])
    artifact, _, _ = run_decision(
        engine, FILTER_INSTRUCTION, SORT_EXISTING_SOURCE, TARGET, "paid_orders"
    )
    assert '    rows.sort(key=lambda r: -r["total"])\n' in artifact.body
    orders = [_order("A1", 10.0, paid=True, shipped=False), _order("A2", 30.0, paid=True, shipped=False)]
    assert _run_orders(artifact.body, orders) == [
        {"order_id": "A2", "total": 30.0},
        {"order_id": "A1", "total": 10.0},
    ]


def test_parse_and_assemble_sort_inside_a_class_method() -> None:
    """类方法里插入排序：缩进跟着方法体走，函数之外一字不动。"""
    source = (
        "class Repo:\n"
        '    """仓库。"""\n'
        "\n"
        "    def items(self, rows):\n"
        '        """返回条目。"""\n'
        "        result = []\n"
        "        for row in rows:\n"
        '            result.append({"code": row["code"], "size": row["size"]})\n'
        "        return result\n"
    )
    candidates, decision = _with_sort(source, "items", '{"r": ["f0", "f1"], "s": "f1", "d": "desc"}')
    out = assemble(source, candidates, decision)
    assert "        result.sort(key=lambda r: r[\"size\"], reverse=True)\n" in out
    assert out.startswith('class Repo:\n    """仓库。"""\n\n    def items(self, rows):')
    ns: dict[str, object] = {}
    exec(compile(out, "<assemble>", "exec"), ns)  # noqa: S102 - 测试专用
    repo = ns["Repo"]  # type: ignore[operator]
    assert repo().items([{"code": "a", "size": 3}, {"code": "b", "size": 9}]) == [  # type: ignore[attr-defined]
        {"code": "b", "size": 9},
        {"code": "a", "size": 3},
    ]
