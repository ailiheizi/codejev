"""assemble() 的独立验收测试：无损手术，只改循环里的两处。

本文件按规格独立编写，不读实现细节：候选 id 一律从候选表按名字查出来，
断言以 ast 结构加真实运行结果为准——把组装后的源码 exec 起来，直接调用函数。
规格要求：过滤条件与被 append 的 dict 之外，每一个字节都必须保留，
包括守卫子句、循环后的 sort、注释和空行。
"""

from __future__ import annotations

import ast
import io
import json
import tokenize
from typing import Any

import pytest

from azfls.decide import (
    Candidate,
    Candidates,
    Decision,
    DecisionError,
    assemble,
    extract,
    parse_decision,
)

# 规格里的头号样本：守卫子句 + 复合条件 + 循环后排序。
HEADLINE_SOURCE = '''"""订单。"""


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

# 同一份样本，函数身后还有别的代码：尾部必须逐字保留。
HEADLINE_WITH_TAIL = HEADLINE_SOURCE + '''

def tail_helper():
    """函数之外的代码必须逐字保留。"""
    return {"x": 1}
'''

# 循环里直接 append，没有任何 if：用来验证“新加一个过滤条件”。
NO_FILTER_SOURCE = '''"""用户列表：当前返回全部字段。"""


def active_users(users):
    """返回用户，保持原顺序。"""
    result = []
    for user in users:
        result.append({"id": user["id"], "name": user["name"], "active": user["active"]})
    return result
'''

# 带 if 的循环：用来验证“去掉过滤条件”。
GUARDED_SOURCE = '''def users(rows):
    """返回有效行。"""
    out = []
    for row in rows:
        if row["active"]:
            out.append({"id": row["id"]})
    return out
'''

# 三个字段的循环：用来验证 dict 只留选中的字段、并按决策要求的顺序。
ORDER_SOURCE = '''def paid_orders(orders):
    """返回订单摘要。"""
    rows = []
    for order in orders:
        if order.paid:
            rows.append({"order_id": order.order_id, "total": order.total, "note": order.note})
    return rows
'''

# 属性风格：order.total。
ATTR_SOURCE = '''def paid_orders(orders):
    """返回金额。"""
    rows = []
    for order in orders:
        if order.paid:
            rows.append({"total": order.total, "note": order.note})
    return rows
'''

# 下标风格：user["name"]。
ITEM_SOURCE = '''def active_users(users):
    """返回名字。"""
    result = []
    for user in users:
        if user["active"]:
            result.append({"name": user["name"], "id": user["id"]})
    return result
'''

# .get(…) 风格：product.get("price")。
GET_SOURCE = '''def visible_products(products):
    """返回商品。"""
    out = []
    for product in products:
        if product.get("visible"):
            out.append({"sku": product["sku"], "price": product.get("price")})
    return out
'''

# 决策与原文行为一致：组装后必须还是同一份行为。
NOOP_SOURCE = '''def users(rows):
    """返回有效用户。"""
    out = []
    for row in rows:
        if row["active"]:
            out.append({"id": row["id"], "name": row["name"]})
    return out
'''

# 不支持的累加形状：+= 、循环里两处 append、append 的不是 dict 字面量。
UNSUPPORTED_SOURCES = [
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

# 候选来自这份源码；它会与下面两份“已经不是同一处函数”的源码配错。
STALE_SOURCE = '''def f(rows):
    """带守卫的函数。"""
    if not rows:
        return []
    out = []
    for row in rows:
        if row["ok"]:
            out.append({"id": row["id"]})
    return out
'''

# 同一函数名，但行范围与候选不一致。
STALE_SOURCE_SHIFTED = '''def f(rows):
    """同名但行数不同。"""
    out = []
    for row in rows:
        if row["ok"]:
            out.append({"id": row["id"], "name": row["name"]})
    return out
'''

# 连函数名都对不上。
STALE_SOURCE_RENAMED = '''def g(rows):
    """另一个函数。"""
    out = []
    for row in rows:
        if row["ok"]:
            out.append({"id": row["id"]})
    return out
'''

# 注释与空行到处都是的样本：除了两处手术，一个字节都不许动。
COMMENTED_SOURCE = '''"""带注释的模块：注释与空行必须逐字保留。"""

# 顶部注释：函数之外，逐字保留。


def paid_orders(orders):
    """返回已付款订单。"""
    # 初始化累加列表

    rows = []

    for order in orders:
        # 仅看是否已付款与是否已发货
        if order.paid and not order.shipped:
            rows.append({"order_id": order.order_id, "total": order.total})  # 行尾注释：必须保留
    # 循环之后再排序
    rows.sort(key=lambda r: -r["total"])

    return rows


# 文件末尾注释：逐字保留。
'''


class _Order:
    """测试用订单对象：属性取值风格。"""

    def __init__(self, order_id: str, total: float, paid: bool, shipped: bool) -> None:
        self.order_id = order_id
        self.total = total
        self.paid = paid
        self.shipped = shipped


# --------------------------------------------------------------------------
# 工具：候选查找、决策构造、运行、ast 观察
# --------------------------------------------------------------------------


def _id_of(candidates: tuple[Candidate, ...], name: str) -> str:
    """按名字从候选表里查 id；查不到说明宿主没提供这个候选，直接报错。"""
    for item in candidates:
        if item.name == name:
            return item.id
    available = ", ".join(f"{item.id}={item.name}" for item in candidates) or "（空）"
    raise AssertionError(f"候选表里没有 {name!r}；现有候选：{available}")


def _decide(
    source: str, function: str, filter_name: str | None, field_names: list[str]
) -> tuple[Candidates, Decision]:
    """按现有测试的写法构造决策：走 parse_decision，让 id 真正被校验。"""
    candidates = extract(source, function)
    filter_id = None if filter_name is None else _id_of(candidates.conditions, filter_name)
    field_ids = [_id_of(candidates.fields, name) for name in field_names]
    decision = parse_decision(
        json.dumps(
            {
                "function": candidates.function_id,
                "filter_field": filter_id,
                "return_fields": field_ids,
            }
        ),
        candidates,
    )
    return candidates, decision


def _assemble(
    source: str, function: str, filter_name: str | None, field_names: list[str]
) -> tuple[str, Candidates, Decision]:
    """提取候选 → 校验决策 → 组装，返回 (组装结果, 候选, 决策)。"""
    candidates, decision = _decide(source, function, filter_name, field_names)
    return assemble(source, candidates, decision), candidates, decision


def _run(source: str, function: str, argument: Any) -> Any:
    """把源码 exec 起来，真的调用那个函数，拿真实运行结果。"""
    namespace: dict[str, Any] = {}
    exec(compile(source, "<assemble>", "exec"), namespace)  # noqa: S102 - 测试专用
    return namespace[function](argument)


def _function(source: str, name: str) -> ast.FunctionDef:
    """组装结果里名叫 name 的函数节点。"""
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"组装结果里没有函数 {name}")


def _for_loop(source: str, name: str) -> ast.For:
    """函数体里的第一个 for 循环。"""
    for stmt in _function(source, name).body:
        if isinstance(stmt, ast.For):
            return stmt
    raise AssertionError("函数体里没有 for 循环")


def _appends(node: ast.AST) -> list[ast.Call]:
    """节点下的全部 `x.append(...)` 调用，按源码顺序。"""
    calls = [
        child
        for child in ast.walk(node)
        if isinstance(child, ast.Call)
        and isinstance(child.func, ast.Attribute)
        and child.func.attr == "append"
    ]
    calls.sort(key=lambda call: (call.lineno, call.col_offset))
    return calls


def _direct_appends(loop: ast.For) -> list[ast.Expr]:
    """直接写在循环体里的 `x.append(...)` 语句。"""
    return [
        stmt
        for stmt in loop.body
        if isinstance(stmt, ast.Expr)
        and isinstance(stmt.value, ast.Call)
        and isinstance(stmt.value.func, ast.Attribute)
        and stmt.value.func.attr == "append"
    ]


def _guard_if(loop: ast.For) -> ast.If | None:
    """循环体里包住 append 的那层 if；没有就是 None。"""
    for stmt in loop.body:
        if isinstance(stmt, ast.If) and _appends(stmt):
            return stmt
    return None


def _appended_dict(source: str, name: str) -> ast.Dict:
    """循环里唯一那处 append 的 dict 实参。"""
    calls = _appends(_for_loop(source, name))
    assert len(calls) == 1, f"循环里应当只有一处 append，实际 {len(calls)} 处"
    argument = calls[0].args[0]
    assert isinstance(argument, ast.Dict), "append 的实参应当是 dict 字面量"
    return argument


def _keys(literal: ast.Dict) -> list[str]:
    """dict 字面量的键，按书写顺序。"""
    names: list[str] = []
    for key in literal.keys:
        assert isinstance(key, ast.Constant) and isinstance(key.value, str), "键应当是字符串字面量"
        names.append(key.value)
    return names


def _access_form(node: ast.expr) -> str:
    """把取值表达式还原成 `obj.field` / `obj["field"]` / `obj.get("field")` 三种风格。

    只比较“取值形式”，引号统一用规格里的双引号写法，不把引号风格变成断言的一部分。
    """
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
        return f"{node.value.id}.{node.attr}"
    if (
        isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Name)
        and isinstance(node.slice, ast.Constant)
        and isinstance(node.slice.value, str)
    ):
        return f'{node.value.id}["{node.slice.value}"]'
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "get"
        and isinstance(node.func.value, ast.Name)
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and isinstance(node.args[0].value, str)
    ):
        return f'{node.func.value.id}.get("{node.args[0].value}")'
    return "<其它>"


def _lines(text: str) -> list[str]:
    """按 \\n 切行并保留行尾，便于逐字节比较。"""
    return text.splitlines(keepends=True)


def _common_prefix(first: str, second: str) -> int:
    limit = min(len(first), len(second))
    index = 0
    while index < limit and first[index] == second[index]:
        index += 1
    return index


def _common_suffix(first: str, second: str) -> int:
    limit = min(len(first), len(second))
    index = 0
    while index < limit and first[-1 - index] == second[-1 - index]:
        index += 1
    return index


def _line_offsets(text: str) -> list[int]:
    """每一行行首的字符偏移，末尾多一项（等于正文长度）。"""
    offsets = [0]
    for line in _lines(text):
        offsets.append(offsets[-1] + len(line))
    return offsets


def _comments(source: str) -> list[str]:
    """源码里的全部注释文本，按出现顺序。"""
    readline = io.StringIO(source).readline
    return [
        token.string
        for token in tokenize.generate_tokens(readline)
        if token.type == tokenize.COMMENT
    ]


def _blank_lines(source: str) -> list[int]:
    """只含空白（或为空）的行号，1 基。"""
    return [index for index, line in enumerate(_lines(source), 1) if not line.strip()]


# --------------------------------------------------------------------------
# 1. 头号样本：守卫子句与循环后排序必须留下
# --------------------------------------------------------------------------


def test_guard_clause_and_sort_survive_the_surgery() -> None:
    """守卫子句、return [] 与循环后的 sort 逐字保留；复合条件换成单一条件。"""
    out, _, _ = _assemble(HEADLINE_SOURCE, "paid_orders", "paid", ["order_id", "total"])
    ast.parse(out)

    # 循环之前与之后的代码：一字未动。
    assert "    if not orders:\n" in out
    assert "        return []\n" in out
    assert '    rows.sort(key=lambda r: -r["total"])\n' in out

    # 复合条件没了，换成一个条件；取值风格仍是属性风格。
    assert "not order.shipped" not in out
    assert "shipped" not in out
    loop = _for_loop(out, "paid_orders")
    guard = _guard_if(loop)
    assert guard is not None, "循环里应当有包住 append 的判断"
    assert isinstance(guard.test, ast.Attribute), "条件应当是对选中字段的一次读取"
    assert guard.test.attr == "paid"
    assert isinstance(guard.test.value, ast.Name) and guard.test.value.id == "order"
    assert len(_appends(loop)) == 1, "循环里应当只有一处 append"


def test_headline_function_filters_on_paid_and_still_sorts_descending() -> None:
    """运行时证据：只看 paid，且循环后的排序真的把结果按 total 从高到低排好。"""
    out, _, _ = _assemble(HEADLINE_SOURCE, "paid_orders", "paid", ["order_id", "total"])
    orders = [
        _Order("A1", 10.0, paid=True, shipped=True),  # paid 就留下，shipped 不参与
        _Order("A2", 30.0, paid=True, shipped=False),
        _Order("A3", 20.0, paid=False, shipped=False),  # 没付款，过滤掉
        _Order("A4", 5.0, paid=True, shipped=False),
    ]
    assert _run(out, "paid_orders", orders) == [
        {"order_id": "A2", "total": 30.0},
        {"order_id": "A1", "total": 10.0},
        {"order_id": "A4", "total": 5.0},
    ]
    # 守卫子句仍然生效：空输入与 None 都走 `if not orders: return []`。
    assert _run(out, "paid_orders", []) == []
    assert _run(out, "paid_orders", None) == []
    assert _run(HEADLINE_SOURCE, "paid_orders", None) == []


# --------------------------------------------------------------------------
# 2. 函数之外逐字节不变
# --------------------------------------------------------------------------


def test_everything_outside_the_function_is_byte_identical() -> None:
    """函数之前、函数身后的另一个函数都逐字节保留。"""
    source = HEADLINE_WITH_TAIL
    out, candidates, _ = _assemble(source, "paid_orders", "paid", ["order_id", "total"])
    source_lines, out_lines = _lines(source), _lines(out)
    head = "".join(source_lines[: candidates.start_line - 1])
    tail = "".join(source_lines[candidates.end_line :])
    assert head.startswith('"""订单。"""')
    assert "def tail_helper" in tail, "样本要保证函数身后确实有代码"

    assert "".join(out_lines[: candidates.start_line - 1]) == head
    assert "".join(out_lines[len(out_lines) - len(_lines(tail)) :]) == tail


# --------------------------------------------------------------------------
# 3. 新加一个过滤条件
# --------------------------------------------------------------------------


def test_filter_is_inserted_when_the_loop_has_no_if() -> None:
    """没有 if 时插入 `if <条件>:`，并把 append 挪进判断体。"""
    out, _, _ = _assemble(NO_FILTER_SOURCE, "active_users", "active", ["id", "name"])
    ast.parse(out)
    loop = _for_loop(out, "active_users")
    guard = _guard_if(loop)
    assert guard is not None, "没有判断时应当插入 `if <条件>:`"
    test = guard.test
    assert isinstance(test, ast.Subscript) and isinstance(test.value, ast.Name)
    assert test.value.id == "user" and test.slice.value == "active"
    assert len(_appends(loop)) == 1, "还是只有一处 append"
    assert len(_appends(guard)) == 1, "append 应当被放进新判断体里"


def test_added_filter_really_filters_at_runtime() -> None:
    """运行时证据：新加的 if 真的在过滤。"""
    out, _, _ = _assemble(NO_FILTER_SOURCE, "active_users", "active", ["id", "name"])
    users = [
        {"id": 1, "name": "Ada", "active": True},
        {"id": 2, "name": "Bob", "active": False},
        {"id": 3, "name": "Cid", "active": True},
    ]
    assert _run(out, "active_users", users) == [
        {"id": 1, "name": "Ada"},
        {"id": 3, "name": "Cid"},
    ]


# --------------------------------------------------------------------------
# 4. 去掉过滤条件
# --------------------------------------------------------------------------


def test_filter_removal_drops_the_if_and_dedents_the_body() -> None:
    """去掉条件：循环里的 if 消失，append 回到循环体第一层。"""
    out, _, _ = _assemble(GUARDED_SOURCE, "users", None, ["id"])
    ast.parse(out)
    loop = _for_loop(out, "users")
    assert not any(isinstance(node, ast.If) for node in ast.walk(loop)), "循环里不该再留下 if"
    assert len(_direct_appends(loop)) == 1, "append 应当直接写在循环体里"
    assert len(_appends(loop)) == 1


def test_removed_filter_returns_every_item_unfiltered() -> None:
    """运行时证据：不再过滤，顺序也保持原样。"""
    out, _, _ = _assemble(GUARDED_SOURCE, "users", None, ["id"])
    rows = [{"id": 1, "active": True}, {"id": 2, "active": False}]
    assert _run(out, "users", rows) == [{"id": 1}, {"id": 2}]


# --------------------------------------------------------------------------
# 5. 替换 dict：只留选中字段，按决策顺序
# --------------------------------------------------------------------------


def test_dict_replacement_keeps_requested_order_and_drops_unselected_fields() -> None:
    """dict 里只剩选中的两个字段，顺序按决策来，未选中的 note 不再出现。"""
    out, _, _ = _assemble(ORDER_SOURCE, "paid_orders", "paid", ["total", "order_id"])
    ast.parse(out)
    literal = _appended_dict(out, "paid_orders")
    assert _keys(literal) == ["total", "order_id"]
    forms = [_access_form(value) for value in literal.values]
    assert forms == ["order.total", "order.order_id"]
    assert "note" not in out, "未选中的字段名不该再出现"


def test_dict_replacement_runtime_keys_follow_requested_order() -> None:
    """运行时证据：返回的 dict 键与顺序都与决策一致。"""
    out, _, _ = _assemble(ORDER_SOURCE, "paid_orders", "paid", ["total", "order_id"])
    records = _run(out, "paid_orders", [_Order("A1", 10.0, paid=True, shipped=False)])
    assert records == [{"total": 10.0, "order_id": "A1"}]
    assert list(records[0]) == ["total", "order_id"]


# --------------------------------------------------------------------------
# 6. 取值风格保留
# --------------------------------------------------------------------------


def test_attribute_access_style_is_preserved() -> None:
    """属性风格：组装后仍是 order.total / order.note 这种写法。"""
    out, _, _ = _assemble(ATTR_SOURCE, "paid_orders", "paid", ["total"])
    ast.parse(out)
    literal = _appended_dict(out, "paid_orders")
    assert _keys(literal) == ["total"]
    assert [_access_form(value) for value in literal.values] == ["order.total"]
    assert "order.note" not in out
    orders = [_Order("A1", 10.0, paid=True, shipped=False)]
    assert _run(out, "paid_orders", orders) == [{"total": 10.0}]


def test_subscript_access_style_is_preserved() -> None:
    """下标风格：组装后仍是 user["name"] 这种写法。"""
    out, _, _ = _assemble(ITEM_SOURCE, "active_users", "active", ["name"])
    ast.parse(out)
    literal = _appended_dict(out, "active_users")
    assert _keys(literal) == ["name"]
    assert [_access_form(value) for value in literal.values] == ['user["name"]']
    users = [{"id": 1, "name": "Ada", "active": True}]
    assert _run(out, "active_users", users) == [{"name": "Ada"}]


def test_get_access_style_is_preserved() -> None:
    """.get() 风格：组装后仍用 .get，缺键不会抛 KeyError。"""
    out, _, _ = _assemble(GET_SOURCE, "visible_products", "visible", ["price", "sku"])
    ast.parse(out)
    literal = _appended_dict(out, "visible_products")
    assert _keys(literal) == ["price", "sku"]
    forms = [_access_form(value) for value in literal.values]
    assert forms == ['product.get("price")', 'product["sku"]']
    products = [
        {"sku": "S1", "price": 3, "visible": True},
        {"sku": "S2", "visible": True},  # 没有 price：只有 .get 才不会炸
    ]
    assert _run(out, "visible_products", products) == [
        {"price": 3, "sku": "S1"},
        {"price": None, "sku": "S2"},
    ]


# --------------------------------------------------------------------------
# 7. 与现状一致的决策：仍是合法代码，行为不变
# --------------------------------------------------------------------------


def test_decision_matching_current_behaviour_is_a_safe_noop() -> None:
    """决策与原文行为一致时：仍是合法 Python，运行结果与原文逐项相同。"""
    out, _, _ = _assemble(NOOP_SOURCE, "users", "active", ["id", "name"])
    ast.parse(out)
    loop = _for_loop(out, "users")
    assert len(_appends(loop)) == 1 and _guard_if(loop) is not None
    for rows in (
        [],
        [{"id": 1, "name": "Ada", "active": True}],
        [
            {"id": 1, "name": "Ada", "active": True},
            {"id": 2, "name": "Bob", "active": False},
        ],
    ):
        assert _run(out, "users", rows) == _run(NOOP_SOURCE, "users", rows)


# --------------------------------------------------------------------------
# 8. 不支持的形状：拒绝，而不是猜
# --------------------------------------------------------------------------


@pytest.mark.parametrize("source", UNSUPPORTED_SOURCES)
def test_unsupported_accumulation_shapes_are_refused(source: str) -> None:
    """+=、循环里两处 append、append 的不是 dict 字面量：一律 DecisionError。"""
    candidates, decision = _decide(source, "f", None, ["id"])
    with pytest.raises(DecisionError):
        assemble(source, candidates, decision)


# --------------------------------------------------------------------------
# 9. 过期候选：拒绝
# --------------------------------------------------------------------------


def test_stale_candidates_are_refused() -> None:
    """候选来自另一份源码：函数名或行范围对不上就拒绝，绝不按旧坐标动刀。"""
    candidates, decision = _decide(STALE_SOURCE, "f", None, ["id"])
    ast.parse(assemble(STALE_SOURCE, candidates, decision))  # 同一份源码照常工作

    moved = extract(STALE_SOURCE_SHIFTED, "f")
    assert (moved.start_line, moved.end_line) != (candidates.start_line, candidates.end_line), (
        "样本本身要保证行范围不同，否则这条测试没有意义"
    )
    with pytest.raises(DecisionError):
        assemble(STALE_SOURCE_SHIFTED, candidates, decision)

    with pytest.raises(DecisionError):
        assemble(STALE_SOURCE_RENAMED, candidates, decision)


# --------------------------------------------------------------------------
# 10. 注释与空行逐字保留（字节层面）
# --------------------------------------------------------------------------


def test_comments_and_blank_lines_survive() -> None:
    """注释一条不少、空行一行不少，改动只发生在循环那几行里。"""
    out, _, _ = _assemble(
        COMMENTED_SOURCE, "paid_orders", "paid", ["total", "order_id"]
    )
    ast.parse(out)

    assert _comments(out) == _comments(COMMENTED_SOURCE)
    assert _blank_lines(out) == _blank_lines(COMMENTED_SOURCE)
    assert "# 行尾注释：必须保留" in out
    assert '    rows.sort(key=lambda r: -r["total"])\n' in out

    # 手术确实发生了：条件换成单个 paid，dict 换成按决策顺序的两个字段。
    assert "not order.shipped" not in out
    assert _keys(_appended_dict(out, "paid_orders")) == ["total", "order_id"]

    # 字节层面的封闭性：公共前缀延伸到循环开始，公共后缀延伸到循环之后，
    # 说明循环以外的每一个字节都没被碰过。
    loop = _for_loop(COMMENTED_SOURCE, "paid_orders")
    offsets = _line_offsets(COMMENTED_SOURCE)
    allowed_start = offsets[loop.lineno - 1]
    allowed_end = offsets[min(loop.end_lineno or loop.lineno, len(offsets) - 1)]
    assert _common_prefix(COMMENTED_SOURCE, out) >= allowed_start, "循环之前的字节被改动了"
    assert (
        len(COMMENTED_SOURCE) - _common_suffix(COMMENTED_SOURCE, out) <= allowed_end
    ), "循环之后的字节被改动了"
