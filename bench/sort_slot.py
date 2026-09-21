"""排序槽位的实测：值不值得留，用四个数字说话。

给选择路线加了第三个槽位（排序：按哪个字段、升序还是降序）之后，按
`docs/12-selection-slots.md` 的规矩要量四件事，本脚本就是这四件事：

1. **正确率**：6 条要求排序的指令，真的 `exec` 组装出来的函数、真的调用，
   检查过滤、返回字段与**顺序**是否都符合要求；不是看代码像不像。
2. **输出 token 的税**：
   A. 决策里**不用**排序槽位时，新提示比旧提示多花了几个输出 token
      （旧提示是加槽位前那句系统提示的逐字副本，改的只有系统提示，用户消息一字未动）；
   B. 决策里**用上**排序槽位时，比同一份源码、同一条指令的不排序版多几个 token。
3. **老任务回归**：只过滤 + 裁字段的指令（决策里不带排序）在新提示下的正确率与耗时，
   以及原文本来就有 `sort` 时是否逐字保住（这正是今天修了一整天的失败模式）。
4. **拒绝质量**：候选表里没有的排序 id 是否明确抛 `DecisionError` 而不是猜一个。

正确性判断一律走运行时行为：`extract → 模型决策 → parse_decision → assemble →
exec → 真调用 → 比对返回值`；另外核对函数之外的字节逐字未动、必须出现的排序行确实
出现、必须消失的旧排序确实消失。温度 0，输出确定，所以重复次数只用来量耗时。

用法（在项目根目录）：

    HF_HUB_OFFLINE=1 .venv/bin/python bench/sort_slot.py --repeats 3
    HF_HUB_OFFLINE=1 .venv/bin/python bench/sort_slot.py --repeats 3 --json bench/sort_slot.json
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:  # 直接以脚本方式运行时，也能 import chooseonly
    sys.path.insert(0, str(ROOT))
# 只用本机模型目录：整个脚本不需要联网。
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from chooseonly.decide import (  # noqa: E402 - 先修好 sys.path 再导入
    DECISION_MAX_TOKENS,
    DECISION_SYSTEM_PROMPT,
    DecisionError,
    assemble,
    build_decision_prompt,
    describe_decision,
    extract,
    parse_decision,
)
from chooseonly.model import MLXEngine  # noqa: E402

MODEL_15B = ROOT / "models" / "Qwen2.5-Coder-1.5B-Instruct-4bit"

# 加排序槽位**之前**的系统提示逐字副本：这是 token 税的对照基线。
# 注意：加槽位时**用户消息也加了一行** `排序槽位：…`，所以只换系统提示不够，
# select_once 会把那行一起去掉；否则两边的差被低估，token 税就不成立。
BASE_SYSTEM_PROMPT = (
    "你只回一个 JSON 对象，不写代码、不解释、不加围栏。\n"
    "只能使用给出的候选 id，不得发明新的 id 或字段名。\n"
    "只回这两个键：f（条件 id，不过滤时用 null）、r（字段 id 的数组，按要求的输出顺序）。\n"
    "正确形状示例：" '{"f": "c2", "r": ["f0", "f1"]}'
)

# 加槽位时才新增到**用户消息**里的那一行（`build_decision_prompt` 的 sort_note）。
SORT_NOTE_PREFIX = "排序槽位："


# --------------------------------------------------------------------------
# 固定任务：源码 + 指令 + 样本 + 期望返回值（期望值一律手写，不由被测代码推导）
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Task:
    """一条固定任务；expect 是运行时的期望返回值，顺序也要对上。"""

    label: str
    kind: str  # "sort" | "control"（同一条指令的不排序版）| "legacy"（老任务类）
    instruction: str
    source: str
    function: str
    sample: list[Any]
    expect: list[Any]
    must_contain: tuple[str, ...] = ()
    must_not_contain: tuple[str, ...] = ()
    pair: str = ""  # 配对键：sort 与它的 control 用同一个键
    guard_none: bool = False  # 空输入 / None 是否仍走守卫返回 []


# 属性取值风格 + 前置守卫 + 循环之后本来就有降序排序：替换已有排序的样本。
ORDERS_SORTED_DESC = '''"""订单。"""


def paid_orders(orders):
    """返回已付款订单的编号与金额，按金额从高到低。"""
    if not orders:
        return []
    rows = []
    for order in orders:
        if order.paid:
            rows.append({"order_id": order.order_id, "total": order.total})
    rows.sort(key=lambda r: -r["total"])
    return rows
'''

# 字典取值风格 + 过滤，循环之后没有任何排序：插入排序的样本。
USERS_FILTERED = '''"""用户。"""


def active_users(users):
    """返回有效用户。"""
    result = []
    for user in users:
        if user["active"]:
            result.append({"id": user["id"], "name": user["name"]})
    return result
'''

# `.get()` 取值风格：排序键要跟着用 `.get()`。
PRODUCTS_GET = '''"""商品。"""


def visible_products(products):
    """返回上架商品。"""
    out = []
    for product in products:
        if product.get("visible"):
            out.append({"sku": product["sku"], "price": product.get("price")})
    return out
'''

# 单引号 + `out = sorted(out, ...)`：整条替换，不能留下第二处排序。
PRICED_SORTED_ASSIGN = '''"""价格表。"""


def priced(rows):
    """返回名称与价格。"""
    out = []
    for row in rows:
        out.append({'name': row['name'], 'price': row['price']})
    out = sorted(out, key=lambda r: r['price'])
    return out
'''


def _order(order_id: str, total: float, *, paid: bool) -> Any:
    """测试订单对象：属性取值风格。"""
    return type("Order", (), {"order_id": order_id, "total": total, "paid": paid})()


ORDERS_SAMPLE = [
    _order("C", 20.0, paid=True),
    _order("A", 10.0, paid=True),
    _order("X", 99.0, paid=False),  # 没付款：任何一条正确产物都不该留下它
    _order("B", 30.0, paid=True),
]

TASKS: tuple[Task, ...] = (
    # ---- 排序任务 1：把原文的降序排序换成升序（替换已有排序） ----
    Task(
        label="orders: 按金额从低到高",
        kind="sort",
        pair="orders-total",
        instruction="只保留已付款订单，返回编号和金额，按金额从低到高。",
        source=ORDERS_SORTED_DESC,
        function="paid_orders",
        sample=ORDERS_SAMPLE,
        expect=[
            {"order_id": "A", "total": 10.0},
            {"order_id": "C", "total": 20.0},
            {"order_id": "B", "total": 30.0},
        ],
        must_contain=('rows.sort(key=lambda r: r["total"])',),
        must_not_contain=('-r["total"]',),
        guard_none=True,
    ),
    Task(
        label="orders 对照: 其他不变（不排序）",
        kind="control",
        pair="orders-total",
        instruction="只保留已付款订单，返回编号和金额，其他不变。",
        source=ORDERS_SORTED_DESC,
        function="paid_orders",
        sample=ORDERS_SAMPLE,
        expect=[
            {"order_id": "B", "total": 30.0},
            {"order_id": "C", "total": 20.0},
            {"order_id": "A", "total": 10.0},
        ],
        must_contain=('rows.sort(key=lambda r: -r["total"])',),  # 原文那条逐字留住
        guard_none=True,
    ),
    # ---- 排序任务 2：本来没有排序，插一条升序 ----
    Task(
        label="users: 按 name 升序",
        kind="sort",
        pair="users-name",
        instruction="只保留 active 为真的项，返回 id 和 name，按名字升序。",
        source=USERS_FILTERED,
        function="active_users",
        sample=[
            {"id": 1, "name": "cid", "active": True},
            {"id": 2, "name": "ada", "active": False},
            {"id": 3, "name": "bob", "active": True},
        ],
        expect=[{"id": 3, "name": "bob"}, {"id": 1, "name": "cid"}],
        must_contain=('result.sort(key=lambda r: r["name"])',),
    ),
    Task(
        label="users 对照: 保持原顺序（不排序）",
        kind="control",
        pair="users-name",
        instruction="只保留 active 为真的项，返回 id 和 name，保持原顺序。",
        source=USERS_FILTERED,
        function="active_users",
        sample=[
            {"id": 1, "name": "cid", "active": True},
            {"id": 2, "name": "ada", "active": False},
            {"id": 3, "name": "bob", "active": True},
        ],
        expect=[{"id": 1, "name": "cid"}, {"id": 3, "name": "bob"}],
        must_not_contain=(".sort(", "sorted("),  # 没有排序就不该凭空多一条
    ),
    # ---- 排序任务 3：换一个排序字段（替换已有排序） ----
    Task(
        label="orders: 按编号升序",
        kind="sort",
        pair="orders-id",
        instruction="只保留已付款订单，返回编号和金额，按编号升序。",
        source=ORDERS_SORTED_DESC,
        function="paid_orders",
        sample=ORDERS_SAMPLE,
        expect=[
            {"order_id": "A", "total": 10.0},
            {"order_id": "B", "total": 30.0},
            {"order_id": "C", "total": 20.0},
        ],
        must_contain=('rows.sort(key=lambda r: r["order_id"])',),
        must_not_contain=('-r["total"]',),
        guard_none=True,
    ),
    # ---- 排序任务 4：`.get()` 风格 + 过滤 + 降序（插入） ----
    Task(
        label="products: 按 price 降序",
        kind="sort",
        pair="products-price",
        instruction="先过滤 visible 为真的商品，返回 sku 和 price，按价格从高到低。",
        source=PRODUCTS_GET,
        function="visible_products",
        sample=[
            {"sku": "S3", "price": 20, "visible": True},
            {"sku": "S1", "price": 10, "visible": False},
            {"sku": "S2", "price": 30, "visible": True},
        ],
        expect=[{"sku": "S2", "price": 30}, {"sku": "S3", "price": 20}],
        must_contain=('out.sort(key=lambda r: r.get("price"), reverse=True)',),
    ),
    Task(
        label="products 对照: 其他不变（不排序）",
        kind="control",
        pair="products-price",
        instruction="只保留 visible 为真的商品，返回 sku 和 price，其他不变。",
        source=PRODUCTS_GET,
        function="visible_products",
        sample=[
            {"sku": "S3", "price": 20, "visible": True},
            {"sku": "S1", "price": 10, "visible": False},
            {"sku": "S2", "price": 30, "visible": True},
        ],
        expect=[{"sku": "S3", "price": 20}, {"sku": "S2", "price": 30}],
        must_not_contain=(".sort(", "sorted("),
    ),
    # ---- 排序任务 5：字典风格 + 降序（插入） ----
    Task(
        label="users: 按 id 降序",
        kind="sort",
        pair="users-id",
        instruction="只保留 active 为真的项，返回 id 和 name，按 id 降序。",
        source=USERS_FILTERED,
        function="active_users",
        sample=[
            {"id": 1, "name": "ada", "active": True},
            {"id": 2, "name": "bob", "active": False},
            {"id": 3, "name": "cid", "active": True},
        ],
        expect=[{"id": 3, "name": "cid"}, {"id": 1, "name": "ada"}],
        must_contain=('result.sort(key=lambda r: r["id"], reverse=True)',),
    ),
    # ---- 排序任务 6：`out = sorted(out, ...)` 换成降序（整条替换） ----
    Task(
        label="priced: 按 price 降序（替换 sorted 赋值）",
        kind="sort",
        pair="priced-price",
        instruction="返回 name 和 price，按价格从高到低。",
        source=PRICED_SORTED_ASSIGN,
        function="priced",
        sample=[{"name": "x", "price": 5}, {"name": "y", "price": 9}],
        expect=[{"name": "y", "price": 9}, {"name": "x", "price": 5}],
        must_contain=("out.sort(key=lambda r: r['price'], reverse=True)",),
        must_not_contain=("sorted(",),
    ),
    Task(
        label="priced 对照: 其他不变（不排序）",
        kind="control",
        pair="priced-price",
        instruction="返回 name 和 price，其他不变。",
        source=PRICED_SORTED_ASSIGN,
        function="priced",
        sample=[{"name": "x", "price": 5}, {"name": "y", "price": 9}],
        expect=[{"name": "x", "price": 5}, {"name": "y", "price": 9}],
        must_contain=("out = sorted(out, key=lambda r: r['price'])",),  # 逐字留住
    ),
)

# 老任务类的固定任务：与 `bench/compare.py` 的任务 A 同源同指令（口径完全一致）。
LEGACY_SOURCE = '''"""用户列表：当前返回全部字段。"""


def active_users(users):
    """返回用户，保持原顺序。"""
    result = []
    for user in users:
        result.append(
            {"id": user["id"], "name": user["name"], "active": user["active"]}
        )
    return result
'''

LEGACY_TASK = Task(
    label="老任务 A：过滤 + 裁字段（不排序）",
    kind="legacy",
    instruction="修改函数：只保留 active 为真的项，返回 id 和 name，保持原顺序，其他不变。",
    source=LEGACY_SOURCE,
    function="active_users",
    sample=[
        {"id": 1, "name": "A", "active": True, "extra": "x"},
        {"id": 2, "name": "B", "active": False, "extra": "y"},
        {"id": 3, "name": "C", "active": True, "extra": "z"},
    ],
    expect=[{"id": 1, "name": "A"}, {"id": 3, "name": "C"}],  # 与 bench/compare.py 口径一致
)


# --------------------------------------------------------------------------
# 判定：运行时行为，不看代码像不像
# --------------------------------------------------------------------------


@dataclass
class Run:
    """一次决策的执行结果：正文、判定理由、耗时与输出 token。"""

    label: str
    ok: bool
    reasons: tuple[str, ...]
    body: str
    raw: str
    tokens: int
    generate_seconds: float
    wall_seconds: float


def check_body(body: str, task: Task) -> tuple[str, ...]:
    """把组装结果真的跑一遍：过滤、字段、顺序、守卫、函数外字节都要对。"""
    reasons: list[str] = []
    try:
        tree = ast.parse(body)
    except SyntaxError as exc:
        return (f"语法错误: {exc.msg}",)
    for line in task.must_contain:
        if line not in body:
            reasons.append(f"缺少应有的排序行: {line}")
    for line in task.must_not_contain:
        if line in body:
            reasons.append(f"不该出现的写法仍在: {line}")

    # 函数之外的字节必须逐字未动。函数可能因为插了一行排序而变长，所以尾部要从
    # **产物里**这个函数的新结束行往后比，不能拿原文的行号直接套。
    try:
        candidates = extract(task.source, task.function)
        source_lines = task.source.splitlines(keepends=True)
        body_lines = body.splitlines(keepends=True)
        if body_lines[: candidates.start_line - 1] != source_lines[: candidates.start_line - 1]:
            reasons.append("函数之前的字节被改动")
        end = next(
            (
                node.end_lineno or node.lineno
                for node in ast.walk(tree)
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name == task.function
            ),
            None,
        )
        if end is None:
            reasons.append(f"产物里找不到函数 {task.function}")
        elif body_lines[end:] != source_lines[candidates.end_line :]:
            reasons.append("函数之后的字节被改动")
    except DecisionError as exc:
        reasons.append(f"复核候选时失败: {exc}")

    namespace: dict[str, Any] = {}
    try:
        exec(compile(tree, "<sort-slot>", "exec"), namespace)  # noqa: S102 - 只跑本脚本的固定样本
    except Exception as exc:  # noqa: BLE001 - 任何执行失败都记成一次失败
        return tuple(reasons + [f"无法执行: {type(exc).__name__}: {exc}"])

    function = namespace.get(task.function)
    if not callable(function):
        return tuple(reasons + [f"找不到可调用函数 {task.function}"])
    try:
        got = function(list(task.sample))
    except Exception as exc:  # noqa: BLE001
        return tuple(reasons + [f"调用失败: {type(exc).__name__}: {exc}"])
    if got != task.expect:
        reasons.append(f"返回值不对: 实际 {got}，期望 {task.expect}")
    if task.guard_none:
        for empty in ([], None):
            try:
                if function(empty) != []:
                    reasons.append(f"守卫失效: 空输入 {empty!r} 没有返回 []")
            except Exception as exc:  # noqa: BLE001
                reasons.append(f"守卫失效: 空输入 {empty!r} 抛 {type(exc).__name__}")
    return tuple(reasons)


def select_once(
    engine: MLXEngine,
    task: Task,
    system_prompt: str | None = None,
    *,
    sort_enabled: bool = False,
) -> tuple[Run, str]:
    """跑一次选择路线；system_prompt 给定时替换系统提示（token 税对照用）。

    ``sort_enabled`` 必须由调用方按任务类型显式传入：排序任务传 True，控制组和
    老任务传 False。**这不是可选的**——宿主在未启用时会在系统提示和用户消息里
    都禁止 s / d，漏传会让排序任务被结构性地判成失败，与模型能力无关。

    返回 (Run, 决策摘要)；决策不合法的回复也记成一次失败，不打断整轮。
    """
    start = time.perf_counter()
    reasons: list[str] = []
    body = ""
    raw = ""
    tokens = 0
    generate_seconds = 0.0
    try:
        candidates = extract(task.source, task.function)
        messages = build_decision_prompt(
            task.instruction, candidates, sort_enabled=sort_enabled
        )
        if system_prompt is not None:
            # 旧提示基线：系统提示换回加槽位前的逐字副本，用户消息也要去掉
            # 加槽位时才新增的那行，否则两边的差被低估，token 税就不成立。
            messages[0]["content"] = system_prompt
            messages[1]["content"] = "\n".join(
                line
                for line in messages[1]["content"].split("\n")
                if not line.startswith(SORT_NOTE_PREFIX)
            )
        raw, stats = engine.generate(messages, max_tokens=DECISION_MAX_TOKENS)
        tokens, generate_seconds = stats.generated_tokens, stats.generate_seconds
        decision = parse_decision(raw, candidates, sort_enabled=sort_enabled)
        summary = describe_decision(candidates, decision)
        body = assemble(task.source, candidates, decision)
        reasons.extend(check_body(body, task))
    except DecisionError as exc:
        summary = f"决策不合法: {exc}"
        reasons.append(f"决策不合法: {exc}")
    except Exception as exc:  # noqa: BLE001 - 单次失败不该中断整轮实测
        summary = f"异常: {type(exc).__name__}: {exc}"
        reasons.append(f"异常: {type(exc).__name__}: {exc}")
    return (
        Run(
            label=task.label,
            ok=not reasons,
            reasons=tuple(reasons),
            body=body,
            raw=raw,
            tokens=tokens,
            generate_seconds=generate_seconds,
            wall_seconds=time.perf_counter() - start,
        ),
        summary,
    )


@dataclass
class Group:
    """一组同名任务的全部重复运行。"""

    label: str
    kind: str
    runs: list[Run] = field(default_factory=list)
    summaries: list[str] = field(default_factory=list)

    @property
    def passed(self) -> int:
        return sum(1 for run in self.runs if run.ok)

    @property
    def tokens(self) -> int:
        """输出 token：温度 0 下每次逐字相同，取众数（等于最大值也等于最小值）。"""
        return max((run.tokens for run in self.runs), default=0)

    def seconds(self) -> tuple[float, float, float]:
        """生成耗时的 (均值, 最小, 最大)。"""
        values = [run.generate_seconds for run in self.runs if run.generate_seconds > 0]
        if not values:
            return (0.0, 0.0, 0.0)
        return (statistics.mean(values), min(values), max(values))


def run_group(
    engine: MLXEngine,
    task: Task,
    repeats: int,
    system_prompt: str | None = None,
    *,
    sort_enabled: bool = False,
) -> Group:
    group = Group(label=task.label, kind=task.kind)
    for _ in range(repeats):
        run, summary = select_once(
            engine, task, system_prompt, sort_enabled=sort_enabled
        )
        group.runs.append(run)
        group.summaries.append(summary)
    return group


# --------------------------------------------------------------------------
# 拒绝质量：候选表里没有的排序 id 必须明确报错
# --------------------------------------------------------------------------

# (说明, 决策 JSON)：前四条是模型可能回的坏 id，后三条是缺一半 / 方向不合法。
REFUSAL_CASES: tuple[tuple[str, str, str], ...] = (
    ("排序字段候选表里没有", USERS_FILTERED, '{"f": "c0", "r": ["f1", "f2"], "s": "f9", "d": "desc"}'),
    ("拿条件 id 当排序字段", USERS_FILTERED, '{"f": "c0", "r": ["f1", "f2"], "s": "c0", "d": "desc"}'),
    ("自造字段名当排序字段", USERS_FILTERED, '{"f": "c0", "r": ["f1", "f2"], "s": "name", "d": "desc"}'),
    ("排序字段不在返回字段里", USERS_FILTERED, '{"f": "c0", "r": ["f1"], "s": "f2", "d": "desc"}'),
    ("只给排序字段没给方向", USERS_FILTERED, '{"f": "c0", "r": ["f1", "f2"], "s": "f2"}'),
    ("只给方向没给排序字段", USERS_FILTERED, '{"f": "c0", "r": ["f1", "f2"], "d": "desc"}'),
    ("方向不是 asc/desc", USERS_FILTERED, '{"f": "c0", "r": ["f1", "f2"], "s": "f2", "d": "横向"}'),
    ("旧长键想夹带排序", USERS_FILTERED,
     '{"function": "fn0", "filter_field": "c0", "return_fields": ["f1", "f2"], "s": "f2", "d": "desc"}'),
)


def refusal_report() -> list[tuple[str, str, str]]:
    """返回 [(说明, 是否拒绝, 拒绝原因或产物)]；不加载模型，纯宿主校验。

    **必须带 `sort_enabled=True`**：否则这些带 s/d 的坏决策会先被「排序槽位未启用」
    这道理拦住，永远走不到真正的排序校验器（未知排序 id、s 不在 r 里、方向不合法…），
    于是 8 条全绿也不代表校验有效。那是一个假通过。
    """
    rows: list[tuple[str, str, str]] = []
    for label, source, text in REFUSAL_CASES:
        try:
            candidates = extract(source, "active_users")
            decision = parse_decision(text, candidates, sort_enabled=True)
            assemble(source, candidates, decision)
            rows.append((label, "没有拒绝（有问题）", text))
        except DecisionError as exc:
            rows.append((label, "已拒绝", str(exc)))
    return rows


# 模型驱动的拒绝探针：指令要求按一个源码里根本不存在的字段排序。
MODEL_REFUSAL_PROBES: tuple[tuple[str, str], ...] = (
    ("按不存在的字段排序（client_rating）", "只保留 active 为真的项，返回 id 和 name，按客户评分降序。"),
    ("按不存在的字段排序（不返回该字段）", "只返回 id，按金额从高到低。"),
)


def model_refusal_probe(engine: MLXEngine, instruction: str) -> tuple[bool, str, str]:
    """让模型自己面对“排不出来的排序”：要么拒绝，要么仍在候选表内选。

    返回 (是否没有猜造 id, 结果说明, 模型原文)。判据只有一条：**宿主绝不接受
    候选表以外的 id**；模型要是编一个不存在的字段名，parse_decision 必须拒绝。
    """
    source = LEGACY_SOURCE
    candidates = extract(source, "active_users")
    # 提示必须启用排序槽位：否则模型被明令禁止输出 s / d，这个探针观察不到任何
    # 排序行为，"没编造 id" 就是一句空话。
    messages = build_decision_prompt(instruction, candidates, sort_enabled=True)
    raw, _stats = engine.generate(messages, max_tokens=DECISION_MAX_TOKENS)
    try:
        decision = parse_decision(raw, candidates, sort_enabled=True)
    except DecisionError as exc:
        return (True, f"模型回复被拒绝（未猜造 id）：{exc}", raw)
    known = {item.name for item in candidates.fields}
    invented = (
        decision.sort_field is not None
        and decision.sort_field not in {item.id for item in candidates.fields}
    )
    names = {item.id: item.name for item in candidates.fields}
    if invented:  # 只有校验被绕过才会走到这里
        return (False, f"接受了候选表以外的排序 id：{decision.sort_field}", raw)
    chosen = "无" if decision.sort_field is None else f"{names[decision.sort_field]}"
    return (
        True,
        f"未编造 id（决策里的排序字段：{chosen}；候选中没有 client_rating / 金额，"
        "模型只能选真实存在的字段或干脆不排序）",
        raw,
    )


# --------------------------------------------------------------------------
# 报告
# --------------------------------------------------------------------------


def _fmt_seconds(values: tuple[float, float, float]) -> str:
    if values[0] <= 0:
        return "—"
    return f"{values[0]:.2f} / {values[1]:.2f} / {values[2]:.2f}"


def _delta(new: int, old: int) -> str:
    diff = new - old
    return f"{diff:+d}" if diff else "0"


def report(
    engine: MLXEngine,
    sort_groups: dict[str, Group],
    controls_old: dict[str, Group],
    controls_new: dict[str, Group],
    legacy_old: Group,
    legacy_new: Group,
    repeats: int,
) -> dict[str, Any]:
    """打印四段报告，并返回给 --json 用的数字。"""
    total = sum(len(group.runs) for group in sort_groups.values())
    passed = sum(group.passed for group in sort_groups.values())

    print("\n=== 1. 排序任务的正确率（运行时行为判定，温度 0） ===")
    print(f"{'指令':<34}{'通过':>7}{'输出tokens':>11}{'生成 s 均值/最小/最大':>26}")
    failures: list[str] = []
    for group in sort_groups.values():
        print(
            f"{group.label:<34}{f'{group.passed}/{len(group.runs)}':>8}"
            f"{group.tokens:>10}{_fmt_seconds(group.seconds()):>26}"
        )
        for run in group.runs:
            if not run.ok:
                failures.append(f"[{group.label}] {'; '.join(run.reasons)}｜模型原文: {run.raw[:160]}")
    print(f"{'合计':<34}{f'{passed}/{total}':>8}")
    if failures:
        print("失败明细：")
        for line in failures:
            print(f"  {line}")

    print("\n=== 2. 输出 token 的税 ===")
    print("A. 决策里**不用**排序槽位：旧提示（加槽位前）vs 新提示，同一批任务")
    print(f"{'任务':<34}{'旧提示':>8}{'新提示':>8}{'差':>6}{'通过(旧/新)':>14}")
    tax_a: list[tuple[str, int, int]] = []
    for key, old in controls_old.items():
        new = controls_new[key]
        tax_a.append((old.label, old.tokens, new.tokens))
        print(
            f"{old.label:<34}{old.tokens:>8}{new.tokens:>8}{_delta(new.tokens, old.tokens):>6}"
            f"{f'{old.passed}/{len(old.runs)} / {new.passed}/{len(new.runs)}':>16}"
        )
    tax_a.append((legacy_old.label, legacy_old.tokens, legacy_new.tokens))
    print(
        f"{legacy_old.label:<34}{legacy_old.tokens:>8}{legacy_new.tokens:>8}"
        f"{_delta(legacy_new.tokens, legacy_old.tokens):>6}"
        f"{f'{legacy_old.passed}/{len(legacy_old.runs)} / {legacy_new.passed}/{len(legacy_new.runs)}':>16}"
    )

    print("\nB. 决策里**用上**排序槽位：与同一份源码、同一条指令的不排序版对比")
    print(f"{'排序任务':<34}{'不排序版':>9}{'排序版':>8}{'差':>6}{'通过(排序版)':>14}")
    tax_b: list[tuple[str, int, int]] = []
    for key, group in sort_groups.items():
        control = controls_new.get(key.split("|")[0])
        if control is None:
            continue
        tax_b.append((group.label, control.tokens, group.tokens))
        print(
            f"{group.label:<34}{control.tokens:>9}{group.tokens:>9}"
            f"{_delta(group.tokens, control.tokens):>6}"
            f"{f'{group.passed}/{len(group.runs)}':>14}"
        )
    all_tokens_a = [new - old for _label, old, new in tax_a]
    all_tokens_b = [new - old for _label, old, new in tax_b]
    print(
        "A 合计：不用排序的决策，新提示多 "
        f"{sum(all_tokens_a)} tokens（{len(all_tokens_a)} 条任务，均值 "
        f"{statistics.mean(all_tokens_a):.2f}，最小 {min(all_tokens_a)}、最大 {max(all_tokens_a)}）"
    )
    print(
        "B 合计：用上排序的决策，比不排序版多 "
        f"{sum(all_tokens_b)} tokens（{len(all_tokens_b)} 条配对，均值 "
        f"{statistics.mean(all_tokens_b):.2f}，最小 {min(all_tokens_b)}、最大 {max(all_tokens_b)}）"
    )
    print("（C. 提示的输入 token 只影响 prefill，不计入这里的“税”；系统提示长了几个字，"
          "决策提示每次多 20 个左右输入 token。）")

    print("\n=== 3. 老任务回归：只用旧槽位的任务（新提示） ===")
    print(f"{'任务':<34}{'通过':>7}{'输出tokens':>11}{'生成 s 均值/最小/最大':>26}")
    regression: list[tuple[str, int, int]] = []
    for group in [*controls_new.values(), legacy_new]:
        print(
            f"{group.label:<34}{f'{group.passed}/{len(group.runs)}':>8}"
            f"{group.tokens:>10}{_fmt_seconds(group.seconds()):>26}"
        )
        for run in group.runs:
            if not run.ok:
                print(f"  失败：{'; '.join(run.reasons)}｜模型原文: {run.raw[:160]}")
        regression.append((group.label, group.passed, len(group.runs)))

    print("\n=== 4. 拒绝质量（候选表里没有的排序 id） ===")
    refusals = refusal_report()
    for label, verdict, detail in refusals:
        print(f"  {'✓' if verdict == '已拒绝' else '✗'} {label}: {verdict}｜{detail}")
    print("模型驱动的探针（真的问模型要一个排不出来的排序）：")
    probes: list[tuple[str, bool, str]] = []
    for label, instruction in MODEL_REFUSAL_PROBES:
        ok, detail, raw = model_refusal_probe(engine, instruction)
        probes.append((label, ok, detail))
        print(f"  {'✓' if ok else '✗'} {label}: {detail}")
        print(f"      模型原文: {' '.join(raw.split())[:160]}")

    return {
        "repeats": repeats,
        "sort_tasks": [
            {
                "label": group.label,
                "passed": group.passed,
                "runs": len(group.runs),
                "tokens": group.tokens,
                "seconds_mean": group.seconds()[0],
                "seconds_min": group.seconds()[1],
                "seconds_max": group.seconds()[2],
            }
            for group in sort_groups.values()
        ],
        "sort_total": {"passed": passed, "runs": total},
        "tax_no_sort": [
            {"label": label, "old_tokens": old, "new_tokens": new, "delta": new - old}
            for label, old, new in tax_a
        ],
        "tax_with_sort": [
            {"label": label, "control_tokens": old, "sort_tokens": new, "delta": new - old}
            for label, old, new in tax_b
        ],
        "regression": [
            {"label": label, "passed": passed_n, "runs": runs}
            for label, passed_n, runs in regression
        ],
        "refusals": [
            {"label": label, "verdict": verdict, "detail": detail}
            for label, verdict, detail in refusals
        ],
        "model_probes": [
            {"label": label, "no_invented_id": ok, "detail": detail}
            for label, ok, detail in probes
        ],
    }


# --------------------------------------------------------------------------
# 入口
# --------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="排序槽位的四项实测（真实模型、运行时判定）")
    parser.add_argument("--repeats", type=int, default=3, help="每条任务重复几次（温度 0，用于耗时）")
    parser.add_argument("--model", default=str(MODEL_15B), help="本机模型目录")
    parser.add_argument("--json", default="", help="把数字写成 JSON 的路径（可选）")
    parser.add_argument("--skip-model", action="store_true", help="只跑拒绝质量（不加载权重）")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    load = os.getloadavg()
    print(f"模型：{args.model}")
    print(
        f"重复次数：{args.repeats}｜温度 0｜开始时的 load average："
        f"{load[0]:.2f} / {load[1]:.2f} / {load[2]:.2f}"
    )
    if args.skip_model:
        print("\n=== 4. 拒绝质量（只跑宿主校验） ===")
        for label, verdict, detail in refusal_report():
            print(f"  {'✓' if verdict == '已拒绝' else '✗'} {label}: {verdict}｜{detail}")
        return 0

    engine = MLXEngine(args.model)
    engine.generate([{"role": "user", "content": "1"}], max_tokens=1)  # 预热：把图编译成本挪走
    engine.generate([{"role": "user", "content": "1"}], max_tokens=1)
    print(f"模型加载 {engine.load_seconds:.2f} s（已预热，不计入下面的耗时）")

    sort_groups: dict[str, Group] = {}
    controls_old: dict[str, Group] = {}
    controls_new: dict[str, Group] = {}
    print("\n开始跑固定任务（旧提示 = 加排序槽位之前的系统提示逐字副本）……")
    for task in TASKS:
        if task.kind == "control":
            controls_old[task.pair] = run_group(engine, task, args.repeats, BASE_SYSTEM_PROMPT)
            controls_new[task.pair] = run_group(engine, task, args.repeats, None)
            print(f"  [对照] {task.label}")
        else:
            # 排序任务必须启用排序槽位；控制组与老任务保持未启用（默认），
            # 这样"未启用时不该出现排序"本身就是一条回归检查。
            sort_groups[f"{task.pair}|sort"] = run_group(
                engine, task, args.repeats, None, sort_enabled=True
            )
            print(f"  [排序] {task.label}")
    legacy_old = run_group(engine, LEGACY_TASK, args.repeats, BASE_SYSTEM_PROMPT)
    legacy_new = run_group(engine, LEGACY_TASK, args.repeats, None)
    print(f"  [老任务] {LEGACY_TASK.label}")

    numbers = report(
        engine, sort_groups, controls_old, controls_new, legacy_old, legacy_new, args.repeats
    )
    end = os.getloadavg()
    print(
        f"\n结束时的 load average：{end[0]:.2f} / {end[1]:.2f} / {end[2]:.2f}"
        "（机器有其他负载时，耗时只能同一次运行内比较）"
    )
    if args.json:
        Path(args.json).write_text(json.dumps(numbers, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"数字已写入 {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
