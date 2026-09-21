"""批处理 vs 顺序调用：N 份独立小工作，放进同一次前向到底省多少墙钟。

要回答的问题（数字全部来自本机真跑）：

1. 顺序调用（现有生产路径 chooseonly.model.MLXEngine，一次一个请求）与批处理
   （N 个请求放进同一次前向）各自的墙钟时间、每次调用时间。
2. 两种模式的总输出 token / 总秒数：解码吞吐到底随批大小上升，还是停在
   ~110 tok/s（GPU 已饱和）。
3. 答案一致性：温度 0 下批处理的答案必须与顺序调用**逐字相同**。不同就说明
   填充 / mask 错了——脚本会逐条报告哪一个请求的答案不一致。
4. 答案正确性：每条答案都走项目现有的运行时检查——取出所选 id，用
   chooseonly.decide.assemble 组装，真的 exec 并调用函数，核对过滤结果与返回字段
   恰好等于要求（不是看代码像不像）。
5. 前缀共享：多个请求共享同一段长上下文时，把共享前缀只 prefill 一次
   （复用 KV），省了多少；以及"在错误位置上读答案"会得到什么，作为
   "每条序列要在自己的末位读"的反证。

批处理路径的实现要点（这是本项目里第一次有批量路径，所以写清楚）：
  * 请求按 token 左侧填充到同一宽度。左填的语义是"所有序列在同一个绝对下标
    结束"，因此第一个 token 可以在同一个位置（width-1）统一读出。
  * KV cache 用 mlx_lm 自带的 BatchKVCache：它的 make_mask 会把每条序列的
    填充段排除在注意力之外，RoPE 位置也按每序列的 offset 走。
  * 解码阶段每条序列各喂自己的上一个 token；已经结束（EOS 或到上限）的序列
    喂 pad，输出丢弃，但绝不因此影响未结束序列（它们的 KV 段是独立的）。
  * 共享前缀变体：前缀只 prefill 一次（batch 1），再 repeat 成 N 条序列的 KV；
    后缀用**右侧填充**，第一个 token 必须逐序列在自己的末位（prefix_len +
    len(suffix_i) - 1）读出，然后 finalize() 把尾部填充滚到前面，后续解码走
    标准的左填 mask。

用法（在项目根目录）：

    HF_HUB_OFFLINE=1 .venv/bin/python bench/batching.py --rounds 3
    HF_HUB_OFFLINE=1 .venv/bin/python bench/batching.py --selfcheck   # 只验 mask 正确性

本脚本只读现有模块，不修改任何 chooseonly 代码；模型只加载一次。
"""

from __future__ import annotations

import argparse
import ast
import importlib
import os
import resource
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:  # 直接以脚本方式运行时也能 import chooseonly
    sys.path.insert(0, str(ROOT))

# 左侧填充用的 token id（Qwen 里就是普通 token，被 mask 挡住，不参与真实位置）。
PAD_ID = 0
# 与 bench/compare.py 的选择路线一致：决策输出几十个 token，64 只是防失控上限。
DEFAULT_MAX_TOKENS = 64
# 固定成本标定：生产路径只生成 1 个 token 的调用，量出与输出长度无关的那一段。
CALIBRATION_TRIALS = 5
# bench/compare.py 的 check_runtime_behaviour 写死了这一个任务的样本；
# 只有这一条能复用它再核一遍，其余任务走本文件里同样口径的通用检查。
COMPARE_TASK = "active_users"


# --------------------------------------------------------------------------
# 独立任务：8 份不同的工作（不同源文件 / 不同函数 / 不同字段）
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Task:
    """一份独立工作：给一个函数加一个过滤条件并裁字段，期望结果由宿主算好。"""

    name: str
    source: str
    function: str
    instruction: str
    cond: str  # 指令要求的过滤字段真实名
    returns: tuple[str, ...]  # 指令要求返回的字段名，按指令里的顺序
    id_field: str  # 核对过滤结果用的主键
    sample: tuple[dict, ...]
    sort_key: str | None = None  # 循环后有降序排序时，期望结果也要排


TASKS: tuple[Task, ...] = (
    Task(
        name="active_users",
        source='''"""用户列表：当前返回全部字段。"""


def active_users(users):
    """返回用户，保持原顺序。"""
    result = []
    for user in users:
        result.append(
            {"id": user["id"], "name": user["name"], "active": user["active"]}
        )
    return result
''',
        function="active_users",
        instruction="修改函数：只保留 active 为真的项，返回 id 和 name，保持原顺序，其他不变。",
        cond="active",
        returns=("id", "name"),
        id_field="id",
        sample=(
            {"id": 1, "name": "A", "active": True, "note": "x"},
            {"id": 2, "name": "B", "active": False, "note": "y"},
            {"id": 3, "name": "C", "active": True, "note": "z"},
            {"id": 4, "name": "D", "active": False, "note": "w"},
        ),
    ),
    Task(
        name="paid_orders",
        source='''"""订单列表：当前返回全部字段，且不过滤。"""


def paid_orders(orders):
    """返回订单，按金额从高到低。"""
    if not orders:
        return []
    rows = []
    for order in orders:
        rows.append(
            {
                "order_id": order["order_id"],
                "total": order["total"],
                "paid": order["paid"],
                "shipped": order["shipped"],
            }
        )
    rows.sort(key=lambda r: -r["total"])
    return rows
''',
        function="paid_orders",
        instruction="修改函数：只保留 paid 为真的项，返回 order_id 和 total，其他不变。",
        cond="paid",
        returns=("order_id", "total"),
        id_field="order_id",
        sample=(
            {"order_id": 101, "total": 30, "paid": True, "shipped": False, "note": "x"},
            {"order_id": 102, "total": 90, "paid": False, "shipped": False, "note": "y"},
            {"order_id": 103, "total": 50, "paid": True, "shipped": True, "note": "z"},
            {"order_id": 104, "total": 10, "paid": True, "shipped": False, "note": "w"},
        ),
        sort_key="total",
    ),
    Task(
        name="in_stock_books",
        source='''"""图书列表：当前返回全部字段。"""


def in_stock_books(books):
    """返回图书，保持原顺序。"""
    result = []
    for book in books:
        result.append(
            {
                "isbn": book["isbn"],
                "title": book["title"],
                "price": book["price"],
                "in_stock": book["in_stock"],
            }
        )
    return result
''',
        function="in_stock_books",
        instruction="修改函数：只保留 in_stock 为真的项，返回 isbn 和 title，保持原顺序，其他不变。",
        cond="in_stock",
        returns=("isbn", "title"),
        id_field="isbn",
        sample=(
            {"isbn": "A1", "title": "T1", "price": 10, "in_stock": True, "note": "x"},
            {"isbn": "A2", "title": "T2", "price": 20, "in_stock": False, "note": "y"},
            {"isbn": "A3", "title": "T3", "price": 30, "in_stock": True, "note": "z"},
        ),
    ),
    Task(
        name="verified_reviews",
        source='''"""评论列表：当前返回全部字段。"""


def verified_reviews(reviews):
    """返回评论，保持原顺序。"""
    result = []
    for review in reviews:
        result.append(
            {
                "review_id": review["review_id"],
                "author": review["author"],
                "rating": review["rating"],
                "verified": review["verified"],
            }
        )
    return result
''',
        function="verified_reviews",
        instruction="修改函数：只保留 verified 为真的项，返回 review_id 和 rating，保持原顺序，其他不变。",
        cond="verified",
        returns=("review_id", "rating"),
        id_field="review_id",
        sample=(
            {"review_id": 7, "author": "A", "rating": 5, "verified": True, "note": "x"},
            {"review_id": 8, "author": "B", "rating": 2, "verified": False, "note": "y"},
            {"review_id": 9, "author": "C", "rating": 4, "verified": True, "note": "z"},
            {"review_id": 10, "author": "D", "rating": 1, "verified": False, "note": "w"},
        ),
    ),
    Task(
        name="enabled_flags",
        source='''"""开关列表：当前返回全部字段。"""


def enabled_flags(flags):
    """返回开关，保持原顺序。"""
    result = []
    for flag in flags:
        result.append(
            {
                "key": flag["key"],
                "description": flag["description"],
                "enabled": flag["enabled"],
                "rollout": flag["rollout"],
            }
        )
    return result
''',
        function="enabled_flags",
        instruction="修改函数：只保留 enabled 为真的项，返回 key 和 description，保持原顺序，其他不变。",
        cond="enabled",
        returns=("key", "description"),
        id_field="key",
        sample=(
            {"key": "k1", "description": "d1", "enabled": True, "rollout": 10, "note": "x"},
            {"key": "k2", "description": "d2", "enabled": False, "rollout": 20, "note": "y"},
            {"key": "k3", "description": "d3", "enabled": True, "rollout": 30, "note": "z"},
        ),
    ),
    Task(
        name="valid_tickets",
        source='''"""工单列表：当前返回全部字段，且不过滤。"""


def valid_tickets(tickets):
    """返回工单，保持原顺序。"""
    if not tickets:
        return []
    result = []
    for ticket in tickets:
        result.append(
            {
                "ticket_id": ticket["ticket_id"],
                "priority": ticket["priority"],
                "valid": ticket["valid"],
                "assignee": ticket["assignee"],
            }
        )
    return result
''',
        function="valid_tickets",
        instruction="修改函数：只保留 valid 为真的项，返回 ticket_id、priority 和 assignee，保持原顺序，其他不变。",
        cond="valid",
        returns=("ticket_id", "priority", "assignee"),
        id_field="ticket_id",
        sample=(
            {"ticket_id": 11, "priority": "high", "valid": True, "assignee": "A", "note": "x"},
            {"ticket_id": 12, "priority": "low", "valid": False, "assignee": "B", "note": "y"},
            {"ticket_id": 13, "priority": "mid", "valid": True, "assignee": "C", "note": "z"},
        ),
    ),
    Task(
        name="ready_jobs",
        source='''"""作业列表：当前返回全部字段。"""


def ready_jobs(jobs):
    """返回作业，保持原顺序。"""
    result = []
    for job in jobs:
        result.append(
            {
                "job_id": job["job_id"],
                "queue": job["queue"],
                "ready": job["ready"],
                "attempts": job["attempts"],
            }
        )
    return result
''',
        function="ready_jobs",
        instruction="修改函数：只保留 ready 为真的项，返回 job_id 和 queue，保持原顺序，其他不变。",
        cond="ready",
        returns=("job_id", "queue"),
        id_field="job_id",
        sample=(
            {"job_id": 21, "queue": "q1", "ready": True, "attempts": 0, "note": "x"},
            {"job_id": 22, "queue": "q2", "ready": False, "attempts": 3, "note": "y"},
            {"job_id": 23, "queue": "q3", "ready": True, "attempts": 1, "note": "z"},
        ),
    ),
    Task(
        name="approved_expenses",
        source='''"""报销单列表：当前取字段的风格是 .get，且不过滤。"""


def approved_expenses(expenses):
    """返回报销单，保持原顺序。"""
    result = []
    for expense in expenses:
        result.append(
            {
                "expense_id": expense.get("expense_id"),
                "amount": expense.get("amount"),
                "approved": expense.get("approved"),
                "category": expense.get("category"),
            }
        )
    return result
''',
        function="approved_expenses",
        instruction="修改函数：只保留 approved 为真的项，返回 expense_id 和 amount，保持原顺序，其他不变。",
        cond="approved",
        returns=("expense_id", "amount"),
        id_field="expense_id",
        sample=(
            {"expense_id": 31, "amount": 100, "approved": True, "category": "c1", "note": "x"},
            {"expense_id": 32, "amount": 200, "approved": False, "category": "c2", "note": "y"},
            {"expense_id": 33, "amount": 300, "approved": True, "category": "c3", "note": "z"},
        ),
    ),
)


# 前缀共享实验用：一个源文件里放四个函数，四个问题问的是同一个文件的不同函数。
SHARED_SOURCE = '''"""账务流水模块：下面是四张流水表各自的处理函数。

模块说明（本模块的历史与约定，供改动时参考）：
- 这个模块从 2023 年起由账务组维护，每次只改一个函数，改动必须是行为等价的
  最小改动：过滤条件、返回字段是本模块唯一允许变化的两个地方。
- 所有函数都返回"列表套字典"，列表顺序与输入顺序一致；金额一律是整数分。
- 每个函数都不做过滤（把输入原样投影到输出），过滤由调用方在 SQL 里做；
  这一层保留下来是为了让上游能在内存里做二次筛选。
- 早期的版本直接返回 ORM 对象，改成字典是为了避免会话关闭后取属性报错。
- 每个字典的键就是下游接口的字段名，改名等于破坏接口，要发版说明。
- 单元测试用固定的三条样例数据，只断言字段名和条数，不断言内容。
- 这个文件里的函数签名不允许改：唯一的参数就是对应的列表。
"""


def billed_invoices(invoices):
    """返回发票，保持原顺序。"""
    if not invoices:
        return []
    result = []
    for invoice in invoices:
        result.append(
            {
                "invoice_id": invoice["invoice_id"],
                "amount": invoice["amount"],
                "billed": invoice["billed"],
                "currency": invoice["currency"],
            }
        )
    return result


def settled_payments(payments):
    """返回付款记录，保持原顺序。"""
    if not payments:
        return []
    result = []
    for payment in payments:
        result.append(
            {
                "payment_id": payment["payment_id"],
                "amount": payment["amount"],
                "settled": payment["settled"],
                "method": payment["method"],
            }
        )
    return result


def matched_ledger(entries):
    """返回流水条目，保持原顺序。"""
    result = []
    for entry in entries:
        result.append(
            {
                "entry_id": entry["entry_id"],
                "account": entry["account"],
                "matched": entry["matched"],
                "delta": entry["delta"],
            }
        )
    return result


def confirmed_refunds(refunds):
    """返回退款单，保持原顺序。"""
    result = []
    for refund in refunds:
        result.append(
            {
                "refund_id": refund["refund_id"],
                "amount": refund["amount"],
                "confirmed": refund["confirmed"],
                "reason": refund["reason"],
            }
        )
    return result
'''

SHARED_TASKS: tuple[Task, ...] = (
    Task(
        name="billed_invoices",
        source=SHARED_SOURCE,
        function="billed_invoices",
        instruction="修改函数：只保留 billed 为真的项，返回 invoice_id 和 amount，保持原顺序，其他不变。",
        cond="billed",
        returns=("invoice_id", "amount"),
        id_field="invoice_id",
        sample=(
            {"invoice_id": 41, "amount": 100, "billed": True, "currency": "CNY", "note": "x"},
            {"invoice_id": 42, "amount": 200, "billed": False, "currency": "USD", "note": "y"},
            {"invoice_id": 43, "amount": 300, "billed": True, "currency": "CNY", "note": "z"},
        ),
    ),
    Task(
        name="settled_payments",
        source=SHARED_SOURCE,
        function="settled_payments",
        instruction="修改函数：只保留 settled 为真的项，返回 payment_id 和 amount，保持原顺序，其他不变。",
        cond="settled",
        returns=("payment_id", "amount"),
        id_field="payment_id",
        sample=(
            {"payment_id": 51, "amount": 10, "settled": True, "method": "card", "note": "x"},
            {"payment_id": 52, "amount": 20, "settled": False, "method": "wire", "note": "y"},
            {"payment_id": 53, "amount": 30, "settled": True, "method": "card", "note": "z"},
        ),
    ),
    Task(
        name="matched_ledger",
        source=SHARED_SOURCE,
        function="matched_ledger",
        instruction="修改函数：只保留 matched 为真的项，返回 entry_id 和 account，保持原顺序，其他不变。",
        cond="matched",
        returns=("entry_id", "account"),
        id_field="entry_id",
        sample=(
            {"entry_id": 61, "account": "a1", "matched": True, "delta": 5, "note": "x"},
            {"entry_id": 62, "account": "a2", "matched": False, "delta": 6, "note": "y"},
            {"entry_id": 63, "account": "a3", "matched": True, "delta": 7, "note": "z"},
        ),
    ),
    Task(
        name="confirmed_refunds",
        source=SHARED_SOURCE,
        function="confirmed_refunds",
        instruction="修改函数：只保留 confirmed 为真的项，返回 refund_id 和 amount，保持原顺序，其他不变。",
        cond="confirmed",
        returns=("refund_id", "amount"),
        id_field="refund_id",
        sample=(
            {"refund_id": 71, "amount": 11, "confirmed": True, "reason": "r1", "note": "x"},
            {"refund_id": 72, "amount": 12, "confirmed": False, "reason": "r2", "note": "y"},
            {"refund_id": 73, "amount": 13, "confirmed": True, "reason": "r3", "note": "z"},
        ),
    ),
)


# --------------------------------------------------------------------------
# 请求与答案
# --------------------------------------------------------------------------


@dataclass
class Request:
    """一次决策请求：消息、token id、期望决策，以及来源任务。"""

    task: Task
    messages: list[dict[str, str]]
    prompt_ids: list[int]
    candidates: object  # chooseonly.decide.Candidates
    expected: object  # chooseonly.decide.Decision


@dataclass
class Answer:
    """一条答案：正文、生成 token 数、耗时。"""

    text: str
    tokens: list[int]
    seconds: float
    prompt_tokens: int


@dataclass
class BatchOutcome:
    """一次批处理的实测：答案、分段耗时、token 数、内存。"""

    answers: list[Answer]
    prefill_seconds: float
    decode_seconds: float
    wall_seconds: float
    output_tokens: int
    prompt_tokens: int
    peak_memory_gb: float
    decode_forwards: int = 0  # 解码阶段真正跑的前向次数（每步 1 次，整批一起）
    step_seconds: list[float] = field(default_factory=list)

    @property
    def decode_tokens(self) -> int:
        """首 token 由 prefill 那一次前向产出；EOS 那一步的前向不在这个计数里。"""
        return sum(max(len(a.tokens) - 1, 0) for a in self.answers)

    @property
    def decode_tokens_per_forward(self) -> float:
        """每次解码前向平均产出几个 token（批越大越高，上限是批大小）。"""
        return self.output_tokens / self.decode_forwards if self.decode_forwards else 0.0


@dataclass
class Quality:
    """一条答案的运行时判定。"""

    ok: bool
    reasons: tuple[str, ...]
    field_order_ok: bool

    def line(self) -> str:
        if self.ok:
            return "通过" if self.field_order_ok else "通过（字段顺序不同）"
        return "不通过：" + "；".join(self.reasons)


def encode_prompt(tokenizer, messages: list[dict[str, str]]) -> list[int]:
    """与 chooseonly.model.MLXEngine.generate 里的编码规则逐字一致（含 add_special_tokens）。"""
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    add_special_tokens = tokenizer.bos_token is None or not prompt.startswith(
        tokenizer.bos_token
    )
    return tokenizer.encode(prompt, add_special_tokens=add_special_tokens)


def build_requests(decide, tasks, tokenizer, shared_context: bool) -> list[Request]:
    """把固定任务变成请求；shared_context=True 时把源文件正文放进共享前缀。"""
    requests: list[Request] = []
    for task in tasks:
        candidates = decide.extract(task.source, task.function)
        messages = decide.build_decision_prompt(task.instruction, candidates)
        if shared_context:
            shared = f"上下文（同一个源文件，可能包含多个函数）：\n\n{task.source}\n\n"
            messages = [
                messages[0],
                {"role": "user", "content": shared + messages[1]["content"]},
            ]
        expected = decide.Decision(
            function_id=decide.FUNCTION_ID,
            filter_field=next(c.id for c in candidates.conditions if c.name == task.cond),
            return_fields=tuple(
                next(c.id for c in candidates.fields if c.name == name)
                for name in task.returns
            ),
        )
        requests.append(
            Request(
                task=task,
                messages=messages,
                prompt_ids=encode_prompt(tokenizer, messages),
                candidates=candidates,
                expected=expected,
            )
        )
    return requests


def expected_rows(task: Task) -> list[dict]:
    """宿主算好的期望结果：只留条件为真的行、只留要求返回的字段，有排序就跟着排。"""
    rows = [
        {name: row[name] for name in task.returns}
        for row in task.sample
        if row[task.cond]
    ]
    if task.sort_key:
        rows.sort(key=lambda r: -r[task.sort_key])
    return rows


def _run_body(task: Task, body: str) -> tuple[list[str], bool]:
    """把组装后的正文真的跑一遍：返回（问题列表，字段顺序是否与期望一致）。

    判据只有运行时行为：过滤结果等于期望、每个字典的键恰好是要求的字段、值也对；
    另外空输入必须仍然返回 []（守卫子句没被弄丢）。
    """
    reasons: list[str] = []
    try:
        tree = ast.parse(body)
    except SyntaxError as exc:
        return [f"语法错误: {exc.msg}"], False
    namespace: dict[str, object] = {}
    try:
        exec(compile(tree, "<candidate>", "exec"), namespace)  # noqa: S102 - 只跑实验用样本
    except Exception as exc:  # noqa: BLE001 - 记录任何执行失败
        return [f"无法执行: {type(exc).__name__}: {exc}"], False
    fn = namespace.get(task.function)
    if not callable(fn):
        return [f"找不到可调用函数 {task.function}"], False

    sample = [dict(row) for row in task.sample]
    try:
        out = fn(sample)
        empty = fn([])
    except Exception as exc:  # noqa: BLE001
        return [f"调用失败: {type(exc).__name__}: {exc}"], False

    if not isinstance(out, list):
        return [f"返回值不是列表: {out!r}"], False
    if not all(isinstance(item, dict) for item in out):
        return [f"元素不全是字典: {out!r}"], False
    want = expected_rows(task)
    if out != want:
        reasons.append(f"结果不正确: {out!r} 期望 {want!r}")
    if empty != []:
        reasons.append(f"空输入没有返回 []: {empty!r}")
    order_ok = (
        all(
            list(item.keys()) == list(row.keys())
            for item, row in zip(out, want)
        )
        if len(out) == len(want)
        else False
    )
    return reasons, order_ok


def check_quality(decide, request: Request, text: str) -> Quality:
    """项目现有的判断口径：解析决策 → 确定性组装 → 真的 exec、真的调用。

    模型回的是决策 JSON（候选 id），代码由 chooseonly.decide.assemble 落成；所以
    "答案正确"分三步查：
    1. 决策合法，且挑中的条件/字段按真实字段名就是指令要求的那些；
    2. 组装出来的正文能 exec、能找到目标函数（assemble 自己找不到安全落点会抛）；
    3. 真的喂样本调用一次，核对过滤结果与返回字段（_run_body）。
    只看代码像不像不算数。active_users 这一条额外过一遍项目自带的
    bench.compare.check_runtime_behaviour——它写死了那一个任务的样本，
    这里复用一次，等于用两套实现核同一条正文。
    """
    candidates = request.candidates
    expected = request.expected
    if not text.strip():
        return Quality(False, ("没有产出正文",), False)
    try:
        decision = decide.parse_decision(text, candidates)
    except Exception as exc:  # noqa: BLE001 - DecisionError 之外也不该让整轮崩掉
        return Quality(False, (f"决策不合法: {type(exc).__name__}: {exc}",), False)

    reasons: list[str] = []
    cond_names = {c.id: c.name for c in candidates.conditions}
    field_names = {c.id: c.name for c in candidates.fields}
    got_cond = cond_names.get(decision.filter_field) if decision.filter_field else None
    want_cond = cond_names.get(expected.filter_field) if expected.filter_field else None
    if got_cond != want_cond:
        reasons.append(f"过滤字段不是要求的那一个: {got_cond!r} 期望 {want_cond!r}")
    got_fields = [field_names.get(fid, fid) for fid in decision.return_fields]
    want_fields = [field_names.get(fid, fid) for fid in expected.return_fields]
    # 按集合比（与 bench/compare.py 的口径一致）；选出来的字段顺序由 _run_body 的
    # order_ok 单独报告，不算失败。
    if sorted(got_fields) != sorted(want_fields):
        reasons.append(f"返回字段不是要求的那几个: {got_fields} 期望 {want_fields}")

    try:
        body = decide.assemble(request.task.source, candidates, decision)
    except Exception as exc:  # noqa: BLE001
        reasons.append(f"组装失败: {type(exc).__name__}: {exc}")
        return Quality(False, tuple(reasons), False)

    runtime_reasons, order_ok = _run_body(request.task, body)
    reasons.extend(runtime_reasons)

    if request.task.name == COMPARE_TASK:
        try:
            from bench.compare import check_runtime_behaviour

            ok, why = check_runtime_behaviour(body, request.task.function)
            if not ok:
                reasons.append("项目对照检查未通过: " + "；".join(why))
        except Exception as exc:  # noqa: BLE001
            reasons.append(f"项目对照检查抛异常: {type(exc).__name__}: {exc}")
    return Quality(not reasons, tuple(reasons), order_ok)


# --------------------------------------------------------------------------
# 顺序路径（现有生产路径）
# --------------------------------------------------------------------------


def run_sequential(engine, requests: list[Request], max_tokens: int) -> list[Answer]:
    """现有生产路径：一次一个请求，逐次计时。"""
    answers: list[Answer] = []
    for request in requests:
        start = time.perf_counter()
        text, stats = engine.generate(request.messages, max_tokens=max_tokens)
        seconds = time.perf_counter() - start
        answers.append(
            Answer(
                text=text,
                tokens=[0] * stats.generated_tokens,  # 生产路径不暴露 token id
                seconds=seconds,
                prompt_tokens=stats.prompt_tokens,
            )
        )
    return answers


def calibration_fixed_cost(engine, request: Request, trials: int) -> tuple[float, float]:
    """只生成 1 个 token：量出与输出长度无关的那一段（调用开销＋prefill＋收尾）。"""
    samples = []
    for _ in range(trials):
        start = time.perf_counter()
        engine.generate(request.messages, max_tokens=1)
        samples.append(time.perf_counter() - start)
    return min(samples), sum(samples) / len(samples)


# --------------------------------------------------------------------------
# 批处理路径
# --------------------------------------------------------------------------


def _last_logits(model, tokens, cache):
    """跑一次前向，只取最后一个位置的 logits（省掉 (B, L, V) 那一大块）。

    生产路径走 model(...) 会算全部位置的 logits；批量路径里 B×L×V 在 1.5B 上
    就是几百 MB 的瞬时分配，16 GB 机器上没必要。transformer 主体逐字相同，
    只有输出头少算了前面的位置——这一点在报告里如实写明。

    Qwen2.5-1.5B 的 config 里 tie_word_embeddings=true，加载出来的 Model 因此
    没有 lm_head，模型自己的 __call__ 用的是 embed_tokens.as_linear；这里按同样
    的算子只算末位（行与行独立，结果与算全部位置再切片一致）。
    """
    body = getattr(model, "model", None)
    if body is None:
        return model(tokens, cache=cache)[:, -1, :]
    head = getattr(model, "lm_head", None)
    if head is None:
        head = getattr(getattr(body, "embed_tokens", None), "as_linear", None)
    if head is None:
        return model(tokens, cache=cache)[:, -1, :]  # 兜底：老实算全部位置
    hidden = body(tokens, cache=cache)
    return head(hidden[:, -1, :])


def _clear_mlx_cache() -> None:
    """把 MLX 的空闲缓存还给系统；16 GB 机器上其他 agent 也在跑，用完就还。"""
    for name in ("clear_cache",):
        fn = getattr(__import__("mlx.core", fromlist=["x"]), name, None)
        if callable(fn):
            try:
                fn()
            except Exception:  # noqa: BLE001 - 还缓存失败不影响测量
                pass
            return


def run_batched(engine, tokenizer, requests: list[Request], max_tokens: int) -> BatchOutcome:
    """批处理：所有请求左侧填充后放进同一次前向，之后每步一次前向。"""
    import mlx.core as mx
    from chooseonly.model import clean_body
    from mlx_lm.generate import generation_stream, wired_limit
    from mlx_lm.models.cache import BatchKVCache
    from mlx_lm.sample_utils import make_sampler

    model = engine._model  # noqa: SLF001 - 加载一次，两种路径共用同一份权重
    width = max(len(r.prompt_ids) for r in requests)
    pads = [width - len(r.prompt_ids) for r in requests]
    padded = mx.array(
        [[PAD_ID] * pad + ids for pad, ids in zip(pads, [r.prompt_ids for r in requests])]
    )
    cache = [BatchKVCache(pads) for _ in range(len(model.layers))]
    eos_ids = set(tokenizer.eos_token_ids)
    sampler = make_sampler(temp=0.0)

    produced: list[list[int]] = [[] for _ in requests]
    done = [False] * len(requests)
    step_seconds: list[float] = []

    mx.reset_peak_memory()
    with wired_limit(model, [generation_stream]), mx.stream(generation_stream):
        start = time.perf_counter()
        logits = _last_logits(model, padded, cache)
        mx.eval(logits)  # 强制 prefill 在计时区间内真的算完
        prefill_seconds = time.perf_counter() - start

        decode_start = time.perf_counter()
        steps = 0
        step_start = time.perf_counter()
        while True:
            tokens = sampler(logits)
            mx.eval(tokens)
            values = [int(t) for t in tokens.tolist()]
            for index, token in enumerate(values):
                if done[index]:
                    continue
                if token in eos_ids:
                    done[index] = True
                else:
                    produced[index].append(token)
                    if len(produced[index]) >= max_tokens:
                        done[index] = True
            steps += 1
            if all(done):
                break
            # 每条序列各喂自己上一个 token；已结束的序列喂它最后那个 token，
            # 输出丢弃。各自的 KV 段互不影响，不会污染未结束的序列。
            inputs = mx.array(
                [
                    [produced[i][-1] if produced[i] else PAD_ID]
                    for i in range(len(requests))
                ]
            )  # 形状 (N, 1)：每条序列一个 token
            logits = _last_logits(model, inputs, cache)
            mx.eval(logits)
            now = time.perf_counter()
            step_seconds.append(now - step_start)  # 一步解码 = 采样 + 下一次前向
            step_start = now
        decode_seconds = time.perf_counter() - decode_start
        wall_seconds = time.perf_counter() - start
        peak_gb = mx.get_peak_memory() / 1e9

    # 与生产路径用同一个归一化（MLXEngine.generate 最后也是 clean_body(decode(tokens))），
    # 逐字比较才有意义：两边差一个字面量空白就等于不一致。
    text_parts = [clean_body(tokenizer.decode(tokens)) for tokens in produced]
    answers = [
        Answer(
            text=text,
            tokens=tokens,
            seconds=wall_seconds,
            prompt_tokens=len(r.prompt_ids),
        )
        for text, tokens, r in zip(text_parts, produced, requests)
    ]
    outcome = BatchOutcome(
        answers=answers,
        prefill_seconds=prefill_seconds,
        decode_seconds=decode_seconds,
        wall_seconds=wall_seconds,
        output_tokens=sum(len(t) for t in produced),
        prompt_tokens=sum(len(r.prompt_ids) for r in requests),
        peak_memory_gb=peak_gb,
        decode_forwards=max(steps - 1, 0),
        step_seconds=step_seconds,
    )
    _clear_mlx_cache()
    return outcome


def longest_common_prefix(lists: list[list[int]]) -> int:
    """所有 token 序列的最长公共前缀长度（这就是"共享前缀"的 token 数）。"""
    if not lists:
        return 0
    shortest = min(len(item) for item in lists)
    first = lists[0]
    for index in range(shortest):
        if any(item[index] != first[index] for item in lists[1:]):
            return index
    return shortest


def run_shared_prefix(
    engine,
    tokenizer,
    requests: list[Request],
    max_tokens: int,
    prefix_len: int,
) -> tuple[BatchOutcome, int, list[int]]:
    """共享前缀：前缀只 prefill 一次，KV 复制给 N 条序列，后缀右侧填充后一次前向。

    返回 (批处理结果, 前缀 prefill 秒数, 用错位置读出来的首 token 列表)。
    "用错位置"是反证：后缀长度不一时，如果在全局最后一个位置（右侧填充的末尾）
    统一读，会读到别的序列的位置——这些 token 与正确答案不同的条数就是
    "必须逐序列在自己的末位读"的证据。
    """
    import mlx.core as mx
    from chooseonly.model import clean_body
    from mlx_lm.generate import generation_stream, wired_limit
    from mlx_lm.models.cache import BatchKVCache, KVCache
    from mlx_lm.sample_utils import make_sampler

    model = engine._model  # noqa: SLF001
    count = len(requests)
    layers = len(model.layers)
    prefix_ids = requests[0].prompt_ids[:prefix_len]
    suffixes = [r.prompt_ids[prefix_len:] for r in requests]
    suffix_width = max(len(s) for s in suffixes)
    right_pads = [suffix_width - len(s) for s in suffixes]
    sampler = make_sampler(temp=0.0)
    eos_ids = set(tokenizer.eos_token_ids)

    produced: list[list[int]] = [[] for _ in requests]
    done = [False] * count
    step_seconds: list[float] = []

    mx.reset_peak_memory()
    with wired_limit(model, [generation_stream]), mx.stream(generation_stream):
        # 1) 前缀只 prefill 一次（batch 1），再复制成 N 条序列的 KV。
        prefix_cache = [KVCache() for _ in range(layers)]
        start = time.perf_counter()
        # 与批量路径同一口径：prefill 只算末位 logits（这里连 logits 都不用，
        # 但保持同一种前向形状，前缀那一趟的耗时才和别处可比）。
        prefix_logits = _last_logits(model, mx.array(prefix_ids)[None], prefix_cache)
        mx.eval(prefix_logits)
        prefix_seconds = time.perf_counter() - start
        del prefix_logits

        batch_cache = []
        for layer_cache in prefix_cache:
            keys, values = layer_cache.state
            batch = BatchKVCache([0] * count)
            batch.state = (
                mx.repeat(keys, count, axis=0),
                mx.repeat(values, count, axis=0),
                mx.array([prefix_len] * count),
                mx.array([0] * count),
            )
            batch_cache.append(batch)
        del prefix_cache

        # 2) 后缀右侧填充后一次前向；每条序列在自己的末位读第一个 token。
        start = time.perf_counter()
        suffix_array = mx.array(
            [s + [PAD_ID] * pad for s, pad in zip(suffixes, right_pads)]
        )
        for layer_cache in batch_cache:
            layer_cache.prepare(right_padding=right_pads)
        logits = model(suffix_array, cache=batch_cache)  # (N, suffix_width, V)
        own_index = mx.array([len(s) - 1 for s in suffixes])
        first = mx.take_along_axis(
            logits, own_index[:, None, None], axis=1
        )[:, 0, :]
        wrong_first = mx.argmax(logits[:, -1, :], axis=-1)
        mx.eval(first, wrong_first)
        prefill_seconds = time.perf_counter() - start
        wrong_tokens = [int(t) for t in wrong_first.tolist()]
        del logits

        # 尾部填充滚到前面（BatchKVCache.finalize），之后的解码走标准左填 mask。
        for layer_cache in batch_cache:
            layer_cache.finalize()

        logits = first
        decode_start = time.perf_counter()
        steps = 0
        step_start = time.perf_counter()
        while True:
            tokens = sampler(logits)
            mx.eval(tokens)
            for index, token in enumerate(int(t) for t in tokens.tolist()):
                if done[index]:
                    continue
                if token in eos_ids:
                    done[index] = True
                else:
                    produced[index].append(token)
                    if len(produced[index]) >= max_tokens:
                        done[index] = True
            steps += 1
            if all(done):
                break
            inputs = mx.array(
                [[produced[i][-1] if produced[i] else PAD_ID] for i in range(count)]
            )  # 形状 (N, 1)
            logits = _last_logits(model, inputs, batch_cache)
            mx.eval(logits)
            now = time.perf_counter()
            step_seconds.append(now - step_start)
            step_start = now
        decode_seconds = time.perf_counter() - decode_start
        wall_seconds = time.perf_counter() - start
        peak_gb = mx.get_peak_memory() / 1e9

    answers = [
        Answer(
            text=clean_body(tokenizer.decode(tokens)),
            tokens=tokens,
            seconds=wall_seconds,
            prompt_tokens=len(r.prompt_ids),
        )
        for tokens, r in zip(produced, requests)
    ]
    # 注意：这里的 wall_seconds 只覆盖"后缀一次前向 + 解码"，不含前缀那一次 prefill；
    # 调用方要把返回的 prefix_seconds 加上去（两处分开计时，谁也不能重复计）。
    outcome = BatchOutcome(
        answers=answers,
        prefill_seconds=prefill_seconds,
        decode_seconds=decode_seconds,
        wall_seconds=wall_seconds,
        output_tokens=sum(len(t) for t in produced),
        prompt_tokens=sum(len(r.prompt_ids) for r in requests),
        peak_memory_gb=peak_gb,
        decode_forwards=max(steps - 1, 0),
        step_seconds=step_seconds,
    )
    _clear_mlx_cache()
    return outcome, prefix_seconds, wrong_tokens


# --------------------------------------------------------------------------
# 统计与打印
# --------------------------------------------------------------------------


def spread(values: list[float]) -> tuple[float, float, float]:
    return min(values), sum(values) / len(values), max(values)


def print_table(headers: list[str], rows: list[list[str]]) -> None:
    widths = [len(h) for h in headers]
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))
    print("| " + " | ".join(h.ljust(w) for h, w in zip(headers, widths)) + " |")
    print("|" + "|".join("-" * (w + 2) for w in widths) + "|")
    for row in rows:
        print("| " + " | ".join(c.ljust(w) for c, w in zip(row, widths)) + " |")


def fmt3(value: float) -> str:
    return f"{value:.3f}"


def fmt2(value: float) -> str:
    return f"{value:.2f}"


def load_note() -> str:
    try:
        one, five, fifteen = os.getloadavg()
    except OSError:
        return "loadavg 不可用"
    return f"loadavg 1/5/15 分钟：{one:.2f}/{five:.2f}/{fifteen:.2f}"


def rss_mb() -> float:
    """本进程最大 RSS（macOS 上 ru_maxrss 单位是字节）。"""
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value / 1e6 if sys.platform == "darwin" else value / 1e3


def import_project_module(name: str, attempts: int = 4):
    """并发改动期间 import 可能正好撞上半个文件；重试几次，不做多余绕行。"""
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            return importlib.import_module(name)
        except Exception as exc:  # noqa: BLE001 - 语法错、ImportError 都在这
            last = exc
            time.sleep(1.5 * (attempt + 1))
    raise ImportError(f"导入 {name} 失败（并发改动中？）：{last!r}")


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="批处理 vs 顺序调用：N 份独立工作的真实成本")
    parser.add_argument("--rounds", type=int, default=3, help="每种批大小重复几轮（取 min 与均值）")
    parser.add_argument("--batches", default="1,2,4,8", help="要测的批大小，逗号分隔")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--model", default=None, help="模型目录（默认用 chooseonly.model.DEFAULT_MODEL）")
    parser.add_argument("--shared-batch", type=int, default=4, help="前缀共享实验的批大小")
    parser.add_argument("--skip-prefix", action="store_true", help="跳过前缀共享实验")
    parser.add_argument("--selfcheck", action="store_true", help="只跑正确性自检（小批、少轮）")
    args = parser.parse_args(argv)

    decide = import_project_module("chooseonly.decide")
    model_mod = import_project_module("chooseonly.model")

    batch_sizes = [int(x) for x in args.batches.split(",") if x.strip()]
    rounds = max(args.rounds, 1)
    if args.selfcheck:
        batch_sizes = [2, 3]
        rounds = 1

    model_path = args.model or model_mod.DEFAULT_MODEL
    print(f"模型：{model_path}")
    print(f"任务：{len(TASKS)} 份独立工作（不同源文件、不同函数、不同字段）")
    engine = model_mod.MLXEngine(model_path)
    print("加载模型（只加载一次，两种路径共用同一份权重）...")
    engine._ensure_loaded()  # noqa: SLF001 - 常驻之后才开始计时
    print(f"加载耗时 {fmt2(engine.load_seconds)}s（不计入任何稳态数字）")
    tokenizer = engine._tokenizer  # noqa: SLF001
    print(f"开始前：{load_note()}\n")

    requests = build_requests(decide, TASKS, tokenizer, shared_context=False)
    print("请求（顺序路径与批处理路径用的是同一批 token）：")
    for request in requests:
        print(
            f"  {request.task.name:20s} 提示 {len(request.prompt_ids):3d} token"
            f"｜期望 {request.expected}"
        )
    print()

    # 预热：MLX 编译图、KV 缓存分配先发生一次，两次都丢弃（不计入任何数字）。
    warm_seq = run_sequential(engine, requests[:1], args.max_tokens)
    warm_bat = run_batched(engine, tokenizer, requests[:2], args.max_tokens)
    print(
        f"预热已丢弃：顺序 {fmt3(warm_seq[0].seconds)}s，"
        f"批处理(2) 墙钟 {fmt3(warm_bat.wall_seconds)}s"
        f"（其中 prefill {fmt3(warm_bat.prefill_seconds)}s / 解码 {fmt3(warm_bat.decode_seconds)}s）\n"
    )

    fixed_min, fixed_mean = calibration_fixed_cost(engine, requests[0], CALIBRATION_TRIALS)
    print(
        f"生产路径固定成本（只生成 1 个 token，{CALIBRATION_TRIALS} 次）："
        f"最小 {fmt3(fixed_min)}s，均值 {fmt3(fixed_mean)}s\n"
    )

    # ---------- 主实验：N vs 顺序 / 批处理 ----------
    print(f"=== 主实验：批大小 {batch_sizes}，每档 {rounds} 轮（轮内顺序与批处理交替）===")
    seq_records: dict[int, list[tuple[float, list[Answer]]]] = {n: [] for n in batch_sizes}
    bat_records: dict[int, list[BatchOutcome]] = {n: [] for n in batch_sizes}
    for round_index in range(rounds):
        for size in batch_sizes:
            if size > len(requests):
                continue
            subset = requests[:size]
            seq_start = time.perf_counter()
            seq_answers = run_sequential(engine, subset, args.max_tokens)
            seq_total = time.perf_counter() - seq_start
            seq_records[size].append((seq_total, seq_answers))
            bat_records[size].append(
                run_batched(engine, tokenizer, subset, args.max_tokens)
            )
        print(f"  第 {round_index + 1}/{rounds} 轮完成：{load_note()}")

    print("\n--- 表 1：墙钟时间（顺序 vs 批处理）---")
    rows = []
    for size in batch_sizes:
        if not seq_records[size]:
            continue
        seq_totals = [total for total, _ in seq_records[size]]
        bat_totals = [outcome.wall_seconds for outcome in bat_records[size]]
        seq_low, seq_mean, seq_high = spread(seq_totals)
        bat_low, bat_mean, bat_high = spread(bat_totals)
        per_call = sum(
            answer.seconds for _, answers in seq_records[size] for answer in answers
        ) / sum(len(answers) for _, answers in seq_records[size])
        rows.append(
            [
                str(size),
                fmt3(seq_mean),
                fmt3(seq_low),
                fmt3(seq_high),
                fmt3(bat_mean),
                fmt3(bat_low),
                fmt3(bat_high),
                f"{seq_mean / bat_mean:.2f}x" if bat_mean > 0 else "-",
                fmt3(per_call),
                fmt3(bat_mean / size),
            ]
        )
    print_table(
        [
            "N",
            "顺序总(s)均值",
            "顺序min",
            "顺序max",
            "批处理总(s)均值",
            "批处理min",
            "批处理max",
            "省下倍数",
            "顺序每次(s)",
            "批处理摊到每次(s)",
        ],
        rows,
    )
    print(
        "说明：批处理总时间 = 一次 prefill ＋ 逐 token 解码；顺序总时间 = N 次独立调用的墙钟之和。"
    )

    print("\n--- 表 2：吞吐（总输出 token / 总秒数）---")
    rows = []
    for size in batch_sizes:
        if not seq_records[size]:
            continue
        seq_seconds = sum(total for total, _ in seq_records[size])
        seq_tokens = sum(
            len(answer.tokens) for _, answers in seq_records[size] for answer in answers
        )
        seq_prompts = sum(
            answer.prompt_tokens for _, answers in seq_records[size] for answer in answers
        )
        bat_wall = sum(outcome.wall_seconds for outcome in bat_records[size])
        bat_tokens = sum(outcome.output_tokens for outcome in bat_records[size])
        bat_decode_tokens = sum(outcome.decode_tokens for outcome in bat_records[size])
        bat_decode_seconds = sum(outcome.decode_seconds for outcome in bat_records[size])
        bat_prefill_seconds = sum(outcome.prefill_seconds for outcome in bat_records[size])
        bat_prompts = sum(outcome.prompt_tokens for outcome in bat_records[size])
        bat_forwards = sum(outcome.decode_forwards for outcome in bat_records[size])
        proc_ms = (
            1000 * bat_decode_seconds / bat_forwards if bat_forwards else 0.0
        )
        rows.append(
            [
                str(size),
                f"{seq_tokens / seq_seconds:.1f}" if seq_seconds else "-",
                f"{bat_tokens / bat_wall:.1f}" if bat_wall else "-",
                f"{bat_decode_tokens / bat_decode_seconds:.1f}" if bat_decode_seconds else "-",
                fmt2(proc_ms),
                f"{bat_forwards / bat_decode_seconds:.1f}" if bat_decode_seconds else "-",
                f"{bat_prompts / bat_prefill_seconds:.0f}" if bat_prefill_seconds else "-",
                f"{bat_tokens / bat_wall / (seq_tokens / seq_seconds):.2f}x"
                if seq_seconds and seq_tokens and bat_wall
                else "-",
            ]
        )
    print_table(
        [
            "N",
            "顺序有效tok/s",
            "批处理有效tok/s",
            "批处理解码tok/s",
            "解码前向(ms)",
            "解码前向/s",
            "批处理prefill tok/s",
            "相对顺序",
        ],
        rows,
    )
    print(
        f"对照：生产路径固定成本 {fmt3(fixed_mean)}s/次（只生成 1 token）；"
        "顺序有效 tok/s 把 prefill 与固定成本都摊进了分母。"
    )

    print("\n--- 表 3：答案一致性与正确性（温度 0，逐条比对）---")
    rows = []
    mismatch_lines: list[str] = []
    for size in batch_sizes:
        if not seq_records[size]:
            continue
        same = 0
        total = 0
        seq_ok = 0
        bat_ok = 0
        tokens_same = 0
        for (_, seq_answers), outcome in zip(seq_records[size], bat_records[size]):
            for request, seq_answer, bat_answer in zip(
                requests[:size], seq_answers, outcome.answers
            ):
                total += 1
                seq_quality = check_quality(decide, request, seq_answer.text)
                bat_quality = check_quality(decide, request, bat_answer.text)
                seq_ok += int(seq_quality.ok)
                bat_ok += int(bat_quality.ok)
                if seq_answer.text == bat_answer.text:
                    same += 1
                elif len(mismatch_lines) < 12:
                    mismatch_lines.append(
                        f"  N={size} {request.task.name}：\n"
                        f"    顺序 {seq_answer.text.strip()[:120]!r}\n"
                        f"    批处理 {bat_answer.text.strip()[:120]!r}"
                    )
                if len(seq_answer.tokens) == len(bat_answer.tokens):
                    tokens_same += 1
        rows.append(
            [
                str(size),
                f"{same}/{total}",
                f"{tokens_same}/{total}",
                f"{seq_ok}/{total}",
                f"{bat_ok}/{total}",
                "是" if same == total else "否",
            ]
        )
    print_table(
        ["N", "正文逐字一致", "输出长度一致", "顺序正确", "批处理正确", "一致"], rows
    )
    if mismatch_lines:
        print("\n不一致明细（这就是 mask / 填充出错的证据）：")
        print("\n".join(mismatch_lines))
    print("\n逐条正确性明细（最后一轮）：")
    for size in batch_sizes:
        if not bat_records[size]:
            continue
        last_seq = seq_records[size][-1][1]
        last_bat = bat_records[size][-1]
        for request, seq_answer, bat_answer in zip(requests[:size], last_seq, last_bat.answers):
            seq_quality = check_quality(decide, request, seq_answer.text)
            bat_quality = check_quality(decide, request, bat_answer.text)
            print(
                f"  N={size} {request.task.name:20s}"
                f"｜顺序 {seq_quality.line()}｜批处理 {bat_quality.line()}"
                f"｜{len(bat_answer.tokens)} tok"
            )

    print("\n--- 表 4：内存（MLX 峰值；16 GB 机器，其他 agent 同时在跑）---")
    rows = []
    for size in batch_sizes:
        if not bat_records[size]:
            continue
        peaks = [outcome.peak_memory_gb for outcome in bat_records[size]]
        low, mean, high = spread(peaks)
        rows.append([str(size), f"{mean:.3f} GB", f"{low:.3f} GB", f"{high:.3f} GB"])
    print_table(["N", "MLX 峰值均值", "最小", "最大"], rows)
    print(f"本进程最大 RSS：{rss_mb():.0f} MB（含 Python、权重、系统分配）")

    # ---------- 前缀共享 ----------
    if not args.skip_prefix:
        size = min(args.shared_batch, len(SHARED_TASKS))
        shared_requests = build_requests(
            decide, SHARED_TASKS[:size], tokenizer, shared_context=True
        )
        prefix_len = longest_common_prefix([r.prompt_ids for r in shared_requests])
        total_lens = [len(r.prompt_ids) for r in shared_requests]
        print(f"\n=== 前缀共享实验（同一个源文件，{size} 个不同问题）===")
        print(
            f"完整提示 token：{total_lens}；最长公共前缀 {prefix_len} token"
            f"（占完整提示的 {100 * prefix_len / max(total_lens):.0f}%）；"
            f"后缀 token：{[length - prefix_len for length in total_lens]}"
        )
        run_sequential(engine, shared_requests[:1], args.max_tokens)  # 该形状的预热
        seq_totals, bat_totals, shared_totals = [], [], []
        prefix_seconds_samples = []
        shared_answers: list[Answer] = []
        wrong_reads: list[int] = []
        for _ in range(rounds):
            seq_start = time.perf_counter()
            seq_answers = run_sequential(engine, shared_requests, args.max_tokens)
            seq_totals.append(time.perf_counter() - seq_start)
            bat_totals.append(
                run_batched(engine, tokenizer, shared_requests, args.max_tokens).wall_seconds
            )
            outcome, prefix_seconds, wrong_tokens = run_shared_prefix(
                engine, tokenizer, shared_requests, args.max_tokens, prefix_len
            )
            # run_shared_prefix 的 wall_seconds 只含后缀那一次前向 + 解码；
            # 前缀那一次 prefill 单独计时，在这里加一次（只加一次）。
            shared_totals.append(outcome.wall_seconds + prefix_seconds)
            prefix_seconds_samples.append(prefix_seconds)
            shared_answers = outcome.answers
            wrong_reads = wrong_tokens
        seq_low, seq_mean, seq_high = spread(seq_totals)
        bat_low, bat_mean, bat_high = spread(bat_totals)
        shr_low, shr_mean, shr_high = spread(shared_totals)
        pre_low, pre_mean, _ = spread(prefix_seconds_samples)
        print_table(
            [
                "模式",
                "总墙钟均值(s)",
                "最小(s)",
                "最大(s)",
                "相对顺序",
                "说明",
            ],
            [
                [
                    "顺序（N 次全提示）",
                    fmt3(seq_mean),
                    fmt3(seq_low),
                    fmt3(seq_high),
                    "1.00x",
                    f"{size} 次独立调用，每次都重新 prefill 共享上下文",
                ],
                [
                    "批处理（全提示）",
                    fmt3(bat_mean),
                    fmt3(bat_low),
                    fmt3(bat_high),
                    f"{seq_mean / bat_mean:.2f}x",
                    f"一次前向，但 N 份提示里各含一份完整上下文",
                ],
                [
                    "批处理（共享前缀）",
                    fmt3(shr_mean),
                    fmt3(shr_low),
                    fmt3(shr_high),
                    f"{seq_mean / shr_mean:.2f}x",
                    f"前缀只 prefill 一次（{fmt3(pre_mean)}s）＋ 后缀一次前向",
                ],
            ],
        )
        print(
            f"前缀 prefill 一次实测：均值 {fmt3(pre_mean)}s，最小 {fmt3(pre_low)}s"
            f"（batch 1，{prefix_len} token）；"
            f"共享前缀 vs 全提示批处理：省下 "
            f"{(1 - shr_mean / bat_mean) * 100:.0f}%（{fmt3(bat_mean)}s → {fmt3(shr_mean)}s）"
        )
        print("共享前缀答案与顺序答案逐条比对：")
        last_seq = run_sequential(engine, shared_requests, args.max_tokens)
        same = sum(
            1
            for seq_answer, shared_answer in zip(last_seq, shared_answers)
            if seq_answer.text == shared_answer.text
        )
        for request, seq_answer, shared_answer in zip(shared_requests, last_seq, shared_answers):
            quality = check_quality(decide, request, shared_answer.text)
            print(
                f"  {request.task.name:20s}｜顺序 {seq_answer.text.strip()[:60]!r}"
                f"｜共享 {shared_answer.text.strip()[:60]!r}"
                f"｜{'一致' if seq_answer.text == shared_answer.text else '不一致'}"
                f"｜正确性 {quality.line()}"
            )
        print(f"共享前缀答案与顺序答案逐字一致：{same}/{len(last_seq)}")
        # 反证：后缀长度不一时，如果在右侧填充的全局末位统一读首 token，
        # 短的那几条读到的是别的序列末尾（或填充位）的预测。
        max_suffix = max(total_lens) - prefix_len
        comparable = [
            index
            for index, request in enumerate(shared_requests)
            if len(request.prompt_ids) - prefix_len < max_suffix
            and shared_answers[index].tokens
        ]
        differing = [
            index
            for index in comparable
            if wrong_reads[index] != shared_answers[index].tokens[0]
        ]
        print(
            f"反证：在「全局最后一个位置」统一读首 token，{len(differing)}/{len(comparable)} 条"
            f"与自己的正确首 token 不同（后缀长度 {[length - prefix_len for length in total_lens]}，"
            f"只有后缀较短的 {len(comparable)} 条可比较）。"
            "后缀长度不一时，答案必须逐序列在自己的末位读。"
        )

    print(f"\n结束前：{load_note()}")
    print(f"本进程最大 RSS：{rss_mb():.0f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
