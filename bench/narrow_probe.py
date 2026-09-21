"""免费变窄实测：同一模型、同一任务，只改"要求产出什么"。

用户提出的约束：拆得太细，每拆一次就多一次完整往返，成本会上去。
所以先测"免费变窄"——**不增加调用次数**，只让模型产出更小的片段：

  A 整函数：要求写出完整函数（现在的生产行为）
  B 只填函数体：给出 def 行与 docstring，只要求输出函数体
  C 只改片段：给出原文，只要求输出要替换的那一小段

三种协议的**调用次数都是 1 次**，区别只在要求的产出范围与宿主组装方式。
判据：运行时行为通过率、输出 token 数、单次耗时。温度 0，本地 Qwen1.5B。
"""

from __future__ import annotations

import ast
import textwrap
import time
from pathlib import Path

from codejev.adapter import build_messages, to_artifact
from codejev.contracts import Action, Brief, Kind
from codejev.model import MLXEngine

MODEL = str(
    Path(__file__).resolve().parent.parent / "models" / "Qwen2.5-Coder-1.5B-Instruct-4bit"
)

# ---- 任务 1：简单形状（现在的基线在这条上是 5/5）----
SIMPLE_SOURCE = '''"""用户列表：当前返回全部字段。"""


def active_users(users):
    """返回用户，保持原顺序。"""
    result = []
    for user in users:
        result.append(
            {"id": user["id"], "name": user["name"], "active": user["active"]}
        )
    return result
'''

# ---- 任务 2：真实形状（守卫 + 复合条件 + 循环后排序）----
# 基线实测：整函数生成 0/5，模型会把原文原样吐回，指令没有生效。
REAL_SOURCE = '''"""订单。"""


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

TASK1 = {
    "name": "简单形状",
    "source": SIMPLE_SOURCE,
    "function": "active_users",
    "instruction": "修改函数：只保留 active 为真的项，返回 id 和 name，保持原顺序，其他不变。",
    "sample": [
        {"id": 1, "name": "A", "active": True},
        {"id": 2, "name": "B", "active": False},
        {"id": 3, "name": "C", "active": True},
    ],
    "expect": [{"id": 1, "name": "A"}, {"id": 3, "name": "C"}],
}

TASK2 = {
    "name": "真实形状",
    "source": REAL_SOURCE,
    "function": "paid_orders",
    # 指令只说"其他不变"，所以原文那条降序排序应当保留：
    # 过滤 paid 后剩 A(5) 和 C(7)，再按 total 降序 → C(7), A(5)。
    "instruction": "修改函数：只保留 paid 为真的项，返回 order_id 和 total，其他不变。",
    "sample": [
        type("O", (), {"paid": True, "shipped": False, "order_id": "A", "total": 5})(),
        type("O", (), {"paid": False, "shipped": False, "order_id": "B", "total": 9})(),
        type("O", (), {"paid": True, "shipped": True, "order_id": "C", "total": 7})(),
    ],
    "expect": [
        {"order_id": "C", "total": 7},
        {"order_id": "A", "total": 5},
    ],
}


def check(source: str, function: str, sample: object, expect: object) -> tuple[bool, tuple[str, ...]]:
    """运行时验证：exec 后真的调用函数。"""
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return False, (f"语法错误: {exc.msg}",)
    namespace: dict[str, object] = {}
    try:
        exec(compile(tree, "<candidate>", "exec"), namespace)  # noqa: S102
    except Exception as exc:  # noqa: BLE001
        return False, (f"无法执行: {type(exc).__name__}",)
    fn = namespace.get(function)
    if not callable(fn):
        return False, (f"缺少函数 {function}",)
    try:
        got = fn(sample)
    except Exception as exc:  # noqa: BLE001
        return False, (f"调用失败: {type(exc).__name__}: {exc}",)
    if got != expect:
        return False, (f"结果不符: {got}",)
    return True, ()


def protocol_a(task: dict) -> str:
    """A：整函数。"""
    return task["instruction"] + "\n只输出修改后的完整函数正文，不要解释。"


def protocol_b(task: dict) -> str:
    """B：只填函数体。宿主负责补 def 行与缩进。"""
    tree = ast.parse(task["source"])
    fn = next(
        node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == task["function"]
    )
    header = task["source"].splitlines()[fn.lineno - 1]
    doc = ""
    if fn.body and isinstance(fn.body[0], ast.Expr) and isinstance(fn.body[0].value, ast.Constant):
        doc = "\n".join(task["source"].splitlines()[fn.body[0].lineno - 1 : (fn.body[0].end_lineno or 0)])
    return (
        f"{task['instruction']}\n"
        f"函数签名（不要重复输出这一行）：{header.strip()}\n"
        + (f"docstring（原样保留，不要重复输出）：{doc.strip()}\n" if doc else "")
        + "只输出函数体（每行四个空格缩进），不要输出 def 行、不要输出 docstring、不要解释。"
    )


def protocol_c(task: dict) -> str:
    """C：只改片段——只输出循环体里要替换的那两行。"""
    return (
        f"{task['instruction']}\n"
        "只输出循环体内部需要替换的语句（每行八个空格缩进），"
        "不要输出 def 行、不要输出循环行、不要输出 return、不要解释。"
    )


def rebuild(task: dict, protocol: str, body: str) -> str:
    """按协议把模型产出拼回完整文件。宿主做组装，模型不负责包装。"""
    source = task["source"]
    if protocol == "A":
        return body
    tree = ast.parse(source)
    fn = next(
        node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == task["function"]
    )
    lines = source.splitlines()
    header = lines[fn.lineno - 1]
    indent = "    "
    if protocol == "B":
        head = [header]
        if fn.body and isinstance(fn.body[0], ast.Expr):
            head.append(lines[fn.body[0].lineno - 1])
        new_body = head + [f"{indent}{line}" if line.strip() else "" for line in body.splitlines()]
        out = lines[: fn.lineno - 1] + new_body + lines[fn.end_lineno or len(lines) :]
        return "\n".join(out) + "\n"
    # C：只替换循环体
    loop = next(node for node in fn.body if isinstance(node, ast.For))
    first = loop.body[0].lineno - 1
    last = (loop.body[-1].end_lineno or loop.body[-1].lineno) - 1
    new_block = [f"{indent}{indent}{line}" if line.strip() else "" for line in body.splitlines()]
    out = lines[:first] + new_block + lines[last + 1 :]
    return "\n".join(out) + "\n"


def main() -> int:
    engine = MLXEngine(MODEL)
    engine._ensure_loaded()
    print(f"模型：{MODEL}")
    print(f"加载：{engine.load_seconds:.2f}s")
    print("三条协议调用次数都是 1 次；区别只在要求的产出范围与宿主组装方式。")

    trials = 5
    for task in (TASK1, TASK2):
        print()
        print("=" * 78)
        print(f"任务：{task['name']}")
        print("=" * 78)
        print(f"  {'协议':22s} {'通过':>6s} {'输出tok':>8s} {'耗时':>8s}  首次失败原因")
        for protocol, label, prompt_fn in (
            ("A", "A 整函数", protocol_a),
            ("B", "B 只填函数体", protocol_b),
            ("C", "C 只改片段", protocol_c),
        ):
            passed = 0
            tokens: list[int] = []
            seconds: list[float] = []
            first_fail = ""
            for index in range(trials):
                brief = Brief(
                    instruction=prompt_fn(task),
                    target="x.py",
                    action=Action.CREATE,
                    kind=Kind.CODE,
                    context=task["source"],
                )
                start = time.perf_counter()
                text, stats = engine.generate(build_messages(brief), max_tokens=400)
                elapsed = time.perf_counter() - start
                body = to_artifact(brief, text).body
                source = rebuild(task, protocol, body)
                ok, why = check(source, task["function"], task["sample"], task["expect"])
                passed += int(ok)
                tokens.append(stats.generated_tokens)
                seconds.append(elapsed)
                if not ok and not first_fail:
                    first_fail = "; ".join(why)[:60]
            avg_tok = sum(tokens) / len(tokens)
            avg_s = sum(seconds) / len(seconds)
            print(
                f"  {label:22s} {passed}/{trials:<4d} {avg_tok:>8.0f} {avg_s:>7.2f}s  {first_fail}"
            )
    print()
    print("读法：三种协议调用次数相同（都是 1 次），所以通过率提升属于「免费变窄」；")
    print("      输出 token 减少同时意味着更快。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
