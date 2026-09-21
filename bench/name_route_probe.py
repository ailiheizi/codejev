"""两种取名字的方式对照：枚举后选择 vs 提议后验证。

背景：原型（bench/general_domain_probe.py）查出的枚举空洞很具体——
推导式/生成器目标（还被 PEP 709 弄成解释器版本相关）、lambda 形参、
闭包单元、装饰器行、运行时才存在的名字（`__class__`）、属性/下标绑定。

用户提出的方向：枚举不全就"拆更细"，让模型直接产出名字本身。
本脚本把它落成一个可测的替代设计——

  A 枚举后选择：宿主枚举候选页 → 模型只回 id → 宿主物化（现有生产协议）
  B 提议后验证：模型直接产出名字 → 宿主用 symtable/AST 判定
               "这个名字在该作用域里可用吗" → 通过才采用

判据：能否覆盖 A 枚举不到的那些空洞；以及 A 能覆盖时 B 是否仍然对。
只用真实本地模型（Qwen1.5B），温度 0；两边都跑同样的任务。
"""

from __future__ import annotations

import ast
import symtable
import textwrap
import time
from pathlib import Path

from codejev.adapter import build_messages
from codejev.contracts import Action, Brief, Kind
from codejev.model import MLXEngine

MODEL = str(
    Path(__file__).resolve().parent.parent / "models" / "Qwen2.5-Coder-1.5B-Instruct-4bit"
)

# 这份源码故意包含原型列出的各种"枚举空洞"。
SOURCE = '''"""包含多种作用域形态的样本。"""

import os

THRESHOLD = 10


def summarize(rows):
    """主函数：含推导式、生成器、lambda、闭包与条件绑定。"""
    kept = [row for row in rows if row["score"] > THRESHOLD]
    names = (row["name"] for row in rows)
    picker = lambda item: item["score"]
    try:
        total = sum(picker(row) for row in kept)
    except TypeError as err:
        total = 0
    def inner():
        # 闭包：引用外层的 kept
        return len(kept)
    return {"kept": kept, "names": names, "total": total, "count": inner(), "err": err}


def scan(rows):
    """另一个函数，用来测目标函数选择。"""
    ordered = sorted(rows, key=lambda entry: entry["name"])
    return ordered
'''

# (任务, 期望名字, 该名字属于哪个作用域, 说明)
TASKS = (
    ("在主函数 summarize 里，哪个名字保存了过滤后的行？只回那个名字。", "kept", "summarize", "普通局部变量"),
    ("summarize 里哪个名字是生成器？只回那个名字。", "names", "summarize", "生成器表达式目标"),
    ("summarize 里哪个名字是那个 lambda 的形参？只回那个名字。", "item", "summarize", "lambda 形参（AST 枚举不到）"),
    ("summarize 里哪个名字是被内层函数闭包捕获的？只回那个名字。", "kept", "summarize", "闭包引用"),
    ("summarize 里哪个名字来自 except as？只回那个名字。", "err", "summarize", "条件绑定"),
    ("scan 里哪个名字是排序后返回的列表？只回那个名字。", "ordered", "scan", "普通局部变量"),
)


def names_in_scope(source: str, function: str) -> set[str]:
    """宿主侧的真实判定：该函数作用域里可用的名字（用 symtable，不靠 AST 近似）。"""
    table = symtable.symtable(source, "<src>", "exec")
    found: set[str] = set()

    def walk(node: symtable.SymbolTable) -> None:
        if node.get_name() == function and node.get_type() == "function":
            for symbol in node.get_symbols():
                if symbol.is_assigned() or symbol.is_parameter() or symbol.is_referenced():
                    found.add(symbol.get_name())
        for child in node.get_children():
            walk(child)

    walk(table)
    return found


def ask(engine, prompt: str, max_tokens: int = 24) -> tuple[str, int, float]:
    brief = Brief(instruction=prompt, target="x.py", action=Action.CREATE, kind=Kind.TEXT, context=SOURCE)
    started = time.perf_counter()
    text, stats = engine.generate(build_messages(brief), max_tokens=max_tokens)
    return text.strip(), stats.generated_tokens, time.perf_counter() - started


def main() -> int:
    engine = MLXEngine(MODEL)
    engine._ensure_loaded()
    print(f"模型：{MODEL}（本地，温度 0）")
    print()
    print("A = 枚举后选择（宿主给候选页，模型只回 id）")
    print("B = 提议后验证（模型直接产出名字，宿主用 symtable 判定是否可用）")
    print()
    print(f"{'任务':34s} {'期望':>8s} {'A':>6s} {'B':>6s}  {'B 是否通过宿主验证':>18s}")

    a_pass = b_pass = b_verified = 0
    for prompt, expected, scope, note in TASKS:
        scope_names = names_in_scope(SOURCE, scope)

        # A：把该作用域里能枚举到的名字做成候选页（AST 近似：只能看到绑定点）
        ast_names = [
            node.id
            for node in ast.walk(ast.parse(SOURCE))
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
        ]
        page = sorted({name for name in ast_names if name in scope_names})
        page_text = " ".join(f"{index}={name}" for index, name in enumerate(page))
        a_raw, _, _ = ask(
            engine,
            f"任务：{prompt}\n候选（只回编号）：{page_text}\n不过滤回 NONE。只回编号：",
        )
        a_name = ""
        if a_raw.strip().isdigit() and int(a_raw.strip()) < len(page):
            a_name = page[int(a_raw.strip())]
        a_ok = a_name == expected

        # B：直接产出名字，宿主验证
        b_raw, _, _ = ask(engine, f"任务：{prompt}\n只回一个名字，不要解释、不要引号：")
        b_name = b_raw.split()[0].strip("`\"'.,") if b_raw.split() else ""
        b_ok = b_name == expected
        verified = b_name in scope_names

        a_pass += int(a_ok)
        b_pass += int(b_ok)
        b_verified += int(verified)
        print(
            f"{note:34s} {expected:>8s} {'✓' if a_ok else '✗':>6s} "
            f"{'✓' if b_ok else '✗':>6s}  {'可用' if verified else '不可用':>18s}"
        )
        if not a_ok:
            print(f"{'':34s} A 得到 {a_name!r}（候选页 {len(page)} 个名字）")
        if not b_ok:
            print(f"{'':34s} B 得到 {b_name!r}")

    total = len(TASKS)
    print()
    print(f"A 枚举后选择：{a_pass}/{total}")
    print(f"B 提议后验证：{b_pass}/{total}（其中 {b_verified}/{total} 通过了宿主作用域验证）")
    print()
    print("读法：A 的上限是枚举器能列出多少名字；B 不需要枚举，但要靠宿主的验证器兜底。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
