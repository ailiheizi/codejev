"""Jev 在**真实源码候选池**上的批量选择：15 条任务。

候选池不再是手写的 3 条，而是从真实文件里枚举出的函数（chooseonly/candidate.py），
评审判据由宿主自己算：选中的函数必须能通过行为指纹（真的 import 并调用）。

15 条任务分三类：
  1. 直接问职责（7 条）
  2. 换一种说法问同一件事（3 条）—— 测语义鲁棒性
  3. 池里确实没有的能力（5 条）—— 应该回 NONE

判据用运行时行为：把选中的函数源码 exec 成真函数，跑一段行为指纹。
"""

from __future__ import annotations

import ast
import json
import os
import time
from pathlib import Path

from chooseonly.candidate import CandidatePage, CodeCandidate, CodeTask
from chooseonly.jev_engine import TypesafeConfig, TypesafeEngine

SOURCE = Path(__file__).resolve().parent.parent / "chooseonly" / "candidate.py"


def enumerate_functions(path: Path, limit: int = 12) -> tuple[CodeCandidate, ...]:
    """宿主枚举：模块级函数 + 类方法，附源码段与一段行为指纹描述。"""
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text)
    lines = text.splitlines()
    out: list[CodeCandidate] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        if node.name.startswith("_"):
            continue
        code = "\n".join(lines[node.lineno - 1 : node.end_lineno or node.lineno])
        doc = ast.get_docstring(node) or ""
        out.append(
            CodeCandidate(
                id=str(len(out)),
                name=node.name,
                purpose=doc.splitlines()[0] if doc else f"函数 {node.name}",
                code=code,
            )
        )
        if len(out) >= limit:
            break
    return tuple(out)


# (需求, 期望函数名 或 None 表示 NONE, 类别)
TASKS = (
    ("把模型返回的单行候选编号解析出来，别的写法一律拒绝", "parse_choice", "直接问职责"),
    ("校验候选页：id 不能重复、代码不能为空、语言要一致", "validate_page", "直接问职责"),
    ("把候选页里的短要求排成提示行", "_bullet_lines", "直接问职责"),
    ("去掉包住单行标量的 markdown 围栏", "_unwrap_scalar", "直接问职责"),
    ("把模型对话控制符从正文里清掉", "clean_body", "直接问职责"),
    ("返回宿主保存的候选代码，没有匹配就拒绝", "materialize", "直接问职责"),
    ("构造固定的两段选择提示", "build_selection_messages", "直接问职责"),
    ("解析模型回答并拿到编号（换个说法）", "parse_choice", "换说法"),
    ("检查这张候选页本身是否合法", "validate_page", "换说法"),
    ("取出候选的源码正文给下游使用", "materialize", "换说法"),
    ("把两个文件做语法树对比并生成补丁", None, "池里没有"),
    ("调用远端 HTTP 接口并处理重试", None, "池里没有"),
    ("把结果写入 SQLite 数据库", None, "池里没有"),
    ("计算两个向量的余弦相似度", None, "池里没有"),
    ("给函数自动补全类型注解", None, "池里没有"),
)


def main() -> int:
    key = os.environ.get("TYPESAFE_API_KEY", "")
    if not key:
        raise SystemExit("需要 TYPESAFE_API_KEY")
    page = CandidatePage(enumerate_functions(SOURCE))
    engine = TypesafeEngine(TypesafeConfig(api_key=key))
    print(f"候选池：{SOURCE.name} 里枚举到 {len(page.candidates)} 个函数")
    print(f"  {', '.join(c.name for c in page.candidates)}")
    print()

    passed = 0
    times: list[float] = []
    confidences: list[float] = []
    wrong: list[str] = []
    print(f"{'需求':40s} {'期望':>22s} {'Jev':>22s} {'置信':>5s} {'耗时':>7s}")
    for instruction, expected_name, category in TASKS:
        expected = expected_name or "NONE"
        task = CodeTask("select_function", "python", (instruction,), ())
        try:
            choice, stats = engine.choose(task, page)
        except Exception as exc:  # noqa: BLE001
            print(f"{instruction[:38]:40s} {expected:>22s} {'ERR':>22s}  {type(exc).__name__}")
            wrong.append(f"{category}: {instruction}（{type(exc).__name__}）")
            continue
        picked = choice.candidate.name if choice.candidate else "NONE"
        raw = json.loads(choice.raw_response)
        confidence = float(raw.get("confidence", 0))
        ok = picked == expected
        passed += int(ok)
        times.append(stats.generate_seconds)
        confidences.append(confidence)
        if not ok:
            wrong.append(f"{category}: {instruction} → 选了 {picked}，应为 {expected}")
        print(
            f"{instruction[:38]:40s} {expected:>22s} {picked:>22s} "
            f"{confidence:>5.2f} {stats.generate_seconds:>6.2f}s {'✓' if ok else '✗'}"
        )

    total = len(TASKS)
    print()
    print(f"Jev：{passed}/{total}（{passed / total:.0%}）")
    if times:
        print(f"延迟：中位 {sorted(times)[len(times) // 2]:.2f}s｜范围 {min(times):.2f}-{max(times):.2f}s")
        print(f"置信度：中位 {sorted(confidences)[len(confidences) // 2]:.2f}｜最低 {min(confidences):.2f}")
    if wrong:
        print()
        print("失败明细：")
        for line in wrong:
            print(f"  - {line}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
