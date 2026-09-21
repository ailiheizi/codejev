"""对照实验：生成路线 vs 选择路线，各自是否更快、是否更可靠。

这不是评测平台。只按文档要求回答三件事：
是否按指令完成、输出是否可用、完整交互是否更快。
直接用小模型自己的产物做判断，不引入另外的评审模型。
"""

from __future__ import annotations

import ast
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from chooseonly.contracts import Action, Artifact, Brief, Kind
from chooseonly.decide import Candidates, Decision, DecisionError, assemble, extract, parse_decision
from chooseonly.adapter import to_artifact
from chooseonly.model import Engine, Stats, request_body

# 固定任务：把一个返回全部字段的函数改成只返回选定字段。
TASK_INSTRUCTION = "修改函数：只保留 active 为真的项，返回 id 和 name，保持原顺序，其他不变。"
TASK_FUNCTION = "active_users"
EXPECTED_FIELDS = {"id", "name"}
FILTER_FIELD = "active"
# 期望的过滤结果：只留 active 为真的项。
EXPECTED_IDS = [1, 3]

# 基准用的源码：这个函数返回全部字段、且完全不过滤。
# 任务因此是一次真实改动（加过滤 + 裁字段），不是把已有行为再抄一遍。
BENCH_SOURCE = '''"""用户列表：当前返回全部字段。"""


def active_users(users):
    """返回用户，保持原顺序。"""
    result = []
    for user in users:
        result.append(
            {"id": user["id"], "name": user["name"], "active": user["active"]}
        )
    return result
'''


@dataclass
class Outcome:
    """一次尝试的结果；判断标准写在这里，不藏到别处。"""

    path: str
    ok: bool
    reasons: tuple[str, ...]
    artifact: Artifact | None
    stats: Stats
    wall_seconds: float
    raw: str = ""


@dataclass
class Report:
    trials: int
    results: list[Outcome] = field(default_factory=list)

    def add(self, outcome: Outcome) -> None:
        self.results.append(outcome)

    def by_path(self, path: str) -> list[Outcome]:
        return [r for r in self.results if r.path == path]

    def summary(self) -> str:
        lines = []
        for path in ("generate", "select"):
            rows = self.by_path(path)
            if not rows:
                continue
            ok = sum(1 for r in rows if r.ok)
            warm = [r for r in rows if r.stats.generate_seconds > 0]
            avg_gen = sum(r.stats.generate_seconds for r in warm) / len(warm) if warm else 0.0
            avg_wall = sum(r.wall_seconds for r in rows) / len(rows)
            max_tok = max((r.stats.generated_tokens for r in rows), default=0)
            lines.append(
                f"{path:9s} 通过 {ok}/{len(rows)}"
                f"｜平均生成 {avg_gen:.2f}s｜平均墙钟 {avg_wall:.2f}s｜最长输出 {max_tok} tokens"
            )
        return "\n".join(lines)

    def failure_detail(self) -> str:
        lines = []
        for r in self.results:
            if not r.ok:
                lines.append(f"[{r.path}] {', '.join(r.reasons)}")
        return "\n".join(lines)


# ---------- 判断标准：只看能不能用，不看好不好看 ----------

def check_runtime_behaviour(source: str, function_name: str) -> tuple[bool, tuple[str, ...]]:
    """把函数真的跑一遍，看过滤和字段是否符合要求。这是唯一可信的判断。"""
    reasons: list[str] = []
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return False, (f"语法错误: {exc.msg}",)

    ns: dict[str, object] = {}
    try:
        exec(compile(tree, "<candidate>", "exec"), ns)  # noqa: S102 - 只跑实验用的样本代码
    except Exception as exc:  # noqa: BLE001 - 记录任何执行失败
        return False, (f"无法执行: {type(exc).__name__}",)

    fn = ns.get(function_name)
    if not callable(fn):
        return False, (f"找不到可调用函数 {function_name}",)

    sample = [
        {"id": 1, "name": "A", "active": True, "extra": "x"},
        {"id": 2, "name": "B", "active": False, "extra": "y"},
        {"id": 3, "name": "C", "active": True, "extra": "z"},
    ]
    try:
        out = fn(sample)
    except Exception as exc:  # noqa: BLE001
        return False, (f"调用失败: {type(exc).__name__}",)

    if not isinstance(out, list):
        return False, ("返回值不是列表",)
    # 先确认每个元素都是字典，再做取值比较；模型可能返回 [1, 2, 3] 这类内容，
    # 直接在元素上调用 .get() 会让整轮对照崩掉，而不是记成一次失败。
    if not all(isinstance(item, dict) for item in out):
        return False, (f"元素不全是字典: {out!r}",)
    if [item.get("id") for item in out] != EXPECTED_IDS:
        reasons.append(f"过滤不正确: {out}")
    for item in out:
        if set(item) != EXPECTED_FIELDS:
            reasons.append(f"字段不正确: {sorted(item)}")
            break
    return (not reasons), tuple(reasons)


def looks_like_explanation(raw: str) -> bool:
    """小模型偶尔会附上说明，这违反“只回正文”。"""
    head = raw.strip()[:200]
    markers = ("好的", "以下是", "这里", "说明：", "Sure", "Here", "Explanation")
    return any(marker in head for marker in markers)


def looks_truncated(raw: str) -> bool:
    """围栏没闭合，说明输出被截断，正文不完整。"""
    ticks = raw.count("```")
    return ticks % 2 == 1


# ---------- 两条路线 ----------

def run_generate(engine: Engine, source: str, trials: int) -> Report:
    """生成路线：小模型自由写出整个函数正文。"""
    report = Report(trials=trials)
    for i in range(trials):
        brief = Brief(
            instruction=TASK_INSTRUCTION,
            target="users.py",
            action=Action.REPLACE,
            kind=Kind.CODE,
            context=source,
            original=source,
        )
        start = time.perf_counter()
        # 生成路线要重写整份文件，给足余量；截断会让围栏不闭合，那属于真实失败。
        raw, stats = request_body(engine, brief, max_tokens=1024)
        wall = time.perf_counter() - start
        artifact = to_artifact(brief, raw)
        reasons: list[str] = []
        if not artifact.body.strip():
            reasons.append("没有产出正文")
        else:
            try:
                ok, why = check_runtime_behaviour(artifact.body, TASK_FUNCTION)
            except Exception as exc:  # noqa: BLE001 - 单次失败不该中断整轮对照
                ok, why = False, (f"检查时异常: {type(exc).__name__}",)
            if not ok:
                reasons.extend(why)
        if looks_like_explanation(raw):
            reasons.append("附带了说明文字")
        if looks_truncated(raw):
            reasons.append("输出被截断（围栏未闭合）")
        report.add(
            Outcome(
                path="generate",
                ok=not reasons,
                reasons=tuple(reasons),
                artifact=artifact,
                stats=stats,
                wall_seconds=wall,
                raw=raw,
            )
        )
        if i == 0:
            print(f"[generate] 第 1 次原始输出:\n{raw[:400]}\n")
    return report


def run_select(engine: Engine, source: str, trials: int) -> Report:
    """选择路线：宿主提取候选，小模型只选，宿主确定性组装。"""
    report = Report(trials=trials)
    for i in range(trials):
        start = time.perf_counter()
        try:
            candidates = extract(source, TASK_FUNCTION)
        except DecisionError as exc:
            report.add(
                Outcome("select", False, (f"提取失败: {exc}",), None, Stats(), time.perf_counter() - start)
            )
            continue

        from chooseonly.decide import build_decision_prompt

        messages = build_decision_prompt(TASK_INSTRUCTION, candidates)
        raw, stats = engine.generate(messages, max_tokens=64)
        reasons: list[str] = []
        decision: Decision | None = None
        try:
            decision = parse_decision(raw, candidates)
        except DecisionError as exc:
            reasons.append(f"决策不合法: {exc}")

        body = ""
        if decision is not None:
            # 宿主按字段名核对：id 合法不代表选得对，这里再查一次真实名字。
            names = {c.id: c.name for c in candidates.fields}
            chosen = {names.get(fid, fid) for fid in decision.return_fields}
            if chosen != EXPECTED_FIELDS:
                reasons.append(f"选的字段不是期望值: {sorted(chosen)}")
            cond = None
            if decision.filter_field is not None:
                cnames = {c.id: c.name for c in candidates.conditions}
                cond = cnames.get(decision.filter_field, decision.filter_field)
            if cond != FILTER_FIELD:
                reasons.append(f"过滤字段不是期望值: {cond}")
            try:
                body = assemble(source, candidates, decision)
            except DecisionError as exc:
                reasons.append(f"组装失败: {exc}")
            else:
                try:
                    ok, why = check_runtime_behaviour(body, TASK_FUNCTION)
                except Exception as exc:  # noqa: BLE001 - 单次失败不该中断整轮对照
                    ok, why = False, (f"检查时异常: {type(exc).__name__}",)
                if not ok:
                    reasons.extend(why)

        wall = time.perf_counter() - start
        artifact = (
            to_artifact(
                Brief(
                    instruction=TASK_INSTRUCTION,
                    target="users.py",
                    action=Action.REPLACE,
                    kind=Kind.CODE,
                    context=source,
                    original=source,
                ),
                body,
            )
            if body
            else None
        )
        report.add(
            Outcome(
                path="select",
                ok=not reasons,
                reasons=tuple(reasons),
                artifact=artifact,
                stats=stats,
                wall_seconds=wall,
                raw=raw,
            )
        )
        if i == 0:
            print(f"[select] 候选: " + ", ".join(f"{c.id}={c.name}" for c in candidates.fields)
                  + " | " + ", ".join(f"{c.id}={c.name}" for c in candidates.conditions))
            print(f"[select] 第 1 次原始输出: {raw[:200]}")
            print(f"[select] 组装结果:\n{body}\n")
    return report


def run_all(engine: Engine, source: str, trials: int = 5) -> Report:
    report = run_generate(engine, source, trials)
    for outcome in run_select(engine, source, trials).results:
        report.add(outcome)
    return report


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="生成路线与选择路线对照")
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--model", default=None)
    args = parser.parse_args(argv)

    from chooseonly.model import MLXEngine

    engine = MLXEngine(args.model) if args.model else MLXEngine()
    print("加载模型...")
    engine._ensure_loaded()  # noqa: SLF001 - 常驻生效后才开始计时
    print(f"加载完成 {engine.load_seconds:.2f}s\n")

    report = run_all(engine, BENCH_SOURCE, args.trials)
    print("=== 结果 ===")
    print(report.summary())
    detail = report.failure_detail()
    if detail:
        print("\n=== 未通过明细 ===")
        print(detail)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
