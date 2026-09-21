"""生成路线三方对照：本地 Qwen / DeepSeek Flash / Mercury（扩散）。

任务是一个"写整个模块"的真实长输出需求（三个函数 + 类型注解 + docstring），
输出规模落在扩散该发挥优势的区间（400+ token）。

三方拿到**完全相同的提示和 max_tokens**，判断一律用运行时行为：
把产物写成临时模块、真的 import、真的用样本数据调用三个函数，检查返回值。
机械问题（围栏等）由生产管线 `to_artifact` 自动清理后仍然失败，才算语义失败。
"""

from __future__ import annotations

import ast
import json
import os
import tempfile
import textwrap
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from chooseonly.adapter import to_artifact
from chooseonly.api_engine import OpenAICompatibleEngine
from chooseonly.contracts import Action, Brief, Kind
from chooseonly.model import MLXEngine

TARGET = "record_ops.py"
INSTRUCTION = (
    "写一个完整的 Python 模块，必须包含下面三个函数，并带上类型注解和 docstring：\n"
    "1. filter_active(users)：只保留 active 为真的项，返回 id 和 name，保持原顺序；\n"
    "2. sort_by_total(orders)：按 total 从高到低排序，保留完整记录；\n"
    "3. group_by_category(rows)：按 category 分组统计数量，返回 category 到数量的字典。\n"
    "只输出代码，不要解释。"
)

# 判定用的样本数据与期望结果。
USERS = [
    {"id": 1, "name": "Ada", "active": True},
    {"id": 2, "name": "Bob", "active": False},
    {"id": 3, "name": "Cid", "active": True},
]
ORDERS = [
    {"id": "A", "total": 5},
    {"id": "B", "total": 9},
    {"id": "C", "total": 7},
]
ROWS = [
    {"category": "x"},
    {"category": "y"},
    {"category": "x"},
    {"category": "x"},
]


def check_module(source: str) -> tuple[bool, tuple[str, ...]]:
    """运行时验证三个函数的真实行为。"""
    reasons: list[str] = []
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return False, (f"语法错误: {exc.msg}",)

    namespace: dict[str, object] = {}
    try:
        exec(compile(tree, "<candidate>", "exec"), namespace)  # noqa: S102 - 只跑候选样本
    except Exception as exc:  # noqa: BLE001
        return False, (f"无法执行: {type(exc).__name__}",)

    for name in ("filter_active", "sort_by_total", "group_by_category"):
        if not callable(namespace.get(name)):
            reasons.append(f"缺少函数 {name}")
    if reasons:
        return False, tuple(reasons)

    try:
        filtered = namespace["filter_active"](USERS)  # type: ignore[operator]
        if [row.get("id") for row in filtered] != [1, 3]:
            reasons.append(f"filter_active 结果错: {filtered}")
        for row in filtered:
            if set(row) != {"id", "name"}:
                reasons.append(f"filter_active 字段错: {sorted(row)}")
                break

        sorted_orders = namespace["sort_by_total"](ORDERS)  # type: ignore[operator]
        if [row.get("total") for row in sorted_orders] != [9, 7, 5]:
            reasons.append(f"sort_by_total 顺序错: {sorted_orders}")
        if sorted_orders and set(sorted_orders[0]) != {"id", "total"}:
            reasons.append(f"sort_by_total 丢字段: {sorted(sorted_orders[0])}")

        grouped = namespace["group_by_category"](ROWS)  # type: ignore[operator]
        if grouped != {"x": 3, "y": 1}:
            reasons.append(f"group_by_category 结果错: {grouped}")
    except Exception as exc:  # noqa: BLE001
        reasons.append(f"调用失败: {type(exc).__name__}: {exc}")
    return (not reasons), tuple(reasons)


@dataclass
class Backend:
    """一个可调用的生成后端。"""

    label: str
    generate: object  # Callable[[list[dict[str,str]], int], tuple[str, int, float]]


MAX_TOKENS = 1200


def mercury_generate(messages: list[dict[str, str]], max_tokens: int) -> tuple[str, int, float]:
    key = os.environ.get("MERCURY_KEY", "")
    if not key:
        raise RuntimeError("需要 MERCURY_KEY")
    last: Exception | None = None
    for attempt in range(4):
        payload = json.dumps(
            {
                "model": "mercury-2.5",
                "reasoning_effort": "none",  # 必须：否则会烧 200+ 推理 token
                "max_tokens": max_tokens,
                "messages": messages,
            }
        ).encode()
        request = urllib.request.Request(
            "https://api.inceptionlabs.ai/v1/chat/completions",
            data=payload,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            method="POST",
        )
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                body = json.loads(response.read().decode())
            elapsed = time.perf_counter() - started
            content = body["choices"][0]["message"].get("content") or ""
            return content, int(body.get("usage", {}).get("completion_tokens", 0)), elapsed
        except urllib.error.HTTPError as exc:
            last = exc
            if exc.code in (400, 429, 500, 502, 503) and attempt < 3:
                time.sleep(6 * (attempt + 1))
                continue
            raise
    raise RuntimeError(f"Mercury 重试后仍失败: {last}")


def deepseek_generate(messages: list[dict[str, str]], max_tokens: int) -> tuple[str, int, float]:
    """走 OpenAI-compatible 执行器；配置来自环境变量（bench/provider_config.py）。"""
    from bench.provider_config import load_provider

    engine = OpenAICompatibleEngine(load_provider())
    raw, stats = engine.generate(messages, max_tokens=max_tokens)
    return raw, stats.generated_tokens, stats.generate_seconds


def local_generate(messages: list[dict[str, str]], max_tokens: int) -> tuple[str, int, float]:
    engine = _LOCAL["engine"]
    if engine is None:
        engine = MLXEngine()
        _LOCAL["engine"] = engine
    raw, stats = engine.generate(messages, max_tokens=max_tokens)
    return raw, stats.generated_tokens, stats.generate_seconds


_LOCAL: dict[str, object] = {"engine": None}


def run_backend(backend: Backend, brief: Brief, trials: int) -> None:
    print()
    print("=" * 84)
    print(f"后端：{backend.label}")
    print("=" * 84)
    print(f"  {'#':>2s} {'结果':>5s} {'输出tok':>8s} {'墙钟s':>7s} {'tok/s':>7s}  说明")
    from chooseonly.adapter import build_messages

    messages = build_messages(brief)
    passed = 0
    times: list[float] = []
    tokens: list[int] = []
    for index in range(trials):
        try:
            raw, tok, elapsed = backend.generate(messages, MAX_TOKENS)  # type: ignore[operator]
        except Exception as exc:  # noqa: BLE001
            print(f"  {index + 1:>2d} {'ERR':>5s} {'-':>8s} {'-':>7s} {'-':>7s}  {type(exc).__name__}: {str(exc)[:70]}")
            continue
        artifact = to_artifact(brief, raw)  # 生产管线：机械围栏在这里被清掉
        ok, why = check_module(artifact.body)
        passed += int(ok)
        times.append(elapsed)
        tokens.append(tok)
        rate = tok / elapsed if elapsed else 0.0
        print(
            f"  {index + 1:>2d} {'PASS' if ok else 'FAIL':>5s} {tok:>8d} {elapsed:>7.2f} {rate:>7.1f}"
            f"  {'' if ok else '; '.join(why)[:90]}"
        )
        time.sleep(1.5)
    if times:
        print(
            f"  → {passed}/{len(times)} 通过｜平均 {sum(times) / len(times):.2f}s｜"
            f"平均 {sum(tokens) / len(tokens):.0f} tok｜"
            f"平均 {sum(tokens) / sum(times):.0f} tok/s"
        )
    else:
        print("  → 全部调用失败")


def main() -> int:
    brief = Brief(
        instruction=INSTRUCTION,
        target=TARGET,
        action=Action.CREATE,
        kind=Kind.CODE,
        context="",
        original=None,
    )
    print(f"任务：写完整模块（三个函数 + 类型注解 + docstring）")
    print(f"max_tokens={MAX_TOKENS}　判断：写成临时模块真的 import 并调用三个函数")
    backends = [
        Backend("本地 Qwen2.5-Coder-1.5B-4bit", local_generate),
        Backend("DeepSeek Flash（API）", deepseek_generate),
        Backend("Mercury 2.5（扩散，API）", mercury_generate),
    ]
    for backend in backends:
        run_backend(backend, brief, trials=5)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
