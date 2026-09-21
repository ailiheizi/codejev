"""Mercury（扩散 LLM）在生成路线上的实测：质量 + 吞吐随输出长度的变化。

候选选择任务输出只有 1-2 个 token，扩散的并行度优势用不上（实测 5.68 s/次）。
这里改测扩散真正该发挥的地方：长输出的代码生成。

两个实验：
  A. 真实函数生成任务（与 bench/compare.py 同题），判断用运行时行为，不看代码像不像。
  B. 输出长度扫描：从短到长要求不同规模的输出，量墙钟与 tokens/s，
     看固定的网络往返摊销之后，扩散的吞吐优势在哪个长度开始出现。
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request

from chooseonly.adapter import to_artifact
from chooseonly.contracts import Action, Brief, Kind
from bench.compare import BENCH_SOURCE, TASK_FUNCTION, TASK_INSTRUCTION, check_runtime_behaviour

BASE_URL = "https://api.inceptionlabs.ai/v1/chat/completions"
MODEL = "mercury-2.5"


def call(messages: list[dict[str, str]], max_tokens: int = 2000) -> tuple[str, dict, float]:
    """一次 Mercury 调用；返回正文、usage、墙钟秒数。"""
    key = os.environ.get("MERCURY_KEY", "")
    if not key:
        raise RuntimeError("需要 MERCURY_KEY 环境变量")
    payload = json.dumps(
        {
            "model": MODEL,
            "reasoning_effort": "none",  # 必须：否则简单任务会烧 200+ 推理 token
            "max_tokens": max_tokens,
            "messages": messages,
        }
    ).encode()
    request = urllib.request.Request(
        BASE_URL,
        data=payload,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=180) as response:
        body = json.loads(response.read().decode())
    elapsed = time.perf_counter() - started
    content = body["choices"][0]["message"].get("content") or ""
    return content, body.get("usage") or {}, elapsed


def experiment_a(trials: int = 3) -> None:
    """真实函数生成任务：只保留 active 为真的项、返回 id 和 name。"""
    print("=" * 78)
    print("实验 A：真实函数生成（与 bench/compare.py 同题，判断用运行时行为）")
    print("=" * 78)
    brief = Brief(
        instruction=TASK_INSTRUCTION,
        target="users.py",
        action=Action.REPLACE,
        kind=Kind.CODE,
        context=BENCH_SOURCE,
        original=BENCH_SOURCE,
    )
    from chooseonly.adapter import build_messages

    passed = 0
    for index in range(trials):
        raw, usage, elapsed = call(build_messages(brief))
        artifact = to_artifact(brief, raw)
        ok, why = check_runtime_behaviour(artifact.body, TASK_FUNCTION)
        passed += int(ok)
        completion = usage.get("completion_tokens", 0)
        rate = completion / elapsed if elapsed else 0.0
        print(
            f"  第{index + 1}次 {'PASS' if ok else 'FAIL'} "
            f"{elapsed:6.2f}s  输出 {completion:4d} tok  {rate:6.1f} tok/s"
            + ("" if ok else f"  原因: {', '.join(why)}")
        )
        if index == 0:
            print("  --- 首次输出 ---")
            for line in artifact.body.splitlines()[:14]:
                print(f"  | {line}")
    print(f"  A 结果：{passed}/{trials}")
    print("  对照：本地 Qwen1.5B 生成 5/5（0.92-1.23s，70 tok）；真实形状函数上 0/5")


def experiment_b() -> None:
    """输出长度扫描：扩散的吞吐优势应该随输出变长而出现。"""
    print()
    print("=" * 78)
    print("实验 B：输出长度扫描（固定往返开销 vs 扩散吞吐）")
    print("=" * 78)
    print(f"  {'要求长度':>8s} {'实际 tok':>9s} {'墙钟 s':>8s} {'tok/s':>8s} {'边际 ms/tok':>12s}")
    previous: tuple[int, float] | None = None
    for target in (50, 200, 500, 1200):
        messages = [
            {
                "role": "user",
                "content": (
                    f"写一个 Python 函数 generate_records(n)，返回一个长度为 n 的列表，"
                    f"每个元素是含 id 和 name 两个键的字典。"
                    f"请给出完整实现，并在函数后面追加大约 {target} 个 token 的详细注释说明每一行。"
                    f"只输出代码，不要解释。"
                ),
            }
        ]
        try:
            content, usage, elapsed = call(messages, max_tokens=max(target * 2, 400))
        except urllib.error.HTTPError as exc:
            print(f"  {target:>8d}  HTTP {exc.code}: {exc.read().decode()[:120]}")
            continue
        completion = usage.get("completion_tokens", 0)
        rate = completion / elapsed if elapsed else 0.0
        if previous is None:
            marginal = "-"
        else:
            d_tok = completion - previous[0]
            d_time = elapsed - previous[1]
            marginal = f"{1000 * d_time / d_tok:.2f}" if d_tok > 0 else "-"
        previous = (completion, elapsed)
        print(f"  {target:>8d} {completion:>9d} {elapsed:>8.2f} {rate:>8.1f} {marginal:>12s}")
    print()
    print("  读法：若固定往返占主导，短输出的 tok/s 会被摊薄；")
    print("       边际 ms/tok 才是扩散真正的每 token 成本，厂商宣称的 1107 tok/s ≈ 0.9 ms/tok。")


def main() -> int:
    print(f"model={MODEL}  base_url={BASE_URL}  reasoning_effort=none")
    experiment_a()
    experiment_b()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
