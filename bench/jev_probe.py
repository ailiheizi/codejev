"""用 Jev（TypeSafe System One API）跑我们自己的候选页。

对照基准（同一份候选页、同一批任务）：
  本地 Qwen1.5B        2-3/4
  LFM2.5-350M + 批打分  1/4（恒定输出 "2"，零区分）
  DeepSeek Flash        5/5
  Jev                   本脚本

契约来源：jev-ultrafast 的 jev_ultrafast/model.py 读出来的真实请求结构
（endpoint https://api.typesafe.ai/v1/systemone，questions.type="choice"）。

关键点：Jev 是**域外迁移**——它在 UI 元素上训练，我们喂的是代码候选。
零样本 Jev 在表单域上是 83.6%（Cua 的对照），所以别预设它在代码上一样好。
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

ENDPOINT = "https://api.typesafe.ai/v1/systemone"
MODEL = "jev-latest"

# 与 bench/rlcd_batch_probe.py 完全相同的候选页
PAGE = (
    ("0", "filter_and_project",
     "过滤 active 为真的行，并按 id、name 投影，保持原顺序。",
     'result = []\nfor row in rows:\n    if row["active"]:\n        result.append({"id": row["id"], "name": row["name"]})\nreturn result'),
    ("1", "sort_rows",
     "按 score 从高到低排序 rows，保留每行的完整内容。",
     'return sorted(rows, key=lambda row: row["score"], reverse=True)'),
    ("2", "group_and_count",
     "按 category 分组计数，返回 category 到数量的映射。",
     'counts = {}\nfor row in rows:\n    key = row["category"]\n    counts[key] = counts.get(key, 0) + 1\nreturn counts'),
)

CASES = (
    ("只保留 active 为真的项，返回 id 和 name，保持原顺序。", "0", "过滤+裁字段"),
    ("把结果按 score 从高到低排序，保留完整记录。", "1", "排序"),
    ("按 category 分组统计数量，返回映射。", "2", "分组计数"),
    ("按 owner_id 关联用户和订单，返回合并记录。", "NONE", "候选页里没有这个能力"),
)


def call(instruction: str, retries: int = 3) -> tuple[dict, float]:
    """一次 Jev 请求：候选页翻成 questions.type=choice。"""
    key = os.environ.get("TYPESAFE_API_KEY", "")
    if not key:
        raise RuntimeError("需要 TYPESAFE_API_KEY")
    criteria = {
        cid: {"name": name, "purpose": purpose, "code": code}
        for cid, name, purpose, code in PAGE
    }
    criteria["NONE"] = {"name": "none", "purpose": "以上候选都无法满足这个需求"}
    body = {
        "model": MODEL,
        "state": {
            "page": {"url": "repo://local", "title": "候选代码页", "text": ""},
            "elements": [],
            "recent_actions": [],
        },
        "questions": {
            "candidate": {
                "type": "choice",
                "criteria": criteria,
                "instructions": {
                    "goal": instruction,
                    "rules": [
                        "只从给出的候选里选一个；都不满足时选 NONE。",
                        "按需求与候选用途/代码的匹配程度判断，不要按编号顺序。",
                    ],
                },
            }
        },
    }
    request = urllib.request.Request(
        ENDPOINT,
        data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        method="POST",
    )
    last: Exception | None = None
    for attempt in range(retries):
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                data = json.loads(response.read().decode())
            return data, time.perf_counter() - started
        except urllib.error.HTTPError as exc:
            last = exc
            if exc.code in (429, 500, 502, 503) and attempt < retries - 1:
                time.sleep(3 * (attempt + 1))
                continue
            raise
    raise RuntimeError(f"重试后仍失败: {last}")


def main() -> int:
    print(f"endpoint={ENDPOINT}  model={MODEL}")
    print("候选页（与 LFM2.5 / DeepSeek Flash 对照用的是同一份）：")
    for cid, name, purpose, _code in PAGE:
        print(f"  [{cid}] {name}：{purpose}")
    print()
    print(f"{'需求':34s} {'期望':>6s} {'Jev':>6s} {'置信度':>7s} {'耗时':>8s}  结果")
    passed = 0
    times: list[float] = []
    for instruction, expected, want in CASES:
        try:
            data, seconds = call(instruction)
        except Exception as exc:  # noqa: BLE001
            print(f"{instruction[:32]:34s} {expected:>6s} {'ERR':>6s} {'-':>7s} {'-':>8s}  {type(exc).__name__}: {str(exc)[:60]}")
            continue
        answer = (data.get("answers") or {}).get("candidate") or {}
        picked = answer.get("choice", "<无>")
        confidence = answer.get("confidence")
        ok = picked == expected
        passed += int(ok)
        times.append(seconds)
        print(
            f"{instruction[:32]:34s} {expected:>6s} {picked:>6s} "
            f"{(f'{confidence:.2f}' if isinstance(confidence, (int, float)) else '-'):>7s} "
            f"{seconds:>7.2f}s  {'✓' if ok else '✗'}（要的是 {want}）"
        )
    total = len(CASES)
    print()
    print(f"Jev：{passed}/{total}")
    if times:
        print(f"延迟：中位 {sorted(times)[len(times) // 2]:.2f}s｜范围 {min(times):.2f}-{max(times):.2f}s（含代理）")
    print()
    print("对照：DeepSeek Flash 5/5｜本地 Qwen1.5B 2-3/4｜LFM2.5-350M 1/4（恒定输出 2）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
