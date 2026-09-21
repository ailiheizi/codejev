"""单次前向选择：不生成 JSON，直接读 logits。

自回归生成 36 个 token 要 36 次前向；候选 id（f0/c1…）各自是单个 token，
所以可以只跑一次前向，把分布限制在候选 id 上取 argmax。
本脚本只回答一件事：这样做准不准、快多少。
"""

from __future__ import annotations

import json
import time

import mlx.core as mx
from mlx_lm import load

from codejev.decide import extract

MODEL = "models/Qwen2.5-Coder-1.5B-Instruct-4bit"

SOURCE = '''"""用户列表。"""


def active_users(users):
    """返回用户，保持原顺序。"""
    result = []
    for user in users:
        result.append({"id": user["id"], "name": user["name"], "active": user["active"]})
    return result
'''

# (指令, 期望的过滤字段, 期望的返回字段)
CASES = [
    ("只保留 active 为真的项，返回 id 和 name", "active", {"id", "name"}),
    ("只保留 name 为真的项，返回 id", "name", {"id"}),
    ("只保留 id 为真的项，返回 name", "id", {"name"}),
    ("不过滤，返回 id 和 name", None, {"id", "name"}),
    ("只保留 active 为真的项，只要 id", "active", {"id"}),
]


def main() -> int:
    print("加载模型...")
    t0 = time.perf_counter()
    model, tok = load(MODEL)
    print(f"加载 {time.perf_counter() - t0:.2f}s\n")

    cands = extract(SOURCE, "active_users")
    # c0 会分成 "c"+"0" 两个 token，没法一次 argmax。
    # 换成裸数字：0/1/2 各自是单 token，宿主自己记住 编号→候选 的对应。
    def digit_ids(n: int) -> dict[str, list[int]]:
        return {str(i): tok.encode(str(i)) for i in range(n)}

    tok_ids = digit_ids(len(cands.conditions))
    # 用 9 当"不过滤"的哨兵；-1 是两 token，不能用于单次 argmax。
    tok_ids["9"] = tok.encode("9")
    print("候选（提示里只出现编号）：",
          " ".join(f"{i}={c.name}" for i, c in enumerate(cands.conditions)))
    print("编号的 token：", tok_ids)
    single = all(len(v) == 1 for v in tok_ids.values())
    print(f"全部是单 token：{single}\n")
    if not single:
        print(">>> 编号不是单 token，这条思路在本形式下不成立")
        return 1

    right = 0
    total_ms = 0.0
    for instruction, want_field, want_returns in CASES:
        # 提示到这里为止，下一个 token 就该是编号
        prompt = (
            f"任务：{instruction}\n"
            f"可选过滤字段（只回编号）：{' '.join(f'{i}={c.name}' for i, c in enumerate(cands.conditions))}\n"
            f"不过滤回 9。过滤字段编号："
        )
        messages = [{"role": "user", "content": prompt}]
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        toks = tok.encode(text)

        t0 = time.perf_counter()
        logits = model(mx.array([toks]))[0, -1, :]
        mx.eval(logits)
        ms = (time.perf_counter() - t0) * 1000
        total_ms += ms

        # 把分布限制在编号的 token 上
        cand_logits = {cid: float(logits[v[0]]) for cid, v in tok_ids.items()}
        best = max(cand_logits, key=cand_logits.get)
        got_field = None if best == "9" else cands.conditions[int(best)].name
        ok = got_field == want_field
        right += ok

        probs = mx.softmax(mx.array(list(cand_logits.values())))
        mx.eval(probs)
        pmap = {cid: float(p) for cid, p in zip(cand_logits, probs)}
        top2 = sorted(pmap.items(), key=lambda kv: -kv[1])[:2]
        margin = top2[0][1] - (top2[1][1] if len(top2) > 1 else 0.0)

        print(
            f"{'✅' if ok else '❌'} {instruction[:28]:30s} "
            f"期望={want_field!s:6s} 得到={got_field!s:6s} "
            f"p={top2[0][1]:.3f} 余量={margin:.3f} {ms:.0f}ms"
        )

    print(f"\n单次前向准确率 {right}/{len(CASES)}｜平均 {total_ms / len(CASES):.0f} ms")
    print("对照：自回归生成 JSON 基线约 600 ms/次（36 token，5/5 正确）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
