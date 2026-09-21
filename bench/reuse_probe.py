"""复用推理实测：同一份上下文 + K 条不同指令，前缀只算一次能省多少。

用户假设：请求之间在低维上很接近，只有少数维度不同。放到本项目里就是——
同一份文件的连续多次修改，每条的提示里"文件上下文 + 候选表"完全相同，
只有那条短指令不同。

对照两条路线（都是本地同一份权重，因此结论只归因于复用机制本身）：
  A 独立：每条请求都把上下文+指令整个预填充一遍（现在的生产行为）
  B 复用：上下文只预填充一次，之后每条请求克隆 KV、只预填充那条短指令

判断标准：
  1. 两条路线的输出必须逐字相同（复用不能改变答案）
  2. 分别量总墙钟、前缀预填充耗时、以及 K 增大时的节省比例
"""

from __future__ import annotations

import time
from pathlib import Path

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import KVCache
from mlx_lm.sample_utils import make_sampler

MODEL = str(
    Path(__file__).resolve().parent.parent / "models" / "Qwen2.5-Coder-1.5B-Instruct-4bit"
)

# 共享上下文：一段真实的项目代码 + 候选表，模拟每次修改都要带的文件背景。
SHARED_CONTEXT = '''"""用户与订单查询：当前返回全部字段，不做过滤。"""


def list_users(users):
    """返回用户，保持原顺序。"""
    result = []
    for user in users:
        result.append(
            {"id": user["id"], "name": user["name"], "active": user["active"],
             "deleted": user["deleted"], "score": user["score"]}
        )
    return result


def list_orders(orders):
    """返回订单，保持原顺序。"""
    rows = []
    for order in orders:
        rows.append(
            {"order_id": order["order_id"], "total": order["total"],
             "paid": order["paid"], "shipped": order["shipped"]}
        )
    return rows
'''

# K 条只在少数维度上不同的指令：同一个函数、同一个候选表，只有要求不同。
INSTRUCTIONS = (
    "只保留 active 为真的项，返回 id 和 name，保持原顺序。",
    "只保留 deleted 为假的项，返回 id 和 name，保持原顺序。",
    "只保留 score 大于 0 的项，返回 id 和 name，保持原顺序。",
    "返回 id、name、score 三个字段，不过滤，保持原顺序。",
    "只保留 active 为真的项，返回 id、name、score，保持原顺序。",
)


def build_prompt_ids(tokenizer, instruction: str) -> list[int]:
    """把共享上下文和一条指令拼成一次请求的完整 prompt。"""
    messages = [
        {
            "role": "user",
            "content": (
                f"{SHARED_CONTEXT}\n"
                f"候选字段：f0=id f1=name f2=active f3=deleted f4=score\n"
                f"任务：{instruction}\n"
                "只输出修改后的完整函数正文。"
            ),
        }
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return tokenizer.encode(text)


def common_prefix_len(sequences: list[list[int]]) -> int:
    """所有 prompt 的**真实**公共前缀长度。

    不能靠"单独编码上下文再数长度"来猜边界：分词器对"任务："和"任务：只保留"
    会切出不同的 token，猜出来的位置和真实边界对不齐，后缀就会错位。
    这里逐 token 比对，取真正一致的前缀。
    """
    shortest = min(len(s) for s in sequences)
    index = 0
    while index < shortest and all(s[index] == sequences[0][index] for s in sequences):
        index += 1
    return index


def decode_from(model, tokenizer, logits, cache, max_tokens: int = 128) -> list[int]:
    """从给定 logits 出发贪心解码。"""
    sampler = make_sampler(temp=0.0)
    eos = set(tokenizer.eos_token_ids)
    produced: list[int] = []
    for _ in range(max_tokens):
        token = int(sampler(logits)[0])
        mx.eval()
        if token in eos:
            break
        produced.append(token)
        out = model(mx.array([[token]]), cache=cache)
        logits = out[:, -1, :]
        mx.eval(logits)
    return produced


def clone(cache: list[KVCache]) -> list[KVCache]:
    """克隆前缀 KV：只读复制，后续请求各自往下写，互不影响。"""
    return [KVCache.from_state(c.state, c.meta_state) for c in cache]


def run_independent(model, tokenizer, prompts: list[list[int]]) -> tuple[list[str], float]:
    """路线 A：每条请求完整预填充（现在的生产行为）。"""
    started = time.perf_counter()
    bodies: list[str] = []
    for ids in prompts:
        cache = [KVCache() for _ in model.layers]
        logits = model(mx.array(ids)[None], cache=cache)[:, -1, :]
        mx.eval(logits)
        tokens = decode_from(model, tokenizer, logits, cache)
        bodies.append(tokenizer.decode(tokens))
    return bodies, time.perf_counter() - started


def run_reuse(
    model, tokenizer, prefix_ids: list[int], suffix_ids: list[list[int]]
) -> tuple[list[str], float, float]:
    """路线 B：前缀只预填充一次，之后每条请求克隆 KV、只算自己的短后缀。"""
    # 1) 前缀预填充一次
    start = time.perf_counter()
    base_cache = [KVCache() for _ in model.layers]
    logits = model(mx.array(prefix_ids)[None], cache=base_cache)[:, -1, :]
    mx.eval(logits)
    prefix_seconds = time.perf_counter() - start
    del logits

    # 2) 每条请求复用前缀 KV，只预填充自己的后缀
    started = time.perf_counter()
    bodies: list[str] = []
    for suffix in suffix_ids:
        cache = clone(base_cache)
        logits = model(mx.array(suffix)[None], cache=cache)[:, -1, :]
        mx.eval(logits)
        tokens = decode_from(model, tokenizer, logits, cache)
        bodies.append(tokenizer.decode(tokens))
    total = time.perf_counter() - started
    return bodies, total, prefix_seconds


def measure_one(model, tokenizer, context: str, label: str) -> None:
    """对一份上下文量一次：独立 vs 复用，并报告节省与预测上限。"""
    messages_ctx = (
        f"{context}\n"
        "候选字段：f0=id f1=name f2=active f3=deleted f4=score\n"
        "任务："
    )

    def build(instruction: str) -> list[int]:
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": f"{messages_ctx}{instruction}\n只输出修改后的完整函数正文。"}],
            tokenize=False,
            add_generation_prompt=True,
        )
        return tokenizer.encode(text)

    prompts = [build(text) for text in INSTRUCTIONS]
    prefix_len = common_prefix_len(prompts)
    prefix_ids = prompts[0][:prefix_len]
    suffix_ids = [ids[prefix_len:] for ids in prompts]

    bodies_a, seconds_a = run_independent(model, tokenizer, prompts)
    bodies_b, seconds_b, prefix_seconds = run_reuse(model, tokenizer, prefix_ids, suffix_ids)

    per_call = seconds_a / len(prompts)
    ceiling = prefix_seconds / per_call  # K→∞ 时的节省上限
    print(f"── {label} ──")
    print(f"  上下文前缀 {prefix_len} token（占提示 {prefix_len / len(prompts[0]):.0%}）｜"
          f"指令后缀 {len(suffix_ids[0])} token｜K={len(prompts)}")
    print(f"  独立 {seconds_a:.2f}s（{per_call:.3f}s/条） → "
          f"复用 {seconds_b + prefix_seconds:.2f}s → 省 "
          f"{1 - (seconds_b + prefix_seconds) / seconds_a:.1%}")
    print(f"  前缀预填充一次 {prefix_seconds:.3f}s｜占单条 {prefix_seconds / per_call:.0%}"
          f" → 即使 K 无限大，节省上限也只有 {ceiling:.1%}")
    print(f"  输出逐字相同：{bodies_a == bodies_b}")
    print()
    return None


def main() -> int:
    print(f"模型：{MODEL}")
    print("加载中...")
    model, tokenizer = load(MODEL)
    print()

    # 短上下文：只有两个函数
    measure_one(model, tokenizer, SHARED_CONTEXT, "短上下文")

    # 长上下文：把同一份代码重复堆成更长的文件背景，模拟大文件
    long_context = SHARED_CONTEXT + "\n\n" + "\n\n".join(
        f"# ---- 历史修订 {i} ----\n" + SHARED_CONTEXT for i in range(1, 6)
    )
    measure_one(model, tokenizer, long_context, "长上下文（5 倍文件背景）")

    print("结论口径：同一份权重、同一台机器，只比较复用机制本身。")
    print("节省上限 = 前缀预填充耗时 / 单条总耗时；它随上下文变长而上升。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
