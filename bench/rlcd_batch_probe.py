"""把开源 RLCD 实现的"批打分"机制移植到 MLX，用我们自己的候选页测。

机制来源：notnotsamuel/LFM2.5-350M-RLCD 的 rlcd/engine.py 的 constrained()。
它的做法是：
  1. 预填充一次上下文
  2. 对【每个候选】拼一个分支后缀
  3. 把预填充的 KV 复制成 N 条序列（fork cache）
  4. 一次批前向，右侧填充 + attention mask
  5. 每个分支算【整个候选值的 log 概率之和】，不取首 token、不做长度归一化
  6. argmax

**总前向次数是 2**（一次预填充 + 一次批打分），与候选数量无关。

本脚本做两件事：
  A. 用 MLX 实现同一机制，跑我们已有的候选页，和自回归选择对照
  B. 量候选数从 3 涨到 255 时的耗时，看它说的"255 候选反而慢 3.23 倍"是否复现
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import BatchKVCache, KVCache

# 先用纯注意力的 Qwen 做**受控对照**：同一个模型、同一个任务，只换推理方式。
# LFM2.5 是混合架构（10 层 ArraysCache 卷积状态 + 6 层 KVCache），
# 复制卷积状态要额外处理，留作后续；把模型换成 LFM2.5 就能跑那条线。
MODEL = str(
    Path(__file__).resolve().parent.parent / "models" / "Qwen2.5-Coder-1.5B-Instruct-4bit"
)

# 复用现有候选页的内容（与 bench/candidate_probe.py 同源）
PAGE = (
    ("0", "filter_and_project", "过滤 active 为真的行，并按 id、name 投影，保持原顺序。"),
    ("1", "sort_rows", "按 score 从高到低排序 rows，保留每行的完整内容。"),
    ("2", "group_and_count", "按 category 分组计数，返回 category 到数量的映射。"),
)

TASK = "任务：只保留 active 为真的项，返回 id 和 name，保持原顺序。"


def build_prefix(tokenizer, task: str, page, extra: int = 0) -> list[int]:
    """公共前缀：任务 + 候选表，末尾停在"只回编号："之后。"""
    lines = [task, "候选（只回编号）："]
    for cid, name, purpose in page:
        lines.append(f"{cid}={name}：{purpose}")
    for index in range(extra):  # 规模实验用：灌入无关候选抬高候选数
        lines.append(f"{1000 + index}=filler_{index}：与任务无关的补充候选。")
    lines.append("不过滤回 NONE。只回编号，不要解释。")
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": "\n".join(lines)}], tokenize=False, add_generation_prompt=True
    )
    return tokenizer.encode(text)


def fork_cache(base: list, count: int) -> list:
    """把预填充的 cache 复制成 count 条序列（对应它们的 fork_cache）。

    LFM2/LFM2.5 是混合架构：注意力层是 KVCache，卷积层是 ArraysCache。
    两类的 state 批维都是 1，都能沿 axis=0 重复。
    这正对应它们注释里说的"reorder_cache handles both"。
    """
    from mlx_lm.models.cache import ArraysCache

    out: list = []
    for layer in base:
        if isinstance(layer, KVCache):
            keys, values = layer.state
            batch = BatchKVCache([0] * count)
            batch.state = (
                mx.repeat(keys, count, axis=0),
                mx.repeat(values, count, axis=0),
                mx.array([keys.shape[2]] * count),
                mx.array([0] * count),
            )
            out.append(batch)
        else:  # ArraysCache：卷积/SSM 的循环状态
            repeated = [mx.repeat(array, count, axis=0) for array in layer.state]
            new = ArraysCache(len(repeated))
            new.state = repeated
            out.append(new)
    return out


def constrained_score(model, tokenizer, prefix: list[int], values: list[str]) -> tuple[str, float, int]:
    """一次预填充 + 一次批打分，返回得分最高的候选。

    返回 (选中的候选, 打分秒数, 前向次数)。
    """
    forward_calls = 0
    # 1) 预填充一次。用模型自己的 make_cache()，这样混合架构（卷积状态）也对。
    base_cache = model.make_cache()
    logits = model(mx.array(prefix)[None], cache=base_cache)[:, -1, :]
    mx.eval(logits)
    forward_calls += 1
    del logits

    # 2) 每个候选拼一个分支：先一个显式 token 边界，再是候选值本身
    branches: list[tuple[list[int], list[int]]] = []
    for value in values:
        suffix = tokenizer.encode(" ")
        value_ids = tokenizer.encode(value)
        branches.append((suffix, value_ids))

    # 3) 复制 KV 成 N 条，一次批前向（右侧填充）
    width = max(len(s) + len(v) for s, v in branches)
    ids = []
    right_pads = []
    for suffix, value_ids in branches:
        row = suffix + value_ids
        pad = width - len(row)
        ids.append(row + [0] * pad)
        right_pads.append(pad)

    started = time.perf_counter()
    cache = fork_cache(base_cache, len(branches))
    prefix_len = len(prefix)
    total_lengths = [prefix_len + len(s) + len(v) for s, v in branches]
    for layer in cache:
        try:
            # 注意力层：按右侧填充声明
            layer.prepare(right_padding=right_pads)
        except TypeError:
            # 卷积/SSM 层（ArraysCache）：只接受 lengths
            layer.prepare(lengths=total_lengths)
    output = model(mx.array(ids), cache=cache)
    mx.eval(output)
    forward_calls += 1

    # 4) 每个分支算整段候选值的 log 概率之和
    # output 形状 (N, width, vocab)：第 i 条序列前 len(suffix)+len(value) 个位置有效
    scores: list[float] = []
    for row, (suffix, value_ids) in enumerate(branches):
        row_logits = output[row].astype(mx.float32)
        # 该 MLX 版本没有 mx.log_softmax，手算：x - logsumexp(x)
        logp = row_logits - mx.logsumexp(row_logits, axis=-1, keepdims=True)
        total = 0.0
        for offset, token in enumerate(value_ids):
            position = len(suffix) + offset - 1
            total += float(logp[position, token])
        scores.append(total)
    del base_cache, cache
    mx.clear_cache()

    best = max(range(len(values)), key=lambda i: scores[i])
    seconds = time.perf_counter() - started
    return values[best], seconds, forward_calls


def autoregressive(model, tokenizer, prefix: list[int], max_tokens: int = 6) -> tuple[str, float]:
    """对照：自回归生成（现在的常见做法）。"""
    from mlx_lm import generate as mlx_generate
    from mlx_lm.sample_utils import make_sampler

    started = time.perf_counter()
    text = mlx_generate(model, tokenizer, prompt=prefix, max_tokens=max_tokens,
                        sampler=make_sampler(temp=0.0), verbose=False)
    return text.strip(), time.perf_counter() - started


def main() -> int:
    print(f"模型：{MODEL}（MLX，4bit）")
    model, tokenizer = load(MODEL)
    print("加载完成")
    print()
    print("机制来自 notnotsamuel/LFM2.5-350M-RLCD 的 constrained()：")
    print("  预填充一次 → KV 复制成 N 条 → 一次批打分 → 取 log 概率之和最大的候选")
    print()

    cases = (
        ("filter-and-project", TASK, "0"),
        ("sort-rows", "任务：把结果按 score 从高到低排序，保留完整记录。", "1"),
        ("group-count", "任务：按 category 分组统计数量，返回映射。", "2"),
        ("no-match-join", "任务：按 owner_id 关联用户和订单，返回合并记录。", "NONE"),
    )
    values = ["0", "1", "2", "NONE"]

    print(f"{'任务':22s} {'期望':>6s} {'打分':>6s} {'自回归':>8s} {'打分耗时':>10s} {'自回归耗时':>12s} 前向")
    score_pass = ar_pass = 0
    for label, task, expected in cases:
        prefix = build_prefix(tokenizer, task, PAGE)
        picked, score_seconds, calls = constrained_score(model, tokenizer, prefix, values)
        ar_text, ar_seconds = autoregressive(model, tokenizer, prefix)
        # 模型会带 <|im_end|> 等控制符，用项目自己的 clean_body 去掉再比对
        from codejev.model import clean_body

        ar_clean = clean_body(ar_text).strip().strip("`").split()[0].strip("\"'.,") if ar_text.strip() else ""
        score_pass += int(picked == expected)
        ar_pass += int(ar_clean == expected)
        print(
            f"{label:22s} {expected:>6s} {picked:>6s} {ar_clean:>8s} "
            f"{score_seconds:>9.3f}s {ar_seconds:>11.3f}s {calls:>4d}"
        )

    total = len(cases)
    print()
    print(f"批打分：{score_pass}/{total}｜自回归：{ar_pass}/{total}")
    print()

    # B：候选规模曲线（它报告 255 候选时反而慢 3.23 倍）
    print("候选规模曲线（同一任务，灌入无关候选抬高候选数）")
    print(f"  {'候选数':>6s} {'前缀tok':>8s} {'打分耗时':>10s} {'选中':>6s}")
    base_prefix = build_prefix(tokenizer, TASK, PAGE)
    for extra in (0, 32, 128, 252):
        prefix = build_prefix(tokenizer, TASK, PAGE, extra=extra)
        picked, seconds, _ = constrained_score(model, tokenizer, prefix, values)
        print(f"  {len(values):>6d} {len(prefix):>8d} {seconds:>9.3f}s {picked:>6s}")
    print()
    print("注意：这里的候选值只有 4 个（编号 + NONE），规模实验抬高的是提示长度，不是候选数。")
    print("      它报告的是 255 个候选值时打分反而慢 3.23 倍——那需要真的把候选值做到 255 个。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
