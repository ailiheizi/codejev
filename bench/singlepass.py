"""单次前向读 logits：验证“不生成、只读分布”这条路是否可用。

被测假设：选择路线的候选 id（f0、c0、fn0）是单 token，所以不必自回归生成
36 个 token，一次前向取最后位置的 logits、在合法候选 token 上 argmax 即可。

实测：假设不成立（token_probe 的输出）。f0=["f","0"]、c1=["c","1"]、
fn0=["fn","0"] 都是 2 个 token，只有 null、数字 0..9、「是」「否」是单 token。
于是把假设改成三个可实现的读法，逐一实测：

  V1  强制 JSON 前缀 + 受限读取：提示与基线完全相同的 chat 提示，末尾接上宿主
      自己写死的 JSON 前缀（'{"function": "fn0", "filter_field": "'），再逐
       token 读分布，在候选串（c0" / c1" / null"）上做受限选择。
      前向次数 = 1 次 prefill + 选项串剩余 token 的步数（实测 2–3 次）。
  V2  把候选 id 重新编号成单个数字 token（宿主本来就在生成 id）：同一个 JSON
      前缀下，答案位置的第一个 token 就是 0..9 之一 → 真正的单次前向。
  V3  每个返回字段一个「要不要」问题，全部问题写进同一条序列；因果注意力让
      每个「答：」位置的分布可以从同一次前向里读出来（天然批处理）。

正确性一律用运行时行为判断：读出的 id 拼回决策 JSON → parse_decision 校验 →
assemble → exec → 真调用函数 → 与期望行比对（与 bench/compare.py 同一套做法）。

本脚本只做实测，不改动任何产品代码；结论无论正负都如实打印。
"""

from __future__ import annotations

import argparse
import ast
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

from codejev.decide import (
    DECISION_MAX_TOKENS,
    Candidates,
    Decision,
    DecisionError,
    assemble,
    build_decision_prompt,
    extract,
    parse_decision,
)
from codejev.model import MLXEngine
from bench.compare import BENCH_SOURCE, TASK_FUNCTION, TASK_INSTRUCTION, check_runtime_behaviour

MODEL = str(
    Path(__file__).resolve().parent.parent / "models" / "Qwen2.5-Coder-1.5B-Instruct-4bit"
)

# 强制前缀：宿主自己写死的 JSON 开头，停在 filter_field 的字符串内部。
# 选项就是字符串的剩余部分（c0" / c1" / null"），实测这三种写法在边界处都不
# 发生 BPE 跨边界合并（尾随空格会合并，所以这里不加空格）。
JSON_TAIL = '{"function": "fn0", "filter_field": "'


# ---------------------------------------------------------------- 任务集
@dataclass(frozen=True)
class Case:
    key: str
    source: str
    function: str
    instruction: str
    expect_filter: str | None  # 期望的过滤字段名；None 表示期望“不过滤”
    expect_fields: frozenset[str]  # 期望的返回字段名集合
    sample: tuple[dict[str, object], ...]
    sort_by: str | None = None  # 组装后函数会按这个键降序排（源码里的 sort 原样保留）
    decision_only: bool = False  # 现有渲染只能表达 `if item[field]:`，这类指令只比决策


SOURCE_A = BENCH_SOURCE  # active_users：字段 id / name / active，完全不过滤

SOURCE_B = '''def paid_orders(orders):
    """返回已付款订单。"""
    rows = []
    if not orders:
        return []
    for order in orders:
        if order["paid"] and not order["shipped"]:
            rows.append({"order_id": order["order_id"], "total": order["total"]})
    rows.sort(key=lambda r: r["total"], reverse=True)
    return rows
'''

SOURCE_C = '''def item_rows(items):
    """返回商品行。"""
    out = []
    for item in items:
        out.append(
            {"in_stock": item["in_stock"], "sku": item["sku"], "price": item["price"]}
        )
    return out
'''

SOURCE_D = '''def event_rows(events):
    """返回事件行。"""
    result = []
    for event in events:
        result.append(
            {"ts": event["ts"], "level": event["level"], "msg": event["msg"], "urgent": event["urgent"]}
        )
    return result
'''

SAMPLE_A = (
    {"id": 1, "name": "A", "active": True, "extra": "x"},
    {"id": 2, "name": "B", "active": False, "extra": "y"},
    {"id": 3, "name": "C", "active": True, "extra": "z"},
)
SAMPLE_B = (
    {"order_id": 11, "total": 20, "paid": True, "shipped": True},
    {"order_id": 22, "total": 50, "paid": False, "shipped": False},
    {"order_id": 33, "total": 30, "paid": True, "shipped": False},
)
SAMPLE_C = (
    {"in_stock": True, "sku": "s1", "price": 10, "qty": 1},
    {"in_stock": False, "sku": "s2", "price": 20, "qty": 2},
    {"in_stock": True, "sku": "s3", "price": 30, "qty": 3},
)
SAMPLE_D = (
    {"ts": 1, "level": "info", "msg": "a", "urgent": False},
    {"ts": 2, "level": "error", "msg": "b", "urgent": True},
    {"ts": 3, "level": "warn", "msg": "c", "urgent": False},
)

CASES: tuple[Case, ...] = (
    Case(
        key="A_active",
        source=SOURCE_A,
        function=TASK_FUNCTION,
        instruction=TASK_INSTRUCTION,
        expect_filter="active",
        expect_fields=frozenset({"id", "name"}),
        sample=SAMPLE_A,
    ),
    Case(
        key="A_no_filter",
        source=SOURCE_A,
        function=TASK_FUNCTION,
        instruction="修改函数：不要加任何过滤条件，返回 id 和 name，保持原顺序。",
        expect_filter=None,
        expect_fields=frozenset({"id", "name"}),
        sample=SAMPLE_A,
    ),
    Case(
        key="B_paid",
        source=SOURCE_B,
        function="paid_orders",
        instruction="修改函数：只保留 paid 为真的项，返回 order_id 和 total，其他不变。",
        expect_filter="paid",
        expect_fields=frozenset({"order_id", "total"}),
        sample=SAMPLE_B,
        sort_by="total",
    ),
    Case(
        key="B_shipped",
        source=SOURCE_B,
        function="paid_orders",
        instruction="修改函数：只保留 shipped 为真的项，返回 order_id，其他不变。",
        expect_filter="shipped",
        expect_fields=frozenset({"order_id"}),
        sample=SAMPLE_B,
        sort_by="total",
    ),
    Case(
        key="C_in_stock",
        source=SOURCE_C,
        function="item_rows",
        instruction="修改函数：只保留 in_stock 为真的项，返回 sku 和 price，其他不变。",
        expect_filter="in_stock",
        expect_fields=frozenset({"sku", "price"}),
        sample=SAMPLE_C,
    ),
    Case(
        key="C_no_filter",
        source=SOURCE_C,
        function="item_rows",
        instruction="修改函数：不要加过滤条件，返回 sku 和 price，其他不变。",
        expect_filter=None,
        expect_fields=frozenset({"sku", "price"}),
        sample=SAMPLE_C,
    ),
    Case(
        key="D_urgent",
        source=SOURCE_D,
        function="event_rows",
        instruction="修改函数：只保留 urgent 为真的项，返回 ts 和 msg，其他不变。",
        expect_filter="urgent",
        expect_fields=frozenset({"ts", "msg"}),
        sample=SAMPLE_D,
    ),
    Case(
        key="D_level",
        source=SOURCE_D,
        function="event_rows",
        instruction="修改函数：只保留 level 为真的项，返回 msg，其他不变。",
        expect_filter="level",
        expect_fields=frozenset({"msg"}),
        sample=SAMPLE_D,
    ),
    # 现有渲染只能表达 `if item[field]:`（真值测试）。下面两条指令的语义与真值
    # 测试不同（比较 / 取反），只比“决策对不对”，不跑运行时。
    Case(
        key="A_name_cmp*",
        source=SOURCE_A,
        function=TASK_FUNCTION,
        instruction="修改函数：只保留 name 是 A 的项，返回 id 和 name，其他不变。",
        expect_filter="name",
        expect_fields=frozenset({"id", "name"}),
        sample=SAMPLE_A,
        decision_only=True,
    ),
    Case(
        key="B_shipped_neg*",
        source=SOURCE_B,
        function="paid_orders",
        instruction="修改函数：只保留 shipped 为假的项，返回 order_id 和 total，其他不变。",
        expect_filter="shipped",
        expect_fields=frozenset({"order_id", "total"}),
        sample=SAMPLE_B,
        sort_by="total",
        decision_only=True,
    ),
    # 指令里没提任何条件的对照：正确答案应当是 null。
    Case(
        key="C_no_mention",
        source=SOURCE_C,
        function="item_rows",
        instruction="修改函数：把每个商品的 sku 放进返回结果，其他不变。",
        expect_filter=None,
        expect_fields=frozenset({"sku"}),
        sample=SAMPLE_C,
    ),
)


# ---------------------------------------------------------------- 基础工具
def forward(model, ids: list[int], cache=None) -> mx.array:
    """一次前向，返回最后一个位置的 logits（强制求值，便于计时）。"""
    out = model(mx.array([ids]), cache=cache)
    logits = out[0, -1, :]
    mx.eval(logits)
    mx.synchronize()
    return logits


def softmax_full(logits: mx.array) -> mx.array:
    return mx.softmax(logits.astype(mx.float32))


def masked_dist(logits: mx.array, token_ids: list[int]) -> list[float]:
    """只在给定 token id 上做 softmax（= 把候选之外的概率全部剪掉）。"""
    sel = mx.array([logits[i] for i in token_ids])
    return mx.softmax(sel.astype(mx.float32)).tolist()


def option_token_seqs(
    tokenizer, options: dict[str, str], context: str = ""
) -> tuple[dict[str, list[int]], dict[str, str]]:
    """每个选项串的 token id 序列，以及它跨边界合并的情况。

    context 是选项前面那段真实文本（强制前缀）。BPE 在边界处会合并，所以选项
    的写法要按“上下文 + 选项”整段编码再减掉上下文长度来取；不能把选项单独
    编码——实测 'c0"' 单独编码是 ['c','0','"']，在 '{"filter_field": "' 之后
    也是 ['c','0','"']，但这必须实测确认，不能假设。
    """
    if not context:
        return (
            {name: tokenizer.encode(text, add_special_tokens=False) for name, text in options.items()},
            {name: "clean" for name in options},
        )
    base = tokenizer.encode(context, add_special_tokens=False)
    seqs: dict[str, list[int]] = {}
    notes: dict[str, str] = {}
    for name, text in options.items():
        full = tokenizer.encode(context + text, add_special_tokens=False)
        if full[: len(base)] != base:
            # 跨边界合并：退化为“整段编码去掉共同前缀”，并如实记录
            common = 0
            while common < len(base) and common < len(full) and base[common] == full[common]:
                common += 1
            seqs[name] = full[common:]
            notes[name] = f"merged@token{common}"
        else:
            seqs[name] = full[len(base) :]
            notes[name] = "clean"
    return seqs, notes


def path_scores(model, prefix_ids: list[int], seqs: dict[str, list[int]]) -> dict[str, dict]:
    """逐个选项走它自己的 token 路径，记录**全词表** softmax 下的串概率。

    每个选项单独 prefill 自己的路径（分支缓存需要复制 MLX 的 KV 缓冲，这里不做）；
    只用于“校准分析”，不计入实用读法的耗时。评分时必须把**上一个** token 喂回去
    再读当前位置的分布——喂错会让“重复刚说过的 token”这种倾向污染分数。
    """
    out: dict[str, dict] = {}
    for name, seq in seqs.items():
        if not seq:
            continue
        cache = make_prompt_cache(model)
        logits = forward(model, prefix_ids, cache=cache)
        logp = 0.0
        for step, tok in enumerate(seq):
            if step:
                logits = forward(model, [seq[step - 1]], cache=cache)
            p = float(softmax_full(logits)[tok])
            logp += float(mx.log(mx.array(max(p, 1e-30))))
        out[name] = {
            "n_tokens": len(seq),
            "p_path": float(mx.exp(mx.array(logp))),
            "logp": logp,
        }
    return out


# ---------------------------------------------------------------- 受限读取
@dataclass
class Readout:
    kind: str
    case: str
    pick: str | None  # 选中的候选 id；"null" 表示不过滤
    expect: str | None
    ok: bool
    detail: dict = field(default_factory=dict)
    seconds: float = 0.0
    forwards: int = 0


def greedy_constrained(
    model, tokenizer, prefix_ids: list[int], seqs: dict[str, list[int]]
) -> tuple[str, dict, int]:
    """受限贪心：共享一个 KV 缓存，每步只在“当前还可能的 token”里取最大。

    这是实用的读法：1 次 prefill + 选项串每个后续 token 各 1 步；当还活着的
    选项剩下同一段后缀（宿主写死的收尾引号）时提前停手，不白跑一次前向。
    """
    cache = make_prompt_cache(model)
    logits = forward(model, prefix_ids, cache=cache)
    forwards = 1
    alive = list(seqs)
    depth = 0
    trace: list[dict] = []
    first_groups: dict[str, float] = {}
    while True:
        next_tokens = sorted({seqs[name][depth] for name in alive})
        probs = masked_dist(logits, next_tokens)
        if depth == 0:
            first_groups = {tokenizer.decode([tok]): round(p, 4) for tok, p in zip(next_tokens, probs)}
        winner_tok = next_tokens[max(range(len(next_tokens)), key=lambda i: probs[i])]
        alive = [name for name in alive if seqs[name][depth] == winner_tok]
        trace.append(
            {
                "depth": depth,
                "options": [tokenizer.decode([t]) for t in next_tokens],
                "probs": [round(p, 4) for p in probs],
                "picked": tokenizer.decode([winner_tok]),
                "alive": list(alive),
            }
        )
        rest = {tuple(seqs[name][depth + 1 :]) for name in alive}
        if len(rest) == 1:
            # 还活着的选项后面是同一段后缀（宿主写死的收尾引号），谁活着谁就是答案
            return alive[0], {"trace": trace, "first_groups": first_groups}, forwards
        depth += 1
        logits = forward(model, [winner_tok], cache=cache)
        forwards += 1


def forced_prefix_ids(tokenizer, messages: list[dict[str, str]], tail: str) -> list[int]:
    """与基线相同的 chat 提示，末尾接上宿主写死的 JSON 前缀。"""
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return tokenizer.encode(text + tail, add_special_tokens=False)


def natural_options(candidates: Candidates) -> dict[str, str]:
    """候选在 JSON 里的真实写法的剩余部分：字符串内容 + 收尾引号。"""
    options = {cand.id: f'{cand.id}"' for cand in candidates.conditions}
    options["null"] = 'null"'
    return options


def digit_messages(instruction: str, candidates: Candidates) -> list[dict[str, str]]:
    """把候选 id 换成单个数字 token 的同款提示（宿主本来就在生成 id）。"""
    conds = candidates.conditions
    fields = candidates.fields
    lines = [
        f"任务：{instruction.strip()}",
        "",
        f"函数：{candidates.function_id}={candidates.function_name}",
        "字段候选（return_fields 只能选这里）："
        + ", ".join(f"{i}={cand.name}" for i, cand in enumerate(fields)),
        "条件候选（filter_field 只能选这里）："
        + ", ".join(f"{i}={cand.name}" for i, cand in enumerate(conds)),
        "只回 JSON，不要解释。",
    ]
    return [
        {
            "role": "system",
            "content": (
                "你只回一个 JSON 对象，不写代码、不解释、不加围栏。\n"
                "只能使用给出的候选编号，不得发明新的编号或字段名。\n"
                "只回这三个键：function（函数 id）、filter_field（条件编号，不过滤时用 null）、"
                "return_fields（字段编号的数组，按要求的输出顺序）。\n"
                "正确形状示例："
                '{"function": "fn0", "filter_field": "1", "return_fields": ["2", "3"]}'
            ),
        },
        {"role": "user", "content": "\n".join(lines)},
    ]


def digit_options(candidates: Candidates) -> dict[str, str]:
    """数字编号下的选项：数字 + 收尾引号；不过滤用 JSON 字符串 "null"。"""
    options = {cand.id: f'{i}"' for i, cand in enumerate(candidates.conditions)}
    options["null"] = 'null"'
    return options


def digits_of(candidates: Candidates) -> dict[str, str]:
    """候选 id -> 数字编号的显示（给报告用）。"""
    return {cand.id: str(i) for i, cand in enumerate(candidates.conditions)}


def run_readout(
    kind: str, model, tokenizer, case: Case, messages: list[dict[str, str]], options: dict[str, str]
) -> tuple[Readout, Candidates]:
    """强制前缀 + 受限读取，输出过滤字段决策。"""
    candidates = extract(case.source, case.function)
    prefix = forced_prefix_ids(tokenizer, messages, JSON_TAIL)
    seqs, notes = option_token_seqs(tokenizer, options, context=JSON_TAIL)

    t0 = time.perf_counter()
    pick, detail, forwards = greedy_constrained(model, tokenizer, prefix, seqs)
    seconds = time.perf_counter() - t0

    scores = path_scores(model, prefix, seqs)
    norm = {k: v["logp"] / v["n_tokens"] for k, v in scores.items()}
    names = {c.id: c.name for c in candidates.conditions}
    expect_id = _id_for(candidates, case.expect_filter)
    return (
        Readout(
            kind=kind,
            case=case.key,
            pick=pick,
            expect=expect_id,
            ok=pick == expect_id,
            detail={
                "filter_name": None if pick == "null" else names.get(pick),
                "expect_id": expect_id,
                "correct_index": _index_of(candidates, expect_id),
                "n_conditions": len(candidates.conditions),
                "digit_ids": digits_of(candidates),
                "option_seq_len": {k: len(v) for k, v in seqs.items()},
                "boundary": notes,
                "first_groups": detail["first_groups"],
                "p_path": {
                    k: float(f"{v['p_path']:.6g}")
                    for k, v in sorted(scores.items(), key=lambda kv: -kv[1]["p_path"])
                },
                "p_path_norm": {
                    k: float(f"{mx.exp(mx.array(v)).item():.6g}")
                    for k, v in sorted(norm.items(), key=lambda kv: -kv[1])
                },
                "p_correct_path": float(f"{scores[expect_id]['p_path']:.6g}"),
                "p_pick_path": float(f"{scores[pick]['p_path']:.6g}"),
                "pick_raw_path": max(scores, key=lambda k: scores[k]["p_path"]) if scores else None,
                "pick_norm": max(norm, key=lambda k: norm[k]) if norm else None,
                "forwards": forwards,
                "trace": detail["trace"],
            },
            seconds=seconds,
            forwards=forwards,
        ),
        candidates,
    )


def run_v1(model, tokenizer, case: Case) -> tuple[Readout, Candidates]:
    """V1：现有 id 写法（c0/c1/…），强制 JSON 前缀后受限读取。"""
    candidates = extract(case.source, case.function)
    messages = build_decision_prompt(case.instruction, candidates)
    return run_readout("V1", model, tokenizer, case, messages, natural_options(candidates))


def run_v2(model, tokenizer, case: Case) -> tuple[Readout, Candidates]:
    """V2：候选重新编号成单个数字 token → 真正的单次前向。"""
    candidates = extract(case.source, case.function)
    messages = digit_messages(case.instruction, candidates)
    return run_readout("V2", model, tokenizer, case, messages, digit_options(candidates))


def _id_for(candidates: Candidates, name: str | None) -> str:
    if name is None:
        return "null"
    for cand in candidates.conditions:
        if cand.name == name:
            return cand.id
    raise KeyError(name)


def _index_of(candidates: Candidates, cand_id: str) -> int:
    for i, cand in enumerate(candidates.conditions):
        if cand.id == cand_id:
            return i
    return -1


# ---------------------------------------------------------------- V3：是/否
def field_question_ids(
    tokenizer, instruction: str, candidates: Candidates
) -> tuple[list[int], list[str], list[int]]:
    """一条序列里问完所有字段；每个问题以「答：」结尾，位置可精确记录。"""
    head = (
        f"任务：{instruction.strip()}\n"
        f"函数：{candidates.function_id}={candidates.function_name}\n"
        "对每个字段回答：这个字段要不要放进返回结果里。只回「是」或「否」。\n"
    )
    ids = tokenizer.encode(
        tokenizer.apply_chat_template(
            [{"role": "user", "content": head}], tokenize=False, add_generation_prompt=True
        ),
        add_special_tokens=False,
    )
    names: list[str] = []
    marks: list[int] = []
    for cand in candidates.fields:
        segment = tokenizer.encode(
            f"返回字段里要不要包含 {cand.name}？答：", add_special_tokens=False
        )
        ids = ids + segment
        names.append(cand.name)
        marks.append(len(ids) - 1)  # 该问题答案分布所在的位置
    return ids, names, marks


def run_v3(model, tokenizer, case: Case) -> tuple[Readout, Candidates]:
    """一次前向读完所有字段的是/否分布。"""
    candidates = extract(case.source, case.function)
    ids, names, marks = field_question_ids(tokenizer, case.instruction, candidates)
    yes_ids = _single(tokenizer, ["是", " 是"])
    no_ids = _single(tokenizer, ["否", " 否"])

    t0 = time.perf_counter()
    logits_all = model(mx.array([ids]))
    mx.eval(logits_all)
    mx.synchronize()
    seconds = time.perf_counter() - t0

    rows = []
    chosen: set[str] = set()
    for name, pos in zip(names, marks):
        row = logits_all[0, pos, :]
        probs = masked_dist(row, yes_ids + no_ids)
        p_yes = sum(probs[: len(yes_ids)])
        p_no = sum(probs[len(yes_ids) :])
        total = p_yes + p_no or 1.0
        p_yes, p_no = p_yes / total, p_no / total
        if p_yes > p_no:
            chosen.add(name)
        rows.append(
            {
                "field": name,
                "p_yes": round(p_yes, 4),
                "p_no": round(p_no, 4),
                "decision": "是" if p_yes > p_no else "否",
            }
        )
    expect_ids = {c.id for c in candidates.fields if c.name in case.expect_fields}
    chosen_ids = {c.id for c in candidates.fields if c.name in chosen}
    return (
        Readout(
            kind="V3",
            case=case.key,
            pick=None,
            expect=None,
            ok=chosen == set(case.expect_fields),
            detail={
                "questions": rows,
                "chosen_names": sorted(chosen),
                "expect_names": sorted(case.expect_fields),
                "chosen_ids": sorted(chosen_ids),
                "expect_ids": sorted(expect_ids),
            },
            seconds=seconds,
            forwards=1,
        ),
        candidates,
    )


def _single(tokenizer, texts: list[str]) -> list[int]:
    """只保留单 token 的写法，返回它们的 id。"""
    out = []
    for text in texts:
        enc = tokenizer.encode(text, add_special_tokens=False)
        if len(enc) == 1:
            out.append(enc[0])
    return out


def timing_breakdown(model, tokenizer, prefix_ids: list[int]) -> dict[str, float]:
    """把一次前向拆成 prefill 与单 token 步：耗时到底花在哪。"""
    t0 = time.perf_counter()
    forward(model, prefix_ids)
    prefill = time.perf_counter() - t0
    cache = make_prompt_cache(model)
    forward(model, prefix_ids, cache=cache)
    tok = prefix_ids[-1:]
    t0 = time.perf_counter()
    for _ in range(5):
        forward(model, tok, cache=cache)
    step = (time.perf_counter() - t0) / 5
    return {"prefill_seconds": prefill, "step_seconds": step, "prefix_tokens": len(prefix_ids)}


# ------------------------------------------------- V4：返回字段按槽位逐个读
def run_v4(
    model, tokenizer, case: Case, filter_pick: str, filter_text_by_id: dict[str, str]
) -> tuple[Readout, Candidates]:
    """返回字段的有序读法：一个槽位一次受限读取，直到「停」选项胜出。

    与 V3 的是/否集合不同，这里直接读 JSON 数组里下一个元素该是谁，因此
    顺序天然带出来；代价是每个槽位一次前向。
    """
    candidates = extract(case.source, case.function)
    messages = digit_messages(case.instruction, candidates)
    chat = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    filter_text = filter_text_by_id.get(filter_pick, filter_pick)
    fields_with_digits = [(str(i), cand) for i, cand in enumerate(candidates.fields)]

    chosen: list[str] = []
    used: set[str] = set()
    forwards = 0
    slots: list[dict] = []
    t0 = time.perf_counter()
    while True:
        prefix = chat + (
            '{"function": "fn0", "filter_field": "'
            + filter_text
            + '", "return_fields": ["'
            + "".join(f'{d}", "' for d in chosen)
        )
        prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
        options = {cand.id: f'{d}"' for d, cand in fields_with_digits if cand.id not in used}
        options["stop"] = '"'  # 结束这个字符串（宿主随后补 ]），数组到此为止
        # 前缀尾部的引号可能与选项合并，所以按完整前缀整段编码再取后缀
        seqs, notes = option_token_seqs(tokenizer, options, context=prefix)
        pick, detail, fw = greedy_constrained(model, tokenizer, prefix_ids, seqs)
        forwards += fw
        if pick == "stop" or not options:
            slots.append({"slot": len(chosen), "picked": "stop", "dist": detail["first_groups"]})
            break
        chosen.append(next(d for d, c in fields_with_digits if c.id == pick))
        used.add(pick)
        slots.append(
            {"slot": len(chosen) - 1, "picked": pick, "name": _name_of_id(candidates, pick), "dist": detail["first_groups"]}
        )
        if len(used) == len(fields_with_digits):
            break
    seconds = time.perf_counter() - t0
    names = [_name_of_id(candidates, fid) for fid in chosen]
    expect_order = list(case.expect_fields)
    ok = set(names) == set(case.expect_fields) and len(names) == len(case.expect_fields)
    return (
        Readout(
            kind="V4",
            case=case.key,
            pick=None,
            expect=None,
            ok=ok,
            detail={
                "chosen_ids": chosen,
                "chosen_names": names,
                "expect_names": sorted(case.expect_fields),
                "order_matches_expect_set": names == expect_order,
                "slots": slots,
                "forwards": forwards,
            },
            seconds=seconds,
            forwards=forwards,
        ),
        candidates,
    )


def _name_of_id(candidates: Candidates, cand_id: str) -> str:
    for cand in candidates.fields:
        if cand.id == cand_id:
            return cand.name
    return cand_id


# ---------------------------------------------------------------- 运行时核对
def decision_text(function_id: str, filter_id: str | None, field_ids: list[str]) -> str:
    """把读出来的 id 拼回决策 JSON，走真实的 parse_decision 校验路径。"""
    return json.dumps(
        {"function": function_id, "filter_field": filter_id, "return_fields": field_ids},
        ensure_ascii=False,
    )


def runtime_check(source: str, candidates: Candidates, decision: Decision, case: Case) -> tuple[bool, str]:
    """assemble → exec → 真调用 → 与期望行比对；与 compare.py 同一套做法。"""
    try:
        body = assemble(source, candidates, decision)
    except DecisionError as exc:
        return False, f"组装失败: {exc}"
    try:
        tree = ast.parse(body)
        ns: dict[str, object] = {}
        exec(compile(tree, "<candidate>", "exec"), ns)  # noqa: S102 - 只跑实验样本
    except Exception as exc:  # noqa: BLE001
        return False, f"无法执行: {type(exc).__name__}: {exc}"
    fn = ns.get(case.function)
    if not callable(fn):
        return False, "找不到函数"
    try:
        out = fn(list(case.sample))
    except Exception as exc:  # noqa: BLE001
        return False, f"调用失败: {type(exc).__name__}: {exc}"
    if not isinstance(out, list) or not all(isinstance(r, dict) for r in out):
        return False, f"返回值不是字典列表: {out!r}"
    if case.expect_filter is None:
        want = [{k: row[k] for k in case.expect_fields} for row in case.sample]
    else:
        want = [
            {k: row[k] for k in case.expect_fields}
            for row in case.sample
            if row[case.expect_filter]
        ]
    if case.sort_by:
        want.sort(key=lambda r: r[case.sort_by], reverse=True)
    ok = out == want
    return ok, "" if ok else f"want={want} got={out}"


def decision_from_picks(candidates: Candidates, filter_pick: str, field_ids: list[str]) -> Decision | None:
    """把读出来的 id 拼成决策 JSON，走真实的 parse_decision 校验路径。"""
    filter_id = None if filter_pick in ("null", None) else filter_pick
    if not field_ids:
        print("    V3 没选出任何字段，return_fields 为空会被 parse_decision 拒绝")
        return None
    try:
        return parse_decision(
            decision_text(candidates.function_id, filter_id, field_ids), candidates
        )
    except DecisionError as exc:
        print(f"    parse_decision 拒绝: {exc}")
        return None


def v3_field_ids(candidates: Candidates, readout: Readout) -> list[str]:
    """V3 只给出集合；顺序按候选表顺序补齐（顺序必须由宿主另想办法）。"""
    chosen = set(readout.detail["chosen_names"])
    return [c.id for c in candidates.fields if c.name in chosen]


def quality_check(case: Case, candidates: Candidates, decision: Decision) -> tuple[bool, str]:
    """任务 A 用 compare.py 的原版检查；其它 case 用通用比对。"""
    if case.key == "A_active":
        body = assemble(case.source, candidates, decision)
        ok, why = check_runtime_behaviour(body, case.function)
        return ok, ("; ".join(why) if why else "")
    return runtime_check(case.source, candidates, decision, case)


# ---------------------------------------------------------------- 分词检查
def token_probe(tokenizer) -> None:
    print("=== 分词检查（假设的关键实测点）===")
    probes = [
        ("fn0", "fn0"),
        ("f0", "f0"),
        ("f1", "f1"),
        ("f2", "f2"),
        ("c0", "c0"),
        ("c1", "c1"),
        ("带空格 f0", " f0"),
        ("带空格 c1", " c1"),
        ("带空格 fn0", " fn0"),
        ("JSON 内 fn0", '"fn0"'),
        ("JSON 内 f0", '"f0"'),
        ("JSON 内 c0", '"c0"'),
        ("null", "null"),
        ("带空格 null", " null"),
        ("数字 0", "0"),
        ("带空格 0", " 0"),
        ("是", "是"),
        ("否", "否"),
    ]
    for label, text in probes:
        ids = tokenizer.encode(text, add_special_tokens=False)
        pieces = [tokenizer.decode([i]) for i in ids]
        flag = "单 token" if len(ids) == 1 else f"{len(ids)} 个 token"
        print(f"  {label:14s} {text!r:10s} ids={ids} pieces={pieces} → {flag}")


# ---------------------------------------------------------------- 主流程
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--token-only", action="store_true")
    parser.add_argument("--baseline-reps", type=int, default=5)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    print(f"加载模型 {args.model} ...")
    t0 = time.perf_counter()
    model, tokenizer = load(args.model)
    print(f"加载完成 {time.perf_counter() - t0:.2f}s\n")

    token_probe(tokenizer)
    if args.token_only:
        return 0

    # 预热：第一次前向带编译开销，先跑掉
    warm = tokenizer.encode("预热", add_special_tokens=False)
    mx.eval(model(mx.array([warm])))
    mx.synchronize()
    forward(model, warm)

    print("\n=== V1：强制 JSON 前缀 + 受限读取（现有 id 写法 c0/c1/…）===")
    v1_rows: list[Readout] = []
    v1_cands: dict[str, Candidates] = {}
    for case in CASES:
        r, cands = run_v1(model, tokenizer, case)
        v1_rows.append(r)
        v1_cands[case.key] = cands
        d = r.detail
        print(
            f"  [{case.key:16s}] {'OK ' if r.ok else 'BAD'} "
            f"贪心选={d['filter_name']} 期望={case.expect_filter} "
            f"首token={d['first_groups']}\n"
            f"      串概率={d['p_path']}\n"
            f"      串概率argmax={d['pick_raw_path']} 归一化argmax={d['pick_norm']} "
            f"p_correct={d['p_correct_path']:.3g} p_pick={d['p_pick_path']:.3g} "
            f"前向{d['forwards']}次 {r.seconds * 1000:.0f}ms 边界={set(d['boundary'].values())}"
        )

    print("\n=== V2：候选重新编号成单个数字 token（真正 1 次前向）===")
    v2_rows: list[Readout] = []
    v2_cands: dict[str, Candidates] = {}
    for case in CASES:
        r, cands = run_v2(model, tokenizer, case)
        v2_rows.append(r)
        v2_cands[case.key] = cands
        d = r.detail
        print(
            f"  [{case.key:16s}] {'OK ' if r.ok else 'BAD'} "
            f"选={d['filter_name']} 期望={case.expect_filter} "
            f"首token={d['first_groups']} p_correct={d['p_correct_path']:.3g} "
            f"前向{d['forwards']}次 {r.seconds * 1000:.0f}ms 边界={set(d['boundary'].values())}"
        )

    print("\n=== V3：是/否 问题写进同一条序列，一次前向读完（返回字段集合）===")
    v3_rows: list[Readout] = []
    for case in CASES:
        r, _ = run_v3(model, tokenizer, case)
        v3_rows.append(r)
        detail = ", ".join(
            f"{q['field']}:{'是' if q['decision'] == '是' else '否'}(p_yes={q['p_yes']})"
            for q in r.detail["questions"]
        )
        print(
            f"  [{case.key:16s}] {'OK ' if r.ok else 'BAD'} "
            f"选了={r.detail['chosen_names']} 期望={r.detail['expect_names']} | {detail} | {r.seconds * 1000:.0f}ms"
        )

    print("\n=== V4：返回字段按槽位逐个读（顺序天然带出，每槽一次前向）===")
    v4_rows: list[Readout] = []
    for case in CASES:
        v2 = next(r for r in v2_rows if r.case == case.key)
        cands = v2_cands[case.key]
        fmap = {cand.id: str(i) for i, cand in enumerate(cands.conditions)}
        r, _ = run_v4(model, tokenizer, case, v2.pick, fmap)
        v4_rows.append(r)
        slots = " → ".join(
            f"{s.get('name', s['picked'])}[{s['dist']}]" for s in r.detail["slots"]
        )
        print(
            f"  [{case.key:16s}] {'OK ' if r.ok else 'BAD'} "
            f"选了={r.detail['chosen_names']} 期望={r.detail['expect_names']} "
            f"顺序={r.detail['order_matches_expect_set']} | {slots} | "
            f"前向{r.forwards}次 {r.seconds * 1000:.0f}ms"
        )

    # ---------------- 一次前向花在哪 ----------------
    print("\n=== 前向耗时拆分（prefill 与单 token 步）===")
    parts = []
    for case in CASES[:4]:
        v1 = next(r for r in v1_rows if r.case == case.key)
        cands = v1_cands[case.key]
        prefix = forced_prefix_ids(
            tokenizer, build_decision_prompt(case.instruction, cands), JSON_TAIL
        )
        parts.append(timing_breakdown(model, tokenizer, prefix))
    avg_prefill = sum(p["prefill_seconds"] for p in parts) / len(parts)
    avg_step = sum(p["step_seconds"] for p in parts) / len(parts)
    avg_tokens = sum(p["prefix_tokens"] for p in parts) / len(parts)
    print(
        f"  前缀平均 {avg_tokens:.0f} tokens：一次 prefill {avg_prefill * 1000:.0f}ms；"
        f"之后每个单 token 步 {avg_step * 1000:.0f}ms（同样条件下 36 步解码 ≈ {avg_step * 36 * 1000:.0f}ms）"
    )

    # ---------------- 端到端运行时核对 ----------------
    print("\n=== 端到端：V1 的过滤 + V3 的字段 → parse_decision → assemble → 真调用 ===")
    combo_ok = 0
    combo_total = 0
    combo_rows: list[dict] = []
    for case in CASES:
        v1 = next(r for r in v1_rows if r.case == case.key)
        v3 = next(r for r in v3_rows if r.case == case.key)
        cands = v1_cands[case.key]
        fids = v3_field_ids(cands, v3)
        decision = decision_from_picks(cands, v1.pick, fids)
        if case.decision_only:
            print(
                f"  [{case.key:16s}] 仅决策：过滤选={v1.pick} 期望={v1.expect} "
                f"{'一致' if v1.ok else '不一致'}（现有渲染表达不了该指令，不跑运行时）"
            )
            continue
        combo_total += 1
        ok, why = (False, "决策不合法") if decision is None else runtime_check(case.source, cands, decision, case)
        combo_ok += ok
        combo_rows.append(
            {"case": case.key, "filter": v1.pick, "fields": fids, "runtime_ok": ok, "why": why}
        )
        print(f"  [{case.key:16s}] 过滤={v1.pick} 字段={fids} → {'OK' if ok else 'BAD ' + why}")

    # ---------------- 头对头：同一固定任务 ----------------
    print("\n=== 头对头（同一固定任务 = bench/compare.py 任务 A）===")
    case_a = CASES[0]
    cands_a = v1_cands[case_a.key]
    head: list[dict] = []

    # 1) 基线：自回归生成 JSON（走产品代码 MLXEngine.generate，复用已加载权重）
    engine = MLXEngine(args.model)
    engine._model, engine._tokenizer = model, tokenizer  # noqa: SLF001 - 复用已加载权重
    gen_ok = 0
    gen_wall: list[float] = []
    gen_tok: list[int] = []
    gen_fail: list[str] = []
    for i in range(args.baseline_reps):
        t0 = time.perf_counter()
        raw, stats = engine.generate(
            build_decision_prompt(case_a.instruction, cands_a), max_tokens=DECISION_MAX_TOKENS
        )
        wall = time.perf_counter() - t0
        gen_wall.append(wall)
        gen_tok.append(stats.generated_tokens)
        if i == 0:
            print(f"  [基线 第1次输出] {raw[:160]!r}")
        try:
            decision = parse_decision(raw, cands_a)
        except DecisionError as exc:
            gen_fail.append(f"决策不合法: {exc}")
            continue
        ok_rt, why = quality_check(case_a, cands_a, decision)
        gen_ok += ok_rt
        if not ok_rt:
            gen_fail.append(why)
    head.append(
        {
            "method": "基线：自回归生成 JSON（现状）",
            "runtime": f"{gen_ok}/{args.baseline_reps}",
            "seconds": sum(gen_wall) / len(gen_wall),
            "tokens": f"{sorted(set(gen_tok))} tokens",
            "forwards": "≈36 次解码步",
            "failures": gen_fail,
        }
    )

    # 2) V1 + V3
    v1a = next(r for r in v1_rows if r.case == case_a.key)
    v3a = next(r for r in v3_rows if r.case == case_a.key)
    fids_v3 = v3_field_ids(cands_a, v3a)
    dec = decision_from_picks(cands_a, v1a.pick, fids_v3)
    ok_rt, why = (False, "决策不合法") if dec is None else quality_check(case_a, cands_a, dec)
    head.append(
        {
            "method": "V1 强制前缀读过滤 + V3 是/否读字段（集合）",
            "runtime": "1/1" if ok_rt else f"0/1（{why}）",
            "seconds": v1a.seconds + v3a.seconds,
            "tokens": "0 生成",
            "forwards": f"{v1a.forwards + v3a.forwards} 次（过滤 {v1a.forwards} + 字段 1）",
            "failures": [] if ok_rt else [why],
        }
    )

    # 3) V2 + V4（数字编号读过滤，字段按槽位逐个读）
    v2a = next(r for r in v2_rows if r.case == case_a.key)
    fmap_a = {cand.id: str(i) for i, cand in enumerate(cands_a.conditions)}
    v4a, _ = run_v4(model, tokenizer, case_a, v2a.pick, fmap_a)
    dec2 = decision_from_picks(cands_a, v2a.pick, v4a.detail["chosen_ids"])
    ok_rt2, why2 = (False, "决策不合法") if dec2 is None else quality_check(case_a, cands_a, dec2)
    head.append(
        {
            "method": "V2 数字编号读过滤 + V4 槽位读字段",
            "runtime": "1/1" if ok_rt2 else f"0/1（{why2}）",
            "seconds": v2a.seconds + v4a.seconds,
            "tokens": "0 生成",
            "forwards": f"{v2a.forwards + v4a.forwards} 次（过滤 {v2a.forwards} + 字段 {v4a.forwards}）",
            "failures": [] if ok_rt2 else [why2],
        }
    )

    # 4) V1 + V4（现有 id 写法）
    dec4 = decision_from_picks(cands_a, v1a.pick, v4a.detail["chosen_ids"])
    ok_rt4, why4 = (False, "决策不合法") if dec4 is None else quality_check(case_a, cands_a, dec4)
    head.append(
        {
            "method": "V1 强制前缀读过滤 + V4 槽位读字段",
            "runtime": "1/1" if ok_rt4 else f"0/1（{why4}）",
            "seconds": v1a.seconds + v4a.seconds,
            "tokens": "0 生成",
            "forwards": f"{v1a.forwards + v4a.forwards} 次（过滤 {v1a.forwards} + 字段 {v4a.forwards}）",
            "failures": [] if ok_rt4 else [why4],
        }
    )

    print(f"{'方法':44s} {'运行时':14s} {'耗时':9s} {'生成':16s} 前向")
    for row in head:
        print(
            f"{row['method']:44s} {row['runtime']:14s} {row['seconds']:.3f}s   "
            f"{row['tokens']:16s} {row['forwards']}"
        )

    # ---------------- 汇总 ----------------
    print("\n=== 汇总 ===")
    for label, rows in (("V1", v1_rows), ("V2", v2_rows), ("V3", v3_rows), ("V4", v4_rows)):
        ok = sum(1 for r in rows if r.ok)
        secs = [r.seconds for r in rows]
        print(
            f"{label}: {ok}/{len(rows)} 通过，平均 {sum(secs) / len(secs) * 1000:.0f}ms，"
            f"前向 {sorted({r.forwards for r in rows})} 次"
        )
    print(
        f"V3 字段集合（是/否 门限 0.5）：{sum(1 for r in v3_rows if r.ok)}/{len(v3_rows)}；"
        f"V4 字段有序槽位：{sum(1 for r in v4_rows if r.ok)}/{len(v4_rows)}"
    )
    for label, rows in (("V1", v1_rows), ("V2", v2_rows)):
        raw_ok = sum(1 for r in rows if r.detail["pick_raw_path"] == r.expect)
        norm_ok = sum(1 for r in rows if r.detail["pick_norm"] == r.expect)
        print(
            f"{label} 变体：贪心受限解码 {sum(1 for r in rows if r.ok)}/{len(rows)}，"
            f"全词表串概率 argmax {raw_ok}/{len(rows)}，长度归一化 argmax {norm_ok}/{len(rows)}"
        )
    print(f"端到端（V1过滤 + V3字段，运行时行为）：{combo_ok}/{combo_total}")
    combo4_ok = 0
    combo4_total = 0
    for case in CASES:
        if case.decision_only:
            continue
        v2 = next(r for r in v2_rows if r.case == case.key)
        v4 = next(r for r in v4_rows if r.case == case.key)
        cands = v2_cands[case.key]
        dec = decision_from_picks(cands, v2.pick, v4.detail["chosen_ids"])
        combo4_total += 1
        if dec is None:
            continue
        ok, _why = runtime_check(case.source, cands, dec, case)
        combo4_ok += ok
    print(f"端到端（V2过滤 + V4字段，运行时行为）：{combo4_ok}/{combo4_total}")

    if args.json:
        print("\n完整明细（JSON）：")
        print(
            json.dumps(
                {
                    "V1": [{"case": r.case, "ok": r.ok, **r.detail} for r in v1_rows],
                    "V2": [{"case": r.case, "ok": r.ok, **r.detail} for r in v2_rows],
                    "V3": [{"case": r.case, "ok": r.ok, **r.detail} for r in v3_rows],
                    "V4": [{"case": r.case, "ok": r.ok, **r.detail} for r in v4_rows],
                    "combo": combo_rows,
                    "head_to_head": head,
                },
                ensure_ascii=False,
                indent=1,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
