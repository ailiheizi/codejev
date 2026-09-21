"""片大小实验：小模型一次调用最多能可靠处理多大的一块。

计划中的架构是"大模型拆任务 → 告诉小模型每一块的做法 → 小模型回结果 → 大模型判断"。
拆分的直接代价是**每一块都要一次完整的往返**（本机基线约 0.60 s/次，模型常驻）。
于是关键未知数是：**一块能有多大，小模型才开始做错**。

本文件产出三条曲线，全部用本机真跑的数字：

  A. 选择路线：候选数量 N（3/6/9/12/15），每个 N 至少 8 次试验，且不少于 N 次，
     保证正确答案在候选表里的**每一个位置**都被测到；
  B. 生成路线：函数体长度（算出来的实际跨度约 6/19/32/58 行；前两档由形状决定——
     校准档无守卫/排序/上限，其余档带，之后只增加填充行），每个大小至少 5 次试验；
  C. 拆分成本：每次调用的固定底价（含提示处理的底），以及 N 块 = N 倍底价的算术。

正确率一律用**运行时行为**判断：真的 exec 产物、真的调用函数，检查过滤结果、
返回字段、守卫子句、循环后的排序、输出上限；不看代码像不像。温度 0。

温度 0 下同一输入逐字相同，所以每次试验都换一份**不同的输入**（不同字段名、
不同函数名、不同正确答案的位置、不同填充文字），否则多次重复只是同一个样本。

用法（项目根目录）：

    HF_HUB_OFFLINE=1 .venv/bin/python -m bench.piecesize --curve all
    HF_HUB_OFFLINE=1 .venv/bin/python -m bench.piecesize --selftest   # 不加载权重

本脚本只读现有模块（extract / build_decision_prompt / parse_decision / assemble 原样复用），
不修改任何 azfls 代码。结果只是单机小样本观测，机器当时还有其他负载，不是通用结论。
"""

from __future__ import annotations

import argparse
import ast
import json
import random
import re
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from string import Template

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:  # 直接以脚本方式运行时，也能 import azfls
    sys.path.insert(0, str(ROOT))

from azfls.adapter import to_artifact  # noqa: E402 - 先修好 sys.path 再导入
from azfls.contracts import Action, Brief, Kind  # noqa: E402
from azfls.decide import (  # noqa: E402
    DECISION_MAX_TOKENS,
    DECISION_SYSTEM_PROMPT,
    Decision,
    DecisionError,
    assemble,
    build_decision_prompt,
    extract,
    parse_decision,
)
from azfls.model import Engine, MLXEngine, Stats, request_body  # noqa: E402
from bench.compare import looks_like_explanation, looks_truncated  # noqa: E402

MODEL_15B = ROOT / "models" / "Qwen2.5-Coder-1.5B-Instruct-4bit"
MODEL_05B = ROOT / "models" / "Qwen2.5-Coder-0.5B-Instruct-4bit"

# 选择路线的输出上限：与生产默认一致（决策只有 30–50 token，上限只是防失控）。
SELECT_MAX_TOKENS = DECISION_MAX_TOKENS
# 生成路线的输出上限：各大小变体用同一个上限，免得"被截断"混进"大小效应"。
GENERATE_MAX_TOKENS = 1536
# 每次试验前先按同一形状空跑一次（max_tokens=1），把 MLX 编译图的一次性成本排除在稳态之外。
WARMUP_MAX_TOKENS = 1
def _system_example(prompt: str) -> str:
    """从系统提示里现取示例 JSON：提示改了这里跟着改，不会留着一份过期的对照串。

    提示里只有一个 JSON 对象（当前的正确形状示例）。取不到就返回空串，
    此时"复述示例"这一类计数恒为 0，而不是拿一句早已不在提示里的话去比对。
    """
    match = re.search(r"\{[^{}]*\}", prompt)
    return match.group(0) if match else ""


# 系统提示里的示例原文：模型犯懒时会逐字复述它，这里单列一类失败。
SYSTEM_EXAMPLE = _system_example(DECISION_SYSTEM_PROMPT)


def _example_ids(example: str) -> tuple[str | None, tuple[str, ...]]:
    """示例里的条件 id 与字段 id；取不到就给空，绝不让诊断字段把整轮跑挂掉。"""
    try:
        obj = json.loads(example)
    except ValueError:
        return None, ()
    if not isinstance(obj, dict):
        return None, ()
    fields = obj.get("r")
    return obj.get("f"), tuple(fields) if isinstance(fields, list) else ()


# 示例本身给出的"答案"：模型照抄示例时，回的就是这一组 id。
# 这正是文档里记过的失败模式（0.5B 在任务 A 上 10/10 都是示例那句），
# 所以按决策 id 判定，而不是按文本是否逐字相同。
EXAMPLE_FILTER_ID, EXAMPLE_FIELD_IDS = _example_ids(SYSTEM_EXAMPLE)


class Record(dict):
    """同时支持 `row["total"]` 与 `row.total` 的记录对象。

    生成路线可能把下标取值改写成属性取值（或反过来），两种写法都能跑才算
    "其他不变"，不该因为取值风格不同就判失败。
    """

    def __getattr__(self, name: str) -> object:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


@dataclass
class Trial:
    """一次试验的实测记录；判断标准与耗时都写在这里，不藏到别处。"""

    curve: str  # "A" / "A2" / "B"
    point: str  # "N=6" / "~30 行"
    index: int
    ok: bool
    reasons: tuple[str, ...]
    raw: str
    expected: str
    wall_seconds: float
    stats: Stats
    detail: str = ""
    extra: dict = field(default_factory=dict)


# --------------------------------------------------------------------------
# 曲线 A：选择路线 —— 候选数量
# --------------------------------------------------------------------------

A_FIELD_POOL = (
    "id", "name", "active", "price", "qty", "sku", "region", "tier", "note", "weight",
    "color", "size", "brand", "stock", "rating", "tag", "owner", "code", "level", "score",
)
# 名字不能和宿主变量撞车：item / items / rows / result 一律不用（池子里没有）。
A_SIZES = (3, 6, 9, 12, 15)
A_TRIALS = 8
A_SHAPES = ("aligned", "misaligned")


@dataclass(frozen=True)
class ASpec:
    """一次选择路线试验的输入：源码、指令、正确答案、运行样本。"""

    n: int
    trial: int
    shape: str  # aligned：原文没有条件；misaligned：原文已有另一个条件
    fn_name: str
    names: tuple[str, ...]
    filter_name: str
    ret_first: str
    ret_second: str
    pre_condition: str | None
    instruction: str
    source: str
    sample: list[dict]
    expected_filter_id: str
    expected_field_ids: tuple[str, str]

    @property
    def expected_json(self) -> str:
        """期望决策的展示串，用生产契约的精简键（模型实际要回的就是这个形状）。"""
        return '{"f": "%s", "r": [%s]}' % (
            self.expected_filter_id,
            ", ".join(f'"{fid}"' for fid in self.expected_field_ids),
        )


def a_source(fn_name: str, names: tuple[str, ...], pre_condition: str | None = None) -> str:
    """N 个键的列表函数；原文没有任何过滤（或已有另一个条件）。"""
    lines = [
        f"def {fn_name}(items):",
        '    """返回记录列表，保持原顺序。"""',
        "    rows = []",
        "    for item in items:",
    ]
    append = [
        "        rows.append(",
        "            {",
    ]
    for name in names:
        append.append(f'                "{name}": item["{name}"],')
    append += ["            }", "        )"]
    if pre_condition is None:
        lines += append
    else:
        lines.append(f'        if item["{pre_condition}"] == "keep":')
        lines += ["    " + line for line in append]
    lines.append("    return rows")
    return "\n".join(lines) + "\n"


def a_sample(names: tuple[str, ...], filter_name: str, pre_condition: str | None) -> list[dict]:
    """5 条记录：过滤字段真真假假，其余字段每条都不同（过滤是否发生看得出来）。"""
    flags = [True, False, True, False, True]
    sample: list[dict] = []
    for index, flag in enumerate(flags):
        row: dict = {}
        for name in names:
            if name == filter_name:
                row[name] = flag
            elif name == pre_condition:
                # 原文已有的条件：取值让四种产物在运行时彼此可分辨，错法藏不住——
                #   保持原条件（== "keep"）→ 一条都不留；把候选当真假值判断 → 只留 1 的那两条；
                #   本次要求的过滤 → 留 flag 为真的三条；完全不过滤 → 五条全留。
                # 若取值与本次过滤恰好重合，"没换条件"在运行时与正确答案分不开。
                row[name] = 1 if index % 2 else 0
            else:
                row[name] = f"{name}-{index}"
        sample.append(row)
    return sample


def a_spec(n: int, trial: int, shape: str = "aligned") -> ASpec:
    """第 trial 次试验：换名字、换正确答案的位置、换返回字段的书写顺序。

    正确答案的位置按 trial % n 轮转；调用方保证每个 N 至少跑 n 次，
    于是候选表里**每一个位置**都当过正确答案（否则 N=15 只测前 8 个位置时，
    "后半段选不对"这类失败会被漏掉，正确率被高估）。
    """
    rng = random.Random(97_000 + 1000 * n + 10 * trial + (0 if shape == "aligned" else 1))
    names = tuple(rng.sample(A_FIELD_POOL, n))
    filter_pos = trial % n
    span = n - 1
    stride1 = 1 + (trial % span)
    stride2 = 1 + ((trial + 1) % span)
    if stride2 == stride1:
        stride2 = stride2 % span + 1
    first_pos = (filter_pos + stride1) % n
    second_pos = (filter_pos + stride2) % n
    filter_name = names[filter_pos]
    if trial % 2 == 0:
        ret_first, ret_second = names[first_pos], names[second_pos]
    else:  # 一半的试验把返回顺序颠倒，模型不能靠"按源码顺序抄"过关
        ret_first, ret_second = names[second_pos], names[first_pos]

    pre_condition = None
    if shape == "misaligned":
        taken = {filter_name, ret_first, ret_second}
        available = [name for name in names if name not in taken]
        if not available:  # 至少 4 个字段才腾得出一个"原文已有的条件"
            raise ValueError(f"候选表错位的形状至少需要 4 个字段，当前 N={n}")
        pre_condition = available[0]

    source = a_source(f"fn_{n}_{trial}", names, pre_condition)
    instruction = (
        f"修改函数：只保留 {filter_name} 为真的项，返回 {ret_first} 和 {ret_second}，"
        "保持原顺序，其他不变。"
    )
    # 正确答案的 id 从真实的候选表里取（字段顺序由 extract 决定，不是名字表的顺序：
    # 原文已有的条件会把那个字段提到前面），别用 names.index 自己猜。
    candidates = extract(source, f"fn_{n}_{trial}")
    field_ids = {c.name: c.id for c in candidates.fields}
    cond_ids = {c.name: c.id for c in candidates.conditions}
    return ASpec(
        n=n,
        trial=trial,
        shape=shape,
        fn_name=f"fn_{n}_{trial}",
        names=names,
        filter_name=filter_name,
        ret_first=ret_first,
        ret_second=ret_second,
        pre_condition=pre_condition,
        instruction=instruction,
        source=source,
        sample=a_sample(names, filter_name, pre_condition),
        expected_filter_id=cond_ids[filter_name],
        expected_field_ids=(field_ids[ret_first], field_ids[ret_second]),
    )


def check_select_runtime(
    body: str, fn_name: str, filter_name: str, ret_first: str, ret_second: str, sample: list[dict]
) -> tuple[bool, tuple[str, ...]]:
    """把组装产物真的跑一遍：过滤、字段集合、字段顺序都要对。"""
    try:
        tree = ast.parse(body)
    except SyntaxError as exc:
        return False, (f"产物语法错误: {exc.msg}",)
    namespace: dict = {}
    try:
        exec(compile(tree, "<assembled>", "exec"), namespace)  # noqa: S102 - 只跑实验样本
    except Exception as exc:  # noqa: BLE001 - 任何执行失败都记成这次失败
        return False, (f"产物无法执行: {type(exc).__name__}: {exc}",)
    fn = namespace.get(fn_name)
    if not callable(fn):
        return False, (f"找不到可调用函数 {fn_name}",)
    try:
        out = fn(list(sample))
    except Exception as exc:  # noqa: BLE001
        return False, (f"调用失败: {type(exc).__name__}: {exc}",)
    if not isinstance(out, list) or not all(isinstance(item, dict) for item in out):
        return False, (f"返回值形状不对: {out!r}",)

    expected = [
        {ret_first: row[ret_first], ret_second: row[ret_second]}
        for row in sample
        if row[filter_name]
    ]
    reasons: list[str] = []
    keys_ok = all(set(item) == {ret_first, ret_second} for item in out)
    if not keys_ok:
        seen_keys = sorted({key for item in out for key in item})
        reasons.append(f"返回字段不正确: 实际含 {seen_keys}，期望 {sorted({ret_first, ret_second})}")
    if keys_ok:
        # 两个字段的取值都要比：只比第一个字段的话，第二个字段被抄成常量或抄错会漏过去。
        for field in (ret_first, ret_second):
            got = [item[field] for item in out]
            want = [row[field] for row in expected]
            if got != want:
                reasons.append(f"过滤或取值不正确（{field}）: 实际 {got!r}，期望 {want!r}")
        if not reasons:
            order = [list(item.keys()) for item in out]
            if any(list(keys) != [ret_first, ret_second] for keys in order):
                reasons.append(
                    f"字段顺序不是要求的顺序: {order[0] if order else []}，期望 {[ret_first, ret_second]}"
                )
    return (not reasons), tuple(reasons)


def run_select_trial(engine: Engine, spec: ASpec, curve: str, point: str) -> Trial:
    """一次选择路线试验：extract → prompt → 模型 → parse → assemble → 运行时判断。"""
    candidates = extract(spec.source, spec.fn_name)
    messages = build_decision_prompt(spec.instruction, candidates)
    _warmup(engine, messages)
    start = time.perf_counter()
    try:
        raw, stats = engine.generate(messages, max_tokens=SELECT_MAX_TOKENS)
    except Exception as exc:  # noqa: BLE001 - 单次失败不该中断整轮
        stats = Stats()
        raw = ""
        reasons = (f"模型调用失败: {type(exc).__name__}: {exc}",)
        return Trial(curve, point, spec.trial, False, reasons, raw, spec.expected_json,
                     time.perf_counter() - start, stats)
    wall = time.perf_counter() - start

    reasons: list[str] = []
    decision: Decision | None = None
    try:
        decision = parse_decision(raw, candidates)
    except DecisionError as exc:
        reasons.append(f"决策不合法: {exc}")
    if decision is not None:
        field_names = {c.id: c.name for c in candidates.fields}
        cond_names = {c.id: c.name for c in candidates.conditions}
        got_filter = None if decision.filter_field is None else cond_names.get(decision.filter_field)
        got_fields = [field_names.get(fid, fid) for fid in decision.return_fields]
        if got_filter != spec.filter_name:
            reasons.append(f"过滤字段不是期望值: {got_filter!r}，期望 {spec.filter_name!r}")
        if got_fields != [spec.ret_first, spec.ret_second]:
            reasons.append(
                f"返回字段不是期望值: {got_fields}，期望 {[spec.ret_first, spec.ret_second]}"
            )
        body = ""
        try:
            body = assemble(spec.source, candidates, decision)
        except DecisionError as exc:
            reasons.append(f"组装失败: {exc}")
        else:
            ok, why = check_select_runtime(
                body, spec.fn_name, spec.filter_name, spec.ret_first, spec.ret_second, spec.sample
            )
            if not ok:
                reasons.extend(why)

    # 诊断（不参与判定，只是把失败按"哪种错法"分堆）：
    # 决定一个试验成败的是上面的运行时判断，这里只回答"它是怎么错的"。
    chosen = () if decision is None else tuple(decision.return_fields)
    chosen_filter = None if decision is None else decision.filter_field
    expected_set = set(spec.expected_field_ids)
    extra = {
        "invalid_id": any("未知的" in reason for reason in reasons),
        # 逐字复述系统提示里那句示例（空白差异不算）：诊断用，本身不判负。
        "echoed_example": bool(SYSTEM_EXAMPLE) and " ".join(raw.split()) == SYSTEM_EXAMPLE,
        # 回了一组与示例完全相同的 id：模型没有在选择，它在照抄提示里的示例答案。
        "example_decision": decision is not None
        and chosen_filter == EXAMPLE_FILTER_ID
        and chosen == EXAMPLE_FIELD_IDS,
        # 本次试验的正确答案恰好就是示例那句（能对上只是撞对了，不是会做决策）。
        "expected_is_example": spec.expected_filter_id == EXAMPLE_FILTER_ID
        and tuple(spec.expected_field_ids) == EXAMPLE_FIELD_IDS,
        # 把要求的字段连同别的候选一起回（"多选"），与"选错"、"顺序反了"分开计。
        "over_selected": bool(chosen)
        and expected_set < set(chosen),
        "wrong_filter": decision is not None and chosen_filter != spec.expected_filter_id,
        "reordered_fields": set(chosen) == expected_set and chosen != tuple(spec.expected_field_ids),
        "filter_pos": spec.names.index(spec.filter_name),
        "ret_first_pos": spec.names.index(spec.ret_first),
        "pre_condition": spec.pre_condition,
    }
    return Trial(curve, point, spec.trial, not reasons, tuple(reasons), raw, spec.expected_json,
                 wall, stats, detail=f"候选 {len(candidates.fields)}", extra=extra)


def run_curve_a(engine: Engine, sizes=A_SIZES, trials: int = A_TRIALS, shape: str = "aligned",
                curve: str = "A", verbose: bool = True) -> list[Trial]:
    records: list[Trial] = []
    for n in sizes:
        # 每个 N 至少 trials 次（默认 8），且不少于 N 次：trial % n 轮转，
        # 于是候选表里每个位置都至少当过一回正确答案。
        count = max(trials, n)
        for index in range(count):
            spec = a_spec(n, index, shape)
            trial = run_select_trial(engine, spec, curve, f"N={n}")
            records.append(trial)
            if verbose:
                mark = "通过" if trial.ok else "失败"
                print(
                    f"[{curve} N={n} #{index}] {mark} {trial.wall_seconds:.2f}s "
                    f"{trial.stats.generated_tokens}tok 期望 {trial.expected} 实际 {trial.raw.strip()[:90]}"
                )
                for reason in trial.reasons:
                    print(f"    - {reason}")
    return records


# --------------------------------------------------------------------------
# 曲线 B：生成路线 —— 函数体长度
# --------------------------------------------------------------------------

B_FN_NAMES = ("collect_orders", "gather_orders", "pick_orders", "list_orders", "scan_orders")
B_FIELDS = (
    ("order_id", "total", "paid"),
    ("ticket_id", "amount", "paid"),
    ("entry_id", "price", "active"),
    ("sku", "cost", "visible"),
    ("user_id", "fee", "enabled"),
)
B_EXTRAS = (
    ("note", "channel"),
    ("memo", "source"),
    ("remark", "origin"),
    ("comment", "route"),
    ("label", "stage"),
)
# 目标函数体行数：9 行是校准点（无守卫、无排序、无辅助函数，形状同 bench/compare.py 的任务 A），
# 其余三档都一样带守卫、排序、上限、模块常量、辅助函数，只有填充长度不同。
B_TARGETS = (9, 15, 30, 60)
B_TRIALS = 5
B_MAX_ROWS = 3

# 填充块：按顺序贪心加到目标行数。每块自洽（只读核心里的名字：orders / rows / order /
# 过滤字段 / 返回字段），不读别的填充块定义的名字，保证任意前缀都合法可跑且不影响结果。
B_BLOCKS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("pre", ("    # 先登记本批的基本信息，便于排查", '    batch_label = "$fn"', "    inspected = 0")),
    ("in", ("        # 每一条都登记一次，便于排查", '        row_note = str(order["$ret1"])')),
    ("post", ("    # 汇总本批规模，只用于日志", '    summary = {"rows": len(rows)}')),
    ("pre", ('    limit = MAX_ROWS', '    phase = "collect"')),
    ("in", ('        if order["$ret2"] >= 100:', "            heavy = True", "        else:", "            heavy = False")),
    ("post", ("    # 下面两个值只用于日志", "    scale = len(rows)", '    stage = "sorted"')),
    ("pre", ("    total_amount = 0.0", "    seen = set()")),
    ("in", ('        amount = order["$ret2"]', '        shape = (amount, order["$filt"])')),
    ("post", ("    checksum = len(rows) % 97", "    window = checksum")),
    ("pre", ("    # 输入规模先记下来，便于比对", "    batch_size = len(orders)", "    skipped = 0")),
    ("in", ('        # 明细字段先拼好，后面日志要用', '        detail = (order["$ret1"], order["$ret2"])')),
    ("post", ("    # 保持原有返回形状，不额外包装", "    kept = len(rows)")),
    ("pre", ("    checksum = 0", "    window = batch_size if batch_size else 0")),
    ("in", ('        if order["$filt"]:', '            kept_hint = "keep"', "        else:", '            kept_hint = "drop"')),
    ("post", ("    if len(rows) == 0:", "        summary = {}", '    else:', '        summary = {"n": len(rows)}')),
)


@dataclass(frozen=True)
class BSpec:
    """一次生成路线试验的输入：整份源码、指令、运行样本、必须保留的东西。"""

    label: str
    target: int
    fn_name: str
    ret_first: str
    ret_second: str
    filter_field: str
    extras: tuple[str, ...]
    max_rows: int
    source: str
    span_lines: int
    file_lines: int
    guarded: bool  # 是否要求保留守卫 / 排序 / 上限 / 辅助函数 / 模块常量
    instruction: str
    sample: list[Record]


def _b_trial_names(trial: int) -> tuple[str, str, str, str, tuple[str, ...]]:
    fn_name = B_FN_NAMES[trial % len(B_FN_NAMES)]
    ret_first, ret_second, filter_field = B_FIELDS[trial % len(B_FIELDS)]
    extras = B_EXTRAS[trial % len(B_EXTRAS)]
    return fn_name, ret_first, ret_second, filter_field, extras


def _b_sample(ret_first: str, ret_second: str, filter_field: str, extras: tuple[str, ...]) -> list[Record]:
    """7 条记录：5 条过过滤，按金额是升序 → 只有循环后的排序还在，输出才是降序。"""
    plan = [
        (True, 30, "a"),
        (False, 90, "b"),
        (True, 10, "c"),
        (True, 50, "d"),
        (False, 20, "e"),
        (True, 40, "f"),
        (True, 20, "g"),
    ]
    sample: list[Record] = []
    for index, (flag, amount, tag) in enumerate(plan, start=1):
        row = {ret_first: f"{ret_first}-{index}", ret_second: amount, filter_field: flag}
        for extra in extras:
            row[extra] = f"{extra}-{index}"
        row["tag"] = tag
        sample.append(Record(row))
    return sample


def _b_core_lines(
    fn_name: str, ret_first: str, ret_second: str, filter_field: str, extras: tuple[str, ...],
    with_extras: bool, compact: bool = False,
) -> list[str]:
    keys = [ret_first, ret_second, filter_field, *extras]
    lines = [
        f"def {fn_name}(orders):",
        '    """返回订单编号与金额。"""',
    ]
    if with_extras:
        lines += [
            "    if not orders:",
            "        return []",
        ]
    lines.append("    rows = []")
    lines.append("    for order in orders:")
    if compact:
        pairs = ", ".join(f'"{key}": order["{key}"]' for key in keys)
        lines.append(f"        rows.append({{{pairs}}})")
    else:
        lines.append("        rows.append(")
        lines.append("            {")
        for key in keys:
            lines.append(f'                "{key}": order["{key}"],')
        lines += ["            }", "        )"]
    if with_extras:
        lines += [
            f'    rows.sort(key=lambda r: -r["{ret_second}"])',
            "    if len(rows) > MAX_ROWS:",
            "        rows = rows[:MAX_ROWS]",
        ]
    lines.append("    return rows")
    return lines


def b_build(target: int, trial: int, with_extras: bool = True) -> BSpec:
    """按目标行数拼出一份源码；行数由宿主算好，不靠人眼估。"""
    fn_name, ret_first, ret_second, filter_field, extras = _b_trial_names(trial)
    template = {"fn": fn_name, "ret1": ret_first, "ret2": ret_second, "filt": filter_field}
    pre: list[str] = []
    inside: list[str] = []
    post: list[str] = []
    core = _b_core_lines(
        fn_name, ret_first, ret_second, filter_field, extras, with_extras,
        compact=not with_extras,  # 校准档写成单行 dict，函数才真的短
    )

    if with_extras and target > len(core):
        for position, block in B_BLOCKS:
            lines = [Template(line).substitute(template) for line in block]
            {"pre": pre, "in": inside, "post": post}[position].extend(lines)
            candidate = "\n".join(_b_compose(core, pre, inside, post))
            if _b_span(candidate, fn_name) >= target:
                break
            if len(pre) + len(inside) + len(post) > 120:  # 兜底，正常到不了
                break

    body = _b_compose(core, pre, inside, post)
    span = _b_span("\n".join(body), fn_name)
    header = ['"""订单工具：处理订单记录。"""', "", f"MAX_ROWS = {B_MAX_ROWS}", ""]
    if with_extras:
        header += [
            "",
            "def _score(order):",
            '    """辅助函数：金额翻倍，供上游对账使用。"""',
            f'    return order["{ret_second}"] * 2',
            "",
        ]
    source = "\n".join(header + body) + "\n"
    instruction = (
        f"修改函数：只保留 {filter_field} 为真的项，返回 {ret_first} 和 {ret_second}，其他不变。"
    )
    return BSpec(
        label=f"函数 {span} 行",
        target=target,
        fn_name=fn_name,
        ret_first=ret_first,
        ret_second=ret_second,
        filter_field=filter_field,
        extras=extras,
        max_rows=B_MAX_ROWS,
        source=source,
        span_lines=_b_span(source, fn_name),
        file_lines=source.count("\n"),
        guarded=with_extras,
        instruction=instruction,
        sample=_b_sample(ret_first, ret_second, filter_field, extras),
    )


def _b_compose(core: list[str], pre: list[str], inside: list[str], post: list[str]) -> list[str]:
    """把填充块插到三个位置：循环前 / 循环里 append 前 / 循环后 return 前。"""
    lines: list[str] = []
    for line in core:
        if line == "    rows = []":
            lines.extend(pre)
        if line.startswith("        rows.append("):
            lines.extend(inside)
        if line == "    return rows":
            lines.extend(post)
        lines.append(line)
    return lines


def _b_span(source: str, fn_name: str) -> int:
    """函数体行数（def 到 return，含首尾）。"""
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == fn_name:
            return (node.end_lineno or node.lineno) - node.lineno + 1
    return 0


def _has_early_return_guard(tree: ast.Module, fn_name: str, param: str) -> bool:
    """函数体第一条可执行语句仍是"参数为空就提前返回空列表"的判断（结构性检查）。"""
    node = next(
        (
            item
            for item in ast.walk(tree)
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == fn_name
        ),
        None,
    )
    if node is None:
        return False
    statements = [
        stmt
        for stmt in node.body
        if not (
            isinstance(stmt, ast.Expr)
            and isinstance(stmt.value, ast.Constant)
            and isinstance(stmt.value.value, str)
        )
    ]
    if not statements or not isinstance(statements[0], ast.If):
        return False
    guard = statements[0]
    names = {child.id for child in ast.walk(guard.test) if isinstance(child, ast.Name)}
    if param not in names:
        return False
    returns_empty = any(
        isinstance(stmt, ast.Return)
        and isinstance(stmt.value, (ast.List, ast.Tuple))
        and not stmt.value.elts
        for stmt in guard.body
    )
    return returns_empty


def b_expected(spec: BSpec) -> list[dict]:
    """要求的产物语义：过滤 → 保留循环后的排序 → 保留循环后的上限 → 只留两个字段。

    校准档（guarded=False）的原文本来就没有排序和上限，期望里也不加。
    """
    kept = [row for row in spec.sample if row[spec.filter_field]]
    if spec.guarded:
        kept.sort(key=lambda row: -row[spec.ret_second])
        kept = kept[: spec.max_rows]
    return [
        {spec.ret_first: row[spec.ret_first], spec.ret_second: row[spec.ret_second]}
        for row in kept
    ]


def check_generate_runtime(body: str, spec: BSpec) -> tuple[bool, tuple[str, ...], dict[str, bool]]:
    """把生成的整份文件真的跑一遍：过滤、字段、守卫、排序、上限、辅助函数、常量。

    判定与之前逐字相同（产物必须与期望逐项相等），只是把差异拆成几条具体的：
    多留字段、留下的行不对、顺序不对、条数不对分开报，不再挤在一条复合消息里
    （旧写法里"只多留了字段"会被读成"排序和上限也坏了"）。

    第三个返回值是各能力项的**行为**是否还在（守卫/排序/上限/常量/辅助/空输入），
    只用于诊断"哪些被保留了"，不参与判定。
    """
    aspects = {
        "fields": False, "filter": False, "order": False, "cap": False,
        "guard": not spec.guarded, "helper": not spec.guarded, "const": not spec.guarded,
        "empty": False,
    }
    try:
        tree = ast.parse(body)
    except SyntaxError as exc:
        return False, (f"语法错误: {exc.msg}",), aspects
    namespace: dict = {}
    try:
        exec(compile(tree, "<candidate>", "exec"), namespace)  # noqa: S102 - 只跑实验样本
    except Exception as exc:  # noqa: BLE001
        return False, (f"无法执行: {type(exc).__name__}: {exc}",), aspects
    fn = namespace.get(spec.fn_name)
    if not callable(fn):
        return False, (f"找不到可调用函数 {spec.fn_name}",), aspects
    try:
        out = fn(list(spec.sample))
    except Exception as exc:  # noqa: BLE001
        return False, (f"调用失败: {type(exc).__name__}: {exc}",), aspects

    reasons: list[str] = []
    expected = b_expected(spec)
    if not isinstance(out, list) or not all(isinstance(item, dict) for item in out):
        reasons.append(f"返回值形状不对: {out!r}")
    else:
        keys_ok = all(set(item) == {spec.ret_first, spec.ret_second} for item in out)
        aspects["fields"] = keys_ok
        if not keys_ok:
            seen_keys = sorted({key for item in out for key in item})
            reasons.append(
                f"返回字段不正确: 实际含 {seen_keys}，期望 {sorted({spec.ret_first, spec.ret_second})}"
            )
        # 过滤：留下的行（按 id 看）对不对；顺序：金额序列对不对；上限：条数对不对。
        got_ids = [item.get(spec.ret_first) for item in out]
        want_ids = [row[spec.ret_first] for row in expected]
        aspects["filter"] = got_ids == want_ids
        if got_ids != want_ids:
            reasons.append(f"留下的行不对（过滤/排序）: 实际 {got_ids}，期望 {want_ids}")
        got_amounts = [item.get(spec.ret_second) for item in out]
        want_amounts = [row[spec.ret_second] for row in expected]
        aspects["order"] = not spec.guarded or got_amounts == want_amounts
        if got_ids == want_ids and got_amounts != want_amounts:
            reasons.append(f"排序或取值不正确: 实际 {got_amounts}，期望 {want_amounts}")
        aspects["cap"] = not spec.guarded or len(out) <= spec.max_rows
        if len(out) != len(expected):
            reasons.append(f"输出条数不对（过滤或上限）: 实际 {len(out)} 条，期望 {len(expected)} 条")
        if not reasons and out != expected:
            reasons.append(
                "产物与期望不一致: 实际 "
                + repr([(item.get(spec.ret_first), item.get(spec.ret_second)) for item in out])
                + "，期望 "
                + repr([(row[spec.ret_first], row[spec.ret_second]) for row in expected])
            )
    try:
        empty = fn([])
    except Exception as exc:  # noqa: BLE001
        reasons.append(f"空输入调用失败: {type(exc).__name__}: {exc}")
    else:
        if empty != []:
            reasons.append(f"守卫子句行为改变: 空输入返回 {empty!r}，期望 []")
        else:
            aspects["empty"] = True
    if spec.guarded:
        aspects["guard"] = _has_early_return_guard(tree, spec.fn_name, "orders")
        if not aspects["guard"]:
            reasons.append("守卫子句丢失（结构性检查：函数开头不再有对 orders 的提前返回）")
        helper = namespace.get("_score")
        if not callable(helper):
            reasons.append("辅助函数 _score 丢失")
        else:
            try:
                doubled = helper(Record({spec.ret_second: 3}))
            except Exception as exc:  # noqa: BLE001
                reasons.append(f"辅助函数 _score 调用失败: {type(exc).__name__}: {exc}")
            else:
                aspects["helper"] = doubled == 6
                if doubled != 6:
                    reasons.append(f"辅助函数 _score 行为改变: 返回 {doubled!r}，期望 6")
        aspects["const"] = namespace.get("MAX_ROWS") == spec.max_rows
        if not aspects["const"]:
            reasons.append(
                f"模块常量 MAX_ROWS 丢失或被改: {namespace.get('MAX_ROWS')!r}，期望 {spec.max_rows}"
            )
    return (not reasons), tuple(reasons), aspects


def run_generate_trial(engine: Engine, spec: BSpec) -> Trial:
    """一次生成路线试验：整份文件交给模型，产物按运行时行为判断。"""
    brief = Brief(
        instruction=spec.instruction,
        target="orders.py",
        action=Action.REPLACE,
        kind=Kind.CODE,
        context=spec.source,
        original=spec.source,
    )
    messages = _brief_messages(brief)
    _warmup(engine, messages)
    start = time.perf_counter()
    try:
        raw, stats = request_body(engine, brief, max_tokens=GENERATE_MAX_TOKENS)
    except Exception as exc:  # noqa: BLE001
        return Trial("B", spec.label, 0, False, (f"模型调用失败: {type(exc).__name__}: {exc}",),
                     "", spec.instruction, time.perf_counter() - start, Stats())
    wall = time.perf_counter() - start

    artifact = to_artifact(brief, raw)
    reasons: list[str] = []
    aspects: dict[str, bool] = {}
    if not artifact.body.strip():
        reasons.append("没有产出正文")
    else:
        ok, why, aspects = check_generate_runtime(artifact.body, spec)
        if not ok:
            reasons.extend(why)
    if looks_like_explanation(raw):
        reasons.append("附带了说明文字")
    if looks_truncated(raw):
        reasons.append("输出被截断（围栏未闭合）")
    if stats.generated_tokens >= GENERATE_MAX_TOKENS:
        reasons.append("输出顶到 max_tokens 上限，可能被截断")
    # "保留了其它"按行为逐项算（守卫/排序/上限/常量/辅助/空输入），
    # 不再靠关键字匹配报错文本——旧写法里"只多留了字段"会被误读成排序和上限也没了。
    extra = {
        "preserved": bool(aspects) and all(
            aspects.get(key, False)
            for key in ("guard", "order", "cap", "const", "helper", "empty")
        ),
        "aspects": aspects,
    }
    return Trial("B", spec.label, 0, not reasons, tuple(reasons), raw, spec.instruction, wall,
                 stats, detail=spec.fn_name, extra=extra)


def _brief_messages(brief: Brief) -> list[dict[str, str]]:
    """生成路线的消息：与生产路径完全一致（复用 adapter.build_messages）。"""
    from azfls.adapter import build_messages

    return build_messages(brief)


def run_curve_b(engine: Engine, targets=B_TARGETS, trials: int = B_TRIALS,
                verbose: bool = True) -> list[Trial]:
    records: list[Trial] = []
    for target in targets:
        with_extras = target > B_TARGETS[0]
        for index in range(trials):
            spec = b_build(target, index, with_extras=with_extras)
            trial = run_generate_trial(engine, spec)
            trial.point = spec.label
            trial.index = index
            trial.extra.update(
                {
                    "span_lines": spec.span_lines,
                    "file_lines": spec.file_lines,
                    "guarded": spec.guarded,
                    "fn_name": spec.fn_name,
                    "fields": [spec.ret_first, spec.ret_second, spec.filter_field],
                }
            )
            records.append(trial)
            if verbose:
                mark = "通过" if trial.ok else "失败"
                print(
                    f"[B {spec.label} #{index}] {mark} {trial.wall_seconds:.2f}s "
                    f"{trial.stats.prompt_tokens}in/{trial.stats.generated_tokens}out "
                    f"（函数 {spec.span_lines} 行，文件 {spec.file_lines} 行）"
                )
                for reason in trial.reasons:
                    print(f"    - {reason}")
    return records


# --------------------------------------------------------------------------
# 曲线 C：拆分成本
# --------------------------------------------------------------------------


def _warmup(engine: Engine, messages: list[dict[str, str]]) -> float:
    """按同一形状空跑一次（max_tokens=1），把 MLX 编译图的一次性成本排除在稳态之外。"""
    start = time.perf_counter()
    try:
        engine.generate(messages, max_tokens=WARMUP_MAX_TOKENS)
    except Exception:  # noqa: BLE001 - 预热失败不致命，计时里会体现
        pass
    return time.perf_counter() - start


def measure_floor(engine: Engine, messages: list[dict[str, str]], repeats: int = 3) -> list[float]:
    """只生成 1 个 token 的墙钟：推理前置开销＋提示处理＋1 步解码，就是每次调用的底。"""
    times: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        engine.generate(messages, max_tokens=1)
        times.append(time.perf_counter() - start)
    return times


def run_curve_c(engine: Engine) -> dict:
    """量每次调用的固定底价，再算 N 块的墙钟。"""
    probes: dict[str, dict] = {}
    specs = {"A 小提示（N=3）": a_spec(3, 0), "A 大提示（N=15）": a_spec(15, 0)}
    for label, spec in specs.items():
        candidates = extract(spec.source, spec.fn_name)
        messages = build_decision_prompt(spec.instruction, candidates)
        engine.generate(messages, max_tokens=1)  # 预热，不计入
        times = measure_floor(engine, messages, repeats=3)
        prompt_tokens = len(engine._tokenizer.encode(  # noqa: SLF001 - 只为读提示长度
            engine._tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        ))
        probes[label] = {
            "times": times,
            "median": _median(times),
            "mean": _mean(times),
            "min": min(times),
            "prompt_tokens": prompt_tokens,
        }
        print(
            f"[C] {label}: 1-token 墙钟 {[f'{t:.3f}' for t in times]}"
            f"（中位 {_median(times):.3f}s，最小 {min(times):.3f}s），提示 {prompt_tokens} tokens"
        )

    minimal = [
        {"role": "system", "content": DECISION_SYSTEM_PROMPT},
        {"role": "user", "content": "只回 JSON，不要解释。"},
    ]
    engine.generate(minimal, max_tokens=1)
    times = measure_floor(engine, minimal, repeats=3)
    probes["最小提示"] = {
        "times": times,
        "median": _median(times),
        "mean": _mean(times),
        "min": min(times),
        "prompt_tokens": 0,
    }
    print(
        f"[C] 最小提示: 1-token 墙钟 {[f'{t:.3f}' for t in times]}"
        f"（中位 {_median(times):.3f}s，最小 {min(times):.3f}s）"
    )
    return probes


# --------------------------------------------------------------------------
# 汇总
# --------------------------------------------------------------------------


def _median(values: list[float]) -> float:
    return statistics.median(values) if values else 0.0


def _mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def summarize_a(records: list[Trial]) -> list[dict]:
    rows = []
    for n in sorted({record.extra.get("n", int(record.point.split("=")[1])) for record in records}):
        group = [record for record in records if int(record.point.split("=")[1]) == n]
        walls = [record.wall_seconds for record in group]
        rows.append(
            {
                "N": n,
                "trials": len(group),
                "passed": sum(1 for record in group if record.ok),
                "wall_median": _median(walls),
                "wall_mean": _mean(walls),
                "wall_min": min(walls) if walls else 0.0,
                "wall_max": max(walls) if walls else 0.0,
                "prompt_tokens": int(_median([record.stats.prompt_tokens for record in group])),
                "output_tokens": int(_median([record.stats.generated_tokens for record in group])),
                "invalid_id": sum(1 for record in group if record.extra.get("invalid_id")),
                "echoed_example": sum(1 for record in group if record.extra.get("echoed_example")),
                "example_decision": sum(
                    1 for record in group if record.extra.get("example_decision")
                ),
                "expected_is_example": sum(
                    1 for record in group if record.extra.get("expected_is_example")
                ),
                "passed_when_expected_is_example": sum(
                    1 for record in group
                    if record.extra.get("expected_is_example") and record.ok
                ),
                "passed_otherwise": sum(
                    1 for record in group
                    if not record.extra.get("expected_is_example") and record.ok
                ),
                "over_selected": sum(1 for record in group if record.extra.get("over_selected")),
                "wrong_filter": sum(1 for record in group if record.extra.get("wrong_filter")),
                "reordered_fields": sum(
                    1 for record in group if record.extra.get("reordered_fields")
                ),
                "by_filter_pos": {
                    str(pos): [
                        sum(1 for record in group if record.extra.get("filter_pos") == pos and record.ok),
                        sum(1 for record in group if record.extra.get("filter_pos") == pos),
                    ]
                    for pos in sorted({record.extra.get("filter_pos", -1) for record in group})
                },
            }
        )
    return rows


def summarize_b(records: list[Trial]) -> list[dict]:
    rows = []
    for label in dict.fromkeys(record.point for record in records):
        group = [record for record in records if record.point == label]
        walls = [record.wall_seconds for record in group]
        rows.append(
            {
                "target": label,
                "trials": len(group),
                "passed": sum(1 for record in group if record.ok),
                "wall_median": _median(walls),
                "wall_mean": _mean(walls),
                "wall_min": min(walls) if walls else 0.0,
                "wall_max": max(walls) if walls else 0.0,
                "prompt_tokens": int(_median([record.stats.prompt_tokens for record in group])),
                "output_tokens": int(_median([record.stats.generated_tokens for record in group])),
                "span_lines": sorted({record.extra.get("span_lines") for record in group}),
                "file_lines": sorted({record.extra.get("file_lines") for record in group}),
                "preserved": sum(1 for record in group if record.extra.get("preserved")),
                "aspects": {
                    key: sum(
                        1
                        for record in group
                        if (record.extra.get("aspects") or {}).get(key)
                    )
                    for key in ("guard", "order", "cap", "const", "helper", "empty", "filter", "fields")
                },
                "failures": [
                    reason for record in group for reason in record.reasons
                ],
            }
        )
    return rows


def print_summary_a(rows: list[dict], title: str) -> None:
    print(f"\n=== {title} ===")
    print(
        "| N | 通过/试验 | 墙钟均值 s | 中位 s | 最小–最大 s | 提示 token | 输出 token | "
        "非法 id | 期望=示例 | 照示例决策 | 字段多选 | 选错条件 | 字段顺序反 |"
    )
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for row in rows:
        print(
            f"| {row['N']} | {row['passed']}/{row['trials']} | {row['wall_mean']:.2f} | "
            f"{row['wall_median']:.2f} | "
            f"{row['wall_min']:.2f}–{row['wall_max']:.2f} | {row['prompt_tokens']} | "
            f"{row['output_tokens']} | {row['invalid_id']} | {row['expected_is_example']} | "
            f"{row['example_decision']} | {row['over_selected']} | {row['wrong_filter']} | "
            f"{row['reordered_fields']} |"
        )
    if rows:
        both = sum(row["expected_is_example"] for row in rows)
        hit = sum(row["passed_when_expected_is_example"] for row in rows)
        other = sum(row["passed_otherwise"] for row in rows)
        print(
            f"\n正确答案恰好等于提示示例的试验 {both} 次（通过 {hit} 次）；"
            f"其余试验 {sum(row['trials'] for row in rows) - both} 次（通过 {other} 次）。"
            "两列差得越远，越说明'通过'来自照抄示例而不是在做选择。"
        )
    print("\n按正确条件下标位置拆开（通过/试验）：")
    for row in rows:
        parts = ", ".join(f"pos{pos}: {ok}/{total}" for pos, (ok, total) in row["by_filter_pos"].items())
        print(f"  N={row['N']}: {parts}")


def print_summary_b(rows: list[dict]) -> None:
    print("\n=== 曲线 B：函数体长度（生成路线） ===")
    print(
        "| 目标 | 实际函数行 | 文件行 | 通过/试验 | 墙钟均值 s | 中位 s | 最小–最大 s | "
        "提示 token | 输出 token | 保留守卫/排序/上限/常量/辅助/空输入 |"
    )
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for row in rows:
        counts = row["aspects"]
        keep = "/".join(
            str(counts[key]) for key in ("guard", "order", "cap", "const", "helper", "empty")
        )
        print(
            f"| {row['target']} | {row['span_lines']} | {row['file_lines']} | {row['passed']}/{row['trials']} | "
            f"{row['wall_mean']:.2f} | {row['wall_median']:.2f} | "
            f"{row['wall_min']:.2f}–{row['wall_max']:.2f} | "
            f"{row['prompt_tokens']} | {row['output_tokens']} | {keep} |"
        )
    print("\n失败明细（每档最多 4 条）：")
    for row in rows:
        if row["failures"]:
            print(f"  {row['target']}: " + "；".join(row["failures"][:4]))


def print_curve_c(probes: dict, a_records: list[Trial], b_records: list[Trial]) -> dict:
    small = probes["A 小提示（N=3）"]
    large = probes["A 大提示（N=15）"]
    floor_small, floor_small_min = small["median"], small["min"]
    floor_large, floor_large_min = large["median"], large["min"]
    a_walls = [record.wall_seconds for record in a_records] or [0.0]
    b_walls = [record.wall_seconds for record in b_records] or [0.0]
    per_call_decision = _median(a_walls)
    per_call_generate = _median(b_walls)
    arithmetic = {
        "floor_small_prompt": floor_small,
        "floor_small_prompt_min": floor_small_min,
        "floor_small_prompt_mean": small["mean"],
        "floor_large_prompt": floor_large,
        "floor_large_prompt_min": floor_large_min,
        "floor_large_prompt_mean": large["mean"],
        "decision_call_median": per_call_decision,
        "decision_call_mean": _mean(a_walls),
        "decision_call_min": min(a_walls),
        "generate_call_median": per_call_generate,
        "generate_call_mean": _mean(b_walls),
        "generate_call_min": min(b_walls),
        "pieces_decision": {n: n * per_call_decision for n in (1, 3, 5, 10)},
        "pieces_floor": {n: n * floor_small for n in (1, 3, 5, 10)},
        "pieces_generate": {n: n * per_call_generate for n in (1, 3, 5, 10)},
    }
    print("\n=== 曲线 C：拆分成本算术 ===")
    print(
        f"每次调用的固定底（1 token，含提示处理，各 3 次）：小提示 {floor_small:.3f}s"
        f"（最小 {floor_small_min:.3f}s），大提示 {floor_large:.3f}s（最小 {floor_large_min:.3f}s）"
    )
    print(
        f"实测单次调用中位（含解码）：选择路线 {per_call_decision:.2f}s，生成路线 {per_call_generate:.2f}s"
    )
    print("| 块数 N | N×固定底 s | N×选择单次 s | N×生成单次 s |")
    print("| --- | --- | --- | --- |")
    for n in (1, 3, 5, 10):
        print(
            f"| {n} | {arithmetic['pieces_floor'][n]:.2f} | "
            f"{arithmetic['pieces_decision'][n]:.2f} | {arithmetic['pieces_generate'][n]:.2f} |"
        )
    return arithmetic


# --------------------------------------------------------------------------
# 自检：不加载权重，确认构造与判断口径本身没错
# --------------------------------------------------------------------------


def selftest() -> int:
    print("=== 自检（不加载权重） ===")
    failures = 0
    for n in A_SIZES:
        for trial in (0, 1, 2):
            for shape in (A_SHAPES if n >= 4 else ("aligned",)):
                spec = a_spec(n, trial, shape)
                candidates = extract(spec.source, spec.fn_name)
                good = Decision("fn0", spec.expected_filter_id, spec.expected_field_ids)
                body = assemble(spec.source, candidates, good)
                ok, why = check_select_runtime(
                    body, spec.fn_name, spec.filter_name, spec.ret_first, spec.ret_second, spec.sample
                )
                if not ok:
                    failures += 1
                    print(f"  [A {shape} N={n} #{trial}] 参考解不通过: {why}")
                # 原样不改应当失败（过滤没做）
                ok_raw, _ = check_select_runtime(
                    spec.source, spec.fn_name, spec.filter_name, spec.ret_first, spec.ret_second, spec.sample
                )
                if ok_raw:
                    failures += 1
                    print(f"  [A {shape} N={n} #{trial}] 未改动的原文被判为通过（判断口径太松）")
                # 换一个过滤条件应当失败
                other = next(c.id for c in candidates.conditions if c.id != spec.expected_filter_id)
                wrong = Decision("fn0", other, spec.expected_field_ids)
                ok_wrong, _ = check_select_runtime(
                    assemble(spec.source, candidates, wrong), spec.fn_name, spec.filter_name,
                    spec.ret_first, spec.ret_second, spec.sample,
                )
                if ok_wrong:
                    failures += 1
                    print(f"  [A {shape} N={n} #{trial}] 错误决策被判为通过（判断口径太松）")
    for target in B_TARGETS:
        for trial in range(B_TRIALS):
            spec = b_build(target, trial, with_extras=target > B_TARGETS[0])
            # 用选择路线的组装器做出参考解（宿主确定性重写），生成路线的判断应当接受它
            candidates = extract(spec.source, spec.fn_name)
            fields = {c.name: c.id for c in candidates.fields}
            conds = {c.name: c.id for c in candidates.conditions}
            decision = Decision(
                "fn0",
                conds[spec.filter_field],
                (fields[spec.ret_first], fields[spec.ret_second]),
            )
            body = assemble(spec.source, candidates, decision)
            ok, why, _aspects = check_generate_runtime(body, spec)
            if not ok:
                failures += 1
                print(f"  [B {target} #{trial}] 参考解不通过: {why}")
            ok_raw, _why_raw, _aspects_raw = check_generate_runtime(spec.source, spec)
            if ok_raw:
                failures += 1
                print(f"  [B {target} #{trial}] 未改动的原文被判为通过（判断口径太松）")
    print(f"自检结束：{'全部通过' if not failures else f'{failures} 处问题'}")
    return 1 if failures else 0


# --------------------------------------------------------------------------
# 入口
# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="片大小实验：小模型一次能可靠处理多大的一块")
    parser.add_argument("--curve", default="all", choices=("a", "b", "c", "all"))
    parser.add_argument("--model", default=str(MODEL_15B))
    parser.add_argument(
        "--trials-a", type=int, default=A_TRIALS,
        help="每个 N 的试验次数下限；实际取 max(该值, N)，保证每个候选位置都当过正确答案",
    )
    parser.add_argument("--trials-b", type=int, default=B_TRIALS)
    parser.add_argument("--sizes", default=",".join(str(n) for n in A_SIZES))
    parser.add_argument("--targets", default=",".join(str(n) for n in B_TARGETS))
    parser.add_argument("--misaligned-probe", action="store_true", help="额外跑一组候选表错位的 N=9 探针")
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--json", action="store_true", help="最后把明细打成一行 JSON")
    args = parser.parse_args(argv)

    if args.selftest:
        return selftest()

    sizes = tuple(int(part) for part in args.sizes.split(",") if part.strip())
    targets = tuple(int(part) for part in args.targets.split(",") if part.strip())

    engine = MLXEngine(args.model)
    print(f"加载模型 {args.model}")
    engine._ensure_loaded()  # noqa: SLF001 - 常驻生效后才开始计时
    print(f"加载完成 {engine.load_seconds:.2f}s\n")

    a_records: list[Trial] = []
    a2_records: list[Trial] = []
    b_records: list[Trial] = []
    probes: dict = {}
    started = time.perf_counter()

    if args.curve in ("a", "all"):
        a_records = run_curve_a(engine, sizes, args.trials_a)
        print_summary_a(summarize_a(a_records), "曲线 A：候选数量（选择路线）")
        if args.misaligned_probe:
            a2_records = run_curve_a(engine, (9,), args.trials_a, shape="misaligned", curve="A2")
            print_summary_a(summarize_a(a2_records), "曲线 A2：候选表错位（原文已有另一个条件）")
    if args.curve in ("b", "all"):
        b_records = run_curve_b(engine, targets, args.trials_b)
        print_summary_b(summarize_b(b_records))
    if args.curve in ("c", "all"):
        probes = run_curve_c(engine)

    arithmetic = {}
    if probes and (a_records or b_records):
        arithmetic = print_curve_c(probes, a_records, b_records)
    print(f"\n总墙钟 {time.perf_counter() - started:.1f}s")

    if args.json:
        payload = {
            "model": args.model,
            "trials": [
                {
                    "curve": record.curve,
                    "point": record.point,
                    "index": record.index,
                    "ok": record.ok,
                    "reasons": list(record.reasons),
                    "raw": record.raw,
                    "expected": record.expected,
                    "wall_seconds": record.wall_seconds,
                    "prompt_tokens": record.stats.prompt_tokens,
                    "generated_tokens": record.stats.generated_tokens,
                    "detail": record.detail,
                    "extra": record.extra,
                }
                for record in a_records + a2_records + b_records
            ],
            "summary_a": summarize_a(a_records) if a_records else [],
            "summary_a2": summarize_a(a2_records) if a2_records else [],
            "summary_b": summarize_b(b_records) if b_records else [],
            "probes": {key: value for key, value in probes.items()},
            "arithmetic": arithmetic,
        }
        print("JSON_BEGIN")
        print(json.dumps(payload, ensure_ascii=False))
        print("JSON_END")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
