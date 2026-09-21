"""速度实验：把模型换小，是不是真的更快、会不会改错。

四个变体，同一份源码、同一条指令、温度 0（引擎内部 make_sampler(temp=0.0)）：

    1.5B + select ／ 0.5B + select ／ 1.5B + ask(generate) ／ 0.5B + ask(generate)

耗时口径分三段，绝不混着报：

  - 加载（load_seconds）：把权重读进内存的一次性成本，只在模型第一次被用到时付；
  - 首次调用（warmup_seconds）：MLX 编译计算图、分配 KV cache 的一次性成本。每个
    变体先空跑一次同形状的请求、不计入均值，单独报出来，避免把编译成本算进稳态；
  - 稳态单次（generate_seconds）：模型常驻、图已编译之后，每一次决策/生成的真实耗时。

正确率一律用运行时行为判断：把产物真的 exec 出来、用样本真的调用一次，看过滤、
返回字段、守卫子句和循环后的排序是否还对。输出变快但改错，等于没有价值。

本文件复用 bench/compare.py 的任务常量与判断口径，那边一个字都不改。
结果只是单机小样本观测，不是通用性能结论。
"""

from __future__ import annotations

import argparse
import ast
import time
from dataclasses import dataclass, field
from pathlib import Path

from codejev.adapter import to_artifact
from codejev.contracts import Action, Brief, Kind
from codejev.decide import (
    DECISION_MAX_TOKENS,
    DECISION_SYSTEM_PROMPT,
    Decision,
    DecisionError,
    assemble,
    build_decision_prompt,
    extract,
    parse_decision,
)
from codejev.model import Engine, MLXEngine, request_body
from bench.compare import (
    BENCH_SOURCE,
    EXPECTED_FIELDS,
    EXPECTED_IDS,
    FILTER_FIELD,
    TASK_FUNCTION,
    TASK_INSTRUCTION,
    looks_like_explanation,
    looks_truncated,
)

ROOT = Path(__file__).resolve().parent.parent
MODEL_15B = ROOT / "models" / "Qwen2.5-Coder-1.5B-Instruct-4bit"
MODEL_05B = ROOT / "models" / "Qwen2.5-Coder-0.5B-Instruct-4bit"

# 与 bench/compare.py 保持一致，数字才能和已有基线直接对比。
# （decide.run_decision 的生产默认是 128；这里用 64 是为了对齐基线口径。）
SELECT_MAX_TOKENS = 64
GENERATE_MAX_TOKENS = 1024

# 任务 B 的源码：带前置守卫、复合条件和循环后排序。
# select 路线一次只能表达一个条件，这里就是要看它能不能在只看 paid 的同时
# 保住 `if not orders: return []` 和循环后的 rows.sort(...)。
TASK_B_INSTRUCTION = "修改函数：只保留 paid 为真的项，返回 order_id 和 total，其他不变。"
TASK_B_FUNCTION = "paid_orders"
TASK_B_SOURCE = '''def paid_orders(orders):
    """返回已付款订单的编号与金额，按金额从高到低。"""
    if not orders:
        return []
    rows = []
    for order in orders:
        if order.paid and not order.shipped:
            rows.append({"order_id": order.order_id, "total": order.total})
    rows.sort(key=lambda r: -r["total"])
    return rows
'''


class Record(dict):
    """同时支持 `order.paid` 与 `order["paid"]` 的记录对象。

    生成路线可能把属性访问改写成下标访问（或反过来），两种写法都能跑才是对
    "其他不变"的正当检验，不该因为取值风格不同就判失败。
    """

    def __getattr__(self, name: str) -> object:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


@dataclass(frozen=True)
class RuntimeSpec:
    """一个任务的运行时判断标准；全部来自真实调用，不看代码像不像。"""

    function_name: str
    instruction: str
    sample: list[dict]
    expected_ids: list
    expected_fields: frozenset[str]
    expected_filter: str
    id_key: str = "id"
    # 任务 B 才启用：守卫子句要求空输入返回 []，循环后的 sort 要求结果降序。
    requires_empty_answer: bool = False
    descending_key: str | None = None
    # 守卫子句在运行时是不可分辨的（有没有它，空输入都返回 []），所以除了空输入
    # 行为检查，再补一条确定性的 AST 检查：函数开头仍有一个"提前返回空结果"的判断。
    guard_param: str | None = None


SPEC_A = RuntimeSpec(
    function_name=TASK_FUNCTION,
    instruction=TASK_INSTRUCTION,
    sample=[
        {"id": 1, "name": "A", "active": True, "extra": "x"},
        {"id": 2, "name": "B", "active": False, "extra": "y"},
        {"id": 3, "name": "C", "active": True, "extra": "z"},
    ],
    expected_ids=EXPECTED_IDS,
    expected_fields=frozenset(EXPECTED_FIELDS),
    expected_filter=FILTER_FIELD,
)

# 样本设计成两处都能真的分辨对错：
#   - id 11 是 paid 且 shipped 的订单：原复合条件会排除它，只看 paid 就该留下它，
#     这样"条件换对了"和"条件没换"的运行时结果不同；
#   - 过滤后剩余项按输入顺序是 total=20、total=30，只有循环后的 sort 还在跑，
#     输出才会是降序 [33, 11]；sort 被删掉就会变成升序 [11, 33]。
SPEC_B = RuntimeSpec(
    function_name=TASK_B_FUNCTION,
    instruction=TASK_B_INSTRUCTION,
    sample=[
        Record(order_id=11, total=20, paid=True, shipped=True),
        Record(order_id=22, total=50, paid=False, shipped=False),
        Record(order_id=33, total=30, paid=True, shipped=False),
    ],
    expected_ids=[33, 11],
    expected_fields=frozenset({"order_id", "total"}),
    expected_filter="paid",
    id_key="order_id",
    requires_empty_answer=True,
    descending_key="total",
    guard_param="orders",
)


def check_runtime(source: str, spec: RuntimeSpec) -> tuple[bool, tuple[str, ...]]:
    """把产物真的跑一遍：过滤、返回字段、空输入守卫、循环后排序都要对。

    任务 A 的判断口径与 bench/compare.py 的 check_runtime_behaviour 一致；
    多出来的空输入与降序两项只在任务 B 启用（那里源码本来就有守卫和 sort）。
    """
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return False, (f"语法错误: {exc.msg}",)

    namespace: dict[str, object] = {}
    try:
        exec(compile(tree, "<candidate>", "exec"), namespace)  # noqa: S102 - 只跑实验样本
    except Exception as exc:  # noqa: BLE001 - 任何执行失败都记成这次失败
        return False, (f"无法执行: {type(exc).__name__}: {exc}",)

    fn = namespace.get(spec.function_name)
    if not callable(fn):
        return False, (f"找不到可调用函数 {spec.function_name}",)

    try:
        out = fn(list(spec.sample))
    except Exception as exc:  # noqa: BLE001
        return False, (f"调用失败: {type(exc).__name__}: {exc}",)

    if not isinstance(out, list):
        return False, ("返回值不是列表",)
    if not all(isinstance(item, dict) for item in out):
        return False, (f"元素不全是字典: {out!r}",)

    reasons: list[str] = []
    ids = [item.get(spec.id_key) for item in out]
    if ids != spec.expected_ids:
        reasons.append(f"过滤/顺序不正确: {ids}，期望 {spec.expected_ids}")
    for item in out:
        if set(item) != spec.expected_fields:
            reasons.append(f"字段不正确: {sorted(item)}，期望 {sorted(spec.expected_fields)}")
            break

    if spec.descending_key is not None:
        values = [item.get(spec.descending_key) for item in out]
        if not all(
            isinstance(a, (int, float)) and isinstance(b, (int, float)) and a > b
            for a, b in zip(values, values[1:])
        ):
            reasons.append(f"循环后的排序丢失: {values}，期望按 {spec.descending_key} 严格降序")

    if spec.requires_empty_answer:
        try:
            empty = fn([])
        except Exception as exc:  # noqa: BLE001
            reasons.append(f"空输入调用失败: {type(exc).__name__}: {exc}")
        else:
            if empty != []:
                reasons.append(f"守卫子句行为改变: 空输入返回 {empty!r}，期望 []")

    if spec.guard_param is not None and not _has_early_return_guard(tree, spec):
        reasons.append(
            f"守卫子句丢失: 函数开头不再有对 {spec.guard_param} 的提前返回（结构性检查，"
            "运行时对空输入不可分辨）"
        )

    return (not reasons), tuple(reasons)


def _has_early_return_guard(tree: ast.Module, spec: RuntimeSpec) -> bool:
    """函数体第一条可执行语句仍是"参数为空就提前返回空列表"的判断。

    不限定写法（`not x` / `len(x) == 0` 都算），只看形状：开头是 if，判断里出现
    参数名，体内有 return 一个空列表字面量。守卫子句有没有，空输入的运行时结果
    都是 []，因此这里用确定性 AST 形状补足，而不是靠人看。
    """
    assert spec.guard_param is not None
    node = next(
        (
            item
            for item in ast.walk(tree)
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
            and item.name == spec.function_name
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
    if spec.guard_param not in {child.id for child in ast.walk(guard.test) if isinstance(child, ast.Name)}:
        return False
    return any(
        isinstance(stmt, ast.Return)
        and isinstance(stmt.value, ast.List)
        and not stmt.value.elts
        for stmt in guard.body
    )


@dataclass
class Trial:
    """一次尝试：产物是否可用，以及三段耗时。"""

    ok: bool
    reasons: tuple[str, ...]
    raw: str
    body: str
    generated_tokens: int
    prompt_tokens: int
    generate_seconds: float
    wall_seconds: float


@dataclass
class Variant:
    """一个变体（模型 × 路线）的全部尝试。"""

    label: str
    route: str
    model: str
    spec: RuntimeSpec
    warmup_seconds: float = 0.0
    trials: list[Trial] = field(default_factory=list)

    def summary(self) -> dict[str, float]:
        rows = self.trials
        if not rows:
            return {}
        seconds = [row.generate_seconds for row in rows]
        tokens = [row.generated_tokens for row in rows]
        rates = [
            row.generated_tokens / row.generate_seconds
            for row in rows
            if row.generate_seconds > 0
        ]
        walls = [row.wall_seconds for row in rows]
        return {
            "passed": sum(1 for row in rows if row.ok),
            "total": len(rows),
            "mean_seconds": sum(seconds) / len(seconds),
            "min_seconds": min(seconds),
            "max_seconds": max(seconds),
            "mean_tokens": sum(tokens) / len(tokens),
            "mean_rate": sum(rates) / len(rates) if rates else 0.0,
            "min_rate": min(rates) if rates else 0.0,
            "max_rate": max(rates) if rates else 0.0,
            "mean_wall": sum(walls) / len(walls),
            "mean_prompt": sum(row.prompt_tokens for row in rows) / len(rows),
        }


# --------------------------------------------------------------------------
# 两条路线各跑一次；逻辑与 bench/compare.py 对应函数保持一致
# --------------------------------------------------------------------------


def select_once(
    engine: Engine, source: str, function_name: str, spec: RuntimeSpec
) -> Trial:
    """选择路线一次：宿主提取候选 → 小模型只选 id → 宿主组装 → 运行时检验。"""
    start = time.perf_counter()
    try:
        candidates = extract(source, function_name)
    except DecisionError as exc:
        return Trial(False, (f"提取失败: {exc}",), "", "", 0, 0, 0.0, time.perf_counter() - start)

    messages = build_decision_prompt(spec.instruction, candidates)
    raw, stats = engine.generate(messages, max_tokens=SELECT_MAX_TOKENS)

    reasons: list[str] = []
    decision: Decision | None = None
    try:
        decision = parse_decision(raw, candidates)
    except DecisionError as exc:
        reasons.append(f"决策不合法: {exc}")

    body = ""
    if decision is not None:
        # id 合法不代表选得对：再按真实字段名核对一次。
        field_names = {item.id: item.name for item in candidates.fields}
        chosen = {field_names.get(fid, fid) for fid in decision.return_fields}
        if chosen != set(spec.expected_fields):
            reasons.append(f"选的字段不是期望值: {sorted(chosen)}")
        condition = None
        if decision.filter_field is not None:
            condition_names = {item.id: item.name for item in candidates.conditions}
            condition = condition_names.get(decision.filter_field, decision.filter_field)
        if condition != spec.expected_filter:
            reasons.append(f"过滤字段不是期望值: {condition}")
        try:
            body = assemble(source, candidates, decision)
        except DecisionError as exc:
            reasons.append(f"组装失败: {exc}")
        else:
            ok, why = check_runtime(body, spec)
            if not ok:
                reasons.extend(why)

    return Trial(
        ok=not reasons,
        reasons=tuple(reasons),
        raw=raw,
        body=body,
        generated_tokens=stats.generated_tokens,
        prompt_tokens=stats.prompt_tokens,
        generate_seconds=stats.generate_seconds,
        wall_seconds=time.perf_counter() - start,
    )


def generate_once(engine: Engine, source: str, spec: RuntimeSpec) -> Trial:
    """生成路线一次：小模型自由写出整个函数正文。"""
    brief = Brief(
        instruction=spec.instruction,
        target="target.py",
        action=Action.REPLACE,
        kind=Kind.CODE,
        context=source,
        original=source,
    )
    start = time.perf_counter()
    raw, stats = request_body(engine, brief, max_tokens=GENERATE_MAX_TOKENS)
    wall = time.perf_counter() - start
    artifact = to_artifact(brief, raw)

    reasons: list[str] = []
    if not artifact.body.strip():
        reasons.append("没有产出正文")
    else:
        try:
            ok, why = check_runtime(artifact.body, spec)
        except Exception as exc:  # noqa: BLE001 - 单次失败不该中断整轮实验
            ok, why = False, (f"检查时异常: {type(exc).__name__}",)
        if not ok:
            reasons.extend(why)
    if looks_like_explanation(raw):
        reasons.append("附带了说明文字")
    if looks_truncated(raw):
        reasons.append("输出被截断（围栏未闭合）")

    return Trial(
        ok=not reasons,
        reasons=tuple(reasons),
        raw=raw,
        body=artifact.body,
        generated_tokens=stats.generated_tokens,
        prompt_tokens=stats.prompt_tokens,
        generate_seconds=stats.generate_seconds,
        wall_seconds=wall,
    )


def run_variant(
    label: str,
    route: str,
    engine: Engine,
    model_name: str,
    source: str,
    function_name: str,
    spec: RuntimeSpec,
    trials: int,
) -> Variant:
    """跑一个变体：先空跑一次预热（单独计时，不计入均值），再跑 trials 次。"""
    variant = Variant(label=label, route=route, model=model_name, spec=spec)
    once = (
        (lambda: select_once(engine, source, function_name, spec))
        if route == "select"
        else (lambda: generate_once(engine, source, spec))
    )
    variant.warmup_seconds = once().generate_seconds
    for _ in range(trials):
        variant.trials.append(once())
    return variant


def print_table(title: str, variants: list[Variant]) -> None:
    """把一个任务下四个变体的实测数字打成一张表。"""
    print(f"\n=== {title} ===")
    header = (
        f"{'变体':<20}{'通过':>7}{'单次生成 s（min–max）':>28}"
        f"{'输出 tok':>10}{'tok/s（min–max）':>22}{'单次墙钟 s':>12}"
    )
    print(header)
    print("-" * len(header))
    for variant in variants:
        row = variant.summary()
        if not row:
            continue
        print(
            f"{variant.label:<20}"
            f"{int(row['passed'])}/{int(row['total']):<4}"
            f"{row['mean_seconds']:>10.2f} ({row['min_seconds']:.2f}–{row['max_seconds']:.2f})   "
            f"{row['mean_tokens']:>8.0f}"
            f"{row['mean_rate']:>10.1f} ({row['min_rate']:.0f}–{row['max_rate']:.0f})   "
            f"{row['mean_wall']:>8.2f}"
        )
    for variant in variants:
        detail = [row for row in variant.trials if not row.ok]
        for row in detail:
            raw = " ".join(row.raw.split())[:160]
            print(f"  失败[{variant.label}] {'; '.join(row.reasons)}｜原始输出: {raw or '（空）'}")
    for variant in variants:
        if variant.trials:
            raw = " ".join(variant.trials[0].raw.split())[:160]
            print(f"  首次原始输出[{variant.label}]: {raw or '（空）'}")


# 诊断用：把系统提示里那行具体示例删掉，其余一字不改。
# 提示本身归 decide.py 管，这里只是复制一份做对照，不改产品代码。
NO_EXAMPLE_SYSTEM_PROMPT = "\n".join(
    line
    for line in DECISION_SYSTEM_PROMPT.splitlines()
    if not line.startswith("正确形状示例")
)


def run_example_diagnostic(
    engines: dict[str, MLXEngine], source: str, function_name: str, spec: RuntimeSpec, trials: int
) -> None:
    """额外诊断（不属于四个变体）：小模型是不是在照抄系统提示里的示例 id。

    同一个候选表、同一条指令，只把系统提示里的示例行删掉再跑一次 select。
    输出和正常变体一样时，说明它照着候选表选；输出跟着示例走，说明它在抄示例。
    这是解释失败原因用的，不是产品提示的改动。
    """
    candidates = extract(source, function_name)
    print("\n=== 诊断：系统提示删掉具体示例行后，select 决策会变成什么 ===")
    print(
        f"候选表：字段 {[(c.id, c.name) for c in candidates.fields]}，"
        f"条件 {[(c.id, c.name) for c in candidates.conditions]}"
    )
    for model_name, engine in engines.items():
        messages = list(build_decision_prompt(spec.instruction, candidates))
        messages[0] = {"role": "system", "content": NO_EXAMPLE_SYSTEM_PROMPT}
        engine.generate(messages, max_tokens=DECISION_MAX_TOKENS)  # 预热，不计时
        for index in range(trials):
            raw, stats = engine.generate(messages, max_tokens=DECISION_MAX_TOKENS)
            try:
                decision = parse_decision(raw, candidates)
                verdict = f"解析通过 {decision.filter_field} {decision.return_fields}"
            except DecisionError as exc:
                verdict = f"解析失败 {exc}"
            print(
                f"  [{model_name}] #{index} {stats.generated_tokens} tok："
                f"{' '.join(raw.split())[:110]}｜{verdict}"
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="模型大小 × 两条路线的速度与正确率实验")
    parser.add_argument("--trials", type=int, default=5, help="每个变体的实测次数（默认 5）")
    parser.add_argument("--model-15b", default=str(MODEL_15B))
    parser.add_argument("--model-05b", default=str(MODEL_05B))
    parser.add_argument(
        "--diagnostic",
        action="store_true",
        help="额外跑一次诊断：删掉系统提示的示例行看决策怎么变（不属于四个变体）",
    )
    args = parser.parse_args(argv)

    engines: dict[str, MLXEngine] = {}
    for name, path in (("1.5B", args.model_15b), ("0.5B", args.model_05b)):
        engine = MLXEngine(path)
        engine._ensure_loaded()  # noqa: SLF001 - 常驻生效后才开始计时
        print(f"加载 {name}（{path}）：{engine.load_seconds:.2f}s")
        engines[name] = engine
    print("以下稳态数字都假设模型已常驻；加载成本只在每次进程冷启动时付一次。")

    for title, source, function_name, spec in (
        ("任务 A：简单改动（同 bench/compare.py 的固定任务）", BENCH_SOURCE, TASK_FUNCTION, SPEC_A),
        ("任务 B：带守卫子句与循环后排序的真实形状", TASK_B_SOURCE, TASK_B_FUNCTION, SPEC_B),
    ):
        variants: list[Variant] = []
        for model_name, engine in engines.items():
            for route in ("select", "generate"):
                variants.append(
                    run_variant(
                        label=f"{model_name} + {route}",
                        route=route,
                        engine=engine,
                        model_name=model_name,
                        source=source,
                        function_name=function_name,
                        spec=spec,
                        trials=args.trials,
                    )
                )
        print_table(title, variants)
        for variant in variants:
            print(
                f"  预热（不计入均值）{variant.label}："
                f"{variant.warmup_seconds:.2f}s，平均输出 {variant.summary()['mean_tokens']:.0f} tokens"
            )
    if args.diagnostic:
        run_example_diagnostic(engines, BENCH_SOURCE, TASK_FUNCTION, SPEC_A, args.trials)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
