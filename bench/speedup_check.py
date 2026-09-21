"""两处提速的实测复核：正文是否逐字一致、选择路径的墙钟前后差多少。

本轮落地的两处改动：

1. `chooseonly/model.py` 直驱 `mlx_lm.generate_step`，只在最后 decode 一次。
   旧调用 `mlx_lm.generate`（内部 `stream_generate`）每次都会新建一个流式
   detokenizer（`TokenizerWrapper.detokenizer` 是 `return self._detokenizer_class(self)`，
   构造要遍历 15 万条词表，不缓存）。
2. `chooseonly/decide.py` 的决策契约换成精简 JSON `{"f": ..., "r": [...]}`，
   旧的长键仍然接受（向后兼容）。

本脚本回答三件事，全部本机真跑：

- 正确性（逐字）：同一提示、温度 0 下，新引擎与旧的 `mlx_lm.generate` 调用返回的正文
  逐字比较；两套契约选出的决策也必须一样。
- 正确性（端到端）：在临时工作区里真跑一次选择路线（`run_decision` → Gate 写盘），
  把写出的文件 exec 起来调用函数，按运行时行为检查过滤、字段、顺序
  （口径直接复用 `bench.compare.check_runtime_behaviour`）。
- 计时：四种组合（旧/新引擎 × 旧/精简契约）按轮次交替执行，每种至少 5 次；
  模型已加载、已预热、温度 0，只报墙钟的均值/最小/最大，不含加载。

用法（在项目根目录）：

    HF_HUB_OFFLINE=1 .venv/bin/python bench/speedup_check.py --trials 7
"""

from __future__ import annotations

import argparse
import ast
import os
import sys
import tempfile
import time
from collections.abc import Callable
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:  # 直接以脚本方式运行时，也能 import chooseonly
    sys.path.insert(0, str(ROOT))
# 只用本机模型目录：整个脚本不需要联网。
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from chooseonly.decide import (  # noqa: E402 - 先修好 sys.path 再导入
    FUNCTION_ID,
    Candidates,
    Decision,
    DecisionError,
    build_decision_prompt,
    extract,
    parse_decision,
    run_decision,
)
from chooseonly.gate import Gate  # noqa: E402
from chooseonly.model import DEFAULT_MODEL, MLXEngine, clean_body  # noqa: E402
from bench.compare import (  # noqa: E402
    BENCH_SOURCE,
    EXPECTED_FIELDS,
    FILTER_FIELD,
    TASK_FUNCTION,
    TASK_INSTRUCTION,
    check_runtime_behaviour,
)

DEFAULT_MAX_TOKENS = 64

# 改动前的系统提示与用户消息标签：冻结副本，只用于 before 基线，模型看到的仍是
# build_decision_prompt 的当前产出。冻结而不是从代码里取，是因为改动后代码里已没有它们。
OLD_SYSTEM_PROMPT = (
    "你只回一个 JSON 对象，不写代码、不解释、不加围栏。\n"
    "只能使用给出的候选 id，不得发明新的 id 或字段名。\n"
    "只回这三个键：function（函数 id）、filter_field（条件 id，不过滤时用 null）、"
    "return_fields（字段 id 的数组，按要求的输出顺序）。\n"
    "正确形状示例："
    '{"function": "fn0", "filter_field": "c1", "return_fields": ["f2", "f3"]}'
)
OLD_FIELD_LABEL = "（return_fields 只能选这里）"
OLD_CONDITION_LABEL = "（filter_field 只能选这里，不过滤时用 null）"
NEW_FIELD_LABEL = "（r 只能选这里）"
NEW_CONDITION_LABEL = "（f 只能选这里，不过滤时用 null）"


# ---------- 一次调用的两种实现 ----------


def call_before(
    engine: MLXEngine, messages: list[dict[str, str]], max_tokens: int
) -> tuple[str, float, int, int]:
    """改动前的逐字复制：走 mlx_lm.generate（每次新建流式 detokenizer）。

    返回 (正文, 生成墙钟, 生成 token 数, 提示 token 数)；正文的 token 数按旧实现
    的口径（重新 encode 正文）计算，好让两边的 token 数字可比。
    """
    from mlx_lm import generate as mlx_generate
    from mlx_lm.sample_utils import make_sampler

    engine._ensure_loaded()  # noqa: SLF001 - 与 bench/compare.py 同一用法
    tokenizer = engine._tokenizer  # noqa: SLF001
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prompt_tokens = len(tokenizer.encode(prompt))

    start = time.perf_counter()
    text = mlx_generate(
        engine._model,  # noqa: SLF001
        tokenizer,
        prompt=prompt,
        max_tokens=max_tokens,
        sampler=make_sampler(temp=0.0),
        verbose=False,
    )
    seconds = time.perf_counter() - start
    body = clean_body(text)
    return body, seconds, len(tokenizer.encode(body)), prompt_tokens


def call_after(
    engine: MLXEngine, messages: list[dict[str, str]], max_tokens: int
) -> tuple[str, float, int, int]:
    """改动后的实现：MLXEngine.generate 直驱 generate_step，最后 decode 一次。"""
    body, stats = engine.generate(messages, max_tokens=max_tokens)
    return body, stats.generate_seconds, stats.generated_tokens, stats.prompt_tokens


# ---------- 契约与期望值 ----------


def legacy_messages(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    """把当前（精简）提示换成改动前的长键契约，其余一字不动。"""
    out = [dict(message) for message in messages]
    out[0]["content"] = OLD_SYSTEM_PROMPT
    user = out[1]["content"]
    for new, old in ((NEW_FIELD_LABEL, OLD_FIELD_LABEL), (NEW_CONDITION_LABEL, OLD_CONDITION_LABEL)):
        if new not in user:  # 基线必须真的是旧契约，替换不上就直接失败
            raise SystemExit(f"当前提示里找不到 {new!r}，无法构造 before 基线")
        user = user.replace(new, old)
    out[1]["content"] = user
    return out


def expected_decision(candidates: Candidates) -> Decision:
    """固定任务的正确决策：按名字取 id，不用猜。"""
    return Decision(
        function_id=FUNCTION_ID,
        filter_field=next(c.id for c in candidates.conditions if c.name == FILTER_FIELD),
        return_fields=tuple(
            c.id for name in ("id", "name") for c in candidates.fields if c.name == name
        ),
    )


def decision_names(candidates: Candidates, decision: Decision) -> tuple[list[str], str | None]:
    """把决策折成真实字段名，用于和期望值比较（id 合法不代表选得对）。"""
    fields = {c.id: c.name for c in candidates.fields}
    conditions = {c.id: c.name for c in candidates.conditions}
    names = [fields.get(fid, fid) for fid in decision.return_fields]
    condition = None if decision.filter_field is None else conditions.get(
        decision.filter_field, decision.filter_field
    )
    return names, condition


# ---------- 一次请求的四种组合 ----------


class Config:
    """一种组合：引擎实现 × 决策契约。"""

    def __init__(
        self,
        name: str,
        call: Callable[[MLXEngine, list[dict[str, str]], int], tuple[str, float, int, int]],
        messages: list[dict[str, str]],
    ) -> None:
        self.name = name
        self.call = call
        self.messages = messages
        self.walls: list[float] = []
        self.gen_tokens = 0
        self.prompt_tokens = 0
        self.parse_failures = 0

    def run(self, engine: MLXEngine, max_tokens: int) -> str:
        start = time.perf_counter()
        body, _seconds, self.gen_tokens, self.prompt_tokens = self.call(
            engine, self.messages, max_tokens
        )
        self.walls.append(time.perf_counter() - start)
        return body


# ---------- 打印与统计 ----------


def spread(values: list[float]) -> tuple[float, float, float]:
    return min(values), sum(values) / len(values), max(values)


def median(values: list[float]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2


def print_table(headers: list[str], rows: list[list[str]]) -> None:
    widths = [len(h) for h in headers]
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))
    print("| " + " | ".join(h.ljust(w) for h, w in zip(headers, widths)) + " |")
    print("|" + "|".join("-" * (w + 2) for w in widths) + "|")
    for row in rows:
        print("| " + " | ".join(c.ljust(w) for c, w in zip(row, widths)) + " |")


# ---------- 正确性：端到端的选择路线 ----------


def end_to_end(engine: MLXEngine, candidates: Candidates, expected: Decision) -> bool:
    """在临时工作区真跑一次选择路线，把写出的文件 exec 起来按运行时行为检查。"""
    print("=== 正确性：选择路线端到端（临时工作区，真实模型调用）===")
    with tempfile.TemporaryDirectory(prefix="chooseonly-speedup-") as scratch:
        workspace = Path(scratch)
        (workspace / "users.py").write_text(BENCH_SOURCE, encoding="utf-8")

        artifact, decision, _ = run_decision(
            engine, TASK_INSTRUCTION, BENCH_SOURCE, "users.py", TASK_FUNCTION
        )
        gate = Gate(workspace)
        proposal = gate.propose(artifact)
        written = gate.apply(gate.approve(proposal, True), proposal)

        text = written.read_text(encoding="utf-8")
        print(f"决策：{decision}")
        print('模型看到的契约：{"f": "c2", "r": ["f0", "f1"]}（精简键）')
        print(f"写盘文件：{written}")
        try:
            ast.parse(text)
            syntax_ok = True
        except SyntaxError as exc:
            print(f"语法错误：{exc}")
            syntax_ok = False

        names, condition = decision_names(candidates, decision)
        selection_ok = names == ["id", "name"] and condition == FILTER_FIELD
        print(f"选出的字段：{names}；条件：{condition}（期望 {sorted(EXPECTED_FIELDS)} / {FILTER_FIELD}）")

        runtime_ok, reasons = check_runtime_behaviour(text, TASK_FUNCTION)
        print(f"运行时行为：{'通过' if runtime_ok else '失败'} {reasons if reasons else ''}")
        print(f"决策与期望一致：{'是' if decision == expected else '否'}")

        namespace: dict[str, object] = {}
        exec(compile(ast.parse(text), "<e2e>", "exec"), namespace)  # noqa: S102 - 实验样本
        sample = [
            {"id": 1, "name": "A", "active": True, "extra": "x"},
            {"id": 2, "name": "B", "active": False, "extra": "y"},
            {"id": 3, "name": "C", "active": True, "extra": "z"},
        ]
        result = namespace[TASK_FUNCTION](sample)  # type: ignore[operator]
        print(f"调用结果：{result}")
        print("组装后的函数：")
        for line in text.splitlines():
            print(f"  {line}")
        print()
        return syntax_ok and selection_ok and runtime_ok


# ---------- 主流程 ----------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="两处提速的实测复核：正文逐字一致 + 墙钟前后对比")
    parser.add_argument("--trials", type=int, default=7, help="每种组合的试验次数（至少 5）")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    args = parser.parse_args(argv)
    trials = max(args.trials, 5)

    candidates = extract(BENCH_SOURCE, TASK_FUNCTION)
    terse = build_decision_prompt(TASK_INSTRUCTION, candidates)
    legacy = legacy_messages(terse)
    expected = expected_decision(candidates)
    print("候选项：" + ", ".join(f"{c.id}={c.name}" for c in candidates.fields))
    print("条件项：" + ", ".join(f"{c.id}={c.name}" for c in candidates.conditions))
    print(f"期望决策：{expected}\n")

    print(f"加载模型：{args.model}")
    engine = MLXEngine(args.model)
    engine._ensure_loaded()  # noqa: SLF001 - 常驻生效后才开始计时
    print(f"加载完成 {engine.load_seconds:.2f}s（不计入任何下面的数字）\n")

    # 单独量一次 detokenizer 构造成本：这是旧调用每次都要付、新实现省掉的固定开销。
    tokenizer = engine._tokenizer  # noqa: SLF001
    samples = []
    for _ in range(3):
        start = time.perf_counter()
        detokenizer = tokenizer.detokenizer
        samples.append(time.perf_counter() - start)
        del detokenizer
    low, mean, high = spread(samples)
    print(f"单独量 detokenizer 构造：均值 {mean:.3f}s，{low:.3f}–{high:.3f}s（每次旧调用付一次）\n")

    # 预热：四种组合各一次，全部丢弃（MLX 编译图、KV cache 分配先发生一次）。
    for messages in (legacy, terse):
        call_before(engine, messages, args.max_tokens)
        call_after(engine, messages, args.max_tokens)
    print("预热 4 次已丢弃（两种契约 × 新旧引擎）\n")

    print("=== 正确性：同一提示下新旧引擎的正文逐字比较 ===")
    body_ok = True
    for name, messages in (("旧契约（长键）", legacy), ("精简契约（f/r）", terse)):
        before_body, _, before_tokens, _ = call_before(engine, messages, args.max_tokens)
        after_body, _, after_tokens, _ = call_after(engine, messages, args.max_tokens)
        same = before_body == after_body
        body_ok = body_ok and same
        print(
            f"[{name}] 正文逐字一致：{'是' if same else '否'}"
            f"｜输出 token {before_tokens} vs {after_tokens}"
        )
        print(f"  旧引擎：{before_body.strip()[:120]!r}")
        if not same:
            print(f"  新引擎：{after_body.strip()[:120]!r}")
        for label, body in (("旧引擎", before_body), ("新引擎", after_body)):
            try:
                decision = parse_decision(body, candidates)
            except DecisionError as exc:
                print(f"  {label} 决策不合法：{exc}")
                body_ok = False
                continue
            ok_names, condition = decision_names(candidates, decision)
            ok = decision == expected
            body_ok = body_ok and ok
            print(
                f"  {label} 决策：{decision}｜字段 {ok_names}、条件 {condition}"
                f"｜{'正确' if ok else '不是期望值'}"
            )
    print()

    configs = [
        Config("旧引擎+旧契约", call_before, legacy),
        Config("旧引擎+精简契约", call_before, terse),
        Config("新引擎+旧契约", call_after, legacy),
        Config("新引擎+精简契约", call_after, terse),
    ]

    print(f"=== 计时：选择路径单次墙钟（每种 {trials} 次，轮转执行，温度 0）===")
    for index in range(trials):
        order = configs if index % 2 == 0 else list(reversed(configs))
        for config in order:
            body = config.run(engine, args.max_tokens)
            try:
                parse_decision(body, candidates)
            except DecisionError:
                config.parse_failures += 1
    print()

    baseline = next(c for c in configs if c.name == "旧引擎+旧契约")
    _, base_mean, _ = spread(baseline.walls)
    rows = []
    for config in configs:
        low, mean, high = spread(config.walls)
        rows.append(
            [
                config.name,
                f"{mean:.3f}",
                f"{low:.3f}",
                f"{high:.3f}",
                str(config.gen_tokens),
                f"{config.prompt_tokens}",
                f"{(1 - mean / base_mean) * 100:+.1f}%",
                f"{config.parse_failures}/{trials}",
            ]
        )
    print_table(
        ["组合", "墙钟均值(s)", "最小(s)", "最大(s)", "输出token†", "提示token", "相对基线", "解析失败"],
        rows,
    )
    print(
        "† 输出 token 的口径与实现一致：旧引擎是重新 encode 正文的长度，"
        "新引擎是真正生成并解码的 token 数；同一段正文两者可以差几个 token"
        "（模型生成了会被重新合并的分段），正文本身仍逐字相同。"
    )

    print()
    for label, a, b in (
        ("WIN 2（精简契约，旧引擎）", configs[0], configs[1]),
        ("WIN 2（精简契约，新引擎）", configs[2], configs[3]),
        ("WIN 1（直驱 generate_step，精简契约）", configs[1], configs[3]),
        ("WIN 1（直驱 generate_step，旧契约）", configs[0], configs[2]),
        ("两处合计（旧引擎+旧契约 → 新引擎+精简契约）", configs[0], configs[3]),
    ):
        _, before_mean, _ = spread(a.walls)
        _, after_mean, _ = spread(b.walls)
        print(
            f"{label}：{before_mean:.3f}s → {after_mean:.3f}s，"
            f"省 {before_mean - after_mean:.3f}s / {(1 - after_mean / before_mean) * 100:.0f}%"
        )

    e2e_ok = end_to_end(engine, candidates, expected)
    ok = body_ok and e2e_ok and all(c.parse_failures == 0 for c in configs)
    print(f"\n=== 判定：{'全部通过' if ok else '有失败项'} ===")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
