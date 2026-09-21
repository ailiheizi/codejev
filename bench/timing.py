"""计时实验：选择路线的延迟预算拆解，以及“更短输出”这个杠杆的真实成本。

回答两个问题，全部用本机真跑的数字，不估算：

1. 选择路线那 0.6 s 花在哪里？逐次调用拆成四段：
   调用前置开销（分词器 detokenizer 构造 + 配置）／提示处理（首 token 时间）／
   解码（首个 token 之后的逐 token 步进）／收尾（同步与 wired limit 复原）。
   模型加载、预热都不计入稳态数字，单独报告。
2. 同一个决策换更短的输出格式（精简 JSON、裸 token）是否真的更快、是否可靠。
   三种格式共用同一份候选表和同一段任务文本，只有输出契约不同，逐条统计能否解析。

用法（在项目根目录）：

    HF_HUB_OFFLINE=1 .venv/bin/python bench/timing.py --trials 9

本脚本只读现有模块（extract / build_decision_prompt / parse_decision 原样复用），
不修改任何 codejev 代码；它测量的是现有选择路径换引擎调用方式之后的结果。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:  # 直接以脚本方式运行时，也能 import codejev
    sys.path.insert(0, str(ROOT))

from codejev.decide import (  # noqa: E402 - 先修好 sys.path 再导入
    DECISION_SYSTEM_PROMPT,
    FUNCTION_ID,
    Candidates,
    Decision,
    DecisionError,
    _json_object,
    build_decision_prompt,
    extract,
    parse_decision,
)
from codejev.model import DEFAULT_MODEL, clean_body  # noqa: E402
from bench.compare import BENCH_SOURCE, TASK_FUNCTION, TASK_INSTRUCTION  # noqa: E402

# 决策任务的输出上限；与 bench/compare.py 的选择路线一致（36 token 左右就会 EOS，
# 上限不参与计时，只用于防止失控输出）。
DEFAULT_MAX_TOKENS = 64


# ---------- 一次请求的计时 ----------


@dataclass
class Timing:
    """一次请求的实测结果：四段拆开记，装饰性数字一个都不混进来。"""

    label: str
    prompt_tokens: int
    prompt_seconds: float  # 首 token 时间：库自报的 prompt_tokens / prompt_tps
    setup_seconds: float  # 首 token 到达之前、库计时起点之前的那一段
    decode_span_seconds: float  # 首个 token 到最后一个 token 之间的步进
    teardown_seconds: float  # 最后一个 token 之后到本次调用返回
    gen_tokens: int
    wall_seconds: float
    lib_decode_seconds: float  # 交叉验证：gen_tokens / generation_tps
    finish_reason: str

    @property
    def decode_tokens(self) -> int:
        """首 token 是提示处理那一段产出的，解码步进实际只有这么多步。"""
        return max(self.gen_tokens - 1, 0)

    @property
    def per_token_ms(self) -> float:
        return 1000 * self.decode_span_seconds / self.decode_tokens if self.decode_tokens else 0.0

    @property
    def decode_tps(self) -> float:
        return self.decode_tokens / self.decode_span_seconds if self.decode_span_seconds > 0 else 0.0

    def share(self, seconds: float) -> float:
        return 100 * seconds / self.wall_seconds if self.wall_seconds > 0 else 0.0


class Streamer:
    """加载一次，之后用流式接口逐 token 计时；另留一个非流式调用做对照。"""

    def __init__(self, model_path: str) -> None:
        self.model_path = model_path
        self.load_seconds = 0.0
        self._model = None
        self._tokenizer = None

    def load(self) -> float:
        from mlx_lm import load

        start = time.perf_counter()
        self._model, self._tokenizer = load(self.model_path)
        self.load_seconds = time.perf_counter() - start
        return self.load_seconds

    def prompt_text(self, messages: list[dict[str, str]]) -> str:
        return self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def detokenizer_seconds(self, rounds: int = 3) -> list[float]:
        """直接量 stream_generate 每次调用都会付的 detokenizer 构造成本。

        库里的 `TokenizerWrapper.detokenizer` 是 `return self._detokenizer_class(self)`，
        每次都新建一个（BPE 版构造要遍历 15 万条词表），不缓存。
        """
        samples = []
        for _ in range(rounds):
            start = time.perf_counter()
            detokenizer = self._tokenizer.detokenizer
            samples.append(time.perf_counter() - start)
            del detokenizer
        return samples

    def run(self, messages: list[dict[str, str]], label: str, max_tokens: int) -> tuple[str, Timing]:
        """流式生成：逐 token 收时间戳，把四段拆开。"""
        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_sampler

        prompt = self.prompt_text(messages)
        arrivals: list[float] = []
        parts: list[str] = []
        first = None
        last = None
        start = time.perf_counter()
        for response in stream_generate(
            self._model,
            self._tokenizer,
            prompt,
            max_tokens=max_tokens,
            sampler=make_sampler(temp=0.0),
        ):
            arrivals.append(time.perf_counter() - start)
            if first is None:
                first = response
            parts.append(response.text)
            last = response
        wall = time.perf_counter() - start
        if first is None or last is None:  # 理论上不会发生
            raise RuntimeError("stream_generate 没有产出任何 token")
        prompt_seconds = first.prompt_tokens / first.prompt_tps if first.prompt_tps > 0 else 0.0
        lib_decode = last.generation_tokens / last.generation_tps if last.generation_tps > 0 else 0.0
        timing = Timing(
            label=label,
            prompt_tokens=first.prompt_tokens,
            prompt_seconds=prompt_seconds,
            setup_seconds=max(arrivals[0] - prompt_seconds, 0.0),
            decode_span_seconds=arrivals[-1] - arrivals[0],
            teardown_seconds=wall - arrivals[-1],
            gen_tokens=last.generation_tokens,
            wall_seconds=wall,
            lib_decode_seconds=lib_decode,
            finish_reason=last.finish_reason or "",
        )
        return "".join(parts), timing

    def run_plain(self, messages: list[dict[str, str]], max_tokens: int) -> tuple[str, float]:
        """非流式对照：与 codejev.model.MLXEngine.generate 同一调用，用来核对基线。"""
        from mlx_lm import generate as mlx_generate
        from mlx_lm.sample_utils import make_sampler

        prompt = self.prompt_text(messages)
        start = time.perf_counter()
        text = mlx_generate(
            self._model,
            self._tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            sampler=make_sampler(temp=0.0),
            verbose=False,
        )
        return text, time.perf_counter() - start


# ---------- 输出格式 ----------


def _canonical(function_id: str, filter_field: object, return_fields: object) -> str:
    """把任何一种格式折成现有 parse_decision 认的规范 JSON，复用它的 id 校验。"""
    return json.dumps(
        {"function": function_id, "filter_field": filter_field, "return_fields": return_fields},
        ensure_ascii=False,
    )


def parse_json_format(text: str, candidates: Candidates) -> Decision:
    """现有格式：直接用生产解析器，不做任何额外宽容。"""
    return parse_decision(text, candidates)


def parse_terse_format(text: str, candidates: Candidates) -> Decision:
    """精简 JSON：只认 f / r 两个键，值仍走同一套 id 校验。"""
    obj = _json_object(text)
    if not isinstance(obj, dict):
        raise DecisionError("必须是 JSON 对象")
    extra = sorted(set(obj) - {"f", "r"})
    if extra:
        raise DecisionError(f"多余键：{', '.join(map(str, extra))}")
    if "f" not in obj or "r" not in obj:
        raise DecisionError("缺少 f 或 r")
    return parse_decision(_canonical(FUNCTION_ID, obj["f"], obj["r"]), candidates)


def parse_bare_format(text: str, candidates: Candidates) -> Decision:
    """裸 token：空白/逗号分隔，首 token 是条件 id（null/- 表示不过滤），其余是字段 id。"""
    line = text.strip().strip("`").strip()
    if not line:
        raise DecisionError("空输出")
    if any(char in line for char in "{}[]:"):
        raise DecisionError(f"不是裸 token 形状：{line[:60]}")
    tokens = [tok.strip("\"'.,;。") for tok in re.split(r"[\s,]+", line) if tok]
    if not tokens or not tokens[0]:
        raise DecisionError("空输出")
    head, rest = tokens[0], tokens[1:]
    filter_field: str | None = None if head.lower() in ("null", "none", "-") else head
    return parse_decision(_canonical(FUNCTION_ID, filter_field, rest), candidates)


# 裸 token 格式与 user 末尾“只回 JSON”那行直接冲突，所以只在这一种格式里把那行换掉；
# 候选表与任务文本逐字不动，系统提示按格式改写（这就是“换输出契约”的全部差异）。
JSON_SUFFIX = "只回 JSON，不要解释。"
BARE_SUFFIX = "只回一行 token，不要解释。"

TERSE_SYSTEM_PROMPT = (
    "你只回一个 JSON 对象，不写代码、不解释、不加围栏。\n"
    "只能使用给出的候选 id，不得发明新的 id 或字段名。\n"
    "只回这两个键：f（条件 id，不过滤时用 null）、r（字段 id 的数组，按要求的输出顺序）。\n"
    "正确形状示例：" '{"f": "c2", "r": ["f0", "f1"]}'
)

BARE_SYSTEM_PROMPT = (
    "你只回一行 token，不写代码、不解释、不加围栏、不加标点。\n"
    "只能使用给出的候选 id，不得发明新的 id 或字段名。\n"
    "形状是：<条件 id> <字段 id 1> <字段 id 2> ...\n"
    "第一个 token 是条件 id，不过滤时写 null；其余 token 是字段 id，按要求的输出顺序。\n"
    "正确形状示例：c2 f0 f1"
)

# 直白版：把上一版实测里模型的照抄行为（抄候选表的 f0=id 写法、抄字段名）
# 明确禁掉，示例改成一句短标签。实测里只有这一版被逐字执行；
# 换成“如果条件是 c2、要返回 f0 和 f1，你只回：…”这类条件句，模型会多回一个 fn0。
BARE2_SYSTEM_PROMPT = (
    "你只回一行 id，用空格分隔，不写代码、不解释、不加围栏、不加标点。\n"
    "只能使用给出的候选 id，不得发明新的 id 或字段名。\n"
    "第一个 id 是条件 id（不过滤时写 null），后面是要返回的字段 id，按要求的输出顺序。\n"
    "不要写字段名、不要写等号、不要写引号、不要写括号。\n"
    "正确的回复示例：c2 f0 f1"
)


@dataclass
class OutputFormat:
    name: str
    system: str
    user_suffix: str
    parse: Callable[[str, Candidates], Decision]
    description: str


FORMATS: tuple[OutputFormat, ...] = (
    OutputFormat(
        "json",
        DECISION_SYSTEM_PROMPT,
        JSON_SUFFIX,
        parse_json_format,
        '{"function": "fn0", "filter_field": "c2", "return_fields": ["f0", "f1"]}',
    ),
    OutputFormat(
        "terse",
        TERSE_SYSTEM_PROMPT,
        JSON_SUFFIX,
        parse_terse_format,
        '{"f": "c2", "r": ["f0", "f1"]}',
    ),
    OutputFormat(
        "bare",
        BARE_SYSTEM_PROMPT,
        BARE_SUFFIX,
        parse_bare_format,
        "c2 f0 f1",
    ),
    OutputFormat(
        "bare2",
        BARE2_SYSTEM_PROMPT,
        BARE_SUFFIX,
        parse_bare_format,
        "c2 f0 f1",
    ),
)


def adapt_messages(base: list[dict[str, str]], fmt: OutputFormat) -> list[dict[str, str]]:
    """复用 build_decision_prompt 的产出，只换系统提示与末尾那行输出要求。"""
    messages = [dict(message) for message in base]
    messages[0]["content"] = fmt.system
    tail = messages[1]["content"].rsplit("\n", 1)
    if len(tail) == 2 and tail[1].strip() == JSON_SUFFIX:
        messages[1]["content"] = f"{tail[0]}\n{fmt.user_suffix}"
    return messages


# ---------- 统计与打印 ----------


def spread(values: list[float]) -> tuple[float, float, float]:
    return min(values), sum(values) / len(values), max(values)


def print_table(headers: list[str], rows: list[list[str]]) -> None:
    widths = [len(h) for h in headers]
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))
    print("| " + " | ".join(h.ljust(w) for h, w in zip(headers, widths)) + " |")
    print("|" + "|".join("-" * (w + 2) for w in widths) + "|")
    for row in rows:
        print("| " + " | ".join(c.ljust(w) for c, w in zip(row, widths)) + " |")


def fmt3(value: float) -> str:
    return f"{value:.3f}"


def fmt2(value: float) -> str:
    return f"{value:.2f}"


def load_note() -> str:
    try:
        one, five, fifteen = os.getloadavg()
    except OSError:  # pragma: no cover - 某些平台没有 loadavg
        return "loadavg 不可用"
    return f"loadavg 1/5/15 分钟：{one:.2f}/{five:.2f}/{fifteen:.2f}"


# ---------- 实验 1：延迟拆解 ----------


def measure_split(streamer: Streamer, messages, trials: int, max_tokens: int) -> list[Timing]:
    timings: list[Timing] = []
    for index in range(trials):
        text, timing = streamer.run(messages, "stream", max_tokens)
        timings.append(timing)
        print(
            f"  #{index + 1} 前置 {fmt3(timing.setup_seconds)}s"
            f" + 提示 {timing.prompt_tokens}tok/{fmt3(timing.prompt_seconds)}s"
            f" + 解码 {timing.decode_tokens}步/{fmt3(timing.decode_span_seconds)}s"
            f"（{fmt2(timing.per_token_ms)} ms/token）"
            f" + 收尾 {fmt3(timing.teardown_seconds)}s"
            f" = {fmt3(timing.wall_seconds)}s"
        )
        if index == 0:
            print(f"     输出：{text.strip()[:160]}")
    return timings


def report_split(timings: list[Timing]) -> None:
    metrics: list[tuple[str, Callable[[Timing], float], int]] = [
        ("前置开销 setup (s)", lambda t: t.setup_seconds, 3),
        ("提示处理 prompt (s)", lambda t: t.prompt_seconds, 3),
        ("解码步进 decode (s)", lambda t: t.decode_span_seconds, 3),
        ("收尾 teardown (s)", lambda t: t.teardown_seconds, 3),
        ("总墙钟 wall (s)", lambda t: t.wall_seconds, 3),
        ("提示 token 数", lambda t: float(t.prompt_tokens), 0),
        ("解码步数（token-1）", lambda t: float(t.decode_tokens), 0),
        ("每 token 解码 (ms)", lambda t: t.per_token_ms, 2),
        ("库报解码速度 (tok/s)", lambda t: t.decode_tokens / t.lib_decode_seconds if t.lib_decode_seconds else 0.0, 1),
    ]
    rows = []
    for name, pick, digits in metrics:
        low, mean, high = spread([pick(t) for t in timings])
        rows.append([name, f"{mean:.{digits}f}", f"{low:.{digits}f}", f"{high:.{digits}f}"])
    for name, pick in (
        ("前置占比 (%)", lambda t: t.share(t.setup_seconds)),
        ("提示占比 (%)", lambda t: t.share(t.prompt_seconds)),
        ("解码占比 (%)", lambda t: t.share(t.decode_span_seconds)),
        ("收尾占比 (%)", lambda t: t.share(t.teardown_seconds)),
    ):
        low, mean, high = spread([pick(t) for t in timings])
        rows.append([name, f"{mean:.1f}", f"{low:.1f}", f"{high:.1f}"])
    print_table(["指标（均值/最小/最大）", "均值", "最小", "最大"], rows)


# ---------- 实验 2：输出格式探针 ----------


@dataclass
class FormatTrial:
    text: str
    timing: Timing
    parsed: Decision | None
    error: str


@dataclass
class FormatResult:
    fmt: OutputFormat
    trials: list[FormatTrial] = field(default_factory=list)

    def parse_ok(self) -> int:
        return sum(1 for t in self.trials if t.parsed is not None)


def measure_formats(
    streamer: Streamer,
    base_messages,
    candidates: Candidates,
    expected: Decision,
    trials: int,
    max_tokens: int,
) -> list[FormatResult]:
    results: list[FormatResult] = []
    prepared: list[list[dict[str, str]]] = []
    for fmt in FORMATS:
        results.append(FormatResult(fmt=fmt))
        prepared.append(adapt_messages(base_messages, fmt))
        print(f"\n[{fmt.name}] 期望形状：{fmt.description}")
    # 轮转着跑：机器的后台负载会漂移，四种格式按轮次交替，漂移就摊到每一种上，
    # 而不是让先跑的那种独占干净的一段。
    for index in range(trials):
        for result, messages in zip(results, prepared):
            fmt = result.fmt
            raw, timing = streamer.run(messages, fmt.name, max_tokens)
            # 与生产路径一致：MLXEngine.generate 先 clean_body 去掉 <|im_end|> 之类的
            # 对话控制符，再把正文交给解析器。少这一步裸 token 会被尾随标记判死。
            text = clean_body(raw)
            parsed: Decision | None = None
            error = ""
            try:
                parsed = fmt.parse(text, candidates)
            except DecisionError as exc:
                error = str(exc)
            result.trials.append(FormatTrial(text, timing, parsed, error))
            state = "解析通过" if parsed is not None else f"解析失败（{error}）"
            if parsed is not None and parsed != expected:
                state += "，但选得不对"
            print(
                f"  #{index + 1} {state}｜{timing.gen_tokens}tok"
                f"｜解码 {timing.decode_tokens}步/{fmt3(timing.decode_span_seconds)}s"
                f"｜墙钟 {fmt3(timing.wall_seconds)}s｜输出 {text.strip()[:80]!r}"
            )
    return results


def report_formats(results: list[FormatResult], trials: int, expected: Decision) -> None:
    rows = []
    for result in results:
        ok = result.parse_ok()
        right = sum(1 for t in result.trials if t.parsed == expected)
        distinct = len({t.text.strip() for t in result.trials})
        tokens_low, tokens_mean, tokens_high = spread([float(t.timing.gen_tokens) for t in result.trials])
        _, decode_mean, _ = spread([t.timing.decode_span_seconds for t in result.trials])
        wall_low, wall_mean, _ = spread([t.timing.wall_seconds for t in result.trials])
        _, per_token_mean, _ = spread([t.timing.per_token_ms for t in result.trials])
        _, prompt_mean, _ = spread([float(t.timing.prompt_tokens) for t in result.trials])
        rows.append(
            [
                result.fmt.name,
                f"{ok}/{trials}",
                f"{right}/{trials}",
                f"{tokens_mean:.1f}",
                f"{tokens_low:.0f}-{tokens_high:.0f}",
                f"{decode_mean:.3f}",
                f"{per_token_mean:.2f}",
                f"{wall_mean:.3f}",
                f"{wall_low:.3f}",
                f"{prompt_mean:.0f}",
                f"{'是' if distinct == 1 else '否'}",
            ]
        )
    print_table(
        [
            "格式",
            "解析通过",
            "选得正确",
            "生成 token",
            "token 范围",
            "解码均值(s)",
            "ms/token",
            "墙钟均值(s)",
            "墙钟最小(s)",
            "提示 token",
            "逐字复现",
        ],
        rows,
    )
    print(
        "\n说明：温度 0 下同一请求逐字复现，所以同一格式的多次试验衡量的是耗时波动，"
        "解析通过率是确定性的形状判断，不是多次独立样本。"
    )


# ---------- 主流程 ----------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="选择路径延迟拆解与输出格式成本探针")
    parser.add_argument("--trials", type=int, default=9, help="每次测量的试验次数（至少 5）")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    args = parser.parse_args(argv)

    trials = max(args.trials, 5)
    candidates = extract(BENCH_SOURCE, TASK_FUNCTION)
    base_messages = build_decision_prompt(TASK_INSTRUCTION, candidates)
    expected = Decision(
        function_id=FUNCTION_ID,
        filter_field=next(c.id for c in candidates.conditions if c.name == "active"),
        return_fields=tuple(
            c.id for name in ("id", "name") for c in candidates.fields if c.name == name
        ),
    )
    print("候选项：" + ", ".join(f"{c.id}={c.name}" for c in candidates.fields))
    print("条件项：" + ", ".join(f"{c.id}={c.name}" for c in candidates.conditions))
    print(f"期望决策：{expected}")
    print(f"开始前：{load_note()}\n")

    print(f"加载模型：{args.model}")
    streamer = Streamer(args.model)
    load_seconds = streamer.load()
    print(f"加载耗时 {fmt2(load_seconds)}s（不计入稳态；之后所有试验都用这个已加载的模型）")

    # 预热：让 MLX 编译、内存分配与 KV 缓存分配先发生一次，两次都丢弃。
    warm_text, warm = streamer.run(base_messages, "warmup", args.max_tokens)
    _, warm_plain = streamer.run_plain(base_messages, args.max_tokens)
    print(
        f"预热 2 次已丢弃：流式墙钟 {fmt3(warm.wall_seconds)}s，"
        f"非流式墙钟 {fmt3(warm_plain)}s\n"
    )

    detok = streamer.detokenizer_seconds()
    detok_low, detok_mean, detok_high = spread(detok)
    print(
        f"单独量 detokenizer 构造：均值 {fmt3(detok_mean)}s，"
        f"{fmt3(detok_low)}–{fmt3(detok_high)}s"
        f"（每次 generate 调用都会付一次，见报告里的“前置开销”）\n"
    )

    print(f"=== 实验 1：延迟拆解（{trials} 次同一决策请求，1.5B 4bit，温度 0）===")
    split_timings = measure_split(streamer, base_messages, trials, args.max_tokens)
    report_split(split_timings)

    plain_seconds = []
    for _ in range(trials):
        _, seconds = streamer.run_plain(base_messages, args.max_tokens)
        plain_seconds.append(seconds)
    low, mean, high = spread(plain_seconds)
    print(
        f"\n非流式对照（现有 MLXEngine 走的同一调用）：均值 {fmt3(mean)}s，{fmt3(low)}–{fmt3(high)}s"
        f"；流式墙钟均值 {fmt3(sum(t.wall_seconds for t in split_timings) / len(split_timings))}s。"
        "两者接近，说明按流式拆出来的四段就是这条路径的真实预算。"
    )

    print(f"\n=== 实验 2：输出格式探针（每种 {trials} 次，同一候选表与任务文本）===")
    format_results = measure_formats(
        streamer, base_messages, candidates, expected, trials, args.max_tokens
    )
    print()
    report_formats(format_results, trials, expected)

    print("\n=== 判定 ===")
    prompt_mean = sum(t.prompt_seconds for t in split_timings) / len(split_timings)
    setup_mean = sum(t.setup_seconds for t in split_timings) / len(split_timings)
    decode_mean = sum(t.decode_span_seconds for t in split_timings) / len(split_timings)
    teardown_mean = sum(t.teardown_seconds for t in split_timings) / len(split_timings)
    wall_mean = sum(t.wall_seconds for t in split_timings) / len(split_timings)
    print(
        f"四段预算：前置 {fmt3(setup_mean)}s（{setup_mean / wall_mean * 100:.1f}%）"
        f" + 提示 {fmt3(prompt_mean)}s（{prompt_mean / wall_mean * 100:.1f}%）"
        f" + 解码 {fmt3(decode_mean)}s（{decode_mean / wall_mean * 100:.1f}%）"
        f" + 收尾 {fmt3(teardown_mean)}s（{teardown_mean / wall_mean * 100:.1f}%）"
        f" = {fmt3(wall_mean)}s"
    )
    floor = setup_mean + prompt_mean + teardown_mean
    print(
        f"输出缩不到零的底：前置 + 提示 + 收尾 = {fmt3(floor)}s"
        f"（{floor / wall_mean * 100:.0f}% 与输出长度无关），"
        f"只有解码那 {fmt3(decode_mean)}s（{decode_mean / wall_mean * 100:.0f}%）是缩短输出能动的"
    )
    json_result = next(r for r in format_results if r.fmt.name == "json")
    json_tokens = sum(t.timing.gen_tokens for t in json_result.trials) / len(json_result.trials)
    json_low, _, _ = spread([t.timing.wall_seconds for t in json_result.trials])
    for result in format_results:
        if result.fmt.name == "json":
            continue
        wall_low, wall, _ = spread([t.timing.wall_seconds for t in result.trials])
        _, tokens, _ = spread([float(t.timing.gen_tokens) for t in result.trials])
        ok = result.parse_ok()
        print(
            f"{result.fmt.name}: 解析 {ok}/{trials}，输出 {tokens:.0f} vs {json_tokens:.0f} token，"
            f"墙钟 {fmt3(wall)}s（最小 {fmt3(wall_low)}s；json {fmt3(wall_mean)}s / 最小 {fmt3(json_low)}s），"
            f"均值口径省下 {fmt3(wall_mean - wall)}s / {(1 - wall / wall_mean) * 100:.0f}%"
        )
    print(f"结束前：{load_note()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
