"""扩散语言模型 vs 自回归基线：在本项目的固定任务上做同题对照。

要回答的问题只有一个：把本机小模型换成扩散模型后，
同一个固定任务是否还做得成、是否更快。不引入评审模型，
判断标准复用 bench/compare.py 的运行时行为检查（exec 后真的调用函数）。

三类后端：
  baseline     —— 现有自回归基线 Qwen2.5-Coder-1.5B-Instruct-4bit（mlx-lm）
  nemotron-ar  —— Nemotron-Labs-Diffusion-3B-4bit 用自回归模式解码（同一份权重）
  diffusion    —— 同一份权重用扩散模式解码

同一权重换解码模式，是为了把"扩散解码本身"和"换了个模型"两件事分开。
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

from azfls.adapter import to_artifact
from azfls.contracts import Action, Brief, Kind
from azfls.model import MLXEngine, Stats, clean_body

from bench.compare import (
    BENCH_SOURCE,
    TASK_FUNCTION,
    check_runtime_behaviour,
    looks_like_explanation,
    looks_truncated,
    run_generate,
    run_select,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DIFFUSION_MODEL = str(PROJECT_ROOT / "models" / "Nemotron-Labs-Diffusion-3B-4bit")


@dataclass
class DiffStats(Stats):
    """在项目原有耗时字段上，补上扩散解码特有的计数。"""

    denoising_forwards: int = 0   # 真正的去噪前向次数（NFE）
    accepted_tokens: int = 0      # 整个画布上被接受的位置数
    generation_mode: str = ""

    @property
    def tpf(self) -> float:
        """tokens per forward：每次前向平均产出多少 token。"""
        return self.accepted_tokens / self.denoising_forwards if self.denoising_forwards else 0.0

    @property
    def seconds_per_forward(self) -> float:
        return self.generate_seconds / self.denoising_forwards if self.denoising_forwards else 0.0


class DiffusionEngine:
    """用 mlx-vlm 驱动 Nemotron-Labs-Diffusion；接口与 azfls.model.Engine 一致。"""

    def __init__(self, model_path: str = DIFFUSION_MODEL, generation_mode: str = "diffusion",
                 gen_kwargs: dict | None = None) -> None:
        self.model_path = model_path
        self.generation_mode = generation_mode
        self.gen_kwargs = dict(gen_kwargs or {})
        self._model = None
        self._processor = None
        self._tokenizer = None
        self.load_seconds = 0.0

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        from mlx_vlm import load

        start = time.perf_counter()
        try:
            self._model, self._processor = load(self.model_path, trust_remote_code=True)
        except TypeError:
            # 该版本 load() 不接受 trust_remote_code 时退回默认签名。
            self._model, self._processor = load(self.model_path)
        self._tokenizer = getattr(self._processor, "tokenizer", self._processor)
        self.load_seconds = time.perf_counter() - start

    def generate(self, messages: list[dict[str, str]], max_tokens: int = 512) -> tuple[str, Stats]:
        from mlx_vlm.generate import generate as vlm_generate
        from mlx_vlm.prompt_utils import apply_chat_template

        self._ensure_loaded()
        config = getattr(self._model, "config", None)
        prompt = apply_chat_template(
            self._processor, config, messages, add_generation_prompt=True
        )

        stats = DiffStats(load_seconds=self.load_seconds, generation_mode=self.generation_mode)
        stats.prompt_tokens = len(self._tokenizer.encode(prompt))

        start = time.perf_counter()
        result = vlm_generate(
            self._model,
            self._processor,
            prompt,
            max_tokens=max_tokens,
            temperature=0.0,  # 与基线一致：按指令产出，不随机发挥
            verbose=False,
            generation_mode=self.generation_mode,
            **self.gen_kwargs,
        )
        stats.generate_seconds = time.perf_counter() - start

        body = clean_body(result.text)
        stats.generated_tokens = len(self._tokenizer.encode(body))
        stats.denoising_forwards = int(getattr(result, "diffusion_denoising_steps", 0) or 0)
        stats.accepted_tokens = int(getattr(result, "diffusion_work_tokens", 0) or 0)
        return body, stats

    def close(self) -> None:
        self._model = None
        self._processor = None
        self._tokenizer = None


class TokenBudgetEngine:
    """bench/compare.py 把 max_tokens 写死成 1024（生成）和 64（选择）。

    扩散解码的开销跟画布长度成正比，不跟真实输出长度成正比，
    所以这里按项目里的原值重新映射，让本次实验能单独调这两个预算。
    """

    def __init__(self, inner, mapping: dict[int, int]) -> None:
        self.inner = inner
        self.mapping = dict(mapping)
        self.load_seconds = 0.0

    def _ensure_loaded(self) -> None:
        self.inner._ensure_loaded()  # noqa: SLF001 - 只是转发
        self.load_seconds = self.inner.load_seconds

    def generate(self, messages: list[dict[str, str]], max_tokens: int = 512):
        return self.inner.generate(messages, max_tokens=self.mapping.get(max_tokens, max_tokens))

    def close(self) -> None:
        self.inner.close()


def summarize(report, path: str, backend: str) -> dict:
    """把一轮 Outcome 汇总成一行可比较的数字。"""
    rows = report.by_path(path)
    if not rows:
        return {}
    ok = sum(1 for r in rows if r.ok)
    walls = [r.wall_seconds for r in rows]

    def _num(attr: str) -> list[float]:
        return [getattr(r.stats, attr, 0) or 0 for r in rows]

    gens = _num("generate_seconds")
    toks = _num("generated_tokens")
    fwds = _num("denoising_forwards")
    acc = _num("accepted_tokens")

    def avg(xs: list[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    # 只对真正产出正文的回合统计速度，避免把"失败但很快"算成优势。
    warm = [r for r in rows if (getattr(r.stats, "generated_tokens", 0) or 0) > 0]
    return {
        "backend": backend,
        "path": path,
        "trials": len(rows),
        "passed": ok,
        "avg_wall_s": round(avg(walls), 3),
        "avg_generate_s": round(avg(gens), 3),
        "avg_tokens": round(avg(toks), 1),
        "max_tokens": max(toks) if toks else 0,
        "avg_forwards": round(avg(fwds), 1) if any(fwds) else None,
        "avg_tpf": round(sum(acc) / sum(fwds), 2) if sum(fwds) else None,
        "avg_s_per_forward": round(sum(gens) / sum(fwds), 4) if sum(fwds) else None,
        "tokens_per_s": round(avg([r.stats.tokens_per_second for r in rows]), 1),
        "warm_trials": len(warm),
        "failures": sorted({reason.split(":")[0] for r in rows if not r.ok for reason in r.reasons}),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="扩散模型 vs 自回归基线同题对照")
    parser.add_argument("--backend", required=True,
                        choices=["baseline", "nemotron-ar", "diffusion"])
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--gen-max-tokens", type=int, default=1024,
                        help="生成路线的 max_tokens；基线默认 1024")
    parser.add_argument("--select-max-tokens", type=int, default=64)
    parser.add_argument("--json", default=None, help="把汇总写到该文件")
    parser.add_argument("--gen-kwargs", default=None,
                        help='透传给 mlx-vlm 的 JSON，例如 \'{"block_length":128}\'')
    parser.add_argument("--show-raw", action="store_true", help="打印每次原始输出")
    args = parser.parse_args(argv)

    if args.backend == "baseline":
        engine = MLXEngine()
    else:
        mode = "ar" if args.backend == "nemotron-ar" else "diffusion"
        extra = json.loads(args.gen_kwargs) if args.gen_kwargs else None
        engine = DiffusionEngine(generation_mode=mode, gen_kwargs=extra)

    engine = TokenBudgetEngine(
        engine, {1024: args.gen_max_tokens, 64: args.select_max_tokens}
    )

    print(f"[{args.backend}] 加载模型...", flush=True)
    engine._ensure_loaded()  # noqa: SLF001 - 常驻生效后才开始计时
    print(f"[{args.backend}] 加载完成 {engine.load_seconds:.2f}s\n", flush=True)

    # 计时口径要一致：两条路线的 max_tokens 由这里统一注入。
    report = _run(engine, args)

    print(f"=== [{args.backend}] 汇总 ===")
    rows = []
    for path in ("generate", "select"):
        row = summarize(report, path, args.backend)
        if row:
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False))
    print()
    print(report.summary())
    detail = report.failure_detail()
    if detail:
        print("\n=== 未通过明细 ===")
        print(detail)

    if args.json:
        Path(args.json).write_text(json.dumps(rows, ensure_ascii=False, indent=2))
    return 0


def _run(engine, args):
    """跑两条路线；max_tokens 在这里统一，保证与基线同题同参。"""
    report = run_generate(engine, BENCH_SOURCE, args.trials)
    for outcome in run_select(engine, BENCH_SOURCE, args.trials).results:
        report.add(outcome)
    return report


if __name__ == "__main__":
    raise SystemExit(main())
