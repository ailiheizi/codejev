"""小模型引擎：常驻进程，只回正文。

大模型发短指令，这里把指令交给本机小代码模型，拿回正文。
模型不做业务判断、不解释、不规划；加载一次后常驻，避免每次重新加载的等待。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from chooseonly.adapter import build_messages
from chooseonly.contracts import Brief

DEFAULT_MODEL = str(
    Path(__file__).resolve().parent.parent / "models" / "Qwen2.5-Coder-1.5B-Instruct-4bit"
)

# 模型偶尔把对话控制符一起吐出来；它们不是正文。
CHAT_MARKERS = ("<|im_end|>", "<|im_start|>", "<|endoftext|>", "<|eot_id|>", "<|end▁of▁sentence|>")


def clean_body(text: str) -> str:
    """去掉对话控制符，只留正文。"""
    for marker in CHAT_MARKERS:
        text = text.replace(marker, "")
    return text.strip()


@dataclass
class Stats:
    """一次请求的耗时；用于判断是否真的更快，而不是只看 token 速度。"""

    load_seconds: float = 0.0
    prompt_tokens: int = 0
    generated_tokens: int = 0
    generate_seconds: float = 0.0

    @property
    def tokens_per_second(self) -> float:
        return self.generated_tokens / self.generate_seconds if self.generate_seconds > 0 else 0.0

    def line(self) -> str:
        if self.generate_seconds <= 0:
            return "耗时：未计时"
        return (
            f"耗时：生成 {self.generate_seconds:.2f}s"
            f"（{self.generated_tokens} tokens, {self.tokens_per_second:.1f} tok/s）"
            f"｜输入 {self.prompt_tokens} tokens"
        )


class Engine(Protocol):
    """任何能接收消息并返回正文的后端。"""

    def generate(self, messages: list[dict[str, str]], max_tokens: int = 512) -> tuple[str, Stats]:
        ...


class MLXEngine:
    """MLX-LM 本地推理；模型只加载一次。"""

    def __init__(self, model_path: str = DEFAULT_MODEL) -> None:
        self.model_path = model_path
        self._model = None
        self._tokenizer = None
        self.load_seconds = 0.0

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        from mlx_lm import load  # 延迟导入：没有模型也能跑测试和干跑

        start = time.perf_counter()
        self._model, self._tokenizer = load(self.model_path)
        self.load_seconds = time.perf_counter() - start

    def generate(self, messages: list[dict[str, str]], max_tokens: int = 512) -> tuple[str, Stats]:
        """生成正文；直驱 generate_step，只在最后解码一次。

        `mlx_lm.generate` 会走 `stream_generate`，而后者每次调用都新建一个流式
        detokenizer（`TokenizerWrapper.detokenizer` 是 `return self._detokenizer_class(self)`，
        构造时要遍历 15 万条词表，不缓存）。实测这次构造成本 0.085–0.10 s/次，
        且与模型、提示、输出长度都无关。这里直接驱动 `generate_step`，
        把整段 token 一次 decode，省掉这笔固定开销；温度 0 下逐 token 与旧实现相同，
        正文逐字一致（见 bench/speedup_check.py 的逐字对比）。
        """
        self._ensure_loaded()
        import mlx.core as mx
        from mlx_lm.generate import generate_step, generation_stream, wired_limit
        from mlx_lm.sample_utils import make_sampler

        prompt = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        # 与 mlx_lm.stream_generate 的编码规则逐字一致（含 add_special_tokens 的推断），
        # 保证送进模型的 token 串与旧实现完全相同。
        add_special_tokens = self._tokenizer.bos_token is None or not prompt.startswith(
            self._tokenizer.bos_token
        )
        prompt_ids = self._tokenizer.encode(prompt, add_special_tokens=add_special_tokens)

        stats = Stats(load_seconds=self.load_seconds)
        stats.prompt_tokens = len(prompt_ids)
        eos_ids = self._tokenizer.eos_token_ids
        sampler = make_sampler(temp=0.0)  # 按指令产出，不要随机发挥
        prompt_array = mx.array(prompt_ids)

        tokens: list[int] = []
        start = time.perf_counter()
        # 与 stream_generate 一样在 wired limit 内生成（对大模型是必要的内存设置）。
        with wired_limit(self._model, [generation_stream]):
            for token, _logprobs in generate_step(
                prompt_array,
                self._model,
                max_tokens=max_tokens,
                sampler=sampler,
            ):
                if token in eos_ids:
                    break  # EOS 不算正文，也不进 token 计数（与 stream_generate 一致）
                tokens.append(token)
        stats.generate_seconds = time.perf_counter() - start
        body = clean_body(self._tokenizer.decode(tokens))
        # 旧实现是重新 encode 正文来数 token；这里直接数真正生成并解码的 token。
        stats.generated_tokens = len(tokens)
        return body, stats

    def close(self) -> None:
        self._model = None
        self._tokenizer = None


@dataclass
class ScriptedEngine:
    """测试与干跑用：按预先给定的正文返回，不加载模型。"""

    responses: list[str]
    calls: list[list[dict[str, str]]] = field(default_factory=list)

    def generate(self, messages: list[dict[str, str]], max_tokens: int = 512) -> tuple[str, Stats]:
        self.calls.append(messages)
        if not self.responses:
            raise RuntimeError("ScriptedEngine 没有更多预设回复")
        return self.responses.pop(0), Stats(generate_seconds=0.0)


def request_body(
    engine: Engine, brief: Brief, max_tokens: int = 512
) -> tuple[str, Stats]:
    """把一条 Brief 变成模型请求，返回原始正文与耗时。"""
    messages = build_messages(brief)
    return engine.generate(messages, max_tokens=max_tokens)
