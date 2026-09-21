"""线性/混合架构的第一次实测：能不能写代码、有多快、长上下文内存是否恒定。

背景：RWKV-7 论文里没有任何 HumanEval/MBPP 数字（检索确认为 0 命中），
所以"线性架构能不能写代码"目前没有权威公开数据。
本脚本在**同一台机器、同一批任务**上对比：

  基线  Qwen2.5-Coder-1.5B-Instruct-4bit（自回归 Transformer）
  候选  LFM2-350M-4bit（极小混合）
        Falcon-H1-0.5B-Instruct-4bit（注意力 + SSM 混合）
        RWKV-7-1.5B（mlx-6bit，纯线性 RNN）

三件事：
  1. 能不能加载、能不能产出合法代码（运行时验证，不看像不像）
  2. 解码速度（tok/s）与单次耗时
  3. **长上下文时峰值内存**——线性架构的真正卖点是状态恒定，
     这直接对应"大模型那部分要塞很大的数据"的需求
"""

from __future__ import annotations

import ast
import time
from pathlib import Path

from azfls.adapter import build_messages, to_artifact
from azfls.contracts import Action, Brief, Kind

MODELS_ROOT = Path(__file__).resolve().parent.parent / "models"

MODELS = (
    ("Qwen1.5B（自回归基线）", "Qwen2.5-Coder-1.5B-Instruct-4bit"),
    ("LFM2-350M（混合）", "LFM2-350M-4bit"),
    ("Falcon-H1-0.5B（混合）", "Falcon-H1-0.5B-Instruct-4bit"),
    ("RWKV-7-1.5B（纯线性）", "rwkv7-1.5B-g1c-20260110-ctx8192-mlx-6bit-test"),
)

SOURCE = '''"""订单列表：当前返回全部字段，且不过滤。"""


def paid_orders(orders):
    """返回订单，保持原顺序。"""
    rows = []
    for order in orders:
        rows.append(
            {"order_id": order.order_id, "total": order.total, "paid": order.paid}
        )
    return rows
'''

INSTRUCTION = "修改函数：只保留 paid 为真的项，返回 order_id 和 total，保持原顺序，其他不变。"

SAMPLE = [
    type("O", (), {"paid": True, "order_id": "A", "total": 5})(),
    type("O", (), {"paid": False, "order_id": "B", "total": 9})(),
    type("O", (), {"paid": True, "order_id": "C", "total": 7})(),
]
EXPECT = [{"order_id": "A", "total": 5}, {"order_id": "C", "total": 7}]


def check(source: str) -> tuple[bool, tuple[str, ...]]:
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return False, (f"语法错误: {exc.msg}",)
    namespace: dict[str, object] = {}
    try:
        exec(compile(tree, "<candidate>", "exec"), namespace)  # noqa: S102
    except Exception as exc:  # noqa: BLE001
        return False, (f"无法执行: {type(exc).__name__}",)
    fn = namespace.get("paid_orders")
    if not callable(fn):
        return False, ("缺少函数 paid_orders",)
    try:
        got = fn(SAMPLE)
    except Exception as exc:  # noqa: BLE001
        return False, (f"调用失败: {type(exc).__name__}",)
    return (got == EXPECT), (() if got == EXPECT else (f"结果不符: {got}",))


def probe(model_dir: str, label: str, trials: int = 3) -> None:
    print()
    print("=" * 84)
    print(f"{label}　（{model_dir}）")
    print("=" * 84)
    path = MODELS_ROOT / model_dir
    if not (path / "config.json").exists():
        print("  未下载，跳过")
        return

    try:
        import mlx.core as mx
        from mlx_lm import load

        started = time.perf_counter()
        model, tokenizer = load(str(path))
        load_seconds = time.perf_counter() - started
    except Exception as exc:  # noqa: BLE001
        print(f"  加载失败：{type(exc).__name__}: {str(exc)[:110]}")
        return

    brief = Brief(
        instruction=INSTRUCTION,
        target="orders.py",
        action=Action.REPLACE,
        kind=Kind.CODE,
        context=SOURCE,
        original=SOURCE,
    )
    messages = build_messages(brief)
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    print(f"  加载 {load_seconds:.2f}s｜提示 {len(tokenizer.encode(prompt))} token")
    passed = 0
    tokens: list[int] = []
    seconds: list[float] = []
    first_fail = ""
    for _ in range(trials):
        from mlx_lm import generate as mlx_generate
        from mlx_lm.sample_utils import make_sampler

        start = time.perf_counter()
        text = mlx_generate(model, tokenizer, prompt=prompt, max_tokens=320,
                            sampler=make_sampler(temp=0.0), verbose=False)
        elapsed = time.perf_counter() - start
        body = to_artifact(brief, text).body
        ok, why = check(body)
        passed += int(ok)
        out_tokens = len(tokenizer.encode(body))
        tokens.append(out_tokens)
        seconds.append(elapsed)
        if not ok and not first_fail:
            first_fail = "; ".join(why)[:70]

    avg_tok = sum(tokens) / len(tokens)
    avg_s = sum(seconds) / len(seconds)
    rate = avg_tok / avg_s if avg_s else 0
    print(f"  通过 {passed}/{trials}｜输出 {avg_tok:.0f} tok｜{avg_s:.2f}s｜{rate:.0f} tok/s"
          + (f"｜首次失败: {first_fail}" if first_fail else ""))

    # 长上下文下的峰值内存：线性架构的状态应当恒定，Transformer 的 KV 随长度线性增长
    print("  长上下文峰值内存：")
    for multiplier in (1, 4, 16):
        long_context = SOURCE * multiplier
        long_brief = Brief(instruction=INSTRUCTION, target="orders.py", action=Action.REPLACE,
                           kind=Kind.CODE, context=long_context, original=SOURCE)
        long_prompt = tokenizer.apply_chat_template(
            build_messages(long_brief), tokenize=False, add_generation_prompt=True
        )
        ids = tokenizer.encode(long_prompt)
        mx.reset_peak_memory()
        try:
            from mlx_lm import generate as mlx_generate
            from mlx_lm.sample_utils import make_sampler

            mlx_generate(model, tokenizer, prompt=long_prompt, max_tokens=16,
                         sampler=make_sampler(temp=0.0), verbose=False)
            peak = mx.get_peak_memory() / 1e9
            print(f"    {len(ids):>6d} token  →  峰值 {peak:.2f} GB")
        except Exception as exc:  # noqa: BLE001
            print(f"    {len(ids):>6d} token  →  失败 {type(exc).__name__}")
    del model, tokenizer
    mx.clear_cache()


def main() -> int:
    print("同一台机器、同一批任务、温度 0；判断用运行时行为。")
    print("任务：只保留 paid 为真的项，返回 order_id 和 total，保持原顺序。")
    for label, model_dir in MODELS:
        probe(model_dir, label)
    print()
    print("读法：线性/混合架构的价值不只在 tok/s，还在长上下文时内存是否保持恒定。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
