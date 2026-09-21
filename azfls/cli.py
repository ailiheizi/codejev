"""命令行入口：一条指令 → 一个产物 → 一次确认 → 一次写入。

见 docs/01-system-design.md 与 docs/06-artifact-protocol.md：
大模型给短指令，小模型只回正文，适配器负责包装与展示差异，用户只在需要取舍时确认；
没有确认就不写盘。本文件只做接线：解析参数、调用小模型、展示 diff、走一次确认门。
不做多轮重试、不做多文件编排，也不把模型输出当成路径或确认。

两条产出路径：ask 让小模型自由写正文；select 走 decide.py 的选择式路径，
宿主提取候选、小模型只选 id、宿主确定性组装，适合固定任务形状。

    .venv/bin/python -m azfls.cli ask -i “只保留 active 的项” -t app/users.py
    .venv/bin/python -m azfls.cli select -i “只保留 active 的项” -t app/users.py
    .venv/bin/python -m azfls.cli check
"""

from __future__ import annotations

import argparse
import importlib.util
import time
from dataclasses import replace
from pathlib import Path

from azfls.adapter import display, render_diff, summarize, to_artifact
from azfls.contracts import Action, Brief, Kind, resolve_target
from azfls.decide import DecisionError, describe_decision, run_decision
from azfls.gate import Gate, GateError
from azfls.model import DEFAULT_MODEL, MLXEngine, Stats, request_body


def _make_engine(args: argparse.Namespace) -> MLXEngine:
    """按参数创建本机引擎；测试在这里换成 ScriptedEngine，不读权重。"""
    return MLXEngine(args.model or DEFAULT_MODEL)


def _read_existing(workspace: Path, target: str) -> str | None:
    """读取工作区内的现有正文，用于真实 diff 与默认原文。

    越界目标、目录或读不到的内容都返回 None；拒绝由确认门统一给出。
    """
    try:
        path = resolve_target(workspace, target)
    except ValueError:
        return None
    if path.is_dir() or not path.exists():
        return None
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None


def _print_timing(args: argparse.Namespace, stats: Stats) -> None:
    """打印一行耗时；--no-timing 时保持安静。"""
    if not args.no_timing:
        print(stats.line())


def _print_wall_clock(args: argparse.Namespace, seconds: float) -> None:
    """打印一行墙钟耗时；--no-timing 时保持安静。

    run_decision 只返回产物、决策和候选，引擎的 Stats 在内部丢弃；
    这里报的是 CLI 自己量到的墙钟（含模型加载），并明确标注口径。
    """
    if not args.no_timing:
        print(f"耗时（墙钟，含加载）：{seconds:.2f}s")


def _ask_confirmation() -> bool:
    """问一次是否写入；只有明确回答 y/yes 才算确认。

    非交互环境下 input 会抛 EOFError；那等于没人确认，按拒绝处理。
    """
    try:
        answer = input("写入？[y/N] ")
    except EOFError:
        return False
    return answer.strip().lower() in ("y", "yes")


def run_ask(args: argparse.Namespace) -> int:
    """ask：一条短指令拿回正文，展示差异，确认后写入目标文件。"""
    target = args.target
    if not target or target.strip() != target:
        print(f"目标路径不合法：{target!r}")
        return 2
    workspace = Path(args.workspace)

    # 原文：--context-file 优先，其次 --context；都没有时再用磁盘上的现有正文。
    context = args.context
    if args.context_file:
        try:
            context = Path(args.context_file).read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            print(f"读取原文文件失败：{exc}")
            return 1

    brief = Brief(
        instruction=args.instruction,
        target=target,
        action=Action(args.action),
        kind=Kind(args.kind),
        context=context,
        keep=tuple(args.keep or ()),
    )
    # 目标已存在时，original 始终取磁盘正文，保证 diff 对着现场文件；
    # 没有给原文时，现有正文也直接当原文用。
    existing = _read_existing(workspace, target)
    if existing is not None:
        brief = replace(brief, original=existing)
        if not context.strip():
            brief = replace(brief, context=existing)

    try:
        engine = _make_engine(args)
        raw, stats = request_body(engine, brief, max_tokens=args.max_tokens)
    except Exception as exc:  # 模型缺失、加载或推理失败：不进写入流程
        print(f"小模型未产出正文：{exc}")
        return 1

    artifact = to_artifact(brief, raw)
    print(summarize(artifact))
    for note in artifact.notes:
        print(f"提示：{note}")
    print(render_diff(brief.original, artifact.body, artifact.target))

    try:
        gate = Gate(workspace)
        proposal = gate.propose(artifact)
    except (GateError, ValueError, OSError) as exc:
        print(f"门控拒绝：{exc}")
        return 2

    if args.dry_run:
        print(display(artifact))
        _print_timing(args, stats)
        print("干跑：未写入。")
        return 0

    if args.yes:
        confirmed = True
    else:
        confirmed = _ask_confirmation()
    try:
        approval = gate.approve(proposal, confirmed)
        written = gate.apply(approval, proposal)
    except (GateError, ValueError, OSError) as exc:
        print(f"门控拒绝：{exc}")
        return 2

    print(f"已写入 {written}")
    _print_timing(args, stats)
    return 0


def run_select(args: argparse.Namespace) -> int:
    """select：走选择式路径，小模型只选候选 id，宿主确定性组装，确认后写入。

    目标文件必须已存在——这条路径重写现有函数，不能新建文件；
    决策不合法退出 3，门控拒绝退出 2，方便调用方区分两种失败。
    """
    target = args.target
    if not target or target.strip() != target:
        print(f"目标路径不合法：{target!r}")
        return 2
    workspace = Path(args.workspace)

    # 选择式路径重写现有函数：目标必须已经在工作区里，读不出来就是硬失败。
    try:
        path = resolve_target(workspace, target)
    except ValueError as exc:
        print(f"目标路径不合法：{exc}")
        return 2
    if path.is_dir() or not path.exists():
        print(f"目标文件不存在：{target}（select 只能重写现有函数，不能新建文件）")
        return 1
    try:
        source = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        print(f"读取目标文件失败：{exc}")
        return 1

    started = time.perf_counter()
    try:
        engine = _make_engine(args)
        artifact, decision, candidates = run_decision(
            engine, args.instruction, source, target, args.function
        )
    except DecisionError as exc:
        print(f"决策不合法：{exc}")
        return 3
    except Exception as exc:  # 模型缺失、加载或推理失败：不进写入流程
        print(f"小模型未产出决策：{exc}")
        return 1

    print(describe_decision(candidates, decision))
    for note in artifact.notes:
        print(f"提示：{note}")
    print(render_diff(source, artifact.body, target))

    try:
        gate = Gate(workspace)
        proposal = gate.propose(artifact)
    except (GateError, ValueError, OSError) as exc:
        print(f"门控拒绝：{exc}")
        return 2

    if args.dry_run:
        print(display(artifact))
        _print_wall_clock(args, time.perf_counter() - started)
        print("干跑：未写入。")
        return 0

    if args.yes:
        confirmed = True
    else:
        confirmed = _ask_confirmation()
    try:
        approval = gate.approve(proposal, confirmed)
        written = gate.apply(approval, proposal)
    except (GateError, ValueError, OSError) as exc:
        print(f"门控拒绝：{exc}")
        return 2

    print(f"已写入 {written}")
    _print_wall_clock(args, time.perf_counter() - started)
    return 0


def run_check(args: argparse.Namespace) -> int:
    """check：只看模型目录与 mlx_lm 是否就绪，不加载模型，保持秒回。"""
    model_dir = Path(DEFAULT_MODEL)
    exists = model_dir.is_dir()
    print(f"{'模型目录：存在' if exists else '模型目录：缺失'} {model_dir}")
    try:
        ready = importlib.util.find_spec("mlx_lm") is not None
    except (ImportError, ValueError):
        ready = False
    print("mlx_lm：可用" if ready else "mlx_lm：不可用（pip install mlx-lm）")
    if not exists:
        print("先准备模型目录，再执行 ask。")
    return 0 if exists else 1


def build_parser() -> argparse.ArgumentParser:
    """构造解析器：ask 自由生成，select 选择式产出，check 环境自检。"""
    parser = argparse.ArgumentParser(
        prog="azfls",
        description="大模型发短指令，小模型快速产出；展示差异，确认后才写入。",
    )
    sub = parser.add_subparsers(dest="command", required=True, metavar="{ask,select,check}")

    ask = sub.add_parser("ask", help="发一条短指令，得到正文，确认后写入目标文件")
    ask.add_argument("-i", "--instruction", required=True, help="短指令：做什么、怎么改")
    ask.add_argument("-t", "--target", required=True, help="目标文件，工作区内的相对路径")
    ask.add_argument("-w", "--workspace", default=".", help="工作区根目录（默认当前目录）")
    ask.add_argument(
        "--action",
        choices=[item.value for item in Action],
        default=Action.REPLACE.value,
        help="create 新建 / replace 整体替换 / edit 只输出改动段（默认 replace）",
    )
    ask.add_argument(
        "--kind",
        choices=[item.value for item in Kind],
        default=Kind.CODE.value,
        help="正文类型：code / json / text（默认 code）",
    )
    ask.add_argument("--context", default="", help="必要原文；也可用 --context-file 从文件读")
    ask.add_argument("--context-file", default=None, help="从文件读取原文（优先于 --context）")
    ask.add_argument("--keep", action="append", default=None, help="必须保留的行为，可重复")
    ask.add_argument("--max-tokens", type=int, default=512, help="本次生成上限（默认 512）")
    ask.add_argument("--model", default=None, help="覆盖模型目录；默认用 model.DEFAULT_MODEL")
    ask.add_argument("--dry-run", action="store_true", help="只展示产物和差异，不询问、不写入")
    ask.add_argument("-y", "--yes", action="store_true", help="预先确认写入（脚本用）；仍走完整校验")
    ask.add_argument("--no-timing", action="store_true", help="不打印耗时")
    ask.set_defaults(handler=run_ask)

    select = sub.add_parser(
        "select", help="选择式产出：宿主给候选，小模型只选 id，宿主组装正文"
    )
    select.add_argument("-i", "--instruction", required=True, help="短指令：做什么、怎么改")
    select.add_argument("-t", "--target", required=True, help="目标文件，工作区内的相对路径（必须已存在）")
    select.add_argument("-w", "--workspace", default=".", help="工作区根目录（默认当前目录）")
    select.add_argument(
        "--function",
        default=None,
        help="只改这个函数；默认由 decide.extract 取第一个公开函数",
    )
    select.add_argument("--model", default=None, help="覆盖模型目录；默认用 model.DEFAULT_MODEL")
    select.add_argument("--dry-run", action="store_true", help="只展示决策和差异，不询问、不写入")
    select.add_argument("-y", "--yes", action="store_true", help="预先确认写入（脚本用）；仍走完整校验")
    select.add_argument("--no-timing", action="store_true", help="不打印耗时")
    select.set_defaults(handler=run_select)

    check = sub.add_parser("check", help="环境自检：模型目录与 mlx_lm 是否就绪")
    check.set_defaults(handler=run_check)
    return parser


def main(argv: list[str] | None = None) -> int:
    """解析参数并执行子命令，返回进程退出码。"""
    args = build_parser().parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
