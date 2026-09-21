"""域无关候选页原型：运行时判据（只读源码，不写文件，不调模型）。

这里的判据刻意不看"代码像不像"，只看"跑起来对不对"：
- 名字候选：在一次真实调用里用代码对象过滤后的跟踪器取局部快照，把选中的名字
  代进去求值/比较，得出行为判据；
- 函数候选：把宿主物化的源码段 exec 成真函数，再对这个真函数跑一组行为指纹。

跟踪器必须按 `frame.f_code is func.__code__` 过滤：不过滤会把推导式/生成器帧的
局部混进来，得出"我观测到了那个名字"的假结论。
"""

from __future__ import annotations

import ast
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping, Sequence

from bench.domain_enum import FunctionEntry


class ObservationError(RuntimeError):
    """观测点定位失败：源文件的结构和判据对不上。"""


class MaterializeError(RuntimeError):
    """物化的源码段不可用。"""


MISSING = object()


# --------------------------------------------------------------------------
# 观测点定位
# --------------------------------------------------------------------------


def source_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def needle_line(lines: Sequence[str], entry: FunctionEntry, needle: str, *, nth: int = 0) -> int:
    """在某个函数的行区间里按文本找观测点行号。

    找不到或有歧义就报错——宁可不跑，也不要悄悄观测错的行。
    """

    hits = [
        index + 1
        for index in range(entry.lineno - 1, entry.end_lineno)
        if needle in lines[index]
    ]
    if not hits:
        raise ObservationError(
            f"{entry.qualname}（{entry.lineno}-{entry.end_lineno}）里找不到观测点 {needle!r}"
        )
    if nth >= len(hits):
        raise ObservationError(
            f"{entry.qualname} 里 {needle!r} 只有 {len(hits)} 处，取不到第 {nth + 1} 处"
        )
    return hits[nth]


# --------------------------------------------------------------------------
# 局部快照
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Snapshot:
    """某一次见证调用、在执行某一行之前的一次局部快照。"""

    line: int
    witness: str
    values: dict[str, object]


@dataclass(frozen=True)
class LocalObservation:
    """一次或多次见证调用里，某个函数自己的局部命名空间的形状。"""

    qualname: str
    witnesses: tuple[str, ...]
    snapshots: dict[int, tuple[Snapshot, ...]]  # 行号 -> 该行每次执行前的快照
    union: dict[str, object]  # 所有见证调用里出现过的名字（保留最后见到的值）
    calls: int

    def at(self, line: int, index: int = 0) -> Snapshot:
        entries = self.snapshots.get(line)
        if not entries:
            raise ObservationError(f"{self.qualname} 在执行第 {line} 行时没有被观测到")
        if index >= len(entries):
            raise ObservationError(
                f"{self.qualname} 第 {line} 行只执行了 {len(entries)} 次，取不到第 {index + 1} 次快照"
            )
        return entries[index]

def observe_locals(
    func: Callable[..., object],
    witnesses: Sequence[tuple[tuple[object, ...], dict[str, object]]],
    *,
    needles: Mapping[str, int] | None = None,
    qualname: str = "",
) -> LocalObservation:
    """跑真实见证调用，按代码对象过滤，收集局部快照。

    `needles` 是 {标签: 行号}，只对这些行做快照；行号来自 `needle_line`。
    抛异常的见证调用也会被跟踪（except 分支里的名字照样能观测到）。
    """

    target = func.__code__
    watch = set((needles or {}).values())
    snapshots: dict[int, list[Snapshot]] = {}
    union: dict[str, object] = {}
    labels: list[str] = []

    def local_tracer(frame, event, arg):  # type: ignore[no-untyped-def]
        if frame.f_code is not target:
            return None
        if event == "line":
            if frame.f_lineno in watch:
                snapshots.setdefault(frame.f_lineno, []).append(
                    Snapshot(frame.f_lineno, label, dict(frame.f_locals))
                )
            for key, value in frame.f_locals.items():
                union[key] = value
        return local_tracer

    def global_tracer(frame, event, arg):  # type: ignore[no-untyped-def]
        if frame.f_code is target and event == "call":
            return local_tracer
        return None

    for args, kwargs in witnesses:
        label = _witness_label(func.__name__, args)
        labels.append(label)
        sys.settrace(global_tracer)
        try:
            func(*args, **kwargs)
        except BaseException:  # 见证调用允许抛错：except 分支正是我们要看的路径
            pass
        finally:
            sys.settrace(None)

    return LocalObservation(
        qualname=qualname or func.__name__,
        witnesses=tuple(labels),
        snapshots={line: tuple(entries) for line, entries in snapshots.items()},
        union=union,
        calls=len(witnesses),
    )


def _witness_label(name: str, args: Sequence[object]) -> str:
    """见证调用的短标签：报告里不需要整张候选页的 repr。"""

    inner = ", ".join(describe_value(value, 18) for value in args)
    if len(inner) > 72:
        inner = inner[:69] + "..."
    return f"{name}({inner})"


def describe_value(value: object, limit: int = 72) -> str:
    """给报告用的短值描述；不执行任何用户代码。"""

    if value is MISSING:
        return "<未绑定>"
    text = repr(value).replace("\n", "\\n")
    if len(text) > limit:
        return text[: limit - 3] + "..."
    return text


def equal_groups(snapshot: Mapping[str, object], names: Sequence[str]) -> tuple[tuple[str, ...], ...]:
    """把快照里值相等的名字分组：值相等的名字在"读值"判据下不可区分。"""

    groups: list[list[str]] = []
    representatives: list[object] = []
    for name in names:
        if name not in snapshot:
            continue
        value = snapshot[name]
        for index, representative in enumerate(representatives):
            if _safe_equal(value, representative):
                groups[index].append(name)
                break
        else:
            groups.append([name])
            representatives.append(value)
    return tuple(tuple(group) for group in groups if len(group) > 1)


def _safe_equal(left: object, right: object) -> bool:
    try:
        return bool(left == right)
    except Exception:
        return False


# --------------------------------------------------------------------------
# 物化：把宿主候选的源码段变成真函数
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Materialized:
    """物化结果：真可调用对象，以及它定义在哪个命名空间里。"""

    code: str
    function: Callable[..., object]
    defined_names: tuple[str, ...]
    globals_used: tuple[str, ...]
    inner_captured: tuple[str, ...]  # 本函数绑定、又被内层作用域引用的单元变量（AST 近似）


def parse_single_function(code: str) -> ast.FunctionDef:
    """候选源码段必须恰好是一个函数定义。"""

    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        raise MaterializeError(f"候选源码段不能解析：{exc.msg}") from None
    if len(tree.body) != 1 or not isinstance(tree.body[0], (ast.FunctionDef, ast.AsyncFunctionDef)):
        kinds = ", ".join(type(node).__name__ for node in tree.body)
        raise MaterializeError(f"候选源码段不是一个函数定义（实际是 {kinds or '空'}）")
    annotate_parents(tree)
    return tree.body[0]


def materialize_code(code: str, module_globals: Mapping[str, object]) -> Materialized:
    """在模块全局的副本里 exec 候选源码段，拿到真函数。

    只复制命名空间，不动被导入模块自己的字典。
    """

    node = parse_single_function(code)
    namespace = dict(module_globals)
    try:
        exec(compile(code, f"<materialized {node.name}>", "exec"), namespace)
    except Exception as exc:
        raise MaterializeError(f"物化后执行失败：{type(exc).__name__}: {exc}") from None
    produced = namespace.get(node.name)
    if not callable(produced):
        raise MaterializeError(f"物化后 {node.name!r} 不是可调用对象")
    defined = tuple(sorted(set(namespace) - set(module_globals)))
    return Materialized(
        code=code,
        function=produced,
        defined_names=defined,
        globals_used=tuple(sorted(loaded_globals(node))),
        inner_captured=tuple(sorted(inner_captured_names(node))),
    )


_INNER_SCOPE_NODES = (
    ast.FunctionDef,
    ast.AsyncFunctionDef,
    ast.Lambda,
    ast.ClassDef,
    ast.ListComp,
    ast.SetComp,
    ast.DictComp,
    ast.GeneratorExp,
)


def _is_inner_scope(node: ast.AST) -> bool:
    return isinstance(node, _INNER_SCOPE_NODES)


def iter_own_scope_nodes(node: ast.AST):
    """遍历函数自己这一层的节点，遇到内层作用域只交出根节点、不进去。"""

    for child in ast.iter_child_nodes(node):
        yield child
        if _is_inner_scope(child):
            continue
        yield from iter_own_scope_nodes(child)


def scope_bound_names(node: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    """函数自己这一层绑定的名字：参数、赋值/循环/with/except 目标、内层 def 名。"""

    names: set[str] = set(_arg_names(node.args))
    for child in iter_own_scope_nodes(node):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(child.name)
        elif isinstance(child, ast.Name) and isinstance(child.ctx, (ast.Store, ast.Del)):
            names.add(child.id)
        elif isinstance(child, ast.ExceptHandler) and child.name:
            names.add(child.name)
    return names


def _arg_names(args: ast.arguments) -> list[str]:
    names = [a.arg for a in (*args.posonlyargs, *args.args, *args.kwonlyargs)]
    if args.vararg is not None:
        names.append(args.vararg.arg)
    if args.kwarg is not None:
        names.append(args.kwarg.arg)
    return names


def inner_scope_names(node: ast.AST) -> set[str]:
    """所有内层作用域里出现过的名字（读写都算，含形参）。"""

    out: set[str] = set()
    for child in iter_own_scope_nodes(node):
        if not _is_inner_scope(child):
            continue
        for sub in ast.walk(child):
            if isinstance(sub, ast.Name):
                out.add(sub.id)
            elif isinstance(sub, ast.arg):
                out.add(sub.arg)
    return out


def inner_captured_names(node: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    """本函数绑定、又被内层作用域引用的名字：闭包要捕获的那些单元变量。

    注意这只是 AST 近似：它看不到"本函数自己从外层函数捕获了什么"——那要看
    解释器的 freevars（symtable 的 is_free）。
    """

    return scope_bound_names(node) & inner_scope_names(node)


def loaded_globals(node: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    """函数体内加载、但本函数（含内层作用域）任何地方都没绑定的名字。"""

    bound = set(_arg_names(node.args))
    for child in ast.walk(node):
        if isinstance(child, ast.Name) and isinstance(child.ctx, (ast.Store, ast.Del)):
            bound.add(child.id)
        elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(child.name)
        elif isinstance(child, ast.arg):
            bound.add(child.arg)
        elif isinstance(child, ast.ExceptHandler) and child.name:
            bound.add(child.name)
    loaded = {
        child.id
        for child in ast.walk(node)
        if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load)
    }
    return loaded - bound


def annotate_parents(tree: ast.AST) -> None:
    """给 AST 补 parent 指针；本模块只用于少量结构判断。"""

    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            child.parent = node  # type: ignore[attr-defined]


def segment_is_file_text(entry: FunctionEntry, code: str) -> bool:
    """物化段必须和第 lineno/end_lineno 行区间里的源码逐字一致。"""

    return code == entry.source


# --------------------------------------------------------------------------
# 判据执行
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Check:
    """一条宿主判据：返回 None 表示通过，否则返回失败原因。"""

    label: str
    run: Callable[[], str | None]


def eq_check(label: str, thunk: Callable[[], object], want: object) -> Check:
    """相等判据：注意 thunk 每次都会真的跑一遍被测代码。"""

    def run() -> str | None:
        got = thunk()
        if got == want:
            return None
        return f"期望 {want!r}，实得 {got!r}"

    return Check(label, run)


def raises_check(label: str, thunk: Callable[[], object], want: type[BaseException]) -> Check:
    """异常判据：必须抛指定的异常类型。"""

    def run() -> str | None:
        try:
            got = thunk()
        except want:
            return None
        except BaseException as exc:
            return f"期望抛 {want.__name__}，实得 {type(exc).__name__}: {exc}"
        return f"期望抛 {want.__name__}，实际正常返回 {got!r}"

    return Check(label, run)


def predicate_check(
    label: str,
    thunk: Callable[[], object],
    predicate: Callable[[object], bool],
    describe: str,
) -> Check:
    """谓词判据：把被测代码的返回值代进宿主写的谓词。"""

    def run() -> str | None:
        got = thunk()
        try:
            ok = bool(predicate(got))
        except Exception as exc:
            return f"谓词自身抛错 {type(exc).__name__}: {exc}"
        if ok:
            return None
        return f"不满足「{describe}」；实得 {got!r}"

    return Check(label, run)


@dataclass
class CheckReport:
    checks: list[tuple[str, bool, str]] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return bool(self.checks) and all(ok for _, ok, _ in self.checks)

    def add(self, label: str, ok: bool, detail: str) -> None:
        self.checks.append((label, ok, detail))

    def lines(self) -> list[str]:
        return [f"{'ok  ' if ok else 'FAIL'} {label}: {detail}" for label, ok, detail in self.checks]


def run_checks(checks: Sequence[Check]) -> CheckReport:
    """逐条跑判据；判据自己抛错也算失败，但原因会写明是谁抛的。"""

    report = CheckReport()
    for check in checks:
        try:
            reason = check.run()
        except Exception as exc:
            report.add(check.label, False, f"宿主判据抛错 {type(exc).__name__}: {exc}")
            continue
        report.add(check.label, reason is None, reason or "通过")
    return report
