"""域无关候选页原型：静态枚举器与覆盖率普查（只读，不调模型）。

原来的候选页只枚举"代码片段"。这个模块试第二个域：从一份真实 Python 源文件里
枚举"可选的目标函数"和"某个函数作用域里可用的名字"，把"宿主能枚举出哪些候选"
从主张变成可核对的清单。

本模块只做静态枚举与对照，不做任何写操作，也不访问网络：
- 用 ast 逐类收集绑定站点（参数、赋值、循环目标、with/except 目标、推导式目标……）；
- 用 symtable（解释器自己的作用域分析）当独立基准，给出漏项与虚报；
- 三档枚举规则 strict / python312 / naive，用它们之间的差异量化"枚举规则"本身
  的代价：贪一档能多认出多少名字，又虚报多少。
"""

from __future__ import annotations

import ast
import symtable
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# 枚举档位。越靠后越贪，能认的名字越多，虚报也越多。
#   strict    ：只认函数作用域体内直接写的绑定，不进入任何子作用域；
#   python312 ：再认 list/set/dict 推导式目标（CPython 3.12 起 PEP 709 把推导式
#               内联进外层函数，目标名真的落在外层函数局部命名空间里）；
#   naive     ：把嵌套 def / lambda / 生成器表达式 / 推导式里的名字一律挂到最近
#               的外层函数上——这是"只看名字出现"的朴素枚举器会做的事。
PROFILES = ("strict", "python312", "naive")

KIND_LABELS = {
    "param": "位置/普通参数",
    "param_kwonly": "仅关键字参数",
    "param_vararg": "可变位置参数",
    "param_kwarg": "可变关键字参数",
    "assign": "局部变量（赋值）",
    "annassign": "局部变量（带注解赋值）",
    "augassign": "局部变量（增量赋值）",
    "for_target": "循环目标",
    "with_as": "with ... as 目标",
    "except_as": "except ... as 目标",
    "import": "局部导入名",
    "walrus": "海象目标",
    "del_target": "del 目标",
    "global": "global 声明名",
    "nonlocal": "nonlocal 声明名",
    "match_capture": "match 捕获名",
    "comp_target": "推导式目标",
    "lambda_param": "lambda 参数",
    "class_attr": "类体属性",
    "nested_def": "嵌套函数名",
    "module_def": "模块级函数名",
    "module_assign": "模块级常量",
    "class_def": "类名",
}


@dataclass(frozen=True)
class ScopeRef:
    """枚举过程中看到的一个作用域。"""

    qualname: str  # "" 表示模块顶层
    kind: str  # module | class | function | comprehension | lambda
    lineno: int


@dataclass(frozen=True)
class FunctionEntry:
    """一个可选的目标函数/方法。"""

    qualname: str
    name: str
    owner_kind: str  # module | class | function（直接父作用域的种类）
    owner_qualname: str
    enclosing_functions: tuple[str, ...]  # 外层函数作用域链，空表示不在任何函数里
    lineno: int
    end_lineno: int
    decorators: tuple[str, ...]
    signature: str
    params: tuple[str, ...]
    docstring: str
    source: str  # ast.get_source_segment：从 def 行开始，不含装饰器行
    source_with_decorators: str

    @property
    def segmented_lines(self) -> int:
        return self.source.count("\n") + 1


@dataclass(frozen=True)
class NameEntry:
    """一个可选的"名字"候选，以及它真正绑在哪个作用域。"""

    name: str
    kind: str  # 绑定类别，见 KIND_LABELS
    scope: str  # 枚举规则把这个名字挂到哪个函数作用域（候选页的分组依据）
    scope_kind: str  # module | class | function
    true_scope: str  # 名字真正绑定到的作用域
    observable: bool  # 枚举规则的声明：这个名字应出现在 scope 的运行时局部里
    inlined_scope: bool  # 子作用域被内联进外层（PEP 709），名字确实在外层
    conditional: bool  # 只在部分执行路径上绑定
    lineno: int
    line: str
    statement: str

    @property
    def label(self) -> str:
        return KIND_LABELS.get(self.kind, self.kind)


@dataclass(frozen=True)
class Enumeration:
    """一次枚举的完整结果。"""

    path: Path
    profile: str
    functions: tuple[FunctionEntry, ...]
    names: tuple[NameEntry, ...]
    scopes: tuple[ScopeRef, ...]

    def functions_in(self, owner_kind: str) -> tuple[FunctionEntry, ...]:
        return tuple(f for f in self.functions if f.owner_kind == owner_kind)

    def names_in(self, scope: str) -> tuple[NameEntry, ...]:
        return tuple(n for n in self.names if n.scope == scope)

    def function_scopes(self) -> tuple[str, ...]:
        seen: list[str] = []
        for ref in self.scopes:
            if ref.kind == "function" and ref.qualname not in seen:
                seen.append(ref.qualname)
        return tuple(seen)


# --------------------------------------------------------------------------
# ast 枚举
# --------------------------------------------------------------------------


class _Walker:
    """按档位收集函数与绑定站点。只读源码。"""

    def __init__(self, source: str, profile: str) -> None:
        if profile not in PROFILES:
            raise ValueError(f"未知枚举档位：{profile!r}")
        self.source = source
        self.lines = source.splitlines()
        self.profile = profile
        self.functions: list[FunctionEntry] = []
        self.names: list[NameEntry] = []
        self.scopes: list[ScopeRef] = []

    # -- 工具 -------------------------------------------------------------
    def _text(self, node: ast.AST) -> str:
        return ast.get_source_segment(self.source, node) or ""

    def _line_text(self, lineno: int) -> str:
        if 1 <= lineno <= len(self.lines):
            return self.lines[lineno - 1].strip()
        return ""

    def _statement_text(self, node: ast.AST) -> str:
        segment = self._text(node)
        return segment if segment else self._line_text(node.lineno)

    def _emit(
        self,
        name: str,
        *,
        kind: str,
        node: ast.AST,
        scope: str,
        scope_kind: str,
        true_scope: str,
        observable: bool,
        inlined_scope: bool = False,
        conditional: bool = False,
        statement_node: ast.AST | None = None,
    ) -> None:
        self.names.append(
            NameEntry(
                name=name,
                kind=kind,
                scope=scope,
                scope_kind=scope_kind,
                true_scope=true_scope,
                observable=observable,
                inlined_scope=inlined_scope,
                conditional=conditional,
                lineno=node.lineno,
                line=self._line_text(node.lineno),
                statement=self._statement_text(statement_node or node),
            )
        )

    def _bind_target(
        self,
        target: ast.AST,
        *,
        kind: str,
        scope: str,
        scope_kind: str,
        true_scope: str,
        observable: bool,
        inlined_scope: bool,
        conditional: bool,
        statement_node: ast.AST,
    ) -> None:
        """赋值/循环/with/推导式目标的解包：Name、Tuple/List、Starred。"""

        if isinstance(target, ast.Name):
            self._emit(
                target.id,
                kind=kind,
                node=target,
                scope=scope,
                scope_kind=scope_kind,
                true_scope=true_scope,
                observable=observable,
                inlined_scope=inlined_scope,
                conditional=conditional,
                statement_node=statement_node,
            )
            return
        if isinstance(target, (ast.Tuple, ast.List)):
            for element in target.elts:
                self._bind_target(
                    element,
                    kind=kind,
                    scope=scope,
                    scope_kind=scope_kind,
                    true_scope=true_scope,
                    observable=observable,
                    inlined_scope=inlined_scope,
                    conditional=conditional,
                    statement_node=statement_node,
                )
            return
        if isinstance(target, ast.Starred):
            self._bind_target(
                target.value,
                kind=kind,
                scope=scope,
                scope_kind=scope_kind,
                true_scope=true_scope,
                observable=observable,
                inlined_scope=inlined_scope,
                conditional=conditional,
                statement_node=statement_node,
            )
            return
        # 属性/下标目标（self.x = ...、d[k] = ...）绑定的不是名字：枚举器不产出候选。

    # -- 作用域遍历 -------------------------------------------------------
    def run(self, tree: ast.Module) -> None:
        self.scopes.append(ScopeRef("", "module", 1))
        self._body(
            tree.body,
            scope="",
            scope_kind="module",
            enclosing=tuple(),
            path_depth=0,
        )

    def _body(
        self,
        body: list[ast.stmt],
        *,
        scope: str,
        scope_kind: str,
        enclosing: tuple[str, ...],
        path_depth: int,
    ) -> None:
        for stmt in body:
            self._stmt(
                stmt,
                scope=scope,
                scope_kind=scope_kind,
                enclosing=enclosing,
                path_depth=path_depth,
            )

    def _stmt(
        self,
        stmt: ast.stmt,
        *,
        scope: str,
        scope_kind: str,
        enclosing: tuple[str, ...],
        path_depth: int,
    ) -> None:
        nested_body = path_depth + 1

        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            self._function(stmt, scope=scope, scope_kind=scope_kind, enclosing=enclosing)
            return

        if isinstance(stmt, ast.ClassDef):
            qualname = f"{scope}.{stmt.name}" if scope else stmt.name
            self._emit(
                stmt.name,
                kind="class_def",
                node=stmt,
                scope=scope,
                scope_kind=scope_kind,
                true_scope=qualname,
                observable=False,
                statement_node=stmt,
            )
            self.scopes.append(ScopeRef(qualname, "class", stmt.lineno))
            self._body(
                stmt.body,
                scope=qualname,
                scope_kind="class",
                enclosing=enclosing,
                path_depth=path_depth,
            )
            return

        if isinstance(stmt, ast.Assign):
            for target in stmt.targets:
                self._bind_target(
                    target,
                    kind="assign",
                    scope=scope,
                    scope_kind=scope_kind,
                    true_scope=scope,
                    observable=scope_kind == "function",
                    inlined_scope=False,
                    conditional=path_depth > 0 or scope_kind == "class",
                    statement_node=stmt,
                )
            return

        if isinstance(stmt, ast.AnnAssign):
            self._bind_target(
                stmt.target,
                kind="annassign",
                scope=scope,
                scope_kind=scope_kind,
                true_scope=scope,
                observable=scope_kind == "function",
                inlined_scope=False,
                conditional=path_depth > 0 or scope_kind == "class",
                statement_node=stmt,
            )
            return

        if isinstance(stmt, ast.AugAssign):
            self._bind_target(
                stmt.target,
                kind="augassign",
                scope=scope,
                scope_kind=scope_kind,
                true_scope=scope,
                observable=scope_kind == "function",
                inlined_scope=False,
                conditional=path_depth > 0,
                statement_node=stmt,
            )
            return

        if isinstance(stmt, (ast.For, ast.AsyncFor)):
            self._bind_target(
                stmt.target,
                kind="for_target",
                scope=scope,
                scope_kind=scope_kind,
                true_scope=scope,
                observable=scope_kind == "function",
                inlined_scope=False,
                conditional=True,
                statement_node=stmt,
            )
            self._body(
                stmt.body,
                scope=scope,
                scope_kind=scope_kind,
                enclosing=enclosing,
                path_depth=nested_body,
            )
            self._body(
                stmt.orelse,
                scope=scope,
                scope_kind=scope_kind,
                enclosing=enclosing,
                path_depth=nested_body,
            )
            return

        if isinstance(stmt, (ast.With, ast.AsyncWith)):
            for item in stmt.items:
                if item.optional_vars is not None:
                    self._bind_target(
                        item.optional_vars,
                        kind="with_as",
                        scope=scope,
                        scope_kind=scope_kind,
                        true_scope=scope,
                        observable=scope_kind == "function",
                        inlined_scope=False,
                        conditional=False,
                        statement_node=stmt,
                    )
            self._body(
                stmt.body,
                scope=scope,
                scope_kind=scope_kind,
                enclosing=enclosing,
                path_depth=nested_body,
            )
            return

        if isinstance(stmt, ast.Try):
            self._body(
                stmt.body,
                scope=scope,
                scope_kind=scope_kind,
                enclosing=enclosing,
                path_depth=nested_body,
            )
            for handler in stmt.handlers:
                if handler.name:
                    self._emit(
                        handler.name,
                        kind="except_as",
                        node=handler,
                        scope=scope,
                        scope_kind=scope_kind,
                        true_scope=scope,
                        observable=scope_kind == "function",
                        conditional=True,
                        statement_node=handler,
                    )
                self._body(
                    handler.body,
                    scope=scope,
                    scope_kind=scope_kind,
                    enclosing=enclosing,
                    path_depth=nested_body,
                )
            self._body(
                stmt.orelse,
                scope=scope,
                scope_kind=scope_kind,
                enclosing=enclosing,
                path_depth=nested_body,
            )
            self._body(
                stmt.finalbody,
                scope=scope,
                scope_kind=scope_kind,
                enclosing=enclosing,
                path_depth=nested_body,
            )
            return

        if isinstance(stmt, (ast.Import, ast.ImportFrom)):
            for alias in stmt.names:
                if alias.name == "*":
                    continue
                bound = alias.asname or alias.name.split(".")[0]
                self._emit(
                    bound,
                    kind="import",
                    node=stmt,
                    scope=scope,
                    scope_kind=scope_kind,
                    true_scope=scope,
                    observable=scope_kind == "function",
                    conditional=path_depth > 0,
                    statement_node=stmt,
                )
            return

        if isinstance(stmt, (ast.Global, ast.Nonlocal)):
            for name in stmt.names:
                kind = "global" if isinstance(stmt, ast.Global) else "nonlocal"
                self._emit(
                    name,
                    kind=kind,
                    node=stmt,
                    scope=scope,
                    scope_kind=scope_kind,
                    true_scope=scope,
                    observable=False,  # 名字绑在别的作用域，不在本作用域的局部里
                    conditional=False,
                    statement_node=stmt,
                )
            return

        if isinstance(stmt, ast.Delete):
            for target in stmt.targets:
                self._bind_target(
                    target,
                    kind="del_target",
                    scope=scope,
                    scope_kind=scope_kind,
                    true_scope=scope,
                    observable=scope_kind == "function",
                    inlined_scope=False,
                    conditional=True,
                    statement_node=stmt,
                )
            return

        if isinstance(stmt, ast.If):
            self._body(
                stmt.body,
                scope=scope,
                scope_kind=scope_kind,
                enclosing=enclosing,
                path_depth=nested_body,
            )
            self._body(
                stmt.orelse,
                scope=scope,
                scope_kind=scope_kind,
                enclosing=enclosing,
                path_depth=nested_body,
            )
            return

        if isinstance(stmt, ast.Match):
            for case in stmt.cases:
                for name in _match_capture_names(case.pattern):
                    self._emit(
                        name,
                        kind="match_capture",
                        node=case.pattern,
                        scope=scope,
                        scope_kind=scope_kind,
                        true_scope=scope,
                        observable=scope_kind == "function",
                        conditional=True,
                        statement_node=case.pattern,
                    )
                self._body(
                    case.body,
                    scope=scope,
                    scope_kind=scope_kind,
                    enclosing=enclosing,
                    path_depth=nested_body,
                )
            return

        if isinstance(stmt, (ast.While,)):
            self._body(
                stmt.body,
                scope=scope,
                scope_kind=scope_kind,
                enclosing=enclosing,
                path_depth=nested_body,
            )
            self._body(
                stmt.orelse,
                scope=scope,
                scope_kind=scope_kind,
                enclosing=enclosing,
                path_depth=nested_body,
            )
            return

        # 其余语句：表达式里可能藏海象、lambda、推导式。
        self._expr_tree(
            stmt,
            scope=scope,
            scope_kind=scope_kind,
            enclosing=enclosing,
            path_depth=path_depth,
        )

    def _expr_tree(
        self,
        node: ast.AST,
        *,
        scope: str,
        scope_kind: str,
        enclosing: tuple[str, ...],
        path_depth: int,
    ) -> None:
        """只找表达式里的绑定站点：海象、lambda 参数、推导式目标。"""

        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.Lambda):
                self._lambda(
                    child,
                    scope=scope,
                    scope_kind=scope_kind,
                    enclosing=enclosing,
                    path_depth=path_depth,
                )
                continue
            if isinstance(child, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
                self._comprehension(
                    child,
                    scope=scope,
                    scope_kind=scope_kind,
                    enclosing=enclosing,
                    path_depth=path_depth,
                )
                continue
            if isinstance(child, ast.NamedExpr) and isinstance(child.target, ast.Name):
                # 海象绑到最近的函数作用域（推导式里的海象绑到外层函数，这是语言规定的）
                self._emit(
                    child.target.id,
                    kind="walrus",
                    node=child.target,
                    scope=scope if scope_kind == "function" else "",
                    scope_kind=scope_kind,
                    true_scope=scope,
                    observable=scope_kind == "function",
                    conditional=True,
                    statement_node=child,
                )
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self._function(child, scope=scope, scope_kind=scope_kind, enclosing=enclosing)
                continue
            if isinstance(child, ast.ClassDef):
                self._stmt(
                    child,
                    scope=scope,
                    scope_kind=scope_kind,
                    enclosing=enclosing,
                    path_depth=path_depth,
                )
                continue
            self._expr_tree(
                child,
                scope=scope,
                scope_kind=scope_kind,
                enclosing=enclosing,
                path_depth=path_depth,
            )

    def _function(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        *,
        scope: str,
        scope_kind: str,
        enclosing: tuple[str, ...],
    ) -> None:
        qualname = f"{scope}.{node.name}" if scope else node.name
        docstring = ast.get_docstring(node) or ""
        entry = FunctionEntry(
            qualname=qualname,
            name=node.name,
            owner_kind=scope_kind,
            owner_qualname=scope,
            enclosing_functions=enclosing,
            lineno=node.lineno,
            end_lineno=node.end_lineno or node.lineno,
            decorators=tuple(self._text(d) for d in node.decorator_list),
            signature=_signature(node),
            params=tuple(_param_names(node.args)),
            docstring=docstring.strip().splitlines()[0] if docstring.strip() else "",
            source=self._text(node),
            source_with_decorators=_block_text(
                self.source,
                min([node.lineno, *(d.lineno for d in node.decorator_list)]),
                node.end_lineno or node.lineno,
            ),
        )
        self.functions.append(entry)
        self.scopes.append(ScopeRef(qualname, "function", node.lineno))

        # 函数名本身也是外层作用域里绑定的一个名字。
        if scope:
            self._emit(
                node.name,
                kind="nested_def",
                node=node,
                scope=scope,
                scope_kind=scope_kind,
                true_scope=qualname,
                observable=False,
                statement_node=node,
            )

        # 参数：属于这个函数自己的作用域。
        self._params(node.args, qualname=qualname, enclosing=enclosing)
        self._body(
            node.body,
            scope=qualname,
            scope_kind="function",
            enclosing=(*enclosing, qualname),
            path_depth=0,
        )

        # 朴素档位：把这个内层函数的名字也挂到外层函数上。
        if self.profile == "naive" and enclosing:
            outer = enclosing[-1]
            for param in _param_names(node.args):
                self._emit(
                    param,
                    kind="param",
                    node=node.args,
                    scope=outer,
                    scope_kind="function",
                    true_scope=qualname,
                    observable=False,
                    statement_node=node,
                )

    def _params(self, args: ast.arguments, *, qualname: str, enclosing: tuple[str, ...]) -> None:
        groups = (
            ("param", args.posonlyargs),
            ("param", args.args),
            ("param_kwonly", args.kwonlyargs),
        )
        for kind, group in groups:
            for arg in group:
                self._emit(
                    arg.arg,
                    kind=kind,
                    node=arg,
                    scope=qualname,
                    scope_kind="function",
                    true_scope=qualname,
                    observable=True,
                    statement_node=arg,
                )
        if args.vararg is not None:
            self._emit(
                args.vararg.arg,
                kind="param_vararg",
                node=args.vararg,
                scope=qualname,
                scope_kind="function",
                true_scope=qualname,
                observable=True,
                statement_node=args.vararg,
            )
        if args.kwarg is not None:
            self._emit(
                args.kwarg.arg,
                kind="param_kwarg",
                node=args.kwarg,
                scope=qualname,
                scope_kind="function",
                true_scope=qualname,
                observable=True,
                statement_node=args.kwarg,
            )

    def _lambda(
        self,
        node: ast.Lambda,
        *,
        scope: str,
        scope_kind: str,
        enclosing: tuple[str, ...],
        path_depth: int,
    ) -> None:
        qualname = f"{scope}.<lambda@{node.lineno}>" if scope else f"<lambda@{node.lineno}>"
        self.scopes.append(ScopeRef(qualname, "lambda", node.lineno))
        self._params(node.args, qualname=qualname, enclosing=enclosing)
        if self.profile == "naive" and enclosing:
            outer = enclosing[-1]
            for param in _param_names(node.args):
                self._emit(
                    param,
                    kind="lambda_param",
                    node=node.args,
                    scope=outer,
                    scope_kind="function",
                    true_scope=qualname,
                    observable=False,
                    statement_node=node,
                )
        self._expr_tree(
            node.body,
            scope=qualname,
            scope_kind="function",
            enclosing=(*enclosing, qualname),
            path_depth=path_depth,
        )

    def _comprehension(
        self,
        node: ast.ListComp | ast.SetComp | ast.DictComp | ast.GeneratorExp,
        *,
        scope: str,
        scope_kind: str,
        enclosing: tuple[str, ...],
        path_depth: int,
    ) -> None:
        inlined = isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp))
        short = {
            ast.ListComp: "listcomp",
            ast.SetComp: "setcomp",
            ast.DictComp: "dictcomp",
            ast.GeneratorExp: "genexpr",
        }[type(node)]
        qualname = f"{scope}.<{short}@{node.lineno}>" if scope else f"<{short}@{node.lineno}>"
        self.scopes.append(ScopeRef(qualname, "comprehension", node.lineno))

        # 目标名的归属：3.12 起 list/set/dict 推导式被内联，目标名落在外层函数局部；
        # 生成器表达式仍然是独立作用域（它是个真正的生成器帧）。
        outer = enclosing[-1] if enclosing else scope
        if inlined and self.profile in ("python312", "naive"):
            # CPython 3.12+：推导式被内联，目标名真的落在外层函数局部命名空间。
            target_scope, target_kind, observable, attached = outer, "function", True, False
        else:
            # 生成器表达式（以及 3.11 及更早的推导式）：独立作用域，
            # 目标名不在外层函数局部里；朴素档位仍把它挂到外层，即虚报。
            hops = self.profile == "naive" and bool(outer)
            target_scope = outer if hops else qualname
            target_kind, observable, attached = "function", False, True

        for generator in node.generators:
            self._bind_target(
                generator.target,
                kind="comp_target",
                scope=target_scope,
                scope_kind=target_kind,
                true_scope=qualname if not inlined else qualname,
                observable=observable,
                inlined_scope=inlined and not attached,
                conditional=True,
                statement_node=node,
            )
            self._expr_tree(
                generator.iter,
                scope=qualname,
                scope_kind="function",
                enclosing=(*enclosing, qualname),
                path_depth=path_depth,
            )
            for condition in generator.ifs:
                self._expr_tree(
                    condition,
                    scope=qualname,
                    scope_kind="function",
                    enclosing=(*enclosing, qualname),
                    path_depth=path_depth,
                )
        if isinstance(node, ast.DictComp):
            self._expr_tree(
                node.value,
                scope=qualname,
                scope_kind="function",
                enclosing=(*enclosing, qualname),
                path_depth=path_depth,
            )
        else:
            self._expr_tree(
                node.elt,
                scope=qualname,
                scope_kind="function",
                enclosing=(*enclosing, qualname),
                path_depth=path_depth,
            )


def _param_names(args: ast.arguments) -> list[str]:
    names = [a.arg for a in (*args.posonlyargs, *args.args, *args.kwonlyargs)]
    if args.vararg is not None:
        names.append(args.vararg.arg)
    if args.kwarg is not None:
        names.append(args.kwarg.arg)
    return names


def _signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    return f"{node.name}({ast.unparse(node.args)})"


def _block_text(source: str, start: int, end: int) -> str:
    lines = source.splitlines()
    return "\n".join(lines[start - 1 : end])


def _match_capture_names(pattern: ast.pattern) -> list[str]:
    names: list[str] = []
    for node in ast.walk(pattern):
        if isinstance(node, ast.MatchAs) and node.name:
            names.append(node.name)
        elif isinstance(node, ast.MatchStar) and node.name:
            names.append(node.name)
        elif isinstance(node, ast.MatchMapping) and node.rest:
            names.append(node.rest)
    return names


def enumerate_source(path: Path, profile: str = "python312") -> Enumeration:
    """对一份真实源文件做一次枚举；只读文件。"""

    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text, filename=str(path))
    walker = _Walker(text, profile)
    walker.run(tree)
    return Enumeration(
        path=path,
        profile=profile,
        functions=tuple(walker.functions),
        names=tuple(walker.names),
        scopes=tuple(walker.scopes),
    )


# --------------------------------------------------------------------------
# symtable 基准与覆盖率
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ScopeSymbols:
    """symtable（解释器自己的作用域分析）给出的一个作用域。"""

    qualname: str
    scope_type: str  # module | class | function
    parent: str
    symbols: dict[str, tuple[str, ...]] = field(default_factory=dict)

    def bound(self) -> tuple[str, ...]:
        """在自己的作用域里真正绑定名字的符号（参数或赋值）。"""

        return tuple(
            sorted(
                name
                for name, flags in self.symbols.items()
                if "is_parameter" in flags or "is_assigned" in flags
            )
        )

    def referenced_only(self) -> tuple[str, ...]:
        """引用了但没在本作用域绑定的名字（全局、内建、自由变量）。"""

        return tuple(
            sorted(
                name
                for name, flags in self.symbols.items()
                if "is_referenced" in flags
                and "is_parameter" not in flags
                and "is_assigned" not in flags
            )
        )

    def free(self) -> tuple[str, ...]:
        return tuple(sorted(n for n, f in self.symbols.items() if "is_free" in f))

    def global_ref(self) -> tuple[str, ...]:
        return tuple(sorted(n for n, f in self.symbols.items() if "is_global" in f))


_FLAG_NAMES = (
    "is_parameter",
    "is_local",
    "is_global",
    "is_free",
    "is_imported",
    "is_referenced",
    "is_assigned",
    "is_namespace",
    "is_annotated",
)


def symtable_scopes(path: Path) -> tuple[ScopeSymbols, ...]:
    """用 symtable 解析同一份文件，作为"作用域真相"的独立基准。"""

    text = path.read_text(encoding="utf-8")
    table = symtable.symtable(text, str(path), "exec")
    out: list[ScopeSymbols] = []

    def visit(current: symtable.SymbolTable, parent: str) -> None:
        name = current.get_name()
        if parent == "" and current.get_type() == "module":
            qualname = ""
        elif name in {"genexpr", "listcomp", "setcomp", "dictcomp", "lambda"}:
            qualname = f"{parent}.<{name}>" if parent else f"<{name}>"
        else:
            qualname = f"{parent}.{name}" if parent else name
        symbols = {
            symbol.get_name(): tuple(f for f in _FLAG_NAMES if getattr(symbol, f)())
            for symbol in current.get_symbols()
        }
        out.append(
            ScopeSymbols(
                qualname=qualname,
                scope_type=current.get_type(),
                parent=parent,
                symbols=symbols,
            )
        )
        for child in current.get_children():
            visit(child, qualname)

    visit(table, "")
    return tuple(out)


@dataclass(frozen=True)
class ScopeCoverage:
    """一个函数作用域上的枚举覆盖率：枚举器 vs symtable vs 运行时观测。"""

    qualname: str
    lineno: int
    truth_bound: tuple[str, ...]  # symtable：本作用域真正绑定的名字
    truth_referenced_only: tuple[str, ...]  # symtable：引用了但没绑定（全局/内建/自由）
    enumerated_strict: tuple[str, ...]
    enumerated_profile: tuple[str, ...]
    enumerated_naive: tuple[str, ...]
    enumerated_observable: tuple[str, ...]  # 枚举器声称运行时可观测的名字
    claimed_unobservable: tuple[str, ...]  # 枚举器自己标了"挂到外层但不在运行时局部"
    missed_by_strict: tuple[str, ...]  # symtable 绑定、strict 档没给
    missed_by_profile: tuple[str, ...]  # symtable 绑定、本次档位也没给
    spurious_naive: tuple[str, ...]  # naive 档给了、symtable 说本作用域没绑定（虚报）
    observed_union: tuple[str, ...] = ()
    observed_only: tuple[str, ...] = ()  # 运行时见过、枚举器没给
    claimed_but_unobserved: tuple[str, ...] = ()  # 枚举器声称可观测、见证调用里没见过


def measure_coverage(
    path: Path,
    *,
    scope_qualname: str,
    strict: Enumeration,
    profile: Enumeration,
    naive: Enumeration | None = None,
    observed_union: tuple[str, ...] = (),
) -> ScopeCoverage:
    """把一次枚举与 symtable 对照，可选叠加运行时观测。"""

    truth = {s.qualname: s for s in symtable_scopes(path)}
    scope = truth.get(scope_qualname)
    if scope is None:
        raise KeyError(f"symtable 里没有作用域：{scope_qualname!r}")

    bound = scope.bound()
    strictly = tuple(sorted({n.name for n in strict.names_in(scope_qualname)}))
    profiled = tuple(sorted({n.name for n in profile.names_in(scope_qualname)}))
    naively = (
        tuple(sorted({n.name for n in naive.names_in(scope_qualname)})) if naive is not None else ()
    )
    observable = tuple(
        sorted({n.name for n in profile.names_in(scope_qualname) if n.observable})
    )
    unobservable = tuple(
        sorted({n.name for n in profile.names_in(scope_qualname) if not n.observable})
    )
    observed = tuple(sorted(set(observed_union)))
    return ScopeCoverage(
        qualname=scope_qualname,
        lineno=next((f.lineno for f in profile.functions if f.qualname == scope_qualname), 0),
        truth_bound=bound,
        truth_referenced_only=scope.referenced_only(),
        enumerated_strict=strictly,
        enumerated_profile=profiled,
        enumerated_naive=naively,
        enumerated_observable=observable,
        claimed_unobservable=unobservable,
        missed_by_strict=tuple(sorted(set(bound) - set(strictly))),
        missed_by_profile=tuple(sorted(set(bound) - set(profiled))),
        spurious_naive=tuple(sorted(set(naively) - set(bound))),
        observed_union=observed,
        observed_only=tuple(sorted(set(observed) - set(profiled))),
        claimed_but_unobserved=tuple(sorted(set(observable) - set(observed))) if observed else (),
    )


# --------------------------------------------------------------------------
# 清单：能枚举 / 枚举不全 / 枚举不了
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class CapabilityRow:
    """一条能力声明：某个类别的决策点枚举到什么程度，带真实例子。"""

    category: str
    verdict: str  # 能枚举 | 枚举不全 | 枚举不了
    enumerated: int
    truth: int
    examples: tuple[str, ...]
    note: str


def _example(entry: FunctionEntry | NameEntry, path: Path) -> str:
    if isinstance(entry, FunctionEntry):
        rel = path.relative_to(REPO_ROOT) if path.is_absolute() else path
        return f"{rel}:{entry.lineno}:{entry.name}"
    rel = path.relative_to(REPO_ROOT) if path.is_absolute() else path
    return f"{rel}:{entry.lineno}:{entry.name}（{entry.label}，真作用域 {entry.true_scope or '<module>'}）"


def capability_rows(
    enumeration: Enumeration,
    strict: Enumeration,
    *,
    limit: int = 4,
) -> tuple[CapabilityRow, ...]:
    """按类别给出枚举能力清单；数字全部来自这一次真实枚举。"""

    path = enumeration.path
    rows: list[CapabilityRow] = []

    top = enumeration.functions_in("module")
    methods = enumeration.functions_in("class")
    nested = enumeration.functions_in("function")
    rows.append(
        CapabilityRow(
            category="模块级函数 / 类方法（作为可选目标）",
            verdict="能枚举",
            enumerated=len(top) + len(methods),
            truth=len(top) + len(methods),
            examples=tuple(_example(f, path) for f in (*top, *methods)[:limit]),
            note="qualname、签名、docstring、源码段都来自 ast；候选 id 由宿主另发，与位置无关。",
        )
    )
    rows.append(
        CapabilityRow(
            category="嵌套函数（函数体内 def，作为可选目标）",
            verdict="枚举不全" if nested else "能枚举",
            enumerated=len(nested),
            truth=len(nested),
            examples=tuple(_example(f, path) for f in nested[:limit]),
            note=(
                "行号与名字能枚举，但和模块级函数放进同一张平坦候选页会撞名，"
                "且物化后拿不到闭包单元。本次源文件里"
                + (f"有 {len(nested)} 个。" if nested else "一个都没有，所以这一格只能靠别的真实文件举证。")
            ),
        )
    )

    fn_scopes = [s for s in enumeration.scopes if s.kind == "function"]
    nested_scopes = [
        s for s in fn_scopes if s.qualname.startswith("<") or ".<" in s.qualname
    ]
    observed_names = [n for n in enumeration.names if n.scope_kind == "function"]
    rows.append(
        CapabilityRow(
            category="函数作用域内的参数（可作为「名字」候选）",
            verdict="能枚举",
            enumerated=len([n for n in observed_names if n.kind.startswith("param")]),
            truth=len([n for n in observed_names if n.kind.startswith("param")]),
            examples=tuple(
                _example(n, path) for n in observed_names if n.kind.startswith("param")
            )[:limit],
            note="参数是 ast.arguments 里显式列出的，不存在漏项。",
        )
    )
    rows.append(
        CapabilityRow(
            category="函数体内的局部绑定（赋值/循环/with/except 目标）",
            verdict="能枚举",
            enumerated=len(
                [
                    n
                    for n in observed_names
                    if n.kind
                    in {"assign", "annassign", "augassign", "for_target", "with_as", "except_as", "import"}
                ]
            ),
            truth=len(
                [
                    n
                    for n in observed_names
                    if n.kind
                    in {"assign", "annassign", "augassign", "for_target", "with_as", "except_as", "import"}
                ]
            ),
            examples=tuple(
                _example(n, path)
                for n in observed_names
                if n.kind in {"assign", "for_target", "except_as"}
            )[:limit],
            note="绑定点能枚举；但条件分支里的绑定（except ... as、循环目标）不保证运行时存在。",
        )
    )

    comp_targets = [n for n in enumeration.names if n.kind == "comp_target"]
    inlined = [n for n in comp_targets if n.inlined_scope]
    separate = [n for n in comp_targets if not n.inlined_scope]
    rows.append(
        CapabilityRow(
            category="推导式 / 生成器表达式的目标名",
            verdict="枚举不全",
            enumerated=len(inlined),
            truth=len(comp_targets),
            examples=tuple(_example(n, path) for n in comp_targets[:limit]),
            note=(
                f"list/set/dict 推导式目标在 CPython 3.12 起被内联，确实落在外层函数局部（{len(inlined)} 个）；"
                f"生成器表达式目标仍是独立作用域，不在外层局部（{len(separate)} 个）。"
                "同一份代码在 3.11 及更早的解释器上归属完全相反——枚举规则绑死在解释器版本上。"
            ),
        )
    )

    lambda_count = len([s for s in enumeration.scopes if s.kind == "lambda"])
    lambda_params = [n for n in enumeration.names if n.kind == "lambda_param"]
    rows.append(
        CapabilityRow(
            category="lambda 形参（作为所在函数的「名字」候选）",
            verdict="枚举不了" if lambda_count == 0 else "枚举不全",
            enumerated=len(lambda_params),
            truth=lambda_count,
            examples=tuple(
                s.qualname for s in enumeration.scopes if s.kind == "lambda"
            )[:limit],
            note="lambda 形参属于 lambda 自己的作用域；挂在最近的外层函数上就是虚报。",
        )
    )

    free_like = [
        n
        for n in strict.names
        if n.kind in {"global", "nonlocal"}
    ]
    rows.append(
        CapabilityRow(
            category="global / nonlocal 声明的名字",
            verdict="枚举不了" if not free_like else "枚举不全",
            enumerated=len(free_like),
            truth=len(free_like),
            examples=tuple(_example(n, path) for n in free_like[:limit]),
            note="声明本身能看见，但名字绑在别的作用域，放进本作用域的候选页会指错对象。",
        )
    )

    rows.append(
        CapabilityRow(
            category="属性 / 下标 / 动态名字（self.x=、d[k]=、setattr、exec）",
            verdict="枚举不了",
            enumerated=0,
            truth=0,
            examples=(),
            note="绑定目标不是 Name 节点，ast 里没有「名字」可枚举；此类决策点必须换一套候选表达。",
        )
    )
    return tuple(rows)


def name_scope_page_source(enumeration: Enumeration, scope: str) -> tuple[NameEntry, ...]:
    """取出某个函数作用域上、按档位归属到这里的全部名字候选。

    同一个名字可能在多处绑定（例如 text 先 strip、后被覆盖）：这里按名字去重，
    保留第一次出现的绑定点，并把绑定点数量记在 note 里由调用方展示。
    """

    entries = [n for n in enumeration.names if n.scope == scope and n.observable]
    dedup: dict[str, NameEntry] = {}
    for entry in entries:
        dedup.setdefault(entry.name, entry)
    return tuple(dedup.values())


def binding_kind_counts(enumeration: Enumeration) -> dict[str, int]:
    counts: dict[str, int] = {}
    for entry in enumeration.names:
        counts[entry.kind] = counts.get(entry.kind, 0) + 1
    return dict(sorted(counts.items()))
