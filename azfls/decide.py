"""选择式产出：宿主提取候选，小模型只做选择，宿主确定性组装。

参考 Cua 的 CUA-S1-FORMS：任务足够固定时，不需要通用生成模型——
宿主先把候选观察清楚，模型只在候选之间做选择，产物由宿主确定性落成。

本模块只做一类固定任务：把“筛选列表函数 + 选择返回字段”的短指令
变成一次候选 id 的选择，再由宿主重写目标函数：

    extract → build_decision_prompt → engine.generate → parse_decision → assemble → make_artifact

模型只回一个 JSON 决策，且只能引用宿主给出的候选 id；未知 id、非 JSON、
缺字段一律拒绝（身份与审批由宿主掌握，模型不产出代码）。本模块不写盘、
不 import 确认门：返回的 Artifact 交给现有 Gate 流程确认后原样写入。

决策有三个槽位：过滤条件、返回字段、排序（按哪个字段、升序还是降序）。
排序是后来加的槽位，代价是每次决策的输出契约都长一点，所以决策里没有排序时
一个字节都不动原文里的排序——删掉正在工作的代码是这条路径最严重的失败模式。
"""

from __future__ import annotations

import ast
import json
from collections.abc import Mapping
from dataclasses import dataclass

import libcst as cst
from libcst import matchers as cstm
from libcst.metadata import CodeRange, MetadataWrapper, PositionProvider

from azfls.contracts import Action, Artifact, Kind, make_artifact, normalize_body, strip_code_fence
from azfls.model import Engine

# 决策只回一个 JSON 对象，几十个 token 足够；提示短是这条路径的全部意义。
DECISION_MAX_TOKENS = 128
# 只有一个函数参与本次决策，id 由宿主固定给出。
FUNCTION_ID = "fn0"
# 精简格式：f = 条件 id（可省，省掉即不过滤），r = 字段 id 数组；
# s / d 只有调用方显式启用排序槽位时才允许出现。
# 实测同一决策 39→29 token、墙钟 0.662→0.556 s，解析 9/9；函数 id 不再回抄
# （候选表里只有一个函数，让模型回声它只花 token 不带信息，身份由宿主校验）。
TERSE_KEYS = frozenset({"f", "r", "s", "d"})
# 旧的长键仍然接受：此前生成的决策与测试夹具不必改写。两套键不许混在一个对象里。
# 旧格式**不能**表达排序：加了 s / d 的对象会被判成混用两套键而拒绝。
LEGACY_KEYS = frozenset({"function", "filter_field", "return_fields"})
# 决策 JSON 只允许这两套键，多一个都拒绝，避免模型夹带别的东西。
DECISION_KEYS = TERSE_KEYS | LEGACY_KEYS

# 排序方向：只认这两组写法，别的一律拒绝（不猜模型想说哪个方向）。
_SORT_ASC = frozenset({"asc", "ascending", "升序", "升"})
_SORT_DESC = frozenset({"desc", "descending", "降序", "降"})

# 基础提示只展示无排序形状；排序槽位由调用方通过 sort_enabled 显式开启，
# 不能让模型从自然语言自行决定是否可以输出 s / d。
#
# **那行 `正确形状示例` 不能删。** 它是 2026-09-20 实测出来的：
# 加排序槽位时把示例删掉、换上一句“未启用时严禁输出 s 或 d”，1.5B 就开始
# 整个省掉 `f` 键（回 `{"r": ["f1", "f2"]}`），于是宿主的“不过滤”语义被静默命中，
# 过滤条件那行代码被删掉而产物仍然是合法 Python。
# `bench/sort_slot.py --repeats 3`（运行时行为判定，两个方向的提示都跑过）：
#   不用排序的 5 条任务：有示例 15/15 次（5/5 条），无示例 3/15 次（1/5 条）；
#   6 条要排序的任务：有示例 18/18 次（6/6 条），无示例 3/18 次（1/6 条，
#   只有原文本来就有排序的 `priced` 通过）。代价是每次决策多约 8 个输出 token
#   （20 → 28），换过滤条件不再被丢掉。
# 提示里**只**允许出现 f / r：历史上基础提示里出现过带 s / d 的示例后，
# 没要求排序的指令被模型全部错误地加上排序；排序怎么写只由
# `_SORT_ENABLED_SYSTEM_PROMPT` 交代，禁用侧的禁令留在用户消息的 sort_note 里
# （实测无害），真正的强制在宿主 `parse_decision(sort_enabled=False)` 那一层，
# 不靠提示词措辞。
DECISION_SYSTEM_PROMPT = (
    "你只回一个 JSON 对象，不写代码、不解释、不加围栏。\n"
    "只能使用给出的候选 id，不得发明新的 id 或字段名。\n"
    "只回这两个键：f（条件 id，不过滤时用 null）、r（字段 id 的数组，按要求的输出顺序）。\n"
    "正确形状示例：" '{"f": "c2", "r": ["f0", "f1"]}'
)

_SORT_ENABLED_SYSTEM_PROMPT = (
    "本次调用方已明确启用排序槽位。需要排序时，额外输出 s（排序字段 id，必须是 r 里的一个字段）"
    "和 d（排序方向：升序 asc，降序 desc），两个键必须同时出现；不要求排序时仍省略 s 和 d。"
)

_FUNC_TYPES = (ast.FunctionDef, ast.AsyncFunctionDef)
# 看起来像方法名的属性不算字段（它们出现在调用位置，不是被读取的数据）。
_ATTR_FALLBACK_QUOTE = '"'
# 排序键的 lambda 参数名；累加列表本身就叫 r 时换一个，免得读起来像在排 lambda 自己。
_LAMBDA_ITEM = "r"
_LAMBDA_ITEM_FALLBACK = "row"


class DecisionError(Exception):
    """决策不合法时抛出（未知候选 id、非 JSON、缺字段等）。"""


@dataclass(frozen=True)
class Candidate:
    """宿主提取到的一个候选；id 由宿主生成，模型只能引用 id。"""

    id: str  # 宿主生成，如 "f0" / "c1"
    name: str  # 真实的字段名，如 "active"
    kind: str  # "field" | "condition"


@dataclass(frozen=True)
class Candidates:
    """一个函数的全部可选候选；模型只能从这里选。"""

    function_id: str  # "fn0"
    function_name: str
    start_line: int
    end_line: int
    fields: tuple[Candidate, ...]  # 可返回的字段候选
    conditions: tuple[Candidate, ...]  # 可用来过滤的条件候选


@dataclass(frozen=True)
class Decision:
    """小模型的全部输出：只选 id，不含代码、路径、哈希。"""

    function_id: str
    filter_field: str | None  # 条件候选 id；None 表示不过滤
    return_fields: tuple[str, ...]  # 字段候选 id，按要求的输出顺序
    sort_field: str | None = None  # 字段候选 id；None 表示不动原文里的排序
    sort_desc: bool = False  # 只在 sort_field 非 None 时有意义


def extract(source: str, function_name: str | None = None) -> Candidates:
    """用 ast 从源码里确定性提取候选；找不到函数时抛 DecisionError。

    字段候选：dict 字面量键、`item["key"]`、`item.key`、`item.get("key")`；
    条件候选：所有观察到的名字都可以当过滤条件——已经在 if / while / 三元 /
    推导式里当过判断的排在最前，其余按出现顺序排后面。

    之所以不把条件候选限制成“原文已经用来判断的字段”，是因为常见指令是
    “加上一个过滤条件”，而原文里根本没有判断可抄；宿主观察到了这个键，
    就有资格把它作为候选提供出来。字段是 f0,f1,…，条件是 c0,c1,…；
    同名可以同时出现在两张表里（id 不同），宿主不猜类型。
    """
    tree = _parse(source)
    node = _find_function(tree, function_name)
    collector = _collect(source, node)
    fields = tuple(
        Candidate(f"f{index}", name, "field") for index, name in enumerate(collector.fields)
    )
    # 先放已经当过判断的，再补上其余观察到的名字；两边都按源码出现顺序。
    ordered = list(collector.conditions) + [
        name for name in collector.fields if name not in collector.conditions
    ]
    conditions = tuple(
        Candidate(f"c{index}", name, "condition") for index, name in enumerate(ordered)
    )
    return Candidates(
        function_id=FUNCTION_ID,
        function_name=node.name,
        start_line=node.lineno,
        end_line=node.end_lineno or node.lineno,
        fields=fields,
        conditions=conditions,
    )


def build_decision_prompt(
    instruction: str, candidates: Candidates, *, sort_enabled: bool = False
) -> list[dict[str, str]]:
    """构造极短消息；排序槽位是否可用由调用方显式传入。

    ``sort_enabled`` 是主模型计划/请求类型传下来的协议位，不从 ``instruction``
    猜测。关闭时系统提示只给 f / r 的真实形状示例、一个 ``s`` / ``d`` 的写法都不出现，
    禁令写在用户消息的 sort_note 里、真正的强制在 ``parse_decision``：避免模型抄到
    排序格式后把旧任务误判成排序任务。
    """
    sort_note = (
        "排序槽位：已启用。若任务明确要求排序才输出 s 和 d，两个键必须同时出现；"
        "否则省略它们。"
        if sort_enabled
        else "排序槽位：未启用。严禁输出 s 或 d；不得因示例或任务措辞改变原文顺序。"
    )
    system = DECISION_SYSTEM_PROMPT
    if sort_enabled:
        system = f"{system}\n{_SORT_ENABLED_SYSTEM_PROMPT}"
    lines = [
        f"任务：{instruction.strip()}",
        "",
        f"函数：{candidates.function_id}={candidates.function_name}",
        f"字段（r 只能选这里）：{_pairs(candidates.fields)}",
        f"条件（f 只能选这里，不过滤时用 null）：{_pairs(candidates.conditions)}",
        sort_note,
        "只回 JSON，不要解释。",
    ]
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": "\n".join(lines)},
    ]


def parse_decision(
    text: str, candidates: Candidates, *, sort_enabled: bool = False
) -> Decision:
    """解析并严格校验；只接受候选表里存在的 id。任何不合格都抛 DecisionError。

    允许回复外面带代码围栏或少量说明文字（取第一个 JSON 对象），
    但不修补、不猜测残缺的 JSON，也不接受模型自造的 id。

    认两套键，但一个对象里只许用一套：
    - 精简格式（现在的契约）：`{"f": "c2", "r": ["f0", "f1"]}`；
      f 可以是 null，也可以整个键省掉（就是不过滤），r 必须是非空字段 id 数组；
      只有调用方显式启用排序槽位时，才允许额外给出成对的 s / d。
    - 旧的长键格式：`function` / `filter_field` / `return_fields`，三者缺一不可，
      继续接受是为了此前生成的决策与测试夹具不必改写；**旧格式表达不了排序**，
      带上 s / d 会被判成混用两套键。
    两套键混在同一个对象里，格式说不清，直接拒绝。

    ``sort_enabled=False`` 是宿主闸门：即使模型回了合法的 s/d，未启用时
    也拒绝，而不是继续执行隐式排序。
    """
    obj = _json_object(text)
    if not isinstance(obj, dict):
        raise DecisionError("决策必须是一个 JSON 对象")
    present = set(obj)
    terse = present & TERSE_KEYS
    legacy = present & LEGACY_KEYS
    if terse and legacy:
        raise DecisionError(
            f"决策混用了两套键：{', '.join(sorted(terse))} 与 {', '.join(sorted(legacy))}"
            "（一个对象里只能用一套：精简键 f/r，或旧的长键 function/filter_field/return_fields）"
        )
    if legacy:
        return _parse_legacy(obj, candidates)
    return _parse_terse(obj, candidates, sort_enabled=sort_enabled)


def _parse_legacy(obj: dict[str, object], candidates: Candidates) -> Decision:
    """旧的长键格式：三个键都要，函数 id 必须与候选表一致。"""
    extra = sorted(set(obj) - LEGACY_KEYS)
    if extra:
        raise DecisionError(
            f"决策有多余的键：{', '.join(extra)}（只接受 {', '.join(sorted(LEGACY_KEYS))}）"
        )
    missing = sorted(LEGACY_KEYS - set(obj))
    if missing:
        raise DecisionError(f"决策缺少字段：{', '.join(missing)}")

    function_id = obj["function"]
    if not isinstance(function_id, str):
        raise DecisionError("function 必须是函数候选 id 字符串")
    filter_field = _parse_filter(obj["filter_field"], "filter_field")
    return_fields = _parse_return_fields(obj["return_fields"], "return_fields")
    return _validate_ids(
        candidates,
        function_id.strip(),
        filter_field,
        return_fields,
    )


def _parse_terse(
    obj: dict[str, object], candidates: Candidates, *, sort_enabled: bool
) -> Decision:
    """精简格式：只认 f / r；排序 s / d 只有显式启用时才可用。"""
    sort_keys = set(obj) & frozenset({"s", "d"})
    if sort_keys and not sort_enabled:
        raise DecisionError(
            "排序槽位未启用：决策不得包含 s 或 d；请由调用方先明确启用排序槽位"
        )
    extra = sorted(set(obj) - TERSE_KEYS)
    if extra:
        raise DecisionError(
            f"决策有多余的键：{', '.join(extra)}（只接受 {', '.join(sorted(TERSE_KEYS))}）"
        )
    if "r" not in obj:
        raise DecisionError("决策缺少字段：r")
    # 精简格式不回抄函数 id：候选表里只有一个函数，身份由宿主从候选表取，
    # assemble 会再用 _match_function 按名字与行范围复核源码，身份不靠模型保证。
    function_id = candidates.function_id
    filter_field = _parse_filter(obj.get("f"), "f")
    return_fields = _parse_return_fields(obj["r"], "r")
    sort_field = _parse_sort_field(obj.get("s"))
    sort_desc = _parse_sort_direction(obj.get("d"), sort=sort_field is not None)
    return _validate_ids(candidates, function_id, filter_field, return_fields, sort_field, sort_desc)


def _parse_filter(value: object, key: str) -> str | None:
    """条件字段：null / "null" 之类的写法都按不过滤处理，其余必须是候选 id 字符串。

    小模型常把 null 写成字符串 "null"；这只是 JSON 写法差异，不是新的 id，
    因此按“不过滤”处理。除此之外的任何字符串都必须匹配真实候选 id。
    """
    if value is None:
        return None
    if not isinstance(value, str):
        raise DecisionError(f"{key} 必须是条件候选 id 或 null")
    if value.strip().lower() in ("null", "none", ""):
        return None
    return value.strip()


def _parse_sort_field(value: object) -> str | None:
    """排序字段：null / "null" 之类的写法都按“不排序”处理，其余必须是候选 id 字符串。

    与 f 同一套写法差异：小模型把 null 写成字符串只是 JSON 写法问题，不是新 id。
    真的要排序时这里的 id 必须能在字段候选表里找到，由 _validate_ids 负责。
    """
    if value is None:
        return None
    if not isinstance(value, str):
        raise DecisionError("s 必须是字段候选 id 或 null（不排序就别给 s）")
    if value.strip().lower() in ("null", "none", ""):
        return None
    return value.strip()


def _parse_sort_direction(value: object, *, sort: bool) -> bool:
    """排序方向：升序给 False、降序给 True。

    给了 s 就必须给 d：不知道方向却替模型挑一个，等于猜。反过来没给 s 却给了 d
    也是自相矛盾，同样拒绝。方向只认 asc / desc 两组写法，别的一律拒绝。
    """
    text = _direction_text(value)
    if text is None:
        if sort:
            raise DecisionError("决策给了排序字段 s 却没给方向 d：升序写 asc，降序写 desc")
        return False
    if not sort:
        raise DecisionError("决策给了排序方向 d 却没给排序字段 s：不排序就别给这两个键")
    if text in _SORT_ASC:
        return False
    if text in _SORT_DESC:
        return True
    raise DecisionError(f"未知的排序方向：{value!r}（只接受 asc / desc）")


def _direction_text(value: object) -> str | None:
    """方向的规范化写法；null 与字符串 "null" 都算“没给方向”。"""
    if value is None:
        return None
    if not isinstance(value, str):
        raise DecisionError("d 必须是 asc 或 desc")
    text = value.strip().lower()
    return None if text in ("null", "none", "") else text


def _parse_return_fields(value: object, key: str) -> tuple[str, ...]:
    """返回字段：非空的字符串数组；空数组、非字符串元素一律拒绝。"""
    if not isinstance(value, list):
        raise DecisionError(f"{key} 必须是字段候选 id 的数组")
    if not value:
        raise DecisionError(f"{key} 不能为空：至少要选一个返回字段")
    for item in value:
        if not isinstance(item, str):
            raise DecisionError(f"{key} 里只能放字段候选 id 字符串")
    return tuple(item.strip() for item in value)


def assemble(source: str, candidates: Candidates, decision: Decision) -> str:
    """确定性重写：用 libcst 做无损手术，函数其余部分逐字保留。

    手术范围只有三处：append 外层的判断条件、被 append 的 dict 字面量（这两处
    只落在产出返回值的那个 for 循环上），以及决策要求的排序。函数里别的循环
    原样保留，不会被顺手改掉。改动落在原文件的 CST 上，只有这三处重新生成，
    守卫子句、注释、空行以及函数之外的字节都不参与重建，因此逐字留在结果里；
    模型没有参与这一步。

    排序槽位按决策分两种走法：决策里没有排序（sort_field 为 None）时，原文里的
    `sort` / `sorted` 一个字节都不动；决策里指定了排序时，替换掉已有的那处排序
    （只有一处、且在函数体顶层、且在产出返回值的循环之后才敢替换），没有就插在
    循环之后、`return 累加列表` 之前。替换不安全（多处排序、排序嵌在分支里、
    排序在循环之前）直接抛 DecisionError。

    找不到安全的 append 落点（不是 `累加列表.append({...})`、循环里有多处
    append、说不清哪个循环产出返回值等）同样直接抛 DecisionError，不做猜测，
    也不静默删代码。
    """
    tree = _parse(source)
    node = _match_function(tree, candidates)  # 名字 + 行范围复核：候选过期就拒绝
    _validate_ids(
        candidates,
        decision.function_id,
        decision.filter_field,
        decision.return_fields,
        decision.sort_field,
        decision.sort_desc,
    )
    styles = _collect(source, node).styles

    wrapper = MetadataWrapper(_parse_module(source), unsafe_skip_copy=True)
    function = _locate_function(wrapper, candidates)
    loop = _returned_loop(function)  # 只改产出返回值的那个循环，别的循环不动
    site = _append_site(loop)
    positions = wrapper.resolve(PositionProvider)
    _reject_rewrite_hazards(function, site, positions)
    # 手术只在目标循环和目标排序上做：模型给的是选择，代码由宿主确定性落成。
    plan = _plan_surgery(candidates, decision, styles, site)
    sort = _plan_sort(candidates, decision, styles, site, function)
    return wrapper.visit(_Surgery(function, loop, plan, sort)).code


@dataclass(frozen=True)
class _AppendSite:
    """循环里被定位的累加点：append 语句、外层判断、累加列表名、dict 字面量。"""

    loop: cst.For  # 承载这次累加的 for 循环
    item: str  # 循环变量名，如 "order"
    accumulator: str  # 累加列表名，如 "rows"
    line: cst.SimpleStatementLine | None  # 独立成行的 append 语句；单行 if 体里是 None
    small: cst.Expr  # `rows.append({...})` 这条表达式语句
    call: cst.Call  # 那条 append 调用
    guard: cst.If | None  # 包住这条 append 的 if；没有就是 None
    literal: cst.Dict  # 被 append 的 dict 字面量

    @property
    def owner(self) -> cst.CSTNode:
        """承载这条 append 的节点：独立一行，或单行 if 体里的小语句。"""
        return self.line if self.line is not None else self.small


@dataclass(frozen=True)
class _Plan:
    """一次手术的定稿：新的过滤条件（None 表示去掉条件）与要 append 的字段元素。"""

    test: cst.BaseExpression | None
    elements: tuple[cst.DictElement, ...]


@dataclass(frozen=True)
class _SortPlan:
    """一次排序手术的定稿：新的排序语句、落在函数体的第几条、替换还是插入。"""

    statement: cst.BaseStatement  # 新写出的 `累加列表.sort(key=..., reverse=...)`
    index: int  # 新语句在函数体顶层语句里的下标
    replace: bool  # True=替换第 index 条（旧排序）；False=插到第 index 条之前
    anchor: cst.CSTNode  # 复核用：替换时是旧排序语句，插入时是它前面那条语句（循环）


# 只认这一种累加写法：`累加列表.append(...)`；接收者是别的形状另行拒绝。
_APPEND_CALL = cstm.Call(func=cstm.Attribute(attr=cstm.Name(value="append")))
# 函数里任何一次调用；用来找已有的排序，以及判断它落在哪条顶层语句里。
_ANY_CALL = cstm.Call()
# append 藏在 try / 嵌套判断 / 内层循环里时定位不到，统一用这句话拒绝。
_NESTED_APPEND = (
    "暂不支持这个函数形状：append 不在循环体里，而在 try / 嵌套判断等更深的层级，无法安全定位"
)


def _append_site(loop: cst.For) -> _AppendSite:
    """定位循环里唯一的 `累加列表.append({...})`，顺带取出循环变量与外层判断。

    只认一种形状：一个 for 循环，循环体里（或循环体里某个顶层 if 的体里）
    恰好一条 `name.append({...})`。其它写法——`+=` / `extend` 之类、循环里
    有两处 append、append 藏在 try 或嵌套判断里——都不猜，直接抛 DecisionError。
    """
    body = loop.body
    if not isinstance(body, cst.IndentedBlock):
        raise DecisionError("暂不支持这个函数形状：for 与 append 写在同一行，无法安全手术")
    calls = list(cstm.findall(loop, _APPEND_CALL))
    if not calls:
        raise DecisionError(
            "暂不支持这个函数形状：循环里没有 `累加列表.append({...})` 这样的写法"
            "（+=、extend 等其它累加方式不在支持范围内）"
        )
    if len(calls) > 1:
        raise DecisionError(
            f"暂不支持这个函数形状：循环里有 {len(calls)} 处 append，无法确定该改哪一处。"
            "请改用生成路线，或先手工调整这个函数"
        )
    call = calls[0]
    owner, guard = _locate_append(call, body)
    if isinstance(owner, cst.SimpleStatementLine) and len(owner.body) != 1:
        raise DecisionError("暂不支持这个函数形状：append 与别的语句写在同一行，无法安全替换")
    small = owner.body[0] if isinstance(owner, cst.SimpleStatementLine) else owner
    if not isinstance(small, cst.Expr) or small.value is not call:
        raise DecisionError(
            "暂不支持这个函数形状：append 不是独立成句的写法（如 `x = rows.append(...)`），"
            "无法安全定位"
        )
    accumulator = _receiver_name(call)
    if accumulator is None:
        raise DecisionError("暂不支持这个函数形状：append 的接收者不是简单名字，无法确定累加列表")
    return _AppendSite(
        loop=loop,
        item=_loop_item(loop),
        accumulator=accumulator,
        line=owner if isinstance(owner, cst.SimpleStatementLine) else None,
        small=small,
        call=call,
        guard=guard,
        literal=_literal_of(call),
    )


def _holds(node: cst.CSTNode, call: cst.Call) -> bool:
    """这条 append 调用是否就在这个节点里。"""
    return any(found is call for found in cstm.findall(node, _APPEND_CALL))


def _locate_append(
    call: cst.Call, body: cst.IndentedBlock
) -> tuple[cst.CSTNode, cst.If | None]:
    """这条 append 所在的语句：直接写在循环体里，或写在顶层 if 的体里。

    只在循环体这一层和它的顶层 if 里找；再深（try、嵌套判断、内层 for）就拒绝，
    不猜哪一层才是本次要改的过滤，也不动别的语句。
    """
    for statement in body.body:
        if not _holds(statement, call):
            continue
        if isinstance(statement, cst.If):
            for inner in statement.body.body:
                if _holds(inner, call):
                    if not isinstance(inner, (cst.SimpleStatementLine, cst.Expr)):
                        raise DecisionError(_NESTED_APPEND)
                    return inner, statement
            raise DecisionError(_NESTED_APPEND)
        if not isinstance(statement, cst.SimpleStatementLine):
            raise DecisionError(_NESTED_APPEND)
        return statement, None
    raise DecisionError(_NESTED_APPEND)


def _loop_item(loop: cst.For) -> str:
    """循环变量名；元组解包取最后一个名字（enumerate / items 这类写法）。"""
    target = loop.target
    if isinstance(target, cst.Name):
        return target.value
    if isinstance(target, (cst.Tuple, cst.List)) and target.elements:
        last = target.elements[-1].value
        if isinstance(last, cst.Name):
            return last.value
    raise DecisionError("暂不支持这个函数形状：循环变量不是简单名字")


def _receiver_name(call: cst.Call) -> str | None:
    """append 的接收者名字：`rows.append(...)` 给 "rows"，别的形状给 None。"""
    func = call.func
    if isinstance(func, cst.Attribute) and isinstance(func.value, cst.Name):
        return func.value.value
    return None


def _literal_of(call: cst.Call) -> cst.Dict:
    """append 的实参必须是单个 dict 字面量；别的形状一律拒绝，不猜要替换什么。"""
    argument = call.args[0] if len(call.args) == 1 else None
    if (
        argument is None
        or argument.keyword is not None
        or argument.star
        or not isinstance(argument.value, cst.Dict)
    ):
        raise DecisionError("暂不支持这个函数形状：append 的实参不是单个 {…} 字面量，无法安全替换")
    return argument.value


def _append_to(accumulator: str) -> cstm.Call:
    """匹配 `累加列表.append(...)` 这种调用。"""
    return cstm.Call(
        func=cstm.Attribute(value=cstm.Name(value=accumulator), attr=cstm.Name(value="append"))
    )


def _reject_rewrite_hazards(
    function: cst.FunctionDef, site: _AppendSite, positions: Mapping[cst.CSTNode, CodeRange]
) -> None:
    """重写前检查：只拦真正无法安全手术的形状，不再拦循环外的普通语句。

    手术只替换循环里的判断和 dict，守卫子句、循环后的 sort、注释、空行都原样
    保留，所以它们在函数体里出现不再是危险。仍然危险的是同一个累加列表在别处
    还有第二处 append 或 `+=`：只改循环里这一处，会留下没按决策处理的数据。
    """
    others = [
        call
        for call in cstm.findall(function, _append_to(site.accumulator))
        if call is not site.call
    ]
    if others:
        raise DecisionError(
            f"暂不支持这个函数形状：{site.accumulator} 在{_line_of(positions, others[0])}"
            "还有一次 append，只改循环里这一处会让两处数据不一致。"
            "请改用生成路线，或先手工调整这个函数"
        )
    aug_assigned = cstm.findall(
        function, cstm.AugAssign(target=cstm.Name(value=site.accumulator))
    )
    if aug_assigned:
        raise DecisionError(
            f"暂不支持这个函数形状：{site.accumulator} 在{_line_of(positions, aug_assigned[0])}"
            "用了 `+=`，只支持 `累加列表.append({...})` 这一种累加写法"
        )


def run_decision(
    engine: Engine,
    instruction: str,
    source: str,
    target: str,
    function_name: str | None = None,
    *,
    sort_enabled: bool = False,
) -> tuple[Artifact, Decision, Candidates]:
    """完整流程：extract → prompt → generate → parse → assemble → make_artifact。

    模型只回一个短 JSON 决策，正文全部由宿主组装；返回的 Artifact 已由宿主
    算好 content_hash（make_artifact 会去掉首尾空行），交给调用方按 Gate
    流程确认后原样写入。engine 只依赖 azfls.model.Engine 协议，测试用
    ScriptedEngine，不加载权重。
    """
    candidates = extract(source, function_name)
    messages = build_decision_prompt(instruction, candidates, sort_enabled=sort_enabled)
    raw, _stats = engine.generate(messages, max_tokens=DECISION_MAX_TOKENS)
    decision = parse_decision(raw, candidates, sort_enabled=sort_enabled)
    body = assemble(source, candidates, decision)
    artifact = make_artifact(
        target=target,
        # 这条路径的正文由宿主组装，不是模型写的；传规范化后的正文，
        # 让 raw_response 与 body 一致（模型的原话是决策 JSON，记在 notes 里）。
        raw_response=normalize_body(body),
        kind=Kind.CODE,
        action=Action.REPLACE,
        notes=(
            f"选择式产出：{describe_decision(candidates, decision)}"
            "（id 已按候选表校验，正文由宿主组装，模型未生成代码）",
            f"模型决策原文：{_one_line(raw)}",
            *_limitation_notes(source, candidates, decision),
        ),
    )
    return artifact, decision, candidates


def _limitation_notes(
    source: str, candidates: Candidates, decision: Decision
) -> tuple[str, ...]:
    """把这条路径表达不了的东西如实说出来，交大模型判断。

    选择式一次只表达一个条件；如果原函数用的是复合条件（and / or），
    重写后条件会变宽或变窄。不静默处理，也不自动改用别的路线。
    函数里还有别的循环在向别的累加列表 append 时，也如实说明本次没有动它们。
    """
    tree = _parse(source)
    node = _find_function(tree, candidates.function_name)
    notes: list[str] = []
    if _has_compound_test(node):
        if decision.filter_field is None:
            notes.append("注意：原函数用了复合条件，本次决策不过滤，条件已被去掉。")
        else:
            notes.append(
                "注意：原函数用了复合条件，而选择式一次只保留一个条件，"
                "过滤范围可能与原来不同，请人工确认。"
            )
    untouched = _untouched_loops_note(source, candidates)
    if untouched:
        notes.append(untouched)
    return tuple(notes)


def _untouched_loops_note(source: str, candidates: Candidates) -> str | None:
    """还有别的循环在向别的累加列表 append 时，说清楚本次没有动它们。

    只改产出返回值的那个循环；别的循环逐字留在原文里，这里如实说明，
    免得看着 diff 以为改错了地方。没有这种情况就不加提示，不制造噪音。
    """
    wrapper = MetadataWrapper(_parse_module(source), unsafe_skip_copy=True)
    function = _locate_function(wrapper, candidates)
    returned = _returned_name(function)
    others = sum(
        1
        for statement in _body_statements(function)
        if isinstance(statement, cst.For) and _loop_receivers(statement) - {returned}
    )
    if not others:
        return None
    return (
        f"注意：函数里还有 {others} 个循环在向别的累加列表 append，"
        "本次只改了产出返回值的那个循环，其余原样保留。"
    )


def _has_compound_test(node: ast.AST) -> bool:
    """函数体里的判断是否用了 and / or（复合条件）。"""
    for child in ast.walk(node):
        if isinstance(child, (ast.BoolOp,)):
            return True
    return False


def describe_decision(candidates: Candidates, decision: Decision) -> str:
    """一行中文摘要：过滤、返回字段、排序各选了谁；给大模型判断和展示用。"""
    if decision.filter_field is None:
        condition = "不过滤"
    else:
        condition = f"filter={decision.filter_field}={_name_of(candidates.conditions, decision.filter_field)}"
    fields = "、".join(
        f"{fid}={_name_of(candidates.fields, fid)}" for fid in decision.return_fields
    )
    line = f"function={candidates.function_id}，{condition}，return_fields={fields}"
    if decision.sort_field is None:
        return f"{line}，排序=不动原文"
    direction = "降序" if decision.sort_desc else "升序"
    return f"{line}，排序={decision.sort_field}={_name_of(candidates.fields, decision.sort_field)} {direction}"


# --------------------------------------------------------------------------
# 候选提取：按源码顺序遍历函数体
# --------------------------------------------------------------------------


class _Collector:
    """按源码顺序遍历函数体，收集字段引用、条件引用和取值风格。

    只有“被读取”的位置才算字段：dict 键、下标取字符串、值位置上的属性、
    `.get("键")`。调用位置上的属性（如 `rows.append`）是方法名，不算字段。
    """

    def __init__(self, source: str) -> None:
        self.source = source
        self.fields: list[str] = []
        self.conditions: list[str] = []
        self.styles: dict[str, tuple[str, str]] = {}  # 名字 -> (风格, 引号)
        self._test_depth = 0

    def visit_body(self, node: ast.AST) -> None:
        for stmt in getattr(node, "body", ()):
            self.visit(stmt)

    def visit(self, node: ast.AST | None, *, call_func: bool = False) -> None:
        """遍历一个节点；call_func 表示它处在被调用位置。"""
        if node is None:
            return
        if isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values):
                if isinstance(key, ast.Constant) and isinstance(key.value, str):
                    self._field(key.value)  # 键名算字段，但不代表读取风格
                else:
                    self.visit(key)
                self.visit(value)
            return
        if isinstance(node, ast.Subscript):
            name, quote = _literal_key(self.source, node.slice)
            if name:
                self._field(name, "item", quote)
            else:
                self.visit(node.slice)
            self.visit(node.value)
            return
        if isinstance(node, ast.Attribute):
            if not call_func:
                self._field(node.attr, "attr")
            self.visit(node.value)
            return
        if isinstance(node, ast.Call):
            self.visit_call(node)
            return
        if isinstance(node, (ast.If, ast.While, ast.IfExp, ast.Assert)):
            self.visit_test(node.test)
            for child in ast.iter_child_nodes(node):
                if child is not node.test:
                    self.visit(child)
            return
        if isinstance(node, ast.comprehension):
            self.visit(node.target)
            self.visit(node.iter)
            for condition in node.ifs:
                self.visit_test(condition)
            return
        for child in ast.iter_child_nodes(node):
            self.visit(child)

    def visit_call(self, node: ast.Call) -> None:
        """调用：方法名不算字段，但 `item.get("k")` 里的键算字段。"""
        func = node.func
        consumed = 0
        if isinstance(func, ast.Attribute) and func.attr == "get":
            self.visit(func.value)
            if node.args:
                name, quote = _literal_key(self.source, node.args[0])
                if name:
                    self._field(name, "get", quote)
                else:
                    self.visit(node.args[0])
                consumed = 1
        else:
            self.visit(func, call_func=True)
        for arg in node.args[consumed:]:
            self.visit(arg)
        for keyword in node.keywords:
            self.visit(keyword.value)

    def visit_test(self, test: ast.AST) -> None:
        """判断位置上的引用同时算条件候选。"""
        self._test_depth += 1
        try:
            self.visit(test)
        finally:
            self._test_depth -= 1

    def _field(self, name: str, style: str | None = None, quote: str = _ATTR_FALLBACK_QUOTE) -> None:
        if name not in self.fields:
            self.fields.append(name)
        if style is not None and name not in self.styles:
            self.styles[name] = (style, quote)
        if self._test_depth and name not in self.conditions:
            self.conditions.append(name)


def _collect(source: str, node: ast.AST) -> _Collector:
    collector = _Collector(source)
    collector.visit_body(node)
    return collector


def _parse(source: str) -> ast.Module:
    """解析源码；语法不合法时抛 DecisionError。"""
    try:
        return ast.parse(source)
    except SyntaxError as exc:
        raise DecisionError(f"源码无法解析：第 {exc.lineno} 行 {exc.msg}") from exc


def _find_function(
    tree: ast.Module, function_name: str | None
) -> ast.FunctionDef | ast.AsyncFunctionDef:
    """按名字找函数；没给名字时优先模块级函数（跳过 _ 开头的内部函数）。"""
    found = [n for n in ast.walk(tree) if isinstance(n, _FUNC_TYPES)]
    if function_name:
        matched = [n for n in found if n.name == function_name]
        if not matched:
            raise DecisionError(f"源码里没有函数 {function_name}")
        return min(matched, key=lambda n: (n.lineno, n.col_offset))
    top_level = [n for n in tree.body if isinstance(n, _FUNC_TYPES)]
    pool = top_level or found
    public = [n for n in pool if not n.name.startswith("_")]
    if public:
        return min(public, key=lambda n: (n.lineno, n.col_offset))
    if pool:
        return min(pool, key=lambda n: (n.lineno, n.col_offset))
    raise DecisionError("源码里没有可用的函数")


def _match_function(
    tree: ast.Module, candidates: Candidates
) -> ast.FunctionDef | ast.AsyncFunctionDef:
    """重写前复核：函数名与行位置都必须和候选一致，否则视为源码已改变。"""
    node = _find_function(tree, candidates.function_name)
    if node.lineno != candidates.start_line or (node.end_lineno or node.lineno) != candidates.end_line:
        raise DecisionError("源码与候选不一致：函数位置已改变，请重新提取候选")
    return node


# --------------------------------------------------------------------------
# 决策校验与 JSON 解析
# --------------------------------------------------------------------------


def _validate_ids(
    candidates: Candidates,
    function_id: str,
    filter_field: str | None,
    return_fields: tuple[str, ...],
    sort_field: str | None = None,
    sort_desc: bool = False,
) -> Decision:
    """所有 id 必须来自候选表；未知 id 一律拒绝，绝不接受模型自造的标识。

    排序字段只能用**字段候选**（f0,f1,…），与 r 同一张表，不另开一套 id；
    而且必须同时出现在 r 里：累加列表里只写了 r 选中的键，排序键读不到别的键。
    """
    if function_id != candidates.function_id:
        raise DecisionError(f"未知的函数 id：{function_id!r}（只能是 {candidates.function_id}）")
    condition_ids = [item.id for item in candidates.conditions]
    if filter_field is not None and filter_field not in condition_ids:
        raise DecisionError(
            f"未知的条件 id：{filter_field!r}（可用：{_ids(condition_ids)}；不过滤用 null）"
        )
    field_ids = [item.id for item in candidates.fields]
    for field_id in return_fields:
        if field_id not in field_ids:
            raise DecisionError(f"未知的字段 id：{field_id!r}（可用：{_ids(field_ids)}）")
    if len(set(return_fields)) != len(return_fields):
        raise DecisionError(f"return_fields 里有重复 id：{_ids(list(return_fields))}")
    if sort_field is not None:
        if sort_field not in field_ids:
            raise DecisionError(
                f"未知的排序字段 id：{sort_field!r}（s 只能选字段候选：{_ids(field_ids)}）"
            )
        if sort_field not in return_fields:
            raise DecisionError(
                f"排序字段 {sort_field!r} 不在返回字段 r 里：累加列表里只有 r 选中的键，"
                "排序键读不到别的键。请把这个字段加进 r，或者不要排序"
            )
    return Decision(
        function_id=function_id,
        filter_field=filter_field,
        return_fields=tuple(return_fields),
        sort_field=sort_field,
        sort_desc=sort_desc,
    )


def _json_object(text: str) -> object:
    """取回复里的第一个 JSON 对象；不修补、不猜测残缺内容。

    允许外面有围栏或说明文字（说明里的花括号会被跳过）；但如果每个候选片段
    都解析不出对象，就拒绝，不去猜模型想说什么。
    """
    cleaned = strip_code_fence(text or "").strip()
    try:
        return json.loads(cleaned)
    except ValueError:
        pass
    for fragment in _balanced_objects(cleaned):
        try:
            obj = json.loads(fragment)
        except ValueError:
            continue  # 说明文字里的花括号，换个起点继续找
        if isinstance(obj, dict):
            return obj
    raise DecisionError("回复不是 JSON：只接受一个 JSON 对象，不接受解释文字")


def _balanced_objects(text: str) -> list[str]:
    """按从左到右的顺序，给出每个 `{` 起头的配平片段（跳过字符串里的括号）。"""
    fragments: list[str] = []
    for start, char in enumerate(text):
        if char != "{":
            continue
        fragment = _balanced_from(text, start)
        if fragment is not None:
            fragments.append(fragment)
    return fragments


def _balanced_from(text: str, start: int) -> str | None:
    """从 start 处的 `{` 起找配平的 `}`；找不到返回 None。"""
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


# --------------------------------------------------------------------------
# 确定性组装
# --------------------------------------------------------------------------


class _Surgery(cst.CSTTransformer):
    """只在目标函数的目标循环与排序上做手术；其它节点原样返回，字节不动。"""

    def __init__(
        self,
        function: cst.FunctionDef,
        loop: cst.For,
        plan: _Plan,
        sort: _SortPlan | None,
    ) -> None:
        self._function = function  # 原始节点：只用来比较身份，不改它
        self._loop = loop  # 同上
        self._plan = plan
        self._sort = sort

    def leave_For(self, original_node: cst.For, updated_node: cst.For) -> cst.For:
        if original_node is not self._loop:
            return updated_node  # 别的循环（别的函数、内层循环）一律不动
        return _surgery(updated_node, _append_site(updated_node), self._plan)

    def leave_FunctionDef(
        self, original_node: cst.FunctionDef, updated_node: cst.FunctionDef
    ) -> cst.FunctionDef:
        if self._sort is None or original_node is not self._function:
            return updated_node  # 决策没要求排序：原文里的 sort / sorted 一个字节都不动
        return _apply_sort(updated_node, original_node, self._sort)


def _apply_sort(
    function: cst.FunctionDef, original: cst.FunctionDef, plan: _SortPlan
) -> cst.FunctionDef:
    """把排序手术落到函数体上：替换旧排序那条语句，或在循环之后插一条。

    逐条重建函数体的语句列表，但只有目标那一条是新节点，其余原对象照搬，
    因此注释、空行、守卫子句和别的语句都逐字保留。插入时节点位置由
    `_SortPlan.index` 给出，libcst 按所在块自动给出缩进。
    """
    body = function.body
    if not isinstance(body, cst.IndentedBlock):
        raise DecisionError("内部错误：函数体不是缩进块，拒绝写出可能错误的正文")
    originals = original.body
    if not isinstance(originals, cst.IndentedBlock):
        raise DecisionError("内部错误：排序无处落笔，拒绝写出可能错误的正文")
    anchor = plan.index if plan.replace else plan.index - 1
    if not 0 <= anchor < len(originals.body) or originals.body[anchor] is not plan.anchor:
        raise DecisionError("内部错误：排序的落点对不上，拒绝写出可能错误的正文")
    statements = list(body.body)
    if plan.replace and isinstance(plan.anchor, cst.SimpleStatementLine):
        # 替换旧排序：它上方的注释与空行留在原处（删掉注释比留下更容易误导）。
        statements[plan.index] = plan.statement.with_changes(
            leading_lines=plan.anchor.leading_lines
        )
    else:
        statements.insert(plan.index, plan.statement)  # 插在产出返回值的循环之后
    return function.with_changes(body=body.with_changes(body=tuple(statements)))


def _surgery(loop: cst.For, site: _AppendSite, plan: _Plan) -> cst.For:
    """把计划落到循环上：换掉 dict，并按计划插入 / 替换 / 去掉 append 外层的判断。"""
    statement = _with_elements(site, plan.elements)
    guard = site.guard
    if plan.test is None:
        if guard is None:
            return _replace_in_loop(loop, site.owner, (statement,))
        _reject_else(guard)
        # 去掉条件：append 提到循环体一层，判断上方的注释留在原处。
        loop = _replace_in_loop(loop, guard, (_unwrap(guard, statement),))
        return _keep_inner_footer(loop, guard)
    if guard is None:
        # 原本没有判断：插入 `if <条件>:`，append 连同自己的注释挪进判断体。
        return _replace_in_loop(loop, site.owner, (_wrap(statement, plan.test),))
    _reject_else(guard)
    new_guard = guard.with_changes(
        test=plan.test,  # 只换判断表达式，行首缩进、冒号和行尾注释都留在原处
        body=guard.body.with_changes(
            body=tuple(statement if node is site.owner else node for node in guard.body.body)
        ),
    )
    return _replace_in_loop(loop, guard, (new_guard,))


def _with_elements(
    site: _AppendSite, elements: tuple[cst.DictElement, ...]
) -> cst.BaseStatement | cst.BaseSmallStatement:
    """把 append 的 dict 换成决策选定的字段，语句的其余部分原样保留。"""
    argument = site.call.args[0]
    call = site.call.with_changes(
        args=(argument.with_changes(value=site.literal.with_changes(elements=elements)),)
    )
    small = site.small.with_changes(value=call)
    if site.line is None:
        return small  # 单行 `if cond: rows.append(...)`：小语句放回 if 体
    return site.line.with_changes(body=(small,))


def _wrap(
    statement: cst.BaseStatement | cst.BaseSmallStatement, test: cst.BaseExpression
) -> cst.If:
    """原本没有判断：插入 `if <条件>:`，append 缩进一级；上方的注释留在原处不动。"""
    if not isinstance(statement, cst.SimpleStatementLine):
        raise DecisionError("内部错误：判断无处插入，拒绝写出可能错误的正文")
    return cst.If(
        test=test,
        body=cst.IndentedBlock(body=(statement.with_changes(leading_lines=()),)),
        leading_lines=statement.leading_lines,
    )


def _unwrap(
    guard: cst.If, statement: cst.BaseStatement | cst.BaseSmallStatement
) -> cst.BaseStatement:
    """去掉判断：append 提回循环体一层，判断上方的注释留在原处。"""
    leading = tuple(guard.leading_lines)
    if isinstance(statement, cst.SimpleStatementLine):
        return statement.with_changes(leading_lines=leading + tuple(statement.leading_lines))
    # 单行 `if cond: rows.append({...})`：小语句要包成独立一行，行尾注释一起搬过来。
    trailing = (
        guard.body.trailing_whitespace
        if isinstance(guard.body, cst.SimpleStatementSuite)
        else cst.TrailingWhitespace()
    )
    return cst.SimpleStatementLine(
        body=(statement,), leading_lines=leading, trailing_whitespace=trailing
    )


def _keep_inner_footer(loop: cst.For, guard: cst.If) -> cst.For:
    """判断体末尾的空行 / 注释跟着提到循环体一层，绝不留在被丢掉的分支里。"""
    inner = guard.body
    if not isinstance(inner, cst.IndentedBlock) or not inner.footer or not isinstance(
        loop.body, cst.IndentedBlock
    ):
        return loop
    return loop.with_changes(
        body=loop.body.with_changes(footer=tuple(inner.footer) + tuple(loop.body.footer))
    )


def _replace_in_loop(
    loop: cst.For, old: cst.CSTNode, new: tuple[cst.BaseStatement, ...]
) -> cst.For:
    """把循环体里的 old 换成 new；同一层的其它语句原样保留。"""
    body = loop.body
    if not isinstance(body, cst.IndentedBlock):
        raise DecisionError("内部错误：循环体不是缩进块，拒绝写出可能错误的正文")
    statements: list[cst.BaseStatement] = []
    replaced = False
    for statement in body.body:
        if statement is old:
            statements.extend(new)
            replaced = True
        else:
            statements.append(statement)
    if not replaced:
        raise DecisionError("内部错误：没有在循环体里找到要替换的语句，拒绝写出可能错误的正文")
    return loop.with_changes(body=body.with_changes(body=tuple(statements)))


def _reject_else(guard: cst.If) -> None:
    """判断带 else / elif 时，改条件会连带改变另一分支的行为，一律拒绝，不静默改语义。"""
    if guard.orelse is not None:
        raise DecisionError(
            "暂不支持这个函数形状：包住 append 的判断带 else / elif 分支，"
            "改条件会连带改变另一分支的行为。请改用生成路线，或先手工调整这个函数"
        )


def _plan_surgery(
    candidates: Candidates,
    decision: Decision,
    styles: dict[str, tuple[str, str]],
    site: _AppendSite,
) -> _Plan:
    """按决策渲染好两处手术的结果：新的过滤条件与新的 dict 字段元素。"""
    quote = _quote_for(styles, candidates, decision)
    test = None
    if decision.filter_field is not None:
        name = _name_of(candidates.conditions, decision.filter_field)
        test = _expression(_access(site.item, name, styles, quote))
    elements: list[cst.DictElement] = []
    for field_id in decision.return_fields:
        name = _name_of(candidates.fields, field_id)
        elements.append(
            cst.DictElement(
                key=cst.SimpleString(f"{quote}{name}{quote}"),
                value=_expression(_access(site.item, name, styles, quote)),
            )
        )
    return _Plan(test=test, elements=tuple(elements))


# --------------------------------------------------------------------------
# 排序槽位：决策要求排序时才动原文里的排序
# --------------------------------------------------------------------------


def _plan_sort(
    candidates: Candidates,
    decision: Decision,
    styles: dict[str, tuple[str, str]],
    site: _AppendSite,
    function: cst.FunctionDef,
) -> _SortPlan | None:
    """按决策渲染排序手术；决策里没有排序时返回 None（原文的排序一个字节都不动）。

    替换已有的排序只在三种情况下做：函数体顶层、独立成句、且在产出返回值的
    循环之后。其余形状（多处排序、排序嵌在分支或循环里、排序在循环之前）
    说不清该替换哪一个，抛 DecisionError，不猜也不留两处排序。
    """
    if decision.sort_field is None:
        return None
    name = _name_of(candidates.fields, decision.sort_field)
    statement = _sort_statement(site.accumulator, name, _sort_quote(styles, name), styles, decision)
    body = _body_statements(function)
    loop_index = _index_of(body, site.loop)
    returned = _returned_statement(function)
    return_index = _index_of(body, returned)
    if loop_index is None or return_index is None or return_index < loop_index:
        raise DecisionError(
            f"暂不支持这个函数形状：`return {site.accumulator}` 不在产出返回值的循环之后，"
            "无法确定排序插在哪里"
        )
    existing = _accumulator_sorts(function, site.accumulator)
    if len(existing) > 1:
        raise DecisionError(
            f"暂不支持这个函数形状：函数里有 {len(existing)} 处对 {site.accumulator} 的排序，"
            "无法确定该替换哪一处。请改用生成路线，或先手工调整这个函数"
        )
    if not existing:
        # 插在产出返回值的循环之后、`return 累加列表` 之前（return 在下标上已复核过）。
        return _SortPlan(
            statement=statement, index=loop_index + 1, replace=False, anchor=site.loop
        )
    owner, _call = existing[0]
    index = _index_of(body, owner)
    if index is None:
        raise DecisionError(_unsafe_sort_shape(site.accumulator))
    if index < loop_index:
        raise DecisionError(
            f"暂不支持这个函数形状：{site.accumulator} 的排序在产出返回值的循环之前，"
            "替换它排的就是循环里的旧数据。请先手工调整这个函数"
        )
    if not _is_replaceable_sort(owner, site.accumulator):
        raise DecisionError(_unsafe_sort_shape(site.accumulator))
    return _SortPlan(statement=statement, index=index, replace=True, anchor=owner)


def _unsafe_sort_shape(accumulator: str) -> str:
    """已有的排序不是能整条替换的写法时的统一拒绝说明。"""
    return (
        f"暂不支持这个函数形状：已有的排序不是函数体顶层独立成句的 "
        f"`{accumulator}.sort(...)` 或 `{accumulator} = sorted({accumulator}, ...)`"
        "（嵌在分支 / 循环里、写在别的表达式里都在此列），无法安全替换。"
        "请改用生成路线，或先手工调整这个函数"
    )


def _sort_statement(
    accumulator: str,
    name: str,
    quote: str,
    styles: dict[str, tuple[str, str]],
    decision: Decision,
) -> cst.SimpleStatementLine:
    """渲染 `累加列表.sort(key=lambda r: <取值>, reverse=True)` 这一条语句。

    排序键读的是累加列表里的 dict 元素，所以只可能是下标或 `.get(...)`：
    源码里这个字段用 `.get` 就跟着用 `.get`，用属性取值也照样写成下标——
    属性取值取的是循环里那个对象，而这里排的是已经攒好的 dict。
    引号跟源码观察到的写法一致。
    """
    item = _LAMBDA_ITEM if accumulator != _LAMBDA_ITEM else _LAMBDA_ITEM_FALLBACK
    if styles.get(name, ("item", quote))[0] == "get":
        key = f"{item}.get({quote}{name}{quote})"
    else:
        key = f"{item}[{quote}{name}{quote}]"
    reverse = ", reverse=True" if decision.sort_desc else ""
    return cst.SimpleStatementLine(
        body=(
            cst.Expr(
                value=_expression(f"{accumulator}.sort(key=lambda {item}: {key}{reverse})")
            ),
        )
    )


def _sort_quote(styles: dict[str, tuple[str, str]], name: str) -> str:
    """排序键用的引号：优先用源码里这个字段观察到的引号，没观察到再用文件默认。"""
    return styles.get(name, ("", _ATTR_FALLBACK_QUOTE))[1] or _ATTR_FALLBACK_QUOTE


def _accumulator_sorts(
    function: cst.FunctionDef, accumulator: str
) -> list[tuple[cst.BaseStatement | None, cst.Call]]:
    """函数里对累加列表的排序调用，连同它所在的那条顶层语句（不在顶层则为 None）。

    只算 `累加列表.sort(...)` 与 `sorted(累加列表, ...)`；排的是别的列表不算，
    那种调用不影响返回值顺序，原样留着。没有向决策请求排序时一个都不看，
    原文里的排序逐字节保留。
    """
    top = _body_statements(function)
    found: list[tuple[cst.BaseStatement | None, cst.Call]] = []
    for call in cstm.findall(function, _ANY_CALL):
        if not _is_accumulator_sort(call, accumulator):
            continue
        owner = next(
            (
                statement
                for statement in top
                if any(item is call for item in cstm.findall(statement, _ANY_CALL))
            ),
            None,
        )
        found.append((owner, call))
    return found


def _is_accumulator_sort(call: cst.Call, accumulator: str) -> bool:
    """这次调用是不是在对累加列表排序：`acc.sort(...)` 或 `sorted(acc, ...)`。"""
    func = call.func
    if isinstance(func, cst.Attribute):
        return (
            isinstance(func.value, cst.Name)
            and func.value.value == accumulator
            and func.attr.value == "sort"
        )
    if isinstance(func, cst.Name) and func.value == "sorted":
        return any(
            arg.keyword is None
            and not arg.star
            and isinstance(arg.value, cst.Name)
            and arg.value.value == accumulator
            for arg in call.args
        )
    return False


def _is_replaceable_sort(statement: cst.BaseStatement, accumulator: str) -> bool:
    """已有的排序是不是可以整条替换的写法：`acc.sort(...)` 或 `acc = sorted(acc, ...)`。"""
    if not isinstance(statement, cst.SimpleStatementLine) or len(statement.body) != 1:
        return False
    small = statement.body[0]
    if isinstance(small, cst.Expr) and isinstance(small.value, cst.Call):
        func = small.value.func
        return (
            isinstance(func, cst.Attribute)
            and isinstance(func.value, cst.Name)
            and func.value.value == accumulator
            and func.attr.value == "sort"
        )
    if isinstance(small, cst.Assign) and len(small.targets) == 1:
        target = small.targets[0].target
        return (
            isinstance(target, cst.Name)
            and target.value == accumulator
            and isinstance(small.value, cst.Call)
            and isinstance(small.value.func, cst.Name)
            and small.value.func.value == "sorted"
        )
    return False


def _index_of(statements: tuple[cst.BaseStatement, ...], node: cst.CSTNode | None) -> int | None:
    """node 在顶层语句列表里的下标；找不到（None 或不在这一层）给 None。"""
    if node is None:
        return None
    for index, statement in enumerate(statements):
        if statement is node:
            return index
    return None


def _expression(text: str) -> cst.BaseExpression:
    """把宿主渲染好的表达式解析成 CST；渲染不出合法表达式时拒绝，不猜。"""
    try:
        return cst.parse_expression(text)
    except cst.ParserSyntaxError as exc:
        raise DecisionError(
            f"内部错误：宿主渲染的表达式不是合法 Python（{text}）：{exc.message}"
        ) from exc


class _FunctionLocator(cst.CSTVisitor):
    """按名字找函数，并记下每个同名函数的起止行。"""

    METADATA_DEPENDENCIES = (PositionProvider,)

    def __init__(self, name: str) -> None:
        self.name = name
        self.found: list[tuple[int, int, cst.FunctionDef]] = []

    def visit_FunctionDef(self, node: cst.FunctionDef) -> bool:
        if node.name.value == self.name:
            span = self.get_metadata(PositionProvider, node)
            self.found.append((span.start.line, span.end.line, node))
        return True


def _locate_function(wrapper: MetadataWrapper, candidates: Candidates) -> cst.FunctionDef:
    """在 CST 上找出与候选同名、同起止行的函数；对不上就是源码已经变了。"""
    locator = _FunctionLocator(candidates.function_name)
    wrapper.visit(locator)
    for start, end, node in locator.found:
        if (start, end) == (candidates.start_line, candidates.end_line):
            return node
    if not locator.found:
        raise DecisionError(f"源码里没有函数 {candidates.function_name}")
    raise DecisionError("源码与候选不一致：函数位置已改变，请重新提取候选")


def _body_statements(function: cst.FunctionDef) -> tuple[cst.BaseStatement, ...]:
    """函数体顶层的语句（不含缩进块自己）。"""
    body = function.body
    return tuple(body.body) if isinstance(body, cst.IndentedBlock) else ()


def _returned_statement(function: cst.FunctionDef) -> cst.SimpleStatementLine | None:
    """函数体里最后一条 `return <名字>` 语句；没有给 None。"""
    found: cst.SimpleStatementLine | None = None
    for statement in _body_statements(function):
        if not isinstance(statement, cst.SimpleStatementLine):
            continue
        for small in statement.body:
            if isinstance(small, cst.Return) and isinstance(small.value, cst.Name):
                found = statement
    return found


def _returned_name(function: cst.FunctionDef) -> str:
    """函数体里最后一条 `return <名字>` 的名字；没有就拒绝，不猜返回值从哪来。"""
    statement = _returned_statement(function)
    name = None
    if statement is not None:
        for small in statement.body:
            if isinstance(small, cst.Return) and isinstance(small.value, cst.Name):
                name = small.value.value
    if name is None:
        raise DecisionError(
            "暂不支持这个函数形状：函数里没有 `return 累加列表` 这样的写法，"
            "无法确定返回值来自哪个循环"
        )
    return name


def _loop_receivers(loop: cst.For) -> set[str]:
    """循环里 `x.append(...)` 的接收者名字，用来判断它累加的是哪个列表。"""
    receivers: set[str] = set()
    for call in cstm.findall(loop, _APPEND_CALL):
        name = _receiver_name(call)
        if name is not None:
            receivers.add(name)
    return receivers


def _returned_loop(function: cst.FunctionDef) -> cst.For:
    """产出返回值的那个循环：向 `return <名字>` 里的累加列表 append 的 for 循环。

    一个函数里可能有好几个循环（先去重、再产出行之类），只有向返回的那个
    累加列表 append 的循环才产出返回值；别的循环原样留着，不动也不报错。
    找不到、或者有多个循环都在向它 append（说不清哪个产出返回值）都不猜，
    直接拒绝——改错循环会安静地改出个坏结果。
    """
    accumulator = _returned_name(function)
    loops = [
        statement
        for statement in _body_statements(function)
        if isinstance(statement, cst.For) and accumulator in _loop_receivers(statement)
    ]
    if not loops:
        raise DecisionError(
            f"暂不支持这个函数形状：函数里没有 for 循环在向 {accumulator} append"
            f"（返回值来自它），无法确定要改哪个循环"
        )
    if len(loops) > 1:
        raise DecisionError(
            f"暂不支持这个函数形状：有 {len(loops)} 个循环都在向 {accumulator} append，"
            "无法确定哪个产出返回值。请改用生成路线，或先手工调整这个函数"
        )
    return loops[0]


def _parse_module(source: str) -> cst.Module:
    """用 libcst 解析源码，拿到无损 CST：改写后只有手术那两处会重新生成。

    ast 在前面已经解析过一次，这里再解析一次是因为分工不同：ast 用来复核函数、
    观察取值风格，libcst 用来做手术。libcst 解析不了（例如很新的语法）就拒绝，
    不退回手写切片，也不猜。
    """
    try:
        return cst.parse_module(source)
    except cst.ParserSyntaxError as exc:
        raise DecisionError(f"源码无法解析：第 {exc.raw_line} 行 {exc.message}") from exc


def _line_of(positions: Mapping[cst.CSTNode, CodeRange], node: cst.CSTNode) -> str:
    """节点所在行；位置拿不到时给一句兜底，绝不因为拼报错信息再抛一次异常。"""
    span = positions.get(node)
    return f"第 {span.start.line} 行" if span is not None else "另一处"


def _access(item: str, name: str, styles: dict[str, tuple[str, str]], quote: str) -> str:
    """按源码里观察到的取值风格渲染一个字段读取。"""
    style = styles.get(name, ("item", quote))[0]
    if style == "attr" and name.isidentifier():
        return f"{item}.{name}"
    if style == "get":
        return f"{item}.get({quote}{name}{quote})"
    return f"{item}[{quote}{name}{quote}]"


def _quote_for(styles: dict[str, tuple[str, str]], candidates: Candidates, decision: Decision) -> str:
    """整个 dict 字面量用同一个引号：取第一个被选字段观察到的引号风格。"""
    for field_id in decision.return_fields:
        name = _name_of(candidates.fields, field_id)
        if name in styles:
            return styles[name][1]
    return _ATTR_FALLBACK_QUOTE


def _literal_key(source: str, node: ast.AST) -> tuple[str, str]:
    """字符串字面量下标/键：返回 (字段名, 引号)；不是字符串时返回空。"""
    if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
        return "", ""
    segment = ast.get_source_segment(source, node) or ""
    quote = segment[0] if segment[:1] in ("'", '"') else _ATTR_FALLBACK_QUOTE
    return node.value, quote


def _name_of(candidates: tuple[Candidate, ...], candidate_id: str) -> str:
    for item in candidates:
        if item.id == candidate_id:
            return item.name
    return "?"


def _pairs(candidates: tuple[Candidate, ...]) -> str:
    """候选表的一行紧凑写法：f0=id, f1=name。"""
    return ", ".join(f"{item.id}={item.name}" for item in candidates) or "（无）"


def _ids(ids: list[str]) -> str:
    return ", ".join(ids) if ids else "无"


def _one_line(text: str) -> str:
    """把模型回复压成一行短摘要，便于放进 notes 展示。"""
    collapsed = " ".join((text or "").split())
    return collapsed[:160] if collapsed else "（空）"
