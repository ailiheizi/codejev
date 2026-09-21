"""共享数据结构。

大模型给短指令，小模型只回正文；本文件定义两者之间的固定交接形状。
身份、路径和内容哈希一律由宿主（本程序）生成，绝不从模型输出接受。
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

CODE_FENCE_RE = re.compile(r"^\s*```[^\n]*\n(.*?)\n?\s*```\s*$", re.DOTALL)
# 单独一行的围栏标记（前后无内容）；用来清掉模型偶尔多吐的孤立围栏。
_FENCE_LINE_RE = re.compile(r"^\s*```")


class Action(str, Enum):
    """小模型这次要产出的东西。"""

    CREATE = "create"  # 新建文件
    REPLACE = "replace"  # 整体替换目标文件
    EDIT = "edit"  # 只输出被改动的那一段


class Kind(str, Enum):
    """正文类型，决定提示词与展示围栏。"""

    CODE = "code"
    JSON = "json"
    TEXT = "text"

    @property
    def fence(self) -> str:
        return {"code": "python", "json": "json", "text": "text"}[self.value]


@dataclass(frozen=True)
class Brief:
    """大模型交给小模型的一条短指令。

    instruction 说做什么、怎么改；context 只放本次需要的原文；
    keep 说明必须保留的行为。字段和命名由大模型决定，不引入 DSL。
    """

    instruction: str
    target: str
    action: Action = Action.REPLACE
    kind: Kind = Kind.CODE
    context: str = ""
    keep: tuple[str, ...] = ()
    # 旧文件正文，由适配器读取后填入，用于生成 diff；不由大模型提供。
    original: str | None = None


@dataclass(frozen=True)
class Artifact:
    """适配器包装好的产物：目标 + 正文 + 宿主计算的哈希。

    artifact_id 与 content_hash 由宿主生成；模型输出里的同名字段一律忽略。
    """

    target: str
    body: str
    kind: Kind
    action: Action
    content_hash: str
    raw_response: str = ""
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def new_content(self) -> str:
        """写盘时的最终内容（统一以换行结尾）。"""
        return self.body if self.body.endswith("\n") else self.body + "\n"


def content_hash(target: str, body: str) -> str:
    """审批令牌：绑定目标路径与正文本身。

    正文或目标变了，哈希就变，旧确认自动失效。
    """
    h = hashlib.sha256()
    h.update(target.encode("utf-8"))
    h.update(b"\x00")
    h.update(body.encode("utf-8"))
    return h.hexdigest()[:16]


def strip_code_fence(text: str) -> str:
    """去掉模型输出的外层代码围栏；围栏只是展示，不是正文。

    处理两种形态：
    1. 完整的一对围栏（首尾都有），取中间内容；
    2. 只有一侧的**孤立围栏行**——实测 Mercury 2.5 有 5/8 次在末尾多吐一个
       关闭标记而没有开头的围栏，这不是语义错误，是纯机械问题。正文里的围栏
       一律不动，只清掉首尾的围栏行。

    这一步必须是确定性的：机械问题用规则修，不该花一次大模型往返。
    """
    lines = text.strip().split("\n")
    while lines and _FENCE_LINE_RE.match(lines[0]):
        lines.pop(0)
    while lines and _FENCE_LINE_RE.match(lines[-1]):
        lines.pop()
    stripped = "\n".join(lines).strip()
    # 清掉首尾围栏行之后还可能剩下一对（内容里带说明时），再按成对规则取一次。
    m = CODE_FENCE_RE.match(stripped)
    return m.group(1) if m else stripped


def normalize_body(text: str) -> str:
    """统一换行、去掉首尾空白行，避免同一内容产生不同哈希。"""
    body = strip_code_fence(text).replace("\r\n", "\n").replace("\r", "\n")
    return body.strip("\n")


def make_artifact(
    *,
    target: str,
    raw_response: str,
    kind: Kind,
    action: Action,
    notes: tuple[str, ...] = (),
) -> Artifact:
    """把模型原始输出变成可信产物；哈希在这里由宿主计算。"""
    body = normalize_body(raw_response)
    return Artifact(
        target=target,
        body=body,
        kind=kind,
        action=action,
        content_hash=content_hash(target, body),
        raw_response=raw_response,
        notes=notes,
    )


def resolve_target(workspace: Path, target: str) -> Path:
    """把目标解析到工作区内；越界路径直接拒绝。"""
    if not target or target.strip() != target:
        raise ValueError(f"目标路径不合法: {target!r}")
    candidate = (workspace / target).resolve()
    root = workspace.resolve()
    if candidate != root and root not in candidate.parents:
        raise ValueError(f"目标超出工作区: {target}")
    return candidate
