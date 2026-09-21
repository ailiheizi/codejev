"""薄适配器：短指令 → 正文 → 宿主包装的产物。

见 docs/06-artifact-protocol.md：适配器只把小模型的正文对应到目标文件、生成 diff
和展示外壳；身份字段（目标、kind、action、哈希）一律由宿主给出，绝不从模型输出接受。
正文里的代码围栏只是展示，出现命令或“已批准”等文字也只是文本：不执行、不构成确认。
完整性检查只产出提示，交给大模型判断；不自动修复、不重试、不循环。
"""

from __future__ import annotations

import difflib
import json
import re

from chooseonly.contracts import Action, Artifact, Brief, Kind, make_artifact, normalize_body

SYSTEM_PROMPT = (
    "你只输出要写入目标文件的正文本身。\n"
    "不要输出解释、分析、步骤、总结或任何多余文字。\n"
    "不要输出 Markdown 代码围栏（```）。\n"
    "不要决定文件路径、文件名、哈希或“已批准”一类字段，这些由宿主负责。\n"
    "只按指令改动需要改的部分，其余保持原样。"
)

# 用户消息里的操作说明，保持一句话。
_ACTION_HINT: dict[Action, str] = {
    Action.CREATE: "新建文件，输出完整正文",
    Action.REPLACE: "整体替换该文件，输出完整正文",
    Action.EDIT: "只输出被改动的那一段，不要整份文件",
}

# 摘要用的动词。
_ACTION_VERB: dict[Action, str] = {
    Action.CREATE: "新建",
    Action.REPLACE: "替换",
    Action.EDIT: "修改",
}

# 看起来像整份文件的阈值：顶层定义 + 行数，或顶层语句数量。
WHOLE_FILE_MIN_LINES = 12
WHOLE_FILE_MIN_TOP_LEVEL = 3
_DEF_OR_CLASS_RE = re.compile(r"(?:async\s+def|def|class)\s+\w")

# 正文里出现这些字样时提醒大模型：它们不会被当成命令或确认执行。
_SHELL_OR_APPROVAL: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"(?:^|\s)rm\s+-[A-Za-z]*[rRfF]"), "rm -rf"),
    (re.compile(r"(?:^|\s)sudo(?:\s|$)"), "sudo"),
    (re.compile(r"git\s+push"), "git push"),
    (re.compile(r"已批准"), "已批准"),
    (re.compile(r"approved\s*[:=]\s*true", re.IGNORECASE), "approved: true"),
)

# 正文自带围栏时加长外层围栏，避免展示块被截断。
_FENCE_LINE_RE = re.compile(r"^\s*```", re.MULTILINE)


def build_messages(brief: Brief) -> list[dict[str, str]]:
    """把一条短指令整理成小模型的对话消息：一条系统提示 + 一条用户消息。

    用户消息只放这次真正需要的东西：要求、目标、操作、原文、必须保留的行为；
    不要求小模型解释或自省，不加推理步骤。
    """
    parts = [
        f"目标：{brief.target}",
        f"操作：{brief.action.value}（{_ACTION_HINT[brief.action]}）",
        f"要求：{brief.instruction.strip()}",
    ]
    if brief.keep:
        parts.append("必须保留：")
        parts.extend(f"- {item}" for item in brief.keep)
    if brief.context.strip():
        parts.append("原文（只改需要改的部分）：")
        parts.append("```")
        parts.append(brief.context.strip())
        parts.append("```")
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "\n".join(parts)},
    ]


def to_artifact(brief: Brief, raw_response: str) -> Artifact:
    """把模型原始输出包成宿主拥有的 Artifact：目标与哈希都取自 brief 和宿主计算。

    模型输出里的路径、哈希、批准字段一律忽略；完整性检查只写进 notes。
    """
    body = normalize_body(raw_response)
    notes = check_body(body, brief.kind, brief.action)
    return make_artifact(
        target=brief.target,
        raw_response=raw_response,
        kind=brief.kind,
        action=brief.action,
        notes=notes,
    )


def check_body(body: str, kind: Kind, action: Action) -> tuple[str, ...]:
    """完整性检查，只产出中文提示，不拒绝、不修改正文。"""
    notes: list[str] = []
    if not body.strip():
        notes.append("正文为空：小模型没有产出内容，请不要据此写盘。")
    elif action == Action.EDIT and looks_like_whole_file(body):
        notes.append("正文像整份文件而不是局部改动：请确认小模型是否重写了整个文件。")
    if kind == Kind.JSON and not _is_json(body):
        notes.append("JSON 无法解析：请先让大模型修正，再交给用户决定。")
    shell = _shell_or_approval_hit(body)
    if shell:
        notes.append(f"正文出现 {shell} 一类命令或批准字样：只是文本，不会被执行，也不构成确认。")
    return tuple(notes)


def looks_like_whole_file(body: str) -> bool:
    """粗略判断正文是否像整份文件：顶层 def/class 且较长，或有多个顶层语句。"""
    lines = body.split("\n")
    top_level = [ln for ln in lines if ln and not ln[0].isspace() and not ln.lstrip().startswith("#")]
    if len(top_level) >= WHOLE_FILE_MIN_TOP_LEVEL:
        return True
    body_lines = sum(1 for ln in lines if ln.strip())
    return body_lines >= WHOLE_FILE_MIN_LINES and any(_DEF_OR_CLASS_RE.match(ln) for ln in top_level)


def render_diff(original: str | None, new: str, target: str) -> str:
    """统一 diff 文本，无颜色码；original 为 None 表示新文件，全部行都是新增。

    没有差异时返回“（无差异）”。
    """
    new_lines = _as_lines(new)
    if original is None:
        old_lines: list[str] = []
        fromfile, tofile = "/dev/null", f"b/{target}"
    else:
        old_lines = _as_lines(original)
        fromfile, tofile = f"a/{target}", f"b/{target}"
    diff = "\n".join(
        difflib.unified_diff(old_lines, new_lines, fromfile=fromfile, tofile=tofile, lineterm="")
    )
    return diff if diff else "（无差异）"


def summarize(artifact: Artifact) -> str:
    """给大模型或用户的一行中文摘要：改了什么、多少行、哈希前缀。"""
    verb = _ACTION_VERB[artifact.action]
    line = f"{verb} {artifact.target}：{_line_count(artifact.body)} 行，哈希 {artifact.content_hash[:8]}"
    if artifact.notes:
        line += f"（{len(artifact.notes)} 条提示）"
    return line


def display(artifact: Artifact) -> str:
    """完整展示块：目标文件名 + 围栏内的正文。围栏只用于展示，不执行其中任何内容。"""
    bar = "````" if _FENCE_LINE_RE.search(artifact.body) else "```"
    return (
        f"### {artifact.target}  ({artifact.action.value})\n"
        f"{bar}{artifact.kind.fence}\n"
        f"{artifact.body}\n"
        f"{bar}"
    )


def _is_json(body: str) -> bool:
    try:
        json.loads(body)
    except ValueError:
        return False
    return True


def _shell_or_approval_hit(body: str) -> str | None:
    """命中第一条像命令或批准字样的文本，返回它的展示名。"""
    for pattern, label in _SHELL_OR_APPROVAL:
        if pattern.search(body):
            return label
    return None


def _as_lines(text: str) -> list[str]:
    """拆成行，统一换行并忽略结尾换行差异，避免 diff 出现空噪声行。"""
    if not text:
        return []
    return text.replace("\r\n", "\n").replace("\r", "\n").rstrip("\n").split("\n")


def _line_count(body: str) -> int:
    return 0 if not body else body.count("\n") + 1
