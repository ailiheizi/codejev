"""contracts.py 的测试：哈希、正文规范化、目标解析。

只测宿主侧的固定形状，不涉及模型或网络。
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from azfls.contracts import (
    Action,
    Artifact,
    Brief,
    Kind,
    content_hash,
    make_artifact,
    normalize_body,
    resolve_target,
    strip_code_fence,
)


def test_brief_defaults() -> None:
    """短指令的默认值：替换整个文件、代码正文、无原文。"""
    brief = Brief(instruction="加一行注释", target="app/util.py")
    assert brief.action is Action.REPLACE
    assert brief.kind is Kind.CODE
    assert brief.context == ""
    assert brief.keep == ()
    assert brief.original is None


def test_kind_fence() -> None:
    assert Kind.CODE.fence == "python"
    assert Kind.JSON.fence == "json"
    assert Kind.TEXT.fence == "text"


def test_content_hash_changes_with_target() -> None:
    """目标变了，哈希必须变，旧确认才自然失效。"""
    assert content_hash("a.py", "x = 1") != content_hash("b.py", "x = 1")


def test_content_hash_changes_with_body() -> None:
    assert content_hash("a.py", "x = 1") != content_hash("a.py", "x = 2")


def test_content_hash_is_deterministic() -> None:
    """同一目标和正文必须每次得到同一个哈希。"""
    first = content_hash("app/util.py", "def f():\n    return 1\n")
    second = content_hash("app/util.py", "def f():\n    return 1\n")
    assert first == second
    assert len(first) == 16
    assert all(c in "0123456789abcdef" for c in first)


def test_content_hash_does_not_hide_boundary() -> None:
    """目标和正文的拼接以 \\x00 分隔，不同切分不会撞哈希。"""
    assert content_hash("a", "bc") != content_hash("ab", "c")


def test_strip_code_fence_removes_python_fence() -> None:
    fenced = "```python\ndef f():\n    return 1\n```"
    assert strip_code_fence(fenced) == "def f():\n    return 1"


def test_strip_code_fence_keeps_plain_text() -> None:
    assert strip_code_fence("  x = 1\n") == "x = 1"


def test_normalize_body_crlf_and_blank_lines() -> None:
    """CRLF 统一成 LF，首尾空行去掉。"""
    assert normalize_body("\r\n\r\nx = 1\r\ny = 2\r\n\r\n") == "x = 1\ny = 2"


def test_normalize_body_strips_fence_first() -> None:
    assert normalize_body("```python\r\nx = 1\r\n```\r\n") == "x = 1"


def test_normalize_body_same_hash_for_crlf_and_lf() -> None:
    """换行风格不同不应产生两个哈希。"""
    assert normalize_body("a\r\nb") == normalize_body("a\nb")


def test_make_artifact_computes_hash_from_normalized_body() -> None:
    """哈希取自规范化后的正文，围栏与 CRLF 不影响它。"""
    artifact = make_artifact(
        target="a.py",
        raw_response="```python\r\nx = 1\r\n```\r\n",
        kind=Kind.CODE,
        action=Action.CREATE,
    )
    assert artifact.body == "x = 1"
    assert artifact.content_hash == content_hash("a.py", "x = 1")
    assert artifact.target == "a.py"
    assert artifact.kind is Kind.CODE
    assert artifact.action is Action.CREATE
    assert artifact.notes == ()


def test_make_artifact_keeps_raw_response() -> None:
    artifact = make_artifact(
        target="a.py",
        raw_response="```python\nx = 1\n```",
        kind=Kind.CODE,
        action=Action.REPLACE,
        notes=("提示",),
    )
    assert artifact.raw_response == "```python\nx = 1\n```"
    assert artifact.notes == ("提示",)


def test_artifact_new_content_ends_with_newline() -> None:
    artifact = Artifact(
        target="a.py",
        body="x = 1",
        kind=Kind.CODE,
        action=Action.CREATE,
        content_hash="0" * 16,
    )
    assert artifact.new_content == "x = 1\n"


def test_resolve_target_accepts_relative_path(tmp_path: Path) -> None:
    nested = tmp_path / "pkg"
    nested.mkdir()
    assert resolve_target(tmp_path, "pkg/mod.py") == (nested / "mod.py").resolve()


def test_resolve_target_accepts_plain_filename(tmp_path: Path) -> None:
    assert resolve_target(tmp_path, "new.py") == (tmp_path / "new.py").resolve()


def test_resolve_target_rejects_parent_escape(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        resolve_target(tmp_path, "../escape")


def test_resolve_target_rejects_deep_escape(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        resolve_target(tmp_path, "pkg/../../outside.py")


def test_resolve_target_rejects_absolute_path_outside_workspace(tmp_path: Path) -> None:
    """绝对路径即使真实存在，只要不在工作区内就拒绝。"""
    workspace = tmp_path / "ws"
    workspace.mkdir()
    outside = tmp_path / "outside.py"
    outside.write_text("x = 1\n", encoding="utf-8")
    assert resolve_target(workspace, "outside.py") == (workspace / "outside.py").resolve()
    with pytest.raises(ValueError):
        resolve_target(workspace, str(outside))


def test_resolve_target_rejects_empty_and_untrimmed(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        resolve_target(tmp_path, "")
    with pytest.raises(ValueError):
        resolve_target(tmp_path, " a.py")


def test_resolve_target_rejects_symlink_escape(tmp_path: Path) -> None:
    """工作区内的软链接指向外部时同样拒绝。"""
    outside = tmp_path.parent / f"{tmp_path.name}-outside"
    outside.mkdir()
    link = tmp_path / "link"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError:  # 平台不支持软链接时跳过
        pytest.skip("本平台不支持创建软链接")
    with pytest.raises(ValueError):
        resolve_target(tmp_path, "link/evil.py")


# --------------------------------------------------------------------------
# 孤立围栏：实测 Mercury 2.5 有 5/8 次在末尾多吐一个关闭标记（纯机械问题）
# --------------------------------------------------------------------------


def test_strip_code_fence_removes_stray_trailing_fence() -> None:
    """只有关闭标记、没有开头的围栏：必须清掉，否则正文语法不合法。"""
    body = 'def f():\n    return 1\n```'
    assert strip_code_fence(body) == "def f():\n    return 1"


def test_strip_code_fence_removes_stray_leading_fence() -> None:
    """只有开头围栏、没有关闭标记：同样清掉。"""
    body = '```python\ndef f():\n    return 1'
    assert strip_code_fence(body) == "def f():\n    return 1"


def test_strip_code_fence_removes_multiple_stray_lines() -> None:
    body = '```\ndef f():\n    return 1\n```\n```'
    assert strip_code_fence(body) == "def f():\n    return 1"


def test_strip_code_fence_keeps_fence_inside_body() -> None:
    """正文中间围栏不是包装，不能动（只清首尾）。"""
    body = 'x = """\n```\n"""\ny = 1'
    assert strip_code_fence(body) == body


def test_normalize_body_cleans_stray_fence_so_code_parses() -> None:
    """端到端：带孤立围栏的模型输出经 normalize_body 后仍是合法 Python。"""
    raw = 'def f():\n    return 1\n```'
    assert ast.parse(normalize_body(raw))
