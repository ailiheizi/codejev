"""确认门：确认什么就写什么。

规则见 docs/06-artifact-protocol.md：文件未经确认不写入；确认只绑定
当时的内容与基线，内容、目标或磁盘状态改变后旧确认立即失效。
哈希与令牌一律由宿主签发，模型输出里的“已批准”之类字段一律忽略。
不做多级审批：一次 propose、一次 confirm、一次 write。
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

from azfls.contracts import Artifact, content_hash, resolve_target


class GateError(Exception):
    """审批流程被拒绝时抛出。"""


@dataclass(frozen=True)
class Proposal:
    """待确认的一次写入：目标、基线，以及目标是否已存在。"""

    artifact: Artifact
    path: Path  # 工作区内的绝对路径
    existed: bool  # 目标文件已存在
    existing: str | None  # 磁盘上的当前正文；新文件为 None
    base_hash: str | None  # 基线哈希；新文件为 None


@dataclass(frozen=True)
class Approval:
    """宿主签发的确认结果；只有它能放行一次写入。"""

    target: str
    content_hash: str  # 本确认绑定的产物哈希
    token: str  # 宿主生成，同时绑定内容与基线
    acknowledged: bool  # 有效确认恒为 True


def _mint_token(proposal: Proposal) -> str:
    """令牌同时绑定产物哈希与基线；两者任一改变都会失配。"""
    return f"{proposal.artifact.content_hash}:{proposal.base_hash or 'new'}"


class Gate:
    """工作区内的薄确认门：propose 只读，apply 才写。"""

    def __init__(self, workspace: str | Path) -> None:
        self.workspace = Path(workspace).resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)

    def propose(self, artifact: Artifact) -> Proposal:
        """解析目标并读取基线，不写任何文件。"""
        path = resolve_target(self.workspace, artifact.target)
        existing = self._read(path)
        base_hash = None if existing is None else content_hash(artifact.target, existing)
        return Proposal(
            artifact=artifact,
            path=path,
            existed=existing is not None,
            existing=existing,
            base_hash=base_hash,
        )

    def approve(self, proposal: Proposal, confirmed: bool) -> Approval:
        """确认由宿主签发；未确认一律拒绝，不产生任何副作用。"""
        if confirmed is not True:
            raise GateError("未确认写入，已拒绝：内容与目标未经确认前不写盘。")
        return Approval(
            target=proposal.artifact.target,
            content_hash=proposal.artifact.content_hash,
            token=_mint_token(proposal),
            acknowledged=True,
        )

    def apply(self, approval: Approval, proposal: Proposal) -> Path:
        """逐项复核后原子写入；任一检查失败都不落盘。"""
        # 1. 确认必须由宿主签发
        if approval.acknowledged is not True:
            raise GateError("确认未生效：缺少宿主签发的确认。")
        # 2. 确认绑定的内容必须与当前产物一致
        if approval.content_hash != proposal.artifact.content_hash:
            raise GateError("确认已失效：内容已改变")
        # 3. 令牌同时覆盖内容与基线
        if approval.token != _mint_token(proposal):
            raise GateError("确认令牌不匹配")
        # 4. 目标重新解析，仍须在工作区内且与原目标一致
        try:
            path = resolve_target(self.workspace, proposal.artifact.target)
        except ValueError as exc:
            raise GateError(f"确认已失效：目标不合法（{exc}）") from exc
        if path != proposal.path:
            raise GateError(f"确认已失效：目标已改变（{path}）")
        # 5. 基线复核：磁盘必须在确认后保持原样
        current = self._read(path)
        current_hash = (
            None if current is None else content_hash(proposal.artifact.target, current)
        )
        if current_hash != proposal.base_hash:
            raise GateError("目标文件已在确认后发生变化，请重新确认")
        # 6. 通过全部检查后才写：同目录临时文件 + 原子替换
        return self._write(path, proposal.artifact.new_content)

    def _read(self, path: Path) -> str | None:
        """读取磁盘现有正文；目录不能作为文件目标。"""
        if path.is_dir():
            raise GateError(f"目标是目录，不能作为文件写入: {path}")
        if not path.exists():
            return None
        try:
            return path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            raise GateError(f"无法读取目标文件: {exc}") from exc

    def _write(self, path: Path, text: str) -> Path:
        """先写同目录临时文件再原子替换；失败不留半截文件。"""
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        )
        try:
            with tmp as fh:
                fh.write(text)
                fh.flush()
                os.fsync(fh.fileno())
            if path.exists():  # 替换已有文件时保留其权限位
                os.chmod(tmp.name, path.stat().st_mode & 0o777)
            os.replace(tmp.name, path)
        except OSError:
            Path(tmp.name).unlink(missing_ok=True)
            raise
        return path
