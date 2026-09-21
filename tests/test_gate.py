"""确认门的测试：只读 propose、宿主签发确认、复核后原子写入。"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from codejev.contracts import Action, Kind, content_hash, make_artifact
from codejev.gate import Approval, Gate, GateError, Proposal


def make_proposal_artifact(target: str, body: str):
    """按宿主规则包装一份产物；哈希由 make_artifact 计算。"""
    return make_artifact(
        target=target,
        raw_response=body,
        kind=Kind.CODE,
        action=Action.REPLACE,
    )


def test_propose_new_file(tmp_path: Path) -> None:
    """新文件：existed 为 False，base_hash 与 existing 均为 None。"""
    gate = Gate(tmp_path)
    proposal = gate.propose(make_proposal_artifact("new.py", "print(1)"))

    assert proposal.existed is False
    assert proposal.existing is None
    assert proposal.base_hash is None
    assert proposal.path == (tmp_path / "new.py").resolve()
    assert not proposal.path.exists()  # propose 不写盘


def test_propose_existing_file(tmp_path: Path) -> None:
    """已有文件：读到磁盘正文并据此计算基线哈希。"""
    path = tmp_path / "old.py"
    path.write_text("print('old')\n", encoding="utf-8")

    gate = Gate(tmp_path)
    proposal = gate.propose(make_proposal_artifact("old.py", "print('new')"))

    assert proposal.existed is True
    assert proposal.existing == "print('old')\n"
    assert proposal.base_hash == content_hash("old.py", "print('old')\n")


def test_propose_rejects_escaping_target(tmp_path: Path) -> None:
    """越界目标（../evil.py）直接拒绝，不解析到工作区外。"""
    gate = Gate(tmp_path)
    with pytest.raises((ValueError, GateError)):
        gate.propose(make_proposal_artifact("../evil.py", "print('evil')"))

    assert not (tmp_path.parent / "evil.py").exists()


def test_propose_rejects_directory_target(tmp_path: Path) -> None:
    """目标已是目录时不能当作文件写入。"""
    (tmp_path / "pkg").mkdir()
    gate = Gate(tmp_path)

    with pytest.raises(GateError):
        gate.propose(make_proposal_artifact("pkg", "x = 1"))


def test_approve_requires_confirmation(tmp_path: Path) -> None:
    """未确认一律拒绝，且不产生任何文件。"""
    gate = Gate(tmp_path)
    proposal = gate.propose(make_proposal_artifact("a.py", "x = 1"))

    with pytest.raises(GateError) as excinfo:
        gate.approve(proposal, confirmed=False)

    assert "确认" in str(excinfo.value)
    assert list(tmp_path.iterdir()) == []


def test_approve_confirmed_mints_approval(tmp_path: Path) -> None:
    """确认由宿主签发：acknowledged 为 True，令牌绑定内容与基线。"""
    gate = Gate(tmp_path)
    proposal = gate.propose(make_proposal_artifact("a.py", "x = 1"))
    approval = gate.approve(proposal, confirmed=True)

    assert approval.acknowledged is True
    assert approval.target == "a.py"
    assert approval.content_hash == proposal.artifact.content_hash
    assert approval.token == f"{proposal.artifact.content_hash}:new"


def test_approve_token_binds_existing_baseline(tmp_path: Path) -> None:
    """已有文件时令牌带回基线哈希，基线变了令牌就不再匹配。"""
    (tmp_path / "a.py").write_text("x = 0\n", encoding="utf-8")
    gate = Gate(tmp_path)
    proposal = gate.propose(make_proposal_artifact("a.py", "x = 1"))
    approval = gate.approve(proposal, confirmed=True)

    assert approval.token == f"{proposal.artifact.content_hash}:{proposal.base_hash}"


def test_apply_writes_content_with_trailing_newline(tmp_path: Path) -> None:
    """写入内容与产物一致，并以换行结尾。"""
    gate = Gate(tmp_path)
    proposal = gate.propose(make_proposal_artifact("a.py", "x = 1"))
    approval = gate.approve(proposal, confirmed=True)

    written = gate.apply(approval, proposal)

    assert written == (tmp_path / "a.py").resolve()
    assert written.read_text(encoding="utf-8") == "x = 1\n"


def test_apply_replaces_existing_file(tmp_path: Path) -> None:
    """已有文件经确认后被整体替换，不留临时文件。"""
    path = tmp_path / "a.py"
    path.write_text("x = 0\n", encoding="utf-8")
    gate = Gate(tmp_path)
    proposal = gate.propose(make_proposal_artifact("a.py", "x = 1"))
    approval = gate.approve(proposal, confirmed=True)

    gate.apply(approval, proposal)

    assert path.read_text(encoding="utf-8") == "x = 1\n"
    assert [p.name for p in tmp_path.iterdir()] == ["a.py"]


def test_apply_creates_parent_directories(tmp_path: Path) -> None:
    """嵌套目标自动建父目录。"""
    gate = Gate(tmp_path)
    artifact = make_proposal_artifact("pkg/sub/mod.py", "def f():\n    return 1")
    proposal = gate.propose(artifact)
    approval = gate.approve(proposal, confirmed=True)

    written = gate.apply(approval, proposal)

    assert written == (tmp_path / "pkg" / "sub" / "mod.py").resolve()
    assert written.read_text(encoding="utf-8") == "def f():\n    return 1\n"


def test_apply_rejects_tampered_content_hash(tmp_path: Path) -> None:
    """确认绑定的内容哈希与产物不符时拒绝写入。"""
    gate = Gate(tmp_path)
    proposal = gate.propose(make_proposal_artifact("a.py", "x = 1"))
    approval = gate.approve(proposal, confirmed=True)
    forged_hash = content_hash("a.py", "x = 999")
    assert forged_hash != proposal.artifact.content_hash
    tampered = replace(approval, content_hash=forged_hash)

    with pytest.raises(GateError) as excinfo:
        gate.apply(tampered, proposal)

    assert "内容已改变" in str(excinfo.value)
    assert not (tmp_path / "a.py").exists()


def test_apply_rejects_stale_baseline(tmp_path: Path) -> None:
    """确认后磁盘文件被改动：旧确认失效，且不覆盖新内容。"""
    path = tmp_path / "a.py"
    path.write_text("x = 0\n", encoding="utf-8")
    gate = Gate(tmp_path)
    proposal = gate.propose(make_proposal_artifact("a.py", "x = 1"))
    approval = gate.approve(proposal, confirmed=True)

    path.write_text("x = 99\n", encoding="utf-8")  # 确认之后被人改动

    with pytest.raises(GateError) as excinfo:
        gate.apply(approval, proposal)

    assert "已在确认后发生变化" in str(excinfo.value)
    assert path.read_text(encoding="utf-8") == "x = 99\n"


def test_apply_rejects_file_not_created_when_proposed_as_new(tmp_path: Path) -> None:
    """propose 时视为新文件，之后凭空出现的文件同样使确认失效。"""
    gate = Gate(tmp_path)
    proposal = gate.propose(make_proposal_artifact("a.py", "x = 1"))
    approval = gate.approve(proposal, confirmed=True)

    (tmp_path / "a.py").write_text("x = 0\n", encoding="utf-8")

    with pytest.raises(GateError):
        gate.apply(approval, proposal)

    assert (tmp_path / "a.py").read_text(encoding="utf-8") == "x = 0\n"


def test_apply_rejects_unacknowledged_approval(tmp_path: Path) -> None:
    """宿主未签发的确认（acknowledged 为 False）不能放行。"""
    gate = Gate(tmp_path)
    proposal = gate.propose(make_proposal_artifact("a.py", "x = 1"))
    approval = gate.approve(proposal, confirmed=True)
    forged = replace(approval, acknowledged=False)

    with pytest.raises(GateError) as excinfo:
        gate.apply(forged, proposal)

    assert "确认" in str(excinfo.value)
    assert not (tmp_path / "a.py").exists()


def test_apply_rejects_token_for_other_content(tmp_path: Path) -> None:
    """为别的内容签发的令牌在这里不通过。"""
    gate = Gate(tmp_path)
    proposal = gate.propose(make_proposal_artifact("a.py", "x = 1"))
    approval = gate.approve(proposal, confirmed=True)

    other = gate.propose(make_proposal_artifact("a.py", "x = 2"))
    other_token = gate.approve(other, confirmed=True).token
    assert other_token != approval.token

    with pytest.raises(GateError) as excinfo:
        gate.apply(replace(approval, token=other_token), proposal)

    assert "令牌不匹配" in str(excinfo.value)
    assert not (tmp_path / "a.py").exists()


def test_apply_rejects_foreign_proposal(tmp_path: Path) -> None:
    """确认与产物必须成对；拿另一份产物来写入直接拒绝。"""
    gate = Gate(tmp_path)
    proposal = gate.propose(make_proposal_artifact("a.py", "x = 1"))
    approval = gate.approve(proposal, confirmed=True)
    other = gate.propose(make_proposal_artifact("a.py", "x = 2"))

    with pytest.raises(GateError):
        gate.apply(approval, other)

    assert not (tmp_path / "a.py").exists()


def test_gate_creates_workspace(tmp_path: Path) -> None:
    """工作区不存在时自动创建。"""
    workspace = tmp_path / "ws"
    gate = Gate(workspace)

    assert gate.workspace == workspace.resolve()
    assert workspace.is_dir()


def test_approval_dataclasses_are_frozen(tmp_path: Path) -> None:
    """确认对象不可被事后改写。"""
    gate = Gate(tmp_path)
    proposal: Proposal = gate.propose(make_proposal_artifact("a.py", "x = 1"))
    approval: Approval = gate.approve(proposal, confirmed=True)

    with pytest.raises(Exception):
        approval.acknowledged = False  # type: ignore[misc]
    with pytest.raises(Exception):
        approval.token = "forged"  # type: ignore[misc]
