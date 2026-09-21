"""cli.py 的测试：一条指令、一次展示、一次确认、一次写入。

全部离线：引擎换成 ScriptedEngine，不加载任何权重；写入只发生在 tmp_path 工作区。
select 的测试同样离线：脚本引擎回一个决策 JSON，宿主组装、确认、写入。
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from azfls import cli
from azfls.fixtures import USERS_MODULE
from azfls.model import ScriptedEngine, clean_body

BODY = "def active(users):\n    return [u for u in users if u['active']]"
BODY2 = "def active(users):\n    return [u for u in users if u['active'] and u['id']]"

# 固定任务形状：USERS_MODULE 里 active_users 的候选 id（f2=id、f3=name、c0=active）。
FILTER_INSTRUCTION = "只保留 active 为真的项，返回 id 和 name，保持原顺序。"
DECISION_JSON = '{"function": "fn0", "filter_field": "c0", "return_fields": ["f2", "f3"]}'


@pytest.fixture
def scripted(monkeypatch: pytest.MonkeyPatch):
    """把 cli 里的引擎工厂换成预设回复；返回引擎以便检查它收到的消息。"""

    def install(*bodies: str) -> ScriptedEngine:
        engine = ScriptedEngine(responses=list(bodies) or [BODY])
        monkeypatch.setattr(cli, "_make_engine", lambda args: engine)
        return engine

    return install


def test_ask_dry_run_on_new_target_shows_diff_without_writing(
    tmp_path: Path, scripted, capsys: pytest.CaptureFixture[str]
) -> None:
    """干跑：展示目标名和新增差异，退出 0，文件不落盘。"""
    scripted()
    code = cli.main(
        ["ask", "-i", "只返回 active 的项", "-t", "app/users.py", "-w", str(tmp_path), "--dry-run"]
    )

    out = capsys.readouterr().out
    assert code == 0
    assert "app/users.py" in out
    assert "--- /dev/null" in out  # 新文件：全部是新增
    assert "+def active(users):" in out
    assert "未写入" in out
    assert not (tmp_path / "app" / "users.py").exists()
    # 结论要短：一条摘要、可选提示、diff、展示块
    assert len(out.splitlines()) < 60


def test_ask_yes_writes_new_file(
    tmp_path: Path, scripted, capsys: pytest.CaptureFixture[str]
) -> None:
    """--yes 预先确认：文件按模型正文创建，以换行结尾。"""
    scripted()
    code = cli.main(
        ["ask", "-i", "写一个 active 过滤函数", "-t", "app/users.py", "-w", str(tmp_path), "--yes"]
    )

    out = capsys.readouterr().out
    assert code == 0
    written = tmp_path / "app" / "users.py"
    assert written.read_text(encoding="utf-8") == BODY + "\n"
    assert f"已写入 {written.resolve()}" in out


def test_ask_yes_replaces_existing_file(tmp_path: Path, scripted) -> None:
    """已有文件：original 取自磁盘，确认后整体替换为新正文。"""
    target = tmp_path / "app" / "users.py"
    target.parent.mkdir(parents=True)
    target.write_text("def active(users):\n    return users\n", encoding="utf-8")
    engine = scripted(BODY2)

    code = cli.main(
        ["ask", "-i", "只返回 active 的项", "-t", "app/users.py", "-w", str(tmp_path), "--yes"]
    )

    assert code == 0
    assert target.read_text(encoding="utf-8") == BODY2 + "\n"
    # 没给原文时用磁盘正文，且文件名出现在消息里而不是整份旧文件
    user = engine.calls[0][1]["content"]
    assert "def active(users):\n    return users" in user


def test_ask_refused_confirmation_writes_nothing(
    tmp_path: Path, scripted, monkeypatch: pytest.MonkeyPatch
) -> None:
    """用户回答 n：不写盘，退出码非 0。"""
    scripted()
    monkeypatch.setattr("builtins.input", lambda *a: "n")

    code = cli.main(
        ["ask", "-i", "只返回 active 的项", "-t", "app/users.py", "-w", str(tmp_path)]
    )

    assert code == 2
    assert not (tmp_path / "app" / "users.py").exists()


def test_ask_accepts_confirmation_from_stdin(
    tmp_path: Path, scripted, monkeypatch: pytest.MonkeyPatch
) -> None:
    """回答 y：确认生效，文件写入。"""
    scripted()
    monkeypatch.setattr("builtins.input", lambda *a: "y")

    code = cli.main(
        ["ask", "-i", "只返回 active 的项", "-t", "app/users.py", "-w", str(tmp_path)]
    )

    assert code == 0
    assert (tmp_path / "app" / "users.py").read_text(encoding="utf-8") == BODY + "\n"


def test_ask_eof_on_confirmation_writes_nothing(
    tmp_path: Path, scripted, monkeypatch: pytest.MonkeyPatch
) -> None:
    """非交互环境读不到回答（EOFError）：等于没人确认，不写盘。"""
    scripted()

    def eof(*args: object) -> str:
        raise EOFError

    monkeypatch.setattr("builtins.input", eof)

    code = cli.main(
        ["ask", "-i", "只返回 active 的项", "-t", "app/users.py", "-w", str(tmp_path)]
    )

    assert code == 2
    assert not (tmp_path / "app" / "users.py").exists()


def test_ask_rejects_escaping_target(
    tmp_path: Path, scripted, capsys: pytest.CaptureFixture[str]
) -> None:
    """越界目标被门控拒绝：非零退出，工作区外不出现任何文件。"""
    scripted()
    workspace = tmp_path / "ws"
    workspace.mkdir()

    code = cli.main(
        ["ask", "-i", "写点什么", "-t", "../evil.py", "-w", str(workspace), "--yes"]
    )

    out = capsys.readouterr().out
    assert code == 2
    assert "拒绝" in out
    assert not (tmp_path / "evil.py").exists()
    assert list(workspace.iterdir()) == []


def test_ask_context_file_reaches_the_prompt(
    tmp_path: Path, scripted, capsys: pytest.CaptureFixture[str]
) -> None:
    """--context-file 的内容作为原文进入提示词；未确认前不写盘。"""
    original = "def active(users):\n    return users\n"
    context_file = tmp_path / "old.py"
    context_file.write_text(original, encoding="utf-8")
    engine = scripted(BODY)

    code = cli.main(
        [
            "ask",
            "-i", "只返回 active 的项",
            "-t", "app/users.py",
            "-w", str(tmp_path),
            "--context-file", str(context_file),
            "--dry-run",
        ]
    )
    capsys.readouterr()

    assert code == 0
    user = engine.calls[0][1]["content"]
    assert original in user  # 原文来自 --context-file
    assert "只返回 active 的项" in user  # 指令本身照常进入消息
    assert not (tmp_path / "app" / "users.py").exists()


def test_ask_keep_rules_and_options_reach_the_prompt(tmp_path: Path, scripted) -> None:
    """--keep 可重复，且与 action/kind 一起进入模型消息。"""
    engine = scripted('{"a": 1}')

    code = cli.main(
        [
            "ask",
            "-i", "返回配置",
            "-t", "conf.json",
            "-w", str(tmp_path),
            "--action", "create",
            "--kind", "json",
            "--keep", "保持原顺序",
            "--keep", "不动字段名",
            "--dry-run",
        ]
    )

    assert code == 0
    user = engine.calls[0][1]["content"]
    assert "保持原顺序" in user
    assert "不动字段名" in user
    assert "create" in user
    assert not (tmp_path / "conf.json").exists()


def test_ask_model_failure_returns_1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """模型加载或推理失败：给出中文短提示，退出 1，不进入写入流程。"""

    class Broken:
        def generate(self, messages, max_tokens=512):
            raise RuntimeError("没有找到模型权重")

    monkeypatch.setattr(cli, "_make_engine", lambda args: Broken())

    code = cli.main(["ask", "-i", "写点什么", "-t", "a.py", "-w", str(tmp_path), "--yes"])

    out = capsys.readouterr().out
    assert code == 1
    assert "小模型" in out
    assert not (tmp_path / "a.py").exists()


def test_ask_max_tokens_is_passed_through(tmp_path: Path, scripted) -> None:
    """--max-tokens 传给引擎。"""
    engine = scripted("x = 1")
    seen: list[int] = []
    inner = engine.generate

    def generate(messages, max_tokens: int = 512):
        seen.append(max_tokens)
        return inner(messages, max_tokens=max_tokens)

    engine.generate = generate  # type: ignore[method-assign]
    cli.main(["ask", "-i", "写", "-t", "a.py", "-w", str(tmp_path), "--max-tokens", "64", "--dry-run"])

    assert seen == [64]


def test_check_reports_ready_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """模型目录存在：退出 0，输出一行中文状态。"""
    monkeypatch.setattr(cli, "DEFAULT_MODEL", str(tmp_path))

    code = cli.main(["check"])

    out = capsys.readouterr().out
    assert code == 0
    assert "模型目录：存在" in out
    assert "mlx_lm" in out


def test_check_returns_1_without_model_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """模型目录缺失：退出 1。"""
    monkeypatch.setattr(cli, "DEFAULT_MODEL", str(tmp_path / "不存在"))

    code = cli.main(["check"])

    assert code == 1
    assert "缺失" in capsys.readouterr().out


def test_check_does_not_load_the_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """自检必须快：不构造引擎。"""
    monkeypatch.setattr(cli, "DEFAULT_MODEL", str(tmp_path))
    monkeypatch.setattr(
        cli, "_make_engine", lambda args: pytest.fail("check 不应构造引擎")
    )

    assert cli.main(["check"]) == 0
    capsys.readouterr()


def test_build_parser_parses_choices() -> None:
    """argparse 接受 action/kind 的合法取值。"""
    parser = cli.build_parser()
    args = parser.parse_args(
        ["ask", "-i", "改", "-t", "app/users.py", "--action", "edit", "--kind", "text", "--keep", "a", "--keep", "b"]
    )

    assert args.action == "edit"
    assert args.kind == "text"
    assert args.keep == ["a", "b"]
    assert args.workspace == "."
    assert args.max_tokens == 512
    assert args.dry_run is False and args.yes is False


def test_build_parser_rejects_invalid_action() -> None:
    """非法 --action 直接报错退出（argparse 约定的 2）。"""
    with pytest.raises(SystemExit) as excinfo:
        cli.build_parser().parse_args(["ask", "-i", "改", "-t", "a.py", "--action", "重写"])

    assert excinfo.value.code == 2


def test_ask_requires_instruction_and_target() -> None:
    """缺少必填参数时报错退出。"""
    with pytest.raises(SystemExit) as excinfo:
        cli.build_parser().parse_args(["ask", "-t", "a.py"])

    assert excinfo.value.code == 2


def test_help_text_is_chinese(capsys: pytest.CaptureFixture[str]) -> None:
    """帮助与子命令说明保持中文。"""
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["--help"])
    out = capsys.readouterr().out
    assert "大模型发短指令" in out
    assert "ask" in out and "check" in out


def test_main_requires_subcommand() -> None:
    """必须给出子命令。"""
    with pytest.raises(SystemExit) as excinfo:
        cli.main([])

    assert excinfo.value.code == 2


# --------------------------------------------------------------------------
# select：选择式产出（宿主给候选、模型只选 id、宿主组装）
# --------------------------------------------------------------------------


def _write_users(tmp_path: Path, name: str = "app/users.py") -> Path:
    """把 USERS_MODULE 写成工作区里的一个真实文件，返回它的路径。"""
    target = tmp_path / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(USERS_MODULE, encoding="utf-8")
    return target


def test_select_dry_run_shows_decision_and_diff_without_writing(
    tmp_path: Path, scripted, capsys: pytest.CaptureFixture[str]
) -> None:
    """干跑：展示决策行与差异，退出 0，磁盘文件保持原样。"""
    scripted(DECISION_JSON)
    target = _write_users(tmp_path)

    code = cli.main(
        [
            "select",
            "-i", FILTER_INSTRUCTION,
            "-t", "app/users.py",
            "-w", str(tmp_path),
            "--function", "active_users",
            "--dry-run",
        ]
    )

    out = capsys.readouterr().out
    assert code == 0
    assert "filter=c0=active" in out  # 决策一行摘要
    assert "return_fields=f2=id、f3=name" in out
    assert "--- a/app/users.py" in out  # 对着磁盘原文的真实 diff
    assert '+        if user["active"]:' in out
    assert "未写入" in out
    assert target.read_text(encoding="utf-8") == USERS_MODULE
    assert len(out.splitlines()) < 60


def test_select_yes_rewrites_file_with_host_assembled_function(
    tmp_path: Path, scripted
) -> None:
    """--yes：宿主按决策重写函数；过滤条件与返回字段都真的生效。"""
    scripted(DECISION_JSON)
    target = _write_users(tmp_path)

    code = cli.main(
        [
            "select",
            "-i", FILTER_INSTRUCTION,
            "-t", "app/users.py",
            "-w", str(tmp_path),
            "--function", "active_users",
            "--yes",
        ]
    )

    assert code == 0
    text = target.read_text(encoding="utf-8")
    ast.parse(text)  # 仍是合法 Python
    namespace: dict[str, object] = {}
    exec(text, namespace)  # noqa: S102 - 测试专用：真跑一次重写后的函数
    rows = namespace["active_users"](namespace["USERS"])  # type: ignore[operator]
    # 只返回选定的两个字段，且只留下 active 为真的项，顺序不变
    assert rows == [{"id": 1, "name": "Ada"}]
    assert 'if user["active"]:' in text
    assert 'result.append({"id": user["id"], "name": user["name"]})' in text
    # 函数外的内容逐字保留
    assert text.startswith('"""用户列表示例：字典风格。"""')
    assert "USERS = [" in text


def test_select_accepts_terse_decision_json(tmp_path: Path, scripted) -> None:
    """模型回精简 JSON（现在的契约）：select 全流程照常工作。"""
    engine = scripted('{"f": "c0", "r": ["f2", "f3"]}')
    target = _write_users(tmp_path)

    code = cli.main(
        [
            "select",
            "-i", FILTER_INSTRUCTION,
            "-t", "app/users.py",
            "-w", str(tmp_path),
            "--function", "active_users",
            "--yes",
        ]
    )

    assert code == 0
    # 提示里给出的是精简契约：只列候选 id，键名是 f / r
    user = engine.calls[0][1]["content"]
    assert "（r 只能选这里）" in user
    assert "（f 只能选这里" in user
    text = target.read_text(encoding="utf-8")
    ast.parse(text)
    namespace: dict[str, object] = {}
    exec(text, namespace)  # noqa: S102 - 测试专用：真跑一次重写后的函数
    assert namespace["active_users"](namespace["USERS"]) == [{"id": 1, "name": "Ada"}]  # type: ignore[operator]


def test_clean_body_post_processing_is_unchanged() -> None:
    """WIN 1 的离线证据：引擎对正文的后处理一个字没变。

    直驱 generate_step 之后唯一变的是“怎么解码”，正文仍要过同一个 clean_body：
    去掉对话控制符、再去掉首尾空白。真跑模型的逐字对比在 bench/speedup_check.py。
    """
    assert clean_body('{"f": "c0", "r": ["f2"]}<|im_end|>\n') == '{"f": "c0", "r": ["f2"]}'
    assert clean_body("  <|im_start|>def f():\n    return 1\n") == "def f():\n    return 1"
    assert clean_body("正文<|endoftext|>") == "正文"
    assert clean_body("   \n  ") == ""


def test_select_missing_target_returns_1_and_creates_nothing(
    tmp_path: Path, scripted, capsys: pytest.CaptureFixture[str]
) -> None:
    """目标不存在：中文短提示、退出 1，不新建文件、不问模型。"""
    engine = scripted(DECISION_JSON)

    code = cli.main(
        ["select", "-i", FILTER_INSTRUCTION, "-t", "app/users.py", "-w", str(tmp_path), "--yes"]
    )

    out = capsys.readouterr().out
    assert code == 1
    assert "不存在" in out
    assert not (tmp_path / "app" / "users.py").exists()
    assert engine.calls == []  # 文件都没有，不该去问模型


def test_select_bogus_candidate_id_returns_3_and_keeps_file(
    tmp_path: Path, scripted, capsys: pytest.CaptureFixture[str]
) -> None:
    """模型自造候选 id：决策不合法，退出 3，文件一个字都不动。"""
    scripted('{"function": "fn0", "filter_field": "c0", "return_fields": ["f99"]}')
    target = _write_users(tmp_path)

    code = cli.main(
        [
            "select",
            "-i", FILTER_INSTRUCTION,
            "-t", "app/users.py",
            "-w", str(tmp_path),
            "--function", "active_users",
            "--yes",
        ]
    )

    out = capsys.readouterr().out
    assert code == 3
    assert "决策不合法" in out
    assert target.read_text(encoding="utf-8") == USERS_MODULE


def test_select_refused_confirmation_writes_nothing(
    tmp_path: Path, scripted, monkeypatch: pytest.MonkeyPatch
) -> None:
    """用户回答 n：门控拒绝，退出 2，文件保持原样。"""
    scripted(DECISION_JSON)
    target = _write_users(tmp_path)
    monkeypatch.setattr("builtins.input", lambda *a: "n")

    code = cli.main(
        [
            "select",
            "-i", FILTER_INSTRUCTION,
            "-t", "app/users.py",
            "-w", str(tmp_path),
            "--function", "active_users",
        ]
    )

    assert code == 2
    assert target.read_text(encoding="utf-8") == USERS_MODULE


def test_select_without_function_uses_first_public_function(
    tmp_path: Path, scripted
) -> None:
    """不给 --function 时由 decide.extract 选第一个公开函数，模型只看到候选 id。"""
    source = '"""两个函数。"""\n\n\ndef first(items):\n    out = []\n    for item in items:\n        out.append({"a": item["a"]})\n    return out\n\n\ndef second(items):\n    out = []\n    for item in items:\n        out.append({"b": item["b"]})\n    return out\n'
    target = tmp_path / "two.py"
    target.write_text(source, encoding="utf-8")
    engine = scripted('{"function": "fn0", "filter_field": "c0", "return_fields": ["f0"]}')

    code = cli.main(["select", "-i", "加过滤", "-t", "two.py", "-w", str(tmp_path), "--yes"])

    assert code == 0
    text = target.read_text(encoding="utf-8")
    ast.parse(text)
    assert 'if item["a"]:' in text  # 改的是第一个公开函数
    assert 'if item["b"]:' not in text  # 第二个函数没被动
    # 模型看到的是候选表，不是整份源码
    assert "def first" not in engine.calls[0][1]["content"]


def test_select_output_stays_short(
    tmp_path: Path, scripted, capsys: pytest.CaptureFixture[str]
) -> None:
    """输出要短：一条决策、几条提示、diff、展示块。"""
    scripted(DECISION_JSON)
    _write_users(tmp_path)

    code = cli.main(
        [
            "select",
            "-i", FILTER_INSTRUCTION,
            "-t", "app/users.py",
            "-w", str(tmp_path),
            "--function", "active_users",
            "--dry-run",
            "--no-timing",
        ]
    )

    out = capsys.readouterr().out
    assert code == 0
    assert "耗时" not in out  # --no-timing 保持安静
    assert len(out.splitlines()) < 60


def test_select_model_failure_returns_1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """模型加载或推理失败：中文短提示、退出 1、不写盘。"""
    _write_users(tmp_path)

    class Broken:
        def generate(self, messages, max_tokens=512):
            raise RuntimeError("没有找到模型权重")

    monkeypatch.setattr(cli, "_make_engine", lambda args: Broken())

    code = cli.main(
        [
            "select",
            "-i", FILTER_INSTRUCTION,
            "-t", "app/users.py",
            "-w", str(tmp_path),
            "--function", "active_users",
            "--yes",
        ]
    )

    out = capsys.readouterr().out
    assert code == 1
    assert "小模型" in out
    assert (tmp_path / "app" / "users.py").read_text(encoding="utf-8") == USERS_MODULE


def test_select_asks_engine_with_decision_max_tokens(tmp_path: Path, scripted) -> None:
    """决策路径自己调用 engine.generate，且用 DECISION_MAX_TOKENS。"""
    _write_users(tmp_path)
    engine = scripted(DECISION_JSON)
    seen: list[int] = []
    inner = engine.generate

    def generate(messages, max_tokens: int = 512):
        seen.append(max_tokens)
        return inner(messages, max_tokens=max_tokens)

    engine.generate = generate  # type: ignore[method-assign]
    cli.main(["select", "-i", "改", "-t", "app/users.py", "-w", str(tmp_path), "--dry-run"])

    assert seen == [128]


def test_build_parser_select_flags() -> None:
    """select 的默认值与选项：workspace 默认当前目录，function 默认 None。"""
    parser = cli.build_parser()
    args = parser.parse_args(["select", "-i", "改", "-t", "app/users.py"])

    assert args.workspace == "."
    assert args.function is None
    assert args.model is None
    assert args.dry_run is False and args.yes is False and args.no_timing is False


def test_build_parser_select_requires_instruction_and_target() -> None:
    """select 的 -i 与 -t 都是必填。"""
    with pytest.raises(SystemExit) as excinfo:
        cli.build_parser().parse_args(["select", "-t", "a.py"])

    assert excinfo.value.code == 2


def test_help_text_lists_select(capsys: pytest.CaptureFixture[str]) -> None:
    """帮助里列出 select 子命令。"""
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["--help"])
    out = capsys.readouterr().out
    assert "select" in out
