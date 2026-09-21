"""候选页协议：大模型定任务，小模型只选宿主候选。"""

from __future__ import annotations

import re
from dataclasses import dataclass

from azfls.model import Engine, Stats


class CandidateError(Exception):
    """候选页或模型选择无效。"""

    def __init__(self, message: str, *, raw_response: str | None = None) -> None:
        super().__init__(message)
        self.raw_response = raw_response


@dataclass(frozen=True)
class CodeTask:
    """大模型已经规范化的短代码任务。"""

    operation: str
    target_language: str
    requirements: tuple[str, ...]
    constraints: tuple[str, ...] = ()


@dataclass(frozen=True)
class CodeCandidate:
    """宿主拥有的稳定代码候选。"""

    id: str
    name: str
    purpose: str
    code: str
    language: str = "python"


@dataclass(frozen=True)
class CandidatePage:
    """宿主展示给小模型的候选代码页。"""

    candidates: tuple[CodeCandidate, ...]
    language: str = "python"


@dataclass(frozen=True)
class CandidateChoice:
    """小模型选择的宿主候选；None 表示没有匹配。"""

    candidate_id: str | None
    candidate: CodeCandidate | None
    raw_response: str


_SYSTEM_PROMPT = (
    "Output ONLY one candidate id from the page or NONE. "
    "Do not output an explanation, code, path, approval, new id, or reasoning. "
    "Do not plan, choose an operation, choose a path, approve, or create a candidate. "
    "Match only the normalized task against the candidates listed in the page."
)



def _error(message: str, raw_response: str | None = None) -> CandidateError:
    """构造带原始回复的协议错误。"""

    return CandidateError(message, raw_response=raw_response)



def validate_page(page: CandidatePage) -> None:
    """校验候选页的宿主边界。"""

    if not isinstance(page, CandidatePage):
        raise CandidateError("候选页类型无效")
    if not isinstance(page.language, str) or not page.language.strip():
        raise CandidateError("页面语言不能为空")

    seen: set[str] = set()
    for candidate in page.candidates:
        if not isinstance(candidate, CodeCandidate):
            raise CandidateError("候选类型无效")
        if not isinstance(candidate.id, str) or not candidate.id.strip():
            raise CandidateError("候选 id 不能为空")
        if candidate.id in seen:
            raise CandidateError(f"候选 id 重复：{candidate.id!r}")
        seen.add(candidate.id)
        if not isinstance(candidate.code, str) or not candidate.code.strip():
            raise CandidateError(f"候选代码不能为空：{candidate.id!r}")
        if candidate.language != page.language:
            raise CandidateError(
                f"候选语言与页面不一致：{candidate.id!r}"
            )



def _bullet_lines(values: tuple[str, ...]) -> list[str]:
    """把短要求按固定顺序排成提示行。"""

    return [f"- {value}" for value in values] or ["- (none)"]



def build_selection_messages(task: CodeTask, page: CandidatePage) -> list[dict[str, str]]:
    """构造固定的两段选择提示。"""

    validate_page(page)
    lines = [
        "TASK",
        f"operation: {task.operation}",
        f"language: {task.target_language}",
        "requirements:",
        *_bullet_lines(task.requirements),
        "constraints:",
        *_bullet_lines(task.constraints),
        "CANDIDATES",
    ]
    valid_ids = ", ".join(candidate.id for candidate in page.candidates)
    lines.extend(
        (
            f"valid outputs: {valid_ids}, NONE",
            "output exactly one valid output token; never write id=, candidate_id=, code, or explanation",
        )
    )
    for candidate in page.candidates:
        lines.extend(
            (
                f"candidate {candidate.id}",
                f"name: {candidate.name}",
                f"purpose: {candidate.purpose}",
                "code:",
                candidate.code,
            )
        )
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": "\n".join(lines)},
    ]



def _unwrap_scalar(text: str) -> str:
    """只去掉完整包住单行标量的 markdown 围栏。"""

    if not (text.startswith("```") or text.endswith("```")):
        return text
    match = re.fullmatch(r"```[^\r\n]*\r?\n([^\r\n]*)\r?\n```", text)
    if match is None:
        raise CandidateError("回复不是完整的单行 fenced scalar")
    return match.group(1).strip()



def parse_choice(raw_response: str, page: CandidatePage) -> CandidateChoice:
    """严格解析单行候选 id 或 NO_MATCH。"""

    validate_page(page)
    if not isinstance(raw_response, str):
        raise _error("模型回复必须是字符串")

    text = raw_response.strip()
    if not text:
        raise _error("模型回复为空", raw_response)
    try:
        text = _unwrap_scalar(text)
    except CandidateError as exc:
        raise _error(str(exc), raw_response) from None
    if not text or "\n" in text or "\r" in text:
        raise _error("模型回复必须只有一行候选 id", raw_response)

    token = text.strip()
    if not token:
        raise _error("模型回复为空", raw_response)
    if token.upper() in {"NONE", "NO_MATCH"}:
        return CandidateChoice(None, None, raw_response)

    candidate = next((item for item in page.candidates if item.id == token), None)
    if candidate is None:
        raise _error(f"未知的候选 id：{token!r}", raw_response)
    return CandidateChoice(candidate.id, candidate, raw_response)



def choose_candidate(
    engine: Engine,
    task: CodeTask,
    page: CandidatePage,
    max_tokens: int = 8,
) -> tuple[CandidateChoice, Stats]:
    """请求小模型选择；不重试、不修复、不执行文件操作。"""

    validate_page(page)
    messages = build_selection_messages(task, page)
    raw_response, stats = engine.generate(messages, max_tokens=max_tokens)
    try:
        choice = parse_choice(raw_response, page)
    except CandidateError as exc:
        if exc.raw_response is None:
            exc.raw_response = raw_response
        raise
    return choice, stats



def materialize(choice: CandidateChoice) -> str:
    """返回宿主候选代码；没有匹配时拒绝物化。"""

    if choice.candidate_id is None:
        raise CandidateError("NO_MATCH 没有可物化的候选")
    if choice.candidate is None:
        raise CandidateError("选择缺少宿主候选")
    return choice.candidate.code
