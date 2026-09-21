"""Jev（TypeSafe System One）执行器：宿主给候选，它只回一个选项。

与另外两个执行器的关系：
  MLXEngine / OpenAICompatibleEngine —— 走 chat-completions，**生成**正文
  TypesafeEngine                      —— 走 System One，**只做选择**，不吐字

Jev 的契约来自 jev-ultrafast 的公开源码（jev_ultrafast/model.py）：
    POST https://api.typesafe.ai/v1/systemone
    {"model": "jev-latest",
     "state": {"page": {...}, "elements": [...], "recent_actions": [...]},
     "questions": {"<名字>": {"type": "choice", "criteria": {id: 描述}, "instructions": {...}}}}
  返回 {"answers": {"<名字>": {"choice": id, "confidence": 0.99, "probabilities": {...}}}}

它是**选择器**而不是打分器，所以能表达"以上都不是"——这一点在实测里是关键：
LFM2.5 的批打分会永远选一个编号，而 Jev 能对当前候选页回 NONE。

密钥只从配置进请求头，不出现在异常、repr 或返回值里。
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable

from chooseonly.candidate import (
    CandidateChoice,
    CandidateError,
    CandidatePage,
    CodeTask,
    validate_page,
)
from chooseonly.model import Stats

ENDPOINT = "https://api.typesafe.ai/v1/systemone"
MODEL = "jev-latest"


class TypesafeError(RuntimeError):
    """Jev 调用或应答不可用时抛出的安全错误（不含密钥）。"""


@dataclass(frozen=True)
class TypesafeConfig:
    """连接设置。api_key 用 field(repr=False) 避免被打印出来。"""

    api_key: str = field(repr=False)
    model: str = MODEL
    endpoint: str = ENDPOINT
    timeout_seconds: float = 120.0


class TypesafeEngine:
    """把 CandidatePage 翻成 Jev 的 questions，再把答案收回成宿主候选。"""

    def __init__(
        self,
        config: TypesafeConfig,
        *,
        opener: Callable[..., Any] | None = None,
    ) -> None:
        self.config = config
        self._opener = opener if opener is not None else urllib.request.urlopen

    # -- 请求构造 ---------------------------------------------------------

    def build_request(self, task: CodeTask, page: CandidatePage) -> dict[str, Any]:
        """候选页 → Jev 请求体。候选内容全部来自宿主，模型不得增补。"""
        validate_page(page)
        criteria: dict[str, dict[str, str]] = {}
        for candidate in page.candidates:
            criteria[candidate.id] = {
                "name": candidate.name,
                "purpose": candidate.purpose,
                "code": candidate.code,
            }
        # 明确的弃权选项：没有它，打分式模型只能强行挑一个。
        criteria["NONE"] = {
            "name": "none",
            "purpose": "以上候选都无法满足这个需求",
            "code": "",
        }
        requirements = "；".join(task.requirements) or task.operation
        return {
            "model": self.config.model,
            "state": {
                "page": {
                    "url": "repo://host-candidate-page",
                    "title": f"候选代码页（{task.target_language}）",
                    "text": f"操作：{task.operation}",
                },
                "elements": [],
                "recent_actions": [],
            },
            "questions": {
                "candidate": {
                    "type": "choice",
                    "criteria": criteria,
                    "instructions": {
                        "goal": requirements,
                        "rules": [
                            "只从给出的候选里选一个；都不满足时选 NONE。",
                            "按需求与候选用途及代码的匹配程度判断，不要按编号顺序。",
                            *(
                                ["必须保留：" + "；".join(task.constraints)]
                                if task.constraints
                                else []
                            ),
                        ],
                    },
                }
            },
        }

    # -- 调用与解析 -------------------------------------------------------

    def _post(self, body: dict[str, Any]) -> tuple[dict[str, Any], float]:
        try:
            payload = json.dumps(body).encode("utf-8")
        except (TypeError, ValueError):
            raise TypesafeError("请求体无法序列化为 JSON") from None
        request = urllib.request.Request(
            self.config.endpoint,
            data=payload,
            headers={
                "Accept": "application/json",
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        started = time.perf_counter()
        try:
            with self._opener(request, timeout=self.config.timeout_seconds) as response:
                raw = response.read()
        except urllib.error.HTTPError as exc:
            status = exc.code if isinstance(exc.code, int) else None
            raise TypesafeError(
                "Jev HTTP 错误" + (f" {status}" if status else "")
            ) from None
        except urllib.error.URLError:
            raise TypesafeError("Jev 网络错误") from None
        except (OSError, TimeoutError):
            raise TypesafeError("Jev 网络错误") from None
        except Exception:
            raise TypesafeError("Jev 调用失败") from None
        elapsed = time.perf_counter() - started
        try:
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")
            return json.loads(raw), elapsed
        except (UnicodeError, TypeError, ValueError):
            raise TypesafeError("Jev 响应不是合法 JSON") from None

    def parse_answer(
        self, body: Any, page: CandidatePage, question: str = "candidate"
    ) -> CandidateChoice:
        """严格解析：选项必须来自宿主候选页，概率与置信度都必须自洽。"""
        if not isinstance(body, dict):
            raise TypesafeError("Jev 响应结构无效")
        answers = body.get("answers")
        if not isinstance(answers, dict):
            raise TypesafeError("Jev 响应缺少 answers")
        answer = answers.get(question)
        if not isinstance(answer, dict):
            raise TypesafeError(f"Jev 响应缺少 answers.{question}")

        choice = answer.get("choice")
        valid = {candidate.id for candidate in page.candidates} | {"NONE"}
        if not isinstance(choice, str) or choice not in valid:
            raise TypesafeError(f"Jev 返回了候选页外的选项：{choice!r}")

        confidence = answer.get("confidence")
        if confidence is not None:
            if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
                raise TypesafeError("confidence 必须是数字")
            if not 0.0 <= float(confidence) <= 1.0:
                raise TypesafeError("confidence 超出 [0,1]")

        probabilities = answer.get("probabilities")
        if probabilities is not None:
            if not isinstance(probabilities, dict):
                raise TypesafeError("probabilities 必须是对象")
            if set(probabilities) != valid:
                raise TypesafeError("probabilities 的键与候选页不一致")
            if abs(sum(float(v) for v in probabilities.values()) - 1.0) > 0.02:
                raise TypesafeError("probabilities 之和不为 1")

        if choice == "NONE":
            return CandidateChoice(None, None, json.dumps(answer, ensure_ascii=False))
        candidate = next(item for item in page.candidates if item.id == choice)
        return CandidateChoice(choice, candidate, json.dumps(answer, ensure_ascii=False))

    def choose(self, task: CodeTask, page: CandidatePage) -> tuple[CandidateChoice, Stats]:
        """一次选择。失败就是失败——不重试、不兜底、不自己造候选。"""
        body, elapsed = self._post(self.build_request(task, page))
        choice = self.parse_answer(body, page)
        usage = body.get("usage") if isinstance(body, dict) else None
        stats = Stats(
            load_seconds=0.0,
            prompt_tokens=int((usage or {}).get("input_tokens", 0) or 0),
            generated_tokens=int((usage or {}).get("output_tokens", 0) or 0),
            generate_seconds=elapsed,
        )
        return choice, stats
