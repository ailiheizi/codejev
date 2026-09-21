"""真实本地模型的候选页选择探针；输出协议边界，不修复模型回复。"""

from __future__ import annotations

import json
import time
from pathlib import Path

from codejev.candidate import (
    CandidateError,
    CandidatePage,
    CodeCandidate,
    CodeTask,
    choose_candidate,
    materialize,
)
from codejev.model import MLXEngine, Stats


MODEL_PATH = (
    Path(__file__).resolve().parent.parent
    / "models"
    / "Qwen2.5-Coder-1.5B-Instruct-4bit"
)

PAGE = CandidatePage(
    (
        CodeCandidate(
            id="0",
            name="filter_and_project",
            purpose="过滤 active 为真的行，并按 id、name 投影，保持原顺序。",
            code=(
                "result = []\n"
                "for row in rows:\n"
                '    if row["active"]:\n'
                '        result.append({"id": row["id"], "name": row["name"]})\n'
                "return result"
            ),
        ),
        CodeCandidate(
            id="1",
            name="sort_rows",
            purpose="按 score 从高到低排序 rows，保留每行的完整内容。",
            code='return sorted(rows, key=lambda row: row["score"], reverse=True)',
        ),
        CodeCandidate(
            id="2",
            name="group_and_count",
            purpose="按 category 分组计数，返回 category 到数量的映射。",
            code=(
                "counts = {}\n"
                "for row in rows:\n"
                '    key = row["category"]\n'
                "    counts[key] = counts.get(key, 0) + 1\n"
                "return counts"
            ),
        ),
    )
)

TASK_FILE = Path(__file__).with_name("candidate_tasks.json")


def load_tasks() -> tuple[tuple[str, str, CodeTask, str], ...]:
    """从宿主保存的真实 CodeTask 输入文件加载任务。"""
    rows = json.loads(TASK_FILE.read_text(encoding="utf-8"))
    return tuple(
        (
            str(row["label"]),
            str(row["expected"]),
            CodeTask(
                operation=str(row["operation"]),
                target_language=str(row["target_language"]),
                requirements=tuple(str(value) for value in row["requirements"]),
                constraints=tuple(str(value) for value in row["constraints"]),
            ),
            str(row["source_request"]),
        )
        for row in rows
    )


def _print_result(
    label: str,
    expected: str,
    actual: str,
    passed: bool,
    elapsed: float,
    stats: Stats,
    raw: str,
) -> None:
    """打印单条探针结果。"""

    print(
        f"{label:24s} expected={expected:7s} actual={actual:9s} "
        f"{'PASS' if passed else 'FAIL':4s} elapsed={elapsed:.3f}s "
        f"load={stats.load_seconds:.3f}s prompt={stats.prompt_tokens} "
        f"generated={stats.generated_tokens} tokens "
        f"generate={stats.generate_seconds:.3f}s raw={raw!r}"
    )


def main() -> int:
    """运行五个已规范化任务并统计严格协议准确率。"""

    print(f"model={MODEL_PATH}")
    print(f"task_file={TASK_FILE}")
    print(f"candidates={', '.join(candidate.id for candidate in PAGE.candidates)}")
    print("label                    expected actual    result elapsed stats raw")

    engine = MLXEngine(str(MODEL_PATH))
    passed_count = 0
    tasks = load_tasks()
    for label, expected, task, source_request in tasks:
        print(f"source_request[{label}]={source_request}")
        start = time.perf_counter()
        stats = Stats()
        raw = "<no response>"
        actual = "ERROR"
        materialized = "<no materialized code>"
        try:
            choice, stats = choose_candidate(engine, task, PAGE, max_tokens=8)
            actual = choice.candidate_id if choice.candidate_id is not None else "NONE"
            raw = choice.raw_response
            if choice.candidate_id is not None:
                # 只允许宿主根据已验证的 choice 物化代码；模型原文不参与路径/代码决定。
                materialized = materialize(choice)
                if materialized != choice.candidate.code:
                    raise RuntimeError("宿主物化代码与候选页不一致")
            else:
                materialized = "<NO_MATCH: no materialization>"
        except CandidateError as exc:
            if exc.raw_response is not None:
                raw = exc.raw_response
            actual = "INVALID"
        except Exception as exc:  # pragma: no cover - exercised only by unavailable MLX/model
            raw = f"{type(exc).__name__}: {exc}"
            actual = "ERROR"
        elapsed = time.perf_counter() - start
        passed = actual == expected
        if passed:
            passed_count += 1
        _print_result(label, expected, actual, passed, elapsed, stats, raw)
        print(f"materialized[{label}]: {materialized}")

    total = len(tasks)
    print(f"accuracy={passed_count}/{total} ({passed_count / total:.1%})")
    print(
        "Strict accuracy counts only an exact candidate id or NONE; "
        "explanations and unknown ids are failures."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
