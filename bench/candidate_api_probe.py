"""用 OpenAI-compatible 执行器跑候选页对照。

不改本地 Qwen 探针：同一 CandidatePage、同一 candidate_tasks.json、同一严格解析器，
只替换执行器，用来区分"协议问题"和"本地小模型能力问题"。

配置从环境变量读，见 bench/provider_config.py：
    export AZFLS_API_BASE=... AZFLS_API_KEY=... AZFLS_MODEL=...
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from codejev.api_engine import OpenAICompatibleEngine
from codejev.candidate import CandidateError, choose_candidate, materialize
from codejev.model import Stats
from bench.candidate_probe import PAGE, load_tasks
from bench.provider_config import DEFAULT_MODEL, load_provider

TASK_FILE = Path(__file__).with_name("candidate_tasks.json")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="DeepSeek Flash 候选页对照探针")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args(argv)

    config = load_provider(args.model)
    engine = OpenAICompatibleEngine(config)
    tasks = load_tasks()
    passed = 0
    print(f"provider=env model={args.model}")
    print("comparison=API executor only; local Qwen baseline is unchanged")
    print(f"task_file={Path(__file__).with_name('candidate_tasks.json')}")
    print("label                    expected actual       result elapsed  raw")

    for label, expected, task, source_request in tasks:
        started = time.perf_counter()
        raw = "<no response>"
        actual = "ERROR"
        materialized = "<no materialized code>"
        stats = Stats()
        try:
            choice, stats = choose_candidate(engine, task, PAGE, max_tokens=16)
            raw = choice.raw_response
            actual = choice.candidate_id if choice.candidate_id is not None else "NONE"
            if choice.candidate_id is None:
                materialized = "<NO_MATCH: no materialization>"
            else:
                materialized = materialize(choice)
                if materialized != choice.candidate.code:
                    raise RuntimeError("host materialize mismatch")
        except CandidateError as exc:
            actual = "INVALID"
            if exc.raw_response is not None:
                raw = exc.raw_response
        except Exception as exc:  # probe must report a row, not hide a provider failure
            actual = f"ERROR:{type(exc).__name__}"
            raw = f"{type(exc).__name__}: {exc}"
        elapsed = time.perf_counter() - started
        ok = actual == expected
        passed += int(ok)
        print(
            f"{label:24s} expected={expected:7s} actual={actual:11s} "
            f"{'PASS' if ok else 'FAIL':4s} {elapsed:7.3f}s "
            f"prompt={stats.prompt_tokens} completion={stats.generated_tokens} raw={raw!r}"
        )
        print(f"source_request[{label}]={source_request}")
        print(f"materialized[{label}]: {materialized}")

    print(f"accuracy={passed}/{len(tasks)} ({passed / len(tasks):.1%})")
    print("Candidate ids and materialized code remain host-owned; API output is parsed strictly.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
