"""测这条回路：关键词库（语句形式）+ 变量 → Jev 选 → 宿主拼接 → 大模型检查。

回路（作者描述）：
  大模型提出需求
    → 给出目标语言的关键词库（该语言的语句形式）
    → 给出需求需要的变量（关键词库里没有的那些）
  → 候选池 = 语句形式 ∪ 变量
  → Jev 只做选择：变量填进哪个槽位
  → 宿主按形式拼接出代码
  → 交回大模型检查

本脚本测两件事：
  A. 正常回路：Jev 在**一次请求**里回答多个槽位问题（Jev 的 questions 支持并列多问），
     宿主拼接后**真的 exec 并调用**，看行为对不对。
  B. 缺口回路：需求需要的东西**不在池里**时，Jev 是否会回 NONE，
     而不是硬凑一个——这是这条回路能不能安全用的关键。

判据全部是运行时行为，不看代码像不像。
"""

from __future__ import annotations

import ast
import json
import os
import time
import urllib.request

ENDPOINT = "https://api.typesafe.ai/v1/systemone"
MODEL = "jev-latest"

# ---- 大模型给出的：目标语言的关键词库（语句形式）----
FORMS = {
    "def": "def {name}({param}):",
    "init": "{acc} = []",
    "loop": "for {item} in {param}:",
    "filter": 'if {item}["{cond}"]:',
    "append": "{acc}.append({{{fields}}})",
    "return": "return {acc}",
}

# ---- 大模型给出的：需求需要的变量（语句形式里没有的）----
VARIABLES = ["rows", "active", "deleted", "id", "name", "email"]

REQUIREMENT = "build active_users(rows): keep rows where active is true, return id and name"


def call(questions: dict, retries: int = 3) -> tuple[dict, float]:
    key = os.environ.get("TYPESAFE_API_KEY", "")
    if not key:
        raise RuntimeError("需要 TYPESAFE_API_KEY")
    body = {
        "model": MODEL,
        "state": {
            "page": {"url": "codejev://form-pool", "title": "keyword library + variables", "text": REQUIREMENT},
            "elements": [],
            "recent_actions": [],
        },
        "questions": questions,
    }
    request = urllib.request.Request(
        ENDPOINT, data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        method="POST",
    )
    last: Exception | None = None
    for attempt in range(retries):
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                return json.loads(response.read().decode()), time.perf_counter() - started
        except Exception as exc:  # noqa: BLE001
            last = exc
            if attempt < retries - 1:
                time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"重试后仍失败: {last}")


def build_questions() -> dict:
    """把槽位问题一次问完：Jev 支持一个请求里并列多个问题。"""
    var_criteria = {v: f"variable {v}" for v in VARIABLES}
    yes_no = {"yes": "yes, it should", "no": "no, it should not"}
    return {
        "input_param": {
            "type": "choice", "criteria": var_criteria,
            "instructions": {"goal": REQUIREMENT,
                             "rules": ["which variable holds the incoming collection the function iterates?"]},
        },
        "filter_var": {
            "type": "choice", "criteria": var_criteria,
            "instructions": {"goal": REQUIREMENT,
                             "rules": ["which variable should the filter condition test for truth?"]},
        },
        "return_id": {
            "type": "choice", "criteria": yes_no,
            "instructions": {"goal": REQUIREMENT, "rules": ["should the returned record include id?"]},
        },
        "return_name": {
            "type": "choice", "criteria": yes_no,
            "instructions": {"goal": REQUIREMENT, "rules": ["should the returned record include name?"]},
        },
        "return_active": {
            "type": "choice", "criteria": yes_no,
            "instructions": {"goal": REQUIREMENT, "rules": ["should the returned record include active?"]},
        },
    }


def assemble(answers: dict) -> str:
    """宿主按固定的语句形式拼代码；变量取值全部来自 Jev 的选择。"""
    param = answers["input_param"]["choice"]
    cond = answers["filter_var"]["choice"]
    fields = [f for f in ("id", "name", "active")
              if answers[f"return_{f}"]["choice"] == "yes"]
    item, acc = "row", "result"
    body = [
        FORMS["def"].format(name="active_users", param=param),
        "    " + FORMS["init"].format(acc=acc),
        "    " + FORMS["loop"].format(item=item, param=param),
        "        " + FORMS["filter"].format(item=item, cond=cond),
        "            " + FORMS["append"].format(
            acc=acc, fields=", ".join(f'"{f}": {item}["{f}"]' for f in fields)),
        "    " + FORMS["return"].format(acc=acc),
    ]
    return "\n".join(body) + "\n"


SAMPLE = [
    {"id": 1, "name": "Ada", "active": True, "email": "a@x"},
    {"id": 2, "name": "Bob", "active": False, "email": "b@x"},
    {"id": 3, "name": "Cid", "active": True, "email": "c@x"},
]
EXPECT = [{"id": 1, "name": "Ada"}, {"id": 3, "name": "Cid"}]


def judge(source: str) -> tuple[bool, tuple[str, ...]]:
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return False, (f"syntax: {exc.msg}",)
    ns: dict[str, object] = {}
    try:
        exec(compile(tree, "<candidate>", "exec"), ns)  # noqa: S102
    except Exception as exc:  # noqa: BLE001
        return False, (f"exec failed: {type(exc).__name__}",)
    fn = ns.get("active_users")
    if not callable(fn):
        return False, ("no active_users function",)
    try:
        got = fn(SAMPLE)
    except Exception as exc:  # noqa: BLE001
        return False, (f"call failed: {type(exc).__name__}: {exc}",)
    if got != EXPECT:
        return False, (f"wrong result: {got}",)
    return True, ()


def part_a() -> None:
    print("=" * 78)
    print("A. normal loop — Jev answers five slot questions in ONE request")
    print("=" * 78)
    print(f"requirement: {REQUIREMENT}")
    print(f"keyword library (forms): {', '.join(FORMS)}")
    print(f"variables supplied by the big model: {', '.join(VARIABLES)}")
    print()
    passed = 0
    times: list[float] = []
    for trial in range(3):
        data, seconds = call(build_questions())
        times.append(seconds)
        answers = data.get("answers", {})
        picked = {k: v.get("choice") for k, v in answers.items()}
        source = assemble(answers)
        ok, why = judge(source)
        passed += int(ok)
        print(f"trial {trial + 1}: {seconds:.2f}s  {'PASS' if ok else 'FAIL'}")
        print(f"  Jev picked: input={picked['input_param']}  filter={picked['filter_var']}  "
              f"return={[f for f in ('id', 'name', 'active') if picked[f'return_{f}'] == 'yes']}")
        conf = {k: round(float(v.get('confidence', 0)), 2) for k, v in answers.items()}
        print(f"  confidence: {conf}")
        if not ok:
            print(f"  reason: {'; '.join(why)}")
        if trial == 0:
            print("  --- host-assembled code ---")
            for line in source.rstrip().splitlines():
                print(f"  | {line}")
    print()
    print(f"A result: {passed}/3   median {sorted(times)[1]:.2f}s")


def part_b() -> None:
    """缺口回路：池里没有需要的东西时，会不会硬凑。"""
    print()
    print("=" * 78)
    print("B. gap loop — the requirement needs something NOT in the pool")
    print("=" * 78)
    cases = (
        ("needs a field that is not in the pool",
         "build active_users(rows): keep rows where active is true, return id and phone",
         ["id", "name", "email", "active"]),
        ("needs a sort form that is not in the library",
         "build active_users(rows): keep active rows, return id and name, sorted by name",
         ["id", "name", "email", "active"]),
    )
    for label, requirement, fields in cases:
        criteria = {f: f"field {f}" for f in fields}
        criteria["NONE"] = "none of these fields can satisfy the requirement"
        questions = {
            "field": {"type": "choice", "criteria": criteria,
                      "instructions": {"goal": requirement,
                                       "rules": ["pick the field the requirement asks for; "
                                                 "if it is not listed, pick NONE"]}},
        }
        data, seconds = call(questions)
        answer = data.get("answers", {}).get("field", {})
        choice = answer.get("choice")
        confidence = answer.get("confidence")
        verdict = "CORRECTLY ABSTAINED" if choice == "NONE" else f"picked {choice} (invented nothing, but did not abstain)"
        print(f"  {label}")
        print(f"    requirement: {requirement}")
        print(f"    pool: {', '.join(fields)} + NONE")
        print(f"    Jev -> {choice}   confidence={confidence}   {seconds:.2f}s   {verdict}")
        print()


def main() -> int:
    print(f"endpoint={ENDPOINT} model={MODEL}")
    print()
    part_a()
    part_b()
    print("judgement is runtime behaviour: the assembled code is exec'd and called.")
    print("the forms are fixed by the host; only the slot values come from the selector.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
