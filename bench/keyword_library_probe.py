"""用完整的关键词库让 Jev 拼一个长模块。

作者描述的回路：
  大模型提出需求 → 给出目标语言的**关键词库**（该语言的语句形式）
  → 再给出需求需要的**变量**（关键词库里没有的）
  → Jev 只做选择 → 宿主按形式拼接 → 交回大模型检查

这里把关键词库做成接近完整的 Python 语法形式（覆盖全部常用关键字的组合形态），
变量由大模型给出，然后让 Jev 拼出一个**多函数的长模块**。

关键设计问题：Jev 每次只回一个选择。要输出变长而不能变成几十次往返，
就要让宿主先给出**结构骨架**（由关键词库里的形式组成），Jev 只选：
  1. 每个函数用哪个骨架
  2. 每个槽位绑哪个变量
这样决策数固定（一次请求并列问完），输出长度由骨架展开。——这正是"组合"的含义。

判据：宿主拼出的模块必须能 exec，且三个函数**真的调用**后行为正确。
"""

from __future__ import annotations

import ast
import json
import os
import time
import urllib.request

ENDPOINT = "https://api.typesafe.ai/v1/systemone"
MODEL = "jev-latest"

# ---- 关键词库：Python 常用关键字，以及由它们构成的语句形式 ----
PYTHON_KEYWORDS = [
    "False", "None", "True", "and", "as", "assert", "async", "await", "break", "class",
    "continue", "def", "del", "elif", "else", "except", "finally", "for", "from",
    "global", "if", "import", "in", "is", "lambda", "nonlocal", "not", "or", "pass",
    "raise", "return", "try", "while", "with", "yield",
]

# 语句形式：每个都是一小段带槽位的代码，由上面的关键字组合而成
FORMS = {
    "def":        "def {name}({param}):",
    "init_list":  "{acc} = []",
    "init_dict":  "{acc} = {{}}",   # {{}} 是字面空字典，转义后才是合法模板
    "loop":       "for {item} in {param}:",
    "if_truthy":  'if {item}["{cond}"]:',
    "if_falsy":   'if not {item}["{cond}"]:',
    "append_dict": '{acc}.append({{{fields}}})',
    "assign_get": '{key} = {item}["{key}"]',
    "count_inc":  "{acc}[{key}] = {acc}.get({key}, 0) + 1",
    "sort_key":   '{acc}.sort(key=lambda {item}: {item}["{key}"], reverse={rev})',
    "return_acc": "return {acc}",
}

# ---- 大模型给出的变量（关键词库里没有的）----
VARIABLES = ["rows", "user", "order", "id", "name", "active", "score", "category"]

# ---- 骨架：由关键词库里的形式组成，宿主事先定义 ----
SKELETONS = {
    # 每项是 (形式, 缩进层级)：结构由宿主定死，不靠启发式猜
    "filter_project": [("def", 0), ("init_list", 1), ("loop", 1), ("if_truthy", 2),
                       ("append_dict", 3), ("return_acc", 1)],
    "filter_sort":    [("def", 0), ("init_list", 1), ("loop", 1), ("if_truthy", 2),
                       ("append_dict", 3), ("sort_key", 1), ("return_acc", 1)],
    "group_count":    [("def", 0), ("init_dict", 1), ("loop", 1), ("assign_get", 2),
                       ("count_inc", 2), ("return_acc", 1)],
}

REQUIREMENT = (
    "write a module with three functions over `rows`, a list of dicts with keys "
    "id / name / active / score / category:\n"
    "  1. active_rows(rows): keep rows where active is true, return id and name\n"
    "  2. top_rows(rows): keep active rows, return id and score, sorted by score descending\n"
    "  3. count_by_category(rows): count rows per category, return category -> count"
)

SAMPLE = [
    {"id": 1, "name": "Ada", "active": True, "score": 5, "category": "x"},
    {"id": 2, "name": "Bob", "active": False, "score": 9, "category": "y"},
    {"id": 3, "name": "Cid", "active": True, "score": 7, "category": "x"},
]


def call(questions: dict, retries: int = 3) -> tuple[dict, float]:
    key = os.environ.get("TYPESAFE_API_KEY", "")
    if not key:
        raise RuntimeError("需要 TYPESAFE_API_KEY")
    body = {
        "model": MODEL,
        "state": {
            "page": {"url": "codejev://keyword-library", "title": "python keyword library", "text": REQUIREMENT},
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
            with urllib.request.urlopen(request, timeout=180) as response:
                return json.loads(response.read().decode()), time.perf_counter() - started
        except Exception as exc:  # noqa: BLE001
            last = exc
            if attempt < retries - 1:
                time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"重试后仍失败: {last}")


def build_questions() -> dict:
    """一次请求问完所有决策：三个函数各用哪个骨架 + 各自的变量绑定。"""
    var = {v: f"variable {v}" for v in VARIABLES}
    skel = {k: "shape: " + " -> ".join(form for form, _depth in v)
            for k, v in SKELETONS.items()}
    q: dict = {
        "shape_1": {"type": "choice", "criteria": skel,
                    "instructions": {"goal": REQUIREMENT,
                                     "rules": ["shape for function 1: keep active rows, return id and name"]}},
        "shape_2": {"type": "choice", "criteria": skel,
                    "instructions": {"goal": REQUIREMENT,
                                     "rules": ["shape for function 2: keep active rows, return id and score, "
                                               "sorted by score descending"]}},
        "shape_3": {"type": "choice", "criteria": skel,
                    "instructions": {"goal": REQUIREMENT,
                                     "rules": ["shape for function 3: count rows per category"]}},
        "loop_item": {"type": "choice", "criteria": var,
                      "instructions": {"goal": REQUIREMENT,
                                       "rules": ["which variable is a single element of `rows`?"]}},
        "filter_var": {"type": "choice", "criteria": var,
                       "instructions": {"goal": REQUIREMENT,
                                        "rules": ["which key is tested for truth in the filter?"]}},
        "sort_key": {"type": "choice", "criteria": var,
                     "instructions": {"goal": REQUIREMENT,
                                      "rules": ["which key is the sort used on?"]}},
        "group_key": {"type": "choice", "criteria": var,
                      "instructions": {"goal": REQUIREMENT,
                                       "rules": ["which key is the grouping done by?"]}},
        "field_a": {"type": "choice", "criteria": var,
                    "instructions": {"goal": REQUIREMENT,
                                     "rules": ["first key of the returned record"]}},
        "field_b": {"type": "choice", "criteria": var,
                    "instructions": {"goal": REQUIREMENT,
                                     "rules": ["second key of the returned record"]}},
    }
    return q


def assemble(answers: dict) -> str:
    """宿主按骨架把形式展开成代码；变量取值全部来自 Jev 的选择。"""
    pick = {k: v.get("choice") for k, v in answers.items()}
    item, acc = "row", "result"
    lines: list[str] = ['"""Generated by host assembly from a keyword library."""', "", ""]

    def expand(func: str, shape: str, fields: list[str], cond: str) -> list[str]:
        """按骨架展开：层级由骨架给出，宿主不做缩进猜测。"""
        out: list[str] = []
        for form_name, depth in SKELETONS[shape]:
            tpl = FORMS[form_name]
            if form_name == "def":
                body = tpl.format(name=func, param="rows")
            elif form_name in ("init_list", "init_dict"):
                body = tpl.format(acc=acc)
            elif form_name == "loop":
                body = tpl.format(item=item, param="rows")
            elif form_name in ("if_truthy", "if_falsy"):
                body = tpl.format(item=item, cond=cond)
            elif form_name == "append_dict":
                body = tpl.format(acc=acc, fields=", ".join(
                    f'"{f}": {item}["{f}"]' for f in fields))
            elif form_name == "assign_get":
                body = tpl.format(key=pick["group_key"], item=item)
            elif form_name == "count_inc":
                body = tpl.format(acc=acc, key=pick["group_key"])
            elif form_name == "sort_key":
                body = tpl.format(acc=acc, item=item, key=pick["sort_key"], rev="True")
            else:
                body = tpl.format(acc=acc)
            out.append("    " * depth + body)
        return out

    lines += expand("active_rows", pick["shape_1"] or "filter_project",
                    [pick["field_a"], pick["field_b"]], pick["filter_var"])
    lines += ["", ""]
    lines += expand("top_rows", pick["shape_2"] or "filter_sort",
                    [pick["field_a"], pick["sort_key"]], pick["filter_var"])
    lines += ["", ""]
    lines += expand("count_by_category", pick["shape_3"] or "group_count",
                    [], "")
    return "\n".join(lines) + "\n"


def judge(source: str) -> tuple[bool, tuple[str, ...]]:
    reasons: list[str] = []
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return False, (f"syntax error line {exc.lineno}: {exc.msg}",)
    ns: dict[str, object] = {}
    try:
        exec(compile(tree, "<module>", "exec"), ns)  # noqa: S102
    except Exception as exc:  # noqa: BLE001
        return False, (f"exec failed: {type(exc).__name__}: {exc}",)

    for name in ("active_rows", "top_rows", "count_by_category"):
        if not callable(ns.get(name)):
            reasons.append(f"missing {name}")
    if reasons:
        return False, tuple(reasons)

    try:
        first = ns["active_rows"](SAMPLE)
        if len(first) != 2:
            reasons.append(f"active_rows kept {len(first)} rows, expected 2")
        second = ns["top_rows"](SAMPLE)
        scores = [r.get("score") for r in second]
        if scores != sorted(scores, reverse=True):
            reasons.append(f"top_rows not sorted descending: {scores}")
        counts = ns["count_by_category"](SAMPLE)
        if counts != {"x": 2, "y": 1}:
            reasons.append(f"count_by_category wrong: {counts}")
    except Exception as exc:  # noqa: BLE001
        reasons.append(f"call failed: {type(exc).__name__}: {exc}")
    return (not reasons), tuple(reasons)


def main() -> int:
    print(f"endpoint={ENDPOINT} model={MODEL}")
    print(f"keyword library: {len(PYTHON_KEYWORDS)} python keywords, "
          f"{len(FORMS)} statement forms, {len(SKELETONS)} shapes")
    print(f"variables supplied by the big model: {', '.join(VARIABLES)}")
    print()
    print(REQUIREMENT)
    print()
    passed = 0
    times: list[float] = []
    for trial in range(3):
        data, seconds = call(build_questions())
        times.append(seconds)
        answers = data.get("answers", {})
        source = assemble(answers)
        ok, why = judge(source)
        passed += int(ok)
        picks = {k: v.get("choice") for k, v in answers.items()}
        conf = [float(v.get("confidence", 0)) for v in answers.values()]
        print(f"trial {trial + 1}: {seconds:.2f}s  {'PASS' if ok else 'FAIL'}  "
              f"({len(source.splitlines())} lines, min confidence {min(conf):.2f})")
        print(f"  picks: shape_1={picks['shape_1']} shape_2={picks['shape_2']} shape_3={picks['shape_3']}")
        print(f"         item={picks['loop_item']} filter={picks['filter_var']} "
              f"sort={picks['sort_key']} group={picks['group_key']} "
              f"fields=({picks['field_a']}, {picks['field_b']})")
        if not ok:
            print(f"  reason: {'; '.join(why)}")
        if trial == 0:
            print("  --- assembled module ---")
            for line in source.rstrip().splitlines():
                print(f"  | {line}")
        print()
    print(f"result: {passed}/3   median {sorted(times)[1]:.2f}s")
    print()
    print("one request, nine decisions in parallel; the host expands the shapes into code.")
    print("judgement is runtime: the module is exec'd and all three functions are called.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
