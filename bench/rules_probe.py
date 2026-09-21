"""规则探针：一条纯正则/关键词的窄规则，能替模型读懂多少条中文选择指令。

选择式路径每次要付约 0.6 s 的小模型调用，这里量它能不能被一条规则跳过：
规则要给出（过滤字段, 是否取反, 返回字段）这组选择。

结果分三种计数：correct（与预期一致，这一次真的能跳过模型）、
WRONG（给了答案但是错的——最危险，会静默改成错误的过滤条件）、
declined（拿不准，回落模型——安全，只是没省下这次调用）。

纯 stdlib，不加载权重、不联网。样本只有 14 条自撰指令，规则也是照着它们写的，
数字只用来判断"这条规则值不值得往下做"，不是通用识别率。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from azfls.decide import extract

# 与 azfls/fixtures.py 的 USERS_MODULE 同形状：字典取值，返回全部字段、不过滤。
SOURCE = '''"""用户列表示例：字典风格。"""


def active_users(users):
    """返回用户，保持原顺序。"""
    result = []
    for user in users:
        result.append({"id": user["id"], "name": user["name"], "active": user["active"]})
    return result
'''

# 小同义词表：词 -> (候选字段名, 词本身是否断言"真")；None 表示只有名字、不含极性。
SYNONYMS: dict[str, tuple[str, bool | None]] = {
    "编号": ("id", None), "序号": ("id", None),
    "名字": ("name", None), "名称": ("name", None), "姓名": ("name", None),
    "激活": ("active", None), "活跃": ("active", None),
    "有效": ("active", True), "启用": ("active", True),
    "无效": ("active", False), "未激活": ("active", False), "停用": ("active", False),
}

# 字段名后面的极性词；分句分隔符（线索跨分句不算）；贴在前面的否定字；丢弃词。
_AFTER_TRUE = re.compile(r"^[\s为是=]{0,4}(真|true)", re.IGNORECASE)
_AFTER_FALSE = re.compile(r"^[\s为是=]{0,4}(假|false)", re.IGNORECASE)
_SEPARATORS = "，,。；;"
_NEG_PREFIX = ("未", "不", "非", "没", "无")
_DROP_CUES = ("去掉", "不要", "排除", "过滤掉", "剔除", "删除", "移除", "忽略", "跳过")


@dataclass(frozen=True)
class Mention:
    """指令里提到的一个字段；drop 表示所在分句里有"去掉/不要"这类丢弃词。"""

    start: int  # 在指令里的位置，用来定返回字段的顺序
    field: str  # 已按候选表校验过的字段名
    said_false: bool | None  # 指令说"这个字段为假"；None 表示没写清真假
    drop: bool


@dataclass(frozen=True)
class Case:
    """一条固定指令与它的正确读法；filter_field 为 None 表示不该加过滤。"""

    label: str
    phrasing: str
    filter_field: str | None
    filter_negated: bool | None
    return_fields: tuple[str, ...]


@dataclass(frozen=True)
class RuleResult:
    """规则给出的完整选择；extract_rule 返回 None 表示 declined。"""

    filter_field: str
    filter_negated: bool  # True 表示结果是 not item["字段"]
    return_fields: tuple[str, ...]


# LITERAL 字段名与真假都写全；PARAPHRASE 只用同义词；TRICKY 否定、倒装，
# 以及"不要"到底是在过滤还是在挑返回字段。
CASES: tuple[Case, ...] = (
    Case("L1", "只保留 active 为真的项，返回 id 和 name", "active", False, ("id", "name")),
    Case("L2", "返回 id 和 name，只保留 active 为真的", "active", False, ("id", "name")),
    Case("L3", "只保留 active 为假的项，返回 id 和 name", "active", True, ("id", "name")),
    Case("L4", "过滤掉 active 为真的项，返回 name", "active", True, ("name",)),
    Case("L5", "筛选 active 为假的记录，返回 name 和 id", "active", True, ("name", "id")),
    Case("P1", "只要有效用户，返回编号和名字", "active", False, ("id", "name")),
    Case("P2", "去掉无效用户，返回编号", "active", False, ("id",)),
    Case("P3", "我需要还在用的用户，给我编号和名称", "active", False, ("id", "name")),
    Case("P4", "只显示启用状态的项，返回名称", "active", False, ("name",)),
    Case("P5", "返回编号，只要有效的", "active", False, ("id",)),
    Case("T1", "去掉未激活的，只留 id 和 name", "active", False, ("id", "name")),
    Case("T2", "active 为真的不要，返回 id 和 name", "active", True, ("id", "name")),
    Case("T3", "返回 id 和 name，过滤掉 active 为假的", "active", False, ("id", "name")),
    Case("T4", "不要 active 字段，返回 id 和 name", None, None, ("id", "name")),
)


def extract_rule(phrasing: str, vocab: tuple[str, ...], *, strict: bool = False) -> RuleResult | None:
    """窄规则：唯一一个带条件线索的字段当过滤条件，其余提到的字段当返回字段。

    线索 = 词自带极性（有效/无效/未激活）、后面跟着"为真/为假"、或同一分句里有
    "去掉/不要"。线索不唯一就返回 None（declined），宁可回落模型也不猜。
    strict 再收紧一档：有丢弃词却没写清真假时也拒绝——那种句子更可能是在挑
    返回字段，不是要过滤。
    """
    mentions = _mentions(phrasing, vocab)
    cues = [m for m in mentions if m.said_false is not None or m.drop]
    if len(cues) != 1:
        return None  # 没有条件线索，或不止一个字段有线索 → 拿不准
    chosen = cues[0]
    if strict and chosen.drop and chosen.said_false is None:
        return None
    returns = tuple(m.field for m in mentions if m.start != chosen.start)
    if not returns:
        return None  # 没说要返回什么，别猜
    # 没说真假时按"要真的"处理：这是整条规则里最危险的一步假设。
    return RuleResult(
        filter_field=chosen.field,
        filter_negated=bool(chosen.said_false) ^ chosen.drop,  # 丢掉"为假"的 = 保留"为真"的
        return_fields=returns,
    )


def _mentions(phrasing: str, vocab: tuple[str, ...]) -> list[Mention]:
    """按出现顺序找提到的字段：重叠的词取更长的，同一字段只留第一次出现。"""
    hits: list[tuple[int, int, str, bool | None]] = []
    for field in vocab:
        words: list[tuple[str, bool | None]] = [(field, None)]  # 英文原名永远可匹配
        words += [(w, pol) for w, (name, pol) in SYNONYMS.items() if name == field]
        for word, asserts_true in words:
            for match in _pattern(word).finditer(phrasing):
                hits.append((match.start(), match.end(), field, asserts_true))
    hits.sort(key=lambda hit: (hit[0], hit[0] - hit[1]))  # 同一起点先长后短
    taken: list[tuple[int, int]] = []
    seen: set[str] = set()
    found: list[Mention] = []
    for start, end, field, asserts_true in hits:
        if field in seen or any(start < b and a < end for a, b in taken):
            continue
        seen.add(field)
        taken.append((start, end))
        found.append(
            Mention(
                start=start,
                field=field,
                said_false=_said_false(phrasing, start, end, asserts_true),
                drop=any(cue in _clause(phrasing, start, end) for cue in _DROP_CUES),
            )
        )
    return found


def _pattern(word: str) -> re.Pattern[str]:
    """英文名按词边界匹配（避免 id 命中别的单词），中文按子串匹配。"""
    if word.isascii():
        return re.compile(rf"(?<![A-Za-z0-9_]){re.escape(word)}(?![A-Za-z0-9_])", re.IGNORECASE)
    return re.compile(re.escape(word))


def _said_false(text: str, start: int, end: int, asserts_true: bool | None) -> bool | None:
    """这个字段被说成"真"还是"假"；都没说就返回 None。"""
    if asserts_true is not None:
        return not asserts_true  # 无效 / 未激活 这类词本身就说了"假"
    window = text[end : end + 8]
    if _AFTER_FALSE.match(window):
        return True
    if _AFTER_TRUE.match(window):
        return False
    if text[start - 1 : start] in _NEG_PREFIX:  # 不活跃 / 未启用
        return True
    return None


def _clause(text: str, start: int, end: int) -> str:
    """字段所在的分句；线索只在同一个分句里算数。"""
    left = max((text.rfind(sep, 0, start) for sep in _SEPARATORS), default=-1)
    right = min((i for i in (text.find(sep, end) for sep in _SEPARATORS) if i != -1), default=len(text))
    return text[left + 1 : right]


def _outcome(case: Case, result: RuleResult | None) -> str:
    """三选一：declined / correct / WRONG。"""
    if result is None:
        return "declined"
    if (
        result.filter_field == case.filter_field
        and result.filter_negated == case.filter_negated
        and result.return_fields == case.return_fields
    ):
        return "correct"
    return "WRONG"


def _describe(result: RuleResult | None) -> str:
    """表格里"提取到的东西"那一列。"""
    if result is None:
        return "（不敢认，回落模型）"
    keep = "保留假值" if result.filter_negated else "保留真值"
    return f"filter={result.filter_field}（{keep}） return=[{', '.join(result.return_fields)}]"


def _pad(text: str, width: int) -> str:
    """按终端显示宽度补齐（中日韩字符算 2 列），否则中文列对不齐。"""
    return text + " " * max(0, width - sum(2 if ord(ch) > 0x2E7F else 1 for ch in text))


def main() -> None:
    """跑 14 条指令，打印表格、三项计数和结论。"""
    vocab = tuple(item.name for item in extract(SOURCE).fields)
    for case in CASES:  # 期望的字段必须真的在候选表里，否则探针本身写错了
        for name in (*case.return_fields, case.filter_field):
            if name is not None and name not in vocab:
                raise SystemExit(f"用例 {case.label} 期望的字段 {name} 不在候选表 {vocab} 里")

    results = [extract_rule(case.phrasing, vocab) for case in CASES]
    outcomes = [_outcome(case, result) for case, result in zip(CASES, results)]
    strict = [_outcome(c, extract_rule(c.phrasing, vocab, strict=True)) for c in CASES]
    label_w = max(len(case.label) for case in CASES)
    text_w = max(sum(2 if ord(ch) > 0x2E7F else 1 for ch in case.phrasing) for case in CASES)

    lines = [
        f"候选字段（来自 azfls.decide.extract）：{', '.join(vocab)}",
        "L=字面  P=换说法（同义词）  T=陷阱（否定/倒装）",
        "",
        f"{_pad('标签', label_w)}  {_pad('指令', text_w)}  {_pad('结果', 8)}  规则提取到的东西",
    ]
    lines.append("-" * sum(2 if ord(ch) > 0x2E7F else 1 for ch in lines[-1]))
    for case, result, outcome in zip(CASES, results, outcomes):
        mark = "  ← 静默改错" if outcome == "WRONG" else ""
        lines.append(
            f"{_pad(case.label, label_w)}  {_pad(case.phrasing, text_w)}  "
            f"{_pad(outcome, 8)}  {_describe(result)}{mark}"
        )

    total = len(CASES)
    correct, wrong, declined = (outcomes.count(name) for name in ("correct", "WRONG", "declined"))
    lines += [
        "",
        f"汇总：correct={correct}  WRONG={wrong}  declined={declined}（共 {total} 条）",
        f"可以跳过模型：{correct}/{total} = {correct / total:.1%}",
        "加了闸（有“去掉/不要”但没写“为真/为假”就拒绝）："
        f"correct={strict.count('correct')}  WRONG={strict.count('WRONG')}  "
        f"declined={strict.count('declined')}",
        "",
        "结论：",
    ]
    bad = [case for case, outcome in zip(CASES, outcomes) if outcome == "WRONG"]
    if bad:
        listed = "、".join(f"{case.label}“{case.phrasing}”" for case in bad)
        lines.append(
            f"1) 14 条里规则读对 {correct} 条、读错 {wrong} 条（{listed}）：读错不是拒绝，规则照样"
            "给出了一个语法合法的过滤条件，diff 看起来很干净，会被静默写入。"
        )
        lines.append("2) 一次静默改错比每次多付 0.6 秒模型调用糟糕得多，所以现在这条规则快路不值得加。")
        if strict.count("WRONG") == 0:
            lines.append(
                "3) 只加一道闸（出现“去掉/不要”但没写“为真/为假”就拒绝）就能把读错压到 0 条："
                f"读对 {strict.count('correct')} 条、回落模型 {strict.count('declined')} 条，"
                f"{strict.count('correct')}/{total} 可以跳过模型——这个形状可以试，但 14 条自撰样本"
                "太小，上线前要用真实指令扩样。"
            )
    else:
        lines.append(
            f"1) 规则没有读错，读对 {correct} 条、回落模型 {declined} 条：在读懂的这个子集上快路"
            "是安全的，剩下的照旧问模型。"
        )
    print("\n".join(lines))


if __name__ == "__main__":
    main()
