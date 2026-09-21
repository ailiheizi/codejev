"""候选页选择准确率对**提示写法**的敏感度：2/5 里有多少是提示造成的（有界实验）。

背景：同一批 5 条任务、同一个候选页、同一个 Qwen2.5-Coder-1.5B（温度 0），
`bench/candidate_probe.py` 记下的现状是 2/5；而 `azfls/decide.py` 的选择路线
实测过“系统提示里少一行工作示例”会把 15/15 打成 3/15（模型整个省掉 `f` 键）。
所以本脚本只问一个问题：候选页的 2/5 有多少来自提示写法，多少来自模型能力。

三个变体（**措辞在跑之前写死**，跑完不看结果换措辞）：

- V0：现状，逐字使用 `azfls.candidate.build_selection_messages` 的产物；
- V1：V0 的**系统提示**末尾追加一条最小工作示例（示例页 + 命中候选 → 回它的 id）；
- V2：V1 的系统提示末尾再追加一条**该回 NONE 的示例**（示例页里没有能完成该任务的候选）。

三者只差系统提示的文本：**用户消息逐字相同**（脚本内 assert），候选页、任务文件、
严格解析器（`azfls.candidate.parse_choice`）一个字都没改。本脚本不写盘、不改生产代码，
产出只有 stdout 上的证据。

严格口径：解析结果精确等于候选 id 或 NONE（模型回 `NO_MATCH` 也算 NONE）才 PASS；
解释、未知 id、多行、代码一律 FAIL。模型明确说“没有匹配”（NO_MATCH）与
“回复无法解析”（INVALID）分开计数。

温度 0 下同一输入输出确定，重复只用来确认确定性并量耗时：回复内容、输出 token 数
是确定性观测，墙钟是耗时噪声。

用法（项目根目录，离线，只用本机模型目录）：

    HF_HUB_OFFLINE=1 .venv/bin/python bench/candidate_prompt_ab.py
    HF_HUB_OFFLINE=1 .venv/bin/python bench/candidate_prompt_ab.py --repeats 2
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:  # 直接以脚本方式运行时，也能 import azfls
    sys.path.insert(0, str(ROOT))
# 只用本机模型目录：整个脚本不需要联网。
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from azfls.candidate import (  # noqa: E402 - 先修好 sys.path 再导入
    _SYSTEM_PROMPT,
    CandidateError,
    CodeTask,
    build_selection_messages,
    parse_choice,
)
from azfls.model import MLXEngine, Stats  # noqa: E402
from bench.candidate_probe import MODEL_PATH, PAGE, load_tasks  # noqa: E402

# 与 `choose_candidate` 的默认值一致：一个 id 或 NONE 就是全部输出。
MAX_TOKENS = 8

# ---------------------------------------------------------------------------
# 变体的提示文本：这里就是本次实验的全部自变量，跑之前写死
# ---------------------------------------------------------------------------

# V1 追加到系统提示末尾的最小工作示例：示例页只有两个候选，
# 任务命中的候选是 0，示范“回命中候选自己的 id”。
_EXAMPLE_MATCH = """Example (a different page; shows only the output shape):
TASK
operation: rename_column
requirements:
- rename column price to cost
CANDIDATES
valid outputs: 0, 1, NONE
candidate 0
name: rename_column
purpose: rename column price to cost.
candidate 1
name: sum_rows
purpose: sum the price column.
Output: 0"""

# V2 在 V1 之上再追加的示例：同一形状，但示例页里没有能做该任务的候选，
# 示范“没有候选能完成就回 NONE”。示例用的是 pivot_rows，不是被测任务里的
# join_rows / filter_and_project，以免把某个具体 operation 的答案送进提示。
_EXAMPLE_NONE = """Example (a different page; shows only the output shape):
TASK
operation: pivot_rows
requirements:
- pivot rows by month
CANDIDATES
valid outputs: 0, 1, NONE
candidate 0
name: rename_column
purpose: rename column price to cost.
candidate 1
name: sum_rows
purpose: sum the price column.
Output: NONE"""

# 变体 id → 系统提示末尾追加的文本（空串 = 不改系统提示）。
_VARIANT_SUFFIX = {
    "V0": "",
    "V1": "\n" + _EXAMPLE_MATCH,
    "V2": "\n" + _EXAMPLE_MATCH + "\n" + _EXAMPLE_NONE,
}
# 固定顺序跑：先跑现状，再跑只加命中示例，最后跑加 NONE 示例的版本。
VARIANT_ORDER = ("V0", "V1", "V2")
