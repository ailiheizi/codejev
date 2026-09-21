# az-fls：把选择权从模型手里拿走，交给宿主

> **English**: When a task is fixed enough, don't let the model write what it can *choose*.
> The host enumerates the candidates; a tiny model only returns an id; the host assembles
> deterministically and verifies. Every number below was measured on one M1 Pro and is reproducible.

**一句话：任务足够固定时，不要让模型去写它能选的东西。**

大模型负责拆任务和判断，宿主负责枚举候选、确定性组装和验证，小模型只做一件事——
**从宿主给的候选里选一个**。它不规划、不解释、不诊断、不决定路径、不批准写入。

这个仓库把这条主张**量出来**了，包括它在哪里成立、在哪里不成立。

## 先看证据

### 同一份候选页，四个执行器

任务：从宿主枚举的候选代码里，选出能满足这条需求的实现。

| 执行器 | 准确率 | 延迟 | 能把"以上都不是"答对吗 |
| --- | --- | --- | --- |
| LFM2.5-350M（开源，批打分） | **1/4** | 0.03–0.11s | ❌ 恒定输出 `2` |
| Qwen2.5-Coder-1.5B（本地） | 2–3/4 | 0.33–0.6s | ❌ 选了 `1` |
| **Jev**（TypeSafe System One） | **4/4** | 0.73s | ✅ `NONE`，置信度 **1.00** |
| DeepSeek Flash（API） | 5/5 | 0.37–2.1s | ✅ |

**350M 那个"恒定输出 `2`"是最直观的一条**：它不是选错，是**根本没有在区分需求**——
均匀随机猜的期望命中率也是 1/4，它正好是 1/4。参数小到某个尺寸，语义匹配这件事就消失了。

而**四个里只有两个能把弃权做对**。这是机制差异，不是能力差异：
批打分是 argmax，永远必须挑一个；选择器能对整页说"都不满足"。

### 选择 vs 自由生成

同一模型、同一份真实源码（带前置守卫、复合条件、循环后排序）：

```text
选择路线（宿主给候选，只回 id）  5/5
自由生成（让模型写整个函数）     0/5   ← 把原文件原样吐回
```

失败是**静默**的：产物是合法且能跑的 Python，看 diff 像一次干净改动。我们没有把它包装成成功。

### 实测地图

| 方向 | 实测结果 | 结论 |
| --- | --- | --- |
| 选择路线（窄任务） | 5/5 | ✅ 主力 |
| 自由生成（宽任务） | 0/5 | ❌ |
| 提示级"变窄"（只填函数体等 5 种变体） | 0/5 | ❌ 模型会丢必要部分 |
| 线性架构 RWKV-7-1.5B | 0/3，44 tok/s | ❌ 比自回归还慢 |
| 混合架构 LFM2-350M / Falcon-H1-0.5B | 0/3 | ❌ |
| 扩散（同一份权重只换解码方式） | 质量 10/10，但慢 2.7–3.7 倍 | ❌ 本机不划算 |
| 跨候选批打分（MLX 实现） | 快 5–6 倍，准确率 3/4→2/4 | ⚠️ 权衡 |
| 前缀 KV 复用 | 长上下文省 **41.5%** | ✅ 纯赚 |

## 想法是什么

```text
用户需求
  ↓
大模型：拆成明确的任务（CodeTask）
  ↓
宿主：枚举候选页（稳定 id 由宿主分配）
  ↓
选择器：只回一个 id 或 NONE
  ↓
宿主：确定性物化 + 边界校验
  ↓
大模型：检查结果 → 需要时重新下指令
```

**所有权划分是核心**：路径、哈希、审批状态一律由宿主生成，
模型输出里的同名字段一律忽略。模型能做的只有"选"，且只能从宿主给的选项里选。

### 它在哪里成立

宿主可枚举 + 单值选择 + 有可执行判据 + 单步。实测到 **417 个候选仍准确**。

### 它在哪里不成立

- **候选枚举不出来**：新函数还没写、动态名字、闭包、装饰器段（清单见 `docs/13`）
- **候选池一大就要检索**：公开数据显示池超过约 100 后检索精度开始下降
- **验证器不存在时**：选错是"合法但错"，只能靠测试发现
- **多步组合会退化成搜索**：每步候选依赖上一步的副作用

**上限在枚举器，不在模型。** 15 条真实任务那次，Jev 的 3 个"错误"全部是我的枚举器
漏了候选（我跳过了 `_` 开头的函数），模型本身 15/15。

## 快速开始

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python mlx-lm libcst pytest

# 全部离线：不加载模型、不联网
.venv/bin/python -m pytest tests/ -q

# 本地 Qwen 探针（先把权重放到 models/，或用 ModelScope 下载）
HF_HUB_OFFLINE=1 .venv/bin/python -m bench.candidate_probe

# 接任意 OpenAI-compatible 执行器
export AZFLS_API_BASE=https://api.deepseek.com/v1
export AZFLS_API_KEY=sk-...
export AZFLS_MODEL=deepseek-chat
.venv/bin/python -m bench.candidate_api_probe

# 接 Jev（TypeSafe System One）
export TYPESAFE_API_KEY=...
.venv/bin/python -m bench.jev_probe          # 手写候选页 4 条
.venv/bin/python -m bench.jev_batch_probe    # 真实源码候选池 15 条

# CLI：生成路线与槽位级选择路线
.venv/bin/python -m azfls.cli ask    --workspace /path/to/proj --target app/users.py \
  --instruction "只保留 active 为真的项，返回 id 和 name" --dry-run
.venv/bin/python -m azfls.cli select --workspace /path/to/proj --target app/users.py \
  --function active_users --instruction "只保留 active 为真的项" --dry-run
.venv/bin/python -m azfls.cli check
```

## 成本（按实测 token 数与官方标价）

| 角色 | 最便宜 | 说明 |
| --- | --- | --- |
| 选择（每次） | **Jev $0.000018** | 输入 $0.042/百万（$42/十亿），输出免费 |
| 生成（每个模块） | **本地 $0.000003** | 电费；API 约 $0.00029–0.00058 |

Jev 在"选择"这个角色上比 API 生成模型便宜约 3–6 倍，但**它的价值不在省钱**
（单任务省下的是万分之几美分），而在**准确率、弃权能力和校准置信度**。
生成那一格它做不了——它不写代码，而生成才是成本大头。

## 代码结构

```text
azfls/
  contracts.py     共享形状：Brief / Artifact / 内容哈希 / 路径解析
  adapter.py       包装正文、生成 diff、完整性检查
  gate.py          确认门：审批绑定目标与内容哈希，内容变了旧确认自动失效
  model.py         本地 MLX 引擎（常驻）+ 计时
  api_engine.py    OpenAI-compatible 执行器
  jev_engine.py    Jev（System One）选择器
  candidate.py     候选页协议：宿主拥有 id 与代码
  decide.py        槽位级选择：ast 提取候选 → 只选 → libcst 无损组装
  cli.py           ask / select / check
bench/             每个数字背后都有可复跑的探针
docs/              设计文档、ADR、实测记录
```

## 诚实边界

- 单机（M1 Pro 16GB）、小样本（4–15 条任务为主）、单一目标语言（Python）。
- **刻意没有做**：错误学习、失败回收训练、强化学习、自动修复闭环、复杂多 agent 编排、
  多语言、扩散训练。这些是范围限制，不是遗漏。
- 成本按标价推算，未核对免费额度用尽后的真实账单。
- 走代理时延迟约 5.8s，直连 0.73s——**差 8 倍**，别把网络延迟算进模型头上。

## 与其他工作的关系

这个方向现在很热，开源实现已经成规模（151M 到 14B 都有）：Jev / jev-ultrafast、
CUA-S1-FORMS（706K 参数，窄任务 99.7%）、Loom（组装式开发）、universal-selector，
以及一批 RLCD 复现。

**本仓库的差别是"把所有权划清楚并量出来"**：候选 id、代码、路径、审批全部由宿主掌握，
模型只能回一个 id；并且给出每个结论的实测数字与失败记录，包括我们自己踩的坑。

## 文档

| 文档 | 内容 |
| --- | --- |
| [核心想法](docs/00-core-idea.md) | 角色分工与"最少的话做最多的事" |
| [实测结果](docs/11-measured-results.md) | 本机全部实测数字 |
| [Jev 实测与成本](docs/14-jev-measured.md) | 选择器对照与按角色算的成本 |
| [槽位设计](docs/12-selection-slots.md) | 怎么把更多任务挪进选择路线 |
| [实验顺序](docs/13-next-experiments.md) | 下一步与两个否定结论 |
| [ADR](docs/adr/0001-brainless-candidate-converter.md) | 为什么小模型只做无脑选择 |

MIT License.
