# codejev

**English** | [中文](#中文说明)

**When a task is fixed enough, don't let the model write what it can *choose*.**

A big model defines the task and the candidate vocabulary. The **host** enumerates concrete
code candidates, owns their ids, assembles the result deterministically, and verifies it.
A small **Jev-style selector** does exactly one thing: return one candidate id — or `NONE`.

```
requirement
   │
   ▼  big model: task + candidate keywords / keyword library
CodeTask { operation, requirements, constraints }
   │
   ▼  host: enumerate concrete code candidates, assign stable ids
CandidatePage
   [0] filter_and_project  "keep rows where active is true, project id/name"
   [1] sort_rows           "sort rows by score, descending"
   [2] group_and_count     "group rows by category, count each"
   NONE                    "none of the above"
   │
   ▼  selector: returns ONE id (no text, no prose, no code)
{"choice": "0", "confidence": 0.84, "probabilities": {...}}
   │
   ▼  host: look up the id, return the code. The model never wrote it.
result = []
for row in rows:
    if row["active"]:
        result.append({"id": row["id"], "name": row["name"]})
return result
   │
   ▼  big model: check the result → re-instruct if wrong
```

The selector never decides paths, never approves writes, never diagnoses failures, never
retries. Every identity field (id, hash, path, approval) is minted by the host; same-named
fields in model output are ignored.

## Why this shape

Because we measured it. Same model, same real source file, only the output shape differs:

| Route | Result |
| --- | --- |
| **Selection** (host gives candidates, model returns an id) | **5/5** |
| **Free generation** (model writes the whole function) | **0/5** — it echoes the input back |

And the failure is **silent**: the artifact is valid, runnable Python. The diff looks like a
clean change. You only catch it by executing the result — which is what this repo's
probes do.

## Measured: four selectors on one candidate page

Requirement: pick the implementation that satisfies the instruction. Candidates are
enumerated by the host from real source.

| Selector | Accuracy | Latency | Can it answer "none of the above"? |
| --- | --- | --- | --- |
| LFM2.5-350M (open weights, batch scoring) | **1/4** | 0.03–0.11s | ❌ constant output `2` |
| Qwen2.5-Coder-1.5B (local, MLX) | 2–3/4 | 0.33–0.6s | ❌ picked `1` |
| **Jev** (TypeSafe System One) | **4/4** | **0.73s** | ✅ `NONE`, confidence **1.00** |
| DeepSeek Flash (API) | 5/5 | 0.37–2.1s | ✅ |

The 350M row is the sharpest evidence: it is not choosing wrong, it is **not discriminating
at all** — it emits `2` for every requirement, including one whose capability is not in the
candidate pool. Uniform random guessing would also score 1/4. Below some size, semantic
matching simply does not happen.

Only two of four can abstain. That is a **mechanism** difference, not a capability one:
argmax scoring must always pick one; a selector can reject the whole page.

### Scale: 6 candidates → 417 candidates

| Candidates | Prompt tokens | Correct? |
| --- | ---: | --- |
| 6 | ~440 | ✅ |
| 417 | 24,517 | ✅ (no degradation) |

### Where the ceiling actually is

Not the model — **the enumerator**.

On 15 real tasks over a real source file, Jev scored 12/15 raw. All three "failures" were
**my enumerator's fault**: it skipped functions starting with `_`, so two expected answers
were never offered, and one expected function lived in a different file (Jev correctly
answered `NONE`). Adjusted for that, the selector is **15/15**.

> Selection accuracy ceiling = enumerator recall. Fix the enumerator before swapping models.

## The flow, end to end

```bash
# 1. host enumerates candidates from real source (ast), assigns stable ids
# 2. selector returns one id — see bench/jev_probe.py for the exact contract
export TYPESAFE_API_KEY=...
python -m bench.jev_probe          # hand-written page, 4 tasks
python -m bench.jev_batch_probe    # real source pool, 15 tasks
```

```bash
# Any OpenAI-compatible selector works through the same protocol
export AZFLS_API_BASE=https://api.deepseek.com/v1 AZFLS_API_KEY=sk-... AZFLS_MODEL=deepseek-chat
python -m bench.candidate_api_probe
```

```bash
# The local path needs no API at all (model weights in models/, not in git)
HF_HUB_OFFLINE=1 python -m bench.candidate_probe
```

## Cost

Jev prices input at **$0.042 / M tokens** (listed as *$42 per billion*) and **output is free** —
it emits no text. Per selection, at our measured token counts:

| Selector | Per selection |
| --- | ---: |
| **Jev** (419 in / 40 out, output free) | **$0.000018** |
| DeepSeek Flash, off-peak (330 in / 2 out) | $0.000051 |
| Local Qwen1.5B | ≈ $0.000003 electricity |

Jev is ~3–6× cheaper per selection than an API generator. But **its value is not the money** —
a single task saves fractions of a cent. Its value is **accuracy, the ability to abstain, and
calibrated confidence**. In our runs the lowest confidence (0.43) landed exactly on the task
whose candidate pool was incomplete — the confidence is tracking real uncertainty.

Generation is where the money goes, and Jev cannot generate. Local generation is ~100× cheaper
than API generation, but our local model scored 0/5 on real-shaped functions.

## Install & test

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt   # or: mlx-lm libcst pytest
python -m pytest tests/ -q          # 351 tests, all offline, no weights, no network
python -m codejev.cli check         # environment self-check
```

## Layout

```
codejev/
  candidate.py     candidate-page protocol: host owns ids and code
  jev_engine.py    Jev (System One) selector — strict parsing, no key leakage
  api_engine.py    OpenAI-compatible selector
  model.py         local MLX engine (resident) + timing
  decide.py        slot-level selection: ast extraction → pick → libcst assembly
  gate.py          confirmation gate: approval bound to target + content hash
  adapter.py       wrapping, diffs, integrity checks
  cli.py           ask / select / check
bench/             every number in this README has a reproducible probe
docs/              design notes, ADR, measurement logs
```

## Honest limits

- One machine (M1 Pro 16GB), small samples (4–15 tasks), one target language (Python).
- **Deliberately not built**: error learning, failure-recycling training, RL, auto-repair
  loops, multi-language, diffusion training, complex agent orchestration.
- Costs are computed from list prices; the real bill after free credit is unverified.
- Through a proxy, Jev latency was ~5.8s; direct it is 0.73s. **8× difference — do not
  attribute network latency to the model.**
- The other experiments in `docs/` (diffusion, linear architectures, prompt-level narrowing)
  are recorded as **negative results** so nobody repeats them. They are not the main line.

## Related work

This direction is crowded and moving fast — open implementations now range from 151M to 14B:
Jev / [jev-ultrafast](https://github.com/browser-use/jev-ultrafast), CUA-S1-FORMS (706K params,
99.7% on one narrow task), Loom (composition over generation), universal-selector, and a
growing set of RLCD replications.

**The difference here is that the ownership boundary is made explicit and measured**:
candidate ids, code, paths and approvals all stay with the host; the model may only return an
id. Every conclusion ships with its numbers, including the failures we caused ourselves.

MIT License.

---

# 中文说明

**任务足够固定时，不要让模型去写它能选的东西。**

大模型定义任务和候选词；宿主枚举具体代码候选、掌握 id、确定性组装并验证；
小模型只做一件事——**回一个候选 id 或 `NONE`**。

## 流程

```text
需求
  → 大模型：任务 + 候选词/关键词库
  → 宿主：枚举候选代码，分配稳定 id
  → 选择器：只回一个 id
  → 宿主：按 id 取出代码（模型从没写过这段代码）
  → 大模型：检查结果，不对就重新下指令
```

## 关键实测

| 路线 | 结果 |
| --- | --- |
| 选择（宿主给候选，只回 id） | **5/5** |
| 自由生成（让模型写整个函数） | **0/5**（把原文件原样吐回） |

四个选择器跑同一份候选页：

| 选择器 | 准确率 | 延迟 | 能弃权吗 |
| --- | --- | --- | --- |
| LFM2.5-350M（开源权重，批打分） | **1/4** | 0.03–0.11s | ❌ 恒定输出 `2` |
| Qwen2.5-Coder-1.5B（本地） | 2–3/4 | 0.33–0.6s | ❌ 选了 `1` |
| **Jev**（System One） | **4/4** | **0.73s** | ✅ `NONE`，置信度 **1.00** |
| DeepSeek Flash（API） | 5/5 | 0.37–2.1s | ✅ |

350M 那行最直观：它不是选错，是**没有在区分需求**——均匀随机猜也是 1/4。
参数小到某个尺寸，语义匹配这件事就消失了。

只有两个能弃权。这是**机制差异**：argmax 打分永远必须挑一个；选择器能对整页说"都不满足"。

候选池从 6 涨到 **417**，准确率没有退化。

**上限在枚举器，不在模型。** 15 条真实任务里 Jev 的 3 个"错误"全是我的枚举器漏了候选
（跳过了下划线开头的函数），模型本身 15/15。

## 成本

Jev 输入 **$0.042/百万 token**（官网写作 *$42 per billion*），**输出免费**（不吐字）。

| 选择器 | 每次选择 |
| --- | ---: |
| **Jev** | **$0.000018** |
| DeepSeek Flash（错峰） | $0.000051 |
| 本地 Qwen1.5B | ≈ $0.000003（电费） |

Jev 比 API 生成模型便宜 3–6 倍，但**它的价值不在省钱**，而在准确率、弃权能力和校准置信度。
实测里最低置信度 0.43 恰好落在候选池不全的那条任务上。

## 诚实边界

- 单机、小样本、单一语言（Python）
- **刻意不做**：错误学习、失败回收训练、强化学习、自动修复闭环、多语言、扩散训练
- `docs/` 里的扩散、线性架构、提示级变窄都是**否定结果**，记录在案避免重复踩坑，不是主线
- 走代理时延迟约 5.8s、直连 0.73s，别把网络延迟算到模型头上
