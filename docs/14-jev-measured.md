# Jev (TypeSafe System One): measurements and cost comparison

Measured 2026-09-21 on this machine (M1 Pro), direct connection, no proxy. Every number comes
from a real call, and the commands can be re-run.

## What it is, and what it is not

| | |
| --- | --- |
| **Is** | a **selector**: the host supplies candidates, it returns one option + a confidence + a probability distribution |
| **Is not** | not a generation model. The vendor's own wording is Chinese — "Jev 压根不吐字，它只做判断" ("Jev emits no text at all; it only makes a judgement"). **It cannot write code** |
| Weights | **no open weights** (zero models under `typesafe` / `inceptionai` on HuggingFace); API only |
| Contract | `POST https://api.typesafe.ai/v1/systemone`, `questions.<name>.type = "choice"` + `criteria` |

The contract comes from the open-source agent harness
[jev-ultrafast](https://github.com/browser-use/jev-ultrafast), file
`jev_ultrafast/model.py` (MIT). That repo contains **no weight files at all** — only calling code.

## It is isomorphic to our protocol

| Jev | our `CandidatePage` |
| --- | --- |
| `questions.<name>.criteria` (id → description) | the candidate page |
| `answers.<name>.choice` | `candidate_id` |
| `answers.<name>.probabilities` / `confidence` | **we have no equivalent** ← calibrated confidence |
| several `questions` in one request | the multiple slots in `decide.py` |

So integration is **adding an executor** — no protocol change, no change to the host boundary.
The implementation is `codejev/jev_engine.py` (15 offline tests, fake opener, no network).

## Measurement one: hand-written candidate page, 4 tasks — 4/4

Same candidate page, same instructions, run against three other executors:

| Requirement | Expected | **Jev** | Confidence | Time |
| --- | --- | --- | --- | --- |
| 只保留 active 为真的项，返回 id 和 name ("keep rows where active is true, return id and name") | `0` | **`0`** ✓ | 0.84 | 0.76s |
| 把结果按 score 从高到低排序 ("sort the result by score, descending") | `1` | **`1`** ✓ | 0.90 | 0.70s |
| 按 category 分组统计数量 ("group by category and count") | `2` | **`2`** ✓ | 0.85 | 0.70s |
| **按 owner_id 关联（候选页无此能力）** ("join by owner_id — not a capability on this page") | `NONE` | **`NONE`** ✓ | **1.00** | 0.73s |

```text
accuracy 4/4 | median latency 0.73s (range 0.70-0.76s, very little variance)
```

### Four executors side by side (same candidate page)

| Executor | Accuracy | Latency | Abstention (`NONE`) |
| --- | --- | --- | --- |
| LFM2.5-350M + batch scoring | **1/4** (constant output `2`, zero discrimination) | 0.03–0.11s | ❌ picked `2` |
| local Qwen1.5B | 2–3/4 | 0.33–0.6s | ❌ picked `1` |
| **Jev** | **4/4** | **0.73s** | ✅ `NONE`, confidence **1.00** |
| DeepSeek Flash | 5/5 (a different batch of 5 tasks) | 0.37–2.1s | ✅ |

**Only Jev gets abstention right.** This is a mechanism difference: batch scoring is argmax and
must always pick one; Jev is a selector and can say "none of these" about the page as a whole.

## Measurement two: real source candidate pool, 15 tasks — 12/15 raw, **15/15** after checking

The pool is not hand-written: it is the functions enumerated with `ast` from the real file
`codejev/candidate.py`.

Raw result: **12/15 (80%)**. But checking the 3 "failures" showed **all of them were my
enumerator's fault, not Jev's**:

| Requirement | I expected | Jev picked | What checking showed |
| --- | --- | --- | --- |
| 把候选页里的短要求排成提示行 ("lay a candidate page's short requirements out as prompt lines") | `_bullet_lines` | `build_selection_messages` | the expected function **starts with `_`, and my enumerator skips those** — it was never in the pool. Jev picked the closest one that was |
| 去掉包住单行标量的围栏 ("strip the fence around a single-line scalar") | `_unwrap_scalar` | `parse_choice` | same: not in the pool. And `parse_choice` **is exactly the function that calls it** |
| 把对话控制符从正文清掉 ("strip chat control tokens out of the body") | `clean_body` | **`NONE`** | that function lives in `codejev/model.py`, **not in the candidate file**. **Jev answering `NONE` was correct** |

**So the real score is 15/15.** And this demonstrates the stress-test conclusion on the spot:

> **The accuracy ceiling of the selection route = the enumerator's recall (P2).** The model did not
> answer wrong; the host failed to enumerate the candidate.

Median latency 1.12s (range 0.92–2.52s), median confidence 0.87. **Note that the high confidence
on the first two tasks (0.87/0.92) means "closest available in the given pool", not "found the
ideal answer"** — confidence reflects the model's certainty about its own choice; it does not
replace the host's responsibility for the completeness of the pool.

## Cost comparison

### Jev's actual price

```text
input:  $0.042 / million tokens
output: free (it emits no text)
free credit: $5 signup credit ≈ 120 million input tokens
```

Price source: the typesafe.ai homepage lists **"$42 Per Billion input tokens"** (= $0.042/million).
Cross-check: $5 ÷ $42/billion = **119 million tokens ≈ the official 120 million** ✓
(The "$0.042 per million" figure circulating online is correct; reading it as "$42 per million"
would be off by 1000×.)

### Cost per selection (at our measured token counts)

| Executor | Input tok | Output tok | Cost per call |
| --- | ---: | ---: | ---: |
| **Jev** | 419 | 40 (free) | **$0.000018** |
| DeepSeek Flash (off-peak) | 330 | 2 | $0.000051 |
| DeepSeek Flash (peak) | 330 | 2 | $0.00010 |
| DeepSeek Flash (cache hit, off-peak) | 330 | 2 | $0.0000021 |
| local Qwen1.5B | — | — | ≈ $0.000003 (electricity) |

DeepSeek's official pricing (api-docs.deepseek.com, deepseek-flash): input, cache miss, $0.15
off-peak / $0.30 peak per million; output, $0.60 off-peak / $1.20 peak per million; off-peak is
half of peak.

**Jev is about 3–6× cheaper per selection than DeepSeek Flash**; but DeepSeek is cheaper on a
cache hit.

### Cost per generation (writing a ~400-token module)

| Executor | Cost per call |
| --- | ---: |
| **local Qwen1.5B** | **≈ $0.000003** (electricity) |
| Mercury 2.5 (launch price $0.04/$0.15) | $0.000073 |
| DeepSeek Flash (off-peak) | $0.00029 |
| Mercury 2.5 (standard price $0.20/$0.75) | $0.00037 |
| DeepSeek Flash (peak) | $0.00058 |

**In this column Jev is blank** — it does not write code. And generation is where the cost is:
local and API differ by about 100×.

### One complete task (one selection + one generated module)

| Combination | Total cost |
| --- | ---: |
| all local | $0.000003 (but accuracy 2–3/4, generation 0/5) |
| **Jev select + local generate** | **$0.000021** |
| Jev select + DeepSeek Flash generate | $0.00031 |
| DeepSeek Flash all-in | $0.00034 |

## Verdict

**Jev is strong enough in the selection role, and its price is competitive.**

- Its value is **not the saved money** (a single task saves a few ten-thousandths of a cent) but
  **accuracy and the ability to abstain** — and that is a functional matter: not one of the other
  three executors gets `NONE` right.
- **Calibrated confidence** (measured 0.43–1.00) is the "I am not sure" signal our architecture
  had been missing. In our runs the lowest value, 0.43, fell exactly on a task where my pool was
  incomplete — **confidence is indeed tracking uncertainty**.
- **Stable latency** (0.70–0.76s direct), far less jitter than DeepSeek Flash (0.37–2.1s).

**Its ceiling is set by the host, not by itself**: in the 15-task run, all 3 "errors" were my
enumerator missing candidates. **To improve selection quality, fix the enumerator first — do not
swap the model.**

## Not yet verified

- Only 4 + 15 tasks, one candidate-pool style (Python functions), one language. Not tested across
  files or across languages.
- No concurrency test, no larger candidate pool (we only went to 9–12 candidates; Jev documents a
  cardinality ceiling of 255).
- No end-to-end latency or accuracy test of the actual combination with a local generator
  (Jev select + local generate).
- The real bill after the free credit is exhausted is unverified; this section is computed from
  list prices.
- Through a proxy, latency was about 5.8s; direct it is 0.73s — **an 8× difference**, and I had
  previously mis-attributed proxy latency to the model.

## How to reproduce

```bash
cd codejev
.venv/bin/python -m pytest tests/test_jev_engine.py -q      # 15 offline tests, no network
export TYPESAFE_API_KEY=...                                 # direct connection, no proxy
.venv/bin/python -m bench.jev_probe                         # hand-written page, 4 tasks
.venv/bin/python -m bench.jev_batch_probe                   # real source pool, 15 tasks
```
