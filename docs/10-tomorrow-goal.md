# 明天唯一验收 Goal：候选页选择器端到端跑通

## 目标

明天只验收一条最小链路，不继续扩展排序槽位、语言支持、扩散、蒸馏或自动训练：

```text
大模型写 CodeTask
→ 宿主准备一页 Python 候选代码
→ Qwen 小模型只返回候选 id 或 NO_MATCH
→ 宿主拿回候选代码
→ 大模型看到候选 id、代码和简短结果，做最终判断
```

这条链路验证的是当前项目最核心的假设：**把小模型当成无状态、低成本、无脑的需求/候选转换器，而不是 Agent 或决策者。**

## 明天要看到的具体结果

本次最小验收使用 `bench/candidate_tasks.json` 作为**主模型已经生成好的规范化 CodeTask 输入**：

- `source_request` 保存真实的自然语言需求，证明每条任务有来源；
- `operation`、`requirements`、`constraints` 是大模型规划后的 CodeTask；
- 本次不调用远端大模型 API，避免把 API 供应商、网络和价格引入最小协议验收；
- 这等价于先固定主模型输出，再独立验收小模型候选选择器。

明天若接入主模型，只替换 `source_request → CodeTask` 这一层，不改变候选页、小模型和宿主物化协议。

运行：

```bash
cd az-fls
HF_HUB_OFFLINE=1 .venv/bin/python -m bench.candidate_probe
```

预期终端会逐条打印：

```text
filter-active            expected=0       actual=0        PASS
sort-score               expected=1       actual=1        PASS
count-category           expected=2       actual=2        PASS
join-by-owner            expected=NONE    actual=NONE     PASS
filter-active-with-total expected=NONE    actual=...      FAIL/NO_MATCH
accuracy=...
```

## 验收标准

### 必须满足

1. `azfls/candidate.py` 的离线协议测试通过。
2. 全量测试通过：

   ```bash
   .venv/bin/python -m pytest tests/ -q
   ```

3. 候选 id、候选代码和候选身份全部由宿主掌握。
4. 小模型只返回一行候选 id 或 `NO_MATCH`。
5. 小模型输出解释、代码、路径、`approved` 或未知 id 时，宿主拒绝，不修复、不重试、不写盘。
6. `NO_MATCH` 返回给大模型，由大模型决定下一步，不由小模型自行生成新代码。
7. 真实本地模型探针输出原始回复、命中率和失败原因，不把失败伪装成成功。

### 不作为明天目标

- 不要求小模型达到生产准确率。
- 不要求明天完成蒸馏或 LoRA。
- 不要求明天完成 C2C/KV 通信。
- 不要求明天支持多语言。
- 不要求明天把候选页接入主 CLI 写盘流程。
- 不要求明天支持所有代码形状。

如果 Qwen 1.5B 在候选页选择上准确率不高，这仍是合格验收结果：它说明协议边界是对的，但通用模型还不是专用选择器。下一步再决定是增加候选摘要、换 API 模型，还是训练一个小打分器。

## 明天不再引入的复杂度

- 不加 CoT、长思考或隐藏推理字段。
- 不让小模型判断任务类型或是否启用槽位。
- 不做自动失败回收训练。
- 不做自动修复循环。
- 不做第二个 orchestrator 层。
- 不把模型返回的文件路径或审批状态当成可信数据。

## 完成后的判断

明天只回答三个问题：

1. 候选页协议是否能稳定传递“CodeTask → 候选 id”？
2. Qwen 1.5B 是否能在这个极窄任务上可靠选择？
3. 候选页失败时，`NO_MATCH` 是否能干净地交回大模型？

三个问题回答完，才决定后续是否：

- 把候选页接入主 CLI；
- 增加成功候选的复用库；
- 训练专用小打分器；
- 或改用便宜 API 模型作为执行器。

## 本次实际验收记录（2026-09-20）

### 测试

```text
.venv/bin/python -m pytest tests/ -q
299 passed in 1.09s
```

### 真实本地模型探针

模型：`models/Qwen2.5-Coder-1.5B-Instruct-4bit`，离线运行。
CodeTask 输入来自 `bench/candidate_tasks.json`，它是**大模型已经生成好的规范化 CodeTask 的固定替代**；本次没有调用远端大模型 API。每条任务同时保留 `source_request`，所以自然语言需求到 CodeTask 的输入证据仍然存在，但主模型转换层未在本次本地验收中执行。`expected` 与宿主物化代码用于本次确定性检查，**本次没有伪造一个远端大模型检查结果**；真实主模型检查留到接入 API 时再做。

```text
filter-active             expected=0    actual=1    FAIL
sort-score                expected=1    actual=1    PASS
count-category            expected=2    actual=2    PASS
join-by-owner             expected=NONE actual=1    FAIL
filter-active-with-total  expected=NONE actual=1    FAIL
accuracy=2/5 (40.0%)
```

探针同时打印每次的原始回复、prompt/generated token、耗时，以及宿主 `materialize()` 的候选代码。真实 Qwen 只返回了单个 id，没有输出路径、审批字段或解释；协议解析和宿主物化边界通过，但候选语义选择还不可靠：两个没有匹配候选的任务被错误强行选成了候选。

### 验收结论

- **协议通过**：CodeTask → CandidatePage → id/NO_MATCH 解析 → 宿主候选代码物化，链路真实跑通。
- **安全边界通过**：候选 id/代码由宿主持有；模型回复不获得路径、审批或写盘能力；`NO_MATCH` 不自动重试、不自动生成。
- **模型能力未通过生产门槛**：当前 Qwen 1.5B 在这 5 条固定任务上为 2/5，不能直接接入主 CLI。
- **下一步选择**：优先换一个更适合候选选择的 API/小打分模型重新测同一协议；若协议在更强执行器上稳定，再决定训练专用候选打分器。当前不做蒸馏、LoRA、自动训练或主 CLI 接入。

### API 执行器 A/B 对照（非本地 Goal 验收）

可复现实验脚本：

```bash
export https_proxy=http://127.0.0.1:7890
export http_proxy=http://127.0.0.1:7890
export all_proxy=socks5://127.0.0.1:7890
.venv/bin/python -m bench.candidate_api_probe
```

使用 CC Switch 已配置的 DMInfra OpenAI-compatible endpoint，模型为 `deepseek-v4-flash`，同一份 `candidate_tasks.json`、同一候选页、`temperature=0`，只替换执行器：

```text
filter-active             0     PASS
sort-score                1     PASS
count-category            2     PASS
join-by-owner            NONE   PASS
filter-active-with-total NONE   PASS
accuracy=5/5 (100%)
```

这说明当前协议和候选页可以被更强的执行器稳定使用；本地 Qwen 1.5B 的 2/5 是执行器能力边界，不是候选页/宿主协议边界。API 调用没有写入项目密钥，也没有改变本地 Goal 的 Qwen 结果。
