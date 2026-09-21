# 参考项目、论文与选型依据

核查日期：2026-09-19。本文把当前资料中的事实与本项目建议分开；没有实测模型性能，也未证明这一组合已经具备商业收益。

## 1. 可选参考，不要求全部接入

| 用途 | 第一选择 | 选择原因 | 还要验证 |
| --- | --- | --- | --- |
| 本机展开基线 | Qwen2.5-Coder-1.5B-Instruct + MLX-LM 4bit | 小尺寸代码模型；Apple Silicon 有针对性工具 | 真实规格任务的质量、内存和完整延迟 |
| 已知代码复用 | Loom 的选择/物化分离 | 减少重复生成，已有文件清单思路 | 当前项目边界、增量修改与授权机制 |
| 候选检索 | universal-selector 的向量预计算与 top-k | 提供低成本候选筛选思路 | 候选是否满足精确语义；批处理实现 |
| 宿主协议边界 | jev + what-to-pick-today | 有限动作、严格校验、宿主身份字段 | 迁移到文件操作后的状态和授权设计 |
| 扩散代码研究 | Dream-Coder + Fast-dLLM + Block Diffusion | 代码任务、缓存并行优化、可变长度分别有参考 | 质量/步数权衡、许可证、实际硬件支持 |
| 专用训练框架研究 | MLX-LM；后续 dLLM | 前者用于本机小规模适配，后者提供扩散训练配方 | 模型/量化支持、内存和训练收益 |

当前只保留“大模型短指令 → 小模型产出 → 大模型判断 → 用户可决策”的流程。表中项目是按需查阅的参考，不是必须集成的组件清单；现成小模型先用，普通 SFT 可选，不做错误学习或复杂编排。

## 2. 用户指定的三个项目

### jev-ultrafast：把决策限制在观察到的候选里

- [仓库](https://github.com/browser-use/jev-ultrafast)，本次源码快照 `1231850`。
- [选择请求](https://github.com/browser-use/jev-ultrafast/blob/1231850/jev_ultrafast/model.py#L81)：一次请求包含操作选择及各操作目标问题，调用 TypeSafe 的 `jev-latest`。
- [返回值校验](https://github.com/browser-use/jev-ultrafast/blob/1231850/jev_ultrafast/model.py#L30)：校验候选身份、概率和一致性，执行时只消费被选操作对应的目标。
- [文本生成](https://github.com/browser-use/jev-ultrafast/blob/1231850/jev_ultrafast/model.py#L160)：单独调用文本模型，以 JSON 对象返回字段值并做严格检查。当前示例配置使用 Mercury 2.5；不能把示例配置等同所有运行路径的默认值。
- [预测与执行分离](https://github.com/browser-use/jev-ultrafast/blob/1231850/jev_ultrafast/agent.py#L86)：有页面新鲜度校验，但默认流程可自动执行，没有本项目保留的强模型确认环节。

可以借鉴有限动作空间、同请求并列问题、缓存明确上下文和确定性执行。不能据此推断 Jev 内部是扩散模型，也不能从仓库取得其训练权重。[性能文档](https://github.com/browser-use/jev-ultrafast/blob/1231850/docs/performance.md)是浏览器任务观测，不能外推代码生成速度。仓库有完整 [MIT LICENSE](https://github.com/browser-use/jev-ultrafast/blob/1231850/LICENSE)，不覆盖外部服务或模型。

### Loom：选择完成后由程序组装文件

- [仓库](https://github.com/ailiheizi/loom)，源码快照 `c299faf`。
- [选择与组装契约](https://github.com/ailiheizi/loom/blob/c299faf/client/src/contracts.ts#L170)：选择、引用、内容 hash 和生成文件字段提供了协议原型。
- [选择到计划](https://github.com/ailiheizi/loom/blob/c299faf/platform/plan_from_choices.py#L31)：程序将选择转为组装计划。
- [文件清单](https://github.com/ailiheizi/loom/blob/c299faf/platform/get_files.py#L154)：返回文件、依赖、环境变量等数据。
- [物化器](https://github.com/ailiheizi/loom/blob/c299faf/client/src/materialize.ts#L67)：写入已有组件；生成分支需要外部正文，不能解释为已训练代码生成器。

有现成实现时可以直接复用，不要求为此搭建检索或选择平台。现有物化主路径涉及重建输出目录，不应直接用于覆盖用户工程；协议里出现 hash 不代表已经实施审批或版本一致性校验。README 和包元数据声明 MIT，但已读快照没有独立 LICENSE 全文。详细资产与缺陷见[代码资产核查](reference-notes/code-corpus.md)。

### universal-selector：低成本候选选择

- [仓库](https://github.com/ailiheizi/universal-selector)，源码快照 `f014818`。
- [编码器](https://github.com/ailiheizi/universal-selector/blob/f014818/src/encoder.py#L7)：使用 multilingual MiniLM 编码文字。
- [选择器](https://github.com/ailiheizi/universal-selector/blob/f014818/src/selector.py#L38)：向量相似度与 top-k，JSON 是程序构造的结果。
- [训练](https://github.com/ailiheizi/universal-selector/blob/f014818/scripts/train.py#L61)：训练查询与候选描述的相似度，不是代码生成。

适合模板、API 组合、代码资产或展开策略的候选发现。候选语义正确性不能由相似度保证。已读多库/批量路径仍有逐条循环，不等同真正并行推理；仓库宣称的毫秒级表现本次没有重跑。README 声明 MIT，已读快照没有独立 LICENSE 全文。

## 3. 额外发现：用户自己的宿主协议实践

[what-to-pick-today](https://github.com/ailiheizi/what-to-pick-today) 的 [schemas.ts](https://github.com/ailiheizi/what-to-pick-today/blob/033f361998c4619bab4948e1bdb25baf970d2735/app/src/lib/harness/schemas.ts) 值得参考：候选解析与宿主提供的身份字段分开，能够避免模型输出伪造内部身份。

真实 commit 已核验为 `033f361998c4619bab4948e1bdb25baf970d2735`，对应 tree 为 `99c02e7c557b06e4f816f09ee0b48494a499ee90`。它是 UI/实验客户端源码参考，不是已经具备权限保证的文件执行器，也不是完整后端训练集。当前用途是借鉴边界设计；直接复制前仍需检查适用许可证。

## 4. 模型与本地工具

### Qwen2.5-Coder-1.5B-Instruct

[官方模型卡](https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct)明确为代码指令模型、因果语言模型架构；模型卡标注 Apache-2.0。它被选择为小模型实验基线，而非最新或最强候选的排他结论。

可先验证标准规格到正文的映射，再做针对性 SFT。模型卡上下文上限不等于 16GB 机器上的建议输入长度；长上下文的内存和延迟需要实际测量。使用/分发前核对权重的实际 LICENSE 和依赖条款。

### MLX-LM

[官方仓库](https://github.com/ml-explore/mlx-lm)，补充核查 commit `9d1e356e7cc6549e7d1697adabe2ea01ff8e062c`。本轮读取官方说明、LoRA 文档和相关 CLI 源码，确认其模型转换、生成、服务与适配训练路线有实际入口。

推荐作为 Apple Silicon 本机实验工具；本次没有安装或测试特定模型的兼容性。量化权重较小不代表训练峰值内存同样小，服务能运行也不代表任意组合训练都可行。具体资源判断见[本地方案](02-local-setup.md)与[训练路线](05-training-and-diffusion.md)。

### Gemini Diffusion 与 Mercury 2.5

- [Gemini Diffusion 官方页面](https://deepmind.google/models/gemini-diffusion/)：核查时仍标为实验文本扩散演示，解释整块生成和迭代修正。页面的采样速度与额外 overhead 分开报告；不应直接当成本地部署能力。
- [Mercury 2.5 官方发布](https://www.inceptionlabs.ai/blog/introducing-mercury-2-5)：提供 API，文档声称可调 reasoning、并行工具调用和 schema 对齐 JSON。可以作为未来远程展开对照，但本轮没有调用 API，公开资料也没有提供可据以自行训练的权重入口。

厂商 token 速度、不同硬件的实验数据和本机任务延迟不能直接横比。选用 API 不等于证明本地模型方案失败，二者应在不同部署条件下各自评测。

## 5. 论文及扩散实现

以下是原论文或项目的机制描述。即使来源采用强化学习、复杂采样或训练流水线，本项目也不因此采用这些机制；当前只保留其代码生成和速度方面的参考价值。

| 来源 | 本次读到的机制 | 对本项目的用途与边界 |
| --- | --- | --- |
| [Dream-Coder 7B 论文](https://arxiv.org/abs/2509.01142)与[实现](https://github.com/DreamLM/Dream-Coder) | 由自回归 checkpoint 适配离散扩散，结合 SFT 与可验证奖励，公开代码训练资料 | 最贴近代码展开的扩散参考；不保证少量去噪步下仍足够准确 |
| [Fast-dLLM](https://arxiv.org/abs/2505.22618) | 分块近似 KV Cache、按置信度并行解码；指出未经优化扩散可能慢于自回归 | 解释为什么“扩散”本身不是提速保证；论文提升不等于端到端任务提升 |
| [Block Diffusion](https://arxiv.org/abs/2503.09573) | 块级顺序与块内并行，支持缓存及可变长度 | 对长文件或分块输出有启发，仍需最终完整审核 |
| [dLLM](https://github.com/ZHZisZZ/dllm) | 统一训练/推理/评测配方，涵盖 LLaDA、Dream、AR 到扩散、Tiny-A2D 等 | 后续训练工程入口；提供配方不等于一键获得通用高质量小代码模型 |
| [XGrammar](https://github.com/mlc-ai/xgrammar) | 约束解码支持 JSON、正则和自定义文法 | 支持后端上的结构保证参考；不保证字段语义，也不能直接假定兼容任意扩散采样 |

补充读取的 Dream-Coder commit 为 `79d43878c55ba4e7474d5e0b6057d110b43acfcd`，dLLM commit 为 `ca176752fbceec49c6b4777a2c18ae88e4eb10ed`。Dream-Coder 官方示例使用 CUDA/BF16；本轮没有证明 Apple Silicon 适配。其权重、训练代码及数据集许可证应分别核验后再使用。

LLaDA、Dream、Tiny-A2D 在 dLLM 官方配方中可见。本项目没有对这些候选逐一做最新性能或完整权重许可审计，因此不把它们列为已验证的本机替代品。

## 6. 相关任务分工：Fast Apply

[opencode-morph-fast-apply](https://github.com/JRedeker/opencode-morph-fast-apply) 的文档描述：将原文件、修改意图和局部修改交给 Morph API，合成完整文件并生成 diff。

它说明“负责决策的模型＋专门完成代码编辑的模型”存在实现参考。代码合并可以利用大量未改变内容，这与从零生成新程序的难度和速度条件不同。已读的是集成项目说明，未运行插件、未验证 API，也未核对其性能数字；不将插件描述当成独立性能证据。借用概念即可，具体复制需另查 LICENSE 和服务条件。

## 7. 证据范围

- 发现渠道：前序使用用户的 Siftline 查询 GitHub 和 HN；HN 用于发现候选，不用于证明技术效果或付费需求。
- 原始材料：GitHub 源码/官方说明、Hugging Face 模型卡、arXiv 摘要、Google DeepMind 与 Inception 官方页面。
- 本轮补充：通过 gh 查用户公开仓库和源码，核查 MLX-LM、Dream-Coder、dLLM。另一次 Qwen 模型卡抓取超时，使用本次会话先前成功取得的原始模型卡副本，不把失败记成新的成功证据。
- 不可外推：未运行候选项目、未复现论文基准、未审计全部数据许可、未验证商业需求、未完成最新模型全市场比较。

来源会更新，后续开始实现时应重新固定实际使用的 revision、许可证和依赖。事实与推导的组合见[参考融合与收敛](09-synthesis-and-decisions.md)。
