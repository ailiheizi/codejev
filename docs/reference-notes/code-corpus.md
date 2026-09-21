# 代码资产调查与可复用点

调查日期：2026-09-19。本文记录实际看过的源码和可用于新项目的思路，不把仓库 README 的性能宣称视作本次测量。没有运行这些项目，没有生成或下载训练数据集。

本文保留此前实际查看的源码事实，便于按需参考，不是当前方案的实现清单。当前主线是大模型短指令、小模型产出、大模型给用户简短结论；不要求建设检索、训练回收或复杂审批系统。

本次文档工作开始时，当前 `codejev` 目录没有源代码。本次没有遍历用户整块磁盘，也没有声称用户其他本地项目缺少代码。已有公开仓库已经能提供几个很具体的起点。

## 1. 查了什么

通过 gh 查询用户公开仓库列表，得到 18 个公开仓库及其描述、主要语言、许可证识别和更新时间。随后选择 Loom、universal-selector 和 what-to-pick-today 做定点调查。

前两个仓库复用此前的公开 archive 快照，未重复克隆。第三个获取 Git tree、README、`app/package.json` 和 `app/src/lib/harness/schemas.ts`，并用 `commits/HEAD` 核实完整 commit。只读取这些定点文件，没有下载整个第三仓库。

仓库列表中另有 Rust、Go、Python、Kotlin 等项目，但这次不扩大到多语言训练。第一期 TypeScript 函数/handler 已足以检验核心假设。

## 2. Loom：优先继承契约与确定性生成

来源：[ailiheizi/loom](https://github.com/ailiheizi/loom)。本次读取 archive 的 commit 前缀为 `c299faf`，本地快照目录为 `/tmp/codejev-repo-review/ailiheizi__loom/src/ailiheizi-loom-c299faf/`。未在这次调查中解析完整 commit；下列链接固定到该已知前缀。

| 已查看文件 | 实际机制 | 对新项目的贡献 |
| --- | --- | --- |
| [README.md](https://github.com/ailiheizi/loom/blob/c299faf/README.md) | 选择候选、组装、物化与回收代码 | 定义“已知资产复用”和“新增实现生成”的边界 |
| [client/src/contracts.ts](https://github.com/ailiheizi/loom/blob/c299faf/client/src/contracts.ts) | Zod 契约；候选接口、文件目标、来源、依赖等元数据 | 作为适配器数据结构的参考，避免模型自由发明字段 |
| [platform/get_files.py](https://github.com/ailiheizi/loom/blob/c299faf/platform/get_files.py) | 候选文件复制、锚点插入、标准模型及页面构建，返回 `files/path/content` | 建立模板/编译器对照组，确定的东西直接由代码生成 |
| [client/src/materialize.ts](https://github.com/ailiheizi/loom/blob/c299faf/client/src/materialize.ts) | 拷贝 base、落候选文件、注册片段、收集依赖；`generate` 使用外部传入的内容 | 小模型可以接到“缺失实现”的入口，但当前函数不是通用审批器 |
| [project-crud-router/files/project.ts](https://github.com/ailiheizi/loom/blob/c299faf/candidates/data.crud_resource/project-crud-router/files/project.ts) | Zod 输入、tRPC protected procedure、Prisma CRUD | TypeScript 后端展开样例和确定性 CRUD 基线 |
| [project-crud-router/meta.json](https://github.com/ailiheizi/loom/blob/c299faf/candidates/data.crud_resource/project-crud-router/meta.json) | 指明接口、目标文件、Prisma model 和注册片段 | 从代码提取 spec/context 的起点 |
| [generic-crud-factory/files/crud-factory.ts](https://github.com/ailiheizi/loom/blob/c299faf/candidates/data.crud_resource/generic-crud-factory/files/crud-factory.ts) | 通过已有工厂绑定具体数据模型 | 证明一部分“关键词组合”完全可以不用生成模型 |

源码给出一个很直接的对照：`project-crud-router` 展开了完整 CRUD，`generic-crud-factory` 把类似模式合并为工厂。第一期实验应同时保留两种确定性实现，测试快模型到底解决了哪类工厂无法覆盖的变化。

必须补充的边界：

- `materialize()` 会在已有输出目录上清空并重新复制 base；新适配器不能把这段行为原封不动套到用户工作目录。借鉴时只保留用户所需的改动和确认后执行，不复制整套目录重建流程。
- 该文件的 `adapt` 注释与实现表明，它仍依赖后续修复补胶水，不是任意代码修改器。
- `project.ts` 使用 `protectedProcedure`，但这个文件里的查询仅展示了排序或 id 条件，没有在该处加入 owner/tenant 条件。不能据此断言整个应用必然越权，也不能把它作为“当前用户只访问自己的项目”需求的已验证正例；那需要完整契约与行为测试。
- README 明确说明其部分成功信号来自 `tsc`，不等于功能完备。本次没有复现 README 中的比例或延迟。

许可证记录：README 声明 MIT，候选 meta 也有 MIT 字段；本次快照未发现独立根 LICENSE 全文，gh 的 `licenseInfo` 为空。它们是“已有声明”的证据，不能当作第三方候选与所有文件都完成来源审核的结论。

## 3. universal-selector：负责路由和召回

来源：[ailiheizi/universal-selector](https://github.com/ailiheizi/universal-selector)。本次读取 archive 的 commit 前缀为 `f014818`，本地快照目录为 `/tmp/codejev-repo-review/ailiheizi__universal-selector/src/ailiheizi-universal-selector-f014818/`。未在这次调查中解析完整 commit。

实际查看：[README.md](https://github.com/ailiheizi/universal-selector/blob/f014818/README.md)、[src/selector.py](https://github.com/ailiheizi/universal-selector/blob/f014818/src/selector.py)。

`select()` 编码 query，归一化向量，通过点积和排序选择候选；返回候选对象与分数。`select_multi()` 是循环调用 `select()`，这份实现没有一次批量编码所有请求。

它适合在新系统里做这些事：

1. 找出已有模板或候选组件。
2. 召回与规格有关的 API 契约、局部实现或示例。
3. 作为可选路由器，缩小需要快模型处理的任务范围。

它不负责生成未见代码，也不证明自己的相似度分数能判断“某个模板一定适用”。路由决策还需要兼容条件、类型约束和阈值校准；未知任务允许返回“没有合适候选”。

如果要批量化，可以在将来的实现里把 query 编码批处理、缓存候选归一化向量；这是新项目建议，不是这份源码已经做到的性能。本次没有运行 ONNX、计时或验证 README 的准确率。

许可证记录：README 声明 MIT License，gh 的 `licenseInfo` 为空，本次快照未发现独立根 LICENSE 全文。README 中提到的第三方素材必须单独保留来源与许可；素材许可不能自动由仓库声明替代。

## 4. what-to-pick-today：宿主掌握产物身份

来源：[ailiheizi/what-to-pick-today](https://github.com/ailiheizi/what-to-pick-today)。经 GitHub `commits/HEAD` 核验，本次完整 commit 是 `033f361998c4619bab4948e1bdb25baf970d2735`，日期为 `2026-08-01T16:25:30Z`；该 commit 对应 tree 为 `99c02e7c557b06e4f816f09ee0b48494a499ee90`。以下文件按固定 commit 读取和引用。

| 已查看内容 | 发现 |
| --- | --- |
| [README.md](https://github.com/ailiheizi/what-to-pick-today/blob/033f361998c4619bab4948e1bdb25baf970d2735/README.md) | 项目是并行生成、比较和挑选 UI 方案的 React + TypeScript 应用 |
| [app/package.json](https://github.com/ailiheizi/what-to-pick-today/blob/033f361998c4619bab4948e1bdb25baf970d2735/app/package.json) | 使用 Zod `^4.3.5`、TypeScript `~5.9.3`；提供 build 和 harness test 命令。这里只核实声明范围，没有读取 lockfile 解析安装版本 |
| [app/src/lib/harness/schemas.ts](https://github.com/ailiheizi/what-to-pick-today/blob/033f361998c4619bab4948e1bdb25baf970d2735/app/src/lib/harness/schemas.ts) | 校验 plan、candidate 与 review；限制路径与依赖；宿主注入身份字段 |
| Git tree | 看到 `test/bindings.test.mjs`、`test/harness.test.mjs`、`test/sandbox-runtime.test.mjs` 等测试路径；没有据此声称测试已读或通过 |

最值得继承的是 `parseCandidate()` 的身份边界：

- 模型只提供 `files`、`entryFile`、`previewProps`、`notes` 等内容字段。
- `id`、`componentId`、`variant`、`agent`、`attemptId` 来自调用方，在解析后由宿主注入。
- 注释明确说明 `attemptId` 不能从模型输出接受，避免模型冒充另一候选或新的一次尝试。

这与新适配器的关键设计一致：快模型不能自己指定审批状态、产物身份或新鲜度凭据。审批必须绑定宿主生成的实际内容和基线，不能信任模型输出的 `approved: true`。

这里的路径检查只是一个已有校验示例，不足以直接证明完整文件执行器安全；新系统还需要按自己的工作区、操作类型和提交方式检查实际解析路径。该项目是 UI 客户端，不应被描述为大量后端 handler 的训练来源。

许可证记录：gh 的 `licenseInfo` 为空，目录树未显示根 LICENSE。`app/package.json` 的 `private: true` 表示 npm 包发布配置，不等于代码是私有仓库，也不是许可证结论。

## 5. 当前只借这些

| 来源 | 当前可借鉴内容 |
| --- | --- |
| Loom | 已有内容直接复用，文件包装由程序完成 |
| selector | 手头已有候选时可以选择，不必搭建独立检索层 |
| what-to-pick-today | 小模型只给内容，宿主维护文件对应关系 |

首个尝试直接从现有项目挑一个局部修改：大模型给原文和明确改法，小模型返回结果，大模型给用户简短判断。上面的源码细节用于需要时查阅，不要求先建设数据平台、模板库或学习循环。

如果以后需要普通 SFT，可以取正常“指令加原文 → 正确结果”配对，保留来源，留些未用于训练的样例做简单检查。无需收集线上错误或修复过程用于学习。

当前完整想法见[交接说明](../10-handoff-brief.md)，简单检查方式见[验证方案](../04-evaluation.md)。
