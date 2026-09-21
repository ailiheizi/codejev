# 先用现成模型，普通 SFT 和扩散都可后置

当前路线很简单：**大模型发短指令，小模型产出正文，大模型判断结果并给用户简短结论。** 先用现成的小代码模型完成这个配合，不从零训练，也不把训练当作开始的条件。

## 如果以后做 SFT

只做普通监督微调，学习“简短指令＋必要原文 → 正确结果”。目标是服从给定修改、保留名称和字段、只输出所需正文。训练材料见[简单数据配对](03-data-pipeline.md)。不做错误学习、失败收集、修复回流或强化学习，也不训练小模型规划、审批或承担 agent 工作。

本机候选仍是 Qwen2.5-Coder-1.5B-Instruct。MLX-LM 提供 LoRA / 量化 LoRA 工具，可在 M1 Pro 16 GB 上考察小规模适配；具体能否运行取决于输入长度和实际内存，当前未验证。先用短样本做小尝试，没有理由直接扩大到全参数训练或复杂训练系统。

SFT 学的是任务映射和输出习惯，不会自动改变自回归模型逐步生成的机制，也不保证推理速度提高。

## 先用简单办法减少等待

优先让模型常驻、只提供相关原文、只生成所需正文，并由适配器补齐文件包装。明确的局部修改就返回约定的局部结果，避免重写无关内容。比较加入大模型判断后的完整等待时间；只有真正需要时才换更小模型或研究其他生成架构。

## 扩散只是备用资料

Dream-Coder 可参考代码扩散生成；Fast-dLLM 可参考缓存与并行解码；dLLM 提供模型转换和训练配方。它们不进入当前必做流程，也不引入其错误反馈或奖励训练路线。

扩散可以并行更新多个位置，但需要多轮计算，不能只凭“扩散”保证更快。现有 Dream-Coder 示例采用 CUDA，不能直接视为已支持这台 Mac。先保留相同输入输出约定，未来有实际加速需求时再替换小模型即可。

## 官方与论文参考

- [Qwen 官方模型卡](https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct)、[MLX-LM LoRA 文档](https://github.com/ml-explore/mlx-lm/blob/9d1e356e7cc6549e7d1697adabe2ea01ff8e062c/mlx_lm/LORA.md)：当前候选与可选普通 SFT 工具。
- [Dream-Coder 实现](https://github.com/DreamLM/Dream-Coder/blob/79d43878c55ba4e7474d5e0b6057d110b43acfcd/README.md)、[论文](https://arxiv.org/abs/2509.01142)：代码扩散资料。
- [Fast-dLLM 论文](https://arxiv.org/abs/2505.22618)、[dLLM 实现](https://github.com/ZHZisZZ/dllm/blob/ca176752fbceec49c6b4777a2c18ae88e4eb10ed/README.md)：未来并行生成与转换路线的参考。
