# SageAttention 本机初筛：2026-10-13

当前配置保持不变。这轮中 SageAttention 对完整生成耗时有小幅收益，未得到关闭它能明显改善画质的证据。数据仅来自一张公开人物图片、一个 seed 和两种时长，不能作为普遍性能结论或 NSFW 专项结论。

## 环境和输入

- 同一张 RTX 4090 24GB 顺序运行两组，开始每组前确认该卡无计算进程。驱动 580.178.04，Python 3.11.14，PyTorch 2.9.1+cu128，SageAttention 1.0.6，comfy-kitchen 0.2.28，comfy-aimdo 0.4.13。
- 独立 ComfyUI worktree，提交 `00d02f2854892ee5b9808bc2f6348b972017886a`；Turbo 节点提交 `55fee864dd7b2976b1c4ce3c3d5f7968f181409f`。独立解释器环境只读复用相同依赖，未升级生产依赖。
- FL2VA Pruned INT8 ConvRot、NVFP4 AWQ 文本编码器、FP16 视频 VAE、FP32 音频 VAE、Larry Turbo v4 step600 EMA。8 步、LoRA 强度 1.0、576×576、seed 42、24fps、预留显存 5GiB。
- 输入来自 scikit-image 的 [astronaut 示例](https://scikit-image.org/docs/stable/api/skimage.data.html#skimage.data.astronaut)，原始 NASA 图片无已知版权限制。使用完全着装、微笑、转头和简短英语对白的非露骨提示词。
- 名义时长 5 秒和 15 秒对应 H3 的 124 帧（5.167 秒）和 362 帧（15.083 秒）。每组先完成 22 帧的启动预热，不纳入下表。
- 每个组均独立启动 worker，仅切换 `--use-sage-attention`，模型、任务 JSONL 和其他启动参数一致。每组的日志分别确认 Sage 或 PyTorch 主后端，均未出现 `using pytorch attention instead` 回退日志；这不能证明所有层都没有未记录的回退。

## 测量结果

耗时来自 ComfyUI 的 execution_start 到 execution_success，包含条件编码、采样、解码和视频保存；不是纯采样内核计时。显存为每秒采样得到的整卡占用峰值，不是 PyTorch allocator 峰值。

| 名义时长 | Sage 开启耗时 | Sage 关闭耗时 | 开启后耗时降低 | 开启／关闭显存峰值 |
| --- | ---: | ---: | ---: | ---: |
| 5 秒 | 72.513 秒 | 78.516 秒 | 7.65% | 20168／20232 MiB |
| 15 秒 | 233.018 秒 | 270.311 秒 | 13.80% | 18760／18760 MiB |

两组各完成 2/2 个测量任务，无生成失败。排队耗时均低于 0.01 秒。对应的速度比为 1.083 和 1.160，中位数 1.121；它仅描述这两次配对。

四个视频均有 H.264 视频与 32kHz 双声道 AAC 音轨，视频帧数和尺寸符合请求。这是容器和输出完整性验证，不能证明对白清晰或音画同步良好。

## 抽帧观察与限制

对四个视频分别抽查首帧、中间帧和末帧。两组人物均可识别，5 秒片段的构图相近；15 秒片段中两组都出现了人物放大、构图变化和衣服徽章细节漂移。未在这次抽帧中观察到“开启 Sage 导致明显片尾崩坏、关闭即可修复”的差异。

抽帧不能评价连续运动中的闪烁、短暂畸变或语音质量。这里只做了可视检查和媒体探测，未给出主观量化评分，也未做完整音频听审。

一次短预热主要用于加载组件，可能仍包含首次目标长度的内核编译成本。两组按开启、关闭的固定顺序运行，没有交错随机化或重复测量；其他 GPU 上的任务和整机负载也没有被完全控制。显存差异可能来自动态加载和缓存，不能据此推断 15 秒任务比 5 秒任务更省显存。

下一轮画质验证应增加多个 seed、不同运动幅度和纹理场景，并使用目标长度的完整预热，交错重复两种后端。保留现有 Sage 开启的默认配置，避免根据这个小样本修改生产参数。

## 复现与证据

通过 `tools.prepare_ab` 生成同输入同 seed 的计划，使用 `tools.run_attention_ab` 运行，通过 `tools.summarize_attention_ab` 汇总。实验 worker 已停止。原始计划、模型哈希、启动日志、显存采样、结果 JSONL、抽帧和视频保存在实验 worktree 的 `.runtime/attention-results` 与 `.runtime/attention-plan`；生成媒体和机器路径不提交到 Git。

实际校验的模型 SHA-256：

| 组件 | SHA-256 |
| --- | --- |
| FL2VA Pruned INT8 ConvRot | `e889202c41dafb67b10d67b97f0d8541508036a6090af23425a5c2615d03c47a` |
| Qwen3-VL NVFP4 AWQ | `35a88d51044231fe332301d7a62aa81e3f2cba62febeb446e2c1e3e0ef76f2c6` |
| 视频 VAE FP16 | `7c1f131492e7eddacaac9069a61b81bdd39de5cc96561e677c5eab1cdce5e522` |
| 音频 VAE FP32 | `8e505d95dd1561d47abd43d4238fd40d9bb1ae9e147ed0a4cba778d76ae4db48` |
| Larry Turbo v4 step600 EMA | `5f3a626cd72c93a8b9318d6760c510bc5092d2ab13aaba1f932c5bab07a416d3` |
