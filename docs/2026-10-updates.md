# 2026-10 H3 可选更新与验证

基于 main `01820910ffc91f14e11b8c092c17dc8982ea14e0`。现有默认不变：I2V Larry 8 步、Ref2VA v0.1 4 步、FP16 视频 VAE、576p 均衡画布。此更新不部署、不重启现有服务、不修改共享 ComfyUI。

## 来源与取舍

核查日期：2026-10-28。

- [MiniMax 官方仓库](https://github.com/MiniMax-AI/MiniMax-H3) 与[模型卡](https://huggingface.co/MiniMaxAI/MiniMax-H3)：未发现新的官方 H3 基础模型版本；不要把衍生蒸馏模型称为 MiniMax 官方升级。
- [ComfyUI v0.43.0](https://github.com/Comfy-Org/ComfyUI/releases/tag/v0.43.0)：2026-10-27 发布，提交 `426f5eea52a7ece1dd599fff4cb6cd1f352ddd24`。用于隔离验证，共享 checkout 保持原版本。
- [LightX2V 文件发布记录](https://huggingface.co/lightx2v/Minimax-h3-Turbo/commits/main)：2026-10-20 提交 `30f485ed874aa44c8183348ad361c5eabad318da` 更新 FL2V 与 Ref2V 权重。该团队不是 MiniMax。采用最新 768p 8 步 v1.1 的任务专用文件作为显式试验选项；不把早期非 768p v1.0 当最新版本。
- [作者配方说明](https://github.com/ModelTC/Minimax-H3-Turbo)与[ComfyUI 示例说明](https://github.com/ModelTC/Minimax-H3-Turbo/blob/main/COMFYUI_SETUP_AND_INFERENCE.md)：768p 家族说明采用 video/audio shift 6/3，示例为 Euler；最新 v1.1 文件未在旧 README 表中单独列出。因此本项目绑定 8 步/Euler/simple/6/3，并标为试验，不能宣称已获作者对每份文件的完整参数认证。两种任务使用自己的基础模型与 LoRA，绝不交叉套用。
- [Comfy 官方 H3 指导](https://docs.comfy.org/tutorials/video/minimax/minimax-h3)与[打包权重](https://huggingface.co/Comfy-Org/MiniMax-H3)：INT8 视频 VAE 是显式选项，要求 v0.36.0+。官方解码提速说法不等于本机端到端提速。音频 VAE 保持 FP32。
- [FastH3 Omni 10月20日发布](https://haoailab.com/blogs/fastvideo-omniref-update/)、[Alibaba PAI PDD](https://huggingface.co/alibaba-pai/MiniMax-H3-Acc-LoRAs)、W6A8、VDN 与 Ref Patch 均留待第二阶段。其运行时、调度或模型族不同，不能依据其他硬件的速度声称改变默认。

## 固定配方

| task | accelerated | backend | LoRA | 步数 / sampler / shifts |
| --- | --- | --- | --- | --- |
| i2v | false | 任意支持值 | 无 | 默认20 / res_multistep / 12,3 |
| reference | false | 任意支持值 | 无 | 默认20 / res_multistep / 12,3 |
| i2v | true | legacy | Larry v4_step600_ema | 8 / 自定义自适应 AV sampler / 12,3 |
| reference | true | legacy | ref2v 4step v0.1 | 4 / res_multistep / 12,3 |
| i2v | true | native | fl2v 8step v1.1 768p | 8 / Euler / 6,3，试验 |
| reference | true | native | ref2v 8step v1.1 768p | 8 / Euler / 6,3，试验 |

调度器均默认 simple；标准模式允许调用者指定 sampler/scheduler。新配方不叠加其他加速 LoRA。`disabled` 禁用对应任务的 Fast，标准模式仍可用。

实际文件与 SHA-256 保存在 `minimax_h3/assets.py`。新 LoRA 每份 1,956,193,000 字节，INT8 VAE 为 2,811,065,184 字节；同时保留旧权重新增约 6.72 GB。来源分别为 ModelScope `lightx2v/Minimax-h3-Turbo` 和 `Comfy-Org/MiniMax-H3`，哈希与 Hugging Face 作者/Comfy-Org LFS 元数据交叉核对。

## 验证方法与限制

离线检查覆盖旧设置兼容、按用户保存与任务冻结、不同配方的绑定、标准模式不受可选 Turbo 缺失影响、同名坏权重拒绝、哈希缓存失效、失败下载不覆盖现有权重及断点保留。GitHub CI 不加载 GPU 模型。

GPU 实验仅使用空闲 GPU 0、隔离 ComfyUI、独立端口 8299 和合成输入；不使用用户私人媒体，不修改 GPU 7 或 Qwen 的 GPU 1。固定 seed `20261028`、5秒请求（124帧/24fps）、相同输入与提示词，在每个任务族内部比较 legacy/native 与 576/768 方形画布。记录节点、文本编码、视频编码、采样、音视频解码、总时间、PyTorch 分配峰值与定期采样的 NVML 显存峰值。采样节点耗时包含权重加载/卸载，不等同于纯 DiT NFE 时间。

运行时：ComfyUI v0.43.0，PyTorch 2.9.1+cu128，comfy-kitchen 0.2.40，comfy-aimdo 0.6.0，SageAttention，reserve-vram=5GiB。VAE 在 GPU 上执行。禁用动态显存：最初的仪表化启动提前导入执行器，造成 aimdo 初始化顺序错误，该失败请求不纳入结果。后续使用静态显存路径。此 cu128 环境会提示 cu130 优化内核不可用，不能代表升级 CUDA/PyTorch 后的最佳性能。

矩阵使用 `cache-none`，防止同 seed 的重复请求只命中结果缓存；标注进程冷启动与后续进程热状态，但每次仍重新执行 loader，不冒充全模型常驻的热启动测试。操作系统磁盘缓存未清空。INT8 VAE 另外以同一保存 latent 与 FP16 对照，避免把采样变化混入解码比较。实验代码、媒体、latent、详细日志保留本地，不提交。

### 已完成的兼容配方基线

以下是同一合成输入与 seed 的首轮样例，各行仅运行一次。576/768 为方形画布的实际边长。采样列是整个 sampler 节点时间（含必要加载/卸载）；视频解码列含少量取样落盘开销。

| 配方 | 画布 | conditioning（文本+图像编码） | sampler 节点 | 视频解码 | 总渲染时间 | 采样 NVML 峰值 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Larry I2V 8步 | 576×576 | 8.83s | 67.71s | 9.87s | 92.04s | 22,046 MiB |
| Larry I2V 8步 | 768×768 | 10.53s | 120.29s | 12.45s | 149.06s | 23,242 MiB |
| Ref2VA v0.1 4步 | 576×576 | 8.92s | 44.98s | 7.00s | 67.85s | 21,286 MiB |
| Ref2VA v0.1 4步 | 768×768 | 9.56s | 59.27s | 11.05s | 87.31s | 22,296 MiB |

这些输出均包含 32kHz 立体声音轨。逐帧检查可见 I2V 保住了平面纸船的主体与配色；多图结果包含纸船和蓝杯，但姿态、比例、画面风格与最初出现的时间存在差异。增大画布不是这个样例的稳定质量改进，不据此更换默认。一次无对白合成样例不能证明真人参考、对白或一般场景质量。

最新 Turbo 和 INT8 VAE 对照仍待权重下载完成。已完成的上述表格是旧配方基线，不是新模型的收益；候选对照补齐前保持 draft 和试验标记。合成纸船/蓝杯样例不覆盖真人面孔、对白、长时运动、极端比例或全部参考组合。

## 部署与回滚

PR/CI 不会自动部署。合并后仍需单独批准切换：保留生产目录、配置和偏好备份，在独立环境完成组件/端到端检查，停止接收新任务并排空队列，再切换 Bot/worker。恢复旧版本与备份配置即可回滚；不要在运行中的共享 ComfyUI 上直接更新。
