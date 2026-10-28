# 2026-10 H3 可选更新与验证

基于 main `01820910ffc91f14e11b8c092c17dc8982ea14e0`。现有默认不变：I2V Larry 8 步、Ref2VA v0.1 4 步、FP16 视频 VAE、576p 均衡画布。此更新不部署、不重启现有服务、不修改共享 ComfyUI。

## 来源与取舍

核查日期：2026-10-28。

- [MiniMax 官方仓库](https://github.com/MiniMax-AI/MiniMax-H3) 与[模型卡](https://huggingface.co/MiniMaxAI/MiniMax-H3)：未发现新的官方 H3 基础模型版本；不要把衍生蒸馏模型称为 MiniMax 官方升级。
- [ComfyUI v0.43.0](https://github.com/Comfy-Org/ComfyUI/releases/tag/v0.43.0)：2026-10-27 发布，提交 `426f5eea52a7ece1dd599fff4cb6cd1f352ddd24`。用于隔离验证，共享 checkout 保持原版本。
- [LightX2V 文件发布记录](https://huggingface.co/lightx2v/Minimax-h3-Turbo/commits/main)：2026-10-20 提交 `30f485ed874aa44c8183348ad361c5eabad318da` 更新 FL2V 与 Ref2V 权重。该团队不是 MiniMax。采用最新 768p 8 步 v1.1 的任务专用文件作为显式试验选项；不把早期非 768p v1.0 当最新版本。优先8步便于与现有单图8步比较，并为参考和音频保留更多求值；新版4步家族留待后续单独评估。
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

## 验证环境与边界

2026-10-28 完成三份文件下载与完整 SHA-256 校验，均为 ModelScope 直连；短期 CDN 签名地址未记录或提交。再次运行本仓库下载入口确认三个文件均 `REUSED`，没有重复下载。新版 LoRA 每份 1.956 GB，INT8 VAE 2.811 GB，总计新增约6.72 GB；有效权重保留，任务自己的重复 Range/SDK 分片已清理。早期非768p v1.0下载的旧分片未擅自删除。

最终对照使用独占的物理 GPU2（RTX 4090 24GB）；完整 UUID、运行结果与进程记录仅留本地。GPU0 的早期记录不作为下表性能依据。GPU2 启动前无计算进程，统筹为本任务独占分配；中后段加入0.5秒进程采样，未观察到其他计算进程。早期匹配基线和VAE配对在连续进程审计加入前采集，依据前后检查和独占分配；逐条记录保留审计覆盖范围，不假称全程都有采样。

ComfyUI v0.43.0 / `426f5ee`，PyTorch 2.9.1+cu128，comfy-kitchen 0.2.40、comfy-aimdo 0.6.0，SageAttention，预留5GiB，静态显存，`cache-none`。仪表化提前导入执行器的早期动态显存失败请求已排除。cu128未启用cu130优化内核，结果不代表升级CUDA/PyTorch后的性能。没有修改共享环境、生产配置或服务。

固定合成纸船、蓝杯两份图片、seed `20261028`、5秒请求（124帧，24fps）。各任务内部在相同输入/提示词/seed/画布下比较完整配方，图像与音频联合生成。全部为方形576×576或768×768，不能外推为1344×768原生16:9的吞吐。每条完整请求一次，VAE同latent对照各两次。I2V旧576基线为进程冷启动，其后为进程热状态；`cache-none`仍重新执行loader，不是模型全常驻测试，磁盘缓存和宿主其它资源未隔离。

## 同卡完整请求结果

时间为秒，峰值为0.5秒采样的GPU总NVML MiB。conditioning包含文本与图像编码；sampler节点包含必要加载/卸载，不是纯DiT NFE时间；视频解码节点包含少量取样落盘。总渲染时间来自ComfyUI执行事件，排队/客户端总时间另存本地。每阶段的独立component计时和PyTorch分配峰值也已保留。

| 任务/配方 | 方形画布 | 步数 | conditioning | sampler节点 | 视频解码节点 | 总渲染 | NVML峰值 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| I2V Larry / FP16 | 576×576 | 8 | 7.94 | 69.16 | 9.90 | 92.46 | 22,030 |
| I2V新版 / FP16 | 576×576 | 8 | 7.06 | 62.76 | 6.24 | 80.68 | 21,106 |
| I2V Larry / FP16 | 768×768 | 8 | 9.46 | 121.26 | 12.78 | 149.42 | 23,226 |
| I2V新版 / FP16 | 768×768 | 8 | 8.08 | 108.19 | 11.28 | 133.55 | 21,994 |
| Ref2VA旧Fast / FP16 | 576×576 | 4 | 8.62 | 37.19 | 6.89 | 59.03 | 21,268 |
| Ref2VA新版 / FP16 | 576×576 | 8 | 7.96 | 66.30 | 6.37 | 86.80 | 21,268 |
| Ref2VA旧Fast / FP16 | 768×768 | 4 | 8.87 | 61.06 | 11.26 | 88.45 | 22,280 |
| Ref2VA新版 / FP16 | 768×768 | 8 | 8.88 | 114.10 | 11.14 | 141.47 | 22,280 |
| Ref2VA标准 / FP16 | 768×768 | 20 | 8.97 | 292.76 | 11.15 | 320.87 | 23,328 |
| I2V Larry / INT8 | 576×576 | 8 | 5.37 | 66.54 | 13.68 | 89.53 | 22,032 |

文本编码例：旧I2V576为4.89s、图像VAE编码3.02s；同图INT8完整请求为4.38s与0.96s。音频解码约0.39–0.60s。单条完整INT8请求总耗时89.53s，对比92.46s仅小幅缩短，且前者进程热、后者进程冷；不将约3%的差异视为可复现提速。端到端峰值22,032/22,030MiB基本相同，不能把解码阶段的显存节省当成整个生成过程的峰值下降。

## 相同latent的VAE对照

直接读取同一个GPU2旧I2V576 sampler输出，保持音频FP32 VAE、帧数与编码器配置。每轮前对私有worker执行模型释放，均在GPU上解码；`cuda:0`是`CUDA_VISIBLE_DEVICES=2`后的逻辑设备，物理卡为GPU2。INT8指权重量化，计算dtype仍为FP16。

| 视频VAE | 独立视频decode计时（两轮） | 完整解码图时间（含loader/音频/封装） | NVML峰值 |
| --- | --- | --- | ---: |
| FP16 | 5.78s / 5.86s | 9.27s / 9.46s | 6,736 |
| INT8 | 11.99s / 11.99s | 15.06s / 14.69s | 5,580 |

当前环境INT8独立视频解码约慢一倍，但此解码测试峰值少1,156MiB（约1.13GiB）。在0、24、60、100号帧上，相对FP16的PSNR为55.26dB、MAE为0.00138，两轮相同；这是同latent解码保真度，不是视频感知质量分数，更不证明真人/复杂纹理无损。

14份输出均检查为124帧/24fps、32kHz立体声，音频数值有限、非静音，无幅度接近满量程的样本。未进行对白、语言、音色或完整主观听感验收。

## 质量观察与默认选择

- 新版I2V在768方形画布用时133.55s，旧Larry为149.42s（本样例约少10.6%）；但约4秒处帆/船边可见重影，576样例水纹与色彩也发生变化，不能宣称质量全面提升。
- 新版Ref2VA在768首帧即包含纸船与蓝杯，旧4步首帧仅有纸船，杯稍后出现；标准20步也能从首帧包含两者。新版8步141.47s，旧4步88.45s，标准20步320.87s。它更接近质量档位取舍，不能将步数/shift/sampler一起改变的收益都归因于LoRA权重。
- INT8保真度在这份合成latent上很高，解码显存更低；当前cu128环境纯解码更慢、端到端峰值基本未降，因此不按上游cu130解码提速宣传改默认。
- 保留旧I2V/Ref2VA配方、FP16 VAE、均衡576p默认。新入口继续显式可选并带试验标记。真人参考、对白、其它seed、宽画幅、长运动与多参考组合未覆盖；后续若考虑改默认，需再由用户选择扩大验证范围。

## 代码/兼容验证与收尾

69项离线测试、Python编译、Shell语法和Git空白检查通过。覆盖旧用户设置、每用户分辨率、提示词收到时的画布/backend/VAE冻结、真实EXIF旋转JPEG、配方绑定与能力检查、同名坏权重拒绝、哈希缓存失效、下载复用/原子安装及失败后续传。没有向Telegram发送测试消息，Bot前端通过离线handler测试验证。

本机真实文件的下载复用通过，选择native/native/INT8的CPU服务预检通过；GPU图实际执行通过。最终GitHub Python3.10/3.13/3.14、工作流及Required checks以PR最新head为准。模型、实验脚本、输入、输出、UUID和进程日志保留本地，不提交。

所有本任务下载器/自动实验队列均已结束，GPU2私有worker与审计进程已停止；未停止其他GPU任务。当前模型权重已完整安装到受管模型目录，新分片缓存只剩目录/锁文件；旧非768p v1.0分片保留。实验材料保留供用户复核。

## 部署与回滚

本轮不合并、不部署、不重启生产；由统筹审阅PR后处理。现有默认和用户偏好未切换。将来部署先备份生产目录、配置与偏好，在隔离环境完成验收，停止接收新任务并排空队列，再切换Bot/worker；可恢复旧版本和备份配置回滚。不能直接更新共享ComfyUI或中断现有任务。
