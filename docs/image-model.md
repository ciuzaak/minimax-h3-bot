# MiniMax H3-Image

2026-11-11：接入原生 H3-Image 文生图和多图编辑。与通过视频模型抽取单帧的社区工作流不同，本模型使用独立图像 DiT、图像 VAE 和 6 步蒸馏采样。

## 24GB RTX 4090 配置

| 组件 | 选择 | 下载量 | 复用情况 |
| --- | --- | --- | --- |
| DiT | `minimax_h3_image_pruned_w4a8.safetensors` | 12.55 GB | 新下载，24GB 默认 |
| 文本编码器 | `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` | 0 | 与视频模型相同，校验后直接复用 |
| 图像 VAE | `minimax_h3_image_vae_fp16.safetensors` | 1.17 GB | 新下载；视频/音频 VAE 不兼容 |
| 可选 DiT | `minimax_h3_image_pruned_int8_convrot.safetensors` | 20.98 GB | 精度对照，需另外下载；24GB 上会增加卸载压力 |

W4A8 给激活值和参考图留出更多空间，BF16 文件约 40.23 GB，不作为单卡 24GB 默认。模型、编码器及 VAE 分阶段加载，不能把文件大小相加后当作峰值显存。8 张 4090 的显存也不自动合并成单任务的 192GB。

图像 worker 与原视频 worker 使用独立 ComfyUI checkout、Python 环境和端口，共享模型目录、输入目录及输出目录。图像 worker 默认自动选一张空闲 GPU，预留 3GB，开启 low-VRAM，使用 PyTorch SDPA；图像 VAE 使用模型内置的自适应解码。启动器同时检查实际显存占用，避免容器看不到宿主机 GPU 进程时误选忙卡。

## 参数档位

所有档位固定 **Euler / simple / 6 步 / CFG 1 / denoise 1**，使用权重内置 PDD sigma 网格，无负向提示词，无视频 Turbo LoRA。

| 档位 | 方形生成画布 | 用途 |
| --- | --- | --- |
| 快速预览 | 2048×2048，约 4MP | 只用于快速尝试构图，不代表最佳画质 |
| 均衡 | 2880×2880，约 8MP | 更多细节，代价高于预览 |
| 原生（默认） | 4096×4096，约 16MP | 质量优先，时间和显存代价最高 |

支持 1:1、4:3、3:4、16:9、9:16；按目标像素面积计算尺寸并对齐 32 像素。原生横屏为约 5472×3072。参考图越多，条件序列越长，耗时和显存占用越高；模型支持最多 16 张，但这不是 24GB 下所有图数、比例与分辨率组合均可运行的保证。遇到 OOM 先减少参考图或退回预览。

**编辑输出默认跟随图 1 的准确尺寸和比例**，按 EXIF 纠正后的显示方向判断；多张参考图以图 1 为编辑基底。档位用于确定生成质量：保持图 1 比例在较大的 32 像素网格画布上生成后，用 Lanczos 缩放回准确的输入尺寸。输出宽高不要求是 32 的倍数。文生图则按档位和所选比例决定输出。

在 Bot `/mode` 中显式选择“编辑：自定义画布”后，编辑输出才改为所选档位/比例的尺寸。CLI 对应 `--edit-size canvas`；也可同时传入 `--output-width`、`--output-height` 指定准确的交付尺寸。

官方建议 4–20MP、约 16MP 最佳；不应通过加到 20/50 步提高画质。当前参数来自官方模型卡和 ComfyUI 模板，4090 的运行时间及峰值显存应以本机 smoke 记录为准，不能套用 5090 社区数据。

## 提示词

描述完整成图、主体、构图、光线、材质及必须准确显示的文字。编辑时用 `<Picture 1>`、`<Picture 2>` 指定各图角色。提示词原样传入，不做视频台词转换，也不追加音频或背景音乐字段。

官方期望详细英文图像描述。可选官方 PE / Context-IR 或 ComfyUI 的 Qwen3.5 9B 扩写能帮助短提示词，但会增加权重、推理时间，可能改写编辑细节。本接入默认关闭扩写；短中文指令的效果不能等同于 MiniMax 在线服务的扩写后结果。

官方编辑模板默认启用扩写，本 Bot 当前未实现这一阶段，因此不能称为完整复现官方编辑流程。手写编辑提示词时参考 [官方编辑提示词指南](https://github.com/MiniMax-AI/MiniMax-H3-Image/blob/main/skills/h3-image-prompt-writing/references/edit-en.txt)：按图片顺序定义来源及 `<Subject N>`，再描述最终画面、每个主体保留或改变的关系。不要机械删除用户的风格词，也不要把一条社区经验当作所有任务的保证。

## 下载与安装

```bash
# 读取 .env 指定的目录；所有模型下载走 ModelScope。
"$MODELSCOPE_PYTHON" -m tools.download_image
# 可选：另外下载 INT8 对照，不影响 W4A8 默认值。
"$MODELSCOPE_PYTHON" -m tools.download_image --quant int8
```

下载器固定 ModelScope revision `b8ebf14be96f02e5c58960620a101de71b0bd9f3`，逐个校验 SHA256，已有且校验正确的文件直接复用。

图像服务需要包含 `MiniMaxH3ImageConditioning` 的 ComfyUI 0.45+。本次部署的隔离 checkout 固定为 `0228a9d`，不更新已有视频 checkout。为它创建独立 Python 环境并安装该 checkout 的依赖；CUDA 优化算子需要 PyTorch cu130 或更新的 CUDA 构建。不要直接升级视频环境中的 torch/comfy-kitchen。

```dotenv
MINIMAX_IMAGE_COMFY_DIR=/path/to/ComfyUI-image
MINIMAX_IMAGE_PYTHON=/path/to/image-venv/bin/python
MINIMAX_IMAGE_PORT=8290
MINIMAX_IMAGE_RESERVE_VRAM=3
```

```bash
./bin/minimax-h3-image-service start
./bin/minimax-h3-image-service status
./bin/minimax-h3-image-service stop
```

图像权重可以在本机从 ModelScope 下载，校验后 scp 到服务器的 `MINIMAX_MODEL_DIR` 同名子目录；先传到临时文件，服务器校验后再改为最终文件名。请勿将断点下载的分片当作成品。

## Bot 与 smoke

`/image` 或 `/mode` 的 H3-Image 按钮进入图像模式：无参考图时直接发提示词文生图；有参考图时进行编辑。`/images clear` 清空后恢复文生图。图像设置按用户保存，收到提示词时冻结；不覆盖个人视频档位。

保留原始输入像素时，请将 PNG、JPEG 或 WebP **作为文件上传**。预览会生成独立缩略图，模型读取原文件。照片方式以上传后 Telegram 实际传来的图片尺寸为准。原文件按正常 Bot API 的 20MB 下载限制处理。

`/mode` 选择分辨率、比例和量化；`INT8` 需要先完成额外下载，否则提交时会明确报缺少权重。`/service_start`、`/service_stop`、`/service_restart` 在图像模式下管理图像 worker；在视频模式下管理原视频 workers。`/progress`、`/stop` 覆盖两类服务。生成结果用 Telegram 文件发送原始 PNG，保留服务器输出。

```bash
python -m minimax_h3.image_jobs --prompt 'A detailed studio photograph of a red ceramic mug and a lemon on a wooden table, soft window light, no text.' --profile preview --seed 42 --results .runtime/image-preview.json
python -m minimax_h3.image_jobs --prompt 'A detailed studio photograph of a red ceramic mug and a lemon on a wooden table, soft window light, no text.' --profile native --seed 42 --results .runtime/image-native.json
python -m minimax_h3.image_jobs --prompt 'Edit <Picture 1>: change the ceramic mug to blue, preserve the lemon, composition, wooden table and lighting.' --image /absolute/path/to/generated.png --profile preview --results .runtime/image-edit.json
```

结果 JSON 记录 seed、实际宽高、精度、模型名、排队时间、执行时间和输出路径。测试状态见部署时写入的 `.runtime/image-*.json`；单元测试通过不等于 GPU 出图通过。

其中 `width` / `height` 是最终输出尺寸，`generation_width` / `generation_height` 是实际生成画布尺寸。编辑尺寸实测：1001×777 PNG 精确输出 1001×777；EXIF 方向 6 的 JPEG 精确输出显示方向的 777×1001。记录位于 `.runtime/h3-image-size-audit/`。

完整运行三个档位及一次参考图编辑，并以约 1 秒间隔采样该卡显存：

```bash
python -m tools.smoke_image --gpu 1 --results .runtime/image-smoke.json
```

`--gpu` 必须是图像 worker 实际使用的物理 GPU（启动日志可见，也可用 `tmux show-environment -t minimax-h3-image MINIMAX_IMAGE_GPU` 查询）。显存记录是采样最大值，不是 CUDA allocator 的严格瞬时峰值。

### 本机实测（2026-11-11）

GPU 0，RTX 4090 24GB，PyTorch 2.12.0+cu130、comfy-kitchen 0.2.41，W4A8 DiT + 复用 NVFP4 文本编码器，low-VRAM，reserve-vram 3。下表是单个静物提示词、每档一次的执行时间，包含编码、采样、解码与 PNG 保存，不包含 Telegram 上传；不是通用性能保证。

| 测试 | 实际尺寸 | 执行耗时 | 显存采样最大值 |
| --- | --- | --- | --- |
| 4MP 文生图（首次加载） | 2048×2048 | 24.79 秒 | 21464 MiB |
| 4MP 文生图（已加载，更换 seed 以实际重算） | 2048×2048 | 4.85 秒 | 21558 MiB |
| 8MP 文生图 | 2880×2880 | 10.39 秒 | 21524 MiB |
| 16MP 文生图 | 4096×4096 | 23.10 秒 | 21532 MiB |
| 单图编辑（红杯改蓝杯） | 2048×2048 | 15.28 秒 | 22350 MiB |
| 双图参考编辑 | 2048×2048 | 22.09 秒 | 22350 MiB |

以上全部完成且 PNG 尺寸正确；已目视确认 16MP 静物与单图编辑结果。量化 CUDA 算子已实际运行。多图组合目前实测到 2 张，16 张是模型输入上限，不能据此推断 24GB 显存下全部组合都可运行。

这些记录验证了运行链路和性能，**不代表画质验收通过**。原先 4MP 编辑图存在过度锐化与材质失真；颜色变化成功不足以证明编辑质量合格。后续画质核查见下节。

原始记录：`.runtime/image-smoke.json`、`.runtime/image-preview-warm.json`、`.runtime/image-multiref.json`。部署状态：`.runtime/h3-image-deployment-state.json`。

## 画质核查与修正（2026-11-11）

- 重新比对 [官方 T2I 模板](https://github.com/Comfy-Org/workflow_templates/blob/main/templates/image_minimax_h3_image_t2i.json) 和 [编辑模板](https://github.com/Comfy-Org/workflow_templates/blob/main/templates/image_minimax_h3_image_edit.json)：6 步、CFG 1、Euler/simple、独立图像 VAE 均一致。没有证据支持把正式配置改成 20/30 步或套用视频 Turbo 参数。
- 同 seed、同 latent 的普通 `VAEDecode` 与之前的 `VAEDecodeTiled` 入口，4096×4096 PNG 的像素完全相同（MAE=0，最大误差=0）。已改用官方普通解码节点以统一工作流；这项改动不是画质修复。
- 将原有 BasicGuider + SamplerCustomAdvanced 替换为官方模板的 KSampler（6 步、CFG 1、Euler/simple），在出现明显失真的原生参考图编辑样本上，结果同样逐像素一致。这组样本没有显示自定义采样图与官方采样图的差异。
- 保持 4MP 来源图和短编辑指令不变，只将输出改成 16MP，本例杯身和桌面过度锐化明显减轻，执行约 32.8 秒。已把 Bot 和 CLI 的默认生成档位从 4MP 改成 16MP；已明确保存的个人预览档位继续保留。需要较小成品时，可以先原生生成再缩图，不应把低分辨率直出视为相同画质。
- 同 seed 的 16MP 文生图中，单独删除 `realistic` 一词没有明显解决质感问题。该建议来自 [一位用户的实测](https://www.reddit.com/r/StableDiffusion/comments/1xsyifz/minimax_h3_image_t2i_tests_and_some_findings/)，未被当作自动改写规则。
- 按官方指南手写完整来源/Subject/保留关系，对这张静物改色只有较小变化。另一张原生 16MP 来源图编辑后依然出现明显纹理失真，说明提高分辨率和补充提示词均不能保证修复。
- [Comfy-Org 社区讨论 2](https://huggingface.co/Comfy-Org/MiniMax-H3-Image/discussions/2)、[讨论 3](https://huggingface.co/Comfy-Org/MiniMax-H3-Image/discussions/3) 和 [讨论 5](https://huggingface.co/Comfy-Org/MiniMax-H3-Image/discussions/5) 有类似画质与编辑一致性反馈，讨论 5 中也有 BF16 用户报告同类问题。这是用户报告，不是对本机问题的根因证明。
- 本机尚未进行图像 DiT 的 W4A8 / INT8 / BF16 同 seed 对照，因此既不将问题直接归因于 W4A8，也不宣称已经排除量化影响。当前 W4A8 的选择依据是 24GB 显存下的可运行性，不是已证明的最高画质。

核查原始记录与图片位于 `.runtime/h3-image-quality-audit/`。这些对照只覆盖一组静物和两个来源图；不能据此宣布模型的整体画质合格，或把调整后的参数称为普遍最佳。

上述原 4MP→16MP 对照发生在编辑尺寸跟随图 1 的修正之前。现在若要复现旧的指定画布输出，须显式传入 `image_edit_size="canvas"` 或 CLI `--edit-size canvas`；默认编辑已不再改变最终交付尺寸。

## 来源

- [MiniMax 官方模型说明：组件复用、6 步蒸馏、分辨率与提示词](https://github.com/MiniMax-AI/MiniMax-H3-Image)
- [ComfyUI 官方教程与采样设置](https://docs.comfy.org/tutorials/image/minimax/minimax-h3-image)
- [Comfy-Org 量化权重](https://huggingface.co/Comfy-Org/MiniMax-H3-Image/tree/main/diffusion_models)
- [ModelScope 下载源](https://modelscope.cn/models/Comfy-Org/MiniMax-H3-Image)
- [KOBO Lab 本人实测：W4A8、复用 NVFP4 编码器、分辨率与提示词扩写对照](https://localai-kobo.com/minimax-h3-image-comfyui/)（RTX 5090，结果不直接外推到 RTX 4090）
