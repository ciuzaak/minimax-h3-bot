# MiniMax H3 多 GPU ComfyUI 服务

一套围绕 [ComfyUI](https://github.com/comfyanonymous/ComfyUI) 和 MiniMax H3 的实用脚本：下载并校验模型、启动单卡或多卡图生视频服务、通过 JSONL 批量提交任务，以及可选地用 Telegram Bot 操作服务。

每张 GPU 会运行一个独立的 ComfyUI worker；提交器自动发现可用 worker，并优先把新任务分配给排队较短的实例。生成结果包含视频和模型生成的音频。

当前稳定版本：`v1.0.0`。

## 功能

- 从 ModelScope 下载 MiniMax H3 I2V / Ref2VA 权重并校验 SHA-256
- 自动生成 ComfyUI `extra_model_paths` 配置，无需修改仓库文件
- 使用 `tmux` 管理 1–8 个独立 GPU worker
- JSONL 批量提交，自动分辨率、随机 seed、队列调度和结果记录
- 支持 MiniMax H3 Turbo LoRA（8 步）与标准模式（默认 20 步）
- 可选 Telegram Bot：单图首帧或最多 9 张参考图、个人设置、选择时长和数量、查看进度及管理 worker
- 提供组件验证和端到端 smoke test

> 本项目是社区工具，不是 MiniMax 或 ComfyUI 的官方项目。模型权重受其原始许可证和使用条款约束。

## 适用场景

这个仓库适合已经能够运行 ComfyUI、希望把 MiniMax H3 图生视频变成可重复执行的本地服务的用户，尤其适用于：

- 一台机器上有多张 GPU，希望每张卡独立处理任务
- 需要从 JSONL 批量生成视频，并保留任务、耗时和输出记录
- 希望通过 Telegram 在手机端提交图片和提示词
- 需要固定模型文件版本并在启动前校验运行环境

它不是云端托管服务，也不负责安装 NVIDIA 驱动、CUDA、PyTorch 或 ComfyUI。单个生成任务只使用一张 GPU；多 GPU 用于并行处理多个任务。

## 环境要求

- Linux、NVIDIA GPU、可用的 CUDA/PyTorch 环境
- Git、`tmux`、`curl`、`sha256sum`
- 已安装的 ComfyUI
- Python 3.10 或更高版本
- 足够的磁盘空间与显存；实际需求取决于模型精度、分辨率、时长和 ComfyUI 配置

服务脚本会检查以下已知可用版本：

- ComfyUI 包含提交 `93cb5ed` 或更高版本
- `ComfyUI-MiniMax-H3-Turbo` 包含提交 `55fee864dd7b2976b1c4ce3c3d5f7968f181409f`
- Turbo LoRA 的 SHA-256 为 `5f3a626cd72c93a8b9318d6760c510bc5092d2ab13aaba1f932c5bab07a416d3`

如果只想使用脚本中的下载或基础验证能力，可以按需跳过 Telegram Bot。当前多卡模式是“一卡一个 worker”，并不会把单个任务拆分到多张卡上。

## 快速开始

### 1. 克隆并准备 Python 环境

```bash
git clone https://github.com/ciuzaak/minimax-h3-bot.git
cd minimax-h3-bot

# 下载模型所需；也可以安装进独立虚拟环境
python3 -m pip install -r requirements/download.txt
```

ComfyUI 自身及其 PyTorch/CUDA 依赖请按照 ComfyUI 官方文档安装。`COMFY_PYTHON` 必须指向能够正常运行该 ComfyUI 的 Python。

### 2. 配置 `.env`

```bash
cp .env.example .env
```

至少修改以下四项：

```dotenv
COMFY_DIR=/opt/ComfyUI
COMFY_PYTHON=/opt/venvs/comfy/bin/python
MODELSCOPE_PYTHON=/opt/venvs/modelscope/bin/python
MINIMAX_MODEL_DIR=/data/models/MiniMax-H3
```

路径配置说明：

| 变量 | 用途 | 未配置时的默认值 |
| --- | --- | --- |
| `COMFY_DIR` | ComfyUI 根目录 | 仓库同级的 `ComfyUI/` |
| `COMFY_PYTHON` | 运行 ComfyUI 与提交器的 Python | 当前 `python3` / Python 解释器 |
| `MODELSCOPE_PYTHON` | 运行下载脚本的 Python | 当前 Python 解释器 |
| `MINIMAX_MODEL_DIR` | H3 模型根目录 | `./models/MiniMax-H3` |
| `MINIMAX_OUTPUT_DIR` | 生成视频目录 | `./output` |
| `MINIMAX_LOG_DIR` | worker 与 Bot 日志目录 | `./logs` |
| `MINIMAX_BOT_INPUT_DIR` | Bot 临时图片目录 | `./bot_input` |
| `MINIMAX_MODEL_PATHS_CONFIG` | 自动生成的 ComfyUI 模型路径文件 | `./.runtime/extra_model_paths.yaml` |

`.env` 已加入 `.gitignore`，不要提交 Bot Token 或本机私密路径。也可以用 `MINIMAX_ENV_FILE=/path/to/another.env` 临时指定另一份配置。

### 3. 下载模型

```bash
set -a
. ./.env
set +a
"$MODELSCOPE_PYTHON" -m tools.download_i2v

# 可选：下载 Ref2VA 权重
"$MODELSCOPE_PYTHON" -m tools.download_ref2va
```

更简单的做法是先激活安装了 ModelScope 的环境，再直接执行：

```bash
python -m tools.download_i2v
python -m tools.download_ref2va

# 多图参考 Fast 的专用 4 步 LoRA（标准 20 步不需要）
python -m tools.download_reference_turbo
```

下载脚本读取 `.env` 中的模型目录，完成后逐个校验文件哈希。可通过 `MODELSCOPE_DOWNLOAD_PARALLEL_WORKERS` 调整并行数。

### 4. 验证安装

```bash
set -a; . ./.env; set +a
"$COMFY_PYTHON" -m tools.validate_i2v
```

如果当前 Shell 没有加载 `.env`，直接写出解释器路径即可。验证脚本会自动创建 `.runtime/extra_model_paths.yaml`，加载 diffusion model、text encoder、视频 VAE 和音频 VAE，并执行一个最小 conditioning 测试。

完整生成 smoke test：

```bash
set -a; . ./.env; set +a
"$COMFY_PYTHON" -m tools.smoke_i2v --lowvram --reserve-vram 5
```

## 启动服务

### 单个 ComfyUI 实例

```bash
./bin/start-comfyui
```

它默认使用 GPU 0、low-VRAM、CPU VAE，并把额外参数原样传给 ComfyUI：

```bash
CUDA_VISIBLE_DEVICES=1 ./bin/start-comfyui --listen 0.0.0.0 --port 8190
```

### 多 GPU worker 服务

安装 Turbo 自定义节点并把 Turbo LoRA 放到下列相对位置：

```bash
git -C "$COMFY_DIR/custom_nodes" clone https://github.com/larryvrh/ComfyUI-MiniMax-H3-Turbo.git
mkdir -p "$COMFY_DIR/models/loras/h3"
# 从 https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora 下载：
# minimax_h3_turbo_v4_step600_ema.safetensors
# 并放入 "$COMFY_DIR/models/loras/h3/"
```

然后运行：

```bash
./bin/minimax-h3-service validate
./bin/minimax-h3-service start 2
./bin/minimax-h3-service status
```

常用命令：

```text
start [N]             启动完整服务，使用 N 张当前空闲的 GPU
stop                  停止完整服务
restart [N]           重启完整服务
status                查看 worker 与 Bot 状态
workers-start [N]     只启动 worker
workers-stop          只停止 worker，保留 Bot
workers-restart [N]   调整 worker 数量
workers-status        查看 worker 状态
validate              检查 ComfyUI、Turbo 节点、LoRA 与 SageAttention
```

worker 从 `MINIMAX_BASE_PORT` 开始监听本机回环地址。默认 GPU 数量、最大数量、预留显存和 SageAttention 可以配置：

```dotenv
MINIMAX_BASE_PORT=8190
MINIMAX_DEFAULT_GPU_COUNT=1
MINIMAX_MAX_GPU_COUNT=8
MINIMAX_RESERVE_VRAM=5
MINIMAX_USE_SAGE_ATTENTION=1
```

设为 `MINIMAX_USE_SAGE_ATTENTION=0` 可关闭 SageAttention 检查和启动参数。

## 批量提交任务

复制示例并将图片改为真实的绝对路径：

```bash
cp examples/jobs.example.jsonl jobs.jsonl
./bin/submit-jobs jobs.jsonl
```

每行是一个 JSON 对象，必填字段为 `image_path`、`prompt`、`seconds`：

```json
{"image_path":"/data/input/example.jpg","prompt":"The subject waves at the camera. Audio: soft room ambience.","seconds":5,"steps":20,"width":576,"height":768,"seed":42,"accelerated":false}
```

- `seconds`：`0.2`–`15`
- `width` / `height`：必须同时提供，且为 32 的正整数倍；省略时按原图比例自动计算
- `seed`：可省略，届时使用随机 64 位 seed
- `steps`：标准模式默认 20
- `accelerated`：`true` 使用 Turbo LoRA，固定 8 步
- `sampler` / `scheduler`：可选的 ComfyUI 采样配置

多图参考任务使用独立的 Ref2VA 模型与有序 `image_paths` 列表：

```json
{"task":"reference","image_paths":["/data/input/person.jpg","/data/input/cat.jpg"],"prompt":"The person from <Picture 1> sits beside the cat from <Picture 2>. Static camera, quiet room ambience.","seconds":5,"seed":42,"accelerated":true}
```

`task` 省略时保持单图 I2V 行为。`reference` 接受 1–9 张图片，`<Picture 1>` 对应列表第一张。标准模式默认 20 步；`accelerated:true` 使用专用 Ref2V 4 步 LoRA、原生 `LoraLoaderModelOnly` 与 `res_multistep`，不会使用单图的 Larry LoRA 或自定义 Turbo sampler。`ref_image_size` 可选 `match`（默认，按输出像素面积缩小参考图，较快）或 `max`（保留更多参考细节，可能明显变慢）。输出宽高省略时按第一张图确定，其他图片保持各自比例。

默认等待所有任务完成，并在输入文件旁写入 `*.results.jsonl`。使用 `--no-wait` 可在入队后立即返回，使用 `--results PATH` 可指定结果文件。

## Telegram Bot（可选）

```bash
python3 -m pip install -r requirements/bot.txt
```

在 `.env` 中配置：

```dotenv
MINIMAX_ENABLE_TELEGRAM=1
BOT_PYTHON=/opt/venvs/bot/bin/python
TG_BOT_TOKEN=123456:replace-me
TG_ALLOWLIST=123456789,987654321
MINIMAX_DEFAULT_MODE=fast
MINIMAX_DEFAULT_SECONDS=5
MINIMAX_DEFAULT_DIALOGUE_LANGUAGE=Chinese
```

`TG_ALLOWLIST` 是允许使用 Bot 的 Telegram 数字用户 ID，多个 ID 用逗号或空格分隔。首次不知道 ID 时，可先启动 Bot 后使用 `/id` 查看，再立即补好白名单并重启。不要将 Token 提交到 Git。

Bot 使用 `/mode` 修改你自己的生成模式、时长（5、10 或 15 秒）和自动台词语言；`/language` 是同一设置面板的快捷入口。每位用户的偏好独立保存，并跨聊天生效；群聊中的设置按钮只允许打开面板的用户操作。语言可选中文、日语、韩语、英语或“无”，分别生成 `[Chinese]`、`[Japanese]`、`[Korean]`、`[English]` 标签；选择“无”时不转换或校验尖括号内容，方便手写多语言 `<d>[Language] ...</d>` 标签。设置会保存在被 Git 忽略的 `.runtime/bot_settings.json`，重启 Bot 后仍然有效。Fast 使用 Turbo LoRA 8 步，原始模型使用 20 步。模型、时长和台词语言在收到提示词时固定到该任务，不受之后的设置修改影响。

升级时，旧的共享设置作为所有用户的初始默认值保留；第一次修改偏好时，文件自动迁移为包含默认值和各用户偏好的版本 2 格式。升级前的设置按钮会提示重新打开 `/mode`。启动时若 Turbo 预检失败，会暂时按原始模型提交，不会覆盖保存的 Fast 偏好；组件恢复后重新选择 Fast 或重启 Bot 即可恢复。回退到旧版本前请备份设置文件，旧版本不支持各用户偏好格式。

### 多图参考

1. 在 `/mode` 的最后一行选择“多图参考”，然后发送相册或逐张上传图片。多图模式会追加图片，最多 9 张；默认“单图首帧”模式用新图片替换旧图片，并拒绝相册，避免只取其中一张。
2. 等待收图完成后的编号预览。相册按 Telegram 消息编号排序，下载完成顺序不会改变图号；收图过程中发送的提示词会提示稍后重试。
3. 用 `/images` 查看图片及编号，可删除任意图片、将图片上移或清空；`/images clear` 直接清空。修改后的图号以新预览为准，旧面板按钮自动失效。
4. 提示词写明每张图的用途。例如人物来自 `<Picture 1>`，猫来自 `<Picture 2>`。不要写 `<图1>` 或 `<Picture1>`，开启自动对白转换时它们会被当作台词。
5. 收到提示词时会固定该任务的全部图片、顺序、模式和时长；之后修改或清空参考图不会改变已经收到的提示词任务。输入集合保存在 `.runtime/bot_inputs.json`，重启后可恢复；该文件及图片均不进入 Git。

多图初始使用标准 20 步，在 `/mode` 可单独选择 Fast 4 步。单图和多图的 Fast 偏好分开保存，切回单图时仍使用原先选择的模式。4 步更快，但可能更锐化、丢失部分参考细节或偏离场景指令，重要结果可切回 20 步复核。

多图 Fast 使用 `minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors`，应放在 ComfyUI 的 `models/loras/` 根目录，或由额外模型路径配置暴露同名文件。先运行 `tools.download_ref2va` 和 `tools.download_reference_turbo` 准备权重。单图与多图的 Fast 预检和运行时回退相互独立；缺少多图 Turbo 时暂用标准 20 步，不覆盖保存的 Fast 偏好。没有 Ref2VA 模型时明确报错，不会悄悄改成单图生成。

模型原生还支持参考视频、视频配套音轨、独立参考音频以及风格提示词；此版本的 Bot 和 JSONL 新入口只接图片。视频或音频参考不能只靠写 `<Video 1>` / `<Audio 1>` 传入。[原生工作流说明](https://docs.comfy.org/tutorials/video/minimax/minimax-h3-native)

数量按钮提供 1–8。也可以在提示词末尾添加 `xN` 直接指定任意正整数数量并跳过数量选择，例如：

```text
人物转向镜头挥手，背景和光线保持稳定。Audio: quiet room ambience. x12
```

`xN` 不会进入最终提示词，每条任务仍使用不同的随机 seed。

如果不需要 Bot，留空 `TG_BOT_TOKEN` 或设置 `MINIMAX_ENABLE_TELEGRAM=0`，多 GPU worker 仍可正常启动。

## 目录结构

```text
bin/                          面向用户的 Shell 命令
  minimax-h3-service          tmux 多 GPU 服务管理
  start-comfyui               单实例 ComfyUI 启动器
  submit-jobs                 JSONL 批量提交入口
minimax_h3/                   核心 Python 包
  config.py                   .env 配置与模型路径生成
  jobs.py                     任务协议、调度与结果收集
  telegram_bot.py             可选 Telegram 前端
tools/                        下载、验证与 smoke test 工具
examples/                     可复制的任务示例
requirements/                 按用途拆分的 Python 依赖
```

运行时生成的 `.env`、`.runtime/`、`models/`、`logs/`、`output/` 和 `bot_input/` 均不会进入 Git。

## 开发与测试

安装测试依赖后，使用与 CI 相同的检查入口；不需要 ComfyUI、模型、GPU 或 Telegram 凭据：

```bash
python3 -m pip install -r requirements/test.txt
bash bin/check
```

GitHub Actions 在 PR、`main` 和 `codex/**` 分支推送时检查 Python 3.10、3.13、3.14、Shell 语法、工作流语法及 Git 空白。`Required checks` 汇总所有结果，失败、取消或跳过不能通过；主线合并需通过该检查。

CI 不自动部署或重启服务。模型与采样变更仍需单独验证实际 GPU 生成效果；实验工具、权重、私人素材和生成结果不提交。完整分支、检查与合并流程见 [开发流程](docs/development.md)。

## 常见问题

**提示找不到模型**  
确认 `.env` 中的 `MINIMAX_MODEL_DIR` 直接包含 `diffusion_models/`、`text_encoders/` 和 `vae/` 三个子目录，然后删除 `.runtime/extra_model_paths.yaml` 并重新启动（脚本也会自动覆盖为当前配置）。

**找不到可用 worker**  
先运行 `./bin/minimax-h3-service status`，并检查 `MINIMAX_BASE_PORT` 与 `MINIMAX_MAX_GPU_COUNT` 是否和提交器一致。日志位于 `MINIMAX_LOG_DIR`。

**没有空闲 GPU**  
服务只选择 `nvidia-smi` 中没有计算进程的 GPU，不会抢占现有任务。停止占用进程或减少启动数量后重试。

**Turbo 预检失败**  
核对自定义节点版本、LoRA 路径与哈希。标准批任务仍可通过普通 ComfyUI 实例执行，但当前 `bin/minimax-h3-service` 的 worker 配置包含 Turbo 运行时预检。

## 已知局限

- 仅面向 Linux 和 NVIDIA GPU，服务管理依赖 `nvidia-smi` 与 `tmux`，尚未提供 Windows、macOS 或容器化部署方案。
- 多 GPU 采用“一张卡一个 worker”的任务并行方式，单个任务不能跨 GPU 拆分，也不支持跨主机调度。
- JSONL 提交器和 Telegram Bot 支持单图首帧 I2V 与 Ref2VA 图片参考；参考视频、参考音频和首尾帧入口尚未接入。
- 相册在最后一个已知下载完成后等待约一秒再展示编号；极迟到的更新可能进入下一批。提交前请以最新预览确认图片集合。
- 切回单图时，如果集合仍有多张图，提交会要求删除多余图片或切回多图，不会静默忽略图片。切换模式、上传图片时不自动提交生成。
- `bin/minimax-h3-service` 会统一预检 Turbo 自定义节点和 LoRA，即使只计划提交标准模式任务也需要满足该运行时检查。
- Telegram Bot 尚不支持按用户设置配额或队列优先级。
- worker 调度主要依据 ComfyUI 队列长度和当前任务进度，不会根据显存占用、预计生成耗时或任务尺寸做资源预测。

## 安全与许可证

- ComfyUI worker 默认只监听 `127.0.0.1`。如果改为监听公网地址，请自行增加防火墙、认证和反向代理，不要直接暴露未经保护的 ComfyUI 接口。
- `.env`、运行日志、生成结果、Bot 临时文件和本地任务文件已被 `.gitignore` 排除。提交前仍建议运行 `git status`，确认没有误加入 Token、用户 ID、私人图片或本机路径。
- 如果凭据曾经进入 Git 历史，仅删除当前文件并不安全；应先撤销并轮换凭据，再清理历史。安全问题的报告方式见 [SECURITY.md](SECURITY.md)。
- 本仓库自行编写的代码采用 [MIT License](LICENSE)。MiniMax H3 权重、Turbo LoRA、ComfyUI、自定义节点及其他第三方依赖不包含在该授权范围内，仍分别遵循其原始许可证和使用条款。
