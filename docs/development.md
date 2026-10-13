# 开发与评测流程

新功能在 `codex/` 分支和独立 worktree 开发，通过 PR 审查后再合并。实验阶段保持现有生成参数不变，不在生产 checkout 更新 ComfyUI、安装节点或重启 worker。

## 无 GPU 的 CI

```bash
python3 -m pip install -r requirements/test.txt
bash bin/check
```

如果测试解释器不在 PATH 中，可以使用 `CHECK_PYTHON=/path/to/python bash bin/check`。该入口禁用 `.env` 加载并清空 Telegram 凭据，执行 Python 编译、单元测试、Shell 语法和 Git 空白检查。测试使用临时图片和模拟的 staging，不启动 Bot，也不访问 ComfyUI。

Actions 在 PR、`main` 和 `codex/**` 分支推送时执行相同检查，覆盖最低支持版本 Python 3.10，以及 3.13、3.14。额外通过固定版本、校验 SHA-256 的 actionlint 校验工作流。权限仅为 `contents: read`，不使用仓库 secrets、生产机器或 GPU。失败和取消都不能通过 `Required checks` 汇总任务。

仓库管理员可以在分支保护或 ruleset 中要求 `Required checks` 成功。这个 PR 不修改远端仓库设置、不自动合并或部署；首次分支 CI 通过后即可创建 PR。PR 事件会再次执行检查。

## 第一轮：准备 SageAttention 对照

这轮只比较注意力后端，保持模型文件、软件版本、画布、时长、提示词、seed 和采样参数一致。使用非露骨样例；不要将私人图片、提示词、实验 JSONL 或输出提交进 Git。

准备本地 JSONL，每行一个场景，图像路径必须存在。例如：

```json
{"image_path":"/data/evaluation/portrait.png","prompt":"A fully clothed adult turns toward the camera and waves. Audio: quiet room ambience.","seconds":5,"accelerated":true,"width":576,"height":768}
```

为了观察时长影响，可以为同一场景分别添加 5 秒和 15 秒条目。不要在同一轮同时换模型、增加步数或修改分辨率。

在开发 worktree 中生成配对任务：

```bash
python3 -m tools.prepare_ab /data/evaluation/scenes.jsonl \
  --output-dir .runtime/evaluations/attention-run-01 \
  --seeds 42 314159 271828
```

生成内容：

```text
attention-run-01/
  plan.json
  sage_on/jobs.jsonl
  sage_off/jobs.jsonl
```

两份任务文件完全一致，输入中的 seed 会被指定的 seed 列表替换；每个场景各生成一条对应每个 seed 的任务。`plan.json` 保存任务索引、场景、seed、图像内容哈希、画布尺寸、任务文件哈希，以及两个组的 worker 环境设置。工具不会启动服务、修改配置、联系 Telegram 或提交 GPU 任务；如果输出目录已存在则拒绝覆盖。

## GPU 评测需要独立运行环境

`MINIMAX_USE_SAGE_ATTENTION` 是 worker 启动设置，不是每条任务字段。直接把两份 JSONL 提交给同一组现有 worker 不构成有效的开关对照。

实际运行前需要准备独立的 ComfyUI checkout、Python 环境、端口范围、日志、输出和输入目录。模型文件可以共享只读路径。显式指定实验用 `MINIMAX_ENV_FILE`，设置 `MINIMAX_ENABLE_TELEGRAM=0`，并在两组分别启动时设置 `MINIMAX_USE_SAGE_ATTENTION=1` 或 `0`。独立端口仍不会隔离 GPU；选择无其他计算进程的 GPU，或顺序运行两组且只停止实验 worker。

启动前核实两组使用相同的 ComfyUI 提交、Turbo 节点提交、Python/CUDA/PyTorch 版本及模型文件哈希。保存启动日志，确认实际注意力后端：SageAttention 层可能回退到 PyTorch，环境开关并不能证明所有层都使用同一种实现。不要直接从生产 `.env` 复制 Token 到实验配置。

提交时让提交器加载对应实验环境，再使用 `python -m minimax_h3.jobs JOBS.jsonl --results RESULTS.jsonl`。结果索引对应 `plan.json` 的 `pairs`，可按场景和 seed 配对。先分别完成预热并单独标记，再比较后续任务；同 seed 不保证不同后端输出逐像素一致。

## 评测记录与后续 PR

每组保留以下证据：

- 软件提交、模型哈希、启动参数、实际后端和输入任务哈希。
- 完成数、失败数、失败原因；完成任务的排队、生成和总耗时。
- 独立采集的峰值显存；现有结果 JSONL 不包含这一项。
- 同一场景和 seed 的人物一致性、运动、片尾变形、纹理稳定性、语音和音画同步人工评分。

生成结果 JSONL 已包含耗时字段，但不要把排队耗时当作采样速度。只有 CPU CI 通过、GPU 功能验证通过，并且对照证据支持收益时，才提出默认参数变更。PR 可以提供汇总表和经过选择的公开非敏感示例，无需上传私人素材。

后续拆分为独立 PR：新模型预设与匹配的采样设置、按用户保存偏好、结果复用、任务级取消，以及 Ref2VA/续接。新增社区节点时同时更新服务的节点白名单；当前 worker 只放行 `ComfyUI-MiniMax-H3-Turbo`。不要将安装节点、修改后端、换模型和交互重构混成一次画质对照。
