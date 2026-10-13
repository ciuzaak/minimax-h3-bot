# 开发与 CI 流程

在 `codex/` 分支和独立 worktree 开发，通过 PR 合并到 `main`。运行中的 Bot 和 GPU worker 使用生产目录；开发、CI 和模型实验不直接修改其环境或重启进程。

## 本地检查

```bash
python3 -m pip install -r requirements/test.txt
bash bin/check
```

可用 `CHECK_PYTHON=/path/to/python bash bin/check` 指定测试解释器。检查入口禁用 `.env` 加载，清空 Telegram 凭据，执行 Python 编译、单元测试、所有 `bin/` 脚本的 Shell 语法和 Git 空白检查。测试使用临时文件及模拟的 API，不启动 Bot、访问 Telegram 或加载 GPU 模型。

## GitHub Actions

PR、`main` 和 `codex/**` 分支推送自动执行：

- Python 3.10、3.13、3.14 的完整离线检查。
- 固定版本并验证下载哈希的 actionlint 工作流检查。
- `Required checks` 汇总任务，要求所有 Python 版本与工作流检查成功；失败、取消或跳过都不会通过。

Actions 权限为 `contents: read`，不使用 secrets、生产服务器或 GPU。checkout 和 setup-python 固定到提交 SHA。新推送会取消同一分支或 PR 的旧检查；合并应以最新提交的结果为准。

## 主线合并

`main` 配置分支保护，要求 `Required checks` 成功并与当前主线同步；管理员也受检查约束。通过 PR 合并，不直接推送主线。不额外要求其他人批准，以便个人维护的仓库仍能正常合并。

收到 `Required checks` 失败时，先打开对应 Actions run，定位失败的 Python 版本或工作流步骤；在同一开发分支修复并推送，等待新提交检查通过。不要将一个历史提交的成功结果当作当前 PR 的通过证明。

保护规则是仓库设置，不能仅靠提交工作流 YAML 自动建立。仓库管理员需保证设置中的必需检查名与工作流中的 `Required checks` 一致，并由 GitHub Actions 提供该结果。

## 模型验证与部署

CPU CI 验证代码与任务协议，不代表视频质量、声音质量或生成速度已通过。涉及模型、LoRA 或 sampler 的改动需要独立 GPU 验证证据，或在 PR 中明确仍未验证。

实验工具、权重、私人输入、提示词和生成结果保留本地。实验使用独立端口、输入、输出及环境，模型可共享只读路径；仅选择空闲 GPU，并只停止自己启动的实验进程。

CI 不自动更新生产目录、安装节点或重启 Bot/worker。合并后按已授权的部署范围更新；重启 Bot 时先停止接收新任务并等待应用已有任务结束，保留 GPU worker 可继续完成生成。更新仅涉及 CI 时无需重启运行服务。
