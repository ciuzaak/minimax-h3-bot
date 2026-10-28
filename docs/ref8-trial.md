# Ref2VA 8步试用档

基于 main `01820910ffc91f14e11b8c092c17dc8982ea14e0`，从完整实验分支中只保留参考图8步入口。I2V保持原实现，视频VAE保持FP16；原参考Fast4、标准20、用户设置及默认576画布算法不变。8步不会自动升到768。标准JSONL仍可指定宽高和步数，Bot标准档固定20步。

## 配方与来源

| 参考档 | LoRA | 步数 / sampler / shift(video,audio) |
| --- | --- | --- |
| Fast | ref2v 4step v0.1 | 4 / res_multistep / 12,3 |
| 8步试用（较慢） | ref2v 8step v1.1 768p | 8 / Euler / 6,3 |
| 标准 | 无 | 默认20 / res_multistep / 12,3 |

三档共用原Ref2VA基础模型、文本编码器和音视频VAE。8步固定simple调度、LoRA强度1，不叠加旧LoRA。

- [LightX2V发布记录](https://huggingface.co/lightx2v/Minimax-h3-Turbo/commits/main)：2026-10-20，revision `30f485ed874aa44c8183348ad361c5eabad318da`；文件 `minimax_h3_ref2v_turbo_8step_v1.1_768p_comfyui_bf16.safetensors`，1,956,193,000字节，SHA256 `e56863a7524e0300729cbb7b64cc8b548d90698c1abbf7d83898b0122fbe1e81`。
- [作者配方](https://github.com/ModelTC/Minimax-H3-Turbo)及[ComfyUI示例](https://github.com/ModelTC/Minimax-H3-Turbo/blob/main/COMFYUI_SETUP_AND_INFERENCE.md)：768p家族采用6/3，示例Euler。旧README表未单独列v1.1文件，因此此绑定继续标为试用，不能宣称作者已逐文件完整认证。
- 新文件已从ModelScope `lightx2v/Minimax-h3-Turbo` 直连获取并完整校验，与作者LFS哈希交叉核对。`python -m tools.download_reference_trial` 只下载这一份，已有合格文件复用；分片保留、校验后原子发布。旧4步下载入口保持不变。

## 使用与兼容

`/mode` 先选择多图参考，再选择Fast4、8步试用或标准20。参考档逐用户保存，I2V偏好独立；收到提示词时冻结档位，后改设置不影响已收任务。8步选项不会清除旧4步能力故障标志，也不会自动替换其他人的Fast选择。

JSONL使用 `task: reference` 和 `reference_mode: trial8`；显式档位优先于旧布尔 `accelerated`。省略新字段时保留旧接口：true为参考4步、false为标准。I2V拒绝参考专用字段，防止误套LoRA。所有结果与Comfy历史包含实际档位、基础模型、LoRA、步数、sampler、shift与新文件校验值。

只有选择8步时检查该文件完整SHA（包括管理目录和Comfy模型目录中同名副本）、ComfyUI v0.31.0+、完整图节点及必需文件。缺8步资产不影响旧4/20步。无需安装新自定义节点、升级共享Comfy或加入INT8视频VAE。

同时补两项小修复：下载I2V入口的Path导入；JSONL自动画布遵循JPEG EXIF方向，匹配Comfy LoadImage。

## 768方形质量对照

2026-10-28，在独占GPU2 RTX4090上，用隔离ComfyUI v0.43.0、PyTorch2.9.1+cu128、SageAttention、FP16视频VAE、静态显存、reserve5GiB、cache-none完成5条。复用本轮Qwen生成的虚构成年女性照片及白色陶瓷杯，不读取私人媒体、不新增下载。每组输入/提示词/seed20261028/768×768/5秒请求相同，输出124帧24fps；首请求进程冷、后续进程热，宿主与磁盘缓存未隔离。审计未见GPU2其他计算进程。

| 场景 | 旧4步 | 新8步 | 标准20步 |
| --- | ---: | ---: | ---: |
| 人物眨眼、轻转头、微笑 | 85.08s | 134.57s | 未运行 |
| 人物拿参考杯至胸前停住 | 89.70s | 143.28s | 321.93s |

5条均按124帧缩略图检查主体连续性，并查看14个较大关键帧及局部；这不是人脸识别评分或人工盲评。两档都大体保留脸、发型和蓝衬衫，未看到明确身份保留优势。新8步在人物组构图更稳定；拿杯组新8和20步都在胸口附近停住，旧4后段举至嘴边。杯子三档均从首帧出现，没有消失或复制；对白、多人及大幅运动不在本次验证范围。杯型与精细手部没有得到量化保真认证。

因此8步有作为可选中间档的价值：本次768参考样例比4步慢约58–60%，拿杯结果在动作终点/构图上更接近20步，耗时约20步的45%。这不代表整体画质等同20步、身份全面改善或所有参考图都会更好；步数、LoRA、sampler和shift同时改变，也不能归因于某一个因素。

## 原运行时与默认576验证

原共享ComfyUI `00d02f2854892ee5b9808bc2f6348b972017886a`、版本0.31.0与原配置COMFY_PYTHON，独立端口/输入/输出/user目录、空闲GPU2、SageAttention、FP16 VAE、reserve5GiB、默认动态显存、cache-none。相同人物+杯输入/提示词/seed，576×576、5秒124帧，旧4步73.488s，新8步101.326s（本例约慢38%）。首条进程冷、后条进程热，没有统计重复；耗时不是速度保证。两条均完整成功，逐帧缩略图未见主体消失或杯复制；172条进程审计未发现其他GPU2计算进程。

默认576未复现明确的整体质量收益：新8杯停得较低、更接近胸前，但构图更近，0–4秒头顶存在裁切，旧4保留完整头部；身份和杯型也未显示明确优势。它仍是显式试用档，不能把768的构图改善外推成默认576改善，也不静默升画布。原运行时另有一条此前已启动的768 Ref8兼容烟测成功（156.173s），不作为576或跨运行时性能比较。

共享Comfy源码、venv和生产配置未改动。自己的worker已停止、8299关闭、GPU2释放；原生产workers仍停止，Qwen保持不动。

## 检查与部署

64项离线测试、Python编译、Shell语法与Git空白检查通过。离线回归覆盖三档实际图路由、JSONL兼容、逐用户持久化与冻结、缺/坏权重不伤旧档、版本/节点失败、EXIF方向、下载复用与失败保留。Bot启动用单次0600私有文件交接已解析凭据，覆盖共享tmux继承别的token、显式调用者优先及新session/新window两条行为。`bot-start`只启动Bot，不启动GPU。

更新前先备份配置、个人偏好及输入集合，确认Bot没有尚未提交或等待交付的任务，并检查worker队列。可仅更新和重启Bot：停止原Bot后运行 `bin/minimax-h3-service bot-start`，核对新进程的运行目录、版本、凭据来源及启动日志。该入口不会启动GPU；worker原有运行或停止状态由操作者保留，其他服务不随Bot更新重启。回滚时恢复旧代码和配置备份，再仅重启Bot。

2026-10-28验证快照：原生产workers停止，实验worker已结束；完整实验分支与本地媒体保留供复核。
