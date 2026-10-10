# 第五部分：Agent 与课程学习助手

这一部分从手写执行循环开始，围绕 Rules、Skills、Knowledge 完成一个课程学习助手：查询教程、给出引用、提供练习、检查数值答案、保存复习记录、恢复任务，并留下可检查的动作轨迹。

正文讲结构与必要片段，完整规则、技能、题库和代码保存在仓库附件。读完后应能为自己的领域定义任务、工具、权限、知识来源、状态和评价方法。

## 章节与产出

| 章节 | 内容 | 产出 |
|---|---|---|
| [20. Agent 的组成与运行过程](20-Agent的组成与运行过程.md) | 目标、工具、状态、三类附件 | 项目范围与验收条件 |
| [21. 编写第一个 Agent 循环](21-编写第一个Agent循环.md) | 动作协议、模型接口、结束预算 | 最小循环与真实适配层 |
| [22. Rules](22-Rules规则与执行边界.md) | 规则与权限、参数、注入边界 | 可执行的检查 |
| [23. Skills](23-Skills让任务方法可以复用.md) | 目录、按需加载、任务方法 | 解释、练习、复习技能 |
| [24. Knowledge](24-Knowledge读取检索与引用.md) | 分块、关键词、向量、引用 | 可追溯课程索引 |
| [25. 状态与记忆](25-状态记忆与多步骤任务.md) | 检查点、事务、恢复、计划 | 可恢复的任务与记录 |
| [26. 组装完整助手](26-组装完整课程学习助手.md) | CLI、HTTP、完整学习流程 | 可运行项目 |
| [27. 测试、评价与扩展](27-测试评价与扩展.md) | 真实失败、分层评价、领域迁移 | 核验与评价报告 |

## 附件与源码

[assets/part05/README.md](../assets/part05/README.md)列出规则、技能和题库全文。技能采用本项目自己的目录与加载约定，并不要求某个商业 Agent 产品。

核心源码：[循环](../script/part05/runner.py)、[模型适配](../script/part05/model.py)、[工具](../script/part05/toolbox.py)、[知识检索](../script/part05/knowledge.py)、[数据库](../script/part05/store.py)、[结果展示](../script/part05/presentation.py)、[命令行](../script/part05/cli.py)、[HTTP 入口](../script/part05/api.py)。

## 环境与依赖

进入本部分前做[Agent 入口自测](../附录/02-学习路线与前置自测.md#6-第五部分入口把数据决定和执行分开)。重点是嵌套 JSON、接口检查与实际工具执行；高级数学不是离线循环的门槛。第25章的状态复制与事务、第26章的服务入口可按需查[工程 Python 补读](../附录/03-读懂后续工程代码.md)。

还没有第四部分的服务环境时，按[环境准备与复现](../附录/01-环境准备与复现.md)准备；离线循环与工具练习可先运行，不需要 GPU 或模型权重。

统一使用 conda + VS Code，在仓库根目录运行命令。默认索引、循环和 SQLite 只用标准库；真实模型客户端需要 requests。已有 base 可以完成大部分操作，不新建环境，也不把 Torch 加载进 Agent 进程。

```powershell
conda activate base
python -m pip install requests==2.32.5 -i https://pypi.tuna.tsinghua.edu.cn/simple
```

若已经有可用 requests，无需重新安装。清华镜像也可用于 Matplotlib、FastAPI 等普通 Python 包；GPU Torch 和容器镜像不能直接套用这个安装地址。镜像来源与第四部分保持一致。

可选 HTTP 入口使用第四部分中新建的 `handmade-llm` conda 环境，其中已有 FastAPI 与 uvicorn。若复用作者已有环境，可将激活命令改为 `conda activate vllm`：

```powershell
conda activate handmade-llm
python -m uvicorn api:app --app-dir script/part05 --host 127.0.0.1 --port 8010 --workers 1
```

若你的环境没有这两个库，可安装本机验证使用的版本：

```powershell
python -m pip install fastapi==0.135.1 uvicorn==0.41.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
```

当前 HTTP 入口是本地单用户示例，learner 只是记录分组，不是身份认证。保持 127.0.0.1 和一个 worker。

## 准备知识索引

```powershell
conda activate base
python script/part05/index_course.py
python tools/verify_part05.py
python script/part05/21_minimal_loop.py
python script/part05/24_vector_demo.py
```

索引默认读取前四部分的19个章节文件，忽略代码围栏与图片。源文档变化后需要重建；知识块保存路径、标题、行号和内容 ID。分块按字符数，不按 token 数。

离线核验使用预设动作测试循环和工具，不能当作大模型成功率。完整报告见[verification.json](../outputs/part05/verification.json)。

## 启动真实模型服务

本机验证复用了第四部分的 WSL2 Ubuntu、RTX 4060 Ti 16GB、vLLM 镜像和 Qwen2.5-1.5B-Instruct 权重，因此本次没有拉取新镜像或下载整模型。读者已有资源时也可复用；首次准备则按[统一环境指南](../附录/01-环境准备与复现.md)拉取镜像与下载权重。模型路径以自己的实际目录为准。

进入 WSL：

```powershell
wsl -d Ubuntu
```

在 Ubuntu 终端执行，保留该终端运行：

```bash
docker run --gpus all --ipc=host \
  --name handmade-llm-part05-vllm \
  -p 127.0.0.1:8002:8000 \
  -v /mnt/g/models/qwen2_5_1.5b_instruct:/models/qwen:ro \
  -e HF_HUB_OFFLINE=1 -e VLLM_NO_USAGE_STATS=1 \
  vllm/vllm-openai:v0.20.1-cu129-ubuntu2404 \
  /models/qwen --served-model-name tutorial-agent \
  --dtype float16 --max-model-len 4096 --max-num-seqs 1 \
  --gpu-memory-utilization 0.4 --enforce-eager
```

容器已经创建但停止时，不重复 run，用：

```bash
docker start -a handmade-llm-part05-vllm
```

第四部分原容器保留，未修改其配置。这次新建 part05 容器使用 8002、4096 上下文和一个并发序列。输入与最多500输出 token 的总量需要在4096内；多轮大资料可能超出，错误会进入报告。

实际验证结束后只停止本部分容器，保留便于读者再次启动。WSL 在作者机器出现过会话启动警告，因此采用前台终端，不能仅因看到警告就认定模型启动失败。

实际环境版本、镜像 ID 和启动参数见[environment.json](../outputs/part05/environment.json)。Agent 客户端使用已有 base，HTTP 入口验证使用已有 vllm conda 环境；本部分没有调整推理库或下载模型。

在 Windows 检查健康：

```powershell
Invoke-RestMethod http://127.0.0.1:8002/health
```

返回成功后再运行 Agent。第四部分自定义 /generate 有适配参数，但旧512输入与128输出限制不足以容纳本项目，完整参考验证使用 vLLM 的结构输出。

## 运行完整学习流程

```powershell
conda activate base
python script/part05/cli.py "请根据教程解释梯度下降，给出出处。" --learner demo
python script/part05/cli.py "请给我两道梯度下降练习，先不要给答案。" --learner demo
python script/part05/cli.py "gd-01 的答案是 2，请检查并保存复习记录。" --learner demo --allow-record
python script/part05/cli.py "请查看我的最近学习记录。" --learner demo --require-tool progress
```

--allow-record 表示本任务要保存产生的评分。省略时不能写记录；开启且产生评分时，结束前必须完成 record。不要在“不保存”的任务中加这个参数。

默认题库有梯度下降两题、概率一题和点积一题。题库工具不返回答案；评分接受有限数值，拒绝表达式。多个题号与复杂文本答案应改用结构化答题入口，当前轻量抽取不是通用自然语言解析器。

暂停与恢复：

```powershell
python script/part05/cli.py "请解释梯度下降，给出处" --stop-after 2
python script/part05/cli.py --resume "输出的任务ID" --learner demo
```

恢复使用原任务权限。done、failed、exhausted 不会自动继续；原始报告与错误用于排查，网络失败不会被替换成预设成功回答。

本机还实际验证了CLI暂停与恢复，结果见[26_cli.json](../outputs/part05/26_cli.json)，同样使用临时数据库。个人数据库默认在 outputs/part05/learning.sqlite3；开发核验可用 AGENT_DB_PATH 指定隔离数据库。

## 核验与真实评价

```powershell
python tools/verify_part05.py
python script/part05/evaluate.py
python script/part05/26_demo.py
```

真实评价必须有上述模型服务，使用临时数据库，不污染个人记录。[最终报告](../outputs/part05/27_evaluation.json)保留原始输出与状态；[初始失败报告](../outputs/part05/27_evaluation_initial.json)对应改进前实现。

26_demo.py 在临时数据库中真实运行“解释—取题—批改保存—查询进度”四个任务，报告见[26_demo.json](../outputs/part05/26_demo.json)。CLI 的正常使用仍保存到个人本地数据库。

这个固定学习流程通过 required_tools 检查必需操作，避免模型未读记录就结束回答。CLI 用 --require-tool 指定同类要求；任务恢复不会改变已保存的要求或权限。

练习、评分和学习记录由入口直接展示工具事实，不拼接可能编造结果的模型说明；原始文本保留在状态与报告中用于诊断。解释类输出仍需核对引用和内容。

本机八条自动检查均通过，但人工检查发现学习率解释错误，详见[内容核对记录](../outputs/part05/27_content_review.json)。通过数只对应这些窄范围检查，不代表教学内容全部正确。

HTTP 入口还实际验证了422、429以及评分保存，报告见[26_http.json](../outputs/part05/26_http.json)。复现时先保持模型服务运行、停止自己占用8010的Agent入口，再在有FastAPI的环境运行 `python tools/verify_part05_http.py`。核验启动并清理自己的入口进程，使用临时数据库。

评价只有八条手写教学用例，部分检查工具边界，部分检查题库或保存结果，不能当作通用 Agent 基准。关键词命中、引用存在、JSON 合法都不能自动证明教学解释正确。

## 可选语义检索

默认没有额外嵌入模型，使用中文二字词元和英文词频检索。手工向量图仅说明余弦。

需要语义扩展时，在独立或已有合适的 conda 环境安装 Sentence Transformers：

```powershell
python -m pip install sentence-transformers -i https://pypi.tuna.tsinghua.edu.cn/simple
python script/part05/semantic_search.py --model-path "已有本地嵌入模型目录" --build
python script/part05/semantic_search.py --model-path "已有本地嵌入模型目录" --query "每步更新太大怎么办"
```

这是可选扩展，作者没有为本部分下载或验证一个新嵌入模型，因此不提供已测语义检索效果。库版本与模型结构兼容需要单独验证；避免把未固定依赖直接装进已经验证的推理环境。

## 文件与运行记录

```text
第五部分-Agent/         # 八章正文与本指南
assets/part05/          # 规则、技能、来源约定与题库全文
script/part05/          # 完整 Agent 与实验
data/part05/            # 真实评价用例
images/part05-*.png     # 原创 matplotlib 插图
outputs/part05/         # 索引、报告、个人数据库
```

个人数据库、last_task、索引缓存与服务日志由 Git 忽略；公开报告只保存教学用例和验证结果，不上传模型权重或读者个人记录。

复现图与正文检查：

```powershell
python tools/draw_part05.py
python tools/review_markdown.py
```

绘图需要已有 base 的 Matplotlib。本地审阅使用 CommonMark、表格与 MathJax，属于 GitHub 渲染的近似检查，不是线上页面验证。

正文代码含局部片段与接口示意，变量由上下文说明；需要直接运行时使用对应完整脚本。向量扩展未做真实模型验证，其余核验范围均在报告中注明。
