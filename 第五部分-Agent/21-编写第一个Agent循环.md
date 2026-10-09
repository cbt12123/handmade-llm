# 第21章 编写第一个 Agent 循环

这一章先看一个只有几十行的循环，再把它接到第四部分的模型服务。核心过程只有四步：读取当前状态、请求下一动作、执行动作、保存结果。完整项目增加的权限、检索和数据库仍围绕这个过程展开。

## 21.1 先给出循环

```python
observation = None
tools = {"multiply": lambda a, b: a * b}

for _ in range(3):
    action = next_action(observation)
    if action["action"] == "finish":
        print(action["answer"])
        break
    observation = tools[action["action"]](**action["args"])
else:
    raise RuntimeError("未在预算内完成")
```

这里的 next_action 可以由模型实现，也可以先用固定函数代替。教学脚本[21_minimal_loop.py](../script/part05/21_minimal_loop.py)用固定函数提出“3×4”，收到 12 后结束；**它只演示协议，运行结果不属于模型能力实验。**

先运行：

```powershell
conda activate base
python script/part05/21_minimal_loop.py
```

第一次得到工具动作，第二次得到 finish。for 后面的 else 只在循环没有 break 时运行，适合检查达到预算却未完成的情况。

## 21.2 模型动作需要什么结构

完整助手的动作统一有四个字段：

```json
{
  "action": "search",
  "args": {"query": "梯度下降"},
  "answer": "",
  "citations": []
}
```

工具动作的 args 保存参数。最终动作使用 action="finish"，args 为空，answer 写回答，citations 列资料 ID。

为什么不用“请帮我查一下梯度下降”作为工具请求？自然语言可以有很多表达，程序难以确定工具和参数。JSON 把接口变成可以逐字段检查的数据结构。

这个自定义动作协议可以帮助初学者看清调用过程。它没有使用服务商原生的 tools/tool_calls 接口；以后迁移到原生接口时，可以保留同一个工具注册表、权限检查和任务状态，只更换模型适配层。

## 21.3 模型怎样知道有哪些工具

model.py 把规则、工具说明和当前任务状态组成请求。工具说明包含固定名字与参数示例：

```text
search: {"query":"一个概念关键词"}
quiz: {"topic":"梯度下降|概率|矩阵","count":2}
grade: {"question_id":"gd-01","answer":"1.6"}
```

当前状态会告诉模型已经加载什么技能、已经拿到什么结果、还有几步可以执行。模型需要依据这些信息提出下一动作；程序不在幕后按用户关键词预先替它完成整个流程。

技能尚未加载时，模型先接受一个简短的分类请求，只看用户任务与技能描述。后续请求再加入规则、当前技能和工具结果。这避免把出题示例混进分类输入，导致小模型把示例当作真实问题。

输出后先用 json.loads 解析，再检查字段和工具参数。解析失败与工具执行失败不同：前者意味着没有接受到有效动作；后者意味着动作已解析，但其名称、权限或参数不符合约定。

## 21.4 接入真实 HTTP 模型

vLLM 请求的关键部分是：

```python
payload = {
    "model": "tutorial-agent",
    "messages": messages,
    "temperature": 0,
    "max_tokens": 500,
    "structured_outputs": {"json": ACTION_SCHEMA},
}
```

结构约束让服务在指定 JSON Schema 范围内生成。本教程仍会在客户端重新检查，工具层还会检查每个动作需要的字段。模型把 grade 的 answer 写成错误数值，即使完全符合 Schema，仍然是错误动作。结构约束也不保证任务能完成。[vLLM 官方说明](https://docs.vllm.ai/en/stable/examples/features/structured_outputs/)介绍了相应服务接口。

最终实现按动作建立不同 Schema 分支，并依据状态限制候选：工具动作的 answer 与 citations 为空，search 只能带 query，已取题后不重复 quiz，已评分后不重复 grade。没有知识结果的解释任务，最终回答必须以“资料不足”开头。具体约定同时接受 Python 检查，完整代码见[model.py](../script/part05/model.py)。

第四部分的容器上下文上限是 512 token，适合短对话；这部分要装入规则、技能和工具结果，参考容器提高到 4096。部署命令与模型路径见[运行指南](README.md)，仍使用已有 Qwen2.5-1.5B-Instruct 和现成镜像。

第四部分的自定义 /generate 接口也有适配，但原服务的 512 输入预算与 128 输出上限通常不足以承载完整助手。本部分实际验证使用 vLLM，不能把这两个入口看作在旧配置下直接等效。

## 21.5 真实循环比示意代码多了什么

[runner.py](../script/part05/runner.py)把工作拆为以下步骤：

1. 根据已提交状态请求模型动作，HTTP 请求在数据库事务之外执行。
2. 复制状态并增加步数，检查动作。
3. 在事务内执行工具，添加观察结果并保存检查点。
4. 工具检查失败时回滚，保存明确错误；模型可以在下一步修正。
5. finish 通过检查后结束；最多八步，超出后状态为 exhausted。

复制状态的原因是工具执行可能修改字典后才失败。数据库可以回滚，但普通字典不会自动回滚；需要丢弃这份候选状态，重新从已提交状态出发。

网络或动作解析失败会结束为 failed，不自动换成预设回答。用户看到失败，可以根据 HTTP 错误、输出截断或 JSON 内容判断问题。这比只显示“暂时不可用”更便于学习和排查。

## 21.6 结束条件与预算

模型可以反复查询同一个词，也可以重复加载技能。循环必须有代码预算，不能只在提示里写“请尽快完成”。

本项目最多八次决策，每次最多生成 500 token，HTTP 有连接和读取超时。八步是教学项目的配置选择，不是所有 Agent 的标准；任务更长时先评估是否应该拆成独立任务。

```text
running   正在运行，或已主动暂停在检查点
done      模型提交的最终动作通过检查
failed    模型调用或协议解析失败
exhausted 达到八步但未完成
```

done 只表示程序流程结束；回答质量还要另行检查。第27章会区分运行状态和任务评价。

## 21.7 如何阅读执行记录

每个 event 保存 step、action 与 observation，失败则保存 error。它记录外部可观察的动作，不要求模型输出内部推理。

```json
{
  "step": 2,
  "action": {"action":"search","args":{"query":"梯度下降"},"answer":"","citations":[]},
  "observation": {"matches": []}
}
```

matches 为空时，模型应改查询或说明资料不足。动作记录可以区分“没有执行检索”“检索没命中”“拿到资料但解释不对”，三个问题需要不同修复。

## 21.8 随堂检查与产出

运行最小循环，把乘法工具改为加法，并观察动作参数如何传入。然后阅读真实循环，找出八步预算、HTTP 错误和工具错误各自在哪里处理。

本章产出是能读懂动作与观察结果的循环。下章再把“能执行”收紧到“在什么条件下能执行”。

[上一章](20-Agent的组成与运行过程.md) · [下一章：Rules](22-Rules规则与执行边界.md)
