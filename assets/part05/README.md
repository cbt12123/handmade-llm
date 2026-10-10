# 第五部分完整附件

正文展示结构和关键片段，这里保留维护者可直接阅读、修改的全文。

| 文件 | 用途 |
|---|---|
| [rules/agent.md](rules/agent.md) | 跨任务规则全文 |
| [skills/catalog.json](skills/catalog.json) | 技能目录与工具权限 |
| [skills/explain/SKILL.md](skills/explain/SKILL.md) | 查询与解释方法 |
| [skills/practice/SKILL.md](skills/practice/SKILL.md) | 练习选择方法 |
| [skills/review/SKILL.md](skills/review/SKILL.md) | 检查答案与记录方法 |
| [knowledge/README.md](knowledge/README.md) | 知识来源与索引约定 |
| [knowledge/questions.json](knowledge/questions.json) | 题库、标准答案、容差与反馈 |

这些文件按本项目加载器的约定读取；不属于所有 Agent 产品通用的装载标准。修改规则和技能无需重新训练模型，修改工具权限、题目或检索逻辑仍应重新核验和评价。

检索与内容评价的完整用例另见[检索回归](../../data/part05/retrieval_cases.json)、[真实解释任务](../../data/part05/evaluation_quality.jsonl)和[历史错误输入](../../data/part05/content_regression.json)。正文讲结构，附件保留完整技能与回归数据；修改后运行检索核验，再检查真实模型答案。
