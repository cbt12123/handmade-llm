"""One-time migration of the original combined draft into three chapters."""
from pathlib import Path
root = Path(__file__).resolve().parents[1]
chapter = root / "第一部分-Python与计算基础" / "01-Python基础.md"
text = chapter.read_text(encoding="utf-8")
if "## 1.19" in text:
    start = text.index("## 1.10")
    classes_start = text.index("## 1.13")
    classes_end = text.index("## 1.14")
    # Retain class fundamentals, move file organization and array lessons to chapters 2/3.
    text = text[:start] + text[classes_start:classes_end].replace("1.13", "1.10")
    text = text.replace("# 第 1 章：Python 与计算基础", "# 第 1 章：Python 基础")
    text = text.replace("逐步讲到读取数据、使用数组和画图", "逐步讲到函数、类与对象")
    route_start = text.index("## 阅读路线")
    route_end = text.index("## 1.1")
    text = text[:route_start] + "## 阅读路线\n\n环境与运行 → 变量与运算 → 容器与控制流程 → 函数 → 类与对象。\n\n文件处理、调试与模块组织在第 2 章学习，NumPy 与画图在第 3 章学习。\n\n" + text[route_end:]
    text += """## 1.11 本章自检

| 问题 | 参考答案 |
|---|---|
| 代码和运行命令分别写在哪里？ | Python 代码写入 .py 文件，运行命令写在终端。 |
| = 与 == 有何不同？ | 赋值与相等比较。 |
| input 返回什么类型？ | 字符串，数字运算前需要转换。 |
| 列表切片包含结束位置吗？ | 不包含。 |
| for 与 while 有何不同？ | 前者遍历元素，后者按条件重复。 |
| print 与 return 有何不同？ | 显示内容与返回结果。 |
| self 表示什么？ | 当前实例。 |

下一章继续学习怎样把代码拆成模块，并处理文件与表格数据。本章不设置章末作品。

[下一章：组织代码与数据处理](02-组织代码与数据处理.md) · [项目目录](../README.md)
"""
    chapter.write_text(text, encoding="utf-8")
readme = root / "README.md"
text = readme.read_text(encoding="utf-8").replace("Python 与计算基础（合并为一个大章节）", "Python 基础、组织代码与数据处理、NumPy 与数据可视化")
text = text.replace("│   └── 01-Python基础.md", "│   ├── 01-Python基础.md\n│   ├── 02-组织代码与数据处理.md\n│   └── 03-NumPy与数据可视化.md")
if "## 第一部分章节" not in text:
    text += "\n## 第一部分章节\n\n1. [Python 基础](第一部分-Python与计算基础/01-Python基础.md)\n2. [组织代码与数据处理](第一部分-Python与计算基础/02-组织代码与数据处理.md)\n3. [NumPy 与数据可视化](第一部分-Python与计算基础/03-NumPy与数据可视化.md)\n"
readme.write_text(text, encoding="utf-8")
