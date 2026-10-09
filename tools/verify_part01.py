"""Execute independent Python examples in isolated temporary directories.

AST inspection provides the two explicitly documented prerequisites: user input
for the input example and a neighboring helpers.py for the import example.
"""
import ast
import contextlib
import io
import os
from pathlib import Path
import re
import tempfile

import matplotlib
matplotlib.use("Agg")
ROOT = Path(__file__).resolve().parents[1]
DOCS = sorted((ROOT / "第一部分-Python与计算基础").glob("*.md"))
blocks = [(doc, code) for doc in DOCS for code in re.findall(r"```python\n(.*?)```", doc.read_text(encoding="utf-8"), re.S)]
original_cwd = Path.cwd()
for index, (DOC, code) in enumerate(blocks, 1):
    tree = ast.parse(code)
    has_input = any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "input" for n in ast.walk(tree))
    with tempfile.TemporaryDirectory(prefix="python-tutorial-check-") as temp:
        os.chdir(temp)
        if "from helpers import double" in code:
            Path("helpers.py").write_text("def double(number):\n    return number * 2\n", encoding="utf-8")
        namespace = {"__name__": "__main__"}
        if has_input:
            namespace["input"] = lambda prompt="": "18"
        if "from helpers import double" in code:
            import sys
            sys.path.insert(0, temp)
        try:
            with contextlib.redirect_stdout(io.StringIO()) as output:
                exec(compile(tree, str(DOC), "exec"), namespace)
            assert "Traceback" not in output.getvalue()
        finally:
            if "from helpers import double" in code:
                sys.path.remove(temp)
                sys.modules.pop("helpers", None)
            os.chdir(original_cwd)
            import matplotlib.pyplot as plt
            plt.close("all")

image_count = 0
for DOC in DOCS:
    text = DOC.read_text(encoding="utf-8")
    refs = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", text)
    for ref in refs:
        assert (DOC.parent / ref).resolve().is_file(), ref
    image_count += len(refs)
    for ref in re.findall(r"(?<!!)\[[^\]]*\]\(([^)]+)\)", text):
        if not ref.startswith(("https://", "http://", "#")):
            assert (DOC.parent / ref).resolve().is_file(), ref
    assert text.count("```") % 2 == 0
print(f"PASS: {len(DOCS)} chapters; {len(blocks)} Python examples executed; {image_count} image references resolved; chapter links resolved.")
