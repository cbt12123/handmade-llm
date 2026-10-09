"""Run math examples with assertions, check local references, execute deliverables."""
import ast
import contextlib
import io
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
docs = sorted((ROOT / "第二部分-面向机器学习的数学基础").glob("*.md"))
blocks_count, images_count = 0, 0
original_cwd = Path.cwd()
for doc in docs:
    text = doc.read_text(encoding="utf-8")
    assert text.count("```") % 2 == 0, doc
    for target in re.findall(r"\[[^\]]*\]\(([^)]+)\)", text):
        if not target.startswith(("https://", "http://", "#")):
            assert (doc.parent / target).resolve().is_file(), (doc, target)
    images_count += len(re.findall(r"!\[", text))
    for index, code in enumerate(re.findall(r"```python\n(.*?)```", text, re.S), 1):
        ast.parse(code)
        with tempfile.TemporaryDirectory(prefix="math-tutorial-") as temp:
            try:
                os.chdir(temp)
                with contextlib.redirect_stdout(io.StringIO()):
                    exec(compile(code, f"{doc.name}:example{index}", "exec"), {"__name__":"__main__"})
            except Exception as error:
                raise RuntimeError(f"{doc.name}, example {index}") from error
            finally:
                os.chdir(original_cwd)
                plt.close("all")
        blocks_count += 1
for script in sorted((ROOT / "script" / "part02").glob("[0-9]*.py")):
    subprocess.run([sys.executable, str(script)], cwd=ROOT, check=True)
for prefix, image_name in [("04-calculus", "04-learning-rates"),("05-linear-algebra","05-transforms"),("06-probability","06-frequency"),("07-discrete","07-graph")]:
    assert (ROOT / "outputs" / "part02" / f"{prefix}.json").is_file()
    assert (ROOT / "outputs" / "part02" / f"{image_name}.png").is_file()
print(f"PASS: {blocks_count} mathematical examples, {images_count} image references, 4 deliverable scripts and 8 output artifacts.")
