"""Build LankaEnsemble_Lab.ipynb from notebook_src.py (percent-format source, like jupytext).
Usage: python build_notebook.py"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).parent
src = (ROOT / "notebook_src.py").read_text(encoding="utf-8")
parts = re.split(r"^# %%(.*)$", src, flags=re.M)
cells = []
for i in range(1, len(parts), 2):
    kind, body = parts[i].strip(), parts[i + 1].strip("\n")
    if kind == "[markdown]":
        text = "\n".join(l[2:] if l.startswith("# ") else l.lstrip("#") for l in body.splitlines())
        cells.append({"cell_type": "markdown", "metadata": {}, "source": text.splitlines(True)})
    else:
        m = re.match(r"# INCLUDE: (.+)", body.strip())
        if m:  # inline the dataset generator so the notebook is self-contained
            g = (ROOT / m.group(1)).read_text(encoding="utf-8")
            body = g.split('if __name__ == "__main__":')[0].rstrip().replace("OUT = Path(__file__).parent\n", "")
        cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                      "source": body.splitlines(True)})
nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}, "colab": {"provenance": []}},
      "nbformat": 4, "nbformat_minor": 5}
(ROOT / "LankaEnsemble_Lab.ipynb").write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"Built LankaEnsemble_Lab.ipynb with {len(cells)} cells")
