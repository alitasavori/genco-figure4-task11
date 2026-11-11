"""One-off: remove duplicate git pull from training cells; keep prep cell only."""

from __future__ import annotations

import json
import re
from pathlib import Path

NOTEBOOK = Path(__file__).resolve().parents[1] / "Task1_1.ipynb"

GIT_TRAIN_BLOCK = re.compile(
    r"if on_colab\(\):\n"
    r"    if not REPO\.is_dir\(\):\n"
    r"        subprocess\.check_call\(\[\"git\", \"clone\", REPO_URL, str\(REPO\)\]\)\n"
    r"    else:\n"
    r"        subprocess\.check_call\(\[\"git\", \"-C\", str\(REPO\), \"fetch\", \"origin\"\]\)\n"
    r"        subprocess\.check_call\(\[\"git\", \"-C\", str\(REPO\), \"reset\", \"--hard\", \"origin/main\"\]\)\n"
    r"    os\.chdir\(REPO\)\n"
    r"    head = subprocess\.check_output\(\n"
    r"        \[\"git\", \"-C\", str\(REPO\), \"log\", \"-1\", \"--oneline\"\], text=True\n"
    r"    \)\.strip\(\)\n"
    r"    print\(f\"Training from \{head\}\", flush=True\)\n",
    re.MULTILINE,
)

COLAB_TRAIN_SETUP = """if on_colab():
    if not REPO.is_dir():
        raise RuntimeError("Run the Colab preparation cell first.")
    import sys

    sys.path.insert(0, str(REPO))
    from colab_bootstrap import ensure_training_cwd, git_head

    ensure_training_cwd()
    print(f"Training from {git_head()}", flush=True)
"""

# Task cells: drive mount is in the same on_colab block after git block.
GIT_TASK_BLOCK = re.compile(
    r"if on_colab\(\):\n"
    r"    if not REPO\.is_dir\(\):\n"
    r"        subprocess\.check_call\(\[\"git\", \"clone\", REPO_URL, str\(REPO\)\]\)\n"
    r"    else:\n"
    r"        subprocess\.check_call\(\[\"git\", \"-C\", str\(REPO\), \"fetch\", \"origin\"\]\)\n"
    r"        subprocess\.check_call\(\[\"git\", \"-C\", str\(REPO\), \"reset\", \"--hard\", \"origin/main\"\]\)\n"
    r"    os\.chdir\(REPO\)\n"
    r"    head = subprocess\.check_output\(\n"
    r"        \[\"git\", \"-C\", str\(REPO\), \"log\", \"-1\", \"--oneline\"\], text=True\n"
    r"    \)\.strip\(\)\n"
    r"    print\(f\"Training from \{head\}\", flush=True\)\n"
    r"    if not Path\(\"/content/drive/MyDrive\"\)\.is_dir\(\):\n"
    r"        from google\.colab import drive\n"
    r"\n"
    r"        drive\.mount\(\"/content/drive\"\)\n"
    r"    DRIVE = Path\(\"/content/drive/MyDrive\"\)\n",
    re.MULTILINE,
)

COLAB_TASK_SETUP = """if on_colab():
    if not REPO.is_dir():
        raise RuntimeError("Run the Colab preparation cell first.")
    import sys

    sys.path.insert(0, str(REPO))
    from colab_bootstrap import ensure_training_cwd, git_head, mount_drive_if_needed

    ensure_training_cwd()
    print(f"Training from {git_head()}", flush=True)
    DRIVE = mount_drive_if_needed()
"""


def patch_source(src: str, cell_id: str) -> str:
    if cell_id == "install-tpgt":
        return src
    if cell_id == "c7diagnose":
        return (
            "import os\n"
            "import sys\n"
            "from pathlib import Path\n\n"
            "TASK = \"1.1\"\n\n"
            "REPO = Path(\"/content/genco-figure4-task11\")\n"
            "if not REPO.is_dir():\n"
            "    raise RuntimeError(\"Run the Colab preparation cell first.\")\n"
            "sys.path.insert(0, str(REPO))\n"
            "from colab_bootstrap import ensure_training_cwd, mount_drive_if_needed\n\n"
            "ensure_training_cwd()\n"
            "mount_drive_if_needed()\n"
            "DATA_ID = \"1.3\" if TASK in (\"1.3\", \"2.1\") else TASK\n"
            "os.environ[\"GENCO_PFDELTA_TASK\" + DATA_ID.replace(\".\", \"\")] = (\n"
            "    f\"/content/drive/MyDrive/Foundation Model/vendor/pfdelta_task{DATA_ID}\"\n"
            ")\n"
            "os.environ[\"GENCO_TPGT_DATA_FRACTION\"] = \"1.0\"\n\n"
            "%run train_tpgt_task11.py --diagnose --task {TASK} --seed 0 --cases 32\n"
        )
    if cell_id == "c5svg":
        return re.sub(
            r"    if on_colab\(\):\n"
            r"        if not REPO\.is_dir\(\):\n"
            r"            subprocess\.check_call\(\[\"git\", \"clone\", REPO_URL, str\(REPO\)\]\)\n"
            r"        os\.chdir\(REPO\)\n",
            "    if on_colab():\n"
            "        if not REPO.is_dir():\n"
            "            raise RuntimeError(\"Run the Colab preparation cell first.\")\n"
            "        os.chdir(REPO)\n",
            src,
            count=1,
        )
    if cell_id == "c8busmaps":
        if "ensure_training_cwd" in src:
            return src
        prefix = (
            "import sys\n"
            "from pathlib import Path\n\n"
            "REPO = Path(\"/content/genco-figure4-task11\")\n"
            "if REPO.is_dir():\n"
            "    sys.path.insert(0, str(REPO))\n"
            "    from colab_bootstrap import ensure_training_cwd\n\n"
            "    ensure_training_cwd()\n\n"
        )
        # drop duplicate leading imports if present
        if src.startswith("import os\n"):
            src = src.split("\n", 3)[-1] if "from pathlib" in src[:80] else src
        return prefix + src

    if GIT_TASK_BLOCK.search(src):
        src = GIT_TASK_BLOCK.sub(COLAB_TASK_SETUP, src, count=1)
    elif GIT_TRAIN_BLOCK.search(src):
        src = GIT_TRAIN_BLOCK.sub(COLAB_TRAIN_SETUP, src, count=1)

    if "subprocess" in src and "subprocess." not in src:
        src = src.replace("import subprocess\n", "")
    return src


def main() -> None:
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        cell_id = cell.get("id", "")
        src = "".join(cell["source"])
        new = patch_source(src, cell_id)
        if new != src:
            cell["source"] = [line if line.endswith("\n") else line + "\n" for line in new.splitlines(keepends=True)]
            if cell["source"] and not cell["source"][-1].endswith("\n"):
                cell["source"][-1] += "\n"

    for cell in nb["cells"]:
        if cell.get("id") == "c0prepmd":
            cell["source"] = [
                "# Colab preparation (one pull for all cells below)\n",
                "\n",
                "Clone on first run, then `git fetch` + `reset --hard origin/main`, mount Google Drive, "
                "install missing packages. **Training, task, diagnose, and SVG cells do not run git**—"
                "run this cell once per Colab session (and after we push updates).\n",
                "\n",
                "Prints the repo commit hash. Data: `My Drive/Foundation Model/vendor/pfdelta_task1.1`.\n",
            ]
        if cell.get("id") == "c2trainmd":
            text = "".join(cell["source"])
            if "preparation cell" in text:
                cell["source"] = [
                    line.replace(
                        "Run the prep cell first",
                        "Run the **preparation** cell first (it is the only cell that runs `git pull`)",
                    )
                    for line in cell["source"]
                ]

    NOTEBOOK.write_text(json.dumps(nb, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("Patched", NOTEBOOK)


if __name__ == "__main__":
    main()
