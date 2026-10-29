"""Install the GENCO graphkit stack used for Task 1.1.

This environment is separate from the PFDelta baseline notebook. Graphkit
requires torch>=2.7.1,<2.9. The PFDelta repository pins torch==2.1.2.
"""

from __future__ import annotations

import os
import sys

from common import LOGS, clone_repo, ensure_dirs, in_colab, report_environment, run_command, work_root


def mount_drive() -> None:
    if not in_colab():
        return
    try:
        from google.colab import drive
    except Exception as exc:
        print(f"Google Drive is not available: {exc}")
        return
    drive.mount("/content/drive")
    print("Mounted Google Drive at /content/drive")


def pip_install(args: list[str], log_name: str) -> None:
    run_command([sys.executable, "-m", "pip", "install", *args], cwd=None, log_path=LOGS / log_name)


def torch_version_ok() -> bool:
    try:
        from packaging.version import Version
        import torch
    except Exception:
        return False
    version = Version(torch.__version__.split("+")[0])
    return Version("2.7.1") <= version < Version("2.9")


def install_torch() -> None:
    if torch_version_ok():
        import torch

        print(f"Keeping existing torch {torch.__version__}")
        return
    index = []
    try:
        import torch

        cuda = torch.version.cuda
    except Exception:
        cuda = os.environ.get("COLAB_GPU")
    if cuda or os.environ.get("COLAB_GPU"):
        # cu124 wheels cover the torch 2.8 pin used with this graphkit commit.
        index = ["--index-url", "https://download.pytorch.org/whl/cu124"]
        print("Installing torch==2.8.0+cu124 to satisfy graphkit torch>=2.7.1,<2.9.")
    else:
        print("Installing torch==2.8.0+cpu to satisfy graphkit torch>=2.7.1,<2.9.")
        index = ["--index-url", "https://download.pytorch.org/whl/cpu"]
    pip_install(["torch==2.8.0", *index], "pip_torch.log")
    if not torch_version_ok():
        raise SystemExit("torch is outside the graphkit range >=2.7.1,<2.9 after installation.")


def install_torch_scatter() -> None:
    import torch

    cuda = torch.version.cuda
    tag = torch.__version__ if cuda else f"{torch.__version__}+cpu"
    url = f"https://data.pyg.org/whl/torch-{tag}.html"
    print(f"Installing torch-scatter from {url}")
    pip_install(["torch-scatter", "-f", url], "pip_torch_scatter.log")


def main() -> None:
    ensure_dirs()
    info = report_environment()
    major, minor = sys.version_info[:2]
    if (major, minor) < (3, 10) or (major, minor) >= (3, 13):
        raise SystemExit(
            f"gridfm-graphkit requires Python >=3.10,<3.13. This runtime is {info['python']}."
        )
    mount_drive()
    pip_install(
        ["pandas", "matplotlib", "seaborn", "pyyaml", "numpy", "pyarrow", "huggingface_hub"],
        "pip_plot_stack.log",
    )
    root = work_root()
    repo = root / "gridfm-graphkit"
    clone_repo("gridfm-graphkit", repo)
    install_torch()
    pip_install(["-e", str(repo)], "pip_graphkit.log")
    if not torch_version_ok():
        print("graphkit's installer moved torch outside >=2.7.1,<2.9. Restoring the pin.")
        install_torch()
        pip_install(["-e", str(repo), "--no-deps"], "pip_graphkit_nodeps.log")
    install_torch_scatter()
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    report_environment()
    note = LOGS / "genco_compatibility.txt"
    note.write_text(
        "\n".join(
            [
                "Clone YAML and model code are not edited.",
                "torch is pinned to >=2.7.1,<2.9 because that is what this graphkit commit declares.",
                "If Colab already had a different torch build, the install log records the replacement.",
                "data.workers may be lowered in a copy under results/configs when the reproduction script runs.",
                "That copy is passed as --config. The file inside the clone stays as released.",
                "PFDelta torch==2.1.2 is intentionally not installed in this runtime.",
                f"Work directory: {root}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"GENCO install finished. Work directory: {root}")


if __name__ == "__main__":
    main()
