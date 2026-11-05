"""Train and test GENCO Base, PFNet, CANOS-PF, and GNS-S for Task 1.1.

Finished seeds are skipped. The figure is written after all four models have
three finite test scores.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from storage import ROOT, mount_and_select


def interpreters() -> tuple[Path, Path]:
    from common import LOGS

    manifest = LOGS / "venvs.json"
    if not manifest.exists():
        raise SystemExit("No training environments are recorded. Run the install cell first.")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    genco = Path(payload["genco"])
    pfdelta = Path(payload["pfdelta"])
    missing = [str(path) for path in (genco, pfdelta) if not path.exists()]
    if missing:
        raise SystemExit(
            "The training environments are not on this runtime: "
            + ", ".join(missing)
            + ". Run the install cell again. Checkpoints already saved on Drive are kept."
        )
    return genco, pfdelta


def run_stack(python: Path, script: str, log_name: str) -> None:
    from common import LOGS, run_command

    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    run_command(
        [str(python), str(ROOT / script)],
        cwd=ROOT,
        log_path=LOGS / log_name,
        env=env,
    )


def show_results() -> None:
    from common import resolve_results

    results = resolve_results()
    summary_path = results / "task1_1_summary.csv"
    figure = results / "figure4_task1.1.svg"
    note = results / "figure4_task1.1_NOT_READY.txt"
    print(f"Summary file: {summary_path}")
    if summary_path.exists():
        print(summary_path.read_text(encoding="utf-8"))
    if figure.exists():
        print(f"Figure: {figure}")
    elif note.exists():
        print(note.read_text(encoding="utf-8"))
    try:
        from IPython import get_ipython
        from IPython.display import Markdown, SVG, display
        import pandas as pd
    except Exception:
        return
    if get_ipython() is None or not summary_path.exists():
        return
    frame = pd.read_csv(summary_path)
    if figure.exists():
        display(Markdown("**Task 1.1 test results.** Each bar is the mean power-balance loss of three trained seeds."))
        display(frame)
        display(SVG(filename=str(figure)))
        return
    display(Markdown("**The four-model figure is not ready.** The table is the test scores written so far."))
    display(frame)
    if note.exists():
        display(Markdown("```\n" + note.read_text(encoding="utf-8") + "\n```"))


def main() -> None:
    mount_and_select()
    from common import use_selected_results

    use_selected_results()
    print("Training and testing all four models.")
    print("Order: GENCO Base (seeds 1, 2, 3), then PFNet, CANOS-PF, and GNS-S (seeds 11, 21, 31).")
    print("Epoch caps come from the released configs: GENCO 300, PFNet 100, CANOS-PF 200, GNS-S 25.")
    print("A Colab session can end before that finishes. Run both cells again to resume.")
    print("A seed that already has its checkpoint and test metrics is skipped.")
    genco, pfdelta = interpreters()
    print(f"GENCO interpreter: {genco}")
    run_stack(genco, "reproduce_genco.py", "train_genco.log")
    print(f"PFDelta interpreter: {pfdelta}")
    run_stack(pfdelta, "reproduce_pfdelta.py", "train_pfdelta.log")
    show_results()


if __name__ == "__main__":
    main()
