"""Show the published Task 1.1 figure and say where the numbers came from.

Running this file does not train or evaluate GENCO, CANOS-PF, GNS-S, or PFNet.
"""

from __future__ import annotations

from pathlib import Path

from common import RESULTS
from plot_task1_1 import plot_task11
from published import write_published_tables


def main() -> None:
    print("These numbers were not produced by training or testing the four models.")
    print("Nothing in this run fitted GENCO Base, CANOS-PF, GNS-S, or PFNet.")
    print("The table and the figure are the authors' published Figure 4 values,")
    print("copied from their LaTeX table and plotted again.")
    print("measured_source is published_figure4_table.")
    print("abs_diff is 0 because the table was copied, not because a new run matched it.")
    write_published_tables(RESULTS)
    plot_task11(RESULTS)
    summary = (RESULTS / "task1_1_summary.csv").read_text(encoding="utf-8")
    print(summary)
    missing = RESULTS / "figure4_task1.1_retrained_NOT_RUN.txt"
    if missing.exists():
        print(missing.read_text(encoding="utf-8"))
    image = RESULTS / "figure4_task1.1_published.png"
    try:
        from IPython import get_ipython
        from IPython.display import Image, Markdown, display
        import pandas as pd
    except Exception:
        print(f"Figure file: {image}")
        return
    if get_ipython() is None:
        print(f"Figure file: {image}")
        return
    display(Markdown(
        "**Published table only.** No model was trained or tested to make this figure."
    ))
    display(pd.read_csv(RESULTS / "task1_1_summary.csv"))
    display(Image(filename=str(image)))
    if missing.exists():
        display(Markdown(missing.read_text(encoding="utf-8")))


if __name__ == "__main__":
    main()
