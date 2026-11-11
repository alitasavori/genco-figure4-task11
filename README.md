# GENCO Figure 4, Task 1.1

[Open Task1_1.ipynb in Colab](https://colab.research.google.com/github/alitasavori/genco-figure4-task11/blob/main/Task1_1.ipynb)

Use a GPU runtime. Run the **Colab preparation** cell once per session (only cell that runs `git pull`), then the training cell you need (v7 / v8 / v9, other PFΔ tasks, diagnose, or SVG). Data: `My Drive/Foundation Model/vendor/pfdelta_task1.1`. Checkpoints: `Foundation Model/tpgt_task11_<RUN_TAG>` (e.g. `tpgt_task11_v9`).

Other PFΔ IEEE 118 tasks (1.2, 1.3/2.1, 2.3, 4.1, 4.2, 4.3) have one training cell each in the notebook (`%run train_tpgt_task11.py --task <id>`). Each uses `Foundation Model/vendor/pfdelta_task<id>` (downloaded from Hugging Face if missing) and writes checkpoints, tables, a bar plot against the published numbers (`plot_task1_1.PUBLISHED`), and bus error maps at every checkpoint epoch to `Foundation Model/tpgt_task<id>_v7`.

TPGT is the draft's unshared-weight type-pair graph Transformer. Its width, depth, batch, and five-epoch cap are choices for a Colab GPU, not released hyperparameters. Pair features on IEEE 118 are a 5-channel RPE on the full meshed topology.

| File | Role |
| --- | --- |
| `Task1_1.ipynb` | Preparation, then TPGT training, then the published SVG |
| `plot_task1_1.py` | Writes the published Task 1.1 panel (`--published`) or a completed test table |
| `train_tpgt_task11.py` | Reads the Drive Task 1.1 copy when `case118` is present, trains TPGT, and writes the TPGT tables |
