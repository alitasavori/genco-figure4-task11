# GENCO Figure 4, Task 1.1

[Open Task1_1.ipynb in Colab](https://colab.research.google.com/github/alitasavori/genco-figure4-task11/blob/main/Task1_1.ipynb)

Use a GPU runtime. Run the install cell, accept Google Drive, then run the training cell. The Task 1.1 files are the copy already on Google Drive at `Foundation Model/vendor/pfdelta_task1.1`. Checkpoints go to `Foundation Model/tpgt_task11` and TPGT tables to `Foundation Model/results`.

TPGT is the draft's unshared-weight type-pair graph Transformer. Its width, depth, batch, and five-epoch cap are choices for a Colab GPU, not released hyperparameters. Pair features on IEEE 118 use a slack-rooted spanning tree.

| File | Role |
| --- | --- |
| `Task1_1.ipynb` | Install cell, then the training cell (published figure, then TPGT) |
| `plot_task1_1.py` | Writes the published Task 1.1 panel (`--published`) or a completed test table |
| `train_tpgt_task11.py` | Reads the Drive Task 1.1 copy when `case118` is present, trains TPGT, and writes the TPGT tables |
