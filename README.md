# GENCO Figure 4, Task 1.1

[Open Task1_1.ipynb in Colab](https://colab.research.google.com/github/alitasavori/genco-figure4-task11/blob/main/Task1_1.ipynb)

Use a GPU runtime. Run the preparation cell first (it clones the repo and mounts Drive), accept the Drive prompt, then run the training cell. The SVG cell draws the published figure and does not train. Data path: `My Drive/Foundation Model/vendor/pfdelta_task1.1`. Checkpoints go to `Foundation Model/tpgt_task11_<RUN_TAG>` (currently `tpgt_task11_v7`) and TPGT tables to `Foundation Model/results/tpgt_task11_<RUN_TAG>`.

TPGT is the draft's unshared-weight type-pair graph Transformer. Its width, depth, batch, and five-epoch cap are choices for a Colab GPU, not released hyperparameters. Pair features on IEEE 118 are a 5-channel RPE on the full meshed topology.

| File | Role |
| --- | --- |
| `Task1_1.ipynb` | Preparation, then TPGT training, then the published SVG |
| `plot_task1_1.py` | Writes the published Task 1.1 panel (`--published`) or a completed test table |
| `train_tpgt_task11.py` | Reads the Drive Task 1.1 copy when `case118` is present, trains TPGT, and writes the TPGT tables |
