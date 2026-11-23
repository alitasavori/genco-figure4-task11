"""NPZ shard format for (J, r, dx*) triples."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.sparse import csr_matrix


def save_triple_npz(
    path: Path,
    j: csr_matrix,
    r: np.ndarray,
    dx: np.ndarray,
    meta: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        indptr=j.indptr,
        indices=j.indices,
        data=j.data,
        shape=np.array(j.shape),
        r=r.astype(np.float64),
        dx=dx.astype(np.float64),
        meta_json=np.array(json.dumps(meta)),
    )


def load_triple_npz(path: Path) -> tuple[csr_matrix, np.ndarray, np.ndarray, dict[str, Any]]:
    z = np.load(path, allow_pickle=False)
    shape = tuple(int(x) for x in z["shape"])
    j = csr_matrix((z["data"], z["indices"], z["indptr"]), shape=shape)
    meta = json.loads(str(z["meta_json"]))
    return j, z["r"], z["dx"], meta