"""Six Phase 3 ablation presets (architecture fixed width/depth)."""

from __future__ import annotations

from jmat.model import JMAT, JMATConfig


ABLATION_ORDER = (
    "mlp_only",
    "sparse_a_only",
    "sparse_a_at",
    "global_attn_only",
    "sparse_a_global",
    "jmat_full",
)


def config_for_ablation(name: str) -> JMATConfig:
    base = dict(d_model=64, n_latent=16, n_heads=4, n_loops=3)
    presets = {
        "mlp_only": dict(use_global_attn=False, use_sparse_a=False, use_sparse_at=False),
        "sparse_a_only": dict(use_global_attn=False, use_sparse_a=True, use_sparse_at=False),
        "sparse_a_at": dict(use_global_attn=False, use_sparse_a=True, use_sparse_at=True),
        "global_attn_only": dict(use_global_attn=True, use_sparse_a=False, use_sparse_at=False),
        "sparse_a_global": dict(use_global_attn=True, use_sparse_a=True, use_sparse_at=False),
        "jmat_full": dict(use_global_attn=True, use_sparse_a=True, use_sparse_at=True),
    }
    if name not in presets:
        raise KeyError(name)
    return JMATConfig(**base, **presets[name])


def build_ablation(name: str) -> JMAT:
    return JMAT(config_for_ablation(name))