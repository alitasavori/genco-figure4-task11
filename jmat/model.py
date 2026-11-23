"""JMAT encoder, shared matrix-action block, and correction head (brief §5)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from jmat.normalize import NormalizedLinearSystem, normalize_linear_system
from jmat.sparse_ops import csr_matmul, csr_transpose_matmul, index_features


@dataclass
class JMATConfig:
    d_model: int = 64
    n_latent: int = 16
    n_heads: int = 4
    n_loops: int = 3
    use_global_attn: bool = True
    use_sparse_a: bool = True
    use_sparse_at: bool = True


class JMAT(nn.Module):
    """f_θ(J, r) → Δx ≈ J⁻¹r via one forward pass (no inner Krylov loop)."""

    def __init__(self, cfg: JMATConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or JMATConfig()
        d = self.cfg.d_model
        self.encoder = nn.Sequential(nn.Linear(4, d), nn.GELU(), nn.Linear(d, d))
        self.latent = nn.Parameter(torch.randn(self.cfg.n_latent, d) * 0.02)
        h = self.cfg.n_heads
        self.attn_pool = nn.MultiheadAttention(d, h, batch_first=True)
        self.attn_latent = nn.MultiheadAttention(d, h, batch_first=True)
        self.attn_broadcast = nn.MultiheadAttention(d, h, batch_first=True)
        self.ln_h = nn.LayerNorm(d)
        self.ln_z = nn.LayerNorm(d)
        self.ln_out = nn.LayerNorm(d)
        mix_in = d * (1 + int(self.cfg.use_sparse_a) + int(self.cfg.use_sparse_at) + int(self.cfg.use_global_attn))
        self.mix_mlp = nn.Sequential(nn.Linear(mix_in, 2 * d), nn.GELU(), nn.Linear(2 * d, d))
        self.out_head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))

    def forward_normalized(
        self,
        norm: NormalizedLinearSystem,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (z_hat, delta_x) in physical units."""
        crow, col, val, n = norm.a_crow, norm.a_col, norm.a_val, norm.n
        q = index_features(crow, col, val, n, norm.b_tilde)
        h = self.encoder(q)
        z = self.latent.unsqueeze(0)
        for _ in range(self.cfg.n_loops):
            x = self.ln_h(h).unsqueeze(0)
            z_n = self.ln_z(z)
            pool, _ = self.attn_pool(z_n, x, x)
            z = z + pool
            z_n2 = self.ln_z(z)
            lat, _ = self.attn_latent(z_n2, z_n2, z_n2)
            z = z + lat
            pieces = [h]
            if self.cfg.use_global_attn:
                g, _ = self.attn_broadcast(x, self.ln_z(z), self.ln_z(z))
                pieces.append(g.squeeze(0))
            if self.cfg.use_sparse_a:
                pieces.append(csr_matmul(crow, col, val, n, h))
            if self.cfg.use_sparse_at:
                pieces.append(csr_transpose_matmul(crow, col, val, n, h))
            mix_in = torch.cat(pieces, dim=-1)
            h = h + self.mix_mlp(mix_in)
        z_hat = self.out_head(self.ln_out(h)).squeeze(-1)
        delta_x = norm.beta * z_hat
        return z_hat, delta_x

    def forward_csr(
        self,
        crow: torch.Tensor,
        col: torch.Tensor,
        val: torch.Tensor,
        n: int,
        r: torch.Tensor,
    ) -> torch.Tensor:
        """Public API: sparse J and mismatch r → Δx (n,); zeros if r == 0."""
        norm = normalize_linear_system(crow, col, val, n, r)
        if norm is None:
            return torch.zeros(n, dtype=val.dtype, device=val.device)
        _, dx = self.forward_normalized(norm)
        return dx