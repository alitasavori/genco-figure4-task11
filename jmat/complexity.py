"""Asymptotic work estimates for one JMAT forward (Phase 2 Task 1)."""

from __future__ import annotations

from dataclasses import dataclass

from jmat.model import JMATConfig


@dataclass(frozen=True)
class JMATComplexityReport:
    n: int
    nnz: int
    m: int
    d: int
    h: int
    t_loops: int
    use_global_attn: bool
    use_sparse_a: bool
    use_sparse_at: bool

    @property
    def attention_rows_per_loop(self) -> int:
        """Latent↔grid attention pairs (order-of-magnitude)."""
        return self.m * self.n + self.m * self.m + self.n * self.m

    def dense_mlp_flops_per_loop(self) -> int:
        """FLOPs from LayerNorm + concat MLP + (optional) output (one loop)."""
        d = self.d
        branches = 1 + int(self.use_global_attn) + int(self.use_sparse_a) + int(self.use_sparse_at)
        mix_in = branches * d
        # LayerNorm ~ 5d per row on H (n rows) + on Z (m rows); rough 5d(n+m)
        ln = 5 * d * (self.n + self.m)
        # MLP mix: (mix_in -> 2d -> d) per row: ~ 2*mix_in*2d + 2*2d*d = 4*mix_in*d + 4*d^2
        mlp = self.n * (4 * mix_in * d + 4 * d * d)
        # Three multi-head attentions: each ~ 4 n m d^2 (QKV + out proj) at leading order
        attn = 3 * self.attention_rows_per_loop * 4 * d * d
        return ln + mlp + attn

    def sparse_action_flops_per_loop(self) -> int:
        """nnz multiply-adds per sparse branch (A and A^T)."""
        ops = 0
        if self.use_sparse_a:
            ops += 2 * self.nnz * self.d
        if self.use_sparse_at:
            ops += 2 * self.nnz * self.d
        return ops

    def encoder_flops(self) -> int:
        """Per-index MLP(4 -> d -> d)."""
        return self.n * (8 * 4 * self.d + 8 * self.d * self.d)

    def output_flops(self) -> int:
        return self.n * (8 * self.d * self.d + 8 * self.d)

    def total_estimated_flops(self) -> int:
        loop = self.dense_mlp_flops_per_loop() + self.sparse_action_flops_per_loop()
        return self.encoder_flops() + self.t_loops * loop + self.output_flops()

    def summary_lines(self) -> list[str]:
        return [
            f"n={self.n}, nnz={self.nnz}, d={self.d}, m={self.m}, T={self.t_loops}",
            f"sparse actions/loop: {self.sparse_action_flops_per_loop():,} FLOPs",
            f"dense attn+MLP+LN/loop: {self.dense_mlp_flops_per_loop():,} FLOPs",
            f"encoder+output: {self.encoder_flops() + self.output_flops():,} FLOPs",
            f"total (estimate): {self.total_estimated_flops():,} FLOPs",
            "Scaling: O(T [ nnz·d + n·d² + (n·m+m²)·d² ]) dense part; O(T·nnz·d) sparse part.",
        ]


def complexity_report(
    n: int,
    nnz: int,
    cfg: JMATConfig,
) -> JMATComplexityReport:
    return JMATComplexityReport(
        n=n,
        nnz=nnz,
        m=cfg.n_latent,
        d=cfg.d_model,
        h=cfg.n_heads,
        t_loops=cfg.n_loops,
        use_global_attn=cfg.use_global_attn,
        use_sparse_a=cfg.use_sparse_a,
        use_sparse_at=cfg.use_sparse_at,
    )