"""Print TPGT parameter breakdown for current ARCH preset (default v9 via env)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ.setdefault("GENCO_TPGT_ARCH", "v9")
import train_tpgt_task11 as t  # noqa: E402

t.apply_architecture_config()


def mlp_params(in_f: int, out_f: int, width: int = t.D) -> int:
    return in_f * width + width + width * width + width + width * out_f + out_f


def stage_attn_params() -> int:
    """Attention weights for one correction stage under the active ``ATTENTION_KIND``."""
    total = 0
    # Shared: W_Q, W_K, W_V per type-pair head; W_O per head.
    for rows, cols in ((t.D_A, t.D), (t.D_A, t.D), (t.D_A, t.D)):
        total += 9 * t.HEADS * rows * cols
    total += t.HEADS * t.D * t.D_A
    if t.ATTENTION_KIND == "joint_mlp":
        score_in = 2 * t.D_A + t.RPE_DIM
        hidden = int(t.ATTENTION_MLP_HIDDEN)
        # W1 (hidden, score_in) + b1; W2 (1, hidden) + b2 — per type-pair head.
        total += 9 * t.HEADS * (hidden * score_in + hidden + 1 * hidden + 1)
    else:
        for extra in (
            (t.D_A, t.D_E),
            (t.D_A, t.D_E),
            (1, t.D_A),
            (t.D_A, t.D_A),
        ):
            total += 9 * t.HEADS * extra[0] * extra[1]
    return total


def main() -> None:
    attn = stage_attn_params()
    update = 3 * mlp_params(t.D + t.D + 2, t.D)
    decode = 3 * mlp_params(t.D, 2)
    ln = 2 * t.D
    per_stage = attn + update + decode + ln

    node_in = t.NODE_FEATURE_DIM if t.USE_NODE_SHUNT else t.NODE_SPECIFIED_DIM
    enc_one = mlp_params(node_in, t.D)
    pe_one = 0 if t.ATTENTION_KIND == "joint_mlp" else mlp_params(t.RPE_DIM, t.D_E)

    model = t.TPGT()
    total = sum(p.numel() for p in model.parameters())

    print(
        f"ARCH={os.environ.get('GENCO_TPGT_ARCH', 'v9')}: "
        f"D={t.D} D_E={t.D_E} D_A={t.D_A} HEADS={t.HEADS} DEPTH={t.DEPTH} "
        f"node_in={node_in} RPE_DIM={t.RPE_DIM} ATTENTION_KIND={t.ATTENTION_KIND}\n"
    )

    rows: list[tuple[str, int, int]] = [
        ("Type encoder MLP (per bus type)", enc_one, 3),
        (
            "Pair encoder MLP (RPE -> D_E)" if pe_one else "Pair encoder (Identity / raw RPE)",
            pe_one,
            1,
        ),
        ("Correction stage: type-pair attention", attn, t.DEPTH),
        ("Correction stage: update MLPs (3 x)", update, t.DEPTH),
        ("Correction stage: decode MLPs (3 x)", decode, t.DEPTH),
        ("Correction stage: LayerNorm(D)", ln, t.DEPTH),
    ]

    print(f"{'Component':<48} {'Per unit':>12} {' x':>4} {'Subtotal':>12}")
    print("-" * 80)
    grand = 0
    for name, per, mult in rows:
        sub = per * mult
        grand += sub
        print(f"{name:<48} {per:>12,} {mult:>4} {sub:>12,}")
    print("-" * 80)
    print(f"{'Total (sum of table)':<48} {'':>12} {'':>4} {grand:>12,}")
    print(f"{'torch.nn count (model)':<48} {'':>12} {'':>4} {total:>12,}")
    print(f"\nPer correction stage subtotal: {per_stage:,}  (x{t.DEPTH} = {per_stage * t.DEPTH:,})")
    print(
        "\nMLP formula (3-layer): in*D + D + D*D + D + D*out + out "
        f"with hidden D={t.D}."
    )
    if t.ATTENTION_KIND == "joint_mlp":
        score_in = 2 * t.D_A + t.RPE_DIM
        print(
            "Attention (joint MLP on raw RPE): 9 type-pairs x HEADS x DEPTH, "
            f"W_Q/W_K/W_V: ({t.D_A},{t.D}), score MLP "
            f"({score_in} -> {t.ATTENTION_MLP_HIDDEN} -> 1) with r_ij dim={t.RPE_DIM}, "
            f"W_O: ({t.D},{t.D_A}). No pair encoder."
        )
    else:
        print(
            "Attention (GRIT): 9 directed type-pairs x HEADS, shapes "
            "W_Q/W_K/W_V: (D_A,D), W_EW/W_EB: (D_A,D_E), W_A: (D_A,), "
            f"W_EV: (D_A,D_A), W_O: ({t.D},{t.D_A})."
        )


if __name__ == "__main__":
    main()
