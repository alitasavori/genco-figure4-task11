"""Jacobian Matrix-Action Transformer (JMAT) — sparse Newton correction learner."""

from jmat.loss import jmat_combined_loss
from jmat.model import JMAT, JMATConfig
from jmat.normalize import NormalizedLinearSystem, normalize_linear_system

__all__ = [
    "JMAT",
    "JMATConfig",
    "NormalizedLinearSystem",
    "normalize_linear_system",
    "jmat_combined_loss",
]