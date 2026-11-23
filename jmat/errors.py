"""JMAT input validation errors."""

from __future__ import annotations


class InvalidLinearSystemError(ValueError):
    """Raised when (J, r) cannot define a meaningful Newton correction."""