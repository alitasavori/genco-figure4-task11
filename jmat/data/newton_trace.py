"""Newton trajectory logger scaffold (engineering brief §4).

Full PFΔ instrumentation (MATPOWER ``newtonpf`` / verified Python port) lands here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class NewtonStepRecord:
    scenario_id: str
    iteration: int
    crow: Any
    col: Any
    values: Any
    r: Any
    dx_star: Any
    x_k: Any
    meta: dict[str, Any]


def not_implemented_trace_from_pfdelta(*_args: Any, **_kwargs: Any) -> list[NewtonStepRecord]:
    raise NotImplementedError(
        "PFΔ Newton trace generation is specified in docs/jmat_engineering_brief_v1.tex; "
        "implement prepare_data.py next."
    )