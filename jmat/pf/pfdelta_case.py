"""Load PFΔ JSON into tensors for Newton logging."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from jsgc_physics import TYPE_PQ, TYPE_PV, TYPE_SLACK
from pfdelta_case_parquet import _bus_gen, _bus_load, _network_and_solution, build_ybus


def _pm_bus_type(bus: dict[str, Any]) -> int:
    """PowerModels bus_type 1=PQ, 2=PV, 3=ref → JSGC 0/1/2."""
    bt = int(bus.get("bus_type", 1))
    if bt == 3:
        return TYPE_SLACK
    if bt == 2:
        return TYPE_PV
    return TYPE_PQ


def case_from_pfdelta_json(pm_case: dict[str, Any]) -> dict[str, Any]:
    """Build per-unit Ybus, specs, and bus types from one PFΔ sample."""
    network, solution, base_mva = _network_and_solution(pm_case)
    bus_ids = sorted(int(b) for b in network["bus"])
    n = len(bus_ids)
    index = {bid: i for i, bid in enumerate(bus_ids)}
    ybus = build_ybus(network)
    g = np.real(ybus)
    b = np.imag(ybus)
    types = np.zeros(n, dtype=np.int64)
    p_spec = np.zeros(n, dtype=np.float64)
    q_spec = np.zeros(n, dtype=np.float64)
    v_set = np.ones(n, dtype=np.float64)

    for bid in bus_ids:
        i = index[bid]
        bus = network["bus"][str(bid)]
        types[i] = _pm_bus_type(bus)
        pd, qd = _bus_load(network, bid)
        pg, qg = _bus_gen(network, solution, bid)
        # Net injection specification (generation − load) in p.u.
        p_spec[i] = pg - pd
        q_spec[i] = qg - qd
        if types[i] in (TYPE_PV, TYPE_SLACK):
            vmin = float(bus.get("vmin", 0.9))
            vmax = float(bus.get("vmax", 1.1))
            v_set[i] = float(np.clip(float(bus.get("vm", 1.0)), vmin, vmax))
            gen_vm = None
            for gen in network.get("gen", {}).values():
                if int(gen.get("gen_bus", -1)) == bid and int(gen.get("gen_status", 1)) == 1:
                    gen_vm = float(gen.get("vg", v_set[i]))
                    break
            if gen_vm is not None:
                v_set[i] = gen_vm

    return {
        "g": torch.tensor(g, dtype=torch.float64),
        "b_mat": torch.tensor(b, dtype=torch.float64),
        "types": torch.tensor(types, dtype=torch.long),
        "p_spec": torch.tensor(p_spec, dtype=torch.float64),
        "q_spec": torch.tensor(q_spec, dtype=torch.float64),
        "v_set": torch.tensor(v_set, dtype=torch.float64),
        "base_mva": base_mva,
        "n_buses": n,
        "network": network,
        "solution": solution,
    }