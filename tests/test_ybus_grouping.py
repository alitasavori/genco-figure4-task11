"""Y-bus fingerprint distinguishes networks for split grouping."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from train_tpgt_task11 import ybus_entries_fingerprint  # noqa: E402


def test_ybus_fingerprint_changes_when_admittance_changes():
    base = [(0, 1, 1.0, 0.5), (1, 0, 1.0, 0.5)]
    changed = [(0, 1, 1.0, 0.6), (1, 0, 1.0, 0.6)]
    assert ybus_entries_fingerprint(base) != ybus_entries_fingerprint(changed)


def test_ybus_fingerprint_is_order_independent():
    entries_a = [(0, 1, 1.0, 0.0), (2, 3, 0.5, -0.1)]
    entries_b = [(2, 3, 0.5, -0.1), (0, 1, 1.0, 0.0)]
    assert ybus_entries_fingerprint(entries_a) == ybus_entries_fingerprint(entries_b)
