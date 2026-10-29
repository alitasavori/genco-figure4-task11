"""Task 1.1 split checks against the converted dataset directory."""

from __future__ import annotations

import json
from pathlib import Path

from common import (
    EXPECTED_TEST_COUNTS,
    EXPECTED_TEST_TOTAL,
    EXPECTED_TRAIN_N,
    EXPECTED_TRAIN_POOL_N,
    EXPECTED_VAL_N,
)
from metrics import assert_no_overlap

TRAIN_ALLOWED = {("n", "feasible")}
TEST_REQUIRED = {
    ("n", "feasible"),
    ("n-1", "feasible"),
    ("n-2", "feasible"),
    ("n", "nose"),
    ("n-1", "nose"),
    ("n-2", "nose"),
}


def _case_root(data_root: Path, split: str) -> Path:
    direct = data_root / split / "case118"
    nested = data_root / "case118" / split / "case118"
    if direct.is_dir():
        return direct
    if nested.is_dir():
        return nested
    raise FileNotFoundError(f"No case118 folder for split {split!r} under {data_root}.")


def iter_condition_dirs(data_root: Path, split: str) -> dict[tuple[str, str], Path]:
    found: dict[tuple[str, str], Path] = {}
    case_root = _case_root(data_root, split)
    for topo in sorted(path.name for path in case_root.iterdir() if path.is_dir()):
        topo_dir = case_root / topo
        for condition in sorted(path.name for path in topo_dir.iterdir() if path.is_dir()):
            found[(topo, condition)] = topo_dir / condition
    return found


def parquet_rows(path: Path) -> int | None:
    try:
        import pyarrow.parquet as pq
    except Exception:
        return None
    return int(pq.ParquetFile(path).metadata.num_rows)


def id_set(condition_dir: Path) -> set[str] | None:
    bus_file = condition_dir / "raw" / "bus_data.parquet"
    if not bus_file.exists():
        return None
    try:
        import pyarrow.parquet as pq
    except Exception:
        return None
    schema = pq.read_schema(bus_file)
    names = set(schema.names)
    column = next((name for name in ("scenario_id", "sample_id", "scenario", "id") if name in names), None)
    if column is None:
        return None
    table = pq.read_table(bus_file, columns=[column])
    return {str(value) for value in table.column(column).to_pylist()}


def check_task11_layout(data_root: Path) -> dict:
    """Check that Task 1.1 train/valid stay on normal feasible topology."""
    report: dict = {"data_root": str(data_root), "splits": {}, "warnings": []}
    train = iter_condition_dirs(data_root, "train")
    valid_split = "valid" if _has_split(data_root, "valid") else "val"
    valid = iter_condition_dirs(data_root, valid_split)
    test = iter_condition_dirs(data_root, "test")
    unexpected_train = sorted(set(train) - TRAIN_ALLOWED)
    unexpected_valid = sorted(set(valid) - TRAIN_ALLOWED)
    if unexpected_train or unexpected_valid:
        raise AssertionError(
            "Task 1.1 training data includes a non-N or non-feasible condition: "
            f"train={unexpected_train}, valid={unexpected_valid}."
        )
    missing_test = sorted(TEST_REQUIRED - set(test))
    if missing_test:
        raise AssertionError(f"Task 1.1 test split is missing {missing_test}.")
    for split_name, mapping in (("train", train), ("valid", valid), ("test", test)):
        counts = {}
        for key, folder in mapping.items():
            bus_file = folder / "raw" / "bus_data.parquet"
            counts["/".join(key)] = parquet_rows(bus_file) if bus_file.exists() else None
        report["splits"][split_name] = counts
    train_ids = id_set(train[("n", "feasible")])
    test_ids: set[str] = set()
    saw_ids = train_ids is not None
    for key, folder in test.items():
        ids = id_set(folder)
        if ids is None:
            saw_ids = False
            break
        test_ids |= ids
    if saw_ids and train_ids is not None:
        assert_no_overlap(train_ids, test_ids)
        report["id_overlap_checked"] = True
    else:
        report["id_overlap_checked"] = False
        report["warnings"].append(
            "Parquet files have no scenario id column, so overlap was checked by split directory only."
        )
    report["expected"] = {
        "train_pool_before_validation": EXPECTED_TRAIN_POOL_N,
        "pfdelta_loader_train": EXPECTED_TRAIN_N,
        "pfdelta_loader_val": EXPECTED_VAL_N,
        "test_total": EXPECTED_TEST_TOTAL,
        "test_by_condition": {f"{cond}/{topo}": count for (cond, topo), count in EXPECTED_TEST_COUNTS.items()},
    }
    return report


def _has_split(data_root: Path, split: str) -> bool:
    return (data_root / split).is_dir() or (data_root / "case118" / split).is_dir()


def write_layout_report(data_root: Path, destination: Path) -> dict:
    report = check_task11_layout(data_root)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report
