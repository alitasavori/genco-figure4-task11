"""Checks for the Task 1.1 metric, splits, and figure sources."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from checks import check_task11_layout
from common import EXPECTED_TEST_COUNTS, EXPECTED_TEST_TOTAL, EXPECTED_TRAIN_N, EXPECTED_TRAIN_POOL_N, EXPECTED_VAL_N
from metrics import (
    BUS_PQ,
    BUS_PV,
    BUS_REF,
    aggregate_c2i,
    assert_no_overlap,
    mean_power_balance_loss,
    sample_mean,
    sample_std,
    specified_quantities,
)
from plot_task1_1 import plot_task11
from published import parse_metric, published_stats, write_published_tables


def _make_tree(root: Path, split: str, conditions: list[tuple[str, str]]) -> None:
    for topo, condition in conditions:
        folder = root / "case118" / split / "case118" / topo / condition / "raw"
        folder.mkdir(parents=True, exist_ok=True)


class MetricTests(unittest.TestCase):
    def test_pure_reactance_flat_start_is_balanced(self) -> None:
        # Equal voltages, zero angle difference, pure reactance, no shunt: flows are zero.
        loss = mean_power_balance_loss(
            voltage=[1.0, 1.0],
            angle=[0.0, 0.0],
            p_net=[0.0, 0.0],
            q_net=[0.0, 0.0],
            shunt_g=[0.0, 0.0],
            shunt_b=[0.0, 0.0],
            src=[0],
            dst=[1],
            resistance=[0.0],
            reactance=[0.1],
            g_from=[0.0],
            b_from=[0.0],
            g_to=[0.0],
            b_to=[0.0],
            tap=[1.0],
            shift=[0.0],
            bus_types=[BUS_REF, BUS_PQ],
        )
        self.assertAlmostEqual(loss, 0.0, places=12)

    def test_reference_bus_stays_in_the_average(self) -> None:
        loss = mean_power_balance_loss(
            voltage=[1.0, 1.0, 1.0],
            angle=[0.0, 0.0, 0.0],
            p_net=[0.3, 0.0, 0.0],
            q_net=[0.0, 0.0, 0.0],
            shunt_g=[0.0, 0.0, 0.0],
            shunt_b=[0.0, 0.0, 0.0],
            src=[0, 1],
            dst=[1, 2],
            resistance=[0.0, 0.0],
            reactance=[0.1, 0.1],
            g_from=[0.0, 0.0],
            b_from=[0.0, 0.0],
            g_to=[0.0, 0.0],
            b_to=[0.0, 0.0],
            tap=[1.0, 1.0],
            shift=[0.0, 0.0],
            bus_types=[BUS_REF, BUS_PV, BUS_PQ],
        )
        self.assertAlmostEqual(loss, 0.1, places=12)
        again = mean_power_balance_loss(
            voltage=[1.0, 1.0, 1.0],
            angle=[0.0, 0.0, 0.0],
            p_net=[0.3, 0.0, 0.0],
            q_net=[0.0, 0.0, 0.0],
            shunt_g=[0.0, 0.0, 0.0],
            shunt_b=[0.0, 0.0, 0.0],
            src=[0, 1],
            dst=[1, 2],
            resistance=[0.0, 0.0],
            reactance=[0.1, 0.1],
            g_from=[0.0, 0.0],
            b_from=[0.0, 0.0],
            g_to=[0.0, 0.0],
            b_to=[0.0, 0.0],
            tap=[1.0, 1.0],
            shift=[0.0, 0.0],
            bus_types=[BUS_REF, BUS_PV, BUS_PQ],
        )
        self.assertEqual(loss, again)

    def test_non_finite_prediction_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            mean_power_balance_loss(
                voltage=[1.0, np.nan],
                angle=[0.0, 0.0],
                p_net=[0.0, 0.0],
                q_net=[0.0, 0.0],
                shunt_g=[0.0, 0.0],
                shunt_b=[0.0, 0.0],
                src=[0],
                dst=[1],
                resistance=[0.0],
                reactance=[0.1],
                g_from=[0.0],
                b_from=[0.0],
                g_to=[0.0],
                b_to=[0.0],
                tap=[1.0],
                shift=[0.0],
            )

    def test_bus_quantities(self) -> None:
        self.assertEqual(specified_quantities(BUS_PQ), ("P", "Q"))
        self.assertEqual(specified_quantities(BUS_PV), ("P", "V"))
        self.assertEqual(specified_quantities(BUS_REF), ("V", "theta"))

    def test_sample_std_matches_unbiased_definition(self) -> None:
        self.assertAlmostEqual(sample_mean([1.0, 2.0, 3.0]), 2.0)
        self.assertAlmostEqual(sample_std([1.0, 2.0, 3.0]), 1.0)
        with self.assertRaises(ValueError):
            sample_std([1.0, float("nan"), 3.0])

    def test_c2i_requires_every_topology(self) -> None:
        with self.assertRaises(ValueError):
            aggregate_c2i({"C2I-N": 1.0, "C2I-N-1": 2.0})
        self.assertAlmostEqual(aggregate_c2i({"C2I-N": 1.0, "C2I-N-1": 2.0, "C2I-N-2": 3.0}), 2.0)
        weighted = aggregate_c2i(
            {"C2I-N": 1.0, "C2I-N-1": 2.0, "C2I-N-2": 3.0},
            {"C2I-N": 1, "C2I-N-1": 1, "C2I-N-2": 2},
        )
        self.assertAlmostEqual(weighted, (1.0 + 2.0 + 6.0) / 4.0)


class SplitTests(unittest.TestCase):
    def test_official_counts(self) -> None:
        self.assertEqual(EXPECTED_TRAIN_POOL_N, 54000)
        self.assertEqual(EXPECTED_TRAIN_N + EXPECTED_VAL_N, EXPECTED_TRAIN_POOL_N)
        self.assertEqual(EXPECTED_TEST_TOTAL, 6600)
        self.assertEqual(sum(EXPECTED_TEST_COUNTS.values()), 6600)

    def test_no_overlap_and_task11_directories(self) -> None:
        assert_no_overlap(["a", "b"], ["c"])
        with self.assertRaises(AssertionError):
            assert_no_overlap(["a", "b"], ["b", "c"])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            allowed = [("n", "feasible")]
            test_conditions = [
                ("n", "feasible"),
                ("n-1", "feasible"),
                ("n-2", "feasible"),
                ("n", "nose"),
                ("n-1", "nose"),
                ("n-2", "nose"),
            ]
            _make_tree(root, "train", allowed)
            _make_tree(root, "valid", allowed)
            _make_tree(root, "test", test_conditions)
            report = check_task11_layout(root)
            self.assertFalse(report["id_overlap_checked"])
            _make_tree(root, "train", [("n-1", "feasible")])
            with self.assertRaises(AssertionError):
                check_task11_layout(root)


class PublishedFigureTests(unittest.TestCase):
    def test_table_values(self) -> None:
        mean, std = parse_metric(r"2.7e-2\std{0.03e-2}")
        self.assertAlmostEqual(mean, 0.027)
        self.assertAlmostEqual(std, 0.0003)
        stats = published_stats()
        self.assertAlmostEqual(stats["GENCO Base"]["N"][0], 0.0014)
        self.assertAlmostEqual(stats["GENCO Base"]["C2I"][0], 0.30)
        self.assertAlmostEqual(stats["PFNet"]["C2I"][0], 1.4)
        self.assertAlmostEqual(stats["CANOS-PF"]["N-1"][0], 0.70)
        self.assertAlmostEqual(stats["GNS-S"]["N"][1], 0.27)

    def test_retrained_figure_is_not_filled_from_the_table(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            write_published_tables(folder)
            plot_task11(folder)
            self.assertTrue((folder / "figure4_task1.1_published.pdf").is_file())
            self.assertTrue((folder / "figure4_task1.1_published.png").is_file())
            self.assertFalse((folder / "figure4_task1.1_retrained.pdf").exists())
            note = (folder / "figure4_task1.1_retrained_NOT_RUN.txt").read_text(encoding="utf-8")
            self.assertIn("Published table values were not copied", note)


if __name__ == "__main__":
    unittest.main()
