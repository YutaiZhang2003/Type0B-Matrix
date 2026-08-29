"""Regression tests for the pure-c five-point cluster driver."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from run_type0b_ns_five_tachyon_cluster import (
    _config_sha256,
    _load_config,
    _tasks,
    _worker_arguments,
    reduce_shards,
)


CODE_ROOT = Path(__file__).resolve().parents[2]
CONFIG = (
    CODE_ROOT
    / "config"
    / "type0b_ns_five_tachyon_c_recursion_order8_small_collar_cluster.json"
)


def _encoded(value: complex) -> dict[str, float]:
    return {"real": complex(value).real, "imag": complex(value).imag}


def _write_mock_shards(output_dir: Path, *, wrong_hash_shard: int | None = None):
    config = _load_config(CONFIG)
    config_hash = _config_sha256(CONFIG)
    radii = tuple(config["subtraction"]["collar_radii"])
    for task in _tasks(config):
        shard_index = int(task["shard_index"])
        results = []
        for radius_index, radius in enumerate(radii):
            bulk_values = (
                complex(10 * radius_index + shard_index, 0.5),
                complex(10 * radius_index + shard_index + 1, 0.5),
            )
            results.append(
                {
                    "recursion_variant": "production",
                    "h_fit_variant": "production",
                    "radius_index": radius_index,
                    "collar_radius": radius,
                    "bulk_estimates": [_encoded(value) for value in bulk_values],
                    "face_estimates": [_encoded(0j), _encoded(0j)],
                    "corner_contribution": _encoded(1.0 + 0.0j),
                    "corner_contribution_computed": shard_index == 0,
                    "face_collar_certificate": (
                        {"passed": False} if shard_index == 0 else None
                    ),
                    "replicates": 2,
                    "bulk_samples_per_replicate": 8,
                    "face_samples_per_replicate": 16,
                }
            )
        payload = {
            "schema": "type0b-ns-fivepoint-coupled-collar-fit-bundle-v1",
            "cluster_task": {
                **task,
                "config_sha256": (
                    "0" * 64 if wrong_hash_shard == shard_index else config_hash
                ),
            },
            "self_dual_coefficient_fit": None,
            "results": results,
        }
        (output_dir / f"task_{shard_index:05d}.json").write_text(
            json.dumps(payload)
        )


class Type0BFivePointClusterTests(unittest.TestCase):
    def test_plan_is_four_shard_pure_c_recursion(self):
        config = _load_config(CONFIG)
        tasks = _tasks(config)
        self.assertEqual(len(tasks), 4)
        self.assertEqual(config["recursion"]["block_backend"], "c")
        arguments = _worker_arguments(
            config, tasks[0], Path("/tmp/type0b-fivepoint-cluster-test.json")
        )
        backend_index = arguments.index("--block-backend")
        self.assertEqual(arguments[backend_index + 1], "c")
        self.assertNotIn("--h-regulator-etas", arguments)
        self.assertNotIn("--include-comparison-fit", arguments)
        self.assertNotIn("--enforce-face-collar-certificate", arguments)
        radii_index = arguments.index("--collar-radii")
        self.assertEqual(
            tuple(map(float, arguments[radii_index + 1 : radii_index + 4])),
            (0.01, 0.005, 0.0025),
        )

    def test_reducer_combines_c_shards_without_h_fit_axis(self):
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            _write_mock_shards(output_dir)
            summary_path = output_dir / "summary.json"
            summary = reduce_shards(CONFIG, output_dir, summary_path)
            self.assertEqual(
                summary["schema"],
                "type0b-ns-fivepoint-order8-c-recursion-summary-v5",
            )
            self.assertEqual(len(summary["radius_summaries"]), 3)
            first = summary["radius_summaries"][0]
            self.assertEqual(first["block_backend"], "c")
            self.assertFalse(first["regulator_extrapolated_per_coefficient"])
            self.assertNotIn("coefficient_fit_shift_mean", first)
            self.assertFalse(first["face_collar_certificates_passed"])
            self.assertEqual(len(summary["collar_stability_differences"]), 2)
            self.assertTrue(summary_path.exists())

    def test_reducer_rejects_an_undeclared_shard_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            _write_mock_shards(output_dir, wrong_hash_shard=2)
            with self.assertRaisesRegex(ValueError, "config hash mismatch"):
                reduce_shards(CONFIG, output_dir, output_dir / "summary.json")


if __name__ == "__main__":
    unittest.main()
