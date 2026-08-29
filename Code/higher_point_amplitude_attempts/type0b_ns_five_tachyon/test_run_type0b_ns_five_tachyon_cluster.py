"""Regression tests for the coupled-radius coefficient-table cluster driver."""

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
CONFIG = CODE_ROOT / "config" / "type0b_ns_five_tachyon_order8_small_collar_cluster.json"


def _encoded(value: complex) -> dict[str, float]:
    return {"real": complex(value).real, "imag": complex(value).imag}


class Type0BFivePointClusterTests(unittest.TestCase):
    def test_plan_couples_all_radii_and_fits_inside_four_shards(self):
        config = _load_config(CONFIG)
        tasks = _tasks(config)
        self.assertEqual(len(tasks), 4)
        self.assertEqual([task["shard_index"] for task in tasks], list(range(4)))
        self.assertTrue(all("collar_radius" not in task for task in tasks))
        self.assertTrue(all("regulator_eta" not in task for task in tasks))

        arguments = _worker_arguments(
            config, tasks[0], Path("/tmp/type0b-fivepoint-cluster-test.json")
        )
        radii_index = arguments.index("--collar-radii")
        self.assertEqual(
            tuple(map(float, arguments[radii_index + 1 : radii_index + 4])),
            (0.01, 0.005, 0.0025),
        )
        self.assertIn("--include-comparison-fit", arguments)
        self.assertIn("--h-regulator-etas", arguments)
        self.assertNotIn("--h-regulator-eta", arguments)
        self.assertNotIn("--enforce-face-collar-certificate", arguments)

    def test_reducer_uses_paired_fit_and_collar_differences(self):
        config = _load_config(CONFIG)
        config_hash = _config_sha256(CONFIG)
        radii = tuple(config["subtraction"]["collar_radii"])
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            for task in _tasks(config):
                shard_index = int(task["shard_index"])
                results = []
                for variant in ("production", "comparison"):
                    fit_offset = 0.0 if variant == "production" else -0.25
                    for radius_index, radius in enumerate(radii):
                        bulk_values = (
                            complex(10 * radius_index + shard_index + fit_offset, 0.5),
                            complex(10 * radius_index + shard_index + 1 + fit_offset, 0.5),
                        )
                        results.append(
                            {
                                "h_fit_variant": variant,
                                "radius_index": radius_index,
                                "collar_radius": radius,
                                "bulk_estimates": [_encoded(value) for value in bulk_values],
                                "face_estimates": [_encoded(0j), _encoded(0j)],
                                "corner_contribution": _encoded(1.0 + 0.0j),
                                "corner_contribution_computed": shard_index == 0,
                                "face_collar_certificate": (
                                    {"passed": False}
                                    if variant == "production" and shard_index == 0
                                    else None
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
                        "config_sha256": config_hash,
                    },
                    "self_dual_coefficient_fit": {"block_count": 1},
                    "results": results,
                }
                (output_dir / f"task_{int(task['task_index']):05d}.json").write_text(
                    json.dumps(payload)
                )

            summary_path = output_dir / "summary.json"
            summary = reduce_shards(CONFIG, output_dir, summary_path)
            self.assertEqual(
                summary["schema"],
                "type0b-ns-fivepoint-order8-coefficient-table-summary-v4",
            )
            self.assertEqual(len(summary["radius_summaries"]), 3)
            first_shift = summary["radius_summaries"][0][
                "coefficient_fit_shift_mean"
            ]
            self.assertAlmostEqual(first_shift["real"], 0.25)
            self.assertAlmostEqual(first_shift["imag"], 0.0)
            self.assertEqual(len(summary["collar_stability_differences"]), 2)
            self.assertFalse(
                summary["radius_summaries"][0][
                    "face_collar_certificates_passed"
                ]
            )
            self.assertTrue(summary_path.exists())

    def test_reducer_accepts_only_the_declared_previous_non_audit_shard_hashes(self):
        config = _load_config(CONFIG)
        config_hash = _config_sha256(CONFIG)
        compatible = config["merge"]["compatible_shard_config_sha256"]
        radii = tuple(config["subtraction"]["collar_radii"])
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            for task in _tasks(config):
                shard_index = int(task["shard_index"])
                results = []
                for variant in ("production", "comparison"):
                    for radius_index, radius in enumerate(radii):
                        results.append(
                            {
                                "h_fit_variant": variant,
                                "radius_index": radius_index,
                                "collar_radius": radius,
                                "bulk_estimates": [_encoded(complex(shard_index, 0.0))] * 2,
                                "face_estimates": [_encoded(0j)] * 2,
                                "corner_contribution": _encoded(1.0 + 0.0j),
                                "corner_contribution_computed": shard_index == 0,
                                "face_collar_certificate": (
                                    {"passed": True}
                                    if variant == "production" and shard_index == 0
                                    else None
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
                            config_hash
                            if shard_index == 0
                            else compatible[str(shard_index)]
                        ),
                    },
                    "self_dual_coefficient_fit": {"block_count": 1},
                    "results": results,
                }
                (output_dir / f"task_{shard_index:05d}.json").write_text(
                    json.dumps(payload)
                )

            summary = reduce_shards(CONFIG, output_dir, output_dir / "summary.json")
            self.assertEqual(summary["merged_shard_config_sha256"]["0"], config_hash)
            for shard_index in range(1, 4):
                self.assertEqual(
                    summary["merged_shard_config_sha256"][str(shard_index)],
                    compatible[str(shard_index)],
                )


if __name__ == "__main__":
    unittest.main()
