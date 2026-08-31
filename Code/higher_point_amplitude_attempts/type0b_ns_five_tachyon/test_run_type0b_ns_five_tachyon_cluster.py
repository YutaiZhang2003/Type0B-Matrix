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
QUICK_CONFIG = CODE_ROOT / "config" / "type0b_ns_five_tachyon_one_hour_preliminary.json"


def _encoded(value: complex) -> dict[str, float]:
    return {"real": complex(value).real, "imag": complex(value).imag}


def _write_mock_shards(output_dir: Path, *, wrong_hash_shard: int | None = None, config_path: Path = CONFIG):
    config = _load_config(config_path)
    config_hash = _config_sha256(config_path)
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
                        {"passed": False} if shard_index < config["collar_certificate"]["audit_shards"] else None
                    ),
                    "replicates": 2,
                    "bulk_samples_per_replicate": 2 ** config["qmc"]["bulk_sobol_power"],
                    "face_samples_per_replicate": 2 ** config["qmc"]["face_sobol_power"],
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
    def test_quick_profile_retains_forest_and_skips_only_optional_audits(self):
        config = _load_config(QUICK_CONFIG)
        tasks = _tasks(config)
        seen = set()
        for task in tasks:
            seeds = set(range(task["seed"], task["seed"] + config["qmc"]["replicates_per_shard"]))
            self.assertFalse(seen & seeds)
            seen.update(seeds)
            args = _worker_arguments(config, task, Path("/tmp/quick-shard.json"))
            self.assertIn("--skip-face-collar-diagnostic", args)
            self.assertEqual("--skip-corner-contribution" in args, task["shard_index"] != 0)
            self.assertIn("--batch-c-evaluation", args)
            self.assertNotIn("--disable-momentum-singularity-subtraction", args)
        self.assertEqual(len(seen), 8)
        self.assertEqual(config["recursion"]["global_max_twice_levels"], [4, 4])

    def test_quick_reducer_does_not_label_an_omitted_audit_as_passed(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            _write_mock_shards(output, config_path=QUICK_CONFIG)
            summary = reduce_shards(QUICK_CONFIG, output, output / "summary.json")
            self.assertEqual(summary["schema"], "type0b-ns-fivepoint-preliminary-c-recursion-summary-v1")
            self.assertEqual(len(summary["radius_summaries"]), 1)
            self.assertIsNone(summary["radius_summaries"][0]["face_collar_certificates_passed"])
            self.assertEqual(summary["radius_summaries"][0]["bulk_samples"], 32)
            self.assertEqual(summary["radius_summaries"][0]["face_samples"], 64)
            self.assertEqual(summary["radius_summaries"][0]["integral_mean"], _encoded(3 + .5j))
            self.assertEqual(summary["collar_stability_differences"], [])

    def test_standard_profile_cannot_silently_skip_audits_or_reduce_cutoff(self):
        for change in ("cutoff", "audit", "seeds"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as directory:
                config = json.loads(CONFIG.read_text())
                if change == "cutoff":
                    config["recursion"]["global_max_twice_levels"] = [4, 4]
                elif change == "audit":
                    config["collar_certificate"]["audit_shards"] = 0
                else:
                    config["array"]["seed_stride"] = 1
                path = Path(directory) / "invalid.json"
                path.write_text(json.dumps(config))
                with self.assertRaises(ValueError):
                    _load_config(path)

    def test_production_config_rejects_h_and_hybrid_backends(self):
        for backend in ("h", "hybrid"):
            with self.subTest(backend=backend), tempfile.TemporaryDirectory() as directory:
                config = json.loads(CONFIG.read_text())
                config["recursion"]["block_backend"] = backend
                path = Path(directory) / "retired-backend.json"
                path.write_text(json.dumps(config))
                with self.assertRaisesRegex(ValueError, "requires all-c recursion"):
                    _load_config(path)

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
        self.assertIn("--c-coefficient-cache", arguments)
        self.assertIn("--checkpoint-directory", arguments)
        self.assertEqual(arguments[arguments.index("--block-cache-limit") + 1], "2048")
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
