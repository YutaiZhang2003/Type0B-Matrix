"""Regression tests for the momentum-sharded torus modular driver."""

from __future__ import annotations

import cmath
import json
import math
from pathlib import Path
import tempfile
import unittest

from super_liouville_torus_modular_cluster import (
    atomic_jobs,
    config_sha256,
    implementation_manifest,
    jobs_for_shard,
    reduce_shards,
    run_shard,
)
from super_liouville_torus_one_point import (
    Type0BNSOnePointQuadrature,
    Type0BROnePointQuadrature,
    ns_lift_sign_from_tau,
)
from plot_torus_modularity_q_scan import (
    plot_scan,
    validate_summary_config,
)


def _small_config() -> dict[str, object]:
    return {
        "schema_version": 1,
        "external_momentum": 0.33,
        "levels": [0, 2],
        "taus": [
            {"name": "test", "tau": [0.45, 0.65]},
        ],
        "studies": [
            {
                "name": "small",
                "quadrature_order": 4,
                "p_max": 3.0,
                "structure_precision": 20,
                "finite_part_samples": 8,
            }
        ],
        "comparisons": [],
        "accuracy_targets": [],
        "default_shard_count": 2,
    }


def _small_spin_orbit_config() -> dict[str, object]:
    payload = _small_config()
    payload["modular_orbit"] = "ns_tilde_to_r"
    return payload


class SuperLiouvilleTorusModularClusterTests(unittest.TestCase):
    def test_shards_partition_atomic_jobs_exactly_once(self):
        payload = _small_config()
        jobs = atomic_jobs(payload)
        sharded = [
            job
            for shard_id in range(3)
            for job in jobs_for_shard(payload, shard_id, 3)
        ]
        self.assertEqual(
            sorted(int(job["job_id"]) for job in sharded),
            list(range(len(jobs))),
        )

    def test_sharded_reduction_matches_serial_quadrature(self):
        payload = _small_config()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = root / "config.json"
            shards = root / "shards"
            summary_path = root / "summary.json"
            config.write_text(json.dumps(payload))
            for shard_id in range(2):
                run_shard(config, shards, shard_id, 2)
            summary = reduce_shards(
                config_path=config,
                input_dir=shards,
                output_path=summary_path,
                shard_count=2,
            )

            tau = 0.45 + 0.65j
            s_tau = -1.0 / tau
            q = cmath.exp(2.0j * math.pi * tau)
            q_tilde = cmath.exp(2.0j * math.pi * s_tau)
            quadrature = Type0BNSOnePointQuadrature(
                external_momentum=0.33,
                max_twice_level=2,
                p_max=3.0,
                quadrature_order=4,
                structure_precision=20,
                finite_part_samples=8,
            )
            serial_q = quadrature.evaluate(
                q,
                lift_sign=ns_lift_sign_from_tau(tau),
                max_twice_level=2,
            )
            serial_q_tilde = quadrature.evaluate(
                q_tilde,
                lift_sign=ns_lift_sign_from_tau(s_tau),
                max_twice_level=2,
            )
            reduced = summary["studies"]["small"]["taus"]["test"]
            reduced_level = reduced["levels"]["2"]
            self.assertEqual(
                (reduced["lift_sign"], reduced["lift_sign_tilde"]),
                (1, -1),
            )
            self.assertAlmostEqual(
                complex(*reduced_level["value_q"]), serial_q, places=14
            )
            self.assertAlmostEqual(
                complex(*reduced_level["value_q_tilde"]),
                serial_q_tilde,
                places=14,
            )
            self.assertTrue(summary["accuracy_targets_passed"])
            self.assertEqual(
                summary["implementation_sha256"],
                implementation_manifest()["sha256"],
            )
            self.assertIn(
                "numpy_version",
                summary["implementation"]["environment"],
            )
            self.assertIn(
                "mpmath_version",
                summary["implementation"]["environment"],
            )
            self.assertTrue(summary_path.exists())

    def test_spin_orbit_reduction_matches_serial_quadratures(self):
        payload = _small_spin_orbit_config()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = root / "config.json"
            shards = root / "shards"
            summary_path = root / "summary.json"
            config.write_text(json.dumps(payload))
            for shard_id in range(2):
                run_shard(config, shards, shard_id, 2)
            summary = reduce_shards(
                config_path=config,
                input_dir=shards,
                output_path=summary_path,
                shard_count=2,
            )

            tau = 0.45 + 0.65j
            s_tau = -1.0 / tau
            q = cmath.exp(2.0j * math.pi * tau)
            q_tilde = cmath.exp(2.0j * math.pi * s_tau)
            ns_quadrature = Type0BNSOnePointQuadrature(
                external_momentum=0.33,
                max_twice_level=2,
                p_max=3.0,
                quadrature_order=4,
                structure_precision=20,
                finite_part_samples=8,
            )
            r_quadrature = Type0BROnePointQuadrature(
                external_momentum=0.33,
                max_level=1,
                p_max=3.0,
                quadrature_order=4,
                structure_precision=20,
                finite_part_samples=8,
            )
            serial_q = ns_quadrature.evaluate(
                q,
                lift_sign=-ns_lift_sign_from_tau(tau),
                max_twice_level=2,
            )
            serial_q_tilde = r_quadrature.evaluate(
                q_tilde,
                max_level=1,
            )
            reduced = summary["studies"]["small"]["taus"]["test"]
            reduced_level = reduced["levels"]["2"]
            self.assertEqual(summary["modular_orbit"], "ns_tilde_to_r")
            self.assertEqual(
                (
                    reduced["direct_spin_structure"],
                    reduced["transformed_spin_structure"],
                ),
                ("NS_tilde", "R"),
            )
            self.assertEqual(
                (reduced["lift_sign"], reduced["lift_sign_tilde"]),
                (1, -1),
            )
            self.assertAlmostEqual(
                complex(*reduced_level["value_q"]), serial_q, places=14
            )
            self.assertAlmostEqual(
                complex(*reduced_level["value_q_tilde"]),
                serial_q_tilde,
                places=14,
            )

    def test_complete_shard_is_idempotent(self):
        payload = _small_config()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = root / "config.json"
            shards = root / "shards"
            config.write_text(json.dumps(payload))
            first = run_shard(config, shards, 0, 2)
            second = run_shard(config, shards, 0, 2)
            self.assertEqual(first["status"], "computed")
            self.assertEqual(second["status"], "already-complete")

    def test_changed_implementation_invalidates_cached_shard(self):
        payload = _small_config()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = root / "config.json"
            shards = root / "shards"
            config.write_text(json.dumps(payload))
            run_shard(config, shards, 0, 2)
            shard_path = shards / "shard-0000.jsonl"
            records = [
                json.loads(line) for line in shard_path.read_text().splitlines()
            ]
            records[0]["implementation_sha256"] = "0" * 64
            shard_path.write_text(
                "".join(
                    json.dumps(record, sort_keys=True) + "\n"
                    for record in records
                )
            )
            refreshed = run_shard(config, shards, 0, 2)
            self.assertEqual(refreshed["status"], "computed")

    def test_reducer_rejects_mixed_implementation_shards(self):
        payload = _small_config()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = root / "config.json"
            shards = root / "shards"
            config.write_text(json.dumps(payload))
            for shard_id in range(2):
                run_shard(config, shards, shard_id, 2)
            shard_path = shards / "shard-0001.jsonl"
            records = [
                json.loads(line) for line in shard_path.read_text().splitlines()
            ]
            records[0]["implementation_sha256"] = "f" * 64
            shard_path.write_text(
                "".join(
                    json.dumps(record, sort_keys=True) + "\n"
                    for record in records
                )
            )
            with self.assertRaisesRegex(
                ValueError, "implementation fingerprint mismatch"
            ):
                reduce_shards(
                    config_path=config,
                    input_dir=shards,
                    output_path=root / "summary.json",
                    shard_count=2,
                )

    def test_plotter_rejects_mismatched_configuration(self):
        payload = _small_config()
        summary = {"config_sha256": config_sha256(payload)}
        self.assertEqual(
            validate_summary_config(summary, payload),
            config_sha256(payload),
        )
        changed = dict(payload)
        changed["external_momentum"] = 0.34
        with self.assertRaisesRegex(
            ValueError, "summary/configuration digest mismatch"
        ):
            validate_summary_config(summary, changed)

    def test_q_scan_certifies_principal_and_nonprincipal_modularity(self):
        config_path = (
            Path(__file__).resolve().parent
            / "config"
            / "type0b_torus_modular_q_scan_cluster.json"
        )
        payload = json.loads(config_path.read_text())
        modular_targets = {
            (str(target["tau"]), int(target["level"]))
            for target in payload["accuracy_targets"]
            if target["kind"] == "modular_residual"
        }
        self.assertIn(("radial_x020_y085", 12), modular_targets)
        self.assertIn(("lift_x045_y065", 12), modular_targets)

    def test_stored_scan_fit_is_reproduced_and_recorded(self):
        run_root = (
            Path(__file__).resolve().parent.parent
            / "Data Set"
            / "results"
            / "type0b_torus_modular_cluster"
            / "cannon_qscan_20260723_v1"
        )
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "scan.svg"
            plot_scan(
                summary_path=run_root / "summary.json",
                config_path=run_root / "config.snapshot.json",
                output_path=output,
            )
            fit = json.loads(
                output.with_name("scan.fit.json").read_text()
            )
            self.assertAlmostEqual(fit["exponent"], 6.461873097, places=9)
            self.assertEqual(fit["included_point_count"], 7)
            self.assertEqual(fit["residual_floor_exclusive"], 1.0e-15)
            self.assertEqual(
                fit["expected_first_omitted_exponent"], 6.5
            )
            self.assertIn(
                "fit 6.4619 (δ &gt; 10⁻¹⁵); expected 6.5",
                output.read_text(),
            )


if __name__ == "__main__":
    unittest.main()
