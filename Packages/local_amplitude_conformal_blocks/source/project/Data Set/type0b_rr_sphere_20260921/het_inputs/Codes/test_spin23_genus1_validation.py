#!/usr/bin/env python3
"""Structural tests for the fixed-moduli genus-one validation bundle."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import unittest

from spin23_genus1_validation import (
    _assembler_options,
    _block_comparison,
    _combine_serialized_spin_evaluations,
    _lower_block_options,
    _serialized_block_comparison,
    _spin_shard_schedule_wall_seconds,
    coordinates_from_sample,
    load_validation_manifest,
    states_from_sample,
)


ROOT = Path(__file__).resolve().parent
MANIFEST = (
    ROOT.parent
    / "Data Set"
    / "spin23_scan_manifests"
    / "spin23_genus1_fixed_moduli_validation.json"
)


class GenusOneValidationManifestTests(unittest.TestCase):
    def test_all_samples_have_valid_kinematics_and_necklace_coordinates(self) -> None:
        manifest = load_validation_manifest(MANIFEST)
        self.assertEqual(len(manifest["samples"]), 6)
        for sample in manifest["samples"]:
            with self.subTest(sample=sample["id"]):
                states = states_from_sample(sample)
                tau, points = coordinates_from_sample(sample)
                self.assertEqual(len(states), len(points))
                self.assertGreater(tau.imag, 0.0)

    def test_validation_profile_has_three_independent_probes(self) -> None:
        profile = load_validation_manifest(MANIFEST)["profiles"]["validation"]
        self.assertLess(
            profile["lower_maximum_twice_levels"],
            profile["maximum_twice_levels"],
        )
        self.assertGreater(profile["refined_order"], profile["quadrature_order"])
        self.assertGreater(profile["extended_p_max"], profile["p_max"])

    def test_lower_block_probe_can_reuse_only_the_base_spectral_rule(self) -> None:
        profile = load_validation_manifest(MANIFEST)["profiles"]["validation"]
        lower_options = _lower_block_options(profile)
        self.assertIsNotNone(lower_options)
        assert lower_options is not None
        self.assertEqual(lower_options["quadrature_order"], 3)
        self.assertEqual(lower_options["p_max"], 1.5)
        self.assertIsNone(lower_options["refined_order"])
        self.assertIsNone(lower_options["extended_p_max"])

    def test_lower_block_probe_preserves_anisotropic_edge_cutoffs(self) -> None:
        profile = dict(
            load_validation_manifest(MANIFEST)["profiles"][
                "endpoint_adaptive_v6"
            ]
        )
        profile["maximum_twice_levels"] = [10, 6]
        profile["lower_maximum_twice_levels"] = [8, 6]
        lower_options = _lower_block_options(profile)
        self.assertIsNotNone(lower_options)
        assert lower_options is not None
        self.assertEqual(lower_options["maximum_twice_levels"], (8, 6))

    def test_recursive_profile_uses_auto_with_explicit_contour_controls(
        self,
    ) -> None:
        manifest = load_validation_manifest(MANIFEST)
        recursive = _assembler_options(
            manifest["profiles"]["recursive_endpoint_v9"]
        )
        self.assertEqual(recursive["block_backend"], "auto")
        self.assertEqual(recursive["recursion_samples"], 8)
        self.assertEqual(recursive["recursion_radius"], 0.035)

        legacy = _assembler_options(manifest["profiles"]["smoke"])
        self.assertEqual(legacy["block_backend"], "direct")
        self.assertEqual(legacy["recursion_samples"], 24)

    def test_nested_profile_uses_embedded_rule_and_disjoint_tail_probe(self) -> None:
        profile = load_validation_manifest(MANIFEST)["profiles"][
            "nested_validation_v2"
        ]
        self.assertEqual(profile["spectral_method"], "gauss_kronrod_3_7")
        self.assertEqual(profile["quadrature_order"], 7)
        self.assertIsNone(profile["refined_order"])
        self.assertGreater(profile["extended_p_max"], profile["p_max"])

        higher_profile = load_validation_manifest(MANIFEST)["profiles"][
            "nested_validation_v3"
        ]
        self.assertEqual(
            higher_profile["spectral_method"],
            "gauss_kronrod_7_15",
        )
        self.assertEqual(higher_profile["quadrature_order"], 15)
        self.assertIsNone(higher_profile["refined_order"])
        self.assertGreater(
            higher_profile["extended_p_max"],
            higher_profile["p_max"],
        )

        gaussian_profile = load_validation_manifest(MANIFEST)["profiles"][
            "gaussian_validation_v4"
        ]
        self.assertEqual(
            gaussian_profile["spectral_method"],
            "gauss_laguerre_7_15",
        )
        self.assertEqual(gaussian_profile["quadrature_order"], 15)
        self.assertIsNone(gaussian_profile["p_max"])
        self.assertIsNone(gaussian_profile["extended_p_max"])

        lower_options = _lower_block_options(gaussian_profile)
        self.assertIsNotNone(lower_options)
        assert lower_options is not None
        self.assertEqual(lower_options["spectral_method"], "gauss_laguerre")
        self.assertEqual(lower_options["quadrature_order"], 15)

        adaptive_profile = load_validation_manifest(MANIFEST)["profiles"][
            "adaptive_gaussian_v5"
        ]
        adaptive_lower = _lower_block_options(adaptive_profile)
        self.assertIsNotNone(adaptive_lower)
        assert adaptive_lower is not None
        self.assertEqual(
            adaptive_lower["spectral_method"],
            "gauss_laguerre_adaptive",
        )
        self.assertEqual(adaptive_lower["quadrature_order"], 7)
        self.assertEqual(adaptive_lower["refined_order"], 24)
        self.assertEqual(adaptive_lower["spectral_relative_tolerance"], 0.001)

        endpoint_profile = load_validation_manifest(MANIFEST)["profiles"][
            "endpoint_adaptive_v6"
        ]
        self.assertEqual(
            endpoint_profile["spectral_method"],
            "gauss_laguerre_adaptive",
        )
        self.assertEqual(endpoint_profile["refined_order"], 24)
        self.assertEqual(
            endpoint_profile["spectral_relative_tolerance"],
            1.0e-6,
        )

    def test_block_diagnostic_does_not_hide_fixed_spin_cancellation(self) -> None:
        current = SimpleNamespace(
            value=1.0 + 0.0j,
            estimated_absolute_error=0.01,
            absolute_spin_scale=4.0,
            fixed_spin={
                "NS": SimpleNamespace(value=2.0 + 0.0j),
                "NS_tilde": SimpleNamespace(value=-1.0 + 0.0j),
                "R": SimpleNamespace(value=1.0 + 0.0j),
            },
        )
        lower = SimpleNamespace(
            fixed_spin={
                "NS": SimpleNamespace(value=1.8 + 0.0j),
                "NS_tilde": SimpleNamespace(value=-0.8 + 0.0j),
                "R": SimpleNamespace(value=1.0 + 0.0j),
            },
            value=1.0 + 0.0j,
        )
        diagnostic = _block_comparison(current, lower)
        self.assertEqual(diagnostic["block_truncation_absolute_change"], 0.0)
        self.assertAlmostEqual(
            diagnostic["block_truncation_conservative_absolute_change"],
            0.2,
        )
        self.assertAlmostEqual(
            diagnostic["combined_diagnostic_absolute_error"],
            0.2,
        )
        self.assertAlmostEqual(
            diagnostic[
                "block_truncation_conservative_change_over_absolute_scale"
            ],
            0.05,
        )
        self.assertAlmostEqual(
            diagnostic["combined_diagnostic_error_over_absolute_scale"],
            0.05,
        )

    def test_spin_shard_reduction_preserves_gso_sum_and_diagnostic(self) -> None:
        def shard(label: str, weighted: complex, raw: complex) -> dict:
            return {
                "value": {"re": weighted.real, "im": weighted.imag},
                "estimated_absolute_error": 0.01,
                "estimated_relative_error": None,
                "fixed_spin": {
                    label: {
                        "value": {"re": raw.real, "im": raw.imag},
                    }
                },
                "coordinates": {"tau": {"re": 0.0, "im": 1.0}},
                "picture_raising_factor": 0.25,
                "includes_string_phase": False,
            }

        current_rows = (
            shard("NS", 1.0 + 0.5j, 2.0 + 1.0j),
            shard("NS_tilde", -0.2 + 0.1j, -0.4 + 0.2j),
            shard("R", 0.3 - 0.4j, 0.6 - 0.8j),
        )
        lower_rows = (
            shard("NS", 0.9 + 0.5j, 1.8 + 1.0j),
            shard("NS_tilde", -0.1 + 0.1j, -0.2 + 0.2j),
            shard("R", 0.3 - 0.4j, 0.6 - 0.8j),
        )
        current = _combine_serialized_spin_evaluations(current_rows)
        lower = _combine_serialized_spin_evaluations(lower_rows)
        self.assertEqual(current["value"], {"re": 1.1, "im": 0.19999999999999996})
        self.assertEqual(tuple(current["fixed_spin"]), ("NS", "NS_tilde", "R"))
        diagnostic = _serialized_block_comparison(current, lower)
        self.assertAlmostEqual(
            diagnostic["block_truncation_absolute_change"],
            abs((1.1 + 0.2j) - (1.1 + 0.2j)),
        )
        self.assertAlmostEqual(
            diagnostic["block_truncation_conservative_absolute_change"],
            0.2,
        )
        self.assertAlmostEqual(
            diagnostic[
                "block_truncation_conservative_change_over_absolute_scale"
            ],
            0.2 / current["absolute_spin_scale"],
        )

    def test_component_shards_merge_one_fixed_spin_without_overlap(self) -> None:
        def shard(component_index: int, weighted: float, raw: float) -> dict:
            indices = [] if component_index == 0 else [0, 1]
            return {
                "value": {"re": weighted, "im": 0.0},
                "estimated_absolute_error": 0.01,
                "estimated_relative_error": None,
                "fixed_spin": {
                    "R": {
                        "value": {"re": raw, "im": 0.0},
                        "estimated_absolute_error": 0.02,
                        "estimated_relative_error": None,
                        "common_free_field_factor": {"re": 3.0, "im": 0.0},
                        "components": [
                            {
                                "time_fermion_indices": indices,
                                "value": {"re": raw, "im": 0.0},
                            }
                        ],
                    }
                },
                "coordinates": {"tau": {"re": 0.0, "im": 1.0}},
                "picture_raising_factor": 0.25,
                "includes_string_phase": False,
            }

        ns_rows = tuple(
            {
                "value": {"re": value, "im": 0.0},
                "estimated_absolute_error": 0.0,
                "estimated_relative_error": None,
                "fixed_spin": {
                    label: {"value": {"re": 2.0 * value, "im": 0.0}}
                },
                "coordinates": {"tau": {"re": 0.0, "im": 1.0}},
                "picture_raising_factor": 0.25,
                "includes_string_phase": False,
            }
            for label, value in (("NS", 1.0), ("NS_tilde", -0.25))
        )
        combined = _combine_serialized_spin_evaluations(
            (*ns_rows, shard(0, 0.1, 0.2), shard(1, 0.3, 0.6))
        )
        self.assertEqual(combined["value"], {"re": 1.15, "im": 0.0})
        self.assertEqual(combined["fixed_spin"]["R"]["value"], {"re": 0.8, "im": 0.0})
        self.assertEqual(
            [
                row["time_fermion_indices"]
                for row in combined["fixed_spin"]["R"]["components"]
            ],
            [[], [0, 1]],
        )

    def test_balanced_schedule_adds_the_two_concurrent_waves(self) -> None:
        runtimes = {
            "NS": 140.0,
            "NS_tilde": 145.0,
            "R_component_0": 330.0,
            "R_component_1": 200.0,
        }
        self.assertEqual(
            _spin_shard_schedule_wall_seconds(
                runtimes,
                sequential_spin_waves=False,
            ),
            330.0,
        )
        self.assertEqual(
            _spin_shard_schedule_wall_seconds(
                runtimes,
                sequential_spin_waves=True,
            ),
            475.0,
        )


if __name__ == "__main__":
    unittest.main()
