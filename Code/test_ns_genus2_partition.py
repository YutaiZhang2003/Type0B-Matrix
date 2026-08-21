#!/usr/bin/env python3
"""Regression checks for the first genus-two NS partition experiment."""

from __future__ import annotations

import cmath
from itertools import product
import unittest

import mpmath

from compare_ns_torus_c_h_recursion import _global_torus_block
from ns_genus2_cannon import (
    SCHEMA,
    _cutoff_pairs,
    _digest,
    _designs,
    _node_data,
    _validate_shard,
    _validate_config_spin_characteristics,
    channel_task_chunks,
    decode_task,
    task_count,
)
from ns_genus2_partition import (
    C_ORDINARY_AT_HAT_C_9,
    GLASSES_CCY_DESCENDANT_EDGE_ORDER,
    GLASSES_GEOMETRY_EDGE_ORDER,
    GLASSES_ORIENTATION,
    GLASSES_TO_THETA_BRANCH_COMPOSED,
    HAT_C_TARGET,
    MAX_RECURSION_ORDER,
    NSGenus2CRecursion,
    THETA_CCY_DESCENDANT_EDGE_ORDER,
    THETA_GEOMETRY_EDGE_ORDER,
    THETA_INTEGER_BRANCH,
    _MPPartialFractionInC,
    _free_scalar_chiral_log,
    _free_superfield_chiral_log,
    _spin_characteristic_from_lifts,
    _theta_geometry_to_ccy_order,
    _theta_global_term,
    _theta_schottky_data,
    _transport_spin_characteristic,
    direct_global_block,
    free_superfield_partition,
    ns_weight,
    resummed_glasses_global_block,
    resummed_theta_global_block,
    run_internal_checks,
)
from ns_global_osp_block import osp_norm, osp_three_point
from ns_human_convention import (
    glasses_primary_parity_rephasing,
    theta_primary_parity_rephasing,
)
from ns_vacuum_schottky import ccy_theta_generators, theta_lift_signs
from plumbing_algorithms import generators_for_glasses, generators_for_theta


class GenusTwoNSPartitionTests(unittest.TestCase):
    @staticmethod
    def _projective_error(left, right) -> float:
        left_entries = (left.a, left.b, left.c, left.d)
        right_entries = (right.a, right.b, right.c, right.d)
        pivot = max(range(4), key=lambda index: abs(right_entries[index]))
        scale = left_entries[pivot] / right_entries[pivot]
        return max(
            abs(a - scale * b) for a, b in zip(left_entries, right_entries)
        ) / max(1.0, *(abs(value) for value in left_entries))

    def test_central_charge_convention_is_explicit(self) -> None:
        self.assertEqual(HAT_C_TARGET, 9.0)
        self.assertEqual(C_ORDINARY_AT_HAT_C_9, 13.5)
        self.assertEqual(C_ORDINARY_AT_HAT_C_9, 1.5 * HAT_C_TARGET)

    def test_generic_primary_global_resummations_match_direct_sums(self) -> None:
        weights = (0.71, 0.83, 0.94)
        q_values = (0.010, 0.013, 0.009)
        lifts = (1, -1, 1)
        for channel in ("theta", "glasses"):
            for sector in (0, 1):
                for primaries in product((0, 1), repeat=3):
                    direct = direct_global_block(
                        channel=channel,
                        weights=weights,
                        q_values=q_values,
                        sector=sector,
                        lifts=lifts,
                        tolerance=1.0e-14,
                        max_total_occupation=18,
                        primary_parities=primaries,
                    )
                    if channel == "theta":
                        resummed = resummed_theta_global_block(
                            weights=weights,
                            q_values=q_values,
                            sector=sector,
                            lifts=lifts,
                            tolerance=1.0e-14,
                            max_total_endpoint_occupation=12,
                            primary_parities=primaries,
                        )
                    else:
                        resummed = resummed_glasses_global_block(
                            weights=weights,
                            q_values=q_values,
                            sector=sector,
                            lifts=lifts,
                            primary_parities=primaries,
                        )
                    self.assertTrue(direct.converged)
                    self.assertLess(
                        abs(direct.value - resummed.value)
                        / max(1.0, abs(resummed.value)),
                        2.0e-11,
                        (channel, sector, primaries),
                    )

    def test_functional_recursion_obeys_generic_primary_rephasing(self) -> None:
        weights = (0.731, 0.913, 1.173)
        q_values = (0.0013, 0.0017, 0.0011)
        lifts = (1, -1, 1)
        reducers = {
            "theta": theta_primary_parity_rephasing,
            "glasses": glasses_primary_parity_rephasing,
        }
        for channel, reducer in reducers.items():
            recursion = NSGenus2CRecursion(
                channel=channel,
                q_values=q_values,
                global_tolerance=1.0e-13,
                global_max_total_occupation=16,
                vacuum_word_length=3,
                vacuum_max_mode=18,
            )
            for sector in (0, 1):
                for primaries in product((0, 1), repeat=3):
                    prefactor, effective_lifts = reducer(lifts, primaries)
                    expected = prefactor * recursion.block(
                        weights=weights,
                        sector=sector,
                        recursion_order=3,
                        lifts=effective_lifts,
                        central_charge=41.3,
                    )
                    observed = recursion.block(
                        weights=weights,
                        sector=sector,
                        recursion_order=3,
                        lifts=lifts,
                        central_charge=41.3,
                        primary_parities=primaries,
                    )
                    self.assertLess(
                        abs(observed - expected),
                        2.0e-13,
                        (channel, sector, primaries),
                    )

    def test_orientation_polynomial(self) -> None:
        self.assertEqual(
            GLASSES_GEOMETRY_EDGE_ORDER, GLASSES_CCY_DESCENDANT_EDGE_ORDER
        )
        self.assertEqual(GLASSES_ORIENTATION.edge_linear_bits, (0, 0, 0))
        for left in (0, 1):
            for right in (0, 1):
                for bridge in (0, 1):
                    self.assertEqual(
                        GLASSES_ORIENTATION.exponent((left, right, bridge)),
                        bridge * (left + right) % 2,
                    )

    def test_separating_and_spin_checks(self) -> None:
        checks = run_internal_checks()
        self.assertLess(checks["theta_descendant_order_relative_error"], 2.0e-14)
        self.assertGreater(checks["theta_old_order_relative_displacement"], 1.0e-2)
        self.assertLess(
            max(checks["theta_resummation_direct_relative_errors"]),
            2.0e-11,
        )
        self.assertLess(checks["separating_global_relative_error"], 2.0e-10)
        self.assertLess(
            max(checks["glasses_resummation_direct_relative_errors"]),
            2.0e-11,
        )
        self.assertLess(checks["handle_residue_torus_relative_error"], 2.0e-12)
        self.assertEqual(
            set(checks["handle_residue_torus_relative_errors"]),
            {
                "3,1,sector=0",
                "3,1,sector=1",
                "2,2,sector=0",
                "2,2,sector=1",
            },
        )
        self.assertLess(
            max(checks["handle_residue_torus_relative_errors"].values()),
            2.0e-12,
        )
        self.assertEqual(
            checks["spin_target_characteristic"],
            {"alpha": [0, 0], "beta": [0, 0]},
        )
        self.assertEqual(checks["theta_edge_lifts"], [1, 1, 1])
        self.assertEqual(checks["same_spin_theta_lifts"], [1, 1, 1])
        self.assertEqual(checks["same_spin_glasses_lifts"], [1, 1, 1])
        self.assertEqual(
            checks["theta_integer_branch"],
            [list(row) for row in THETA_INTEGER_BRANCH],
        )
        self.assertEqual(
            checks["symplectic_matrix"],
            [list(row) for row in GLASSES_TO_THETA_BRANCH_COMPOSED],
        )

    def test_functional_recursion_preserves_order_twelve_benchmarks(self) -> None:
        self.assertEqual(MAX_RECURSION_ORDER, 24)
        weights = (0.731, 0.913, 1.173)
        q_values = (0.07 + 0.002j, 0.11 - 0.003j, 0.09 + 0.001j)
        expected = {
            "theta": {
                # Literal human-note theta orientation and lifts.
                8: 1.097163792567563 + 0.0019394577120379352j,
                12: 1.0971613337433175 + 0.0019395813465742041j,
            },
            "glasses": {
                # Human-note fixed-parity rho_a convention.  These differ
                # from the former component-ordered Ward benchmarks only in
                # the odd handle/bridge transport signs.
                8: 1.6587896314716546 - 0.004469940511235218j,
                12: 1.6587911258930406 - 0.004469282970501153j,
            },
        }
        lifts = {"theta": (1, -1, -1), "glasses": (1, 1, 1)}
        for channel in ("theta", "glasses"):
            recursion = NSGenus2CRecursion(
                channel=channel,
                q_values=q_values,
                global_max_total_occupation=22,
                vacuum_word_length=3,
                vacuum_max_mode=20,
            )
            for order in (8, 12):
                with self.subTest(channel=channel, order=order):
                    observed = recursion.block(
                        weights=weights,
                        sector=0,
                        recursion_order=order,
                        lifts=lifts[channel],
                        central_charge=41.3,
                    )
                    self.assertLess(abs(observed - expected[channel][order]), 2.0e-13)
                    collision_aware = recursion.collision_aware_block(
                        weights=weights,
                        sector=0,
                        recursion_order=order,
                        lifts=lifts[channel],
                        central_charge=41.3,
                    )
                    self.assertLess(abs(collision_aware - observed), 3.0e-12)
                    if order == 8:
                        collision_aware_mp = recursion.collision_aware_block_mp(
                            weights=weights,
                            sector=0,
                            recursion_order=order,
                            lifts=lifts[channel],
                            central_charge=41.3,
                            working_precision=50,
                        )
                        self.assertLess(
                            abs(collision_aware_mp - observed), 3.0e-12
                        )
            with self.assertRaisesRegex(ValueError, "0..24"):
                recursion.block(
                    weights=weights,
                    sector=0,
                    recursion_order=25,
                    lifts=lifts[channel],
                    central_charge=41.3,
                )

    def test_fixed_difference_family_is_combined_by_confluent_moments(self) -> None:
        with mpmath.workdps(90):
            anchor = mpmath.mpf("1.5")
            evaluation_point = mpmath.mpf("13.5")
            pole_deltas = tuple(
                mpmath.mpf(value)
                for value in (
                    "-0.0012",
                    "-0.00053",
                    "-0.00030",
                    "-0.00019",
                )
            )
            diagnostics = {
                "moment_groups": 0,
                "direct_groups": 0,
                "max_moment_terms": 0,
                "max_moment_ratio": mpmath.mpf(0),
            }
            partial_fraction = _MPPartialFractionInC(
                moment_diagnostics=diagnostics
            )
            tolerance = mpmath.mpf("1e-60")
            for index, pole_delta in enumerate(pole_deltas):
                barycentric_weight = 1 / mpmath.fprod(
                    pole_delta - other
                    for other_index, other in enumerate(pole_deltas)
                    if other_index != index
                )
                partial_fraction.add_pole_coefficient(
                    anchor + pole_delta,
                    1,
                    barycentric_weight,
                    tolerance,
                    family_key=2,
                )
            observed = partial_fraction.value(evaluation_point, tolerance)
            expected = 1 / mpmath.fprod(
                evaluation_point - anchor - pole_delta
                for pole_delta in pole_deltas
            )
            self.assertLess(abs(observed - expected), mpmath.mpf("1e-55"))
            self.assertEqual(diagnostics["moment_groups"], 1)
            self.assertEqual(diagnostics["direct_groups"], 0)
            self.assertGreater(diagnostics["max_moment_terms"], 3)

    def test_order_twenty_four_threshold_family_uses_moments(self) -> None:
        q_values = (
            0.15388585893452059 + 0.00028976276925814107j,
            0.15105853512050485 + 0.005374602305150904j,
            0.15290700987239295 - 0.005617044175995109j,
        )
        momenta = (
            0.1999067241791299,
            0.19895686082147337,
            0.19960253618399287,
        )
        recursion = NSGenus2CRecursion(
            channel="glasses",
            q_values=q_values,
            global_method="auto",
            global_tolerance=2.0e-8,
            global_max_total_occupation=22,
            vacuum_word_length=8,
            vacuum_max_mode=50,
        )
        observed = recursion.collision_aware_block_mp(
            weights=tuple(ns_weight(momentum) for momentum in momenta),
            sector=0,
            recursion_order=24,
            lifts=(1, 1, 1),
            working_precision=50,
        )
        expected = 2.291673200615892 + 0.027492883704842413j
        self.assertLess(abs(observed - expected), 3.0e-14)
        self.assertGreaterEqual(recursion.confluent_moment_groups, 1)
        self.assertGreater(recursion.confluent_max_moment_terms, 3)
        self.assertLessEqual(recursion.confluent_max_moment_ratio, 0.2)

    def test_cannon_designs_include_recursion_order(self) -> None:
        config = {
            "points": [{"id": "audit"}],
            "recursion_orders": [10, 12],
            "quadrature_orders": [8],
        }
        designs = _designs(config)
        self.assertEqual(task_count(config), 2 * 2 * 8**3)
        self.assertEqual(
            [(row["recursion_order"], row["channel"]) for row in designs],
            [
                (10, "theta"),
                (10, "glasses"),
                (12, "theta"),
                (12, "glasses"),
            ],
        )

        axis_config = {
            "points": config["points"],
            "convergence_designs": [
                {"recursion_order": 20, "quadrature_order": 10},
                {"recursion_order": 22, "quadrature_order": 10},
                {"recursion_order": 24, "quadrature_order": 8},
                {"recursion_order": 24, "quadrature_order": 10},
                {"recursion_order": 24, "quadrature_order": 12},
            ],
        }
        self.assertEqual(
            _cutoff_pairs(axis_config),
            ((20, 10), (22, 10), (24, 8), (24, 10), (24, 12)),
        )
        self.assertEqual(
            task_count(axis_config),
            2 * (10**3 + 10**3 + 8**3 + 10**3 + 12**3),
        )

    def test_cannon_shard_identity_validation(self) -> None:
        config = {
            "points": [
                {
                    "id": "audit",
                    "q_values": {
                        "theta": [0.11, 0.12, 0.13],
                        "glasses": [0.14, 0.15, 0.16],
                    },
                }
            ],
            "recursion_order": 8,
            "quadrature_orders": [2],
        }
        design = _designs(config)[0]
        _, indices, momenta, measure = _node_data(config, design, 0)
        shard = {
            "schema": SCHEMA,
            "task_index": 0,
            "node_index": 0,
            "config_digest": _digest(config),
            "implementation_fingerprint": "review-fingerprint",
            **design,
            "indices": list(indices),
            "momenta": list(momenta),
            "measure": measure,
            "q_edge_order": list(THETA_GEOMETRY_EDGE_ORDER),
            "descendant_tensor_edge_order": list(
                THETA_CCY_DESCENDANT_EDGE_ORDER
            ),
        }
        _validate_shard(config, 0, shard, "review-fingerprint")

        wrong_node = dict(shard, node_index=1)
        with self.assertRaisesRegex(RuntimeError, "node_index mismatch"):
            _validate_shard(config, 0, wrong_node, "review-fingerprint")
        wrong_implementation = dict(
            shard, implementation_fingerprint="other-fingerprint"
        )
        with self.assertRaisesRegex(RuntimeError, "implementation mismatch"):
            _validate_shard(
                config, 0, wrong_implementation, "review-fingerprint"
            )
        wrong_indices = dict(shard, indices=[1, 0, 0])
        with self.assertRaisesRegex(RuntimeError, "indices mismatch"):
            _validate_shard(config, 0, wrong_indices, "review-fingerprint")

    def test_theta_geometry_is_reversed_only_at_ccy_tensor_boundary(self) -> None:
        self.assertEqual(
            _theta_geometry_to_ccy_order(THETA_GEOMETRY_EDGE_ORDER),
            THETA_CCY_DESCENDANT_EDGE_ORDER,
        )
        weights = (0.71, 1.23, 0.94)
        q_values = (0.073 + 0.004j, 0.121 - 0.006j, 0.097 + 0.003j)
        occupations = (0, 1, 0)
        fermions = (0, 0, 0)
        lifts = (1, -1, -1)
        observed = _theta_global_term(
            weights, q_values, occupations, fermions, lifts
        )
        ccy_occupations = _theta_geometry_to_ccy_order(occupations)
        ccy_fermions = _theta_geometry_to_ccy_order(fermions)
        ccy_weights = _theta_geometry_to_ccy_order(weights)
        ccy_rho = osp_three_point(
            n1=int(ccy_occupations[0]),
            n2=int(ccy_occupations[1]),
            n3=int(ccy_occupations[2]),
            epsilon1=int(ccy_fermions[0]),
            epsilon2=int(ccy_fermions[1]),
            epsilon3=int(ccy_fermions[2]),
            d1=ccy_weights[0],
            d2=ccy_weights[1],
            d3=ccy_weights[2],
        )
        expected = q_values[1] * ccy_rho**2 / osp_norm(
            weights[1], occupations[1], fermions[1]
        )
        self.assertLess(abs(observed - expected), 2.0e-14)

        old_rho = osp_three_point(
            n1=occupations[0],
            n2=occupations[1],
            n3=occupations[2],
            epsilon1=fermions[0],
            epsilon2=fermions[1],
            epsilon3=fermions[2],
            d1=weights[0],
            d2=weights[1],
            d3=weights[2],
        )
        old_value = q_values[1] * old_rho**2 / osp_norm(
            weights[1], occupations[1], fermions[1]
        )
        self.assertGreater(abs(observed - old_value), 1.0e-2)

    def test_odd_glasses_sector_starts_on_bridge(self) -> None:
        result = direct_global_block(
            channel="glasses",
            weights=(0.71, 0.83, 0.64),
            q_values=(0.12, 0.14, 0.0),
            sector=1,
            lifts=(1, 1, 1),
            tolerance=1.0e-12,
            max_total_occupation=10,
        )
        self.assertEqual(result.value, 0.0j)

    def test_resummed_glasses_block_matches_direct_sum(self) -> None:
        weights = (1.30, 0.57, 2.10)
        q_values = (0.13 + 0.011j, 0.09 - 0.007j, 0.14 + 0.005j)
        lifts = (-1, 1, -1)
        for sector in (0, 1):
            with self.subTest(sector=sector):
                direct = direct_global_block(
                    channel="glasses",
                    weights=weights,
                    q_values=q_values,
                    sector=sector,
                    lifts=lifts,
                    tolerance=2.0e-14,
                    max_total_occupation=34,
                )
                resummed = resummed_glasses_global_block(
                    weights=weights,
                    q_values=q_values,
                    sector=sector,
                    lifts=lifts,
                )
                self.assertTrue(direct.converged)
                self.assertLess(
                    abs(direct.value - resummed.value)
                    / max(1.0, abs(resummed.value)),
                    8.0e-14,
                )

    def test_resummed_glasses_separating_limit(self) -> None:
        weights = (0.73, 0.91, 0.62)
        q_values = (0.11 + 0.006j, 0.14 - 0.004j, 0.0)
        even = resummed_glasses_global_block(
            weights=weights,
            q_values=q_values,
            sector=0,
            lifts=(1, 1, 1),
        )
        expected = complex(
            _global_torus_block(q_values[0], 1, weights[0], weights[2])
            * _global_torus_block(q_values[1], 1, weights[1], weights[2])
        )
        self.assertLess(abs(even.value - expected), 2.0e-14)
        odd = resummed_glasses_global_block(
            weights=weights,
            q_values=q_values,
            sector=1,
            lifts=(1, 1, 1),
        )
        self.assertEqual(odd.value, 0.0j)

    def test_resummed_regular_seed_matches_direct_audit_path(self) -> None:
        settings = {
            "channel": "glasses",
            "q_values": (0.09 + 0.002j, 0.11 - 0.003j, 0.07 + 0.001j),
            "global_tolerance": 2.0e-13,
            "global_max_total_occupation": 30,
            "vacuum_word_length": 4,
            "vacuum_max_mode": 25,
        }
        resummed = NSGenus2CRecursion(**settings, global_method="auto")
        direct = NSGenus2CRecursion(**settings, global_method="direct")
        weights = (0.71 + 0.01j, 0.83 - 0.02j, 0.64 + 0.005j)
        lifts = (1, -1, -1)
        for sector in (0, 1):
            with self.subTest(sector=sector):
                exact_value = resummed._regular(weights, sector, lifts)
                direct_value = direct._regular(weights, sector, lifts)
                self.assertLess(
                    abs(exact_value - direct_value)
                    / max(1.0, abs(exact_value)),
                    2.0e-13,
                )

    def test_resummed_theta_block_matches_direct_sum(self) -> None:
        weights = (0.71, 0.83, 0.64)
        q_values = (0.09 + 0.002j, 0.11 - 0.003j, 0.07 + 0.001j)
        lifts = (1, -1, -1)
        for sector in (0, 1):
            with self.subTest(sector=sector):
                direct = direct_global_block(
                    channel="theta",
                    weights=weights,
                    q_values=q_values,
                    sector=sector,
                    lifts=lifts,
                    tolerance=2.0e-13,
                    max_total_occupation=38,
                )
                resummed = resummed_theta_global_block(
                    weights=weights,
                    q_values=q_values,
                    sector=sector,
                    lifts=lifts,
                    tolerance=2.0e-13,
                    max_total_endpoint_occupation=38,
                )
                self.assertTrue(direct.converged)
                self.assertTrue(resummed.converged)
                self.assertLess(
                    abs(direct.value - resummed.value)
                    / max(1.0, abs(resummed.value)),
                    3.0e-13,
                )

    def test_resummed_theta_regular_seed_matches_direct_audit_path(self) -> None:
        settings = {
            "channel": "theta",
            "q_values": (0.07 + 0.002j, 0.10 - 0.003j, 0.06 + 0.001j),
            "global_tolerance": 2.0e-13,
            "global_max_total_occupation": 34,
            "vacuum_word_length": 4,
            "vacuum_max_mode": 25,
        }
        resummed = NSGenus2CRecursion(**settings, global_method="auto")
        direct = NSGenus2CRecursion(**settings, global_method="direct")
        weights = (0.71 + 0.01j, 0.83 - 0.02j, 0.64 + 0.005j)
        lifts = (1, -1, -1)
        for sector in (0, 1):
            with self.subTest(sector=sector):
                exact_value = resummed._regular(weights, sector, lifts)
                direct_value = direct._regular(weights, sector, lifts)
                self.assertLess(
                    abs(exact_value - direct_value)
                    / max(1.0, abs(exact_value)),
                    4.0e-13,
                )

    def test_nonconverged_global_seed_is_a_hard_failure(self) -> None:
        recursion = NSGenus2CRecursion(
            channel="theta",
            q_values=(0.12, 0.18, 0.11),
            global_method="resummed",
            global_tolerance=1.0e-15,
            global_max_total_occupation=0,
            vacuum_word_length=2,
            vacuum_max_mode=8,
        )
        with self.assertRaisesRegex(RuntimeError, "pointwise convergence"):
            recursion._global((0.71, 0.83, 0.64), 0, (1, 1, 1))

    def test_finite_part_sampling_rejects_laurent_aliasing(self) -> None:
        recursion = NSGenus2CRecursion(
            channel="glasses",
            q_values=(0.12, 0.13, 0.14),
            global_method="resummed",
            vacuum_word_length=2,
            vacuum_max_mode=8,
        )
        with self.assertRaisesRegex(ValueError, "samples >= 2"):
            recursion.finite_part_block(
                momenta=(0.2, 0.3, 0.4),
                sector=0,
                recursion_order=12,
                lifts=(1, 1, 1),
                samples=23,
            )

    def test_theta_schottky_marking_matches_period_coordinates(self) -> None:
        q_values = (0.073 + 0.004j, 0.121 - 0.006j, 0.097 + 0.003j)
        edge_lifts = (1, -1, -1)
        generators, signs = _theta_schottky_data(q_values, edge_lifts)
        expected = generators_for_theta(*q_values)
        self.assertEqual(signs, (1, -1))
        for observed, target in zip(generators, expected):
            self.assertLess(
                self._projective_error(observed.gamma, target.gamma), 1.0e-14
            )

        # The same marked surface in the CCY frame swaps q_0 and q_infinity
        # and reverses the second Schottky generator.
        ccy = ccy_theta_generators(q_values[2], q_values[1], q_values[0])
        self.assertLess(
            self._projective_error(generators[0].gamma, ccy[0].gamma), 1.0e-14
        )
        self.assertLess(
            self._projective_error(generators[1].gamma, ccy[1].gamma.inv()),
            1.0e-14,
        )

    def test_theta_free_product_is_identical_in_both_markings(self) -> None:
        q_values = (0.073 + 0.004j, 0.121 - 0.006j, 0.097 + 0.003j)
        ccy_generators = ccy_theta_generators(
            q_values[2], q_values[1], q_values[0]
        )
        for edge_lifts in product((-1, 1), repeat=3):
            generators, signs = _theta_schottky_data(q_values, edge_lifts)
            period_log, _ = _free_superfield_chiral_log(
                generators, signs, max_word_length=5, max_mode=30
            )

            swapped_lifts = (edge_lifts[2], edge_lifts[1], edge_lifts[0])
            ccy_log, _ = _free_superfield_chiral_log(
                ccy_generators,
                theta_lift_signs(swapped_lifts),
                max_word_length=5,
                max_mode=30,
            )
            self.assertLess(abs(period_log - ccy_log), 2.0e-13)

        # Lock against the original erroneous unswapped call.  This is a
        # genuine surface mismatch, not a numerically invisible refactoring.
        edge_lifts = (1, -1, -1)
        generators, signs = _theta_schottky_data(q_values, edge_lifts)
        period_log, _ = _free_superfield_chiral_log(
            generators, signs, max_word_length=5, max_mode=30
        )
        wrong_log, _ = _free_superfield_chiral_log(
            ccy_theta_generators(*q_values),
            theta_lift_signs(edge_lifts),
            max_word_length=5,
            max_mode=30,
        )
        self.assertGreater(abs(cmath.exp(period_log - wrong_log) - 1.0), 1.0e-4)

    def test_transported_physical_spin_characteristics(self) -> None:
        q_values = (0.11, 0.12, 0.13)
        self.assertEqual(
            _spin_characteristic_from_lifts(
                "glasses", q_values, (1, 1, 1)
            ),
            ((0, 0), (0, 0)),
        )
        self.assertEqual(
            _spin_characteristic_from_lifts(
                "theta", q_values, (1, 1, 1)
            ),
            ((0, 0), (0, 0)),
        )
        self.assertEqual(
            _spin_characteristic_from_lifts(
                "theta", q_values, (1, 1, -1)
            ),
            ((0, 0), (1, 1)),
        )
        self.assertEqual(
            _spin_characteristic_from_lifts(
                "theta", q_values, (1, -1, -1)
            ),
            ((0, 0), (1, 0)),
        )
        self.assertEqual(
            _spin_characteristic_from_lifts(
                "theta", q_values, (-1, 1, 1)
            ),
            ((0, 0), (1, 0)),
        )

        branch_composed = (
            (0, 0, -1, -1),
            (0, 0, 0, -1),
            (1, 0, 0, 0),
            (-1, 1, 0, 0),
        )
        self.assertEqual(
            _transport_spin_characteristic(
                branch_composed, ((0, 0), (0, 0))
            ),
            ((0, 0), (0, 0)),
        )

    def test_cannon_config_fails_closed_on_spin_mismatch(self) -> None:
        config = {
            "points": [
                {
                    "id": "spin-probe",
                    "q_values": {
                        "theta": [0.11, 0.12, 0.13],
                        "glasses": [0.14, 0.15, 0.16],
                    },
                    "omega": {
                        "glasses": [[1j, 0j], [0j, 1j]],
                        "theta": [[2j, 1j], [1j, 1j]],
                    },
                }
            ],
            "physical_lifts": {
                "theta": [1, 1, 1],
                "glasses": [1, 1, 1],
            },
            "expected_spin_characteristics": {
                "theta": {"alpha": [0, 0], "beta": [0, 0]},
                "glasses": {"alpha": [0, 0], "beta": [0, 0]},
            },
            "provenance": {
                "symplectic_matrix_glasses_to_theta_after_branch": [
                    [0, 0, -1, -1],
                    [0, 0, 0, -1],
                    [1, 0, 0, 0],
                    [-1, 1, 0, 0],
                ],
                "spin_transport_source_channel": "glasses",
                "spin_transport_target_channel": "theta",
                "spin_transport_period_tolerance": 1.0e-12,
            },
        }
        ledger = _validate_config_spin_characteristics(config)
        self.assertEqual(
            ledger["spin-probe"]["theta"],
            {"alpha": [0, 0], "beta": [0, 0]},
        )
        config["physical_lifts"]["theta"] = [1, -1, 1]
        config["expected_spin_characteristics"]["theta"]["beta"] = [0, 1]
        with self.assertRaisesRegex(ValueError, "modular spin mismatch"):
            _validate_config_spin_characteristics(config)
        config["physical_lifts"]["theta"] = [1, 1, 1]
        config["expected_spin_characteristics"]["theta"]["beta"] = [0, 0]
        config["expected_spin_characteristics"]["theta"]["beta"] = [1, 1]
        with self.assertRaisesRegex(ValueError, "spin characteristic mismatch"):
            _validate_config_spin_characteristics(config)

        del config["expected_spin_characteristics"]
        with self.assertRaisesRegex(ValueError, "must specify"):
            _validate_config_spin_characteristics(config)

    def test_channel_task_chunks_cover_only_requested_channel(self) -> None:
        config = {
            "points": [{"id": "p0"}, {"id": "p1"}],
            "recursion_orders": [2],
            "quadrature_orders": [2],
        }
        chunks = channel_task_chunks(config, "theta", 3)
        flattened = [
            task_index
            for start, stop in chunks
            for task_index in range(start, stop + 1)
        ]
        expected = [
            task_index
            for task_index in range(task_count(config))
            if decode_task(config, task_index)[0]["channel"] == "theta"
        ]
        self.assertEqual(flattened, expected)
        self.assertEqual(len(chunks), 6)
        self.assertTrue(all(stop - start + 1 <= 3 for start, stop in chunks))

    def test_bosonized_free_superfield_at_overlap_point(self) -> None:
        q_values = (
            0.15388585893452059 + 0.00028976276925814107j,
            0.15105853512050485 + 0.005374602305150904j,
            0.15290700987239295 - 0.005617044175995109j,
        )
        omega = [
            [
                0.00023985886060702268 + 0.2957553455196915j,
                0.0010668836604170971 + 0.026580859449807027j,
            ],
            [
                0.0010668836604170971 + 0.026580859449807027j,
                0.005492055485793113 + 0.29855499576571304j,
            ],
        ]
        result = free_superfield_partition(
            channel="glasses",
            q_values=q_values,
            omega=omega,
            physical_lifts=(1, 1, 1),
            max_word_length=9,
            max_mode=70,
        )
        self.assertLess(
            abs(result.value - 36.7446750673705) / result.value,
            2.0e-13,
        )
        scalar_log, _ = _free_scalar_chiral_log(
            generators_for_glasses(*q_values),
            max_word_length=9,
            max_mode=70,
        )
        self.assertLess(
            abs(cmath.exp(scalar_log) - (1.4763960934445186 + 0.013013146374300157j)),
            3.0e-13,
        )


if __name__ == "__main__":
    unittest.main()
