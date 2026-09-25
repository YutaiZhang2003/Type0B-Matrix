#!/usr/bin/env python3
"""Regression tests for the NS sphere four/five-point recursion layers."""

from __future__ import annotations

import unittest
from itertools import product

import sympy as sp

from ns_algebra.ns_fusion import c_xi
from spin23_ns_sphere_blocks import (
    direct_sphere_block_series,
    global_sphere_block_beta_series,
    relative_series_difference,
    sphere_h_pole_data,
)
from spin23_ns_sphere_five_point import (
    direct_five_point_block_series,
    five_point_h_pole_data,
    global_five_point_beta_series,
    recursive_five_point_block_series,
    relative_five_point_series_difference,
)


class NSFivePointBlockTest(unittest.TestCase):
    c = 11.3
    internal = (0.73, 0.91)
    external = (0.17, 0.29, 0.41, 0.37, 0.23)

    def test_all_component_choices_match_direct_oracle_through_level_two(self) -> None:
        for alphas in product((0, 1), repeat=3):
            for betas in product((0, 1), repeat=2):
                with self.subTest(alphas=alphas, betas=betas):
                    direct = direct_five_point_block_series(
                        c=self.c,
                        internal_weights=self.internal,
                        external_weights=self.external,
                        maximum_twice_levels=(4, 4),
                        component_parities=betas,
                        external_alphas=alphas,
                    )
                    recursive = recursive_five_point_block_series(
                        c=self.c,
                        internal_weights=self.internal,
                        external_weights=self.external,
                        maximum_twice_levels=(4, 4),
                        component_parities=betas,
                        external_alphas=alphas,
                    )
                    self.assertLess(
                        relative_five_point_series_difference(direct, recursive),
                        2.0e-12,
                    )

    def test_complex_level_three_and_edge_order(self) -> None:
        parameters = dict(
            c=10.7 + 0.2j,
            internal_weights=(0.83 + 0.07j, 1.11 - 0.04j),
            external_weights=(0.19, 0.31, 0.47, 0.28, 0.22),
            maximum_twice_levels=(6, 6),
            component_parities=(0, 1),
            external_alphas=(1, 0, 1),
        )
        direct = direct_five_point_block_series(**parameters)
        left_first = recursive_five_point_block_series(
            **parameters, edge_order="left_first"
        )
        right_first = recursive_five_point_block_series(
            **parameters, edge_order="right_first"
        )
        self.assertLess(
            relative_five_point_series_difference(direct, left_first), 2.0e-12
        )
        self.assertLess(
            relative_five_point_series_difference(left_first, right_first), 2.0e-12
        )

    def test_primary_q2_zero_slice_reduces_to_four_point_block(self) -> None:
        five = direct_five_point_block_series(
            c=self.c,
            internal_weights=self.internal,
            external_weights=self.external,
            maximum_twice_levels=(6, 0),
            component_parities=(0, 0),
        )
        four = direct_sphere_block_series(
            c=self.c,
            h_internal=self.internal[0],
            external_weights=(
                self.external[0],
                self.external[1],
                self.external[2],
                self.internal[1],
            ),
            maximum_twice_level=6,
            component="even",
        )
        for (left_level, right_level), coefficient in five.coefficients.items():
            self.assertEqual(right_level, 0)
            self.assertAlmostEqual(coefficient, four.coefficients[left_level], places=12)

    def test_beta_resolved_global_seeds_are_large_c_limits(self) -> None:
        five_global = global_five_point_beta_series(
            internal_weights=self.internal,
            external_weights=self.external,
            maximum_twice_levels=(4, 4),
            beta_labels=(1, 0),
            external_alphas=(1, 1, 0),
        )
        five_large_c = direct_five_point_block_series(
            c=1.0e12,
            internal_weights=self.internal,
            external_weights=self.external,
            maximum_twice_levels=(4, 4),
            component_parities=(1, 0),
            external_alphas=(1, 1, 0),
        )
        self.assertLess(
            relative_five_point_series_difference(five_global, five_large_c),
            2.0e-10,
        )

        four_global = global_sphere_block_beta_series(
            h_internal=self.internal[0],
            external_weights=self.external[:4],
            maximum_twice_level=6,
            beta=1,
        )
        four_large_c = direct_sphere_block_series(
            c=1.0e12,
            h_internal=self.internal[0],
            external_weights=self.external[:4],
            maximum_twice_level=6,
            component="odd",
        )
        self.assertLess(relative_series_difference(four_global, four_large_c), 2.0e-10)

    def test_even_graph_reflection(self) -> None:
        forward = direct_five_point_block_series(
            c=self.c,
            internal_weights=self.internal,
            external_weights=self.external,
            maximum_twice_levels=(4, 4),
            component_parities=(0, 0),
        )
        reflected = direct_five_point_block_series(
            c=self.c,
            internal_weights=self.internal[::-1],
            external_weights=self.external[::-1],
            maximum_twice_levels=(4, 4),
            component_parities=(0, 0),
        )
        for (left_level, right_level), coefficient in forward.coefficients.items():
            self.assertAlmostEqual(
                coefficient,
                reflected.coefficients[(right_level, left_level)],
                places=12,
            )

    def test_fixed_c_h_pole_residues(self) -> None:
        b = 0.8
        c = complex(sp.N(c_xi(sp.Float(b)), 40))
        epsilon = 1.0e-7

        four_pole = sphere_h_pole_data(
            b=b,
            r=3,
            s=1,
            external_weights=self.external[:4],
            component="odd",
        )
        four_near = direct_sphere_block_series(
            c=c,
            h_internal=four_pole.h_pole + epsilon,
            external_weights=self.external[:4],
            maximum_twice_level=3,
            component="odd",
        )
        self.assertLess(
            abs(
                epsilon * four_near.coefficients[3]
                - four_pole.residue
            ),
            2.0e-7,
        )

        for edge, parities, alphas in (
            ("left", (1, 1), (0, 1, 0)),
            ("right", (1, 1), (0, 0, 1)),
        ):
            with self.subTest(edge=edge):
                pole = five_point_h_pole_data(
                    b=b,
                    edge=edge,
                    r=3,
                    s=1,
                    internal_weights=self.internal,
                    external_weights=self.external,
                    component_parities=parities,
                    external_alphas=alphas,
                )
                pole_index = 0 if edge == "left" else 1
                near_weights = list(self.internal)
                near_weights[pole_index] = pole.h_pole + epsilon
                target_key = [parities[0], parities[1]]
                target_key[pole_index] = pole.null_twice_level
                near = direct_five_point_block_series(
                    c=c,
                    internal_weights=near_weights,
                    external_weights=self.external,
                    maximum_twice_levels=target_key,
                    component_parities=parities,
                    external_alphas=alphas,
                )

                shifted_weights = list(self.internal)
                shifted_weights[pole_index] = (
                    pole.h_pole + pole.null_twice_level / 2
                )
                sub_cutoffs = [parities[0], parities[1]]
                sub_cutoffs[pole_index] = 0
                sub_key = [parities[0], parities[1]]
                sub_key[pole_index] = 0
                sub = direct_five_point_block_series(
                    c=c,
                    internal_weights=shifted_weights,
                    external_weights=self.external,
                    maximum_twice_levels=sub_cutoffs,
                    component_parities=pole.shifted_component_parities,
                    external_alphas=alphas,
                )
                expected = pole.residue * sub.coefficients[tuple(sub_key)]
                observed = epsilon * near.coefficients[tuple(target_key)]
                self.assertLess(abs(observed - expected), 2.0e-7)


if __name__ == "__main__":
    unittest.main()
