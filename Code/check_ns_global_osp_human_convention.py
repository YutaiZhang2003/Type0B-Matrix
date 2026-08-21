#!/usr/bin/env python3
"""Directly check the theta global osp(1|2) block in the Human Note convention.

The check implements the convention used in ``Human Notes/SCblock.tex``.
For clarity it algebraically factors the displayed lift monomial as
``eta_i**(p_i+a_i) = eta_i**p_i * eta_i**a_i``:

* ``p_i`` is an intrinsic primary parity;
* ``prod_i eta_i**p_i`` is tracked as the primary prefactor;
* the lift inside ``F^{osp(1|2)}_{a_1 a_2 a_3}`` is only
  ``prod_i eta_i**a_i``;
* the theta orientation for arbitrary primary parity is
  ``(-1)**sum_{i<j}((p_i+a_i)*(p_j+a_j))``.

Here ``a_i`` is the parity of the global descendant
``L_-1**k_i G_-1/2**a_i phi_i`` relative to ``phi_i``.  In particular, none
of the lift powers inside the global block includes ``p_i``; primary parity
does, however, enter the orientation polynomial and the absolute trilinear
parity.  The block/fusion label itself remains the relative ``a``.

Three independent calculations of every global trinion are compared:

1. the exact NS PBW/contour Ward engine;
2. the closed global osp(1|2) formula used by the symbolic recursion; and
3. direct differentiation of the superspace three-point invariant.

The sewn coefficient is then compared with both the exact symbolic seed and
the numerical production theta term.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from itertools import product
from math import comb
from typing import Sequence

import sympy as sp

from ns_genus2_partition import _theta_global_term
from ns_genus2_symbolic_low_order import (
    ExactNSDescendantThreeForm,
    ExactNSVermaModule,
    State,
    exact_global_theta_coefficient,
    exact_osp_norm,
    exact_osp_three_point,
)
from ns_osp_superspace import superspace_three_point


G_HALF: State = (("G", -1),)
L_ONE = ("L", -2)


def _clean(value: sp.Expr) -> sp.Expr:
    return sp.cancel(sp.together(value))


def _bits(values: Sequence[int], name: str) -> tuple[int, int, int]:
    if len(values) != 3 or any(value not in (0, 1) for value in values):
        raise ValueError(f"{name} must contain three bits")
    return int(values[0]), int(values[1]), int(values[2])


def _lifts(values: Sequence[int]) -> tuple[int, int, int]:
    if len(values) != 3 or any(value not in (-1, 1) for value in values):
        raise ValueError("lifts must contain three signs")
    return int(values[0]), int(values[1]), int(values[2])


def global_state(occupation: int, descendant_parity: int) -> State:
    r"""Return ``G_-1/2**a L_-1**k |h>`` in the repository PBW order."""

    if occupation < 0 or descendant_parity not in (0, 1):
        raise ValueError("invalid global osp(1|2) state")
    return (G_HALF if descendant_parity else ()) + (L_ONE,) * occupation


def theta_orientation_exponent(
    primary_parities: Sequence[int], descendant_parities: Sequence[int]
) -> int:
    r"""Return ``sum_{i<j}(p_i+a_i)(p_j+a_j)`` modulo two."""

    primaries = _bits(primary_parities, "primary_parities")
    descendants = _bits(descendant_parities, "descendant_parities")
    total = tuple(
        primary ^ descendant
        for primary, descendant in zip(primaries, descendants)
    )
    first, second, third = total
    return (first * second + first * third + second * third) % 2


def theta_cross_exponent(
    primary_parities: Sequence[int], descendant_parities: Sequence[int]
) -> int:
    r"""Return the polarization cross term ``B(p,a)`` modulo two."""

    p1, p2, p3 = _bits(primary_parities, "primary_parities")
    a1, a2, a3 = _bits(descendant_parities, "descendant_parities")
    return (
        p1 * (a2 + a3) + p2 * (a1 + a3) + p3 * (a1 + a2)
    ) % 2


def required_vacuum_lift_rephasing(
    primary_parities: Sequence[int], descendant_parities: Sequence[int]
) -> tuple[int, int, int]:
    r"""Return the large-c vacuum rephasing ``(-1)**(p_i+a_i)``.

    For an even vacuum parity vector ``v``, polarization gives

    ``Q(v+p+a)=Q(v)+Q(p+a)+sum_i v_i(p_i+a_i)``.

    Therefore the vacuum block in the large-c convolution must be evaluated
    at ``eta_i -> (-1)**(p_i+a_i) eta_i``.
    """

    primaries = _bits(primary_parities, "primary_parities")
    descendants = _bits(descendant_parities, "descendant_parities")
    return tuple(
        (-1) ** (primary ^ descendant)
        for primary, descendant in zip(primaries, descendants)
    )  # type: ignore[return-value]


def extracted_primary_lift_factor(
    primary_parities: Sequence[int], lifts: Sequence[int]
) -> int:
    r"""Return the factor ``prod_i eta_i**p_i`` outside the global block."""

    parities = _bits(primary_parities, "primary_parities")
    lift_signs = _lifts(lifts)
    return int(
        sp.prod(lift**parity for lift, parity in zip(lift_signs, parities))
    )


def relative_global_label(
    absolute_parity: int, primary_parities: Sequence[int]
) -> int:
    """Recover the relative label ``a`` from absolute trilinear parity."""

    if absolute_parity not in (0, 1):
        raise ValueError("absolute_parity must be zero or one")
    parities = _bits(primary_parities, "primary_parities")
    return absolute_parity ^ (sum(parities) % 2)


# Compatibility spelling used by the first version of this standalone audit.
relative_global_sector = relative_global_label


def pbw_global_three_point(
    *,
    form: ExactNSDescendantThreeForm,
    occupations: Sequence[int],
    descendant_parities: Sequence[int],
) -> sp.Expr:
    """Evaluate one global trinion with the full exact PBW/Ward engine."""

    if len(occupations) != 3 or any(value < 0 for value in occupations):
        raise ValueError("occupations must contain three non-negative integers")
    parities = _bits(descendant_parities, "descendant_parities")
    states = tuple(
        global_state(int(occupation), parity)
        for occupation, parity in zip(occupations, parities)
    )
    return _clean(form.value(*states))


def human_note_global_osp_coefficient(
    *,
    weights: Sequence[sp.Expr],
    occupations: Sequence[int],
    descendant_parities: Sequence[int],
    primary_parities: Sequence[int] = (0, 0, 0),
    rho: sp.Expr | None = None,
) -> sp.Expr:
    r"""Return the coefficient in ``F^{osp}_{a1 a2 a3}`` before ``q,eta``.

    This is the literal Human Note coefficient

    ``(-1)^Q rho^2 prod_i [k_i! (2 h_i)_(k_i+a_i)]^-1``.

    The primary parities enter only through ``Q(p+a)``.  The lift monomial is
    assembled separately, so ``eta_i**p_i`` can remain outside the block.
    """

    if len(weights) != 3:
        raise ValueError("weights must contain three entries")
    if len(occupations) != 3 or any(value < 0 for value in occupations):
        raise ValueError("occupations must contain three non-negative integers")
    parities = _bits(descendant_parities, "descendant_parities")
    primaries = _bits(primary_parities, "primary_parities")
    occupations = tuple(int(value) for value in occupations)
    if rho is None:
        rho = exact_osp_three_point(
            n1=occupations[0],
            n2=occupations[1],
            n3=occupations[2],
            epsilon1=parities[0],
            epsilon2=parities[1],
            epsilon3=parities[2],
            d1=weights[0],
            d2=weights[1],
            d3=weights[2],
        )
    denominator = sp.prod(
        exact_osp_norm(weight, occupation, parity)
        for weight, occupation, parity in zip(weights, occupations, parities)
    )
    orientation = (-1) ** theta_orientation_exponent(primaries, parities)
    return _clean(orientation * rho**2 / denominator)


def truncated_human_note_global_osp_block(
    *,
    weights: Sequence[sp.Expr],
    plumbing_square_roots: Sequence[sp.Expr],
    lifts: Sequence[int],
    descendant_parities: Sequence[int],
    maximum_total_occupation: int,
    primary_parities: Sequence[int] = (0, 0, 0),
    vertex: str = "closed",
    c: sp.Expr = sp.Rational(21, 2),
) -> sp.Expr:
    r"""Truncate one fixed ``F^{osp}_{a1 a2 a3}`` by total occupation.

    ``plumbing_square_roots[i]`` is ``sqrt(q_i)``.  Every lift inside this
    function occurs only as ``eta_i**a_i``.  To restore the primary factor,
    multiply the result by :func:`extracted_primary_lift_factor`.
    """

    if maximum_total_occupation < 0:
        raise ValueError("maximum_total_occupation must be non-negative")
    if len(weights) != 3 or len(plumbing_square_roots) != 3:
        raise ValueError("weights and plumbing_square_roots need three entries")
    parities = _bits(descendant_parities, "descendant_parities")
    primaries = _bits(primary_parities, "primary_parities")
    lift_signs = _lifts(lifts)
    if vertex not in {"closed", "pbw", "superspace"}:
        raise ValueError("vertex must be 'closed', 'pbw', or 'superspace'")
    form = (
        ExactNSDescendantThreeForm(c=c, weights=tuple(weights))
        if vertex == "pbw"
        else None
    )

    total = sp.S.Zero
    for occupations in product(range(maximum_total_occupation + 1), repeat=3):
        if sum(occupations) > maximum_total_occupation:
            continue
        if vertex == "pbw":
            assert form is not None
            rho = pbw_global_three_point(
                form=form,
                occupations=occupations,
                descendant_parities=parities,
            )
        elif vertex == "superspace":
            rho = superspace_three_point(
                n1=occupations[0],
                n2=occupations[1],
                n3=occupations[2],
                epsilon1=parities[0],
                epsilon2=parities[1],
                epsilon3=parities[2],
                d1=weights[0],
                d2=weights[1],
                d3=weights[2],
            )
        else:
            rho = None
        coefficient = human_note_global_osp_coefficient(
            weights=weights,
            occupations=occupations,
            descendant_parities=parities,
            primary_parities=primaries,
            rho=rho,
        )
        plumbing = sp.prod(
            square_root ** (2 * occupation + parity) * lift**parity
            for square_root, occupation, parity, lift in zip(
                plumbing_square_roots, occupations, parities, lift_signs
            )
        )
        total += plumbing * coefficient
    return _clean(total)


@dataclass(frozen=True)
class GlobalOSPHumanConventionSummary:
    maximum_total_occupation: int
    ground_table_identities: int
    pbw_vertex_identities: int
    superspace_vertex_identities: int
    global_norm_identities: int
    sewn_coefficient_identities: int
    exact_production_seed_identities: int
    numerical_production_term_identities: int
    truncated_block_identities: int
    arbitrary_primary_coefficient_identities: int
    arbitrary_primary_block_identities: int
    orientation_polarization_identities: int
    large_c_vacuum_rephasing_identities: int
    adapted_production_term_identities: int
    extracted_primary_lift_identities: int
    maximum_numerical_production_error: float


def run_checks(maximum_total_occupation: int = 2) -> GlobalOSPHumanConventionSummary:
    """Run exact direct checks of the Human Note global block."""

    if maximum_total_occupation < 0:
        raise ValueError("maximum_total_occupation must be non-negative")

    weights = (
        sp.Rational(7, 10),
        sp.Rational(11, 13),
        sp.Rational(17, 19),
    )
    c = sp.Rational(21, 2)
    form = ExactNSDescendantThreeForm(c=c, weights=weights)
    modules = tuple(ExactNSVermaModule(c=c, weight=weight) for weight in weights)
    occupations = tuple(
        values
        for values in product(range(maximum_total_occupation + 1), repeat=3)
        if sum(values) <= maximum_total_occupation
    )

    h1, h2, h3 = weights
    expected_ground_table = {
        (0, 0, 0): sp.S.One,
        (1, 0, 0): sp.S.One,
        (0, 1, 0): sp.S.One,
        (0, 0, 1): -sp.S.One,
        (1, 1, 0): h1 + h2 - h3,
        (1, 0, 1): h1 - h2 + h3,
        (0, 1, 1): h1 - h2 - h3,
        (1, 1, 1): -(h1 + h2 + h3 - sp.Rational(1, 2)),
    }
    for parities, expected in expected_ground_table.items():
        actual = exact_osp_three_point(
            n1=0,
            n2=0,
            n3=0,
            epsilon1=parities[0],
            epsilon2=parities[1],
            epsilon3=parities[2],
            d1=h1,
            d2=h2,
            d3=h3,
        )
        if _clean(actual - expected) != 0:
            raise AssertionError(f"Human Note ground table mismatch at {parities}")

    norm_count = 0
    for module, weight in zip(modules, weights):
        for occupation in range(maximum_total_occupation + 1):
            for parity in (0, 1):
                state = global_state(occupation, parity)
                pbw_norm = module.inner_product(state, state)
                osp_norm = exact_osp_norm(weight, occupation, parity)
                if _clean(pbw_norm - osp_norm) != 0:
                    raise AssertionError(
                        "PBW/global norm mismatch: "
                        f"h={weight}, k={occupation}, a={parity}"
                    )
                norm_count += 1

    pbw_count = 0
    superspace_count = 0
    coefficient_count = 0
    production_seed_count = 0
    production_term_count = 0
    arbitrary_coefficient_count = 0
    polarization_count = 0
    vacuum_rephasing_count = 0
    adapted_production_count = 0
    production_errors: list[float] = []
    geometric_q = (0.013, 0.017, 0.019)  # order: zero, one, infinity
    geometric_lifts = (-1, 1, -1)

    for occupation_values in occupations:
        for parities in product((0, 1), repeat=3):
            arguments = dict(
                n1=occupation_values[0],
                n2=occupation_values[1],
                n3=occupation_values[2],
                epsilon1=parities[0],
                epsilon2=parities[1],
                epsilon3=parities[2],
                d1=weights[0],
                d2=weights[1],
                d3=weights[2],
            )
            closed_rho = exact_osp_three_point(**arguments)
            pbw_rho = pbw_global_three_point(
                form=form,
                occupations=occupation_values,
                descendant_parities=parities,
            )
            superspace_rho = superspace_three_point(**arguments)
            if _clean(pbw_rho - closed_rho) != 0:
                raise AssertionError(
                    "PBW/closed global vertex mismatch: "
                    f"k={occupation_values}, a={parities}"
                )
            if _clean(superspace_rho - closed_rho) != 0:
                raise AssertionError(
                    "superspace/closed global vertex mismatch: "
                    f"k={occupation_values}, a={parities}"
                )
            pbw_count += 1
            superspace_count += 1

            closed_coefficient = human_note_global_osp_coefficient(
                weights=weights,
                occupations=occupation_values,
                descendant_parities=parities,
                primary_parities=(0, 0, 0),
                rho=closed_rho,
            )
            pbw_coefficient = human_note_global_osp_coefficient(
                weights=weights,
                occupations=occupation_values,
                descendant_parities=parities,
                primary_parities=(0, 0, 0),
                rho=pbw_rho,
            )
            if _clean(pbw_coefficient - closed_coefficient) != 0:
                raise AssertionError(
                    f"direct sewn coefficient mismatch at {occupation_values}, {parities}"
                )
            coefficient_count += 1

            twice_levels = tuple(
                2 * occupation + parity
                for occupation, parity in zip(occupation_values, parities)
            )
            relative_label = sum(parities) % 2
            production_seed = exact_global_theta_coefficient(
                weights=weights,
                levels=twice_levels,
                sectors=(relative_label, relative_label),
            )
            if _clean(production_seed - closed_coefficient) != 0:
                raise AssertionError(
                    f"symbolic production seed mismatch at {twice_levels}"
                )
            production_seed_count += 1

            # The production routine stores edges in (zero, one, infinity),
            # while the Human Note trinion is (infinity, one, zero).
            geometric_weights = tuple(reversed(weights))
            geometric_occupations = tuple(reversed(occupation_values))
            geometric_parities = tuple(reversed(parities))
            production_term = _theta_global_term(
                geometric_weights,
                geometric_q,
                geometric_occupations,
                geometric_parities,
                geometric_lifts,
            )
            expected_plumbing = sp.prod(
                q ** (occupation + sp.Rational(parity, 2)) * lift**parity
                for q, occupation, parity, lift in zip(
                    geometric_q,
                    geometric_occupations,
                    geometric_parities,
                    geometric_lifts,
                )
            )
            expected_term = complex(sp.N(closed_coefficient * expected_plumbing, 17))
            error = abs(complex(production_term) - expected_term)
            production_errors.append(float(error))
            if error > 1.0e-13:
                raise AssertionError(
                    "numerical production term mismatch: "
                    f"k={occupation_values}, a={parities}, error={error}"
                )
            production_term_count += 1

            q_of_primary = {
                primary_parities: theta_orientation_exponent(
                    primary_parities, (0, 0, 0)
                )
                for primary_parities in product((0, 1), repeat=3)
            }
            q_of_descendant = theta_orientation_exponent(
                (0, 0, 0), parities
            )
            ccy_lifts = tuple(reversed(geometric_lifts))
            for primary_parities in product((0, 1), repeat=3):
                arbitrary_coefficient = human_note_global_osp_coefficient(
                    weights=weights,
                    occupations=occupation_values,
                    descendant_parities=parities,
                    primary_parities=primary_parities,
                    rho=closed_rho,
                )
                arbitrary_pbw = human_note_global_osp_coefficient(
                    weights=weights,
                    occupations=occupation_values,
                    descendant_parities=parities,
                    primary_parities=primary_parities,
                    rho=pbw_rho,
                )
                if _clean(arbitrary_coefficient - arbitrary_pbw) != 0:
                    raise AssertionError(
                        "arbitrary-primary PBW coefficient mismatch: "
                        f"p={primary_parities}, a={parities}"
                    )
                cross = theta_cross_exponent(primary_parities, parities)
                total_q = theta_orientation_exponent(
                    primary_parities, parities
                )
                if total_q != (
                    q_of_primary[primary_parities]
                    + q_of_descendant
                    + cross
                ) % 2:
                    raise AssertionError(
                        f"orientation polarization failed for p={primary_parities}, a={parities}"
                    )
                polarization_count += 1
                arbitrary_coefficient_count += 1

                # Current production evaluates the p_i=0 term.  The exact
                # wrapper below is what is needed for arbitrary p_i when the
                # primary lift monomial is restored outside the block.
                primary_lift = extracted_primary_lift_factor(
                    primary_parities, ccy_lifts
                )
                orientation_ratio = (-1) ** (total_q + q_of_descendant)
                adapted_production = (
                    primary_lift * orientation_ratio * complex(production_term)
                )
                arbitrary_expected = complex(
                    sp.N(
                        arbitrary_coefficient
                        * expected_plumbing
                        * primary_lift,
                        17,
                    )
                )
                adapted_error = abs(adapted_production - arbitrary_expected)
                production_errors.append(float(adapted_error))
                if adapted_error > 1.0e-13:
                    raise AssertionError(
                        "adapted arbitrary-primary production mismatch: "
                        f"p={primary_parities}, a={parities}, error={adapted_error}"
                    )
                adapted_production_count += 1

    # Check the cross term between the even large-c vacuum state and the
    # global state.  This is distinct from the internal Q(p+a) sign.
    even_vacuum_parities = tuple(
        values
        for values in product((0, 1), repeat=3)
        if sum(values) % 2 == 0
    )
    for primary_parities in product((0, 1), repeat=3):
        for parities in product((0, 1), repeat=3):
            rephasing = required_vacuum_lift_rephasing(
                primary_parities, parities
            )
            total_global = tuple(
                primary ^ descendant
                for primary, descendant in zip(primary_parities, parities)
            )
            global_q = theta_orientation_exponent(primary_parities, parities)
            for vacuum_parities in even_vacuum_parities:
                full_parities = tuple(
                    vacuum ^ global_parity
                    for vacuum, global_parity in zip(
                        vacuum_parities, total_global
                    )
                )
                full_q = theta_orientation_exponent(
                    (0, 0, 0), full_parities
                )
                vacuum_q = theta_orientation_exponent(
                    (0, 0, 0), vacuum_parities
                )
                rephasing_exponent = sum(
                    vacuum * int(sign == -1)
                    for vacuum, sign in zip(vacuum_parities, rephasing)
                ) % 2
                if full_q != (vacuum_q + global_q + rephasing_exponent) % 2:
                    raise AssertionError(
                        "large-c vacuum/global polarization failed: "
                        f"p={primary_parities}, a={parities}, v={vacuum_parities}"
                    )
                vacuum_rephasing_count += 1

    square_roots = (
        sp.Rational(1, 5),
        sp.Rational(1, 7),
        sp.Rational(1, 11),
    )
    block_count = 0
    arbitrary_block_count = 0
    extraction_count = 0
    for parities in product((0, 1), repeat=3):
        for lifts in product((1, -1), repeat=3):
            closed_block = truncated_human_note_global_osp_block(
                weights=weights,
                plumbing_square_roots=square_roots,
                lifts=lifts,
                descendant_parities=parities,
                primary_parities=(0, 0, 0),
                maximum_total_occupation=maximum_total_occupation,
                vertex="closed",
                c=c,
            )
            pbw_block = truncated_human_note_global_osp_block(
                weights=weights,
                plumbing_square_roots=square_roots,
                lifts=lifts,
                descendant_parities=parities,
                primary_parities=(0, 0, 0),
                maximum_total_occupation=maximum_total_occupation,
                vertex="pbw",
                c=c,
            )
            superspace_block = truncated_human_note_global_osp_block(
                weights=weights,
                plumbing_square_roots=square_roots,
                lifts=lifts,
                descendant_parities=parities,
                primary_parities=(0, 0, 0),
                maximum_total_occupation=maximum_total_occupation,
                vertex="superspace",
                c=c,
            )
            if _clean(closed_block - pbw_block) != 0:
                raise AssertionError(f"truncated PBW block mismatch at a={parities}")
            if _clean(closed_block - superspace_block) != 0:
                raise AssertionError(
                    f"truncated superspace block mismatch at a={parities}"
                )
            block_count += 1

            # In the lift monomial, primary parity changes only the extracted
            # eta**p factor.  It separately changes the block through Q(p+a).
            descendant_lift = sp.prod(
                lift**parity for lift, parity in zip(lifts, parities)
            )
            for primary_parities in product((0, 1), repeat=3):
                primary_factor = extracted_primary_lift_factor(
                    primary_parities, lifts
                )
                arbitrary_block = truncated_human_note_global_osp_block(
                    weights=weights,
                    plumbing_square_roots=square_roots,
                    lifts=lifts,
                    descendant_parities=parities,
                    primary_parities=primary_parities,
                    maximum_total_occupation=maximum_total_occupation,
                    vertex="closed",
                    c=c,
                )
                orientation_ratio = (-1) ** (
                    theta_orientation_exponent(primary_parities, parities)
                    + theta_orientation_exponent((0, 0, 0), parities)
                )
                if _clean(arbitrary_block - orientation_ratio * closed_block) != 0:
                    raise AssertionError(
                        "arbitrary-primary truncated block mismatch: "
                        f"p={primary_parities}, a={parities}"
                    )
                arbitrary_block_count += 1
                unfactored_lift = sp.prod(
                    lift ** (primary + parity)
                    for lift, primary, parity in zip(
                        lifts, primary_parities, parities
                    )
                )
                if unfactored_lift != primary_factor * descendant_lift:
                    raise AssertionError(
                        "primary lift extraction mismatch: "
                        f"p={primary_parities}, a={parities}, eta={lifts}"
                    )
                # The completed large-c contribution is obtained only after
                # the p-dependent factor is put back outside this block.
                completed = primary_factor * arbitrary_block
                if _clean(
                    completed
                    - primary_factor * orientation_ratio * pbw_block
                ) != 0:
                    raise AssertionError("completed PBW contribution mismatch")
                absolute_parity = (
                    sum(primary_parities) + sum(parities)
                ) % 2
                if relative_global_label(
                    absolute_parity, primary_parities
                ) != sum(parities) % 2:
                    raise AssertionError("absolute/relative parity conversion failed")
                extraction_count += 1

    expected_vertex_count = 8 * comb(maximum_total_occupation + 3, 3)
    if pbw_count != expected_vertex_count:
        raise AssertionError("internal occupation-shell count changed")
    return GlobalOSPHumanConventionSummary(
        maximum_total_occupation=maximum_total_occupation,
        ground_table_identities=8,
        pbw_vertex_identities=pbw_count,
        superspace_vertex_identities=superspace_count,
        global_norm_identities=norm_count,
        sewn_coefficient_identities=coefficient_count,
        exact_production_seed_identities=production_seed_count,
        numerical_production_term_identities=production_term_count,
        truncated_block_identities=block_count,
        arbitrary_primary_coefficient_identities=arbitrary_coefficient_count,
        arbitrary_primary_block_identities=arbitrary_block_count,
        orientation_polarization_identities=polarization_count,
        large_c_vacuum_rephasing_identities=vacuum_rephasing_count,
        adapted_production_term_identities=adapted_production_count,
        extracted_primary_lift_identities=extraction_count,
        maximum_numerical_production_error=max(production_errors, default=0.0),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-occupation", type=int, default=2)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    summary = run_checks(args.max_occupation)
    if args.json:
        print(json.dumps(asdict(summary), indent=2))
        return
    print("Human Note global osp(1|2) convention check: PASS")
    print("  lift convention: eta_i^(p_i+a_i) = eta_i^p_i eta_i^a_i")
    print("  descendant lift factor: prod eta_i^a_i")
    print("  orientation convention: (-1)^Q(p+a), including all cross terms")
    print(
        "  exact PBW/closed/superspace vertices: "
        f"{summary.pbw_vertex_identities}"
    )
    print(f"  exact global norms: {summary.global_norm_identities}")
    print(f"  exact sewn coefficients: {summary.sewn_coefficient_identities}")
    print(
        "  production theta terms: "
        f"{summary.numerical_production_term_identities}"
    )
    print(
        "  extracted primary-lift identities: "
        f"{summary.extracted_primary_lift_identities}"
    )
    print(
        "  arbitrary-primary coefficients / blocks: "
        f"{summary.arbitrary_primary_coefficient_identities} / "
        f"{summary.arbitrary_primary_block_identities}"
    )
    print(
        "  large-c vacuum/global cross-sign identities: "
        f"{summary.large_c_vacuum_rephasing_identities}"
    )
    print(
        "  maximum numerical production error: "
        f"{summary.maximum_numerical_production_error:.3e}"
    )


if __name__ == "__main__":
    main()
