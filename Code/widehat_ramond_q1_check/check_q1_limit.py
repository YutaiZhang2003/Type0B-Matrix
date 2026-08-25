#!/usr/bin/env python3
"""Check the q1 -> 0 limit of the extended Ramond block in two ways.

Method 1 evaluates equation (rextendedblock) of SCblock.tex at q1 = 0.
Only n1 = 0 survives.  The required Ramond branching coefficients are
obtained from the current main-notes Ward system implemented in
python/ramond_branching_recursion/compute_target.py, and the two ordinary
Virasoro torus two-point blocks are summed directly over descendants.

Method 2 computes the Ramond SCA torus two-point block by extending the
internal-weight pole reconstruction of Hadasz--Jaskolski--Suchanek,
arXiv:1207.5740, from one to two insertions in the necklace channel.  At the
present low-level stage, independent large-momentum descendant evaluations
fix the regular polynomial; the target momenta are not sampled.  It then
multiplies by the q1 = 0 free-Majorana block.

The comparison made here is for f = 0, eta = eta' = + and positive plumbing
spin signs.  In this sector the Ramond convolution becomes the ordinary
product after q1 is set to zero.
"""

from __future__ import annotations

import argparse
import cmath
import json
import math
import sys
import time
from fractions import Fraction
from functools import lru_cache
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parent.parent
sys.path.insert(0, str(WORKSPACE / "python" / "ramond_branching_recursion"))

import compute_target as branching_core  # noqa: E402
from compute_target import (  # noqa: E402
    BranchWeights,
    FreeFieldModule,
    VirasoroThreePoint,
    finite_ward_solution,
    norm_product,
    partitions,
    solve_ns_l1,
    solve_ramond_lminus,
    strict_partitions,
)
from direct_state_check import (  # noqa: E402
    AuxiliaryThreePoint,
    PBWModule,
    PhysicalThreePoint,
)


TOLERANCE = 1.0e-7


def add(series, key, value):
    value = series.get(key, 0.0j) + complex(value)
    if abs(value) < 1.0e-14:
        series.pop(key, None)
    else:
        series[key] = value


def multiply(first, second, maximum2, maximum3):
    answer = {}
    for (level2a, level3a), value_a in first.items():
        for (level2b, level3b), value_b in second.items():
            level2 = level2a + level2b
            level3 = level3a + level3b
            if level2 <= maximum2 and level3 <= maximum3:
                add(answer, (level2, level3), value_a * value_b)
    return answer


def shift(series, shift2, shift3, maximum2, maximum3, factor=1.0):
    answer = {}
    for (level2, level3), value in series.items():
        target = (level2 + shift2, level3 + shift3)
        if target[0] <= maximum2 and target[1] <= maximum3:
            add(answer, target, factor * value)
    return answer


def complex_record(value):
    value = complex(value)
    return {"real": float(value.real), "imag": float(value.imag)}


def parse_fraction(text):
    return float(Fraction(text))


def gram_entry(left, right, weight, central_charge):
    """BPZ Gram entry for L_{-left}|h> and L_{-right}|h>."""
    evaluator = VirasoroThreePoint((weight, 0.0, 0.0), central_charge)
    expression = {tuple(right): 1.0 + 0.0j}
    # If left=(a1,...,ak), then (L_-a1 ... L_-ak)^dagger acts on the
    # ket first with L_a1, then with L_a2, and so on.
    for mode in left:
        next_expression = {}
        for word, outer in expression.items():
            for reduced, inner in evaluator.act(0, mode, word).items():
                next_expression[reduced] = next_expression.get(reduced, 0.0j) + outer * inner
        expression = next_expression
    return expression.get((), 0.0j)


@lru_cache(None)
def inverse_gram(level, weight, central_charge):
    basis = partitions(level)
    matrix = np.asarray(
        [
            [gram_entry(left, right, weight, central_charge) for right in basis]
            for left in basis
        ],
        dtype=np.complex128,
    )
    return basis, np.linalg.inv(matrix)


@lru_cache(None)
def ordinary_torus_two_point_series(
    external_weight,
    second_weight,
    third_weight,
    central_charge,
    maximum2,
    maximum3,
):
    """Direct necklace-channel Virasoro torus two-point block."""
    answer = {}
    evaluator = VirasoroThreePoint(
        (external_weight, second_weight, third_weight), central_charge
    )
    for level2 in range(maximum2 + 1):
        basis2, inverse2 = inverse_gram(level2, second_weight, central_charge)
        for level3 in range(maximum3 + 1):
            basis3, inverse3 = inverse_gram(level3, third_weight, central_charge)
            rho = np.asarray(
                [
                    [evaluator.value((), word2, word3) for word3 in basis3]
                    for word2 in basis2
                ],
                dtype=np.complex128,
            )
            coefficient = np.einsum(
                "ab,cd,ac,bd->", inverse2, inverse3, rho, rho, optimize=True
            )
            answer[(level2, level3)] = complex(coefficient)
    return answer


class MainConventionAuxiliaryThreePoint(AuxiliaryThreePoint):
    """Free-fermion form with the two ground values fixed in SCblock.tex."""

    def base_value(self, states):
        ground2 = states[1][1]
        ground3 = states[2][1]
        if ground2 != ground3:
            return 0.0j
        return 1.0 + 0.0j if ground2 == 0 else 1.0j


def corrected_ground_value(
    second_module,
    third_module,
    second_label,
    third_label,
    alpha2,
    alpha3,
    eta,
):
    """The exact extended ground form in the conventions of SCblock.tex."""
    second = second_module.r_branch(second_label, alpha2)
    third = third_module.r_branch(third_label, alpha3)
    form_parity = (alpha2 + alpha3) % 2
    ramond_odd_phase = cmath.exp(3j * math.pi / 4)
    answer = 0.0j
    for state2, coefficient2 in second.items():
        for state3, coefficient3 in third.items():
            if state2[0] or state2[2] or state2[3] or state3[0] or state3[2] or state3[3]:
                raise AssertionError("A ground anchor contains an oscillator excitation.")
            auxiliary2, physical2 = state2[1], state2[4]
            auxiliary3, physical3 = state3[1], state3[4]
            if auxiliary2 != auxiliary3:
                continue
            auxiliary_form = 1.0 if auxiliary2 == 0 else 1.0j
            if form_parity == 0:
                if (physical2, physical3) == (0, 0):
                    physical_form = 1.0
                elif (physical2, physical3) == (1, 1):
                    physical_form = ramond_odd_phase**2 * eta
                else:
                    physical_form = 0.0
            else:
                if (physical2, physical3) == (0, 1):
                    physical_form = ramond_odd_phase
                elif (physical2, physical3) == (1, 0):
                    physical_form = ramond_odd_phase * 1j * eta
                else:
                    physical_form = 0.0
            tensor_sign = (-1) ** (physical2 * auxiliary3)
            answer += (
                coefficient2
                * coefficient3
                * tensor_sign
                * auxiliary_form
                * physical_form
            )
    return answer


def build_branching_values(b, momenta, labels):
    """Compute low branches from the first L1 Ward identity and exact anchors."""
    ns_module = FreeFieldModule("NS", b, momenta[0])
    second_module = FreeFieldModule("R", b, momenta[1])
    third_module = FreeFieldModule("R", b, momenta[2])
    weights = BranchWeights(b, momenta)
    ns_actions = {Fraction(0): []}
    for label in (1, 2):
        ns_actions[Fraction(label)] = solve_ns_l1(ns_module, label)[0]
    second_domain = (
        Fraction(-3, 4), Fraction(-1, 4), Fraction(1, 4),
        Fraction(3, 4), Fraction(7, 4),
    )
    third_domain = (
        Fraction(-3, 4), Fraction(-1, 4), Fraction(1, 4),
        Fraction(3, 4), Fraction(5, 4),
    )
    values = {}
    diagnostics = []
    original_ground_value = branching_core.direct_ground_value
    branching_core.direct_ground_value = corrected_ground_value
    for alpha in (0, 1):
        second_actions = {
            label: solve_ramond_lminus(second_module, label, alpha)[0]
            for label in second_domain
        }
        third_actions = {
            label: solve_ramond_lminus(third_module, label, alpha)[0]
            for label in third_domain
        }
        ward = finite_ward_solution(
            weights,
            ns_actions,
            second_actions,
            third_actions,
            second_module,
            third_module,
            alpha,
            alpha,
            1,
        )
        index = {state: position for position, state in enumerate(ward["unknowns"])}
        for second_label in labels:
            for third_label in labels:
                state = (Fraction(0), second_label, third_label)
                values[(second_label, third_label, alpha)] = ward["values"][
                    index[state]
                ] / norm_product(
                    state, alpha, alpha, b, momenta
                )
        diagnostics.append(
            {
                "alpha": alpha,
                "method": "first L1 Ward identity with exact tensor-ground anchors",
                "rows": ward["rows"],
                "columns": ward["columns"],
                "rank": ward["rank"],
                "relative_residual": ward["relative_residual"],
            }
        )
    branching_core.direct_ground_value = original_ground_value
    return weights, values, diagnostics


def branch_level(label):
    return int(2 * label * label - Fraction(1, 8))


def double_virasoro_series(b, momenta, maximum2, maximum3):
    labels = tuple(
        label
        for label in (
            Fraction(-3, 4),
            Fraction(-1, 4),
            Fraction(1, 4),
            Fraction(3, 4),
        )
        if branch_level(label) <= max(maximum2, maximum3)
    )
    weights, branching, diagnostics = build_branching_values(b, momenta, labels)
    answer = {}
    branch_terms = []

    for second_label in labels:
        base2 = branch_level(second_label)
        if base2 > maximum2:
            continue
        for third_label in labels:
            base3 = branch_level(third_label)
            if base3 > maximum3:
                continue
            branch_labels = (Fraction(0), second_label, third_label)
            residual2 = maximum2 - base2
            residual3 = maximum3 - base3
            copy_blocks = []
            for copy in (0, 1):
                external, second, third = weights.triple(branch_labels, copy)
                copy_blocks.append(
                    ordinary_torus_two_point_series(
                        external,
                        second,
                        third,
                        weights.central_charges[copy],
                        residual2,
                        residual3,
                    )
                )
            virasoro_product = multiply(
                copy_blocks[0], copy_blocks[1], residual2, residual3
            )
            for alpha in (0, 1):
                branching_value = branching[(second_label, third_label, alpha)]
                # At n1=0 and alpha2=alpha3=alpha, the sign in
                # (rextendedblock) is (-1)^(alpha2 alpha3)=(-1)^alpha.
                prefactor = (-1) ** alpha * branching_value * branching_value
                contribution = shift(
                    virasoro_product,
                    base2,
                    base3,
                    maximum2,
                    maximum3,
                    prefactor,
                )
                for key, value in contribution.items():
                    add(answer, key, value)
                branch_terms.append(
                    {
                        "n2": str(second_label),
                        "n3": str(third_label),
                        "alpha": alpha,
                        "base_level": [base2, base3],
                        "B": complex_record(branching_value),
                        "prefactor": complex_record(prefactor),
                    }
                )
    return answer, branch_terms, diagnostics


def ramond_partition_coefficients(maximum, fermion_sign=1):
    """Coefficients of prod_(n>=1) (1+s Q^n)/(1-Q^n)."""
    coefficients = [0] * (maximum + 1)
    coefficients[0] = 1
    for mode in range(1, maximum + 1):
        updated = [0] * (maximum + 1)
        for old_level, old_value in enumerate(coefficients):
            if not old_value:
                continue
            boson_count = 0
            while old_level + boson_count * mode <= maximum:
                level = old_level + boson_count * mode
                updated[level] += old_value
                if level + mode <= maximum:
                    updated[level + mode] += fermion_sign * old_value
                boson_count += 1
        coefficients = updated
    return coefficients


def distinct_partition_coefficients(maximum, sign=1):
    """Coefficients of prod_(n>=1) (1+sign Q^n)."""
    coefficients = [0] * (maximum + 1)
    coefficients[0] = 1
    for mode in range(1, maximum + 1):
        for level in range(maximum, mode - 1, -1):
            coefficients[level] += sign * coefficients[level - mode]
    return coefficients


def a_rs(r, s, b):
    """Ramond inverse-null-norm coefficient in arXiv:1207.5740."""
    answer = 2.0 ** (r * s - 2)
    for m in range(1 - r, r + 1):
        for n in range(1 - s, s + 1):
            if (m + n) % 2 != 1:
                continue
            if (m, n) in ((0, 0), (r, s)):
                continue
            answer /= m * b + n / b
    return complex(answer)


def beta_rs(r, s, b):
    return (r * b + s / b) / (2 * math.sqrt(2))


def beta_prime_rs(r, s, b):
    return (-1) ** s * (r * b - s / b) / (2 * math.sqrt(2))


def delta_rs(r, s, b, central_charge):
    beta = beta_rs(r, s, b)
    return central_charge / 24 - beta * beta


def fusion_r(lambda_external, beta_other, eta, r, s, b):
    """The polynomial P_{rs}^{R,eta} in the appendix of SCblock.tex."""
    answer = 1.0 + 0.0j
    for k in range(r):
        for ell in range(s):
            lattice = (1 - r + 2 * k) * b + (1 - s + 2 * ell) / b
            if (k + ell) % 2 == 0:
                numerator = lambda_external - 2 * math.sqrt(2) * eta * beta_other - lattice
            else:
                numerator = lambda_external + 2 * math.sqrt(2) * eta * beta_other - lattice
            answer *= numerator / (2 * math.sqrt(2))
    return complex(answer)


def continued_beta(beta, delta, new_delta, central_charge):
    """Continue a chosen beta sheet while Delta=c/24-beta^2 changes."""
    old_root = cmath.sqrt(central_charge / 24 - delta)
    if abs(old_root) < 1.0e-13:
        sheet = 1.0 + 0.0j
    else:
        sheet = beta / old_root
        sheet /= abs(sheet)
    return sheet * cmath.sqrt(central_charge / 24 - new_delta)


def inverse_physical_gram(module, level):
    basis = module.basis(level)[2]
    matrix = np.asarray(
        [[module.inner(left, right) for right in basis] for left in basis],
        dtype=np.complex128,
    )
    return basis, np.linalg.inv(matrix)


def direct_sca_two_point_series(b, momenta, maximum2, maximum3):
    """Direct physical SCA block, used only to determine the regular part."""
    free_modules = (
        FreeFieldModule("NS", b, momenta[0]),
        FreeFieldModule("R", b, momenta[1]),
        FreeFieldModule("R", b, momenta[2]),
    )
    modules = tuple(PBWModule(module) for module in free_modules)
    form = PhysicalThreePoint(modules, 0, 1)
    external = ((), ())
    second_data = [inverse_physical_gram(modules[1], level) for level in range(maximum2 + 1)]
    third_data = [inverse_physical_gram(modules[2], level) for level in range(maximum3 + 1)]
    answer = {}
    for level2 in range(maximum2 + 1):
        basis2, inverse2 = second_data[level2]
        for level3 in range(maximum3 + 1):
            basis3, inverse3 = third_data[level3]
            rho = np.zeros((len(basis2), len(basis3)), dtype=np.complex128)
            sewing = np.zeros_like(rho)
            for row, second in enumerate(basis2):
                parity2 = modules[1].parity(second)
                for column, third in enumerate(basis3):
                    parity3 = modules[2].parity(third)
                    if (parity2 + parity3) % 2:
                        continue
                    rho[row, column] = form.value((external, second, third))
                    sewing[row, column] = (-1) ** (parity2 * parity3)
            answer[(level2, level3)] = complex(
                np.einsum(
                    "ij,kl,ik,jl,ik->",
                    inverse2,
                    inverse3,
                    rho,
                    rho,
                    sewing,
                    optimize=True,
                )
            )
    return answer


def free_fermion_two_point_series(b, momenta, maximum2, maximum3):
    modules = (
        FreeFieldModule("NS", b, momenta[0]),
        FreeFieldModule("R", b, momenta[1]),
        FreeFieldModule("R", b, momenta[2]),
    )
    form = MainConventionAuxiliaryThreePoint(modules)
    answer = {}
    for level2 in range(maximum2 + 1):
        second_states = tuple(
            (modes, ground)
            for modes in strict_partitions(level2)
            for ground in (0, 1)
        )
        for level3 in range(maximum3 + 1):
            third_states = tuple(
                (modes, ground)
                for modes in strict_partitions(level3)
                for ground in (0, 1)
            )
            coefficient = 0.0j
            for second in second_states:
                parity2 = modules[1].auxiliary_parity(second)
                for third in third_states:
                    parity3 = modules[2].auxiliary_parity(third)
                    if (parity2 + parity3) % 2:
                        continue
                    rho = form.value(((), second, third))
                    coefficient += (-1) ** (parity2 * parity3) * rho * rho
            answer[(level2, level3)] = coefficient
    return answer


class RamondTwoPointHRecursion:
    """Pole reconstruction along beta2=t, beta3=t+d.

    Hadasz's Ramond recursion identifies the simple poles at
    beta=+/-beta_rs.  For a two-point necklace there are two pole families.
    Choosing beta2=t and beta3=t+d makes every pole position linear in t and
    removes the square-root sheets that obstruct a recursion directly in a
    common conformal weight.  At fixed q2 and q3 levels, the non-polar part is
    a polynomial in t.  Its degree and coefficients are fixed from
    large-|t| samples; the values at the target momenta are never sampled.
    """

    def __init__(self, b, external_momentum, central_charge, maximum2, maximum3):
        self.b = float(b)
        self.external_momentum = float(external_momentum)
        self.central_charge = float(central_charge)
        self.maximum2 = int(maximum2)
        self.maximum3 = int(maximum3)
        self.calls = 0
        self._direct_cache = {}
        self._fit_cache = {}
        self.diagnostics = []

    def degenerate_pairs(self, maximum_level):
        pairs = []
        for r in range(1, 2 * maximum_level + 1):
            for s in range(1, 2 * maximum_level + 1):
                if (r + s) % 2 != 1 or (r * s) % 2:
                    continue
                level = r * s // 2
                if 1 <= level <= maximum_level:
                    pairs.append((r, s, level))
        return tuple(pairs)

    def pole_data(self, difference, level2, level3):
        poles = []
        for r, s, null_level in self.degenerate_pairs(level2):
            beta = beta_rs(r, s, self.b)
            poles.extend(
                [
                    (beta, "edge2", r, s, null_level, 1),
                    (-beta, "edge2", r, s, null_level, -1),
                ]
            )
        for r, s, null_level in self.degenerate_pairs(level3):
            beta = beta_rs(r, s, self.b)
            poles.extend(
                [
                    (beta - difference, "edge3", r, s, null_level, 1),
                    (-beta - difference, "edge3", r, s, null_level, -1),
                ]
            )
        positions = [item[0] for item in poles]
        if len(positions) != len(set(round(value, 12) for value in positions)):
            raise ValueError("Two degenerate pole positions collided at this sample point.")
        return tuple(poles)

    def direct_series(self, parameter, difference):
        key = (float(parameter), float(difference))
        if key not in self._direct_cache:
            momenta = (
                self.external_momentum,
                math.sqrt(2) * parameter,
                math.sqrt(2) * (parameter + difference),
            )
            self._direct_cache[key] = direct_sca_two_point_series(
                self.b, momenta, self.maximum2, self.maximum3
            )
        return self._direct_cache[key]

    def inverse_null_norm(self, r, s):
        """Inverse null norm in the no-G0, two-ground-state basis of the notes."""
        if r * s != 2:
            raise NotImplementedError(
                "The beta-dependent basis conversion is presently derived at level one."
            )
        beta = beta_rs(r, s, self.b)
        weight = self.central_charge / 24 - beta * beta
        derivative = (
            2
            + 2 * weight / (beta * beta)
            - 32 * weight * weight / (9 * beta * beta)
        )
        return 1 / derivative

    def analytic_residue(self, difference, level2, level3, pole):
        position, edge, r, s, null_level, sign = pole
        beta = beta_rs(r, s, self.b)
        shifted_beta = sign * beta_prime_rs(r, s, self.b)
        if edge == "edge2":
            other_beta = sign * beta + difference
            shifted = self.coefficient(
                shifted_beta,
                other_beta - shifted_beta,
                level2 - null_level,
                level3,
            )
        else:
            other_beta = sign * beta - difference
            shifted = self.coefficient(
                other_beta,
                shifted_beta - other_beta,
                level2,
                level3 - null_level,
            )
        polynomial = fusion_r(
            2 * self.external_momentum,
            other_beta,
            sign,
            r,
            s,
            self.b,
        )
        # Delta-Delta_rs = -(beta^2-beta_rs^2).  Consequently its
        # conversion to a residue in t=beta2 (or beta3=t+d) is
        # -sign/(2 beta_rs).
        return (
            -sign
            * self.inverse_null_norm(r, s)
            * polynomial
            * polynomial
            * shifted
            / (2 * beta)
        )

    def fit(self, difference, level2, level3):
        key = (float(difference), int(level2), int(level3))
        if key in self._fit_cache:
            return self._fit_cache[key]
        poles = self.pole_data(difference, level2, level3)
        degree = 2 * (level2 + level3)
        unknowns = degree + 1
        residues = tuple(
            self.analytic_residue(difference, level2, level3, pole)
            for pole in poles
        )
        candidates = tuple(-4.0 + 0.1 * index for index in range(81))
        training = tuple(
            value
            for index, value in enumerate(candidates)
            if index % 2 == 0
            and min((abs(value - pole[0]) for pole in poles), default=1.0) > 0.08
        )
        validation = tuple(
            value
            for index, value in enumerate(candidates)
            if index % 2 == 1
            and min((abs(value - pole[0]) for pole in poles), default=1.0) > 0.08
        )
        if len(training) < unknowns + 4:
            raise AssertionError("Too few safe interpolation samples remain.")
        scale = max(abs(value) for value in training)

        def row(parameter):
            return [(parameter / scale) ** power for power in range(degree + 1)]

        def polar(parameter):
            return sum(
                residue / (parameter - pole[0])
                for residue, pole in zip(residues, poles)
            )

        matrix = np.asarray([row(value) for value in training], dtype=np.complex128)
        values = np.asarray(
            [
                self.direct_series(value, difference)[(level2, level3)] - polar(value)
                for value in training
            ],
            dtype=np.complex128,
        )
        column_norms = np.linalg.norm(matrix, axis=0)
        normalized_matrix = matrix / column_norms
        normalized_coefficients, _, rank, singular_values = np.linalg.lstsq(
            normalized_matrix, values, rcond=1.0e-14
        )
        coefficients = normalized_coefficients / column_norms
        if rank != unknowns:
            raise AssertionError("The h-recursion interpolation matrix lost rank.")
        training_relative = float(np.linalg.norm(matrix @ coefficients - values)) / max(
            float(np.linalg.norm(values)), 1.0
        )
        validation_errors = []
        for parameter in validation:
            predicted = np.dot(np.asarray(row(parameter)), coefficients) + polar(parameter)
            direct = self.direct_series(parameter, difference)[(level2, level3)]
            validation_errors.append(abs(predicted - direct) / max(abs(direct), 1.0))

        residue_checks = []
        for pole, residue in zip(poles, residues):
            epsilon_values = (1.0e-4, 5.0e-5, 2.5e-5)
            sampled = np.asarray(
                [
                    epsilon
                    * self.direct_series(pole[0] + epsilon, difference)[
                        (level2, level3)
                    ]
                    for epsilon in epsilon_values
                ],
                dtype=np.complex128,
            )
            extrapolation = np.asarray(
                [[1.0, epsilon, epsilon * epsilon] for epsilon in epsilon_values],
                dtype=np.complex128,
            )
            direct_residue = np.linalg.solve(extrapolation, sampled)[0]
            residue_checks.append(
                {
                    "position": float(pole[0]),
                    "edge": pole[1],
                    "r": pole[2],
                    "s": pole[3],
                    "null_level": pole[4],
                    "sign": pole[5],
                    "analytic_residue": complex_record(residue),
                    "direct_residue": complex_record(direct_residue),
                    "relative_difference": float(abs(residue - direct_residue))
                    / max(float(abs(direct_residue)), 1.0),
                }
            )
        diagnostic = {
            "difference": float(difference),
            "q2_level": int(level2),
            "q3_level": int(level3),
            "polynomial_degree": degree,
            "pole_count": len(poles),
            "rank": int(rank),
            "smallest_singular_value": float(singular_values[-1]),
            "training_relative_residual": training_relative,
            "maximum_validation_relative_error": float(max(validation_errors, default=0.0)),
            "residue_checks": residue_checks,
        }
        self.diagnostics.append(diagnostic)
        result = (tuple(coefficients), scale, poles, residues)
        self._fit_cache[key] = result
        return result

    def coefficient(self, parameter, difference, level2, level3):
        coefficients, scale, poles, residues = self.fit(difference, level2, level3)
        degree = 2 * (level2 + level3)
        regular = sum(
            coefficient * (parameter / scale) ** power
            for power, coefficient in enumerate(coefficients)
        )
        polar = sum(
            residue / (parameter - pole[0])
            for residue, pole in zip(residues, poles)
        )
        self.calls += 1
        return complex(regular + polar)

    def series(self, beta2, beta3):
        difference = beta3 - beta2
        return {
            (level2, level3): self.coefficient(
                beta2, difference, level2, level3
            )
            for level2 in range(self.maximum2 + 1)
            for level3 in range(self.maximum3 + 1)
        }


def h_recursion_extended_series(b, momenta, maximum2, maximum3):
    q = b + 1 / b
    central_charge = 1.5 + 3 * q * q
    h1 = q * q / 8 - momenta[0] * momenta[0] / 2
    beta2 = momenta[1] / math.sqrt(2)
    beta3 = momenta[2] / math.sqrt(2)
    delta2 = central_charge / 24 - beta2 * beta2
    delta3 = central_charge / 24 - beta3 * beta3

    recursion = RamondTwoPointHRecursion(
        b, momenta[0], central_charge, maximum2, maximum3
    )
    sca = recursion.series(beta2, beta3)

    free = free_fermion_two_point_series(
        b, momenta, maximum2, maximum3
    )
    extended = multiply(sca, free, maximum2, maximum3)
    return extended, sca, free, recursion.calls, recursion.diagnostics, {
        "c": central_charge,
        "h1": h1,
        "beta2": beta2,
        "beta3": beta3,
        "Delta2": delta2,
        "Delta3": delta3,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--b", default="7/5")
    parser.add_argument("--p1", default="11/23")
    parser.add_argument("--p2", default="13/29")
    parser.add_argument("--p3", default="17/31")
    parser.add_argument("--level", type=int, default=1)
    parser.add_argument("--json", type=Path, default=HERE / "results.json")
    arguments = parser.parse_args()

    b = parse_fraction(arguments.b)
    momenta = tuple(parse_fraction(value) for value in (arguments.p1, arguments.p2, arguments.p3))
    maximum2 = maximum3 = arguments.level

    started = time.perf_counter()
    double_started = time.perf_counter()
    double, branch_terms, ward_diagnostics = double_virasoro_series(
        b, momenta, maximum2, maximum3
    )
    double_seconds = time.perf_counter() - double_started

    recursion_started = time.perf_counter()
    recursive, sca, free, recursion_calls, regular_diagnostics, parameters = h_recursion_extended_series(
        b, momenta, maximum2, maximum3
    )
    recursion_seconds = time.perf_counter() - recursion_started

    comparisons = []
    maximum_absolute = 0.0
    maximum_relative = 0.0
    for level2 in range(maximum2 + 1):
        for level3 in range(maximum3 + 1):
            first = double.get((level2, level3), 0.0j)
            second = recursive.get((level2, level3), 0.0j)
            absolute = abs(first - second)
            relative = absolute / max(abs(first), abs(second), 1.0)
            maximum_absolute = max(maximum_absolute, absolute)
            maximum_relative = max(maximum_relative, relative)
            comparisons.append(
                {
                    "q2_level": level2,
                    "q3_level": level3,
                    "double_virasoro": complex_record(first),
                    "h_recursion_times_fermion": complex_record(second),
                    "absolute_difference": float(absolute),
                    "relative_difference": float(relative),
                }
            )

    passed = maximum_relative < TOLERANCE
    payload = {
        "scope": {
            "limit": "q1 -> 0",
            "f": 0,
            "eta": 1,
            "eta_prime": 1,
            "plumbing_spin_signs": [1, 1, 1],
            "maximum_q2_level": maximum2,
            "maximum_q3_level": maximum3,
        },
        "input": {
            "b": arguments.b,
            "P1": arguments.p1,
            "P2": arguments.p2,
            "P3": arguments.p3,
        },
        "parameters": {key: complex_record(value) for key, value in parameters.items()},
        "ward_diagnostics": ward_diagnostics,
        "branch_terms": branch_terms,
        "sca_h_recursion": {
            f"{level2},{level3}": complex_record(value)
            for (level2, level3), value in sorted(sca.items())
        },
        "free_fermion": {
            f"{level2},{level3}": complex_record(value)
            for (level2, level3), value in sorted(free.items())
        },
        "comparisons": comparisons,
        "maximum_absolute_difference": maximum_absolute,
        "maximum_relative_difference": maximum_relative,
        "h_recursion_calls": recursion_calls,
        "regular_part_diagnostics": regular_diagnostics,
        "double_virasoro_seconds": double_seconds,
        "h_recursion_seconds": recursion_seconds,
        "total_seconds": time.perf_counter() - started,
        "passed": passed,
    }
    arguments.json.write_text(json.dumps(payload, indent=2) + "\n")

    for item in comparisons:
        left = complex(
            item["double_virasoro"]["real"], item["double_virasoro"]["imag"]
        )
        right = complex(
            item["h_recursion_times_fermion"]["real"],
            item["h_recursion_times_fermion"]["imag"],
        )
        print(
            f"q2^{item['q2_level']} q3^{item['q3_level']}: "
            f"double Vir={left:.12g}, h-recursion={right:.12g}, "
            f"|difference|={item['absolute_difference']:.3e}"
        )
    print(f"maximum relative difference: {maximum_relative:.3e}")
    print(f"double-Virasoro runtime: {double_seconds:.3f} s")
    print(f"h-recursion runtime: {recursion_seconds:.3f} s")
    print("PASS" if passed else "FAIL")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
