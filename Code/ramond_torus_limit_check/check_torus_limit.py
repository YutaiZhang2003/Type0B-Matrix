#!/usr/bin/env python3
"""Check the q1 -> 0 Ramond theta block through total level six.

There are two independent constructions.

1.  Specialize the sewn SCA block and the auxiliary-fermion block to q1=0,
    compute both from their PBW bases and inverse Gram matrices, and combine
    them with the Ramond convolution of SCblock.tex.
2.  Put n1=0 in the double-Virasoro decomposition, insert the normalized
    Ramond branching coefficients, and compute the two ordinary Virasoro
    torus two-point blocks directly from Virasoro Ward identities.

The output keeps the two spin parities separate.  Thus equality of the formal
coefficients implies equality for all four choices (eta2, eta3) in {+1,-1}^2.
Only the canonical pairs rho_0^(+) and rho_1^(-), for which the direct Ward
evaluator in the branching implementation is defined, are used.
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
BRANCHING = HERE.parent / "ramond_branching_recursion"
sys.path.insert(0, str(BRANCHING))

from compute_target import (  # noqa: E402
    BranchWeights,
    FreeFieldModule,
    TOLERANCE,
    VirasoroThreePoint,
    norm_product,
    ordinary_factor,
    partitions,
    solve_ns_l1,
    solve_ramond_lminus,
    strict_partitions,
)
from direct_state_check import (  # noqa: E402
    AuxiliaryThreePoint,
    DirectBranchingCoefficient,
    PBWModule,
    PhysicalThreePoint,
    branch_in_pbw,
)


SeriesKey = tuple[int, int, int, int]
Series = dict[SeriesKey, complex]
BivariateSeries = dict[tuple[int, int], complex]


class RamondAuxiliaryThreePoint(AuxiliaryThreePoint):
    """Auxiliary NS-R-R form in the BPZ convention of the main notes."""

    def base_value(self, states):
        ground2 = states[1][1]
        ground3 = states[2][1]
        if ground2 != ground3:
            return 0.0j
        # This is a chiral three-point phase, not the BPZ norm of u^1.
        # The latter remains -1 in AuxiliaryThreePoint.inner.
        return 1.0 + 0.0j if ground2 == 0 else 1.0j


def add(series, key, value, tolerance=1.0e-12):
    value = series.get(key, 0.0j) + complex(value)
    if abs(value) < tolerance:
        series.pop(key, None)
    else:
        series[key] = value


def ramond_star(left: Series, right: Series, cutoff: int) -> Series:
    """The q1=0 restriction of star_R in SCblock.tex."""
    answer: Series = {}
    for (r2, r3, epsilon2, epsilon3), first in left.items():
        for (s2, s3, delta2, delta3), second in right.items():
            level2 = r2 + s2
            level3 = r3 + s3
            if level2 + level3 > cutoff:
                continue
            sign = (-1) ** (
                epsilon2 * delta3 + delta2 * epsilon3
            )
            key = (
                level2,
                level3,
                (epsilon2 + delta2) % 2,
                (epsilon3 + delta3) % 2,
            )
            add(answer, key, sign * first * second)
    return answer


def bivariate_product(
    left: BivariateSeries, right: BivariateSeries, cutoff: int
) -> BivariateSeries:
    answer: BivariateSeries = {}
    for (r2, r3), first in left.items():
        for (s2, s3), second in right.items():
            key = (r2 + s2, r3 + s3)
            if sum(key) <= cutoff:
                add(answer, key, first * second)
    return answer


def encode(value: complex):
    return {"real": float(value.real), "imag": float(value.imag)}


def encode_series(series: Series):
    return [
        {
            "level_2": key[0],
            "level_3": key[1],
            "eta_2_parity": key[2],
            "eta_3_parity": key[3],
            "coefficient": encode(value),
        }
        for key, value in sorted(series.items())
    ]


class VirasoroTorusTwoPoint:
    """The q1=0 specialization of the ordinary theta block."""

    def __init__(self, weights, central_charge, internal_slots=(1, 2)):
        self.weights = tuple(complex(value) for value in weights)
        self.central_charge = complex(central_charge)
        self.internal_slots = tuple(internal_slots)
        self.three_point = VirasoroThreePoint(
            self.weights, self.central_charge
        )
        self._gram_cache = {}

    @lru_cache(None)
    def _module_evaluator(self, slot):
        weights = [0.0j, 0.0j, 0.0j]
        weights[slot] = self.weights[slot]
        return VirasoroThreePoint(tuple(weights), self.central_charge)

    def inner(self, slot, left, right):
        evaluator = self._module_evaluator(slot)
        expression = {tuple(right): 1.0 + 0.0j}
        for mode in left:
            following = {}
            for word, outer in expression.items():
                for final, inner in evaluator.act(slot, mode, word).items():
                    add(following, final, outer * inner)
            expression = following
        return expression.get((), 0.0j)

    def inverse_gram(self, slot, level):
        key = (slot, level)
        if key in self._gram_cache:
            return self._gram_cache[key]
        basis = partitions(level)
        gram = np.asarray(
            [
                [self.inner(slot, left, right) for right in basis]
                for left in basis
            ],
            dtype=np.complex128,
        )
        inverse = np.linalg.inv(gram)
        self._gram_cache[key] = (basis, inverse)
        return basis, inverse

    def series(self, cutoff):
        answer: BivariateSeries = {}
        for level2 in range(cutoff + 1):
            for level3 in range(cutoff - level2 + 1):
                slot2, slot3 = self.internal_slots
                basis2, inverse2 = self.inverse_gram(slot2, level2)
                basis3, inverse3 = self.inverse_gram(slot3, level3)
                rho = np.asarray(
                    [
                        [
                            self.three_point.value(
                                *tuple(
                                    state2 if slot == slot2
                                    else state3 if slot == slot3
                                    else ()
                                    for slot in range(3)
                                )
                            )
                            for state3 in basis3
                        ]
                        for state2 in basis2
                    ],
                    dtype=np.complex128,
                )
                coefficient = np.einsum(
                    "ik,ij,kl,jl->",
                    rho,
                    inverse2,
                    inverse3,
                    rho,
                    optimize=True,
                )
                add(answer, (level2, level3), coefficient)
        return answer


class DirectTorusTwoPoint:
    """Direct SCA and free-fermion torus two-point sewing contractions."""

    def __init__(self, b, momenta):
        self.free_modules = (
            FreeFieldModule("NS", b, momenta[0]),
            FreeFieldModule("R", b, momenta[1]),
            FreeFieldModule("R", b, momenta[2]),
        )
        self.pbw_modules = tuple(PBWModule(module) for module in self.free_modules)
        self.auxiliary_form = RamondAuxiliaryThreePoint(self.free_modules)
        self._physical_inverse_cache = {}
        self._auxiliary_inverse_cache = {}

    def physical_inverse_gram(self, slot, level):
        key = (slot, level)
        if key in self._physical_inverse_cache:
            return self._physical_inverse_cache[key]
        module = self.pbw_modules[slot]
        basis = module.basis(level)[2]
        gram = np.asarray(
            [[module.inner(left, right) for right in basis] for left in basis],
            dtype=np.complex128,
        )
        inverse = np.linalg.inv(gram)
        answer = (basis, inverse)
        self._physical_inverse_cache[key] = answer
        return answer

    @staticmethod
    @lru_cache(None)
    def auxiliary_basis(level):
        return tuple(
            (modes, ground)
            for modes in strict_partitions(level)
            for ground in (0, 1)
        )

    def auxiliary_inverse_gram(self, slot, level):
        key = (slot, level)
        if key in self._auxiliary_inverse_cache:
            return self._auxiliary_inverse_cache[key]
        basis = self.auxiliary_basis(level)
        gram = np.asarray(
            [
                [self.auxiliary_form.inner(slot, left, right) for right in basis]
                for left in basis
            ],
            dtype=np.complex128,
        )
        inverse = np.linalg.inv(gram)
        answer = (basis, inverse)
        self._auxiliary_inverse_cache[key] = answer
        return answer

    def physical_series(self, cutoff, form_parity, eta):
        form = PhysicalThreePoint(self.pbw_modules, form_parity, eta)
        answer: Series = {}
        for level2 in range(cutoff + 1):
            for level3 in range(cutoff - level2 + 1):
                basis2, inverse2 = self.physical_inverse_gram(1, level2)
                basis3, inverse3 = self.physical_inverse_gram(2, level3)
                for parity2 in (0, 1):
                    parity3 = (form_parity - parity2) % 2
                    indices2 = [
                        index for index, state in enumerate(basis2)
                        if self.pbw_modules[1].parity(state) == parity2
                    ]
                    indices3 = [
                        index for index, state in enumerate(basis3)
                        if self.pbw_modules[2].parity(state) == parity3
                    ]
                    if not indices2 or not indices3:
                        continue
                    rho = np.asarray(
                        [
                            [form.value((((), ()), basis2[i], basis3[k]))
                             for k in indices3]
                            for i in indices2
                        ],
                        dtype=np.complex128,
                    )
                    block2 = inverse2[np.ix_(indices2, indices2)]
                    block3 = inverse3[np.ix_(indices3, indices3)]
                    coefficient = (-1) ** (parity2 * parity3) * np.einsum(
                        "ik,ij,kl,jl->",
                        rho,
                        block2,
                        block3,
                        rho,
                        optimize=True,
                    )
                    add(
                        answer,
                        (level2, level3, parity2, parity3),
                        coefficient,
                    )
        return answer

    def auxiliary_series(self, cutoff):
        answer: Series = {}
        vacuum = ()
        for level2 in range(cutoff + 1):
            for level3 in range(cutoff - level2 + 1):
                basis2, inverse2 = self.auxiliary_inverse_gram(1, level2)
                basis3, inverse3 = self.auxiliary_inverse_gram(2, level3)
                for parity2 in (0, 1):
                    parity3 = parity2
                    indices2 = [
                        index for index, state in enumerate(basis2)
                        if self.auxiliary_form.parity(1, state) == parity2
                    ]
                    indices3 = [
                        index for index, state in enumerate(basis3)
                        if self.auxiliary_form.parity(2, state) == parity3
                    ]
                    rho = np.asarray(
                        [
                            [self.auxiliary_form.value(
                                (vacuum, basis2[i], basis3[k])
                            ) for k in indices3]
                            for i in indices2
                        ],
                        dtype=np.complex128,
                    )
                    block2 = inverse2[np.ix_(indices2, indices2)]
                    block3 = inverse3[np.ix_(indices3, indices3)]
                    coefficient = (-1) ** (parity2 * parity3) * np.einsum(
                        "ik,ij,kl,jl->",
                        rho,
                        block2,
                        block3,
                        rho,
                        optimize=True,
                    )
                    add(
                        answer,
                        (level2, level3, parity2, parity3),
                        coefficient,
                    )
        return answer


class BranchingTorusLimit:
    """Double-Virasoro q1=0 expansion with the first-Ward coefficients."""

    def __init__(self, b, momenta):
        self.b = float(b)
        self.momenta = tuple(float(value) for value in momenta)
        self.weights = BranchWeights(self.b, self.momenta)
        self.modules = (
            FreeFieldModule("NS", self.b, self.momenta[0]),
            FreeFieldModule("R", self.b, self.momenta[1]),
            FreeFieldModule("R", self.b, self.momenta[2]),
        )
        self._virasoro_cache = {}
        self._action_cache = {}
        self._ward_cache = {}

    @staticmethod
    def branch_shift(label):
        label = Fraction(label)
        shift = 2 * label * label - Fraction(1, 8)
        if shift.denominator != 1:
            raise AssertionError("A Ramond branch shift must be integral.")
        return int(shift)

    @staticmethod
    def labels(cutoff):
        answer = []
        numerator = 1
        while True:
            label = Fraction(numerator, 4)
            shift = BranchingTorusLimit.branch_shift(label)
            if shift > cutoff:
                break
            answer.extend((-label, label))
            numerator += 2
        return tuple(answer)

    def virasoro_series(self, copy, label2, label3, cutoff):
        key = (copy, Fraction(label2), Fraction(label3), cutoff)
        if key not in self._virasoro_cache:
            labels = (Fraction(0), Fraction(label2), Fraction(label3))
            triple = self.weights.triple(labels, copy)
            block = VirasoroTorusTwoPoint(
                triple,
                self.weights.central_charges[copy],
            )
            self._virasoro_cache[key] = block.series(cutoff)
        return self._virasoro_cache[key]

    def ns_actions(self):
        if "NS" not in self._action_cache:
            actions = {Fraction(0): ()}
            for label in (Fraction(1), Fraction(2)):
                actions[label] = solve_ns_l1(self.modules[0], label)[0]
            self._action_cache["NS"] = actions
        return self._action_cache["NS"]

    def ramond_actions(self, slot, alpha, labels):
        key = ("R", slot, alpha, tuple(labels))
        if key not in self._action_cache:
            self._action_cache[key] = {
                label: solve_ramond_lminus(
                    self.modules[slot], label, alpha
                )[0]
                for label in labels
            }
        return self._action_cache[key]

    def ground_value(self, label2, label3, alpha2, alpha3, eta):
        """The four tensor-ground anchors in the main-note w^+/- basis."""
        second = self.modules[1].r_branch(label2, alpha2)
        third = self.modules[2].r_branch(label3, alpha3)
        form_parity = (alpha2 + alpha3) % 2
        # The unit-norm oscillator ground is f^1=exp(3 pi i/4) w^-.
        # Indeed BPZ self-adjointness of G_0 and the convention in the main
        # notes give <w^-|w^->=i<w^+|w^+>.
        odd_phase = cmath.exp(3j * math.pi / 4)
        answer = 0.0j
        for state2, coefficient2 in second.items():
            for state3, coefficient3 in third.items():
                if (
                    state2[0] or state2[2] or state2[3]
                    or state3[0] or state3[2] or state3[3]
                ):
                    raise AssertionError("A ground anchor contains oscillators.")
                auxiliary2, physical2 = state2[1], state2[4]
                auxiliary3, physical3 = state3[1], state3[4]
                if auxiliary2 != auxiliary3:
                    continue
                auxiliary_form = 1 if auxiliary2 == 0 else 1j
                tensor_sign = (-1) ** (physical2 * auxiliary3)
                if form_parity == 0:
                    physical_form = (
                        1 if (physical2, physical3) == (0, 0)
                        else eta if (physical2, physical3) == (1, 1)
                        else 0
                    )
                else:
                    physical_form = (
                        1 if (physical2, physical3) == (0, 1)
                        else 1j * eta if (physical2, physical3) == (1, 0)
                        else 0
                    )
                answer += (
                    coefficient2
                    * coefficient3
                    * tensor_sign
                    * auxiliary_form
                    * odd_phase ** (physical2 + physical3)
                    * physical_form
                )
        return answer

    def ward_values(self, cutoff, alpha2, alpha3, eta):
        """Solve the closed first-Ward system for all labels needed at q1=0.

        The direct PBW evaluator is intentionally not used away from the four
        tensor-ground anchors.  At mixed reflected labels its old free-field
        implementation does not obey the embedded-Virasoro Ward identity;
        the finite system is the branching-coefficient algorithm of the main
        notes and fixes those values from the anchors.
        """
        cache_key = (cutoff, alpha2, alpha3, eta)
        if cache_key in self._ward_cache:
            return self._ward_cache[cache_key]
        labels1 = (Fraction(0), Fraction(1), Fraction(2))
        # The first-Ward lattice must not be truncated at the requested block
        # order: doing so creates an artificial boundary and changes even the
        # level-one coefficients.  The level-six check requires the complete
        # symmetric window through |n|=7/4, which we also retain in lower-order
        # diagnostic runs.
        ramond_labels = tuple(sorted(self.labels(max(cutoff, 6))))
        unknowns = tuple(
            (first, second, third)
            for first in labels1
            for second in ramond_labels
            for third in ramond_labels
        )
        index = {labels: position for position, labels in enumerate(unknowns)}
        ns_actions = self.ns_actions()
        second_actions = self.ramond_actions(1, alpha2, ramond_labels)
        third_actions = self.ramond_actions(2, alpha3, ramond_labels)
        rows = []
        right_hand_sides = []

        def append_equation(equation):
            norm = np.linalg.norm(equation)
            if norm > TOLERANCE:
                rows.append(equation / norm)
                right_hand_sides.append(0.0j)

        for labels in unknowns:
            equation = np.zeros(len(unknowns), dtype=np.complex128)
            for term in ns_actions[labels[0]]:
                changed, coefficient = ordinary_factor(
                    self.weights, labels, 0, term
                )
                equation[index[changed]] += coefficient
            for term in second_actions[labels[1]]:
                changed, coefficient = ordinary_factor(
                    self.weights, labels, 1, term
                )
                equation[index[changed]] -= coefficient
            for term in third_actions[labels[2]]:
                changed, coefficient = ordinary_factor(
                    self.weights, labels, 2, term
                )
                equation[index[changed]] -= coefficient
            append_equation(equation)

        for label2 in (Fraction(-1, 4), Fraction(1, 4)):
            for label3 in (Fraction(-1, 4), Fraction(1, 4)):
                labels = (Fraction(0), label2, label3)
                row = np.zeros(len(unknowns), dtype=np.complex128)
                row[index[labels]] = 1.0
                rows.append(row)
                right_hand_sides.append(
                    self.ground_value(
                        label2,
                        label3,
                        alpha2,
                        alpha3,
                        eta,
                    )
                )

        matrix = np.asarray(rows, dtype=np.complex128)
        vector = np.asarray(right_hand_sides, dtype=np.complex128)
        column_norms = np.linalg.norm(matrix, axis=0)
        if np.any(column_norms == 0):
            raise AssertionError("The finite Ward system has an empty column.")
        scaled = matrix / column_norms
        scaled_solution, _, rank, singular_values = np.linalg.lstsq(
            scaled, vector, rcond=1.0e-13
        )
        solution = scaled_solution / column_norms
        residual = matrix @ solution - vector
        if rank != len(unknowns):
            raise AssertionError(
                f"The finite Ward system has rank {rank}, expected {len(unknowns)}."
            )
        values = {
            labels: solution[position]
            / norm_product(
                labels,
                alpha2,
                alpha3,
                self.b,
                self.momenta,
            )
            for position, labels in enumerate(unknowns)
            if labels[0] == 0
        }
        answer = {
            "values": values,
            "rows": int(matrix.shape[0]),
            "columns": int(matrix.shape[1]),
            "rank": int(rank),
            "relative_residual": float(np.linalg.norm(residual))
            / max(float(np.linalg.norm(vector)), 1.0),
            "smallest_singular_value": float(singular_values[rank - 1]),
        }
        self._ward_cache[cache_key] = answer
        return answer

    def series(self, cutoff, form_parity, eta):
        answer: Series = {}
        labels = self.labels(cutoff)
        for alpha2 in (0, 1):
            alpha3 = (form_parity - alpha2) % 2
            ward = self.ward_values(cutoff, alpha2, alpha3, eta)
            for label2 in labels:
                shift2 = self.branch_shift(label2)
                for label3 in labels:
                    shift3 = self.branch_shift(label3)
                    base_level = shift2 + shift3
                    if base_level > cutoff:
                        continue
                    residual = cutoff - base_level
                    first = self.virasoro_series(0, label2, label3, residual)
                    second = self.virasoro_series(1, label2, label3, residual)
                    product = bivariate_product(first, second, residual)
                    coefficient = ward["values"][
                        (Fraction(0), label2, label3)
                    ]
                    prefactor = (-1) ** (alpha2 * alpha3) * coefficient**2
                    for (level2, level3), value in product.items():
                        add(
                            answer,
                            (
                                shift2 + level2,
                                shift3 + level3,
                                alpha2,
                                alpha3,
                            ),
                            prefactor * value,
                        )
        return answer


def comparison(left: Series, right: Series, cutoff: int):
    rows = []
    maximum_absolute = 0.0
    maximum_relative = 0.0
    by_total = {}
    keys = sorted(set(left) | set(right))
    for key in keys:
        first = left.get(key, 0.0j)
        second = right.get(key, 0.0j)
        absolute = abs(first - second)
        scale = max(abs(first), abs(second), 1.0)
        relative = absolute / scale
        maximum_absolute = max(maximum_absolute, absolute)
        maximum_relative = max(maximum_relative, relative)
        total = key[0] + key[1]
        level = by_total.setdefault(
            total,
            {"maximum_absolute_error": 0.0, "maximum_relative_error": 0.0},
        )
        level["maximum_absolute_error"] = max(
            level["maximum_absolute_error"], absolute
        )
        level["maximum_relative_error"] = max(
            level["maximum_relative_error"], relative
        )
        rows.append(
            {
                "level_2": key[0],
                "level_3": key[1],
                "eta_2_parity": key[2],
                "eta_3_parity": key[3],
                "direct": encode(first),
                "branching": encode(second),
                "absolute_error": float(absolute),
                "relative_error": float(relative),
            }
        )
    for total in range(cutoff + 1):
        by_total.setdefault(
            total,
            {"maximum_absolute_error": 0.0, "maximum_relative_error": 0.0},
        )
    return {
        "coefficient_count": len(keys),
        "maximum_absolute_error": float(maximum_absolute),
        "maximum_relative_error": float(maximum_relative),
        "by_total_level": [
            {"total_level": level, **by_total[level]}
            for level in sorted(by_total)
        ],
        "coefficients": rows,
    }


def level_one_factorization_diagnostic(b, momenta):
    """Test the first embedded-Virasoro descendant before sewing a block."""
    evaluator = DirectBranchingCoefficient(b, momenta)
    evaluator.auxiliary_form = RamondAuxiliaryThreePoint(
        evaluator.free_modules
    )
    weights = BranchWeights(b, momenta)
    form_parity = 0
    eta = 1

    def physical_form(label2, label3):
        del label2, label3
        return PhysicalThreePoint(
            evaluator.pbw_modules, form_parity, eta
        )

    def evaluate(second, third, form):
        first = evaluator.branch(0, Fraction(0))
        answer = 0.0j
        for (auxiliary1, physical1), coefficient1 in first.items():
            parity_physical1 = evaluator.pbw_modules[0].parity(physical1)
            parity_auxiliary1 = evaluator.free_modules[0].auxiliary_parity(
                auxiliary1
            )
            for (auxiliary2, physical2), coefficient2 in second.items():
                parity_physical2 = evaluator.pbw_modules[1].parity(physical2)
                for (auxiliary3, physical3), coefficient3 in third.items():
                    parity_auxiliary3 = evaluator.free_modules[2].auxiliary_parity(
                        auxiliary3
                    )
                    sign = (-1) ** (
                        parity_physical1 * parity_auxiliary1
                        + parity_physical2 * parity_auxiliary3
                    )
                    answer += (
                        coefficient1
                        * coefficient2
                        * coefficient3
                        * sign
                        * evaluator.auxiliary_form.value(
                            (auxiliary1, auxiliary2, auxiliary3)
                        )
                        * form.value(
                            (physical1, physical2, physical3)
                        )
                    )
        return answer

    def descendant_value(label2, label3):
        form = physical_form(label2, label3)
        raw = evaluator.free_modules[1].r_branch(label2, 0)
        descendant = evaluator.free_modules[1].descendant(raw, (1,), ())
        second = branch_in_pbw(
            evaluator.free_modules[1], evaluator.pbw_modules[1], descendant
        )
        third = evaluator.branch(2, label3, 0)
        direct = evaluate(second, third, form)
        labels = (Fraction(0), label2, label3)
        primary = evaluate(
            evaluator.branch(1, label2, 0),
            evaluator.branch(2, label3, 0),
            form,
        )
        virasoro = VirasoroThreePoint(
            weights.triple(labels, 0), weights.central_charges[0]
        )
        factorized = primary * virasoro.value((), (1,), ())
        return {
            "n2": str(label2),
            "n3": str(label3),
            "direct": encode(direct),
            "factorized": encode(factorized),
            "absolute_error": float(abs(direct - factorized)),
        }

    module = evaluator.free_modules[1]
    pbw = evaluator.pbw_modules[1]
    product_basis = []
    for auxiliary_level in range(2):
        auxiliary_basis = tuple(
            (modes, ground)
            for modes in strict_partitions(auxiliary_level)
            for ground in (0, 1)
        )
        physical_basis = pbw.basis(1 - auxiliary_level)[2]
        product_basis.extend(
            (auxiliary, physical)
            for auxiliary in auxiliary_basis
            for physical in physical_basis
        )
    branch_columns = []
    for label in (Fraction(-1, 4), Fraction(1, 4)):
        for alpha in (0, 1):
            raw = module.r_branch(label, alpha)
            for first_word, second_word in (((1,), ()), ((), (1,))):
                expression = branch_in_pbw(
                    module,
                    pbw,
                    module.descendant(raw, first_word, second_word),
                )
                branch_columns.append(
                    [expression.get(state, 0.0j) for state in product_basis]
                )
    for label in (Fraction(-3, 4), Fraction(3, 4)):
        for alpha in (0, 1):
            expression = evaluator.branch(1, label, alpha)
            branch_columns.append(
                [expression.get(state, 0.0j) for state in product_basis]
            )
    transition = np.asarray(branch_columns, dtype=np.complex128).T
    return {
        "dimension": int(transition.shape[0]),
        "rank": int(np.linalg.matrix_rank(transition)),
        "condition_number": float(np.linalg.cond(transition)),
        "mixed_reflection": descendant_value(
            Fraction(-1, 4), Fraction(1, 4)
        ),
        "same_chamber": descendant_value(
            Fraction(1, 4), Fraction(1, 4)
        ),
    }


def parse_fraction(text):
    return float(Fraction(text))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cutoff", type=int, default=6)
    parser.add_argument("--b", default="7/5")
    parser.add_argument("--p1", default="11/23")
    parser.add_argument("--p2", default="13/29")
    parser.add_argument("--p3", default="17/31")
    parser.add_argument("--output", type=Path, default=HERE / "results.json")
    arguments = parser.parse_args()
    momenta = tuple(
        parse_fraction(value)
        for value in (arguments.p1, arguments.p2, arguments.p3)
    )
    b = parse_fraction(arguments.b)
    cutoff = arguments.cutoff

    timing = {}
    started = time.perf_counter()
    direct = DirectTorusTwoPoint(b, momenta)

    mark = time.perf_counter()
    auxiliary = direct.auxiliary_series(cutoff)
    timing["auxiliary_fermion_seconds"] = time.perf_counter() - mark

    branching = BranchingTorusLimit(b, momenta)
    sectors = []
    for form_parity, eta in ((0, 1), (1, -1)):
        mark = time.perf_counter()
        physical = direct.physical_series(cutoff, form_parity, eta)
        timing[f"direct_sca_f{form_parity}_seconds"] = time.perf_counter() - mark

        mark = time.perf_counter()
        enlarged_direct = ramond_star(auxiliary, physical, cutoff)
        timing[f"convolution_f{form_parity}_seconds"] = time.perf_counter() - mark

        mark = time.perf_counter()
        enlarged_branching = branching.series(
            cutoff, form_parity, eta
        )
        timing[f"branching_f{form_parity}_seconds"] = time.perf_counter() - mark

        sectors.append(
            {
                "form_parity": form_parity,
                "eta": eta,
                "physical_sca_torus_two_point": encode_series(physical),
                "enlarged_direct": encode_series(enlarged_direct),
                "enlarged_branching": encode_series(enlarged_branching),
                "comparison": comparison(
                    enlarged_direct, enlarged_branching, cutoff
                ),
            }
        )

    mark = time.perf_counter()
    factorization_diagnostic = level_one_factorization_diagnostic(b, momenta)
    timing["level_one_diagnostic_seconds"] = time.perf_counter() - mark
    timing["total_seconds"] = time.perf_counter() - started
    payload = {
        "description": "q1 -> 0 Ramond theta-block cross-check",
        "cutoff": cutoff,
        "parameters": {
            "b": arguments.b,
            "P1": arguments.p1,
            "P2": arguments.p2,
            "P3": arguments.p3,
        },
        "central_charge": 1.5 + 3 * (b + 1 / b) ** 2,
        "auxiliary_fermion": encode_series(auxiliary),
        "sectors": sectors,
        "level_one_factorization_diagnostic": factorization_diagnostic,
        "timing_seconds": timing,
    }
    arguments.output.write_text(json.dumps(payload, indent=2) + "\n")

    for sector in sectors:
        result = sector["comparison"]
        print(
            f"f={sector['form_parity']}, eta={sector['eta']:+d}: "
            f"{result['coefficient_count']} coefficients, "
            f"max abs={result['maximum_absolute_error']:.3e}, "
            f"max rel={result['maximum_relative_error']:.3e}"
        )
    print(f"total runtime: {timing['total_seconds']:.3f} s")
    print(f"wrote {arguments.output}")


if __name__ == "__main__":
    main()
