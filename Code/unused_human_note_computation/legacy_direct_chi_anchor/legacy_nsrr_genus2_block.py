#!/usr/bin/env python3
"""Low-order genus-two NS--R--R block from ``Vir x Vir`` branching.

This is the coefficientwise implementation of the Ramond theta-channel
formula in ``Human Notes/SCblock.tex``.  It keeps the three plumbing parities
open, in the component order

    p_NS + 2 p_R1 + 4 p_R2,

and computes the enlarged ``SCA x Majorana`` block as a sum over

* ``n_NS in Z/2`` and ``n_R in Z/2+1/4`` branching labels;
* the two Ramond-copy parities;
* the internally checked numerical branching recursion; and
* two ordinary Virasoro genus-two theta blocks.

For an independent check it also sews the auxiliary Majorana block and the
finite-level NS--R--R PBW block, then measures

    enlarged = auxiliary star physical.

The auxiliary ground vector is singular in four star-spectral channels, so
the forward identity is the unambiguous check; a full formal star quotient is
not unique without additional spin-structure data.  Integral NS labels use
the internally checked branching recursion.  Half-integral NS labels use an independent
boundary layer that expands the free-field chi strings, converts their
physical endpoints to the NS/R PBW bases, and evaluates the two three-point
forms with the Majorana and super-Virasoro Ward identities.

All public level triples are twice-levels.  Thus the Ramond entries are even
and ``(a,b,c)`` represents the monomial
``q_NS**(a/2) q_R1**(b/2) q_R2**(c/2)``.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from dataclasses import asdict, dataclass
from fractions import Fraction
from functools import lru_cache
from itertools import combinations
from pathlib import Path
import sys
from typing import Iterable, Mapping, Sequence

import numpy as np
import sympy as sp


THIS_DIR = Path(__file__).resolve().parent
CODE_DIR = THIS_DIR.parent
REPOSITORY = CODE_DIR.parent
for directory in (
    CODE_DIR / "c_Recursion",
    CODE_DIR / "genus_2_cross_channel",
    CODE_DIR / "ramond_branching_recursion",
):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from mixed_ns_ramond_descendant_blocks import NSVermaModule  # noqa: E402
from half_ns_anchor import DirectHalfNSAnchor  # noqa: E402
from ramond_pbw_generalized_ward import (  # noqa: E402
    GeneralizedNRRWard,
    RamondPBWModule,
)
from theta_star_algebra import star_multiply, theta_quadratic_sign  # noqa: E402
from torus_descendant_blocks import gram_matrix  # noqa: E402
import compute_target as BRANCHING_RECURSION  # noqa: E402

VirasoroThreePoint = BRANCHING_RECURSION.VirasoroThreePoint


Level = tuple[int, int, int]
ParityVector = tuple[complex, ...]
ComponentSeries = dict[Level, ParityVector]

ZERO_VECTOR: ParityVector = (0.0j,) * 8
HUMAN_CONVENTION_ONLY = True
PBW_DOUBLE_VIRASORO_MATCH_VERIFIED = False


def _load_chi_branching():
    source = REPOSITORY / "python 2" / "nsrr_chi_branching" / "nsrr_chi_formula.py"
    specification = importlib.util.spec_from_file_location(
        "_type0b_nsrr_chi_formula", source
    )
    if specification is None or specification.loader is None:
        raise ImportError(f"cannot load NSRR branching code from {source}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


CHI_BRANCHING = _load_chi_branching()


def _exact_real(value: object) -> sp.Expr:
    """Convert a real input to a compact exact SymPy number."""

    if isinstance(value, sp.Expr):
        expression = value
    elif isinstance(value, Fraction):
        expression = sp.Rational(value.numerator, value.denominator)
    elif isinstance(value, float):
        expression = sp.Rational(str(value))
    else:
        expression = sp.sympify(value)
    if expression.is_real is False:
        raise ValueError("the exact NSRR branching backend currently requires real data")
    return sp.cancel(expression)


def _complex_number(value: sp.Expr, precision: int) -> complex:
    return complex(sp.N(value, int(precision)))


def level_triples(maximum_total_twice_level: int) -> Iterable[Level]:
    """Yield NSRR-compatible triples through the total twice-level cutoff."""

    cutoff = int(maximum_total_twice_level)
    if cutoff < 0 or cutoff > 2:
        raise ValueError(
            "the low-order audit engine supports total twice-level 0 through 2"
        )
    for total in range(cutoff + 1):
        for ns_level in range(total + 1):
            for r1_level in range(0, total - ns_level + 1, 2):
                r2_level = total - ns_level - r1_level
                if r2_level % 2 == 0:
                    yield ns_level, r1_level, r2_level


def _add_vectors(left: Sequence[complex], right: Sequence[complex]) -> ParityVector:
    return tuple(complex(a) + complex(b) for a, b in zip(left, right))


def _scale_vector(scale: complex, vector: Sequence[complex]) -> ParityVector:
    return tuple(complex(scale) * complex(value) for value in vector)


def _put_component(
    series: ComponentSeries, levels: Level, component: int, value: complex
) -> None:
    current = list(series.get(levels, ZERO_VECTOR))
    current[int(component)] += complex(value)
    series[levels] = tuple(current)


def star_convolve_series(
    left: Mapping[Level, Sequence[complex]],
    right: Mapping[Level, Sequence[complex]],
    *,
    maximum_total_twice_level: int,
) -> ComponentSeries:
    """Multiply two truncated three-variable series in the R star algebra."""

    cutoff = int(maximum_total_twice_level)
    answer: ComponentSeries = {}
    for left_levels, left_vector in left.items():
        for right_levels, right_vector in right.items():
            levels = tuple(
                int(left_levels[edge]) + int(right_levels[edge])
                for edge in range(3)
            )
            if sum(levels) > cutoff:
                continue
            product = tuple(star_multiply(left_vector, right_vector))
            answer[levels] = _add_vectors(answer.get(levels, ZERO_VECTOR), product)
    return answer


def two_virasoro_parameters(
    *, momentum: complex, branch_label: Fraction, b: complex
) -> tuple[tuple[complex, complex], tuple[complex, complex]]:
    """Return ``((c1,h1),(c2,h2))`` in the Human-Note convention.

    The implementation uses only squared quantities, avoiding an arbitrary
    choice of the two correlated square-root branches in equation
    ``branchweights``.
    """

    momentum = complex(momentum)
    b = complex(b)
    twice_label = 2.0 * float(Fraction(branch_label))
    d1_squared = 2.0 - 2.0 * b * b
    b1_squared = 4.0 * b * b / d1_squared
    q1_squared = b1_squared + 2.0 + 1.0 / b1_squared
    h1 = q1_squared / 4.0 - (momentum + twice_label * b) ** 2 / d1_squared

    d2_squared = 2.0 - 2.0 / (b * b)
    inverse_b2_squared = 4.0 / (b * b * d2_squared)
    q2_squared = inverse_b2_squared + 2.0 + 1.0 / inverse_b2_squared
    h2 = (
        q2_squared / 4.0
        - (momentum + twice_label / b) ** 2 / d2_squared
    )
    return (
        (1.0 + 6.0 * q1_squared, h1),
        (1.0 + 6.0 * q2_squared, h2),
    )


class OrdinaryVirasoroThetaBlock:
    """Direct low-order PBW sewing of one ordinary Virasoro theta block."""

    def __init__(self, *, central_charge: complex, weights: Sequence[complex]):
        if len(weights) != 3:
            raise ValueError("a theta trinion has three Virasoro weights")
        self.central_charge = complex(central_charge)
        self.weights = tuple(complex(value) for value in weights)
        self.form = VirasoroThreePoint(self.weights, self.central_charge)

    @lru_cache(maxsize=None)
    def basis_and_inverse(self, edge: int, level: int) -> tuple[tuple, np.ndarray]:
        basis, matrix = gram_matrix(
            self.weights[int(edge)], self.central_charge, int(level)
        )
        return tuple(basis), np.linalg.inv(np.asarray(matrix, dtype=np.complex128))

    @lru_cache(maxsize=None)
    def coefficient(self, levels: Level) -> complex:
        levels = tuple(int(value) for value in levels)  # type: ignore[assignment]
        data = tuple(self.basis_and_inverse(edge, levels[edge]) for edge in range(3))
        bases = tuple(item[0] for item in data)
        inverses = tuple(item[1] for item in data)
        tensor = np.empty(tuple(len(basis) for basis in bases), dtype=np.complex128)
        for first_index, first in enumerate(bases[0]):
            for second_index, second in enumerate(bases[1]):
                for third_index, third in enumerate(bases[2]):
                    tensor[first_index, second_index, third_index] = self.form.value(
                        first, second, third
                    )
        return complex(
            np.einsum(
                "abc,ad,be,cf,def->",
                tensor,
                inverses[0],
                inverses[1],
                inverses[2],
                tensor,
                optimize=True,
            )
        )

    def series(self, maximum_total_level: int) -> dict[Level, complex]:
        cutoff = int(maximum_total_level)
        answer: dict[Level, complex] = {}
        for total in range(cutoff + 1):
            for first in range(total + 1):
                for second in range(total - first + 1):
                    levels = (first, second, total - first - second)
                    answer[levels] = self.coefficient(levels)
        return answer


def _ordinary_convolution(
    left: Mapping[Level, complex],
    right: Mapping[Level, complex],
    *,
    cutoff: int,
) -> dict[Level, complex]:
    answer: dict[Level, complex] = {}
    for left_levels, left_value in left.items():
        for right_levels, right_value in right.items():
            levels = tuple(
                left_levels[edge] + right_levels[edge] for edge in range(3)
            )
            if sum(levels) <= cutoff:
                answer[levels] = answer.get(levels, 0.0j) + left_value * right_value
    return answer


def ns_branch_labels(maximum_total_twice_level: int) -> tuple[Fraction, ...]:
    """Return every NS branch label that can contribute at the cutoff."""

    maximum_twice_label = math.isqrt(int(maximum_total_twice_level))
    return tuple(
        Fraction(integer, 2)
        for integer in range(-maximum_twice_label, maximum_twice_label + 1)
    )


def ramond_branch_labels(maximum_total_twice_level: int) -> tuple[Fraction, ...]:
    cutoff = int(maximum_total_twice_level)
    maximum_odd = math.isqrt(4 * cutoff + 1)
    if maximum_odd % 2 == 0:
        maximum_odd -= 1
    return tuple(
        Fraction(odd, 4)
        for odd in range(-maximum_odd, maximum_odd + 1, 2)
        if odd and (odd * odd - 1) // 4 <= cutoff
    )


def branch_base_levels(n_ns: Fraction, n_r1: Fraction, n_r2: Fraction) -> Level:
    ns = 4 * n_ns * n_ns
    r1 = 4 * n_r1 * n_r1 - Fraction(1, 4)
    r2 = 4 * n_r2 * n_r2 - Fraction(1, 4)
    values = (ns, r1, r2)
    if any(value.denominator != 1 or value < 0 for value in values):
        raise ValueError("invalid NSRR branching lattice")
    return tuple(int(value) for value in values)  # type: ignore[return-value]


class CertifiedRamondBranching:
    """Thin cached adapter to ``ramond_branching_recursion/compute_target.py``."""

    def __init__(self, *, b: sp.Expr, momenta: Sequence[sp.Expr]):
        if len(momenta) != 3:
            raise ValueError("the branching recursion needs three momenta")
        self.b_exact = sp.sympify(b)
        self.momenta_exact = tuple(sp.sympify(value) for value in momenta)
        self.b = BRANCHING_RECURSION.real_number(float(b))
        self.momenta = tuple(
            BRANCHING_RECURSION.real_number(float(value)) for value in momenta
        )

    @lru_cache(maxsize=None)
    def table(
        self, n_ns: Fraction, n_r1: Fraction, n_r2: Fraction
    ) -> dict[tuple[int, int, int], complex]:
        target = BRANCHING_RECURSION.validate_target((n_ns, n_r1, n_r2))
        labels_ns = BRANCHING_RECURSION.ns_label_closure(target[0])
        labels_r1 = BRANCHING_RECURSION.ramond_label_closure(target[1])
        labels_r2 = BRANCHING_RECURSION.ramond_label_closure(target[2])
        module_ns = BRANCHING_RECURSION.FreeFieldModule(
            "NS", self.b, self.momenta[0]
        )
        module_r1 = BRANCHING_RECURSION.FreeFieldModule(
            "R", self.b, self.momenta[1]
        )
        module_r2 = BRANCHING_RECURSION.FreeFieldModule(
            "R", self.b, self.momenta[2]
        )
        ns_actions = {Fraction(0): []}
        for label in labels_ns[1:]:
            ns_actions[label] = BRANCHING_RECURSION.solve_ns_l1(
                module_ns, int(label)
            )[0]
        weights = BRANCHING_RECURSION.BranchWeights(self.b, self.momenta)
        answer: dict[tuple[int, int, int], complex] = {}
        for parity_r1 in (0, 1):
            actions_r1 = {
                label: BRANCHING_RECURSION.solve_ramond_lminus(
                    module_r1, label, parity_r1
                )[0]
                for label in labels_r1
            }
            for parity_r2 in (0, 1):
                actions_r2 = {
                    label: BRANCHING_RECURSION.solve_ramond_lminus(
                        module_r2, label, parity_r2
                    )[0]
                    for label in labels_r2
                }
                recursion = BRANCHING_RECURSION.BranchingRecursion(
                    self.b,
                    self.momenta,
                    weights,
                    ns_actions,
                    actions_r1,
                    actions_r2,
                )
                expansion = recursion.expansion(target)
                for eta in (1, -1):
                    ward = BRANCHING_RECURSION.finite_ward_solution(
                        weights,
                        ns_actions,
                        actions_r1,
                        actions_r2,
                        labels_ns,
                        labels_r1,
                        labels_r2,
                        module_r1,
                        module_r2,
                        parity_r1,
                        parity_r2,
                        eta,
                    )
                    ward_index = {
                        labels: position
                        for position, labels in enumerate(ward["unknowns"])
                    }
                    boundary = {
                        labels: ward["values"][ward_index[labels]]
                        / BRANCHING_RECURSION.norm_product(
                            labels,
                            parity_r1,
                            parity_r2,
                            self.b,
                            self.momenta,
                        )
                        for labels in expansion
                    }
                    value = sum(
                        coefficient * boundary[labels]
                        for labels, coefficient in expansion.items()
                    )
                    answer[(parity_r1, parity_r2, eta)] = complex(value)
        return answer

    def product(
        self,
        *,
        labels: tuple[Fraction, Fraction, Fraction],
        parities: tuple[int, int],
        etas: tuple[int, int],
    ) -> complex:
        table = self.table(*labels)
        left = table[(parities[0], parities[1], etas[0])]
        right = table[(parities[0], parities[1], etas[1])]
        # ``compute_target`` keeps the literal ell(x,4n) continuation on a
        # negative Ramond recursion boundary.  The block resolution in the
        # Human Note instead defines v_n(P)=v_-n(-P).  Convert the normalized
        # B product (two three-forms) by backend_norm/reflected_norm on each
        # Ramond edge.  Positive labels have conversion one.
        normalization_conversion = 1.0 + 0.0j
        for edge, (label, parity) in enumerate(
            zip(labels[1:], parities), start=1
        ):
            backend_norm = BRANCHING_RECURSION.ramond_norm_squared(
                label, parity, self.b, self.momenta[edge]
            )
            reflected_norm = CHI_BRANCHING.ramond_norm(
                sp.Rational(label.numerator, label.denominator),
                parity,
                self.b_exact,
                self.momenta_exact[edge],
            )
            normalization_conversion *= complex(backend_norm) / complex(
                sp.N(reflected_norm, 60)
            )
        return complex(left * right * normalization_conversion)


def enlarged_double_virasoro_series(
    *,
    b: object,
    momenta: Sequence[object],
    form_parity: int,
    etas: Sequence[int] = (1, 1),
    maximum_total_twice_level: int = 2,
    precision: int = 60,
) -> ComponentSeries:
    """Compute the Human-Note enlarged NSRR theta block."""

    cutoff = int(maximum_total_twice_level)
    if cutoff < 0 or cutoff > 2:
        raise ValueError(
            "the low-order audit engine supports total twice-level 0 through 2"
        )
    form_parity = int(form_parity)
    if form_parity not in (0, 1):
        raise ValueError("form_parity must be 0 or 1")
    etas = tuple(int(value) for value in etas)
    if len(etas) != 2 or any(value not in (-1, 1) for value in etas):
        raise ValueError("etas must contain two Human-Note form signs")
    if len(momenta) != 3:
        raise ValueError("momenta must contain the NS, R1, and R2 momenta")
    exact = (_exact_real(b), *(_exact_real(value) for value in momenta))
    b_exact, p_ns, p_r1, p_r2 = exact
    branching = CertifiedRamondBranching(
        b=b_exact, momenta=(p_ns, p_r1, p_r2)
    )
    half_ns_anchor = DirectHalfNSAnchor(
        b=b_exact, momenta=(p_ns, p_r1, p_r2)
    )
    b_numeric = _complex_number(b_exact, precision)
    momenta_numeric = tuple(_complex_number(value, precision) for value in exact[1:])

    answer: ComponentSeries = {}
    block_cache: dict[
        tuple[complex, tuple[complex, complex, complex], int], dict[Level, complex]
    ] = {}

    def virasoro_series(
        central_charge: complex, weights: Sequence[complex], descendant_cutoff: int
    ) -> dict[Level, complex]:
        key = (
            complex(central_charge),
            tuple(complex(value) for value in weights),
            int(descendant_cutoff),
        )
        if key not in block_cache:
            block_cache[key] = OrdinaryVirasoroThetaBlock(
                central_charge=key[0], weights=key[1]
            ).series(key[2])
        return block_cache[key]

    for n_ns in ns_branch_labels(cutoff):
        for n_r1 in ramond_branch_labels(cutoff):
            for n_r2 in ramond_branch_labels(cutoff):
                base = branch_base_levels(n_ns, n_r1, n_r2)
                base_total = sum(base)
                if base_total > cutoff:
                    continue
                ns_parity = int(2 * n_ns) % 2
                descendant_cutoff = (cutoff - base_total) // 2

                copy_central_charges: list[complex] = []
                copy_weights: list[list[complex]] = [[], []]
                for momentum, label in zip(
                    momenta_numeric, (n_ns, n_r1, n_r2)
                ):
                    parameters = two_virasoro_parameters(
                        momentum=momentum, branch_label=label, b=b_numeric
                    )
                    if not copy_central_charges:
                        copy_central_charges.extend(
                            (parameters[0][0], parameters[1][0])
                        )
                    copy_weights[0].append(parameters[0][1])
                    copy_weights[1].append(parameters[1][1])
                first = virasoro_series(
                    copy_central_charges[0], copy_weights[0], descendant_cutoff
                )
                second = virasoro_series(
                    copy_central_charges[1], copy_weights[1], descendant_cutoff
                )
                ordinary_product = _ordinary_convolution(
                    first, second, cutoff=descendant_cutoff
                )

                for parity_r1 in (0, 1):
                    for parity_r2 in (0, 1):
                        if (ns_parity + parity_r1 + parity_r2) % 2 != form_parity:
                            continue
                        component = ns_parity | (parity_r1 << 1) | (parity_r2 << 2)
                        if n_ns.denominator == 1:
                            branch = branching.product(
                                labels=(n_ns, n_r1, n_r2),
                                parities=(parity_r1, parity_r2),
                                etas=etas,  # type: ignore[arg-type]
                            )
                        else:
                            branch = _complex_number(
                                half_ns_anchor.branching_product(
                                    labels=(n_ns, n_r1, n_r2),
                                    parities=(parity_r1, parity_r2),
                                    form_parity=form_parity,
                                    etas=etas,
                                ),
                                precision,
                            )
                        branch *= theta_quadratic_sign(component)
                        for descendant, coefficient in ordinary_product.items():
                            levels = tuple(
                                base[edge] + 2 * descendant[edge]
                                for edge in range(3)
                            )
                            if sum(levels) <= cutoff:
                                _put_component(
                                    answer,
                                    levels,  # type: ignore[arg-type]
                                    component,
                                    branch * coefficient,
                                )
    return answer


def _distinct_mode_sets(total: int, allowed: Sequence[int]) -> tuple[tuple[int, ...], ...]:
    answer: list[tuple[int, ...]] = []
    for count in range(len(allowed) + 1):
        for subset in combinations(allowed, count):
            if sum(subset) == total:
                answer.append(tuple(sorted(subset, reverse=True)))
    return tuple(answer)


@lru_cache(maxsize=None)
def ns_fermion_states(twice_level: int) -> tuple[tuple[sp.Rational, ...], ...]:
    target = int(twice_level)
    modes = _distinct_mode_sets(target, tuple(range(1, target + 1, 2)))
    return tuple(
        tuple(sp.Rational(value, 2) for value in state) for state in modes
    )


@lru_cache(maxsize=None)
def ramond_fermion_states(
    level: int,
) -> tuple[tuple[tuple[sp.Integer, ...], int], ...]:
    target = int(level)
    modes = _distinct_mode_sets(target, tuple(range(1, target + 1)))
    return tuple(
        (tuple(sp.Integer(value) for value in state), ground)
        for state in modes
        for ground in (0, 1)
    )


def auxiliary_majorana_nsrr_series(
    *, maximum_total_twice_level: int
) -> ComponentSeries:
    """Sew the auxiliary Majorana NSRR block in the canonical spin frames."""

    answer: ComponentSeries = {}
    for levels in level_triples(maximum_total_twice_level):
        ns_level, r1_twice, r2_twice = levels
        for first_modes in ns_fermion_states(ns_level):
            parity_ns = len(first_modes) % 2
            for second_modes, ground_1 in ramond_fermion_states(r1_twice // 2):
                parity_r1 = (len(second_modes) + ground_1) % 2
                for third_modes, ground_2 in ramond_fermion_states(r2_twice // 2):
                    parity_r2 = (len(third_modes) + ground_2) % 2
                    if (parity_ns + parity_r1 + parity_r2) % 2:
                        continue
                    rho = CHI_BRANCHING.stored.fermion_value(
                        0,
                        first_modes,
                        second_modes,
                        ground_1,
                        third_modes,
                        ground_2,
                    )
                    if rho == 0:
                        continue
                    component = parity_ns | (parity_r1 << 1) | (parity_r2 << 2)
                    value = theta_quadratic_sign(component) * complex(sp.N(rho * rho, 60))
                    _put_component(answer, levels, component, value)
    return answer


class HumanNSRRThetaOracle:
    """Direct PBW theta sewing in the literal Human-Note slot order NS,R,R."""

    def __init__(
        self,
        *,
        central_charge: sp.Expr,
        h_ns: sp.Expr,
        beta_r1: sp.Expr,
        beta_r2: sp.Expr,
        form_parity: int,
        etas: Sequence[int],
    ) -> None:
        self.central_charge = sp.sympify(central_charge)
        self.h_ns = sp.sympify(h_ns)
        self.betas = (sp.sympify(beta_r1), sp.sympify(beta_r2))
        self.r_weights = tuple(
            sp.cancel(self.central_charge / 24 - beta**2) for beta in self.betas
        )
        self.form_parity = int(form_parity)
        self.etas = tuple(int(value) for value in etas)
        self.ns_module = NSVermaModule(
            c=complex(sp.N(self.central_charge, 60)),
            weight=complex(sp.N(self.h_ns, 60)),
        )
        self.r_modules = tuple(
            RamondPBWModule(weight, beta, self.central_charge)
            for weight, beta in zip(self.r_weights, self.betas)
        )
        self.forms = tuple(
            GeneralizedNRRWard(
                p_phi=0,
                form_parity=self.form_parity,
                eta=eta,
                h_ns=self.h_ns,
                h_second=self.r_weights[0],
                h_third=self.r_weights[1],
                beta_second=self.betas[0],
                beta_third=self.betas[1],
                central_charge=self.central_charge,
            )
            for eta in self.etas
        )

    @staticmethod
    def _ns_word(state) -> tuple[tuple[str, sp.Expr], ...]:
        return tuple(
            (kind, sp.Rational(mode, 2)) for kind, mode in state
        )

    @lru_cache(maxsize=None)
    def ns_basis_inverse(self, twice_level: int):
        basis = tuple(self.ns_module.basis(int(twice_level)))
        inverse = np.linalg.inv(
            np.asarray(
                self.ns_module.gram_matrix(int(twice_level)), dtype=np.complex128
            )
        )
        return basis, inverse

    @lru_cache(maxsize=None)
    def r_basis_inverse(self, edge: int, level: int, parity: int):
        basis, gram = self.r_modules[int(edge)].gram_matrix(
            int(level), int(parity)
        )
        inverse = np.linalg.inv(
            np.asarray(gram.evalf(60).tolist(), dtype=np.complex128)
        )
        return tuple(basis), inverse

    @lru_cache(maxsize=None)
    def coefficient_components(
        self, ns_twice_level: int, r1_level: int, r2_level: int
    ) -> ParityVector:
        ns_twice_level = int(ns_twice_level)
        ns_basis, inverse_ns = self.ns_basis_inverse(ns_twice_level)
        ns_parity = ns_twice_level % 2
        answer = [0.0j] * 8
        for parity_r1 in (0, 1):
            parity_r2 = (
                self.form_parity + ns_parity + parity_r1
            ) % 2
            basis_r1, inverse_r1 = self.r_basis_inverse(
                0, int(r1_level), parity_r1
            )
            basis_r2, inverse_r2 = self.r_basis_inverse(
                1, int(r2_level), parity_r2
            )
            tensors = []
            for form in self.forms:
                tensor = np.empty(
                    (len(ns_basis), len(basis_r1), len(basis_r2)),
                    dtype=np.complex128,
                )
                for i, state_ns in enumerate(ns_basis):
                    word_ns = self._ns_word(state_ns)
                    for j, state_r1 in enumerate(basis_r1):
                        for k, state_r2 in enumerate(basis_r2):
                            tensor[i, j, k] = complex(
                                sp.N(
                                    form.value(
                                        word_ns,
                                        state_r1.word,
                                        state_r1.ground,
                                        state_r2.word,
                                        state_r2.ground,
                                    ),
                                    60,
                                )
                            )
                tensors.append(tensor)
            contracted = np.einsum(
                "abc,ad,be,cf,def->",
                tensors[0],
                inverse_ns,
                inverse_r1,
                inverse_r2,
                tensors[1],
                optimize=True,
            )
            component = ns_parity | (parity_r1 << 1) | (parity_r2 << 2)
            # This is the physical SCA block appearing on the right of the
            # Section 8 convolution.  It retains the quadratic theta sewing
            # sign from the Human Note's Ramond-block definition.
            answer[component] = theta_quadratic_sign(component) * contracted
        return tuple(complex(value) for value in answer)


def direct_pbw_nsrr_series(
    *,
    b: object,
    momenta: Sequence[object],
    form_parity: int,
    etas: Sequence[int] = (1, 1),
    maximum_total_twice_level: int = 2,
) -> ComponentSeries:
    """Return the independent finite-level physical NSRR PBW block."""

    if len(momenta) != 3:
        raise ValueError("momenta must contain the NS, R1, and R2 momenta")
    b_exact = _exact_real(b)
    p_ns, p_r1, p_r2 = tuple(_exact_real(value) for value in momenta)
    q_value = sp.cancel(b_exact + 1 / b_exact)
    central_charge = sp.cancel(sp.Rational(3, 2) + 3 * q_value**2)
    h_ns = sp.cancel((q_value**2 / 4 - p_ns**2) / 2)
    oracle = HumanNSRRThetaOracle(
        central_charge=central_charge,
        h_ns=h_ns,
        beta_r1=p_r1 / sp.sqrt(2),
        beta_r2=p_r2 / sp.sqrt(2),
        etas=etas,
        form_parity=int(form_parity),
    )
    return {
        levels: oracle.coefficient_components(
            levels[0], levels[1] // 2, levels[2] // 2
        )
        for levels in level_triples(maximum_total_twice_level)
    }


@dataclass(frozen=True)
class NSRRComparison:
    maximum_total_twice_level: int
    form_parity: int
    etas: tuple[int, int]
    coefficient_count: int
    unsupported_level_triples: tuple[Level, ...]
    maximum_absolute_error: float
    maximum_relative_error: float
    worst_levels: Level
    worst_component: int
    double_virasoro_value: complex
    factorized_pbw_value: complex

    def as_json(self) -> dict[str, object]:
        result = asdict(self)
        for key in ("double_virasoro_value", "factorized_pbw_value"):
            value = complex(result[key])
            result[key] = {"real": value.real, "imaginary": value.imag}
        return result


def compare_nsrr_low_order(
    *,
    b: object = sp.Rational(7, 5),
    momenta: Sequence[object] = (
        sp.Rational(11, 23),
        sp.Rational(13, 29),
        sp.Rational(17, 31),
    ),
    form_parity: int = 0,
    etas: Sequence[int] = (1, 1),
    maximum_total_twice_level: int = 2,
    precision: int = 60,
) -> NSRRComparison:
    """Run the double-Virasoro versus direct Ramond-PBW comparison."""

    etas_tuple = tuple(int(value) for value in etas)
    enlarged = enlarged_double_virasoro_series(
        b=b,
        momenta=momenta,
        form_parity=form_parity,
        etas=etas_tuple,
        maximum_total_twice_level=maximum_total_twice_level,
        precision=precision,
    )
    auxiliary = auxiliary_majorana_nsrr_series(
        maximum_total_twice_level=maximum_total_twice_level
    )
    physical = direct_pbw_nsrr_series(
        b=b,
        momenta=momenta,
        form_parity=form_parity,
        etas=etas_tuple,
        maximum_total_twice_level=maximum_total_twice_level,
    )
    factorized = star_convolve_series(
        auxiliary,
        physical,
        maximum_total_twice_level=maximum_total_twice_level,
    )

    maximum_absolute = 0.0
    maximum_relative = 0.0
    worst_levels: Level = (0, 0, 0)
    worst_component = 0
    worst_values = (0.0j, 0.0j)
    coefficient_count = 0
    unsupported: tuple[Level, ...] = ()
    for levels in level_triples(maximum_total_twice_level):
        if levels in unsupported:
            continue
        left = enlarged.get(levels, ZERO_VECTOR)
        right = factorized.get(levels, ZERO_VECTOR)
        for component in range(8):
            coefficient_count += 1
            absolute = abs(left[component] - right[component])
            relative = absolute / max(1.0, abs(left[component]), abs(right[component]))
            if relative > maximum_relative or (
                relative == maximum_relative and absolute > maximum_absolute
            ):
                maximum_absolute = float(absolute)
                maximum_relative = float(relative)
                worst_levels = levels
                worst_component = component
                worst_values = (complex(left[component]), complex(right[component]))
    return NSRRComparison(
        maximum_total_twice_level=int(maximum_total_twice_level),
        form_parity=int(form_parity),
        etas=etas_tuple,  # type: ignore[arg-type]
        coefficient_count=coefficient_count,
        unsupported_level_triples=unsupported,
        maximum_absolute_error=maximum_absolute,
        maximum_relative_error=maximum_relative,
        worst_levels=worst_levels,
        worst_component=worst_component,
        double_virasoro_value=worst_values[0],
        factorized_pbw_value=worst_values[1],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-total-twice-level", type=int, default=2)
    parser.add_argument("--form-parity", type=int, choices=(0, 1), default=0)
    parser.add_argument("--eta-left", type=int, choices=(-1, 1), default=1)
    parser.add_argument("--eta-right", type=int, choices=(-1, 1), default=1)
    parser.add_argument("--precision", type=int, default=60)
    parser.add_argument("--json", action="store_true")
    arguments = parser.parse_args()
    comparison = compare_nsrr_low_order(
        form_parity=arguments.form_parity,
        etas=(arguments.eta_left, arguments.eta_right),
        maximum_total_twice_level=arguments.max_total_twice_level,
        precision=arguments.precision,
    )
    if arguments.json:
        print(json.dumps(comparison.as_json(), indent=2))
        return
    print("genus-two NSRR double-Virasoro / direct-PBW comparison")
    print(f"  maximum total twice-level: {comparison.maximum_total_twice_level}")
    print(
        "  form parity / Human-Note signs: "
        f"{comparison.form_parity} / {comparison.etas}"
    )
    print(f"  checked parity coefficients: {comparison.coefficient_count}")
    print(f"  unsupported level triples: {comparison.unsupported_level_triples}")
    print(
        "  maximum absolute / relative error: "
        f"{comparison.maximum_absolute_error:.3e} / "
        f"{comparison.maximum_relative_error:.3e}"
    )
    print(
        "  worst (levels, component): "
        f"{comparison.worst_levels}, {comparison.worst_component}"
    )


if __name__ == "__main__":
    main()
