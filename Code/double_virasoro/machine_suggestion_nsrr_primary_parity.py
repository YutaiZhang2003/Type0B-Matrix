#!/usr/bin/env python3
"""MACHINE SUGGESTION -- NOT YET HUMAN-VERIFIED.

Provisional intrinsic-NS-parity lift of the genus-two NS--R--R
double-Virasoro formula in Section 8 of ``Human Notes/SCblock.tex``.

This file is deliberately separate from :mod:`nsrr_genus2_block`.  It tests
the machine-suggested replacements

    2 n_NS -> p_NS + 2 n_NS,

in the theta monomial and quadratic sewing sign.  In particular, a branch
with Ramond-copy parities ``alpha_2, alpha_3`` receives the extra sign

    (-1)**(p_NS * (alpha_2 + alpha_3)).

The parity-dependent branching numerator is not inferred from the even
primary.  It is recomputed directly: free-field chi strings are expanded in
the auxiliary-Majorana x SCA basis and the physical endpoints are evaluated
with the generalized NS--R--R Ward identities using ``p_phi=p_NS``.  This
makes the module a sign audit, not an independent certification of the
branching recursion.

The comparison performed here, through total twice-level four, is

    machine-suggested double Virasoro
        == auxiliary Majorana star direct physical PBW.

Any mismatch is reported.  The direct PBW side also provides a useful
parity-covariance check under ``p_NS: 0 -> 1`` independently of the proposed
double-Virasoro branching sum.  No fitted signs, phases, or normalizations
are introduced to force equality.  Do not use this module as a production
block until the conventions have been reviewed by a human.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from functools import lru_cache
from fractions import Fraction
import json
from typing import Sequence

import numpy as np
import sympy as sp

import nsrr_genus2_block as certified
from half_ns_anchor import CHI, DirectHalfNSAnchor, SQRT2
from ramond_pbw_generalized_ward import GeneralizedNRRWard
from theta_star_algebra import theta_quadratic_sign


MACHINE_SUGGESTION = True
HUMAN_VERIFIED = False
DISCLAIMER = (
    "MACHINE SUGGESTION: intrinsic NS-primary parity signs are provisional "
    "and have not been fully human-verified."
)

Level = certified.Level
ParityVector = certified.ParityVector
ComponentSeries = certified.ComponentSeries
ZERO_VECTOR = certified.ZERO_VECTOR
MAXIMUM_MACHINE_SUGGESTED_TWICE_LEVEL = 4


def _bit(value: int, name: str) -> int:
    value = int(value)
    if value not in (0, 1):
        raise ValueError(f"{name} must be 0 or 1")
    return value


def branch_relative_ns_parity(n_ns: Fraction) -> int:
    """Return ``2*n_NS mod 2`` on the NS branching lattice."""

    twice = 2 * Fraction(n_ns)
    if twice.denominator != 1:
        raise ValueError("n_NS must lie in Z/2")
    return int(twice) % 2


def branch_component(
    *, primary_parity: int, n_ns: Fraction, parity_r1: int, parity_r2: int
) -> int:
    """Return the full theta component ``p_NS+2p_R1+4p_R2``."""

    primary_parity = _bit(primary_parity, "primary_parity")
    parity_r1 = _bit(parity_r1, "parity_r1")
    parity_r2 = _bit(parity_r2, "parity_r2")
    parity_ns = primary_parity ^ branch_relative_ns_parity(n_ns)
    return parity_ns | (parity_r1 << 1) | (parity_r2 << 2)


def proposed_extra_branch_sign(
    primary_parity: int, parity_r1: int, parity_r2: int
) -> int:
    """Return the proposed ``(-1)^(p_NS*(alpha_2+alpha_3))``."""

    primary_parity = _bit(primary_parity, "primary_parity")
    parity_r1 = _bit(parity_r1, "parity_r1")
    parity_r2 = _bit(parity_r2, "parity_r2")
    return (-1) ** (primary_parity * (parity_r1 + parity_r2))


def machine_suggested_level_triples(
    maximum_total_twice_level: int,
):
    """Yield the NSRR level lattice used only by this provisional audit.

    The production checker deliberately stops at total twice-level two.  The
    machine-suggested audit extends the same lattice through twice-level four
    without changing that certified bound.
    """

    cutoff = int(maximum_total_twice_level)
    if cutoff < 0 or cutoff > MAXIMUM_MACHINE_SUGGESTED_TWICE_LEVEL:
        raise ValueError(
            "the provisional audit supports total twice-level 0 through "
            f"{MAXIMUM_MACHINE_SUGGESTED_TWICE_LEVEL}"
        )
    for total in range(cutoff + 1):
        for ns_level in range(total + 1):
            for r1_level in range(0, total - ns_level + 1, 2):
                r2_level = total - ns_level - r1_level
                if r2_level % 2 == 0:
                    yield ns_level, r1_level, r2_level


def machine_suggested_auxiliary_majorana_series(
    *, maximum_total_twice_level: int
) -> ComponentSeries:
    """Extend the unchanged auxiliary Majorana sewing through level four.

    This is a literal extension of the production routine's state sum.  No
    intrinsic-NS-parity hypothesis enters this factor.
    """

    answer: ComponentSeries = {}
    for levels in machine_suggested_level_triples(
        maximum_total_twice_level
    ):
        ns_level, r1_twice, r2_twice = levels
        for first_modes in certified.ns_fermion_states(ns_level):
            parity_ns = len(first_modes) % 2
            for second_modes, ground_1 in certified.ramond_fermion_states(
                r1_twice // 2
            ):
                parity_r1 = (len(second_modes) + ground_1) % 2
                for third_modes, ground_2 in certified.ramond_fermion_states(
                    r2_twice // 2
                ):
                    parity_r2 = (len(third_modes) + ground_2) % 2
                    if (parity_ns + parity_r1 + parity_r2) % 2:
                        continue
                    rho = CHI.stored.fermion_value(
                        0,
                        first_modes,
                        second_modes,
                        ground_1,
                        third_modes,
                        ground_2,
                    )
                    if rho == 0:
                        continue
                    component = (
                        parity_ns | (parity_r1 << 1) | (parity_r2 << 2)
                    )
                    value = theta_quadratic_sign(component) * complex(
                        sp.N(rho * rho, 60)
                    )
                    certified._put_component(
                        answer, levels, component, value
                    )
    return answer


class MachineSuggestedParityAnchor(DirectHalfNSAnchor):
    """Direct free-field/PBW branching numerator for intrinsic parity p_NS.

    This is intentionally an audit backend.  Unlike the production adapter,
    it is used for every low-order NS branch label, including ``n_NS=0``.
    """

    def __init__(
        self,
        *,
        b: object,
        momenta: Sequence[object],
        primary_parity: int,
        primary_in_tensor_koszul: bool = True,
    ) -> None:
        super().__init__(b=b, momenta=momenta)
        self.primary_parity = _bit(primary_parity, "primary_parity")
        self.primary_in_tensor_koszul = bool(primary_in_tensor_koszul)

    @lru_cache(maxsize=None)
    def numerator(
        self,
        n_ns,
        n_r1,
        n_r2,
        parity_r1: int,
        parity_r2: int,
        form_parity: int,
        eta: int,
    ) -> sp.Expr:
        """Return the parity-lifted unsquared branching numerator exactly.

        ``form_parity`` is the relative Human-Note label f.  The generalized
        Ward evaluator instead receives the absolute form parity p_NS+f.
        """

        n_ns, n_r1, n_r2 = self._validate_labels(n_ns, n_r1, n_r2)
        parity_r1 = _bit(parity_r1, "parity_r1")
        parity_r2 = _bit(parity_r2, "parity_r2")
        form_parity = _bit(form_parity, "form_parity")
        eta = int(eta)
        if eta not in (-1, 1):
            raise ValueError("eta must be +1 or -1")

        physical_form = GeneralizedNRRWard(
            p_phi=self.primary_parity,
            form_parity=self.primary_parity ^ form_parity,
            eta=eta,
            h_ns=self.h_ns,
            h_second=self.h_ramond[0],
            h_third=self.h_ramond[1],
            beta_second=self.momenta[1] / SQRT2,
            beta_third=self.momenta[2] / SQRT2,
            central_charge=self.central_charge,
        )
        first = CHI.ns_path_components(n_ns, self.q, self.momenta[0])
        second = CHI.ramond_path_components(
            n_r1, parity_r1, self.q, self.momenta[1]
        )
        third = CHI.ramond_path_components(
            n_r2, parity_r2, self.q, self.momenta[2]
        )

        # Relative branch parity and relative physical-form parity are used
        # here.  The intrinsic p_NS occurs on both sides and cancels.
        auxiliary_form_parity = (
            int(2 * n_ns) + parity_r1 + parity_r2 - form_parity
        ) % 2

        answer = sp.S.Zero
        for auxiliary_1, word_1, coefficient_1 in first:
            physical_parity_1 = (
                self.primary_parity * self.primary_in_tensor_koszul
                + CHI.stored.state_parity(word_1)
            ) % 2
            for (
                auxiliary_2,
                auxiliary_ground_2,
                word_2,
                physical_ground_2,
                coefficient_2,
            ) in second:
                physical_parity_2 = CHI.stored.state_parity(
                    word_2, physical_ground_2
                )
                auxiliary_parity_2 = (
                    len(auxiliary_2) + auxiliary_ground_2
                ) % 2
                for (
                    auxiliary_3,
                    auxiliary_ground_3,
                    word_3,
                    physical_ground_3,
                    coefficient_3,
                ) in third:
                    auxiliary_parity_3 = (
                        len(auxiliary_3) + auxiliary_ground_3
                    ) % 2
                    auxiliary_value = CHI.stored.fermion_value(
                        auxiliary_form_parity,
                        auxiliary_1,
                        auxiliary_2,
                        auxiliary_ground_2,
                        auxiliary_3,
                        auxiliary_ground_3,
                    )
                    if auxiliary_value == 0:
                        continue
                    physical_value = physical_form.value(
                        word_1,
                        word_2,
                        physical_ground_2,
                        word_3,
                        physical_ground_3,
                    )
                    if physical_value == 0:
                        continue
                    hatted_form_sign = (-1) ** (
                        physical_parity_1 * (len(auxiliary_1) % 2)
                        + physical_parity_2 * auxiliary_parity_3
                        + (self.primary_parity ^ form_parity)
                        * auxiliary_parity_3
                    )
                    answer += (
                        hatted_form_sign
                        * coefficient_1
                        * coefficient_2
                        * coefficient_3
                        * auxiliary_value
                        * physical_value
                    )
        return sp.factor(
            sp.cancel(
                CHI.ns_v_scale(n_ns, self.b, self.momenta[0]) * answer
            )
        )


class MachineSuggestedPBWOracle(certified.HumanNSRRThetaOracle):
    """Direct physical NS--R--R PBW sewing with intrinsic NS parity."""

    def __init__(
        self,
        *,
        central_charge: sp.Expr,
        h_ns: sp.Expr,
        beta_r1: sp.Expr,
        beta_r2: sp.Expr,
        relative_form_parity: int,
        etas: Sequence[int],
        primary_parity: int,
    ) -> None:
        relative_form_parity = _bit(
            relative_form_parity, "relative_form_parity"
        )
        primary_parity = _bit(primary_parity, "primary_parity")
        super().__init__(
            central_charge=central_charge,
            h_ns=h_ns,
            beta_r1=beta_r1,
            beta_r2=beta_r2,
            form_parity=relative_form_parity,
            etas=etas,
        )
        self.primary_parity = primary_parity
        # Keep self.form_parity as the relative label used by the block sum.
        self.forms = tuple(
            GeneralizedNRRWard(
                p_phi=primary_parity,
                form_parity=primary_parity ^ relative_form_parity,
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

    @lru_cache(maxsize=None)
    def coefficient_components(
        self, ns_twice_level: int, r1_level: int, r2_level: int
    ) -> ParityVector:
        ns_twice_level = int(ns_twice_level)
        ns_basis, inverse_ns = self.ns_basis_inverse(ns_twice_level)
        relative_ns_parity = ns_twice_level % 2
        full_ns_parity = self.primary_parity ^ relative_ns_parity
        answer = [0.0j] * 8
        for parity_r1 in (0, 1):
            parity_r2 = (
                self.form_parity + relative_ns_parity + parity_r1
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
            component = full_ns_parity | (parity_r1 << 1) | (parity_r2 << 2)
            # Keep the complete Human-Note quadratic sewing sign.  Since the
            # component contains the intrinsic NS parity, its ratio to the
            # p_NS=0 sign is automatically the proposed parity lift.
            answer[component] = theta_quadratic_sign(component) * contracted
        return tuple(complex(value) for value in answer)


def machine_suggested_direct_pbw_series(
    *,
    b: object,
    momenta: Sequence[object],
    form_parity: int,
    primary_parity: int,
    etas: Sequence[int] = (1, 1),
    maximum_total_twice_level: int = 2,
) -> ComponentSeries:
    """Return the provisional parity-aware direct physical PBW series."""

    if len(momenta) != 3:
        raise ValueError("momenta must contain the NS, R1, and R2 momenta")
    primary_parity = _bit(primary_parity, "primary_parity")
    b_exact = certified._exact_real(b)
    p_ns, p_r1, p_r2 = tuple(
        certified._exact_real(value) for value in momenta
    )
    q_value = sp.cancel(b_exact + 1 / b_exact)
    central_charge = sp.cancel(sp.Rational(3, 2) + 3 * q_value**2)
    h_ns = sp.cancel((q_value**2 / 4 - p_ns**2) / 2)
    oracle = MachineSuggestedPBWOracle(
        central_charge=central_charge,
        h_ns=h_ns,
        beta_r1=p_r1 / sp.sqrt(2),
        beta_r2=p_r2 / sp.sqrt(2),
        etas=etas,
        relative_form_parity=int(form_parity),
        primary_parity=primary_parity,
    )
    return {
        levels: oracle.coefficient_components(
            levels[0], levels[1] // 2, levels[2] // 2
        )
        for levels in machine_suggested_level_triples(
            maximum_total_twice_level
        )
    }


def machine_suggested_double_virasoro_series(
    *,
    b: object,
    momenta: Sequence[object],
    form_parity: int,
    primary_parity: int,
    etas: Sequence[int] = (1, 1),
    maximum_total_twice_level: int = 2,
    precision: int = 60,
    primary_in_tensor_koszul: bool = True,
) -> ComponentSeries:
    """Evaluate the provisional parity-lifted Section 8 formula.

    Every low-order branching product is supplied by
    :class:`MachineSuggestedParityAnchor`; this isolates the proposed final
    sewing signs from the still-unverified generic-parity branching recursion.

    ``primary_in_tensor_koszul=False`` is a negative-control convention: it
    omits the intrinsic NS parity when regrouping auxiliary and physical
    tensor factors.  It is exposed only so the two possible placements of
    that sign can be compared against direct PBW data.
    """

    cutoff = int(maximum_total_twice_level)
    if cutoff < 0 or cutoff > MAXIMUM_MACHINE_SUGGESTED_TWICE_LEVEL:
        raise ValueError(
            "the provisional audit supports total twice-level 0 through "
            f"{MAXIMUM_MACHINE_SUGGESTED_TWICE_LEVEL}"
        )
    form_parity = _bit(form_parity, "form_parity")
    primary_parity = _bit(primary_parity, "primary_parity")
    etas = tuple(int(value) for value in etas)
    if len(etas) != 2 or any(value not in (-1, 1) for value in etas):
        raise ValueError("etas must contain two Human-Note form signs")
    if len(momenta) != 3:
        raise ValueError("momenta must contain the NS, R1, and R2 momenta")

    exact = (certified._exact_real(b),) + tuple(
        certified._exact_real(value) for value in momenta
    )
    b_exact, p_ns, p_r1, p_r2 = exact
    branching = MachineSuggestedParityAnchor(
        b=b_exact,
        momenta=(p_ns, p_r1, p_r2),
        primary_parity=primary_parity,
        primary_in_tensor_koszul=primary_in_tensor_koszul,
    )
    b_numeric = certified._complex_number(b_exact, precision)
    momenta_numeric = tuple(
        certified._complex_number(value, precision) for value in exact[1:]
    )
    answer: ComponentSeries = {}
    block_cache: dict[
        tuple[complex, tuple[complex, complex, complex], int],
        dict[Level, complex],
    ] = {}

    def virasoro_series(
        central_charge: complex,
        weights: Sequence[complex],
        descendant_cutoff: int,
    ) -> dict[Level, complex]:
        key = (
            complex(central_charge),
            tuple(complex(value) for value in weights),
            int(descendant_cutoff),
        )
        if key not in block_cache:
            block_cache[key] = certified.OrdinaryVirasoroThetaBlock(
                central_charge=key[0], weights=key[1]
            ).series(key[2])
        return block_cache[key]

    for n_ns in certified.ns_branch_labels(cutoff):
        for n_r1 in certified.ramond_branch_labels(cutoff):
            for n_r2 in certified.ramond_branch_labels(cutoff):
                base = certified.branch_base_levels(n_ns, n_r1, n_r2)
                base_total = sum(base)
                if base_total > cutoff:
                    continue
                relative_ns_parity = branch_relative_ns_parity(n_ns)
                descendant_cutoff = (cutoff - base_total) // 2

                copy_central_charges: list[complex] = []
                copy_weights: list[list[complex]] = [[], []]
                for momentum, label in zip(
                    momenta_numeric, (n_ns, n_r1, n_r2)
                ):
                    parameters = certified.two_virasoro_parameters(
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
                ordinary_product = certified._ordinary_convolution(
                    first, second, cutoff=descendant_cutoff
                )

                for parity_r1 in (0, 1):
                    for parity_r2 in (0, 1):
                        if (
                            relative_ns_parity + parity_r1 + parity_r2
                        ) % 2 != form_parity:
                            continue
                        component = branch_component(
                            primary_parity=primary_parity,
                            n_ns=n_ns,
                            parity_r1=parity_r1,
                            parity_r2=parity_r2,
                        )
                        branch = certified._complex_number(
                            branching.branching_product(
                                labels=(n_ns, n_r1, n_r2),
                                parities=(parity_r1, parity_r2),
                                form_parity=form_parity,
                                etas=etas,
                            ),
                            precision,
                        )
                        # Q(p_NS+2n_NS, alpha_2, alpha_3).  Relative to
                        # p_NS=0 this contains the proposed extra sign.
                        branch *= theta_quadratic_sign(component)
                        for descendant, coefficient in ordinary_product.items():
                            levels = tuple(
                                base[edge] + 2 * descendant[edge]
                                for edge in range(3)
                            )
                            if sum(levels) <= cutoff:
                                certified._put_component(
                                    answer,
                                    levels,  # type: ignore[arg-type]
                                    component,
                                    branch * coefficient,
                                )
    return answer


@dataclass(frozen=True)
class MachineSuggestedComparison:
    disclaimer: str
    primary_parity: int
    form_parity: int
    etas: tuple[int, int]
    maximum_total_twice_level: int
    coefficient_count: int
    maximum_absolute_error: float
    maximum_relative_error: float
    worst_levels: Level
    worst_component: int
    worst_relative_levels: Level
    worst_relative_component: int
    double_virasoro_value: complex
    factorized_pbw_value: complex

    def as_json(self) -> dict[str, object]:
        result = asdict(self)
        for key in ("double_virasoro_value", "factorized_pbw_value"):
            value = complex(result[key])
            result[key] = {"real": value.real, "imaginary": value.imag}
        return result


def compare_machine_suggestion_to_direct_pbw(
    *,
    b: object = sp.Rational(7, 5),
    momenta: Sequence[object] = (
        sp.Rational(11, 23),
        sp.Rational(13, 29),
        sp.Rational(17, 31),
    ),
    form_parity: int = 0,
    primary_parity: int = 1,
    etas: Sequence[int] = (1, 1),
    maximum_total_twice_level: int = 2,
    precision: int = 60,
) -> MachineSuggestedComparison:
    """Compare the proposed formula with direct physical PBW sewing."""

    etas_tuple = tuple(int(value) for value in etas)
    double_virasoro = machine_suggested_double_virasoro_series(
        b=b,
        momenta=momenta,
        form_parity=form_parity,
        primary_parity=primary_parity,
        etas=etas_tuple,
        maximum_total_twice_level=maximum_total_twice_level,
        precision=precision,
    )
    auxiliary = machine_suggested_auxiliary_majorana_series(
        maximum_total_twice_level=maximum_total_twice_level
    )
    physical = machine_suggested_direct_pbw_series(
        b=b,
        momenta=momenta,
        form_parity=form_parity,
        primary_parity=primary_parity,
        etas=etas_tuple,
        maximum_total_twice_level=maximum_total_twice_level,
    )
    factorized = certified.star_convolve_series(
        auxiliary,
        physical,
        maximum_total_twice_level=maximum_total_twice_level,
    )

    maximum_absolute = 0.0
    maximum_relative = 0.0
    worst_levels: Level = (0, 0, 0)
    worst_component = 0
    worst_relative_levels: Level = (0, 0, 0)
    worst_relative_component = 0
    worst_values = (0.0j, 0.0j)
    coefficient_count = 0
    for levels in machine_suggested_level_triples(
        maximum_total_twice_level
    ):
        left = double_virasoro.get(levels, ZERO_VECTOR)
        right = factorized.get(levels, ZERO_VECTOR)
        for component in range(8):
            coefficient_count += 1
            absolute = abs(left[component] - right[component])
            relative = absolute / max(
                1.0, abs(left[component]), abs(right[component])
            )
            if absolute > maximum_absolute:
                maximum_absolute = float(absolute)
                worst_levels = levels
                worst_component = component
                worst_values = (
                    complex(left[component]),
                    complex(right[component]),
                )
            if relative > maximum_relative:
                maximum_relative = float(relative)
                worst_relative_levels = levels
                worst_relative_component = component
    return MachineSuggestedComparison(
        disclaimer=DISCLAIMER,
        primary_parity=_bit(primary_parity, "primary_parity"),
        form_parity=_bit(form_parity, "form_parity"),
        etas=etas_tuple,  # type: ignore[arg-type]
        maximum_total_twice_level=int(maximum_total_twice_level),
        coefficient_count=coefficient_count,
        maximum_absolute_error=maximum_absolute,
        maximum_relative_error=maximum_relative,
        worst_levels=worst_levels,
        worst_component=worst_component,
        worst_relative_levels=worst_relative_levels,
        worst_relative_component=worst_relative_component,
        double_virasoro_value=worst_values[0],
        factorized_pbw_value=worst_values[1],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary-parity", type=int, choices=(0, 1), default=1)
    parser.add_argument("--form-parity", type=int, choices=(0, 1), default=0)
    parser.add_argument("--eta-left", type=int, choices=(-1, 1), default=1)
    parser.add_argument("--eta-right", type=int, choices=(-1, 1), default=1)
    parser.add_argument("--max-total-twice-level", type=int, default=2)
    parser.add_argument("--precision", type=int, default=60)
    parser.add_argument("--json", action="store_true")
    arguments = parser.parse_args()
    comparison = compare_machine_suggestion_to_direct_pbw(
        primary_parity=arguments.primary_parity,
        form_parity=arguments.form_parity,
        etas=(arguments.eta_left, arguments.eta_right),
        maximum_total_twice_level=arguments.max_total_twice_level,
        precision=arguments.precision,
    )
    if arguments.json:
        print(json.dumps(comparison.as_json(), indent=2))
        return
    print(DISCLAIMER)
    print("provisional NS-primary-parity double-Virasoro / direct-PBW audit")
    print(f"  primary / relative form parity: {comparison.primary_parity} / {comparison.form_parity}")
    print(f"  Human-Note form signs: {comparison.etas}")
    print(f"  maximum total twice-level: {comparison.maximum_total_twice_level}")
    print(f"  checked parity coefficients: {comparison.coefficient_count}")
    print(
        "  maximum absolute / relative error: "
        f"{comparison.maximum_absolute_error:.3e} / "
        f"{comparison.maximum_relative_error:.3e}"
    )
    print(
        "  worst absolute (levels, component): "
        f"{comparison.worst_levels}, {comparison.worst_component}"
    )
    print(
        "  worst relative (levels, component): "
        f"{comparison.worst_relative_levels}, "
        f"{comparison.worst_relative_component}"
    )


if __name__ == "__main__":
    main()
