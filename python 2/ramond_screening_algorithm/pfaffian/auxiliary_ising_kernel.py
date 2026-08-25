"""Ground-resolved *native* Ising NS--R--R kernel.

This module is deliberately independent of the Virasoro Ward evaluator used
by :mod:`python.ramond_three_point_grid.compute_grid`.  It evaluates a Fock
matrix element directly, by coefficient extraction from the two-spin
Majorana kernel and one exact Pfaffian.

The native functional is the free-Majorana two-spin correlator.  The Fock to
Virasoro conversion agrees with it only after the BPZ normalization, the
infinity--middle cocycle, and the middle-chart transport are included.  The
production implementation lives in :mod:`native_spin_kernel` and uses one
ordinary or bordered Pfaffian.

For diagnosis only, this file also contains a corrected version of the old
Virasoro evaluator.  The correction is important: negative Virasoro modes
do not commute, and ``L_0`` acts with the descendant level included.  The
old grid violated both rules.  The executable audit also checks the three
frame/cocycle corrections above on every stored endpoint.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import itertools
import sys

import sympy as sp


ROOT = Path(__file__).resolve().parents[3]
for directory in (
    ROOT / "python 2" / "ramond_three_point_grid",
    ROOT / "python 2" / "ramond_branching_coefficient_check",
):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import compute_grid as grid  # noqa: E402

from .native_spin_kernel import (
    FLIP,
    canonical_ground_matrix,
    canonical_ising_value,
)


I = sp.I
SQRT2 = sp.sqrt(2)


@lru_cache(None)
def canonicalize_virasoro(word):
    """Put ``L_-word`` in descending-partition order, exactly.

    ``word=(a,b,...)`` denotes ``L_-a L_-b ...``.  For an inversion
    ``a<b`` we use

        L_-a L_-b = L_-b L_-a + (b-a) L_-(a+b).

    The returned dictionary retains the commutator term.  Merely sorting
    the tuple is incorrect and is the source of the old mode-two mismatch.
    """

    word = tuple(int(value) for value in word)
    for position in range(len(word) - 1):
        first = word[position]
        second = word[position + 1]
        if first >= second:
            continue
        out = {}
        exchanged = word[:position] + (second, first) + word[position + 2 :]
        for canonical, coefficient in canonicalize_virasoro(exchanged).items():
            grid.add_term(out, canonical, coefficient)
        bracket = word[:position] + (first + second,) + word[position + 2 :]
        for canonical, coefficient in canonicalize_virasoro(bracket).items():
            grid.add_term(out, canonical, (second - first) * coefficient)
        return out
    return {word: sp.Integer(1)}


class CorrectAuxiliaryVirasoroEvaluator(grid.AuxiliaryVirasoroEvaluator):
    """Independent audit oracle with exact negative-mode ordering."""

    @lru_cache(None)
    def act(self, slot, mode, word):
        mode = int(mode)
        word = tuple(int(value) for value in word)
        if mode < 0:
            return canonicalize_virasoro((-mode,) + word)
        if not word:
            if mode == 0:
                return {(): self.weights[slot]}
            return {}

        first = word[0]
        rest = word[1:]
        out = {}

        # L_mode L_-first rest = L_-first L_mode rest + [L_mode,L_-first]rest.
        for reduced, coefficient in self.act(slot, mode, rest).items():
            for canonical, ordering in canonicalize_virasoro((first,) + reduced).items():
                grid.add_term(out, canonical, coefficient * ordering)

        bracket_coefficient = mode + first
        replacement = mode - first
        if replacement < 0:
            acted = canonicalize_virasoro((-replacement,) + rest)
            for reduced, coefficient in acted.items():
                grid.add_term(out, reduced, bracket_coefficient * coefficient)
        elif replacement == 0:
            # L_0 acts on the descendant ``rest``, not on the primary.
            # Omitting ``sum(rest)`` is invisible when the equal mode is
            # rightmost but fails as soon as, e.g., L_2 meets L_-2 L_-1.
            grid.add_term(
                out,
                rest,
                bracket_coefficient * (self.weights[slot] + sum(rest)),
            )
        else:
            for reduced, coefficient in self.act(slot, replacement, rest).items():
                grid.add_term(out, reduced, bracket_coefficient * coefficient)
        if mode == first:
            grid.add_term(
                out,
                rest,
                self.central_charge * (mode**3 - mode) / 12,
            )
        return out


class LegacyAuxiliaryVirasoroEvaluator(grid.AuxiliaryVirasoroEvaluator):
    """Frozen pre-repair evaluator, retained only to quantify old outputs."""

    def base(self):
        """Pre-repair first-primary normalization (including its BPZ bug)."""

        if self.first_primary == 0:
            return grid.boundary.fermion_ground(
                self.form_parity,
                self.second_ground,
                self.third_ground,
            )
        return (
            I
            / SQRT2
            * grid.boundary.fermion_ground(
                1 - self.form_parity,
                self.second_ground,
                self.third_ground,
            )
        )

    @lru_cache(None)
    def act(self, slot, mode, word):
        if mode < 0:
            new_word = tuple(sorted((-int(mode),) + word, reverse=True))
            return {new_word: sp.Integer(1)}
        if not word:
            if mode == 0:
                return {(): self.weights[slot]}
            return {}
        first = word[0]
        rest = word[1:]
        out = {}
        for reduced, coefficient in self.act(slot, mode, rest).items():
            grid.add_term(out, (first,) + reduced, coefficient)
        bracket_coefficient = mode + first
        replacement = mode - first
        if replacement < 0:
            for reduced, coefficient in self.act(slot, replacement, rest).items():
                grid.add_term(out, reduced, bracket_coefficient * coefficient)
        elif replacement == 0:
            grid.add_term(
                out, rest, bracket_coefficient * self.weights[slot]
            )
        else:
            for reduced, coefficient in self.act(slot, replacement, rest).items():
                grid.add_term(out, reduced, bracket_coefficient * coefficient)
        if mode == first:
            grid.add_term(
                out,
                rest,
                self.central_charge * (mode**3 - mode) / 12,
            )
        return out


def _converted_virasoro_value(evaluator_type, data):
    """Evaluate one converted Fock endpoint with ``evaluator_type``."""

    (
        form_parity,
        first_modes,
        second_modes,
        second_ground,
        third_modes,
        third_ground,
    ) = data
    first_primary, basis1, coefficients1 = grid.auxiliary_to_virasoro(
        "NS", first_modes, 0
    )
    second_primary, basis2, coefficients2 = grid.auxiliary_to_virasoro(
        "R", second_modes, second_ground
    )
    third_primary, basis3, coefficients3 = grid.auxiliary_to_virasoro(
        "R", third_modes, third_ground
    )
    evaluator = evaluator_type(
        first_primary, form_parity, second_primary, third_primary
    )
    answer = sp.Integer(0)
    for word1, coefficient1 in zip(basis1, coefficients1):
        for word2, coefficient2 in zip(basis2, coefficients2):
            for word3, coefficient3 in zip(basis3, coefficients3):
                answer += (
                    coefficient1
                    * coefficient2
                    * coefficient3
                    * evaluator.value(word1, word2, word3)
                )
    return sp.factor(sp.cancel(answer))


@lru_cache(None)
def legacy_fermion_value_virasoro(
    form_parity,
    first_modes,
    second_modes,
    second_ground,
    third_modes,
    third_ground,
):
    """Value produced by the pre-repair Virasoro implementation."""

    return _converted_virasoro_value(
        LegacyAuxiliaryVirasoroEvaluator,
        (
            form_parity,
            first_modes,
            second_modes,
            second_ground,
            third_modes,
            third_ground,
        ),
    )


@lru_cache(None)
def corrected_fermion_value_virasoro(
    form_parity,
    first_modes,
    second_modes,
    second_ground,
    third_modes,
    third_ground,
):
    """Correct Virasoro-conversion oracle used only by the audit."""

    return sp.factor(
        grid.auxiliary_virasoro_transport_phase(first_modes, second_modes)
        * _converted_virasoro_value(
        CorrectAuxiliaryVirasoroEvaluator,
        (
            form_parity,
            first_modes,
            second_modes,
            second_ground,
            third_modes,
            third_ground,
        ),
        )
    )


@lru_cache(None)
def auxiliary_ising_value(
    form_parity,
    first_modes,
    second_modes,
    second_ground,
    third_modes,
    third_ground,
):
    """Evaluate the canonical free-Ising form by one exact Pfaffian.

    In the ordered Ramond ground basis the even matrices are

    ``Gamma_0=diag(1,-1)`` and ``Gamma_1=((0,1),(-1,0))``.

    For an odd number of fermions the common ground matrix is
    ``Gamma_(1-f)``.  Relative to its scalar one-fermion coefficient, the
    source phases on ``(infinity, one, zero)`` are ``(+1,-1,+1)``.  These
    signs follow directly from

    ``X Gamma_f=-Gamma_(1-f)`` and
    ``Gamma_f X=+Gamma_(1-f)``,

    where ``X`` is the Ramond zero-mode ground flip.  They are therefore
    ground-resolved OPE data, not a fit to the old ``rho`` table.
    """

    return canonical_ising_value(
        int(form_parity),
        tuple(sp.Rational(mode) for mode in first_modes),
        tuple(sp.Rational(mode) for mode in second_modes),
        int(second_ground),
        tuple(sp.Rational(mode) for mode in third_modes),
        int(third_ground),
    )


def native_odd_source_matrix(form_parity, leg):
    """Ground matrix multiplying the scalar odd source on ``leg``.

    Legs ``1,2,3`` mean ``infinity, one, zero``.  This small public helper
    makes the OPE phase ledger explicit for audits and downstream callers.
    """

    form_parity = int(form_parity)
    leg = int(leg)
    ground = canonical_ground_matrix(form_parity)
    if leg == 1:
        return canonical_ground_matrix(1 - form_parity)
    if leg == 2:
        return FLIP * ground
    if leg == 3:
        return ground * FLIP
    raise ValueError("leg must be 1 (infinity), 2 (one), or 3 (zero)")


def stored_auxiliary_endpoints(sample=None):
    """Return every distinct auxiliary endpoint in the 432 restrictions.

    The lift sign ``eta`` does not enter the auxiliary Majorana factor, so
    its two choices do not create new endpoint keys.  A generic exact sample
    is used only to build the nonzero branch components.
    """

    if sample is None:
        sample = grid.SAMPLES[0]
    b_value, p1, p2, p3 = map(sp.sympify, sample)
    q_value = sp.cancel(b_value + 1 / b_value)
    endpoints = set()
    for n1, n2, n3 in itertools.product(
        grid.GRID_NS_LEVELS, grid.GRID_R_LEVELS, grid.GRID_R_LEVELS
    ):
        first = grid.ns_components(n1, q_value, p1)
        ns_parity = int(2 * sp.Rational(n1)) % 2
        for epsilon2, epsilon3, form_parity in itertools.product(
            (0, 1), (0, 1), (0, 1)
        ):
            second = grid.ramond_components(n2, epsilon2, q_value, p2)
            third = grid.ramond_components(n3, epsilon3, q_value, p3)
            auxiliary_parity = (
                ns_parity + epsilon2 + epsilon3 - form_parity
            ) % 2
            for auxiliary1, _word1, _coefficient1 in first:
                for (
                    auxiliary2,
                    ground2,
                    _word2,
                    _physical2,
                    _coefficient2,
                ) in second:
                    for (
                        auxiliary3,
                        ground3,
                        _word3,
                        _physical3,
                        _coefficient3,
                    ) in third:
                        endpoints.add(
                            (
                                auxiliary_parity,
                                tuple(auxiliary1),
                                tuple(auxiliary2),
                                int(ground2),
                                tuple(auxiliary3),
                                int(ground3),
                            )
                        )
    return tuple(sorted(endpoints, key=repr))


def audit_stored_endpoints(compare_virasoro=True):
    """Audit the native kernel on the complete stored auxiliary corpus.

    The literal recursive Wick expansion is the native comparator.  The
    optional Ward comparison is diagnostic only and reports how many old
    stored-Ward values are changed by repairing its Virasoro algebra.
    """

    endpoints = stored_auxiliary_endpoints()
    native_mismatches = []
    old_ward_changes = []
    production_ward_mismatches = []
    native_ward_differences = []
    for endpoint in endpoints:
        calculated = auxiliary_ising_value(*endpoint)
        expected = grid.fermion_value(*endpoint)
        if sp.simplify(calculated - expected) != 0:
            native_mismatches.append((endpoint, calculated, expected))
        if compare_virasoro:
            old = legacy_fermion_value_virasoro(*endpoint)
            corrected = corrected_fermion_value_virasoro(*endpoint)
            production = grid.fermion_value_virasoro(*endpoint)
            if sp.simplify(old - corrected) != 0:
                old_ward_changes.append((endpoint, old, corrected))
            if sp.simplify(production - corrected) != 0:
                production_ward_mismatches.append(
                    (endpoint, production, corrected)
                )
            if sp.simplify(calculated - corrected) != 0:
                native_ward_differences.append((endpoint, calculated, corrected))
    if native_mismatches:
        raise AssertionError(native_mismatches[0])
    if production_ward_mismatches:
        raise AssertionError(production_ward_mismatches[0])
    return {
        "endpoint_count": len(endpoints),
        "native_mismatch_count": len(native_mismatches),
        "old_ward_change_count": len(old_ward_changes),
        "production_ward_mismatch_count": len(production_ward_mismatches),
        "native_corrected_ward_difference_count": len(native_ward_differences),
        "old_ward_changes": tuple(old_ward_changes),
        "native_corrected_ward_differences": tuple(native_ward_differences),
    }


def _small_audit():
    """Quick algebra, OPE-phase, and channel-separation regressions."""

    assert canonicalize_virasoro((1, 2)) == {
        (2, 1): sp.Integer(1),
        (3,): sp.Integer(1),
    }

    gamma0 = canonical_ground_matrix(0)
    gamma1 = canonical_ground_matrix(1)
    for form_parity, flipped in (
        (0, gamma1),
        (1, gamma0),
    ):
        assert native_odd_source_matrix(form_parity, 1) == flipped
        assert native_odd_source_matrix(form_parity, 2) == -flipped
        assert native_odd_source_matrix(form_parity, 3) == flipped

    # The omitted [L_-1,L_-2] term accounts for the whole old discrepancy.
    old = legacy_fermion_value_virasoro(0, (), (2,), 0, (2,), 0)
    corrected = corrected_fermion_value_virasoro(0, (), (2,), 0, (2,), 0)
    direct = auxiliary_ising_value(0, (), (2,), 0, (2,), 0)
    assert old == -sp.Rational(361, 128)
    assert corrected == -sp.Rational(425, 128)
    assert direct == corrected

    # BPZ normalization at infinity and middle-chart transport now make the
    # Virasoro expansion identical to the direct spin-field OPE.
    assert auxiliary_ising_value(
        0, (sp.Rational(1, 2),), (), 0, (), 1
    ) == -I / SQRT2
    assert corrected_fermion_value_virasoro(
        0, (sp.Rational(1, 2),), (), 0, (), 1
    ) == -I / SQRT2
    assert auxiliary_ising_value(0, (), (2, 1), 0, (), 0) == sp.Rational(1, 32)
    assert corrected_fermion_value_virasoro(
        0, (), (2, 1), 0, (), 0
    ) == sp.Rational(1, 32)

    samples = (
        ((), (2, 1), 0, (), 0),
        ((), (4, 3, 2, 1), 0, (), 0),
        ((sp.Rational(5, 2),), (), 0, (3,), 0),
        ((sp.Rational(1, 2),), (2, 1), 0, (1,), 0),
    )
    for first, second, ground2, third, ground3 in samples:
        for form_parity in (0, 1):
            direct = auxiliary_ising_value(
                form_parity, first, second, ground2, third, ground3
            )
            wick = grid.fermion_value(
                form_parity, first, second, ground2, third, ground3
            )
            if sp.factor(direct - wick) != 0:
                raise AssertionError((first, second, third, direct, wick))
    print("auxiliary Ising kernel: quick native-OPE audit passed")


if __name__ == "__main__":
    _small_audit()
    result = audit_stored_endpoints()
    print(
        "stored auxiliary endpoints: "
        f"{result['endpoint_count']} total, "
        f"{result['native_mismatch_count']} native mismatches, "
        f"{result['old_ward_change_count']} repaired-Ward changes, "
        f"{result['production_ward_mismatch_count']} production-Ward mismatches, "
        f"{result['native_corrected_ward_difference_count']} "
        "native/stored-Ward differences"
    )
