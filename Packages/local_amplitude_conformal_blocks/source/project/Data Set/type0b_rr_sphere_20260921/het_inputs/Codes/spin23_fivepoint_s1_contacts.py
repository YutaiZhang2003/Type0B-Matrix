#!/usr/bin/env python3
r"""External top-component contacts at the first five-point resonance.

This module evaluates the explicit component-action contact branch of a
zero-picture singlet.  It is intentionally separate from the screened KLT
continuation: the two are alternative local prescriptions until their overlap
is matched, and blindly adding them could double count a collision stratum.

At ``b=1`` the interacting top component is

``W_a = a**2 psi psibar exp(a phi) - 2 i pi mu a exp((a+1) phi)``.

After stripping the same ``-2 i mu`` Yukawa-wall coefficient as the screened
representative, a contact on leg ``j`` therefore has coefficient ``pi*a_j``.
The contacted charge is shifted by one and the contacted leg is omitted from
both fermion Pfaffians.  All three physical PCO magnitudes are retained, so
the final KLT value carries ``1/8``.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
import math
from typing import Callable, Mapping, Sequence

from spin23_factored_ibp import (
    FactoredPowerSum,
    factored_meromorphic_cube_integral,
    hepp_factored_term,
)
from spin23_fivepoint_resonance_channels import (
    FivePointChannelProjection,
    resonant_worldsheet_states,
)
from spin23_linear_dilaton_fivepoint import FermionMatchingTerm, liouville_charge
from spin23_twisted_periods import klt_bilinear, sine_momentum_kernel


GapDivisor = tuple[int, ...]
Word = tuple[int, int]
WORDS: tuple[Word, ...] = ((2, 3), (3, 2))
ZERO_PICTURE_WORLD_LEGS = (2, 3, 4)
DEFAULT_CONTACT_TWIST_DIRECTION: dict[tuple[int, int], complex] = {
    pair: complex(0.173 * (index + 1), 0.061 * (index + 2))
    for index, pair in enumerate(
        ((1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4))
    )
}


@dataclass(frozen=True)
class ContactSimplexTerm:
    coefficient: complex
    powers: Mapping[GapDivisor, complex]

    def evaluate_gaps(self, gaps: Sequence[float]) -> complex:
        """Evaluate the ordered-simplex integrand at positive unit-sum gaps."""

        normalized = tuple(float(value) for value in gaps)
        if len(normalized) != 3 or any(value <= 0 for value in normalized):
            raise ValueError("three positive ordered-simplex gaps are required")
        if abs(sum(normalized) - 1.0) > 1.0e-10:
            raise ValueError("simplex gaps must sum to one")
        result = complex(self.coefficient)
        for divisor, power in self.powers.items():
            result *= sum(normalized[index] for index in divisor) ** complex(power)
        return result


@dataclass(frozen=True)
class ContactPeriod:
    value: complex
    absolute_error: float
    quadrature_calls: int
    converged: bool


@dataclass(frozen=True)
class ContactBranchResult:
    contact_world_leg: int
    original_charge: complex
    raw_klt_value: complex
    pco_normalized_integral: complex
    wall_stripped_value: complex
    absolute_error_bound: float
    periods_converged: bool
    quadrature_calls: int


@dataclass(frozen=True)
class ContactProjectionResult:
    projection_label: str
    twist_regulator: complex
    branches: tuple[ContactBranchResult, ...]
    wall_stripped_value: complex
    absolute_error_bound: float
    periods_converged: bool
    quadrature_calls: int


def _add_power(
    powers: dict[GapDivisor, complex],
    divisor: Sequence[int],
    value: complex,
) -> None:
    key = tuple(sorted({int(index) for index in divisor}))
    if not key or any(index not in range(3) for index in key):
        raise ValueError("a contact divisor must use gap indices 0,1,2")
    powers[key] = powers.get(key, 0.0j) + complex(value)
    if abs(powers[key]) < 1.0e-15:
        del powers[key]


def _between(first: int, second: int) -> GapDivisor:
    low, high = sorted((int(first), int(second)))
    if low == high:
        raise ValueError("distinct punctures cannot occupy one position")
    return tuple(range(low, high))


def _kappa(
    exponents: Mapping[tuple[int, int], complex], first: int, second: int
) -> complex:
    return complex(exponents[tuple(sorted((int(first), int(second))))])


def _matching_terms(
    ordered_labels: tuple[int, ...],
    contraction: Callable[[int, int], complex],
) -> tuple[FermionMatchingTerm, ...]:
    if not ordered_labels:
        return (FermionMatchingTerm(1.0 + 0.0j, ()),)
    if len(ordered_labels) % 2:
        return ()
    first = ordered_labels[0]
    result: list[FermionMatchingTerm] = []
    for offset, second in enumerate(ordered_labels[1:], start=1):
        pair = complex(contraction(first, second))
        if pair == 0:
            continue
        rest = ordered_labels[1:offset] + ordered_labels[offset + 1 :]
        sign = -1 if (offset + 1) % 2 else 1
        for subterm in _matching_terms(rest, contraction):
            result.append(
                FermionMatchingTerm(
                    sign * pair * subterm.coefficient,
                    ((first, second),) + subterm.edges,
                )
            )
    return tuple(result)


def contact_twist_exponents(
    projection: FivePointChannelProjection,
    outgoing_momenta: Sequence[complex],
    contact_world_leg: int,
    twist_regulator: complex = 0.0j,
    *,
    direction: Mapping[
        tuple[int, int], complex
    ] = DEFAULT_CONTACT_TWIST_DIRECTION,
) -> dict[tuple[int, int], complex]:
    """Return finite-puncture powers after one external charge shift."""

    states = resonant_worldsheet_states(projection, outgoing_momenta)
    contact = int(contact_world_leg)
    if contact not in ZERO_PICTURE_WORLD_LEGS:
        raise ValueError("a contact leg must be one of the three zero-picture legs")
    if states[contact - 1].kind != "singlet":
        raise ValueError("only a zero-picture singlet has this contact branch")
    charges = {
        label: liouville_charge(states[label - 1].liouville_momentum)
        for label in range(1, 6)
    }
    charges[contact] += 1.0
    times = {
        label: complex(states[label - 1].time_momentum)
        for label in range(1, 6)
    }
    if abs(sum(charges.values()) - 2.0) > 1.0e-10:
        raise AssertionError("the contacted five-point exponentials are not neutral")
    result = {
        (first, second): -(
            charges[first] * charges[second] + times[first] * times[second]
        )
        for first in range(1, 5)
        for second in range(first + 1, 5)
    }
    epsilon = complex(twist_regulator)
    for raw_pair, coefficient in direction.items():
        pair = tuple(sorted((int(raw_pair[0]), int(raw_pair[1]))))
        if pair not in result:
            raise ValueError(f"unknown finite contact-twist pair {pair!r}")
        result[pair] += epsilon * complex(coefficient)
    return result


def _contact_matchings(
    projection: FivePointChannelProjection,
    outgoing_momenta: Sequence[complex],
    contact_world_leg: int,
) -> tuple[tuple[FermionMatchingTerm, ...], tuple[FermionMatchingTerm, ...]]:
    states = resonant_worldsheet_states(projection, outgoing_momenta)
    contact = int(contact_world_leg)
    charges = {
        label: liouville_charge(states[label - 1].liouville_momentum)
        for label in range(1, 6)
    }
    times = {
        label: complex(states[label - 1].time_momentum)
        for label in range(1, 6)
    }

    def holomorphic(first: int, second: int) -> complex:
        return charges[first] * charges[second] + times[first] * times[second]

    def antiholomorphic(first: int, second: int) -> complex:
        first_state = states[first - 1]
        second_state = states[second - 1]
        if first_state.kind != second_state.kind:
            return 0.0j
        if first_state.kind == "singlet":
            return charges[first] * charges[second]
        return 1.0 if first_state.flavor == second_state.flavor else 0.0j

    holomorphic_order = tuple(
        label
        for label in reversed(ZERO_PICTURE_WORLD_LEGS)
        if label != contact
    )
    antiholomorphic_order = tuple(
        label for label in (5, 4, 3, 2, 1) if label != contact
    )
    return (
        _matching_terms(holomorphic_order, holomorphic),
        _matching_terms(antiholomorphic_order, antiholomorphic),
    )


def _positive_terms(
    exponents: Mapping[tuple[int, int], complex],
    word: Word,
    matchings: Sequence[FermionMatchingTerm],
) -> tuple[ContactSimplexTerm, ...]:
    finite_order = (1,) + tuple(word) + (4,)
    positions = {label: index for index, label in enumerate(finite_order)}
    common: dict[GapDivisor, complex] = {}
    for offset, first in enumerate(finite_order):
        for second in finite_order[offset + 1 :]:
            _add_power(
                common,
                _between(positions[first], positions[second]),
                _kappa(exponents, first, second),
            )
    result = []
    for matching in matchings:
        powers = dict(common)
        coefficient = complex(matching.coefficient)
        for first, second in matching.edges:
            if first == 5 or second == 5:
                coefficient *= 1 if first == 5 else -1
                continue
            if positions[first] < positions[second]:
                coefficient = -coefficient
            _add_power(
                powers,
                _between(positions[first], positions[second]),
                -1,
            )
        result.append(ContactSimplexTerm(coefficient, powers))
    return tuple(result)


def _negative_geometry(word: Word) -> tuple[dict[int, int], dict[int, int]]:
    q_order = (1, word[1], word[0], 0)
    q_positions = {label: index for index, label in enumerate(q_order[:-1])}
    y_ranks = {word[0]: -2, word[1]: -1, 1: 0, 4: 1}
    return q_positions, y_ranks


def _q_zero(q_positions: Mapping[int, int], label: int) -> GapDivisor:
    return tuple(range(q_positions[label]))


def _q_one(q_positions: Mapping[int, int], label: int) -> GapDivisor:
    return tuple(range(q_positions[label], 3))


def _negative_terms(
    exponents: Mapping[tuple[int, int], complex],
    word: Word,
    matchings: Sequence[FermionMatchingTerm],
) -> tuple[ContactSimplexTerm, ...]:
    q_positions, y_ranks = _negative_geometry(word)
    common: dict[GapDivisor, complex] = {}
    for label in word:
        _add_power(common, _q_zero(q_positions, label), _kappa(exponents, 1, label))
        _add_power(
            common,
            _q_one(q_positions, label),
            -_kappa(exponents, 1, label)
            - _kappa(exponents, label, 4)
            - 2,
        )
    first, second = word
    _add_power(
        common,
        _between(q_positions[first], q_positions[second]),
        _kappa(exponents, first, second),
    )
    _add_power(common, _q_one(q_positions, first), -_kappa(exponents, first, second))
    _add_power(common, _q_one(q_positions, second), -_kappa(exponents, first, second))

    result = []
    moving = set(word)
    for matching in matchings:
        powers = dict(common)
        coefficient = complex(matching.coefficient)
        for first, second in matching.edges:
            if first == 5 or second == 5:
                coefficient *= 1 if first == 5 else -1
                continue
            if y_ranks[first] < y_ranks[second]:
                coefficient = -coefficient
            if first in moving and second in moving:
                _add_power(
                    powers,
                    _between(q_positions[first], q_positions[second]),
                    -1,
                )
                _add_power(powers, _q_one(q_positions, first), 1)
                _add_power(powers, _q_one(q_positions, second), 1)
            elif first in moving or second in moving:
                label = first if first in moving else second
                fixed = second if first in moving else first
                if fixed == 1:
                    _add_power(powers, _q_zero(q_positions, label), -1)
                    _add_power(powers, _q_one(q_positions, label), 1)
                elif fixed == 4:
                    _add_power(powers, _q_one(q_positions, label), 1)
                else:
                    raise AssertionError("unexpected fixed contact-cycle label")
        result.append(ContactSimplexTerm(coefficient, powers))
    return tuple(result)


def _mellin_parameters(
    permutation: Sequence[int], powers: Mapping[GapDivisor, complex]
) -> tuple[complex, complex]:
    pi = tuple(int(value) for value in permutation)
    rank = {gap: index for index, gap in enumerate(pi)}
    parameters: list[complex] = [2.0 + 0.0j, 1.0 + 0.0j]
    for divisor, power in powers.items():
        leading = min(rank[index] for index in divisor)
        for axis in range(2):
            if leading >= axis + 1:
                parameters[axis] += complex(power)
    return tuple(parameters)  # type: ignore[return-value]


def _integrate_terms(
    terms: Sequence[ContactSimplexTerm],
    *,
    rule: str,
    rtol: float,
    atol: float,
) -> ContactPeriod:
    value = 0.0j
    error = 0.0
    calls = 0
    converged = True
    for permutation in permutations(range(3)):
        for term in terms:
            exponents = _mellin_parameters(permutation, term.powers)
            expression = FactoredPowerSum(
                (
                    hepp_factored_term(
                        permutation,
                        term.powers,
                        monomial_shifts=(0, 0),
                        coefficient=term.coefficient,
                    ),
                )
            )
            diagnostic = factored_meromorphic_cube_integral(
                expression,
                exponents,
                rule=rule,
                rtol=rtol,
                atol=atol / 6,
            )
            value += complex(diagnostic.value)
            error += float(diagnostic.absolute_error)
            calls += diagnostic.quadrature_calls
            converged = converged and diagnostic.converged
    return ContactPeriod(value, error, calls, converged)


def evaluate_contact_projection(
    projection: FivePointChannelProjection,
    outgoing_momenta: Sequence[complex],
    *,
    twist_regulator: complex,
    twist_direction: Mapping[
        tuple[int, int], complex
    ] = DEFAULT_CONTACT_TWIST_DIRECTION,
    rule: str = "gauss-jacobi-10",
    rtol: float = 1.0e-6,
    atol: float = 1.0e-9,
) -> ContactProjectionResult:
    """Evaluate every explicit zero-picture-singlet contact branch."""

    states = resonant_worldsheet_states(projection, outgoing_momenta)
    contacts = tuple(
        label
        for label in ZERO_PICTURE_WORLD_LEGS
        if states[label - 1].kind == "singlet"
    )
    branches: list[ContactBranchResult] = []
    for contact in contacts:
        charge = liouville_charge(states[contact - 1].liouville_momentum)
        exponents = contact_twist_exponents(
            projection,
            outgoing_momenta,
            contact,
            twist_regulator,
            direction=twist_direction,
        )
        holomorphic, antiholomorphic = _contact_matchings(
            projection, outgoing_momenta, contact
        )
        left = []
        right = []
        left_errors = []
        right_errors = []
        diagnostics: list[ContactPeriod] = []
        for word in WORDS:
            left_diagnostic = _integrate_terms(
                _positive_terms(exponents, word, holomorphic),
                rule=rule,
                rtol=rtol,
                atol=atol,
            )
            right_diagnostic = _integrate_terms(
                _negative_terms(exponents, word, antiholomorphic),
                rule=rule,
                rtol=rtol,
                atol=atol,
            )
            diagnostics.extend((left_diagnostic, right_diagnostic))
            left.append(left_diagnostic.value)
            right.append(right_diagnostic.value)
            left_errors.append(left_diagnostic.absolute_error)
            right_errors.append(right_diagnostic.absolute_error)

        left_map = dict(zip(WORDS, left))
        right_map = dict(zip(WORDS, right))
        # ``klt_bilinear`` contains the six-point (three-modulus) global
        # sign ``(-1)**3``.  A contacted five-point branch has two complex
        # moduli and hence the raw-area sign ``(-1)**2``.  The extra minus
        # below is fixed independently by the factorized two-beta test.
        raw = -complex(klt_bilinear(left_map, right_map, exponents, root=1))
        propagated = 0.0
        for sigma, left_value, left_error in zip(WORDS, left, left_errors):
            for gamma, right_value, right_error in zip(
                WORDS, right, right_errors
            ):
                kernel = abs(sine_momentum_kernel(gamma, sigma, exponents, root=1))
                propagated += kernel * (
                    abs(right_value) * left_error
                    + abs(left_value) * right_error
                    + left_error * right_error
                )
        pco_integral = raw / 8.0
        wall_value = math.pi * charge * pco_integral
        branches.append(
            ContactBranchResult(
                contact_world_leg=contact,
                original_charge=charge,
                raw_klt_value=raw,
                pco_normalized_integral=pco_integral,
                wall_stripped_value=wall_value,
                absolute_error_bound=math.pi * abs(charge) * propagated / 8.0,
                periods_converged=all(item.converged for item in diagnostics),
                quadrature_calls=sum(item.quadrature_calls for item in diagnostics),
            )
        )
    return ContactProjectionResult(
        projection_label=projection.label,
        twist_regulator=complex(twist_regulator),
        branches=tuple(branches),
        wall_stripped_value=sum(
            (branch.wall_stripped_value for branch in branches), 0.0j
        ),
        absolute_error_bound=sum(
            branch.absolute_error_bound for branch in branches
        ),
        periods_converged=all(branch.periods_converged for branch in branches),
        quadrature_calls=sum(branch.quadrature_calls for branch in branches),
    )


__all__ = [
    "ContactBranchResult",
    "ContactProjectionResult",
    "ContactSimplexTerm",
    "DEFAULT_CONTACT_TWIST_DIRECTION",
    "WORDS",
    "ZERO_PICTURE_WORLD_LEGS",
    "contact_twist_exponents",
    "evaluate_contact_projection",
]
