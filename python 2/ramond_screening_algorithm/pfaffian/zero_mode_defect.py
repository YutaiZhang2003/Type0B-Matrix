"""Prototype for the non-natural Ramond copy as an opposite zero-mode defect.

For positive Ramond branch label ``n``, the 2016 staircase of natural parity
contains ``chi^-_0 chi^-_-1 ...``.  The other parity copy appends a rightmost
``chi^+_0``.  This module keeps that extra operator as one marked contour
row.  It is deliberately separate from the validated natural-holonomy code:
the zero-mode-resolved spin frame is still under audit.

The important structural fact is already exact.  Natural--natural and
defect--defect external contractions cancel between the two Majoranas,
whereas natural--defect contractions form a matrix of rank at most two per
defect.  There are at most two defects (one on each Ramond leg), so the
Pfaffian correction has fixed rank at every descendant level.  After the
Vandermonde is removed its Schur width is therefore bounded independently
of the level and is compatible with ``selberg_elementary``.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations

import sympy as sp

from .core import pfaffian
from .screening_pfaffian import (
    ExternalRow,
    I,
    SQRT2,
    _row_one_point,
    _row_pair_with_screening,
    _screen_kernel,
    external_rows,
    natural_ramond_parity,
    vandermonde,
)
from .selberg_jack import (
    _jack_transition,
    monomial_coefficients,
    partitions,
    variables,
)


@dataclass(frozen=True)
class ZeroModeDefect:
    """The rightmost opposite ``chi^+_0`` on one Ramond leg."""

    leg: str
    mode: sp.Rational = sp.Integer(0)
    auxiliary_coefficient: sp.Expr = sp.Integer(1)
    physical_coefficient: sp.Expr = I


def local_ground_action():
    r"""Return ``chi^-_0 chi^+_0 |0,0>`` in the tensor ground basis.

    Keys are ``(auxiliary_ground, physical_ground)``.  Koszul signs are
    included, so this is a direct finite Clifford-algebra check of the
    same-leg contact used below.
    """

    # Apply chi+ first (it is the rightmost operator), then chi-.
    state = {(0, 0): sp.Integer(1)}
    for realization in (1, -1):
        next_state = {}
        for (auxiliary, physical), outer in state.items():
            key = (1 - auxiliary, physical)
            next_state[key] = next_state.get(key, 0) + outer / SQRT2

            # chi^realization = f - i eta in the natural (-) case and
            # f + i eta in the defect (+) case.  eta crosses the auxiliary
            # ground when the latter is odd.
            physical_sign = I if realization == 1 else -I
            coefficient = outer * physical_sign * (-1) ** auxiliary / SQRT2
            key = (auxiliary, 1 - physical)
            next_state[key] = next_state.get(key, 0) + coefficient
        state = next_state
    return {key: sp.simplify(value) for key, value in state.items() if value != 0}


def rows_with_defects(n1, n2, n3, epsilon2, epsilon3):
    """Return radial-order rows with zero or one marked defect per R leg."""

    natural2 = natural_ramond_parity(n2)
    natural3 = natural_ramond_parity(n3)
    epsilon2, epsilon3 = int(epsilon2), int(epsilon3)
    if epsilon2 not in (0, 1) or epsilon3 not in (0, 1):
        raise ValueError((epsilon2, epsilon3))
    infinity, one, zero = external_rows(
        n1, n2, n3, epsilon2=natural2, epsilon3=natural3
    )
    if epsilon2 != natural2:
        one = one + (ZeroModeDefect("one"),)
    if epsilon3 != natural3:
        zero = zero + (ZeroModeDefect("zero"),)
    return infinity, one, zero


def _is_defect(row):
    return isinstance(row, ZeroModeDefect)


def _endpoint(leg):
    if leg == "one":
        return sp.Integer(1)
    if leg == "zero":
        return sp.Integer(0)
    raise ValueError(leg)


def _natural_defect_covariance(left, right):
    """Combined-Majorana covariance for two ordered external rows."""

    if _is_defect(left) == _is_defect(right):
        return sp.Integer(0)
    defect = left if _is_defect(left) else right
    natural = right if _is_defect(left) else left

    if defect.leg == natural.leg:
        # Both creation modes lie on the same ket.  Only the ordered zero
        # modes contract.  Each Majorana contributes 1/2.
        return sp.Integer(1) if natural.mode == 0 else sp.Integer(0)

    endpoint = _endpoint(defect.leg)
    bare = _row_pair_with_screening(natural.leg, natural.mode, endpoint)
    # 1*1 + (-i)*(+i) = 2 for chi^- against chi+.
    return sp.factor(2 * bare)


def _combined_defect_pfaffian(objects, auxiliary_parity, physical_parity, eta):
    """One Pfaffian including the fixed-rank opposite-zero-mode update."""

    objects = tuple(objects)
    size = len(objects)
    covariance = [[sp.Integer(0) for _ in range(size)] for _ in range(size)]
    mean_aux = [sp.Integer(0)] * size
    mean_phys = [sp.Integer(0)] * size

    for index, (kind, item) in enumerate(objects):
        if kind == "external":
            mean_aux[index] = item.auxiliary_coefficient * _row_one_point(
                item.leg, item.mode, "auxiliary", eta
            )
            mean_phys[index] = item.physical_coefficient * _row_one_point(
                item.leg, item.mode, "physical", eta
            )
        else:
            mean_phys[index] = 1 / SQRT2

    for left in range(size):
        kind_left, item_left = objects[left]
        for right in range(left + 1, size):
            kind_right, item_right = objects[right]
            if kind_left == kind_right == "screening":
                value = _screen_kernel(item_left, item_right)
            elif kind_left == "external" and kind_right == "screening":
                value = item_left.physical_coefficient * _row_pair_with_screening(
                    item_left.leg, item_left.mode, item_right
                )
            elif kind_left == "screening" and kind_right == "external":
                value = item_right.physical_coefficient * _row_pair_with_screening(
                    item_right.leg, item_right.mode, item_left
                )
            else:
                value = _natural_defect_covariance(item_left, item_right)
            covariance[left][right] = value
            covariance[right][left] = -value

    auxiliary_parity = int(auxiliary_parity) % 2
    physical_parity = int(physical_parity) % 2
    if auxiliary_parity and physical_parity:
        for left in range(size):
            for right in range(left + 1, size):
                update = mean_aux[left] * mean_phys[right] - mean_phys[left] * mean_aux[right]
                covariance[left][right] += update
                covariance[right][left] -= update
        return pfaffian(covariance)
    if auxiliary_parity or physical_parity:
        border = mean_aux if auxiliary_parity else mean_phys
        augmented = [row + [border[index]] for index, row in enumerate(covariance)]
        augmented.append([-entry for entry in border] + [sp.Integer(0)])
        return pfaffian(augmented)
    return pfaffian(covariance)


def defect_contour_polynomial(
    n1, n2, n3, epsilon2, epsilon3, form_parity, eta, screenings=None
):
    """Return the polynomialized contour integrand with marked defects."""

    n1, n2, n3 = map(sp.Rational, (n1, n2, n3))
    if screenings is None:
        screenings = 2 * (n1 + n2 + n3)
    screenings = sp.Rational(screenings)
    if not screenings.is_integer or screenings < 0:
        raise ValueError(screenings)
    count = int(screenings)
    infinity, one, zero = rows_with_defects(
        n1, n2, n3, epsilon2, epsilon3
    )
    rows = infinity + one + zero
    ts = sp.symbols(f"t0:{count}")
    external_parity = len(rows) % 2
    auxiliary_form = (external_parity - int(form_parity)) % 2
    physical_screening_form = (int(form_parity) + count) % 2
    objects = (
        tuple(("external", row) for row in infinity + one)
        + tuple(("screening", t) for t in ts)
        + tuple(("external", row) for row in zero)
    )
    correlator = _combined_defect_pfaffian(
        objects, auxiliary_form, physical_screening_form, eta
    )
    laurent = sp.factor(sp.cancel(vandermonde(ts) * correlator))
    shift_A = int(2 * n3 - sp.Rational(1, 2))
    shift_B = int(2 * n2 - sp.Rational(1, 2))
    clearing = sp.prod(t**shift_A * (1 - t) ** shift_B for t in ts)
    polynomial = sp.factor(sp.cancel(clearing * laurent))
    _, denominator = sp.fraction(polynomial)
    if set(ts) & denominator.free_symbols:
        raise AssertionError(f"defect Pfaffian did not polynomialize: {denominator}")
    return ts, sp.expand(polynomial), shift_A, shift_B


def schur_coefficients(polynomial, xs):
    """Exact Schur decomposition of a small symmetric test polynomial."""

    xs = tuple(xs)
    monomials = monomial_coefficients(polynomial, xs)
    by_degree = {}
    for partition, coefficient in monomials.items():
        by_degree.setdefault(sum(partition), {})[partition] = coefficient
    answer = {}
    for degree, coefficients in by_degree.items():
        parts, transition = _jack_transition(len(xs), degree, sp.Integer(1))
        vector = sp.Matrix([coefficients.get(partition, 0) for partition in parts])
        schur_vector = transition.inv() * vector
        for partition, coefficient in zip(parts, schur_vector):
            if coefficient:
                answer[partition] = sp.factor(coefficient)
    return answer


def reduced_schur_data(*arguments):
    """Divide by ``Delta^2`` and return its Schur support (small N audit)."""

    ts, polynomial, shift_A, shift_B = defect_contour_polynomial(*arguments)
    reduced = sp.factor(sp.cancel(polynomial / vandermonde(ts) ** 2))
    _, denominator = sp.fraction(reduced)
    if set(ts) & denominator.free_symbols:
        raise AssertionError(f"Delta^2 does not divide the defect insertion: {denominator}")
    coefficients = schur_coefficients(sp.expand(reduced), ts)
    return len(ts), shift_A, shift_B, coefficients


def one_defect_schur_coefficients(
    n1, n2, n3, epsilon2, epsilon3, form_parity, eta
):
    """Univariate-minor reduction for exactly one opposite zero mode.

    This version requires the auxiliary and physical Gaussian functionals
    to be even.  The external block then has rank two, so exactly one
    natural--defect pair is used and the screening--screening block cannot
    enter.  It costs a polynomial number of univariate determinants and
    never creates a multivariate Pfaffian.
    """

    n1, n2, n3 = map(sp.Rational, (n1, n2, n3))
    count = int(2 * (n1 + n2 + n3))
    infinity, one, zero = rows_with_defects(
        n1, n2, n3, epsilon2, epsilon3
    )
    rows = infinity + one + zero
    defects = [index for index, row in enumerate(rows) if _is_defect(row)]
    if len(defects) != 1 or len(rows) != count + 2:
        raise ValueError("this reduction requires exactly one zero-mode defect")
    external_parity = len(rows) % 2
    auxiliary_form = (external_parity - int(form_parity)) % 2
    physical_screening_form = (int(form_parity) + count) % 2
    if auxiliary_form or physical_screening_form:
        raise ValueError(
            "choose form_parity=N mod 2 for the even--even one-defect block"
        )

    shift_A = int(2 * n3 - sp.Rational(1, 2))
    shift_B = int(2 * n2 - sp.Rational(1, 2))
    t = sp.symbols("t")
    clearing = t**shift_A * (1 - t) ** shift_B
    row_polynomials = []
    for row in rows:
        value = row.physical_coefficient * _row_pair_with_screening(
            row.leg, row.mode, t
        )
        if row.leg == "zero":
            value = -value
        polynomial = sp.cancel(clearing * value)
        numerator, denominator = sp.fraction(polynomial)
        if t in denominator.free_symbols:
            raise AssertionError((row, denominator))
        row_polynomials.append(sp.Poly(sp.expand(numerator / denominator), t))

    maximum_degree = max(polynomial.degree() for polynomial in row_polynomials)
    coefficient_matrix = sp.zeros(len(rows), maximum_degree + 1)
    for row_index, polynomial in enumerate(row_polynomials):
        for (degree,), value in polynomial.terms():
            coefficient_matrix[row_index, degree] = value

    # Move the zero-leg external rows through all screening rows.  The two
    # signs from the bipartite Pfaffian and alternant orientation cancel.
    reorder_sign = (-1) ** (count * len(zero))
    answer = {}
    for left in range(len(rows)):
        for right in range(left + 1, len(rows)):
            external_value = _natural_defect_covariance(rows[left], rows[right])
            if external_value == 0:
                continue
            pair_sign = (-1) ** (left + right - 1)
            kept_rows = [
                index for index in range(len(rows)) if index not in (left, right)
            ]
            reduced_matrix = coefficient_matrix[kept_rows, :]
            for degrees in __import__("itertools").combinations(
                range(maximum_degree + 1), count
            ):
                minor = reduced_matrix[:, degrees].det(method="domain-ge")
                if minor == 0:
                    continue
                reverse_degrees = tuple(reversed(degrees))
                partition = tuple(
                    reverse_degrees[index] - (count - index - 1)
                    for index in range(count)
                )
                partition = tuple(value for value in partition if value)
                value = sp.factor(
                    reorder_sign * pair_sign * external_value * minor
                )
                answer[partition] = sp.factor(answer.get(partition, 0) + value)
                if answer[partition] == 0:
                    del answer[partition]
    return count, shift_A, shift_B, answer


def audit():
    expected_ground = {(0, 0): 1, (1, 1): I}
    if local_ground_action() != expected_ground:
        raise AssertionError((local_ground_action(), expected_ground))
    print("opposite zero mode: chi0- chi0+ |00> = |00> + i |11>")

    # The one-defect even--even block is reduced entirely with univariate
    # minors.  Its form parity is N mod 2.
    cases = (
        (0, sp.Rational(1, 4), sp.Rational(1, 4)),
        (0, sp.Rational(3, 4), sp.Rational(3, 4)),
        (sp.Rational(1, 2), sp.Rational(1, 4), sp.Rational(3, 4)),
        (2, sp.Rational(7, 4), sp.Rational(7, 4)),
    )
    for labels in cases:
        natural2 = natural_ramond_parity(labels[1])
        natural3 = natural_ramond_parity(labels[2])
        for defects in ((1, 0), (0, 1)):
            epsilon2 = natural2 ^ defects[0]
            epsilon3 = natural3 ^ defects[1]
            count = int(2 * sum(labels))
            count, _, _, coefficients = one_defect_schur_coefficients(
                *labels, epsilon2, epsilon3, count % 2, 1
            )
            width = max(
                (partition[0] if partition else 0 for partition in coefficients),
                default=0,
            )
            print(
                f"labels={labels}, defect_leg={'one' if defects[0] else 'zero'}, N={count}, "
                f"Schur_terms={len(coefficients)}, width={width}"
            )

    # At one screening the full multivariate Pfaffian is cheap; compare it
    # coefficient by coefficient with the minor formula.
    labels = (0, sp.Rational(1, 4), sp.Rational(1, 4))
    natural2 = natural_ramond_parity(labels[1])
    natural3 = natural_ramond_parity(labels[2])
    for defects in ((1, 0), (0, 1)):
        epsilon2 = natural2 ^ defects[0]
        epsilon3 = natural3 ^ defects[1]
        direct = reduced_schur_data(
            *labels, epsilon2, epsilon3, 1, 1
        )[3]
        minor = one_defect_schur_coefficients(
            *labels, epsilon2, epsilon3, 1, 1
        )[3]
        if direct != minor:
            raise AssertionError((defects, direct, minor))
    print("one-defect minors: exact literal-Pfaffian contact check passed")


if __name__ == "__main__":
    audit()
