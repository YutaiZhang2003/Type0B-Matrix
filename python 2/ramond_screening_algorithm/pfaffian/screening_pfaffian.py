"""Contour Pfaffian for the positive 2016 NS--R--R chi strings.

This module keeps the consecutive ``chi=psi-i*eta`` strings as contour
insertions.  The auxiliary Majorana ``psi`` and the free-field Majorana
``eta`` have the same two-spin kernel.  Hence every external--external
contraction cancels for an unmodified positive chi string.  Only the
external--screening block, the screening--screening block, and (when the
total functional is odd) one border remain.  A single Pfaffian therefore
does the sum over all auxiliary/physical assignments.

After multiplication by the screening Vandermonde the answer is a finite
symmetric polynomial.  ``selberg_ratio`` averages this polynomial exactly
with the Jack/Kadell backend.  No SCA state, transition matrix, or Ward
recursion occurs in this file.

The implementation presently exposes the *natural* Ramond copy, i.e. the
copy whose parity is the length of the 2016 consecutive string.  The other
copy has the extra opposite zero mode and requires the coincident-contour
contact term; it is deliberately rejected rather than silently guessed.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import itertools

import sympy as sp

from .core import pfaffian
from .ising_polynomial import spin_pfaffian_polynomial
from .selberg_jack import (
    normalized_selberg_average,
    normalized_selberg_average_schur,
)
from .special_oracle import ordinary_selberg, physical_nsrr_selberg


I = sp.I
SQRT2 = sp.sqrt(2)


def vandermonde(xs):
    answer = sp.Integer(1)
    for left in range(len(xs)):
        for right in range(left + 1, len(xs)):
            answer *= xs[left] - xs[right]
    return answer


def coefficient(expression, variable, power):
    """Exact Taylor coefficient at zero (all requested powers are >=0)."""

    power = int(power)
    return sp.expand(sp.series(expression, variable, 0, power + 1).removeO()).coeff(
        variable, power
    )


@dataclass(frozen=True)
class ExternalRow:
    """One BPZ-ordered external chi contour.

    ``leg`` is ``inf``, ``one``, or ``zero``.  ``mode`` is the positive
    magnitude of the creation mode on the ket before BPZ.  The coefficients
    already include BPZ on the infinity leg.
    """

    leg: str
    mode: sp.Rational
    auxiliary_coefficient: sp.Expr
    physical_coefficient: sp.Expr


def natural_ramond_parity(branch_label):
    branch_label = sp.Rational(branch_label)
    if branch_label <= 0 or not (2 * branch_label - sp.Rational(1, 2)).is_integer:
        raise ValueError("this contour implementation expects positive n in Z/2+1/4")
    length = int(2 * branch_label + sp.Rational(1, 2))
    return length % 2


def external_rows(n1, n2, n3, epsilon2=None, epsilon3=None):
    """Return rows in radial product order: infinity, one, zero.

    Screenings are inserted between the ``one`` and ``zero`` groups by
    ``contour_polynomial``.  Positive NS strings are BPZ reversed at
    infinity.  Positive Ramond strings are ``chi^-_0 chi^-_-1 ...``.
    """

    n1, n2, n3 = map(sp.Rational, (n1, n2, n3))
    if n1 < 0 or not (2 * n1).is_integer:
        raise ValueError("the NS label must be a nonnegative half-integer")
    natural2 = natural_ramond_parity(n2)
    natural3 = natural_ramond_parity(n3)
    if epsilon2 is not None and int(epsilon2) != natural2:
        raise NotImplementedError("the non-natural Ramond copy needs the zero-mode contact term")
    if epsilon3 is not None and int(epsilon3) != natural3:
        raise NotImplementedError("the non-natural Ramond copy needs the zero-mode contact term")

    infinity = tuple(
        ExternalRow("inf", sp.Rational(2 * index + 1, 2), -1, I)
        for index in range(int(2 * n1))
    )
    count2 = int(2 * n2 + sp.Rational(1, 2))
    count3 = int(2 * n3 + sp.Rational(1, 2))
    one = tuple(ExternalRow("one", sp.Integer(mode), 1, -I) for mode in range(count2))
    zero = tuple(ExternalRow("zero", sp.Integer(mode), 1, -I) for mode in range(count3))
    return infinity, one, zero


@lru_cache(None)
def _row_pair_with_screening(leg, mode, t):
    """Two-spin kernel after the external contour and screen radical.

    The result includes the repository plumbing spin frame and the radial
    ordering cocycle, but not the chi coefficient ``-i`` (or its BPZ image).
    It is a polynomial in ``t`` for every consecutive mode used here.
    """

    mode = sp.Rational(mode)
    z = sp.symbols("z")
    kernel_hat = (z + t - 2 * z * t) / (2 * (z - t))
    if leg == "zero":
        # The row itself is later than the screening in radial product order;
        # this function returns K(screen,zero), hence the extra minus sign.
        return sp.factor(-coefficient(kernel_hat / sp.sqrt(1 - z), z, mode))
    if leg == "one":
        y = sp.symbols("y")
        at_one = kernel_hat.subs(z, 1 + y) / sp.sqrt(1 + y)
        # The ordered one--zero spin fields carry the canonical cocycle -1.
        return sp.factor(-coefficient(at_one, y, mode))
    if leg == "inf":
        u = sp.symbols("u")
        at_infinity = (
            (1 - 2 * t + t * u) / (2 * (1 - t * u) * sp.sqrt(1 - u))
        )
        # The local infinity spin frame is +i before the BPZ coefficient
        # stored in ExternalRow.
        index = mode - sp.Rational(1, 2)
        return sp.factor(I * coefficient(at_infinity, u, index))
    raise ValueError(leg)


@lru_cache(None)
def _row_one_point(leg, mode, sector, eta):
    """Odd two-spin functional in the canonical SCblock spin frame."""

    mode = sp.Rational(mode)
    eta = int(eta)
    x = sp.symbols("x")
    if leg == "zero":
        base = coefficient(1 / (SQRT2 * sp.sqrt(1 - x)), x, mode)
        ground = 1
    elif leg == "one":
        base = coefficient(1 / (SQRT2 * sp.sqrt(1 + x)), x, mode)
        # Gamma_1^F(1,0)=-1, whereas the physical odd ground matrix has
        # Gamma_1^(eta)(1,0)=i eta.
        ground = -1 if sector == "auxiliary" else I * eta
    elif leg == "inf":
        index = mode - sp.Rational(1, 2)
        base = I * coefficient(1 / (SQRT2 * sp.sqrt(1 - x)), x, index)
        ground = 1
    else:
        raise ValueError(leg)
    return sp.factor(ground * base)


def _screen_kernel(left, right):
    return (left + right - 2 * left * right) / (2 * (left - right))


def _combined_pfaffian(objects, auxiliary_parity, physical_parity, eta):
    """Tensor-product Majorana correlator as one Pfaffian.

    Odd--odd functionals are represented by the rank-two skew update
    ``mu_aux wedge mu_phys``.  This is the exterior-algebra form of the
    Koszul sign in the product three-point form.
    """

    objects = tuple(objects)
    size = len(objects)
    covariance = [[sp.Integer(0) for _ in range(size)] for _ in range(size)]
    mean_aux = [sp.Integer(0)] * size
    mean_phys = [sp.Integer(0)] * size

    for index, (kind, item) in enumerate(objects):
        if kind == "external":
            row = item
            mean_aux[index] = row.auxiliary_coefficient * _row_one_point(
                row.leg, row.mode, "auxiliary", eta
            )
            mean_phys[index] = row.physical_coefficient * _row_one_point(
                row.leg, row.mode, "physical", eta
            )
        else:
            # The global odd kernel on the ordered screening interval.
            mean_phys[index] = 1 / SQRT2

    for left in range(size):
        kind_left, item_left = objects[left]
        for right in range(left + 1, size):
            kind_right, item_right = objects[right]
            value = sp.Integer(0)
            if kind_left == kind_right == "screening":
                value = _screen_kernel(item_left, item_right)
            elif kind_left == "external" and kind_right == "screening":
                value = (
                    item_left.physical_coefficient
                    * _row_pair_with_screening(item_left.leg, item_left.mode, item_right)
                )
            elif kind_left == "screening" and kind_right == "external":
                # Kept for callers which provide the literal radial order.
                value = (
                    item_right.physical_coefficient
                    * _row_pair_with_screening(item_right.leg, item_right.mode, item_left)
                )
            else:
                # For two natural chi^- rows, 1+(-i)^2=0.  This assertion
                # prevents an unnoticed use of an opposite zero mode.
                factor = (
                    item_left.auxiliary_coefficient * item_right.auxiliary_coefficient
                    + item_left.physical_coefficient * item_right.physical_coefficient
                )
                if sp.simplify(factor) != 0:
                    raise NotImplementedError("non-cancelling external contact block")
            covariance[left][right] = value
            covariance[right][left] = -value

    auxiliary_parity = int(auxiliary_parity) % 2
    physical_parity = int(physical_parity) % 2
    if auxiliary_parity and physical_parity:
        for left in range(size):
            for right in range(left + 1, size):
                update = (
                    mean_aux[left] * mean_phys[right]
                    - mean_phys[left] * mean_aux[right]
                )
                covariance[left][right] += update
                covariance[right][left] -= update
        return pfaffian(covariance)

    if auxiliary_parity or physical_parity:
        border = mean_aux if auxiliary_parity else mean_phys
        external_indices = [
            index for index, (kind, _) in enumerate(objects) if kind == "external"
        ]
        screening_indices = [
            index for index, (kind, _) in enumerate(objects) if kind == "screening"
        ]
        # On the maximal-screening plane there are N+1 external contours
        # and N screenings.  After the odd border is appended, the zero
        # external block is square against (screenings,border), so
        #
        #   Pf [[0,X],[-X^T,D]] = (-1)^(E(E-1)/2) det X.
        #
        # This removes the screening kernel D altogether and is the main
        # reason the contour algorithm remains small at W_{7/4} and above.
        if len(external_indices) == len(screening_indices) + 1:
            cross = sp.zeros(len(external_indices), len(external_indices))
            for row, external_index in enumerate(external_indices):
                for column, screening_index in enumerate(screening_indices):
                    cross[row, column] = covariance[external_index][screening_index]
                cross[row, -1] = border[external_index]
            inversions = sum(
                screening_index < external_index
                for screening_index in screening_indices
                for external_index in external_indices
            )
            exponent = inversions + len(external_indices) * (len(external_indices) - 1) // 2
            return (-1) ** exponent * cross.det(method="domain-ge")
        augmented = [row + [border[index]] for index, row in enumerate(covariance)]
        augmented.append([-entry for entry in border] + [sp.Integer(0)])
        return pfaffian(augmented)

    return pfaffian(covariance)


def contour_polynomial(
    n1,
    n2,
    n3,
    form_parity,
    eta,
    screenings=None,
    epsilon2=None,
    epsilon3=None,
):
    """Return the polynomialized full rationalized chi correlator.

    The return value is ``(t, polynomial, shift_A, shift_B)``.  Ramond
    contour modes produce only the common Laurent denominator
    ``prod(t)^shift_A prod(1-t)^shift_B``; it is cleared here and restored
    as a shift of the Selberg exponents by ``selberg_ratio``.
    """

    n1, n2, n3 = map(sp.Rational, (n1, n2, n3))
    if screenings is None:
        screenings = 2 * (n1 + n2 + n3)
    screenings = sp.Rational(screenings)
    if not screenings.is_integer or screenings < 0:
        raise ValueError("the screening number must be a nonnegative integer")
    screening_count = int(screenings)
    infinity, one, zero = external_rows(n1, n2, n3, epsilon2, epsilon3)
    # Literal radial order.  Keeping the zero rows after the screenings is
    # important for the sign already built into _row_pair_with_screening.
    rows = infinity + one + zero
    ts = sp.symbols(f"t0:{screening_count}")
    external_parity = len(rows) % 2
    auxiliary_form = (external_parity - int(form_parity)) % 2
    physical_screening_form = (int(form_parity) + screening_count) % 2
    objects = (
        tuple(("external", row) for row in infinity + one)
        + tuple(("screening", t) for t in ts)
        + tuple(("external", row) for row in zero)
    )
    correlator = _combined_pfaffian(
        objects, auxiliary_form, physical_screening_form, eta
    )
    laurent = sp.factor(sp.cancel(vandermonde(ts) * correlator))
    shift_A = int(2 * n3 - sp.Rational(1, 2))
    shift_B = int(2 * n2 - sp.Rational(1, 2))
    clearing = sp.prod(t**shift_A * (1 - t) ** shift_B for t in ts)
    polynomial = sp.factor(sp.cancel(clearing * laurent))
    _, denominator = sp.fraction(polynomial)
    if set(ts) & denominator.free_symbols:
        raise AssertionError(f"the contour Pfaffian did not polynomialize: {denominator}")
    return ts, sp.expand(polynomial), shift_A, shift_B


def alternant_schur_coefficients(
    n1,
    n2,
    n3,
    form_parity,
    eta,
    screenings=None,
    epsilon2=None,
    epsilon3=None,
):
    """Return the reduced maximal-screening insertion in Schur form.

    If ``N=2(n1+n2+n3)`` and the natural Ramond copies are used, write the
    polynomialized contour insertion as

      Delta(t)^2 * sum_lambda coefficient[lambda] s_lambda(t).

    The returned tuple is ``(N, shift_A, shift_B, coefficient)``.  It is
    obtained from univariate row coefficients and maximal minors; no
    multivariate determinant is expanded.
    """

    n1, n2, n3 = map(sp.Rational, (n1, n2, n3))
    if screenings is None:
        screenings = 2 * (n1 + n2 + n3)
    screenings = sp.Rational(screenings)
    if not screenings.is_integer or screenings < 0:
        raise ValueError("the screening number must be a nonnegative integer")
    count = int(screenings)
    infinity, one, zero = external_rows(n1, n2, n3, epsilon2, epsilon3)
    rows = infinity + one + zero
    if len(rows) != count + 1:
        raise ValueError("the alternant reduction requires the maximal-screening plane")

    external_parity = len(rows) % 2
    auxiliary_form = (external_parity - int(form_parity)) % 2
    physical_screening_form = (int(form_parity) + count) % 2
    if auxiliary_form + physical_screening_form != 1:
        raise AssertionError("the maximal positive string must have one odd border")
    odd_sector = "auxiliary" if auxiliary_form else "physical"

    shift_A = int(2 * n3 - sp.Rational(1, 2))
    shift_B = int(2 * n2 - sp.Rational(1, 2))
    t = sp.symbols("t")
    clearing = t**shift_A * (1 - t) ** shift_B
    row_polynomials = []
    border = []
    for row in rows:
        value = row.physical_coefficient * _row_pair_with_screening(
            row.leg, row.mode, t
        )
        if row.leg == "zero":
            # Move the zero rows across the screening rows to make the
            # external block contiguous.
            value = -value
        polynomial = sp.cancel(clearing * value)
        numerator, denominator = sp.fraction(polynomial)
        if t in denominator.free_symbols:
            raise AssertionError((row, denominator))
        row_polynomials.append(sp.Poly(sp.expand(numerator / denominator), t))
        coefficient = (
            row.auxiliary_coefficient
            if odd_sector == "auxiliary"
            else row.physical_coefficient
        )
        border.append(
            sp.factor(
                coefficient * _row_one_point(row.leg, row.mode, odd_sector, eta)
            )
        )

    maximum_degree = max((polynomial.degree() for polynomial in row_polynomials), default=-1)
    coefficient_matrix = sp.zeros(len(rows), maximum_degree + 1)
    for row_index, polynomial in enumerate(row_polynomials):
        for (degree,), value in polynomial.terms():
            coefficient_matrix[row_index, degree] = value

    # Pfaffian block sign, including the permutation which moves all zero
    # contours through all screening contours.
    block_sign = (-1) ** (
        count * len(zero) + len(rows) * (len(rows) - 1) // 2
    )
    alternant_sign = (-1) ** (count * (count - 1) // 2)
    answer = {}
    border_column = sp.Matrix(border)
    for degrees in itertools.combinations(range(maximum_degree + 1), count):
        matrix = coefficient_matrix[:, degrees].row_join(border_column)
        minor = matrix.det(method="domain-ge")
        if minor == 0:
            continue
        reverse_degrees = tuple(reversed(degrees))
        partition = tuple(
            reverse_degrees[index] - (count - index - 1)
            for index in range(count)
        )
        partition = tuple(value for value in partition if value)
        value = sp.factor(block_sign * alternant_sign * minor)
        answer[partition] = sp.factor(answer.get(partition, 0) + value)
        if answer[partition] == 0:
            del answer[partition]
    return count, shift_A, shift_B, answer


def selberg_ratio(
    n1,
    n2,
    n3,
    form_parity,
    eta,
    A,
    B,
    g,
    screenings=None,
    epsilon2=None,
    epsilon3=None,
):
    """Exact contour integral divided by the primary two-spin integral."""

    ts, insertion, shift_A, shift_B = contour_polynomial(
        n1,
        n2,
        n3,
        form_parity,
        eta,
        screenings,
        epsilon2,
        epsilon3,
    )
    primary = spin_pfaffian_polynomial(ts)
    numerator = normalized_selberg_average(
        insertion, ts, A - shift_A, B - shift_B, g
    )
    numerator *= ordinary_selberg(len(ts), A - shift_A, B - shift_B, g)
    denominator = normalized_selberg_average(primary, ts, A, B, g)
    denominator *= ordinary_selberg(len(ts), A, B, g)
    return sp.factor(
        sp.powsimp(sp.cancel(sp.expand_func(numerator / denominator)), force=True)
    )


def alternant_selberg_ratio(
    n1,
    n2,
    n3,
    form_parity,
    eta,
    A,
    B,
    g,
    epsilon2=None,
    epsilon3=None,
):
    """Fast exact maximal-screening value from the reduced alternant.

    This is mathematically identical to ``selberg_ratio`` but never forms
    ``Delta^2``.  The Vandermonde square shifts ``g`` to ``g+1`` and the
    sparse Schur remainder is averaged directly in the Jack basis.
    """

    count, shift_A, shift_B, coefficients = alternant_schur_coefficients(
        n1,
        n2,
        n3,
        form_parity,
        eta,
        epsilon2=epsilon2,
        epsilon3=epsilon3,
    )
    shifted_A = A - shift_A
    shifted_B = B - shift_B
    numerator = ordinary_selberg(count, shifted_A, shifted_B, g + 1)
    numerator *= normalized_selberg_average_schur(
        coefficients, count, shifted_A, shifted_B, g + 1
    )
    # ``physical_nsrr_selberg`` uses the BFL polynomial normalization.
    # The rationalized Majorana Pfaffian is smaller by 1/sqrt(2) for odd N.
    denominator = physical_nsrr_selberg(count, A, B, g)
    if count % 2:
        denominator /= SQRT2
    return sp.factor(
        sp.powsimp(sp.cancel(sp.expand_func(numerator / denominator)), force=True)
    )


def audit_small_polynomials():
    """Cheap state-free polynomial and symmetry checks."""

    for labels in (
        (0, sp.Rational(1, 4), sp.Rational(1, 4)),
        (0, sp.Rational(3, 4), sp.Rational(3, 4)),
        (2, sp.Rational(1, 4), sp.Rational(1, 4)),
        (0, sp.Rational(7, 4), sp.Rational(1, 4)),
    ):
        n1, n2, n3 = labels
        epsilon2 = natural_ramond_parity(n2)
        epsilon3 = natural_ramond_parity(n3)
        ts, polynomial, shift_A, shift_B = contour_polynomial(
            *labels, 0, 1, epsilon2=epsilon2, epsilon3=epsilon3
        )
        # The Jack backend performs the stronger orbit reconstruction.
        from .selberg_jack import monomial_coefficients

        monomial_coefficients(polynomial, ts)
        print(
            f"labels={labels}, screenings={len(ts)}, "
            f"terms={len(sp.Poly(polynomial, *ts).terms())}, "
            f"shifts=({shift_A},{shift_B})"
        )


if __name__ == "__main__":
    audit_small_polynomials()
