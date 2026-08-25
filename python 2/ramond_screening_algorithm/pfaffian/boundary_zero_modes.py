"""Ramond-zero-mode-resolved screening Pfaffian.

The two Ramond zero modes must act on the two ground doublets.  Treating
them as ordinary bulk fermions replaces the trivalent ground matrices by
a Gaussian contraction and is already wrong at ``(0,5/4,5/4)``.  Here the
two ``chi_0`` factors are expanded explicitly (four choices), while every
nonzero consecutive chi mode and every screening fermion is still summed
by one Pfaffian.  Thus the boundary multiplicity is constant and no SCA
PBW state is ever constructed.

This file is an experimental calibration layer.  Its formulas are kept
separate from ``screening_pfaffian.py`` until every spin-frame phase has
been checked against the stored Ward data.
"""

from __future__ import annotations

from functools import lru_cache
import itertools

import sympy as sp

from .core import pfaffian, pfaffian_recursive
from .screening_pfaffian import (
    ExternalRow,
    I,
    SQRT2,
    _row_pair_with_screening,
    _screen_kernel,
    coefficient,
    natural_ramond_parity,
    vandermonde,
)
from .special_oracle import ordinary_selberg, physical_nsrr_selberg


FOCK_TO_SCBLOCK_MINUS = -(1 - I) / SQRT2


def auxiliary_ground(form_parity, second_ground, third_ground):
    if int(form_parity) % 2 == 0:
        return {(0, 0): 1, (1, 1): -1}.get(
            (int(second_ground), int(third_ground)), 0
        )
    return {(0, 1): 1, (1, 0): -1}.get(
        (int(second_ground), int(third_ground)), 0
    )


def physical_ground(form_parity, eta, second_ground, third_ground):
    if int(form_parity) % 2 == 0:
        return {(0, 0): 1, (1, 1): int(eta)}.get(
            (int(second_ground), int(third_ground)), 0
        )
    return {(0, 1): 1, (1, 0): I * int(eta)}.get(
        (int(second_ground), int(third_ground)), 0
    )


def _ground(sector, form_parity, eta, second_ground, third_ground):
    if sector == "auxiliary":
        return auxiliary_ground(form_parity, second_ground, third_ground)
    return physical_ground(
        form_parity, eta, second_ground, third_ground
    )


def _local_one_point(leg, mode):
    mode = sp.Rational(mode)
    x = sp.symbols("x")
    if leg == "zero":
        return coefficient(1 / (SQRT2 * sp.sqrt(1 - x)), x, mode)
    if leg == "one":
        return coefficient(1 / (SQRT2 * sp.sqrt(1 + x)), x, mode)
    if leg == "inf":
        return I * coefficient(
            1 / (SQRT2 * sp.sqrt(1 - x)),
            x,
            mode - sp.Rational(1, 2),
        )
    if leg == "screen":
        return 1 / SQRT2
    raise ValueError(leg)


def _odd_mean(leg, mode, sector, form_parity, eta, ground2, ground3):
    """One unpaired fermion with the correct boundary ground flip."""

    if leg == "inf":
        ground_value = _ground(
            sector, 1 - int(form_parity), eta, ground2, ground3
        )
    elif leg == "one":
        ground_value = _ground(
            sector, form_parity, eta, 1 - int(ground2), ground3
        )
    else:  # zero and a screening ordered on the zero side of the cut
        ground_value = _ground(
            sector, form_parity, eta, ground2, 1 - int(ground3)
        )
    return sp.factor(_local_one_point(leg, mode) * ground_value)


def _local_data(leg, variable):
    if leg == "zero":
        return variable**2, variable / sp.sqrt(1 - variable**2)
    if leg == "one":
        return 1 + variable**2, -I * sp.sqrt(1 + variable**2) / variable
    if leg == "inf":
        return 1 / variable**2, -I / sp.sqrt(1 - variable**2)
    raise ValueError(leg)


def _local_power(leg, mode):
    mode = sp.Rational(mode)
    return int(2 * mode + (1 if leg == "inf" else -1))


def _series_coefficient(expression, variable, power, order=None):
    if order is None:
        order = max(8, int(power) + 6)
    expanded = sp.series(expression, variable, 0, order).removeO().expand()
    return expanded.coeff(variable, int(power))


@lru_cache(None)
def external_pair(leg_a, mode_a, leg_b, mode_b):
    """Ordered two-spin kernel coefficient in the repository spin frame."""

    y, x, ratio = sp.symbols("y x ratio")
    z_a, square_a = _local_data(leg_a, y)
    z_b, square_b = _local_data(leg_b, x)
    kernel = (square_a / square_b + square_b / square_a) / (
        2 * (z_a - z_b)
    )
    power_a = _local_power(leg_a, mode_a)
    power_b = _local_power(leg_b, mode_b)
    frames = {"inf": -1, "one": I, "zero": 1}
    cocycle = -1 if (leg_a, leg_b) == ("one", "zero") else 1
    if leg_a == leg_b == "inf":
        nested = _series_coefficient(kernel.subs(y, ratio * x), ratio, power_a)
        value = _series_coefficient(nested, x, power_a + power_b)
    elif leg_a == leg_b:
        nested = _series_coefficient(kernel.subs(x, ratio * y), ratio, power_b)
        value = _series_coefficient(nested, y, power_a + power_b)
    else:
        value = _series_coefficient(
            _series_coefficient(kernel, y, power_a), x, power_b
        )
    return sp.factor(frames[leg_a] * frames[leg_b] * cocycle * value)


def _nonzero_rows(n1, n2, n3, auxiliary_ground2, auxiliary_ground3):
    """Rows after moving each Ramond zero mode to the rightmost position."""

    n1, n2, n3 = map(sp.Rational, (n1, n2, n3))
    infinity = tuple(
        ExternalRow(
            "inf",
            sp.Rational(2 * index + 1, 2),
            -1,
            I * (-1) ** (int(auxiliary_ground2) + int(auxiliary_ground3)),
        )
        for index in range(int(2 * n1))
    )
    maximum2 = int(2 * n2 - sp.Rational(1, 2))
    maximum3 = int(2 * n3 - sp.Rational(1, 2))
    one = tuple(
        ExternalRow(
            "one",
            sp.Integer(mode),
            1,
            -I * (-1) ** (int(auxiliary_ground2) + int(auxiliary_ground3)),
        )
        for mode in range(maximum2, 0, -1)
    )
    zero = tuple(
        ExternalRow(
            "zero",
            sp.Integer(mode),
            1,
            I * (-1) ** (int(auxiliary_ground2) + int(auxiliary_ground3)),
        )
        for mode in range(maximum3, 0, -1)
    )
    return infinity, one, zero


def _fixed_boundary_correlator(
    n1,
    n2,
    n3,
    form_parity,
    eta,
    screenings,
    auxiliary_ground2,
    auxiliary_ground3,
    physical_twist="right",
    screening_one_point_sign=1,
    border_component="full",
):
    """One of the four zero-mode boundary sectors."""

    physical_ground2 = 1 - int(auxiliary_ground2)
    physical_ground3 = 1 - int(auxiliary_ground3)
    infinity, one, zero = _nonzero_rows(
        n1, n2, n3, auxiliary_ground2, auxiliary_ground3
    )
    ts = tuple(screenings)
    # Screenings belong only to the physical form.  Put every external chi
    # row before them for the product-theory Koszul sign, then compensate
    # the physical Wick order by (-1)^N on each zero-leg physical row.
    zero = tuple(
        ExternalRow(
            row.leg,
            row.mode,
            row.auxiliary_coefficient,
            row.physical_coefficient * (-1) ** len(ts),
        )
        for row in zero
    )
    objects = (
        tuple(("external", row) for row in infinity + one + zero)
        + tuple(("screening", t) for t in ts)
    )
    external_total_parity = int(
        (2 * (sp.Rational(n1) + sp.Rational(n2) + sp.Rational(n3)) + 1)
        % 2
    )
    auxiliary_form = (external_total_parity - int(form_parity)) % 2
    physical_form = int(form_parity) % 2
    # The two constant-rank Ramond holonomies put X^N on the right or on
    # the left of Gamma_f.  They toggle respectively the third or second
    # ground index and coincide when N is even.
    if physical_twist == "right":
        physical_ground2_effective = physical_ground2
        physical_ground3_effective = physical_ground3 ^ (len(ts) % 2)
    elif physical_twist == "left":
        physical_ground2_effective = physical_ground2 ^ (len(ts) % 2)
        physical_ground3_effective = physical_ground3
    else:
        raise ValueError("physical_twist must be 'right' or 'left'")
    auxiliary_required = (
        auxiliary_form + int(auxiliary_ground2) + int(auxiliary_ground3)
    ) % 2
    physical_required = (
        physical_form + physical_ground2_effective + physical_ground3_effective
    ) % 2
    if auxiliary_required + physical_required != 1:
        raise AssertionError((auxiliary_required, physical_required))

    size = len(objects)
    matrix = [[sp.Integer(0) for _ in range(size)] for _ in range(size)]
    mean = [sp.Integer(0)] * size
    odd_sector = "auxiliary" if auxiliary_required else "physical"
    even_sector = "physical" if auxiliary_required else "auxiliary"
    even_form = physical_form if even_sector == "physical" else auxiliary_form
    even_ground2 = (
        physical_ground2_effective
        if even_sector == "physical"
        else auxiliary_ground2
    )
    even_ground3 = (
        physical_ground3_effective
        if even_sector == "physical"
        else auxiliary_ground3
    )
    even_scalar = _ground(
        even_sector, even_form, eta, even_ground2, even_ground3
    )
    if even_scalar == 0:
        raise AssertionError("the purported even boundary functional vanished")

    odd_form = auxiliary_form if odd_sector == "auxiliary" else physical_form
    odd_ground2 = (
        auxiliary_ground2
        if odd_sector == "auxiliary"
        else physical_ground2_effective
    )
    odd_ground3 = (
        auxiliary_ground3
        if odd_sector == "auxiliary"
        else physical_ground3_effective
    )
    if border_component == "screening" and odd_sector != "physical":
        return sp.Integer(0)
    if border_component not in ("full", "screening"):
        raise ValueError("border_component must be 'full' or 'screening'")
    for index, (kind, item) in enumerate(objects):
        if kind == "screening":
            if odd_sector == "physical":
                mean[index] = int(screening_one_point_sign) * _odd_mean(
                    "screen", 0, odd_sector, odd_form, eta, odd_ground2, odd_ground3
                )
            continue
        if border_component == "screening":
            continue
        coefficient_value = (
            item.auxiliary_coefficient
            if odd_sector == "auxiliary"
            else item.physical_coefficient
        )
        mean[index] = coefficient_value * _odd_mean(
            item.leg,
            item.mode,
            odd_sector,
            odd_form,
            eta,
            odd_ground2,
            odd_ground3,
        )

    for left in range(size):
        kind_left, item_left = objects[left]
        for right in range(left + 1, size):
            kind_right, item_right = objects[right]
            if kind_left == kind_right == "screening":
                value = _screen_kernel(item_left, item_right)
            elif kind_left == "external" and kind_right == "screening":
                pair = _row_pair_with_screening(
                    item_left.leg, item_left.mode, item_right
                )
                if item_left.leg == "zero":
                    pair = -pair
                value = item_left.physical_coefficient * pair
            elif kind_left == "screening" and kind_right == "external":
                value = item_right.physical_coefficient * _row_pair_with_screening(
                    item_right.leg, item_right.mode, item_left
                )
            else:
                pair = external_pair(
                    item_left.leg, item_left.mode, item_right.leg, item_right.mode
                )
                value = pair * (
                    item_left.auxiliary_coefficient
                    * item_right.auxiliary_coefficient
                    + item_left.physical_coefficient
                    * item_right.physical_coefficient
                )
            matrix[left][right] = value
            matrix[right][left] = -value

    augmented = [row + [mean[index]] for index, row in enumerate(matrix)]
    augmented.append([-value for value in mean] + [sp.Integer(0)])
    evaluator = pfaffian_recursive if len(augmented) <= 10 else pfaffian
    return even_scalar * evaluator(augmented)


def resolved_contour_laurent(
    n1,
    n2,
    n3,
    form_parity,
    eta,
    screenings=None,
    physical_twist="right",
    screening_one_point_sign=1,
    border_component="full",
):
    """Sum the four exact Ramond zero-mode boundary sectors."""

    n1, n2, n3 = map(sp.Rational, (n1, n2, n3))
    if screenings is None:
        screening_count = int(2 * (n1 + n2 + n3))
        ts = sp.symbols(f"t0:{screening_count}")
    elif isinstance(screenings, (int, sp.Integer)):
        screening_count = int(screenings)
        ts = sp.symbols(f"t0:{screening_count}")
    else:
        ts = tuple(map(sp.sympify, screenings))
        screening_count = len(ts)
    maximum2 = int(2 * n2 - sp.Rational(1, 2))
    maximum3 = int(2 * n3 - sp.Rational(1, 2))
    answer = sp.Integer(0)
    for auxiliary_ground2, auxiliary_ground3 in itertools.product((0, 1), repeat=2):
        physical_ground2 = 1 - auxiliary_ground2
        physical_ground3 = 1 - auxiliary_ground3
        reference2 = (
            (-1) ** (maximum2 * (maximum2 + 1) // 2)
            / SQRT2
            * (1 if auxiliary_ground2 else -I)
        )
        reference3 = (
            (-1) ** (maximum3 * (maximum3 + 1) // 2)
            / SQRT2
            * (1 if auxiliary_ground3 else -I)
        )
        # Product-form Koszul sign of the reference component in which all
        # nonzero Ramond modes are auxiliary.
        zero_coefficient = (
            reference2
            * reference3
            * (-1) ** (physical_ground2 * (maximum3 + auxiliary_ground3))
            * FOCK_TO_SCBLOCK_MINUS
            ** (physical_ground2 + physical_ground3)
        )
        answer += zero_coefficient * _fixed_boundary_correlator(
            n1,
            n2,
            n3,
            form_parity,
            eta,
            ts,
            auxiliary_ground2,
            auxiliary_ground3,
            physical_twist,
            screening_one_point_sign,
            border_component,
        )
    return ts, answer


def projected_contour_laurent(
    n1,
    n2,
    n3,
    form_parity,
    eta,
    screenings=None,
    physical_twist="right",
):
    """The half-difference projection, evaluated with one border only."""

    return resolved_contour_laurent(
        n1,
        n2,
        n3,
        form_parity,
        eta,
        screenings=screenings,
        physical_twist=physical_twist,
        screening_one_point_sign=1,
        border_component="screening",
    )


def projected_vandermonde_constant(
    n1,
    n2,
    n3,
    form_parity,
    eta,
    sample_shift=0,
    physical_twist="right",
):
    """Sample the projected insertion divided by ``Delta^2`` exactly.

    This routine avoids a multivariate expansion: it evaluates the
    fixed-sector Pfaffian at exact rational nodes and returns
    ``clearing*correlator/Delta``.  A single value is not a proof that the
    quotient is constant.  :func:`projected_determinant_constant` proves
    constancy in the supported zero-external-block orderings; this sampler
    is an independent audit of its coefficient and sign.
    """

    n1, n2, n3 = map(sp.Rational, (n1, n2, n3))
    count = int(2 * (n1 + n2 + n3))
    denominator = count + 2 + int(sample_shift)
    nodes = tuple(
        sp.Rational(index + 1 + int(sample_shift), denominator + int(sample_shift))
        for index in range(count)
    )
    plus_t, projected = projected_contour_laurent(
        n1,
        n2,
        n3,
        form_parity,
        eta,
        screenings=nodes,
        physical_twist=physical_twist,
    )
    shift_A = int(2 * n3 - sp.Rational(1, 2))
    shift_B = int(2 * n2 - sp.Rational(1, 2))
    clearing = sp.prod(
        t**shift_A * (1 - t) ** shift_B for t in plus_t
    )
    return sp.factor(sp.cancel(clearing * projected / vandermonde(plus_t)))


@lru_cache(None)
def projected_determinant_constant(
    n1,
    n2,
    n3,
    form_parity,
    eta,
    physical_twist="right",
):
    """Return the projected ``Delta**2`` coefficient in cubic time.

    On the maximal-screening plane there are ``N-1`` nonzero external
    ``chi`` rows and ``N`` screening rows.  In the projected boundary
    component the only odd mean is the screening mean.  Moreover the
    external--external block vanishes in the supported charge ordering.
    Every perfect matching therefore pairs all
    ``N-1`` external rows and the one border row with the ``N`` screenings.
    The Pfaffian is exactly one ``N by N`` determinant; its
    screening--screening block never enters.

    Clearing the endpoint poles makes every determinant row a polynomial
    of degree at most ``N-1``.  If ``R`` is their coefficient matrix, then

    ``det(p_i(t_j)) = (-1)**(N*(N-1)/2) det(R) Delta(t)``.

    Combining this with the Pfaffian block-ordering sign leaves the simple
    factor ``(-1)**N det(R)`` below.  Unlike
    :func:`projected_vandermonde_constant`, this is a symbolic identity and
    does not evaluate the quotient at sample nodes.
    """

    n1, n2, n3 = map(sp.Rational, (n1, n2, n3))
    count = int(2 * (n1 + n2 + n3))
    maximum2 = int(2 * n2 - sp.Rational(1, 2))
    maximum3 = int(2 * n3 - sp.Rational(1, 2))
    shift_A = maximum3
    shift_B = maximum2
    t = sp.symbols("t")
    clearing = t**shift_A * (1 - t) ** shift_B
    external_total_parity = int(
        (2 * (n1 + n2 + n3) + 1) % 2
    )
    auxiliary_form = (external_total_parity - int(form_parity)) % 2
    physical_form = int(form_parity) % 2
    answer = sp.Integer(0)

    for auxiliary_ground2, auxiliary_ground3 in itertools.product((0, 1), repeat=2):
        physical_ground2 = 1 - auxiliary_ground2
        physical_ground3 = 1 - auxiliary_ground3
        infinity, one, zero = _nonzero_rows(
            n1, n2, n3, auxiliary_ground2, auxiliary_ground3
        )
        zero = tuple(
            ExternalRow(
                row.leg,
                row.mode,
                row.auxiliary_coefficient,
                row.physical_coefficient * (-1) ** count,
            )
            for row in zero
        )
        rows = infinity + one + zero
        if len(rows) != count - 1:
            raise AssertionError((len(rows), count))
        for left, row_left in enumerate(rows):
            for row_right in rows[left + 1 :]:
                coefficient_sum = (
                    row_left.auxiliary_coefficient
                    * row_right.auxiliary_coefficient
                    + row_left.physical_coefficient
                    * row_right.physical_coefficient
                )
                if coefficient_sum and external_pair(
                    row_left.leg,
                    row_left.mode,
                    row_right.leg,
                    row_right.mode,
                ):
                    raise NotImplementedError(
                        "the projected external block is nonzero; use the "
                        "bounded-width Schur/Pfaffian route for this charge ordering"
                    )

        if physical_twist == "right":
            physical_ground2_effective = physical_ground2
            physical_ground3_effective = physical_ground3 ^ (count % 2)
        elif physical_twist == "left":
            physical_ground2_effective = physical_ground2 ^ (count % 2)
            physical_ground3_effective = physical_ground3
        else:
            raise ValueError("physical_twist must be 'right' or 'left'")

        auxiliary_required = (
            auxiliary_form + auxiliary_ground2 + auxiliary_ground3
        ) % 2
        physical_required = (
            physical_form
            + physical_ground2_effective
            + physical_ground3_effective
        ) % 2
        # ``border_component='screening'`` keeps precisely an odd physical
        # functional and an even auxiliary functional.
        if auxiliary_required or physical_required != 1:
            continue
        even_scalar = auxiliary_ground(
            auxiliary_form, auxiliary_ground2, auxiliary_ground3
        )
        if even_scalar == 0:
            continue
        border_mean = _odd_mean(
            "screen",
            0,
            "physical",
            physical_form,
            eta,
            physical_ground2_effective,
            physical_ground3_effective,
        )

        row_polynomials = []
        for row in rows:
            value = row.physical_coefficient * _row_pair_with_screening(
                row.leg, row.mode, t
            )
            if row.leg == "zero":
                value = -value
            row_polynomials.append(sp.cancel(clearing * value))
        # In the augmented skew matrix the final border-to-screen entry is
        # ``-mean`` (the screen-to-border entry is ``+mean``).
        row_polynomials.append(sp.cancel(-clearing * border_mean))

        coefficient_matrix = sp.zeros(count)
        for row_index, expression in enumerate(row_polynomials):
            numerator, denominator = sp.fraction(expression)
            if t in denominator.free_symbols:
                raise AssertionError((row_index, denominator))
            polynomial = sp.Poly(sp.expand(numerator / denominator), t)
            if polynomial.degree() >= count:
                raise AssertionError((row_index, polynomial.degree(), count))
            for (degree,), coefficient_value in polynomial.terms():
                coefficient_matrix[row_index, degree] = coefficient_value

        reference2 = (
            (-1) ** (maximum2 * (maximum2 + 1) // 2)
            / SQRT2
            * (1 if auxiliary_ground2 else -I)
        )
        reference3 = (
            (-1) ** (maximum3 * (maximum3 + 1) // 2)
            / SQRT2
            * (1 if auxiliary_ground3 else -I)
        )
        zero_coefficient = (
            reference2
            * reference3
            * (-1) ** (physical_ground2 * (maximum3 + auxiliary_ground3))
            * FOCK_TO_SCBLOCK_MINUS
            ** (physical_ground2 + physical_ground3)
        )
        answer += (
            zero_coefficient
            * even_scalar
            * (-1) ** count
            * coefficient_matrix.det()
        )

    return sp.factor(sp.cancel(answer))


def projected_selberg_ratio(
    n1,
    n2,
    n3,
    form_parity,
    eta,
    A,
    B,
    g,
    physical_twist="right",
):
    """Integrate the certified factorized projection without PBW states.

    For consecutive ``chi`` strings the denominator-cleared projected
    insertion is

    ``C * Delta(t)**2``.

    Consequently its integral is one ordinary Selberg product with
    coupling ``g+1``.  The denominator is the primary BFL two-spin
    integral in the same rationalized-Majorana convention.  This routine
    deliberately returns only the charge-preserving,
    factorized channel.  The crossed channel requires the reflected
    Ramond fermion kernel and is not supplied by the ordinary two-spin
    covariance used in this module.
    """

    n1, n2, n3 = map(sp.Rational, (n1, n2, n3))
    count = int(2 * (n1 + n2 + n3))
    shift_A = int(2 * n3 - sp.Rational(1, 2))
    shift_B = int(2 * n2 - sp.Rational(1, 2))
    constant = projected_determinant_constant(
        n1,
        n2,
        n3,
        form_parity,
        eta,
        physical_twist=physical_twist,
    )
    numerator = constant * ordinary_selberg(
        count, A - shift_A, B - shift_B, g + 1
    )
    denominator = physical_nsrr_selberg(count, A, B, g)
    # ``physical_nsrr_selberg`` uses the BFL polynomial normalization,
    # whereas the rationalized Majorana Pfaffian is smaller by 1/sqrt(2)
    # for odd screening number.
    if count % 2:
        denominator /= SQRT2
    return sp.factor(
        sp.powsimp(
            sp.cancel(sp.expand_func(numerator / denominator)), force=True
        )
    )


def resolved_contour_polynomial(
    n1,
    n2,
    n3,
    form_parity,
    eta,
    physical_twist="right",
    screening_one_point_sign=1,
):
    ts, laurent = resolved_contour_laurent(
        n1,
        n2,
        n3,
        form_parity,
        eta,
        physical_twist=physical_twist,
        screening_one_point_sign=screening_one_point_sign,
    )
    n2, n3 = sp.Rational(n2), sp.Rational(n3)
    shift_A = int(2 * n3 - sp.Rational(1, 2))
    shift_B = int(2 * n2 - sp.Rational(1, 2))
    clearing = sp.prod(t**shift_A * (1 - t) ** shift_B for t in ts)
    polynomial = sp.cancel(vandermonde(ts) * clearing * laurent)
    numerator, denominator = sp.fraction(polynomial)
    if set(ts) & denominator.free_symbols:
        raise AssertionError(denominator)
    return ts, sp.expand(numerator / denominator), shift_A, shift_B


def projected_contour_polynomial(
    n1, n2, n3, form_parity, eta, physical_twist="right"
):
    """Charge-preserving half-difference of the two screening borders.

    This is the component fixed by the standard BFL screening contour.  It
    is exactly the literal Fock-path projection, not an ell-product ansatz.
    The crossed Ramond holonomy is an independent trivalent datum.
    """

    plus_t, projected = projected_contour_laurent(
        n1,
        n2,
        n3,
        form_parity,
        eta,
        physical_twist=physical_twist,
    )
    ts = plus_t
    n2, n3 = sp.Rational(n2), sp.Rational(n3)
    shift_A = int(2 * n3 - sp.Rational(1, 2))
    shift_B = int(2 * n2 - sp.Rational(1, 2))
    clearing = sp.prod(t**shift_A * (1 - t) ** shift_B for t in ts)
    polynomial = sp.cancel(
        vandermonde(ts) * clearing * projected
    )
    numerator, denominator = sp.fraction(polynomial)
    if set(ts) & denominator.free_symbols:
        raise AssertionError(denominator)
    return ts, sp.expand(numerator / denominator), shift_A, shift_B


def audit_ground():
    for form_parity in (0, 1):
        for eta in (1, -1):
            ts, polynomial, _, _ = resolved_contour_polynomial(
                0, sp.Rational(1, 4), sp.Rational(1, 4), form_parity, eta
            )
            print(form_parity, eta, ts, sp.factor(polynomial))


if __name__ == "__main__":
    audit_ground()
