#!/usr/bin/env python3
"""Fit direct NS--R--R restrictions to signed-momentum product sectors.

This is an exploratory exact-arithmetic companion to ``compute_grid.py``.
It works with the raw chi-string states, multiplies away the three direct
leg denominators, and asks whether the result is a momentum-independent
linear combination of the four numerator products obtained from
``(P_2,P_3) -> (+/- P_2,+/- P_3)``.
"""

from __future__ import annotations

import itertools

import numpy as np
import sympy as sp

import compute_grid as grid


FIT_SAMPLES = (
    grid.SAMPLES[0],
    grid.SAMPLES[1],
    (sp.Rational(7, 5), sp.Rational(2, 7), sp.Rational(4, 9), sp.Rational(5, 11)),
    (sp.Rational(4, 3), sp.Rational(3, 10), sp.Rational(5, 12), sp.Rational(7, 13)),
    (sp.Rational(9, 5), sp.Rational(2, 9), sp.Rational(3, 11), sp.Rational(4, 13)),
    (sp.Rational(8, 5), sp.Rational(1, 5), sp.Rational(4, 11), sp.Rational(6, 13)),
    (sp.Rational(11, 7), sp.Rational(3, 8), sp.Rational(2, 9), sp.Rational(5, 14)),
    (sp.Rational(13, 8), sp.Rational(2, 11), sp.Rational(5, 13), sp.Rational(7, 15)),
    (sp.Rational(14, 9), sp.Rational(3, 13), sp.Rational(4, 15), sp.Rational(5, 17)),
    (sp.Rational(15, 11), sp.Rational(2, 13), sp.Rational(5, 16), sp.Rational(7, 18)),
    (sp.Rational(16, 9), sp.Rational(4, 15), sp.Rational(3, 14), sp.Rational(5, 19)),
    (sp.Rational(17, 12), sp.Rational(5, 18), sp.Rational(4, 17), sp.Rational(7, 20)),
    (sp.Rational(18, 11), sp.Rational(3, 16), sp.Rational(5, 19), sp.Rational(8, 21)),
    (sp.Rational(19, 13), sp.Rational(4, 19), sp.Rational(6, 17), sp.Rational(5, 22)),
    (sp.Rational(20, 13), sp.Rational(5, 21), sp.Rational(3, 17), sp.Rational(7, 23)),
    (sp.Rational(21, 13), sp.Rational(2, 17), sp.Rational(7, 20), sp.Rational(8, 23)),
    (sp.Rational(22, 15), sp.Rational(5, 19), sp.Rational(4, 21), sp.Rational(7, 24)),
    (sp.Rational(23, 14), sp.Rational(3, 20), sp.Rational(6, 23), sp.Rational(8, 25)),
    (sp.Rational(24, 17), sp.Rational(4, 23), sp.Rational(5, 22), sp.Rational(9, 26)),
    (sp.Rational(25, 16), sp.Rational(5, 24), sp.Rational(7, 23), sp.Rational(8, 27)),
)

SHEETS = tuple(itertools.product((1, -1), repeat=2))
CORRELATION_SECTORS = tuple(itertools.product((1, -1), repeat=4))


def ell_degree(index):
    """Number of linear factors in ell(x,index)."""

    return (abs(int(index)) ** 2) // 4


def degree_four_index_patterns():
    """All four-factor ell-index patterns of total polynomial degree four."""

    patterns = []
    for pattern in itertools.product(range(-4, 5), repeat=4):
        if sum(ell_degree(index) for index in pattern) != 4:
            continue
        # ell(x,0) and ell(x,+/-1) are all momentum independent.  Keep one
        # representative so the numerical matrix is not needlessly singular.
        canonical = tuple(0 if abs(index) <= 1 else index for index in pattern)
        if canonical == pattern:
            patterns.append(pattern)
    return tuple(patterns)


DEGREE_FOUR_PATTERNS = degree_four_index_patterns()


def four_argument_ell_product(pattern, sample):
    b_value, p1, p2, p3 = sample
    q_value = b_value + 1 / b_value
    arguments = (
        q_value / 2 + p1 + p2 + p3,
        q_value / 2 + p1 + p2 - p3,
        q_value / 2 + p1 - p2 + p3,
        q_value / 2 + p1 - p2 - p3,
    )
    return sp.prod(
        grid.boundary.ell(argument, index, b_value)
        for argument, index in zip(arguments, pattern)
    )


def degree_four_sparse_search(labels, discrete):
    """Search a scaled raw restriction for one or two screening products."""

    samples = FIT_SAMPLES[:12]
    target = np.array(
        [complex(sp.N(scaled_raw(labels, discrete, sample), 40)) for sample in samples]
    )
    matrix = np.array(
        [
            [
                complex(sp.N(four_argument_ell_product(pattern, sample), 40))
                for pattern in DEGREE_FOUR_PATTERNS
            ]
            for sample in samples
        ],
        dtype=complex,
    )
    hits = []
    for index, pattern in enumerate(DEGREE_FOUR_PATTERNS):
        if np.max(np.abs(matrix[:, index])) == 0:
            continue
        coefficient = np.vdot(matrix[:, index], target) / np.vdot(
            matrix[:, index], matrix[:, index]
        )
        residual = np.max(np.abs(coefficient * matrix[:, index] - target))
        if residual < 1e-15:
            hits.append((residual, pattern, coefficient))
    print("degree-four single-product hits:", hits)

    best = []
    for first, second in itertools.combinations(range(len(DEGREE_FOUR_PATTERNS)), 2):
        pair = matrix[:, [first, second]]
        coefficients, _, rank, _ = np.linalg.lstsq(pair, target, rcond=1e-12)
        if rank < 2:
            continue
        residual = np.max(np.abs(pair @ coefficients - target))
        if residual < 1e-12:
            best.append(
                (
                    residual,
                    DEGREE_FOUR_PATTERNS[first],
                    DEGREE_FOUR_PATTERNS[second],
                    tuple(coefficients),
                )
            )
    print("degree-four two-product hits:")
    for hit in sorted(best, key=lambda item: item[0])[:30]:
        print(" ", hit)


def scaled_raw(labels, discrete, sample):
    n1, n2, n3 = labels
    epsilon2, epsilon3, form_parity, eta = discrete
    b_value, p1, p2, p3 = sample
    _, raw = grid.enlarged_raw_three_point(
        n1,
        n2,
        n3,
        epsilon2,
        epsilon3,
        form_parity,
        eta,
        b_value,
        p1,
        p2,
        p3,
    )
    q_value = b_value + 1 / b_value
    denominator = sp.prod(
        grid.boundary.ell(q_value + 2 * momentum, int(4 * label), b_value)
        for label, momentum in zip(labels, (p1, p2, p3))
    )
    return sp.factor(sp.cancel(raw * denominator))


def signed_products(labels, sample):
    n1, n2, n3 = labels
    b_value, p1, p2, p3 = sample
    return tuple(
        grid.boundary.numerator_product(
            n1,
            n2,
            n3,
            p1,
            second_sheet * p2,
            third_sheet * p3,
            b_value,
        )
        for second_sheet, third_sheet in SHEETS
    )


def branch_signed_numerators(labels, sample):
    n1, n2, n3 = labels
    b_value, p1, p2, p3 = sample
    return tuple(
        grid.boundary.numerator_product(
            n1,
            second_sign * n2,
            third_sign * n3,
            p1,
            p2,
            p3,
            b_value,
        )
        for second_sign, third_sign in SHEETS
    )


def correlation_numerators(labels, sample):
    n1, n2, n3 = labels
    b_value, p1, p2, p3 = sample
    return tuple(
        grid.boundary.numerator_product(
            n1,
            second_label_sign * n2,
            third_label_sign * n3,
            p1,
            second_momentum_sign * p2,
            third_momentum_sign * p3,
            b_value,
        )
        for (
            second_label_sign,
            third_label_sign,
            second_momentum_sign,
            third_momentum_sign,
        ) in CORRELATION_SECTORS
    )


def sparse_correlation_fit(labels, targets):
    rows = [correlation_numerators(labels, sample) for sample in FIT_SAMPLES]
    matches = []
    for first, second in itertools.combinations(range(len(CORRELATION_SECTORS)), 2):
        leading = sp.Matrix(
            [
                [rows[0][first], rows[0][second]],
                [rows[1][first], rows[1][second]],
            ]
        )
        if leading.det() == 0:
            continue
        coefficients = leading.inv() * sp.Matrix(targets[:2])
        if all(
            sp.factor(
                sp.cancel(
                    coefficients[0] * row[first]
                    + coefficients[1] * row[second]
                    - target
                )
            )
            == 0
            for row, target in zip(rows, targets)
        ):
            matches.append(
                (
                    CORRELATION_SECTORS[first],
                    CORRELATION_SECTORS[second],
                    tuple(sp.factor(value) for value in coefficients),
                )
            )
    print("two-correlation-sector matches:")
    for match in matches:
        print(" ", match)


def normalized_raw(labels, discrete, sample):
    n1, n2, n3 = labels
    epsilon2, epsilon3, form_parity, eta = discrete
    b_value, p1, p2, p3 = sample
    _, raw = grid.enlarged_raw_three_point(
        n1,
        n2,
        n3,
        epsilon2,
        epsilon3,
        form_parity,
        eta,
        b_value,
        p1,
        p2,
        p3,
    )
    norms = grid.raw_norms(
        n1,
        n2,
        n3,
        epsilon2,
        epsilon3,
        b_value,
        p1,
        p2,
        p3,
    )
    return raw / sp.sqrt(sp.prod(norms))


def normalized_signed_products(labels, sample, signed_legs):
    n1, n2, n3 = labels
    b_value, p1, p2, p3 = sample
    answer = []
    for second_sheet, third_sheet in SHEETS:
        momenta = (p1, second_sheet * p2, third_sheet * p3)
        numerator = grid.boundary.numerator_product(
            n1, n2, n3, *momenta, b_value
        )
        if signed_legs:
            denominator_momenta = momenta
        else:
            denominator_momenta = (p1, p2, p3)
        denominator = sp.sqrt(
            sp.prod(
                grid.boundary.leg_product(momentum, label, b_value)
                for momentum, label in zip(denominator_momenta, labels)
            )
        )
        answer.append(numerator / denominator)
    return tuple(answer)


def one_sided_signed_products(labels, sample):
    n1, n2, n3 = labels
    b_value, p1, p2, p3 = sample
    q_value = b_value + 1 / b_value
    answer = []
    for second_sheet, third_sheet in SHEETS:
        momenta = (p1, second_sheet * p2, third_sheet * p3)
        numerator = grid.boundary.numerator_product(
            n1, n2, n3, *momenta, b_value
        )
        denominator = sp.prod(
            grid.boundary.ell(
                q_value + 2 * momentum, int(4 * label), b_value
            )
            for momentum, label in zip(momenta, labels)
        )
        answer.append(sp.cancel(numerator / denominator))
    return tuple(answer)


def one_sided_correlation_products(labels, sample):
    n1, n2, n3 = labels
    b_value, p1, p2, p3 = sample
    q_value = b_value + 1 / b_value
    answer = []
    for (
        second_label_sign,
        third_label_sign,
        second_momentum_sign,
        third_momentum_sign,
    ) in CORRELATION_SECTORS:
        signed_labels = (
            n1,
            second_label_sign * n2,
            third_label_sign * n3,
        )
        momenta = (p1, second_momentum_sign * p2, third_momentum_sign * p3)
        numerator = grid.boundary.numerator_product(
            *signed_labels, *momenta, b_value
        )
        denominator = sp.prod(
            grid.boundary.ell(
                q_value + 2 * momentum, int(4 * label), b_value
            )
            for momentum, label in zip(momenta, labels)
        )
        answer.append(sp.cancel(numerator / denominator))
    return tuple(answer)


def one_screening_factor(x_value, screening_number, b_value, sector):
    """The 2013 screening product, also for half-integral arguments.

    The strict inequality is ``r+s < 2 screening_number``.  The old
    exploratory version used the same cutoff in the even and odd sectors;
    that omitted the last odd diagonal and was therefore not the product
    conjectured in the paper.  A negative half-integral argument is
    reflected in the same way as an integral one.  Its undetermined
    momentum-independent reflection phase is deliberately set to one,
    since every fit below is only up to kappa.
    """

    screening_number = sp.Rational(screening_number)
    if screening_number < 0:
        reflected = one_screening_factor(
            b_value + 1 / b_value - x_value,
            -screening_number,
            b_value,
            sector,
        )
        if sector == "even" and screening_number.is_integer:
            reflected *= (-1) ** int(-screening_number)
        return reflected
    threshold = int(2 * screening_number)
    answer = sp.Integer(1)
    required_parity = 0 if sector == "even" else 1
    for r in range(threshold):
        for s in range(threshold - r):
            if (r + s) % 2 == required_parity:
                answer *= x_value + r * b_value + s / b_value
    return sp.factor(answer)


def ramond_screening_sector(labels):
    """Return the even/odd 2013 sector for one NS and two R labels."""

    total = sp.Rational(2 * sum(labels))
    if not total.is_integer:
        raise ValueError(f"Nonintegral screening parity {total}.")
    return "even" if int(total) % 2 == 0 else "odd"


def generalized_screening_product(labels, momenta, b_value):
    """Normalized product with the 2013 half-integral R continuation."""

    sector = ramond_screening_sector(labels)
    numerator = screening_numerator(labels, momenta, b_value, sector)
    denominator = sp.sqrt(
        sp.prod(
            grid.boundary.leg_product(momentum, label, b_value)
            for momentum, label in zip(momenta, labels)
        )
    )
    return sp.cancel(numerator / denominator)


def generalized_signed_products(labels, sample, reflect_momenta):
    """Four choices of the two Ramond branch signs."""

    n1, n2, n3 = labels
    b_value, p1, p2, p3 = sample
    answer = []
    for second_sign, third_sign in SHEETS:
        signed_labels = (n1, second_sign * n2, third_sign * n3)
        momenta = (
            p1,
            second_sign * p2 if reflect_momenta else p2,
            third_sign * p3 if reflect_momenta else p3,
        )
        answer.append(
            generalized_screening_product(signed_labels, momenta, b_value)
        )
    return tuple(answer)


def screening_numerator(labels, momenta, b_value, sector):
    n1, n2, n3 = labels
    p1, p2, p3 = momenta
    q_value = b_value + 1 / b_value
    answer = sp.Integer(1)
    for second_sign, third_sign in SHEETS:
        argument = (
            q_value / 2
            + p1
            + second_sign * p2
            + third_sign * p3
        )
        screening_number = (
            n1 + second_sign * n2 + third_sign * n3
        )
        answer *= one_screening_factor(
            argument, screening_number, b_value, sector
        )
    return sp.factor(answer)


def one_sided_screening_sectors(labels, sample, sector):
    n1, n2, n3 = labels
    b_value, p1, p2, p3 = sample
    q_value = b_value + 1 / b_value
    answer = []
    for (
        second_label_sign,
        third_label_sign,
        second_momentum_sign,
        third_momentum_sign,
    ) in CORRELATION_SECTORS:
        signed_labels = (
            n1,
            second_label_sign * n2,
            third_label_sign * n3,
        )
        momenta = (p1, second_momentum_sign * p2, third_momentum_sign * p3)
        numerator = screening_numerator(
            signed_labels, momenta, b_value, sector
        )
        denominator = sp.prod(
            grid.boundary.ell(
                q_value + 2 * momentum, int(4 * label), b_value
            )
            for momentum, label in zip(momenta, labels)
        )
        answer.append(sp.cancel(numerator / denominator))
    return tuple(answer)


def screening_sparse_fit(labels, discrete):
    samples = FIT_SAMPLES[:10]
    raw_targets = []
    odd_rows = []
    even_rows = []
    for sample in samples:
        n1, n2, n3 = labels
        epsilon2, epsilon3, form_parity, eta = discrete
        raw_targets.append(
            grid.enlarged_raw_three_point(
                n1,
                n2,
                n3,
                epsilon2,
                epsilon3,
                form_parity,
                eta,
                *sample,
            )[1]
        )
        odd_rows.append(one_sided_screening_sectors(labels, sample, "odd"))
        even_rows.append(one_sided_screening_sectors(labels, sample, "even"))
    matches = []
    for odd_index, even_index in itertools.product(range(16), repeat=2):
        leading = sp.Matrix(
            [
                [odd_rows[0][odd_index], even_rows[0][even_index]],
                [odd_rows[1][odd_index], even_rows[1][even_index]],
            ]
        )
        if leading.det() == 0:
            continue
        coefficients = leading.inv() * sp.Matrix(raw_targets[:2])
        if all(
            sp.factor(
                sp.cancel(
                    coefficients[0] * odd_row[odd_index]
                    + coefficients[1] * even_row[even_index]
                    - target
                )
            )
            == 0
            for odd_row, even_row, target in zip(
                odd_rows, even_rows, raw_targets
            )
        ):
            matches.append(
                (
                    CORRELATION_SECTORS[odd_index],
                    CORRELATION_SECTORS[even_index],
                    tuple(sp.factor(value) for value in coefficients),
                )
            )
    print("odd/even screening-sector matches:")
    for match in matches:
        print(" ", match)


def cross_term_sparse_fit(labels):
    samples = FIT_SAMPLES[:10]
    remainders = []
    candidate_rows = []
    candidate_names = []
    for sector in CORRELATION_SECTORS:
        for leg, momentum_sign in itertools.product((2, 3), (1, -1)):
            candidate_names.append((sector, leg, momentum_sign))
    for sample in samples:
        b_value, p1, p2, p3 = sample
        q_value = b_value + 1 / b_value
        minus = scaled_raw(labels, (0, 0, 0, -1), sample)
        plus = scaled_raw(labels, (0, 0, 0, 1), sample)
        remainders.append(sp.cancel(minus + sp.I * plus))
        row = []
        for (
            second_label_sign,
            third_label_sign,
            second_momentum_sign,
            third_momentum_sign,
        ), leg, leg_momentum_sign in candidate_names:
            signed_labels = (
                labels[0],
                second_label_sign * labels[1],
                third_label_sign * labels[2],
            )
            momenta = (
                p1,
                second_momentum_sign * p2,
                third_momentum_sign * p3,
            )
            even_product = screening_numerator(
                signed_labels, momenta, b_value, "even"
            )
            leg_label = labels[leg - 1]
            leg_momentum = (p2, p3)[leg - 2]
            leg_factor = grid.boundary.ell(
                q_value + 2 * leg_momentum_sign * leg_momentum,
                int(4 * leg_label),
                b_value,
            )
            row.append(sp.cancel(even_product * leg_factor))
        candidate_rows.append(tuple(row))

    matches = []
    for first, second in itertools.combinations(range(len(candidate_names)), 2):
        leading = sp.Matrix(
            [
                [candidate_rows[0][first], candidate_rows[0][second]],
                [candidate_rows[1][first], candidate_rows[1][second]],
            ]
        )
        if leading.det() == 0:
            continue
        coefficients = leading.inv() * sp.Matrix(remainders[:2])
        if all(
            sp.factor(
                sp.cancel(
                    coefficients[0] * row[first]
                    + coefficients[1] * row[second]
                    - target
                )
            )
            == 0
            for row, target in zip(candidate_rows, remainders)
        ):
            matches.append(
                (
                    candidate_names[first],
                    candidate_names[second],
                    tuple(sp.factor(value) for value in coefficients),
                )
            )
    print("two oriented cross-term matches:")
    for match in matches:
        print(" ", match)


def finite_product(labels, momenta, b_value):
    numerator = grid.boundary.numerator_product(*labels, *momenta, b_value)
    denominator = sp.sqrt(
        sp.prod(
            grid.boundary.leg_product(momentum, label, b_value)
            for momentum, label in zip(momenta, labels)
        )
    )
    return numerator / denominator


def branch_signed_products(labels, sample, reflect_momenta):
    n1, n2, n3 = labels
    b_value, p1, p2, p3 = sample
    answer = []
    for second_sign, third_sign in SHEETS:
        signed_labels = (n1, second_sign * n2, third_sign * n3)
        if reflect_momenta:
            momenta = (p1, second_sign * p2, third_sign * p3)
        else:
            momenta = (p1, p2, p3)
        answer.append(finite_product(signed_labels, momenta, b_value))
    return tuple(answer)


def numerical_linear_fit(rows, targets):
    numerical_matrix = sp.Matrix(
        [[complex(sp.N(value, 30)) for value in row] for row in rows]
    )
    numerical_target = sp.Matrix(
        [complex(sp.N(value, 30)) for value in targets]
    )
    if numerical_matrix[:4, :].rank() < numerical_matrix.cols:
        return (), float("inf")
    fitted = numerical_matrix[:4, :].inv() * numerical_target[:4, :]
    residual = numerical_matrix * fitted - numerical_target
    return (
        tuple(complex(value) for value in fitted),
        max(abs(complex(value)) for value in residual),
    )


def fit(labels, discrete):
    rows = [signed_products(labels, sample) for sample in FIT_SAMPLES]
    targets = [scaled_raw(labels, discrete, sample) for sample in FIT_SAMPLES]
    matrix = sp.Matrix(rows)
    target = sp.Matrix(targets)
    coefficients = sp.linsolve((matrix, target))
    print("labels=", labels, "discrete=", discrete)
    print("rank=", matrix.rank(), "augmented rank=", matrix.row_join(target).rank())
    print("coefficients=", coefficients)

    branch_rows = [
        branch_signed_numerators(labels, sample) for sample in FIT_SAMPLES
    ]
    branch_matrix = sp.Matrix(branch_rows)
    branch_coefficients = sp.linsolve((branch_matrix, target))
    print(
        "branch-sign rank=",
        branch_matrix.rank(),
        "augmented rank=",
        branch_matrix.row_join(target).rank(),
    )
    print("branch-sign coefficients=", branch_coefficients)
    sparse_correlation_fit(labels, targets)

    raw_targets = []
    one_sided_rows = []
    for sample in FIT_SAMPLES:
        n1, n2, n3 = labels
        epsilon2, epsilon3, form_parity, eta = discrete
        raw_targets.append(
            grid.enlarged_raw_three_point(
                n1,
                n2,
                n3,
                epsilon2,
                epsilon3,
                form_parity,
                eta,
                *sample,
            )[1]
        )
        one_sided_rows.append(one_sided_signed_products(labels, sample))
    one_sided_matrix = sp.Matrix(one_sided_rows)
    one_sided_target = sp.Matrix(raw_targets)
    print(
        "one-sided rank=",
        one_sided_matrix.rank(),
        "augmented rank=",
        one_sided_matrix.row_join(one_sided_target).rank(),
    )
    print(
        "one-sided coefficients=",
        sp.linsolve((one_sided_matrix, one_sided_target)),
    )
    correlation_rows = [
        one_sided_correlation_products(labels, sample)
        for sample in FIT_SAMPLES
    ]
    numerical_correlation = np.array(
        [
            [complex(sp.N(value, 30)) for value in row]
            for row in correlation_rows
        ],
        dtype=complex,
    )
    numerical_target = np.array(
        [complex(sp.N(value, 30)) for value in one_sided_target],
        dtype=complex,
    )
    numerical_coefficients, _, numerical_rank, singular_values = np.linalg.lstsq(
        numerical_correlation, numerical_target, rcond=1e-12
    )
    numerical_residual = np.max(
        np.abs(numerical_correlation @ numerical_coefficients - numerical_target)
    )
    print(
        "one-sided correlation numerical rank=",
        numerical_rank,
        "max residual=",
        numerical_residual,
        "smallest singular value=",
        singular_values[-1],
    )
    print("one-sided correlation coefficients=", tuple(numerical_coefficients))

    for signed_legs in (False, True):
        normalized_rows = [
            normalized_signed_products(labels, sample, signed_legs)
            for sample in FIT_SAMPLES
        ]
        normalized_targets = [
            normalized_raw(labels, discrete, sample) for sample in FIT_SAMPLES
        ]
        fitted, residual = numerical_linear_fit(
            normalized_rows, normalized_targets
        )
        print(
            "normalized signed legs=",
            signed_legs,
            "coefficients=",
            fitted,
            "max residual=",
            residual,
        )


def reflection_search(labels, discrete):
    """Search direct B^2 for the reflected representative of one sector."""

    target_values = [
        grid.normalized_branching_square(labels, discrete, sample)[1]
        for sample in grid.SAMPLES
    ]
    matches = []
    n1, n2, n3 = labels
    for second_label_sign, third_label_sign in SHEETS:
        candidate_labels = (
            n1,
            second_label_sign * n2,
            third_label_sign * n3,
        )
        for second_momentum_sign, third_momentum_sign in SHEETS:
            candidate_samples = [
                (
                    sample[0],
                    sample[1],
                    second_momentum_sign * sample[2],
                    third_momentum_sign * sample[3],
                )
                for sample in grid.SAMPLES
            ]
            for epsilon2, epsilon3, form_parity in itertools.product(
                (0, 1), repeat=3
            ):
                candidate_discrete = (
                    epsilon2,
                    epsilon3,
                    form_parity,
                    1,
                )
                candidate_values = [
                    grid.normalized_branching_square(
                        candidate_labels, candidate_discrete, sample
                    )[1]
                    for sample in candidate_samples
                ]
                if any(value == 0 for value in candidate_values):
                    continue
                ratios = [
                    sp.factor(sp.cancel(target / candidate))
                    for target, candidate in zip(target_values, candidate_values)
                ]
                if sp.factor(sp.cancel(ratios[0] - ratios[1])) == 0:
                    matches.append(
                        (
                            (second_label_sign, third_label_sign),
                            (second_momentum_sign, third_momentum_sign),
                            (epsilon2, epsilon3, form_parity),
                            ratios[0],
                        )
                    )
    print("reflection matches:")
    for match in matches:
        print(" ", match)

def main():
    labels = (sp.Integer(0), sp.Rational(3, 4), sp.Rational(3, 4))
    cross_term_sparse_fit(labels)


if __name__ == "__main__":
    main()
