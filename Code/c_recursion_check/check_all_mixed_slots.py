#!/usr/bin/env python3
"""Low-level certificates for ordered and self-plumbed three-point forms.

The calculations use the SCblock ground-state convention

    rho_0^eta(++ ) = 1,  rho_0^eta(-- ) = eta,
    rho_1^eta(+- ) = 1, rho_1^eta(-+ ) = i eta,

with the Ramond signs read from left to right.  In particular, the script
does not assume that a cyclically reordered eta-basis is unchanged.
"""

import sympy as sp


def assert_zero(label: str, value: sp.Expr) -> None:
    value = sp.factor(sp.cancel(value))
    if value != 0:
        raise AssertionError(f"{label}: expected zero, got {value}")


I = sp.I
u = (1 + I) / sp.sqrt(2)


def check_sl2_mobius_representatives() -> None:
    """Check determinant, fractional-linear action, and slot permutation."""

    z = sp.symbols("z")
    representatives = (
        (sp.Matrix([[0, 1], [-1, 1]]), 1 / (1 - z), (2, 3, 1)),
        (sp.Matrix([[1, -1], [1, 0]]), 1 - 1 / z, (3, 1, 2)),
        (sp.Matrix([[-I, I], [0, I]]), 1 - z, (1, 3, 2)),
    )
    old_points = (sp.oo, sp.Integer(1), sp.Integer(0))
    displayed_points = (sp.oo, sp.Integer(1), sp.Integer(0))

    def image_of(matrix: sp.Matrix, point: sp.Expr) -> sp.Expr:
        a, b, c, d = matrix[0, 0], matrix[0, 1], matrix[1, 0], matrix[1, 1]
        if point == sp.oo:
            return sp.oo if c == 0 else sp.simplify(a / c)
        denominator = sp.simplify(c * point + d)
        if denominator == 0:
            return sp.oo
        return sp.simplify((a * point + b) / denominator)

    for number, (matrix, expected_action, expected_order) in enumerate(
        representatives, start=1
    ):
        assert_zero(f"SL2 representative {number} determinant", matrix.det() - 1)
        action = sp.cancel(
            (matrix[0, 0] * z + matrix[0, 1])
            / (matrix[1, 0] * z + matrix[1, 1])
        )
        assert_zero(
            f"SL2 representative {number} action", action - expected_action
        )
        images = tuple(image_of(matrix, point) for point in old_points)
        transported_order = tuple(images.index(point) + 1 for point in displayed_points)
        if transported_order != expected_order:
            raise AssertionError(
                f"SL2 representative {number} slot order: "
                f"expected {expected_order}, got {transported_order}"
            )


def ground(form_parity: int, eta: int, left_sign: int, right_sign: int) -> sp.Expr:
    """SCblock ground tensor; + is 0 and - is 1."""

    if form_parity == 0:
        if (left_sign, right_sign) == (0, 0):
            return sp.Integer(1)
        if (left_sign, right_sign) == (1, 1):
            return sp.Integer(eta)
        return sp.Integer(0)
    if (left_sign, right_sign) == (0, 1):
        return sp.Integer(1)
    if (left_sign, right_sign) == (1, 0):
        return I * eta
    return sp.Integer(0)


def ns_ground(
    form_parity: int,
    first_fermion: int,
    second_fermion: int,
    third_fermion: int,
    h1: sp.Expr,
    h2: sp.Expr,
    h3: sp.Expr,
) -> sp.Expr:
    """The eight SCblock global NS components with no L_-1 modes."""

    occupation = (first_fermion, second_fermion, third_fermion)
    if form_parity == 0:
        return {
            (0, 0, 0): sp.Integer(1),
            (1, 1, 0): h1 + h2 - h3,
            (1, 0, 1): h1 - h2 + h3,
            (0, 1, 1): h1 - h2 - h3,
        }.get(occupation, sp.Integer(0))
    return {
        (1, 0, 0): sp.Integer(1),
        (0, 1, 0): sp.Integer(1),
        (0, 0, 1): sp.Integer(1),
        (1, 1, 1): h1 + h2 + h3 - sp.Rational(1, 2),
    }.get(occupation, sp.Integer(0))


def all_ns_polynomial(
    r: int,
    s: int,
    form_parity: int,
    lambda_i: sp.Expr,
    lambda_j: sp.Expr,
    b: sp.Expr,
) -> sp.Expr:
    """P_rs^f from the two selected NS fusion lattices."""

    result = sp.Integer(1)
    required_residue = 2 if form_parity == 0 else 0
    for p in range(1 - r, r, 2):
        for q in range(1 - s, s, 2):
            if (p + q - (r + s)) % 4 != required_residue:
                continue
            lattice = p * b + q / b
            result *= (lambda_i - lambda_j + lattice) / (2 * sp.sqrt(2))
            result *= (lambda_i + lambda_j + lattice) / (2 * sp.sqrt(2))
    return sp.factor(result)


def check_double_ns_restrictions() -> None:
    """Check the three self-plumbed NS pairs and restriction-order signs."""

    h1, h2, h3 = sp.symbols("h1 h2 h3")
    direct_checks = (
        (
            "double NS slots 1,2 f=0",
            ns_ground(0, 1, 1, 0, 0, 0, h3),
            -h3,
        ),
        (
            "double NS slots 1,2 f=1",
            ns_ground(1, 1, 1, 1, 0, 0, h3),
            h3 - sp.Rational(1, 2),
        ),
        (
            "double NS slots 1,3 f=0",
            ns_ground(0, 1, 0, 1, 0, h2, 0),
            -h2,
        ),
        (
            "double NS slots 1,3 f=1",
            ns_ground(1, 1, 1, 1, 0, h2, 0),
            h2 - sp.Rational(1, 2),
        ),
        (
            "double NS slots 2,3 f=0",
            ns_ground(0, 0, 1, 1, h1, 0, 0),
            h1,
        ),
        (
            "double NS slots 2,3 f=1",
            ns_ground(1, 1, 1, 1, h1, 0, 0),
            h1 - sp.Rational(1, 2),
        ),
    )
    for label, direct, expected in direct_checks:
        assert_zero(label, direct - expected)

    b, lambda_2 = sp.symbols("b lambda_2", nonzero=True)
    for r, s in ((1, 1), (3, 1), (1, 3), (2, 2), (5, 1)):
        null_parity = (r * s) % 2
        lambda_null = r * b + s / b
        lambda_shifted = r * b - s / b
        for form_parity in (0, 1):
            first_order = (
                (-1) ** null_parity
                * all_ns_polynomial(
                    r, s, form_parity, lambda_null, lambda_2, b
                )
                * all_ns_polynomial(
                    r,
                    s,
                    (form_parity + null_parity) % 2,
                    lambda_2,
                    lambda_shifted,
                    b,
                )
            )
            reverse_order = all_ns_polynomial(
                r, s, form_parity, lambda_2, lambda_null, b
            ) * all_ns_polynomial(
                r,
                s,
                (form_parity + null_parity) % 2,
                lambda_shifted,
                lambda_2,
                b,
            )
            assert_zero(
                f"double NS order independence ({r},{s}) f={form_parity}",
                first_order - reverse_order,
            )


def check_compact_component_sums() -> None:
    """Check that the scalar epsilon-sums reproduce every former 2x2 map."""

    p_plus, p_minus = sp.symbols("P_plus P_minus")
    polynomial = {1: p_plus, -1: p_minus}

    even_matrix = sp.Matrix(
        [
            [p_plus + p_minus, -I * (p_plus - p_minus)],
            [I * (p_plus - p_minus), p_plus + p_minus],
        ]
    ) / 2
    odd_zero_matrix = sp.Matrix(
        [
            [p_plus + p_minus, I * (p_plus - p_minus)],
            [I * (p_plus - p_minus), -(p_plus + p_minus)],
        ]
    ) / 2
    odd_one_matrix = sp.Matrix(
        [
            [I * (p_plus + p_minus), -(p_plus - p_minus)],
            [-(p_plus - p_minus), -I * (p_plus + p_minus)],
        ]
    ) / 2

    for row, eta in enumerate((1, -1)):
        same = row
        opposite = 1 - row
        compact_even = [sp.Integer(0), sp.Integer(0)]
        compact_odd_zero = [sp.Integer(0), sp.Integer(0)]
        compact_odd_one = [sp.Integer(0), sp.Integer(0)]
        for epsilon in (1, -1):
            compact_even[same] += polynomial[epsilon] / 2
            compact_even[opposite] += -I * eta * epsilon * polynomial[epsilon] / 2
            compact_odd_zero[same] += eta * polynomial[epsilon] / 2
            compact_odd_zero[opposite] += I * epsilon * polynomial[epsilon] / 2
            compact_odd_one[same] += I * eta * polynomial[epsilon] / 2
            compact_odd_one[opposite] += -epsilon * polynomial[epsilon] / 2
        for column in range(2):
            assert_zero(
                f"compact even component sum entry={row},{column}",
                compact_even[column] - even_matrix[row, column],
            )
            assert_zero(
                f"compact odd f=0 component sum entry={row},{column}",
                compact_odd_zero[column] - odd_zero_matrix[row, column],
            )
            assert_zero(
                f"compact odd f=1 component sum entry={row},{column}",
                compact_odd_one[column] - odd_one_matrix[row, column],
            )


def g0(beta: sp.Expr, sign: int) -> tuple[sp.Expr, int]:
    """Return the coefficient and flipped sign in G_0 w^sign."""

    return (u * beta if sign == 0 else I * u * beta, 1 - sign)


def middle_g_half(
    form_parity: int,
    eta: int,
    beta_left: sp.Expr,
    beta_right: sp.Expr,
    left_sign: int,
    right_sign: int,
) -> sp.Expr:
    """Ward identity for rho(R,G_{-1/2}phi,R) at (infinity,1,0)."""

    left_coefficient, flipped_left = g0(beta_left, left_sign)
    right_coefficient, flipped_right = g0(beta_right, right_sign)
    return sp.expand(
        left_coefficient * ground(form_parity, eta, flipped_left, right_sign)
        - (-1) ** form_parity
        * right_coefficient
        * ground(form_parity, eta, left_sign, flipped_right)
    )


def decompose(
    target_parity: int,
    plus_component: sp.Expr,
    minus_component: sp.Expr,
) -> tuple[sp.Expr, sp.Expr]:
    """Decompose two allowed components in the eta=+,- ground basis."""

    if target_parity == 0:
        # Components are ++ and --, with columns (1,1) and (1,-1).
        return (
            sp.expand((plus_component + minus_component) / 2),
            sp.expand((plus_component - minus_component) / 2),
        )
    # Components are +- and -+, with columns (1,i) and (1,-i).
    return (
        sp.expand((plus_component - I * minus_component) / 2),
        sp.expand((plus_component + I * minus_component) / 2),
    )


def check_middle_ns_level_half() -> None:
    """Check the complete 2x2 factorization at the (1,1) NS pole."""

    x, y = sp.symbols("beta_left beta_right")
    p_plus = x - y
    p_minus = x + y

    expected = {
        0: u
        * sp.Matrix(
            [
                [(p_plus + p_minus) / 2, I * (p_plus - p_minus) / 2],
                [I * (p_plus - p_minus) / 2, -(p_plus + p_minus) / 2],
            ]
        ),
        1: u
        * sp.Matrix(
            [
                [I * (p_plus + p_minus) / 2, -(p_plus - p_minus) / 2],
                [-(p_plus - p_minus) / 2, -I * (p_plus + p_minus) / 2],
            ]
        ),
    }

    for form_parity in (0, 1):
        target_parity = 1 - form_parity
        rows = []
        for eta in (1, -1):
            if form_parity == 0:
                value_1 = middle_g_half(form_parity, eta, x, y, 0, 1)
                value_2 = middle_g_half(form_parity, eta, x, y, 1, 0)
            else:
                value_1 = middle_g_half(form_parity, eta, x, y, 0, 0)
                value_2 = middle_g_half(form_parity, eta, x, y, 1, 1)
            rows.append(decompose(target_parity, value_1, value_2))
        direct = sp.Matrix(rows)
        for row in range(2):
            for column in range(2):
                assert_zero(
                    f"middle NS (1,1) f={form_parity} entry={row},{column}",
                    direct[row, column] - expected[form_parity][row, column],
                )


def check_ns_third_slot_basis_conversion() -> None:
    """Convert the literature R--R--NS basis to SCblock left-to-right order."""

    p_plus, p_minus = sp.symbols("P_plus P_minus")
    phase = (1 + I) / sp.sqrt(2)
    phase_inverse = (1 - I) / sp.sqrt(2)
    # HJS R--R--NS odd forms obey H_eta=i*eta*SC_{-eta}.
    hjs_from_sc_odd = sp.Matrix([[0, I], [-I, 0]])
    diagonal = sp.diag(p_plus, p_minus)

    # Even rs: f=0 keeps eta; f=1 uses the opposite polynomial label.
    converted_even_f1 = sp.simplify(
        hjs_from_sc_odd * diagonal * hjs_from_sc_odd
    )
    expected_even_f1 = sp.diag(p_minus, p_plus)
    for row in range(2):
        for column in range(2):
            assert_zero(
                f"R--R--NS even basis conversion entry={row},{column}",
                converted_even_f1[row, column] - expected_even_f1[row, column],
            )

    # Odd rs, f=0: eta*e^{i*pi/4} P_eta and no eta flip in SCblock.
    hjs_odd_f0 = sp.Matrix(
        [[0, -phase_inverse * p_plus], [-phase_inverse * p_minus, 0]]
    )
    converted_odd_f0 = sp.simplify(hjs_odd_f0 * hjs_from_sc_odd)
    expected_odd_f0 = sp.diag(phase * p_plus, -phase * p_minus)

    # Odd rs, f=1: eta*e^{-i*pi/4} P_{-eta}.
    hjs_odd_f1 = sp.Matrix(
        [[0, -phase * p_plus], [-phase * p_minus, 0]]
    )
    converted_odd_f1 = sp.simplify(hjs_from_sc_odd * hjs_odd_f1)
    expected_odd_f1 = sp.diag(phase_inverse * p_minus, -phase_inverse * p_plus)
    for name, direct, expected in (
        ("f=0", converted_odd_f0, expected_odd_f0),
        ("f=1", converted_odd_f1, expected_odd_f1),
    ):
        for row in range(2):
            for column in range(2):
                assert_zero(
                    f"R--R--NS odd basis conversion {name} entry={row},{column}",
                    direct[row, column] - expected[row, column],
                )


def middle_g_three_halves_g_half(
    form_parity: int,
    eta: int,
    beta_left: sp.Expr,
    beta_right: sp.Expr,
    h_left: sp.Expr,
    h_middle: sp.Expr,
    h_right: sp.Expr,
    left_sign: int,
    right_sign: int,
) -> sp.Expr:
    """Ward reduction of rho(R,G_{-3/2}G_{-1/2}phi,R)."""

    exponent = h_left - h_middle - h_right
    right_coefficient, flipped_right = g0(beta_right, right_sign)
    g_half_with_right_g0 = right_coefficient * middle_g_half(
        form_parity,
        eta,
        beta_left,
        beta_right,
        left_sign,
        flipped_right,
    )
    return sp.expand(
        (-1) ** (form_parity + 1) * g_half_with_right_g0
        - exponent * ground(form_parity, eta, left_sign, right_sign) / 2
        + h_middle * ground(form_parity, eta, left_sign, right_sign) / 4
    )


def ns_22_polynomial(
    eta: int,
    beta_left: sp.Expr,
    beta_right: sp.Expr,
    b: sp.Expr,
) -> sp.Expr:
    """P_22^{NS,eta}(beta_right,beta_left) in the note's convention."""

    result = sp.Integer(1)
    for k in range(2):
        for ell in range(2):
            lattice = (-1 + 2 * k) * b + (-1 + 2 * ell) / b
            combination = (
                beta_left - eta * beta_right
                if (k + ell) % 2 == 0
                else beta_left + eta * beta_right
            )
            result *= (2 * sp.sqrt(2) * combination - lattice) / (2 * sp.sqrt(2))
    return sp.factor(result)


def ns_polynomial(
    r: int,
    s: int,
    eta: int,
    beta_left: sp.Expr,
    beta_right: sp.Expr,
    b: sp.Expr,
) -> sp.Expr:
    """P_rs^{NS,eta}(beta_right,beta_left)."""

    result = sp.Integer(1)
    for k in range(r):
        for ell in range(s):
            lattice = (1 - r + 2 * k) * b + (1 - s + 2 * ell) / b
            combination = (
                beta_left - eta * beta_right
                if (k + ell) % 2 == 0
                else beta_left + eta * beta_right
            )
            result *= (2 * sp.sqrt(2) * combination - lattice) / (2 * sp.sqrt(2))
    return sp.factor(result)


def check_middle_ns_level_two() -> None:
    """Compute the first even NS singular vector in the middle slot."""

    b, x, y = sp.symbols("b beta_left beta_right", nonzero=True)
    c = sp.Rational(3, 2) + 3 * (b + 1 / b) ** 2
    h_middle = (3 - 2 * c) / 16
    h_left = c / 24 - x**2
    h_right = c / 24 - y**2
    exponent = sp.factor(h_left - h_middle - h_right)
    l_minus_two = h_middle + 2 * h_right - h_left

    # chi_22 = [L_-1^2-(4h/3)L_-2-G_-3/2 G_-1/2] phi.
    p_plus = ns_22_polynomial(1, x, y, b)
    p_minus = ns_22_polynomial(-1, x, y, b)
    polynomial_sum = p_plus + p_minus
    polynomial_difference = p_plus - p_minus

    expected = {
        0: sp.Matrix(
            [
                [polynomial_sum / 2, -I * polynomial_difference / 2],
                [I * polynomial_difference / 2, polynomial_sum / 2],
            ]
        ),
        1: sp.Matrix(
            [
                [polynomial_sum / 2, -I * polynomial_difference / 2],
                [I * polynomial_difference / 2, polynomial_sum / 2],
            ]
        ),
    }

    for form_parity in (0, 1):
        rows = []
        component_signs = (
            ((0, 0), (1, 1))
            if form_parity == 0
            else ((0, 1), (1, 0))
        )
        for eta in (1, -1):
            values = []
            for left_sign, right_sign in component_signs:
                base = ground(form_parity, eta, left_sign, right_sign)
                g_term = middle_g_three_halves_g_half(
                    form_parity,
                    eta,
                    x,
                    y,
                    h_left,
                    h_middle,
                    h_right,
                    left_sign,
                    right_sign,
                )
                values.append(
                    sp.expand(
                        exponent * (exponent - 1) * base
                        - 4 * h_middle * l_minus_two * base / 3
                        - g_term
                    )
                )
            rows.append(decompose(form_parity, values[0], values[1]))

        direct = sp.simplify(sp.Matrix(rows))
        for row in range(2):
            for column in range(2):
                assert_zero(
                    f"middle NS (2,2) f={form_parity} entry={row},{column}",
                    direct[row, column] - expected[form_parity][row, column],
                )


def check_middle_ns_level_three_halves() -> None:
    """Check the (3,1) odd NS null, not only the vacuum (1,1) null."""

    b, x, y = sp.symbols("b beta_left beta_right", nonzero=True)
    c = sp.Rational(3, 2) + 3 * (b + 1 / b) ** 2
    h_middle = -b**2 - sp.Rational(1, 2)
    h_left = c / 24 - x**2
    h_right = c / 24 - y**2
    exponent = sp.factor(h_left - h_middle - h_right)
    p_plus = ns_polynomial(3, 1, 1, x, y, b)
    p_minus = ns_polynomial(3, 1, -1, x, y, b)
    polynomial_sum = p_plus + p_minus
    polynomial_difference = p_plus - p_minus

    # The extra minus sign is (-1)^((rs-1)/2) at rs=3.  It is the
    # orientation factor of the cyclic local coordinate at the NS puncture.
    expected = {
        0: -u
        * sp.Matrix(
            [
                [polynomial_sum / 2, I * polynomial_difference / 2],
                [I * polynomial_difference / 2, -polynomial_sum / 2],
            ]
        ),
        1: -u
        * sp.Matrix(
            [
                [I * polynomial_sum / 2, -polynomial_difference / 2],
                [-polynomial_difference / 2, -I * polynomial_sum / 2],
            ]
        ),
    }

    for form_parity in (0, 1):
        target_parity = 1 - form_parity
        component_signs = (
            ((0, 1), (1, 0))
            if form_parity == 0
            else ((0, 0), (1, 1))
        )
        rows = []
        for eta in (1, -1):
            values = []
            for left_sign, right_sign in component_signs:
                g_half = middle_g_half(
                    form_parity, eta, x, y, left_sign, right_sign
                )
                right_coefficient, flipped_right = g0(y, right_sign)
                right_g0 = right_coefficient * ground(
                    form_parity, eta, left_sign, flipped_right
                )
                g_three_halves = (
                    (-1) ** form_parity * right_g0 - g_half / 2
                )
                # chi_31=(L_-1 G_-1/2+b^2 G_-3/2)phi.
                values.append(
                    sp.expand(
                        (exponent - sp.Rational(1, 2)) * g_half
                        + b**2 * g_three_halves
                    )
                )
            rows.append(decompose(target_parity, values[0], values[1]))

        direct = sp.simplify(sp.Matrix(rows))
        for row in range(2):
            for column in range(2):
                assert_zero(
                    f"middle NS (3,1) f={form_parity} entry={row},{column}",
                    direct[row, column] - expected[form_parity][row, column],
                )


def ramond_21_polynomial(
    eta: int, lambda_ns: sp.Expr, beta_spectator: sp.Expr, b: sp.Expr
) -> sp.Expr:
    """The note's P_21^{R,eta}(h_NS,beta_spectator)."""

    return sp.factor(
        (lambda_ns - 2 * sp.sqrt(2) * eta * beta_spectator + b)
        * (lambda_ns + 2 * sp.sqrt(2) * eta * beta_spectator - b)
        / 8
    )


def check_middle_order_ramond_level_one() -> None:
    """Check the six ordered R-pole factors at the first R null level.

    The direct Ward anchor in NS--R--R order is checked in
    check_poles_and_signs.py.  Here we check all remaining changes of
    ordered eta-basis and the R--NS--R orientation sign.
    """

    b, beta_spectator, lambda_ns = sp.symbols(
        "b beta_spectator lambda_ns", nonzero=True
    )
    p_plus = ramond_21_polynomial(1, lambda_ns, beta_spectator, b)
    p_minus = ramond_21_polynomial(-1, lambda_ns, beta_spectator, b)
    diagonal = sp.diag(p_plus, p_minus)
    flipped = sp.diag(p_minus, p_plus)
    component_transpose = sp.Matrix([[0, -I], [I, 0]])
    for row in range(2):
        for column in range(2):
            assert_zero(
                f"R (2,1) f=1 eta-basis transpose entry={row},{column}",
                (component_transpose * diagonal * component_transpose)[row, column]
                - flipped[row, column],
            )

    # At rs/2=1 the R--NS--R fusion map has the extra sign -1.
    middle_matrix = -sp.Matrix(
        [
            [p_plus + p_minus, -I * (p_plus - p_minus)],
            [I * (p_plus - p_minus), p_plus + p_minus],
        ]
    ) / 2
    # Exchanging the two R punctures leaves this matrix unchanged.
    assert_zero(
        "R (2,1) middle-order exchange",
        (component_transpose * middle_matrix * component_transpose)[0, 1]
        - middle_matrix[0, 1],
    )

    # b -> b^{-1} is the independent (1,2) null check.
    p_12_plus = ramond_21_polynomial(
        1, lambda_ns, beta_spectator, 1 / b
    )
    expected_12_plus = (
        (lambda_ns - 2 * sp.sqrt(2) * beta_spectator + 1 / b)
        * (lambda_ns + 2 * sp.sqrt(2) * beta_spectator - 1 / b)
        / 8
    )
    assert_zero("R (1,2) fusion factor", p_12_plus - expected_12_plus)

    # Independent direct R--NS--R Ward reduction of the left R null.
    c = sp.Rational(3, 2) + 3 * (b + 1 / b) ** 2
    beta_null = (2 * b + 1 / b) / (2 * sp.sqrt(2))
    h_null = c / 24 - beta_null**2
    h_spectator = c / 24 - beta_spectator**2
    h_ns = ((b + 1 / b) ** 2 - lambda_ns**2) / 8
    null_coefficient = -6 / (8 * h_null + c)
    for form_parity in (0, 1):
        component_signs = (
            ((0, 0), (1, 1))
            if form_parity == 0
            else ((0, 1), (1, 0))
        )
        rows = []
        for eta in (1, -1):
            values = []
            for left_sign, right_sign in component_signs:
                base = ground(form_parity, eta, left_sign, right_sign)
                l_minus_one = (h_null + h_ns - h_spectator) * base
                g0_coefficient, flipped_left = g0(beta_null, left_sign)
                g_minus_one_g0 = g0_coefficient * middle_g_half(
                    form_parity,
                    eta,
                    beta_null,
                    beta_spectator,
                    flipped_left,
                    right_sign,
                )
                values.append(
                    sp.expand(l_minus_one + null_coefficient * g_minus_one_g0)
                )
            rows.append(decompose(form_parity, values[0], values[1]))
        direct = sp.simplify(sp.Matrix(rows))
        for row in range(2):
            for column in range(2):
                assert_zero(
                    f"direct R--NS--R R21 f={form_parity} entry={row},{column}",
                    direct[row, column] - middle_matrix[row, column],
                )


def check_double_ramond_restrictions() -> None:
    """Check simultaneous (2,1) and (1,2) R nulls in all slot orders."""

    b, lambda_ns = sp.symbols("b lambda_ns", nonzero=True)
    component_transpose = sp.Matrix([[0, -I], [I, 0]])

    for r, s, fusion_shift in ((2, 1, b), (1, 2, 1 / b)):
        beta_null = (r * b + s / b) / (2 * sp.sqrt(2))
        beta_shifted = (
            (-1) ** s * (r * b - s / b) / (2 * sp.sqrt(2))
        )
        p_null = {
            eta: ramond_21_polynomial(
                eta, lambda_ns, beta_null, fusion_shift
            )
            for eta in (1, -1)
        }
        p_shifted = {
            eta: ramond_21_polynomial(
                eta, lambda_ns, beta_shifted, fusion_shift
            )
            for eta in (1, -1)
        }
        q_plus = sp.factor(p_null[1] * p_shifted[1])
        q_minus = sp.factor(p_null[-1] * p_shifted[-1])
        q_diagonal = sp.diag(q_plus, q_minus)
        q_flipped = sp.diag(q_minus, q_plus)

        def middle_map(p_plus: sp.Expr, p_minus: sp.Expr) -> sp.Matrix:
            orientation = (-1) ** (r * s // 2)
            return orientation * sp.Matrix(
                [
                    [p_plus + p_minus, -I * (p_plus - p_minus)],
                    [I * (p_plus - p_minus), p_plus + p_minus],
                ]
            ) / 2

        first_map = middle_map(p_null[1], p_null[-1])
        second_map = middle_map(p_shifted[1], p_shifted[-1])
        expected_middle = sp.Matrix(
            [
                [q_plus + q_minus, -I * (q_plus - q_minus)],
                [I * (q_plus - q_minus), q_plus + q_minus],
            ]
        ) / 2
        for row in range(2):
            for column in range(2):
                assert_zero(
                    f"double R ({r},{s}) middle product entry={row},{column}",
                    (first_map * second_map)[row, column]
                    - expected_middle[row, column],
                )
                assert_zero(
                    f"double R ({r},{s}) restriction commutator entry={row},{column}",
                    (first_map * second_map - second_map * first_map)[row, column],
                )

        # NS--R--R is diagonal for both parities.  In R--R--NS the odd
        # left-to-right basis exchanges the two labels at both restrictions.
        for row in range(2):
            for column in range(2):
                assert_zero(
                    f"double R ({r},{s}) RRN odd basis entry={row},{column}",
                    (component_transpose * q_diagonal * component_transpose)[
                        row, column
                    ]
                    - q_flipped[row, column],
                )


if __name__ == "__main__":
    check_sl2_mobius_representatives()
    check_double_ns_restrictions()
    check_compact_component_sums()
    check_middle_ns_level_half()
    check_ns_third_slot_basis_conversion()
    check_middle_ns_level_two()
    check_middle_ns_level_three_halves()
    check_middle_order_ramond_level_one()
    check_double_ramond_restrictions()
    print("all ordered mixed-slot checks passed")
