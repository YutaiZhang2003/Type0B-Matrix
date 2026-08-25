#!/usr/bin/env python3
"""Symbolic certificates used by agent_notes/c_recursion_derivation.tex.

These checks do not evaluate conformal blocks.  They verify the algebraic
identities behind the NS and Ramond c-pole locations, the two Jacobians, the
Ramond shifted module, the NS-R-R ground-state Ward phases, and the odd-null
theta-channel sewing signs.  They also check that the two Ramond three-point
structure labels at the theta vertices are independent, and that the
all-NS fusion polynomial has the correct spectator order in every slot.
"""

from itertools import product

import sympy as sp


def assert_zero(label: str, value: sp.Expr) -> None:
    value = sp.factor(sp.cancel(value))
    if value != 0:
        raise AssertionError(f"{label}: expected zero, got {value}")


def check_ns_pole() -> None:
    b, h = sp.symbols("b h", nonzero=True)
    r, s = sp.symbols("r s", nonzero=True)
    x = sp.symbols("x", nonzero=True)

    h_rs = ((b + 1 / b) ** 2 - (r * b + s / b) ** 2) / 8
    ns_quadratic = (r**2 - 1) * x**2 + (8 * h + 2 * (r * s - 1)) * x + s**2 - 1
    assert_zero("NS Kac quadratic", ns_quadratic.subs({x: b**2, h: h_rs}))

    discriminant = sp.sqrt(16 * h**2 + 8 * (r * s - 1) * h + (r - s) ** 2)
    x_branch = (4 * h + r * s - 1 + discriminant) / (1 - r**2)
    assert_zero("NS chosen x branch", ns_quadratic.subs(x, x_branch))

    c = sp.Rational(3, 2) + 3 * (b + 1 / b) ** 2
    dc_dh = sp.diff(c, b) / sp.diff(h_rs, b)
    inverse_denominator_slope = -1 / (sp.diff(h_rs, b) / sp.diff(c, b))
    assert_zero("NS minus Jacobian", inverse_denominator_slope + dc_dh)


def check_ns_level_three_halves_null_residue() -> None:
    """Check A_31 from the full level-3/2 NS Gram matrix."""

    b, h = sp.symbols("b h", nonzero=True)
    c = sp.Rational(3, 2) + 3 * (b + 1 / b) ** 2
    h_31 = -b**2 - sp.Rational(1, 2)

    # Basis: (G_{-3/2}|h>, L_{-1}G_{-1/2}|h>).  The null vector is
    # b^2 G_{-3/2}|h> + L_{-1}G_{-1/2}|h>, whose leading
    # G_{-1/2}^3 coefficient is one.
    gram = sp.Matrix(
        [
            [2 * h + 2 * c / 3, 4 * h],
            [4 * h, 4 * h * (h + sp.Rational(1, 2))],
        ]
    )
    null_vector = sp.Matrix([b**2, 1])
    null_norm = sp.expand((null_vector.T * gram * null_vector)[0])
    inverse_slope = sp.factor(1 / sp.diff(null_norm, h).subs(h, h_31))
    product_formula = 1 / (2 * (b - 1) * (b + 1) * (b**2 + 1))
    assert_zero(
        "NS (3,1) inverse null norm",
        inverse_slope - product_formula,
    )


def check_all_ns_slot_fusion_signs() -> None:
    """Check lattice reflection and the three (1,1) slot anchors.

    For the SCblock order (infinity, 1, 0), the first- and second-slot
    factors are P(h3,h2) and P(h3,h1).  The third-slot factor is
    (-1)^(rs(A+C)) P(h1,h2).  The level-1/2 vacuum null checks the graded
    factor, while the level-3/2 (3,1) null distinguishes the natural
    argument order from the reversed one.
    """

    for r in range(1, 7):
        for s in range(1, 7):
            if (r + s) % 2:
                continue
            lattices = {0: [], 1: []}
            for p in range(1 - r, r, 2):
                for q in range(1 - s, s, 2):
                    residue = (p + q - (r + s)) % 4
                    if residue == 2:
                        lattices[0].append((p, q))
                    elif residue == 0:
                        lattices[1].append((p, q))
                    else:
                        raise AssertionError("NS fusion lattice partition")

            expected_counts = {
                0: (r * s + 1) // 2,
                1: (r * s) // 2,
            }
            for form_parity in (0, 1):
                lattice = set(lattices[form_parity])
                if len(lattice) != expected_counts[form_parity]:
                    raise AssertionError("NS fusion lattice count")
                if {(-p, -q) for p, q in lattice} != lattice:
                    raise AssertionError("NS fusion lattice reflection")

    h1, h2, h3 = sp.symbols("h1 h2 h3")

    # P_11^0(hi,hj)=hj-hi and P_11^1=1.
    p11_even = lambda hi, hj: hj - hi
    p11_odd = lambda hi, hj: sp.Integer(1)

    # Direct global Ward identities with chi_11=G_{-1/2}|0>.
    assert_zero("all-NS first-slot (1,1) even anchor", (h2 - h3) - p11_even(h3, h2))
    assert_zero("all-NS first-slot (1,1) odd anchor", 1 - p11_odd(h3, h2))
    assert_zero("all-NS second-slot (1,1) even anchor", (h1 - h3) - p11_even(h3, h1))
    assert_zero("all-NS second-slot (1,1) odd anchor", 1 - p11_odd(h3, h1))
    assert_zero(
        "all-NS third-slot (1,1) even graded anchor",
        (h1 - h2) + p11_even(h1, h2),
    )
    assert_zero(
        "all-NS third-slot (1,1) odd graded anchor",
        1 - p11_odd(h1, h2),
    )

    # chi_31=(L_-1 G_-1/2+b^2 G_-3/2)|-b^2-1/2>.  In the third slot,
    # the two odd-anchor terms are h2-b^2-h1 and b^2.
    b = sp.symbols("b", nonzero=True)
    third_31_odd = h2 - b**2 - h1 + b**2
    p31_odd_natural = h2 - h1
    assert_zero(
        "all-NS third-slot (3,1) natural argument order",
        third_31_odd - p31_odd_natural,
    )
    if sp.simplify(third_31_odd - (h1 - h2)) == 0:
        raise AssertionError("reversed third-slot (3,1) spectator order passed")

    if sp.simplify((h2 - h3) - p11_even(h2, h3)) == 0:
        raise AssertionError("reversed first-slot spectator order passed")


def check_ramond_pole() -> None:
    b, beta = sp.symbols("b beta", nonzero=True)
    r, s = sp.symbols("r s", nonzero=True)
    x = sp.symbols("x", nonzero=True)

    beta_rs = (r * b + s / b) / (2 * sp.sqrt(2))
    r_quadratic = r**2 * x**2 + (2 * r * s - 8 * beta**2) * x + s**2
    assert_zero("Ramond Kac quadratic", r_quadratic.subs({x: b**2, beta: beta_rs}))

    discriminant = sp.sqrt(2 * beta**2 - r * s)
    x_branch = (4 * beta**2 - r * s + 2 * sp.sqrt(2) * beta * discriminant) / r**2
    assert_zero("Ramond chosen x branch", r_quadratic.subs(x, x_branch))
    x_swapped = (4 * beta**2 - r * s + 2 * sp.sqrt(2) * beta * discriminant) / s**2
    assert_zero(
        "Ramond second ordered-label root",
        r_quadratic.subs(x, 1 / x_swapped),
    )

    c = sp.Rational(3, 2) + 3 * (b + 1 / b) ** 2
    beta_shifted = (r * b - s / b) / (2 * sp.sqrt(2))
    assert_zero(
        "Ramond shifted weight",
        (c / 24 - beta_shifted**2) - (c / 24 - beta_rs**2) - r * s / 2,
    )

    jacobian = 24 * (b + 1 / b) * (1 - 1 / b**2) / (
        (r * b + s / b) * (r - s / b**2)
    )
    assert_zero(
        "Ramond fixed-beta Jacobian",
        jacobian - sp.diff(c, b) / sp.diff(beta_rs**2, b),
    )


def check_ramond_ground_phases() -> None:
    beta = sp.symbols("beta")
    g0_plus_to_minus = sp.I * beta * sp.exp(-sp.I * sp.pi / 4)
    g0_minus_to_plus = sp.I * beta * sp.exp(sp.I * sp.pi / 4)
    assert_zero(
        "Ramond G0 square",
        g0_plus_to_minus * g0_minus_to_plus + beta**2,
    )

    # The chiral BPZ pairing is bilinear.  With <w+|w+>=1 and G0
    # self-adjoint, its odd partner has norm i; no coefficient is conjugated.
    odd_ground_norm = g0_minus_to_plus / g0_plus_to_minus
    assert_zero(
        "Ramond odd-ground BPZ norm",
        sp.expand_complex(odd_ground_norm - sp.I),
    )

    for structure_sign in (1, -1):
        rho_even_pp = sp.Integer(1)
        rho_odd_pm = sp.Integer(1)
        rho_odd_mp = structure_sign * sp.I
        assert_zero("Ramond right-ground normalization", rho_odd_pm - rho_even_pp)
        assert_zero(
            "Ramond left-ground normalization",
            rho_odd_mp - structure_sign * sp.I * rho_even_pp,
        )


def check_ramond_level_one_null_residue() -> None:
    b, h = sp.symbols("b h", nonzero=True)
    c = sp.Rational(3, 2) + 3 * (b + 1 / b) ** 2
    beta_21 = (2 * b + 1 / b) / (2 * sp.sqrt(2))
    h_21 = sp.factor(c / 24 - beta_21**2)
    kappa_squared = h - c / 24

    gram = sp.Matrix(
        [
            [2 * h, sp.Rational(3, 2) * kappa_squared],
            [
                sp.Rational(3, 2) * kappa_squared,
                kappa_squared * (2 * h + c / 4),
            ],
        ]
    )
    null_coefficient = sp.factor(-6 / (8 * h_21 + c))
    null_vector = sp.Matrix([1, null_coefficient])
    null_norm = sp.expand((null_vector.T * gram * null_vector)[0])
    inverse_slope = sp.factor(1 / sp.diff(null_norm, h).subs(h, h_21))
    even_sublattice_product = -(
        2 * b**2 + 1
    ) / (2 * (b**2 - 1) * (b**2 + 1))
    assert_zero(
        "Ramond (2,1) even-sublattice inverse null norm",
        inverse_slope - even_sublattice_product,
    )

    # This is the value produced by the previously printed odd-sublattice
    # product.  It must not agree generically.
    if sp.factor(inverse_slope + 1 / b) == 0:
        raise AssertionError("Ramond odd-sublattice product passed unexpectedly")


def check_ramond_level_one_fusion_anchors() -> None:
    """Check both R pole slots against the direct level-one Ward vector.

    At either (2,1) or (1,2), the even null is
    L_{-1}w+ - 6(8h+c)^{-1}G_{-1}G0w+.  In the NS-first order, the
    direct ground-state Ward vector has entries

        h+h_j-h_i,  -beta^2/2-eta*beta*beta_j.

    Its contraction with the null vector must equal the printed Ramond
    fusion polynomial.  The same vector occurs in the second and third
    Ramond slots after the normalized Mobius transport.
    """

    b, beta_j, lam = sp.symbols("b beta_j lambda", nonzero=True)
    c = sp.Rational(3, 2) + 3 * (b + 1 / b) ** 2
    h_i = ((b + 1 / b) ** 2 - lam**2) / 8
    h_j = c / 24 - beta_j**2

    for shift, beta in (
        (b, (2 * b + 1 / b) / (2 * sp.sqrt(2))),
        (1 / b, (b + 2 / b) / (2 * sp.sqrt(2))),
    ):
        h = c / 24 - beta**2
        null_coefficient = -6 / (8 * h + c)
        for eta in (1, -1):
            direct = (
                h
                + h_j
                - h_i
                + null_coefficient
                * (-beta**2 / 2 - eta * beta * beta_j)
            )
            polynomial = (
                (lam - 2 * sp.sqrt(2) * eta * beta_j + shift)
                * (lam + 2 * sp.sqrt(2) * eta * beta_j - shift)
                / 8
            )
            assert_zero(
                f"Ramond level-one fusion anchor shift={shift} eta={eta}",
                direct - polynomial,
            )


def check_theta_transport() -> None:
    for descendant, edge2, edge3, null_parity in product((0, 1), repeat=4):
        before = (
            (descendant + null_parity) * edge2
            + (descendant + null_parity) * edge3
            + edge2 * edge3
        ) % 2
        shifted = (
            descendant * edge2
            + descendant * edge3
            + edge2 * edge3
            + null_parity * (edge2 + edge3)
        ) % 2
        if before != shifted:
            raise AssertionError("odd-null theta transport")


def check_ns_between_ramond_phase_anchors() -> None:
    """Historical even-primary anchor, retained only to replay old checks.

    The authoritative implementation is now the mandatory-``p_phi``
    generalized Ward system in ``ramond_pbw_generalized_ward.py``.

    The slots are (NS, R_2, R_3) at (infinity, 1, 0).  For an even NS
    primary the fixed-parity identity is

        rho(G_{-1/2} phi, u, v)
          = (-1)^|v| rho(phi, G0 u, v) - i rho(phi, u, G0 v).

    It must reproduce the phases in the odd NS-null factorization and the
    eta -> -eta change.
    """

    beta_2, beta_3 = sp.symbols("beta_2 beta_3", real=True)
    eighth = sp.exp(sp.I * sp.pi / 4)

    def ground_rho(form_parity: int, eta: int, second: int, third: int) -> sp.Expr:
        if form_parity == 0:
            if (second, third) == (0, 0):
                return sp.Integer(1)
            if (second, third) == (1, 1):
                return sp.Integer(eta)
            return sp.Integer(0)
        if (second, third) == (0, 1):
            return sp.Integer(1)
        if (second, third) == (1, 0):
            return sp.I * eta
        return sp.Integer(0)

    # Coefficients of G0 w^alpha in either Ramond slot.
    second_coeff = {
        0: beta_2 * eighth,
        1: sp.I * beta_2 * eighth,
    }
    third_coeff = {
        0: beta_3 * eighth,
        1: sp.I * beta_3 * eighth,
    }

    for eta, second, third in product((1, -1), (0, 1), (0, 1)):
        form_parity = (second + third + 1) % 2
        actual = (
            (-1) ** third
            * second_coeff[second]
            * ground_rho(form_parity, eta, 1 - second, third)
            - sp.I
            * third_coeff[third]
            * ground_rho(form_parity, eta, second, 1 - third)
        )

        seed = beta_3 - eta * beta_2
        shifted_form_parity = 1 - form_parity
        phase = sp.exp(sp.I * (-1) ** form_parity * sp.pi / 4)
        expected = (
            phase
            * seed
            * ground_rho(shifted_form_parity, -eta, second, third)
        )
        assert_zero(
            f"NS-first Ward phase eta={eta} second={second} third={third}",
            sp.expand_complex(actual - expected),
        )

    for f in (0, 1):
        compact_phase = sp.exp(sp.I * (-1) ** f * sp.pi / 4)
        component_phase = eighth if f == 0 else 1 / eighth
        assert_zero(
            f"numeric-f endpoint phase f={f}",
            sp.simplify(compact_phase - component_phase),
        )
        assert_zero(
            f"numeric-f two-vertex phase f={f}",
            sp.simplify(compact_phase**2 - sp.I * (-1) ** f),
        )


def check_ns_first_lift_actions() -> None:
    """Check the second- and third-slot lift identities on ground states."""

    def ground_rho(form_parity: int, eta: int, second: int, third: int) -> sp.Expr:
        if form_parity == 0:
            return {(0, 0): 1, (1, 1): eta}.get((second, third), 0)
        return {(0, 1): 1, (1, 0): sp.I * eta}.get((second, third), 0)

    for lift, eta, second, third in product((1, -1), (1, -1), (0, 1), (0, 1)):
        form_parity = (second + third) % 2
        value = ground_rho(form_parity, eta, second, third)
        assert_zero(
            "second-slot lift",
            lift**second * value
            - ground_rho(form_parity, lift * eta, second, third),
        )
        assert_zero(
            "third-slot lift",
            lift**third * value
            - lift**form_parity
            * ground_rho(form_parity, lift * eta, second, third),
        )


def check_mixed_theta_ns_null_transport() -> None:
    for descendant, edge2, edge3, null_parity in product((0, 1), repeat=4):
        residue_sign = (
            (descendant + null_parity) * edge2
            + (descendant + null_parity) * edge3
            + edge2 * edge3
        ) % 2
        shifted_sign_with_lift_flips = (
            descendant * edge2
            + descendant * edge3
            + edge2 * edge3
            + null_parity * edge2
            + null_parity * edge3
        ) % 2
        if residue_sign != shifted_sign_with_lift_flips:
            raise AssertionError("mixed theta NS-null Koszul transport")

        original_form = (descendant + null_parity + edge2 + edge3) % 2
        shifted_form = (descendant + edge2 + edge3) % 2
        if shifted_form != (original_form + null_parity) % 2:
            raise AssertionError("mixed theta NS-null form parity")

    for form_parity in (0, 1):
        endpoint_phase = sp.exp(
            sp.I * (-1) ** form_parity * sp.pi / 4
        )
        assert_zero(
            f"mixed theta two-endpoint phase f={form_parity}",
            sp.simplify(endpoint_phase**2 - sp.I * (-1) ** form_parity),
        )


def check_mixed_theta_lowest_ns_null_residue() -> None:
    """Contract the (1,1) NS null explicitly at both theta vertices.

    Although the vacuum (1,1) pole is omitted from the c-recursion, its
    level-1/2 null vector G_{-1/2}|0> is the sharpest local test of every
    phase, chiral-label flip, inverse ground metric, Koszul sign, and lift
    flip in the odd NS-edge residue.
    """

    beta_2, beta_3 = sp.symbols("beta_2 beta_3", real=True)
    eighth = sp.exp(sp.I * sp.pi / 4)
    inverse_ground_metric = {0: sp.Integer(1), 1: -sp.I}

    def ground_rho(form_parity: int, eta: int, second: int, third: int) -> sp.Expr:
        if form_parity == 0:
            return {(0, 0): 1, (1, 1): eta}.get((second, third), 0)
        return {(0, 1): 1, (1, 0): sp.I * eta}.get((second, third), 0)

    def g0_coefficient(beta: sp.Expr, parity: int) -> sp.Expr:
        return beta * eighth if parity == 0 else sp.I * beta * eighth

    def ward_value(form_parity: int, eta: int, second: int, third: int) -> sp.Expr:
        return (
            (-1) ** third
            * g0_coefficient(beta_2, second)
            * ground_rho(form_parity, eta, 1 - second, third)
            - sp.I
            * g0_coefficient(beta_3, third)
            * ground_rho(form_parity, eta, second, 1 - third)
        )

    for form_parity, eta, eta_prime, lift_2, lift_3 in product(
        (0, 1), (1, -1), (1, -1), (1, -1), (1, -1)
    ):
        direct = sp.Integer(0)
        shifted_ground_block = sp.Integer(0)
        for second, third in product((0, 1), repeat=2):
            if (1 + second + third) % 2 != form_parity:
                continue
            metric = (
                inverse_ground_metric[second]
                * inverse_ground_metric[third]
            )
            direct += (
                lift_2**second
                * lift_3**third
                * (-1) ** (second + third + second * third)
                * metric
                * ward_value(form_parity, eta, second, third)
                * ward_value(form_parity, eta_prime, second, third)
            )
            shifted_ground_block += (
                (-lift_2) ** second
                * (-lift_3) ** third
                * (-1) ** (second * third)
                * metric
                * ground_rho(1 - form_parity, -eta, second, third)
                * ground_rho(1 - form_parity, -eta_prime, second, third)
            )

        polynomial = (beta_3 - eta * beta_2) * (
            beta_3 - eta_prime * beta_2
        )
        expected = (
            sp.I
            * (-1) ** form_parity
            * polynomial
            * shifted_ground_block
        )
        assert_zero(
            "lowest mixed-theta NS-null residue",
            sp.expand_complex(direct - expected),
        )


def check_two_vertex_structure_labels() -> None:
    """Contract the Ramond ground fibers with independent eta labels.

    The inverse ground metric is diag(1,-i).  The Koszul sign in the theta
    block contributes (-1)^(|alpha||gamma|).  This directly checks the two
    constant terms printed in the notes and supplies a counterexample to a
    putative delta_{eta,eta'} sewing rule.
    """

    eta, eta_prime, lift_2, lift_3 = sp.symbols(
        "eta eta_prime lift_2 lift_3"
    )
    inverse_ground_metric = {0: sp.Integer(1), 1: -sp.I}

    def ground_rho(form_parity: int, structure: sp.Expr, left: int, right: int):
        if form_parity == 0:
            return {(0, 0): sp.Integer(1), (1, 1): structure}.get(
                (left, right), sp.Integer(0)
            )
        return {(0, 1): sp.Integer(1), (1, 0): sp.I * structure}.get(
            (left, right), sp.Integer(0)
        )

    contractions = []
    for form_parity in (0, 1):
        value = sp.Integer(0)
        for left, right in product((0, 1), repeat=2):
            value += (
                (-1) ** (left * right)
                * lift_2**left
                * lift_3**right
                * inverse_ground_metric[left]
                * inverse_ground_metric[right]
                * ground_rho(form_parity, eta, left, right)
                * ground_rho(form_parity, eta_prime, left, right)
            )
        contractions.append(sp.expand(value))

    assert_zero(
        "two-label even ground contraction",
        contractions[0] - (1 + eta * eta_prime * lift_2 * lift_3),
    )
    assert_zero(
        "two-label odd ground contraction",
        contractions[1] - sp.I * (eta * eta_prime * lift_2 - lift_3),
    )

    mixed_odd = contractions[1].subs(
        {eta: 1, eta_prime: -1, lift_2: 1, lift_3: 1}
    )
    if mixed_odd == 0:
        raise AssertionError("mixed eta theta block vanished unexpectedly")


def main() -> None:
    checks = [
        ("NS pole and minus Jacobian", check_ns_pole),
        ("NS level-three-halves inverse null norm", check_ns_level_three_halves_null_residue),
        ("all-NS slot fusion signs", check_all_ns_slot_fusion_signs),
        ("Ramond pole, shift, and fixed-beta Jacobian", check_ramond_pole),
        ("Ramond ground-state phases", check_ramond_ground_phases),
        ("Ramond level-one inverse null norm", check_ramond_level_one_null_residue),
        ("Ramond level-one fusion anchors", check_ramond_level_one_fusion_anchors),
        ("theta odd-null sewing transport", check_theta_transport),
        ("NS-between-Ramond Ward phases", check_ns_between_ramond_phase_anchors),
        ("NS-first Ramond lift actions", check_ns_first_lift_actions),
        ("mixed-theta NS-null transport", check_mixed_theta_ns_null_transport),
        ("lowest mixed-theta NS-null residue", check_mixed_theta_lowest_ns_null_residue),
        ("independent theta-vertex structures", check_two_vertex_structure_labels),
    ]
    for label, check in checks:
        check()
        print(f"PASS: {label}")


if __name__ == "__main__":
    main()
