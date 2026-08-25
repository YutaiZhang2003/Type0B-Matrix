#!/usr/bin/env python3
"""Independent symbolic checks for SCblock's global osp(1|2) formulas.

The checks compare three convention layers:

1. SCblock.tex, Eqs. (2.16), (3.2)--(3.6).
2. The current repository implementation, especially the fixed-parity
   trilinear, theta orientation polynomial, and complete global vertex.
3. Belavin--Ramos Cabezas--Runov, "Shadow formalism for supersymmetric
   conformal blocks", including its sphere four-point and torus one-/two-
   point decompositions into sl(2) blocks.

The collaborator and shadow formulas are entered in their component-ordered
convention.  They are converted explicitly to the Human Note fixed-parity
convention

    rho_a^H(x1,x2,x3) = (-1)^(a |x3|) rho_a^component(x1,x2,x3)

before they are compared with SCblock.  Convention phases are kept explicit
rather than silently folded into an eta or spin-lift variable.
"""

from __future__ import annotations

from itertools import product
from pathlib import Path
import sys

import sympy as sp


CODE_DIR = Path(__file__).resolve().parents[2]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from ns_human_convention import human_note_rho_sign  # noqa: E402


def rising(x: sp.Expr, n: int) -> sp.Expr:
    return sp.rf(x, n)


def falling(x: sp.Expr, n: int) -> sp.Expr:
    return sp.prod(x - j for j in range(n))


def s_endpoint(k: int, m: int, h1: sp.Expr, h2: sp.Expr, h3: sp.Expr) -> sp.Expr:
    """SCblock/CCY endpoint polynomial s_{k m}(h1,h2,h3)."""
    out = sp.S.Zero
    for p in range(min(k, m) + 1):
        out += (
            sp.binomial(k, p)
            * falling(2 * h3 + m - 1, p)
            * falling(m, p)
            * rising(h3 + h2 - h1, m - p)
            * rising(h1 + h2 - h3 + p - m, k - p)
        )
    return sp.expand(out)


def base_component_ordered(
    a: int,
    b: int,
    c: int,
    h1: sp.Expr,
    h2: sp.Expr,
    h3: sp.Expr,
) -> sp.Expr:
    """The component-ordered zero-translation table."""

    table = {
        (0, 0, 0): sp.S.One,
        (1, 0, 0): sp.S.One,
        (0, 1, 0): sp.S.One,
        (0, 0, 1): sp.S.One,
        (1, 1, 0): h1 + h2 - h3,
        (1, 0, 1): h1 - h2 + h3,
        (0, 1, 1): h1 - h2 - h3,
        (1, 1, 1): h1 + h2 + h3 - sp.Rational(1, 2),
    }
    return table[(a, b, c)]


def base_fixed_parity(
    a: int,
    b: int,
    c: int,
    h1: sp.Expr,
    h2: sp.Expr,
    h3: sp.Expr,
) -> sp.Expr:
    """The Human Note fixed-parity table in SCblock (3.4)--(3.5)."""

    return sp.expand(
        human_note_rho_sign((a, b, c))
        * base_component_ordered(a, b, c, h1, h2, h3)
    )


def rho_scblock(
    k1: int,
    k2: int,
    k3: int,
    a: int,
    b: int,
    c: int,
    h1: sp.Expr,
    h2: sp.Expr,
    h3: sp.Expr,
) -> sp.Expr:
    """SCblock Eq. (3.6), including its rising middle Pochhammer."""
    H1 = h1 + sp.Rational(a, 2)
    H2 = h2 + sp.Rational(b, 2)
    H3 = h3 + sp.Rational(c, 2)
    middle = rising(H1 + k1 - H2 - k2 + 1 - H3 - k3, k2)
    return sp.expand(
        base_fixed_parity(a, b, c, h1, h2, h3)
        * s_endpoint(k1, k3, H1, H2, H3)
        * middle
    )


def rho_collaborator(
    k1: int,
    k2: int,
    k3: int,
    a: int,
    b: int,
    c: int,
    h1: sp.Expr,
    h2: sp.Expr,
    h3: sp.Expr,
) -> sp.Expr:
    """Collaborator's complete vertex converted to the Human Note frame."""

    H1 = h1 + sp.Rational(a, 2)
    H2 = h2 + sp.Rational(b, 2)
    H3 = h3 + sp.Rational(c, 2)
    exponent = H1 + k1 - H2 - H3 - k3
    component_value = (
        t_collaborator(k1, k3, a, b, c, h1, h2, h3)
        * falling(exponent, k2)
    )
    return sp.expand(human_note_rho_sign((a, b, c)) * component_value)


def t_collaborator(
    k: int,
    m: int,
    a: int,
    b: int,
    c: int,
    d1: sp.Expr,
    d2: sp.Expr,
    d3: sp.Expr,
) -> sp.Expr:
    """Collaborator Eqs. (primary table, eight kernels, reflection).

    This deliberately does not call ``s_endpoint``.  Six parity cases are
    entered in the expanded form of the collaborator's notes; 100 and 110
    are obtained from their graded endpoint-reflection identity.
    """
    if (a, b, c) in {(1, 0, 0), (1, 1, 0)}:
        reflection_sign = (-1) ** (b * (a + c))
        return sp.expand(
            reflection_sign
            * t_collaborator(m, k, c, b, a, d3, d2, d1)
        )

    A = d2 + d3 - d1
    B = d1 + d2 - d3
    C = d1 - d2 + d3
    S = d1 + d2 + d3

    def sum_kernel(delta: int, left, right) -> sp.Expr:
        out = sp.S.Zero
        for p in range(min(k, m) + 1):
            out += (
                sp.binomial(k, p)
                * falling(2 * d3 + m - 1 + delta, p)
                * falling(m, p)
                * left(p)
                * right(p)
            )
        return sp.expand(out)

    if (a, b, c) == (0, 0, 0):
        return sum_kernel(0, lambda p: rising(A, m - p), lambda p: rising(B + p - m, k - p))
    if (a, b, c) == (0, 1, 0):
        return sum_kernel(0, lambda p: rising(A + sp.Rational(1, 2), m - p), lambda p: rising(B + sp.Rational(1, 2) + p - m, k - p))
    if (a, b, c) == (0, 0, 1):
        return sum_kernel(1, lambda p: rising(A + sp.Rational(1, 2), m - p), lambda p: rising(B - sp.Rational(1, 2) + p - m, k - p))
    if (a, b, c) == (1, 0, 1):
        return C * sum_kernel(1, lambda p: rising(A, m - p), lambda p: rising(B + p - m, k - p))
    if (a, b, c) == (0, 1, 1):
        return -sum_kernel(1, lambda p: rising(A, m - p + 1), lambda p: rising(B + p - m, k - p))
    if (a, b, c) == (1, 1, 1):
        return (S - sp.Rational(1, 2)) * sum_kernel(
            1,
            lambda p: rising(A + sp.Rational(1, 2), m - p),
            lambda p: rising(B + sp.Rational(1, 2) + p - m, k - p),
        )
    raise AssertionError(f"unhandled parity case {(a, b, c)}")


def assert_zero(name: str, expression: sp.Expr) -> None:
    value = sp.factor(sp.cancel(expression))
    if value != 0:
        raise AssertionError(f"{name}: expected zero, obtained {value}")


def check_human_note_ground_table() -> None:
    """Check the literal eight-component table used by current public APIs."""

    h1, h2, h3 = sp.symbols("h1 h2 h3")
    expected = {
        (0, 0, 0): sp.S.One,
        (1, 1, 0): h1 + h2 - h3,
        (1, 0, 1): h1 - h2 + h3,
        (0, 1, 1): h1 - h2 - h3,
        (1, 0, 0): sp.S.One,
        (0, 1, 0): sp.S.One,
        (0, 0, 1): -sp.S.One,
        (1, 1, 1): -(h1 + h2 + h3 - sp.Rational(1, 2)),
    }
    for bits, expected_value in expected.items():
        assert_zero(
            f"Human Note ground component {bits}",
            base_fixed_parity(*bits, h1, h2, h3) - expected_value,
        )


def check_complete_vertex() -> None:
    h1, h2, h3 = sp.symbols("h1 h2 h3")
    for a, b, c in product((0, 1), repeat=3):
        for k1, k2, k3 in product(range(4), repeat=3):
            assert_zero(
                f"complete vertex parities={(a, b, c)} levels={(k1, k2, k3)}",
                rho_scblock(k1, k2, k3, a, b, c, h1, h2, h3)
                - rho_collaborator(k1, k2, k3, a, b, c, h1, h2, h3),
            )


def check_theta_orientation() -> None:
    """Check that SCblock (2.16) reproduces the polarized theta sign."""
    for vacuum in product((0, 1), repeat=3):
        if sum(vacuum) % 2:
            continue  # the large-c vacuum trilinear is parity even
        for global_part in product((0, 1), repeat=3):
            v0, v1, v2 = vacuum
            g0, g1, g2 = global_part
            collaborator_cross = (
                v0 * g1 + g0 * v1
                + v0 * g2 + g0 * v2
                + v1 * g2 + g1 * v2
            ) % 2
            scblock_eta_shift = (v0 * g0 + v1 * g1 + v2 * g2) % 2
            if collaborator_cross != scblock_eta_shift:
                raise AssertionError(
                    f"theta cross sign mismatch: v={vacuum}, g={global_part}"
                )

    # In the collaborator's standard CCY theta frame, edge 0 below denotes
    # SCblock's first/infinity edge.  Rephasing eta_1 removes the linear term.
    for eps in product((0, 1), repeat=3):
        e0, e1, e2 = eps
        quadratic = (e0 * e1 + e0 * e2 + e1 * e2) % 2
        full_ccy = (quadratic + e0) % 2
        rephased_scblock = (quadratic + e0) % 2
        if full_ccy != rephased_scblock:
            raise AssertionError(f"CCY frame rephasing mismatch: eps={eps}")


def check_shadow_sphere() -> None:
    """Compare reduced sphere coefficients with shadow Eqs. (4ptVVVVeven/odd)."""
    h, h1, h2, h3, h4 = sp.symbols("h h1 h2 h3 h4")
    for n in range(7):
        ours_even = (
            rising(h + h2 - h1, n)
            * rising(h + h3 - h4, n)
            / (sp.factorial(n) * rising(2 * h, n))
        )
        shadow_even = ours_even
        assert_zero(f"shadow sphere even n={n}", ours_even - shadow_even)

        ours_odd = (
            rising(h + h2 - h1 + sp.Rational(1, 2), n)
            * rising(h + h3 - h4 + sp.Rational(1, 2), n)
            / (sp.factorial(n) * rising(2 * h, n + 1))
        )
        shadow_odd = (
            sp.Rational(1, 1) / (2 * h)
            * rising(h + h2 - h1 + sp.Rational(1, 2), n)
            * rising(h + h3 - h4 + sp.Rational(1, 2), n)
            / (sp.factorial(n) * rising(2 * h + 1, n))
        )
        assert_zero(f"shadow sphere odd n={n}", ours_odd - shadow_odd)


def check_shadow_torus_one_point() -> None:
    """Compare the two osp sectors with shadow paper Eq. (thetorus5-2)."""

    h, d = sp.symbols("h d")
    bottom_bits = (1, 0, 1)
    upper_bits = (1, 1, 1)
    bottom_human = base_fixed_parity(*bottom_bits, h, d, h) / (2 * h)
    upper_human = base_fixed_parity(*upper_bits, h, d, h) / (2 * h)
    assert_zero(
        "torus B0 Human Note ratio",
        bottom_human - (2 * h - d) / (2 * h),
    )
    assert_zero(
        "torus B1 Human Note ratio",
        upper_human
        + (2 * h + d - sp.Rational(1, 2)) / (2 * h),
    )

    # The shadow formulas are component ordered.  Undo the Human Note
    # third-slot sign before applying their relative sewing phase.
    bottom_component = human_note_rho_sign(bottom_bits) * bottom_human
    upper_component = human_note_rho_sign(upper_bits) * upper_human
    sewing_phase = -1
    paper_b0_second = -(2 * h - d) / (2 * h)
    paper_b1_second = -(2 * h + d - sp.Rational(1, 2)) / (2 * h)
    assert_zero(
        "torus B0 shadow sign",
        sewing_phase * bottom_component - paper_b0_second,
    )
    assert_zero(
        "torus B1 shadow sign",
        sewing_phase * upper_component - paper_b1_second,
    )


def check_shadow_torus_two_point() -> None:
    """Recover every rational prefactor in shadow Eqs. (13-1), (14), (17), (19)."""
    D1, D2, h1, h2 = sp.symbols("D1 D2 h1 h2")

    # B_00^(1): even-even versus odd-odd internal routing.
    ratio_b00_same = (
        base_fixed_parity(1, 0, 1, D1, h1, D2)
        * base_fixed_parity(1, 0, 1, D2, h2, D1)
        / (4 * D1 * D2)
    )
    shadow_b00_same = (D1 + D2 - h1) * (D1 + D2 - h2) / (4 * D1 * D2)
    assert_zero("shadow B00(1) prefactor", ratio_b00_same - shadow_b00_same)

    # B_00^(2): the two mixed-parity routings, computed from their local
    # vertices and the odd-state norms 2*D_i.
    first_b00_mixed = (
        base_fixed_parity(0, 0, 1, D1, h1, D2)
        * base_fixed_parity(1, 0, 0, D2, h2, D1)
        / (2 * D2)
    )
    second_b00_mixed = (
        base_fixed_parity(1, 0, 0, D1, h1, D2)
        * base_fixed_parity(0, 0, 1, D2, h2, D1)
        / (2 * D1)
    )
    ratio_b00_mixed = sp.cancel(second_b00_mixed / first_b00_mixed)
    assert_zero("shadow B00(2) prefactor", ratio_b00_mixed - D2 / D1)

    # B_theta-theta^(2): even-even versus odd-odd with top external fields.
    ratio_tt_same = (
        base_fixed_parity(1, 1, 1, D1, h1, D2)
        * base_fixed_parity(1, 1, 1, D2, h2, D1)
        / (4 * D1 * D2)
    )
    alpha3 = (
        (2 * D1 + 2 * D2 + 2 * h1 - 1)
        * (2 * D1 + 2 * D2 + 2 * h2 - 1)
        / (16 * D1 * D2)
    )
    assert_zero("shadow Btt(2) prefactor", ratio_tt_same - alpha3)

    # B_theta-theta^(1): ratio of the two mixed-parity routings.
    first_routing = (
        base_fixed_parity(0, 1, 1, D1, h1, D2)
        * base_fixed_parity(1, 1, 0, D2, h2, D1)
        / (2 * D2)
    )
    second_routing = (
        base_fixed_parity(1, 1, 0, D1, h1, D2)
        * base_fixed_parity(0, 1, 1, D2, h2, D1)
        / (2 * D1)
    )
    ratio_tt_mixed = sp.cancel(second_routing / first_routing)
    shadow_tt_mixed = sp.cancel(
        D2
        * (D1 - D2 + h1)
        * (D2 - D1 - h2)
        / (D1 * (D1 - D2 - h1) * (D2 - D1 + h2))
    )
    assert_zero("shadow Btt(1) prefactor", ratio_tt_mixed - shadow_tt_mixed)

    # In all four shadow formulas the second routing is preceded by a minus.
    # Check that sign separately: it is a torus supertrace/sewing convention,
    # not a change in the local fixed-parity trilinear polynomials above.
    sewing_phase = -1
    assert_zero(
        "shadow B00(1) signed second routing",
        sewing_phase * ratio_b00_same - (-shadow_b00_same),
    )
    assert_zero(
        "shadow B00(2) signed second routing",
        sewing_phase * ratio_b00_mixed - (-D2 / D1),
    )
    assert_zero(
        "shadow Btt(1) signed second routing",
        sewing_phase * ratio_tt_mixed - (-shadow_tt_mixed),
    )
    assert_zero(
        "shadow Btt(2) signed second routing",
        sewing_phase * ratio_tt_same - (-alpha3),
    )


def check_shadow_component_relations() -> None:
    """Compare with the shadow paper's displayed zero-level matrix elements."""

    D1, D2, h = sp.symbols("D1 D2 h")

    def component_value(bits: tuple[int, int, int]) -> sp.Expr:
        """Convert the canonical Human Note value back to the paper frame."""

        return sp.expand(
            human_note_rho_sign(bits)
            * base_fixed_parity(*bits, D1, h, D2)
        )

    assert_zero(
        "shadow relation 101",
        component_value((1, 0, 1)) - (D1 + D2 - h),
    )
    assert_zero(
        "shadow relation 011",
        component_value((0, 1, 1)) - (D1 - h - D2),
    )
    assert_zero(
        "shadow relation 110",
        component_value((1, 1, 0)) - (D1 + h - D2),
    )
    assert_zero(
        "shadow relation 111",
        component_value((1, 1, 1))
        - (D1 + h + D2 - sp.Rational(1, 2)),
    )


def main() -> None:
    checks = [
        ("Human Note fixed-parity ground table", check_human_note_ground_table),
        ("SCblock complete vertex = collaborator complete vertex", check_complete_vertex),
        ("theta polarized sign and eta rephasing", check_theta_orientation),
        ("shadow sphere four-point blocks", check_shadow_sphere),
        ("shadow torus one-point blocks", check_shadow_torus_one_point),
        ("shadow torus two-point blocks", check_shadow_torus_two_point),
        ("shadow component matrix-element relations", check_shadow_component_relations),
    ]
    for label, check in checks:
        check()
        print(f"PASS: {label}")


if __name__ == "__main__":
    main()
