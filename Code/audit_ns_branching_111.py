#!/usr/bin/env python3
"""Direct audit of the NS double-Virasoro branching coefficient (1,1,1).

The labels in this file are the half-integers n_i themselves: ``111`` means
``n_1=n_2=n_3=1`` (equivalently k_i=2 n_i=2 in code which uses integer
labels).  Nothing in this file changes the branching coefficient used by the
genus-two implementation.

The calculation is deliberately constructed from the explicit level-two
primary, the auxiliary Majorana three-form, and the NS Ward recursion.  The
ell product is evaluated only at the final comparison step.
"""

from __future__ import annotations

import argparse
from itertools import product

import sympy as sp

from check_first_virasoro_primary import ell, simplify
from check_second_virasoro_primary import (
    TensorVector,
    current_fermion_three_point,
    current_sca_three_point,
    product_norm,
    vector_norm,
    verify_v1_primary,
)
from check_ungraded_branching_low_level import ungraded_tensor_three_point
from ns_genus2_symbolic_low_order import (
    ExactNSDescendantThreeForm,
    ExactNSVermaModule,
    state_parity,
)


def module_at(b: sp.Expr, momentum: sp.Expr) -> ExactNSVermaModule:
    q = b + 1 / b
    c = sp.Rational(3, 2) + 3 * q**2
    h = q**2 / 8 - momentum**2 / 2
    return ExactNSVermaModule(c=c, weight=h)


def v1_explicit(b: sp.Expr, momentum: sp.Expr) -> TensorVector:
    r"""The human-note normalized level-two primary v_1(P).

    The tensor basis is auxiliary-first.  SCA states are stored with the
    leftmost operator acting first, as in the other exact Ward-code files.
    """

    q = b + 1 / b
    a = q / 2 + momentum
    d = (a + b) * (a + 1 / b)
    r = q + momentum
    return {
        ((), (("L", -4),)): sp.expand(4 * d),
        ((), (("L", -2), ("L", -2))): sp.Integer(-2),
        ((), (("G", -1), ("G", -3))): sp.expand(-2 * (a**2 + d)),
        ((1,), (("G", -1), ("L", -2))): sp.expand(-4 * r),
        ((1,), (("G", -3),)): sp.expand(-4 * a**2 * r),
        ((2,), (("G", -1),)): sp.expand(4 * d * r),
        ((1, 2), ()): sp.expand(-4 * a * d * r),
    }


def split_v1(b: sp.Expr, momentum: sp.Expr) -> dict[str, TensorVector]:
    """Split v_1 into its four auxiliary-Fock components."""

    names = {(): "0", (1,): "1", (2,): "3", (1, 2): "13"}
    result: dict[str, TensorVector] = {name: {} for name in names.values()}
    for (fermion, sca), coefficient in v1_explicit(b, momentum).items():
        result[names[fermion]][(fermion, sca)] = coefficient
    return result


def fermion_crossing_table() -> tuple[tuple[str, str, str, sp.Expr, int, sp.Expr], ...]:
    """Return all nonzero terms and the ungraded matrix-element sign.

    Every summand of v_1 is even, hence the SCA parity equals its auxiliary
    parity.  With even external primaries, the crossing exponent therefore is

        p_2 p_3.
    """

    states = {"0": (), "1": (1,), "3": (2,), "13": (1, 2)}
    rows = []
    for labels in product(states, repeat=3):
        fermions = tuple(states[label] for label in labels)
        rho_f = current_fermion_three_point(fermions)
        if rho_f == 0:
            continue
        _p1, p2, p3 = (len(state) % 2 for state in fermions)
        exponent = (p2 * p3) % 2
        signed = (-1 if exponent else 1) * rho_f
        rows.append((*labels, rho_f, exponent, signed))
    return tuple(rows)


def direct_111_from_blocks(
    b: sp.Expr, momenta: tuple[sp.Expr, sp.Expr, sp.Expr]
) -> tuple[sp.Expr, tuple[tuple[str, str, str, sp.Expr], ...]]:
    """Evaluate the 25-term direct 111 sum block by block."""

    q = b + 1 / b
    c = sp.Rational(3, 2) + 3 * q**2
    weights = tuple(q**2 / 8 - momentum**2 / 2 for momentum in momenta)
    form = ExactNSDescendantThreeForm(c=c, weights=weights)
    blocks = tuple(split_v1(b, momentum) for momentum in momenta)
    rows = []
    result = sp.S.Zero
    for name1, name2, name3, _rho_f, _exponent, signed_f in fermion_crossing_table():
        vectors = (blocks[0][name1], blocks[1][name2], blocks[2][name3])
        # Each block has a fixed auxiliary state.  Strip off that state and
        # sum the SCA descendants directly, so the displayed coefficient is
        # exactly signed_f times the SCA Ward three-form.
        block_value = sp.S.Zero
        for terms in product(*(tuple(vector.items()) for vector in vectors)):
            coefficient = sp.prod(term[1] for term in terms)
            sca_states = tuple(term[0][1] for term in terms)
            block_value += coefficient * signed_f * current_sca_three_point(
                form, sca_states
            )
        block_value = simplify(block_value)
        rows.append((name1, name2, name3, block_value))
        result += block_value
    return simplify(result), tuple(rows)


def ell_111(b: sp.Expr, momenta: tuple[sp.Expr, sp.Expr, sp.Expr]) -> sp.Expr:
    """Equation (7.38) specialized to n_1=n_2=n_3=1, a=0."""

    q = b + 1 / b
    p1, p2, p3 = momenta
    s = q / 2 + p1 + p2 + p3
    a = q / 2 - p1 + p2 + p3
    c = q / 2 + p1 + p2 - p3
    d = q / 2 - p1 + p2 - p3
    raw = -sp.Rational(1, 8) * ell(s, 6, b, q) * ell(a, 2, b, q)
    raw *= ell(c, 2, b, q) * ell(d, -2, b, q)
    return simplify(raw)


def ell_111_reduced(b: sp.Expr, momenta: tuple[sp.Expr, sp.Expr, sp.Expr]) -> sp.Expr:
    """The same product after ell(x,2)=x and ell(x,-2)=-(Q-x)."""

    q = b + 1 / b
    p1, p2, p3 = momenta
    s = q / 2 + p1 + p2 + p3
    a = q / 2 - p1 + p2 + p3
    c = q / 2 + p1 + p2 - p3
    reflected = q / 2 + p1 - p2 + p3
    return simplify(sp.Rational(1, 8) * ell(s, 6, b, q) * a * c * reflected)


def exact_sample(
    b: sp.Rational, momenta: tuple[sp.Rational, sp.Rational, sp.Rational]
) -> None:
    q = b + 1 / b
    print(f"b={b}, P={momenta}")
    for slot, momentum in enumerate(momenta, start=1):
        module = module_at(b, momentum)
        vector = v1_explicit(b, momentum)
        defining_checks = verify_v1_primary(module, b, momentum, vector)
        norm = vector_norm(module, vector)
        target_norm = product_norm(2, momentum, b, q)
        assert simplify(norm - target_norm) == 0
        print(
            f"  leg {slot}: primary equations={defining_checks}/5, "
            f"norm residual={simplify(norm-target_norm)}"
        )

    direct, rows = direct_111_from_blocks(b, momenta)
    # Independently exercise the unsplit matrix-element implementation.  This
    # guards against mistakes in the displayed term table and is not an
    # ell-product comparison.
    c = sp.Rational(3, 2) + 3 * q**2
    weights = tuple(q**2 / 8 - momentum**2 / 2 for momentum in momenta)
    form = ExactNSDescendantThreeForm(c=c, weights=weights)
    unsplit = ungraded_tensor_three_point(
        form, tuple(v1_explicit(b, momentum) for momentum in momenta)
    )
    assert simplify(direct - unsplit) == 0
    target = ell_111(b, momenta)
    target_reduced = ell_111_reduced(b, momenta)
    assert simplify(target - target_reduced) == 0
    print(f"  nonzero auxiliary blocks={len(rows)}")
    print(f"  rho_direct={sp.factor(direct)}")
    print(f"  rho_ell={sp.factor(target)}")
    print(f"  ratio={sp.factor(sp.cancel(direct/target))}")
    print(f"  residual={sp.factor(direct-target)}")
    norm_product = sp.prod(
        product_norm(2, momentum, b, q) for momentum in momenta
    )
    print(f"  B_direct^2={sp.factor(sp.cancel(direct**2/norm_product))}")
    print(f"  B_ell^2={sp.factor(sp.cancel(target**2/norm_product))}")


def symbolic_primary_and_norm() -> None:
    """Prove the five defining equations and (7.36) for generic b,P."""

    b, momentum = sp.symbols("b P", nonzero=True)
    q = b + 1 / b
    module = module_at(b, momentum)
    vector = v1_explicit(b, momentum)
    checks = verify_v1_primary(module, b, momentum, vector)
    direct = sp.factor(sp.cancel(vector_norm(module, vector)))
    expected = sp.factor(sp.cancel(product_norm(2, momentum, b, q)))
    if checks != 5 or sp.factor(sp.cancel(direct - expected)) != 0:
        raise AssertionError("generic v_1 primary/norm check failed")
    print(f"Generic v_1 defining equations: {checks}/5")
    print(f"Generic norm: {direct}")
    print("Generic norm residual: 0")


def symbolic_fixed_b(b: sp.Rational) -> None:
    momenta = sp.symbols("P_1 P_2 P_3")
    print(f"Computing the exact symbolic 111 residual at b={b} ...")
    direct, _rows = direct_111_from_blocks(b, momenta)
    target = ell_111(b, momenta)
    print("rho_direct =", sp.factor(direct))
    print("rho_ell =", sp.factor(target))
    print("residual =", sp.factor(sp.cancel(direct - target)))


def symbolic_compact_line() -> None:
    """A compact one-parameter exact comparison used in the machine note."""

    b = sp.Integer(3)
    t = sp.symbols("t")
    momenta = (-sp.Rational(3, 2), t, -sp.Rational(3, 2))
    direct, _rows = direct_111_from_blocks(b, momenta)
    target = ell_111(b, momenta)
    print("Compact symbolic line b=3, (P_1,P_2,P_3)=(-3/2,t,-3/2)")
    print("rho_direct =", sp.factor(direct))
    print("rho_ell =", sp.factor(target))
    print("residual =", sp.factor(sp.cancel(direct - target)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--symbolic-fixed-b",
        action="store_true",
        help="also factor the exact residual with symbolic P_i at b=3/2",
    )
    parser.add_argument(
        "--compact-line",
        action="store_true",
        help="factor the one-parameter line displayed in the machine note",
    )
    args = parser.parse_args()

    symbolic_primary_and_norm()
    print("Nonzero auxiliary-fermion/crossing table")
    print(" A  B  C | rho_F  E  (-1)^E rho_F")
    for row in fermion_crossing_table():
        print(f"{row[0]:>2} {row[1]:>2} {row[2]:>2} | {str(row[3]):>5}  {row[4]}  {str(row[5]):>12}")

    exact_sample(
        sp.Integer(3),
        (-sp.Rational(3, 2), sp.Rational(1, 3), -sp.Rational(3, 2)),
    )
    exact_sample(
        sp.Rational(5, 4),
        (sp.Rational(1, 6), sp.Rational(2, 7), -sp.Rational(3, 8)),
    )
    if args.symbolic_fixed_b:
        symbolic_fixed_b(sp.Rational(3, 2))
    if args.compact_line:
        symbolic_compact_line()


if __name__ == "__main__":
    main()
