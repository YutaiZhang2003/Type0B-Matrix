#!/usr/bin/env python3
"""Direct symbolic audit of the first NS branching coefficient.

This script implements the conventions currently written in
``Human Notes/SCblock.tex``.  It does not import the double-Virasoro code and
does not compare two versions of the product formula.  Instead it performs
the following independent low-level calculation.

* Solve the first Virasoro primary in the ordered
  auxiliary-fermion x SCA basis

      X = 1_F x G_{-1/2}|P>,   Y = psi_{-1/2}1_F x |P>.

  The result is

      v_{1/2}(P) = X + (Q/2 + P) Y.

* Compute its BPZ norm using

      <X,X> = 2h,   <Y,Y> = -1.

* Expand every three-point function with n_i in {-1/2,0,1/2} into its
  auxiliary-fermion and SCA components.  The pairing is ungraded.  The only
  crossing retained is the tensor-product vertex-action sign,

      (-1)^[ |x2||u3| ].

  In particular, there is no BPZ-leg crossing
  ``|x1|(|u2|+|u3|)``; retaining it would make the identity insertion return
  the graded rather than the ungraded tensor-product norm.

* Compare both

      B_a^2 = rho_hat_a^2 / (N_1 N_2 N_3)

  and the unsquared B_a with the four-ell branching formula in the note.  For
  the unsquared comparison we make the uniform analytic choice

      ||v_n|| = i^(2|n|) 2^(-|n|)
                sqrt[ell(2P_eff,4|n|) ell(Q+2P_eff,4|n|)],

  where P_eff=P for n>=0 and P_eff=-P for n<0.  Its square is exactly the
  reflected BPZ norm.  Stating this choice is essential: the three complex
  norm squares do not by themselves determine an unsquared coefficient.

The ell function, including its negative-index continuation, is reproduced
literally from the note.  Negative labels are evaluated by the note's
reflection rule v_n(P)=v_{-n}(-P), not by inserting n<0 into a formula stated
for positively normalized vectors.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import sympy as sp


HALF = sp.Rational(1, 2)


def simplify(expression: sp.Expr) -> sp.Expr:
    """Put a rational symbolic residual into a stable factored form."""

    return sp.factor(sp.cancel(sp.expand(expression)))


def require_zero(name: str, expression: sp.Expr | sp.MatrixBase) -> None:
    """Raise an informative assertion unless every residual is zero."""

    entries = (
        list(expression)
        if isinstance(expression, sp.MatrixBase)
        else [expression]
    )
    residuals = [simplify(entry) for entry in entries]
    if any(residual != 0 for residual in residuals):
        raise AssertionError(f"{name}: nonzero residuals {residuals}")


def ell(x: sp.Expr, m: int, b: sp.Expr, q: sp.Expr) -> sp.Expr:
    r"""The function ell(x,m) exactly as defined in the current note."""

    if not isinstance(m, int):
        raise TypeError(f"ell index must be an integer, got {m!r}")
    if m == 0:
        return sp.Integer(1)
    if m < 0:
        reflected = ell(q - x, -m, b, q)
        return (-1) ** (-m // 2) * reflected if m % 2 == 0 else reflected

    value = sp.Pow(2, sp.Rational(1, 8)) if m % 2 else sp.Integer(1)
    required_parity = 1 if m % 2 else 0
    for r in range(m):
        for s in range(m - r):
            if (r + s) % 2 == required_parity:
                value *= x + r * b + s / b
    return sp.factor(value)


@dataclass(frozen=True)
class Component:
    """One term of v_n: coefficient times u_F tensor x_SCA."""

    auxiliary_parity: int
    sca_parity: int
    coefficient: sp.Expr
    name: str


def primary_components(label: int, gamma: sp.Expr) -> tuple[Component, ...]:
    """Return v_n for label=2n in {-1,0,1}."""

    if label == 0:
        return (Component(0, 0, sp.Integer(1), "1 x phi"),)
    if abs(label) == 1:
        return (
            Component(0, 1, sp.Integer(1), "1 x G phi"),
            Component(1, 0, gamma, "psi x phi"),
        )
    raise ValueError("this direct audit supports only n=-1/2, 0, and 1/2")


def fermion_rho(auxiliary_parities: tuple[int, int, int]) -> sp.Expr:
    r"""Level-1/2 auxiliary-fermion trilinear form in note slot order."""

    table = {
        (0, 0, 0): sp.Integer(1),
        (1, 1, 0): -sp.Integer(1),
        (1, 0, 1): -sp.Integer(1),
        (0, 1, 1): sp.Integer(1),
    }
    return table.get(auxiliary_parities, sp.Integer(0))


def sca_rho(
    sca_parities: tuple[int, int, int],
    weights: tuple[sp.Expr, sp.Expr, sp.Expr],
) -> sp.Expr:
    r"""The current note's fixed-parity SCA component table."""

    h1, h2, h3 = weights
    table = {
        (0, 0, 0): sp.Integer(1),
        (1, 1, 0): h1 + h2 - h3,
        (1, 0, 1): h1 - h2 + h3,
        (0, 1, 1): h1 - h2 - h3,
        (1, 0, 0): sp.Integer(1),
        (0, 1, 0): sp.Integer(1),
        (0, 0, 1): -sp.Integer(1),
        (1, 1, 1): -(h1 + h2 + h3 - HALF),
    }
    return table[sca_parities]


def crossing_sign(
    sca_parities: tuple[int, int, int],
    auxiliary_parities: tuple[int, int, int],
) -> int:
    """The vertex-action sign compatible with the ungraded pairing."""

    _x1, x2, _x3 = sca_parities
    _u1, _u2, u3 = auxiliary_parities
    exponent = x2 * u3
    return -1 if exponent % 2 else 1


def direct_rho(
    labels: tuple[int, int, int],
    gammas: tuple[sp.Expr, sp.Expr, sp.Expr],
    weights: tuple[sp.Expr, sp.Expr, sp.Expr],
) -> tuple[sp.Expr, tuple[tuple[str, sp.Expr], ...]]:
    r"""Expand rho_hat_a(v_1,v_2,v_3) term by term."""

    terms: list[tuple[str, sp.Expr]] = []
    for components in product(
        *(primary_components(label, gamma) for label, gamma in zip(labels, gammas))
    ):
        auxiliary = tuple(component.auxiliary_parity for component in components)
        fermion_value = fermion_rho(auxiliary)
        if fermion_value == 0:
            continue
        sca = tuple(component.sca_parity for component in components)
        coefficient = sp.prod(component.coefficient for component in components)
        value = (
            coefficient
            * crossing_sign(sca, auxiliary)
            * fermion_value
            * sca_rho(sca, weights)
        )
        names = ", ".join(component.name for component in components)
        terms.append((names, simplify(value)))
    return simplify(sum(value for _name, value in terms)), tuple(terms)


def direct_norm(label: int, weight: sp.Expr, gamma: sp.Expr) -> sp.Expr:
    """Direct BPZ norm of v_0 or v_{+/-1/2}."""

    if label == 0:
        return sp.Integer(1)
    if abs(label) == 1:
        return simplify(2 * weight - gamma**2)
    raise ValueError("this direct audit supports only n=-1/2, 0, and 1/2")


def reflect_to_nonnegative(
    labels: tuple[int, int, int],
    momenta: tuple[sp.Expr, sp.Expr, sp.Expr],
) -> tuple[tuple[int, int, int], tuple[sp.Expr, sp.Expr, sp.Expr]]:
    """Apply v_n(P)=v_{-n}(-P) independently on all three legs."""

    effective_labels = tuple(abs(label) for label in labels)
    effective_momenta = tuple(
        -momentum if label < 0 else momentum
        for label, momentum in zip(labels, momenta)
    )
    return effective_labels, effective_momenta


def product_norm(label: int, momentum: sp.Expr, b: sp.Expr, q: sp.Expr) -> sp.Expr:
    """Norm formula with negative labels evaluated by reflection."""

    if label < 0:
        return product_norm(-label, -momentum, b, q)

    return simplify(
        (-1) ** label
        * sp.Pow(2, -label)
        * ell(2 * momentum, 2 * label, b, q)
        * ell(q + 2 * momentum, 2 * label, b, q)
    )


def branching_ell_product(
    labels: tuple[int, int, int],
    momenta: tuple[sp.Expr, sp.Expr, sp.Expr],
    b: sp.Expr,
    q: sp.Expr,
) -> sp.Expr:
    """The four distinct ell factors in the note's branching formula."""

    effective_labels, effective_momenta = reflect_to_nonnegative(
        labels, momenta
    )
    k1, k2, k3 = effective_labels
    p1, p2, p3 = effective_momenta
    arguments = (
        q / 2 + p1 + p2 + p3,
        q / 2 - p1 + p2 + p3,
        q / 2 + p1 + p2 - p3,
        q / 2 - p1 + p2 - p3,
    )
    indices = (
        k1 + k2 + k3,
        -k1 + k2 + k3,
        k1 + k2 - k3,
        -k1 + k2 - k3,
    )
    return simplify(
        sp.prod(ell(x, m, b, q) for x, m in zip(arguments, indices))
    )


def branching_formula_squared(
    labels: tuple[int, int, int],
    momenta: tuple[sp.Expr, sp.Expr, sp.Expr],
    b: sp.Expr,
    q: sp.Expr,
) -> sp.Expr:
    r"""Square the displayed B_a formula without choosing square roots."""

    effective_labels, effective_momenta = reflect_to_nonnegative(
        labels, momenta
    )
    channel_parity = sum(effective_labels) % 2
    numerator = branching_ell_product(labels, momenta, b, q)
    denominator = sp.prod(
        ell(2 * momentum, 2 * label, b, q)
        * ell(q + 2 * momentum, 2 * label, b, q)
        for momentum, label in zip(effective_momenta, effective_labels)
    )
    return simplify((-1) ** channel_parity * numerator**2 / denominator)


def direct_branching_numerator(
    labels: tuple[int, int, int],
    rho: sp.Expr,
) -> sp.Expr:
    r"""Numerator of direct B_a over the common product of ell square roots."""

    active_count = sum(label != 0 for label in labels)
    return simplify(
        rho * sp.Pow(2, sp.Rational(active_count, 2)) * sp.I ** (-active_count)
    )


def displayed_branching_numerator(
    labels: tuple[int, int, int],
    momenta: tuple[sp.Expr, sp.Expr, sp.Expr],
    b: sp.Expr,
    q: sp.Expr,
) -> sp.Expr:
    r"""Numerator of the displayed unsquared B_a over the same roots."""

    channel_parity = sum(label != 0 for label in labels) % 2
    return simplify(
        (-sp.I) ** channel_parity
        * branching_ell_product(labels, momenta, b, q)
    )


def solve_first_primary(b: sp.Expr, momentum: sp.Expr) -> None:
    """Derive v_{+/-1/2} from the two Virasoro zero modes."""

    q = b + 1 / b
    weight = q**2 / 8 - momentum**2 / 2
    denominator = 1 / b - b
    l0_one = sp.Matrix(
        [
            [(weight + HALF) / b, -1],
            [2 * weight, (weight - HALF) / b - b],
        ]
    ) / denominator

    gamma = sp.symbols("gamma")
    trial = sp.Matrix([1, gamma])
    image = l0_one * trial
    equation = simplify(image[1] - gamma * image[0])
    require_zero(
        "Virasoro-primary equation",
        equation - (gamma**2 - q * gamma + 2 * weight) / denominator,
    )
    require_zero(
        "v_{1/2} eigenvector",
        (gamma**2 - q * gamma + 2 * weight).subs(gamma, q / 2 + momentum),
    )
    require_zero(
        "v_{-1/2} eigenvector",
        (gamma**2 - q * gamma + 2 * weight).subs(gamma, q / 2 - momentum),
    )


def main() -> None:
    b = sp.symbols("b", nonzero=True)
    p1, p2, p3 = sp.symbols("P_1 P_2 P_3")
    q = b + 1 / b
    momenta = (p1, p2, p3)
    weights = tuple(q**2 / 8 - momentum**2 / 2 for momentum in momenta)
    solve_first_primary(b, p1)

    print("First positive Virasoro primary:")
    print("  v_{1/2}(P) = 1 x G_{-1/2}|P> + (Q/2+P) psi_{-1/2} x |P>")
    print("\nNorm check against the ell formula:")
    norm_check_count = 0
    for slot, (momentum, weight) in enumerate(zip(momenta, weights), start=1):
        for label in (-1, 0, 1):
            gamma = q / 2 + label * momentum if label else sp.S.Zero
            direct = direct_norm(label, weight, gamma)
            formula = product_norm(label, momentum, b, q)
            residual = simplify(direct - formula)
            print(
                f"  slot {slot}, n={sp.Rational(label, 2)}: "
                f"N_direct = {direct}; residual = {residual}"
            )
            require_zero(f"slot-{slot}, label-{label} norm formula", residual)
            norm_check_count += 1

    print("\nDirect comparison with the four-ell branching formula:")
    squared_mismatches: list[
        tuple[
            tuple[int, int, int],
            sp.Expr,
            tuple[tuple[str, sp.Expr], ...],
        ]
    ] = []
    unsquared_mismatches: list[tuple[tuple[int, int, int], sp.Expr]] = []
    corrected_unsquared_mismatches: list[
        tuple[tuple[int, int, int], sp.Expr]
    ] = []
    for labels in product((-1, 0, 1), repeat=3):
        gammas = tuple(
            q / 2 + label * momentum if label else sp.S.Zero
            for label, momentum in zip(labels, momenta)
        )
        rho, terms = direct_rho(labels, gammas, weights)
        norms = sp.prod(
            direct_norm(label, weight, gamma)
            for label, weight, gamma in zip(labels, weights, gammas)
        )
        direct_b_squared = simplify(rho**2 / norms)
        formula_b_squared = branching_formula_squared(labels, momenta, b, q)
        squared_residual = simplify(direct_b_squared - formula_b_squared)
        direct_b_numerator = direct_branching_numerator(labels, rho)
        formula_b_numerator = displayed_branching_numerator(
            labels, momenta, b, q
        )
        unsquared_residual = simplify(
            direct_b_numerator - formula_b_numerator
        )
        corrected_formula_b_numerator = simplify(
            (-1) ** abs(labels[2]) * formula_b_numerator
        )
        corrected_unsquared_residual = simplify(
            direct_b_numerator - corrected_formula_b_numerator
        )
        squared_status = "PASS" if squared_residual == 0 else "FAIL"
        unsquared_status = "PASS" if unsquared_residual == 0 else "FAIL"
        n_labels = tuple(sp.Rational(label, 2) for label in labels)
        print(
            f"  n={n_labels}: B^2 {squared_status}; B {unsquared_status}; "
            f"rho_direct={rho}"
        )
        if squared_residual != 0:
            squared_mismatches.append((labels, squared_residual, terms))
        if unsquared_residual != 0:
            unsquared_mismatches.append((labels, unsquared_residual))
        if corrected_unsquared_residual != 0:
            corrected_unsquared_mismatches.append(
                (labels, corrected_unsquared_residual)
            )

    total_branch_checks = 27
    print(f"\nNorm summary: {norm_check_count}/{norm_check_count} checks pass.")
    print(
        "Squared branching summary: "
        f"{total_branch_checks - len(squared_mismatches)}/"
        f"{total_branch_checks} checks pass."
    )
    print(
        "Unsquared displayed-formula summary: "
        f"{total_branch_checks - len(unsquared_mismatches)}/"
        f"{total_branch_checks} checks pass."
    )
    print(
        "Unsquared formula with (-1)^(2 n_3): "
        f"{total_branch_checks - len(corrected_unsquared_mismatches)}/"
        f"{total_branch_checks} checks pass."
    )

    if squared_mismatches:
        print("Squared-coefficient mismatching component expansions:")
        for labels, residual, terms in squared_mismatches:
            print(f"  labels k=2n={labels}")
            for names, value in terms:
                print(f"    {names}: {value}")
            print(f"    squared-coefficient residual: {residual}")
        raise AssertionError(
            "the squared four-ell branching formula does not agree with "
            "the direct calculation"
        )

    if corrected_unsquared_mismatches:
        raise AssertionError(
            "the proposed third-slot correction does not resolve every "
            "direct unsquared coefficient"
        )

    if unsquared_mismatches:
        first_labels, first_residual = unsquared_mismatches[0]
        print(
            "The formula without the third-leg phase is retained only as a "
            "negative control.  Its first mismatch is at "
            f"k=2n={first_labels}: {first_residual}"
        )

    print(
        "All direct norm, squared-branching, and "
        "(-1)^(2 n_3)-corrected unsquared checks passed."
    )


if __name__ == "__main__":
    main()
