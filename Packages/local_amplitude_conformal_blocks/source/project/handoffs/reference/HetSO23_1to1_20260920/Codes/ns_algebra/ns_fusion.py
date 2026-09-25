#!/usr/bin/env python3
"""NS degenerate-representation and fusion data in Xi Yin's convention.

The ordinary Virasoro central charge used throughout this module is

    c = 3/2 + 3 Q**2,              Q = b + b**(-1).

For positive integers ``r,s`` with ``r+s`` even,

    h_rs = (Q**2 - (r*b + s/b)**2)/8

has an NS singular vector at level ``r*s/2``.  All functions return SymPy
expressions.  Exact inputs therefore stay exact, while floating-point or
complex inputs give numerical SymPy expressions that can be evaluated with
``numeric``.

The value of ``c`` does not determine the branch of ``b``.  Residue
functions consequently take ``b`` itself, and the fixed-weight helpers
return the selected ``b`` branch explicitly.
"""

from __future__ import annotations

from typing import Iterator

import sympy as sp


__all__ = [
    "Q",
    "c_xi",
    "h_of_lambda",
    "lambda_from_h",
    "h_rs",
    "b2_rs",
    "b_rs",
    "c_rs",
    "J_rs",
    "A_rs",
    "P",
    "P0",
    "P1",
    "sigma",
    "S",
    "mu",
    "numeric",
]


def _validate_label(r: int, s: int) -> None:
    """Validate an NS Kac label without imposing the recursion choice r >= 2."""

    if not isinstance(r, int) or not isinstance(s, int):
        raise TypeError("r and s must be integers")
    if r <= 0 or s <= 0:
        raise ValueError("r and s must be positive")
    if (r + s) % 2:
        raise ValueError("an NS Kac label must have r+s even")


def _validate_fixed_h_label(r: int, s: int) -> None:
    """Validate a Kac label for the standard fixed-h recursion branch."""

    _validate_label(r, s)
    if r < 2:
        raise ValueError(
            "the standard fixed-h c-recursion representative requires r>=2"
        )


def _validate_alpha(alpha: int) -> None:
    """Validate a lower/upper three-point-structure index."""

    if alpha not in (0, 1):
        raise ValueError("alpha must be 0 or 1")


def _validate_sign(sign: int, name: str) -> None:
    """Validate a two-valued algebraic branch sign."""

    if sign not in (-1, 1):
        raise ValueError(f"{name} must be +1 or -1")


def Q(b: sp.Expr) -> sp.Expr:
    """Return the background charge ``Q=b+b**(-1)``."""

    b = sp.sympify(b)
    if b == 0:
        raise ValueError("b must be nonzero")
    return b + 1 / b


def c_xi(b: sp.Expr) -> sp.Expr:
    """Return Xi Yin's ordinary central charge ``c=3/2+3Q**2``."""

    charge = Q(b)
    return sp.Rational(3, 2) + 3 * charge**2


def h_of_lambda(lam: sp.Expr, b: sp.Expr) -> sp.Expr:
    """Return ``h(lambda)=(Q**2-lambda**2)/8``."""

    lam = sp.sympify(lam)
    charge = Q(b)
    return (charge**2 - lam**2) / 8


def lambda_from_h(
    h: sp.Expr,
    b: sp.Expr,
    *,
    sign: int = 1,
) -> sp.Expr:
    """Recover a momentum branch ``lambda=sign*sqrt(Q**2-8h)``."""

    _validate_sign(sign, "sign")
    h = sp.sympify(h)
    return sign * sp.sqrt(Q(b) ** 2 - 8 * h)


def h_rs(r: int, s: int, b: sp.Expr) -> sp.Expr:
    """Return the NS degenerate weight with label ``(r,s)``."""

    _validate_label(r, s)
    b = sp.sympify(b)
    lam = r * b + s / b
    return h_of_lambda(lam, b)


def b2_rs(
    r: int,
    s: int,
    h: sp.Expr,
    *,
    root: int = 1,
) -> sp.Expr:
    """Solve ``h=h_rs`` for ``b**2`` on one of the two quadratic roots.

    ``root=+1`` is the standard branch used after choosing ``r>=2`` in
    c-recursion.  ``root=-1`` returns the other algebraic solution.
    """

    _validate_fixed_h_label(r, s)
    _validate_sign(root, "root")
    h = sp.sympify(h)
    discriminant = (r - s) ** 2 + 8 * (r * s - 1) * h + 16 * h**2
    numerator = r * s - 1 + 4 * h + root * sp.sqrt(discriminant)
    return numerator / (1 - r**2)


def b_rs(
    r: int,
    s: int,
    h: sp.Expr,
    *,
    root: int = 1,
    b_sign: int = 1,
) -> sp.Expr:
    """Return a branch of ``b`` solving the fixed-weight pole equation."""

    _validate_sign(b_sign, "b_sign")
    return b_sign * sp.sqrt(b2_rs(r, s, h, root=root))


def c_rs(
    r: int,
    s: int,
    h: sp.Expr,
    *,
    root: int = 1,
) -> sp.Expr:
    """Return the fixed-weight pole position in Xi's central charge."""

    x = b2_rs(r, s, h, root=root)
    return sp.Rational(15, 2) + 3 * (x + 1 / x)


def J_rs(
    r: int,
    s: int,
    h: sp.Expr,
    *,
    root: int = 1,
) -> sp.Expr:
    """Return ``J_rs(h)=-d c_rs(h)/dh`` for the selected root."""

    x = b2_rs(r, s, h, root=root)
    numerator = -24 * (x**2 - 1)
    denominator = (1 - r**2) * x**2 - (1 - s**2)
    return numerator / denominator


def A_rs(r: int, s: int, b: sp.Expr) -> sp.Expr:
    """Return the inverse linearized norm of the normalized NS null state.

    The normalization is

    ``1/2 * product[(p*b+q/b)/sqrt(2)]**(-1)``,

    with ``p=1-r,...,r``, ``q=1-s,...,s``, ``p+q`` even, and the pairs
    ``(0,0)`` and ``(r,s)`` omitted.
    """

    _validate_label(r, s)
    b = sp.sympify(b)
    if b == 0:
        raise ValueError("b must be nonzero")
    factors: list[sp.Expr] = []
    for p in range(1 - r, r + 1):
        for q in range(1 - s, s + 1):
            if (p + q) % 2:
                continue
            if (p, q) in ((0, 0), (r, s)):
                continue
            factors.append(sp.sqrt(2) / (p * b + q / b))
    return sp.Rational(1, 2) * sp.prod(factors)


def _fusion_pairs(r: int, s: int, alpha: int) -> Iterator[tuple[int, int]]:
    """Yield the parity-selected lattice pairs in ``P^alpha_rs``."""

    target = 2 if alpha == 0 else 0
    for p in range(1 - r, r, 2):
        for q in range(1 - s, s, 2):
            if (p + q - (r + s)) % 4 == target:
                yield p, q


def P(
    r: int,
    s: int,
    alpha: int,
    d_i: sp.Expr,
    d_j: sp.Expr,
    b: sp.Expr,
    *,
    lambda_i: sp.Expr | None = None,
    lambda_j: sp.Expr | None = None,
) -> sp.Expr:
    """Return the NS fusion polynomial ``P^alpha_rs(d_i,d_j;b)``.

    If a momentum is omitted, the principal symbolic square root returned
    by ``lambda_from_h`` is used.  Passing ``lambda_i`` and ``lambda_j``
    explicitly is useful when a calculation tracks momentum branches.
    """

    _validate_label(r, s)
    _validate_alpha(alpha)
    d_i = sp.sympify(d_i)
    d_j = sp.sympify(d_j)
    b = sp.sympify(b)
    if lambda_i is None:
        lambda_i = lambda_from_h(d_i, b)
    else:
        lambda_i = sp.sympify(lambda_i)
    if lambda_j is None:
        lambda_j = lambda_from_h(d_j, b)
    else:
        lambda_j = sp.sympify(lambda_j)

    denominator = 2 * sp.sqrt(2)
    factors: list[sp.Expr] = []
    for p, q in _fusion_pairs(r, s, alpha):
        shift = p * b + q / b
        factors.append((lambda_i - lambda_j + shift) / denominator)
        factors.append((lambda_i + lambda_j + shift) / denominator)
    return sp.prod(factors)


def P0(
    r: int,
    s: int,
    d_i: sp.Expr,
    d_j: sp.Expr,
    b: sp.Expr,
    *,
    lambda_i: sp.Expr | None = None,
    lambda_j: sp.Expr | None = None,
) -> sp.Expr:
    """Return the even fusion polynomial ``P^0_rs``."""

    return P(
        r,
        s,
        0,
        d_i,
        d_j,
        b,
        lambda_i=lambda_i,
        lambda_j=lambda_j,
    )


def P1(
    r: int,
    s: int,
    d_i: sp.Expr,
    d_j: sp.Expr,
    b: sp.Expr,
    *,
    lambda_i: sp.Expr | None = None,
    lambda_j: sp.Expr | None = None,
) -> sp.Expr:
    """Return the odd fusion polynomial ``P^1_rs``."""

    return P(
        r,
        s,
        1,
        d_i,
        d_j,
        b,
        lambda_i=lambda_i,
        lambda_j=lambda_j,
    )


def sigma(
    r: int,
    s: int,
    alpha: int,
    d_i: sp.Expr,
    d_j: sp.Expr,
    b: sp.Expr,
    *,
    lambda_i: sp.Expr | None = None,
    lambda_j: sp.Expr | None = None,
) -> sp.Expr:
    """Return the first-slot singular-vector matrix element.

    It equals ``P^alpha`` for even ``r*s`` and ``P^(1-alpha)`` for odd
    ``r*s``.
    """

    _validate_label(r, s)
    _validate_alpha(alpha)
    polynomial_index = alpha if (r * s) % 2 == 0 else 1 - alpha
    return P(
        r,
        s,
        polynomial_index,
        d_i,
        d_j,
        b,
        lambda_i=lambda_i,
        lambda_j=lambda_j,
    )


def S(n: int, alpha: int) -> int:
    """Return the third-slot reflection factor ``S_n^alpha``."""

    if not isinstance(n, int):
        raise TypeError("n must be an integer")
    _validate_alpha(alpha)
    return 1 if alpha == 0 else (-1) ** n


def mu(
    r: int,
    s: int,
    alpha: int,
    d: sp.Expr,
    h: sp.Expr,
    b: sp.Expr,
    *,
    lambda_d: sp.Expr | None = None,
    lambda_h: sp.Expr | None = None,
    lambda_shifted_h: sp.Expr | None = None,
) -> sp.Expr:
    """Return the torus/self-sewing fusion factor ``mu^alpha_rs(d;h)``.

    With ``N=r*s/2``, the even-null formula is

    ``P^alpha(d,h+N) P^alpha(d,h)``.

    For an odd null it is

    ``S_rs^alpha P^(1-alpha)(d,h+N) P^alpha(d,h)``.
    """

    _validate_label(r, s)
    _validate_alpha(alpha)
    d = sp.sympify(d)
    h = sp.sympify(h)
    level = sp.Rational(r * s, 2)
    shifted_h = h + level
    shifted_index = alpha if (r * s) % 2 == 0 else 1 - alpha
    reflected = S(r * s, alpha) if (r * s) % 2 else 1
    shifted_factor = P(
        r,
        s,
        shifted_index,
        d,
        shifted_h,
        b,
        lambda_i=lambda_d,
        lambda_j=lambda_shifted_h,
    )
    unshifted_factor = P(
        r,
        s,
        alpha,
        d,
        h,
        b,
        lambda_i=lambda_d,
        lambda_j=lambda_h,
    )
    return reflected * shifted_factor * unshifted_factor


def numeric(expression: sp.Expr, digits: int = 30) -> sp.Expr:
    """Evaluate a symbolic result to ``digits`` decimal digits."""

    if not isinstance(digits, int):
        raise TypeError("digits must be an integer")
    if digits <= 0:
        raise ValueError("digits must be positive")
    return sp.N(sp.sympify(expression), digits)
