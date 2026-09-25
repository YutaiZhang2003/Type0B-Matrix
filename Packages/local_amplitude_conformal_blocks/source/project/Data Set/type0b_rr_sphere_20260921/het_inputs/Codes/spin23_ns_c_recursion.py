"""Fixed-h NS sphere c-recursion with external upper supermultiplet components.

Leg order is (1,2,3,4) at (0,z,1,infinity).  An alpha of one means
G_{-1/2}|h_i>, NOT a superconformal primary of weight h_i+1/2.  Coefficients
are in the fixed-parity trilinear convention of ns_three_point, before the
physical component phases in spin23_singlet_amplitudes.

The recursion is Hadasz--Jaskolski--Suchanek, hep-th/0611266, section 6.
The global seed is the osp(1|2) endpoint contraction (cf. Belavin--Geiko,
1806.09563, Appendix A), including endpoint reflection for fixed-parity
forms.  No Gram matrices, symbolic Ward reduction, or fitted coefficients
are used here.  The eight even-total-parity external component patterns
are supported; higher external descendants are outside this API.
"""

from __future__ import annotations

import cmath
import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import mpmath as mp


Alphas = tuple[int, int, int, int]


class RecursionPoleCollision(ArithmeticError):
    """A pole/coalescence needs a confluent limit, not an arbitrary regulator."""


def _alphas(values: Sequence[int]) -> Alphas:
    values = tuple(values)
    if len(values) != 4 or any(a not in (0, 1) for a in values):
        raise ValueError("external_alphas must contain four zeros or ones")
    if sum(values) % 2:
        raise NotImplementedError("only even total external fermion parity is supported")
    return values  # type: ignore[return-value]


def _rf(x, n: int):
    result = 1
    for j in range(n):
        result *= x + j
    return result


def _global_endpoint(k, m, alpha, d1, d2, d3):
    """Global trilinear tensor with descendants on at most one endpoint.

    This is the p=0 term of the eight-case osp tensor.  The middle leg has
    no L_-1 modes.  Reflect (100) and (110) to implement the fixed-parity
    convention, rather than confusing it with a component-map convention.
    """
    if alpha in ((1, 0, 0), (1, 1, 0)):
        sign = -1 if alpha[1] else 1
        return sign * _global_endpoint(m, k, alpha[::-1], d3, d2, d1)
    a, b = d2 + d3 - d1, d1 + d2 - d3 - m
    if alpha == (0, 0, 0):
        return _rf(a, m) * _rf(b, k)
    if alpha == (0, 1, 0):
        return _rf(a + 0.5, m) * _rf(b + 0.5, k)
    if alpha == (0, 0, 1):
        return _rf(a + 0.5, m) * _rf(b - 0.5, k)
    if alpha == (1, 0, 1):
        return _rf(a, m) * _rf(b, k) * (d1 - d2 + d3)
    if alpha == (0, 1, 1):
        return -_rf(a, m + 1) * _rf(b, k)
    if alpha == (1, 1, 1):
        return _rf(a + 0.5, m) * _rf(b + 0.5, k) * (d1 + d2 + d3 - 0.5)
    raise AssertionError(alpha)


def global_seed(level2: int, h, weights: Sequence, external_alphas: Sequence[int]):
    """Coefficient of z^(n+beta/2) at c=infinity, at fixed primary weights.

    g = rho(w4,w3,L_-1^n G_-1/2^beta h)
        rho(L_-1^n G_-1/2^beta h,w2,w1) / [n! (2h)_(n+beta)].
    In particular the M-pattern odd leading term is
    -(h+h3-h4)(h+h2-h1)/(2h), not the primary seed 1/(2h).
    """
    a1, a2, a3, a4 = _alphas(external_alphas)
    h1, h2, h3, h4 = weights
    n, beta = divmod(level2, 2)
    left = _global_endpoint(0, n, (a4, a3, beta), h4, h3, h)
    right = _global_endpoint(n, 0, (beta, a2, a1), h, h2, h1)
    return left * right / (math.factorial(n) * _rf(2 * h, n + beta))


class _Recursor:
    """Memoized scalar coefficients; pole/fusion data are shared across patterns."""

    def __init__(self, weights: Sequence[complex], *, digits: int = 0, pole_tolerance=0.0):
        self.ctx = mp.mp.clone() if digits else None
        if self.ctx is not None:
            self.ctx.dps = digits
        self.number = complex if self.ctx is None else self.ctx.mpc
        self.sqrt = cmath.sqrt if self.ctx is None else self.ctx.sqrt
        self.sqrt2 = math.sqrt(2) if self.ctx is None else self.ctx.sqrt(2)
        self.weights = tuple(self.number(w) for w in weights)
        if len(self.weights) != 4:
            raise ValueError("four external primary weights are required")
        self.pole_tolerance = pole_tolerance
        self.memo = {}
        self.poles = {}
        self.fusions = {}
        self.max_condition = 1.0

    def pole(self, h, r, s):
        key = (h, r, s)
        if key not in self.poles:
            root = self.sqrt(16 * h * h + 8 * (r * s - 1) * h + (r - s) ** 2)
            y = -(4 * h + r * s - 1 + root) / (r * r - 1)
            dy = -(4 + (16 * h + 4 * (r * s - 1)) / root) / (r * r - 1)
            pole = 7.5 + 3 * y + 3 / y
            jacobian = -3 * dy * (1 - 1 / (y * y))
            b = self.sqrt(y)
            norm = self.number(0.5)
            for p in range(1 - r, r + 1):
                for q in range(1 - s, s + 1):
                    if (p + q) % 2 or (p, q) in ((0, 0), (r, s)):
                        continue
                    norm *= self.sqrt2 / (p * b + q / b)
            self.poles[key] = pole, jacobian * norm, b
        return self.poles[key]

    def fusion_pair(self, h, r, s, parity):
        key = (h, r, s, parity)
        if key not in self.fusions:
            _, ja, b = self.pole(h, r, s)
            background = b + 1 / b
            product = ja
            h1, h2, h3, h4 = self.weights
            for ha, hb in ((h1, h2), (h4, h3)):
                a = self.sqrt(background * background / 4 - 2 * ha)
                d = self.sqrt(background * background / 4 - 2 * hb)
                for p in range(1 - r, r, 2):
                    for q in range(1 - s, s, 2):
                        if (p + q - r - s) % 4 != (0 if parity else 2):
                            continue
                        x = p * b + q / b
                        product *= (2 * a - 2 * d - x) / (2 * self.sqrt2)
                        product *= (2 * a + 2 * d + x) / (2 * self.sqrt2)
            self.fusions[key] = product
        return self.fusions[key]

    def coefficient(self, level2: int, h, c, alphas: Alphas):
        key = (level2, h, c, alphas)
        if key in self.memo:
            return self.memo[key]
        value = global_seed(level2, h, self.weights, alphas)
        terms = [value]
        # Both endpoints have the same structure parity since sum(alpha) is even.
        fusion_parity = alphas[0] ^ alphas[1] ^ (level2 % 2)
        for r in range(2, level2 + 1):
            for s in range(1, level2 // r + 1):
                if (r + s) % 2:
                    continue
                rs = r * s
                pole, _, _ = self.pole(h, r, s)
                denominator = c - pole
                if abs(denominator) <= self.pole_tolerance * max(1, abs(c), abs(pole)):
                    raise RecursionPoleCollision(f"c={c} collides with ({r},{s}) at h={h}")
                residue = self.fusion_pair(h, r, s, fusion_parity)
                # Null-vector transport in the fixed-parity endpoint convention.
                # With stars on the outer legs this is NOT simply alpha_3.
                if (rs * alphas[1]) % 2:
                    residue = -residue
                terms.append(residue / denominator * self.coefficient(
                    level2 - rs, h + self.number(rs) / 2, pole, alphas
                ))
        if self.ctx is None:
            value = complex(math.fsum(v.real for v in terms), math.fsum(v.imag for v in terms))
            finite = math.isfinite(value.real) and math.isfinite(value.imag)
        else:
            value = self.ctx.fsum(terms)
            finite = self.ctx.isfinite(value)
        if not finite:
            raise ArithmeticError(f"nonfinite c-recursion coefficient at twice level {level2}")
        absolute_sum = sum(abs(v) for v in terms)
        # A coefficient that vanishes by symmetry needs an absolute, not an
        # infinite relative-error test.  This is a mixed absolute/relative scale.
        condition = absolute_sum / max(1, abs(value))
        self.max_condition = max(self.max_condition, float(condition))
        self.memo[key] = value
        return value


@dataclass(frozen=True)
class CRecursionResult:
    coefficients: Mapping[Alphas, tuple[complex, ...]]
    maximum_cancellation: float | None
    high_precision: bool


def block_coefficients(
    *,
    c: complex,
    h_internal: complex,
    external_weights: Sequence[complex],
    external_patterns: Sequence[Sequence[int]],
    maximum_twice_level: int,
    force_high_precision: bool = False,
    digits: int = 70,
    cancellation_limit: float = 1.0e6,
    pole_tolerance: float = 0.0,
) -> CRecursionResult:
    """Evaluate all requested components, retrying risky native recursion in mp.

    The mp context is private (no global dps mutation).  Exact pole collisions
    raise; no imaginary regulator, interpolation, or inverse-Gram fallback is
    hidden in this routine.  `maximum_cancellation` describes the native pass
    when attempted, not an estimate of the full amplitude's quadrature error.
    """
    if not isinstance(maximum_twice_level, int) or maximum_twice_level < 0:
        raise ValueError("maximum_twice_level must be a nonnegative integer")
    if digits < 30 or not math.isfinite(cancellation_limit) or cancellation_limit <= 0:
        raise ValueError("digits must be >=30 and cancellation_limit positive and finite")
    patterns = tuple(dict.fromkeys(_alphas(p) for p in external_patterns))

    def evaluate(engine):
        h, central = engine.number(h_internal), engine.number(c)
        return {a: tuple(complex(engine.coefficient(n, h, central, a))
                         for n in range(maximum_twice_level + 1)) for a in patterns}

    condition = None
    if not force_high_precision:
        native = _Recursor(external_weights, pole_tolerance=pole_tolerance)
        try:
            coefficients = evaluate(native)
            condition = native.max_condition
            if condition <= cancellation_limit:
                return CRecursionResult(coefficients, condition, False)
        except (ArithmeticError, ValueError):
            # Near coalescences and nonfinite intermediates need a fresh mp pass.
            condition = native.max_condition
    precise = _Recursor(external_weights, digits=digits, pole_tolerance=pole_tolerance)
    try:
        coefficients = evaluate(precise)
    except ZeroDivisionError as error:
        raise RecursionPoleCollision("coincident c-recursion poles require a confluent limit") from error
    # Keep at least 20 digits after estimated cancellation before rounding to complex128.
    if precise.max_condition > 10.0 ** (digits - 20):
        raise ArithmeticError("insufficient c-recursion precision; increase digits")
    return CRecursionResult(coefficients, condition, True)
