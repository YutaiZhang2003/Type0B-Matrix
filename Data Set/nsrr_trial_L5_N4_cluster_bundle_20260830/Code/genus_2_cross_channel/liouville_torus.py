#!/usr/bin/env python3
"""Liouville torus one-point functions from Virasoro blocks.

This module keeps the CFT data outside ``virasoro_blocks.py``.  It implements
the real-``b`` Upsilon_b special function, the DOZZ structure constant, and the
one-dimensional Liouville momentum integral for a diagonal torus one-point
function.  The public torus integral follows the Balthazar-Rodriguez-Yin
normalization: primaries are labelled by ``V_P``, have chiral weight
``h=Q^2/4+P^2``, and are two-point normalized to ``pi delta(P-P')``.
"""

from __future__ import annotations

import argparse
import cmath
import math
from dataclasses import dataclass
from typing import Iterable

import mpmath as mp
import numpy as np

try:
    from virasoro_blocks import TorusOnePointVirasoroBlock
except ImportError:  # pragma: no cover - supports package-style execution
    from plumbing.virasoro_blocks import TorusOnePointVirasoroBlock


TWO_PI_I = 2.0j * math.pi


def parse_complex(value: str) -> complex:
    return complex(value.replace("i", "j"))


def format_complex(value: complex) -> str:
    return f"{value.real:+.12e}{value.imag:+.12e}j"


def _to_python_complex(value: mp.mpc | complex | float) -> complex:
    value = mp.mpc(value)
    return complex(float(mp.re(value)), float(mp.im(value)))


def q_from_tau(tau: complex) -> complex:
    return cmath.exp(TWO_PI_I * tau)


@dataclass
class UpsilonB:
    """Numerical Upsilon_b evaluator for positive real b.

    The strip integral is used for ``0 < Re(x) < Q`` and the standard shift
    relations analytically continue to nearby strips.  The normalization is
    Upsilon_b(Q/2)=1, and Upsilon'_b(0)=Upsilon_b(b) in this normalization.
    """

    b: float
    dps: int = 40
    max_shift_steps: int = 200

    def __post_init__(self) -> None:
        if self.b <= 0:
            raise ValueError("b must be positive and real for this Upsilon_b implementation")
        self._log_cache: dict[tuple[str, str], mp.mpc] = {}

    @property
    def q_background(self) -> mp.mpf:
        self._set_precision()
        b = mp.mpf(self.b)
        return b + 1 / b

    def _set_precision(self) -> None:
        mp.mp.dps = self.dps

    def _cache_key(self, value: mp.mpc) -> tuple[str, str]:
        return (
            mp.nstr(mp.re(value), self.dps + 8, min_fixed=0, max_fixed=0),
            mp.nstr(mp.im(value), self.dps + 8, min_fixed=0, max_fixed=0),
        )

    def log_gamma_ratio(self, value: complex | mp.mpc) -> mp.mpc:
        """Return log gamma(value) - log gamma(1-value)."""
        self._set_precision()
        value = mp.mpc(value)
        return mp.loggamma(value) - mp.loggamma(1 - value)

    def gamma_ratio(self, value: complex | mp.mpc) -> mp.mpc:
        return mp.e ** self.log_gamma_ratio(value)

    def _log_upsilon_strip(self, value: mp.mpc) -> mp.mpc:
        self._set_precision()
        b = mp.mpf(self.b)
        q_background = b + 1 / b
        half_shifted = q_background / 2 - value

        def integrand(t: mp.mpf) -> mp.mpc:
            if t == 0:
                return -(half_shifted * half_shifted)
            denominator = mp.sinh(b * t / 2) * mp.sinh(t / (2 * b))
            numerator = (
                half_shifted * half_shifted * mp.e ** (-t)
                - mp.sinh(half_shifted * t / 2) ** 2 / denominator
            )
            return numerator / t

        return mp.quad(integrand, [0, 1, mp.inf])

    def log_upsilon(self, value: complex | mp.mpc) -> mp.mpc:
        """Return log Upsilon_b(value), analytically continued by shifts."""
        self._set_precision()
        value = mp.mpc(value)
        key = self._cache_key(value)
        if key in self._log_cache:
            return self._log_cache[key]

        original_value = value
        b = mp.mpf(self.b)
        q_background = b + 1 / b
        log_b = mp.log(b)
        accumulated = mp.mpc(0)

        for _ in range(self.max_shift_steps):
            real_part = mp.re(value)
            if 0 < real_part < q_background:
                result = accumulated + self._log_upsilon_strip(value)
                self._log_cache[key] = result
                return result

            if real_part >= q_background:
                shifted = value - b
                accumulated += self.log_gamma_ratio(b * shifted) + (1 - 2 * b * shifted) * log_b
                value = shifted
            else:
                shifted = value + b
                accumulated -= self.log_gamma_ratio(b * value) + (1 - 2 * b * value) * log_b
                value = shifted

        raise RuntimeError(f"failed to shift Upsilon_b argument {original_value!r} into the strip")

    def upsilon(self, value: complex | mp.mpc) -> mp.mpc:
        return mp.e ** self.log_upsilon(value)

    def log_upsilon_prime_zero(self) -> mp.mpc:
        return self.log_upsilon(mp.mpf(self.b))

    def upsilon_prime_zero(self) -> mp.mpc:
        return mp.e ** self.log_upsilon_prime_zero()


def liouville_weight_from_lambda(b: float, lam: complex) -> complex:
    q_background = b + 1.0 / b
    return 0.25 * (q_background * q_background - lam * lam)


def liouville_weight_from_alpha(b: float, alpha: complex) -> complex:
    q_background = b + 1.0 / b
    return alpha * (q_background - alpha)


def lambda_from_alpha(b: float, alpha: complex) -> complex:
    return 2.0 * alpha - (b + 1.0 / b)


def alpha_from_lambda(b: float, lam: complex) -> complex:
    return 0.5 * (b + 1.0 / b + lam)


def alpha_from_yin_momentum(b: float, momentum: complex) -> complex:
    return 0.5 * (b + 1.0 / b) + 1.0j * momentum


def lambda_from_yin_momentum(momentum: complex) -> complex:
    return 2.0j * momentum


def dozz_structure_constant_alpha(
    special: UpsilonB,
    alpha1: complex,
    alpha2: complex,
    alpha3: complex,
    *,
    mu: complex = 1.0,
    include_cosmological_prefactor: bool = True,
) -> complex:
    """Return the standard DOZZ constant C(alpha1, alpha2, alpha3)."""
    special._set_precision()
    b = mp.mpf(special.b)
    q_background = b + 1 / b
    alpha_values = [mp.mpc(alpha1), mp.mpc(alpha2), mp.mpc(alpha3)]

    log_constant = special.log_upsilon_prime_zero()
    for alpha in alpha_values:
        log_constant += special.log_upsilon(2 * alpha)

    denominator_args = [
        sum(alpha_values) - q_background,
        alpha_values[0] + alpha_values[1] - alpha_values[2],
        alpha_values[0] + alpha_values[2] - alpha_values[1],
        alpha_values[1] + alpha_values[2] - alpha_values[0],
    ]
    for arg in denominator_args:
        log_constant -= special.log_upsilon(arg)

    if include_cosmological_prefactor:
        log_base = (
            mp.log(mp.pi * mp.mpc(mu))
            + special.log_gamma_ratio(b * b)
            + (2 - 2 * b * b) * mp.log(b)
        )
        log_constant += (q_background - sum(alpha_values)) / b * log_base

    return _to_python_complex(mp.e ** log_constant)


def dozz_structure_constant_lambda(
    special: UpsilonB,
    external_lambda: complex,
    internal_lambda: complex,
    *,
    mu: complex = 1.0,
    include_cosmological_prefactor: bool = True,
) -> complex:
    """Return the torus one-point Liouville coefficient in HJS conventions.

    ``internal_lambda`` is integrated along ``i R_+`` and gives
    ``Delta=(Q^2-internal_lambda^2)/4``.  ``external_lambda`` is the momentum of
    the inserted primary.  This is the DOZZ constant with

        alpha_1=(Q+internal_lambda)/2,
        alpha_2=(Q+external_lambda)/2,
        alpha_3=(Q-internal_lambda)/2.
    """
    if abs(complex(internal_lambda)) == 0:
        return 0.0 + 0.0j

    special._set_precision()
    q_background = special.q_background
    return dozz_structure_constant_alpha(
        special,
        0.5 * (q_background + mp.mpc(internal_lambda)),
        0.5 * (q_background + mp.mpc(external_lambda)),
        0.5 * (q_background - mp.mpc(internal_lambda)),
        mu=mu,
        include_cosmological_prefactor=include_cosmological_prefactor,
    )


def yin_structure_constant_momentum(
    special: UpsilonB,
    p1: complex,
    p2: complex,
    p3: complex,
    *,
    mu: complex = 1.0,
    include_cosmological_prefactor: bool = False,
) -> complex:
    """Return the delta-normalized structure constant of Balthazar-Rodriguez-Yin.

    The vertex operators are labelled by Liouville momenta P and normalized so
    that the two-point function is ``pi delta(P-P')``.  With the default
    ``include_cosmological_prefactor=False`` this is the coefficient ``C`` in
    their equation (2.6).  The optional prefactor multiplies the result by the
    momentum-independent factor shown in their equation (2.5).
    """
    special._set_precision()
    b = mp.mpf(special.b)
    q_background = b + 1 / b
    p_values = [mp.mpc(p1), mp.mpc(p2), mp.mpc(p3)]
    total = sum(p_values)
    if b == 1 and include_cosmological_prefactor:
        raise ValueError(
            "the bare DOZZ cosmological prefactor is singular at b=1; "
            "use the BRY-renormalized coefficient with "
            "include_cosmological_prefactor=False"
        )
    if b == 1:
        # BRY (2.9) fixes the analytic square-root branch in (2.6).  In
        # particular, continuing any P_i through zero changes the sign of C.
        if any(momentum == 0 for momentum in p_values):
            return 0.0 + 0.0j
        log_constant = -special.log_upsilon(1 + 1j * total)
        for momentum in p_values:
            other_sum = total - 2 * momentum
            log_constant += (
                mp.log(2 * momentum)
                + special.log_upsilon(1 + 2j * momentum)
                - special.log_upsilon(1 + 1j * other_sum)
            )
    else:
        log_constant = (
            special.log_upsilon_prime_zero()
            - special.log_upsilon(q_background / 2 + 1j * total)
        )
        for momentum in p_values:
            other_sum = total - 2 * momentum
            log_constant += (
                0.5 * special.log_upsilon(2j * momentum)
                + 0.5 * special.log_upsilon(-2j * momentum)
                - special.log_upsilon(q_background / 2 + 1j * other_sum)
            )

    if include_cosmological_prefactor:
        log_base = (
            mp.log(mp.pi * mp.mpc(mu))
            + special.log_gamma_ratio(b * b)
            + (2 - 2 * b * b) * mp.log(b)
        )
        log_constant -= q_background / (2 * b) * log_base

    return _to_python_complex(mp.e ** log_constant)


def estimate_p_max(q: complex, tail_tolerance: float = 1e-14, safety_margin: float = 1.0) -> float:
    """Gaussian tail estimate from the factor |q|^(2 P^2)."""
    abs_q = abs(q)
    if not 0 < abs_q < 1:
        raise ValueError("the Liouville P integral expects 0 < |q| < 1")
    gaussian_scale = math.sqrt(max(1.0, -math.log(tail_tolerance)) / (-2.0 * math.log(abs_q)))
    return max(2.0, gaussian_scale + safety_margin)


def validate_nonresonant_b_for_block(b: float, block_order: int, tolerance: float = 1e-13) -> None:
    """Fail early when the generic residue formula has colliding Kac labels."""
    for r in range(1, block_order + 1):
        for s in range(1, block_order // r + 1):
            for p_index in range(1 - r, r + 1):
                for ell_index in range(1 - s, s + 1):
                    if (p_index, ell_index) in {(0, 0), (r, s)}:
                        continue
                    if abs(p_index * b + ell_index / b) < tolerance:
                        raise ValueError(
                            "the generic Zamolodchikov recursion is resonant at this b "
                            f"through order {block_order}; p*b+ell/b vanishes for "
                            f"(p,ell)=({p_index},{ell_index}). Use a nearby b regulator "
                            "and extrapolate, or implement the analytic collision limit."
                        )


def liouville_torus_one_point_integrand(
    p: float,
    *,
    b: float,
    external_momentum: complex,
    q: complex,
    block_order: int,
    special: UpsilonB | None = None,
    mu: complex = 1.0,
    include_cosmological_prefactor: bool = False,
) -> complex:
    """Return the Xi-normalized P-integrand pi^(-1) C(P,P_ext,P) |F_P(q)|^2."""
    if p == 0:
        return 0.0 + 0.0j
    if special is None:
        special = UpsilonB(b)

    q_background = b + 1.0 / b
    c = 1.0 + 6.0 * q_background * q_background
    internal_weight = 0.25 * q_background * q_background + p * p
    external_lambda = lambda_from_yin_momentum(external_momentum)
    external_weight = liouville_weight_from_lambda(b, external_lambda)
    structure_constant = yin_structure_constant_momentum(
        special,
        p,
        external_momentum,
        p,
        mu=mu,
        include_cosmological_prefactor=include_cosmological_prefactor,
    )
    block = TorusOnePointVirasoroBlock(
        c,
        internal_weight,
        external_weight,
        b=b,
        external_lambda=external_lambda,
    )
    chiral_block = block.chiral_block_exact_eta(q, block_order)
    return structure_constant * abs(chiral_block) ** 2 / math.pi


@dataclass(frozen=True)
class LiouvilleTorusQuadratureChannel:
    p: float
    weighted_structure_constant: complex
    block: TorusOnePointVirasoroBlock


class LiouvilleTorusOnePointQuadrature:
    """Reusable Xi-normalized P quadrature with DOZZ data cached per channel."""

    def __init__(
        self,
        *,
        b: float,
        external_momentum: complex,
        block_order: int,
        p_max: float,
        quadrature_order: int = 32,
        dps: int = 40,
        mu: complex = 1.0,
        include_cosmological_prefactor: bool = False,
    ) -> None:
        if quadrature_order <= 0:
            raise ValueError("quadrature_order must be positive")
        if block_order < 0:
            raise ValueError("block_order must be non-negative")
        if p_max <= 0:
            raise ValueError("p_max must be positive")
        validate_nonresonant_b_for_block(b, block_order)

        self.b = b
        self.external_momentum = complex(external_momentum)
        self.block_order = block_order
        self.p_max = p_max
        self.quadrature_order = quadrature_order
        self.dps = dps
        self.mu = complex(mu)
        self.include_cosmological_prefactor = include_cosmological_prefactor

        self.q_background = b + 1.0 / b
        self.central_charge = 1.0 + 6.0 * self.q_background * self.q_background
        self.external_lambda = lambda_from_yin_momentum(self.external_momentum)
        self.external_weight = liouville_weight_from_lambda(b, self.external_lambda)
        self.special = UpsilonB(b=b, dps=dps)
        self.channels = self._build_channels()

    @classmethod
    def for_q_values(
        cls,
        *,
        b: float,
        external_momentum: complex,
        q_values: Iterable[complex],
        block_order: int,
        p_max: float | None = None,
        quadrature_order: int = 32,
        dps: int = 40,
        mu: complex = 1.0,
        include_cosmological_prefactor: bool = False,
        tail_tolerance: float = 1.0e-14,
        safety_margin: float = 1.0,
    ) -> "LiouvilleTorusOnePointQuadrature":
        q_values = [complex(q) for q in q_values]
        if not q_values:
            raise ValueError("q_values must be non-empty")
        if p_max is None:
            p_max = max(
                estimate_p_max(q, tail_tolerance=tail_tolerance, safety_margin=safety_margin)
                for q in q_values
            )
        return cls(
            b=b,
            external_momentum=external_momentum,
            block_order=block_order,
            p_max=p_max,
            quadrature_order=quadrature_order,
            dps=dps,
            mu=mu,
            include_cosmological_prefactor=include_cosmological_prefactor,
        )

    def _build_channels(self) -> list[LiouvilleTorusQuadratureChannel]:
        nodes, weights = np.polynomial.legendre.leggauss(self.quadrature_order)
        midpoint = 0.5 * self.p_max
        channels: list[LiouvilleTorusQuadratureChannel] = []
        for node, weight in zip(nodes, weights):
            p = midpoint * (float(node) + 1.0)
            internal_weight = 0.25 * self.q_background * self.q_background + p * p
            structure_constant = yin_structure_constant_momentum(
                self.special,
                p,
                self.external_momentum,
                p,
                mu=self.mu,
                include_cosmological_prefactor=self.include_cosmological_prefactor,
            )
            block = TorusOnePointVirasoroBlock(
                self.central_charge,
                internal_weight,
                self.external_weight,
                b=self.b,
                external_lambda=self.external_lambda,
            )
            channels.append(
                LiouvilleTorusQuadratureChannel(
                    p=p,
                    weighted_structure_constant=midpoint * float(weight) * structure_constant / math.pi,
                    block=block,
                )
            )
        return channels

    def full_one_point(self, q: complex) -> complex:
        """Evaluate the full torus one-point function with cached DOZZ data."""
        q = complex(q)
        if not 0 < abs(q) < 1:
            raise ValueError("the Liouville P integral expects 0 < |q| < 1")
        return sum(
            channel.weighted_structure_constant
            * abs(channel.block.chiral_block_exact_eta(q, self.block_order)) ** 2
            for channel in self.channels
        )

    def hjs_stripped_integral(self, q: complex) -> complex:
        """Return the eta-stripped integral appearing in HJS equation (25)."""
        q = complex(q)
        if not 0 < abs(q) < 1:
            raise ValueError("the Liouville P integral expects 0 < |q| < 1")
        return sum(
            channel.weighted_structure_constant
            * abs((q ** (channel.p * channel.p)) * channel.block.elliptic_block(q, self.block_order)) ** 2
            for channel in self.channels
        )

    def scan_full_one_point(self, q_values: Iterable[complex]) -> list[tuple[complex, complex]]:
        return [(complex(q), self.full_one_point(q)) for q in q_values]


@dataclass(frozen=True)
class LiouvilleTorusIntegralResult:
    value: complex
    q: complex
    b: float
    external_momentum: complex
    external_lambda: complex
    block_order: int
    p_max: float
    quadrature_order: int
    dps: int
    include_cosmological_prefactor: bool


def liouville_torus_one_point(
    *,
    b: float,
    external_momentum: complex,
    q: complex,
    block_order: int,
    p_max: float | None = None,
    quadrature_order: int = 32,
    dps: int = 40,
    mu: complex = 1.0,
    include_cosmological_prefactor: bool = False,
) -> LiouvilleTorusIntegralResult:
    """Compute the Xi-normalized diagonal Liouville torus one-point function."""
    q = complex(q)
    if not 0 < abs(q) < 1:
        raise ValueError("the Liouville P integral expects 0 < |q| < 1")
    quadrature = LiouvilleTorusOnePointQuadrature.for_q_values(
        b=b,
        external_momentum=external_momentum,
        q_values=[q],
        block_order=block_order,
        p_max=p_max,
        quadrature_order=quadrature_order,
        dps=dps,
        mu=mu,
        include_cosmological_prefactor=include_cosmological_prefactor,
    )

    return LiouvilleTorusIntegralResult(
        value=quadrature.full_one_point(q),
        q=q,
        b=b,
        external_momentum=external_momentum,
        external_lambda=quadrature.external_lambda,
        block_order=block_order,
        p_max=quadrature.p_max,
        quadrature_order=quadrature_order,
        dps=dps,
        include_cosmological_prefactor=include_cosmological_prefactor,
    )


def _resolve_external_momentum(args: argparse.Namespace, b: float) -> complex:
    provided = [
        args.external_lambda is not None,
        args.external_alpha is not None,
        args.external_momentum is not None,
    ]
    if sum(provided) != 1:
        raise ValueError("provide exactly one of --external-lambda, --external-alpha, --external-momentum")
    if args.external_lambda is not None:
        return args.external_lambda / (2.0j)
    if args.external_alpha is not None:
        return lambda_from_alpha(b, args.external_alpha) / (2.0j)
    return args.external_momentum


def _resolve_q(args: argparse.Namespace) -> complex:
    if (args.q is None) == (args.tau is None):
        raise ValueError("provide exactly one of --q or --tau")
    return args.q if args.q is not None else q_from_tau(args.tau)


def run(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate a Liouville torus one-point integral.")
    parser.add_argument("--b", type=float, required=True)
    parser.add_argument("--external-lambda", type=parse_complex)
    parser.add_argument("--external-alpha", type=parse_complex)
    parser.add_argument("--external-momentum", type=parse_complex)
    parser.add_argument("--q", type=parse_complex)
    parser.add_argument("--tau", type=parse_complex)
    parser.add_argument("--block-order", type=int, default=6)
    parser.add_argument("--quadrature-order", type=int, default=32)
    parser.add_argument("--p-max", type=float)
    parser.add_argument("--dps", type=int, default=40)
    parser.add_argument("--mu", type=parse_complex, default=1.0 + 0.0j)
    parser.add_argument(
        "--include-cosmological-prefactor",
        action="store_true",
        help="include the momentum-independent prefactor in BRY eq. (2.5)",
    )
    args = parser.parse_args(argv)

    external_momentum = _resolve_external_momentum(args, args.b)
    q = _resolve_q(args)
    try:
        result = liouville_torus_one_point(
            b=args.b,
            external_momentum=external_momentum,
            q=q,
            block_order=args.block_order,
            p_max=args.p_max,
            quadrature_order=args.quadrature_order,
            dps=args.dps,
            mu=args.mu,
            include_cosmological_prefactor=args.include_cosmological_prefactor,
        )
    except ValueError as exc:
        parser.error(str(exc))

    q_background = args.b + 1.0 / args.b
    print("Liouville torus one-point integral")
    print(f"  b={args.b:.12g}")
    print(f"  Q={q_background:.12g}")
    print(f"  c={1.0 + 6.0 * q_background * q_background:.12g}")
    print(f"  external momentum P={format_complex(external_momentum)}")
    print(f"  external lambda={format_complex(result.external_lambda)}")
    print(f"  external weight={format_complex(liouville_weight_from_lambda(args.b, result.external_lambda))}")
    print(f"  q={format_complex(result.q)}")
    print(f"  block order={result.block_order}")
    print(f"  P cutoff={result.p_max:.12g}")
    print(f"  quadrature order={result.quadrature_order}")
    print(f"  value={format_complex(result.value)}")


if __name__ == "__main__":
    run()
