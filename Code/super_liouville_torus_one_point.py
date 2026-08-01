"""Type-0B torus one-point correlators and modular S diagnostics.

The ordinary NS trace (no fermion-parity insertion) is invariant as a spin
structure under S.  This makes it the clean first modular test of the exact
``c=27/2`` toric block:

    G_NS(tau) = integral_0^infinity dP/pi C(P,P_ext,P)
                |F_P^NS(q)|^2,
    q = exp(2 pi i tau).

For a bottom-component NS primary of chiral weight d,

    G_NS(-1/tau) = |tau|^(2d) G_NS(tau).

Individual fixed-P blocks do not obey this relation; the spectral integral
implements the modular-kernel mixing.

The second even-spin orbit exchanges the NS trace with fermion parity and
the ordinary Ramond trace:

    G_NS_tilde(-1/tau) = |tau|^(2d) G_R(tau),

up to the equivalent interchange of the two frames.  In the HJS
factorization of a bottom-component NS insertion, the ordinary R trace keeps
only the ``+`` chiral branch.  With the BRY delta normalization this gives

    G_R(tau) = 2 integral_0^infinity dP/pi C_even(P,P;P_ext)
               |F_{P,e}^{R,+}(q)|^2.

HJS write this as ``4 C^(+)``.  Their nonchiral coefficient is
``C^(+)=(C_+ + C_-)/2=C_even/2`` in the BRY local/defect-family
normalization, hence the net BRY coefficient is ``2 C_even``.  This factor
is fixed by the cyclic contraction rather than fitted to modularity.
"""

from __future__ import annotations

import cmath
from dataclasses import dataclass
import math
from typing import List, Mapping, Tuple

import numpy

from super_liouville_structure_constants import (
    ns_structure_constant,
    rr_ns_structure_constants,
)
from superconformal_blocks import central_charge, ns_liouville_weight
from superconformal_torus_blocks import (
    NSPlumbingParameter,
    RamondPlumbingParameter,
    SelfDualNSTorusOnePointBlock,
    SelfDualRamondTorusOnePointBlock,
)


@dataclass(frozen=True)
class Type0BNSTorusChannel:
    momentum: float
    weighted_structure_constant: complex
    block: SelfDualNSTorusOnePointBlock


@dataclass(frozen=True)
class Type0BRTorusChannel:
    momentum: float
    weighted_structure_constant: complex
    block: SelfDualRamondTorusOnePointBlock


def type0b_ns_gauss_legendre_rule(
    p_max: float, quadrature_order: int
) -> Tuple[Tuple[float, float], ...]:
    """Return ``(P, dP-weight)`` pairs on ``[0, p_max]``."""

    p_max = float(p_max)
    quadrature_order = int(quadrature_order)
    if p_max <= 0.0 or not math.isfinite(p_max):
        raise ValueError("p_max must be finite and positive")
    if quadrature_order < 2:
        raise ValueError("quadrature_order must be at least two")
    nodes, weights = numpy.polynomial.legendre.leggauss(quadrature_order)
    midpoint = 0.5 * p_max
    return tuple(
        (
            midpoint * (float(node) + 1.0),
            midpoint * float(weight),
        )
        for node, weight in zip(nodes, weights)
    )


def build_type0b_ns_torus_channel(
    *,
    momentum: float,
    spectral_weight: float,
    external_momentum: float,
    structure_precision: int,
    finite_part_samples: int,
) -> Type0BNSTorusChannel:
    """Prepare one independently computable momentum-node contribution."""

    momentum = float(momentum)
    spectral_weight = float(spectral_weight)
    structure_constant = ns_structure_constant(
        momentum,
        external_momentum,
        momentum,
        structure_precision,
    )
    return Type0BNSTorusChannel(
        momentum=momentum,
        weighted_structure_constant=(
            spectral_weight * structure_constant / math.pi
        ),
        block=SelfDualNSTorusOnePointBlock(
            internal_momentum=momentum,
            external_momentum=external_momentum,
            samples=finite_part_samples,
        ),
    )


def type0b_ns_channel_contribution(
    channel: Type0BNSTorusChannel,
    plumbing: NSPlumbingParameter,
    raw_coefficients: Mapping[int, complex],
    max_twice_level: int,
    *,
    c: complex | float | None = None,
) -> complex:
    """Evaluate one prepared spectral node at a chosen recursion cutoff."""

    max_twice_level = int(max_twice_level)
    if max_twice_level < 0:
        raise ValueError("max_twice_level must be nonnegative")
    central = central_charge(1.0) if c is None else complex(c)
    chiral_value = (
        plumbing.q
        ** (channel.block.internal_weight - central / 24.0)
        * sum(
            coefficient * plumbing.level_factor(twice_level)
            for twice_level, coefficient in raw_coefficients.items()
            if twice_level <= max_twice_level
        )
    )
    return channel.weighted_structure_constant * abs(chiral_value) ** 2


def build_type0b_r_torus_channel(
    *,
    momentum: float,
    spectral_weight: float,
    external_momentum: float,
    structure_precision: int,
    finite_part_samples: int,
) -> Type0BRTorusChannel:
    """Prepare one ordinary-R momentum-node contribution.

    HJS eqs. (2.10)--(2.12) imply that a bottom-component NS one-point
    function in the ordinary R spin structure contains only the even
    ``+`` block and is proportional to ``4 C^(+)``.  In BRY normalization,
    ``C^(+)=C_even/2``, so the spectral coefficient is ``2 C_even``.
    """

    momentum = float(momentum)
    spectral_weight = float(spectral_weight)
    c_even, _ = rr_ns_structure_constants(
        momentum,
        momentum,
        external_momentum,
        structure_precision,
    )
    return Type0BRTorusChannel(
        momentum=momentum,
        weighted_structure_constant=(
            2.0 * spectral_weight * c_even / math.pi
        ),
        block=SelfDualRamondTorusOnePointBlock(
            internal_momentum=momentum,
            external_momentum=external_momentum,
            sign=1,
            samples=finite_part_samples,
        ),
    )


def type0b_r_channel_contribution(
    channel: Type0BRTorusChannel,
    plumbing: RamondPlumbingParameter,
    raw_coefficients: Tuple[complex, ...],
    max_level: int,
    *,
    c: complex | float | None = None,
) -> complex:
    """Evaluate one ordinary-R spectral node at a recursion cutoff."""

    max_level = int(max_level)
    if max_level < 0:
        raise ValueError("max_level must be nonnegative")
    central = central_charge(1.0) if c is None else complex(c)
    chiral_value = (
        plumbing.q
        ** (channel.block.internal_weight - central / 24.0)
        * sum(
            coefficient * plumbing.level_factor(level)
            for level, coefficient in enumerate(raw_coefficients)
            if level <= max_level
        )
    )
    return channel.weighted_structure_constant * abs(chiral_value) ** 2


class Type0BNSOnePointQuadrature:
    """Reusable BRY-normalized spectral quadrature for the ordinary NS trace."""

    def __init__(
        self,
        *,
        external_momentum: float,
        max_twice_level: int = 12,
        p_max: float = 4.5,
        quadrature_order: int = 48,
        structure_precision: int = 35,
        finite_part_samples: int = 24,
    ) -> None:
        if max_twice_level < 0:
            raise ValueError("max_twice_level must be nonnegative")
        if p_max <= 0.0 or not math.isfinite(p_max):
            raise ValueError("p_max must be finite and positive")
        if quadrature_order < 2:
            raise ValueError("quadrature_order must be at least two")
        if structure_precision < 15:
            raise ValueError("structure_precision must be at least 15")
        if finite_part_samples < 8:
            raise ValueError("finite_part_samples must be at least eight")

        self.external_momentum = float(external_momentum)
        self.max_twice_level = int(max_twice_level)
        self.p_max = float(p_max)
        self.quadrature_order = int(quadrature_order)
        self.structure_precision = int(structure_precision)
        self.finite_part_samples = int(finite_part_samples)
        self.external_weight = ns_liouville_weight(
            self.external_momentum, 1.0
        )
        self.central_charge = central_charge(1.0)
        self.channels = self._build_channels()

    def _build_channels(self) -> List[Type0BNSTorusChannel]:
        return [
            build_type0b_ns_torus_channel(
                momentum=momentum,
                spectral_weight=spectral_weight,
                external_momentum=self.external_momentum,
                structure_precision=self.structure_precision,
                finite_part_samples=self.finite_part_samples,
            )
            for momentum, spectral_weight in type0b_ns_gauss_legendre_rule(
                self.p_max, self.quadrature_order
            )
        ]

    def evaluate(
        self,
        q: complex,
        *,
        lift_sign: int = 1,
        max_twice_level: int | None = None,
    ) -> complex:
        """Evaluate the NS-spin one-point function through a chosen level.

        ``max_twice_level=12`` includes every NS term through ``q^6``.
        Coefficients are prepared through the production order stored on this
        quadrature, so several lower cutoffs can be compared without
        rebuilding the BRY structure constants.
        """

        plumbing = NSPlumbingParameter(q, lift_sign)
        cutoff = (
            self.max_twice_level
            if max_twice_level is None
            else int(max_twice_level)
        )
        if cutoff < 0 or cutoff > self.max_twice_level:
            raise ValueError(
                "max_twice_level must lie between zero and the configured "
                "production order"
            )
        return sum(
            type0b_ns_channel_contribution(
                channel,
                plumbing,
                channel.block.raw_coefficients(self.max_twice_level),
                cutoff,
                c=self.central_charge,
            )
            for channel in self.channels
        )


class Type0BROnePointQuadrature:
    """Reusable BRY-normalized spectral quadrature for the ordinary R trace."""

    def __init__(
        self,
        *,
        external_momentum: float,
        max_level: int = 6,
        p_max: float = 4.5,
        quadrature_order: int = 48,
        structure_precision: int = 35,
        finite_part_samples: int = 24,
    ) -> None:
        if max_level < 0:
            raise ValueError("max_level must be nonnegative")
        if p_max <= 0.0 or not math.isfinite(p_max):
            raise ValueError("p_max must be finite and positive")
        if quadrature_order < 2:
            raise ValueError("quadrature_order must be at least two")
        if structure_precision < 15:
            raise ValueError("structure_precision must be at least 15")
        if finite_part_samples < 8:
            raise ValueError("finite_part_samples must be at least eight")

        self.external_momentum = float(external_momentum)
        self.max_level = int(max_level)
        self.p_max = float(p_max)
        self.quadrature_order = int(quadrature_order)
        self.structure_precision = int(structure_precision)
        self.finite_part_samples = int(finite_part_samples)
        self.external_weight = ns_liouville_weight(
            self.external_momentum, 1.0
        )
        self.central_charge = central_charge(1.0)
        self.channels = self._build_channels()

    def _build_channels(self) -> List[Type0BRTorusChannel]:
        return [
            build_type0b_r_torus_channel(
                momentum=momentum,
                spectral_weight=spectral_weight,
                external_momentum=self.external_momentum,
                structure_precision=self.structure_precision,
                finite_part_samples=self.finite_part_samples,
            )
            for momentum, spectral_weight in type0b_ns_gauss_legendre_rule(
                self.p_max, self.quadrature_order
            )
        ]

    def evaluate(
        self,
        q: complex,
        *,
        max_level: int | None = None,
    ) -> complex:
        """Evaluate the ordinary R one-point function.

        ``max_level=6`` is the R-sector companion of NS
        ``max_twice_level=12``: both retain powers through ``q^6``.
        """

        plumbing = RamondPlumbingParameter(q, "identity")
        cutoff = self.max_level if max_level is None else int(max_level)
        if cutoff < 0 or cutoff > self.max_level:
            raise ValueError(
                "max_level must lie between zero and the configured "
                "production order"
            )
        return sum(
            type0b_r_channel_contribution(
                channel,
                plumbing,
                channel.block.raw_even_coefficients(self.max_level),
                cutoff,
                c=self.central_charge,
            )
            for channel in self.channels
        )


@dataclass(frozen=True)
class Type0BNSModularSResult:
    tau: complex
    s_tau: complex
    q: complex
    q_tilde: complex
    lift_sign: int
    lift_sign_tilde: int
    value_q: complex
    value_q_tilde: complex
    external_weight: complex
    numeric_ratio: complex
    expected_ratio: float
    relative_error: complex
    max_twice_level: int


@dataclass(frozen=True)
class Type0BNSTildeRModularSResult:
    tau: complex
    s_tau: complex
    q: complex
    q_tilde: complex
    ns_lift_sign: int
    ns_tilde_lift_sign: int
    value_ns_tilde_q: complex
    value_r_q_tilde: complex
    external_weight: complex
    numeric_ratio: complex
    expected_ratio: float
    relative_error: complex
    max_twice_level: int
    max_r_level: int


def ns_lift_sign_from_tau(tau: complex) -> int:
    """Recover the NS square-root lift that belongs to ``tau``.

    The reduced plumbing coordinate ``q = exp(2 pi i tau)`` is unchanged by
    ``tau -> tau + 1``, whereas its NS lift ``exp(pi i tau)`` changes sign.
    ``NSPlumbingParameter`` uses the principal square root of ``q`` together
    with an explicit sign, so choose that sign by matching the two possible
    roots to ``exp(pi i tau)``.
    """

    tau = complex(tau)
    q = cmath.exp(2.0j * math.pi * tau)
    principal_lift = q**0.5
    tau_lift = cmath.exp(1.0j * math.pi * tau)
    return (
        1
        if abs(principal_lift - tau_lift)
        <= abs(principal_lift + tau_lift)
        else -1
    )


def run_type0b_ns_modular_s_check(
    *,
    tau: complex = 0.2 + 0.9j,
    external_momentum: float = 0.33,
    max_twice_level: int = 12,
    p_max: float = 4.5,
    quadrature_order: int = 48,
    structure_precision: int = 35,
    finite_part_samples: int = 24,
) -> Type0BNSModularSResult:
    """Compare the direct and S-transformed plumbing frames."""

    tau = complex(tau)
    if tau.imag <= 0.0:
        raise ValueError("tau must lie in the upper half-plane")
    s_tau = -1.0 / tau
    q = cmath.exp(2.0j * math.pi * tau)
    q_tilde = cmath.exp(2.0j * math.pi * s_tau)
    lift_sign = ns_lift_sign_from_tau(tau)
    lift_sign_tilde = ns_lift_sign_from_tau(s_tau)

    quadrature = Type0BNSOnePointQuadrature(
        external_momentum=external_momentum,
        max_twice_level=max_twice_level,
        p_max=p_max,
        quadrature_order=quadrature_order,
        structure_precision=structure_precision,
        finite_part_samples=finite_part_samples,
    )
    value_q = quadrature.evaluate(q, lift_sign=lift_sign)
    value_q_tilde = quadrature.evaluate(
        q_tilde, lift_sign=lift_sign_tilde
    )
    numeric_ratio = value_q_tilde / value_q
    expected_ratio = abs(tau) ** (2.0 * quadrature.external_weight.real)
    relative_error = numeric_ratio / expected_ratio - 1.0
    return Type0BNSModularSResult(
        tau=tau,
        s_tau=s_tau,
        q=q,
        q_tilde=q_tilde,
        lift_sign=lift_sign,
        lift_sign_tilde=lift_sign_tilde,
        value_q=value_q,
        value_q_tilde=value_q_tilde,
        external_weight=quadrature.external_weight,
        numeric_ratio=numeric_ratio,
        expected_ratio=expected_ratio,
        relative_error=relative_error,
        max_twice_level=max_twice_level,
    )


def run_type0b_ns_modular_s_convergence(
    *,
    levels: Tuple[int, ...] = (6, 8, 10, 12),
    tau: complex = 0.2 + 0.9j,
    external_momentum: float = 0.33,
    p_max: float = 4.5,
    quadrature_order: int = 48,
    structure_precision: int = 35,
    finite_part_samples: int = 24,
) -> Tuple[Type0BNSModularSResult, ...]:
    """Return a shared-quadrature recursion-order convergence ledger."""

    if not levels or any(
        not isinstance(level, int) or level < 0 for level in levels
    ):
        raise ValueError("levels must be a nonempty tuple of nonnegative integers")
    tau = complex(tau)
    if tau.imag <= 0.0:
        raise ValueError("tau must lie in the upper half-plane")
    maximum_level = max(levels)
    s_tau = -1.0 / tau
    q = cmath.exp(2.0j * math.pi * tau)
    q_tilde = cmath.exp(2.0j * math.pi * s_tau)
    lift_sign = ns_lift_sign_from_tau(tau)
    lift_sign_tilde = ns_lift_sign_from_tau(s_tau)
    quadrature = Type0BNSOnePointQuadrature(
        external_momentum=external_momentum,
        max_twice_level=maximum_level,
        p_max=p_max,
        quadrature_order=quadrature_order,
        structure_precision=structure_precision,
        finite_part_samples=finite_part_samples,
    )
    expected_ratio = abs(tau) ** (2.0 * quadrature.external_weight.real)
    results = []
    for level in levels:
        value_q = quadrature.evaluate(
            q,
            lift_sign=lift_sign,
            max_twice_level=level,
        )
        value_q_tilde = quadrature.evaluate(
            q_tilde,
            lift_sign=lift_sign_tilde,
            max_twice_level=level,
        )
        numeric_ratio = value_q_tilde / value_q
        results.append(
            Type0BNSModularSResult(
                tau=tau,
                s_tau=s_tau,
                q=q,
                q_tilde=q_tilde,
                lift_sign=lift_sign,
                lift_sign_tilde=lift_sign_tilde,
                value_q=value_q,
                value_q_tilde=value_q_tilde,
                external_weight=quadrature.external_weight,
                numeric_ratio=numeric_ratio,
                expected_ratio=expected_ratio,
                relative_error=numeric_ratio / expected_ratio - 1.0,
                max_twice_level=level,
            )
        )
    return tuple(results)


def run_type0b_ns_tilde_r_modular_s_check(
    *,
    tau: complex = 0.2 + 0.9j,
    external_momentum: float = 0.33,
    max_twice_level: int = 12,
    p_max: float = 4.5,
    quadrature_order: int = 48,
    structure_precision: int = 35,
    finite_part_samples: int = 24,
) -> Type0BNSTildeRModularSResult:
    r"""Test the modular orbit \(\widetilde{\rm NS}\leftrightarrow{\rm R}\).

    The direct frame is the NS trace with ``(-1)^F`` at ``tau``.  The
    transformed frame is the ordinary R trace at ``-1/tau``.  An even
    ``max_twice_level`` is required so the two sectors share the same
    maximum power of ``q``.
    """

    tau = complex(tau)
    if tau.imag <= 0.0:
        raise ValueError("tau must lie in the upper half-plane")
    max_twice_level = int(max_twice_level)
    if max_twice_level < 0 or max_twice_level % 2:
        raise ValueError("max_twice_level must be a nonnegative even integer")

    s_tau = -1.0 / tau
    q = cmath.exp(2.0j * math.pi * tau)
    q_tilde = cmath.exp(2.0j * math.pi * s_tau)
    ns_lift_sign = ns_lift_sign_from_tau(tau)
    ns_tilde_lift_sign = -ns_lift_sign
    max_r_level = max_twice_level // 2

    ns_quadrature = Type0BNSOnePointQuadrature(
        external_momentum=external_momentum,
        max_twice_level=max_twice_level,
        p_max=p_max,
        quadrature_order=quadrature_order,
        structure_precision=structure_precision,
        finite_part_samples=finite_part_samples,
    )
    r_quadrature = Type0BROnePointQuadrature(
        external_momentum=external_momentum,
        max_level=max_r_level,
        p_max=p_max,
        quadrature_order=quadrature_order,
        structure_precision=structure_precision,
        finite_part_samples=finite_part_samples,
    )
    value_ns_tilde_q = ns_quadrature.evaluate(
        q,
        lift_sign=ns_tilde_lift_sign,
    )
    value_r_q_tilde = r_quadrature.evaluate(q_tilde)
    numeric_ratio = value_r_q_tilde / value_ns_tilde_q
    expected_ratio = abs(tau) ** (
        2.0 * ns_quadrature.external_weight.real
    )
    return Type0BNSTildeRModularSResult(
        tau=tau,
        s_tau=s_tau,
        q=q,
        q_tilde=q_tilde,
        ns_lift_sign=ns_lift_sign,
        ns_tilde_lift_sign=ns_tilde_lift_sign,
        value_ns_tilde_q=value_ns_tilde_q,
        value_r_q_tilde=value_r_q_tilde,
        external_weight=ns_quadrature.external_weight,
        numeric_ratio=numeric_ratio,
        expected_ratio=expected_ratio,
        relative_error=numeric_ratio / expected_ratio - 1.0,
        max_twice_level=max_twice_level,
        max_r_level=max_r_level,
    )


def run_type0b_ns_tilde_r_modular_s_convergence(
    *,
    levels: Tuple[int, ...] = (6, 8, 10, 12),
    tau: complex = 0.2 + 0.9j,
    external_momentum: float = 0.33,
    p_max: float = 4.5,
    quadrature_order: int = 48,
    structure_precision: int = 35,
    finite_part_samples: int = 24,
) -> Tuple[Type0BNSTildeRModularSResult, ...]:
    """Return a shared-quadrature spin-orbit convergence ledger."""

    if not levels or any(
        not isinstance(level, int) or level < 0 or level % 2
        for level in levels
    ):
        raise ValueError(
            "levels must be a nonempty tuple of nonnegative even integers"
        )
    tau = complex(tau)
    if tau.imag <= 0.0:
        raise ValueError("tau must lie in the upper half-plane")

    maximum_twice_level = max(levels)
    maximum_r_level = maximum_twice_level // 2
    s_tau = -1.0 / tau
    q = cmath.exp(2.0j * math.pi * tau)
    q_tilde = cmath.exp(2.0j * math.pi * s_tau)
    ns_lift_sign = ns_lift_sign_from_tau(tau)
    ns_tilde_lift_sign = -ns_lift_sign
    ns_quadrature = Type0BNSOnePointQuadrature(
        external_momentum=external_momentum,
        max_twice_level=maximum_twice_level,
        p_max=p_max,
        quadrature_order=quadrature_order,
        structure_precision=structure_precision,
        finite_part_samples=finite_part_samples,
    )
    r_quadrature = Type0BROnePointQuadrature(
        external_momentum=external_momentum,
        max_level=maximum_r_level,
        p_max=p_max,
        quadrature_order=quadrature_order,
        structure_precision=structure_precision,
        finite_part_samples=finite_part_samples,
    )
    expected_ratio = abs(tau) ** (
        2.0 * ns_quadrature.external_weight.real
    )

    results = []
    for level in levels:
        value_ns_tilde_q = ns_quadrature.evaluate(
            q,
            lift_sign=ns_tilde_lift_sign,
            max_twice_level=level,
        )
        value_r_q_tilde = r_quadrature.evaluate(
            q_tilde,
            max_level=level // 2,
        )
        numeric_ratio = value_r_q_tilde / value_ns_tilde_q
        results.append(
            Type0BNSTildeRModularSResult(
                tau=tau,
                s_tau=s_tau,
                q=q,
                q_tilde=q_tilde,
                ns_lift_sign=ns_lift_sign,
                ns_tilde_lift_sign=ns_tilde_lift_sign,
                value_ns_tilde_q=value_ns_tilde_q,
                value_r_q_tilde=value_r_q_tilde,
                external_weight=ns_quadrature.external_weight,
                numeric_ratio=numeric_ratio,
                expected_ratio=expected_ratio,
                relative_error=numeric_ratio / expected_ratio - 1.0,
                max_twice_level=level,
                max_r_level=level // 2,
            )
        )
    return tuple(results)


__all__ = [
    "Type0BNSModularSResult",
    "Type0BNSOnePointQuadrature",
    "Type0BNSTildeRModularSResult",
    "Type0BNSTorusChannel",
    "Type0BROnePointQuadrature",
    "Type0BRTorusChannel",
    "build_type0b_ns_torus_channel",
    "build_type0b_r_torus_channel",
    "ns_lift_sign_from_tau",
    "run_type0b_ns_modular_s_check",
    "run_type0b_ns_modular_s_convergence",
    "run_type0b_ns_tilde_r_modular_s_check",
    "run_type0b_ns_tilde_r_modular_s_convergence",
    "type0b_ns_channel_contribution",
    "type0b_ns_gauss_legendre_rule",
    "type0b_r_channel_contribution",
]
