#!/usr/bin/env python3
r"""Elliptic ``h``-recursion for ``NS NS | R R`` blocks.

The puncture order in this module is

``(NS_zero, NS_w, R_one, R_infinity) = (0, w, 1, infinity)``.

This is the crossed channel used by the SO(7) two-Ramond integrand.  The
recursion is Suchanek's mixed two-NS/two-Ramond recursion, arXiv:1012.2974,
eqs. (6.5)--(6.7), after interchanging its two trinions.  That interchange
is not phase-free.  In the repository's HJS conventions an odd null carries

``-exp(-i*pi/4)`` in the even equation and
``-exp(+i*pi/4)`` in the odd equation.

The phases are conjugated in the antiholomorphic HJS convention.  These
choices are checked coefficient by coefficient against the independent
inverse-Gram implementation in :mod:`so7e8_ramond_fourpoint`.

The primary block and either of the two one-star blocks (one external
``G_{-1/2}`` descendant) are covered by the closed recursion below.  The
two-star block is certified through ``t**7``.  A picture-changed NS state is
*not* obtained by replacing its weight by ``h+1/2``: its regular
large-internal-weight term and its null-vector endpoint both change.

For two-star requests above ``t**7``, this module also exposes a separate
cached inverse-Gram coefficient-table path.  It does not extrapolate the
unknown regular seed.  One base HJS block with Ramond ground parities
``(0,0)`` is computed and elliptically prefactored through the requested
order; all other ground components and the antiholomorphic convention are
then obtained by the already certified constant component phases.  The
table is reusable across the many repeated ground-component requests made by
one spectral scan and has been checked through ``t**15``.

At ``c=27/2`` one has ``b=1``.  Individual Kac terms of the recursion are
then confluent even though their sum is finite.  :func:`b_one_limit_series`
evaluates the Weyl-symmetric regulator ``b=exp(+-epsilon)`` at high precision
and extrapolates in ``epsilon**2``.  It returns both extrapolation and
``b <-> 1/b`` diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass
import cmath
from functools import lru_cache
import math
from typing import Literal, Sequence

import mpmath as mp
import numpy as np

from sphere_block_uniformization import _cut_lip_argument, elliptic_nome


Component = Literal["even", "odd"]
Chirality = Literal["holomorphic", "antiholomorphic"]

TWO_STAR_RECURSION_CERTIFIED_MAXIMUM_T_ORDER = 7
TWO_STAR_INVERSE_GRAM_CACHE_VALIDATED_T_ORDER = 15


def _validate_component(component: str) -> Component:
    if component not in ("even", "odd"):
        raise ValueError("component must be 'even' or 'odd'")
    return component  # type: ignore[return-value]


def _validate_chirality(chirality: str) -> Chirality:
    if chirality not in ("holomorphic", "antiholomorphic"):
        raise ValueError("chirality must be 'holomorphic' or 'antiholomorphic'")
    return chirality  # type: ignore[return-value]


def _validate_sign(sign: int) -> int:
    if sign not in (-1, 1):
        raise ValueError("rr_structure_sign must be +1 or -1")
    return int(sign)


def _validate_order(maximum_t_order: int) -> int:
    if not isinstance(maximum_t_order, int) or isinstance(maximum_t_order, bool):
        raise TypeError("maximum_t_order must be an integer")
    if maximum_t_order < 0:
        raise ValueError("maximum_t_order must be nonnegative")
    return maximum_t_order


def _ns_word_is_star(word: object) -> bool:
    """Map the existing block API's ``()``/``(G_-1/2,)`` words to a flag."""

    materialized = tuple(word)  # type: ignore[arg-type]
    if not materialized:
        return False
    if len(materialized) != 1:
        raise ValueError("only () and (G_-1/2,) external NS words are supported")
    mode = materialized[0]
    if getattr(mode, "kind", None) != "G" or complex(
        getattr(mode, "index", math.nan)
    ) != -0.5:
        raise ValueError("only () and (G_-1/2,) external NS words are supported")
    return True


def _canonical_b_from_c(c: complex) -> complex:
    """Choose one member of the dual pair ``(b,1/b)`` from ``c``."""

    central_charge = complex(c)
    background_charge = cmath.sqrt((central_charge - 1.5) / 3.0)
    discriminant = cmath.sqrt(background_charge**2 - 4.0)
    roots = (
        (background_charge + discriminant) / 2.0,
        (background_charge - discriminant) / 2.0,
    )
    nonzero = tuple(root for root in roots if root != 0)
    if not nonzero:
        raise ValueError("central charge does not determine a nonzero b")
    # The recursion is exactly b <-> 1/b invariant.  Choosing the larger
    # modulus makes the branch deterministic and gives b>=1 on the real
    # super-Liouville line.
    return max(nonzero, key=lambda root: (abs(root), root.real, root.imag))


def _finite_complex(value: complex, name: str) -> complex:
    result = complex(value)
    if not (math.isfinite(result.real) and math.isfinite(result.imag)):
        raise ValueError(f"{name} must be finite")
    return result


def _ground_component_phase(
    component: Component,
    structure_sign: int,
    ground_parities: tuple[int, int],
    chirality: Chirality,
) -> complex:
    r"""HJS phase relative to the ``(w+,w+)`` external block.

    The parities are ordered ``(R_one,R_infinity)``.  The table follows
    directly from the normalized HJS ground tensor and the RN orientation
    in HJS 0810.1203v2 (4.10).  Turning the NS--R--R form contributes
    (-1)^(parity_infinity * internal_NS_parity).
    """

    parity_one, parity_infinity = ground_parities
    if parity_one not in (0, 1) or parity_infinity not in (0, 1):
        raise ValueError("Ramond ground parities must be zero or one")
    convention_i = 1.0j if chirality == "holomorphic" else -1.0j
    if component == "even":
        if parity_one == 0:
            return 1.0 + 0.0j
        if parity_infinity == 0:
            return structure_sign * convention_i
        return complex(structure_sign)
    if parity_one == 0 and parity_infinity == 1:
        return convention_i
    if parity_one == 1:
        return complex(structure_sign * (-1)**parity_infinity)
    return 1.0 + 0.0j


@dataclass(frozen=True)
class MixedEllipticBlockSeries:
    r"""A finite mixed block ``H(t)``, with ``t=sqrt(q)``."""

    h_coefficients: tuple[complex, ...]
    component: Component
    chirality: Chirality
    rr_structure_sign: int
    ramond_ground_parities: tuple[int, int]
    b: complex
    c: complex
    h_internal: complex
    h_ns_zero: complex
    h_ns_moving: complex
    ns_descendants: tuple[bool, bool]
    beta_one: complex
    beta_infinity: complex
    maximum_t_order: int

    @property
    def base_external_weights(self) -> tuple[complex, complex, complex, complex]:
        return (
            self.h_ns_zero,
            self.h_ns_moving,
            self.c / 24.0 - self.beta_one**2,
            self.c / 24.0 - self.beta_infinity**2,
        )

    @property
    def external_weights(self) -> tuple[complex, complex, complex, complex]:
        """Effective weights, including the two possible external stars."""

        h0, hw, h1, hinfinity = self.base_external_weights
        star_zero, star_moving = self.ns_descendants
        return (
            h0 + (0.5 if star_zero else 0.0),
            hw + (0.5 if star_moving else 0.0),
            h1,
            hinfinity,
        )

    def value_from_t(
        self,
        t: complex | np.ndarray,
    ) -> complex | np.ndarray:
        argument = np.asarray(t, dtype=np.complex128)
        scalar = argument.ndim == 0
        value = np.zeros_like(argument)
        for coefficient in reversed(self.h_coefficients):
            value = value * argument + coefficient
        if scalar:
            return complex(value.item())
        return value

    @staticmethod
    def _theta3(q: np.ndarray) -> np.ndarray:
        if np.any(np.abs(q) >= 1.0):
            raise ArithmeticError("elliptic nome left the open unit disk")
        value = np.ones_like(q)
        for index in range(1, 10000):
            term = 2.0 * np.power(q, index * index)
            value += term
            if np.max(np.abs(term), initial=0.0) < 2.0e-15:
                return value
        raise ArithmeticError("theta3 sum did not converge")

    def value(
        self,
        w: complex | np.ndarray,
        *,
        cut_side: Literal["upper", "lower"] = "upper",
    ) -> complex | np.ndarray:
        r"""Evaluate the full chiral block with Suchanek's mixed prefactor."""

        points = np.asarray(w, dtype=np.complex128)
        scalar = points.ndim == 0
        if np.any((points == 0.0) | (points == 1.0)):
            raise ValueError("the elliptic block is singular at zero or one")
        q = np.asarray(elliptic_nome(points, cut_side=cut_side), dtype=np.complex128)
        theta = self._theta3(q)
        branch_w = _cut_lip_argument(points.reshape(-1), cut_side).reshape(
            points.shape
        )
        h0, hw, h1, hinfinity = self.external_weights
        c0 = (self.c - 1.5) / 24.0
        theta_exponent = (
            (self.c - 1.5) / 2.0
            - 4.0 * (h0 + hw + h1 + hinfinity)
            + 0.5
        )
        logarithm = (
            (self.h_internal - c0) * np.log(16.0 * q)
            + (c0 - h0 - hw) * np.log(branch_w)
            + (c0 - hw - h1 + 1.0 / 16.0) * np.log(1.0 - branch_w)
            + theta_exponent * np.log(theta)
        )
        result = np.exp(logarithm) * self.value_from_t(np.sqrt(q))
        if np.any(~np.isfinite(result)):
            raise ArithmeticError("non-finite mixed elliptic block value")
        if scalar:
            return complex(np.asarray(result).item())
        return np.asarray(result, dtype=np.complex128)


@dataclass(frozen=True)
class BOneLimitResult:
    """Confluent ``b=1`` limit and its numerical continuation diagnostics."""

    series: MixedEllipticBlockSeries
    regulator_epsilons: tuple[float, ...]
    coefficient_error_estimates: tuple[float, ...]
    coefficient_duality_defects: tuple[float, ...]
    working_digits: int

    @property
    def maximum_error_estimate(self) -> float:
        return max(self.coefficient_error_estimates, default=0.0)

    @property
    def maximum_duality_defect(self) -> float:
        return max(self.coefficient_duality_defects, default=0.0)


class MixedHRecursionFamily:
    r"""Reusable fixed-external-data evaluator for varying internal weight.

    A momentum quadrature changes ``h_internal`` but leaves every Kac tail
    and fusion polynomial fixed.  This object retains those tails.  At
    ``b=1`` it also retains the twelve regulated recursion engines, so only
    the inexpensive top-level rational evaluation and extrapolation are
    repeated at each momentum node.
    """

    def __init__(
        self,
        *,
        b: complex,
        h_ns_zero: complex,
        h_ns_moving: complex,
        beta_one: complex,
        beta_infinity: complex,
        maximum_t_order: int,
        component: Component,
        rr_structure_sign: int,
        chirality: Chirality = "holomorphic",
        ramond_ground_parities: Sequence[int] = (0, 0),
        ns_descendants: Sequence[bool] = (False, False),
        initial_epsilon: float = 0.2,
        sample_count: int = 6,
        working_digits: int | None = None,
    ) -> None:
        self.maximum_t_order = _validate_order(maximum_t_order)
        self.component = _validate_component(component)
        self.chirality = _validate_chirality(chirality)
        self.rr_structure_sign = _validate_sign(rr_structure_sign)
        self.ramond_ground_parities = tuple(
            int(value) for value in ramond_ground_parities
        )
        if len(self.ramond_ground_parities) != 2 or any(
            value not in (0, 1) for value in self.ramond_ground_parities
        ):
            raise ValueError(
                "ramond_ground_parities must contain two zero/one entries"
            )
        self.ns_descendants = tuple(bool(value) for value in ns_descendants)
        if len(self.ns_descendants) != 2:
            raise ValueError("ns_descendants must contain (star_zero,star_moving)")
        if (
            all(self.ns_descendants)
            and self.maximum_t_order > TWO_STAR_RECURSION_CERTIFIED_MAXIMUM_T_ORDER
        ):
            raise NotImplementedError(
                "the two-star mixed regular term is certified only through t**7"
            )
        self.b = _finite_complex(b, "b")
        if self.b == 0:
            raise ValueError("b must be nonzero")
        self.h_ns_zero = _finite_complex(h_ns_zero, "h_ns_zero")
        self.h_ns_moving = _finite_complex(h_ns_moving, "h_ns_moving")
        self.beta_one = _finite_complex(beta_one, "beta_one")
        self.beta_infinity = _finite_complex(beta_infinity, "beta_infinity")
        if not math.isfinite(initial_epsilon) or not 0 < initial_epsilon < 0.5:
            raise ValueError("initial_epsilon must lie in (0,0.5)")
        if not isinstance(sample_count, int) or sample_count < 4:
            raise ValueError("sample_count must be an integer at least four")
        default_digits = max(80, 4 * self.maximum_t_order + 40)
        self.working_digits = (
            default_digits if working_digits is None else working_digits
        )
        if not isinstance(self.working_digits, int) or self.working_digits < 50:
            raise ValueError("working_digits must be an integer at least 50")
        self._confluent = (
            abs(self.b - 1.0) < 1.0e-12 or abs(self.b + 1.0) < 1.0e-12
        )
        self._epsilons: tuple[mp.mpf, ...] = ()
        self._engines: tuple[
            tuple[_MixedRecursionEngine, _MixedRecursionEngine], ...
        ] = ()
        self._engine: _MixedRecursionEngine | None = None
        with mp.workdps(self.working_digits):
            if self._confluent:
                self._epsilons = tuple(
                    mp.mpf(str(initial_epsilon)) / mp.power(2, index)
                    for index in range(sample_count)
                )
                self._engines = tuple(
                    (
                        self._make_engine(mp.exp(epsilon)),
                        self._make_engine(mp.exp(-epsilon)),
                    )
                    for epsilon in self._epsilons
                )
            else:
                self._engine = self._make_engine(mp.mpc(self.b))

    def _make_engine(self, b: mp.mpc) -> "_MixedRecursionEngine":
        return _MixedRecursionEngine(
            b=b,
            h_internal=mp.mpc(0),
            h_ns_zero=mp.mpc(self.h_ns_zero),
            h_ns_moving=mp.mpc(self.h_ns_moving),
            beta_one=mp.mpc(self.beta_one),
            beta_infinity=mp.mpc(self.beta_infinity),
            rr_structure_sign=self.rr_structure_sign,
            chirality=self.chirality,
            lambda_ns_zero=None,
            lambda_ns_moving=None,
            ns_descendants=self.ns_descendants,  # type: ignore[arg-type]
        )

    def _engine_coefficients(
        self,
        engine: "_MixedRecursionEngine",
        h_internal: mp.mpc,
    ) -> tuple[mp.mpc, ...]:
        return engine.recurse(
            self.component,
            self.rr_structure_sign,
            h_internal,
            self.maximum_t_order,
        )

    def series(
        self,
        h_internal: complex,
        *,
        ramond_ground_parities: Sequence[int] | None = None,
    ) -> MixedEllipticBlockSeries:
        """Evaluate the cached family at one internal conformal weight."""

        internal_weight = _finite_complex(h_internal, "h_internal")
        selected_parities = (
            self.ramond_ground_parities
            if ramond_ground_parities is None
            else tuple(int(value) for value in ramond_ground_parities)
        )
        if len(selected_parities) != 2 or any(
            value not in (0, 1) for value in selected_parities
        ):
            raise ValueError(
                "ramond_ground_parities must contain two zero/one entries"
            )
        phase = _ground_component_phase(
            self.component,
            self.rr_structure_sign,
            selected_parities,  # type: ignore[arg-type]
            self.chirality,
        )
        with mp.workdps(self.working_digits):
            h_mp = mp.mpc(internal_weight)
            if self._confluent:
                symmetric: list[tuple[mp.mpc, ...]] = []
                for plus_engine, minus_engine in self._engines:
                    plus = self._engine_coefficients(plus_engine, h_mp)
                    minus = self._engine_coefficients(minus_engine, h_mp)
                    symmetric.append(
                        tuple((left + right) / 2 for left, right in zip(plus, minus))
                    )
                xs = [epsilon**2 for epsilon in self._epsilons]
                coefficients = tuple(
                    _interpolate_at_zero(
                        xs,
                        [row[index] for row in symmetric],
                    )
                    for index in range(self.maximum_t_order + 1)
                )
                central_charge = 13.5 + 0.0j
                b_output = 1.0 + 0.0j
            else:
                if self._engine is None:
                    raise AssertionError("generic recursion engine was not initialized")
                coefficients = self._engine_coefficients(self._engine, h_mp)
                charge = self.b + 1.0 / self.b
                central_charge = 1.5 + 3.0 * charge**2
                b_output = self.b
        return MixedEllipticBlockSeries(
            h_coefficients=tuple(phase * complex(value) for value in coefficients),
            component=self.component,
            chirality=self.chirality,
            rr_structure_sign=self.rr_structure_sign,
            ramond_ground_parities=selected_parities,  # type: ignore[arg-type]
            b=b_output,
            c=central_charge,
            h_internal=internal_weight,
            h_ns_zero=self.h_ns_zero,
            h_ns_moving=self.h_ns_moving,
            ns_descendants=self.ns_descendants,  # type: ignore[arg-type]
            beta_one=self.beta_one,
            beta_infinity=self.beta_infinity,
            maximum_t_order=self.maximum_t_order,
        )


class _MixedRecursionEngine:
    def __init__(
        self,
        *,
        b: mp.mpc,
        h_internal: mp.mpc,
        h_ns_zero: mp.mpc,
        h_ns_moving: mp.mpc,
        beta_one: mp.mpc,
        beta_infinity: mp.mpc,
        rr_structure_sign: int,
        chirality: Chirality,
        lambda_ns_zero: mp.mpc | None,
        lambda_ns_moving: mp.mpc | None,
        ns_descendants: tuple[bool, bool],
    ) -> None:
        if b == 0:
            raise ValueError("b must be nonzero")
        self.b = b
        self.q_background = b + 1 / b
        self.h_internal = h_internal
        self.h_ns_zero = h_ns_zero
        self.h_ns_moving = h_ns_moving
        self.beta_one = beta_one
        self.beta_infinity = beta_infinity
        self.structure_sign = rr_structure_sign
        self.chirality = chirality
        self.star_zero, self.star_moving = ns_descendants
        self.lambda_zero = (
            mp.sqrt(self.q_background**2 - 8 * h_ns_zero)
            if lambda_ns_zero is None
            else lambda_ns_zero
        )
        self.lambda_moving = (
            mp.sqrt(self.q_background**2 - 8 * h_ns_moving)
            if lambda_ns_moving is None
            else lambda_ns_moving
        )
        self._a_cache: dict[tuple[int, int], mp.mpc] = {}
        self._h_cache: dict[tuple[int, int], mp.mpc] = {}
        self._ns_cache: dict[tuple[int, int, bool, bool], mp.mpc] = {}
        self._rr_cache: dict[tuple[int, int, int], mp.mpc] = {}
        self._recursion_cache: dict[
            tuple[str, int, str, int], tuple[mp.mpc, ...]
        ] = {}

    def h_rs(self, r: int, s: int) -> mp.mpc:
        key = (r, s)
        if key not in self._h_cache:
            momentum = r * self.b + s / self.b
            self._h_cache[key] = (self.q_background**2 - momentum**2) / 8
        return self._h_cache[key]

    def a_rs(self, r: int, s: int) -> mp.mpc:
        key = (r, s)
        if key in self._a_cache:
            return self._a_cache[key]
        value = mp.mpc(mp.mpf("0.5"))
        root_two = mp.sqrt(2)
        for p in range(1 - r, r + 1):
            for q in range(1 - s, s + 1):
                if (p + q) % 2:
                    continue
                if (p, q) in ((0, 0), (r, s)):
                    continue
                denominator = p * self.b + q / self.b
                if denominator == 0:
                    raise ZeroDivisionError(
                        f"the ({r},{s}) inverse null norm is confluent at b={self.b}"
                    )
                value *= root_two / denominator
        self._a_cache[key] = value
        return value

    def ns_fusion(
        self,
        r: int,
        s: int,
        *,
        starred: bool,
        reflected: bool = False,
    ) -> mp.mpc:
        key = (r, s, starred, reflected)
        if key in self._ns_cache:
            return self._ns_cache[key]
        value = mp.mpc(1)
        lambda_zero = self.lambda_moving if reflected else self.lambda_zero
        lambda_moving = self.lambda_zero if reflected else self.lambda_moving
        selected_parity = 1 if starred else 0
        denominator = 2 * mp.sqrt(2)
        for k in range(r):
            for ell in range(s):
                if (k + ell) % 2 != selected_parity:
                    continue
                p = 1 - r + 2 * k
                q = 1 - s + 2 * ell
                shift = p * self.b + q / self.b
                value *= (
                    (lambda_zero + lambda_moving - shift) / denominator
                )
                value *= (
                    (lambda_zero - lambda_moving - shift) / denominator
                )
        self._ns_cache[key] = value
        return value

    def rr_fusion(self, r: int, s: int, structure_sign: int) -> mp.mpc:
        key = (r, s, structure_sign)
        if key in self._rr_cache:
            return self._rr_cache[key]
        value = mp.mpc(1)
        denominator = 2 * mp.sqrt(2)
        for k in range(r):
            for ell in range(s):
                p = 1 - r + 2 * k
                q = 1 - s + 2 * ell
                shift = (p * self.b + q / self.b) / denominator
                if (k + ell) % 2 == 0:
                    value *= (
                        self.beta_infinity
                        - structure_sign * self.beta_one
                        - shift
                    )
                else:
                    value *= (
                        self.beta_infinity
                        + structure_sign * self.beta_one
                        - shift
                    )
        self._rr_cache[key] = value
        return value

    def _odd_null_phase(self, component: Component) -> mp.mpc:
        chirality_sign = 1 if self.chirality == "holomorphic" else -1
        exponent_sign = -chirality_sign if component == "even" else chirality_sign
        return -mp.exp(mp.j * exponent_sign * mp.pi / 4)

    def _regular_seed(
        self,
        component: Component,
        structure_sign: int,
        current_h: mp.mpc,
        maximum_t_order: int,
    ) -> list[mp.mpc]:
        """Regular-in-``h`` term in the crossed repository orientation.

        The two-star seed is certified through ``t**7``.  In contrast to the
        primary and one-star cases, its even component is a degree-one
        polynomial in the current internal weight.  The lower coefficients
        were obtained by exact pole subtraction and independently checked
        against the inverse-Gram series for generic ``b`` and at ``b=1``.
        """

        total = [mp.mpc(0) for _ in range(maximum_t_order + 1)]
        if component == "even":
            if self.star_zero and self.star_moving:
                c = mp.mpf("1.5") + 3 * self.q_background**2
                h0 = self.h_ns_zero
                hw = self.h_ns_moving
                beta_one_sq = self.beta_one**2
                beta_infinity_sq = self.beta_infinity**2
                beta_product = structure_sign * self.beta_one * self.beta_infinity
                total[0] = current_h - h0 - hw
                correction_two = (
                    8 * (hw - h0) * (beta_infinity_sq - beta_one_sq)
                )
                correction_four = (
                    -mp.mpf(35) / 64
                    - mp.mpf(17) / 2 * (h0 + hw)
                    - mp.mpf(23) / 2 * (beta_one_sq + beta_infinity_sq)
                    + mp.mpf(19) / 48 * c
                    + 12 * beta_product
                    + 4 * (h0 - hw) ** 2
                    - 16 * (h0 + hw) * (beta_one_sq + beta_infinity_sq)
                    + c * (h0 + hw) / 3
                    + 4 * (beta_one_sq - beta_infinity_sq) ** 2
                    + c * (beta_one_sq + beta_infinity_sq)
                    - c**2 / 48
                )
                if maximum_t_order >= 2:
                    total[2] = correction_two
                if maximum_t_order >= 4:
                    total[4] = correction_four
                if maximum_t_order >= 6:
                    total[6] = 4 * correction_two
                return total
            total[0] = mp.mpc(1)
            return total
        if not (self.star_zero or self.star_moving):
            return total
        chirality_sign = 1 if self.chirality == "holomorphic" else -1
        amplitude = (
            -2
            * mp.exp(mp.j * chirality_sign * mp.pi / 4)
            * (self.beta_infinity - structure_sign * self.beta_one)
        )
        # theta_2(q)^2/4, q=t^2:
        # t * (sum_{n>=0} t^(2*n*(n+1)))^2.
        # The old t*theta_3(q^2) agrees through t^7 but misses the t^9
        # coefficient, and its error propagates into even residues at t^10.
        # The squared theta seed is independently checked against Gram
        # coefficients and the on-shell supercurrent Ward identity.
        n = 0
        while 1 + 2*n*(n+1) <= maximum_t_order:
            m = 0
            while 1 + 2*n*(n+1) + 2*m*(m+1) <= maximum_t_order:
                total[1 + 2*n*(n+1) + 2*m*(m+1)] += amplitude
                m += 1
            n += 1
        return total

    def _zero_star_reflection_phase(
        self,
        component: Component,
        null_level: int,
        *,
        even_null: bool,
    ) -> mp.mpc:
        r"""Residue phase induced by reflecting the one-star NS trinion.

        Coefficientwise the local-frame map is
        ``H_e(t)->H_e(i*t)``, ``H_o(t)->-i*H_o(i*t)``.  Taking the parent to
        tail phase ratio gives the multipliers below.
        """

        if not (self.star_zero and not self.star_moving):
            return mp.mpc(1)
        if even_null:
            exponent = null_level
        elif component == "even":
            exponent = null_level + 1
        else:
            exponent = null_level - 1
        return mp.power(mp.j, exponent)

    def _two_star_null_phase(self, *, even_null: bool) -> mp.mpc:
        """Extra endpoint-reflection sign for the two-star trinion."""

        if self.star_zero and self.star_moving and not even_null:
            return mp.mpc(-1)
        return mp.mpc(1)

    def recurse(
        self,
        component: Component,
        structure_sign: int,
        current_h: mp.mpc,
        maximum_t_order: int,
    ) -> tuple[mp.mpc, ...]:
        key = (
            component,
            structure_sign,
            mp.nstr(current_h, mp.mp.dps + 8),
            maximum_t_order,
        )
        if key in self._recursion_cache:
            return self._recursion_cache[key]
        total = self._regular_seed(
            component,
            structure_sign,
            current_h,
            maximum_t_order,
        )

        for r in range(1, maximum_t_order + 1):
            for s in range(1, maximum_t_order // r + 1):
                if (r - s) % 2:
                    continue
                null_level = r * s
                pole = self.h_rs(r, s)
                denominator = current_h - pole
                if denominator == 0:
                    raise ZeroDivisionError(
                        f"the recursion internal weight hit the ({r},{s}) Kac pole"
                    )
                even_null = (r % 2 == 0)
                if even_null:
                    shifted_component = component
                    shifted_sign = structure_sign
                    phase = mp.mpc(1)
                else:
                    shifted_component = "odd" if component == "even" else "even"
                    shifted_sign = -structure_sign
                    phase = self._odd_null_phase(component)
                residue = (
                    mp.power(4, null_level)
                    * phase
                    * self._zero_star_reflection_phase(
                        component,
                        null_level,
                        even_null=even_null,
                    )
                    * self._two_star_null_phase(even_null=even_null)
                    * self.a_rs(r, s)
                    * self.ns_fusion(
                        r,
                        s,
                        starred=(
                            (component == "odd")
                            ^ self.star_moving
                            ^ self.star_zero
                        ),
                        reflected=(self.star_zero and not self.star_moving),
                    )
                    * self.rr_fusion(r, s, structure_sign)
                    / denominator
                )
                shifted_h = pole + mp.mpf(null_level) / 2
                tail = self.recurse(
                    shifted_component,
                    shifted_sign,
                    shifted_h,
                    maximum_t_order - null_level,
                )
                for index, coefficient in enumerate(tail):
                    total[index + null_level] += residue * coefficient
        result = tuple(total)
        self._recursion_cache[key] = result
        return result


def _mp_coefficients(
    *,
    b: complex | mp.mpc,
    h_internal: complex,
    h_ns_zero: complex,
    h_ns_moving: complex,
    beta_one: complex,
    beta_infinity: complex,
    maximum_t_order: int,
    component: Component,
    rr_structure_sign: int,
    chirality: Chirality,
    ramond_ground_parities: tuple[int, int],
    lambda_ns_zero: complex | None,
    lambda_ns_moving: complex | None,
    ns_descendants: tuple[bool, bool],
) -> tuple[mp.mpc, ...]:
    engine = _MixedRecursionEngine(
        b=mp.mpc(b),
        h_internal=mp.mpc(h_internal),
        h_ns_zero=mp.mpc(h_ns_zero),
        h_ns_moving=mp.mpc(h_ns_moving),
        beta_one=mp.mpc(beta_one),
        beta_infinity=mp.mpc(beta_infinity),
        rr_structure_sign=rr_structure_sign,
        chirality=chirality,
        lambda_ns_zero=(
            None if lambda_ns_zero is None else mp.mpc(lambda_ns_zero)
        ),
        lambda_ns_moving=(
            None if lambda_ns_moving is None else mp.mpc(lambda_ns_moving)
        ),
        ns_descendants=ns_descendants,
    )
    coefficients = engine.recurse(
        component,
        rr_structure_sign,
        mp.mpc(h_internal),
        maximum_t_order,
    )
    phase = mp.mpc(
        _ground_component_phase(
            component,
            rr_structure_sign,
            ramond_ground_parities,
            chirality,
        )
    )
    return tuple(phase * coefficient for coefficient in coefficients)


def mixed_h_series(
    *,
    b: complex,
    h_internal: complex,
    h_ns_zero: complex,
    h_ns_moving: complex,
    beta_one: complex,
    beta_infinity: complex,
    maximum_t_order: int,
    component: Component,
    rr_structure_sign: int,
    chirality: Chirality = "holomorphic",
    ramond_ground_parities: Sequence[int] = (0, 0),
    lambda_ns_zero: complex | None = None,
    lambda_ns_moving: complex | None = None,
    ns_descendants: Sequence[bool] = (False, False),
    working_digits: int = 60,
) -> MixedEllipticBlockSeries:
    r"""Return a generic-``b`` mixed elliptic recursion series.

    ``ns_descendants`` is ordered ``(NS_zero,NS_moving)``.  The primary and
    the two one-star choices are supported at arbitrary finite order.  The
    two-star regular seed is presently certified only through ``t**7``.
    """

    order = _validate_order(maximum_t_order)
    selected_component = _validate_component(component)
    selected_chirality = _validate_chirality(chirality)
    sign = _validate_sign(rr_structure_sign)
    parities = tuple(int(value) for value in ramond_ground_parities)
    if len(parities) != 2 or any(value not in (0, 1) for value in parities):
        raise ValueError("ramond_ground_parities must contain two zero/one entries")
    descendants = tuple(bool(value) for value in ns_descendants)
    if len(descendants) != 2:
        raise ValueError("ns_descendants must contain (star_zero,star_moving)")
    if all(descendants) and order > TWO_STAR_RECURSION_CERTIFIED_MAXIMUM_T_ORDER:
        raise NotImplementedError(
            "the two-star mixed regular term is certified only through t**7"
        )
    if not isinstance(working_digits, int) or working_digits < 30:
        raise ValueError("working_digits must be an integer at least 30")
    b_value = _finite_complex(b, "b")
    if b_value == 0:
        raise ValueError("b must be nonzero")
    inputs = {
        "h_internal": _finite_complex(h_internal, "h_internal"),
        "h_ns_zero": _finite_complex(h_ns_zero, "h_ns_zero"),
        "h_ns_moving": _finite_complex(h_ns_moving, "h_ns_moving"),
        "beta_one": _finite_complex(beta_one, "beta_one"),
        "beta_infinity": _finite_complex(beta_infinity, "beta_infinity"),
    }
    with mp.workdps(working_digits):
        coefficients_mp = _mp_coefficients(
            b=b_value,
            maximum_t_order=order,
            component=selected_component,
            rr_structure_sign=sign,
            chirality=selected_chirality,
            ramond_ground_parities=parities,  # type: ignore[arg-type]
            lambda_ns_zero=lambda_ns_zero,
            lambda_ns_moving=lambda_ns_moving,
            ns_descendants=descendants,  # type: ignore[arg-type]
            **inputs,
        )
    charge = b_value + 1.0 / b_value
    central_charge = 1.5 + 3.0 * charge**2
    return MixedEllipticBlockSeries(
        h_coefficients=tuple(complex(value) for value in coefficients_mp),
        component=selected_component,
        chirality=selected_chirality,
        rr_structure_sign=sign,
        ramond_ground_parities=parities,  # type: ignore[arg-type]
        b=b_value,
        c=central_charge,
        maximum_t_order=order,
        ns_descendants=descendants,  # type: ignore[arg-type]
        **inputs,
    )


def mixed_primary_h_series(**kwargs: object) -> MixedEllipticBlockSeries:
    """Backward-compatible primary-only entry point."""

    if "ns_descendants" in kwargs:
        raise TypeError("mixed_primary_h_series does not accept ns_descendants")
    return mixed_h_series(ns_descendants=(False, False), **kwargs)  # type: ignore[arg-type]


def _interpolate_at_zero(xs: Sequence[mp.mpf], ys: Sequence[mp.mpc]) -> mp.mpc:
    if len(xs) != len(ys) or not xs:
        raise ValueError("interpolation needs equally sized nonempty data")
    result = mp.mpc(0)
    for index, x_value in enumerate(xs):
        weight = mp.mpf(1)
        for other_index, other_x in enumerate(xs):
            if other_index != index:
                weight *= -other_x / (x_value - other_x)
        result += weight * ys[index]
    return result


def b_one_limit_series(
    *,
    h_internal: complex,
    h_ns_zero: complex,
    h_ns_moving: complex,
    beta_one: complex,
    beta_infinity: complex,
    maximum_t_order: int,
    component: Component,
    rr_structure_sign: int,
    chirality: Chirality = "holomorphic",
    ramond_ground_parities: Sequence[int] = (0, 0),
    initial_epsilon: float = 0.2,
    sample_count: int = 6,
    working_digits: int | None = None,
    ns_descendants: Sequence[bool] = (False, False),
) -> BOneLimitResult:
    r"""Return the confluent ``b=1`` limit by symmetric continuation.

    The external weights and betas are held fixed along the regulator.  The
    endpoint is therefore the requested block as an analytic function of
    ``(c,h_i,beta_i)``.  Coefficients are averaged at ``b=exp(+-epsilon)``
    and polynomially extrapolated in ``epsilon**2``.
    """

    order = _validate_order(maximum_t_order)
    selected_component = _validate_component(component)
    selected_chirality = _validate_chirality(chirality)
    sign = _validate_sign(rr_structure_sign)
    parities = tuple(int(value) for value in ramond_ground_parities)
    if len(parities) != 2 or any(value not in (0, 1) for value in parities):
        raise ValueError("ramond_ground_parities must contain two zero/one entries")
    descendants = tuple(bool(value) for value in ns_descendants)
    if len(descendants) != 2:
        raise ValueError("ns_descendants must contain (star_zero,star_moving)")
    if all(descendants) and order > TWO_STAR_RECURSION_CERTIFIED_MAXIMUM_T_ORDER:
        raise NotImplementedError(
            "the two-star mixed regular term is certified only through t**7"
        )
    if not math.isfinite(initial_epsilon) or not 0 < initial_epsilon < 0.5:
        raise ValueError("initial_epsilon must lie in (0, 0.5)")
    if not isinstance(sample_count, int) or sample_count < 4:
        raise ValueError("sample_count must be an integer at least four")
    digits = max(80, 4 * order + 40) if working_digits is None else working_digits
    if not isinstance(digits, int) or digits < 50:
        raise ValueError("working_digits must be an integer at least 50")
    inputs = {
        "h_internal": _finite_complex(h_internal, "h_internal"),
        "h_ns_zero": _finite_complex(h_ns_zero, "h_ns_zero"),
        "h_ns_moving": _finite_complex(h_ns_moving, "h_ns_moving"),
        "beta_one": _finite_complex(beta_one, "beta_one"),
        "beta_infinity": _finite_complex(beta_infinity, "beta_infinity"),
    }

    with mp.workdps(digits):
        epsilons_mp = [
            mp.mpf(str(initial_epsilon)) / mp.power(2, index)
            for index in range(sample_count)
        ]
        symmetric_samples: list[tuple[mp.mpc, ...]] = []
        defects: list[tuple[mp.mpf, ...]] = []
        for epsilon in epsilons_mp:
            plus = _mp_coefficients(
                b=mp.exp(epsilon),
                maximum_t_order=order,
                component=selected_component,
                rr_structure_sign=sign,
                chirality=selected_chirality,
                ramond_ground_parities=parities,  # type: ignore[arg-type]
                lambda_ns_zero=None,
                lambda_ns_moving=None,
                ns_descendants=descendants,  # type: ignore[arg-type]
                **inputs,
            )
            minus = _mp_coefficients(
                b=mp.exp(-epsilon),
                maximum_t_order=order,
                component=selected_component,
                rr_structure_sign=sign,
                chirality=selected_chirality,
                ramond_ground_parities=parities,  # type: ignore[arg-type]
                lambda_ns_zero=None,
                lambda_ns_moving=None,
                ns_descendants=descendants,  # type: ignore[arg-type]
                **inputs,
            )
            symmetric_samples.append(
                tuple((left + right) / 2 for left, right in zip(plus, minus))
            )
            defects.append(tuple(abs(left - right) for left, right in zip(plus, minus)))

        xs = [epsilon**2 for epsilon in epsilons_mp]
        coefficients: list[mp.mpc] = []
        errors: list[mp.mpf] = []
        duality: list[mp.mpf] = []
        for coefficient_index in range(order + 1):
            values = [sample[coefficient_index] for sample in symmetric_samples]
            full = _interpolate_at_zero(xs, values)
            reduced = _interpolate_at_zero(xs[1:], values[1:])
            coefficients.append(full)
            errors.append(abs(full - reduced))
            duality.append(max(row[coefficient_index] for row in defects))

    series = MixedEllipticBlockSeries(
        h_coefficients=tuple(complex(value) for value in coefficients),
        component=selected_component,
        chirality=selected_chirality,
        rr_structure_sign=sign,
        ramond_ground_parities=parities,  # type: ignore[arg-type]
        b=1.0 + 0.0j,
        c=13.5 + 0.0j,
        maximum_t_order=order,
        ns_descendants=descendants,  # type: ignore[arg-type]
        **inputs,
    )
    return BOneLimitResult(
        series=series,
        regulator_epsilons=tuple(float(value) for value in epsilons_mp),
        coefficient_error_estimates=tuple(float(value) for value in errors),
        coefficient_duality_defects=tuple(float(value) for value in duality),
        working_digits=digits,
    )


@lru_cache(maxsize=4096)
def _cached_two_star_inverse_gram_base_coefficients(
    c: complex,
    h_internal: complex,
    h_ns_zero: complex,
    h_ns_moving: complex,
    beta_one: complex,
    beta_infinity: complex,
    maximum_t_order: int,
    component: Component,
    rr_structure_sign: int,
    digits: int,
    condition_limit: float,
) -> tuple[complex, ...]:
    r"""Generate one base two-star H-table by direct inverse Gram.

    The expensive table is always built in the holomorphic HJS convention
    with Ramond ground parities ``(0,0)``.  Direct coefficient checks show
    that convention conjugation leaves the even table unchanged and
    multiplies the odd table by ``-i``.  All other external Ramond ground
    components are constant phases supplied by
    :func:`_ground_component_phase`; neither operation changes a physical
    momentum or an external weight.
    """

    from ns_algebra.ns_sca import G
    import sympy as sp

    from ramond_sphere_uniformization import (
        elliptically_prefactor_two_ramond_internal_ns_series,
    )
    from so7e8_ramond_fourpoint import hjs_two_ramond_internal_ns_block_series

    star = (G(sp.Rational(-1, 2)),)
    direct = hjs_two_ramond_internal_ns_block_series(
        c=c,
        h_internal=h_internal,
        beta_one=beta_one,
        beta_infinity=beta_infinity,
        h_ns_zero=h_ns_zero,
        h_ns_w=h_ns_moving,
        maximum_twice_level=maximum_t_order,
        component=component,
        rr_structure_sign=rr_structure_sign,
        chirality="holomorphic",
        ns_words=(star, star),
        ramond_ground_parities=(0, 0),
        digits=digits,
        condition_limit=condition_limit,
    )
    prefactored = elliptically_prefactor_two_ramond_internal_ns_series(direct)
    return tuple(map(complex, prefactored.h_coefficients[: maximum_t_order + 1]))


def cached_two_star_inverse_gram_h_series(
    *,
    c: complex,
    h_internal: complex,
    beta_one: complex,
    beta_infinity: complex,
    h_ns_zero: complex,
    h_ns_moving: complex,
    maximum_t_order: int,
    component: Component,
    rr_structure_sign: int,
    chirality: Chirality = "holomorphic",
    ramond_ground_parities: Sequence[int] = (0, 0),
    digits: int = 50,
    condition_limit: float = 1.0e13,
) -> MixedEllipticBlockSeries:
    r"""Return a cached inverse-Gram two-star elliptic coefficient table.

    This is the certified fallback above the closed recursion's ``t**7``
    boundary.  It uses the direct inverse-Gram/Ward definition rather than a
    guessed descendant regular seed.  The in-memory LRU key contains all
    conformal data, truncation data, and numerical guardrails.  Repeated
    ground components and antiholomorphic requests therefore reuse the same
    expensive base table without changing the analytically continued
    momenta.
    """

    order = _validate_order(maximum_t_order)
    selected_component = _validate_component(component)
    selected_chirality = _validate_chirality(chirality)
    sign = _validate_sign(rr_structure_sign)
    parities = tuple(int(value) for value in ramond_ground_parities)
    if len(parities) != 2 or any(value not in (0, 1) for value in parities):
        raise ValueError("ramond_ground_parities must contain two zero/one entries")
    if not isinstance(digits, int) or digits < 30:
        raise ValueError("digits must be an integer at least 30")
    if not math.isfinite(condition_limit) or condition_limit <= 0:
        raise ValueError("condition_limit must be finite and positive")
    inputs = {
        "c": _finite_complex(c, "c"),
        "h_internal": _finite_complex(h_internal, "h_internal"),
        "h_ns_zero": _finite_complex(h_ns_zero, "h_ns_zero"),
        "h_ns_moving": _finite_complex(h_ns_moving, "h_ns_moving"),
        "beta_one": _finite_complex(beta_one, "beta_one"),
        "beta_infinity": _finite_complex(beta_infinity, "beta_infinity"),
    }
    base = _cached_two_star_inverse_gram_base_coefficients(
        inputs["c"],
        inputs["h_internal"],
        inputs["h_ns_zero"],
        inputs["h_ns_moving"],
        inputs["beta_one"],
        inputs["beta_infinity"],
        order,
        selected_component,
        sign,
        int(digits),
        float(condition_limit),
    )
    convention_phase = (
        -1.0j
        if selected_chirality == "antiholomorphic" and selected_component == "odd"
        else 1.0 + 0.0j
    )
    phase = convention_phase * _ground_component_phase(
        selected_component,
        sign,
        parities,  # type: ignore[arg-type]
        selected_chirality,
    )
    b_value = _canonical_b_from_c(inputs["c"])
    return MixedEllipticBlockSeries(
        h_coefficients=tuple(phase * value for value in base),
        component=selected_component,
        chirality=selected_chirality,
        rr_structure_sign=sign,
        ramond_ground_parities=parities,  # type: ignore[arg-type]
        b=b_value,
        c=inputs["c"],
        h_internal=inputs["h_internal"],
        h_ns_zero=inputs["h_ns_zero"],
        h_ns_moving=inputs["h_ns_moving"],
        ns_descendants=(True, True),
        beta_one=inputs["beta_one"],
        beta_infinity=inputs["beta_infinity"],
        maximum_t_order=order,
    )


def clear_two_star_inverse_gram_cache() -> None:
    """Clear the reusable two-star inverse-Gram coefficient tables."""

    _cached_two_star_inverse_gram_base_coefficients.cache_clear()


def two_star_inverse_gram_cache_info():
    """Return the standard ``functools`` cache statistics object."""

    return _cached_two_star_inverse_gram_base_coefficients.cache_info()


def hjs_two_ramond_internal_ns_elliptic_h_series(
    *,
    c: complex,
    h_internal: complex,
    beta_one: complex,
    beta_infinity: complex,
    h_ns_zero: complex,
    h_ns_w: complex,
    maximum_twice_level: int,
    component: Component,
    rr_structure_sign: int,
    chirality: Chirality = "holomorphic",
    ns_words: Sequence[object] = ((), ()),
    ramond_ground_parities: Sequence[int] = (0, 0),
    b: complex | None = None,
    digits: int = 80,
) -> MixedEllipticBlockSeries:
    r"""Adapter for the existing crossed HJS block request API.

    The argument names and puncture ordering match
    :func:`so7e8_ramond_fourpoint.hjs_two_ramond_internal_ns_block_series`.
    The return value is already elliptically prefactored: its
    ``h_coefficients`` are coefficients of ``H(t)``, and ``value(w)``
    evaluates the complete chiral block.

    Supplying ``b`` removes the harmless ``b <-> 1/b`` branch choice.  If it
    is omitted, a canonical root is reconstructed from ``c``.  At the
    confluent point ``c=27/2`` the adapter automatically uses the symmetric
    high-precision continuation.
    """

    words = tuple(ns_words)
    if len(words) != 2:
        raise ValueError("ns_words must contain (word_NS_zero,word_NS_w)")
    descendants = tuple(_ns_word_is_star(word) for word in words)
    central_charge = _finite_complex(c, "c")
    b_value = _canonical_b_from_c(central_charge) if b is None else _finite_complex(b, "b")
    if b_value == 0:
        raise ValueError("b must be nonzero")
    reconstructed_c = 1.5 + 3.0 * (b_value + 1.0 / b_value) ** 2
    scale = max(1.0, abs(central_charge), abs(reconstructed_c))
    if abs(reconstructed_c - central_charge) > 2.0e-10 * scale:
        raise ValueError("the supplied b and c are inconsistent")
    common = dict(
        h_internal=h_internal,
        h_ns_zero=h_ns_zero,
        h_ns_moving=h_ns_w,
        beta_one=beta_one,
        beta_infinity=beta_infinity,
        maximum_t_order=maximum_twice_level,
        component=component,
        rr_structure_sign=rr_structure_sign,
        chirality=chirality,
        ramond_ground_parities=ramond_ground_parities,
        ns_descendants=descendants,
    )
    if abs(b_value - 1.0) < 1.0e-12 or abs(b_value + 1.0) < 1.0e-12:
        return b_one_limit_series(
            working_digits=max(80, digits),
            **common,
        ).series
    return mixed_h_series(
        b=b_value,
        working_digits=max(30, digits),
        **common,
    )


@lru_cache(maxsize=256)
def _cached_b_one_sld_family(
    h_ns_zero: complex,
    h_ns_moving: complex,
    beta_one: complex,
    beta_infinity: complex,
    maximum_t_order: int,
    component: Component,
    rr_structure_sign: int,
    chirality: Chirality,
    ns_descendants: tuple[bool, bool],
    working_digits: int,
) -> MixedHRecursionFamily:
    """Cache all Kac tails shared by one SLD momentum integral."""

    return MixedHRecursionFamily(
        b=1.0,
        h_ns_zero=h_ns_zero,
        h_ns_moving=h_ns_moving,
        beta_one=beta_one,
        beta_infinity=beta_infinity,
        maximum_t_order=maximum_t_order,
        component=component,
        rr_structure_sign=rr_structure_sign,
        chirality=chirality,
        ramond_ground_parities=(0, 0),
        ns_descendants=ns_descendants,
        working_digits=working_digits,
    )


def two_ramond_internal_ns_sld_elliptic_h_series(
    internal_momentum: complex,
    *,
    external_momenta: Sequence[complex],
    maximum_twice_level: int,
    component: Component,
    rr_structure_sign: int,
    chirality: Chirality = "holomorphic",
    ns_words: Sequence[object] = ((), ()),
    ramond_ground_parities: Sequence[int] = (0, 0),
    digits: int = 80,
    condition_limit: float = 1.0e13,
) -> MixedEllipticBlockSeries:
    r"""Crossed SLD adapter in the original ``R--NS--NS--R`` ordering.

    Inputs follow ``(R1,NS2,NS3,R4)=(0,z,1,infinity)`` exactly as in
    :func:`so7e8_ramond_fourpoint.two_ramond_internal_ns_sld_chiral_block`.
    The crossed standardized order is ``(NS3,NS2,R1,R4)``, so this function
    performs the required external-word swap before selecting the recursion
    family.  ``condition_limit`` is accepted for drop-in backend routing but
    is irrelevant because no Gram matrix is constructed.
    """

    del condition_limit
    momenta = tuple(_finite_complex(value, "external momentum") for value in external_momenta)
    if len(momenta) != 4:
        raise ValueError("external_momenta must contain (p_R1,p_NS2,p_NS3,p_R4)")
    words = tuple(ns_words)
    if len(words) != 2:
        raise ValueError("ns_words must contain the words on original legs (2,3)")
    parities = tuple(int(value) for value in ramond_ground_parities)
    if len(parities) != 2 or any(value not in (0, 1) for value in parities):
        raise ValueError(
            "ramond_ground_parities must contain the parities at (one,infinity)"
        )
    p_r1, p_ns2, p_ns3, p_r4 = momenta
    scale = math.sqrt(2.0)
    from spin23_super_liouville_data import ns_weight

    # Original words are (NS2,NS3); the crossed recursion expects (NS3,NS2).
    descendants = (
        _ns_word_is_star(words[1]),
        _ns_word_is_star(words[0]),
    )
    family = _cached_b_one_sld_family(
        complex(ns_weight(p_ns3)),
        complex(ns_weight(p_ns2)),
        1.0j * p_r1 / scale,
        1.0j * p_r4 / scale,
        _validate_order(maximum_twice_level),
        _validate_component(component),
        _validate_sign(rr_structure_sign),
        _validate_chirality(chirality),
        descendants,
        max(80, int(digits)),
    )
    return family.series(
        complex(ns_weight(internal_momentum)),
        ramond_ground_parities=parities,
    )


def two_star_internal_ns_sld_cached_elliptic_h_series(
    internal_momentum: complex,
    *,
    external_momenta: Sequence[complex],
    maximum_twice_level: int,
    component: Component,
    rr_structure_sign: int,
    chirality: Chirality = "holomorphic",
    ramond_ground_parities: Sequence[int] = (0, 0),
    digits: int = 50,
    condition_limit: float = 1.0e13,
) -> MixedEllipticBlockSeries:
    r"""Physical-order adapter for the cached two-star reference table.

    Inputs are ``(R1,NS2,NS3,R4)=(0,z,1,infinity)``.  As in the recursive
    adapter, the standardized crossed order is ``(NS3,NS2,R1,R4)``.  This
    function is deliberately two-star only; callers select it only when both
    original NS words are ``(G_-1/2,)``.
    """

    momenta = tuple(
        _finite_complex(value, "external momentum") for value in external_momenta
    )
    if len(momenta) != 4:
        raise ValueError("external_momenta must contain (p_R1,p_NS2,p_NS3,p_R4)")
    p_r1, p_ns2, p_ns3, p_r4 = momenta
    from spin23_super_liouville_data import ns_weight

    scale = math.sqrt(2.0)
    return cached_two_star_inverse_gram_h_series(
        c=13.5,
        h_internal=complex(ns_weight(internal_momentum)),
        beta_one=1.0j * p_r1 / scale,
        beta_infinity=1.0j * p_r4 / scale,
        h_ns_zero=complex(ns_weight(p_ns3)),
        h_ns_moving=complex(ns_weight(p_ns2)),
        maximum_t_order=maximum_twice_level,
        component=component,
        rr_structure_sign=rr_structure_sign,
        chirality=chirality,
        ramond_ground_parities=ramond_ground_parities,
        digits=digits,
        condition_limit=condition_limit,
    )


__all__ = [
    "BOneLimitResult",
    "Chirality",
    "Component",
    "MixedHRecursionFamily",
    "MixedEllipticBlockSeries",
    "TWO_STAR_INVERSE_GRAM_CACHE_VALIDATED_T_ORDER",
    "TWO_STAR_RECURSION_CERTIFIED_MAXIMUM_T_ORDER",
    "b_one_limit_series",
    "cached_two_star_inverse_gram_h_series",
    "clear_two_star_inverse_gram_cache",
    "hjs_two_ramond_internal_ns_elliptic_h_series",
    "mixed_h_series",
    "mixed_primary_h_series",
    "two_star_internal_ns_sld_cached_elliptic_h_series",
    "two_star_inverse_gram_cache_info",
    "two_ramond_internal_ns_sld_elliptic_h_series",
]
