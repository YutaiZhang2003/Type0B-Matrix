#!/usr/bin/env python3
r"""HJS four-Ramond/internal-NS elliptic ``h``-recursion.

The puncture order is ``(R1,R2,R3,R4)=(0,z,1,infinity)`` and
``t=sqrt(q)``.  This implements the recurrence around eqs. (6.14)--(6.16)
of Hadasz--Jaskolski--Suchanek, arXiv:0810.1203, in the conventions of
``general_four_ramond_sld_chiral_block``.

The repository's normalized odd ``w+`` block differs from the printed HJS
odd block by ``left_structure_sign*right_structure_sign``.  Arbitrary HJS
ground components introduce only the exact constant phase returned by
:func:`four_ramond_ground_map`; neither structure sign is changed.

At ``c=27/2`` (``b=1``), individual Kac residues are confluent.  The public
continuation and reusable family evaluate the symmetric pair
``b=exp(+-epsilon)`` and extrapolate in ``epsilon**2``.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import cmath
import math
from typing import Literal, Sequence

import mpmath as mp
import numpy as np

from sphere_block_uniformization import _cut_lip_argument, elliptic_nome


Component = Literal["even", "odd"]
Chirality = Literal["holomorphic", "antiholomorphic"]


def _component(value: str) -> Component:
    if value not in ("even", "odd"):
        raise ValueError("component must be 'even' or 'odd'")
    return value  # type: ignore[return-value]


def _chirality(value: str) -> Chirality:
    if value not in ("holomorphic", "antiholomorphic"):
        raise ValueError("chirality must be 'holomorphic' or 'antiholomorphic'")
    return value  # type: ignore[return-value]


def _sign(value: int, name: str) -> int:
    if value not in (-1, 1):
        raise ValueError(f"{name} must be +1 or -1")
    return int(value)


def _order(value: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError("maximum_t_order must be a nonnegative integer")
    return value


def _finite(value: complex, name: str) -> complex:
    result = complex(value)
    if not (math.isfinite(result.real) and math.isfinite(result.imag)):
        raise ValueError(f"{name} must be finite")
    return result


def _parities(values: Sequence[int]) -> tuple[int, int, int, int]:
    result = tuple(int(value) for value in values)
    if len(result) != 4 or any(value not in (0, 1) for value in result):
        raise ValueError("external_ground_parities must contain four zero/one entries")
    return result  # type: ignore[return-value]


def _validate_ground_normalization(
    external_betas: Sequence[complex],
    parities: tuple[int, int, int, int],
) -> None:
    for index, (beta, parity) in enumerate(zip(external_betas, parities, strict=True)):
        if parity and complex(beta) == 0:
            raise ValueError(
                "an odd HJS external ground state is singular at beta=0 "
                f"(external index {index})"
            )


def _canonical_b_from_c(c: complex) -> complex:
    central_charge = complex(c)
    charge = cmath.sqrt((central_charge - 1.5) / 3.0)
    discriminant = cmath.sqrt(charge**2 - 4.0)
    roots = ((charge + discriminant) / 2.0, (charge - discriminant) / 2.0)
    nonzero = tuple(root for root in roots if root != 0)
    if not nonzero:
        raise ValueError("central charge does not determine a nonzero b")
    return max(nonzero, key=lambda root: (abs(root), root.real, root.imag))


@dataclass(frozen=True)
class FourRamondGroundMap:
    """Exact reduction of one arbitrary-ground block to a standard HJS block."""

    phase: complex
    left_structure_sign: int
    right_structure_sign: int
    component: Component
    chirality: Chirality


def _trinion_ground_phase(
    middle_parity: int,
    zero_parity: int,
    structure_sign: int,
    component_parity: int,
    chirality: Chirality,
) -> complex:
    """Normalized NS--R_middle--R_zero component relative to ``w+,w+``."""

    key = (middle_parity, zero_parity)
    if key == (0, 0):
        phase = 1.0 + 0.0j
    elif key == (0, 1):
        phase = 1.0 + 0.0j if component_parity == 0 else -1.0j
    elif key == (1, 0):
        phase = structure_sign * (
            1.0j if component_parity == 0 else 1.0
        )
    elif key == (1, 1):
        phase = complex(structure_sign)
    else:
        raise ValueError("Ramond ground parities must be zero or one")
    return phase if chirality == "holomorphic" else phase.conjugate()


def four_ramond_ground_map(
    external_ground_parities: Sequence[int],
    *,
    component: Component,
    left_structure_sign: int,
    right_structure_sign: int,
    chirality: Chirality = "holomorphic",
) -> FourRamondGroundMap:
    r"""Map an arbitrary external ground word to the standard ``w+`` block.

    The external ordering is ``(R1,R2,R3,R4)``.  The right NS--RR trinion
    has slots ``(R2,R1)``.  Turning the left RR--NS trinion into NS--RR
    slots ``(R3,R4)`` also supplies ``(-1)**(a4*q)``.  This is the
    ``(-1)**#I`` in HJS 0810.1203v2, (4.10), absent from the right-endpoint
    identity (4.9) for an external ground state.  The structure signs are
    unchanged.
    """

    parities = _parities(external_ground_parities)
    selected_component = _component(component)
    selected_chirality = _chirality(chirality)
    left_sign = _sign(left_structure_sign, "left_structure_sign")
    right_sign = _sign(right_structure_sign, "right_structure_sign")
    component_parity = 0 if selected_component == "even" else 1
    a1, a2, a3, a4 = parities
    phase = (-1) ** (a4 * component_parity) * _trinion_ground_phase(
        a3,
        a4,
        left_sign,
        component_parity,
        selected_chirality,
    ) * _trinion_ground_phase(
        a2,
        a1,
        right_sign,
        component_parity,
        selected_chirality,
    )
    return FourRamondGroundMap(
        phase=phase,
        left_structure_sign=left_sign,
        right_structure_sign=right_sign,
        component=selected_component,
        chirality=selected_chirality,
    )


@dataclass(frozen=True)
class FourRamondEllipticBlockSeries:
    """Finite HJS four-R elliptic block ``H(t)``, including ground metadata."""

    h_coefficients: tuple[complex, ...]
    component: Component
    chirality: Chirality
    left_structure_sign: int
    right_structure_sign: int
    external_ground_parities: tuple[int, int, int, int]
    b: complex
    c: complex
    h_internal: complex
    external_betas: tuple[complex, complex, complex, complex]
    maximum_t_order: int

    @property
    def external_weights(self) -> tuple[complex, complex, complex, complex]:
        return tuple(
            self.c / 24.0 - beta**2 for beta in self.external_betas
        )  # type: ignore[return-value]

    def value_from_t(
        self,
        t: complex | np.ndarray,
    ) -> complex | np.ndarray:
        argument = np.asarray(t, dtype=np.complex128)
        scalar = argument.ndim == 0
        result = np.zeros_like(argument)
        for coefficient in reversed(self.h_coefficients):
            result = result * argument + coefficient
        if scalar:
            return complex(result.item())
        return result

    @staticmethod
    def _theta3(q: np.ndarray) -> np.ndarray:
        if np.any(np.abs(q) >= 1.0):
            raise ArithmeticError("elliptic nome left the open unit disk")
        result = np.ones_like(q)
        for index in range(1, 10000):
            term = 2.0 * np.power(q, index * index)
            result += term
            if np.max(np.abs(term), initial=0.0) < 2.0e-15:
                return result
        raise ArithmeticError("theta3 sum did not converge")

    def value(
        self,
        z: complex | np.ndarray,
        *,
        cut_side: Literal["upper", "lower"] = "upper",
    ) -> complex | np.ndarray:
        """Evaluate the complete chiral block with the HJS prefactor."""

        points = np.asarray(z, dtype=np.complex128)
        scalar = points.ndim == 0
        if np.any((points == 0.0) | (points == 1.0)):
            raise ValueError("the elliptic block is singular at zero or one")
        q = np.asarray(elliptic_nome(points, cut_side=cut_side), dtype=np.complex128)
        theta = self._theta3(q)
        branch_z = _cut_lip_argument(points.reshape(-1), cut_side).reshape(
            points.shape
        )
        h1, h2, h3, h4 = self.external_weights
        c0 = (self.c - 1.5) / 24.0
        theta_exponent = (self.c - 1.5) / 2.0 - 4.0 * (
            h1 + h2 + h3 + h4
        )
        logarithm = (
            (self.h_internal - c0) * np.log(16.0 * q)
            + (c0 - h1 - h2) * np.log(branch_z)
            + (c0 - h2 - h3) * np.log(1.0 - branch_z)
            + theta_exponent * np.log(theta)
        )
        result = np.exp(logarithm) * self.value_from_t(np.sqrt(q))
        if np.any(~np.isfinite(result)):
            raise ArithmeticError("non-finite four-R elliptic block value")
        if scalar:
            return complex(np.asarray(result).item())
        return np.asarray(result, dtype=np.complex128)


class _FourRamondRecursionEngine:
    def __init__(
        self,
        *,
        b: mp.mpc,
        external_betas: tuple[mp.mpc, mp.mpc, mp.mpc, mp.mpc],
    ) -> None:
        if b == 0:
            raise ValueError("b must be nonzero")
        self.b = b
        self.q_background = b + 1 / b
        self.external_betas = external_betas
        self._h_cache: dict[tuple[int, int], mp.mpc] = {}
        self._a_cache: dict[tuple[int, int], mp.mpc] = {}
        self._rr_cache: dict[tuple[int, int, int, str], mp.mpc] = {}
        self._recursion_cache: dict[
            tuple[str, int, int, str, int], tuple[mp.mpc, ...]
        ] = {}

    def h_rs(self, r: int, s: int) -> mp.mpc:
        key = (r, s)
        if key not in self._h_cache:
            momentum = r * self.b + s / self.b
            self._h_cache[key] = (
                self.q_background**2 - momentum**2
            ) / 8
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

    def rr_fusion(
        self,
        r: int,
        s: int,
        structure_sign: int,
        endpoint: Literal["left", "right"],
    ) -> mp.mpc:
        key = (r, s, structure_sign, endpoint)
        if key in self._rr_cache:
            return self._rr_cache[key]
        beta1, beta2, beta3, beta4 = self.external_betas
        beta_zero, beta_moving = (
            (beta4, beta3) if endpoint == "left" else (beta1, beta2)
        )
        value = mp.mpc(1)
        denominator = 2 * mp.sqrt(2)
        for k in range(r):
            for ell in range(s):
                p = 1 - r + 2 * k
                q = 1 - s + 2 * ell
                shift = (p * self.b + q / self.b) / denominator
                if (k + ell) % 2 == 0:
                    value *= beta_moving - structure_sign * beta_zero - shift
                else:
                    value *= beta_moving + structure_sign * beta_zero - shift
        self._rr_cache[key] = value
        return value

    def recurse(
        self,
        component: Component,
        left_structure_sign: int,
        right_structure_sign: int,
        current_h: mp.mpc,
        maximum_t_order: int,
    ) -> tuple[mp.mpc, ...]:
        key = (
            component,
            left_structure_sign,
            right_structure_sign,
            mp.nstr(current_h, mp.mp.dps + 8),
            maximum_t_order,
        )
        if key in self._recursion_cache:
            return self._recursion_cache[key]
        total = [mp.mpc(0) for _ in range(maximum_t_order + 1)]
        if component == "even":
            total[0] = mp.mpc(1)
        for r in range(1, maximum_t_order + 1):
            for s in range(1, maximum_t_order // r + 1):
                if (r - s) % 2:
                    continue
                null_level = r * s
                pole = self.h_rs(r, s)
                denominator = current_h - pole
                if denominator == 0:
                    raise ZeroDivisionError(
                        f"the internal weight hit the ({r},{s}) Kac pole"
                    )
                even_null = r % 2 == 0
                if even_null:
                    tail_component = component
                    tail_left_sign = left_structure_sign
                    tail_right_sign = right_structure_sign
                    null_phase = mp.mpc(1)
                else:
                    tail_component = "odd" if component == "even" else "even"
                    tail_left_sign = -left_structure_sign
                    tail_right_sign = -right_structure_sign
                    null_phase = mp.mpc(-1)
                residue = (
                    mp.power(4, null_level)
                    * null_phase
                    * self.a_rs(r, s)
                    * self.rr_fusion(
                        r,
                        s,
                        left_structure_sign,
                        "left",
                    )
                    * self.rr_fusion(
                        r,
                        s,
                        right_structure_sign,
                        "right",
                    )
                    / denominator
                )
                shifted_h = pole + mp.mpf(null_level) / 2
                tail = self.recurse(
                    tail_component,
                    tail_left_sign,
                    tail_right_sign,
                    shifted_h,
                    maximum_t_order - null_level,
                )
                for index, coefficient in enumerate(tail):
                    total[index + null_level] += residue * coefficient
        result = tuple(total)
        self._recursion_cache[key] = result
        return result


def _standard_coefficients(
    engine: _FourRamondRecursionEngine,
    *,
    h_internal: mp.mpc,
    maximum_t_order: int,
    component: Component,
    left_structure_sign: int,
    right_structure_sign: int,
) -> tuple[mp.mpc, ...]:
    coefficients = engine.recurse(
        component,
        left_structure_sign,
        right_structure_sign,
        h_internal,
        maximum_t_order,
    )
    # The repository normalizes its odd standard block by s_L*s_R relative
    # to the printed HJS F^(1/2) block.
    normalization = (
        1 if component == "even" else left_structure_sign * right_structure_sign
    )
    return tuple(normalization * coefficient for coefficient in coefficients)


def _interpolate_zero(xs: Sequence[mp.mpf], ys: Sequence[mp.mpc]) -> mp.mpc:
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


def _series_object(
    coefficients: Sequence[mp.mpc | complex],
    *,
    b: complex,
    h_internal: complex,
    external_betas: tuple[complex, complex, complex, complex],
    maximum_t_order: int,
    component: Component,
    left_structure_sign: int,
    right_structure_sign: int,
    chirality: Chirality,
    external_ground_parities: tuple[int, int, int, int],
) -> FourRamondEllipticBlockSeries:
    _validate_ground_normalization(external_betas, external_ground_parities)
    charge = b + 1.0 / b
    central_charge = 1.5 + 3.0 * charge**2
    ground_map = four_ramond_ground_map(
        external_ground_parities,
        component=component,
        left_structure_sign=left_structure_sign,
        right_structure_sign=right_structure_sign,
        chirality=chirality,
    )
    return FourRamondEllipticBlockSeries(
        h_coefficients=tuple(
            ground_map.phase * complex(value) for value in coefficients
        ),
        component=component,
        chirality=chirality,
        left_structure_sign=left_structure_sign,
        right_structure_sign=right_structure_sign,
        external_ground_parities=external_ground_parities,
        b=b,
        c=central_charge,
        h_internal=h_internal,
        external_betas=external_betas,
        maximum_t_order=maximum_t_order,
    )


def four_ramond_h_series(
    *,
    b: complex,
    h_internal: complex,
    external_betas: Sequence[complex],
    maximum_t_order: int,
    component: Component,
    left_structure_sign: int,
    right_structure_sign: int,
    chirality: Chirality = "holomorphic",
    external_ground_parities: Sequence[int] = (0, 0, 0, 0),
    working_digits: int = 60,
) -> FourRamondEllipticBlockSeries:
    """Return one generic-``b`` four-R elliptic recursion series."""

    order = _order(maximum_t_order)
    selected_component = _component(component)
    selected_chirality = _chirality(chirality)
    left_sign = _sign(left_structure_sign, "left_structure_sign")
    right_sign = _sign(right_structure_sign, "right_structure_sign")
    parities = _parities(external_ground_parities)
    beta_values = tuple(_finite(value, "external beta") for value in external_betas)
    if len(beta_values) != 4:
        raise ValueError("external_betas must contain four entries")
    b_value = _finite(b, "b")
    internal_weight = _finite(h_internal, "h_internal")
    if b_value == 0:
        raise ValueError("b must be nonzero")
    if not isinstance(working_digits, int) or working_digits < 30:
        raise ValueError("working_digits must be an integer at least 30")
    with mp.workdps(working_digits):
        engine = _FourRamondRecursionEngine(
            b=mp.mpc(b_value),
            external_betas=tuple(mp.mpc(value) for value in beta_values),  # type: ignore[arg-type]
        )
        coefficients = _standard_coefficients(
            engine,
            h_internal=mp.mpc(internal_weight),
            maximum_t_order=order,
            component=selected_component,
            left_structure_sign=left_sign,
            right_structure_sign=right_sign,
        )
    return _series_object(
        coefficients,
        b=b_value,
        h_internal=internal_weight,
        external_betas=beta_values,  # type: ignore[arg-type]
        maximum_t_order=order,
        component=selected_component,
        left_structure_sign=left_sign,
        right_structure_sign=right_sign,
        chirality=selected_chirality,
        external_ground_parities=parities,
    )


@dataclass(frozen=True)
class FourRamondBOneLimitResult:
    series: FourRamondEllipticBlockSeries
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


def four_ramond_b_one_limit_series(
    *,
    h_internal: complex,
    external_betas: Sequence[complex],
    maximum_t_order: int,
    component: Component,
    left_structure_sign: int,
    right_structure_sign: int,
    chirality: Chirality = "holomorphic",
    external_ground_parities: Sequence[int] = (0, 0, 0, 0),
    initial_epsilon: float = 0.2,
    sample_count: int = 6,
    working_digits: int | None = None,
) -> FourRamondBOneLimitResult:
    """Return the confluent self-dual series by symmetric continuation."""

    order = _order(maximum_t_order)
    selected_component = _component(component)
    selected_chirality = _chirality(chirality)
    left_sign = _sign(left_structure_sign, "left_structure_sign")
    right_sign = _sign(right_structure_sign, "right_structure_sign")
    parities = _parities(external_ground_parities)
    beta_values = tuple(_finite(value, "external beta") for value in external_betas)
    if len(beta_values) != 4:
        raise ValueError("external_betas must contain four entries")
    internal_weight = _finite(h_internal, "h_internal")
    if not math.isfinite(initial_epsilon) or not 0 < initial_epsilon < 0.5:
        raise ValueError("initial_epsilon must lie in (0,0.5)")
    if not isinstance(sample_count, int) or sample_count < 4:
        raise ValueError("sample_count must be an integer at least four")
    digits = max(80, 4 * order + 40) if working_digits is None else working_digits
    if not isinstance(digits, int) or digits < 50:
        raise ValueError("working_digits must be an integer at least 50")
    with mp.workdps(digits):
        epsilons = tuple(
            mp.mpf(str(initial_epsilon)) / mp.power(2, index)
            for index in range(sample_count)
        )
        symmetric: list[tuple[mp.mpc, ...]] = []
        defects: list[tuple[mp.mpf, ...]] = []
        for epsilon in epsilons:
            rows = []
            for regulated_b in (mp.exp(epsilon), mp.exp(-epsilon)):
                engine = _FourRamondRecursionEngine(
                    b=regulated_b,
                    external_betas=tuple(mp.mpc(value) for value in beta_values),  # type: ignore[arg-type]
                )
                rows.append(
                    _standard_coefficients(
                        engine,
                        h_internal=mp.mpc(internal_weight),
                        maximum_t_order=order,
                        component=selected_component,
                        left_structure_sign=left_sign,
                        right_structure_sign=right_sign,
                    )
                )
            plus, minus = rows
            symmetric.append(
                tuple((left + right) / 2 for left, right in zip(plus, minus))
            )
            defects.append(
                tuple(abs(left - right) for left, right in zip(plus, minus))
            )
        xs = [epsilon**2 for epsilon in epsilons]
        coefficients: list[mp.mpc] = []
        errors: list[mp.mpf] = []
        duality: list[mp.mpf] = []
        for index in range(order + 1):
            values = [row[index] for row in symmetric]
            full = _interpolate_zero(xs, values)
            reduced = _interpolate_zero(xs[1:], values[1:])
            coefficients.append(full)
            errors.append(abs(full - reduced))
            duality.append(max(row[index] for row in defects))
    series = _series_object(
        coefficients,
        b=1.0 + 0.0j,
        h_internal=internal_weight,
        external_betas=beta_values,  # type: ignore[arg-type]
        maximum_t_order=order,
        component=selected_component,
        left_structure_sign=left_sign,
        right_structure_sign=right_sign,
        chirality=selected_chirality,
        external_ground_parities=parities,
    )
    return FourRamondBOneLimitResult(
        series=series,
        regulator_epsilons=tuple(float(value) for value in epsilons),
        coefficient_error_estimates=tuple(float(value) for value in errors),
        coefficient_duality_defects=tuple(float(value) for value in duality),
        working_digits=digits,
    )


class FourRamondHRecursionFamily:
    """Reusable fixed-external four-R recursion for varying internal weight."""

    def __init__(
        self,
        *,
        b: complex,
        external_betas: Sequence[complex],
        maximum_t_order: int,
        component: Component,
        left_structure_sign: int,
        right_structure_sign: int,
        chirality: Chirality = "holomorphic",
        initial_epsilon: float = 0.2,
        sample_count: int = 6,
        working_digits: int | None = None,
    ) -> None:
        self.b = _finite(b, "b")
        if self.b == 0:
            raise ValueError("b must be nonzero")
        self.external_betas = tuple(
            _finite(value, "external beta") for value in external_betas
        )
        if len(self.external_betas) != 4:
            raise ValueError("external_betas must contain four entries")
        self.maximum_t_order = _order(maximum_t_order)
        self.component = _component(component)
        self.left_structure_sign = _sign(
            left_structure_sign,
            "left_structure_sign",
        )
        self.right_structure_sign = _sign(
            right_structure_sign,
            "right_structure_sign",
        )
        self.chirality = _chirality(chirality)
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
            tuple[_FourRamondRecursionEngine, _FourRamondRecursionEngine], ...
        ] = ()
        self._engine: _FourRamondRecursionEngine | None = None
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

    def _make_engine(self, b: mp.mpc) -> _FourRamondRecursionEngine:
        return _FourRamondRecursionEngine(
            b=b,
            external_betas=tuple(
                mp.mpc(value) for value in self.external_betas
            ),  # type: ignore[arg-type]
        )

    def _coefficients(
        self,
        engine: _FourRamondRecursionEngine,
        h_internal: mp.mpc,
    ) -> tuple[mp.mpc, ...]:
        return _standard_coefficients(
            engine,
            h_internal=h_internal,
            maximum_t_order=self.maximum_t_order,
            component=self.component,
            left_structure_sign=self.left_structure_sign,
            right_structure_sign=self.right_structure_sign,
        )

    def series(
        self,
        h_internal: complex,
        *,
        external_ground_parities: Sequence[int] = (0, 0, 0, 0),
    ) -> FourRamondEllipticBlockSeries:
        internal_weight = _finite(h_internal, "h_internal")
        parities = _parities(external_ground_parities)
        with mp.workdps(self.working_digits):
            h_mp = mp.mpc(internal_weight)
            if self._confluent:
                symmetric = []
                for plus_engine, minus_engine in self._engines:
                    plus = self._coefficients(plus_engine, h_mp)
                    minus = self._coefficients(minus_engine, h_mp)
                    symmetric.append(
                        tuple((left + right) / 2 for left, right in zip(plus, minus))
                    )
                xs = [epsilon**2 for epsilon in self._epsilons]
                coefficients = tuple(
                    _interpolate_zero(
                        xs,
                        [row[index] for row in symmetric],
                    )
                    for index in range(self.maximum_t_order + 1)
                )
                b_output = 1.0 + 0.0j
            else:
                if self._engine is None:
                    raise AssertionError("generic recursion engine was not initialized")
                coefficients = self._coefficients(self._engine, h_mp)
                b_output = self.b
        return _series_object(
            coefficients,
            b=b_output,
            h_internal=internal_weight,
            external_betas=self.external_betas,  # type: ignore[arg-type]
            maximum_t_order=self.maximum_t_order,
            component=self.component,
            left_structure_sign=self.left_structure_sign,
            right_structure_sign=self.right_structure_sign,
            chirality=self.chirality,
            external_ground_parities=parities,
        )


def hjs_four_ramond_ns_elliptic_h_series(
    *,
    c: complex,
    h_internal: complex,
    external_betas: Sequence[complex],
    external_ground_parities: Sequence[int] = (0, 0, 0, 0),
    maximum_twice_level: int,
    component: Component,
    left_structure_sign: int,
    right_structure_sign: int,
    chirality: Chirality = "holomorphic",
    b: complex | None = None,
    digits: int = 80,
) -> FourRamondEllipticBlockSeries:
    """Adapter matching the general inverse-Gram four-R request API."""

    order = _order(maximum_twice_level)
    central_charge = _finite(c, "c")
    b_value = _canonical_b_from_c(central_charge) if b is None else _finite(b, "b")
    reconstructed = 1.5 + 3.0 * (b_value + 1.0 / b_value) ** 2
    scale = max(1.0, abs(central_charge), abs(reconstructed))
    if abs(reconstructed - central_charge) > 2.0e-10 * scale:
        raise ValueError("the supplied b and c are inconsistent")
    common = dict(
        h_internal=h_internal,
        external_betas=external_betas,
        maximum_t_order=order,
        component=component,
        left_structure_sign=left_structure_sign,
        right_structure_sign=right_structure_sign,
        chirality=chirality,
        external_ground_parities=external_ground_parities,
    )
    if abs(b_value - 1.0) < 1.0e-12 or abs(b_value + 1.0) < 1.0e-12:
        return four_ramond_b_one_limit_series(
            working_digits=max(80, 4 * order + 40, digits),
            **common,
        ).series
    return four_ramond_h_series(
        b=b_value,
        working_digits=max(30, digits),
        **common,
    )


@lru_cache(maxsize=256)
def _cached_sld_family(
    external_betas: tuple[complex, complex, complex, complex],
    maximum_t_order: int,
    component: Component,
    left_structure_sign: int,
    right_structure_sign: int,
    chirality: Chirality,
    working_digits: int,
) -> FourRamondHRecursionFamily:
    return FourRamondHRecursionFamily(
        b=1.0,
        external_betas=external_betas,
        maximum_t_order=maximum_t_order,
        component=component,
        left_structure_sign=left_structure_sign,
        right_structure_sign=right_structure_sign,
        chirality=chirality,
        working_digits=working_digits,
    )


def general_four_ramond_sld_elliptic_h_series(
    internal_momentum: complex,
    *,
    external_momenta: Sequence[complex],
    external_ground_parities: Sequence[int],
    maximum_twice_level: int,
    component: Component,
    left_structure_sign: int,
    right_structure_sign: int,
    chirality: Chirality = "holomorphic",
    digits: int = 80,
    condition_limit: float = 1.0e13,
) -> FourRamondEllipticBlockSeries:
    """Self-dual SLD adapter matching ``general_four_ramond_sld_chiral_block``."""

    del condition_limit
    momenta = tuple(_finite(value, "external momentum") for value in external_momenta)
    if len(momenta) != 4:
        raise ValueError("external_momenta must contain four entries")
    order = _order(maximum_twice_level)
    betas = tuple(1.0j * momentum / math.sqrt(2.0) for momentum in momenta)
    from spin23_super_liouville_data import ns_weight

    family = _cached_sld_family(
        betas,  # type: ignore[arg-type]
        order,
        _component(component),
        _sign(left_structure_sign, "left_structure_sign"),
        _sign(right_structure_sign, "right_structure_sign"),
        _chirality(chirality),
        max(80, 4 * order + 40, int(digits)),
    )
    return family.series(
        complex(ns_weight(internal_momentum)),
        external_ground_parities=external_ground_parities,
    )


__all__ = [
    "Chirality",
    "Component",
    "FourRamondBOneLimitResult",
    "FourRamondEllipticBlockSeries",
    "FourRamondGroundMap",
    "FourRamondHRecursionFamily",
    "four_ramond_b_one_limit_series",
    "four_ramond_ground_map",
    "four_ramond_h_series",
    "general_four_ramond_sld_elliptic_h_series",
    "hjs_four_ramond_ns_elliptic_h_series",
]
