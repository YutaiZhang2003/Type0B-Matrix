r"""Algebraic elliptic-nome acceleration of direct sphere-block series.

This module is deliberately independent of any particular NS or Ramond
recursion.  A direct chiral block is supplied in the normalized form

.. math::

   {\cal F}(z)=z^\alpha\sum_{\ell\geq\ell_0}c_\ell z^{\ell/2},

where ``ell`` is the twice-level used by the direct inverse-Gram block
builders.  The descendant series is re-expanded in

.. math::

   q(z)=\exp[-\pi K(1-z)/K(z)],\qquad t=\sqrt q,

using ``z = lambda(q) = 16 t**2 + O(t**4)``.  This is an algebraic change of
variable, not an elliptic ``h``-recursion ansatz, so it applies equally to
mixed and Ramond blocks once their direct coefficients are known.

Branch convention
-----------------
``elliptic_nome`` takes the upper lip of the real cuts by default.  Together
with the principal square root this gives ``sqrt(z) = 4*t + O(t**3)`` near
zero.  Passing ``cut_side="lower"`` gives the conjugate lower lip.  Away from
the cuts the argument itself fixes the analytic continuation.

Truncation guarantee
--------------------
``level_step`` declares the lattice on which direct coefficients can occur.
For example, an even NS component has levels ``0,2,4,...`` and an odd one has
``1,3,5,...``, both with ``level_step=2``.  If direct data are known through
twice-level ``L``, the first unknown term starts at ``t**(L+level_step)``.
The returned coefficients are consequently exact through
``L+level_step-1``.  An omitted coefficient on the declared, already-known
lattice is interpreted as an exact zero.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral
from typing import Literal, Mapping

import mpmath as mp
import numpy as np
import scipy.special as sps

from heterotic_so23_1to3_vvvv_fit_bundle.heterotic_so23_1to3 import (
    _modular_series,
    _series_power,
)


CutSide = Literal["upper", "lower"]


def _validate_cut_side(cut_side: str) -> CutSide:
    if cut_side not in ("upper", "lower"):
        raise ValueError("cut_side must be 'upper' or 'lower'")
    return cut_side


def _cut_lip_argument(z: np.ndarray, cut_side: CutSide) -> np.ndarray:
    """Move exact real-cut points to a numerically resolvable lip."""

    out = np.asarray(z, dtype=np.complex128).copy()
    real_cut = (out.imag == 0.0) & ((out.real < 0.0) | (out.real > 1.0))
    if np.any(real_cut):
        # Carlson RF is undefined numerically when one argument lies exactly
        # on its negative-real cut.  A few ulps choose the boundary value
        # without affecting the quadrature at binary64 accuracy.
        scale = np.maximum(1.0, np.abs(out.real[real_cut]))
        sign = 1.0 if cut_side == "upper" else -1.0
        out[real_cut] += 1j * sign * (8.0 * np.finfo(float).eps) * scale
    return out


def _mpmath_nome(z: complex) -> complex:
    """High-precision scalar fallback for exceptional Carlson-RF points."""

    with mp.workdps(50):
        value = mp.mpc(z.real, z.imag)
        result = mp.exp(-mp.pi * mp.ellipk(1 - value) / mp.ellipk(value))
        return complex(result)


def elliptic_nome(
    z: complex | np.ndarray,
    *,
    cut_side: CutSide = "upper",
) -> complex | np.ndarray:
    r"""Return ``q(z)=exp(-pi*K(1-z)/K(z))`` for scalars or arrays.

    The Carlson symmetric form supplies the fast vectorized path.  Exact
    points on ``(-infinity,0)`` or ``(1,infinity)`` are evaluated on the
    requested lip, and rare non-finite Carlson results fall back to mpmath.
    The removable endpoint limits are defined as ``q(0)=0`` and ``q(1)=1``.
    """

    side = _validate_cut_side(cut_side)
    original = np.asarray(z, dtype=np.complex128)
    scalar = original.ndim == 0
    shape = original.shape
    flat = original.reshape(-1)
    if np.any(~np.isfinite(flat)):
        raise ValueError("z must contain only finite complex values")

    work = _cut_lip_argument(flat, side)
    result = np.empty_like(work)
    at_zero = work == 0.0
    at_one = work == 1.0
    regular = ~(at_zero | at_one)
    result[at_zero] = 0.0
    result[at_one] = 1.0

    if np.any(regular):
        values = work[regular]
        zeros = np.zeros_like(values)
        ones = np.ones_like(values)
        with np.errstate(all="ignore"):
            k_z = sps.elliprf(zeros, 1.0 - values, ones)
            k_one_minus_z = sps.elliprf(zeros, values, ones)
            q_values = np.exp(-np.pi * k_one_minus_z / k_z)

        bad = ~np.isfinite(q_values)
        if np.any(bad):
            q_values = q_values.copy()
            for index in np.flatnonzero(bad):
                q_values[index] = _mpmath_nome(complex(values[index]))
        result[regular] = q_values

    result = result.reshape(shape)
    if scalar:
        return complex(result.item())
    return result


def elliptic_nome_sqrt(
    z: complex | np.ndarray,
    *,
    cut_side: CutSide = "upper",
) -> complex | np.ndarray:
    r"""Return the branch of ``t=sqrt(q(z))`` compatible with ``sqrt(z)``.

    On an exact real cut, ``cut_side`` is also used to retain the sign of the
    infinitesimal imaginary part before taking the principal square root.
    """

    q = elliptic_nome(z, cut_side=cut_side)
    if np.isscalar(q):
        return complex(np.sqrt(np.complex128(q)))
    return np.sqrt(np.asarray(q, dtype=np.complex128))


@dataclass(frozen=True)
class NomeUniformizedBlockSeries:
    """A finite descendant block polynomial in ``t=sqrt(q)``."""

    t_coefficients: tuple[complex, ...]
    direct_coefficients: Mapping[int, complex]
    level_offset: int
    level_step: int
    known_through_twice_level: int
    guaranteed_t_order: int

    def value_from_t(self, t: complex | np.ndarray) -> complex | np.ndarray:
        """Evaluate the descendant polynomial at a supplied nome square root."""

        argument = np.asarray(t, dtype=np.complex128)
        scalar = argument.ndim == 0
        value = np.zeros_like(argument)
        for coefficient in reversed(self.t_coefficients):
            value = value * argument + coefficient
        if scalar:
            return complex(value.item())
        return value

    def descendant_value(
        self,
        z: complex | np.ndarray,
        *,
        cut_side: CutSide = "upper",
    ) -> complex | np.ndarray:
        """Evaluate the uniformized descendant series on the declared branch."""

        return self.value_from_t(elliptic_nome_sqrt(z, cut_side=cut_side))

    def value(
        self,
        z: complex | np.ndarray,
        *,
        primary_exponent: complex = 0.0,
        cut_side: CutSide = "upper",
    ) -> complex | np.ndarray:
        r"""Evaluate ``z**primary_exponent`` times the descendant series.

        The same lip prescription is used for the primary logarithm.  At
        ``z=0`` a nonzero primary exponent is intentionally rejected rather
        than guessing a singular or vanishing limit.
        """

        side = _validate_cut_side(cut_side)
        original = np.asarray(z, dtype=np.complex128)
        scalar = original.ndim == 0
        exponent = complex(primary_exponent)
        descendant = np.asarray(
            self.descendant_value(original, cut_side=side),
            dtype=np.complex128,
        )
        if exponent == 0.0:
            result = descendant
        else:
            if np.any(original == 0.0):
                raise ValueError(
                    "z=0 cannot be evaluated with a nonzero primary exponent"
                )
            branch_argument = _cut_lip_argument(original.reshape(-1), side).reshape(
                original.shape
            )
            result = np.exp(exponent * np.log(branch_argument)) * descendant
        if scalar:
            return complex(result.item())
        return result


def uniformize_direct_block(
    coefficients: Mapping[int, complex],
    *,
    level_step: int = 1,
    level_offset: int | None = None,
    known_through_twice_level: int | None = None,
) -> NomeUniformizedBlockSeries:
    r"""Convert direct ``z**(ell/2)`` coefficients to a guaranteed t-series.

    Parameters
    ----------
    coefficients:
        Mapping from nonnegative twice-level ``ell`` to ``c_ell``.
    level_step:
        Spacing of the allowed level lattice.  Use ``2`` for a fixed even or
        odd NS component, and ``1`` when both parities may occur.
    level_offset:
        Residue class modulo ``level_step``.  It defaults to the residue of
        the smallest supplied level.
    known_through_twice_level:
        Last direct level known on that lattice.  It defaults to the largest
        supplied key.  Missing entries below it are treated as exact zeros.

    Returns
    -------
    NomeUniformizedBlockSeries
        Coefficients through ``known_through + level_step - 1``.  Every
        returned coefficient is independent of all uncomputed direct levels.
    """

    if not isinstance(level_step, Integral) or isinstance(level_step, bool):
        raise TypeError("level_step must be an integer")
    step = int(level_step)
    if step <= 0:
        raise ValueError("level_step must be positive")
    if not coefficients:
        raise ValueError("coefficients must not be empty")

    normalized: dict[int, complex] = {}
    for level, coefficient in coefficients.items():
        if not isinstance(level, Integral) or isinstance(level, bool):
            raise TypeError("coefficient keys must be integer twice-levels")
        level_int = int(level)
        if level_int < 0:
            raise ValueError("coefficient twice-levels must be nonnegative")
        value = complex(coefficient)
        if not (math.isfinite(value.real) and math.isfinite(value.imag)):
            raise ValueError("direct coefficients must be finite")
        normalized[level_int] = value

    smallest = min(normalized)
    if level_offset is None:
        offset = smallest % step
    else:
        if not isinstance(level_offset, Integral) or isinstance(level_offset, bool):
            raise TypeError("level_offset must be an integer")
        offset = int(level_offset)
        if not 0 <= offset < step:
            raise ValueError("level_offset must satisfy 0 <= offset < level_step")
    if any(level % step != offset for level in normalized):
        raise ValueError("a coefficient key lies outside the declared level lattice")

    if known_through_twice_level is None:
        known_through = max(normalized)
    else:
        if not isinstance(known_through_twice_level, Integral) or isinstance(
            known_through_twice_level, bool
        ):
            raise TypeError("known_through_twice_level must be an integer")
        known_through = int(known_through_twice_level)
    if known_through < 0 or known_through % step != offset:
        raise ValueError("known_through_twice_level must lie on the level lattice")
    if max(normalized) > known_through:
        raise ValueError("a coefficient key exceeds known_through_twice_level")

    guaranteed_order = known_through + step - 1
    length = guaranteed_order + 1
    _, reduced_lambda, _ = _modular_series(length)
    t_coefficients = np.zeros(length, dtype=np.complex128)

    for level in range(offset, known_through + 1, step):
        coefficient = normalized.get(level, 0.0j)
        if coefficient == 0.0:
            continue
        available = length - level
        reduced_power = _series_power(
            reduced_lambda,
            0.5 * level,
            available,
        )
        t_coefficients[level:] += (
            coefficient * (4.0**level) * reduced_power[:available]
        )

    return NomeUniformizedBlockSeries(
        t_coefficients=tuple(complex(value) for value in t_coefficients),
        direct_coefficients=dict(normalized),
        level_offset=offset,
        level_step=step,
        known_through_twice_level=known_through,
        guaranteed_t_order=guaranteed_order,
    )


__all__ = [
    "CutSide",
    "NomeUniformizedBlockSeries",
    "elliptic_nome",
    "elliptic_nome_sqrt",
    "uniformize_direct_block",
]
