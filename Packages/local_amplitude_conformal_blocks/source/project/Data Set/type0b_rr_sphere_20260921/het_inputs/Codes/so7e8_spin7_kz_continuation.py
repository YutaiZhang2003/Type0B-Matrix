#!/usr/bin/env python3
r"""Stable direct-channel continuation of the :math:`Spin(7)_1` KZ blocks.

The power series returned by
``so7e8_four_ramond_assembly.spin7_spectator_block_series`` is an excellent
local definition at ``z=0`` but converges unacceptably slowly in the
``z -> 1`` lens.  This module keeps the *same* ``(0,z)|(1,infinity)``
channel and continues that Frobenius solution by integrating the exact KZ
system

.. math::

   {d f\over dz}=\left({A\over z}+{B\over z-1}\right)f.

No Fierz transformation, external-vertex permutation, or Ramond cocycle
transport enters this operation.  It is consequently useful for auditing a
four-Ramond Liouville integral without assuming an as-yet uncertified local
``w=1-z`` component map.
"""

from __future__ import annotations

from dataclasses import dataclass
import cmath
import math
from typing import Sequence

import numpy as np
from scipy.integrate import solve_ivp

from so7e8_four_ramond_assembly import (
    SPIN7_KZ_PAIR_0Z,
    SPIN7_KZ_PAIR_Z1,
    Spin7AffineChannel,
    spin7_spectator_block_series,
)


_A_MATRIX = np.asarray(SPIN7_KZ_PAIR_0Z, dtype=np.complex128) / 6.0
_B_MATRIX = np.asarray(SPIN7_KZ_PAIR_Z1, dtype=np.complex128) / 6.0


@dataclass(frozen=True)
class Spin7KZContinuation:
    """One continued block value together with integration diagnostics."""

    channel: Spin7AffineChannel
    z: complex
    value: tuple[complex, complex, complex, complex]
    used_ode: bool
    starting_radius: float
    accepted_steps: int
    function_evaluations: int

    def as_array(self) -> np.ndarray:
        return np.asarray(self.value, dtype=np.complex128)


def _validate_channel(channel: str) -> Spin7AffineChannel:
    if channel not in ("vacuum", "vector"):
        raise ValueError("channel must be 'vacuum' or 'vector'")
    return channel  # type: ignore[return-value]


def continue_spin7_spectator_block(
    channel: Spin7AffineChannel,
    z: complex,
    *,
    starting_radius: float = 1.0e-5,
    initial_series_order: int = 12,
    direct_series_radius: float = 0.08,
    relative_tolerance: float = 2.0e-12,
    absolute_tolerance: float = 2.0e-13,
) -> Spin7KZContinuation:
    r"""Continue one normalized affine block along a principal radial path.

    The target is restricted to the punctured open unit disk, which is the
    folded sphere region used by the amplitude evaluator.  At small radius
    the local Frobenius series is returned directly.  Otherwise it supplies
    initial data at ``starting_radius * exp(i arg(z))`` and the exact KZ
    system is integrated radially outwards with an eighth-order method.

    A radial path never encircles either Fuchsian singularity.  Its initial
    power uses Python's principal logarithm, so the result has exactly the
    same upper/lower-lip convention as the local series.
    """

    affine_channel = _validate_channel(channel)
    point = complex(z)
    if not (math.isfinite(point.real) and math.isfinite(point.imag)):
        raise ValueError("z must be finite")
    modulus = abs(point)
    if not 0.0 < modulus < 1.0:
        raise ValueError("the KZ continuation requires 0 < |z| < 1")
    if not 0.0 < starting_radius < 1.0:
        raise ValueError("starting_radius must lie in (0,1)")
    if not isinstance(initial_series_order, int) or initial_series_order < 1:
        raise ValueError("initial_series_order must be a positive integer")
    if not 0.0 < direct_series_radius < 1.0:
        raise ValueError("direct_series_radius must lie in (0,1)")
    if relative_tolerance <= 0.0 or absolute_tolerance <= 0.0:
        raise ValueError("ODE tolerances must be positive")

    local_series = spin7_spectator_block_series(
        affine_channel,
        maximum_order=initial_series_order,
    )
    if modulus <= direct_series_radius:
        direct = local_series.value(point)
        return Spin7KZContinuation(
            channel=affine_channel,
            z=point,
            value=tuple(complex(entry) for entry in direct),  # type: ignore[arg-type]
            used_ode=False,
            starting_radius=modulus,
            accepted_steps=0,
            function_evaluations=0,
        )

    # Near z=1, integrating the radius of z asks the ODE solver to take
    # sub-ulp steps near radius 1.  Use log|w|, w=1-z, instead.  Matching at
    # |w|=1/4 fixes the same branch without a permutation of external legs.
    w_target = 1.0 - point
    if abs(w_target) < 0.125:
        w_direction = w_target / abs(w_target)
        match_radius = 0.25
        anchor = continue_spin7_spectator_block(
            affine_channel, 0.75,
            starting_radius=starting_radius,
            initial_series_order=initial_series_order,
            direct_series_radius=direct_series_radius,
            relative_tolerance=relative_tolerance,
            absolute_tolerance=absolute_tolerance,
        )
        angle = cmath.phase(w_target)
        def arc_equation(phi, vector):
            w = match_radius * cmath.exp(1j * phi)
            return 1j * ((_B_MATRIX - w / (1.0 - w) * _A_MATRIX) @ vector)
        arc = solve_ivp(arc_equation, (0.0, angle), anchor.as_array(),
                        method="DOP853", rtol=relative_tolerance,
                        atol=absolute_tolerance)
        if not arc.success:
            raise ArithmeticError(f"Spin(7) KZ matching failed: {arc.message}")
        initial_near = arc.y[:, -1]
        prior_steps = anchor.accepted_steps + arc.t.size
        prior_nfev = anchor.function_evaluations + arc.nfev

        def logarithmic_equation(log_radius, vector):
            w = math.exp(log_radius) * w_direction
            return (_B_MATRIX - w / (1.0 - w) * _A_MATRIX) @ vector

        solution = solve_ivp(
            logarithmic_equation,
            (math.log(match_radius), math.log(abs(w_target))), initial_near,
            method="DOP853", rtol=relative_tolerance, atol=absolute_tolerance,
        )
        if not solution.success or not np.isfinite(solution.y[:, -1]).all():
            raise ArithmeticError(f"Spin(7) KZ continuation failed: {solution.message}")
        return Spin7KZContinuation(
            channel=affine_channel, z=point,
            value=tuple(complex(entry) for entry in solution.y[:, -1]),
            used_ode=True, starting_radius=starting_radius,
            accepted_steps=int(prior_steps + solution.t.size),
            function_evaluations=int(prior_nfev + solution.nfev),
        )

    radius0 = min(float(starting_radius), 0.25 * modulus)
    direction = cmath.exp(1.0j * cmath.phase(point))
    start = radius0 * direction
    initial = local_series.value(start)

    def radial_equation(radius: float, vector: np.ndarray) -> np.ndarray:
        coordinate = radius * direction
        connection = _A_MATRIX / coordinate + _B_MATRIX / (coordinate - 1.0)
        return direction * (connection @ vector)

    solution = solve_ivp(
        radial_equation,
        (radius0, modulus),
        np.asarray(initial, dtype=np.complex128),
        method="DOP853",
        rtol=float(relative_tolerance),
        atol=float(absolute_tolerance),
    )
    if not solution.success:
        raise ArithmeticError(f"Spin(7) KZ continuation failed: {solution.message}")
    value = np.asarray(solution.y[:, -1], dtype=np.complex128)
    if np.any(~np.isfinite(value)):
        raise ArithmeticError("Spin(7) KZ continuation returned a non-finite value")
    return Spin7KZContinuation(
        channel=affine_channel,
        z=point,
        value=tuple(complex(entry) for entry in value),  # type: ignore[arg-type]
        used_ode=True,
        starting_radius=radius0,
        accepted_steps=int(solution.t.size),
        function_evaluations=int(solution.nfev),
    )


def continue_spin7_spectator_block_many(
    channel: Spin7AffineChannel,
    z_values: Sequence[complex],
    **options: object,
) -> np.ndarray:
    """Return a ``(number_of_points,4)`` array of continued block values."""

    rows = [
        continue_spin7_spectator_block(channel, z, **options).as_array()
        for z in z_values
    ]
    if not rows:
        return np.empty((0, 4), dtype=np.complex128)
    return np.stack(rows, axis=0)


__all__ = [
    "Spin7KZContinuation",
    "continue_spin7_spectator_block",
    "continue_spin7_spectator_block_many",
]
