"""Direct exterior-domain audits for the SO(7) sphere amplitudes.

The change of integration variable z=1/u has Jacobian |u|^-4. Evaluating
the original correlator at 1/u retains all external labels and pictures;
it requires no inferred permutation/cocycle matrix. Finite block order and
spectral quadrature still require independent convergence checks.
"""
from functools import lru_cache
import cmath
import math
import numpy as np
from scipy.integrate import solve_ivp

from so7e8_four_ramond_assembly import (
    SPIN7_KZ_PAIR_0Z, SPIN7_KZ_PAIR_Z1, spin7_spectator_block_series,
)
from so7e8_spin7_kz_continuation import continue_spin7_spectator_block


def spin7_global_value(channel, z, *, rtol=2e-12, atol=2e-13):
    """Continue the fixed-channel KZ block along a radial, cut-free path.

    Logarithmic radius is used to resolve both small and large |z|. Exact
    real points beyond 1 are rejected because they require a specified lip.
    No target on the integration grids lies on that cut.
    """
    point = complex(z)
    if not np.isfinite(point) or point in (0, 1):
        raise ValueError("z must be finite and avoid the punctures")
    if point.imag == 0 and point.real >= 1:
        raise ValueError("a point on the exterior real cut needs a lip")
    radius = abs(point)
    if radius < 1:
        return continue_spin7_spectator_block(
            channel, point, relative_tolerance=rtol,
            absolute_tolerance=atol).as_array()
    local = spin7_spectator_block_series(channel, maximum_order=12)
    direction = cmath.exp(1j*cmath.phase(point))
    start = 1e-5
    initial = local.value(start*direction)
    a = np.asarray(SPIN7_KZ_PAIR_0Z, complex)/6
    b = np.asarray(SPIN7_KZ_PAIR_Z1, complex)/6

    if abs(1-point) < .125:
        w_target = 1-point
        w_direction = w_target/abs(w_target)
        anchor = continue_spin7_spectator_block(
            channel, .75, relative_tolerance=rtol,
            absolute_tolerance=atol).as_array()
        def arc_rhs(phi, y):
            w = .25 * cmath.exp(1j*phi)
            return 1j*((b-w/(1-w)*a) @ y)
        arc = solve_ivp(arc_rhs, (0, cmath.phase(w_target)), anchor,
                        method="DOP853", rtol=rtol, atol=atol)
        if not arc.success:
            raise ArithmeticError("global Spin(7) matching failed: " + arc.message)
        def near_rhs(log_radius, y):
            w = math.exp(log_radius)*w_direction
            return (b-w/(1-w)*a) @ y
        sol = solve_ivp(near_rhs, (math.log(.25), math.log(abs(w_target))),
                        arc.y[:, -1], method="DOP853", rtol=rtol, atol=atol)
        if not sol.success or not np.isfinite(sol.y[:, -1]).all():
            raise ArithmeticError("global Spin(7) near-one continuation failed: " + sol.message)
        return sol.y[:, -1]

    def rhs(log_radius, y):
        coordinate = math.exp(log_radius)*direction
        return (a + coordinate/(coordinate-1)*b) @ y

    solution = solve_ivp(rhs, (math.log(start), math.log(radius)), initial,
                         method="DOP853", rtol=rtol, atol=atol)
    if not solution.success or not np.isfinite(solution.y[:, -1]).all():
        raise ArithmeticError("global Spin(7) KZ continuation failed: " + solution.message)
    return solution.y[:, -1]


@lru_cache(maxsize=24)
def spin7_global_grid(coordinates):
    """Geometry-only spectator arrays, reusable across energies and P."""
    result = {channel: np.asarray([spin7_global_value(channel, z) for z in coordinates])
              for channel in ("vacuum", "vector")}
    for values in result.values():
        values.setflags(write=False)
    return result


def exterior_grid(z, weights):
    points, areas = np.asarray(z, complex), np.asarray(weights, float)
    if np.any(points == 0):
        raise ValueError("inversion excludes the origin")
    return 1/points, areas/abs(points)**4
