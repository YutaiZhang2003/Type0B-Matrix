"""Exact polar changes of variables for the four-point boundary cell."""

import cmath
import math


def stratified_face_sample(radial, angular, collar_radius, stratum):
    r"""Map a square to one of four disjoint parts of the face cell.

    The cell is ``|z-1|<1, 0<Re(z)<1/2``. Its radial bands are
    ``(0,rho), (rho,4rho), (4rho,1/2), (1/2,1)``. The first uses uniform
    squared radius, and the other three use uniform log radius. The returned
    Jacobian integrates THIS band; equal sampling of four bands therefore
    requires either summing their means or multiplying each weight by four.
    """
    rho = float(collar_radius)
    if not math.isfinite(rho) or not 0 < rho < .125:
        raise ValueError("the stratified face map needs 0 < collar_radius < 1/8")
    if stratum not in (0, 1, 2, 3):
        raise ValueError("face stratum must be 0, 1, 2, or 3")
    u, v = float(radial), float(angular)
    if not 0 <= u <= 1 or not 0 <= v <= 1:
        raise ValueError("sample coordinates must lie in [0,1]")
    # Scrambled Sobol points have finite resolution. Endpoints have measure
    # zero; keep their images strictly inside the open integration cell.
    u = min(max(u, 2.0**-53), 1.0 - 2.0**-53)
    v = min(max(v, 2.0**-53), 1.0 - 2.0**-53)
    if stratum == 0:
        radius = rho * math.sqrt(u)
        radial_area = rho * rho / 2
    else:
        bounds = (rho, 4*rho, .5, 1.)
        lo, hi = bounds[stratum-1:stratum+1]
        log_ratio = math.log(hi/lo)
        radius = lo * math.exp(u * log_ratio)
        radial_area = radius * radius * log_ratio
    theta_max = math.acos(radius/2)
    theta_min = 0.0 if radius <= .5 else math.acos(.5/radius)
    width = theta_max - theta_min
    theta = (-theta_max + 2*v*width if v < .5
             else theta_min + (2*v-1)*width)
    return complex(radius*cmath.exp(1j*theta)), 2*width*radial_area
