"""Local collision subtraction and exact sampling maps for the VV torus.

Coordinates have period one.  These utilities distinguish an analytic OPE
finite part from a completed BRST amplitude.  In particular the latter also
requires the picture-changing descent term on every moduli-space boundary.
"""
from dataclasses import dataclass
import math

import numpy as np
from scipy.special import erfc, erfcx


def nearest_lattice_image(z, tau):
    """Shortest displacement from a torus puncture, including lattice images.

    Valid for the modular fundamental domain (including its T translates).
    A radial exclusion is different from excluding an entire necklace strip.
    """
    z, tau = np.broadcast_arrays(np.asarray(z, complex), np.asarray(tau, complex))
    if np.any(tau.imag <= 0) or not np.isfinite(z).all() or not np.isfinite(tau).all():
        raise ValueError('finite punctures and upper-half-plane tori required')
    b = np.rint(z.imag/tau.imag)
    candidates = []
    for shift in (-1, 0, 1):
        w = z-(b+shift)*tau
        candidates.append(w-np.rint(w.real))
    candidates = np.stack(candidates)
    return np.take_along_axis(candidates, np.argmin(abs(candidates), axis=0)[None], axis=0)[0]


def primary_disc(h, radius):
    r"""Analytic finite part of integral (1-h)|z|^(2h-4) on a disk.

    Continue the *product* from Re(h)>1; do not divide two floating-point
    quantities that vanish at h=1.  The h=1 value is the continued contact
    limit, rather than the pointwise integral of a zero integrand.
    """
    h = np.asarray(h, complex)
    if not math.isfinite(radius) or radius <= 0 or not np.isfinite(h).all():
        raise ValueError('finite weights and positive disk radius required')
    return -math.pi*np.exp((2*h-2)*math.log(radius))


def primary_annulus(h, inner, outer):
    """Ordinary finite annulus integral; stable also when h is close to one."""
    h = np.asarray(h, complex)
    if not 0 < inner < outer or not math.isfinite(outer):
        raise ValueError('ordered positive finite radii required')
    power = 2*h-2
    return -math.pi*np.exp(power*math.log(inner))*np.expm1(power*math.log(outer/inner))


def primary_inner_boundary(h, radius):
    """Inner-boundary term that cancels the primary power divergence.

    The local NS superdistance-cutoff derivation is implemented separately
    in superdistance_boundary. This is the same term as the restored even
    analytic disk, so it must not be added a second time. It does not by
    itself certify the full global BRST/gluing prescription.
    """
    from spin23_genus1_vv_superboundary import superdistance_boundary
    return superdistance_boundary(h, radius)


@dataclass(frozen=True)
class CollisionPictureAssignment:
    """NS separating degeneration: (g,n)=(0,3) joined to (1,1)."""
    sphere_pcos: int = 1
    torus_pcos: int = 1

    def validate(self):
        if (self.sphere_pcos, self.torus_pcos) != (1, 1):
            raise ValueError('NS sewing requires one PCO on each component')

    def positions(self, separation, torus_position, alpha=1.):
        self.validate()
        if not np.isfinite([separation, torus_position, alpha]).all() or torus_position == 0:
            raise ValueError('finite positions and a nonzero torus PCO position required')
        return complex(alpha*separation), complex(torus_position)


def tail_height_and_jacobian(unit, start, exponent=2.):
    """Exact [start,infinity) map, as in the c=1 three-point sampler.

    The exponent chooses a sampling density, never a fitted physical tail.
    """
    u = np.asarray(unit, float)
    if not np.isfinite(u).all() or np.any((u < 0) | (u >= 1)):
        raise ValueError('tail coordinates must be in [0,1)')
    if not math.isfinite(start) or start <= 1 or not math.isfinite(exponent) or exponent <= 1:
        raise ValueError('finite tail start and proposal exponent greater than one required')
    a = exponent-1
    y = start*np.exp(-np.log1p(-u)/a)
    jac = y/(a*(1-u))
    if not np.isfinite(y).all() or not np.isfinite(jac).all():
        raise ArithmeticError('tail map exceeds floating-point range')
    return y, jac


def tube_propagator_integral(a, c, lower, *, shift=0.):
    r"""Exact integral int_lower^infty dy y^-1/2 exp[-a(y-shift)-c/y].

    a=2*pi*P_long^2 and c=-2*pi*kappa^2*Im(z)^2 in the Euclidean VV
    calculation.  Combining exponentials before exponentiation prevents
    the individual exp(+a*shift) and erfc factors from overflowing.
    """
    a, c, lower, shift = np.broadcast_arrays(*map(lambda x: np.asarray(x, float), (a, c, lower, shift)))
    if not all(np.isfinite(x).all() for x in (a, c, lower, shift)) or np.any(a <= 0) or np.any(lower <= 0) or np.any(shift > lower):
        raise ValueError('a>0, real c, lower>0, shift<=lower required')
    if np.any(c < 0):
        # Analytic continuation in c is entire for a positive lower bound.
        # For c<0 the erfcx terms are conjugates; no numerical cancellation.
        out = np.empty(a.shape)
        negative = c < 0
        aa = np.sqrt(a[negative]*lower[negative])
        cc = np.sqrt(-c[negative]/lower[negative])
        out[negative] = np.sqrt(math.pi/a[negative])*np.exp(
            -a[negative]*(lower[negative]-shift[negative])-c[negative]/lower[negative])*erfcx(aa+1j*cc).real
        if np.any(~negative):
            out[~negative] = tube_propagator_integral(a[~negative],c[~negative],lower[~negative],shift=shift[~negative])
        return out
    aa = np.sqrt(a*lower); cc = np.sqrt(c/lower)
    minus = aa-cc; plus = aa+cc
    log_common = -a*(lower-shift)-c/lower
    first = np.empty(a.shape)
    positive = minus >= 0
    first[positive] = np.exp(log_common[positive])*erfcx(minus[positive])
    first[~positive] = np.exp((a*shift-2*np.sqrt(a*c))[~positive])*erfc(minus[~positive])
    second = np.exp(log_common)*erfcx(plus)
    return np.sqrt(math.pi)/(2*np.sqrt(a))*(first+second)


def validation_status(*, spectral=False, collision_overlap=False, pco_descent=False,
                      cusp_overlap=False, continuation=False, integrated_matching=False,
                      target='euclidean'):
    """Keep mathematical subtraction separate from physical certification."""
    if target not in ('euclidean','real'):
        raise ValueError('choose the Euclidean or real-energy target')
    checks = dict(spectral=spectral, collision_overlap=collision_overlap,
                  pco_descent=pco_descent, cusp_overlap=cusp_overlap,
                  integrated_matching=integrated_matching)
    if target=='real': checks['continuation']=continuation
    if any(type(v) is not bool for v in checks.values()):
        raise ValueError('Boolean validation records required')
    return dict(target=target, checks=checks, physical_amplitude_certified=all(checks.values()))
