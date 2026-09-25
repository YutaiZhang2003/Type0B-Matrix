"""Level-matched leading long-tube term from an existing recursive VV bank.

The tau1 integral and diagonal GSO projection are performed algebraically
before numerical evaluation.  This avoids subtracting exponentially large
NS and NS-tilde vacuum terms.  All short-edge coefficients remain recursive.
The retained long-edge term is an asymptotic approximation, with overlap
checks required; it is not an exact replacement at arbitrary torus height.
"""
import math
import numpy as np

from spin23_genus1_vv_boundaries import tube_propagator_integral
from spin23_genus1_vv_conventions import component_conversion


def _short_factors(bank, z, cutoff):
    """[node, PCO] coefficient of the massless closed-tube propagation.

    With t=exp(pi*i*tau), the NS vacuum prefactor is bar(t)^-1.
    GSO retains odd total t,bar(t) degree; tau1 level matching selects
    holomorphic degree zero, antiholomorphic degree one at leading order.
    The latter is supplied by either the right long-edge half level or
    23 spectator oscillators and the Szego numerator/denominator.
    """
    if hasattr(bank, 'short_factors'):
        return bank.short_factors(z, cutoff)
    if cutoff not in range(bank.metadata['cutoff']+1):
        raise ValueError('short-edge cutoff absent from bank')
    k = bank.ns_levels
    use = k.sum(axis=1) <= cutoff
    zero = use & (k[:, 1] == 0)
    half = use & (k[:, 1] == 1)
    logq = 2j*math.pi*z
    left = bank.ns[..., zero] @ np.exp(k[zero, 0]*logq/2)
    # Explicit slicing preserves [node,form,level] under NumPy indexing.
    right0 = bank.ns[:, 1][:, :, zero] @ np.exp(k[zero, 0]*logq.conjugate()/2)
    right1 = bank.ns[:, 1][:, :, half] @ np.exp((k[half, 0]-1)*logq.conjugate()/2)
    barred_free = 21+2*np.cos(2*math.pi*z.conjugate())
    return np.sum(bank.ns_weights*left*(right1+barred_free*right0)[:, None, :], axis=-1)


def _prefactor(z, omega):
    # Cylinder prime form and Szego kernel in period-one coordinates.
    sine = np.sin(math.pi*z)
    # Existing vertex order uses delta=z_0-z_1=-z.
    szego = -math.pi/sine
    hsum = 1+omega**2
    frame = np.array([np.exp((hsum+1)*np.log(2j*math.pi)+hsum*np.log(-2j*math.pi)),
                      np.exp(hsum*(np.log(2j*math.pi)+np.log(-2j*math.pi)))])
    wick = np.array([1., -omega**2*szego])
    # 1/16 is the original normalization before the factor 2 from NS-NS~.
    conversion=np.array([component_conversion((1,1)),component_conversion((0,0))])
    return conversion*frame*wick*szego.conjugate()*np.exp(2*omega**2*np.log(abs(sine/math.pi)))/(8*np.sqrt(8*math.pi**2))


def leading_cusp_density(bank, z, height, cutoff=8):
    """Tau1-averaged density at fixed rectangular z, retaining both PCO terms."""
    z = complex(z); height = float(height)
    if not 0 < z.imag < height or height <= 1:
        raise ValueError('puncture must be strictly inside the long tube')
    omega = complex(*bank.metadata['energy'])
    if max(z.imag, abs(z.real)) > 90:
        raise ValueError('use integrated tail for extended tubes; cylinder factors require |z|<90')
    coeff = _short_factors(bank, z, cutoff)
    gaussian = np.exp(-2*math.pi*(bank.momenta[:, 0]**2*z.imag+bank.momenta[:, 1]**2*(height-z.imag)))
    time_zero = np.exp(-2*math.pi*omega**2*z.imag**2/height)/math.sqrt(height)
    return _prefactor(z, omega)*time_zero*np.einsum('mw,m,m->w', coeff, gaussian, bank.weights)


def integrated_cusp_density(bank, z, start, cutoff=8):
    """Analytically integrate the leading tube height, keeping x and Im(z).

    The lower bound max(start,2*Im(z)) defines one half of the torus; the
    full z integral has an additional factor two.  Only imaginary external
    energy is supported here; continuation is a separate operation.
    """
    z = complex(z); omega = complex(*bank.metadata['energy'])
    if omega.real != 0 or not 0 < omega.imag < .5 or not 0 < z.imag < 90 or start <= 1:
        raise ValueError('Euclidean energy, 0<Im(z)<90, and tail start>1 required')
    lower = max(float(start), 2*z.imag)
    a = 2*math.pi*bank.momenta[:, 1]**2
    c = -2*math.pi*omega.imag**2*z.imag**2
    propagated = tube_propagator_integral(a, c, lower, shift=z.imag)
    propagated *= np.exp(-2*math.pi*bank.momenta[:, 0]**2*z.imag)
    coeff = _short_factors(bank, z, cutoff)
    return _prefactor(z, omega)*np.einsum('mw,m,m->w', coeff, propagated, bank.weights)
