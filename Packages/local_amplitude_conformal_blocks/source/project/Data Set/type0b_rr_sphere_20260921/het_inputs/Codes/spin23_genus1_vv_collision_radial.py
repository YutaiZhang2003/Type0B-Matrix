"""First angularly averaged regular correction to the collision disk.

The time zero-mode factor is exp(-2*pi*omega**2*Im(z)**2/tau2).
Its angular mean is 1-pi*omega**2*r**2/tau2+O(r**4).
Linear one-point descendants vanish by torus translation invariance;
purely holomorphic/antiholomorphic quadratic terms have zero angular mean.
This coefficient is fixed, never determined from radius matching.
"""
import math
import numpy as np
from scipy.special import erfcx


def correction_radial_weights(momenta, radius):
    if not 0<radius<.5:
        raise ValueError('a local positive disk radius required')
    h=(1+np.asarray(momenta)**2)/2
    return np.column_stack((math.pi*(1-h)/h*radius**(2*h),
                            math.pi*radius**(2*h+1)))


def inverse_height_propagator(a, lower):
    """Integral_lower^infinity y^-3/2 exp(-a*y) dy, including a=0."""
    a=np.asarray(a,float)
    if not np.isfinite(a).all() or np.any(a<0) or not math.isfinite(lower) or lower<=0:
        raise ValueError('nonnegative finite a and a positive lower height required')
    x=np.sqrt(a*lower)
    return 2*np.exp(-x*x)/math.sqrt(lower)*(1-math.sqrt(math.pi)*x*erfcx(x))


def disk_correction(bank, tau, radius, cutoff=None):
    """All spin structures and both leading bridge families, compact torus."""
    omega=complex(*bank.metadata['energy'])
    if cutoff is None: cutoff=bank.metadata['cutoff']
    radial=correction_radial_weights(bank.momenta[:,0],radius)
    return -math.pi*omega**2/complex(tau).imag*np.einsum(
        'msf,mf,m->sf',bank.coefficients_at(tau,cutoff),radial,bank.weights)


def cusp_disk_correction(bank, start, radius):
    """The same angular correction with the long height integrated exactly."""
    omega=complex(*bank.metadata['energy'])
    radial=correction_radial_weights(bank.momenta[:,0],radius)
    prop=inverse_height_propagator(2*math.pi*bank.momenta[:,1]**2,start)
    return -math.pi*omega**2*np.einsum('mf,mf,m,m->f',bank.coefficients,
        radial,prop,bank.weights)/(8*np.sqrt(8*math.pi**2))
