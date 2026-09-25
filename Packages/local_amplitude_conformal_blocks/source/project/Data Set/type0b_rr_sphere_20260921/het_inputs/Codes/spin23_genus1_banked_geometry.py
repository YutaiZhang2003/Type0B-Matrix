"""Batched binary64 free-field factors in the corrected SVV trace convention.

The defining theta sums are the same as Codes/spin23_genus1_free_fields.py.
They are shared across energies and levels, with no mpmath or block evaluation
in this module. Use the original high-precision assembler as an independent
audit. This implementation is restricted to the compact necklace chart.
"""
from dataclasses import dataclass
import math
import numpy as np

from spin23_genus1_svv_operator_order import grouping_sign

PAIRS = ((0, 1), (0, 2), (1, 2))
SELECTED = ((),) + PAIRS


def theta(a, b, z, tau, derivative=0):
    z, tau = np.broadcast_arrays(np.asarray(z, complex), np.asarray(tau, complex))
    # Conservative absolute tail target including growth away from the real axis.
    target = 20*math.log(10.)
    cutoff = int(np.ceil(np.max(abs(z.imag)/tau.imag + abs(a) + np.sqrt(
        (target+math.pi*z.imag**2/tau.imag)/(math.pi*tau.imag)))))+3
    shifted = (np.arange(-cutoff, cutoff+1)+a).reshape((-1,)+(1,)*z.ndim)
    terms = np.exp(1j*math.pi*shifted**2*tau+2j*math.pi*shifted*(z+b))
    if derivative: terms *= (2j*math.pi*shifted)**derivative
    return terms.sum(axis=0)


@dataclass
class GeometryBatch:
    tau: np.ndarray
    points: np.ndarray
    plumbing: np.ndarray
    lifts: np.ndarray
    greens: np.ndarray
    szego: np.ndarray
    common: np.ndarray

    @classmethod
    def build(cls, tau, points):
        tau = np.asarray(tau, complex); z = np.asarray(points, complex)
        if tau.ndim != 1 or z.shape != (len(tau), 3) or not len(tau):
            raise ValueError('a nonempty batch of tori and three points is required')
        if not np.isfinite(tau).all() or not np.isfinite(z).all() or np.any(tau.imag < .15):
            raise ValueError('finite tori with Im(tau)>=0.15 required for this evaluator')
        increments = np.column_stack((z[:, 1]-z[:, 0], z[:, 2]-z[:, 1], tau+z[:, 0]-z[:, 2]))
        if np.any(increments.imag <= 0):
            raise ValueError('points must be strictly ordered in the necklace strip')
        plumbing = np.exp(2j*math.pi*increments)
        windings = np.rint((2*math.pi*increments.real-np.angle(plumbing))/(2*math.pi)).astype(int)
        lifts = (-1.)**windings
        q = np.exp(2j*math.pi*tau)
        eta_cut = int(np.ceil(20*math.log(10.)/(-np.log(abs(q))).min()))+3
        eta = np.exp(1j*math.pi*tau/12)*np.prod(1-q[None, :]**np.arange(1, eta_cut+1)[:, None], axis=0)
        differences = np.column_stack([z[:, i]-z[:, j] for i, j in PAIRS])
        odd = theta(.5, .5, differences, tau[:, None])
        derivative = theta(.5, .5, 0., tau, derivative=1)
        if np.any(abs(odd) == 0) or np.any(abs(derivative) == 0):
            raise ArithmeticError('singular prime form')
        greens = -2*np.log(abs(odd/derivative[:, None])) + 2*math.pi*differences.imag**2/tau.imag[:, None]
        kernels=[]; common=[]
        for a, b in ((0., 0.), (0., .5), (.5, 0.)):
            constant = theta(a, b, 0., tau)
            kernel = theta(a, b, differences, tau[:, None])*derivative[:, None]/(constant[:, None]*odd)
            ratio = constant/eta
            # Preserve principal branches of each determinant separately.
            time_partition = np.exp(.5*np.log(ratio))
            spectator = np.exp(11.5*np.log(ratio))*kernel[:, 2]
            ghost = (1 if a == b == 0 else -1)/ratio
            boson_partition = 1/(np.sqrt(8*math.pi**2*tau.imag)*abs(eta)**2)
            common.append(.125*time_partition*spectator.conj()*ghost*.5*abs(eta)**4*boson_partition)
            kernels.append(kernel)
        return cls(tau, z, plumbing, lifts.astype(int), greens,
                   np.stack(kernels, axis=1), np.stack(common, axis=1))

    def multipliers(self, energy):
        """[geometry, spin, PCO], including string phase and GSO exactly once."""
        p, q = map(float, energy); external = np.array([p+q, p, q]); k=external.copy(); k[0]*=-1
        if not np.isfinite(external).all() or min(p, q)<=0:
            raise ValueError('positive real outgoing energies required')
        oscillator = np.exp(self.greens @ np.array([k[i]*k[j] for i, j in PAIRS]))
        result = np.empty((len(self.tau), 3, 4), complex)
        hsum = ((1+external**2)/2).sum()
        for ci, selected in enumerate(SELECTED):
            frame = np.exp((hsum+(3-len(selected))/2)*np.log(2j*math.pi)
                           +(hsum+.5)*np.log(-2j*math.pi))
            wick = 1. if ci == 0 else self.szego[:, :, ci-1]
            result[:, :, ci] = (-.5j*grouping_sign(selected)*math.prod(k[i] for i in selected)*frame
                                *self.common*oscillator[:, None]*wick)
        if not np.isfinite(result).all():
            raise ArithmeticError('non-finite free-field assembly')
        return result


def evaluate_density(bank, geometry, cutoffs=(6, 8), *, momentum_chunk=128):
    values = bank.evaluate(geometry.plumbing, geometry.lifts, cutoffs,
                           momentum_chunk=momentum_chunk)
    return values*geometry.multipliers(bank.metadata['energy'])[:, None, :, :]
