"""Leading even and odd NS collision families, with recursive handle banks.

Both PCO words are included through their sphere Ward coefficients.  The
odd family is locally integrable after momentum integration but is retained:
its disk integral can converge only logarithmically as the radius shrinks.
Higher bridge descendants and global PCO boundary terms still require an
overlap/BRST audit before this local approximation defines an amplitude.
"""
import math
import time
import numpy as np

from spin23_genus1_coefficient_bank import CoefficientBank
from spin23_genus1_banked_geometry import theta
from spin23_genus1_vv_boundaries import primary_disc, tube_propagator_integral
from spin23_genus1_vv_conventions import CONVENTION, collision_ward_coefficients

SCHEMA = 'spin23-vv-ope-leading-families-v1'


def prepare_ope_node(momentum, omega, *, cutoff=8, precision=24):
    from spin23_genus1_onepoint_recursion import ns_onepoint, ramond_onepoint
    from spin23_super_liouville_data import ns_structure_constants, rr_ns_structure_constants
    bridge, loop = map(float, momentum); omega = complex(omega)
    if min(bridge, loop) <= 0 or not np.isfinite([bridge, loop, omega]).all():
        raise ValueError('positive internal momenta and finite external energy required')
    if cutoff not in (0, 2, 4, 6, 8):
        raise ValueError('integer handle levels through four required')
    started = time.perf_counter()
    ns, nd = ns_onepoint(loop, bridge, cutoff)
    r, rd = ramond_onepoint(loop, bridge, cutoff)
    sphere = np.array(ns_structure_constants(omega, omega, bridge, precision=precision))
    handle = np.array(ns_structure_constants(loop, bridge, loop, precision=precision))
    rhandle = np.array(rr_ns_structure_constants(loop, loop, bridge, precision=precision))
    norm = np.array([1., (1+bridge**2)**2])
    nw = sphere*handle/norm
    # Ordinary-R one-point graded pairing: 1 for P/P and +i for G/G.
    rw = sphere[:, None]/norm[:, None]*np.array([1., 1j])[:, None]*rhandle[None]/2
    bank = OPEBank(dict(schema=SCHEMA, channel='leading-ope', energy=[omega.real, omega.imag],
        physical_component_convention=CONVENTION,
        cutoff=cutoff, precision=precision, ns_checks=nd, ramond_checks=rd,
        preparation_seconds=time.perf_counter()-started), np.array([[bridge, loop]]),
        np.ones(1), np.arange(cutoff+1), np.arange(0, cutoff+1, 2),
        ns[None], r[None], nw[None], rw[None])
    bank.validate()
    return bank


class OPEBank(CoefficientBank):
    """Reuse hash-checked atomic artifact IO, with its own array schema."""
    def validate(self):
        if self.metadata.get('schema') != SCHEMA or self.metadata.get('channel') != 'leading-ope':
            raise ValueError('unsupported collision bank')
        n = len(self.momenta); c = self.metadata['cutoff']
        shapes = dict(momenta=(n, 2), weights=(n,), ns=(n, 2, c+1),
            r=(n, 2, 2, c//2+1), ns_weights=(n, 2), r_weights=(n, 2, 2))
        for name, shape in shapes.items():
            v = getattr(self, name)
            if v.shape != shape or not np.isfinite(v).all():
                raise ValueError(f'invalid collision-bank {name}')
        if not n or np.any(self.momenta <= 0) or np.any(self.weights <= 0):
            raise ValueError('positive spectral nodes and weights required')
        if not np.array_equal(self.ns_levels, np.arange(c+1)) or not np.array_equal(self.r_levels, np.arange(0, c+1, 2)):
            raise ValueError('incomplete collision handle levels')

    def coefficients_at(self, tau, cutoff=8):
        """K(P_bridge,P_loop,tau) by spin and collision family, before r powers."""
        tau = complex(tau)
        if not 0 <= cutoff <= self.metadata['cutoff'] or not .15 <= tau.imag <= 8:
            raise ValueError('retained handle level and compact/moderate torus required')
        logq = 2j*math.pi*tau; q = np.exp(logq)
        eta = np.exp(logq/24)*np.prod(1-q**np.arange(1, 65))
        bridge, loop = self.momenta.T
        h = (1+bridge**2)/2
        frame = np.exp((2*h[:, None]+np.arange(2)[None])*math.log(2*math.pi))
        result = np.empty((len(bridge), 3, 2), complex)
        for spin, (a, b) in enumerate(((0., 0.), (0., .5), (.5, 0.))):
            ratio = complex(theta(a, b, 0., tau))/eta
            # There are TWO ordered local poles. For x=exp(2*pi*i*z),
            # (2*pi*i)/(1-x) ~ -1/z; the spectator pole is -1/bar(z).
            # Their product has positive sign. Counting only the spectator
            # minus would double-count a relative coordinate orientation.
            common = (1 if spin == 0 else -1)*np.exp(-.5*np.log(ratio)+11.5*np.log(ratio).conjugate())*abs(eta)**2/(16*np.sqrt(8*math.pi**2*tau.imag))
            if spin < 2:
                k = self.ns_levels[self.ns_levels <= cutoff]
                powers = np.exp(k*logq/2)*(1 if spin == 0 else (-1.)**k)
                left = self.ns[..., :len(k)] @ powers
                right = self.ns[..., :len(k)] @ powers.conj()
                value = self.ns_weights*left*right
                value *= np.exp(logq.real*(loop**2-1/8))[:, None]
            else:
                k = self.r_levels[self.r_levels <= cutoff]
                powers = np.exp(k*logq/2)
                left = self.r[..., :len(k)] @ powers
                right = self.r[..., :len(k)] @ powers.conj()
                value = np.sum(self.r_weights*left*right, axis=-1)
                value *= np.exp(logq.real*loop**2)[:, None]
            result[:, spin] = common*frame*value
        if not np.isfinite(result).all():
            raise ArithmeticError('non-finite OPE coefficients')
        return result

    def density(self, z, tau, cutoff=8):
        """Leading collision density [spin,family], including the PCO sum."""
        return self.density_components(z,tau,cutoff).sum(axis=-1)

    def density_components(self,z,tau,cutoff=8):
        """Keep [spin,family,PCO] separate for physical overlap checks."""
        radius = abs(z)
        if not 0 < radius < .5:
            raise ValueError('nonzero local collision coordinate required')
        h = (1+self.momenta[:, 0]**2)/2
        radial = np.column_stack((radius**(2*h-4),radius**(2*h-3)))
        ward=collision_ward_coefficients(complex(*self.metadata['energy']),h)
        return np.einsum('msf,mf,mfc,m->sfc',self.coefficients_at(tau,cutoff),radial,ward,self.weights)

    def disc(self, tau, radius, cutoff=8):
        """Analytic even finite part plus the ordinary integrated odd family."""
        if not 0 < radius < .5:
            raise ValueError('a local disk radius in (0,.5) required')
        h = (1+self.momenta[:, 0]**2)/2
        radial = np.column_stack((primary_disc(h, radius),
            math.pi*(h+.5)/(h-.5)*np.exp((2*h-1)*math.log(radius))))
        return np.einsum('msf,mf,m->sf', self.coefficients_at(tau, cutoff), radial, self.weights)

    def disc_tail(self, start, radius, cutoff=8):
        """Leading level-matched, height-integrated collision disk in the cusp."""
        if start <= 1 or not 0 < radius < .5 or not 1 <= cutoff <= self.metadata['cutoff']:
            raise ValueError('tail start>1, local radius and retained half-level required')
        bridge, loop = self.momenta.T; h = (1+bridge**2)/2
        # No spectator pole remains after angular integration: its oscillator
        # contribution is 23, in place of 21+2*cos(2*pi*z) in the bulk tail.
        coeff = self.ns_weights*self.ns[:, :, 0]*(self.ns[:, :, 1]+23*self.ns[:, :, 0])
        coeff *= np.exp((2*h[:, None]+np.arange(2)[None])*math.log(2*math.pi))
        radial = np.column_stack((primary_disc(h, radius),
            math.pi*(h+.5)/(h-.5)*np.exp((2*h-1)*math.log(radius))))
        propagated = tube_propagator_integral(2*math.pi*loop**2, 0., start)
        return np.einsum('mf,mf,m,m->f', coeff, radial, propagated, self.weights)/(8*np.sqrt(8*math.pi**2))


def merge_ope_nodes(banks, momenta, weights, metadata=None):
    if not banks or len(banks) != len(momenta) or len(weights) != len(momenta):
        raise ValueError('all collision nodes are required')
    first = banks[0]
    for bank, p in zip(banks, momenta):
        bank.validate()
        if not np.array_equal(bank.momenta, [p]) or any(bank.metadata[k] != first.metadata[k]
                for k in ('schema', 'energy', 'cutoff', 'precision')):
            raise ValueError('incompatible collision nodes')
    arrays = {k: np.concatenate([getattr(b, k) for b in banks])
              for k in ('ns', 'r', 'ns_weights', 'r_weights')}
    out = OPEBank(dict(first.metadata, **(metadata or {})), np.asarray(momenta), np.asarray(weights),
                  first.ns_levels, first.r_levels, **arrays)
    out.validate()
    return out
