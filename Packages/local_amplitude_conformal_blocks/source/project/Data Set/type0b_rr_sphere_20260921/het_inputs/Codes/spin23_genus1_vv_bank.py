"""Recursive, geometry-independent banks for the torus vector two-point function.

The two PCO words are GG and PP; the right Liouville word is always PP.
Only preparation imports recursion. The inherited artifact IO checks every
array hash and publishes atomically. The spectral measure is dP0 dP1 / pi**2.
"""
from itertools import product
import math
import time

import numpy as np

from spin23_genus1_coefficient_bank import CoefficientBank, _align
from spin23_genus1_banked_geometry import theta
from spin23_genus1_vv_conventions import component_conversion

SCHEMA = 'spin23-vv-recursive-bank-v1'
WORDS = ((1, 1), (0, 0))
FORMS = ((0, 0), (1, 1))
SIGNS = tuple(product((1, -1), repeat=2))


def levels(cutoff, step):
    return np.array([k for k in product(range(0, cutoff+1, step), repeat=2)
                     if sum(k) <= cutoff], dtype=np.int16)


def spectral_rule(order=4, p_max=4., power=1.25):
    if type(order) is not int or order < 1 or not np.isfinite(p_max) or p_max <= 0:
        raise ValueError('positive order and finite momentum cutoff required')
    if not np.isfinite(power) or power < 1:
        raise ValueError('finite endpoint power >= 1 required')
    edges = []
    for edge in range(2):
        x, w = np.polynomial.legendre.leggauss(order+edge)
        t = (x+1)/2
        exponent = power+edge*1e-6
        edges.append(tuple(zip(p_max*t**exponent,
            w/2*p_max*exponent*t**(exponent-1)/math.pi)))
    nodes = tuple(product(*edges))
    return (np.array([[p for p, _ in row] for row in nodes]),
            np.array([math.prod(w for _, w in row) for row in nodes]))


def reference_laguerre_rule(order=8, scale=math.pi):
    """Full-half-line fixed nodes; remove the reference Gaussian from weights.

    The actual geometry Gaussian remains in VVBank.evaluate. Thus these are
    reusable coefficient nodes, not geometry-dependent coefficient calls.
    """
    from scipy.special import roots_genlaguerre
    if type(order) is not int or order < 1 or not math.isfinite(scale) or scale <= 0:
        raise ValueError('positive order and Gaussian reference scale required')
    edges = []
    for n in (order, order+1):
        x, w = roots_genlaguerre(n, -.5)
        edges.append(tuple(zip(np.sqrt(x/scale),
            np.exp(np.log(w)+x)/(2*math.pi*math.sqrt(scale)))))
    nodes = tuple(product(*edges))
    return (np.array([[p for p, _ in row] for row in nodes]),
            np.array([math.prod(w for _, w in row) for row in nodes]))


def phase(forms, word, ramond=False):
    # Right external descendants vanish, so only the tensor crossing remains.
    crossing = forms[0]*((forms[1]+word[1]) % 2)
    return (-1)**crossing * (1 if ramond else (-1j)**sum(forms))


def prepare_node(momentum, omega, *, cutoff=8, precision=24, continuation=None):
    from spin23_genus1_recursive_sewing import ns_coefficients
    from spin23_genus1_branch_recursion import checked_ramond_tables
    from spin23_super_liouville_data import ns_structure_constants, rr_ns_structure_constants
    p = tuple(map(float, momentum)); omega = complex(omega)
    if len(p) != 2 or any(not math.isfinite(x) or x <= 0 for x in p):
        raise ValueError('two positive internal momenta required')
    if not np.isfinite(omega) or omega == 0:
        raise ValueError('a finite nonzero external energy required')
    if type(cutoff) is not int or cutoff < 0 or cutoff > 8 or cutoff % 2:
        raise ValueError('integer total level zero through four required')
    started = time.perf_counter(); external = (omega, omega)
    nl, rl = levels(cutoff, 1), levels(cutoff, 2)
    ns = np.zeros((2, 2, len(nl)), complex)
    ramond = np.zeros((2, 2, 4, len(rl)), complex)
    nw = np.zeros((2, 2), complex); rw = np.zeros((2, 2, 4), complex)
    nc = [ns_structure_constants(p[v-1], omega, p[v], precision=precision) for v in range(2)]
    rc = [rr_ns_structure_constants(p[v-1], p[v], omega, precision=precision) for v in range(2)]
    checks = []
    for wi, word in enumerate(WORDS):
        for fi, f in enumerate(FORMS):
            k, values = ns_coefficients(p, external, word, f, (cutoff,)*2, cutoff)
            ns[wi, fi] = _align(k, values, nl)
            nw[wi, fi] = phase(f, word)*math.prod(c[j] for c, j in zip(nc, f))
        k, values, forms, signs, check = checked_ramond_tables(
            internal_momenta=p, external_momenta=external, external_descendants=word,
            maximum_twice_levels=cutoff, maximum_total_twice_level=cutoff,
            **dict(precision_fallback=True, **(continuation or {})))
        checks.append(check)
        for fi, f in enumerate(FORMS):
            for si, s in enumerate(SIGNS):
                ramond[wi, fi, si] = _align(k, values[forms.index(f), signs.index(s)], rl)
                rw[wi, fi, si] = phase(f, word, True)*math.prod(
                    c[0 if sign == 1 else 1]/2 for c, sign in zip(rc, s))
    bank = VVBank(dict(schema=SCHEMA, channel='necklace',
        energy=[omega.real, omega.imag], cutoff=cutoff, precision=precision,
        ramond_checks=checks, preparation_seconds=time.perf_counter()-started),
        np.array([p]), np.ones(1), nl, rl, ns[None], ramond[None], nw[None], rw[None])
    bank.validate()
    return bank


class VVBank(CoefficientBank):
    """Two-edge specialization; reuse only the generic checked artifact IO."""

    def validate(self):
        if self.metadata.get('schema') != SCHEMA or self.metadata.get('channel') != 'necklace':
            raise ValueError('unsupported vector two-point bank')
        n = len(self.momenta)
        shapes = dict(momenta=(n, 2), weights=(n,), ns=(n, 2, 2, len(self.ns_levels)),
            r=(n, 2, 2, 4, len(self.r_levels)), ns_weights=(n, 2, 2), r_weights=(n, 2, 2, 4))
        for key, shape in shapes.items():
            array = getattr(self, key)
            if array.shape != shape or not np.isfinite(array).all():
                raise ValueError(f'invalid {key} array')
        if n == 0 or np.any(self.momenta <= 0) or np.any(self.weights <= 0):
            raise ValueError('positive spectral nodes and weights required')
        for name, step in (('ns_levels', 1), ('r_levels', 2)):
            if not np.array_equal(getattr(self, name), levels(self.metadata['cutoff'], step)):
                raise ValueError(f'incomplete {name}')

    def evaluate(self, plumbing, lifts, cutoffs=(6, 8), *, momentum_chunk=128):
        q = np.asarray(plumbing, complex); lifts = np.asarray(lifts, int)
        if q.ndim != 2 or q.shape[1] != 2 or lifts.shape != q.shape:
            raise ValueError('geometry-by-two plumbing and lift arrays required')
        if not np.isfinite(q).all() or np.any(abs(q) <= 0) or np.any(abs(q) >= 1):
            raise ValueError('necklace powers must be strictly convergent')
        if not np.isin(lifts, (-1, 1)).all() or momentum_chunk < 1:
            raise ValueError('invalid lifts or momentum chunk')
        if not cutoffs or any(type(c) is not int or c < 0 or c > self.metadata['cutoff'] for c in cutoffs):
            raise ValueError('requested level is absent')
        logs = np.log(q).T
        result = np.zeros((len(q), len(cutoffs), 3, 2), complex)
        ns_primary = np.exp(-np.log(abs(q)).sum(axis=1)/8)
        for li, cutoff in enumerate(cutoffs):
            for spin in range(3):
                is_r = spin == 2
                retained = self.r_levels if is_r else self.ns_levels
                mask = retained.sum(axis=1) <= cutoff
                powers = np.exp(retained[mask] @ logs/2)
                if not is_r:
                    edge_lifts = lifts.copy()
                    if spin == 1: edge_lifts[:, -1] *= -1
                    powers *= np.prod(edge_lifts[None, :, :]**retained[mask, None, :], axis=-1)
                coefficients = self.r if is_r else self.ns
                coupling = self.r_weights if is_r else self.ns_weights
                for start in range(0, len(self.momenta), momentum_chunk):
                    sl = slice(start, start+momentum_chunk)
                    c = coefficients[sl][..., mask]
                    left = c @ powers
                    right = c[:, 1:2] @ powers.conj()
                    products = coupling[sl][..., None]*left*right
                    products = products.sum(axis=tuple(range(2, products.ndim-1)))
                    gaussian = np.exp(self.momenta[sl]**2 @ np.log(abs(q)).T)
                    result[:, li, spin] += np.einsum('mwg,mg,m->gw', products, gaussian, self.weights[sl])
                if not is_r: result[:, li, spin] *= ns_primary[:, None]
        if not np.isfinite(result).all():
            raise ArithmeticError('non-finite spectral integral')
        return result


def merge_nodes(banks, momenta, weights, metadata):
    if not banks or len(banks) != len(momenta) or len(weights) != len(momenta):
        raise ValueError('incomplete spectral bank')
    first = banks[0]
    for bank, p in zip(banks, momenta):
        bank.validate()
        if not np.array_equal(bank.momenta, [p]) or any(bank.metadata[k] != first.metadata[k]
                for k in ('energy', 'cutoff', 'precision', 'channel', 'schema')):
            raise ValueError('incompatible or misordered nodes')
    arrays = {k: np.concatenate([getattr(b, k) for b in banks])
              for k in ('ns', 'r', 'ns_weights', 'r_weights')}
    result = VVBank(dict(first.metadata, **metadata), np.asarray(momenta), np.asarray(weights),
                    first.ns_levels, first.r_levels, **arrays)
    result.validate()
    return result


class VVGeometry:
    """Shared theta functions, determinants, trace frame, and graded PCO signs."""
    def __init__(self, tau, points):
        self.tau = tau = np.asarray(tau, complex); z = np.asarray(points, complex)
        if tau.ndim != 1 or z.shape != (len(tau), 2) or not len(tau):
            raise ValueError('nonempty tori and two-point batch required')
        if not np.isfinite(tau).all() or not np.isfinite(z).all() or np.any(tau.imag < .15):
            raise ValueError('finite tori with Im(tau)>=0.15 required')
        increments = np.column_stack((z[:, 1]-z[:, 0], tau+z[:, 0]-z[:, 1]))
        if np.any(increments.imag <= 0):
            raise ValueError('strictly ordered punctures required')
        self.plumbing = np.exp(2j*math.pi*increments)
        windings = np.rint((2*math.pi*increments.real-np.angle(self.plumbing))/(2*math.pi)).astype(int)
        self.lifts = ((-1.)**windings).astype(int)
        q = np.exp(2j*math.pi*tau)
        count = int(np.ceil(20*math.log(10.)/(-np.log(abs(q))).min()))+3
        eta = np.exp(1j*math.pi*tau/12)*np.prod(1-q[None, :]**np.arange(1, count+1)[:, None], axis=0)
        delta = z[:, 0]-z[:, 1]
        odd = theta(.5, .5, delta, tau)
        derivative = theta(.5, .5, 0., tau, derivative=1)
        if np.any(abs(odd) == 0): raise ArithmeticError('singular prime form')
        self.green = -2*np.log(abs(odd/derivative))+2*math.pi*delta.imag**2/tau.imag
        kernels = []; common = []
        for a, b in ((0., 0.), (0., .5), (.5, 0.)):
            constant = theta(a, b, 0., tau)
            kernel = theta(a, b, delta, tau)*derivative/(constant*odd)
            ratio = constant/eta
            time_partition = np.exp(.5*np.log(ratio))
            spectator = np.exp(11.5*np.log(ratio))*kernel
            ghost = (1 if a == b == 0 else -1)/ratio
            boson = 1/(np.sqrt(8*math.pi**2*tau.imag)*abs(eta)**2)
            common.append(.25*time_partition*spectator.conj()*ghost*.5*abs(eta)**4*boson)
            kernels.append(kernel)
        self.common = np.stack(common, axis=1)
        self.szego = np.stack(kernels, axis=1)

    def multipliers(self, omega):
        omega = complex(omega); hsum = 1+omega**2
        result = np.empty((len(self.tau), 3, 2), complex)
        oscillator = np.exp(-omega**2*self.green)
        for ci, word in enumerate(WORDS):
            frame = np.exp((hsum+sum(word)/2)*np.log(2j*math.pi)+hsum*np.log(-2j*math.pi))
            # i**2 from string convention, 1/2 GSO, and -1 from moving the
            # first spectator past the second holomorphic odd field.
            wick = 1 if ci == 0 else -omega**2*self.szego
            # Saved chiral tables use fixed-parity components; the physical
            # sphere GG/PP convention requires an additional GG minus sign.
            result[:, :, ci] = component_conversion(word)*.5*frame*self.common*oscillator[:, None]*wick
        if not np.isfinite(result).all(): raise ArithmeticError('non-finite free fields')
        return result


def evaluate_density(bank, geometry, cutoffs=(6, 8)):
    omega = complex(*bank.metadata['energy'])
    return bank.evaluate(geometry.plumbing, geometry.lifts, cutoffs)*geometry.multipliers(omega)[:, None]
