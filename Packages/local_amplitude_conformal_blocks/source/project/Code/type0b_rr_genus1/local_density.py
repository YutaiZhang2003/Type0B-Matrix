"""Local RR OPE density in a contractible puncture patch.

The four theta sectors, BRY spin/disorder states, both NS bridge families,
Liouville couplings, picture change, and free determinants are assembled.
The result uses the canonical ordered G_h G_a convention of the RR Ward
engine. Its conversion to a BRY S-matrix coefficient is deliberately not
performed here. This local expansion is not a global integration callback.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import product
from pathlib import Path
import sys

import mpmath as mp
import numpy as np

from ope_channel import CODE, K, TorusRROPEBlock
from physical_components import GROUND_KETS
from picture_changed import free_outer_factor, raised_combination

sys.path.insert(0, str(CODE/'c_Recursion'))
from super_liouville_structure_constants import (ns_structure_constant,
    ns_tilde_structure_constant, rr_ns_structure_constants)


THETAS = (1, 2, 3, 4)
ETAS = (1, -1)
GROUNDS = tuple(product((0, 1), repeat=2))
EMBEDDING = np.asarray(GROUND_KETS, complex)


def disorder_sign(delta):
    """<mu mu>_delta / <sigma sigma>_delta for a shrinking disorder line.

    Di Francesco--Saleur--Zuber, NPB290 (1987), section 5.6, (5.40--42).
    Winding a puncture changes the line and must also transport this sign.
    """
    if delta not in THETAS:
        raise ValueError('theta characteristic must be 1, 2, 3, or 4')
    return -1 if delta == 1 else 1


def sphere_pairing(family, *, raised):
    """Full NSRR vertex cocycle times the inverse NS BPZ metric sign.

    Axes: physical R family, holomorphic ground pair, anti ground pair.
    Both external modes of a raised component are odd, including G0.
    The inverse norm's magnitudes are already in the chiral OPE blocks.
    """
    if family not in (0, 1):
        raise ValueError('NS bridge family must be zero or one')
    odd = int(raised)
    result = np.zeros((2, 4, 4), complex)
    for e, (hi, (zh, oh)), (ai, (za, oa)) in product(
            (0, 1), enumerate(GROUNDS), enumerate(GROUNDS)):
        # Canonical W_h W_a on the puncture-0 ground tensor.
        endpoint_sign = (-1)**(odd*oh)
        # The NS outgoing bra self-crossing and trinion regrouping.
        vertex_sign = (-1)**(family*(family+zh+oh+odd)+za*(oh+odd))
        inverse_gram_sign = (-1)**family
        result[e, hi, ai] = (EMBEDDING[2*zh+za, e]*EMBEDDING[2*oh+oa, e]
            *endpoint_sign*vertex_sign*inverse_gram_sign)
    return result


@dataclass(frozen=True)
class LiouvilleCouplings:
    """Analytic couplings: sphere c_eta, NS (C,Ctilde), R handle c_eta."""
    sphere: tuple
    ns_handle: tuple
    r_handle: tuple

    def __post_init__(self):
        for name in ('sphere', 'ns_handle', 'r_handle'):
            value = tuple(complex(x) for x in getattr(self, name))
            if len(value) != 2 or not np.isfinite(value).all():
                raise ValueError(name+' needs two finite constants')
            object.__setattr__(self, name, value)

    @classmethod
    def evaluate(cls, bridge, loop, omega, *, dps=30):
        return cls(tuple(x/2 for x in rr_ns_structure_constants(omega, omega, bridge, dps)),
            (ns_structure_constant(loop, bridge, loop, dps),
             ns_tilde_structure_constant(loop, bridge, loop, dps)),
            tuple(x/2 for x in rr_ns_structure_constants(loop, loop, bridge, dps)))

    def handle_weight(self, delta, family, handle_eta):
        """Canonical nonchiral one-point coefficient, before chiral traces.

        NS: odd state uses C_HN^(1)=i*Ctilde_BRY. R: i^f implements
        the star pairing; 1/2 removes the doubled Clifford multiplicity
        in a product of FULL chiral traces. For f=0 the ground trace is
        C^+ + C^- (ordinary) or C^+ - C^- (supertrace), not twice this.
        These are local state normalizations, not fitted amplitude factors.
        """
        disorder_sign(delta)
        if family not in (0, 1) or handle_eta not in ETAS:
            raise ValueError('invalid bridge family or HJS sign')
        if delta in (3, 4):
            return (1j)**family*self.ns_handle[family] if handle_eta == 1 else 0j
        return (1j)**family*self.r_handle[ETAS.index(handle_eta)]/2


def local_patch(tau, z):
    """Conservative domain inside the nearest collision images.

    This is a patch restriction, not a truncation-error bound. No puncture
    translation is silently applied: it would wind the disorder line.
    """
    tau, z = complex(tau), complex(z)
    if not np.isfinite([tau, z]).all() or tau.imag <= 0 or z == 0:
        raise ValueError('finite upper-half-plane tau and separated punctures required')
    # The supported modular strip avoids claiming a shortest-vector search.
    if abs(tau.real) > .5 or abs(tau) < 1-1e-12:
        raise ValueError('choose tau in the standard modular fundamental domain')
    if abs(z) >= .25:
        raise ValueError('local RR OPE callback requires 0 < |z| < 1/4')


class LocalRROPE:
    """A fixed-spectral-momentum, four-spin RR density evaluator.

    The RR sphere Ward tensor is conjugated at conjugated PARAMETERS.
    The handle uses the separately normalized reflected trace convention,
    with its nonchiral star phase in LiouvilleCouplings.handle_weight.
    This conjugates structural i's and geometry, never the physical energy.
    Both bridge families retain their own primary powers and inverse norms.
    """
    def __init__(self, bridge, loop, omega, *, bridge_order=3,
                 ns_twice_cutoff=4, r_cutoff=2, couplings=None, dps=30):
        self.bridge, self.loop, self.omega = float(bridge), float(loop), complex(omega)
        if not np.isfinite([self.bridge, self.loop, self.omega]).all() or self.omega == 0:
            raise ValueError('finite momenta and nonzero analytic energy required')
        self.cutoffs = dict(bridge_order=bridge_order,
            ns_twice_cutoff=ns_twice_cutoff, r_cutoff=r_cutoff)
        self.holo = TorusRROPEBlock(self.bridge, self.loop, self.omega, **self.cutoffs)
        # beta=i*omega/sqrt(2). To conjugate beta, the chiral constructor
        # must receive -conj(omega), NOT conj(omega). The latter gives the
        # right weights but flips Ramond zero-mode/odd-family phases.
        self.anti = TorusRROPEBlock(self.bridge, self.loop, -self.omega.conjugate(), **self.cutoffs)
        self.couplings = couplings or LiouvilleCouplings.evaluate(
            self.bridge, self.loop, self.omega, dps=dps)

    @lru_cache(None)
    def chiral_values(self, tau, z, delta, eta, handle_eta, mode, anti=False,
                      handle_ground_only=False):
        block = self.anti if anti else self.holo
        values = np.array([block.evaluate(tau, z, delta=delta, eta=eta,
            handle_eta=handle_eta, mode=mode, grounds=ground, form_parity=0,
            handle_ground_only=handle_ground_only) for ground in GROUNDS])
        return values.conjugate() if anti else values

    def liouville(self, tau, z, delta, *, modes=(None, None), handle_ground_only=False):
        """Return (L^{++},L^{--}); the two time mixed correlators vanish.

        modes=(None,None) is an unraised control. Each physical C_pq
        instead uses p,q in {0,1}, corresponding to G0 and G_-1 at zero.
        """
        local_patch(tau, z)
        disorder_sign(delta)
        if len(modes) != 2 or not (modes == (None, None) or all(m in (0, 1) for m in modes)):
            raise ValueError('use a bare pair or one odd mode in each chirality')
        tau, z = complex(tau), complex(z)
        raised = modes != (None, None)
        result = np.zeros(2, complex)
        for ei, eta in enumerate(ETAS):
            for handle_eta in ((1,) if delta in (3, 4) else ETAS):
                h = self.chiral_values(tau, z, delta, eta, handle_eta, modes[0],
                                       False, handle_ground_only)
                a = self.chiral_values(tau, z, delta, eta, handle_eta, modes[1],
                                       True, handle_ground_only)
                for f in (0, 1):
                    weight = self.couplings.sphere[ei]*self.couplings.handle_weight(delta, f, handle_eta)
                    result += weight*np.einsum('eij,i,j->e', sphere_pairing(f, raised=raised),
                                               h[:, f], a[:, f])
        if not np.isfinite(result).all():
            raise ArithmeticError('nonfinite local Liouville component')
        return result

    def components(self, tau, z, delta, *, handle_ground_only=False):
        """BRY incoming/outgoing C_pq in the contractible theta frame.

        The minus on the outgoing mu V^- term cancels the graded exchange
        sign. The time spin magnitude is already in free_outer_factor.
        """
        sign = disorder_sign(delta)
        return {(p, q): (lambda l: (l[0]+sign*l[1])/2)(self.liouville(
            tau, z, delta, modes=(p, q), handle_ground_only=handle_ground_only))
            for p, q in product((0, 1), repeat=2)}

    def evaluate(self, tau, z, *, handle_ground_only=False):
        """Four contributions and their sum, per d²tau d²z dP dp/pi².

        Each free factor already includes the Type 0B GSO 1/2; do not
        average these entries a second time. No S-matrix phase is applied.
        """
        local_patch(tau, z)
        sectors, components = {}, {}
        with mp.workdps(35):
            for delta in THETAS:
                c = self.components(tau, z, delta, handle_ground_only=handle_ground_only)
                components[delta] = c
                sectors[delta] = complex(free_outer_factor(omega=self.omega, delta=delta, z=z, tau=tau)
                    *raised_combination(c, omega=self.omega, delta=delta, z=z, tau=tau))
        return dict(spins=sectors, total=sum(sectors.values()), components=components)

    def component_polynomials(self, tau, delta, *, order=None, handle_ground_only=False):
        """BRY C_pq Taylor arrays, by NS family, truncated by total degree.

        Remove z^(h+f/2-2*h_R-p) zbar^(h+f/2-2*h_R-q).
        All cylinder factors and the SINGLE handle Casimir are included.
        The arrays can be supplied directly to RRCollisionKernel.
        """
        local_patch(tau, .01)
        sign = disorder_sign(delta)
        order = self.cutoffs['bridge_order'] if order is None else order
        if not isinstance(order, int) or not 0 <= order <= self.cutoffs['bridge_order']:
            raise ValueError('Taylor order must not exceed the bridge relative order')
        h = (1+self.bridge**2)/2
        out = [{key: np.zeros((order+1, order+1), complex)
                for key in product((0, 1), repeat=2)} for f in (0, 1)]
        for ei, eta in enumerate(ETAS):
            for handle_eta in ((1,) if delta in (3, 4) else ETAS):
                arrays = []
                for anti, block in ((False, self.holo), (True, self.anti)):
                    levels = block.handle.levels(delta)
                    q = np.exp(K*complex(tau)*(block.handle.exponent(delta)+levels))
                    if handle_ground_only:
                        q[1:] = 0
                    values = np.array([[block.local(mode=mode, delta=delta, eta=eta,
                        handle_eta=handle_eta, grounds=g, form_parity=0)[..., :len(q)]@q
                        for g in GROUNDS] for mode in (0, 1)])
                    arrays.append(values.conjugate() if anti else values)
                for f in (0, 1):
                    pairing = sphere_pairing(f, raised=True)
                    pairing = (pairing[0]+sign*pairing[1])/2
                    weight = (self.couplings.sphere[ei]*self.couplings.handle_weight(delta, f, handle_eta)
                              *(2*np.pi)**(2*h+f))
                    for p, q in product((0, 1), repeat=2):
                        value = np.einsum('ab,ai,bj->ij', pairing,
                            arrays[0][p, :, f, :order+1], arrays[1][q, :, f, :order+1])
                        out[f][p, q] += weight*value
        mask = np.add.outer(np.arange(order+1), np.arange(order+1)) <= order
        for family in out:
            for key in family:
                family[key] *= mask
        return out

    def collision_polynomials(self, tau, delta, *, order=None, handle_ground_only=False):
        """Actual coupled local density polynomials for disk/annulus primitives.

        This constructs counterterm DATA; selecting a global finite-part
        prescription and performing the moduli integral are separate steps.
        """
        from collision_kernel import RRCollisionKernel
        order = self.cutoffs['bridge_order'] if order is None else order
        kernel = RRCollisionKernel(tau=tau, omega=self.omega, delta=delta, order=order)
        components = self.component_polynomials(tau, delta, order=order,
            handle_ground_only=handle_ground_only)
        return [dict(power=kernel.density_power(self.bridge, f),
                     polynomial=kernel.raise_components(components[f])) for f in (0, 1)]

    def close(self):
        self.chiral_values.cache_clear()
        self.holo.close()
        self.anti.close()

    def __enter__(self):
        return self

    def __exit__(self, *exception):
        self.close()
