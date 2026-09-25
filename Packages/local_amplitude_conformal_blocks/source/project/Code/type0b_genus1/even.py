"""Type 0B even-spin two-point sewing, independently of heterotic free fields.

The frozen banks supply CHIRAL Liouville blocks, not string integrands. Both
external momenta are continued as omega, including in the antichiral block.
The four components are GG/GG, GG/PP, PP/GG, PP/PP. Global normalization and
the missing odd-spin and degeneration terms are deliberately not inferred.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = ROOT / "handoffs/reference/HetSO23_1to1_20260920"
BANK_ROOT = REFERENCE / "data_exports/spin23_genus1_vv"
SPINS = ("NS", "NS_tilde", "R")
COMPONENTS = ("GG/GG", "GG/PP", "PP/GG", "PP/PP")
CHARACTERISTICS = ((0., 0.), (0., .5), (.5, 0.))


def theta(a, b, z, tau, derivative=0):
    z, tau = np.broadcast_arrays(np.asarray(z, complex), np.asarray(tau, complex))
    if np.any(tau.imag <= 0) or not np.isfinite(tau).all():
        raise ValueError("positive finite torus height required")
    # Includes the displaced Gaussian centre for points outside the rectangle.
    count = int(np.ceil(np.max(abs(z.imag)/tau.imag + np.sqrt(46/(math.pi*tau.imag)))))+3
    n = np.arange(-count, count+1).reshape((-1,)+(1,)*z.ndim)+a
    return np.sum((2j*math.pi*n)**derivative * np.exp(
        1j*math.pi*n*n*tau+2j*math.pi*n*(z+b)), axis=0)


def eta(tau):
    tau = np.asarray(tau, complex)
    if np.any(tau.imag <= 0):
        raise ValueError("positive torus height required")
    q = np.exp(2j*math.pi*tau)
    count = int(np.ceil(46/(2*math.pi*np.min(tau.imag))))+2
    n = np.arange(1, count+1).reshape((-1,)+(1,)*tau.ndim)
    return np.exp(1j*math.pi*tau/12)*np.prod(1-q**n, axis=0)


def pairing_phase(forms, left, right, *, ramond=False):
    """Ordered graded-tensor phase, before physical external Wick ordering."""
    f, a, b = tuple(forms), tuple(left), tuple(right)
    lp = tuple((x+y)%2 for x,y in zip(f,a))
    rp = tuple((x+y)%2 for x,y in zip(f,b))
    crossings = sum(rp[i]*lp[j] for i in range(len(f)) for j in range(i+1,len(f)))
    return (-1j)**(sum(b)+(0 if ramond else sum(f))) * (-1)**(
        sum(x*y for x,y in zip(f,b))+crossings)


def factored_coefficients(coefficients, levels, exponent):
    """Multiply by (1-q0)^a(1-q1)^a, retaining the original series order."""
    b = np.ones(int(levels.max())//2+1, complex)
    for k in range(1,len(b)):
        b[k] = b[k-1]*(k-1-exponent)/k
    delta = levels[:,None]-levels[None,:]
    valid = np.all(delta >= 0,axis=-1)&np.all(delta%2 == 0,axis=-1)
    ix = np.maximum(delta//2,0)
    return coefficients @ (valid*b[ix[...,0]]*b[ix[...,1]]).T


@dataclass
class ChiralBank:
    metadata: dict
    arrays: dict
    path: Path
    file_sha256: str

    @classmethod
    def load(cls, path):
        path = Path(path).resolve()
        with np.load(path,allow_pickle=False) as data:
            meta = json.loads(str(data["metadata"]))
            arrays = {k:data[k].copy() for k in data.files if k != "metadata"}
        if meta["schema"] not in ("spin23-vv-recursive-bank-v1","type0b-even-chiral-v1"):
            raise ValueError("unsupported chiral archive")
        for k,a in arrays.items():
            if hashlib.sha256(a.tobytes()).hexdigest() != meta["array_sha256"][k]:
                raise ValueError(f"bank hash mismatch: {k}")
            if not np.isfinite(a).all():
                raise ValueError(f"non-finite bank: {k}")
        n = len(arrays["weights"])
        for k,shape in {"momenta":(n,2),"ns":(n,2,2,len(arrays["ns_levels"])),
                        "r":(n,2,2,4,len(arrays["r_levels"]))}.items():
            if arrays[k].shape != shape: raise ValueError(f"invalid {k} shape")
        return cls(meta,arrays,path,hashlib.sha256(path.read_bytes()).hexdigest())

    @property
    def omega(self):
        return complex(*self.metadata["energy"])

    def components(self,tau,z,cutoffs=(2,4,6,8),*,factored=True,chunk=32):
        """Candidate even-spin density per d²tau d²z, with 1/2 GSO included.

        An independent torus normalization multiplying g_s² is excluded.
        This method does not integrate a collision disk or restore the cusp.
        """
        if self.metadata.get('channel')=='leading-cusp':raise ValueError('ground-handle bank cannot evaluate a full torus')
        tau,z = np.broadcast_arrays(np.atleast_1d(tau).astype(complex),
                                    np.atleast_1d(z).astype(complex))
        if tau.ndim != 1 or np.any(z.imag <= 0) or np.any(z.imag >= tau.imag):
            raise ValueError("ordered punctures strictly inside the torus required")
        if any(c < 0 or c > self.metadata["cutoff"] for c in cutoffs):
            raise ValueError("requested cutoff absent from bank")
        increments = np.column_stack((z,tau-z))
        logq = 2j*math.pi*increments
        q = np.exp(logq)
        et = eta(tau)
        prime = theta(.5,.5,-z,tau)/theta(.5,.5,0,tau,1)
        green = -2*np.log(abs(prime))+2*math.pi*z.imag*z.imag/tau.imag
        exponents = np.array([1.5+self.omega**2,.5+self.omega**2])
        hsum = 1+self.omega**2
        frame0 = np.exp(2*hsum*math.log(2*math.pi))
        oscillator = np.exp(-self.omega**2*green)
        # bc/automorphism: |eta|^4/2; X oscillator/zero-mode:
        # 1/(sqrt(8*pi²*tau2)|eta|²); two PCOs per chirality: 1/16.
        # Combined Majorana and superghost factor: |eta/theta_delta|.
        common0 = frame0*oscillator*abs(et)**2/(64*np.sqrt(8*math.pi**2*tau.imag))
        out = np.zeros((len(tau),len(cutoffs),3,4),complex)
        for spin,(aa,bb) in enumerate(CHARACTERISTICS):
            th = theta(aa,bb,0,tau)
            szego = theta(aa,bb,-z,tau)/(th*prime)
            common = common0*abs(et/th)
            sector = "r" if spin == 2 else "ns"
            levels = self.arrays[sector+"_levels"]
            coeffs = self.arrays[sector]
            if factored:
                coeffs = np.stack([factored_coefficients(coeffs[:,w],levels,exponents[w])
                                   for w in range(2)],axis=1)
            powers = np.exp(levels @ logq.T/2)
            if spin == 1: powers *= (-1.)**levels[:,-1,None]
            # Recover scalar sewing products from saved GG/PP phases. The
            # PP/PP phase is then reinstated, independent of heterotic factors.
            coupling = self.arrays[sector+"_weights"][:,1]
            maskshape = (1,2)+(1,)*(coupling.ndim-2)
            gg_sign = np.array([-1.,1.]).reshape(maskshape)
            left_free = np.stack((np.full(len(z),2j*math.pi),-self.omega**2*szego))
            right_free = np.stack((np.full(len(z),-2j*math.pi),-self.omega**2*szego.conjugate()))
            if factored:
                ll = np.log1p(-q).sum(axis=1)
                left_free *= np.exp(-exponents[:,None]*ll)
                right_free *= np.exp(-exponents[:,None]*ll.conjugate())
            for ci,cutoff in enumerate(cutoffs):
                keep = levels.sum(axis=1) <= cutoff
                for start in range(0,len(coeffs),chunk):
                    sl = slice(start,start+chunk)
                    left = coeffs[sl][...,keep] @ powers[keep]
                    right = coeffs[sl][...,keep] @ powers[keep].conjugate()
                    left[:,0] *= gg_sign[...,None]
                    right[:,0] *= gg_sign[...,None]
                    gauss = np.exp(self.arrays["momenta"][sl]**2 @ logq.real.T)
                    if spin < 2: gauss *= np.exp(-logq.real.sum(axis=1)/8)
                    for comp,(a,b) in enumerate(((0,0),(0,1),(1,0),(1,1))):
                        term = coupling[sl][...,None]*left[:,a]*right[:,b]
                        term = term.sum(axis=tuple(range(1,term.ndim-1)))
                        out[:,ci,spin,comp] += np.einsum("mg,mg,m->g",term,gauss,
                            self.arrays["weights"][sl])*left_free[a]*right_free[b]*common
        if not np.isfinite(out).all(): raise ArithmeticError("non-finite density")
        return out


def require_complete_amplitude(*,even,odd=None,collision=None,cusp=None,normalization=None):
    missing = [k for k,v in dict(odd=odd,collision=collision,cusp=cusp,
                                 normalization=normalization).items() if v is None]
    if missing:
        raise RuntimeError("No physical genus-one amplitude: unresolved "+", ".join(missing))
    return normalization*(even+odd+collision+cusp)
