"""Type 0B matter projections in the stored small-representation frame.

The four-R projection uses chiral Type 0B spinors and is checked against
the sixteen-component graded tensor. Multiplying the two nonchiral
correlators without transporting their odd forms is insufficient. The
mixed projection is kept term-resolved for picture and Ward diagnostics.
No matrix-model amplitude enters this module.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import product
import math

import numpy as np

from frozen import ramond_state, ramond_vertex, crossing_phase, SECTORS


@lru_cache(None)
def ramond_weight(external, qh, qa, sl, sr):
    """Literal graded small-representation contraction, including endpoint frames."""
    total = 0j
    for ((a0, b0), ket), ((a4, b4), bra) in product(
            ramond_state(external[0]), ramond_state(external[3])):
        rf, brf = (qh+a0)%2, (qa+b0)%2
        lf, blf = (qh+a4)%2, (qa+b4)%2
        total += (ket*complex(bra).conjugate()
                  * ramond_vertex(external[1], sr, rf, brf)
                  * ramond_vertex(external[2], sl, lf, blf)
                  * (-1)**(brf*a0+blf*qh)
                  * 1j**(qh*(a4-a0)) * (-1j)**(qa*(b4-b0)))
    return complex(total)


def ising_blocks(z):
    """Identity and fermion blocks, each with leading coefficient one."""
    z = np.asarray(z, complex)
    lz, l1 = np.log(z), np.log(1-z)
    plus = np.sqrt((1+np.exp(l1/2))/2)
    common = np.exp(-(lz+l1)/8)
    return np.array([common*plus, common*np.exp(lz/2)/plus])


def ising_correlator(external, z):
    f = ising_blocks(z)
    return sum(ramond_weight(tuple(external), h, a, 1, 1)*.5**(h+a)
               *f[h]*f[a].conjugate() for h, a in product((0, 1), repeat=2))


@lru_cache(None)
def four_r_finite_terms(eta=(-1, -1, -1, 1)):
    """Finite graded projection, transported to the canonical chiral blocks.

    Pairing the two defect lines and changing the odd RR chiral form to
    the physical spinor frame are distinct operations. The former gives
    (-1)^(r0*r_infinity+r_z*r_1); the latter gives (sL*sR)^(qh+qa).
    Omitting either fails a crossing transformation. This tensor is also
    checked against the factorized chiral GSO expression below.
    """
    out = []
    for h, t, a, b, sl, sr in product(*([(0, 1)]*4+[(-1, 1)]*2)):
        c = sum(math.prod(e**r for e, r in zip(eta, ext))/4
                *ramond_weight(ext, h, a, sl, sr)
                *ramond_weight(ext, t, b, 1, 1)*.5**(t+b)
                *(-1)**(ext[0]*ext[3]+ext[1]*ext[2])
                for ext in product((0, 1), repeat=4))
        c *= (sl*sr)**(h+a)
        if abs(c) > 1e-14:
            out.append((h, t, a, b, sl, sr, c))
    return tuple(out)


@lru_cache(None)
def four_r_terms():
    """A^- A^- A^- A^+ in a chiral Type 0B basis.

    K_0 = F_even I_0 - F_odd I_1/2, with (sL,sR)=(-,+).
    K_1 = F_odd I_0 - F_even I_1/2, with (sL,sR)=(+,-).
    Sew K_0 Kbar_0 and K_1 Kbar_1 against their separate SL densities.
    I_1 has unit leading coefficient; its physical Ising OPE coefficient
    is 1/2. The sign branches follow RRNS three-point factorization.
    No equality with the NSNS or mixed string amplitude is imposed.
    """
    out = []
    for parity, sl, sr in ((0, -1, 1), (1, 1, -1)):
        chiral = ((parity, 0, 1.), (1-parity, 1, -.5))
        for (h, t, ch), (a, b, ca) in product(chiral, repeat=2):
            out.append((h, t, a, b, sl, sr, complex(ch*ca)))
    return tuple(out)


@dataclass(frozen=True)
class Outer:
    external: tuple
    branch: str
    coefficient: complex
    hz: complex
    h1: complex
    az: complex
    a1: complex

    def reflected(self):
        e = self.external
        return Outer((e[2], e[1], e[0], e[3]), self.branch,
                     self.coefficient*crossing_phase(SECTORS['mixed_ns'], e),
                     self.h1, self.hz, self.a1, self.az)

    def value(self, z):
        lz, l1 = np.log(z), np.log(1-z)
        return self.coefficient*np.exp(self.hz*lz+self.h1*l1
                                       +self.az*lz.conjugate()+self.a1*l1.conjugate())


def mixed_terms(times, picture='one', eta=(-1, 1)):
    """Term-resolved projection; times are outgoing positive, incoming negative.

    The common PCO normalization is deliberately external to this tensor.
    Preserve the four branches separately until the picture audit passes.
    """
    if picture not in ('one', 'infinity'):
        raise ValueError('picture must be one or infinity')
    k = tuple(times)
    if len(k) != 4 or len(eta) != 2 or any(e not in (-1, 1) for e in eta):
        raise ValueError('four time momenta and two RR in/out signs required')
    j = 2 if picture == 'one' else 3
    energy = -k[j]  # sign in exp(+i energy X^0), BRY (2.6)
    out = []
    for r, s, a, b in product((0, 1), repeat=4):
        u, v = 1-a, 1-b  # time fermions in the two chiralities
        if (r+s-u-v)%2:
            continue
        if u == v == 0:
            c, branch = 1, 'W'
        elif u == v == 1:
            c, branch = .5j*(-1)**r*energy**2, 'psi_psibar'
        elif u:
            # In the graded HJS product, transporting the time holomorphic
            # fermion past the anti-Liouville odd state changes the sign
            # of BRY's explicitly ordered -psi^0 Lambda_bar term.
            c, branch = energy*np.exp(-1j*np.pi*(1-2*r)/4)/math.sqrt(2), 'psi_Lambdabar'
        else:
            c, branch = -energy*np.exp(1j*np.pi*(1-2*r)/4)/math.sqrt(2), 'psibar_Lambda'
        if j == 3:
            # Bra transport: reverse the two SL odd modes and transport
            # each free fermion to its infinity frame. This is needed
            # before comparing different picture placements.
            c *= (-1)**(a*b+u+v)
        c *= eta[0]**r*eta[1]**s/2
        e = (r, s, (a, b) if j == 2 else (0, 0),
             (a, b) if j == 3 else (0, 0))
        hz, az = -k[0]*k[1]-.375+u/2, -k[0]*k[1]-.375+v/2
        # Picture -1 NS field supplies the remaining superghost factors.
        h1 = -k[1]*k[2]-(.5 if j == 3 else u/2)
        a1 = -k[1]*k[2]-(.5 if j == 3 else v/2)
        out.append(Outer(e, branch, c, hz, h1, az, a1))
    return tuple(out)


def chart_terms(times, channel, picture, eta=(-1, 1)):
    if channel == 's':
        return mixed_terms(times, picture, eta)
    if channel == 'u':
        times = (times[0], times[1], times[3], times[2])
        picture = 'infinity' if picture == 'one' else 'one'
    return tuple(t.reflected() for t in mixed_terms(times, picture, eta))
