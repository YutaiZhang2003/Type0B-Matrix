#!/usr/bin/env python3
"""Isolated primary-correlator benchmark for Suchanek, arXiv:1012.2974v2.

This module deliberately imports no repository block, sewing, normalization,
Ward, or human-note code. Equation numbers below refer to that paper. The
benchmark evaluates (74) and (77) using the defining structure constants
(10)--(13),(28). The Ramond norm product is taken from the paper's reference
[36], Belavin--Zamolodchikov hep-th/0610316, (46),(48): its index sum is EVEN.
The odd index sum printed on p.22 of 1012.2974 is inconsistent with that
reference and already with the level-one Gram matrix. See the accompanying
report and independent norm test. No normalization is fitted to crossing.

Only the paper's real external momenta, c=3, and principal slit z-plane are
supported. This is stage 1, not a heterotic amplitude or convention translator.
"""
from __future__ import annotations

import argparse
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import time

import mpmath as mp
import numpy as np
from numpy.polynomial.legendre import leggauss

SIGNS = (-1, 1)


def gauss_rule(edges, order):
    x, w = leggauss(order)
    return tuple(np.concatenate([((hi - lo) * v + (hi + lo if k == 0 else 0)) / 2
                                 for lo, hi in zip(edges[:-1], edges[1:])])
                 for k, v in enumerate((x, w)))


class PaperBlocks:
    """Primary recurrences (59),(60),(62), below (70), with the cited norm."""

    def __init__(self, kind, momenta, order, dps=50):
        self.ctx = mp.mp.clone()
        self.ctx.dps = dps
        m = self.ctx
        self.b = (m.sqrt(2) + m.j * m.sqrt(14)) / 4
        self.Q = self.b + 1 / self.b
        self.c = m.mpf(3)
        self.kind = kind
        self.order = order
        self.p = tuple(m.mpf(str(p)) for p in momenta)
        self.beta = tuple(-m.j * p / m.sqrt(2) for p in self.p)  # (6)
        self.lam = tuple(-2 * m.j * p for p in self.p)
        self.labels = [(r, s) for r in range(1, order + 1)
                       for s in range(1, order // r + 1)
                       if (r + s) % 2 == (kind == 'mixed_r')]

    @lru_cache(None)
    def kac(self, r, s):
        m, b = self.ctx, self.b
        if self.kind == 'mixed_r':
            pole = (r * b + s / b) / (2 * m.sqrt(2))
            tail = (-1)**s * (r * b - s / b) / (2 * m.sqrt(2))
            residue = -m.power(2, m.mpf(r * s) - m.mpf('1.5'))
        else:
            pole = (self.Q**2 - (r * b + s / b)**2) / 8
            tail = pole + m.mpf(r * s) / 2
            residue = m.power(2, r * s - 2)
        for k in range(1 - r, r + 1):
            for ell in range(1 - s, s + 1):
                # For R use BZ (46),(48), not the misprinted odd parity on
                # Suchanek p.22. The R set excludes (0,0), and (r,s) is odd
                # so is already absent. For NS also exclude (r,s).
                if (k + ell) % 2:
                    continue
                if (k, ell) == (0, 0):
                    continue
                if self.kind != 'mixed_r' and (k, ell) == (r, s):
                    continue
                residue /= k * b + ell / b
        return pole, tail, residue * m.power(4, r * s)

    @lru_cache(None)
    def fusion(self, r, s, family, first, second, sign=1, odd=0):
        """A.3/A.6; first is lower slot, second is upper slot."""
        m = self.ctx
        value = m.mpc(1)
        den = 2 * m.sqrt(2)
        for k in range(r):
            for ell in range(s):
                parity = (k + ell) % 2
                shift = ((1 - r + 2 * k) * self.b + (1 - s + 2 * ell) / self.b)
                if family == 'nsns':
                    if parity == odd:
                        value *= (first + second - shift) * (first - second - shift) / 8
                elif family == 'rr':
                    value *= first + (1 if parity else -1) * sign * second - shift / den
                elif family == 'nsr':
                    value *= first / den + (1 if parity else -1) * sign * second - shift / den
                else:
                    raise ValueError(family)
        return value

    @lru_cache(None)
    def residue(self, r, s, parity, left, right):
        if self.kind == 'rrrr':
            return (self.fusion(r, s, 'rr', self.beta[3], self.beta[2], left)
                    * self.fusion(r, s, 'rr', self.beta[0], self.beta[1], right))
        if self.kind == 'mixed_ns':
            return (self.fusion(r, s, 'nsns', self.lam[3], self.lam[2], odd=parity)
                    * self.fusion(r, s, 'rr', self.beta[0], self.beta[1], right))
        return (self.fusion(r, s, 'nsr', self.lam[3], self.beta[2], left)
                * self.fusion(r, s, 'nsr', self.lam[0], self.beta[1], right))

    @lru_cache(None)
    def series(self, weight, parity, left, right, order):
        m = self.ctx
        out = [m.mpc(0)] * (order + 1)
        out[0] = m.mpc(parity == 0)
        for r, s in self.labels:
            n = r * s
            if n > order:
                continue
            pole, tail_weight, a = self.kac(r, s)
            if self.kind == 'mixed_r':
                terms = ((1, left, right, weight - pole),
                         (-1, -left, -right, weight + pole))
                for phase, sl, sr, denominator in terms:
                    factor = phase * a * self.residue(r, s, 0, sl, sr) / denominator
                    tail = self.series(tail_weight, 0, sl, sr, order - n)
                    for j, val in enumerate(tail):
                        out[j + n] += factor * val
            else:
                flip = r % 2
                if self.kind == 'rrrr':
                    phase = -1 if flip else 1
                else:
                    phase = m.exp((1 if parity == 0 else -1) * m.j * m.pi / 4) if flip else 1
                factor = phase * a * self.residue(r, s, parity, left, right) / (weight - pole)
                tail = self.series(tail_weight, parity ^ flip,
                                   -left if flip else left, -right if flip else right, order - n)
                for j, val in enumerate(tail):
                    out[j + n] += factor * val
        return tuple(out)

    def coefficients(self, P, parity=0, left=1, right=1):
        m = self.ctx
        p = m.mpf(str(P))
        weight = -m.j * p / m.sqrt(2) if self.kind == 'mixed_r' else self.Q**2 / 8 + p**2 / 2
        return np.array([complex(x) for x in self.series(weight, parity, left, right, self.order)])


class PaperConstants:
    """Three-point products directly from (10)--(13),(28).

    We remove the *same* external factor
      (1/4) M**((Q-i sum(p_i))/b) Upsilon_0**2 prod_i Upsilon_sector(2a_i)
    on both sides. Nothing depending on internal momentum or on channel is
    removed. In particular -i*C_tilde has the relative coefficient 2 from (11).
    """

    def __init__(self, order=32, t_max=256):
        # long double is platform dependent, so also expose quadrature refinement.
        self.b = (np.sqrt(np.longdouble(2)) + 1j * np.sqrt(np.longdouble(14))) / 4
        self.Q = self.b + 1 / self.b
        edges = [0, .05, .2, .5, 1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256]
        edges = [e for e in edges if e < t_max] + [t_max]
        t, w = gauss_rule(edges, order)
        self.t = t.astype(np.longdouble)
        self.w = w.astype(np.longdouble)
        self.den = np.sinh(self.b * self.t / 2) * np.sinh(self.t / (2 * self.b))

    @lru_cache(None)
    def log_u(self, x):
        """(13) inside its strip, shifted by b at the two boundary numerators."""
        x = np.clongdouble(x)
        if abs(x.real) < 1.e-12:
            # U(x+b)=gamma(b*x) b**(1-2*b*x) U(x); no fitted constants.
            z = complex(self.b * x)
            with mp.workdps(40):
                shift = complex(mp.loggamma(z) - mp.loggamma(1-z)
                                + (1 - 2*z) * mp.log(complex(self.b)))
            return self.log_u(complex(x + self.b)) - shift
        if abs(x.real - self.Q.real) < 1.e-12:
            return self.log_u(complex(self.Q - x))
        if not 0 < x.real < self.Q.real:
            raise ValueError(f'Upsilon argument outside implemented strip: {x}')
        d = self.Q / 2 - x
        terms = (d*d * np.exp(-self.t) - np.sinh(d * self.t / 2)**2 / self.den) / self.t
        return complex(np.sum(self.w * terms))

    def log_sector(self, x, sector):
        if sector == 'NS':
            return self.log_u(complex(x/2)) + self.log_u(complex((x+self.Q)/2))
        return self.log_u(complex((x+self.b)/2)) + self.log_u(complex((x+1/self.b)/2))

    def a(self, p):
        return self.Q / 2 + 1j * p

    def log_ns_denominator(self, a, odd=0):
        sector = 'R' if odd else 'NS'
        total = sum(a)
        return self.log_sector(total - self.Q, sector) + sum(self.log_sector(total-2*x, sector) for x in a)

    def log_rr_denominator(self, ns, r2, r1, sign):
        # C^(+)=(C^+ + C^-)/2 is the FIRST term in (12).
        same, other = ('R', 'NS') if sign == 1 else ('NS', 'R')
        return (self.log_sector(ns+r2+r1-self.Q, same)
                + self.log_sector(r1+r2-ns, same)
                + self.log_sector(ns+r2-r1, other)
                + self.log_sector(ns+r1-r2, other))

    def numerator(self, P, sector):
        return self.log_sector(2*self.a(P), sector) + self.log_sector(2*self.a(-P), sector)

    def density(self, family, p, P, parity=0, left=1, right=1):
        a1,a2,a3,a4 = map(self.a, p)
        ap, am = self.a(P), self.a(-P)
        if family == 'rrrr':
            logval = self.numerator(P, 'NS') - self.log_rr_denominator(ap,a3,a4,left) - self.log_rr_denominator(am,a2,a1,right)
            factor = 1
        elif family == 'mixed_ns':
            logval = self.numerator(P, 'NS') - self.log_ns_denominator((a4,a3,ap), parity) - self.log_rr_denominator(am,a2,a1,right)
            factor = 2 if parity else 1
        elif family == 'mixed_r':
            logval = self.numerator(P, 'R') - self.log_rr_denominator(a4,a3,ap,left) - self.log_rr_denominator(a1,a2,am,right)
            factor = 1
        else:
            raise ValueError(family)
        val = factor * np.exp(logval)
        if abs(val.imag) > 1.e-8 * max(abs(val), 1.e-100):
            raise ArithmeticError(f'Unexpected complex density: {val}')
        return float(val.real)


def coordinate(z):
    """One principal lift; q^(1/2)=exp(i*pi*tau/2), never re-root q."""
    z = complex(z)
    if z.imag == 0 and not 0 < z.real < 1:
        raise ValueError('Cut lips require an explicit continuation and are not supported here')
    tau = complex(1j * mp.ellipk(1-z) / mp.ellipk(z))
    return tau, np.exp(1j*np.pi*tau), np.exp(1j*np.pi*tau/2)


def full_prefactor(family, p, z):
    """P-independent |prefactor|^2 from (58),(61),(68)."""
    c0 = 1.5 / 24
    sectors = {'rrrr': (1,1,1,1), 'mixed_ns': (1,1,0,0), 'mixed_r': (0,1,1,0)}[family]
    weights = [c0 + pi*pi/2 + r/16 for pi,r in zip(p,sectors)]
    _,q,_ = coordinate(z)
    theta = complex(mp.jtheta(3, 0, q))
    h1,h2,h3,h4 = weights
    a = c0-h1-h2 + (1/16 if family == 'mixed_r' else 0)
    b = c0-h2-h3 + (1/16 if family == 'mixed_ns' else 0)
    k = .75 - 4*sum(weights) + (.5 if family != 'rrrr' else 0)
    return float(np.exp(2*(a*np.log(complex(z)) + b*np.log(1-complex(z)) + k*np.log(theta)).real))


DEFAULT_POINTS = (.2,.35,.5,.65,.8,.35+.12j,.35-.12j,.65+.12j,.5+.1j)


def run(order=16, p_order=16, t_order=32, p_max=6., t_max=256., points=DEFAULT_POINTS, dps=50):
    started = time.time()
    p_edges = [0,.25,.5,1,1.5,2,3,4]
    p_edges = [p for p in p_edges if p < p_max] + [p_max]
    nodes, pw = gauss_rule(p_edges, p_order)
    constants = PaperConstants(t_order, t_max)
    families = {'rrrr':(.3,.5,.3,-.5), 'mixed_ns':(.2,.4,.3,-.3), 'mixed_r':(.3,.4,.2,-.3)}
    blocks = {k:PaperBlocks(k,p,order,dps) for k,p in families.items()}
    orders = sorted(set(n for n in (4,8,12,16,18,20,24,28,order) if n <= order))
    sums = {k:{n:np.zeros(len(points),dtype=float) for n in orders} for k in families}
    symmetry_error = 0.
    for family,p in families.items():
        coordinates = [coordinate(1-z if family == 'mixed_r' else z) for z in points]
        # Additional rrrr evaluations in the crossed coordinate.
        if family == 'rrrr':
            coordinates += [coordinate(1-z) for z in points]
            sums[family] = {n:np.zeros(2*len(points),dtype=float) for n in orders}
        tvals = np.array([v[2] for v in coordinates])
        qabs = np.array([abs(16*v[1]) for v in coordinates])
        for ni,(P,wP) in enumerate(zip(nodes,pw)):
            configs = [(parity,sl,sr) for parity in ((0,) if family == 'mixed_r' else (0,1))
                       for sl in ((1,) if family == 'mixed_ns' else SIGNS) for sr in SIGNS]
            for parity,sl,sr in configs:
                density = constants.density(family,p,P,parity,sl,sr)
                # Eq. (47)/(77) uses -epsilon_2 beta_2 in the R block.
                coeff = blocks[family].coefficients(P,parity,sl,-sr if family == 'mixed_r' else sr)
                for n in orders:
                    h = np.polynomial.polynomial.polyval(tvals, coeff[:n+1])
                    sums[family][n] += wP * density * qabs**(P*P) * abs(h)**2
            if ni == len(nodes)//2:
                # P->-P permutes R structure signs and beta; compare sewn density*H.
                q, t = coordinates[0][1:]
                vals=[]
                for pp in (P,-P):
                    val=0.
                    for parity,sl,sr in configs:
                        co=blocks[family].coefficients(pp,parity,sl,-sr if family=='mixed_r' else sr)
                        val += constants.density(family,p,pp,parity,sl,sr)*abs(np.polynomial.polynomial.polyval(t,co))**2
                    vals.append(val)
                symmetry_error=max(symmetry_error,abs(vals[0]/vals[1]-1))
        print(f'{family}: {len(nodes)} P nodes, t-order {order}, {time.time()-started:.1f}s',flush=True)
    rows=[]
    for n in orders:
        for j,z in enumerate(points):
            tau,_,_=coordinate(z)
            # Eqs. (74),(77); the modular weight is not an optional normalization.
            for family,kappa in (('rrrr',.375-4*(.125+.3**2/2+.125+.5**2/2)),
                                 ('mixed',.375-2*(.125+.2**2/2+.125+.4**2/2+2*(.0625+.3**2/2))+.25)):
                lhs=sums['rrrr' if family=='rrrr' else 'mixed_ns'][n][j]
                rawrhs=sums['rrrr'][n][j+len(points)] if family=='rrrr' else sums['mixed_r'][n][j]
                rhs=abs(tau)**(2*kappa)*rawrhs
                f1='rrrr' if family=='rrrr' else 'mixed_ns'
                f2='rrrr' if family=='rrrr' else 'mixed_r'
                full_ratio=lhs*full_prefactor(f1,families[f1],z)/(rawrhs*full_prefactor(f2,families[f2],1-z))
                rows.append(dict(family=family,z=[complex(z).real,complex(z).imag],maximum_t_order=n,
                                 lhs=float(lhs),rhs=float(rhs),ratio=float(lhs/rhs),
                                 relative_error=float(abs(lhs/rhs-1)),full_correlator_ratio=float(full_ratio),
                                 prefactor_identity_error=float(abs(full_ratio-lhs/rhs))))
    return dict(source='https://arxiv.org/abs/1012.2974v2',stage='literature-native only',
                ramond_norm_source='https://arxiv.org/abs/hep-th/0610316; equations (46),(48), even index sum',
                c=3,b=[2**.5/4,14**.5/4],momenta=families,
                branch='principal slit plane; beta=-i*p/sqrt(2); sqrt(q)=exp(i*pi*tau/2)',
                normalization='common external Upsilon factor stripped; no channel-dependent adjustment',
                integration='internal P only, symmetric full-line folded to P>0 with common factor 2 omitted; no moduli integration',
                p_order=p_order,p_nodes=len(nodes),p_max=p_max,t_order=t_order,t_max=t_max,dps=dps,
                p_reflection_error=float(symmetry_error),elapsed_seconds=time.time()-started,rows=rows,
                code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--order',type=int,default=16,help='highest power of sqrt(q), twice the q order')
    parser.add_argument('--p-order',type=int,default=16)
    parser.add_argument('--t-order',type=int,default=32)
    parser.add_argument('--p-max',type=float,default=6.)
    parser.add_argument('--t-max',type=float,default=256.)
    parser.add_argument('--z',type=complex,action='append',help='fixed insertion coordinate; may be repeated')
    parser.add_argument('--dps',type=int,default=50)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=run(args.order,args.p_order,args.t_order,args.p_max,args.t_max,
               points=args.z or DEFAULT_POINTS,dps=args.dps)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    for family in ('rrrr','mixed'):
        for n in sorted(set(row['maximum_t_order'] for row in result['rows'])):
            error=max(row['relative_error'] for row in result['rows']
                      if row['family']==family and row['maximum_t_order']==n)
            print(f'{family}: sqrt(q) order {n}, maximum crossing error {error:.4g}')


if __name__=='__main__':
    main()
