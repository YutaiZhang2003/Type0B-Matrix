"""Experimental physical NS/R radial trace for two Ramond punctures.

The bare trace is assembled in a Hermitian state basis and continued in
the external momentum analytically. This independent bulk construction
must be matched to the OPE before it is used as a string integrand.
"""
from __future__ import annotations

from functools import lru_cache
from itertools import product
import sys
from pathlib import Path

import numpy as np

from numeric_nrr import NRR, Module, parity

CODE = Path(__file__).resolve().parents[1]
for directory in (CODE/'c_Recursion', CODE/'double_virasoro/nsrr'):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))
from ns_pbw_basis import ns_pbw_basis
from ramond_pbw_generalized_ward import RamondPBWModule
from super_liouville_structure_constants import rr_ns_structure_constants

E0 = np.array([[1, 0], [0, 1], [0, 1], [-1j, 0]])/np.sqrt(2)
K = 2j*np.pi


@lru_cache(None)
def ns_words(n):
    return tuple(tuple((k, m/2) for k, m in w) for w in ns_pbw_basis(n))


@lru_cache(None)
def r_words(n):
    return tuple(dict.fromkeys(tuple((k, int(m)) for k, m in st.word)
                              for st in RamondPBWModule.basis(n)))


class RadialTrace:
    def __init__(self, p_ns, p_r, omega, *, c=13.5, couplings=None, dps=30):
        self.p_ns, self.p_r, self.omega = float(p_ns), float(p_r), complex(omega)
        self.c = float(c)
        self.h_ns = (self.c-1.5)/24+self.p_ns**2/2
        self.h_r = self.c/24+self.p_r**2/2
        self.h_ext = self.c/24+self.omega**2/2
        self.beta_r = 1j*self.p_r/np.sqrt(2)
        self.couplings = tuple(complex(x) for x in (
            couplings if couplings is not None else
            (x/2 for x in rr_ns_structure_constants(self.omega, self.p_r, self.p_ns, dps))))

    @lru_cache(None)
    def form(self, eta, *, anti=False, reflected=False):
        omega = self.omega.conjugate() if reflected else self.omega
        he, be, br = self.c/24+omega**2/2, 1j*omega/np.sqrt(2), self.beta_r
        if anti:
            he, be, br = he.conjugate(), be.conjugate(), br.conjugate()
        return NRR(self.h_ns, he, self.h_r, be, br, self.c, eta=eta)

    @lru_cache(None)
    def r_basis_inverse(self, r, ar):
        hw, aw = r_words(r), r_words(ar)
        hb, ab = tuple(product(hw, (0, 1))), tuple(product(aw, (0, 1)))
        mh = Module(self.h_r, self.c, self.beta_r)
        ma = Module(self.h_r, self.c, self.beta_r.conjugate())
        hh = np.array([[mh.inner(x, y, hermitian=True) for y in hb] for x in hb])
        ha = np.array([[ma.inner(x, y, hermitian=True).conjugate() for y in ab] for x in ab])
        basis = tuple(product(hw, aw, (0, 1)))
        em = np.zeros((len(hb)*len(ab), len(basis)), complex)
        for col, (w, v, family) in enumerate(basis):
            for gh, ga in product((0, 1), repeat=2):
                em[hb.index((w, gh))*len(ab)+ab.index((v, ga)), col] = (
                    (-1)**(parity(v)*gh)*E0[2*gh+ga, family])
        gram = em.conjugate().T@np.kron(hh, ha)@em
        if not np.allclose(gram, gram.conjugate().T, atol=1e-10):
            raise ArithmeticError('physical Ramond Gram is not Hermitian')
        mh.clear()
        ma.clear()
        return basis, np.linalg.inv(gram)

    @lru_cache(None)
    def ns_inverse(self, n):
        m = Module(self.h_ns, self.c)
        basis = ns_words(n)
        gram = np.array([[m.inner((x, 0), (y, 0)) for y in basis] for x in basis])
        m.clear()
        return np.linalg.inv(gram)

    @lru_cache(None)
    def vertex(self, nh, na, rh, ra, family_r, family_e, eta,
               *, words_e=((), ()), reflected=False):
        """rho(NS at infinity, external R at one, loop R at zero)."""
        eh, ea = words_e
        hform = self.form(eta, reflected=reflected)
        aform = self.form(eta, anti=True, reflected=reflected)
        # Odd zero modes are reduced before invoking the negative-word Ward.
        def entries(form, w, g):
            if w == (('G', 0),):
                return form.modules[1].act('G', 0, (), g).items()
            return (((w, g), 1.),)
        out = 0j
        for gh, ga, lh, la in product((0, 1), repeat=4):
            coefficient = E0[2*gh+ga, family_e]*E0[2*lh+la, family_r]
            if coefficient == 0:
                continue
            coefficient *= (-1)**(parity(ea)*gh+parity(ra)*lh)
            hp, ap, lp = (gh+parity(eh)) % 2, (ga+parity(ea)) % 2, (lh+parity(rh)) % 2
            coefficient *= (-1)**(parity(na)*(parity(nh)+hp+lp)+ap*lp)
            for (weh, geh), ch in entries(hform, eh, gh):
                for (wea, gea), ca in entries(aform, ea, ga):
                    out += coefficient*ch*ca.conjugate()*hform.value(nh, weh, geh, rh, lh)*aform.value(na, wea, gea, ra, la).conjugate()
        return out

    @lru_cache(None)
    def matrix(self, n, an, r, ar, family_e, *, words_e=((), ()), reflected=False):
        basis, _ = self.r_basis_inverse(r, ar)
        ns = tuple(product(ns_words(n), ns_words(an)))
        # For real momenta the constants are real; reflecting the analytic
        # vertex conjugates these constants. The outer adjoint reverses it.
        constants = np.conjugate(self.couplings) if reflected else self.couplings
        return np.array([[sum(co*self.vertex(nh, na, rh, ra, fr, family_e, eta,
                    words_e=words_e, reflected=reflected)
                    for co, eta in zip(constants, (1, -1)))
                for rh, ra, fr in basis] for nh, na in ns])

    @lru_cache(None)
    def bare_coefficient(self, n, an, r, ar, family_e, supertrace_r=False):
        basis, gr = self.r_basis_inverse(r, ar)
        gn = np.kron(self.ns_inverse(n), self.ns_inverse(an))
        left = self.matrix(n, an, r, ar, family_e)
        right = self.matrix(n, an, r, ar, family_e, reflected=True).conjugate()
        signs = np.array([(-1)**(parity(h)+parity(a)+f) if supertrace_r else 1.
                          for h, a, f in basis])
        return np.einsum('ij,ik,jl,kl->', left*signs[None, :], gn, gr, right, optimize=True)

    @lru_cache(None)
    def flat_raised_matrix(self, n, an, r, ar, family_e, p, q, reflected=False):
        """Cylinder G0 or G_-1, in canonical holomorphic-first order."""
        terms = lambda m, k: ((1., (('G', 0),)),) if m == 0 else (
            (k, (('G', -1),)), (.75*k, (('G', 0),)))
        return sum(ch*ca*self.matrix(n, an, r, ar, family_e,
                                    words_e=(wh, wa), reflected=reflected)
                   for (ch, wh), (ca, wa) in product(terms(p, K), terms(q, K.conjugate())))

    @lru_cache(None)
    def raised_coefficient(self, n, an, r, ar, family_e, p, q, sector, supertrace=False):
        """Experimental radial adjoint of a raised external insertion.

        Reflection reverses the two odd operators. On the cylinder the
        G_-1 + 3 G0/4 combination has the same adjoint phase as G0.
        This prescription is subject to bulk/OPE matching.
        """
        basis, gr = self.r_basis_inverse(r, ar)
        gn = np.kron(self.ns_inverse(n), self.ns_inverse(an))
        if sector == 'R':
            left = self.matrix(n, an, r, ar, family_e)
            right = -self.flat_raised_matrix(n, an, r, ar, family_e, p, q, True).conjugate()
        elif sector == 'NS':
            left = self.flat_raised_matrix(n, an, r, ar, family_e, p, q)
            right = self.matrix(n, an, r, ar, family_e, reflected=True).conjugate()
        else:
            raise ValueError('sector must be NS or R')
        signs = np.array([(-1)**(parity(h)+parity(a)+f) if supertrace and sector == 'R' else 1.
                          for h, a, f in basis])
        return np.einsum('ij,ik,jl,kl->', left*signs[None, :], gn, gr, right, optimize=True)

    def raised(self, tau, z, delta, *, ns_cutoff=4, r_cutoff=1):
        if delta not in (1, 2, 3, 4) or not 0 < complex(z).imag < complex(tau).imag:
            raise ValueError('radial patch needs 0 < Im(z) < Im(tau) and a theta characteristic')
        sector = 'R' if delta in (1, 2) else 'NS'
        ns_time, r_time = (z, tau-z) if sector == 'R' else (tau-z, z)
        powers = lambda t, h, n, a: np.exp(K*t*(h+n-self.c/24)-K*complex(t).conjugate()*(h+a-self.c/24))
        out = np.zeros((2, 2, 2), complex)
        for n, an, r, ar in product(range(ns_cutoff+1), range(ns_cutoff+1), range(r_cutoff+1), range(r_cutoff+1)):
            weight = powers(ns_time, self.h_ns, n/2, an/2)*powers(r_time, self.h_r, r, ar)
            if delta == 4:
                weight *= (-1)**(n+an)
            for e, p, q in product((0, 1), repeat=3):
                out[e, p, q] += weight*self.raised_coefficient(n, an, r, ar, e, p, q, sector, delta in (1, 4))
        return np.exp(4*self.h_ext*np.log(2*np.pi))*out

    def bare(self, tau, z, delta, *, ns_cutoff=4, r_cutoff=1):
        if delta not in (1, 2, 3, 4) or not 0 < complex(z).imag < complex(tau).imag:
            raise ValueError('radial patch needs 0 < Im(z) < Im(tau) and a theta characteristic')
        ns_time, r_time = (z, tau-z) if delta in (1, 2) else (tau-z, z)
        powers = lambda t, h, n, a: np.exp(K*t*(h+n-self.c/24)-K*complex(t).conjugate()*(h+a-self.c/24))
        out = np.zeros(2, complex)
        for n, an, r, ar in product(range(ns_cutoff+1), range(ns_cutoff+1), range(r_cutoff+1), range(r_cutoff+1)):
            weight = powers(ns_time, self.h_ns, n/2, an/2)*powers(r_time, self.h_r, r, ar)
            if delta == 4:
                weight *= (-1)**(n+an)
            for e in (0, 1):
                out[e] += weight*self.bare_coefficient(n, an, r, ar, e, delta == 1)
        return np.exp(4*self.h_ext*np.log(2*np.pi))*out

    def close(self):
        for method in (self.form, self.r_basis_inverse, self.ns_inverse, self.vertex,
                       self.matrix, self.bare_coefficient, self.flat_raised_matrix,
                       self.raised_coefficient):
            method.cache_clear()
        NRR.value.cache_clear()
        Module.act.cache_clear()
        Module.canonical.cache_clear()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
