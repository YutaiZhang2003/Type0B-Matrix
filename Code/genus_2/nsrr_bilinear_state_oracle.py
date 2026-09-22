"""Independent finite-state physical NSRR BPZ oracle (not a block assembler).

Physical vertices are obtained by successive Ward reduction of the two
superalgebras. Grams are constructed in a Clifford-adapted multiplicity
basis and transported by transpose. The full theta sign is applied to
physical state parities. No M matrix or chiral block is used here.
"""
from __future__ import annotations

from functools import lru_cache
from itertools import product
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
for directory in ("Code", "Code/c_Recursion", "Code/double_virasoro/nsrr"):
    sys.path.insert(0, str(ROOT/directory))
import numpy as np
import sympy as sp
from ns_genus2_symbolic_low_order import ExactNSVermaModule
from ramond_pbw_generalized_ward import GeneralizedNRRWard, RamondPBWModule, word_parity
from nsrr_bilinear_sewing import physical_signs

I = sp.I
U8, U8BAR = (1-I)/sp.sqrt(2), (1+I)/sp.sqrt(2)
J = sp.Matrix([[0, U8], [U8BAR, 0]])
JBAR = sp.conjugate(J)
E0 = sp.Matrix([[1, 0], [0, 1], [0, 1], [-I, 0]])/sp.sqrt(2)
ERJ = sp.Matrix([[1, 0], [0, U8BAR], [0, U8], [-I, 0]])/sp.sqrt(2)
ELJ = sp.Matrix([[1, 0], [0, -U8], [0, -U8BAR], [-I, 0]])/sp.sqrt(2)
DJ = sp.diag(1, -1, -1, -1)


@lru_cache(None)
def clifford_change(p, q):
    """Columns: J-adapted physical kets; rows: Wh Wa |R^epsilon>.

    This is a coordinate change, obtained by coefficient comparison, not
    an inner product. It depends on word parities, not levels or weights.
    """
    old = sp.Matrix(4, 2, lambda row, e: (-1)**(q*(row//2))*E0[row, e])
    result = sp.zeros(2)
    for epsilon in (0, 1):
        new = sp.zeros(4, 1)
        for a, b in product((0, 1), repeat=2):
            gh, ga = p ^ a, q ^ b
            ch = (-1)**(p*a)*(J[gh, p] if a else 1)
            ca = (-1)**(q*b)*(JBAR[ga, q] if b else 1)
            new[2*gh+ga] = ERJ[2*a+b, epsilon]*ch*ca
        eold = epsilon ^ p ^ q
        row = next(k for k in range(4) if old[k, eold] != 0)
        result[eold, epsilon] = sp.simplify(new[row]/old[row, eold])
        assert sp.simplify(old*result[:, epsilon]-new) == sp.zeros(4, 1)
    return result


class PhysicalNSRRBPZ:
    def __init__(self, *, h=sp.Rational(7, 10), c=sp.Rational(81, 5),
                 beta2=I*sp.Rational(2, 5), beta3=I*sp.Rational(7, 10),
                 anti_h=None, anti_c=None, anti_beta2=None, anti_beta3=None):
        self.h, self.c, beta2, beta3 = map(sp.sympify, (h, c, beta2, beta3))
        self.betas = beta2, beta3
        # Anti parameters are independent analytic data. Conjugating a
        # Ward engine at conjugated inputs conjugates ONLY its structural
        # i's: anti G0=-i beta exp(+/-i pi/4), Btilde(w-,w-)=-i.
        self.anti_h = sp.sympify(h if anti_h is None else anti_h)
        self.anti_c = sp.sympify(c if anti_c is None else anti_c)
        self.anti_betas = (sp.sympify(beta2 if anti_beta2 is None else anti_beta2),
                           sp.sympify(beta3 if anti_beta3 is None else anti_beta3))
        self.forms = {}
        for anti, p, eta in product((False, True), (0, 1), (1, -1)):
            ac = sp.conjugate(self.anti_c) if anti else self.c
            ah = sp.conjugate(self.anti_h) if anti else self.h
            bs = tuple(map(sp.conjugate, self.anti_betas)) if anti else self.betas
            self.forms[anti, p, eta] = GeneralizedNRRWard(
                p_phi=p, form_parity=0, eta=eta, h_ns=ah,
                h_second=ac/24-bs[0]**2, h_third=ac/24-bs[1]**2,
                beta_second=bs[0], beta_third=bs[1], central_charge=ac)
        self.modules = tuple(RamondPBWModule(self.c/24-b*b, b, self.c) for b in self.betas)
        self.anti_modules = tuple(RamondPBWModule(sp.conjugate(self.anti_c/24-b*b),
                                  sp.conjugate(b), sp.conjugate(self.anti_c)) for b in self.anti_betas)
        self.ns = ExactNSVermaModule(c=self.c, weight=self.h)
        self.ans = ExactNSVermaModule(c=self.anti_c, weight=self.anti_h)

    @lru_cache(None)
    def anti_primary(self, words, a, b, eta):
        return sp.conjugate(self.forms[True, 0, eta].value(words[0], words[1], a, words[2], b))

    @lru_cache(None)
    def vertex(self, hwords, awords, epsilon2, epsilon3, eta):
        """Unit c_eta physical trinion, from two successive Ward reductions."""
        parities = tuple(word_parity(w) for w in awords)
        def base(a, b):
            factor = sp.prod(-I if delta and alpha else 1
                             for delta, alpha in zip(parities[1:], (a, b)))
            return factor*self.anti_primary(awords, a ^ parities[1], b ^ parities[2], eta)
        if parities[0] == 0:
            co = {1: (base(0, 0)+base(1, 1))/2, -1: (base(0, 0)-base(1, 1))/2}
        else:
            co = {1: (base(0, 1)-I*base(1, 0))/2, -1: (base(0, 1)+I*base(1, 0))/2}
        factor = sp.prod(I if delta and epsilon == 0 else 1
                         for delta, epsilon in zip(parities[1:], (epsilon2, epsilon3)))
        a, b = epsilon2 ^ parities[1], epsilon3 ^ parities[2]
        return sp.simplify(factor*sum(co[t]*self.forms[False, parities[0], t].value(
            hwords[0], hwords[1], a, hwords[2], b) for t in (1, -1)))

    @lru_cache(None)
    def ramond_basis_gram(self, edge, level, anti_level):
        """Exact BPZ Gram in the canonical physical descendant ket basis."""
        module, amodule = self.modules[edge], self.anti_modules[edge]
        hb, ab = tuple(module.basis(level, 0)), tuple(amodule.basis(anti_level, 0))
        physical = tuple((x.word, y.word, e) for x, y, e in product(hb, ab, (0, 1)))
        bh = sp.Matrix([[module.inner_product(x, y) for y in hb] for x in hb])
        ba = sp.conjugate(sp.Matrix([[amodule.inner_product(x, y) for y in ab] for x in ab]))
        bj = sp.kronecker_product(bh, ba, sp.eye(2))
        blocks = [clifford_change(word_parity(x.word), word_parity(y.word)) for x, y in product(hb, ab)]
        change = sp.diag(*blocks)
        inverse_change = change.inv()
        gram = sp.simplify(inverse_change.T*bj*inverse_change)
        return physical, gram

    @lru_cache(None)
    def ramond_basis_inverse(self, edge, level, anti_level):
        basis, gram = self.ramond_basis_gram(edge, level, anti_level)
        return basis, np.linalg.inv(np.asarray(gram.evalf(40), complex))

    @lru_cache(None)
    def tensors_and_grams(self, levels, anti_levels):
        hb, ab = self.ns.basis(levels[0]), self.ans.basis(anti_levels[0])
        word = lambda st: tuple((kind, sp.Rational(mode, 2)) for kind, mode in st)
        nb = tuple((word(h), word(a)) for h, a in product(hb, ab))
        gn = np.linalg.inv(np.asarray(sp.kronecker_product(
            self.ns.gram_matrix(levels[0]), self.ans.gram_matrix(anti_levels[0])).evalf(40), complex))
        gn *= (-1)**(levels[0]*anti_levels[0])
        b2, g2 = self.ramond_basis_inverse(0, levels[1], anti_levels[1])
        b3, g3 = self.ramond_basis_inverse(1, levels[2], anti_levels[2])
        ts = {}
        for eta in (1, -1):
            ts[eta] = np.array([[[complex(self.vertex((hw, w2, w3), (aw, a2, a3), e2, e3, eta))
                for w3, a3, e3 in b3] for w2, a2, e2 in b2] for hw, aw in nb])
        return b2, b3, (gn, g2, g3), ts

    def coefficient(self, levels, anti_levels, *, physical_lifts_slots):
        """All four c_L,eta c_R,eta' coefficients; primary powers stripped."""
        physical_signs(physical_lifts_slots)
        b2, b3, grams, tensors = self.tensors_and_grams(tuple(levels), tuple(anti_levels))
        n = (levels[0]+anti_levels[0]) % 2
        omega = tuple(physical_lifts_slots)
        signs = np.empty((len(b2), len(b3)), complex)
        for i, (hw, aw, e) in enumerate(b2):
            p = (word_parity(hw)+word_parity(aw)+e) % 2
            for j, (hw3, aw3, e3) in enumerate(b3):
                q = (word_parity(hw3)+word_parity(aw3)+e3) % 2
                signs[i, j] = (-1)**(n*p+n*q+p*q)*omega[0]**n*omega[1]**p*omega[2]**q
        return {(eta, etap): complex(np.einsum('abc,ad,be,cf,def->',
            tensors[eta]*signs[None, :, :], *grams, tensors[etap], optimize=True))
            for eta, etap in product((1, -1), repeat=2)}
