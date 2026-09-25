"""Local nonchiral RR tensors with the BRY spin/disorder ground states.

This fixes the local product-algebra cocycles in the marked NS-R-R frame.
It does not identify tube parity signs with flat-torus spin characteristics.
The latter identification, the time Majorana and the ghosts are required
before this tensor can be used as a physical string integrand.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import product
from pathlib import Path
import sys

import numpy as np
import sympy as sp

from mixed_blocks import MixedNSRamondPlumbingBlock

CODE = Path(__file__).resolve().parents[1]
if str(CODE / "genus_2") not in sys.path:
    sys.path.insert(0, str(CODE / "genus_2"))

from nsrr_bilinear_state_oracle import PhysicalNSRRBPZ
from ramond_pbw_generalized_ward import GeneralizedNRRWard, word_parity


@dataclass(frozen=True)
class FullRamondState:
    """Canonical state W_h W_a |R^family>, family 0=+, 1=-."""

    hword: tuple = ()
    aword: tuple = ()
    family: int = 0

    def __post_init__(self):
        if self.family not in (0, 1):
            raise ValueError("Ramond family must be 0 (+) or 1 (-)")

    @property
    def parity(self):
        return (word_parity(self.hword)+word_parity(self.aword)+self.family) % 2


# Ground product order (++,+-,-+,--). This is the local ket embedding,
# not a BPZ dual or a Hermitian projection.
GROUND_KETS = sp.Matrix([[1, 0], [0, 1], [0, 1], [-sp.I, 0]])/sp.sqrt(2)


class PhysicalMixedPlumbing:
    def __init__(self, *, p_ns, p_r, omega, b=sp.S.One):
        self.chiral = MixedNSRamondPlumbingBlock(p_ns=p_ns, p_r=p_r, omega=omega, b=b)
        x = self.chiral
        # Reuse the established BPZ Grams only; vertices below are built
        # directly in the graded chiral product representation.
        self.bp = PhysicalNSRRBPZ(c=x.c, h=x.h_ns, beta2=x.beta_r, beta3=x.beta_ext)

    @lru_cache(None)
    def anti_form(self, eta):
        x = self.chiral
        return GeneralizedNRRWard(p_phi=0, form_parity=0, eta=eta,
            h_ns=sp.conjugate(x.h_ns), h_second=sp.conjugate(x.h_r),
            h_third=sp.conjugate(x.h_ext), beta_second=sp.conjugate(x.beta_r),
            beta_third=sp.conjugate(x.beta_ext), central_charge=sp.conjugate(x.c))

    @lru_cache(None)
    def vertex(self, ns_hword, ns_aword, internal, external, eta):
        """Unit c_eta full vertex; actual c_+=C_even/2, c_-=C_odd/2.

        Conjugating the anti Ward result evaluated at conjugated parameters
        changes its fixed convention i's but preserves analytic momenta.
        """
        if eta not in (1, -1):
            raise ValueError("eta must be +1 or -1")
        a, b = word_parity(ns_hword), word_parity(ns_aword)
        hform, aform = self.chiral.form(0, eta), self.anti_form(eta)
        result = sp.S.Zero
        for gh, ga, eh, ea in product((0, 1), repeat=4):
            coefficient = (GROUND_KETS[2*gh+ga, internal.family]
                           *GROUND_KETS[2*eh+ea, external.family])
            if coefficient == 0:
                continue
            # W_h W_a acting on an ambient ground tensor.
            coefficient *= (-1)**(word_parity(internal.aword)*gh
                                    +word_parity(external.aword)*eh)
            rh = (gh+word_parity(internal.hword)) % 2
            ra = (ga+word_parity(internal.aword)) % 2
            sh = (eh+word_parity(external.hword)) % 2
            # Group all holomorphic entries before all antiholomorphic
            # entries, including the outgoing NS bra self-term a*b.
            coefficient *= (-1)**(b*(a+rh+sh)+ra*sh)
            hvalue = hform.value(ns_hword, internal.hword, gh, external.hword, eh)
            avalue = sp.conjugate(aform.value(ns_aword,
                internal.aword, ga, external.aword, ea))
            result += coefficient*hvalue*avalue
        return sp.simplify(result)

    def bry_vertex(self, ns_hword, ns_aword, internal, external, even, odd):
        return sp.simplify(sum(c*self.vertex(ns_hword, ns_aword, internal, external, eta)/2
                               for eta, c in ((1, even), (-1, odd))))

    @lru_cache(None)
    def internal_data(self, ns_level, r_level, anti_ns_level, anti_r_level):
        hb, ab = self.bp.ns.basis(ns_level), self.bp.ans.basis(anti_ns_level)
        word = lambda st: tuple((kind, sp.Rational(mode, 2)) for kind, mode in st)
        ns_basis = tuple((word(h), word(a)) for h, a in product(hb, ab))
        ns_gram = sp.kronecker_product(self.bp.ns.gram_matrix(ns_level),
                                       self.bp.ans.gram_matrix(anti_ns_level))
        ns_inverse = (-1)**(ns_level*anti_ns_level)*np.asarray(ns_gram.inv().evalf(40), complex)
        r_basis, r_inverse = self.bp.ramond_basis_inverse(0, r_level, anti_r_level)
        r_states = tuple(FullRamondState(h, a, e) for h, a, e in r_basis)
        return ns_basis, r_states, ns_inverse, r_inverse

    @lru_cache(None)
    def vertex_array(self, levels, anti_levels, external, eta):
        ns, rs, _, _ = self.internal_data(*levels, *anti_levels)
        return np.array([[complex(self.vertex(hw, aw, st, external, eta))
                          for st in rs] for hw, aw in ns], dtype=complex)

    def coefficient(self, levels, anti_levels, external1, external2, *,
                    etas=(1, 1), physical_lifts=(1, 1)):
        """Open full-state tensor, primary powers stripped.

        levels=(NS twice-level, R level), with independent anti levels.
        etas label unit c_eta factors on the two pants. Physical lift signs
        refer to the two marked sewn edges, not theta_1,...,theta_4.
        """
        if any(t not in (1, -1) for t in (*etas, *physical_lifts)):
            raise ValueError("All eta and lift signs must be +/-1")
        levels, anti_levels = tuple(levels), tuple(anti_levels)
        _, states, gn, gr = self.internal_data(*levels, *anti_levels)
        left = self.vertex_array(levels, anti_levels, external1, etas[0])
        right = self.vertex_array(levels, anti_levels, external2, etas[1])
        ns_parity = (levels[0]+anti_levels[0]) % 2
        signs = np.array([(-1)**(ns_parity*st.parity)
            *physical_lifts[0]**ns_parity*physical_lifts[1]**st.parity for st in states])
        return complex(np.einsum("ab,ac,bd,cd->", left*signs[None, :],
                                 gn, gr, right, optimize=True))

    def close_external(self, levels, anti_levels, external_level, anti_external_level, *,
                       etas=(1, 1), physical_lifts=(1, 1, 1)):
        basis, inverse = self.bp.ramond_basis_inverse(1, external_level, anti_external_level)
        states = [FullRamondState(h, a, e) for h, a, e in basis]
        # At a nonzero local vertex, p_internal=p_NS+p_external, so the
        # two additional edge crossings reduce to (-1)^p_external.
        return sum((-1)**a.parity*physical_lifts[2]**a.parity*inverse[i, j]
            *self.coefficient(levels, anti_levels, a, b, etas=etas,
                               physical_lifts=physical_lifts[:2])
            for i, a in enumerate(states) for j, b in enumerate(states))
