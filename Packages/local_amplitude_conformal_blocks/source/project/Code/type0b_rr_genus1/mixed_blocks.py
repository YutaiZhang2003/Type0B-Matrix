"""Open Ramond-edge sewing tensors for a torus with two R punctures.

The local coordinates are inherited from the NS-R-R theta pants: cut the
third (Ramond) tube and leave its two states uncontracted. This is a marked
plumbing block, not the period-one necklace block or a string integrand.
All descendant matrix elements use the existing generalized NRR Ward
identities; there is no input from the tachyon amplitude or matrix model.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import sys

import numpy as np
import sympy as sp

CODE = Path(__file__).resolve().parents[1]
for directory in (CODE / "double_virasoro/nsrr", CODE / "c_Recursion"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from nsrr_genus2_block import HumanNSRRThetaOracle
from ramond_pbw_generalized_ward import GeneralizedNRRWard, RamondState


class MixedNSRamondPlumbingBlock:
    """Sew one NS edge and one R edge, keeping both R punctures open.

    ``omega`` is the external Liouville momentum, the same at both ends.
    ``p_ns,p_r`` are independent integration momenta. Mode indices in
    external ``RamondState`` words are ordinary integer R indices.
    NS truncation indices are twice-levels; R indices are integer levels.
    A parity insertion acts on the complete R state, including its ground.
    """

    def __init__(self, *, p_ns, p_r, omega, b=sp.S.One):
        self.b = sp.sympify(b)
        self.p_ns, self.p_r, self.omega = map(sp.sympify, (p_ns, p_r, omega))
        q = self.b + 1/self.b
        self.c = sp.simplify(sp.Rational(3, 2) + 3*q*q)
        self.h_ns = sp.simplify((q*q/4 + self.p_ns**2)/2)
        self.beta_r = sp.I*self.p_r/sp.sqrt(2)
        self.beta_ext = sp.I*self.omega/sp.sqrt(2)
        self.h_r = sp.simplify(self.c/24 - self.beta_r**2)
        self.h_ext = sp.simplify(self.c/24 - self.beta_ext**2)
        if self.p_r == 0:
            raise ValueError("The long-R integration edge must have nonzero momentum")
        self.oracle = HumanNSRRThetaOracle(central_charge=self.c, h_ns=self.h_ns,
            beta_r1=self.beta_r, beta_r2=self.beta_ext, form_parity=0,
            primary_parity=0, etas=(1, 1))

    @lru_cache(None)
    def form(self, parity, eta):
        return GeneralizedNRRWard(p_phi=0, form_parity=parity, eta=eta,
            h_ns=self.h_ns, h_second=self.h_r, h_third=self.h_ext,
            beta_second=self.beta_r, beta_third=self.beta_ext, central_charge=self.c)

    @lru_cache(None)
    def vertex(self, ns_twice_level, r_level, r_parity, state, form_parity, eta):
        ns_basis, _ = self.oracle.ns_basis_inverse(ns_twice_level)
        r_basis, _ = self.oracle.r_basis_inverse(0, r_level, r_parity)
        rho = self.form(form_parity, eta)
        value = np.zeros((len(ns_basis), len(r_basis)), dtype=complex)
        for i, ns in enumerate(ns_basis):
            word = self.oracle._ns_word(ns)
            for j, ramond in enumerate(r_basis):
                value[i, j] = complex(sp.N(rho.value(word,
                    ramond.word, ramond.ground, state.word, state.ground), 50))
        return value

    def coefficient(self, ns_twice_level, r_level, state1, state2, *,
                    forms=(0, 0), etas=(1, 1), ns_lift=1, r_parity_insertion=1,
                    r_parity=None):
        if any(x not in (-1, 1) for x in (*etas, ns_lift, r_parity_insertion)):
            raise ValueError("Signs must be +1 or -1")
        if any(f not in (0, 1) for f in forms):
            raise ValueError("Three-form parities must be 0 or 1")
        if min(ns_twice_level, r_level) < 0:
            raise ValueError("Descendant levels must be nonnegative")
        _, inverse_ns = self.oracle.ns_basis_inverse(ns_twice_level)
        ns_parity = ns_twice_level % 2
        result = 0j
        for parity in ((0, 1) if r_parity is None else (r_parity,)):
            if parity not in (0, 1):
                raise ValueError("R state parity must be 0 or 1")
            _, inverse_r = self.oracle.r_basis_inverse(0, r_level, parity)
            left = self.vertex(ns_twice_level, r_level, parity, state1, forms[0], etas[0])
            right = self.vertex(ns_twice_level, r_level, parity, state2, forms[1], etas[1])
            # Two sewn odd edges cross once. The external edge is open;
            # its extra crossings are inserted only in the closure check.
            orientation = (-1)**(ns_parity*parity)
            spin = ns_lift**ns_parity * r_parity_insertion**parity
            result += orientation*spin*np.einsum("ab,ac,bd,cd->", left,
                inverse_ns, inverse_r, right, optimize=True)
        return complex(result)

    def ground_matrix(self, ns_twice_level, r_level, **kwargs):
        states = [RamondState((), ground) for ground in (0, 1)]
        return np.array([[self.coefficient(ns_twice_level, r_level, a, b, **kwargs)
                          for b in states] for a in states])

    def series(self, max_twice_level, state1, state2, **kwargs):
        return {(n, r): self.coefficient(n, r, state1, state2, **kwargs)
                for n in range(max_twice_level+1)
                for r in range((max_twice_level-n)//2+1)}

    def evaluate(self, log_q_ns, log_q_r, max_twice_level, state1, state2, **kwargs):
        """Plane-plumbing block, including q_NS^h_NS q_R^h_R.

        The two puncture coordinates on each pant are plane coordinates.
        There is no separate Casimir subtraction on either sewing edge.
        ``evaluate_flat_ground`` supplies the ONE torus Casimir factor.
        All logarithms must be continued in the chosen marked patch.
        """
        values = self.series(max_twice_level, state1, state2, **kwargs)
        power_ns = complex(self.h_ns)
        power_r = complex(self.h_r)
        return np.exp(power_ns*log_q_ns+power_r*log_q_r)*sum(
            coefficient*np.exp(n*log_q_ns/2+r*log_q_r)
            for (n, r), coefficient in values.items())

    def evaluate_flat_ground(self, coordinates, max_twice_level, *, grounds=(0, 0),
                             log_derivative_product=None, **kwargs):
        """Transport a ground tensor to the period-one torus, locally.

        The normalization is Q^(-c/24) (f'_w f'_v)^(-h_ext) times the
        plane-plumbing block. The free boson/Majorana check fixes this
        factor without input from a string amplitude. Principal plumbing
        logs and derivative-product log are suitable only in the checked
        patch; this method does not implement global spin transport.
        """
        import cmath
        g = coordinates
        derivative_log = (cmath.log(g.derivative_w*g.derivative_v)
                          if log_derivative_product is None
                          else log_derivative_product)
        # Use tau, rather than log(Q), to retain the cylinder continuation.
        normalization = np.exp(-complex(self.c)/24*2j*np.pi*g.tau
                               -complex(self.h_ext)*derivative_log)
        return normalization*self.evaluate(cmath.log(g.q_ns), cmath.log(g.q_r),
            max_twice_level, *(RamondState((), e) for e in grounds), **kwargs)

    def closed_theta_component(self, ns_twice_level, r_level, external_level,
                               r_parity, external_parity, *, form_parity=0, etas=(1, 1)):
        """Re-sew the open edge, for comparison with the existing theta oracle."""
        states, inverse = self.oracle.r_basis_inverse(1, external_level, external_parity)
        coefficients = np.array([[self.coefficient(ns_twice_level, r_level, a, b,
            forms=(form_parity, form_parity), etas=etas, r_parity=r_parity)
            for b in states] for a in states])
        extra_crossings = (-1)**(external_parity*(ns_twice_level % 2+r_parity))
        return extra_crossings*np.einsum("ab,ab->", coefficients, inverse)
