"""RR-pair OPE on an NS bridge, with NS/R torus handle traces.

The two cutoffs are independent. These are chiral tensors, before the
nonchiral BRY external-state contraction or any momentum integration.
The handle Ward/Gram engine is the one used by the TT collision calculation;
the external RR Ward vector is evaluated afresh here.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import sys

import numpy as np
from scipy.linalg import solve_triangular
import sympy as sp

CODE = Path(__file__).resolve().parents[1]
for directory in (CODE/'type0b_genus1/ope_channel', CODE/'double_virasoro/nsrr'):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from bridge_descendants import nr, W, ns_edge, bridge_factor, cylinder_matrix, K
from ramond_pbw_generalized_ward import GeneralizedNRRWard, word_parity

G_HALF = (('G', -sp.Rational(1, 2)),)


def ward_word(word):
    return tuple((m.kind, sp.Rational(m.twice_index, 2)) for m in word)


def _cutoff(value, name):
    if not isinstance(value, int) or value < 0:
        raise ValueError(name+' must be a nonnegative integer')
    return value


class RRPairVertex:
    """rho(NS_bridge at infinity, R_z at 1, R_0 at 0).

    Both R momenta equal omega for reflection; distinct values are supported
    for normalization tests. G0 is reduced in the ground representation,
    rather than being mistaken for a positive-level descendant.
    """

    def __init__(self, bridge, omega, *, other_omega=None):
        self.bridge, self.omega = sp.sympify(bridge), sp.sympify(omega)
        self.other = self.omega if other_omega is None else sp.sympify(other_omega)
        self.h, self.c = (1+self.bridge**2)/2, sp.Rational(27, 2)
        self.beta_z, self.beta_0 = sp.I*self.other/sp.sqrt(2), sp.I*self.omega/sp.sqrt(2)
        self.h_z, self.h_0 = self.c/24-self.beta_z**2, self.c/24-self.beta_0**2

    @lru_cache(None)
    def form(self, parity, eta):
        return GeneralizedNRRWard(p_phi=0, form_parity=parity, eta=eta,
            h_ns=self.h, h_second=self.h_z, h_third=self.h_0,
            beta_second=self.beta_z, beta_third=self.beta_0, central_charge=self.c)

    def value(self, ns_word, *, eta, grounds=(0, 0), word_z=(), word_0=(),
              form_parity=None):
        if any(g not in (0, 1) for g in grounds):
            raise ValueError('ground labels must be 0 or 1')
        parity = (word_parity(ns_word)+word_parity(word_z)+word_parity(word_0)
                  +sum(grounds)) % 2 if form_parity is None else form_parity
        form = self.form(parity, eta)
        if word_0 == (('G', 0),):
            terms = form.modules[2].act('G', sp.S.Zero, (), grounds[1])
            return sum(c*form.value(ns_word, word_z, grounds[0], w, g)
                       for (w, g), c in terms.items())
        return form.value(ns_word, word_z, grounds[0], word_0, grounds[1])

    def vector(self, basis, **kwargs):
        return np.array([complex(self.value(ward_word(w), **kwargs)) for w in basis])


class HandleTraces:
    """Energy-independent, unsummed one-point trace coefficients.

    theta3 = NS trace; theta4 = NS supertrace; theta2 = R trace;
    theta1 = R supertrace. Both HJS signs are retained in the R handle.
    This identifies the spin on the closed handle of the OPE channel;
    it does not identify spin lifts in the mixed necklace channel.
    """

    def __init__(self, bridge, loop, *, ns_twice_cutoff=4, r_cutoff=2):
        self.bridge, self.loop = float(bridge), float(loop)
        if self.bridge <= 0 or self.loop <= 0:
            raise ValueError('use positive continuum momenta, away from endpoint null states')
        self.h = (1+self.bridge**2)/2
        self.ns_cutoff = _cutoff(ns_twice_cutoff, 'NS twice cutoff')
        self.r_cutoff = _cutoff(r_cutoff, 'R cutoff')
        self.rvertex = nr.Vertex(self.loop, self.loop, self.h)

    @lru_cache(None)
    def trace(self, word, delta, eta=1):
        if delta not in (1, 2, 3, 4) or eta not in (-1, 1):
            raise ValueError('invalid theta characteristic or HJS sign')
        if delta in (3, 4):
            evaluate = W['_three_point_ward_cached']
            hl = (1+self.loop**2)/2
            out = []
            for level in range(self.ns_cutoff+1):
                basis, scale, chol = ns_edge(level, self.loop)
                matrix = np.array([[evaluate(a, word, b, hl, self.h, hl, 13.5)
                                    for b in basis] for a in basis], complex)
                matrix *= scale[:, None]*scale[None, :]
                matrix = solve_triangular(chol, matrix, lower=True, check_finite=False)
                matrix = solve_triangular(chol, matrix.T, lower=True, check_finite=False).T
                out.append(np.trace(matrix)*((-1)**level if delta == 4 else 1))
        else:
            out = []
            for level in range(self.r_cutoff+1):
                edge = nr.edge(2*level, self.loop)
                matrix = self.rvertex.matrix(edge, edge, word)[int(eta == 1)]
                signs = (-1.)**edge[3] if delta == 1 else np.ones(len(edge[0]))
                out.append(np.diag(matrix)@signs)
        answer = np.array(out, complex)
        if not np.isfinite(answer).all():
            raise ArithmeticError('nonfinite torus trace')
        return answer

    def levels(self, delta):
        return (np.arange(self.ns_cutoff+1)/2 if delta in (3, 4)
                else np.arange(self.r_cutoff+1))

    def exponent(self, delta):
        return self.loop**2/2-(1/16 if delta in (3, 4) else 0)

    def close(self):
        self.trace.cache_clear()
        self.rvertex.middle.cache_clear()
        self.rvertex.evaluate.cache_clear()


class TorusRROPEBlock:
    """Independent RR fusion and handle truncations, with flat transport."""

    def __init__(self, bridge, loop, omega, *, bridge_order=4,
                 ns_twice_cutoff=4, r_cutoff=2, other_omega=None):
        self.order = _cutoff(bridge_order, 'bridge order')
        self.vertex = RRPairVertex(bridge, omega, other_omega=other_omega)
        self.handle = HandleTraces(bridge, loop, ns_twice_cutoff=ns_twice_cutoff,
                                   r_cutoff=r_cutoff)

    @lru_cache(None)
    def plane(self, *, delta, eta=1, handle_eta=1, grounds=(0, 0),
              word_z=(), word_0=(), form_parity=None):
        """Arrays [bridge family, relative level, handle level].

        The leading half-integer bridge contains its inverse norm 1/(2h).
        No compensating 2h or structure constant is hidden in this bank.
        """
        result = np.zeros((2, self.order+1, len(self.handle.levels(delta))), complex)
        for family in (0, 1):
            for j in range(self.order+1):
                basis, scale, chol = bridge_factor(2*j+family, self.handle.bridge)
                sphere = self.vertex.vector(basis, eta=eta, grounds=grounds,
                    word_z=word_z, word_0=word_0, form_parity=form_parity)
                solved = solve_triangular(chol, sphere*scale, lower=True, check_finite=False)
                solved = scale*solve_triangular(chol.T, solved, lower=False, check_finite=False)
                traces = np.array([self.handle.trace(w, delta, handle_eta) for w in basis])
                result[family, j] = solved@traces
        if not np.isfinite(result).all():
            raise ArithmeticError('nonfinite RR OPE coefficients')
        return result

    @lru_cache(None)
    def local(self, *, mode=None, **kwargs):
        """Flat-coordinate polynomials, with G_-1/G0 mixing at puncture 0.

        mode=None: primary; mode=0: G0; mode=1: G_-1. The returned family f
        multiplies K^(h+f/2) z^(h+f/2-h_z-h_0-mode), with mode=None
        interpreted as 0 in this exponent. The handle Casimir is separate.
        """
        if mode not in (None, 0, 1):
            raise ValueError('mode must be None, 0, or 1')
        if 'word_0' in kwargs or 'word_z' in kwargs:
            raise ValueError('local transport fixes the external words')
        word = () if mode is None else (('G', -mode),)
        plane = self.plane(word_0=word, **kwargs)
        h, hz, h0 = map(complex, (self.vertex.h, self.vertex.h_z, self.vertex.h_0))
        out = np.zeros_like(plane)
        for f in (0, 1):
            matrix = cylinder_matrix(h, hz, h0+(mode or 0), f, self.order)
            out[f] = matrix@plane[f]
        if mode == 1:
            zero = self.local(mode=0, **kwargs)
            out[:, 1:] += .75*K*zero[:, :-1]
        return out

    def evaluate(self, tau, z, *, mode=None, handle_ground_only=False, **kwargs):
        """Evaluate both bridge families as separate chiral numbers."""
        if complex(tau).imag <= 0 or z == 0:
            raise ValueError('positive torus height and separated punctures required')
        delta = kwargs['delta']
        coeff = self.local(mode=mode, **kwargs)
        levels = self.handle.levels(delta)
        if handle_ground_only:
            coeff, levels = coeff[..., :1], levels[:1]
        logq = K*complex(tau)
        q = np.exp(logq*(self.handle.exponent(delta)+levels))
        powers = complex(z)**np.arange(self.order+1)
        h, hz, h0 = map(complex, (self.vertex.h, self.vertex.h_z, self.vertex.h_0))
        return np.array([np.exp((h+f/2)*np.log(K))
            *complex(z)**(h+f/2-hz-h0-(mode or 0))*powers@coeff[f]@q
            for f in (0, 1)])

    def close(self):
        self.plane.cache_clear()
        self.local.cache_clear()
        self.handle.close()
