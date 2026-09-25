"""Numerical version of the literal generalized NS-R-R Ward identities.

Words use ordinary mode numbers (half integers for NS). This is the same
ordered NR system as ramond_pbw_generalized_ward.GeneralizedNRRWard, with
complex arithmetic instead of symbolic simplification. It does not choose
a torus spin frame or a nonchiral sewing prescription.
"""
from __future__ import annotations

from functools import lru_cache
import cmath


def parity(word):
    return sum(k == 'G' for k, _ in word) % 2


def level(word):
    return sum(-m for _, m in word)


@lru_cache(None)
def binomial(x, n):
    value = 1.
    for j in range(n):
        value *= (x-j)/(j+1)
    return value


def add(out, key, value):
    if value:
        out[key] = out.get(key, 0j)+value
        if out[key] == 0:
            del out[key]


class Module:
    def __init__(self, h, c, beta=None):
        self.h, self.c = complex(h), complex(c)
        self.beta = None if beta is None else complex(beta)

    def bracket(self, a, b):
        k, m = a
        l, n = b
        if k == l == 'L':
            result = [(m-n, (('L', m+n),))]
            if m+n == 0:
                result.append((self.c*(m**3-m)/12, ()))
        elif k == 'L':
            result = [(m/2-n, (('G', m+n),))]
        elif l == 'L':
            result = [(m-n/2, (('G', m+n),))]
        else:
            result = [(2, (('L', m+n),))]
            if m+n == 0:
                result.append((self.c*(m*m-.25)/3, ()))
        return result

    @lru_cache(None)
    def canonical(self, word):
        if any(m >= 0 for _, m in word):
            raise ValueError('only negative modes can be canonicalized')
        key = lambda x: (int(x[0] == 'G'), x[1])
        for j in range(len(word)-1):
            a, b = word[j:j+2]
            if a == b and a[0] == 'G':
                return self.canonical(word[:j]+(('L', 2*a[1]),)+word[j+2:])
            if key(a) <= key(b):
                continue
            out = {}
            sign = -1 if a[0] == b[0] == 'G' else 1
            for w, v in self.canonical(word[:j]+(b, a)+word[j+2:]).items():
                add(out, w, sign*v)
            for coefficient, replacement in self.bracket(a, b):
                for w, v in self.canonical(word[:j]+replacement+word[j+2:]).items():
                    add(out, w, coefficient*v)
            return out
        return {word: 1.}

    @lru_cache(None)
    def act(self, k, m, word, ground):
        if m < 0:
            return {(w, ground): v for w, v in self.canonical(((k, m),)+word).items()}
        if not word:
            if k == 'L':
                return {((), ground): self.h} if m == 0 else {}
            if m != 0:
                return {}
            if self.beta is None:
                raise ValueError('the NS module has no zero mode')
            return {((), 1-ground): self.beta*cmath.exp(1j*cmath.pi/4)*1j**ground}
        a, rest = word[0], word[1:]
        out = {}
        sign = -1 if k == a[0] == 'G' else 1
        for (w, g), v in self.act(k, m, rest, ground).items():
            for ww, vv in self.canonical((a,)+w).items():
                add(out, (ww, g), sign*v*vv)
        for coefficient, replacement in self.bracket((k, m), a):
            if not replacement:
                add(out, (rest, ground), coefficient)
            else:
                for state, v in self.act(*replacement[0], rest, ground).items():
                    add(out, state, coefficient*v)
        return out

    def inner(self, left, right, *, hermitian=False):
        lw, lg = left
        rw, rg = right
        if level(lw) != level(rw) or (parity(lw)+lg-parity(rw)-rg) % 2:
            return 0j
        states = {(rw, rg): 1.}
        for k, m in lw:
            out = {}
            for (w, g), v in states.items():
                for state, coefficient in self.act(k, -m, w, g).items():
                    add(out, state, coefficient*v)
            states = out
        pairing = (1j)**lg if self.beta is not None else 1.
        if hermitian and self.beta is not None:
            pairing *= (-1j)**lg
        return pairing*states.get(((), lg), 0j)

    def clear(self):
        self.canonical.cache_clear()
        self.act.cache_clear()


class NRR:
    def __init__(self, h_ns, h_ext, h_r, beta_ext, beta_r, c=13.5,
                 *, eta=1, p_phi=0, form_parity=0):
        if eta not in (-1, 1) or p_phi not in (0, 1) or form_parity not in (0, 1):
            raise ValueError('invalid NRR form labels')
        self.modules = (Module(h_ns, c), Module(h_ext, c, beta_ext), Module(h_r, c, beta_r))
        self.eta, self.p_phi, self.f = eta, p_phi, form_parity

    def epsilon(self, w1, w3, g3):
        return -1j*(-1)**(self.p_phi+parity(w1)+parity(w3)+g3+1)

    def pair(self, w1, states2, states3):
        return sum(c2*c3*self.value(w1, w2, g2, w3, g3)
                   for (w2, g2), c2 in states2.items() for (w3, g3), c3 in states3.items())

    def first(self, states1, w2, g2, w3, g3):
        return sum(c*self.value(w1, w2, g2, w3, g3) for (w1, _), c in states1.items())

    @staticmethod
    def cutoff(*levels):
        return int(max(0, *levels))+4

    @lru_cache(None)
    def value(self, w1, w2, g2, w3, g3):
        if (self.p_phi+parity(w1)+parity(w2)+g2+parity(w3)+g3) % 2 != self.f:
            return 0j
        a, b, c = self.modules
        if w1 and w1[0][0] == 'G':
            r, rest = -w1[0][1], w1[1:]
            epsilon, out = self.epsilon(rest, w3, g3), 0j
            for p in range(self.cutoff(r+level(rest), level(w2), level(w3))+1):
                out += binomial(r, p)*self.pair(rest, b.act('G', p, w2, g2), {(w3, g3): 1.})
                if p:
                    out -= binomial(.5, p)*(-1)**p*self.first(a.act('G', p-r, rest, 0), w2, g2, w3, g3)
                out -= epsilon*binomial(.5, p)*(-1)**p*self.pair(
                    rest, {(w2, g2): 1.}, c.act('G', r-.5+p, w3, g3))
            return out
        if w2:
            k, mode = w2[0]
            rest, n = w2[1:], int(-mode)
            if k == 'L':
                if n == 1:
                    power = a.h+level(w1)-b.h-level(rest)-c.h-level(w3)
                    return power*self.value(w1, rest, g2, w3, g3)
                out = 0j
                for p in range(self.cutoff(level(w1)-n, level(w3)+1)+1):
                    ward = binomial(n-2+p, n-2)
                    out += ward*self.first(a.act('L', n+p, w1, 0), rest, g2, w3, g3)
                    out += ward*(-1)**n*self.pair(w1, {(rest, g2): 1.}, c.act('L', p-1, w3, g3))
                return out
            out, epsilon = 0j, self.epsilon(w1, w3, g3)
            for p in range(self.cutoff(n+level(rest), level(w1)-n+.5, level(w3))+1):
                ward = binomial(.5-n, p)
                out += ward*(-1)**p*self.first(a.act('G', p+n-.5, w1, 0), rest, g2, w3, g3)
                out += epsilon*ward*(-1)**(n+p)*self.pair(w1, {(rest, g2): 1.}, c.act('G', p, w3, g3))
                if p:
                    out -= binomial(.5, p)*self.pair(w1, b.act('G', p-n, rest, g2), {(w3, g3): 1.})
            return out
        if w1:
            k, mode = w1[0]
            rest, n = w1[1:], int(-mode)
            if k != 'L':
                raise AssertionError('outer G must have been removed')
            out = self.pair(rest, {((), g2): 1.}, c.act('L', n, w3, g3))
            for m in range(-1, n+1):
                out += binomial(n+1, m+1)*self.pair(rest, b.act('L', m, (), g2), {(w3, g3): 1.})
            return out
        if w3:
            k, mode = w3[0]
            rest, m = w3[1:], int(-mode)
            if k == 'L':
                return (c.h+level(rest)+m*b.h-a.h)*self.value((), (), g2, rest, g3)
            out, epsilon = 0j, self.epsilon((), rest, g3)
            for p in range(self.cutoff(m+level(rest))+1):
                out += self.pair((), b.act('G', p, (), g2), {(rest, g3): 1.})*binomial(.5-m, p)/epsilon
                out -= (-1)**p*binomial(.5, p)*self.first(a.act('G', m-.5+p, (), 0), (), g2, rest, g3)/epsilon
                if p:
                    out -= binomial(.5, p)*(-1)**p*self.pair((), {((), g2): 1.}, c.act('G', -m+p, rest, g3))
            return out
        return {(0, 0): 1., (1, 1): self.eta, (0, 1): 1., (1, 0): 1j*self.eta}[g2, g3]

    def clear(self):
        self.value.cache_clear()
        for module in self.modules:
            module.clear()
