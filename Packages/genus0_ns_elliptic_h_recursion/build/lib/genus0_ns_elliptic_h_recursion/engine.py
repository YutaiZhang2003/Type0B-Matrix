"""Fixed-c, common-weight NS elliptic recursion in human-note conventions.

Numerical port of the independently audited research engine. Recursion
states retain exact integer shift labels, not rounded shifted weights.
The engine is private; the public API establishes a fixed precision and
clears its per-instance memoization after extracting the coefficient table.
"""

from functools import lru_cache

import mpmath as mp


class KacPoleError(ZeroDivisionError):
    """A Kac/common-weight pole or null-norm collision needs a limiting prescription."""


class RecursionEngine:
    def __init__(self, b, external, internal, tolerance):
        self.b, self.external, self.internal = b, tuple(external), tuple(internal)
        self.m, self.Q2, self.tolerance = len(internal), (b+1/b)**2, tolerance
        self.minimum_denominator = mp.inf
        self.minimum_pole = None
        self._memoized = []
        for name in ("pole", "norm_slope", "fusion", "weights", "residue", "coefficient"):
            cached = lru_cache(None)(getattr(self, name))
            setattr(self, name, cached)
            self._memoized.append(cached)

    def clear(self):
        for function in self._memoized:
            function.cache_clear()

    def pole(self, r, s):
        return (self.Q2-(r*self.b+s/self.b)**2)/8

    def norm_slope(self, r, s):
        result = mp.mpf(2)**(r*s-2)
        for u in range(1-r, r+1):
            for v in range(1-s, s+1):
                if (u+v)%2 == 0 and (u,v) not in ((0,0),(r,s)):
                    factor = u*self.b+v/self.b
                    if abs(factor) <= self.tolerance:
                        raise KacPoleError(f"colliding null-norm factors at (r,s)=({r},{s}), (u,v)=({u},{v})")
                    result /= factor
        return result

    def fusion(self, r, s, alpha, x, y):
        sites = {(u,v) for u in range(1-r,r,2) for v in range(1-s,s,2)
                 if (u+v-r-s-2*(1-alpha))%4 == 0}
        result = mp.mpf(1)
        lx, ly = self.Q2-8*x, self.Q2-8*y
        while sites:
            u,v = min(sites)
            sites.remove((u,v))
            if (u,v) == (0,0):
                result *= y-x
            else:
                sites.remove((-u,-v))
                linear = u*self.b+v/self.b
                result *= ((lx+linear**2-ly)**2-4*lx*linear**2)/64
        return result

    def weights(self, state):
        if state is None:
            return self.internal
        edge,r,s,shifts = state
        base = self.pole(r,s)+mp.mpf(r*s)/2
        return tuple(base+self.internal[j]-self.internal[edge]
                     +mp.mpf(shifts[j]-shifts[edge])/2 for j in range(self.m))

    def residue(self, edge, r, s, alpha_left, alpha_right, shifts):
        pole = self.pole(r,s)
        def weight(j):
            return pole+self.internal[j]-self.internal[edge]+mp.mpf(shifts[j]-shifts[edge])/2
        left = (self.external[0],self.external[1]) if edge == 0 else (weight(edge-1),self.external[edge+1])
        right = (self.external[-1],self.external[-2]) if edge == self.m-1 else (weight(edge+1),self.external[edge+2])
        return ((-1)**(r*s)*self.norm_slope(r,s)
                *self.fusion(r,s,alpha_left,*left)*self.fusion(r,s,alpha_right,*right))

    def coefficient(self, levels, state=None):
        if not any(levels):
            return mp.mpf(1)
        epsilon = tuple(n%2 for n in levels)
        labels = (epsilon[0],)+tuple(a^b for a,b in zip(epsilon,epsilon[1:]))+(epsilon[-1],)
        shifts = (0,)*self.m if state is None else state[-1]
        weights = self.weights(state)
        result = mp.mpf(0)
        for edge,available in enumerate(levels):
            for r in range(1,available+1):
                for s in range(1,available//r+1):
                    if (r+s)%2:
                        continue
                    product = r*s
                    child_levels = list(levels)
                    child_levels[edge] -= product
                    child_shifts = list(shifts)
                    child_shifts[edge] += product
                    child_shifts = tuple(v-child_shifts[0] for v in child_shifts)
                    child = self.coefficient(tuple(child_levels),(edge,r,s,child_shifts))
                    if child == 0:
                        continue
                    denominator = weights[edge]-self.pole(r,s)
                    if abs(denominator) < self.minimum_denominator:
                        self.minimum_denominator = abs(denominator)
                        self.minimum_pole = {"edge":edge+1,"r":r,"s":s,"twice_levels":levels,
                                             "denominator":denominator}
                    if abs(denominator) <= self.tolerance:
                        raise KacPoleError(f"small recursion denominator on edge {edge+1}, (r,s)=({r},{s}), twice_levels={levels}")
                    endpoint = 4**product if self.m == 1 else (2**product if edge in (0,self.m-1) else 1)
                    result += endpoint*self.residue(edge,r,s,labels[edge],labels[edge+1],shifts)*child/denominator
        return result
