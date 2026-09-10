"""Coefficient-level CCY c-recursion for the punctured theta graph.

The ordered vertices are (h1,hL,h3), (h1,hR,h3), (hR,hpsi,hL).
The returned reduced block has the universal large-c vacuum factor removed.
All finite-c descendant contributions are obtained by null-vector residues;
there are no finite-c Gram matrices or PBW contractions in this module.
"""

from __future__ import annotations

from functools import lru_cache
from fractions import Fraction
from math import comb, factorial
from collections import Counter
import mpmath


def downward_indices(cutoff, degree_weights=(1, 1, 1, 1)):
    """Nonnegative integer levels in a weighted total-degree truncation."""
    if len(degree_weights) != 4 or min(degree_weights) <= 0:
        raise ValueError("four positive degree weights are required")
    out = []
    for a in range(int(cutoff / degree_weights[0]) + 1):
        for b in range(int((cutoff-degree_weights[0]*a) / degree_weights[1]) + 1):
            for c in range(int((cutoff-degree_weights[0]*a-degree_weights[1]*b) / degree_weights[2]) + 1):
                bound = cutoff-degree_weights[0]*a-degree_weights[1]*b-degree_weights[2]*c
                for d in range(int(bound / degree_weights[3]) + 1):
                    out.append((a,b,c,d))
    return tuple(sorted(out, key=lambda n: (sum(n), n)))


class PuncturedCCY:
    """Multiprecision CCY recursion with fixed external weight.

    ``reduced_series(indices=...)`` accepts any downward-closed finite set.
    Its keys are the *integer Virasoro descendant levels* on the four edges.
    Precision is local to this instance.  To regulate colliding poles, pass
    regulated generic input weights; exact poles are never silently dropped.
    """

    def __init__(self, central_charge, weights, external_weight, dps=80):
        self.mp = mpmath.mp.clone()
        self.mp.dps = int(dps)
        self.c = self._number(central_charge)
        if len(weights) != 4:
            raise ValueError("weights must be (h1,hL,hR,h3)")
        self.weights = tuple(self._number(h) for h in weights)
        self.external_weight = self._number(external_weight)
        self._coefficient = lru_cache(maxsize=None)(self._coefficient_uncached)
        self._global = lru_cache(maxsize=None)(self._global_uncached)
        self._pole = lru_cache(maxsize=None)(self._pole_uncached)
        self._rho = lru_cache(maxsize=None)(self._rho_uncached)
        self._norm = lru_cache(maxsize=None)(self._norm_uncached)

    def _number(self,value):
        if isinstance(value,Fraction):
            return self.mp.mpc(value.numerator)/value.denominator
        return self.mp.mpc(value)

    def _rising(self, x, n):
        value = self.mp.mpf(1)
        for j in range(n):
            value *= x+j
        return value

    def _falling(self, x, n):
        value = self.mp.mpf(1)
        for j in range(n):
            value *= x-j
        return value

    def _norm_uncached(self, h, n):
        return factorial(n)*self._rising(2*h,n)

    def _rho_uncached(self, i,j,k,h1,h2,h3):
        # CCY's closed SL(2) three-point matrix element; no Virasoro Ward sum.
        total = self.mp.mpc(0)
        for p in range(min(i,k)+1):
            total += (comb(i,p)*self._falling(2*h3+k-1,p)
                      * self._falling(k,p)*self._rising(h3+h2-h1,k-p)
                      * self._rising(h1+h2-h3+p-k,i-p))
        return total*self._rising(h1+i-h2-j+1-h3-k,j)

    def _global_uncached(self, shifts, levels):
        h1,hL,hR,h3 = tuple(h+s for h,s in zip(self.weights,shifts))
        a,b,c,d = levels
        numerator = (self._rho(a,b,d,h1,hL,h3)
                     *self._rho(a,c,d,h1,hR,h3)
                     *self._rho(c,0,b,hR,self.external_weight,hL))
        denominator = (self._norm(h1,a)*self._norm(hL,b)
                       *self._norm(hR,c)*self._norm(h3,d))
        if not denominator:
            raise ZeroDivisionError("degenerate global Gram norm: use a specified generic-weight limit")
        return numerator/denominator

    def _fusion(self,r,s,b,top,bottom):
        q = b+1/b
        lt = self.mp.sqrt(q*q-4*top)
        lb = self.mp.sqrt(q*q-4*bottom)
        product = self.mp.mpc(1)
        for p in range(1-r,r,2):
            for t in range(1-s,s,2):
                shift = p*b+t/b
                product *= (lt+lb+shift)*(lt-lb+shift)/4
        return product

    def _pole_uncached(self, shifts, edge, r,s):
        weights = tuple(h+j for h,j in zip(self.weights,shifts))
        h = weights[edge]
        radical = (r-s)**2+4*(r*s-1)*h+4*h*h
        root = self.mp.sqrt(radical)
        roots = ((r*s-1+2*h+root)/(1-r*r),
                 (r*s-1+2*h-root)/(1-r*r))
        # Continue the noncancelling branch, including negative h and s=1.
        x = max(roots,key=abs)
        if not x:
            raise ZeroDivisionError("c-recursion pole at infinity requires a generic-weight limit")
        b = self.mp.sqrt(x)
        c = 13+6*(x+1/x)
        # Cancel universal (x-1)(x+1) factors before numerical evaluation.
        numerator = -12*x**(2*r*s-1)
        denominator = (1-r*r)*x*x-(1-s*s)
        factors = [(p,t) for p in range(1-r,r+1) for t in range(1-s,s+1)
                   if (p,t) not in ((0,0),(r,s))]
        for target in (-1,1):
            for j,(p,t) in enumerate(factors):
                if p and t == p*target:
                    denominator *= p
                    factors.pop(j)
                    break
            else:
                numerator *= x+target
        for p,t in factors:
            denominator *= p*x+t
        if not denominator:
            raise ZeroDivisionError("confluent Kac residue requires a generic-weight limit")
        h1,hL,hR,h3 = weights
        # Null slot infinity -> (zero,one); one -> (zero,infinity);
        # zero -> (infinity,one).  All three vertices use (infinity,1,0).
        pairs = (((h3,hL),(h3,hR)),
                 ((h3,h1),(hR,self.external_weight)),
                 ((h3,h1),(hL,self.external_weight)),
                 ((h1,hL),(h1,hR)))
        residue = numerator/denominator
        for top,bottom in pairs[edge]:
            residue *= self._fusion(r,s,b,top,bottom)
        return c,residue

    def _coefficient_uncached(self,c,shifts,levels):
        total = self._global(shifts,levels)
        for edge,k in enumerate(levels):
            for r in range(2,k+1):
                for s in range(1,k//r+1):
                    level = r*s
                    pole,residue = self._pole(shifts,edge,r,s)
                    if not residue:
                        continue
                    denominator = c-pole
                    if abs(denominator) <= self.mp.power(10,10-self.mp.dps)*max(1,abs(c),abs(pole)):
                        raise ZeroDivisionError("coincident or unresolved c-recursion poles require a generic-weight limit or more precision")
                    lower = list(levels)
                    lower[edge] -= level
                    shifted = list(shifts)
                    shifted[edge] += level
                    total += residue/denominator*self._coefficient(pole,tuple(shifted),tuple(lower))
        return total

    def reduced_coefficient(self,levels):
        levels = tuple(int(n) for n in levels)
        if len(levels) != 4 or min(levels) < 0:
            raise ValueError("four nonnegative descendant levels are required")
        return self._coefficient(self.c,(0,0,0,0),levels)

    def reduced_series(self,cutoff=None,*,indices=None,degree_weights=(1,1,1,1)):
        if indices is None:
            if cutoff is None:
                raise ValueError("supply cutoff or indices")
            indices = downward_indices(cutoff,degree_weights)
        return {tuple(n):self.reduced_coefficient(n) for n in indices}

    def full_series(self,cutoff=None,*,indices=None,degree_weights=(1,1,1,1)):
        if indices is None:
            if cutoff is None:
                raise ValueError("supply cutoff or indices")
            indices = downward_indices(cutoff,degree_weights)
        indices = tuple(tuple(n) for n in indices)
        reduced = self.reduced_series(indices=indices)
        seed = theta_vacuum_seed(max((a+b+d for a,b,c,d in indices if b==c),default=0))
        result = {}
        for n in indices:
            total = self.mp.mpc(0)
            for (a,b,d),coefficient in seed.items():
                m = (n[0]-a,n[1]-b,n[2]-b,n[3]-d)
                if min(m)>=0:
                    total += self._number(coefficient)*reduced[m]
            result[n] = total
        return result

    def cache_info(self):
        return {"coefficient":self._coefficient.cache_info()._asdict(),
                "global":self._global.cache_info()._asdict(),
                "pole":self._pole.cache_info()._asdict()}

    def clear_caches(self):
        """Release one branch's recursion state after its series is assembled."""
        for cache in (self._coefficient,self._global,self._pole,self._rho,self._norm):
            cache.cache_clear()


@lru_cache(maxsize=None)
def _vacuum_partitions(level,minimum=2):
    if level == 0:
        return ((),)
    return tuple((m,)+tail for m in range(minimum,level+1)
                 for tail in _vacuum_partitions(level-m,m))


@lru_cache(maxsize=None)
def _stress_covariance(slot_a,m,slot_b,n):
    """Covariances of L_-m/sqrt(c), from <T(z)T(w)>=c/(2(z-w)^4)."""
    if slot_a>slot_b:
        return _stress_covariance(slot_b,n,slot_a,m)
    if slot_a==slot_b:
        return Fraction(0)
    if (slot_a,slot_b)==(0,2):
        return Fraction(m*(m*m-1),12) if m==n else Fraction(0)
    if (slot_a,slot_b)==(0,1):
        return Fraction(m*(m*m-1)*comb(m-2,n-2),12) if m>=n else Fraction(0)
    return Fraction((-1)**m*factorial(m+n-1),12*factorial(m-2)*factorial(n-2))


@lru_cache(maxsize=None)
def _stress_wick(operators):
    if not operators:
        return Fraction(1)
    if len(operators)%2:
        return Fraction(0)
    first = operators[0]
    total = Fraction(0)
    for j in range(1,len(operators)):
        other = operators[j]
        covariance = _stress_covariance(*first,*other)
        if covariance:
            total += covariance*_stress_wick(operators[1:j]+operators[j+1:])
    return total


def _stress_norm(partition):
    value = Fraction(1)
    for m,count in Counter(partition).items():
        value *= factorial(count)*Fraction(m*(m*m-1),12)**count
    return value


@lru_cache(maxsize=None)
def _theta_vacuum_coefficient(a,b,c):
    total = Fraction(0)
    for first in _vacuum_partitions(a):
        for second in _vacuum_partitions(b):
            for third in _vacuum_partitions(c):
                operators = tuple((slot,m) for slot,part in enumerate((first,second,third)) for m in part)
                rho = _stress_wick(operators)
                if rho:
                    total += rho*rho/(_stress_norm(first)*_stress_norm(second)*_stress_norm(third))
    return total


def theta_vacuum_seed(cutoff):
    """Exact CCY large-c seed, integer theta exponents (q1,q2,q3).

    The large-c Virasoro algebra splits into global SL(2) and Gaussian
    stress-tensor oscillators.  Wick contraction computes the latter exactly;
    no finite-c vacuum Gram matrix, extrapolation, or rounded data is used.
    Splitting edge 2 pulls this seed back by q2=qL*qR.
    """
    result = {}
    for a in range(cutoff+1):
        for b in range(cutoff-a+1):
            for c in range(cutoff-a-b+1):
                value = _theta_vacuum_coefficient(a,b,c)
                if value:
                    result[(a,b,c)] = value
    return result
