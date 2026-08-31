"""Arbitrary-order pillow coordinates and the proposed unit-seed NS recursion.

Series keys are twice the powers of p_i.  All coordinate units have even
keys; the sewn NS block also has odd keys.  Uses the human note's actual
central charge, geometric Lambda^(c), and unit-seed H throughout.
"""

from functools import lru_cache
from math import isqrt

import mpmath as mp
import sympy as sp


def indices(count, maximum):
    if count == 0:
        yield ()
        return
    for first in range(maximum+1):
        for tail in indices(count-1, maximum-first):
            yield (first,)+tail


class Series:
    def __init__(self, count, degree, symbolic=False):
        self.m, self.bound, self.symbolic = count, 2*degree, symbolic
        self.zero = (0,)*count
        self.scalar_one = sp.S.One if symbolic else mp.mpf(1)
        self.one = {self.zero:self.scalar_one}

    def add(self, left, right, scale=1):
        out = dict(left)
        for key,value in right.items():
            out[key] = out.get(key,0)+scale*value
        return {k:v for k,v in out.items() if v != 0}

    def mul(self, left, right, bound=None):
        bound = self.bound if bound is None else bound
        if bound < 0:
            return {}
        out = {}
        ls = [(k,v,sum(k)) for k,v in left.items() if sum(k)<=bound]
        rs = [(k,v,sum(k)) for k,v in right.items() if sum(k)<=bound]
        for k,v,n in ls:
            for l,w,t in rs:
                if n+t<=bound:
                    key = tuple(a+b for a,b in zip(k,l))
                    out[key] = out.get(key,0)+v*w
        return {k:v for k,v in out.items() if v != 0}

    def shift(self, series, key, scale=1):
        return {tuple(a+b for a,b in zip(k,key)):scale*v
                for k,v in series.items() if sum(k)+sum(key)<=self.bound}

    def power(self, series, exponent, bound=None):
        bound = self.bound if bound is None else bound
        assert series.get(self.zero,0)==1
        rest = {k:v for k,v in series.items() if k != self.zero and sum(k)<=bound}
        if not rest or exponent == 0:
            return dict(self.one)
        minimum = min(map(sum,rest))
        out, term, coefficient = dict(self.one), dict(self.one), self.scalar_one
        for n in range(1,bound//minimum+1):
            coefficient = coefficient*(exponent-n+1)/n
            if coefficient == 0:
                break
            term = self.mul(term,rest,bound)
            out = self.add(out,term,coefficient)
        return out

    def factor(self, key, exponent, sign=1):
        if sum(key)>self.bound:
            return dict(self.one)
        return self.power({self.zero:self.scalar_one,key:sign*self.scalar_one},exponent)


class PillowMap:
    def __init__(self, count, degree, symbolic=False):
        self.s = Series(count,degree,symbolic)
        self.m, self.degree = count,degree
        s = self.s
        qkey = (2,)*count
        scale_key = lambda key,n:tuple(n*v for v in key)
        self.zunit = dict(s.one)
        for n in range(1,degree+1):
            for power,exponent in ((2*n,8),(2*n-1,-8)):
                self.zunit = s.mul(self.zunit,s.factor(scale_key(qkey,power),exponent))
        self.yunits = []
        self.mobile_factors = []
        for j in range(1,count):
            left = tuple(2 if i<j else 0 for i in range(count))
            right = tuple(0 if i<j else 2 for i in range(count))
            y = s.factor(left,2)
            for n in range(1,degree+1):
                factors = (
                    (scale_key(qkey,2*n),4),
                    (scale_key(qkey,2*n-1),-4),
                    (tuple(a+2*n*q for a,q in zip(left,qkey)),2),
                    (tuple(a+(2*n-1)*q for a,q in zip(right,qkey)),2),
                    (tuple(a+(2*n-1)*q for a,q in zip(left,qkey)),-2),
                    (tuple(a+(2*n-2)*q for a,q in zip(right,qkey)),-2),
                )
                for key,exponent in factors:
                    y = s.mul(y,s.factor(key,exponent))
            self.yunits.append(y)
            t = s.shift(y,right,4)
            z_over_t = s.shift(s.mul(self.zunit,s.power(y,-1)),left,4)
            self.mobile_factors.append(s.mul(s.add(s.one,t,-1),s.add(s.one,z_over_t,-1)))
        self.xunits = []
        previous = self.zunit
        for y in self.yunits:
            self.xunits.append(s.mul(previous,s.power(y,-1)))
            previous = y
        self.xunits.append(previous)
        self.one_minus_z = s.add(s.one,s.shift(self.zunit,qkey,16),-1)
        self.theta3 = dict(s.one)
        self.theta3_q2 = dict(s.one)
        for n in range(1,isqrt(degree)+1):
            self.theta3 = s.add(self.theta3,s.shift(s.one,scale_key(qkey,n*n),2))
            self.theta3_q2 = s.add(self.theta3_q2,s.shift(s.one,scale_key(qkey,2*n*n),2))
        self.inverse_character = dict(s.one)
        for n in range(1,degree+1):
            self.inverse_character = s.mul(self.inverse_character,s.factor(scale_key(qkey,2*n),3*s.scalar_one/4,-1))
        self.inverse_character = s.mul(self.inverse_character,s.power(self.theta3_q2,-1))

    def prefactor(self,c,external,internal):
        s = self.s
        d,h = external,internal
        theta_exponent = c/2-4*(d[0]+d[1]+d[-2]+d[-1])-2*sum(d[2:-2])
        factors = [(self.zunit,h[0]-c/24),
                   (self.theta3,-theta_exponent),
                   (self.one_minus_z,-c/24+d[1]+d[-2])]
        factors.extend((unit,h[i+1]-h[i]) for i,unit in enumerate(self.yunits))
        factors.extend((unit,d[i+2]/2) for i,unit in enumerate(self.mobile_factors))
        result = dict(s.one)
        for unit,power in factors:
            result = s.mul(result,s.power(unit,power))
        return s.mul(result,self.inverse_character)

    def pullback(self,plane,c,external,internal):
        s = self.s
        powers = {}
        for edge in range(self.m):
            for n in range(s.bound+1):
                powers[edge,n] = s.power(self.xunits[edge],n*s.scalar_one/2)
        descendant = {}
        for levels,value in plane.items():
            room = s.bound-sum(levels)
            unit = dict(s.one)
            for edge,n in enumerate(levels):
                unit = s.mul(unit,powers[edge,n],room)
            factor = 4**levels[0] if self.m==1 else 2**(levels[0]+levels[-1])
            descendant = s.add(descendant,s.shift(unit,levels,factor*value))
        return s.mul(self.prefactor(c,external,internal),descendant)


class NSEllipticRecursion:
    """Memoized unit-seed recursion with exact discrete weight-shift keys."""

    def __init__(self,b,external,internal,symbolic=False):
        self.b,self.external,self.internal = b,tuple(external),tuple(internal)
        self.symbolic,self.m = symbolic,len(internal)
        self.one = sp.S.One if symbolic else mp.mpf(1)
        self.Q2 = (b+1/b)**2
        self.minimum_denominator = mp.inf

    def simplify(self,value):
        if not self.symbolic:
            return value
        numerator,denominator = sp.cancel(value).as_numer_denom()
        # Keep shared pole factors visible before adding the next residues.
        # Expanded denominators otherwise produce huge spurious products in
        # SymPy's rational-function simplifier, even at total degree three.
        return numerator/sp.factor(denominator)

    @lru_cache(maxsize=None)
    def pole(self,r,s):
        return self.simplify((self.Q2-(r*self.b+s/self.b)**2)/8)

    @lru_cache(maxsize=None)
    def norm_slope(self,r,s):
        result = self.one*2**max(0,r*s-2)/2**max(0,2-r*s)
        for u in range(1-r,r+1):
            for v in range(1-s,s+1):
                if (u+v)%2==0 and (u,v) not in ((0,0),(r,s)):
                    result /= u*self.b+v/self.b
        return self.simplify(result)

    @lru_cache(maxsize=None)
    def fusion(self,r,s,alpha,x,y):
        sites = {(u,v) for u in range(1-r,r,2) for v in range(1-s,s,2)
                 if (u+v-r-s-2*(1-alpha))%4==0}
        result = self.one
        lx,ly = self.Q2-8*x,self.Q2-8*y
        while sites:
            u,v = min(sites)
            sites.remove((u,v))
            if (u,v)==(0,0):
                result *= y-x
                continue
            sites.remove((-u,-v))
            linear = u*self.b+v/self.b
            result *= ((lx+linear**2-ly)**2-4*lx*linear**2)/64
        return self.simplify(result)

    @lru_cache(maxsize=None)
    def weights(self,state):
        if state is None:
            return self.internal
        edge,r,s,shifts = state
        base = self.pole(r,s)+self.one*r*s/2
        return tuple(self.simplify(base+self.internal[j]-self.internal[edge]+self.one*(shifts[j]-shifts[edge])/2)
                     for j in range(self.m))

    @lru_cache(maxsize=None)
    def residue(self,edge,r,s,alpha_left,alpha_right,shifts):
        pole = self.pole(r,s)
        def weight(j):
            return self.simplify(pole+self.internal[j]-self.internal[edge]+self.one*(shifts[j]-shifts[edge])/2)
        left = (self.external[0],self.external[1]) if edge==0 else (weight(edge-1),self.external[edge+1])
        right = (self.external[-1],self.external[-2]) if edge==self.m-1 else (weight(edge+1),self.external[edge+2])
        return self.simplify((-1)**(r*s)*self.norm_slope(r,s)
                            *self.fusion(r,s,alpha_left,*left)*self.fusion(r,s,alpha_right,*right))

    @lru_cache(maxsize=None)
    def coefficient(self,levels,state=None):
        if not any(levels):
            return self.one
        parities = tuple(n%2 for n in levels)
        labels = (parities[0],)+tuple(a^b for a,b in zip(parities,parities[1:]))+(parities[-1],)
        shifts = (0,)*self.m if state is None else state[-1]
        weights = self.weights(state)
        result = 0*self.one
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
                    if not self.symbolic:
                        self.minimum_denominator = min(self.minimum_denominator,abs(denominator))
                    endpoint = 4**product if self.m==1 else (2**product if edge in (0,self.m-1) else 1)
                    term = endpoint*self.residue(edge,r,s,labels[edge],labels[edge+1],shifts)*child/denominator
                    # Reduce pairwise: expanding a sum of all pole terms at
                    # once creates the product of their denominators first.
                    result = self.simplify(result+term) if self.symbolic else result+term
        return result
