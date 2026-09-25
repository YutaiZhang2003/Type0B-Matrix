"""Numerical evaluation of the frozen PBW Ward recursions at fixed weights.

The reference's symbolic high-level fallback builds polynomials in all
external weights. This module evaluates those same recursions numerically,
sharing each vertex's Ward cache across the entire matrix. Ground tensors,
normalization, parity ordering and Cholesky sewing match the frozen backend.
No reference module is modified.
"""
from functools import lru_cache
import inspect
import math
from types import FunctionType, SimpleNamespace
import sys
from pathlib import Path

import numpy as np
from scipy.linalg import solve_triangular

PARENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PARENT))
from ramond_descendants import enable_oracle
enable_oracle()
from ramond_algebra import ramond_sca as rsca
from ns_algebra import ns_sca as nsca


def _clone_actions(module):
    number = SimpleNamespace(S=SimpleNamespace(Zero=0., One=1.), sympify=lambda x:x,
        Integer=int, Rational=lambda x,y=1:float(x)/y, expand=lambda x:x)
    scope = dict(vars(module), sp=number)
    for name, wrapped in vars(module).items():
        f = getattr(wrapped, '__wrapped__', wrapped)
        if not inspect.isfunction(f) or f.__module__ != module.__name__:
            continue
        new = FunctionType(f.__code__, scope, f.__name__, f.__defaults__, f.__closure__)
        new.__kwdefaults__ = f.__kwdefaults__
        if hasattr(wrapped, 'cache_info'):
            new = lru_cache(maxsize=100000)(new)
        scope[name] = new
    return scope


R = _clone_actions(rsca)
N = _clone_actions(nsca)
# NS Mode.index is symbolic in the reference class; use doubled integer
# indices in the bracket so the arithmetic stays numerical.
def _ns_bracket(a, b, c):
    m, n = a.twice_index/2, b.twice_index/2
    k = a.twice_index+b.twice_index
    if a.kind == b.kind == 'L':
        result = [(m-n, nsca.Mode('L', k))] if m != n else []
        if k == 0: result.append((c*m*(m*m-1)/12, None))
    elif a.kind == 'L':
        result = [(m/2-n, nsca.Mode('G', k))]
    elif b.kind == 'L':
        result = [(m-n/2, nsca.Mode('G', k))]
    else:
        result = [(2., nsca.Mode('L', k))]
        if k == 0: result.append((c*(4*m*m-1)/12, None))
    return tuple((v,w) for v,w in result if v != 0)
N['_super_bracket'] = _ns_bracket
N['G'] = lambda x:nsca.Mode('G', int(round(2*float(x))))
N['L'] = lambda x:nsca.Mode('L', 2*int(x))


@lru_cache(maxsize=None)
def binomial(a, k):
    result = 1.
    for j in range(int(k)):
        result *= (a-j)/(j+1)
    return result


class Vertex:
    def __init__(self, pl, pr, h):
        self.pl, self.pr, self.h = pl, pr, h
        self.hl, self.hr, self.c = 13.5/24+pl*pl/2, 13.5/24+pr*pr/2, 13.5
        # A+sB of the reference polynomial ground tensor, without a basis change.
        self.ground = np.array([[[1,0],[(-1+1j)*pr/2,0]],
                                [[0,(1-1j)*pl/2],[0,pl*pr/2]]], complex)
        self.middle = lru_cache(maxsize=None)(self._middle)
        self.evaluate = lru_cache(maxsize=None)(self._evaluate)

    def _middle(self, word, left, right):
        if not word: return self.ground[left,right]
        mode, rest = word[0], word[1:]
        level = nsca.twice_level(rest)/2 if rest else 0.
        if mode.kind == 'L':
            n = -mode.twice_index//2
            return (-1)**n*(n*self.hr+self.h+level-self.hl)*self.middle(rest,left,right)
        k = -mode.twice_index/2
        value = np.zeros(2,complex)
        if k == .5:
            value += (1 if left == 0 else self.pl*self.pl/2)*self.middle(rest,left^1,right)
        value += (-1)**int(left+right+.5-k)*(1 if right == 0 else self.pr*self.pr/2)*self.middle(rest,left,right^1)
        for p in range(1, math.floor(k+level)+1):
            for output, coeff in N['act_mode'](nsca.Mode('G',int(2*(p-k))), {rest:1.}, h=self.h, c=self.c).items():
                value -= binomial(.5,p)*coeff*self.middle(output,left,right)
        return value

    def _evaluate(self, left, middle, right):
        if not left.word and not right.word:
            return self.middle(middle,left.ground_parity,right.ground_parity)
        is_left = bool(left.word)
        state = left if is_left else right
        first = state.word[0];n = -first.index
        rest = rsca.PBWState(state.word[1:],state.ground_parity)
        l,r = (rest,right) if is_left else (left,rest)
        value = np.zeros(2,complex)
        opposite = right if is_left else left
        weight = self.hr if is_left else self.hl
        sign = 1 if first.kind == 'L' else (-1)**(opposite.parity+rest.parity+1)
        for output, coeff in R['act_mode'](rsca.Mode(first.kind,n), {opposite:1.},h=weight,c=self.c).items():
            ll,rr = (rest,output) if is_left else (output,rest)
            value += sign*coeff*self.evaluate(ll,middle,rr)
        ml = nsca.twice_level(middle) if middle else 0
        if first.kind == 'L':
            maximum = n if is_left else ml//2
            for m in range(-1,maximum+1):
                factor = binomial(n+1 if is_left else 1-n,m+1)*(1 if is_left else -1)
                if factor == 0: continue
                for output, coeff in N['act_mode'](nsca.Mode('L',2*m),{middle:1.},h=self.h,c=self.c).items():
                    value += factor*coeff*self.evaluate(l,output,r)
        else:
            for twice in range(-1,ml+1,2):
                factor = binomial(n+.5 if is_left else .5-n,(twice+1)//2)*(1 if is_left else -sign)
                for output, coeff in N['act_mode'](nsca.Mode('G',twice),{middle:1.},h=self.h,c=self.c).items():
                    value += factor*coeff*self.evaluate(l,output,r)
        return value

    def matrix(self, le, re, word):
        lb, ls, lc, lp, _ = le
        rb, rs, rc, rp, _ = re
        out = np.array([[self.evaluate(a,word,b) for b in rb] for a in lb]).transpose(2,0,1)
        out *= (np.array([math.sqrt(2)/self.pl if b.ground_parity else 1 for b in lb])*ls)[None,:,None]
        out *= (np.array([math.sqrt(2)/self.pr if b.ground_parity else 1 for b in rb])*rs)[None,None,:]
        out = solve_triangular(lc,out.transpose(1,0,2).reshape(len(lb),-1),lower=True,check_finite=False).reshape(len(lb),2,len(rb)).transpose(1,0,2)
        out = solve_triangular(rc,out.transpose(2,0,1).reshape(len(rb),-1),lower=True,check_finite=False).reshape(len(rb),2,len(lb)).transpose(1,2,0)
        return np.stack((out[0]-out[1],out[0]+out[1]))


@lru_cache(maxsize=512)
def edge(level, p):
    basis = tuple(sorted(rsca.pbw_basis(level),key=lambda b:b.parity))
    h = 13.5/24+p*p/2
    scale = np.array([math.sqrt(2)/p if b.ground_parity else 1 for b in basis])
    gram = np.array([[R['descendant_inner_product'](a,b,h=h,c=13.5) for b in basis] for a in basis],float)
    gram *= scale[:,None]*scale[None,:]
    symmetry = np.max(abs(gram-gram.T))/max(1.,np.max(abs(gram)))
    if symmetry > 2e-10: raise ArithmeticError(f'Gram symmetry error {symmetry}')
    gram = (gram+gram.T)/2
    eq = 1/np.sqrt(np.diag(gram));g = gram*eq[:,None]*eq[None,:]
    split = sum(b.parity == 0 for b in basis);chol = np.zeros_like(g)
    cond = 1.
    for sl in (slice(0,split),slice(split,len(basis))):
        chol[sl,sl] = np.linalg.cholesky(g[sl,sl])
        cond = max(cond,float(np.linalg.cond(g[sl,sl])))
    if cond > 1e11: raise ArithmeticError(f'Gram condition {cond}')
    residual = np.max(abs(chol@chol.T-g))
    if residual > 1e-11: raise ArithmeticError(f'Gram factor residual {residual}')
    return basis,eq,chol,np.array([b.parity for b in basis]),dict(condition=cond,symmetry=symmetry,residual=residual)


def tables(momenta,omega,cutoff=16):
    from itertools import product
    p = tuple(map(float,momenta));h = (1+complex(omega)**2)/2
    levels = np.array([v for v in product(range(0,cutoff+1,2),repeat=2) if sum(v)<=cutoff],int)
    words = ((1,1),(0,0),(3,0),(1,0),(0,1))
    signs = tuple(product((-1,1),repeat=2))
    vertices = [Vertex(p[v-1],p[v],h) for v in range(2)]
    matrices = [{},{}];outputs = []
    for word in words:
        forms = ((0,0),(1,1)) if word in words[:2] else ((0,1),(1,0))
        values = np.zeros((2,4,len(levels)),complex)
        for ix,lev in enumerate(levels):
            es = [edge(int(l),x) for l,x in zip(lev,p)]
            vs=[]
            for v in range(2):
                key=(int(lev[v-1]),int(lev[v]),word[v])
                if key not in matrices[v]:
                    middle=() if word[v]==0 else (nsca.Mode('G',-word[v]),)
                    matrices[v][key]=vertices[v].matrix(es[v-1],es[v],middle)
                vs.append(matrices[v][key])
            for fi,f in enumerate(forms):
                a,b=[vs[v]*((es[v-1][3][:,None]+es[v][3][None,:]+(word[v]!=0))%2==f[v])[None] for v in range(2)]
                if word not in words[:2]: b=b*(-1.)**es[1][3][None,None,:]
                # Existing banks enumerate (+,+),(+,-),(-,+),(-,-).
                for si,(s,t) in enumerate(product((1,0),repeat=2)):
                    values[fi,si,ix]=np.einsum('ij,ji->',a[s],b[t])
        outputs.append(values)
    return levels,np.array(outputs[:2]),np.array(outputs[2:])
