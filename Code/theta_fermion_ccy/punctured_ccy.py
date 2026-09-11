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
import mpmath
from schottky_vacuum import theta_vacuum_seed


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
    """CCY recursion with fixed external weight; dps=0 uses native binary64.

    ``reduced_series(indices=...)`` accepts any downward-closed finite set.
    Its keys are the *integer Virasoro descendant levels* on the four edges.
    Precision is local to this instance.  To regulate colliding poles, pass
    regulated generic input weights; exact poles are never silently dropped.
    """

    def __init__(self, central_charge, weights, external_weight, dps=80, precision_bits=None):
        self.mp = mpmath.mp.clone() if dps else mpmath.fp
        if dps:
            if precision_bits is None:
                self.mp.dps = int(dps)
            else:
                self.mp.prec = int(precision_bits)
        self.c = self._number(central_charge)
        if len(weights) != 4:
            raise ValueError("weights must be (h1,hL,hR,h3)")
        self.weights = tuple(self._number(h) for h in weights)
        self.external_weight = self._number(external_weight)
        # Equal multiprecision central charges have one small cache key.
        # This avoids hashing a complex number on every descendant lookup.
        self._central_values = [self.c]
        self._central_ids = {self.c: 0}
        self._pole_tolerance = (self.mp.power(10, 10-self.mp.dps) if dps
                                else 64*self.mp.eps)
        # Cache the c-independent pole expansion, not a separate recursion
        # result for every evaluation central charge. Each pole expansion
        # retains its already evaluated lower blocks.
        self._coefficient = self._coefficient_uncached
        self._rational = lru_cache(maxsize=None)(self._rational_uncached)
        self._templates = []
        self._template = lru_cache(maxsize=None)(self._template_uncached)
        self._factor_rows = {}
        self._numeric_evaluations = 0
        self._factor_lookups = 0
        self._factor_misses = 0
        # Only terminal (all levels <=1) global coefficients need a separate
        # cache. Nonterminal seeds live in their c-independent pole expansion.
        self._global = lru_cache(maxsize=None)(self._global_indexed)
        self._pole_geometry = lru_cache(maxsize=None)(self._pole_geometry_uncached)
        self._fusion = lru_cache(maxsize=32768)(self._fusion_uncached)
        self._rho = lru_cache(maxsize=None)(self._rho_uncached)
        self._rho_core = lru_cache(maxsize=None)(self._rho_core_uncached)
        self._rho_normalized = lru_cache(maxsize=32768)(self._rho_normalized_uncached)
        self._shifted_weights = lru_cache(maxsize=32768)(self._shifted_weights_uncached)
        self._norm = lru_cache(maxsize=None)(self._norm_uncached)

    def _number(self,value):
        if isinstance(value,Fraction):
            return self.mp.mpf(value.numerator)/value.denominator
        value = self.mp.mpc(value)
        # Retain every nonzero imaginary part. Real arithmetic promotes to
        # complex automatically if a subsequent Kac square root requires it.
        return value if value.imag else value.real

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

    def _rho_core_uncached(self, i,k,h1,h2,h3):
        # The sum in CCY's closed SL(2) three-point expression is independent
        # of the middle descendant level j. Build its finite products once,
        # with no divisions by factors that can vanish at generic inputs.
        rising = [self.mp.mpf(1)]
        for t in range(k):
            rising.append(rising[-1]*(h3+h2-h1+t))
        suffix = [self.mp.mpf(1)]*(i+1)
        for t in range(i-1,-1,-1):
            suffix[t] = suffix[t+1]*(h1+h2-h3-k+t)
        total = self.mp.mpf(0)
        falling_h3 = self.mp.mpf(1)
        falling_k = 1
        for p in range(min(i,k)+1):
            total += comb(i,p)*falling_h3*falling_k*rising[k-p]*suffix[p]
            falling_h3 *= 2*h3+k-1-p
            falling_k *= k-p
        return total

    def _rho_uncached(self, i,j,k,h1,h2,h3):
        # CCY's closed SL(2) three-point matrix element; no Virasoro Ward sum.
        return (self._rho_core(i,k,h1,h2,h3)
                *self._rising(h1+i-h2-j+1-h3-k,j))

    def _rho_normalized_uncached(self,i,j,k,h1,h2,h3):
        denominator = self._norm(h1,i)*self._norm(h3,k)
        if not denominator:
            raise ZeroDivisionError("degenerate global Gram norm: use a specified generic-weight limit")
        return self._rho(i,j,k,h1,h2,h3)/denominator

    def _shifted_weights_uncached(self,shifts):
        return tuple(h+s for h,s in zip(self.weights,shifts))

    def _global_uncached(self, shifts, levels):
        h1,hL,hR,h3 = self._shifted_weights(shifts)
        a,b,c,d = levels
        # Assign the edge-1/edge-3 norms to the first outer vertex and the
        # two split-edge norms to the middle vertex. These normalized
        # vertices recur for many spectator levels and shifts.
        return (self._rho_normalized(a,b,d,h1,hL,h3)
                *self._rho(a,c,d,h1,hR,h3)
                *self._rho_normalized(c,0,b,hR,self.external_weight,hL))

    def _global_indexed(self, shifts, levels):
        return self._global_uncached(_vectors[shifts],_vectors[levels])

    def _fusion_uncached(self,r,s,x,top,bottom):
        # The fusion lattice is invariant under (p,t) -> (-p,-t).
        # Pair the two factors before evaluation, eliminating both square
        # roots. Here x=b^2 and u^2=(p*b+t/b)^2. The unpaired origin, when
        # present, is (lambda_top^2-lambda_bottom^2)/4 = bottom-top.
        lambda_top_squared = x+2+1/x-4*top
        difference = 4*(bottom-top)
        product = self.mp.mpf(1)
        for p in range(1-r,r,2):
            for t in range(1-s,s,2):
                if p == 0 and t == 0:
                    product *= bottom-top
                elif p > 0 or (p == 0 and t > 0):
                    shift_squared = p*p*x+2*p*t+t*t/x
                    product *= ((difference+shift_squared)**2
                                -4*lambda_top_squared*shift_squared)/16
        return product

    def _pole_geometry_uncached(self, edge, own_shift, r,s):
        # Kac pole position and universal residue factor depend on the null
        # edge only, never on the three spectator shifts.
        h = self.weights[edge]+own_shift
        radical = (r-s)**2+4*(r*s-1)*h+4*h*h
        root = self.mp.sqrt(radical)
        roots = ((r*s-1+2*h+root)/(1-r*r),
                 (r*s-1+2*h-root)/(1-r*r))
        # Continue the noncancelling branch, including negative h and s=1.
        x = max(roots,key=abs)
        if not x:
            raise ZeroDivisionError("c-recursion pole at infinity requires a generic-weight limit")
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
        return c,numerator/denominator,x

    def _pole_uncached(self, shifts, edge, r,s):
        c,residue,x = self._pole_geometry(edge,shifts[edge],r,s)
        weights = self._shifted_weights(shifts)
        h1,hL,hR,h3 = weights
        # Null slot infinity -> (zero,one); one -> (zero,infinity);
        # zero -> (infinity,one).  All three vertices use (infinity,1,0).
        pairs = (((h3,hL),(h3,hR)),
                 ((h3,h1),(hR,self.external_weight)),
                 ((h3,h1),(hL,self.external_weight)),
                 ((h1,hL),(h1,hR)))
        for top,bottom in pairs[edge]:
            residue *= self._fusion(r,s,x,top,bottom)
        return c,residue

    def _template_uncached(self,shifts,edge,r,s):
        """Pole data independent of the evaluation central charge and levels."""
        shift = _vectors[shifts]
        pole,residue = self._pole_uncached(shift,edge,r,s)
        if not residue:
            return None
        pole_id = self._central_ids.get(pole)
        if pole_id is None:
            pole_id = len(self._central_values)
            self._central_ids[pole] = pole_id
            self._central_values.append(pole)
        shifted = list(shift)
        shifted[edge] += r*s
        template = len(self._templates)
        self._templates.append((pole_id,residue,_vector_id(tuple(shifted))))
        return template

    def _factor_uncached(self,c_id,template):
        pole_id,residue,_ = self._templates[template]
        c,pole = self._central_values[c_id],self._central_values[pole_id]
        denominator = c-pole
        if abs(denominator) <= self._pole_tolerance*max(1,abs(c),abs(pole)):
            raise ZeroDivisionError("coincident or unresolved c-recursion poles require a generic-weight limit or more precision")
        return residue/denominator

    def _rational_uncached(self,shifts,levels):
        """Build the complete ordered pole list once, for all evaluation c.

        Each lower coefficient is evaluated at its pole central charge,
        independently of the parent's evaluation c. Keep it separate from
        the residue to preserve (residue/(c-pole))*child arithmetic order.
        """
        seed = self._global_indexed(shifts,levels)
        terms = []
        for edge,r,s,lower,terminal in _level_steps_indexed(levels):
            template = self._template(shifts,edge,r,s)
            if template is None:
                continue
            pole_id,_,shifted = self._templates[template]
            child = (self._global(shifted,lower) if terminal else
                     self._coefficient(pole_id,shifted,lower))
            terms.append((template,child))
        return seed,tuple(terms)

    def _coefficient_uncached(self,c_id,shifts,levels):
        self._numeric_evaluations += 1
        total,terms = self._rational(shifts,levels)
        self._factor_lookups += len(terms)
        factors = self._factor_rows.setdefault(c_id,{})
        for template,child in terms:
            try:
                factor = factors[template]
            except KeyError:
                factor = self._factor_uncached(c_id,template)
                factors[template] = factor
                self._factor_misses += 1
            total += factor*child
        return total

    def reduced_coefficient(self,levels):
        levels = tuple(int(n) for n in levels)
        if len(levels) != 4 or min(levels) < 0:
            raise ValueError("four nonnegative descendant levels are required")
        level_id = _vector_id(levels)
        if max(levels) < 2:
            return self._global(0,level_id)
        return self._coefficient(0,0,level_id)

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
            total = self.mp.mpf(0)
            for (a,b,d),coefficient in seed.items():
                m = (n[0]-a,n[1]-b,n[2]-b,n[3]-d)
                if min(m)>=0:
                    total += self._number(coefficient)*reduced[m]
            result[n] = total
        return result

    def cache_info(self):
        return {"precision_bits":self.mp.prec,
                "recursion_algorithm":"c-independent pole expansions, Python",
                "rational_blocks":self._rational.cache_info()._asdict(),
                "pole_templates":self._template.cache_info()._asdict(),
                "numeric_evaluations":self._numeric_evaluations,
                "shared_integer_vectors":len(_vectors),
                "shared_level_steps":_level_steps_indexed.cache_info()._asdict(),
                "coefficient":{"hits":0,"misses":self._numeric_evaluations,
                               "maxsize":0,"currsize":0},
                "global":self._global.cache_info()._asdict(),
                "pole_geometry":self._pole_geometry.cache_info()._asdict(),
                "fusion":self._fusion.cache_info()._asdict(),
                "transition":{"hits":self._factor_lookups-self._factor_misses,
                              "misses":self._factor_misses,"maxsize":None,
                              "currsize":sum(map(len,self._factor_rows.values()))},
                "rho":self._rho.cache_info()._asdict(),
                "rho_core":self._rho_core.cache_info()._asdict(),
                "rho_normalized":self._rho_normalized.cache_info()._asdict(),
                "shifted_weights":self._shifted_weights.cache_info()._asdict(),
                "norm":self._norm.cache_info()._asdict(),
                "central_charge_states":len(self._central_values)}

    def clear_caches(self):
        """Release one branch's recursion state after its series is assembled."""
        for cache in (self._rational,self._template,self._global,
                      self._pole_geometry,self._fusion,
                      self._rho,self._rho_core,self._rho_normalized,
                      self._shifted_weights,self._norm):
            cache.cache_clear()
        self._templates.clear()
        self._factor_rows.clear()
        self._numeric_evaluations = self._factor_lookups = self._factor_misses = 0
        self._central_values = [self.c]
        self._central_ids = {self.c: 0}


@lru_cache(maxsize=None)
def _residue_labels(level):
    """Original CCY residue order, reused at every recursion state."""
    return tuple((r,s,r*s) for r in range(2,level+1)
                 for s in range(1,level//r+1))


@lru_cache(maxsize=None)
def _level_steps(levels):
    """Weight-independent recursion steps, in the original residue order.

    Lower-level tuples and terminal flags depend only on the four integer
    levels. Share them between all coefficient states, n branches and copies.
    There are no floating values or precision-dependent objects in this cache.
    """
    steps = []
    for edge,k in enumerate(levels):
        for r,s,level in _residue_labels(k):
            lower = list(levels)
            lower[edge] -= level
            lower = tuple(lower)
            steps.append((edge,r,s,lower,max(lower) < 2))
    return tuple(steps)


# Exact integer vectors only: IDs are independent of weights, central charge,
# precision and branch. The zero vector has ID zero. Never clear this table
# while engines exist, since their local numeric caches refer to these IDs.
_vectors = [(0,0,0,0)]
_vector_ids = {_vectors[0]:0}


def _vector_id(vector):
    index = _vector_ids.get(vector)
    if index is None:
        index = len(_vectors)
        _vectors.append(vector)
        _vector_ids[vector] = index
    return index


@lru_cache(maxsize=None)
def _level_steps_indexed(level):
    return tuple((edge,r,s,_vector_id(lower),terminal)
                 for edge,r,s,lower,terminal in _level_steps(_vectors[level]))
