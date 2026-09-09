"""Research extension of the pillow block to external NS upper components.

Only the human-note rho convention is used.  Unlike the all-bottom block,
an upper-component block generally has a nonconstant polynomial regular
part in the common internal weight.  The recursion requires an explicit
seed provider; it never silently substitutes the bottom-component seed.

ExactPBWSeeds computes that polynomial part algebraically at a finite
cutoff.  This is a constructive reference prescription, not an independent
large-weight formula or an optimized production seed.
"""

from functools import lru_cache

import mpmath as mp
import sympy as sp

from ns_pillow_direct_pbw import DirectNSSpherePBW
from ns_pillow_elliptic_audit import NSEllipticRecursion, PillowMap, indices


def component_markings(markings, count):
    markings = (0,)*count if markings is None else tuple(markings)
    if len(markings) != count or any(type(a) is not int or a not in (0,1) for a in markings):
        raise ValueError("one zero/one component marking is required per external field")
    return markings


def component_vertex_labels(levels, markings):
    """Incidence parity in the oriented zero-to-infinity sphere comb."""
    if not levels or any(type(n) is not int or n < 0 for n in levels):
        raise ValueError("levels must be nonempty nonnegative twice-level integers")
    markings = component_markings(markings,len(levels)+3)
    epsilon = tuple(n%2 for n in levels)
    return ((epsilon[0]^markings[0]^markings[1],)
            + tuple(epsilon[i-1]^epsilon[i]^markings[i+1] for i in range(1,len(levels)))
            + (epsilon[-1]^markings[-2]^markings[-1],))


def effective_weights(external, markings):
    """Virasoro-primary weights in the geometric factor only."""
    markings = component_markings(markings,len(external))
    return tuple(d+sp.Rational(a,2) for d,a in zip(external,markings))


def component_pullback(plane, c, external, internal, markings, degree, *, symbolic=False):
    """H = F/[Lambda(c; d+a/2) prod rho_i^(h_i-c/24) C_NS]."""
    one = sp.S.One if symbolic else mp.mpf(1)
    d_eff = tuple(d+one*a/2 for d,a in zip(external,component_markings(markings,len(external))))
    return PillowMap(len(internal),degree,symbolic).pullback(plane,c,d_eff,internal)


def polynomial_part(expression, variable):
    """Exact nonnegative Laurent powers at variable=infinity, not a fit."""
    numerator,denominator = sp.cancel(expression).as_numer_denom()
    quotient,_ = sp.div(sp.Poly(numerator,variable),sp.Poly(denominator,variable))
    return sp.factor(quotient.as_expr())


class ExactPBWSeeds:
    """Finite-order regular polynomials, reusable for all common-weight shifts.

    The external parameters and c can be symbols or exact numbers.  The
    internal variables remain generic, so recursive changes in differences
    are evaluated correctly.  No pole data is used to extract the seed.
    """

    def __init__(self,c,external,markings,degree):
        if len(external)<4 or type(degree) is not int or degree<0:
            raise ValueError("at least four fields and a nonnegative integer degree are required")
        self.c,self.external = c,tuple(external)
        self.markings = component_markings(markings,len(external))
        self.m,self.degree = len(external)-3,degree
        self.internal = sp.symbols(f"seed_h1:{self.m+1}")
        self.base = sp.Dummy("common_weight")
        self.differences = sp.symbols(f"seed_a2:{self.m+1}")
        self.keys = tuple(sorted(indices(self.m,2*degree),key=lambda k:(sum(k),k)))
        self.direct = DirectNSSpherePBW(c,external,self.internal,True,
                                      external_descendants=self.markings)
        self._seeds = None
        self._pulled = None

    def build(self):
        if self._seeds is not None:
            return self
        plane = {key:self.direct.coefficient(key) for key in self.keys}
        self._pulled = component_pullback(plane,self.c,self.external,self.internal,
                                         self.markings,self.degree,symbolic=True)
        shift = dict(zip(self.internal,(self.base,*(self.base+a for a in self.differences))))
        self._seeds = {key:polynomial_part(sp.cancel(self._pulled.get(key,0)).subs(shift,simultaneous=True),self.base)
                       for key in self.keys}
        return self

    def expression(self,levels):
        self.build()
        if tuple(levels) not in self._seeds:
            raise ValueError("coefficient exceeds the explicitly computed seed cutoff")
        return self._seeds[tuple(levels)]

    def coefficient(self,levels,weights):
        expression = self.expression(levels)
        replacements = {self.base:weights[0]}
        replacements.update({a:h-weights[0] for a,h in zip(self.differences,weights[1:])})
        return expression.subs(replacements,simultaneous=True)

    def numeric(self, *, dps=80):
        """Compile the exact polynomials once for repeated numerical h values.

        The parameters c,d must already be exact numbers.  Conversion is
        done at the stated precision; no SymPy substitution occurs during
        numerical recursion.  Internal differences are still variables.
        """
        self.build()
        if type(dps) is not int or dps<30:
            raise ValueError("numeric seeds require at least 30 decimal digits")
        with mp.workdps(dps):
            terms = {}
            for key,expression in self._seeds.items():
                polynomial = sp.Poly(expression,self.base,*self.differences)
                entries = []
                for powers,value in polynomial.terms():
                    if value.free_symbols:
                        raise ValueError("substitute numeric c and external weights before compilation")
                    real,imag = value.as_real_imag()
                    entries.append((powers,mp.mpc(str(real.evalf(dps+10)),str(imag.evalf(dps+10)))))
                terms[key] = tuple(entries)
            def convert(value):
                real,imag=sp.sympify(value).as_real_imag()
                if real.free_symbols or imag.free_symbols:
                    raise ValueError("numeric seeds require numerical parameters")
                return mp.mpc(str(real.evalf(dps+10)),str(imag.evalf(dps+10)))
            return NumericPolynomialSeeds(self.m,self.degree,dps,self.markings,terms,
                                          convert(self.c),tuple(map(convert,self.external)))


class NumericPolynomialSeeds:
    """Compiled finite-cutoff polynomials; use with an active workdps context."""

    def __init__(self,edges,degree,dps,markings,terms,c,external):
        self.m,self.degree,self.dps,self.markings,self.terms = edges,degree,dps,markings,terms
        self.c,self.external=c,external

    def coefficient(self,levels,weights):
        if tuple(levels) not in self.terms:
            raise ValueError("coefficient exceeds the explicitly computed seed cutoff")
        if len(weights)!=self.m:
            raise ValueError("wrong number of internal weights for this seed")
        if mp.mp.dps>self.dps:
            raise ValueError("recompile the numeric seeds at the requested higher precision")
        variables = (weights[0],*(h-weights[0] for h in weights[1:]))
        return mp.fsum(coefficient*mp.fprod(v**n for v,n in zip(variables,powers))
                       for powers,coefficient in self.terms[tuple(levels)])


class InteriorUnitSeed:
    """Candidate fast seed when the four pillow-cap fields remain bottom.

    Upper components are allowed only at mobile insertions (indices 2:-2).
    This is an explicit large-common-weight assumption, tested independently
    by the unit-seed audit; arbitrary cap components must not use this seed.
    """

    def __init__(self,markings):
        self.markings=component_markings(markings,len(markings))
        if len(markings)<4 or any(self.markings[i] for i in (0,1,len(markings)-2,len(markings)-1)):
            raise ValueError("the unit-seed extension requires four bottom pillow-cap fields")

    def coefficient(self,levels,weights):
        return 0 if any(levels) else 1


class ComponentEllipticRecursion(NSEllipticRecursion):
    """Fixed-c h-pole recursion, with explicitly supplied component seeds.

    Seed providers implement coefficient(levels, internal_weights).  At
    nonzero markings the provider is mandatory.  It must cover every
    requested level, parity and recursive internal-weight difference.
    """

    def __init__(self,b,external,internal,markings=None,*,seed=None,symbolic=False):
        if not symbolic and mp.mp.dps<30:
            raise ValueError("construct and evaluate numerical recursion inside mp.workdps(30 or higher)")
        super().__init__(b,external,internal,symbolic)
        self.markings = component_markings(markings,len(external))
        if any(self.markings) and seed is None:
            raise ValueError("upper-component recursion requires its own regular seed")
        if seed is not None and hasattr(seed,"markings") and tuple(seed.markings)!=self.markings:
            raise ValueError("seed component markings do not match the block")
        if len(external)!=len(internal)+3 or not internal:
            raise ValueError("a sphere comb requires n-3 internal weights")
        if isinstance(seed,NumericPolynomialSeeds):
            tolerance=mp.power(10,-min(mp.mp.dps,seed.dps)+8)
            actual=(mp.mpf('1.5')+3*self.Q2,*external)
            expected=(seed.c,*seed.external)
            if len(actual)!=len(expected) or any(abs(a-e)>tolerance*max(1,abs(a),abs(e)) for a,e in zip(actual,expected)):
                raise ValueError("seed c/external parameters do not match the recursion")
        self.seed = seed

    @lru_cache(maxsize=None)
    def coefficient(self,levels,state=None):
        if len(levels)!=self.m or any(type(n) is not int or n<0 for n in levels):
            raise ValueError("invalid nonnegative twice-level tuple")
        weights = self.weights(state)
        result = ((self.one if not any(levels) else 0*self.one) if self.seed is None
                  else self.seed.coefficient(levels,weights))
        labels = component_vertex_labels(levels,self.markings)
        shifts = (0,)*self.m if state is None else state[-1]
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
                    if denominator == 0:
                        raise ZeroDivisionError("component h-recursion encountered a Kac/common-weight pole")
                    endpoint = 4**product if self.m==1 else (2**product if edge in (0,self.m-1) else 1)
                    term = endpoint*self.residue(edge,r,s,labels[edge],labels[edge+1],shifts)*child/denominator
                    result = self.simplify(result+term) if self.symbolic else result+term
        return result
