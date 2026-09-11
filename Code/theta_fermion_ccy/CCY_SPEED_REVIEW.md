# CCY engine: algebraic reuse and finite-product optimization

The changes in `punctured_ccy.py` preserve coefficient-level CCY central-charge
recursion, the complete caller-supplied descendant index set, the instance
precision, and the existing errors for unresolved poles. No finite-central-charge
Gram matrix, physical PBW block, interpolation, or numerical coefficient cutoff
is introduced. The direct auxiliary fermion factor is unaffected.

This document records a static algebra/source review. It does not itself report
a numerical comparison or a speed measurement. The only numerical validation
requested for the complete implementation remains the physical PBW comparison
through level 5 and the fixed-product split-coordinate comparison.

## Pole geometry

At a residue on edge `edge`, the Kac equation uses only
`weights[edge] + shifts[edge]` and the positive labels `(r,s)`. Neither its chosen
root, its central-charge value, nor the universal null normalization depends on
the shifts on the other edges. The new `_pole_geometry` cache therefore has key
`(edge, own_shift, r, s)`. The original root-selection rule and the existing
cancellation of universal factors in the null normalization are unchanged.

The full residue still depends on neighboring shifted weights. `_pole` retains
the entire shift tuple and multiplies the same two ordered vertex fusion
polynomials. The fusion cache uses `(r,s,x,top,bottom)`, where `x=b²` is the Kac
root. Top and bottom are not interchanged in that key.

## Fusion polynomials without square roots

For one fusion polynomial, write temporarily

\[
A=(b+b^{-1})^2-4h_{\mathrm{top}},\qquad
B=(b+b^{-1})^2-4h_{\mathrm{bottom}},\qquad
u=pb+t/b.
\]

These letters are local to this implementation explanation, not new notation
in the manuscript. The old factor for one lattice point is

\[
\frac{(\sqrt A+\sqrt B+u)(\sqrt A-\sqrt B+u)}4
=\frac{A-B+u^2+2u\sqrt A}{4}.
\]

The fusion lattice `p=1-r,3-r,...,r-1` and
`t=1-s,3-s,...,s-1` is closed under `(p,t)→(-p,-t)`. Every nonzero
pair therefore contributes exactly

\[
\frac{(A-B+u^2)^2-4Au^2}{16},\qquad
u^2=p^2x+2pt+t^2/x.
\]

The representative condition `p>0 or (p==0 and t>0)` chooses every such pair
once. If both `r` and `s` are odd, the remaining origin contributes
`(A-B)/4 = bottom-top`. This fixes the sign, including odd-degree fusion
polynomials. The pairing eliminates both square roots without choosing or
altering their branches. It is a polynomial identity, not a large-weight
approximation. In particular, no tolerance is used to declare a residue zero.

## Global seed

In the existing closed CCY expression for
`rho(i,j,k,h1,h2,h3)`, the finite sum over `p` is independent of `j`.
It is cached separately as `_rho_core(i,k,h1,h2,h3)`. The original final
rising factorial, which contains `j`, is then multiplied back in.

Within that finite sum, the implementation builds the rising products

\[
\prod_{t=0}^{m-1}(h_3+h_2-h_1+t)
\]

as prefixes, and the products

\[
\prod_{t=p}^{i-1}(h_1+h_2-h_3-k+t)
\]

as suffixes. The latter are precisely
`(h1+h2-h3+p-k)_(i-p)` in the original expression. The two falling products
are advanced by multiplication as `p` increases. Thus the same finite sum is
formed with linear product construction rather than rebuilding several
factorials inside each summand. There are no divisions by intermediate
factors, so a vanishing rising/falling factor does not create a new singularity.

The seed is still the exact global `SL(2)` seed. The separately restored
universal vacuum factor is unchanged. Floating-point multiplication order is
different; accuracy is assessed by the already authorized final-block checks.

The second optimization pass additionally caches

\[
\frac{\rho(i,j,k;h_1,h_2,h_3)}
{i!(2h_1)_i\,k!(2h_3)_k}.
\]

Using this normalized value for the first outer vertex assigns it precisely
the edge-1 and edge-3 inverse norms. Using it for the middle vertex assigns
that vertex precisely the two split-edge inverse norms. The ordinary second
outer vertex is unchanged. Their product is consequently the original global
seed, with every internal inverse norm appearing exactly once. Each normalized
value checks its denominator before division; a zero norm therefore retains
the original error even if another vertex factor vanishes. No square roots of
norms or phase convention are introduced. This optional scalar cache is bounded
at 32,768 entries. Shifted weight tuples are cached with the same bound.

## Real inputs and residue-free states

Input conversion preserves an arbitrary nonzero imaginary part exactly. A
converted number with an exactly zero imaginary part is stored as a local
multiprecision real instead. Finite-product and sum accumulators start as local
real one or zero, and mpmath promotes them to complex numbers whenever a complex
input or a Kac square root requires it. There is no float conversion, tolerance
test on imaginary parts, or restriction of the allowed complex input domain.
The precision remains the instance precision.

When every remaining descendant level is zero or one, the conditions `r>=2`
and `s>=1` exclude every CCY residue. The coefficient is then exactly its global
seed independently of the current central charge. Such states return the
already cached global seed directly rather than adding identical entries to
the coefficient cache for multiple central charges. The parent transition's
pole guard has already run before this child shortcut is considered. The same
observation handles residue-free public `reduced_coefficient` calls, which have
no pole terms to inspect. This does not omit any admissible residue.

## Reuse of residue transitions

For fixed current central charge, full internal shift tuple, edge, and `(r,s)`,
the multiplier `residue/(c-pole)` and the target shift tuple do not depend on the
remaining descendant levels. `_transition` caches that result. A coefficient
still visits every original `(edge,r,s)` with `r*s <= levels[edge]` in the same
order, lowers exactly that edge by `r*s`, and recurses at the same pole.

The near-pole guard is unchanged:

\[
|c-c_{rs}|\leq10^{10-\mathrm{dps}}
\max(1,|c|,|c_{rs}|)
\]

raises the same error for every nonzero residue. Its instance-constant
prefactor is computed once. A transition can be cached only after this guard
has passed. Exact zero residues retain the prior behavior of contributing
nothing. The cache is bounded at 32,768 entries; eviction merely recomputes an
unchanged transition. The optional fusion cache has the same bound.

The original `(r,s,r*s)` lists are cached by the remaining integer edge level.
This changes neither the level cutoff nor residue order.

## Canonical central-charge keys

The old coefficient cache used a multiprecision complex central charge as one
part of every key. The new engine interns each encountered central charge in
an instance-local dictionary and uses its integer identifier instead. Two
charges receive one identifier if and only if Python dictionary equality of
the stored multiprecision numbers equates them. There is no rounding,
tolerance grouping, or identification of nearby poles.

Consequently `(c_id, shifts, levels)` names the same numerical recursion state
as `(c, shifts, levels)` did previously. The numerical charge remains available
in `_central_values` for the unchanged denominator calculation. The initial
charge has identifier zero. Pole guards precede recursive use of an identifier,
so interning cannot hide a collision with the initial charge or another pole.

All instance caches and the interned-charge dictionaries are cleared after the
branch has been assembled. The additional caches retain scalars or individual
transition tuples, not expanded descendant-term lists. Existing `cache_info`
fields are retained, with counters for geometry, fusion, transition, seed
three-point functions, norms, and distinct central charges added for profiling.

## Source verification

`python -m py_compile Code/theta_fermion_ccy/punctured_ccy.py` passed after the
edit. This is a syntax check only. Numerical validation and runtime evidence
are recorded by the parent pipeline runs, not inferred from this source review.

An independent source/algebra audit by `/root/speed_review` passed the finite
product identities, fusion-lattice signs, cache dependencies, exact-value charge
interning, and residue order. It also checked that the external callers use the
public `reduced_coefficient` interface, whose signature is unchanged. That audit
performed no numerical checks.

The same independent reviewer subsequently audited the second optimization
pass and passed the real/complex promotion, assignment of all four inverse
norms, zero-norm guards, full shifted-weight cache key, residue-free base-state
criterion, and cache clearing. Syntax compilation passed again. No numerical
calculations were performed for either static audit.
