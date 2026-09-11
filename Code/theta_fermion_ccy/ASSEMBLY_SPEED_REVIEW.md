# Exact reuse in the four-edge assembly

This is a static algebra and source review. It performs no numerical block
comparison. The requested validation remains the two final-physical-block
checks: independent PBW through total level 5 and independence of the split
position at fixed product. The implementation must continue to construct every
allowed four-variable coefficient; no equality of split exponents is imposed.

## Exchange of the two pieces of the split tube

An ordinary Virasoro block has the ordered outer vertices
`(h1,hL,h3)`, `(h1,hR,h3)` and middle vertex `(hR,hpsi,hL)`.
Interchanging `(hL,qL)` with `(hR,qR)` exchanges the two outer scalar
three-point factors. The middle global matrix element is unchanged under
transposition. More explicitly,

\[
\rho(i,0,k;h_1,h_2,h_3)
=i!k![x^iy^k]
(1-x)^{-h_1-h_2+h_3}
(1-y)^{-h_2-h_3+h_1}
(1-xy)^{-h_1-h_3+h_2}.
\]

The right side is invariant under `(x,h1) <-> (y,h3)`.
There is no extra phase. The norm denominator transforms in the same way.
In the CCY residue formula, interchanging split edges merely permutes their
two residue sums and the corresponding spectator weights in every fusion
polynomial. Induction on total descendant level therefore gives the full
ordinary-block identity, including all finite-central-charge corrections.

The product of the two ordinary Virasoro blocks consequently has the same
exchange symmetry. It is enough to compute that product once for
`(n1, unordered pair {incoming,outgoing}, n3)`. The reverse ordered pair uses
the coefficient with its left and right split exponents exchanged. Both
orientations have the same primary-level sum and the same allowed descendant
cutoff. Their branching coefficients, norms, and parity factors must still be
evaluated separately; the symmetry concerns the scalar ordinary blocks only.

The two split labels differ by one half and are never equal. With both
orientations present, this removes half the ordinary-block evaluations and
their scalar products. A product can be discarded as soon as the reverse
orientation is assembled. This is a graph automorphism optimization, not an
assumption about the physical answer depending only on `qL*qR`.

## Algebra kernels and truncation

Several further reorganizations are exact and apply to any nonnegative
multivariate plumbing series:

1. Store each term's total degree once, sort or bucket terms by degree, and
   stop a product's inner loop when the sum exceeds the cutoff. This avoids
   testing all forbidden pairs and repeatedly summing exponent tuples.
2. In triangular division, one may propagate a solved coefficient forward
   to its key plus each positive auxiliary shift. Since every such shift has
   positive total degree, these updates affect only later coefficients. This
   is equivalent to the existing backward convolution recurrence and avoids
   scanning impossible componentwise differences.
3. Precompute the auxiliary vectors' nonzero parity entries and use the
   existing `STAR_SIGN` table to update residual entries directly. Skip only
   exact zero coefficients; do not apply a tolerance cutoff. Retain the
   existing sector-residual guard without replacing the answer by its
   projection.

The universal seed has support `(2*a,b,b,2*c)`, but multiplying it by the
quotient still requires every allowed four-variable quotient coefficient.
Its special support does not justify restricting the quotient to equal split
powers. The selected direct auxiliary is the full fermion factor, so two
copies of the universal seed are still required after division.

## Independent audit of the CCY kernel changes

The changes documented in `CCY_SPEED_REVIEW.md` were independently inspected.
The paired fusion factor, its unpaired-origin sign, the global three-point
prefix/suffix products, the pole-geometry cache dependencies, and the full
transition key all preserve the original algebra. Central-charge identifiers
merge only equal stored numbers, and the unresolved-pole guard remains in
place. The public coefficient interface is unchanged. No numerical check was
performed in this audit.

## What the available profile does and does not show

`results/profile_ccy_baseline_cut16.json` profiles one specified ordinary
block at weighted descendant cutoff 16. Under the profiler it took 87.85 s:
the global SL(2) seed accounts for 18.80 s and pole assembly for 13.28 s,
including 7.93 s in fusion polynomials. Thus seed construction is a material
cost but does not alone explain the runtime. The separately restored
universal vacuum seed is not evaluated in this profiled stage. These figures
are not an unprofiled speed estimate or a full level-10 pipeline timing.
