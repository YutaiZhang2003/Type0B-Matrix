# BRY super-Liouville sphere check of the long-draft sewing prescription

The subsequent [NSRR normalization resolution](Genus%202/NSRR_NORMALIZATION_RESOLUTION_2026-09-24.md)
records the mixed sphere and actual Ramond identity checks of 2026-09-24.
Those fix the native equal-metric coefficient dictionary to `c=(E,O)`.
This earlier audit retains its all-NS scope and odd-component phase rule.

This check was performed after the general graded-tensor audit. It specializes
the prescription to **NS external supermultiplets at b=1**, including bottom
fields, top components, and the fermionic descendant terms in BRY's G, H, J.
It does not test physical Ramond external states or a Ramond state-space
reduction.

The physical inputs are BRY's real structure constants C and Ctilde and
their continuum measure dP/pi. The source equations are (3.1)--(3.4) and
(A.1)--(A.4) of
[Balthazar, Rodriguez and Yin, arXiv:2201.05621](https://arxiv.org/html/2201.05621).
No overall normalization or leg factor was fitted. The central charge is
exactly 27/2 in this check; there is no displaced-c regulator.

## What the conversion fixes

Denote the normalized even and odd full three-point *sewing couplings* in
the graded prescription by Gamma_0 and Gamma_1. They must not be identified
with BRY's two real numerical functions without a convention conversion.
For four even NS bottom fields, the graded formula gives

\[
\mathcal I_G(P;z)=\frac1\pi\left[
\Gamma_{L,0}\Gamma_{R,0}|P_0(z)|^2
-\Gamma_{L,1}\Gamma_{R,1}|P_1(z)|^2\right],
\]

where P_0,P_1 are the long-draft sphere blocks, including their primary
coordinate factors. The draft's odd block has leading coefficient -1/(2h).
Its overall sign disappears in the diagonal product, so it cannot by itself
remove the minus sign in this expression.

Matching BRY (A.1) fixes the **coefficient products** to

\[
\Gamma_{L,0}\Gamma_{R,0}=C_LC_R,\qquad
\Gamma_{L,1}\Gamma_{R,1}=-\widetilde C_L\widetilde C_R.
\]

A consistent choice at the two-vertex sewing boundary is

\[
\boxed{\Gamma_0=C_{\rm BRY},\qquad
\Gamma_1=i\widetilde C_{\rm BRY}.}
\tag{1}
\]

The same sign of i is used on both vertices: this is a bilinear product,
not Gamma_L times the complex conjugate of Gamma_R. The common replacement
Gamma_1 -> -Gamma_1 is indistinguishable in the observables tested here.
The BRY numerical structure-constant functions themselves are unchanged.

This is also the phase convention already recorded in
[conventions.md, section 4.1](</Users/yutaizhang/Desktop/Type0B-Matrix/Machine Notes/conventions.md:249>).
There is a separate analytic check of its relative phase. Write D_NS and D_R
for the four-factor Upsilon denominators. Equations (10)--(11) of
[Suchanek, arXiv:1012.2974](https://arxiv.org/pdf/1012.2974) give
tilde_C_S/C_S = 2i D_NS/D_R. BRY (3.4) gives
tilde_C_BRY/C_BRY = 2 D_NS/D_R at b=1. The denominators coincide for
a_j=1+iP_j, and the common primary leg normalization cancels in each ratio.
Matching the even coefficient therefore gives the same odd phase as (1).
It is not a parameter fitted to the numerical crossing results.

Equation (1) specifies the sewing dictionary used in this check. This audit
does not reconstruct the complete local BRY operator/cocycle map, or prove
an operator identity between W and a bare tensor-product state with
unspecified BPZ/antichiral conventions. Apply the conversion once when
importing BRY constants into the graded basis; do not add it again to a
BRY-native correlator formula.

## Descendant and external-parity check

Write D_0,D_1 for the draft's blocks with G_{-1/2} on external legs 3 and 2.
For either chiral factor, internal parity r selects the vertex labels

\[
a_L=r+d_3,\qquad a_R=r+d_2\pmod2,
\]

and the full external-order exponent is tilde_d3*d2. These rules, the
generic coefficient_matrix function, and the same dictionary (1) give

\[
\pi\mathcal I_G=C_LC_R P_0\bar P_0
+\widetilde C_L\widetilde C_R P_1\bar P_1,
\]
\[
\pi\mathcal I_H=-\widetilde C_L\widetilde C_R D_0\bar D_0
-C_LC_R D_1\bar D_1,
\]
\[
\begin{aligned}
\pi\mathcal I_J={}&C_LC_R\left[
\frac{D_1\bar P_0}{1-\bar z}+\frac{P_0\bar D_1}{1-z}\right]\\
&+\widetilde C_L\widetilde C_R\left[
\frac{D_0\bar P_1}{1-\bar z}+\frac{P_1\bar D_0}{1-z}\right].
\end{aligned}
\tag{2}
\]

In this notation the barred blocks use the conjugate chiral frame at real
weights. The P and D odd blocks each differ by a minus sign from BRY's
corresponding displayed odd blocks. Consequently (2) reproduces all three
published decompositions, including the relative signs of the mixed products
in J. The two terms in J check correlators with a pair of odd Liouville
external descendants, with the free-fermion factors included as in BRY.
No new phase was chosen for H or J after fixing (1) from G.

## What was computed

The [native probe](/Users/yutaizhang/Desktop/Type0B-Matrix/C++/experiments/long_draft_sewing_2026-09-23/bry_sphere_probe.cpp)
uses the frozen C++ Ward and Gram implementation matching the long draft.
It computes the sphere blocks by direct PBW summation, independently of the
Python recursion. At three internal momenta, for both the direct and crossed
external weights, all four choices of external stars on legs 2 and 3 were
checked through internal level 6. All **312 coefficients** match the
independent c-recursion, with maximum scaled error **2.34e-57** at 60 decimal
digits. This licenses using that recursion to evaluate deeper elliptic
expansions in this NS convention; the high-order integrals are not fresh
high-order C++ PBW sums.

The [nonchiral check](/Users/yutaizhang/Desktop/Type0B-Matrix/C++/experiments/long_draft_sewing_2026-09-23/bry_sphere_check.py)
then constructs full coefficient matrices with the generic graded prescription
and compares each momentum integrand against the separate BRY G,H,J assembler.
The high-order block evaluator is shared at this stage, while the coefficient
assembly is separate. Thus the pointwise comparison checks the convention
dictionary, not two independent high-order block algorithms. The crossing
comparison additionally tests the actual spectral integrals in different
channels.

External momenta are (P1,P2,P3,P4)=(1/2,1/3,1/4,3/5). Both z=0.37 and
z=0.37+0.11i are tested. G is compared under z -> 1-z with P1 <-> P3.
At the complex point H and J are also compared under z -> 1/z with
P2 <-> P3 and their required coordinate prefactor. The code varies elliptic
order and quadrature separately and evaluates the extra momentum interval
5 < P < 6.

The numerical ledger, all refinement rows, the negative control using a real
Gamma_1, native coefficients, and build commands are saved under
[bry_sphere](/Users/yutaizhang/Desktop/Type0B-Matrix/C++/experiments/long_draft_sewing_2026-09-23/bry_sphere/results.json).
That ledger is authoritative for the truncation-dependent numerical errors.

## Numerical result

At elliptic order 8, with 36 Gauss--Legendre momentum nodes on 0 < P < 5,
the four-primary correlator gives:

| z | Direct G | Crossed G | Relative crossing mismatch |
|---|---:|---:|---:|
| 0.37 | 0.221609720986587 | 0.221609720986640 | 2.39e-13 |
| 0.37+0.11i | 0.211719203999165 | 0.211719203999662 | 2.35e-12 |

The **1,260 G,H,J momentum-integrand comparisons** have maximum scaled
error **3.72e-16**. Omitting the phase in (1) gives a discrepancy as large
as **0.797 relative** in the four-primary integrand. Thus the convention
conversion has a measurable effect on the full answer.

At the complex point, the H and J inversion-channel mismatches are
**2.34e-5** and **1.96e-5**. At order 6 they were 9.32e-4 and 8.04e-4;
the improvement with order supports the descendant crossing check, but
does not establish the same precision as the primary check.

Changing quadrature from 24 to 36 nodes at order 8 shifts the direct G
values by about 1.2e-7 relative and H,J by less than 6.3e-8. The final
crossing residuals are observed channel agreements, **not certified absolute
error bounds**. The extra direct-channel interval 5 < P < 6 contributes
less than 7.3e-36 in absolute value to each tested correlator; the remaining
infinite tail has not been rigorously bounded.

To reproduce from the repository root:

```sh
VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 C++/experiments/long_draft_sewing_2026-09-23/bry_sphere_check.py
```

This establishes the NS sphere specialization with the stated sewing
dictionary. It does not validate the earlier genus-two factor of four or
provide a completed BRY-to-long-draft Ramond local-tensor dictionary.
