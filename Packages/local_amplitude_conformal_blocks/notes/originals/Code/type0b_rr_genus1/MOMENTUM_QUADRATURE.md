# Momentum quadrature for Ramond punctures

The RR calculation needs a separate spectral convergence study. The
previous 8-by-8 overlap grid and the 8/12-node modular grids did not
separate endpoint resolution, changes of momentum cutoff, and block
truncation. Their discrepancies must not be treated as amplitude error
estimates or, by themselves, proof of a normalization error.

This document records the raw quadrature study before the relative
normalization was fixed. Those drivers and reports retain their original
coupling and Gram conventions. The subsequent
[correlator normalization layer](INTERNAL_NORMALIZATION.md) multiplies
completed radial correlators by two and OPE sectors by `(2,2,1,1)`.
It leaves primary metrics, trinion couplings and descendant sums unchanged.
The normalized copies of both modular banks pass within `1.10e-6`.
This constant changes neither the endpoint powers nor the quadrature
design. No matrix amplitude is used as input.

## Endpoint behavior of the full sewing

An R puncture intertwines an NS module with an R module. The radial
channel therefore has independent momenta \(P_{\rm NS},P_R\). The
collision channel instead has an NS bridge and an NS or R handle.

From [BRY I, (3.5) and (3.9)](https://arxiv.org/html/2201.05621v2#S3.SS1),
the leg factors used in this repository obey

\[
N_{\rm NS}(P)=iP+O(P^3),\qquad
N_R(0)=\Upsilon_1(1/2)^2=0.4160281585833496\ldots.
\]

An NS edge couples to two trinions and supplies two linear zeros. An R
edge has no such universal suppression. This is an endpoint statement
about the coupled integrand, not a change of the continuum measure
\(dP/\pi\) assumed in the current sewing dictionary. Projected state sums,
inverse Grams and PCO descendants must be included before assigning the
final power; individual blocks or leg factors alone are insufficient.

At \(\omega=i/4\), the implemented coupled local components have the
following endpoint powers at generic fixed other momentum:

| Channel and edge | Leading generic power in the integrand |
| --- | --- |
| Radial NS edge | \(P_{\rm NS}^2\) |
| Radial R edge | \(P_R^0\) |
| OPE NS bridge, all four theta sectors | \(P_{\rm bridge}^2\) |
| OPE R handle, theta 1 and 2 | \(P_{\rm handle}^0\) |
| OPE NS handle, theta 3 and 4 | \(P_{\rm handle}^2\) |

`check_momentum_design.py` checks these powers in the bare and all four
raised components at momenta 0.02, 0.01 and 0.005, holding the other edge
at 0.7. It retains the physical couplings and finite descendant sums.
Possible further cancellations in the complete spin-summed string
density must be checked separately.

The saved endpoint run contains 12 spectral points, each with eight bare
and 32 raised components. On the last halving step, the bare NS exponents
are within `9.54e-4` of two and the bare R exponents are within `1.71e-3`
of zero. Across raised components the largest exponent deviation is
`2.18e-2`, in a small radial R component. That deviation falls by a factor
`4.17` on halving P; the largest deviations in the other three edge tests
fall by factors `3.98–4.03`. This is consistent with a regular even
expansion about the stated powers, including the raised puncture.
The raw values and successive exponents are in
[`RR_MOMENTUM_DESIGN_CHECKS.json`](../../Data%20Set/type0b_rr_genus1_20260921/RR_MOMENTUM_DESIGN_CHECKS.json).

For a Gaussian variable \(x=aP^2\), if the full component is
\(f(P)=P^\beta e^{-aP^2}r(P)\),

\[
\int_0^\infty\frac{dP}{\pi}f(P)
=\frac{a^{-(\beta+1)/2}}{2\pi}
\int_0^\infty dx\,x^{(\beta-1)/2}e^{-x}
r\bigl(\sqrt{x/a}\bigr).
\]

Thus the independent `spectral/gaussian.py` backend uses
\(\alpha_{\rm NS}=+1/2\) and \(\alpha_R=-1/2\) for these generic
components. Applying an NS \(\alpha=+1/2\) rule to an R component without
dividing out its artificial weight leaves a singular residual or changes
the integral. The corresponding elementary long-tube integrals scale as
\(a^{-3/2}\) and \(a^{-1/2}\); this is not yet a statement about the fully
contracted string cusp, which includes other factors and cancellations.

The Gaussian backend integrates over the full positive axis. It supplies
both raw `dP/pi` weights and factored weights for the residual
`f/(P**beta*exp(-a*P**2))`; these two calling conventions must not be mixed.
Its largest sampled momentum is not a cutoff. For a sharply varying
residual, a large Gaussian order or an alternative panelled rule is still
needed. `spectral/check_gaussian.py` tests exact Gaussian moments at
`a=1,20,2000` and a physical OPE handle slice at fixed bridge momentum 0.75,
with separate R and NS nodes at orders 12, 24 and 36.

## Pole distances and unequal propagation scales

At \(b=1\), the zero lattices of \(\Upsilon_{\rm NS}\) lie at
\(0,-2,-4,\ldots\) and \(2,4,6,\ldots\); those of
\(\Upsilon_R\) lie at \(-1,-3,-5,\ldots\) and
\(3,5,7,\ldots\). Substituting \(\omega=it\), \(0<t<1/2\),
in the RRNS denominators gives:

| Sewing factor | Nearest relevant denominator pole candidates | Distance at \(t=1/4\) |
| --- | --- | ---: |
| External RR pair, NS bridge | \(P=\pm i(1-2t)\) in \(C_{\rm odd}\) | 0.5 |
| Mixed radial trinion | \(P_{\rm NS}-P_R=\pm i(1-t)\), plus sum-momentum images | 0.75 |
| OPE handle | \(P_{\rm handle}=\pm P_{\rm bridge}/2\pm i/2\), in NS \(C\) and R \(C_{\rm odd}\) | 0.5 |

These are inferred from the published structure constants; some other
denominator zeros cancel against numerators. The design checks approach
three representative uncancelled poles and record their residue limits.
The indicated moving poles do not cross the real continuum on the
specified continuation to \(i/4\). This audit does not authorize arbitrary
complex-energy continuation without contour tracking.

The exact radial primary Gaussian factors are

\[
e^{-a_{\rm NS}P_{\rm NS}^2-a_RP_R^2},\qquad
(a_{\rm NS},a_R)=
\begin{cases}
2\pi(y,\tau_2-y),&\delta=1,2,\\
2\pi(\tau_2-y,y),&\delta=3,4.
\end{cases}
\]

At the overlap geometry \(\tau=0.13+1.2i,z=0.04+0.18i\), these are
approximately \((1.131,6.409)\), or the reverse. The handle Gaussian in
the OPE chart is \(e^{-2\pi\tau_2P_{\rm handle}^2}\). Couplings and
descendants also affect the tail, so these factors alone do not bound it.
In the cusp the long edge should be rescaled with \(\sqrt a\); the short
edge requires its own resolution and tail test.

## Implemented rules and error separation

`momentum_quadrature.py` supplies two independent open rules:

1. Panelled Gauss-Kronrod, with embedded Gauss nodes and weights. It acts
   on the original integrand in \(P\), allowing a finite R endpoint.
2. A Gauss-Jacobi endpoint panel followed by Gauss-Legendre panels. The
   endpoint weight is divided out in the returned weights. Its NS edge
   can use \(\beta=2\); its R edge uses \(\beta=0\).

Both return weights for **\(dP/\pi\)**, avoid \(P=0\), and accept
independent panel boundaries and orders for each edge. No extra factor of
\(P^2\), \(\pi\), or two belongs in the caller.

`check_momentum_quadrature.py` evaluates each point once and records
\(I_{LL},I_{HL},I_{LH},I_{HH}\), where the letters select the low or high
rule on each edge. It compares \(I_{HH}-I_{LH}\) and
\(I_{HH}-I_{HL}\) independently at **fixed descendant cutoffs**. It
retains all component values, signed panel integrals and absolute panel
integrals to expose cancellation. The shell contributions with one or
both momenta in \([3.5,5]\) are separated from the \([0,3.5]^2\) core.
The region beyond five is not represented as bounded.

The driver supports the local modular test, radial channel, and OPE side
of the overlap test; the last two also support all four raised components.
The modular test uses bare primaries, for which the scalar S-transport
comparison is appropriate. Raised descendants need their own transport
matrix and are deliberately excluded from that scalar test.

Every node is checkpointed against settings and source hashes. The merged
bank records the two momentum arrays, their weights, raw component
values and array hashes. Old TT production, RR sphere data, and previous
RR genus-one pilots are unchanged.

## Results at omega = i/4

The full modular comparison keeps the OPE bridge order at four, the NS
handle through level two and the R handle through level two. The two
momentum integrations use different variables from the radial channel;
their results are not fixed-momentum block comparisons.

| Rule | First / second edge nodes | Total points | Domain |
| --- | ---: | ---: | --- |
| Gauss-Kronrod panels `[0,1.5,3.5,5]` | 37 / 37 | 1369 | `[0,5]^2` |
| Jacobi/Legendre panels `[0,0.5,1.5,3.5,5]` | 26 / 26 | 676 | `[0,5]^2` |

The second rule uses endpoint beta two on the NS bridge and beta zero
on the common handle rule. Beta zero allows both the finite R-handle
limit and the vanishing NS-handle limit; the separate infinite-domain
check below gives the NS handle its own beta-two nodes.

Across all 16 bare components (four theta sectors, two external families,
two local geometries), the rules agree to **`8.95e-8` relative**. The
embedded low/high comparisons vary just one edge at a time: their maximum
changes are `5.66e-4` on the bridge and `9.35e-4` on the handle. These
larger changes compare the high rule with its much coarser embedded rule;
they are retained as convergence evidence, not suppressed in favor of
the independent-rule comparison.

The high-rule S-transport ratios for `R+ R+` are

\[
(1.0000000039,\;2.0000014154,\;1.0000001696,\;0.4999995847).
\]

For `R- R-` they are

\[
(0.9999999925,\;2.0000015099,\;1.0000003696,\;0.4999994590).
\]

The independent Jacobi/Legendre rule reproduces this pattern. The
factor-two discrepancy therefore survives the new quadrature tests; it
is not explained by the old coarse momentum grid. This establishes an
interacting consistency failure of the present finite-level sewing
dictionary. It does not by itself determine which normalization or
transport convention needs correction, and no correction is applied.

The sum of absolute sampled integrands over the shell
`[0,5]^2` outside `[0,3.5]^2`, divided componentwise by the full integral,
is at most `1.27e-15` for Kronrod and `7.83e-16` for Jacobi/Legendre.
This shell is numerically negligible **at this geometry and these block
cutoffs**. It is not a rigorous bound beyond five or a cusp-tail bound.

The separate handle slice at bridge momentum 0.75 integrates the full
positive handle axis with sector-specific generalized Laguerre rules.
Its largest relative changes from 24 to 36 nodes are `3.09e-10` for R and
`1.26e-9` for NS. The 36-node values differ from the direct Kronrod panels
by at most `1.84e-8` and `7.01e-8`, respectively. Exact Gaussian moment
controls, including long-tube scales through `a=2000`, pass within
`1.44e-15` relative.

The aggregate record is
[`RR_MOMENTUM_QUADRATURE_SUMMARY.json`](../../Data%20Set/type0b_rr_genus1_20260921/RR_MOMENTUM_QUADRATURE_SUMMARY.json).
Raw point banks are retained beside the modular and Gaussian reports.
The raised radial/OPE overlap still needs its own two-edge convergence
study; the present modular result cannot supply its error estimate.

## Integrated identity normalization check

`spectral/check_identity_distribution.py` tests the stored couplings as
distributions, complementing the earlier pointwise ratio audit. With
\(P_I=i(1-\epsilon)\), it integrates

\[
\int_{P-w}^{P+w}dP'\,
\frac{C(P,P_I,P')}{2i\epsilon}\,\phi(P'),\qquad
\int_{P-w}^{P+w}dP'\,
\frac{C^\pm(P,P';P_I)}{2i\epsilon}\,\phi(P'),
\]

where \(\phi(P')=(1+P'^2)e^{-P'^2}\) and
\(w=\min(0.2,P/2)\). The same identity rescaling and test function are
used in all sectors. The substitution
\(P'-P=\epsilon\tan\theta\) resolves the narrow peak. Each test uses
64 and 128 nodes, at \(P=0.3,0.7,1.2\) and
\(\epsilon=0.01,0.003,0.001,0.0003\).

At the smallest epsilon and 128 nodes, the integrals divided by
\(\pi\phi(P)\) are:

| P | NS | R+ | R− |
| ---: | ---: | ---: | ---: |
| 0.3 | 0.9983877804 | 0.4990258997 | 0.4989026938 |
| 0.7 | 0.9989140539 | 0.4994021786 | 0.4993368754 |
| 1.2 | 0.9991662536 | 0.4995759063 | 0.4995338568 |

The NS areas approach one and the R areas approach one half as epsilon
decreases. The maximum relative 64-to-128-node change is `6.67e-6`, much
smaller than the factor-two effect. Finite-window/finite-epsilon effects
are retained; these numbers are not reported as exact distribution norms.

The local form of the implemented constants explains this behavior:
for fixed positive P and fixed u,

\[
C_{\rm NS}(P,P_I,P+\epsilon u)
\longrightarrow\frac{2i}{1+u^2},\qquad
C_R^\pm(P,P+\epsilon u;P_I)
\longrightarrow\frac{i}{1+u^2}.
\]

This identifies a relative identity-peak normalization of one half in
the stored coupling formulas. It does not establish that a primary
metric should be changed: primary normalization and the overall
partition/correlation-function constant belong to separate layers.
The [adopted prescription](INTERNAL_NORMALIZATION.md) keeps the original
primary/descendant construction and multiplies the completed correlators.
It passes the modular comparison. The remaining common amplitude
constant is reserved for string factorization.

The raw integrals are in
[`RR_IDENTITY_DISTRIBUTION_CHECKS.json`](../../Data%20Set/type0b_rr_genus1_20260921/RR_IDENTITY_DISTRIBUTION_CHECKS.json).

## Reproduce

From the repository root, using the project Python environment:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /private/tmp/hetso23-logic-review-py311/bin/python -B -u Code/type0b_rr_genus1/check_momentum_design.py --workers 2 --output 'Data Set/type0b_rr_genus1_20260921/RR_MOMENTUM_DESIGN_CHECKS.json'
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /private/tmp/hetso23-logic-review-py311/bin/python -B -u Code/type0b_rr_genus1/check_momentum_quadrature.py --channel modular --workers 3 --output 'Data Set/type0b_rr_genus1_20260921/RR_MOMENTUM_MODULAR_GK.json'
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /private/tmp/hetso23-logic-review-py311/bin/python -B -u Code/type0b_rr_genus1/check_momentum_quadrature.py --channel modular --scheme jacobi --workers 3 --output 'Data Set/type0b_rr_genus1_20260921/RR_MOMENTUM_MODULAR_JACOBI.json'
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /private/tmp/hetso23-logic-review-py311/bin/python -B -u Code/type0b_rr_genus1/spectral/check_gaussian.py --workers 2 --output 'Data Set/type0b_rr_genus1_20260921/RR_MOMENTUM_GAUSSIAN_SLICE.json'
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /private/tmp/hetso23-logic-review-py311/bin/python -B -u Code/type0b_rr_genus1/spectral/check_identity_distribution.py --workers 3 --output 'Data Set/type0b_rr_genus1_20260921/RR_IDENTITY_DISTRIBUTION_CHECKS.json'
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /private/tmp/hetso23-logic-review-py311/bin/python -B -u Code/type0b_rr_genus1/spectral/report_quadrature.py --directory 'Data Set/type0b_rr_genus1_20260921'
```

The finite-interval rule controls include known polynomial moments, which
test the measure and Jacobian, and Gaussian functions with nearby complex
poles. These controls do not certify the physical momentum integral.
Likewise a successful spectral refinement does not establish block,
global transport, cusp, moduli, or absolute S-matrix convergence.
