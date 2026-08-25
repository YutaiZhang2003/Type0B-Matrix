# Liouville Genus-Two Pair-Of-Tori Layer

This note accompanies `liouville_genus2.py`.  It is the first code layer for
computing a genus-two Liouville partition function by plumbing two genus-one
surfaces.

## 1. Channel

The channel is the separating pair-of-tori plumbing.  The two genus-one
components have plumbing moduli \(q_1,q_2\), and the bridge has modulus
\(q_3\).  In CFT language the bridge inserts a complete set of Liouville states
between two one-punctured tori.

The current implementation computes the primary bridge term

\[
Z^{(0)}_{2}(q_1,q_2,q_3)
=
\int_0^\infty {dP_3\over \pi}\,
|q_3|^{2(h_{P_3}-c/24)}
G(P_3,q_1)G(P_3,q_2),
\]

where

\[
G(P_{\rm ext},q)=
\int_0^\infty {dP\over\pi}\,
C(P,P_{\rm ext},P)
\left|F^{h_{P_{\rm ext}}}_{c,h_P}(q)\right|^2 .
\]

The function \(G\) is exactly the torus one-point integral already implemented
in `liouville_torus.py`.  Therefore this first genus-two layer already includes
the full descendant sums around the two genus-one handles.

## 2. What Is Still Missing

The full genus-two Virasoro block has bridge descendants.  In this notation the
implemented answer is the zeroth term of the bridge expansion,

\[
\mathcal F_{2}
=
q_3^{h_{P_3}-c/24}
F_{\rm torus}(P_3,q_1)
F_{\rm torus}(P_3,q_2)
\left(1+O(q_3)\right).
\]

Those bridge-descendant terms are the next natural step.  They will introduce
the twist-angle dependence of \(q_3\); the primary term depends only on
\(|q_3|\).

## 2a. CCY Two-Holed-Disc Channel

The module `ccy_genus2_block.py` implements the genus-two Virasoro block in the
two-holed-disc plumbing frame of Cho-Collier-Yin,

\[
\mathcal F_{\rm CCY}
=\sum q_1^{|A|}q_2^{|C|}q_3^{|E|}
G_{h_1}^{AB}G_{h_2}^{CD}G_{h_3}^{EF}
\rho(L_{-A}h_1,L_{-C}h_2,L_{-E}h_3)
\rho(L_{-B}h_1,L_{-D}h_2,L_{-F}h_3).
\]

The evaluator follows their central-charge recursion.  Its regular part is the
large-\(c\) seed: the global \(SL(2)\) block times an optional finite Schottky
product approximation to the \(c=\infty\) vacuum block.  Production evaluations
resum the full global-descendant tower before truncating the pole recursion. In
the theta channel one edge is a Gauss \({}_2F_1\) and the two remaining level
sums are converged adaptively.  In the glasses channel the entire global block
factorizes into three Gauss functions.  A legacy strict total-degree global
sum remains available for coefficient checks.  The pole terms use

\[
c_{r,s}(h)=1+6\left(b_{r,s}+b_{r,s}^{-1}\right)^2,\qquad
b_{r,s}(h)^2=
{rs-1+2h+\sqrt{(r-s)^2+4(rs-1)h+4h^2}\over 1-r^2},
\]

and the genus-two residues

\[
q_i^{rs}A_{r,s}^{c_{r,s}}
\left(P^{r,s}_{c_{r,s}}\right)^2
{ -\partial_h c_{r,s}(h_i)\over c-c_{r,s}(h_i)}
\mathcal F(h_i\to h_i+rs,c\to c_{r,s}(h_i)).
\]

The Liouville wrapper `liouville_genus2_ccy.py` evaluates the corresponding
raw plumbing-frame three-momentum integral.  Following CCY, the block
\(\mathcal F_{\rm CCY}\) contains the descendant powers \(q_i^{N_i}\), while
the separated primary propagation factors \(\prod_i q_i^{h_i}\) multiply the
block outside the recursion:

\[
Z_{\rm CCY,raw}^{(N)}
=\int_0^\infty {dP_1dP_2dP_3\over \pi^3}\,
C(P_1,P_2,P_3)^2
\left|
\prod_{i=1}^3 q_i^{h_i}
\mathcal F_{\rm CCY}^{(N)}(h_1,h_2,h_3;c;q_i)
\right|^2.
\]

This is not yet a modular-invariant genus-two partition function; it is the raw
CCY two-holed-disc conformal frame object.  Cylinder Casimir factors or other
conformal-frame anomaly factors must be derived separately before comparing
different plumbing frames as physical partition functions.

Run the local recursion checks with:

```bash
.venv/bin/python plumbing/ccy_genus2_block_checks.py
```

Run a tiny Liouville smoke sample with:

```bash
.venv/bin/python plumbing/liouville_genus2_ccy.py \
  --b 0.8 \
  --q1 0.003+0.001i \
  --q2 0.002-0.0005i \
  --q3 0.001 \
  --block-order 1 \
  --quadrature-order 2 \
  --p-max 1.2 \
  --dps 25
```

## 3. Liouville Data

The code uses the same Xi/Yin normalization as `liouville_torus.py`:

\[
Q=b+b^{-1},\qquad c=1+6Q^2,\qquad h_P={Q^2\over4}+P^2,
\]

with two-point normalization \(\pi\delta(P-P')\).  This gives the completeness
measure \(dP/\pi\) for every Liouville momentum integral.

For generic real \(b>0\), the Virasoro-block recursion is non-degenerate.  At
exactly \(b=1\), the generic recursion has colliding Kac labels, so the code
keeps the existing guard from `liouville_torus.py`.  Use a nearby regulator such
as \(b=1+\epsilon\) until the analytic collision limit is implemented.

## 4. Usage

From Python:

```python
from plumbing.liouville_genus2 import liouville_genus2_pair_of_tori

result = liouville_genus2_pair_of_tori(
    b=0.8,
    q1=0.02,
    q2=0.018,
    q_bridge=0.01,
    block_order=2,
    bridge_quadrature_order=4,
    handle_quadrature_order=6,
    dps=25,
)
print(result.value)
```

From the command line:

```bash
.venv/bin/python plumbing/liouville_genus2.py \
  --b 0.8 \
  --q1 0.02 \
  --q2 0.018 \
  --q-bridge 0.01 \
  --block-order 2 \
  --bridge-quadrature-order 4 \
  --handle-quadrature-order 6 \
  --dps 25
```

Run the smoke checks with:

```bash
.venv/bin/python plumbing/liouville_genus2_checks.py
```

## 5. Modular-Covariance Diagnostic

The module `liouville_genus2_modular_check.py` tests modular transformations in
period-matrix variables:

\[
\Omega'=(A\Omega+B)(C\Omega+D)^{-1},
\qquad
\begin{pmatrix}A&B\\ C&D\end{pmatrix}\in Sp(4,\mathbb Z).
\]

It uses `schottky_glasses_period_matrix` for the forward map
\((q_1,q_2,q_3)\mapsto\Omega\), applies the selected \(Sp(4,\mathbb Z)\)
transformation, and then tries to invert \(\Omega'\) back to plumbing
coordinates.

The only faithful mode currently implemented is `--target-chart original`,
which performs a numerical inverse in the original glasses plumbing chart.  It
is therefore a real numerical test only for modular images that still lie in
that chart.  Large transformations such as the full \(S\)-move generally report
chart failures.

The option `--target-chart modular-image` is intentionally labelled
bookkeeping-only.  It pulls \(\Omega'\) back by \(G^{-1}\), recovers the original
glasses coordinates, and pushes the represented period matrix forward again.
This is useful for checking symplectic bookkeeping, but it is not an independent
CFT modular-invariance test and is not counted as a pass.

For the Liouville matter factor by itself, without the \(bc\) ghost or another
choice of determinant-line trivialization, the expected covariance is the
absolute square of the common chiral section convention,

\[
Z_{\rm chiral\ section}(\Omega',\bar\Omega')
=|\det(C\Omega+D)|^{-c}Z_{\rm chiral\ section}(\Omega,\bar\Omega).
\]

A scalar invariant comparison,

\[
Z(\Omega',\bar\Omega')=Z(\Omega,\bar\Omega),
\]

is only appropriate after anomaly cancellation or after explicitly dividing by
the corresponding determinant-line factor.

The honest generator-level diagnostic uses the standard generators consisting
of the three symmetric translations, two \(GL(2,\mathbb Z)\) basis moves, and the
full \(S\)-move:

```bash
.venv/bin/python plumbing/liouville_genus2_modular_check.py \
  --expected-law chiral-section \
  --suite sp4-generators
```

It reports pass, chart-failure, modular-mismatch, or bookkeeping-only for each
generator.  A chart failure means the current code does not yet have the needed
independent plumbing chart for that modular image.

The bookkeeping-only comparison is available explicitly:

```bash
.venv/bin/python plumbing/liouville_genus2_modular_check.py \
  --expected-law chiral-section \
  --suite sp4-generators \
  --target-chart modular-image
```

The embedded handle \(S\)-move is also available as `--transform handle-s-1` or
`--transform handle-s-2`.  In practice this is the first nontrivial test to try,
but the current leading primary-bridge approximation does not have the full
matter automorphy factor.  The diagnostic
`liouville_genus2_handle_s_leading_check.py` makes this visible by comparing the
small-\(q_3\) leading term to both the expected Liouville matter factor
\(|J|^{-c}\) and the primary-term trend.
