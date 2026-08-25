# Genus-one two-point Liouville momentum sampler

## Scope

This audit replaces the fixed production rule

`P = 6 u^2`, 16 Gauss--Legendre nodes per internal momentum,

by a local, correlated rule derived from the threshold and large-momentum
geometry of the genus-one two-point integrand.  The old high-statistics
worldsheet calculation is retained as a paired control variate; it is not
treated as the reference value of the Liouville integral.

## Exact threshold and propagation factors

For both the necklace and ordinary OPE channels, every real internal momentum
has the exact `b=1` BRY/DOZZ threshold zero.  The two-momentum density can be
written

\[
 dP_1 dP_2\;P_1^2P_2^2
 e^{-a_1P_1^2-a_2P_2^2}R(P_1,P_2),
 \qquad a_i=-2\log|q_i|.
\]

The propagation coordinates are

- necklace: `q1=exp(i z)`, `q2=exp(i(2 pi tau-z))`;
- OPE: `q_loop=exp(2 pi i tau)`, `q_ope=exp(-i z)-1`.

The OPE representation is used only when `|q_ope|<1`.  A point inside the old
geometric patch but outside that convergence domain is evaluated in the
necklace channel and explicitly labelled as a channel fallback.

For the analytically integrated collision disc,

\[
 \int_{|z|<\delta}d^2z\,|z|^{-2+2P_o^2}
 =\frac{\pi\delta^{2P_o^2}}{P_o^2}.
\]

The factor `1/P_o^2` cancels the OPE momentum's two threshold zeros.  The
correct disc reference measure is therefore

\[
 dP_\ell dP_o\;P_\ell^2
 e^{-a_\ell P_\ell^2-a_oP_o^2},
\]

not the ordinary two-threshold measure.  This distinction is covered by an
analytic quadrature normalization test.

## Large-momentum structure

At `b=1`,

\[
 \log|\Upsilon_1(1+i x)|
 =-x^2\log|x|+\frac32x^2+O(\log|x|).
\]

In a DOZZ constant the common `lambda^2 log lambda` and `lambda^2` pieces
cancel when several momenta are scaled together.  The surviving homogeneous
quadratic large-deviation function depends on their ratios.  For the
necklace constant with a fixed external momentum and
`(P1,P2)=lambda(r1,r2)`, its leading part is

\[
 \begin{split}
 \log|C(P_{\rm ext},P_1,P_2)|\sim\lambda^2\{&
 -4r_1^2\log(2r_1)-4r_2^2\log(2r_2)\\
 &+2(r_1+r_2)^2\log(r_1+r_2)
 +2(r_1-r_2)^2\log|r_1-r_2|\}.
 \end{split}
\]

It vanishes on the diagonal `P1=P2` and suppresses the transverse direction.
Thus the tail is a ridge, not a product of independent one-dimensional
Gaussians.  In the OPE channel,
`C(P_ext,P_ext,P_o)` additionally supplies the one-large-momentum factor
`exp(-4 log(2) P_o^2)`; the second constant
`C(P_loop,P_loop,P_o)` again produces a joint angular ridge.

## Correlated rule

Define scaled coordinates

\[
 x_i=\sqrt{a_i}P_i,\qquad
 v=x_1^2+x_2^2,\qquad
 y=\frac{x_1^2-x_2^2}{v}.
\]

For ordinary edges, the exact reference weight becomes

\[
 v^2e^{-v}\sqrt{1-y^2}\,dv\,dy,
\]

so the sampler uses generalized Laguerre `alpha=2` radially and Jacobi
`(alpha,beta)=(1/2,1/2)` angularly.  For the collision disc the reference is

\[
 ve^{-v}(1-y)^{-1/2}(1+y)^{1/2}\,dv\,dy,
\]

giving Laguerre `alpha=1` and Jacobi
`(alpha,beta)=(-1/2,1/2)`.  All Laguerre/Jacobi proposal factors are undone in
the returned weights.  Analytic model integrals close at machine precision.

## Measured geometry at `x=0.4`

The complete production-order block was decomposed on a `10 x 20` polar
rule.  The contribution sign was constant at every point (`|sum|/sum|term|=1`),
so the problem is concentration rather than cancellation.

| worldsheet sector | mean `v` | mean `y` | fitted common decay scale | fitted Jacobi `(alpha,beta)` |
|---|---:|---:|---:|---:|
| moderate bulk necklace | 2.1353 | -0.5295 | 1.4050 | (2.688, 0.135) |
| moderate OPE | 2.3007 | +0.4416 | 1.3040 | (0.137, 1.936) |
| moderate analytic disc | 1.9158 | +0.5999 | 1.0440 | (-0.497, 1.012) |
| cusp bulk necklace | 2.8791 | -0.0271 | 1.0420 | (0.581, 0.498) |
| cusp OPE | 2.1446 | +0.3459 | 1.3988 | (0.103, 1.269) |
| cusp analytic disc | 1.8169 | +0.6182 | 1.1008 | (-0.541, 0.944) |

The opposite signs of the necklace and OPE angular shifts directly exhibit
the channel-dependent DOZZ ridges.  A single global `q_eff` cannot represent
both.

## Fixed-point convergence audit

At the moderate bulk point, the product threshold sequence requires about
`Q=30` (900 nodes) to reach a `1.96e-5` adjacent step.  The correlated polar
sequence reaches a `3.84e-5` step at `20 x 22` (440 nodes).  The moderate OPE
sector reaches `6.36e-6` at `20 x 22`; the analytic disc is already stable to
`3.50e-7` at `12 x 14`.  Cusp bulk, OPE, and disc reach respectively
`8.54e-11`, `7.85e-6`, and `3.75e-7` by `12 x 14`, `16 x 18`, and `12 x 14`.

Relative to the old production `Q=16` rule, representative local shifts are

| sector | converged/new minus old |
|---|---:|
| moderate bulk | -1.08% |
| moderate OPE | -0.50% |
| moderate disc | -0.37% |
| cusp bulk | +4.55% |
| cusp OPE | +5.40% |
| cusp disc | +4.70% |

These opposite movements are why pointwise evidence cannot be converted into
an amplitude correction without re-integrating the worldsheet distribution.

## Amplitude campaign

The production correction uses eight scrambled-Sobol replicas at each of
`x=0.2,0.4,0.6,0.8`, with 256 paired points for the finite region at
`tau2<=8` and 256 paired points at each tail slice
`tau2=8,10,12,16,20`.  The local polar order schedule is

`8x10 -> 12x14 -> 16x18 -> 20x22 -> 24x26`,

with a `5e-5` adjacent drift gate.  Disc orders stop at `16x18`.  The saved
high-statistics fixed-rule replica is combined with the paired
`new-old` correction before computing the replica standard error.  A
conservative momentum residual is propagated separately.

The Cannon array jobs are `40252525` (192 tasks, throttle 96) and `40252529`
(dependent finalizer).  At submission they were pending for priority with no
scheduler start estimate.
