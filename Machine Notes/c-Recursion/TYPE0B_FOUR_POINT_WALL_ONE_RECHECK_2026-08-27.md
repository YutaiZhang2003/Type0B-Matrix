# Type-0B four-point wall-one numerical recheck (2026-08-27)

## Verdict

The ten-point `sobol_power=6` table is not a converged amplitude table and
must not be used for a matrix-model test.  All ten points fail a target-blind
15% relative-standard-error gate; their relative randomized-QMC errors range
from 23.7% to 344%.  The nominal aggregate chi-square previously quoted from
that table is therefore invalid for inference.

The large deviations are not explained by the BRY normalization, momentum
order, conformal-block order, hybrid routing, or wall-one Laurent residue.
They arise primarily from insufficient resolution of logarithmic radial
oscillations near the moduli boundary.

## BRY normalization

Writing the continued reduced moduli integral as

\[
\mathcal M=\int_{\mathbb C}d^2z\,\mathcal I_T(z),
\]

BRY's sphere prefactor and dictionary give, after stripping
\(\delta(\omega-\sum_i\omega_i)\mu_F^{-2}\),

\[
\mathcal A_{\rm WS}=\frac{8i}{\pi}\mathcal M,
\qquad
\mathcal A_{\rm MQM}=8i\omega\omega_1\omega_2\omega_3(1+2i\omega).
\]

No additional leg-pole factor is present.  These formulas agree with BRY
Eqs. (2.13), (4.13), and (4.14).

## Intermediate checks at \(x=0.270,t=0.604\)

The following tests use the 30-node composite momentum rule.

- The 30-node and global 96-node total densities agree to relative errors
  `3.20e-4` and `3.39e-4` at `z=0.37+0.28i` and `z=0.55+0.18i`.
- Increasing the block cutoff from twice-level 8 to 12 changes the same
  pointwise densities by `1.26e-6` and `9.77e-6`.
- Hybrid and pure c-recursion densities, including the complex-momentum
  residue block, agree to displayed precision at three representative
  direct/inversion points.
- The wall-one Laurent coefficient is unchanged to displayed precision when
  its extraction radius varies from `1e-2` to `1e-4` and the circular sample
  count increases from 32 to 64.
- The complete continued correlator has frame-0/frame-1 crossing spread
  `8.23e-4` at the audit point.

## Source of the variance

Near the raised--raised divisor, the continued powers include

\[
\beta_{\rm cont}=0.167664+1.30464i,
\qquad
\beta_{\rm res}=0.416+1.08i.
\]

The importance sampler cancels the small positive real power but not the
large imaginary part.  The residual factor oscillates rapidly as
\(\exp(i\,\operatorname{Im}\beta\log|1-z|)\).  At 64 Sobol points, the largest
weights all lie in the `z=1` chart and have magnitude about 8--11 with widely
varying phases.  The desired answer is obtained from cancellations among
these terms, so four 64-point replicates cannot resolve it.

## Nested-depth anchor

For the old negative-real diagnostic sheet, the original point
`x=0.270,t=0.604` gave a 115.4% central discrepancy.  At 512 samples per
replicate, keeping every other numerical setting fixed,

\[
\mathcal M_{\rm WS}=0.16779149(1948)-0.20082897(2732)i,
\]

while

\[
\mathcal M_{\rm MQM}=0.16392175-0.17723104i.
\]

The central relative discrepancy falls to 9.91%, and the two-component
residual has \(\chi^2=0.785\).  This is statistically consistent but is still
not a percent-level confirmation.

## Corrections to the workflow

1. Future scans use `+x+i*t`, connected to BRY's positive real energy axis,
   rather than the legacy `-x+i*t` sheet.
2. The default depth is raised from 64 to 512 samples per replicate while
   retaining the 30-node momentum rule.
3. A target-blind 15% relative-standard-error gate is applied before a point
   is marked integrated.
4. Matrix comparison output is labelled `unconverged-moduli-scan` and its
   chi-square is invalidated whenever this gate fails.
5. Percent-level work should next integrate the leading complex OPE powers
   analytically (or at fixed momentum before the final momentum quadrature)
   instead of resolving logarithmic radial oscillations by brute-force QMC.
