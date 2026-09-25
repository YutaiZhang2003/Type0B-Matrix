# Correlator normalization and the remaining factorization constant

The relative factor two belongs to the **overall partition/correlation
function**, separate from primary-state normalization and descendant
sewing. The primary metrics, inverse Gram matrices, trinion coefficients
and normalized conformal-block coefficients keep their original values.

For the existing radial and OPE conventions, write

\[
L^{\rm radial}_{\delta,\mathrm{normalized}}
 =2\,L^{\rm radial}_{\delta,\mathrm{raw}},\qquad
L^{\rm OPE}_{\delta,\mathrm{normalized}}
 =(2,2,1,1)_\delta L^{\rm OPE}_{\delta,\mathrm{raw}}.
\]

The multiplier sits **outside the completed descendant sum**. It is
constant in momentum, energy, geometry and descendant level. The OPE
NS-handle correlator sets the relative reference; the factors reconcile
the raw channel and spin-sector correlator conventions. They are not a
rule assigning a factor to each internal R line, and are not a change to
the Type 0B GSO weights or the spectral measure `dP/pi`.

## Implementation

[`internal_normalization.py`](internal_normalization.py) defines
`CORRELATOR`, `CorrelatorRadialTrace` and `CorrelatorLocalRROPE`:

- `CorrelatorRadialTrace.bare` and `.raised` first execute the original
  state sum, then multiply the returned correlator by two. They inherit
  the original inverse Grams, vertices and descendant coefficients.
- `CorrelatorLocalRROPE.liouville` first assembles the original OPE
  correlator, then applies its theta-sector prefactor. Its couplings and
  chiral block engines are unchanged.
- Collision polynomials are Taylor coefficients of that assembled
  correlator, so they receive its overall prefactor once. This does not
  modify the chiral descendant coefficients used to construct them.
  Inherited PCO contractions and ground-handle limits preserve this
  separation and do not apply a second multiplier.

The existing Clifford `1/2` in `LiouvilleCouplings.handle_weight` remains
unchanged. It belongs to the original state-trace construction. There
is no new `g_R=1/2` metric, inverse-propagator factor, or external-state
rescaling in this layer.

## Checks and saved data

[`RR_INTERNAL_NORMALIZATION_CHECKS.json`](../../Data%20Set/type0b_rr_genus1_20260921/RR_INTERNAL_NORMALIZATION_CHECKS.json)
checks both levels separately at \(\omega=i/4\) and \(0.2+i/4\):

1. NS/R inverse Grams, bare and raised descendant coefficients, chiral
   OPE arrays and trinion weights must equal the original values exactly.
2. Completed bare/raised correlators, PCO densities and collision
   polynomials must equal their original values times the declared
   overall prefactor, with the physical zero-mode relations preserved.

The independent bare modular comparison is at \(\tau=i\),
\(z=0.04+0.1i\) and \(z/\tau\), with theta permutation `(1,4,3,2)`.
The retained 1369-node Kronrod and 676-node Jacobi banks are converted
by multiplying their completed correlators. Both have maximum relative
modular residual below **`1.10e-6`** at the stated block cutoffs. The
independent quadratures agree within `8.95e-8`.
`RR_INTERNAL_MODULAR_GK.npz` and `RR_INTERNAL_MODULAR_JACOBI.npz` retain
the original sampling provenance and record the new prefactor explicitly.
These are finite-cutoff checks at the specified geometry, not a proof
of full moduli transport or full-amplitude convergence.

The retained 64-node bulk/OPE comparison, with the same correlator
prefactors, has maximum bare discrepancy `1.82e-3` and raised discrepancy
`5.74e-2`. The factor-two mismatch is removed, while raised components
still require independent block and momentum refinement.
`normalized/overlap.py` now uses the corrected correlator wrappers for
that study. The interrupted inverse-metric trial was stopped and archived;
its partial grid is not reported as a completed overlap calculation.

`RR_LOCAL_DENSITY_INTERNAL_PILOT.npz` multiplies the assembled correlators
and their collision expansions in the original local bank. It retains
all original couplings and the limits of the 16-node spectral pilot.

The historical identity-limit peak areas in
[MOMENTUM_QUADRATURE.md](MOMENTUM_QUADRATURE.md) do not by themselves
identify which normalization layer caused the factor two. In particular,
this prescription does not infer a replacement primary metric from them.

## One common constant from factorization

Define the eventual physical coefficient by

\[
a^{(1)}_{AA}(\omega)=C_{RR}\,I_{\mathrm{normalized}}(\omega).
\]

Overlap and modular covariance fix the relative correlator factors;
the common constant cancels in these comparisons. The code leaves it
`None` until a string factorization limit fixes it. The matrix prediction
and the TT conversion `2*pi*i` do not supply a fitted value.

In the **nonseparating torus degeneration**, cutting the handle gives a
sphere with the two external RR vertices and two intermediate vertices.
Compare the torus coefficient with that independently normalized sphere
state sum, using the same external states, pictures and plumbing measure.
With \(q=e^{2\pi i\tau}\),

\[
d^2\tau=\frac{d^2q}{4\pi^2|q|^2},\qquad
C_{RR}=\lim_{q\to0}
\frac{D^{\rm sphere}_s(q)}{D^{\rm normalized}_s(q)}.
\]

Here the same propagation factors and Jacobians have been removed from
both coefficients. The ratio must approach one common constant across
independent degeneration points or intermediate sectors. Time, ghosts,
GSO projection, external-state factors and sewing multiplicities must be
included. The original primary norms and descendant sums are used on
both sides.

For a Ramond degeneration, integration over the odd gluing modulus in
each chirality supplies the corresponding supercurrent zero mode on
the propagator. A matter-only state trace omits that string contribution;
see [Witten, section 5.4](https://arxiv.org/html/1306.3621#S5.SS4).
The RR collision limit gives an RRNS sphere vertex times an NS torus
one-point function. It cannot fix a common genus-one constant if that
one-point function carries the same undetermined constant.

[`factorization_normalization.py`](factorization_normalization.py)
records the matching convention and computes individual ratios without
averaging inconsistent limits. The prescription is saved in
[`RR_FACTORIZATION_NORMALIZATION.json`](../../Data%20Set/type0b_rr_genus1_20260921/RR_FACTORIZATION_NORMALIZATION.json).
A full-string degeneration comparison has not yet been evaluated;
the common constant and the integrated RR amplitude remain unset.

## Reproduce

From the repository root, using the project Python environment:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -B -u Code/type0b_rr_genus1/check_internal_normalization.py --directory 'Data Set/type0b_rr_genus1_20260921'
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -B -u Code/type0b_rr_genus1/normalized/overlap.py --workers 4 --output 'Data Set/type0b_rr_genus1_20260921/RR_INTERNAL_RAISED_OVERLAP_J12.json'
```

`history/QUADRATURE_V6_*` preserves the original raw quadrature checkpoint.
The superseded inverse-metric implementation and its partial run are
archived under `history/STATE_METRIC_TRIAL_20260922/`, explicitly marked
as superseded. Current reports use correlator-prefactor normalization.
