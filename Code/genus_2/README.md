# Type-0B genus two: theta and glasses channels

## Provisional NSRR normalization (2026-09-16)

The current factor-four experiment explicitly uses `M_trial = 4 M_local`.
This is a **user-requested assumption**, motivated by the quadrature limit,
and is not inferred from the locally checked BPZ or torus normalization.
`nsrr_normalization.py` applies it once to the full new bilinear matrix,
including interference, and labels every result `normalization_status="assumption"`.
The coefficient in each supported eta sector becomes `B_L B_R/2`.
Primary powers, supplied BRY constants and complex block phases are unchanged.

Use `contract_normalized_nsrr` or select
`--sewing-convention human-bilinear --normalization provisional-times-four`
in `run_nsrr_resummed_frozen_grid.py`, with explicit physical tube signs.
The pointwise API retains the factor-one default for historical callers;
the new convergence study explicitly selects factor four.
This option is distinct from `legacy-times-four`, which uses the old matrix.

`check_nsrr_factor4_convergence.py` recombines saved complex L8 coefficients
on complete 343-node and 1728-node source grids, reuses the completed
N3–N10 quadrature study, and adds an all-NS R16 control at N3.
See [the provisional comparison](../../Data%20Set/nsrr_provisional_factor4_20260916/README.md).
The factor-one results below are preserved as historical references.

## Human Note bilinear convention (2026-09-15)

`human_bilinear_sewing.py` assembles `F^T M Ftilde` with independently
specified antiholomorphic data. For even NS primaries the literal theta
matrix is `diag(C_L^(0) C_R^(0), -C_L^(1) C_R^(1))`. The quadratic parity
sign stays inside the blocks. Each pant uses `C_HN^(1)=i*Ctilde_BRY`.
All primary powers remain explicit, outside the blocks and matrix.
The finite basis rule is `M'=U^(-T) M V^(-1)`; a weight-changing map also
requires `F'=D_(P')^(-1) U D_P F`.

`audit_human_bilinear_sewing.py` compares this all-NS assembly with full
graded state sewing, including descendants, complex coefficients, all four
NS lift choices, and unequal holomorphic/antiholomorphic primary factors.
It also checks the general NSRR tensor formula before physical restriction.
See [the bilinear audit](../../Data%20Set/human_bilinear_sewing_20260915/README.md).

`nsrr_bilinear_sewing.py` now supplies the **physical NSRR** matrix. In each
eta-pair sector it is `B_L,eta B_R,eta'/8 * [[1,s],[s,1]]`, supported on
`eta*eta'=-r`. Here `s=omega_NS*omega_R0`, `r=omega_R1*omega_R0` are physical
full-state tube signs, separate from the two-lift block projection. The
descendant-compatible two-family BPZ dual has been derived and checked
against independent two-point Ward sewing. The audit includes 1,024
descendant coefficients, the torus trace, and a single-Majorana phase check.
See [the NSRR derivation](../../Machine%20Notes/Genus%202/NSRR_BILINEAR_PAIRING_2026-09-15.md).

`nsrr_resummed_sewing.py` uses this matrix by default, with explicit physical
tube signs and primary factors. `legacy-times-four` is an explicit historical
option. `physical_nsrr_sewing.py` retains the old candidate for reproduction.
The raw marked-spin all-NS/NSRR cross-channel comparison is a separate
transport problem: even in the free theory, the literal NSRR combination
is `P*G_s=D_00-i*s*D_11`, not one raw spin block.

`test_nsrr_bilinear_cross_channel.py` now tests the two even RR components
against fresh complex all-NS lift vectors, using separately checked free
spin transport. **This test fails:** normalized ratios are `0.251–0.253`
across five surfaces, and the phase-independent spin sum has the same gap.
The central N3→N4 quadrature change is below 0.28%; the block-cutoff changes
are smaller. This invalidates a claim of established crossing for the
proposed assembly, without isolating the problem to the chiral
double-Virasoro blocks. The interacting off-diagonal spin transport remains
part of the tested proposal. No fitted factor four is applied. See
[the full test and cutoff controls](../../Data%20Set/nsrr_bilinear_cross_channel_20260915/README.md).

The subsequent [quadrature refinement](../../Data%20Set/nsrr_bilinear_quadrature_20260915/FINAL_REPORT.md)
shows that the N3/N4 values were not stabilized. At the central surface
`t=0.60`, the source stabilizes at N7 and the target at N10, with L5/R12
held fixed. The spin-sum ratio is `0.249928019`; the two sign ratios are
`0.250052595` and `0.249813244`. Each integral and the full complex spin
matrices pass two successive relative changes below 0.01%. The last target
change is 0.00162%. The other four surfaces have not received this refinement.
No factor four is applied. Reproduce the completed study with
`reproduce_nsrr_bilinear_quadrature.py`, followed by
`check_nsrr_quadrature_matrices.py` and `report_nsrr_quadrature_limit.py`.

## Numerical block evaluation

The pointwise NSRR geometry adapter defaults to independently resummed
double-Virasoro blocks. `nsrr_resummed_sewing.py` supplies these blocks to the
existing nonchiral contraction; `run_nsrr_resummed_frozen_grid.py` evaluates
the existing saved momentum inputs and requires a complete grid before
publishing a ratio. See
[the resummation guide](../full_ramond_block_runtime/GLOBAL_RESUMMATION.md).
Historical coefficient-bank scripts and frozen cluster bundles remain
finite-polynomial calculations.

This folder contains the parity-sensitive assembly of the all-NS genus-two
theta- and glasses-channel contributions.  The theta sewing convention comes
from `Human Notes/SCblock.tex`, especially `NSblockThetaDefinition`; the
derived glasses recursion and its PBW audit are recorded in
`Machine Notes/c-Recursion/ns_genus_c_recursion.tex`.

The chiral block uses the relative label

```text
a = A + C + E mod 2,
```

while the nonchiral partition term carries

```text
(-1)^(a + p1 + p2 + p3).
```

The antiholomorphic block is selected by

```text
a + sum(p_i) = a_tilde + sum(p_tilde_i) mod 2.
```

For the diagonal propagating Type-0B NS continuum, the internal primaries are
even in both chiral halves.  The theta numerator at each momentum node is
therefore assembled as

```text
C_0^2 |q^h F_0|^2 - C_1^2 |q^h F_1|^2,
```

not as the old unsigned sum.  `theta_partition.py` also records the discrete
transport in the chiral `c`-recursion: an odd level-`rs/2` null flips the
relative sector and the two spectator plumbing lifts, and contributes the
lift on the null edge.

The finite-c blocks, fusion polynomials, Schottky vacuum seed, and numerical
quadrature remain in `Code/c_Recursion`.  Both the local evaluator and Cannon
worker import the assembly functions from this folder.  Existing production
*totals* use the old unsigned formula and must not be mixed with new totals.
An archive that retains separate sector contributions can, however, be
corrected exactly without reevaluating its conformal blocks:

```bash
PYTHONPATH=Code python3 Code/genus_2/recombine_theta_parity.py \
  Machine\ Notes/Genus\ 2/Archives/ns-genus2-fivepoint-r24-n10-production-repro.tar.gz \
  --output Data\ Set/ns_genus2_fivepoint_r24_n10_theta_parity_corrected.json
```

The utility verifies the archive hash-independent provenance fields, all
10,000 task indices, the sector decomposition of every theta node, and the
reconstruction of each old unsigned summary row before applying the sign.

Run the focused tests from the repository root with

```bash
PYTHONPATH=Code python3 -m unittest discover -s Code/genus_2 -p 'test_*.py' -v
```

This is the corrected all-NS contribution in one theta plumbing spin sector.
It is not, by itself, the full BRST-complete genus-two Type-0B amplitude: the
remaining NS/R channels, GSO sum, superghost measure, and odd-moduli treatment
must be supplied before comparing a complete genus-two free energy with the
matrix model.

## Complex single-Majorana free factor

Use `fixed_spin_free_plumbing.fixed_spin_chiral_partition` when retaining
the complex amplitude of one Majorana:

```python
from fixed_spin_free_plumbing import fixed_spin_chiral_partition

free = fixed_spin_chiral_partition(
    q_values, omega_marked, characteristic, period_branch=B,
)
majorana = free["majorana_chiral"]
boson_times_majorana = free["superfield_chiral_oscillator"]
```

The evaluator computes `sqrt_cont(P * xi_B * theta_marked)`, with the
theta-translation phase derived from the supplied integer period branch.
For the saved NSRR spin `[11|00]` and `B=diag(0,1)`, `xi_B=exp(i*pi/4)`:
one Majorana carries the `exp(i*pi/8)` correction. This factor is already
included in the returned amplitude; do not multiply it a second time.

The initial sign is fixed at degeneration: `M_NS=1+...` and
`M_RR=sqrt(2)*prod_R q_e^(1/16)*(1+...)`. Propagators are `q^L0`.
For successive samples in one continuously marked chart, pass the previous
result as `previous=free`. Its `chiral_branch` records the continued plumbing
logs, windings, and normalization. The input `B` always relates the
principal-log charge period to the supplied marked period, so it must track
log cuts consistently with that marking. Spin or unaccounted period-marking
changes are rejected; the API does not infer a modular chart multiplier.

The existing `fixed_spin_partition` remains the nonchiral evaluator.
Its `Z_majorana` and `Z_free` are unchanged; it now also returns
`theta_translation_phase`, the phase-correct
`dirac_chiral_from_marked_theta`, and a complex theta comparison.
Odd-spin vacuum amplitudes vanish and their chiral phase is `None`.

`audit_free_bosonization_phase.py` now exercises the chiral evaluator against
independent Heisenberg and Clifford-mode sewing on the ten saved surfaces,
including a Ramond winding path. See
[the norm and phase report](../../Data%20Set/free_bosonization_phase_20260915/README.md).
Focused regressions are in `test_fixed_spin_chiral_partition.py` and
`test_fixed_spin_free_plumbing.py`.

## Fixed-spin NSRR sewing

All physical NSRR blocks now use the native C++ double-Virasoro backend in
`Code/full_ramond_block_runtime/nsrr_cpp_backend.py`. Equal and opposite HJS
signs use its ordinary and inserted pipelines, respectively, at 40 digits by
default. The factorized, refined, cluster, and off-axis source evaluators share
this backend; none completes a physical block using PBW. The analytic ground
and half-level normalization checks remain active. Old PBW-assisted source
shards are rejected as reusable production input.

After `make -C C++`, rerun the ten-surface fixed-normalization comparison with:

```sh
PYTHONPATH=Code:Code/genus_2:Code/full_ramond_block_runtime:Code/c_Recursion:Code/genus_2_cross_channel \
  /path/to/python Code/genus_2/rerun_nsrr_double_virasoro.py \
  --baseline 'Data Set/nsrr_nsnsns_generic_10point_N4_20260904' \
  --output 'Data Set/nsrr_double_virasoro_N4_L8_20260911' \
  --levels 5 6 7 8 --workers 6
```

This computes every NSRR block through total source order 8 and evaluates
cumulative orders 5, 6, 7, and 8 from that expansion. The momentum quadrature
order is independent: N4 uses 64 momentum nodes and N5 uses 125. The verified
saved all-NS target at recursion twice-level 16 is the control. Geometry,
quadrature, spin lifts, free factors, and the fixed normalization are unchanged.
The old protected-kernel snapshot is retained in `nsrr_checked_kernel_manifest.json`
alongside the refreshed snapshot following the upstream update; native source
and executable hashes are recorded separately in each new run.

The [order-8 rerun report](../../Data%20Set/nsrr_double_virasoro_L8_20260911/README.md)
separates block-order convergence from momentum-quadrature refinement. At
source order 8 the maximum channel differences are `1.023925%` (N4) and
`0.908404%` (N5). The largest relative source change from order 7 to 8 is
`2.53e-6`, while changing the momentum rule from N4 to N5 changes the source
by up to `0.8675%`; momentum convergence remains unresolved.
The [initial order-5 rerun report](../../Data%20Set/nsrr_double_virasoro_20260911/README.md)
contains the ten-surface N4 comparison, the five-surface N5 refinement, and
the residual disagreement at each source order. The order-3 source reproduces
the previous PBW-assisted result within `6.9e-15`; the largest remaining
cross-channel difference at source order 5 is `1.024%` (N4) and `0.9085%` (N5).

For direct momentum refinement, `run_nsrr_nsnsns_offaxis_constant_scan.py
prepare-native-order` accepts momentum orders through N7 and a separate
`--source-level` (default 5). It imports the saved marked geometry and fixed
normalization, fingerprints the current numerical implementations, and
recomputes both channels. The all-NS target remains at recursion twice-level
16. For example, prepare a new output directory with:

```sh
PYTHONPATH=Code:Code/genus_2:Code/full_ramond_block_runtime:Code/c_Recursion:Code/genus_2_cross_channel \
  /path/to/python Code/genus_2/run_nsrr_nsnsns_offaxis_constant_scan.py prepare-native-order \
  --base-config 'Data Set/nsrr_nsnsns_generic_10point_N4_20260904/config.json' \
  --orders 7 --source-level 5 --output-dir /tmp/nsrr-n7-l5
```

Use the same script's `run --output-dir /tmp/nsrr-n7-l5` command to compute
and resume the two channels. The [N7 comparison report](../../Data%20Set/nsrr_double_virasoro_N7_L5_20260911/README.md)
records the fixed-order momentum convergence.

The recorded N7 calculation uses the calibrated lower-depth target via
`Data Set/nsrr_double_virasoro_N7_L5_20260911/run_depth5.py`: null level 5
(R=10), endpoint cutoff 8, and a cutoff-10 control at every node. The full
N4 integration changes by at most `1.25e-7` relative to the original deeper
target. The [depth audit](../../Data%20Set/nsrr_double_virasoro_N7_L5_20260911/validation/all_ns_depth/README.md)
distinguishes the endpoint cutoff and null level from a strict total-q
truncation. The standard runner above retains its R=16 reference settings.
The completed N7 run has maximum cross-channel discrepancy `0.8576%`;
raising the target endpoint cutoff from 8 to 10 changes the integral by only
`4.37e-11` relative. Some surfaces worsen under momentum refinement, so
the direct N7 test alone does not establish momentum convergence.

The subsequent [momentum-integration comparison](../../Data%20Set/nsrr_momentum_integration_20260911/README.md)
keeps NSRR order 5 and the all-NS R=10/K=8 setup. It integrates inexpensive
controls with Gaussian rules directly on positive momentum at N9 and N12,
and retains the full-order correction on the original N7 grid. N12 is the
largest block momentum grid for this round. The N9-to-N12 changes are at most
`1.49e-6` in the source and `7.85e-7` in the target; changing the correction
grid from N4 to N7 moves the channel ratio by at most `7.46e-6`. The remaining
maximum cross-channel discrepancy is `0.8593%`, much larger than these
measured integration changes. These are convergence diagnostics, not
rigorous error bounds or evidence of cross-channel agreement.

The [independent threshold audit](../../Data%20Set/nsrr_momentum_integration_20260911/threshold_audit.md)
also recomputes the full-order correction using the threshold half-Gaussian
N7 rule. The order-convergence driver now uses that validated rule by default
for both grids:

```sh
PYTHONPATH=Code:Code/genus_2:Code/full_ramond_block_runtime:Code/c_Recursion:Code/genus_2_cross_channel \
  /path/to/python Code/genus_2/run_nsrr_threshold_order_scan.py
```

It varies NSRR total order and all-NS null level through 5, 6, 7, and 8, while
holding the N12 controls, N7 correction nodes and weights, K8 endpoint cap,
vacuum cutoffs, precision, geometry, and normalization fixed. It checks that
the run manifest differs from the threshold baseline only in the block-order
lists, and that the order-5 restrictions reproduce every saved baseline node.
The report distinguishes changes within each channel (expected to approach
zero) from the cross-channel ratio (expected to approach one). Use `--status`
for progress or `--reduce-only` to regenerate the results without new block
evaluations. The numerical options expose block orders only.

The completed [fixed-grid order check](../../Data%20Set/nsrr_threshold_order_scan_20260911/README.md)
uses 343 fresh nodes per channel at orders 5 through 8. Raising both block
cutoffs changes the maximum discrepancy from `0.859304620%` to `0.859254104%`.
The largest relative NSRR step decreases from `4.506e-5` (5 to 6) to
`2.516e-6` (7 to 8); the corresponding all-NS steps are `1.216e-7` and
`1.620e-9`. All recomputed order-5 node values reproduce the threshold
baseline exactly. No other integration or block-resummation parameter moves.

The historical candidate in `physical_nsrr_sewing.py`, retained for
reproduction only, projects geometry
lifts as `(F_(+,+,+)+F_(+,-,+))/sqrt(2)`, maps BRY coefficients by
`(C_even,C_odd) -> (C_even/2,C_odd/2)`, and contracts the two chiral
form-parity blocks as

```text
(1/4) |F_0 + i eta_left eta_right F_1|^2.
```

The identity-NS degeneration gives `1^2+1^2=2`; the unscaled modulus gives
`8`. That limit does not test the nonzero NS half-level term which the
candidate incorrectly cancels in radial reflection. It does not establish
the Human Note bilinear dual or a physical fixed-spin genus-two matrix.
`recombine_physical_nsrr_saved.py` applies this formula to the completed
L3/N5 and L5/N3 shard sets, and can additionally reuse the saved central
source-N6/target-N7 refinement.  It computes no new conformal-block nodes:

```bash
PYTHONPATH=Code:Code/genus_2 python3 \
  Code/genus_2/recombine_physical_nsrr_saved.py \
  --central-refinement 'Data Set/nsrr_spin_quadrature_t060_20260830'
```

The historical derivation and comparison are recorded in
`Machine Notes/Genus 2/NSRR_ORDER8_CONVENTION_TROUBLESHOOTING_2026-09-02.md`.

## Glasses channel

At either glasses trinion the handle primary occurs twice, so its intrinsic
parity cancels.  The absolute parity and nonchiral sign are

```text
a_abs = a + p_bridge mod 2,
sign  = (-1)^a_abs.
```

Thus the Type-0B even-primary continuum again uses the even-minus-odd sector
sum.  `glasses_partition.py` is the single source of truth for this sewing
sign and for odd-null transport.  An odd handle null leaves the sector fixed
and flips the bridge lift; an odd bridge null toggles the sector and leaves
all lifts fixed.

`glasses_c_recursion_pbw.py` implements the independently testable
coefficient recursion.  It also fixes the glasses large-c seed: the two
self-loop trace-factorization signs cancel the graph polarization, so the
vacuum and global glasses functions multiply ordinarily.  The old extra
odd-sector flips of the vacuum handle lifts were a double counting.

Run the complete PBW audit through total physical level 4 with

```bash
PYTHONPATH=Code/c_Recursion:Code python3 \
  Code/genus_2/glasses_c_recursion_pbw.py
```

## Cross-sewing production check

`prepare_cross_sewing_config.py` derives a one-design production config from
the five-point convergence scan and certifies the branch-composed spin
transport before any numerical work is submitted.  The matched physical
lifts are `(+,-,+)` in the human-note theta edge order
`(zero,one,infinity)` and `(+,+,+)` in glasses.  Both represent `[00|00]`
after the affine genus-two characteristic transport.  The theta second beta
bit carries an affine shift; using `(+,+,+)` in both channels selects
different physical Majorana spin structures.

The denominator of

```text
Q_L = Z_L / Z_(X+psi)^9
```

is the physical partition function of one noncompact real scalar plus one
physical NS Majorana.  It is now evaluated entirely in the plumbing frame,

```text
Z_(X+psi)^pl = G_X^pl |P_X^pl|^2 |F_psi^pl|^2.
```

Here `F_psi^pl` is the physical-Majorana Fredholm determinant with the Human
Note descendant sign, and `G_X^pl` is the two-loop charge Gaussian derived
from charged free-boson pants sewing with `h(alpha)=alpha^2/2` and measure
`d alpha_1 d alpha_2`.  No period matrix or Riemann theta constant defines
this result.  This denominator is unrelated to the auxiliary Majorana block
`F_F` whose star inverse appears only inside the double-Virasoro computation
of an NS superconformal block.

```bash
PYTHONPATH=Code/c_Recursion:Code:Code/genus_2_cross_channel python3 \
  Code/genus_2/prepare_cross_sewing_config.py \
  Code/config/ns_genus2_cannon_fivepoint_r20_24_n8_12_axis.json \
  --output Code/config/ns_genus2_cross_sewing_r24_n10_human_note_spin00.json
```

For the focused local rerun used to check signs before launching fresh
order-24 shards:

```bash
PYTHONPATH=Code:Code/c_Recursion:Code/genus_2_cross_channel python3 \
  Code/genus_2/rerun_human_note_genus2.py
```

To reuse an already completed numerator and recompute only the independent
physical free denominator at several mode cutoffs:

```bash
PYTHONPATH=Code:Code/c_Recursion:Code/genus_2_cross_channel python3 \
  Code/genus_2/recompute_ql_plumbing_free.py
```

After reducing fresh theta and glasses shards, use
`summarize_cross_sewing.py` to compare the result with both the old unsigned
theta assembly and the intermediate theta-sign-only correction.  The audit
fails closed if either channel does not carry `[00|00]`.
