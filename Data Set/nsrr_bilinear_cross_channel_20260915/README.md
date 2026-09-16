# New NSRR pairing: cross-channel test fails

**Quadrature update:** the [central-surface refinement](../nsrr_bilinear_quadrature_20260915/FINAL_REPORT.md)
stabilizes at spin-sum ratio `0.249928019`, using source N7 and target N10
with L5/R12 fixed. Both integrals and the full complex spin matrices pass
two successive relative changes below 0.01%. The five-point table below
records the original low-order quadrature; those values were not the
quadrature limit. The other four surfaces have not been refined.

The new local NSRR matrix does not yield cross-channel agreement in this experiment. The complex all-NS lift vector was recomputed and both even RR components were transported separately. The ratios remain close to one quarter. No fitted multiplier was applied.

## Comparison definition

$\mathcal R=(Z_{\rm NSRR}/Z_{\rm NSNSNS})(Z_{{\rm free},t}/Z_{{\rm free},s})^\kappa$, with $b=1.4$ and $\kappa=1+2(b+b^{-1})^2=9.940408163265307$. Agreement requires $\mathcal R=1$.

The source uses the proposed `B_L B_R/8 * [[1,s],[s,1]]` on the `r=-1` channels. In the raw spin basis this is the normalized combination `(D_00-i*s*D_11)/sqrt(2)`. The all-NS target is assembled from all four complex Human-Note lift blocks, with its exact sector-dependent quadratic-parity basis transform and the matching spin combination. Each pant retains its supplied BRY constants. The bilinear anti data specialize to complex conjugates only on the real physical slice.

The marked source spins `[11|00]`, `[11|11]` transport to target `[00|00]`, `[00|10]`. In the target charge marking these are raw lifts `(-,+,+)` and `(+,+,+)`. The independently continued single-Majorana roots give relative transport phase +1 across all five surfaces, with maximum complex discrepancy `1.020e-10`. The two matched free-field frame ratios agree. The other two RR characteristics map to a target with a Ramond edge and are not compared to an all-NS target.

Primary factors `exp(sum h_i Log q_i)` are outside blocks and matrices. The source R weights include 1/16; the target has three NS weights. The source and target measures are independently checked as `d^3p/pi^3`.

## Five complete, matched-quadrature comparisons

Source: total descendant cutoff L=5; target: recursion twice-level R=12 with resummed global blocks. Both channels use N=3 nodes per momentum direction.

| t | R(s=+1) | R(s=-1) | R(spin sum) |
|---:|---:|---:|---:|
| 0.52 | 0.251461747 | 0.251166853 | 0.251308055 |
| 0.56 | 0.252437897 | 0.252246849 | 0.252338372 |
| 0.60 | 0.252925745 | 0.252836380 | 0.252879225 |
| 0.64 | 0.252916112 | 0.252965657 | 0.252941875 |
| 0.68 | 0.252456067 | 0.252714296 | 0.252590118 |

The spin sum is the trace of the two-by-two integrated spin pairing: `Z_+ + Z_- = Z_00 + Z_11`. It contains no cross-spin interference. Its mismatch therefore cannot be repaired by changing only the relative Majorana phase.

## Cutoff controls at t=0.60

| Comparison | Source L3→L5 at N3 | Target R8→R12 at N3 | Joint N3→N4 at L3 |
|---|---:|---:|---:|
| s=+1 | 2.650e-04 | -3.915e-07 | -2.447e-03 |
| s=-1 | 1.287e-04 | -1.910e-08 | -2.776e-03 |
| spin_sum | 1.941e-04 | -1.976e-07 | -2.618e-03 |

The N4, L3/R12 spin-sum ratio is `0.252168291`. These variations are much smaller than the roughly 75% discrepancy. They are convergence diagnostics, not rigorous remainder bounds.

## Independent numerical checks

- 91 complete fresh target nodes: 27 nodes at five surfaces, plus 64 nodes at the central surface. Every point retains both three-form sectors, four complex lifts, and both target recursion cutoffs. No partial-grid ratios are reported.
- 334 source/target node measures reconstructed independently; maximum relative measure discrepancy `6.661e-16`.
- 120 source-block spot comparisons against the current native double-Virasoro engine through L5. The extreme tail reaches a scaled block difference of `8.117e-08`; the largest single checked node correction divided by the full integral is only `3.727e-12`. This is a three-node check, not a complete native recomputation.
- Source primary-prefactor comparison is exact at saved precision; target primaries, weights and logarithms are recorded separately.
- All target global sums converge. The spin-pairing matrices are Hermitian and positive within numerical precision.

## What this does and does not establish

**The proposal fails this cross-channel test.** Its local BPZ and descendant checks remain local checks; they are insufficient to certify the full genus-two decomposition. The global pairing/defect prescription and relative channel normalization need to be revisited. The result does not isolate an error in the chiral double-Virasoro procedure, which was not modified.

The off-diagonal interacting transport uses the minimal extension of the checked free spin dictionary. That extension is a tested proposal, not an independent proof of the interacting fusion kernel. The separately reported spin-sum discrepancy survives elimination of the relative-phase interference.

BRY equations (3.6) and (3.8) retain the common `pi delta` two-point normalization and `C^±=(E±O)/2`; no change to these inputs is inferred from the numerical gap. [BRY §3.1](https://arxiv.org/html/2201.05621#S3.SS1).

## Artifacts and reproduction

- `result.json`, `summary.csv`: conclusions, five-point table and cutoff controls.
- `comparison.json`, `comparison.csv`: full source/target combinations at the tested cutoffs.
- `config.json`: geometry, spin transport, primary convention and implementation hashes.
- `target/`: complete fresh complex all-NS block vectors, primaries and constants.
- `source.json`: recombined saved complex NSRR data.
- `fresh_source_checks.json`: native source spot checks and weighted impact.
- `provenance.json`, `target_provenance.json`, `report_provenance.json`: input SHA-256 records.

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  python Code/genus_2/test_nsrr_bilinear_cross_channel.py --workers 3
python Code/genus_2/check_cross_channel_source_blocks.py
python Code/genus_2/report_nsrr_bilinear_cross_channel.py
```
