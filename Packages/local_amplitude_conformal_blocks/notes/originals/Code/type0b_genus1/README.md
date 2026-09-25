# Direct Type 0B genus-one T → T calculation

**Status: implemented and numerically exercised, but no certified physical
one-loop amplitude yet.** Do not cite a `trial_sum` as a BRY-normalized
S-matrix coefficient. The outstanding issues are the completed odd-spin
boundary prescription, the matched Ramond infrared/collision limit, and
the absolute punctured-torus normalization.

The [21 September diagnostic](ope_channel/HET_MATCHING_LESSONS.md) used
compact necklace data that were not internally converged. It is not an
accepted crossing or Weyl-factor comparison, and the smaller L6 discrepancy
does not select a preferred answer. [Internal CFT convergence work](convergence/README.md)
identified and repaired NS coefficient precision loss in the saved L8 bank.
The corrected bank's sampled L7-to-L8 changes satisfy the user's accepted
0.5% production tolerance. L8 is retained, with no larger-L tests requested.
Production using this bank is saved separately in
`Data Set/type0b_genus1_production_L8_precise_20260921/`. The old L8 values
remain superseded; the tolerance is not a full-integral error bound.

- [DERIVATION.md](DERIVATION.md): external basis, explicit diagonal spin
  sum, PCO/ghost factors, nonvanishing allowed odd-spin words, continuation,
  collision primitives and the new cusp calculation.
- [RESULTS.md](RESULTS.md): numbers actually obtained and their limitations.
- [REPRODUCE.md](REPRODUCE.md): commands for preparation, evaluation and
  verification.
- [matched_production/README.md](matched_production/README.md): the L8,
  height-five production with own-cusp subtraction throughout the complete
  strip and independent elliptic-order-eight restoration. Its outputs are
  in `Data Set/type0b_genus1_matched_L8_q8_Y5_20260920/`.
- [ope_channel/README.md](ope_channel/README.md): the completed disk
  sensitivity calculation with collision-bridge descendants through
  relative orders two and four, holding that matched production fixed.
  Its outputs are in `Data Set/type0b_genus1_ope_descendants_20260921/`.
- [ope_channel/HET_MATCHING_LESSONS.md](ope_channel/HET_MATCHING_LESSONS.md):
  the actual Het frame/component and annulus comparison design, and its
  new application to the saved Type 0B even-spin banks.

The original outputs are in `Data Set/type0b_genus1_20260920/`. All code in this
directory is new to this calculation. Existing Type0B work was preserved.
The handoff is isolated at `handoffs/reference/HetSO23_1to1_20260920/`.
Its ZIP hash and package manifests were verified; no Het amplitude was
rerun and its closed numerical review was not reopened.

## Code

| File | Responsibility |
| --- | --- |
| `prepare.py`, `ns_recursion.py` | New coefficient banks using this checkout's NS c-recursion and BRY structure constants |
| `ramond_descendants.py` | Pure Ramond PBW sewing, including G−3/2 and full-state parity insertion |
| `even.py` | Three even-spin components with both left and right PCO sums |
| `odd.py` | New odd-spin banks and explicitly provisional AA/AB/BA/BB/contact assembly |
| `collision.py` | Type0B squared Ward coefficients, continued disks, new odd-supertrace one-point data |
| `integrate.py` | Modular cap/strip integration and finite-height collision matching |
| `cusp.py` | Independent long-handle approximation and analytic height integration |
| `validate.py` | Saved modular, collision, cusp-overlap, quadrature and safety-gate evidence |
| `report.py` | Rebuild numerical summary from saved component records |
| `provenance.py` | Final-delivery source closure and artifact hash manifest |

The frozen reference supplies validated pure CFT blocks and a conditioned
Ramond representation-theory backend. It does not supply the Type0B free
field integrand or spin weights. The old recursion checkout is neither
installed nor substituted for the current one. The broken historical Het
tail preparation API is not called.

No matrix-model amplitude or empirical resummation is imported into any
integrator. The code has no public function returning a physical amplitude
while its required contributions or normalization are absent.
