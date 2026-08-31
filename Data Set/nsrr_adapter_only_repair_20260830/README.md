# Adapter-only NSRR repair — 2026-08-30

Status: the supported chiral wrapper and its geometry boundary are repaired.
The full nonchiral NSRR–NSNSNS Q comparison is **not yet repaired** and remains
disabled. There is no new partition value or modular-agreement claim here.

## Protected implementation

No edits were made in this repair to the checked branching recursion,
ordinary Virasoro c-recursion, PBW oracle, or all-NS recursion. The eight
critical files are recorded in
`Code/genus_2/nsrr_checked_kernel_manifest.json`; the boundary regression
checks their SHA-256 hashes. The snapshot preserves the files as they were
at the start of this request, including existing uncommitted changes. It
does not silently restore or replace them from Git.

Only the newly written NSRR wrapper, geometry/comparison adapters, their
tests, and new diagnostic artifacts were changed. Historical results were
not overwritten.

## Implemented corrections

1. **Canonical branching API only.** Both HJS signs now use the package's
   original `solve(alpha2, alpha3)` interface, with no `eta` or `form_parity`
   override. The negative HJS sign is obtained by the reflected Ramond
   momentum and branch label:

   `raw_minus(P2; n2) = raw_plus(-P2; -n2)`.

   This implements the Human Note's Ramond reflection basis. The chiral
   momentum-sign/HJS-sign relation is also discussed in
   [Suchanek, equation (89)](https://arxiv.org/html/0810.1203).
   The odd form is transported by the already tested Ramond ground-partner
   Ward identity. The earlier noncanonical odd-grid extension is not used.

2. **Explicit geometry boundary.** `nsrr_plumbing_adapter.py` requires a
   matched geometric `(R,R,NS)` chart in `(zero,one,infinity)` order. It maps
   q, literal lift signs, momenta, and primary powers together into the
   package's `(infinity,one,zero)=(NS,R,R)` slots. An old NS-at-zero chart
   is rejected instead of silently permuted. The five re-plumbed charts
   still pass the independent forward period check below `5.53e-13`.

3. **No hidden PBW production fallback.** Equal-sign physical components
   are recovered using the supported star quotient and the Ward support
   relation, then the ordinary lift sum is applied. Opposite-sign components
   are not determined by that auxiliary-star identity. Production rejects
   them. PBW completion must explicitly request `completion="pbw_diagnostic"`.
   Tests replace the PBW constructor with a failure to ensure the supported
   production route never invokes it.

4. **The old factor-four contraction is retired.** It cannot be evaluated
   against the corrected blocks and called a historical reproduction. The
   previous numerical data remain available; the historical constant is
   retained only for interpreting old metadata.

## Numerical checks

The final regression run passed **71 tests**. `verification_summary.json`
records the protected-kernel hash checks, matching certificate fingerprint,
coefficient counts, and numerical errors.

`checked_core_p0_L3.json` and `checked_core_p1_L3.json` rerun the unchanged
original driver at its generic reference point. Each checks 240 parity
entries against PBW. Maximum errors are `3.704e-12` and `9.599e-13`.

`block_certificate_L3.json` checks the new wrapper at b=1.4 and physical
momenta `(0.21,0.37,0.52)` through total level 3:

- 1,920 equal-HJS-sign parity entries are independently checked against PBW.
- Another 1,920 entries test explicitly PBW-backed diagnostic completion;
  they are **not** an independent double-Virasoro determination.
- Maximum component error: `6.66e-14`.
- Maximum forward auxiliary-star identity error: `2.61e-13`.
- All eight literal lift choices are evaluated on all five charts, with
  maximum block-value difference `5.79e-15`.

The certificate records an implementation fingerprint and refuses a run
during which those implementation files change.

## Additional error in my free-factor adapter

The old source code assumed that multiplying an all-NS plumbing Majorana
factor by `|theta_R/theta_NS|` produced its matching Ramond factor. That
assumption requires `Z_psi(q,delta)/|theta[delta](Omega)|` to be independent
of the chosen all-NS reference characteristic in these conventions.

`free_spin_conversion_audit.json` tests this necessary condition using the
existing all-NS evaluator and its existing characteristic map. It fails
at every source point. At t=0.60 the maximum relative incompatibility
against the first reference is `0.459725`. This is a discrepancy between
candidate frame factors, **not** an error estimate for Q or evidence
against the double-Virasoro/PBW calculation.

Consequently the new geometry output records the theta-ratio result only
as `candidate_theta_ratio_free_superfield`, with an explicit failed
compatibility audit. It no longer exports it as a certified physical
Ramond denominator. The existing free evaluator and spin map were not edited.

## What remains

Restoring physical Q production requires deriving and checking the closed
Ramond nonchiral contraction and a compatible physical spin/free-factor
conversion in the same plumbing frame. These are unresolved adapter tasks,
not reasons to modify the protected package. A scalar factor, a star
character, or an all-NS reference lift must not stand in for those data.

## Reproduction

From the repository root, with `OPENBLAS_NUM_THREADS=1`,
`OMP_NUM_THREADS=1`, and `PYTHONDONTWRITEBYTECODE=1`:

```sh
python3 Code/genus_2/nsrr_human_note_geometry.py \
  --baseline 'Data Set/nsrr_nsnsns_fivepoint_L4_N5_20260830/config.json' \
  --output 'Data Set/nsrr_adapter_only_repair_20260830/geometry.json'
python3 Code/genus_2/certify_nsrr_human_note_blocks.py \
  --cutoff 3 \
  --geometry 'Data Set/nsrr_adapter_only_repair_20260830/geometry.json' \
  --output 'Data Set/nsrr_adapter_only_repair_20260830/block_certificate_L3.json'
python3 Code/genus_2/audit_nsrr_free_spin_conversion.py \
  --config 'Data Set/nsrr_nsnsns_fivepoint_L4_N5_20260830/config.json' \
  --output 'Data Set/nsrr_adapter_only_repair_20260830/free_spin_conversion_audit.json'
```
