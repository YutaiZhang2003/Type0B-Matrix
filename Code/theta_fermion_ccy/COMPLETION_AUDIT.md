# Completion audit: selected Ramond fermion-insertion algorithm

Audit date: September 10, 2026. This document audits the current saved
sources and results. It does not run another conformal-block calculation
or numerical comparison. The complete physical level-ten timing remains
pending at this snapshot. The renewed presentation review is closed.

## Authoritative scope

The user amended the auxiliary requirement after requesting its direct
definition and level-ten benchmark: the production denominator now uses
direct fermion sewing. The enlarged numerator must still use two ordinary
Virasoro blocks computed by genuine CCY central-charge recursion. This
selected method does not require an all-level irreducible Ising
central-charge recursion. The exact Ising decomposition remains useful
for identifying representations and parity coefficients.

The required numerical comparisons are exactly (1) the recovered physical
superconformal block against an independent physical PBW calculation
through total level five, and (2) the recovered block at different split
parameters with fixed product. The level-ten calculation is a production
calculation and timing, not an additional comparison suite.

## Requirement-by-requirement status

| Requirement | Status | Authoritative evidence and qualification |
|---|---|---|
| Derive the subdivided tube geometry and spin transport | Complete | `Machine Notes/theta_fermion_ramond_recovery.tex` derives `q2=qL*qR` and explicitly adopts the transported Ramond spin frame. |
| Use the human normalization of the external state | Complete | The notes impose the irreducible physical vacuum relation and use `v_(1/2)(Q/2)=Q*psi`; `Q != 0` is stated with the embedding hypotheses. |
| Derive the full enlarged block as branching coefficients times ordinary Virasoro blocks | Complete | The notes give the raw four-cut contraction, its four internal norms, the normalized middle relation, and the fusion condition `nprime=n+/-1/2`. The omitted native-to-human contour phase is stated explicitly. |
| Compute the middle primary coefficients recursively | Complete | `middle_branching.py` implements the two orientations of the physical `L1` Ward recurrence, starting from the ground anchors and using reusable `RamondActions`. Higher primary matrix elements are not evaluated by a full physical PBW block. |
| Reuse action coefficients and handle negative Ramond labels | Complete | The cached positive-chart action data are transported by simultaneous momentum/label reflection. `outer_branching.py` uses this reflected provider. The notes distinguish the middle level-six action bound from the outer level-eight preparatory action at physical level ten. |
| Compute the full descendants by CCY recursion | Complete | `punctured_ccy.py` uses null-vector residues, fixed external weight, the global `SL(2)` seed, and the exact universal large-central-charge factor. It does not replace the ordinary Virasoro blocks by finite-central-charge Gram contractions. |
| Evaluate the full auxiliary fermion insertion from its definition | Complete through level ten | `direct_fermion.py` retains all nonzero modes and both split-edge orientations, using exact rational mode Ward forms and sparse Fock sewing. Complete reusable coefficients are in `results/direct_fermion_level10.json`. |
| Make the auxiliary ingredients generalizable to arbitrary genus/plumbing graphs | Complete at the stated local-construction scope | The direct-definition subsection explains local Fock bases, pairings, three-point forms, insertions, and graph-dependent spin signs. Only the specified theta assembler is implemented. A complete sector-recovery construction on another graph additionally needs that graph's insertion/projector choices and sector identity. |
| Derive SCA–fermion convolution and recover the physical block | Complete | The notes derive the physical identity contraction, parity cocycle, all-level sector identity, and triangular minus-sector division. `series_algebra.py` retains all four-variable coefficients and raises on a failed sector condition rather than imposing a projected answer. |
| Restore the correct universal factor | Complete | `pipeline.py` divides the reduced numerator by the full direct auxiliary and then restores the universal factor squared. The saved production metadata records `universal_seed_power_after_recovery=2`. |
| First authorized comparison: final physical PBW through level five | Passed | The reflected production and validation artifacts below record all 4,648 parity-coefficient slots, including off-diagonal split exponents. Maximum scaled difference: `4.713175155062888e-52`. This sample has `p=f=0`, `(eta,eta')=(+,-)`. |
| Second authorized comparison: vary the split at fixed product | Passed | The same validation artifact records `u=0.5,1,2` and all eight spin-sign evaluations at `q1=0.013`, `q2=0.007`, `q3=0.009`. Maximum scaled variation: `2.951691834664898e-60`. |
| Compute and report a complete physical level-ten block and its measured runtime | **Pending** | Parent reports the production run is active in session `22826`. No completed level-ten physical result or runtime is claimed in this audit. The auxiliary-only benchmark and level-five runtime do not discharge this requirement. |
| Supply separate detailed LaTeX notes | Complete, pending the level-ten timing entry | The main file and its direct-fermion and exact-Ising inputs are in `Machine Notes`. This deliverable does not require modifying the existing draft or Human Notes. The revised 16-page notes compile without warnings. |
| Renew adversarial mathematical review of the selected algorithm | Complete | `FINAL_MATH_REVIEW.md`, `FINAL_BRANCHING_REVIEW.md`, and `REVIEW_DIRECT_FERMION_SIGNS.md` review the selected direct-denominator construction. The CCY and branching reviewers discussed normalization, seed power, reflection, and the arbitrary-genus scope boundary. No remaining generic-parameter theta inconsistency was found. |
| Renew adversarial presentation review and revise accordingly | Complete | `FINAL_PRESENTATION_REVIEW.md` and `FINAL_PRESENTATION_DIRECT.md` record the debate, agreed revisions and their application. The middle reviewer independently verified the final edits and closed the review. |

## Saved numerical evidence

The authoritative production file is
`results/ccy_direct_reflected_level5.json`, and the corresponding two-check
report is `results/validation_ccy_direct_reflected_level5.json`. The latter
has status `passed`. The production file retains its literal status
`computed; validation pending` because validation is written to a separate
artifact; this is not a failed or unfinished production calculation.

The benchmark parameters are `b=7/5`,
`P=(11/23,13/29,17/31)`, `p=f=0`, `(eta,eta')=(1,-1)`. The production
settings are 100 decimal digits for CCY and 70 for the outer Ward solve,
with exact rational auxiliary coefficients before restoring the common
`Q/sqrt(2)` factor. The independent physical reference uses its separate
384-bit FLINT implementation. These error figures are measured numerical
differences, not rigorous interval bounds or a claim of coverage of other
external parity/form choices.

The level-five computation records `82.19680833300117` seconds: outer
coefficients `52.123118624998824`, middle recurrence `2.6851382569948328`,
CCY blocks `26.199639883996497`, auxiliary `0.08250241699897742`, and
division with seed restoration `0.6678412080000271` seconds. The balance
is assembly and intermediate serialization. The script's `total_seconds`
starts after module imports and argument parsing and stops before final
physical-series encoding and the final JSON save. It measures the complete
mathematical production stages; a full-process timing should be labeled
separately if supplied by an external timer.

The standalone auxiliary level-ten benchmark is
`results/direct_fermion_level10.json`, with status
`computed_without_crosschecks`. Its coefficient construction took
`0.186281209000299` seconds; process time through the initial save was
`0.2902146670003276` seconds. Peak resident memory was `46.765625 MiB`.
It stores 4,793 nonzero exponent tuples and 9,586 rational parity entries.
This establishes the auxiliary cost and reusable data only.

## Source provenance

This audit recomputed SHA-256 digests of the files named in the saved
manifests, without importing the production modules or running any block
calculations. All 14 entries in the reflected level-five production
manifest match the current files. The copied manifest in its validation
report also matches all 14 current files. Thus the passing checks refer
to the current reflected outer implementation, not an earlier result.

Both entries in the standalone auxiliary benchmark manifest match the
current files when its basename keys are resolved relative to
`Code/theta_fermion_ccy`. In particular, the benchmark and integrated
production record the identical `direct_fermion.py` digest. These are
checks of the supplied manifests, not a claim that the manifest records
every transitive dependency in the Python environment.

Selected current/saved digests:

| Source | SHA-256 |
|---|---|
| `pipeline.py` | `8ec42e45ae5f1053256369f575e105fde79f2f3248633873803575698788f59c` |
| `outer_branching.py` | `f9bcbc679d9799c197d10992850e3c3db9d9d9ac2cfb615d46601a1573d004c9` |
| `middle_branching.py` | `485a69a36fba4789480ecc3b830ada7767393b50782547c85775dace612a948f` |
| `punctured_ccy.py` | `1a8b0dfc465e700be9e64be46df450822fb8e24154db594574c7c7b8f4e5d32a` |
| `direct_fermion.py` | `5bf3eab304bcb0a6e66281217a326d1b4df1e975dd8debdb17431f16550a04bd` |
| `series_algebra.py` | `3e7c5502b372e1668592dda97c7b060bb372fedcc59b4c3fecbc3e41b46c23e1` |
| `validate_requested.py` | `ece44aff49d7faf1fe4bfa5496c09ae39d028eb049bda886c9a5cfd126d0e32e` |

## Remaining actions before declaring completion

1. Receive the completed physical level-ten artifact and actual timing
   from the active production run. Record parameters, precision, all
   production stages, the saved coefficient path, and timing scope.
2. Add the measured level-ten timing to the notes, then update this audit
   and the README checklist with the completed result and source provenance.

No completion of the overall goal is asserted until the pending items
have actual supporting evidence.
