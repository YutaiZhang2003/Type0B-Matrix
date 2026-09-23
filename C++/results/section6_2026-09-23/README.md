# Section 6 validation

All requested coefficient comparisons are covered with the agreed cutoff policy. Neither manuscript was edited.

| Channel | Reference | Cutoff | Precision | Maximum scaled error |
| --- | --- | --- | --- | --- |
| theta NS-R-R | physical PBW | per-edge 5 | 40 | 1.395e-20 |
| theta-ns | physical PBW | per-edge 5 | 40 | 1.079e-31 |
| glasses-ns | physical PBW | per-edge 5 | 40 | 8.669e-25 |
| glasses-ns-r | physical PBW | per-edge 5 | 40 | 2.407e-21 |
| glasses-r | physical PBW | per-edge 5 | 40; case 4 repeated at 60 | 2.629e-19 |
| tetrahedron, triangular R face | physical PBW | total 5 | 40 | 3.341e-25 |
| theta all NS | SCA c-recursion | per-edge 10 | 40 | 7.318e-19 |
| tetrahedron all NS | SCA c-recursion | total 10 | 40 | 2.053e-25 |

The tetrahedron all-NS level-10 result is the existing completed run, fully re-compared with output hashes checked. The new frontend reproduces its DV prefix through total level 3 and the new batched SCA reference reproduces its coefficients through total level 6. It was not rerun at level 10.

One glasses case initially gave `5.719e-18` at 40 digits, above the `1e-18` release threshold. Repeating both DV and PBW at 60 digits reduced the discrepancy to `3.009e-38`. The original failure remains recorded; see `precision_refinement.json`.

The fresh all-NS theta level-10 check contains all 9,261 multidegrees. Its maximum absolute discrepancy is `5.108e-7`; the maximum error scaled by `max(1,abs(DV),abs(SCA))` is `7.318e-19`. The large absolute discrepancy occurs for large coefficients; the report retains the worst scaled coefficient and both values.

The all-NS theta DV run took 966.742 s; the optimized SCA reference took 131.407 s, including 11.180 s for the Schottky seed. Validation jobs and compilation overlapped, so these are elapsed validation times rather than controlled performance benchmarks.

## Nonchiral comparison

The partition-function equality claim is not established by this release. The completed quadrature study, with source N7 and target N10 at fixed block cutoffs, satisfies its two-step 0.01% stability criterion for both integrals and complex spin matrices. The two sign ratios are `0.250052595` and `0.249813244`; the spin-sum ratio is `0.249928019`. Multiplying the full NS-R-R amplitude by four is explicitly an assumption. The subsequent source-level-8 study with this assumption gives spin-sum ratio `0.999709394` and a largest resolved-spin discrepancy of about `0.0933%`. The old roughly 1% result used a superseded nonchiral contraction. Correction: the initial release report read an intermediate summary and incorrectly described the completed quadrature as unfinished.

`partition_audit.json` recomputes these saved ratios without fitting or modifying them. The production frozen-grid assembler and normalization metadata are included in the source release; this audit does not claim a new momentum integration.

## Reproduction

See [the release guide](../../SECTION6.md). `validation_summary.json` contains the detailed scope, precision escalation, directed checks, and final source hashes. Per-run coefficient files and logs remain in this directory; the source archive includes compact reports, not numerical caches.
