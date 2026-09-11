# Complete CCY vertex factors and inverse-denominator reuse

The inserted global coefficient is the product of three vertex factors.
Each factor depends only on its incident edges, including the local shifts
and descendant levels. The old cache stored only the core finite sum and
recomputed the remaining rising product and normalization on every call.
The current cache retains each complete normalized factor and returns it by
reference. All factors are evaluated with the same formulas at the same
precision. This trades additional memory within an individual Virasoro engine
for reuse across spectator-edge choices; its caches are released between engines.

The MPC assembly uses one reusable temporary for the three factor products
and the incoming amplitude, avoiding repeated mantissa allocation in the inner
sum. The accumulation order and precision are unchanged.

Forward recursion additionally stores `1/(c_in-c_pole)` by the ordered pair
of central-charge identifiers. The existing small-denominator guard is checked
when the inverse is first constructed. Its common residue is factored out of
the incoming sum: `sum(w * R / (c_in-c_pole)) = R * sum(w / (c_in-c_pole))`.
This changes floating-point rounding slightly but preserves the exact recurrence.
These caches are local to one Virasoro engine and do not mix different weights.

## Directed 40-digit measurements

These are individual inserted Virasoro factors with independent descendant
cutoffs, b=7/5, (P1,P2,P3)=(11/23,13/29,17/31), and branch labels
(n1,n,n',n3)=(0,-1/4,1/4,1/4). Both copies use their actual branching weights
and the external v_(1/2) weight. Timers cover the reduced CCY calculation;
compilation and result serialization are excluded. The older full-pipeline
timing process remained running during these short tests.

| Cutoff per Virasoro edge | Copy | Before (s) | Current (s) | Ratio |
| --- | --- | ---: | ---: | ---: |
| 6 | 1 | 1.94524 | 0.232037 | 8.38 |
| 6 | 2 | 1.94798 | 0.219560 | 8.87 |
| 8 | 1 | 16.0589 | 1.54465 | 10.40 |

The cutoff-6 factors contain 2,401 coefficients and 234,256 global-seed
summands each. The cutoff-8 factor contains 6,561 coefficients and 1,874,161
global-seed summands. The original and current versions perform the same
number of recursion transitions and retain the same coefficients.
Maximum scaled differences are 4.22e-33 and 1.10e-34 for the two cutoff-6
copies, and 8.88e-32 for the cutoff-8 copy.

An intermediate version using complete vertex caches and MPC scratch reuse,
before inverse-denominator reuse, took 0.481988 s and 0.445333 s at cutoff 6.
Its coefficients were unchanged. These intermediate results isolate the
two sources of improvement; they are not additional production algorithms.

Both full physical pipelines at independent cutoff 2 were also compared with
their saved pre-change results. All 360 physical components in each case agree
within 1.97e-38 (ordinary) and 2.97e-35 (inserted) in scaled difference.
No physical PBW block was computed.

## Completed fresh full runs

Both optimized sectors completed with independent level-10 cutoffs on q1, q2,
and q3, using 40 digits (136 bits). Each sector started in a separate sequential
process with empty numerical caches, storing and reusing branching values within
that process. Each output contains all 2,541 monomials and 20,328 physical
components. Compilation and directed validation are outside these timings.

| Stage (seconds) | Ordinary, eta eta' = +1 | Inserted, eta eta' = -1 |
| --- | ---: | ---: |
| Branching, including initial actions | 9.938 | 10.044 |
| Middle recurrence | 0 | 0.013 |
| Reduced CCY | 296.593 | 1460.243 |
| Virasoro products | 26.759 | 88.559 |
| Branch assembly | 0.356 | 0.577 |
| Direct fermion | 5.220 | 5.221 |
| Sector division | 3.115 | 3.130 |
| Schottky vacuum and square | 72.379 | 72.492 |
| Vacuum restoration | 0.375 | 0.380 |
| Other pipeline overhead | 1.801 | 14.620 |
| Startup, serialization and exit | 0.081 | 0.081 |
| Full wall time | **416.619 (6 min 57 s)** | **1655.360 (27 min 35 s)** |

Other pipeline overhead is the difference between the internal total and the
individually timed stages; it includes work and cache cleanup outside their timers.

The ordinary wall time decreased from the completed baseline's 508.004 seconds
to 416.619 seconds. Its CCY time decreased from 390.230 to 296.593 seconds.
The original inserted run was interrupted and had projected to about 4.5 hours;
that is an estimated baseline, not a measured completion time. The optimized
inserted run performs 1,740,820,224 global-seed summands and 839,918,556 CCY
transitions, evaluates 1,620 Virasoro blocks, and reuses 781 transposed products.

Reading the saved ordinary results gives maximum scaled difference 2.107e-20
and maximum absolute difference 8.907e-9 across all 20,328 components; coefficients
reach 4.166e12. The scale is max(1, abs(before), abs(after)). This comparison
checks the implementation change at the same working precision; it does not
certify coefficient accuracy independently. Both sectors have finite outputs.
Maximum recorded recovery-sector residuals are 2.226e-15 (ordinary) and
8.837e-16 (inserted). No new physical PBW block was computed.

The valid run records, all output coefficients, and source/executable hashes are
in `../per_edge_level10_2026-09-11/optimized_full_pipeline_40dps/`. The parent
directory contains the complete timing report, saved-output inspection, and
the previously computed matched-precision PBW estimates.

All benchmark coefficient files and the physical validation report are in this
directory. The driver is `C++/tests/ccy_vertex_benchmark.cpp`; the pre-change
headers and executable are retained under `/private/tmp/ramond_ccy_vertex_baseline/`
and `/private/tmp/ramond_per_edge_before_ccy_reuse` for this session.
