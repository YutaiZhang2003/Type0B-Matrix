# Level-five NSRR toy: cluster preparation and local fallback

The user requested a three-hour toy cluster run raising the total chiral
block cutoff to five, then requested a local point if cluster queueing delays
the calculation. These are two separate, explicitly labelled runs of the
**unchanged factorized-sign trial**.

## Designs

| Run | Momentum order | Nodes | Workers | Chiral levels | Status |
|---|---:|---:|---:|---|---|
| Cluster package | N=4 | 64 | 8 | every half-level through 5 | prepared, not submitted |
| Local fallback | N=3 | 27 | 2 | every half-level through 5 | complete |

Both use `b=1.4`, the saved source plumbing, and all five original surfaces
`Omega_11=Omega_22=i`, `Omega_12=t+i/2`, `t=0.52,0.56,0.60,0.64,0.68`.
The local focus is `t=0.60`. Computing the momentum-dependent coefficients
dominates cost; evaluating those coefficients on the other four surfaces is
cheap, so their values are retained rather than discarding them.

The same momentum nodes, measures, and 30-digit-source BRY coefficients are
reused from the audited `L=3` dataset. The two vertex factors are still
multiplied rather than absolute-squared; the extra odd vertex phases and
separate `(-1)^f` sewing sign are unchanged. The diagonal nonchiral vertex
and conjugate-antiholomorphic hypotheses remain hypotheses. No physical
Ramond projector is inserted or inferred. No normalization is fitted.

Equal-sign blocks use the checked branching recursion and product of two
ordinary Virasoro c-recursions. The missing mixed-sign components still use
the **explicit PBW diagnostic completion**, now through the same level five.
No protected kernel, original trial runner, or old data is edited.

Every new node retains `L=3,4,5` at fixed momentum order, and must reproduce
the saved `L=3` blocks on all five surfaces and all four lifts. This is the
appropriate block-cutoff comparison. Comparing a local `N=3` value directly
with the previous `N=5` value would also include a quadrature change.

## Timing and cluster status

An unequal-momentum L5 benchmark at `(P_NS,P_R1,P_R2)=(.31,.43,.57)` took
157.62 seconds with 191 MB peak RSS. Setup and equal-sign evaluations took
less than a second; the four mixed-sign completions dominated runtime.
The first two actual `N=3` grid nodes took 165.89 and 166.09 seconds and
peaked at approximately 199 MB RSS.

The cluster package requests a **single eight-core, 16-GiB allocation with
a three-hour wall limit**. Eight waves of `N=4` nodes, a factor-three allowance
for slower cluster execution, and 15 minutes of overhead give a planning
estimate around 80–100 minutes. This is not a measured Cannon compute-node
runtime or a guarantee that queue time fits inside three hours.

The driver imposes a 15-minute timeout on each node and a 2h45m compute
deadline, leaving 15 minutes for reduction/plotting/cleanup. Each node runs
in a fresh single-threaded process. Failed or missing nodes prevent final
reduction; completed shards are retained and validated on resume. A file
lock prevents concurrent drivers writing the same output directory.

A read-only Cannon check found the `yin` partition up with mixed-use nodes
and pending jobs in the user's queue. The existing remote Python runtime
was checked: Python 3.12.11, NumPy 2.0.2, SciPy 1.13.1, mpmath 1.3.0.
There is no reliable immediate-start promise from this snapshot.
**No new cluster job has been submitted or remote directory staged.**

## Files and validation

- Driver: `Code/genus_2/nsrr_trial_cluster.py`.
- Configurations: `Code/config/nsrr_trial_L5_N3_local_20260830.json` and
  `Code/config/nsrr_trial_L5_N4_cluster_20260830.json`.
- Cluster instructions: `Code/cluster/README_nsrr_trial_L5_3h.md`.
- Immutable 35-MB bundle:
  `Data Set/nsrr_trial_L5_N4_cluster_bundle_20260830/`.
- Local results: `Data Set/nsrr_trial_L5_N3_local_20260830/`.
- Independent new-level check: `Code/genus_2/audit_nsrr_L5_blocks.py`.

All **56 regression tests pass**. The frozen bundle has 1,866 checksummed
files, contains its StringMC plumbing and vendored SymPy dependencies, and
has no runtime `/Users/...` reference-data paths. Its relocated free-factor
preflight and ten regression tests pass. Both source and target physical
free factors reproduce their saved values within `5.56e-16` relatively.

The independent L5 PBW audit has also completed: all four equal-sign channels
agree across all 91 exponent triples (and their parity components), with
maximum scaled error `5.29e-13`. Its result is saved as
`equal_sign_L5_PBW_audit.json` in the local output directory. This checks the
new level-four/five coefficients, not just reproduction of lower cutoffs.

## Local numerical result

All 27 local nodes completed in **39.78 minutes**, including 108 explicit
mixed-sign PBW completions. At the requested focus point `t=0.60`, with
the momentum grid fixed at `N=3`, the result is

| Total chiral cutoff | Q_NSrr_trial | Change from preceding cutoff |
|---:|---:|---:|
| L=3 | 1.8598784911714955e-7 | — |
| L=4 | 1.8600819385660130e-7 | +0.01093875% |
| L=5 | 1.8600530639101945e-7 | -0.00155233% |

The total `L=3 -> 5` change is **+0.00938624%**. Thus the observed block-cutoff
effect is far smaller than the percent-level diagnostic difference from
NSNSNS. This is a convergence observation, not a rigorous remainder bound.

At this point `Z_NSrr_trial(L5,N3)=7.648912888680161e-10`. The unchanged
NSNSNS diagnostic is `1.949426255891526e-7`, giving
`Q_NSrr_trial/Q_NSNSNS_diagnostic-1=-4.58458953%`. This percentage uses the
**coarser N=3** source grid and must not be read as a deterioration relative
to the previous `N=5` source result. The same-grid L3 comparison is the
appropriate measure of the new block-order effect.

The other surfaces were retained at negligible extra block cost:

| t | Q_NSrr_trial, L5/N3 | L3 to L4 | L4 to L5 | Difference from NSNSNS diagnostic |
|---:|---:|---:|---:|---:|
| 0.52 | 2.930228600326259e-7 | +0.02611844% | -0.00439360% | -3.80326% |
| 0.56 | 2.335064450456282e-7 | +0.01951603% | -0.00304925% | -4.10105% |
| 0.60 | 1.860053063910195e-7 | +0.01093875% | -0.00155233% | -4.58459% |
| 0.64 | 1.477554571096859e-7 | +0.00526730% | -0.00080813% | -5.21724% |
| 0.68 | 1.168745553987028e-7 | +0.00372201% | -0.00060035% | -5.95108% |

Across the five surfaces the final `L4 -> L5` change is only
0.00060–0.00439 percent in magnitude. The lower-cutoff curves nearly
coincide in the plot, as expected from these small changes.

![Local L5 cutoff comparison](</Users/yutaizhang/Desktop/Type0B-Matrix/Data Set/nsrr_trial_L5_N3_local_20260830/comparison.png>)

### Final audit

- Independent reconstruction of the saved contraction terms agrees to
  `8.14e-16`; independently reconstructed primary powers agree to `4.00e-15`.
- All archived L3 blocks are reproduced to `3.42e-13` (scaled), and the
  maximum branching Ward residual is `3.77e-12`.
- All four equal-sign channels at L5 independently match PBW, with maximum
  scaled coefficient error `5.29e-13` over 91 exponent triples per channel.
- The maximum lift spread is `1.62e-14`, so the previously noted physical
  spin/assembly limitation remains; numerical refinement does not resolve it.
- All eight protected kernel hashes remain unchanged. Peak node RSS is
  below 200 MB and maximum node wall time is 175.59 seconds.

The portable cluster package is still **prepared, not submitted**. It would
repeat the same `L=3,4,5` test at the finer `N=4` integration order under the
single-allocation three-hour limit. Its benchmark-based runtime estimate is
not a guarantee of queue or compute-node timing.

Outputs are `summary.json`, `comparison.csv`, `comparison.svg/png`,
`verification.json`, `equal_sign_L5_PBW_audit.json`, `status.json`, and 27
provenance-checked shards. The final independent integration audit can be
reproduced with `Code/genus_2/audit_nsrr_L5_run.py --run-dir` pointing at the
local output directory, using the same Python path as the run command.
