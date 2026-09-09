# Exploratory Type-0B NS sphere five-point amplitude

This directory contains the attempted genus-zero (1\to4) all-NS Type-0B
worldsheet computation: the PCO integrand, boundary-domain ledger, numerical
drivers, plotter, and tests.  It is deliberately separated from the reusable
`c_Recursion` library because the integrated five-point amplitude has not
been numerically certified or frozen.

The implementation imports the general multipoint NS recursions and the
BRY-normalized Liouville utilities from `Code/c_Recursion/`, and reuses the
plumbing atlas from the supplied bosonic c=1 reference implementation.  At
each moduli point the atlas compares all 120 oriented representatives of the
15 five-leaf trivalent trees and selects the linear chart that minimizes
`max(|q1|,|q2|)`. Production now uses fixed-weight **c-recursion in every
chart**, with no h-regulator fit. The shared kernel default is also `c`.
Amplitude CLI selectors accept only `c`, and the cluster loader rejects old
h/hybrid production bundles. Explicit h/hybrid Python APIs remain historical
research diagnostics, not amplitude routes. Here `q1=z1/z2` and `q2=z2` are the ordinary CCY
sphere linear-channel plumbing coordinates, not elliptic nomes.

For a face primary projection, the exact algebraic reduction to a four-point
block is enabled for the production c-series.  Regression tests compare it
with the unfactorized five-point coefficient.  The h-recursion audit retains
the full five-point block because its reduced four-point object does not share
the selected-edge recursion normalization at finite depth.

Run the attempt-specific tests from the repository root with:

```bash
PYTHONPATH='Code/higher_point_amplitude_attempts/type0b_ns_five_tachyon:Code/c_Recursion:Code' \
python3 -m unittest \
  Code/higher_point_amplitude_attempts/type0b_ns_five_tachyon/test_type0b_ns_five_tachyon.py \
  Code/higher_point_amplitude_attempts/type0b_ns_five_tachyon/test_type0b_ns_five_tachyon_domain.py
```

The generic multipoint-recursion tests remain in `Code/c_Recursion/`.

The active physical-domain driver is
`evaluate_type0b_ns_five_tachyon_physical_i_epsilon.py`.  It keeps
`omega_a=E_a+i*epsilon*nu_a` with positive real energies and tilt weights,
forms the incoming energy as their exact sum, leaves both internal Liouville
contours on the positive real axis, and applies the direct local finite-part
forest on all ten faces and fifteen compatible corners.  The default
benchmark is `E_1=...=E_4=1/4`, for which the complete divergence ledger
requires degree zero on every face.  The driver fails closed if a requested
kinematic point needs positive-degree diagonal counterterms.

The files whose names contain `one_divisor_path` or
`minimal_subtraction_path` are retained only as historical continuation-ray
audits.  Their outputs are excluded from the physical worldsheet freeze.

## Chart atlas plus polynomial subtraction

The physical driver adds the BRY polynomial-subtraction layer directly on top
of the c=1 chart atlas.  It does not excise the full recursive correlator from
a collar.  In the best local chart it numerically integrates

```text
F_remainder = F - chi_1 P_1 - chi_2 P_2 + chi_1 chi_2 P_12,
```

then restores the face and corner polynomials by analytic radial finite
parts.  The face integral applies the same construction recursively:
`A_D-A_DE` is retained numerically in a tangential corner collar and the
double finite part `A_DE` is added once.  Thus higher normal powers supplied
by c-recursion remain in the numerical remainder. Agreement between the
degree-zero polynomial and the untruncated c-recursive value at the collar
boundary is recorded only as a diagnostic and is not a production equality
condition.

The prepared matrix-blind cluster bundle uses c-recursion at edgewise
**twice-level** `(8,8)`, total twice-level 16, in the best chart. This means
physical descendant levels up to `(4,4)`; the historical job label "order8"
refers to twice-level, not physical level eight. The three collars
`rho=(0.01,0.005,0.0025)` share coefficients and Sobol points. The momentum
quadrature is `(6,7)` per unit panel: unlike `(5,7)`, its two smooth node
sets do not coincide. One shard records the `(8,8)` versus `(6,6)` c-series
collar diagnostic. This remains non-fatal and does not impose equality
between the degree-zero subtraction polynomial and the full CFT value:

```bash
python3 run_type0b_ns_five_tachyon_cluster.py \
  --config ../../config/type0b_ns_five_tachyon_c_recursion_order8_small_collar_cluster.json \
  plan
```

From the repository root, stage and validate it without submission with:

```bash
TYPE0B_5PT_STAGE_ONLY=1 \
Code/cluster/stage_submit_type0b_ns_fivepoint_order8.sh \
  SSH_HOST REMOTE_RUN_ROOT REMOTE_PYTHON
```

After inspecting the staging record, omit `TYPE0B_5PT_STAGE_ONLY` to submit:

```bash
Code/cluster/stage_submit_type0b_ns_fivepoint_order8.sh \
  SSH_HOST REMOTE_RUN_ROOT REMOTE_PYTHON
```

This creates four independent array tasks followed by a deterministic reducer.
Each shard evaluates all three collars with common random numbers. There are
sixteen RQMC replicates per shard; only one shard computes the common
deterministic corner terms.
The resulting summary remains matrix-model blind and is not labeled frozen.

## Bounded-memory c-series runtime

The old all-c array `42734882` exceeded 12 GB per shard. It retained unbounded
collections of blocks and recursive intermediate states. The revised implementation in
`fivepoint_runtime.py` separates compilation from evaluation:

1. Compile only the required final coefficients using the unchanged recurrence.
   Release the intermediate recursion dictionary after each table extension,
   including on exceptions. At the configured cutoff a fixed-parity five-point
   block has at most 25 final coefficients.
2. Evaluate that polynomial by Horner contraction, retaining the original
   mpmath precision and the supplied holomorphic/antiholomorphic logarithm lift.
   Polynomial masks and total-level cutoffs are unchanged.
3. Keep at most 2,048 blocks and 4,096 entries in each auxiliary cache. Evicted
   c tables are reloaded from a per-shard SQLite file, not recomputed. Only
   final coefficients are serialized, with enough decimal digits for an exact
   round trip at the configured precision. Source hashes and all block
   parameters enter the key. A process lock rejects concurrent writers to the
   same shard database. SQLite uses a 2 MiB page cache and rollback journaling,
   not a shared WAL on the cluster filesystem.
4. Atomically checkpoint each completed bulk sample, face sample, and corner
   orbit. Restarting with the same source/config/seed reuses those values;
   changed source or sampling settings fail closed. A completed face sample
   includes its corner subtraction, so partial forest terms are never reused
   as a completed sample. Collar diagnostics may be repeated on restart.

The worker emits progress, peak RSS, resident coefficient counts, evictions,
and disk-hit counts after each sample. Files are `task_NNNNN.coefficients.sqlite`
and `task_NNNNN.checkpoints/`, beside the final shard JSON. The 12-hour cap and
existing memory allocation have not been increased; a production-scale pilot
is still needed before certifying a smaller allocation or a completion time.

The tests in `test_fivepoint_runtime.py` cover parity/branch/mask preservation,
45-digit disk round trips, bounded eviction, concurrent-writer rejection,
and interrupted integration followed by exact sample reuse. Run the isolated
process benchmarks from the repository root with:

```bash
python3 -B Code/higher_point_amplitude_attempts/type0b_ns_five_tachyon/benchmark_fivepoint_runtime.py \
  --blocks 96 --cache-limit 16 --output /tmp/fivepoint-tables.json
python3 -B Code/higher_point_amplitude_attempts/type0b_ns_five_tachyon/benchmark_fivepoint_runtime.py \
  --integrand --cache-limit 16 --output /tmp/fivepoint-integrand.json
```

The table benchmark uses production momentum nodes and cutoff; the integrand
benchmark uses a reduced `(2,3)` momentum rule at one point. Neither is a full
integration or a convergence certificate. Timings include allocation tracing.
