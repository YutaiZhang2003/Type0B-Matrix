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
`max(|q1|,|q2|)`.  Following the attached five-point review, the production
backend is h-recursion at `b=exp(eta)`, followed by a polynomial extrapolation
in `eta^2` to the self-dual point.  Fixed-weight c-recursion remains the
descendant-validated low-order collar check; legacy `"hybrid"` mode is an
overlap audit.  Here `q1=z1/z2` and `q2=z2` are the ordinary CCY sphere
linear-channel plumbing coordinates.

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
by h/c recursion remain in the numerical remainder.  Agreement between the
degree-zero polynomial and the untruncated c-recursive value at the collar
boundary is recorded only as a diagnostic and is not a production equality
condition.

The prepared matrix-blind cluster bundle uses regulated h-recursion at
edgewise twice-level `(8,8)`, total twice-level 16, in the best chart,
retaining the corner coefficient at `(8,8)`.  It evaluates the five regulator
values `eta=(0.16,0.13,0.10,0.075,0.055)` at fixed physical weights and fits
every recursion coefficient in `eta^2` before moduli integration.  Degree
three is the production table and degree two is integrated on the same Sobol
points as its regulator systematic.  The three collars
`rho=(0.01,0.005,0.0025)` share the same fitted CFT tables and Sobol points.
One shard also records the configured tolerance-based face-CFT versus
c-recursion diagnostic at total level eight against the preceding total-level-six
series at every collar.  It is deliberately non-fatal so that reference-series
non-convergence is reported alongside, rather than substituted for, the first
moduli estimate;
exact boundary equality is not required because the full block contains the
higher normal powers left in the numerical remainder:

```bash
python3 run_type0b_ns_five_tachyon_cluster.py \
  --config ../../config/type0b_ns_five_tachyon_order8_small_collar_cluster.json \
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
Each shard evaluates all three collars and both coefficient fits with common
random numbers.  A hash-addressed, lock-protected shared cache ensures that an
order-eight fitted coefficient table is constructed only once across the
array.  There are sixteen RQMC replicates per shard; only one shard computes
the common deterministic corner terms.
The resulting summary remains matrix-model blind and is not labeled frozen.
