# c-recursion

Central-charge recursion code for NS and Ramond blocks lives here together
with its global/regular seeds, direct finite-level oracles, sphere and
genus-two consumers, plotting tools, and regression tests.

**NS sphere amplitude policy (2026-08-30): all c-recursion.** The four-/five-point
amplitude entry points use component-aware c-recursion in every selected
chart, including every PCO component, face, and corner. There is no automatic
`|q|<0.3` h-recursion gate. Polynomial subtraction and its analytic restoration
remain above the same block layer and are unchanged. Shared correlator/kernel
defaults are also `c`; explicit h/hybrid Python APIs and their tests are retained
only for historical research comparisons. Amplitude CLI backend selectors
accept only `c`, and the cluster loader rejects retired h/hybrid bundles.
The policy and verification scope are recorded in
`Machine Notes/c-Recursion/NS_SPHERE_ALL_C_PRODUCTION_POLICY_2026-08-30.md`.

`ns_multipoint_h_recursion.py` is the fixed-difference internal-weight
recursion for all-NS sphere linear channels.  It supports bottom components,
external `G_-1/2` markings, coefficient tables, and direct functional
evaluation with per-edge and total level cutoffs.  Resonant self-dual
`b=1` evaluations are defined by an even-in-`log(b)` confluent limit.  The
four- and five-point regressions compare it with the independent c-recursion
at generic weights and at the self-dual point.

`ns_multipoint_c_recursion.py` contains the coefficient-level all-NS
generalization to the sphere linear channel and torus necklace channel. It
uses the shared fixed-weight Kac/fusion kernels and global `osp(1|2)`
vertices. For example,

```python
from c_Recursion.ns_multipoint_c_recursion import NSSphereLinearCRecursion

block = NSSphereLinearCRecursion(
    central_charge=14.2,
    external_weights=(0.31, 0.42, 0.53, 0.47, 0.28),
    internal_weights=(0.73, 0.81),
    vertex_sectors=(1, 1, 0),
)
coefficient = block.coefficient((3, 0))  # q1^(3/2) q2^0
```

The sphere driver accepts bottom components and external `G_-1/2` markings;
the xor of the trinion sectors is matched to the xor of those markings.  It
assumes generic non-confluent `c`-poles. The torus driver currently retains
bottom external primaries and supports two or more necklace vertices; the
one-point self-loop uses the specialized torus code.

`sphere_multipoint.py` contracts the sphere block with the self-dual
super-Liouville `C` and `tilde C` structure constants, sums all allowed
vertex sectors, integrates one `dP/pi` continuum per internal edge, and
restores the Mobius-frame covariance factor. Its default is c-recursion in
every chart. Chart selection still improves the coordinate expansion, but
does not switch the production backend. Explicit h/hybrid API selections
remain research diagnostics only.
The reproducible five-point
reassociation test is

```bash
PYTHONPATH=Code/c_Recursion:Code python3 \
  Code/c_Recursion/stress_ns_multipoint_crossing.py
```

It compares the orders `(0,1,2,3,4)` and `(2,1,0,3,4)` at the same five
physical punctures. This is the NS Liouville matter correlator needed for a
Type-0B amplitude, not the complete string integrand: timelike matter,
ghosts, picture changing, and the supermoduli measure are outside this
module. At the default momentum rule the observed crossing residual falls
from `4.11e-3` to `1.82e-3` to `7.94e-4` across the three displayed block
cutoffs. This is a finite-cutoff convergence result, not a certified error
bound for the infinite recursion and continuum integral. The derivation,
tables, and limitations are recorded in
`Machine Notes/c-Recursion/NS_SPHERE_MULTIPOINT_CROSSING_2026-08-25.md`.

The exploratory
`../higher_point_amplitude_attempts/type0b_ns_five_tachyon/` folder contains
the attempted BRY all-NS five-tachyon local
matter density: three picture-changing pairs, the timelike boson and fermion
factors, component superblocks, coherent Mobius spin lifts, the infinity BPZ
phase, and the 120-chart sphere atlas.  Its two-PCO specialization reproduces
BRY's `G,H,J` four-point combination exactly.  The same module also enforces
an important trust boundary: in equal imaginary kinematics, raw PCO-collision
convergence requires `t>=1/2`, while the undeformed positive-real Liouville
contour is residue-free only for `t<1/5`.  Its driver therefore refuses
to write an unsubtracted result until finite-part/vertical-integration layers
or the crossed-pole residues are implemented.  See
`Machine Notes/c-Recursion/TYPE0B_NS_SPHERE_FIVE_TACHYON_1TO4_2026-08-25.md`.

Use `PYTHONPATH=Code` when running from the repository root so the historical
basename imports can resolve modules in the sibling purpose folders.

## Type-0B sphere four-point test (all-c route)

`type0b_sphere_four_point_hybrid.py` implements the one-modulus four-tachyon
test. For the fixed internal contour and purely imaginary outgoing energies
`omega_a=i*t_a`, the three continuum OPE radial powers are
`(t1+t2)^2`, `(t2+t3)^2-1`, and `(t1+t3)^2`. At the equal point the raised
pair is integrable only for `t>1/2`, while absence of the first crossed
`C`-sector pole requires `t<1/2`. Thus the equal `t=0.6` fixed-contour result
is only a diagnostic: its positive continuum margin does not certify the
analytically continued Type-0B amplitude.

The evaluator now defaults to the complete continued contour in the certified
tilted complex-energy chamber. It performs a pointwise frame-0/frame-1
crossing comparison on the same c-recursion route used by the integral and refuses to begin
the moduli integral if that gate fails. This is intentional: a positive
endpoint audit alone is insufficient. Direct, inversion, and degeneration
frames all use c-recursion. Historical filenames containing `hybrid` are
retained; their amplitude drivers now explicitly select `c`.

Run the crossing-gated continued candidate with

```bash
PYTHONPATH=Code/c_Recursion python3 \
  Code/c_Recursion/evaluate_type0b_sphere_four_point_hybrid.py
```

The old equal point can be inspected, but not certified as an amplitude, with

```bash
PYTHONPATH=Code/c_Recursion python3 \
  Code/c_Recursion/evaluate_type0b_sphere_four_point_hybrid.py \
  --mode fixed-diagnostic --crossing-only
```

An explicit `--allow-crossing-failure` is required to form its channel-patched
folded integral. The evaluator reports the reduced BRY moduli integral before
the overall string normalization. Block order, momentum order and cutoff,
and Sobol depth are independent convergence axes and are
stored separately in the JSON output.
Legacy hybrid-radius/nome options are inactive under the all-c route.

The larger fixed-ledger domain

```text
omega_a = (1,0.98,0.92)_a * (-x+i*t),
0.965 < x < 1.055,
1.185 < t < 1.265
```

is certified separately by `certify_residue_convergent_ray_rectangle()`.
The analytic rectangle minima are `0.051232` for the continuum and
`0.1066`, `4.006`, `11.8406`, and `14.4852` for residue walls 1 through 4;
the pole-wall clearance is `0.0665`. Coincident trinion poles raise the
largest full-product orders to 2, 4, 3, and 4 respectively, hence at most
three powers of `log|q|`; the strictly positive radial margins make all of
them integrable. This is a convergence certificate, not yet a production
amplitude domain: the numerical residue evaluator currently supports only
full-product order two, and the continued correlator must still pass the
crossing gate after the order-three/four extension.

A simpler wall-1-only chamber is available on the more unequal ray

```text
omega_a = (0.1,1,1)_a * (+x+i*t),
0.238 < x < 0.304,
0.596 < t < 0.628.
```

Its continuum margin is at least `0.0512`, its five wall-1 residue records
have margin at least `0.056584`, and its pole-wall clearance is `0.0532`.
The largest combined pole order is two, so this chamber stays within the
present residue-order capability. The nearly equal `(1,0.98,0.92)` ray cannot
reach a continuum-convergent chamber without also crossing wall 2; reducing
the residue ledger to wall 1 requires the deliberately smaller first energy.

The analytic-continuation-first optimizer is implemented in
`type0b_sphere_four_point_continuation.py`, with the executable driver
`optimize_type0b_sphere_four_point_continuation.py`.  It certifies the complete
continuum-plus-residue contour on an entire `(x,t)` rectangle and separately
records whether the present residue evaluator supports the largest coincident
pole order.  For every fixed-ledger plumbing exponent it proves a lower bound
on the real part and an upper bound on the absolute imaginary part.  The latter
controls the log-radial oscillation left after power-law importance sampling;
formal convergence alone is not used as a numerical-quality certificate.

Run the deterministic chamber search with

```bash
PYTHONPATH=Code/c_Recursion python3 \
  Code/c_Recursion/optimize_type0b_sphere_four_point_continuation.py
```

The default 144-candidate search selects the positive-sheet ray
`(0.1,1,1)` and the interior rectangle

```text
0.258 <= x <= 0.282,
0.612 <= t <= 0.628.
```

Its complete-contour margin is at least `0.139912`, compared with `0.0512`
on the larger wall-one rectangle, and its conservative phase-to-margin bound
is `8.76847`, compared with `29.83`.  It crosses only wall 1, contains five
residue records, and has maximum full-product pole order two.  The search
writes
`results/type0b_sphere_four_point_continuation_optimization.json`.  Add
`--integrate-best` to run the existing crossing-gated 30-node momentum and
randomized-Sobol moduli integral at the optimized box center.  That output
also applies the BRY normalization and reports the matrix-model comparison;
it is labelled `integrated-low-precision` unless it passes the target-blind
relative-standard-error gate.

The production-depth center run (maximum twice-level eight, 30 momentum nodes, 512
Sobol samples in each of four independent scramblings) is stored in
`results/type0b_sphere_four_point_continuation_optimization_production.json`.
It passes the crossing gate with relative spread `9.97e-4` and the statistical
gate with relative randomized-QMC error `8.40%`.  In BRY units it gives

```text
A_WS  = -0.461004 + 0.471135 i,
A_MQM = -0.533377 + 0.437950 i.
```

The central relative discrepancy is `11.54%`.  This is a valid
subtraction-free numerical point, but not yet a precision agreement test:
the four-replica component errors are anisotropic, and block, momentum-cutoff,
and momentum-quadrature systematics must still be varied independently.  The
matrix-model value is never used in the chamber search or its precision gate.

The ten-point scan in this chamber uses a 30-node composite
momentum rule rather than the memory-intensive 96-node global rule.  Its
node counts on `P` intervals `(0,.3)`, `(.3,.7)`, `(.7,1.1)`, `(1.1,1.5)`,
and `(1.5,3)` are respectively `3,7,14,3,3`.  Run it with

```bash
PYTHONPATH=Code/c_Recursion python3 \
  Code/c_Recursion/evaluate_type0b_sphere_four_point_wall_one_scan.py
```

The default uses 512 Sobol points per replicate, only two worker processes,
and writes
`results/type0b_sphere_four_point_wall_one_ten_point_scan_positive_sheet_m30.json`, including
the domain certificate, pointwise crossing audits, independent randomized
Sobol estimates, and their real/imaginary standard errors.  A point is not
marked integrated unless its target-blind relative standard error is at most
15%.  The 30-node rule
was checked against the 96-node reference at representative points; its
frame-0/frame-1 crossing spreads remain below the one-percent gate.

The earlier `...scan_m30.json` artifact used only 64 Sobol points per
replicate and began at negative real Liouville momenta.  Every point in that
artifact fails the new precision gate, so it is retained only as an
unconverged diagnostic and must not be used as a matrix-model test.

Apply the BRY sphere normalization and compare the scan with the tree-level
matrix-model coefficient using

```bash
PYTHONPATH=Code/c_Recursion python3 \
  Code/c_Recursion/compare_type0b_sphere_four_point_wall_one_to_matrix_model.py
```

The comparison strips the common
`delta(omega-sum omega_i) * mu_F^(-2)` factor and uses
`A_WS=(8i/pi) M` and
`A_MQM=8i omega omega1 omega2 omega3 (1+2i omega)`.  In accordance with BRY,
no additional leg-pole factor is inserted.  The output records componentwise
randomized-QMC errors and labels the aggregate chi-square probability as
nominal because it neglects real/imaginary error covariance.
