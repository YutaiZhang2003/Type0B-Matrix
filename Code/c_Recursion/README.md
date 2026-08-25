# c-recursion

Central-charge recursion code for NS and Ramond blocks lives here together
with its global/regular seeds, direct finite-level oracles, sphere and
genus-two consumers, plotting tools, and regression tests.

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
restores the Mobius-frame covariance factor. The reproducible five-point
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
