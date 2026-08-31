# Human-Note NSRR repair: scope and certificates

The literal chiral-block discrepancy is repaired. **The physical
NSRR–NSNSNS partition-function discrepancy is not yet resolved.** No new
partition value or modular-agreement claim is made by this directory.
The Human Note itself and all historical scan data were left unchanged.

## Corrections supported by independent PBW checks

1. Parity coefficients already contain the Human-Note theta quadratic sign.
   A physical fixed-lift chiral block is their ordinary Walsh sum, not their
   star spectrum. The latter is only a tool for dividing the auxiliary.
2. The auxiliary ground vector is `(1,0,0,0,0,0,1,0)`, whose star spectrum
   is `(0,0,2,2,2,2,0,0)`. Equal-HJS-sign physical data lie in the supported
   ideal. Opposite-sign data lie in the kernel and cannot be recovered by
   division. The repair supplies those missing channels explicitly using
   the packaged PBW oracle, with a default cost cap at level 3.
3. The old odd-form extension of the branching grid disagreed with PBW
   starting at NS level 1/2. It is no longer used by the physical-block
   wrapper. Instead the exact Ramond ground-partner identity is applied to
   the even double-Virasoro result. With component bits `(NS,R_one,R_zero)`,

   `B_1[p xor 4] = -i (-1)^(p_NS+p_Rone) B_0[p]`.

   This follows from `Jw^+=i w^-`, `Jw^-=w^+`, `JG=-GJ` and
   `J^T Gram J=-i Gram`, together with the already included theta sign.
   Equivalently, `F_1=-i e_4 star F_0`. See the component Ward relations in
   [HJS, equations (80) and (82)](https://arxiv.org/html/0810.1203).

The supported series still use the branching-coefficient recursion and
products of ordinary Virasoro c-recursion blocks. The nullspace completion
is explicitly **PBW**, not a claimed pure double-Virasoro computation.

At `b=1.4`, physical momenta `(0.21,0.37,0.52)`, through total physical
level 3, the certificate covers both intrinsic NS-primary parities, both
form parities, all four HJS sign pairs and all eight lift choices:

| Check | Result |
| --- | ---: |
| Parity coefficients compared | 3,840 |
| Maximum parity-coefficient difference | 6.66e-14 |
| Maximum forward auxiliary-star identity difference | 1.96e-13 |
| Five-chart, eight-lift block evaluations compared | 640 |
| Maximum evaluated-block difference | 4.70e-15 |

For nullspace components, PBW is the input, so their agreement is a
completion/projection regression, not an independent two-algorithm
determination. The supported components and forward star identity are the
independent branching/double-Virasoro versus PBW checks.

The focused regression suite passes 63 tests, including the original even
double-Virasoro/PBW check, the corrected physical projection, all-NS odd
coefficient/decomposition signs, chart transport and production guards.

## Re-solved five-point geometry

The plumbing geometry uses `(zero,one,infinity)`; the package uses
`(infinity,one,zero)=(NS,R,R)`. The old source marking had the NS cut at
zero. Simply reversing its q tuple does not preserve the local charts.

The corrected chart first exchanges the zero/infinity cuts by
`T=[[-1,-1],[0,1]]`, `A'=T A`, `B'=T^{-T} B`. It then applies the fixed
collocation B-path shift `K=[[0,1],[1,1]]`. Both operations are included in
the saved symplectic matrix, and the characteristic is transported with
the full affine action. The resulting source spin is `[11|00]`, with
geometric sectors `(R,R,NS)`. The source-to-target matrix is updated so that
the target period matrix and `[00|00]` characteristic are unchanged.

The inverse period problem is solved afresh at all five original values of
`t=Re Omega_12`. A higher-order forward collocation check gives:

| t | Maximum period-matrix residual |
| --- | ---: |
| 0.52 | 5.53e-13 |
| 0.56 | 5.13e-13 |
| 0.60 | 2.53e-13 |
| 0.64 | 3.63e-13 |
| 0.68 | 3.38e-13 |

The complete q tuples, fixed integer branches, spin transport and
cross-ratio checks are in `geometry.json`. At t=0.60 the new geometric
q tuple differs from the naively reversed old one by as much as 0.00445.

The physical scalar+Majorana denominator is recomputed in this new local
frame: the directly sewn all-NS reference is multiplied by the exact
Majorana theta-constant ratio for `[11|00]/[00|00]`. Modes 36 and 44 are
compared. The Weyl-cancelling exponent remains
`kappa=c_SL/(3/2)=1+2(b+1/b)^2`. The all-NS reference lifts are recorded only
as **reference** lifts, not asserted to be the physical NSRR lift dictionary.

## What still prevents a physical Q comparison

The Human Note's nonchiral partition derivation explicitly treats all-NS
tubes. Its Ramond section defines the chiral NSRR blocks, but the inspected
note and package do not supply the matching nonchiral Ramond ground-state
projector for this fixed-spin calculation. The old assembler copied a
sphere HJS factor and inserted an additional multiplicity of four. Neither
is a derivation of the required closed-edge projector.

This distinction matters: the irreducible nonchiral Ramond module is not
the unrestricted product of two chiral ground doublets; see
[HJS, equation (16)](https://arxiv.org/html/0810.1203). BRY also distinguish
the local Ramond spin field from its disorder/defect counterpart in
[section 2.1](https://arxiv.org/html/2201.05621). Their nonchiral sewing and
topological-line prescription must be matched to the chosen spin marking.

Consequently `nsrr_node` and the source scan entry point now fail explicitly
instead of exporting the historical assembly as a repaired partition
function. Private functions retain the old assembly only for labelled
diagnostic regressions. Target all-NS computation remains available. Its
`i*tilde_C` coefficient conversion and independent decomposition sign have
not been changed.

The next required input is the intended nonchiral Ramond sewing/projector
prescription (and the associated R-sector lift dictionary), or its precise
location in another Human Note. No fitted multiplicity, phase or spin sum
has been used to manufacture modular agreement.

## Reproduction

From the repository root, with `OPENBLAS_NUM_THREADS=1` and
`PYTHONDONTWRITEBYTECODE=1`:

```sh
PYTHONPATH=Code/genus_2 python3 Code/genus_2/nsrr_human_note_geometry.py \
  --baseline 'Data Set/nsrr_nsnsns_fivepoint_L4_N5_20260830/config.json' \
  --output 'Data Set/nsrr_nsnsns_human_note_repair_20260830/geometry.json'
PYTHONPATH=Code/genus_2 python3 Code/genus_2/certify_nsrr_human_note_blocks.py \
  --cutoff 3 \
  --geometry 'Data Set/nsrr_nsnsns_human_note_repair_20260830/geometry.json' \
  --output 'Data Set/nsrr_nsnsns_human_note_repair_20260830/block_certificate_L3.json'
```
