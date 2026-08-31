# Independent double-Virasoro branching-cutoff test

## Question and fixed inputs

The user requested a direct test of the contribution of additional
double-Virasoro branching sectors. The previous `L=5` calculation imposed
the combined condition `branching shift + descendant level <= 5`. It did
not independently vary the size of the branching sum at fixed descendant
accuracy.

This test uses `b=1.4`, `t=Re Omega_original,12=0.60`, and the same `N=3`
momentum quadrature, three-point coefficients, primary powers, and physical
free denominator as the completed L5 local trial. The original period
matrix is `[[i, 0.60+i/2], [0.60+i/2, i]]`. The re-marked NS-at-infinity
source sewing parameters, in Human-Note `(NS,R,R)` slot order, are

```
q_NS = -0.03515917339490496 + 0.025344924433414715 i
q_R1 = -0.04059269805965829 + 0.02978808739157108 i
q_R2 = -0.03938929794343916 - 0.02508339199638473 i
```

Their saved forward-period residual is `2.54e-13` or smaller. No new
plumbing fit or fitted normalization is introduced.

## Independent cutoffs

For each branching triple define

\[
s(\mathbf n)=2n_{\rm NS}^2+2n_{\rm R1}^2+2n_{\rm R2}^2-\frac14.
\]

The new sum includes all triples with `s <= K`. Each triple receives the
product of the two ordinary Virasoro blocks through the **same combined
descendant order D**, including their universal vacuum factors. There is
no subsequent total-L5 cutoff: terms through `s+D` are retained.

| K | Branch-label triples before parity multiplicity |
|---:|---:|
| 3 | 80 |
| 4 | 112 |
| 5 | 152 |
| 6 | 196 |
| 8 | 300 |
| 10 | 388 |

The primary sweep is `K=5,6,8,10` at fixed `D=5`. `D=4,6` and `K=3,4` are
controls. The high-K branching grid is shared across K values at a given
momentum, so changes in K do not change already included coefficients.
Branching coefficients use the checked Ward recursion, and Virasoro
descendants use the checked c-recursion. No protected kernel is edited.

## What the reported quantity means

Only the equal-HJS-sign components are determined by the present
double-Virasoro interface. For them, the enlarged series is evaluated at
the fixed sewing parameters; supported star characters are divided by
the auxiliary Majorana, and the checked equal-sign Ward support is used
to recover the literal block components. Unsupported-character leakage
is recorded, rather than silently treated as physical data.

The auxiliary is held fixed at level 16. The level-14/16 change is
`2.29e-13` in the saved scaled norm. This auxiliary quotient is distinct
from the physical free superfield denominator of Q, which is unchanged.

The opposite-HJS-sign PBW blocks are held fixed at their saved **total
level 5** values. The hybrid diagnostic combines those unchanged mixed
terms with the varied equal-sign terms. It preserves the original
`i^f` factors at both vertices and the separate `(-1)^f` sewing sign.
It is not a uniformly high-order NSRR partition function, and the
previously unresolved physical spin/nonchiral dictionary remains
unresolved. `physical_Q` and `physical_Z` are deliberately null.

`K=5,D=5` differs from the old total-`L=5` approximation: the new calculation
keeps additional descendants of non-ground branching sectors and uses a
pointwise auxiliary quotient. Consequently, only **fixed-D changes in K**
isolate the added branching sectors. The old L5 calculation is reproduced
separately by formally restricting the independently assembled series to
total level five.

## Code and checks

- `Code/genus_2/nsrr_branching_cutoff_probe.py`: separate test driver.
- `Code/genus_2/test_nsrr_branching_cutoff_probe.py`: reconstruction,
  counting, independent-cutoff, and star-quotient tests.
- `Code/genus_2/audit_nsrr_branching_cutoff_probe.py`: independent
  reconstruction of saved shell contributions, trial contraction,
  quadrature reduction, and plotting.
- Output directory: `Data Set/nsrr_branching_cutoff_t060_N3_20260830/`.

The seven dedicated unit tests pass, including coefficientwise reconstruction
of the protected total-level series and a synthetic test whose `s=4.5`
sector retains its level-one descendant above total level five. All 46
combined tests including the original trial, protected-kernel boundary,
graded sewing, and free-factor tests also pass. Each actual momentum node additionally checks high-grid L5
coefficients against a separate low-grid calculation, and compares the
latter with the archived L5 blocks on all four lifts.

The driver caches only the literal one-module `L1`/`L-1` action outputs
from the protected solver. Each entry is keyed by the kernel hash, sector,
momentum, b, realization, label, parity, and precision. Payload hashes and
per-entry locks guard reuse. The Ward system is still solved separately
for every momentum triple. Tests establish exact equality of cached and
uncached actions and branching coefficients.

An initial run exposed a node-index metadata bug: a parity-loop temporary
overwrote the node index. It was stopped before reduction; its files were
preserved under the `_initial_metadata_bug` suffix. The numerical probe
itself completed, but it is not used in the final integration. The corrected
driver has a specific regression test and reruns that node. No old L5 trial
data or protected code was modified.

## Completed result and smoking-gun control

The corrected run completed **all 27 nodes in 554.67 seconds (9.24 minutes)**
with two local workers. The uncached first-node benchmark took 180 seconds;
reusing repeated actions reduced the complete-grid cost substantially.

At fixed combined Virasoro descendant order **D=5**:

| Branching cutoff | Branch-label triples | Q_hybrid |
|---:|---:|---:|
| 0 (positive control) | 4 | 1.848277401683597e-7 |
| 5 | 152 | 1.860050995134766e-7 |
| 6 | 196 | 1.860051067100018e-7 |
| 8 | 300 | 1.860051071133188e-7 |
| 10 | 388 | 1.860051071121721e-7 |

The positive control is responsive: restoring the low branching sectors
from **K=0 to K=5 changes Q_hybrid by +0.6370036%**. In contrast, adding
the 236 higher label triples from **K=5 to K=10 changes it by only
+0.0000040852%**. These comparisons use identical momentum quadrature,
descendant order, vertex coefficients, auxiliary quotient, physical free
factor, and mixed-sign PBW terms.

To check that a small net result is not simply cancellation between
momentum nodes or successive half-level shells, the audit computes, for
each added shell and equal-sign component,

\[
 |c_\eta^2|\left(2|F_{\rm old}|\,|\Delta F|
                         +|\Delta F|^2\right),
\]

with the unchanged positive primary/measure factors, and sums these
triangle-inequality bounds over all momentum nodes and all added shells.
Relative to Q_hybrid(K=5,D=5), that bound is **1.17503e-7**, or
**0.0000117503%**. Coherent sums of branch terms *within* each shell are
preserved. This bounds the computed finite-shell change, not the infinite
tail beyond K=10 and not unquantified arithmetic errors.

The all-NS diagnostic comparison remains approximately **-4.58469%**.
Even the absolute shellwise bound is about 390,000 times smaller than
that gap. The tested higher branching sectors therefore cannot account
for the percent-level mismatch in this fixed-D hybrid calculation.

### Independent descendant-order control

| D | K5 to K10 change in Q_hybrid |
|---:|---:|
| 4 | +0.0000040075% |
| 5 | +0.0000040852% |
| 6 | +0.0000040578% |

At fixed K=10, raising D=5 to D=6 changes Q_hybrid by **+0.000472013%**.
Thus the small high-branch contribution is also stable under the tested
descendant-order refinement. The K8-to-K10 increment is around `6.2e-12`
relatively at D=5; its tiny sign should not be overinterpreted at the
present binary64 conditioning level.

![Independent branching-cutoff sweep](</Users/yutaizhang/Desktop/Type0B-Matrix/Data Set/nsrr_branching_cutoff_t060_N3_20260830/comparison.png>)

### Audit and scope of the conclusion

- Independent reconstruction errors: block `5.41e-16`, contraction
  `9.19e-16`, frozen mixed contribution `1.82e-16`, reduction `2.85e-16`.
- The corrected cached K10 node reproduces the uncached node's blocks,
  shell sums, and integrands **bit for bit**. Its metadata is corrected.
- The 198 cached one-module actions pass their identity/payload hash audit;
  their maximum recorded fit residual is `9.95e-12`.
- All archived L5 physical blocks are reproduced exactly by the separate
  low-grid runtime. The high-grid assembly reproduces their enlarged
  total-L5 series to `4.53e-7` in maximum scaled coefficient norm and
  `1.06e-10` in the evaluated parity-component norm at this plumbing point.
  These are conditioning checks, not a certified infinite-order error bar.
- Maximum high-grid branching Ward residual: `2.80e-9`; maximum recorded
  unsupported-character leakage: `5.42e-11`.
- All eight protected kernel hashes are unchanged; 46 regression tests pass.

**Conclusion:** the controlled test strongly disfavors an insufficient
equal-sign double-Virasoro branching cutoff as the explanation of the
remaining mismatch at this point. It does **not** identify the remaining
cause, establish convergence of the mixed-sign PBW tail, certify the
physical Ramond sewing projector, or prove equality with the NSNSNS
partition. The mixed-sign and physical-assembly limitations must remain
explicit; numerical convergence of this hybrid trial does not remove them.

Artifacts in the output directory include `summary.json`, `verification.json`,
`comparison.csv`, `branch_shells_D5.csv`, `comparison.svg/png`, `status.json`,
27 node shards, and the hashed action cache. The verification records both
the positive control and the absolute shellwise bound under `smoking_gun`.
