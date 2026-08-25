# Double Virasoro

This folder contains only active two-Virasoro and independent PBW
verification code, separated by role:

* `all_ns/` implements and tests the all-NS branching/fusion data.
* `nsrr/` contains the independent Ramond PBW/Ward oracle, auxiliary-Majorana
  sewing, and their focused tests.
* `audits/` contains executable factorization and production-boundary audits.
* `../full_ramond_block_runtime/` is the certified double-Virasoro q-expansion
  driver and uses Yuchen's finite `L_1` branching grid from
  `../ramond_branching_recursion/`.

The physical Ramond free-field ground labels are `w^+` and `w^-` from the
start.  Their zero-mode phases are imposed directly from equation (5.1), the
free-field-to-PBW transition is built in the same basis, and there is no
endpoint conversion or second physical Ramond basis.  The auxiliary ground
three-point values are `(1,i)` as in Section 8.

The current Human-note definitions are implemented directly:

```
enlarged first-tube sign:       (-1)^(A + mathsf_A)
auxiliary first-tube sign:      (-1)^mathsf_A
double-Virasoro branch sign:    (-1)^(2*n_1)
physical PBW additional sign:   none
```

There is no post-processing `eta_1` flip.  With these signs built into the
series generators, the unrestricted q^6 check has
1,120 parity-resolved coefficients and maximum errors `9.4132e-10` for
`p1=0` and `4.1311e-10` for `p1=1`.

The intrinsic NS-primary parity is selected with `--primary-parity p1`.  The
`L_{\pm1}` action matrices are parity-independent; only their direct boundary
values change.  The production grid agrees with PBW at the fundamental and
half-NS boundary layers to `5.8e-15` for both `p1` values.

Superseded and exploratory Human-note computations are retained outside the
active import graph in `../unused_human_note_computation/`.
