# The `q1 -> 0` check of the extended Ramond block

This directory checks the same coefficient of
`widehat{mathbb F}_0^{(+,+)}` in two independent representations.

1. Set `q1=0` in the double-Virasoro decomposition in `SCblock.tex`.  Only
   `n1=0` and the level-zero state on the first tube survive.  The code uses
   the main-notes Ward system for the required branching coefficients and
   sums the two ordinary Virasoro torus two-point blocks over descendants.

2. Regard the result before adjoining the free Majorana fermion as a torus
   two-point Ramond super-Virasoro block.  The code reconstructs every block
   coefficient from its Ramond null poles and its large-momentum polynomial,
   following the two ingredients of the Hadasz--Jaskolski--Suchanek
   `h`-recursion, and then convolves it with the free-Majorana block.  At the
   present low-level stage, the coefficients of the regular polynomial are
   fixed by independent super-Virasoro descendant evaluations at
   large-momentum sample points; no value at the target momenta is used.

The 2012 paper itself treats torus one-point blocks, not torus two-point
blocks.  Consequently, this is a verified low-level extension of its pole
reconstruction, not a quotation or implementation of a closed two-point
formula from that paper.  A standalone recursion at arbitrary level still
requires a general formula for the two-point large-weight seed.  Moreover,
the present block depends on the signs of the two Ramond momenta.  A
recursion directly in one common conformal weight therefore introduces
square-root sheets.  The code instead sets

```text
beta2=t,  beta3=t+d
```

and holds `d` fixed.  The two pole families are then the simple poles
`t=+/-beta_rs` and `t=+/-beta_rs-d`.  At fixed `(q2,q3)` level, the remaining
part is a polynomial in `t`; it is fixed from large-momentum samples that do
not include the target point.  Independent held-out samples verify the
rational reconstruction.

There is also a convention conversion that cannot be skipped.  Hadasz's PBW
basis permits `G0` in a lowering string, while the main notes exclude `G0`
and keep `w+` and `w-` as separate ground states.  This change of basis is
`beta`-dependent.  At the level-one nulls used here, the inverse-null-norm
coefficient in the main-notes basis is derived directly from the two-by-two
Gram matrix.  Both signs of every `beta_rs` residue are checked independently
against the descendant algebra.

The initial smoke test uses `f=0`, `eta=eta'=+`, and positive plumbing spin
signs.  At `q1=0` the graded convolution sign is then trivial.  The auxiliary
factor is itself a bivariate torus two-point block; it is not a character in
the single variable `q2 q3`.

Run

```bash
python3 python/widehat_ramond_q1_check/check_q1_limit.py --level 1
```

The present analytic `G0`-basis conversion is implemented through level one.
The machine-readable coefficient comparison, pole checks, held-out errors,
and timings are written to `results.json`.

## Current result

At the default generic point

```text
b=7/5, P1=11/23, P2=13/29, P3=17/31,
```

the coefficients through `q2^1 q3^1` agree as follows:

| coefficient | double Virasoro | `h`-recursion times fermion |
|---|---:|---:|
| `1` | `4.00000000025` | `4.00000000000` |
| `q3` | `1.74840350398` | `1.74840350512` |
| `q2` | `1.57981944205` | `1.57981944296` |
| `q2 q3` | `10.8452092557` | `10.8452092933` |

The maximum relative difference is `3.46e-9`.  The maximum direct-versus-
analytic null-residue difference is `5.08e-10`, and the maximum held-out
rational-reconstruction error is `8.11e-13`.  The double-Virasoro and
`h`-recursion stages take about `0.69 s` and `4.20 s`, respectively, on the
current machine.
