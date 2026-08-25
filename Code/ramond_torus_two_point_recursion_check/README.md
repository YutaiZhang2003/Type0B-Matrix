# q1 -> 0 Ramond block check with torus two-point c-recursion

This folder compares the two sides of the Ramond double-Virasoro identity at
`q1=0` through total level 6 in `(q2,q3)`.

The left side is computed by direct PBW sewing of the Ramond SCA torus
two-point block and the auxiliary Ramond-fermion block, followed by the
Ramond convolution.  The right side uses the Ramond branching coefficients
and two ordinary Virasoro torus two-point blocks computed as formal bivariate
series by central-charge recursion.

Run:

```sh
python3 python/ramond_torus_two_point_recursion_check/check_recursion.py \
  --cutoff 6 \
  --output python/ramond_torus_two_point_recursion_check/results.json
```

At `b=7/5` and `P=(11/23,13/29,17/31)`, all 112 parity-resolved block
coefficients agree.  The maximum relative errors are

- `4.214e-7` for `f=0, eta=+1`;
- `4.548e-7` for `f=1, eta=-1`.

As an independent check of the new recursion, all 88 ordinary Virasoro torus
two-point series were also compared with direct Virasoro sewing.  Their
maximum relative error is `2.420e-13`.  The complete run takes `3.548 s`.

`results.json` contains both sides coefficient by coefficient, errors grouped
by total level, the ordinary-recursion audit, and timings.
