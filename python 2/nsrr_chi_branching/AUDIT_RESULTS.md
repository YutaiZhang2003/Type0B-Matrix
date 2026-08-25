# Exact audit results

The exhaustive command

```bash
python3 -u python/nsrr_chi_branching/audit_stored_values.py --full --samples 2
```

completed with:

- 12/12 direct Ramond norm residuals equal to zero;
- 864/864 stored NS–R–R three-point residuals equal to zero;
- 16/16 additional residuals at `(n1,n2,n3)=(3/2,3/4,3/4)` equal to zero.

The stored grid consists of

```text
n1 = 0, 1/2, 1
n2,n3 = 1/4, 3/4, 5/4
epsilon2,epsilon3,f = 0,1
eta = +1,-1
```

at both exact samples defined in `ramond_three_point_grid/compute_grid.py`.
The run uses exact rational functions and `Q(i,sqrt(2))`; no floating-point
tolerance is involved.

The signed-sheet audit independently checked 64/64 residuals for
`n2,n3=+/-1/4,+/-3/4` at the first exact sample.  The quick audit also checks
the explicit crossed closed form at `(0,3/4,3/4)` and gives 8/8 zero
residuals.
