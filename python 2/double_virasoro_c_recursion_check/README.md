# Double Virasoro versus NS c-recursion

This directory implements the end-of-Section-6 comparison through total
plumbing level four.

Run from the repository root:

```sh
python3 python/double_virasoro_c_recursion_check/check_double_virasoro_c_recursion.py
```

The command computes both parities for all eight lift choices and writes:

- `level4_results.json`: every coefficient and comparison diagnostic;
- `level4_tables.tex`: the complete unit-lift coefficient tables included by
  `agent_notes/double_virasoro_c_recursion_level4.tex`.

The factorization uses the theta-graded convolution dictated by the quadratic
theta sewing sign.  With this product, all 165 retained coefficients agree for
each of the eight lift choices.  The JSON output also records the first failure
obtained from the naive ordinary quotient by the free-fermion block.

Use `--require-agreement` to make the calculation an executable regression
test:

```sh
python3 python/double_virasoro_c_recursion_check/check_double_virasoro_c_recursion.py \
  --require-agreement
```
