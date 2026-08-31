# NS torus two-point h-recursion level-six check

This check compares two independent constructions of the bottom-component
NS two-point necklace block:

1. exact PBW sewing from NS Shapovalov matrices and fixed-parity Ward
   three-point forms;
2. the fixed-weight-difference internal-weight recursion implemented by
   `NSTorusTwoPointHRecursionBlock`.

The default command checks every allowed monomial

\[
q_1^{n_1}q_2^{n_2},\qquad n_1+n_2\leq 6,
\qquad 2n_1\equiv 2n_2\pmod 2.
\]

The parity condition follows because both external insertions are bottom
components of even NS superprimaries. Levels are otherwise allowed to be
half-integral.

Run from the repository root:

```bash
python3 Code/ns_torus_two_point_h_recursion_check/check_level6.py
```

The PBW side does not import any Kac weights, fusion polynomials, null-vector
residues, character seed, or h-recursion function. It contracts

\[
\operatorname{Tr}\!\left(
B_{h_1}^{-1}\rho(h_1,d_1,h_2)
B_{h_2}^{-1}\rho(h_2,d_2,h_1)
\right)
\]

at each pair of levels. The recursion side is evaluated only after this exact
rational result has been constructed.

The checked-in certificates are:

- `results_level6.json`, for the default rational sample;
- `results_level6_second_sample.json`, for the independent sample
  \((b,h_1,h_2,d_1,d_2)=(137/100,79/100,97/100,31/100,43/100)\).

To test generic internal weights rather than only two hand-picked points, run

```bash
python3 Code/ns_torus_two_point_h_recursion_check/check_generic_weights.py
```

This uses seed `20260827` to draw eight independent rational pairs
\((h_1,h_2)\), holds \((b,d_1,d_2)=(127/100,27/100,9/25)\) fixed, and checks
all 49 coefficients through total level six at every pair.  The full
coefficient-by-coefficient ledger is written to
`results_generic_h_sweep_level6.json`.

For an analytic rather than sampled check, run

```bash
python3 Code/ns_torus_two_point_h_recursion_check/check_symbolic_level3.py
```

This fixes \((b,d_1,d_2)=(127/100,27/100,9/25)\), hence fixes the central
charge to an exact rational value, but leaves \(h_1,h_2\) symbolic.  For all
16 allowed coefficients through total level three it clears the denominators
of \(F^{\rm PBW}_{n_1,n_2}-F^{h\text{-rec}}_{n_1,n_2}\) and checks that the
remaining numerator is the zero polynomial.  Full rational expressions and
their hashes are recorded in `results_symbolic_level3.json`.
