# Restricted inverse versus fermion zero-mode recovery

The two saved physical blocks **agree coefficient by coefficient through
total level 10** at the stored benchmark:
\(\mathbb F_0^{(+,+)}\), \(p_1=0\), \(b=7/5\),
\((P_1,P_2,P_3)=(11/23,13/29,17/31)\).

All **506 plumbing monomials and 4,048 parity coefficient slots** were
compared, including zeros. Both computations used 384-bit arithmetic.
The comparison parsed their saved decimal midpoints at 160 decimal digits;
it did not round them to binary64 or apply small-difference clipping.

| Quantity | Result |
| --- | ---: |
| Maximum absolute difference | 2.29682e-106 |
| Maximum scaled difference | 3.1067634e-108 |
| Coefficients failing scaled tolerance 1e-70 | 0 |
| Comparison runtime, including loading and parsing | 0.07053 seconds |

The scaled difference is \(|a-b|/\max(1,|a|,|b|)\). The largest absolute
difference occurs in the coefficient of \(q_1^2q_2^4q_3^4\) with parity
\((0,0,0)\); the largest scaled difference occurs at \(q_1^3q_2^7\),
also with parity \((0,0,0)\).

Only the requested comparison was performed. Neither block was recomputed,
and no direct physical PBW sewing or other cross-check was run. The two
methods share primary and Ward routines, and the original computations
discard scalar terms below 1e-80. These discrepancies describe agreement
of their stored coefficients, not an independent bound on common errors.

- [Detailed comparison, including results at each total level](restricted_vs_zero_mode_level10.json)
- [Restricted-inverse physical block](restricted_level10.json)
- [Zero-mode physical block](complex_level10.cold.json)
- [Comparison script](compare_saved_recovery_methods.py)

Reproduce without recomputing either block:

```sh
python3 Code/ramond_zero_mode_recovery/compare_saved_recovery_methods.py \
  --json Code/ramond_zero_mode_recovery/restricted_vs_zero_mode_level10.json
```
