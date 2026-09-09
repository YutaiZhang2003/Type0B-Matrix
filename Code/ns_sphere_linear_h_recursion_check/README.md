# NS sphere linear-channel h-recursion check

This directory compares the correlated fixed-difference NS h-recursion with
an independent direct PBW/Ward sewing calculation for sphere four-point and
five-point blocks.

Run from the repository root:

```bash
python3 Code/ns_sphere_linear_h_recursion_check/check_order3.py
```

The default cutoff retains every monomial through physical level three on
each internal edge.  Thus the five-point test is rectangular in
\((q_1,q_2)\) and includes \((n_1,n_2)=(3,3)\).  It checks every admissible
bottom-component trinion-sector routing:

- \((0,0)\) and \((1,1)\) for four points;
- \((0,0,0)\), \((0,1,1)\), \((1,0,1)\), and \((1,1,0)\) for five points.

The PBW side imports only the NS algebra, exact Shapovalov matrices, and the
human-note fixed-parity Ward three-form.  It does not import Kac weights,
inverse null slopes, fusion polynomials, or the h-recursion.  The recursion
side uses exact rational/algebraic arithmetic and implements the base-edge
and non-base-edge fixed-difference substitutions separately.

In the human-note fixed-parity convention, the ordinary-edge residue contains
the transport phase \((-1)^{rs}\) multiplying the two raw weight-only fusion
polynomials.  The four-point \((1,1)\) routing at level \(1/2\) changes sign
if this phase is omitted, so the certificate tests it directly.

The complete coefficient ledger is written to `results_order3.json`.

## Numerical four-point check against c-recursion

Run

```bash
python3 Code/ns_sphere_linear_h_recursion_check/check_four_point_numerical_c_recursion.py
```

This second certificate contains no PBW calculation.  It compares the
fixed-difference h-recursion directly with the production BRY c-recursion for
two generic parameter sets.  Every even and odd local coefficient through
physical level 16 is checked at 80-digit working precision.  The resulting
chiral blocks are also evaluated at two real cross ratios and one complex
cross ratio.  For the pointwise comparison the c-recursion is independently
resummed with exact global osp(1|2) blocks at its leaves; its result is
compared with the level-16 h-series.  The complete ledger is written to
`results_four_point_numerical_c_recursion.json`.
