# Independent NS pillow PBW audit

This audit uses only the human note's convention:

`F = Lambda^(c) prod(varrho_i^(h_i-c/24)) C_NS H`,

with `c = 3/2 + 3*(b+1/b)^2`. The human note is a read-only reference.

## Independent calculations

- `ns_pillow_direct_pbw.py`: direct NS algebra, PBW Gram matrices, primary
  Ward identities, and ordered descendant sewing. It does **not** use
  Kac weights, fusion polynomials, or any c/h-recursion.
- `ns_pillow_elliptic_audit.py`: arbitrary-order pillow coordinate products,
  the complete geometric prefactor, regular product, and proposed
  unit-seed elliptic recursion. Memoization uses exact discrete shift
  labels, not approximate floating-point weight keys.
- `check_ns_pillow_direct_pbw.py`: coefficientwise comparison for four,
  five, or six points; exact symbolic or arbitrary-precision numerical.
- `test_ns_pillow_direct_pbw.py`: quick sign, Gram, coordinate-truncation,
  leading-coefficient, and low-order comparison regression tests.

## Meaning of order

Keys `(N1,...,Nm)` denote powers `prod(p_i**(Ni/2))`, not integer powers.
Degree `D` includes **all** tuples with `sum(Ni) <= 2*D`, including all
parity sectors. This is a total-degree cutoff, not a rectangular cutoff.
The coefficient counts at degree three are 7, 28, 84 for four, five,
six points; at degree ten they are 21, 231, 1771.

The exact test keeps `b`, all `d_i`, and all `h_i` symbolic. It constructs
the PBW answer first with independent `c`, substitutes the central-charge
relation only for comparison, and reduces every difference to zero as a
rational function. No random substitutions are used for this test.

The optional `--symbolic-method residues` instead certifies the same
identity exactly by induction: it checks that the PBW denominator divides
the full allowed simple-pole product, checks every residue against the
proposed kernel times a lower-degree PBW coefficient, and computes the
large-common-weight regular part from the PBW numerator and denominator.
Absence of unaccounted poles and equality of the regular part establish
the full rational-function identity, not only residue agreement. This
form is substantially smaller for the generic six-point test.

## Reproduce from the project root

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover \
  -s Code/sphere_five_kummer_h_recursion -p test_ns_pillow_direct_pbw.py -v

python3 Code/sphere_five_kummer_h_recursion/check_ns_pillow_direct_pbw.py \
  --mode oracle --output 'Data Set/h-Recursion/ns_pillow_pbw_algebra_oracle_20260830.json'

python3 Code/sphere_five_kummer_h_recursion/check_ns_pillow_direct_pbw.py \
  --mode symbolic --points 5 --degree 3 \
  --output 'Data Set/h-Recursion/ns_pillow_pbw_symbolic_n5_degree3_20260830.json'

python3 Code/sphere_five_kummer_h_recursion/check_ns_pillow_direct_pbw.py \
  --mode symbolic --symbolic-method residues --points 6 --degree 3 \
  --output 'Data Set/h-Recursion/ns_pillow_pbw_symbolic_residues_n6_degree3_20260830.json'

python3 Code/sphere_five_kummer_h_recursion/check_ns_pillow_direct_pbw.py \
  --mode numeric --points 6 --degree 10 --case A --dps 80 \
  --output 'Data Set/h-Recursion/ns_pillow_pbw_n6_A_degree10_dps80_20260830.json'
```

Repeat the numerical command for points 5 and 6 and cases A, B, C, D.
The 110-digit repeat uses `--points 6 --case C --dps 110`. The four-point
control uses `--points 4 --case A`, with the same degree options.
Dependencies are Python 3, SymPy, and mpmath; no computer-algebra server
or external block code is required. The optional `oracle` test imports
the repository's existing exact NS descendant implementation.
The numerical PBW engine uses diagonally scaled Cholesky solves for the
positive-definite Gram matrices of these real positive-weight fixtures;
it is not a general-purpose nonunitary/complex-weight numerical solver.

Numerical JSON files contain both independently obtained values for
every coefficient, errors by parity and degree, generic parameter values,
precision, and the minimum visited recursion denominator. Symbolic JSON
files record exact-zero status for each coefficient. The tolerance is
`1e-50` by default, and the program exits nonzero on a failed comparison.

The TeX machine note in `Machine Notes/h-Recursion/` records the formulas
and audit results. Finite-order checks do not prove the general-n seed,
give a finite-coordinate truncation error, or cover Ramond/upper external
components. The older `check_ns_sphere_five_elliptic_h_recursion.py`
compares with c-recursion, not direct PBW, and uses a low-order coordinate
helper: it should not be repurposed as the degree-ten PBW test.

## Independent comparison with the literature c-recursion

`check_ns_pillow_c_recursion.py` uses the separate implementation of
Belavin--Geiko, [arXiv:1806.09563](https://arxiv.org/abs/1806.09563),
equations (3.6)--(3.18) and Appendix A, in `Code/c_Recursion/`.
The regular part there is the global osp(1|2) block, not the pillow
product. Its moving-c residue calculation never calls the elliptic
h-recursion or a PBW solver. The runner adds only per-instance caches
and an entry point selecting all eight (six-point) parity sectors.

Both c-recursive plane coefficients and h-recursive elliptic coefficients
are recomputed in each run. The exact pillow coordinate products and
full conformal factor from `ns_pillow_elliptic_audit.py` convert the former
to the same unit-seed H as the latter. Belavin--Geiko's central charge
is multiplied by 3/2, including both pole values and the Jacobian.
Their plane plumbing ratios are the reverse-ordered list of our x_i;
they must not be identified with the elliptic p_i or their product.
The interior line of arXiv v1 equation (3.18) has an indexing offset:
the vertices beside h_i in Figure 3 are d_(i+1), d_(i+2), not the printed
d_(i+2), d_(i+3). The existing library uses the adjacency of that figure
and the defining sewing equation (3.15). The machine note records this
explicitly; the test is not a literal transcription of the misindexed line.

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover \
  -s Code/sphere_five_kummer_h_recursion -p test_ns_pillow_c_recursion.py -v

PYTHONDONTWRITEBYTECODE=1 python3 \
  Code/sphere_five_kummer_h_recursion/check_ns_pillow_c_recursion.py \
  --points 6 --case A --degree 10 --dps 80 \
  --pbw-ledger 'Data Set/h-Recursion/ns_pillow_pbw_n6_A_degree10_dps80_20260830.json' \
  --output 'Data Set/h-Recursion/ns_pillow_c_comparison_n6_A_degree10_dps80_20260830.json'
```

Repeat for five and six points and cases A--D. Use case C at 110 digits
for the precision repeat, with the matching 110-digit PBW ledger.
The four-point control is case A, degree 10, 80 digits. The optional
`--pbw-ledger` reuses the previously computed independent PBW H values
only as a third comparison; omit it to run solely c versus h. It is
never an input to either recursion. Ledger metadata and coverage are
validated before comparison.

Outputs record every plane c-recursive coefficient, its transformed H,
the freshly computed h-recursive H, errors by degree and parity, and
source hashes. Optional PBW-reference values and errors are also stored.
No special prescription for confluent moving-c poles is implemented;
the independent library raises on such a collision.

The completed degree-ten audit passes all four five- and six-point
fixtures. At 80 digits the maximum c-versus-h scaled errors are
`3.361e-70` (five points) and `2.415e-67` (six points). The six-point
case-C repeat at 110 digits gives `1.475e-97`; the four-point control
also passes. Full results, source-file hashes, and the 80/110-digit
stability comparison are in
`Data Set/h-Recursion/ns_pillow_c_comparison_summary_20260830.json`.
This is a finite-order comparison, not an all-orders proof or a bound
on the error at fixed nonzero coordinates.
