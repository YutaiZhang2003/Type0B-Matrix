# Validation and reproduction

All comparisons use the human-note ordinary central charge and the same
unit-seed H, including the complete sphere-to-pillow conformal factor.
Cutoffs are total physical degree, including half-integer edge powers.

## Established research audit

The development audit retained independent symbolic b,d_i,h_i:

| Points | Through degree 3 | Exact method |
| --- | ---: | --- |
| 4 | 7 coefficients | Expanded rational differences |
| 5 | 28 coefficients | Expanded differences and pole/regular-part certificate |
| 6 | 84 coefficients | Pole/regular-part certificate |

The certificates include all 106 (five-point) and 342 (six-point)
candidate residue identities, all allowed denominator factors, and the
large-common-weight polynomial part. They are generic symbolic checks,
not numerical sampling. An independent algebra oracle additionally
checks 57 Gram entries and 110 Ward entries at low level.

At degree ten, four generic parameter fixtures A--D were checked for
both five and six points, including all parity sectors. Every comparison
passed a 1e-50 scaled-error threshold at 80-digit working precision:

| Comparison, worst across A--D | Five points (231 each) | Six points (1771 each) |
| --- | ---: | ---: |
| Direct PBW versus elliptic h | 3.37e-70 | 2.42e-67 |
| Literature c versus elliptic h | 3.37e-70 | 2.42e-67 |
| Literature c versus saved PBW | 1.47e-73 | 3.58e-74 |

The six-point case-C repeat at 110 digits gives c-versus-h error
1.48e-97 and c-versus-PBW error 2.34e-104. The four-point degree-ten
control also passes. Error means abs(a-b)/max(1,abs(a),abs(b)).
These are coefficient comparisons, not accuracy bounds at fixed moduli.

The independent c-recursion is our implementation of
[Belavin--Geiko, arXiv:1806.09563](https://arxiv.org/abs/1806.09563),
not a supplied data set from the authors. Its seed is the global
osp(1|2) network, not the proposed pillow product. The machine note
records the central-charge/Jacobian conversion and the graph-consistent
interpretation of the interior indices in their equation (3.18).

## Port validation

The lightweight package engine is a numerical port of the audited
research engine. It adds input checks, pole diagnostics, per-instance
cache cleanup, a parity-resolved API, geometry, and JSON serialization.
It never calls the independent validation engines.

The portable unit suite checks every generic fixture through degree
three against all three stored independent constructions, correct odd
signs, c/b parametrization, b-inverse duality, JSON round-trips,
branch lifts, invalid inputs/poles, the NS product, coordinate inversion,
and full reconstruction near the OPE cell. A seven-point low-order test
is structural only and is not claimed as an independent PBW audit.

The release also recomputes the packaged engine through degree ten for
all A--D five-/six-point fixtures, the four-point control, and the
110-digit six-point repeat. Results and source hashes are under
validation/runtime/. The release summary is release_validation.json.

## Quick tests

From the extracted package root after installation:

    python -m unittest discover -s tests -v
    python tools/verify_manifest.py

With SymPy installed, from the validation directory:

    python -m unittest discover -s Code/sphere_five_kummer_h_recursion -p 'test_ns_pillow*.py' -v

These ten preserved audit regression tests include direct Gram/Ward
checks, exact leading coefficients, all low-order parities, coordinate
truncation, and deliberate wrong-residue/seed/Jacobian negative tests.

## Recompute the packaged runtime at order ten

From the package root:

    python tools/check_runtime_against_ledgers.py --points 6 --case A --degree 10 --dps 80 --output result.json

Repeat for points 5 and 6 and cases A--D. The four-point control uses
case A. The 110-digit repeat uses points 6 and case C. The comparison
targets the preserved PBW and c-recursion values, not just the older
h-recursive implementation.

## Recompute independent PBW and c-recursion

Run these from the validation directory. They create new outputs rather
than overwriting the preserved ledgers unless explicitly directed there.

    python Code/sphere_five_kummer_h_recursion/check_ns_pillow_direct_pbw.py --mode symbolic --points 5 --degree 3 --output symbolic-five.json

    python Code/sphere_five_kummer_h_recursion/check_ns_pillow_direct_pbw.py --mode symbolic --symbolic-method residues --points 6 --degree 3 --output symbolic-six.json

    python Code/sphere_five_kummer_h_recursion/check_ns_pillow_direct_pbw.py --mode numeric --points 6 --case A --degree 10 --dps 80 --output pbw-six.json

    python Code/sphere_five_kummer_h_recursion/check_ns_pillow_c_recursion.py --points 6 --case A --degree 10 --dps 80 --pbw-ledger 'Data Set/h-Recursion/ns_pillow_pbw_n6_A_degree10_dps80_20260830.json' --output c-six.json

    python Code/sphere_five_kummer_h_recursion/check_ns_pillow_direct_pbw.py --mode oracle --output algebra-oracle.json

Symbolic multipoint checks can take substantially longer than numerical
ones. The preserved numerical PBW solver uses positive-definite Gram
matrices and Cholesky solves for these positive-weight fixtures; it is
not a general complex-weight or nonunitary numerical solver.

## Evidence limitations

The bundle does not prove the arbitrary-n large-h seed, a cap Ward
identity, or convergence/error estimates at arbitrary nonzero positions.
It also does not establish a universal speedup relative to c-recursion:
that requires a fixed-accuracy benchmark at the actual moduli. The
packaged tables facilitate reuse over moduli scans without repeating the
weight-dependent recursion.

The copied machine note has a historical low-order c-comparison subsection.
That superseded driver is not bundled; the full degree-ten replacement
and all current independent checks are included above.
