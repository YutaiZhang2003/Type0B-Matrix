# Genus-two NS c-recursion: order-eight stress test

Date: 2026-08-01

## Verdict

The previous total-order-six check is faithful for generic weights.  Before
the order-eight changes, the unmodified program reproduced all 455 direct
theta-sewing coefficients with maximum absolute error
`5.360156762890256e-13`.  The faster cached double-precision Ward solver used
for the order-eight run reproduces the same order-six test with maximum
absolute error `1.7033041643799152e-12`.

At the genuinely generic order-eight benchmark

```text
c = 41.3
(h_0,h_1,h_infinity) = (0.731,0.913,1.173)
(xi_0,xi_1,xi_infinity) = (1,1,1)
```

all 969 coefficients with `n_0+n_1+n_infinity <= 16` agree between direct
finite-level sewing and the sector-coupled c-recursion.  The maximum absolute
error is `1.0713847586885095e-09`, at twice-level `(0,5,11)`:

```text
direct     = 767.8733692113085
recursion  = 767.8733692123799
```

The maximum scale-normalized error over the 969 coefficients is
`9.124115995894651e-11`.

## Why the order-six comparison is independent enough

The two sides share only convention-level boundary data: the eight primary
three-form normalizations and the fixed theta-graph orientation.  Their
level-dependent constructions are different.

| Direct level truncation | Recursive computation |
|---|---|
| Enumerates NS PBW states | Enumerates admissible Kac labels `(r,s)` |
| Normal-orders the super-Virasoro algebra | Uses `c_(r,s)(h)`, `J_(r,s)`, and `A_(r,s)` |
| Builds and inverts finite-c Gram matrices | Shifts one edge by `rs/2` and lowers its coefficient level |
| Generates the full descendant trinion tensor from Ward identities | Uses two local fusion polynomials and sector transport |
| Contracts the two theta trinions directly | Starts from the independently assembled vacuum/global regular block |

The direct generic block is not used to fit the recursive answer.  The vacuum
regular term is separately computed in the irreducible `h=0` quotient, and
the global regular term is computed from the closed osp(1|2) coefficient.
The polarized vacuum/global sign is falsifiable: deleting it already produces
an order-one error at order six, as documented in the recursion note.

## Exact symbolic low-order check

`Code/ns_genus2_symbolic_low_order.py` performs the coefficient comparison in
the rational function field

```text
Q(c,h_0,h_1,h_infinity)
```

through total twice-level six (physical total order three).  It builds the PBW
Gram matrices and Ward tensors symbolically and compares them with a separate
symbolic implementation of the triangular recursion.  All 84 differences
simplify identically to zero.  This cutoff contains the first three NS Kac
channels, `(3,1)`, `(2,2)`, and `(5,1)`, nested residues on distinct edges,
and the first nontrivial theta-vacuum coefficients.

The low-order pole data rationalize to

```text
c_(3,1)(h) = 6 - 3 h - 6/(2 h + 1),
c_(2,2)(h) = 3/2 - 8 h,
c_(5,1)(h) = 13/2 - h - 9/(h+1).
```

For example, with edge zero singular and with `U` denoting the exact global
coefficient,

```text
D_(3,0,0) - U_(3,0,0)
  = 6 (h_1-h_infinity)^2
    /[(2 h_0+1)((2 h_0+1)c+6 h_0^2-9 h_0)]
  = R_(0;3,1)^(odd)/(c-c_(3,1)(h_0)).
```

At twice-level `(3,1,0)`, put

```text
A = 2 h_0 h_1 + 2 h_0 h_infinity - h_0
    - 2 h_1^2 + 4 h_1 h_infinity + h_1
    - 2 h_infinity^2 + h_infinity.
```

Then direct sewing gives

```text
D_(3,1,0) - U_(3,1,0)
  = -3 A^2/[4 h_1(2 h_0+1)
             ((2 h_0+1)c+6 h_0^2-9 h_0)].
```

The recursion gives the same expression as

```text
- R_(0;3,1)^(even)/(c-c_(3,1)(h_0))
  * U_(0,1,0)(h_0+3/2,h_1,h_infinity),
```

where the minus sign is the exact theta orientation transport and the shifted
global coefficient is `U_(0,1,0)=1/(2 h_1)`.  Finally, `(4,0,0)` checks the
two pole families simultaneously:

```text
D_(4,0,0)
  = U_(4,0,0)
    + R_(0;3,1)^(even)/(c-c_(3,1)(h_0)) * 1/(2 h_0+3)
    + R_(0;2,2)^(even)/(c-c_(2,2)(h_0)).
```

SymPy reduces the difference between the two sides of each displayed identity,
and of all other coefficients at the cutoff, to the literal integer zero.
At `(5,0,0)` this includes the new `(5,1)` pole together with the lower
`(3,1)` channel, while `(3,3,0)` includes nested residues on both nonzero
edges.

The direct large-`c` limit also isolates the first vacuum seed exactly:

```text
lim_(c->infinity) D_(0,3,3) - G_(0,3,3) = +1,
lim_(c->infinity) D_(3,0,3) - G_(3,0,3) = +1,
lim_(c->infinity) D_(3,3,0) - G_(3,3,0) = -1.
```

These are precisely the three ordered theta-vacuum coefficients.  Hence the
order-three symbolic test checks the vacuum/global regular prescription and
its orientation signs, rather than only the Kac-pole part of the recursion.

## Order-eight regular seed

The irreducible vacuum quotient was extracted independently at five large
central charges by a polynomial fit in `1/c`.  Through physical total order
eight there are 97 nonzero integer coefficients (31 through order six and 66
new coefficients at orders seven and eight).  The maximum difference between
the numerical extraction and the stored integer table is
`2.8421709430404007e-13`.

As a second, non-Gram-matrix check, the lifted Schottky primitive product was
compared with the complete order-eight table for all eight edge-lift choices.
Under uniform plumbing scales `t=0.08` and `t=0.04`, the maximum remainders are

```text
1.364465446318519e-05
3.932199965639427e-08
```

Their ratio is `0.0028818611539405`, consistent with the first omitted
physical total level being nine (the leading expectation is `2^(-9)`).

## Error by total twice-level

For total twice-level `0,1,...,16`, the maximum direct-versus-recursive errors
are

```text
(0,
 0,
 4.440892098500626e-16,
 3.3306690738754696e-16,
 2.220446049250313e-15,
 1.1102230246251565e-15,
 4.440892098500626e-15,
 5.329070518200751e-15,
 1.4210854715202004e-14,
 2.842170943040401e-14,
 6.838973831690964e-14,
 1.2567724638756772e-13,
 8.455458555545192e-13,
 2.1316282072803006e-12,
 2.5941915282601258e-11,
 2.8535396268125623e-11,
 1.0713847586885095e-09)
```

The genus-one trace continues to equal the generic NS character exactly
through physical level eight.

## Coincident-pole discovery

The old order-six benchmark weights `(0.73,0.91,1.17)` are not generic at the
higher cutoff.  The first failure occurs in the coefficient with top
twice-level `(10,3,0)`.  A shifted `(5,1)` pole on edge zero lands at

```text
c = 1.1423404255319145
```

which equals the `(3,1)` pole on edge one.  Direct scalar evaluation therefore
attempts a zero denominator in the shifted subblock.  This does not contradict
the generic recursion.  It shows instead that an exceptional-locus extension
must retain higher pole orders.  Already at twice-level `(3,3,0)`, exact PBW
sewing with `h_1=h_0` gives the reduced denominator

```text
4 h_0^2 ((2 h_0 + 1) c + 6 h_0^2 - 9 h_0)^2,
```

with no numerator cancellation.  Merging equal simple-pole keys is therefore
insufficient.  A correct implementation must analytically detune the weights
before taking the combined limit or propagate higher-order Laurent jets and
their derivative terms.  The scalar code reports the unsupported collision
explicitly instead of emitting an unexplained division by zero.

Thus the order-eight result certifies the generic recursion and its
order-by-order global/vacuum regular block.  A separate detuned or
higher-order Laurent implementation is still required to extend the recursion
to the old benchmark point beyond total order six.

## Reproduction

Original order-six benchmark:

```bash
python3 Code/ns_genus12_finite_c_check.py
```

Generic order-eight benchmark:

```bash
python3 Code/ns_genus12_finite_c_check.py \
  --c 41.3 \
  --weights 0.731 0.913 1.173 \
  --genus-one-order 8 \
  --genus-two-order 8
```

Independent lifted-Schottky check through order eight:

```bash
python3 Code/ns_vacuum_schottky.py
```

Exact symbolic check through total physical order three:

```bash
python3 Code/ns_genus2_symbolic_low_order.py
```
