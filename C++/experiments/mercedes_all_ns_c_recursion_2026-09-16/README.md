# All-NS Mercedes: double Virasoro versus NS c-recursion

The two calculations agree through **total physical level 6**, at 40-digit
precision. The maximum coefficient error is

- scaled: `1.0473014012548597e-28`, with scale `max(1, abs(x), abs(y))`;
- absolute: `1.1778739568850004e-28`.

There are no failed components at tolerance `1e-18`. This is an independent
NS c-recursion comparison; no full physical SCA PBW block was computed.

## Exact scope

All six internal edges are NS, with even intrinsic primary parities. In
one-based edge notation the ordered vertices are

```
(1,2,3), (1,4,6), (2,5,4), (3,6,5).
```

We use `b=7/5`, `Q=b+1/b`, `c=3/2+3Q^2`, and momenta

```
(11/23, 13/29, 17/31, 19/37, 23/41, 29/43),
h_e = Q^2/8 - P_e^2/2.
```

The domain is every `q_1^(k_1/2) ... q_6^(k_6/2)` with integer `k_e>=0`
and `sum(k_e)<=12`: **18,564 multidegrees**. Each edge parity is `k_e mod 2`.
All eight permitted vertex-form assignments are included. Cases 0 through 7
encode `(a_1,a_2,a_3)` in binary and set `a_0=a_1+a_2+a_3 mod 2` at the
central sphere. The all-even case has 2,394 permitted multidegrees; each
other case has 2,310. All other sector/parity components are checked as
zeros. Since an NS multidegree fixes its edge parities, equality of these
coefficients implies equality for all 64 spin-lift evaluations; separate
numerical spin-evaluation runs are unnecessary.

## The independent calculations

`compare.cpp` includes the existing, unchanged double-Virasoro implementation
from `../total_level6_2026-09-16/graph_validate.cpp`, replaces its graph setup
with six NS edges, and supplies a separate main program. It uses cached NS
branching coefficients, the two full Virasoro CCY recursions, and their
Schottky vacuum factors. It obtains the physical superconformal block by
convolving out the directly sewn auxiliary fermion factor. It retains every
branch whose starting level fits the requested physical domain and reduces
each Virasoro cutoff by that starting level. The run visits 10,237 branch
tuples and 114,720 Virasoro residue transitions.

`ns_recursion.hpp` independently implements the fixed-weight NS c-recursion
from the NS sections of `Human Notes/SCblock.tex`. Its ingredients are:

1. The Kac pole `c_(r,s)(h)`, Jacobian `-dc/dh`, inverse null slope, and
   the two fusion polynomials, with `r>=2`, `s>=1`, `r+s` even.
2. Null shifts by `rs/2` on the selected edge. An odd null toggles both
   incident vertex-form labels. The fusion polynomial uses the parent label.
3. The explicit eight global Ward values and the finite Pochhammer formula
   for global `osp(1|2)` descendants, with norms `n! (2h)_(n+epsilon)`.
4. The nonconstant NS Schottky vacuum series described below, combined with
   that global block using the full local and graph fermion signs.

Pole geometry, fusion polynomials, global vertex values, global norms, and
recursion states are cached. Opposite momentum shifts are paired in the
fusion products, avoiding independent momentum square-root choices. No
double-Virasoro coefficient or auxiliary factor enters this recursion.

### Null-state signs in the chosen convention

The recursive global blocks use even intrinsic primaries. The human notes'
odd-null factorization first produces an intrinsically odd shifted primary.
Returning to the even-primary convention uses
`rho(p1,p2,p3)=(-1)^(p1*A+p2*C) rho(0,0,0)`. Combining these two steps gives
the incidence signs `1`, `(-1)^C`, and `-1` for a null in slots
`(infinity,1,0)`, respectively. The recursion also includes the ratio of
the parent and child graph permutation signs. The local form label and
the primary parity must not be conflated.

### Schottky seed and its moving-operator sign

`ns_schottky.hpp` enumerates primitive unoriented cycles of the actual ordered
plumbing graph and multiplies their projective plumbing matrices. It expands
the first fermionic oscillator (weight `3/2`) and the first bosonic oscillator
(weight `2`) in the same PBW sewing frame as the global vertices. The cycle's
fermion sign is fixed by the local supercurrent two-point forms and the graph
permutation, not by fitting a coefficient to the other method.

Through total level 6, only four primitive triangles and three primitive
four-edge cycles contribute. For a triangle, the fermionic leading monomial
is at level `9/2`, and the normalized inverse trace cubed supplies its level
`11/2` corrections. Its bosonic leading monomial is at level 6. Each
four-edge cycle first contributes a fermionic monomial at level 6. All higher
oscillators and products of two nonconstant primitive factors lie beyond
this cutoff. Thus the finite expansion is complete, including the
nonconstant Schottky seed; it is not a global-only approximation. This
isolated seed implementation explicitly restricts itself to Mercedes and
total level at most 6.

When multiplying vacuum parity `epsilon'` and global parity `epsilon`, the
sign is

```
(-1)^[K(epsilon+epsilon') - K(epsilon) - K(epsilon')
       + sum_v epsilon_(v,2) epsilon'_(v,3)].
```

The last term comes from moving the middle-slot odd global operators past
the ket-slot odd vacuum operators. The BPZ bra crossing cancels the bra
contribution. This is the same local moving-operator sign used in the
enlarged-block convolution. Keeping only the polarization of `K` misses this
term. It cancels between the two theta vertices, but survives on Mercedes.
The initial diagnostic omitted it and failed in 130 coefficients, beginning
at total level 5. Its output is retained with the prefix
`diagnostic_missing_local_seed_sign_`. The corrected calculation passes all
18,564 multidegrees. Neither the draft nor the human notes was edited.

## Runtime and saved results

The final fresh C++ run starts with empty in-process caches. Compilation is
excluded. Each stage retains and reuses the values it computes.

| Stage | Wall time (s) |
|---|---:|
| Double-Virasoro enlarged blocks, all eight sectors | 10.029 |
| Direct auxiliary fermion factor | 0.013 |
| Convolution recovery | 0.162 |
| NS c-recursion, coefficient comparisons, and output | 5.578 |
| Entire comparison | 15.783 |

The last stage includes validation and file-writing overhead, so its timing
is not a pure recursion-kernel benchmark. No conclusion about high-level
asymptotic speed follows from this low-level comparison.

- `results_L6.json`: errors by sector and stage timings.
- `coefficients_L6.jsonl`: recovered physical double-Virasoro coefficients.
- `c_recursion_L6.jsonl`: independently computed NS coefficients.
- `schottky_seed_L6.jsonl`: primitive cycles and their transported signs.
- `coverage.json`: saved-coefficient coverage and errors by total level.
- `manifest.json`: source and result hashes.
- `level6.log`: progress and error summary.

Reproduce from this directory with:

```sh
make
./compare 6 > level6.log 2>&1
python3 audit_saved.py
```

The level-3 directed intermediate comparison is also saved. It tests the
residue transport below the first nonconstant vacuum contribution. The
passed level-6 results are the requested final comparison.
