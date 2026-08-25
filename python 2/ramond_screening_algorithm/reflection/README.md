# Reflected Ramond fermion without constructing `W_n`

This directory implements the free-field reflection map itself.  It is the
missing local ingredient when one Ramond branch is put on the reflected
`P -> -P` sheet.  The result is not a scalar replacement for the ordinary
fermion propagator: already at mode one it mixes a fermion with a bosonic
current and flips the Ramond ground index.

## Defining equations

Hadasz--Jaskolski, *Super-Liouville -- double Liouville correspondence*,
[arXiv:1312.4520](https://arxiv.org/abs/1312.4520), define the reflection
map in equations (2.17)--(2.19):

```text
L_m(-P) r(P) = r(P) L_m(P),
G_m(-P) r(P) = r(P) G_m(P),

c_m^R = r(P)^(-1) c_m r(P),
psi_m^R = r(P)^(-1) psi_m r(P).
```

Their equation (2.20) writes an oscillator vector in either free-field
chart.  The usual finite-level implementation of that equation is
`S_-(P) S_+(P)^(-1)`, but it constructs the whole super-Virasoro transition
matrix.

The direct recurrence here uses only the first line.  Let `A_s(X_n;l)` be
the positive-mode Fock matrix from level `l` to level `l-n`, where `s=+/-`
is the free-field realization.  If `R_l` maps the plus chart to the minus
chart, then

```text
A_-(X_n;l) R_l = R_(l-n) A_+(X_n;l),  X=L,G, n=1,...,l.       (1)
```

`R_0=1` on each Ramond ground.  Stack every equation (1).  At generic
momentum the stacked left matrix has full column rank, so any maximal
independent row set is square and determines `R_l` uniquely.  This is a
finite-level version of the oscillator recursion in Zhu--Matsuo,
*Yangian associated with 2D N=1 SCFT*,
[arXiv:1504.04150](https://arxiv.org/abs/1504.04150): their equations
(41)--(43) define the two free-field realizations and their equation (45)
is the inverse-momentum reflection recurrence.  Equation (1) solves the
same intertwining problem at fixed level and finite momentum, without an
inverse-momentum truncation.

Positive modes are obtained by BPZ from sparse negative-mode actions.  In
the conventions of this repository the 2013 Hermitian momentum is `p=iP`;
therefore BPZ sends

```text
i -> -i,  P -> -P,  Q -> Q.
```

Omitting `P -> -P` gives the wrong Ramond Kac denominator.

## First nontrivial reflected kernel

Order the level-one Fock basis as

```text
c_-1|0>, c_-1|1>, psi_-1|0>, psi_-1|1>.
```

The recurrence gives

```text
R_1 = [[a,0,0,b],
       [0,a,b,0],
       [0,b,c,0],
       [b,0,0,c]],

d = 4 P^2 - 6 P Q + 2 Q^2 + 1,
a = -(4 P^2 - 2 P Q - 2 Q^2 + 1)/d,
b = 2 sqrt(2) i Q/d,
c = -(4 P^2 + 2 P Q - 2 Q^2 + 1)/d.
```

For example,

```text
R_1 psi_-1|0>_+ = c psi_-1|0>_- + b c_-1|1>_-.
```

This explains the failed scalar-Pfaffian attempts.  Against a vertex and
screening fields, the `c_-1` term is evaluated by the ordinary Heisenberg
Wick rule and depends on the other external momenta and on all screening
coordinates.  Thus the effective reflected fermion covariance is a matrix
in the two Ramond ground indices and contains bosonic-current insertions.
Replacing one scalar external covariance cannot reproduce it.

The next mode shows the growing current content explicitly.  With

```text
d1 = 4 P^2 - 6 P Q + 2 Q^2 + 1,
d2 = 4 P^2 - 10 P Q + 4 Q^2 + 9,
```

the exact level-two recurrence gives, for either ground index `g`,

```text
R_2 psi_-2|g>_+ = A psi_-2|g>_- + B c_-1 psi_-1|g>_-
                  + C c_-1^2|1-g>_- + D c_-2|1-g>_-,

A = -(16 P^4 - 44 P^2 Q^2 + 40 P^2 + 36 P Q^3 + 16 P Q
      - 8 Q^4 - 18 Q^2 + 9)/(d1 d2),
B = 12 i Q (4 P^2 - 2 P Q + 1)/(d1 d2),
C = 4 sqrt(2) Q (4 P + Q)/(d1 d2),
D = 4 sqrt(2) i Q (4 P^2 - 4 P Q + Q^2 + 3)/(d1 d2).
```

Thus the first correction relevant to the longer `W_5/4` strings already
requires one-current/one-fermion and two-current contractions.  A modified
fermion pair kernel cannot absorb these terms either.

The same rule works for any number of screenings: first apply `R_l` to the
reflected external Fock endpoint, then contract every resulting bosonic and
fermionic oscillator with the appropriate free spin-field vertex,
exponential, and screening fields.  The required spin-field-to-SCblock
chiral-structure map is a separate trivalent datum; it is not supplied by
the one-leg reflection recurrence.
The reflection block is local to the external leg and is independent of the
number and positions of the screenings.

## Executable interface and basis order

The public finite-field entry points are

```text
reflection_blocks_mod(maximum_level, Q, P, sqrt_minus_one, sqrt_two, prime)
reflect_fock_expression_mod(expression, blocks, prime)
```

`R_l` maps coefficients in the `+` realization to coefficients in the `-`
realization.  A sparse free-Fock vector is a dictionary

```text
(boson_partition, strict_fermion_partition, ground_index) -> coefficient.
```

Both partitions contain positive mode numbers in decreasing order, and
`ground_index` is `0` or `1`.  The conserved block parity is
`(number_of_fermions + ground_index) mod 2`.  The matrix order returned by
`reflection_blocks_mod` is exactly
`parity_basis(level, parity)`: increasing total fermion grade, then the
repository's decreasing-partition order for the bosons and strict
fermions, with ground indices in the order `0,1` before the parity filter.
`reflect_fock_expression_mod` performs the grouping and matrix application
and returns a sparse expression in the same state format.  There is no
super-Virasoro/PBW object at this interface.

## Complete finite algorithm for one branching coefficient

At fixed branch labels the exact algorithm is the following.

1. Read each ordered 2016 chi chain.  For a Ramond label `n`, it is the
   consecutive string `0,-1,...,-M`, where `M=2|n|-1/2`, followed, when
   required, by the opposite zero mode that selects the other parity copy.
   Keep the auxiliary-fermion and physical-fermion choices in an exterior
   generating vector; literal binary expansion is needed only for a small
   audit.
2. On every reflected physical leg, group the physical endpoints by Fock
   level and total parity and apply `R_l` with
   `reflect_fock_expression_mod`.  Ordinary legs are left unchanged.
3. Every output term is now an ordinary-chart product of bosonic modes,
   physical fermion modes, an auxiliary fermion string, and one of the two
   Ramond ground indices.  Use the native ground-resolved kernel of the
   chosen Coulomb chart; the two charges `Q/2+P2` and `Q/2-P2` give the two
   chiral structures.  Reusing one ordinary Majorana covariance while
   merely changing an SCblock ground matrix is not valid.  Commute each
   bosonic mode through the vertex exponentials
   and screening currents by the Heisenberg relation; this turns it into an
   explicit power sum of external points and screening coordinates.  This
   is where the `b c_-1|1-g>` part of `R_1` enters.
4. Multiply by the ordinary Coulomb-gas weight, expand the resulting
   symmetric Laurent polynomial in the finite Schur/Jack basis, and apply
   the exact Selberg functional term by term.  Sum the finitely many Ramond
   ground sectors with the repository's Fock-to-SCblock and radial-order
   Koszul phases.
5. Divide the result by the primary three-point form and by the directly
   normalized branch-state norms.  Repeat on enough exact charge-neutral
   hyperplanes and reconstruct the generic momentum polynomial using its
   Ward degree bound.  A final unused prime checks the CRT/rational
   reconstruction.

`mixed_sheet_mod.py` implements steps 1--3 on the zero-screening chart.  On

```text
Q/2 + P1 + P2 + P3 = 0
```

the native form is `f=0, eta=-`, and

```text
c_-m -> i (Q/2 + P2).
```

Its physical Fock ground matrix is `diag(1,i)` and must be inserted before
an unpaired fermion flips a ground index.  Conditional on the second ground
`g2`, an even physical Wick pair gets local phases `i(2*g2-1)` at one and
`1` at zero.  This is sufficient to reproduce the irreducible hard
polynomial `H` without using the SCA Ward evaluator on the calculated side.
The complementary `Q/2-P2` chart is a separate Coulomb intertwiner, not an
ad hoc change of `eta` in this covariance.

At `W_5/4` a second, sharply localized issue appears.  The native auxiliary
Majorana spin-OPE kernel differs from the Virasoro-primary sewing convention
on some raw mode-string labels; for example,

```text
psi_-1/2 at infinity:  -i/sqrt(2) versus +i/sqrt(2),
psi_-2 psi_-1 on one leg:  +1/32  versus -1/32.
```

The production `fermion_value_virasoro` has separately been repaired to
retain negative-mode commutators and descendant levels; its old
`-361/128` value for the crossed mode-two pair is now `-425/128`.  The
executable uses this corrected sewing functional as an exact auxiliary
fallback.  It is not yet the requested pure Pfaffian/Selberg production
algorithm because the explicit local-coordinate transport from the native
spin-OPE kernel to the sewing convention has not been supplied.  For
nonzero screenings the already implemented Heisenberg map
is

```text
c_-m -> i (Q/2 + P2 + b sum_j t_j^(-m));
```

its symmetric Laurent output still has to be routed through the bounded-
width Schur/Selberg extractor.

## Exactness, checks, and cost

Run

```bash
python3 -m python.ramond_screening_algorithm.reflection.intertwiner_recurrence
python3 -m python.ramond_screening_algorithm.reflection.audit_hard_channel
python3 -m python.ramond_screening_algorithm.reflection.mixed_sheet_mod
```

The checks establish:

- the symbolic level-one recurrence equals `S_-(P) S_+(P)^(-1)` entry by
  entry;
- finite-field blocks through level three equal the transition result as
  complete matrices, so every low-grid endpoint through `W_5/4` is covered;
- after applying `R_1`, the independent Ward contraction for the mixed-sheet
  `(0,3/4,-3/4)` channel agrees in 16 exact restrictions at the two stored
  samples, including eight restrictions containing the irreducible crossed
  polynomial `H` (this is not a screening-backend check);
- the direct reflection--Heisenberg--physical-Pfaffian calculation reproduces
  the native `f=0, eta=-` hard form for all four Ramond-copy choices at both
  stored samples; its calculated side does not call the SCA Ward evaluator;
- with the explicitly labelled auxiliary Virasoro fallback, the same driver
  reproduces all four copy choices at both stored samples through `W_5/4`;
- all four `W_7/4` copies run through the complete level-six reflected
  endpoint over one prime; there is no stored Ward value at that level, so
  this last item is a benchmark rather than a claimed comparison;
- all blocks through the complete `W_7/4` endpoint level six take about
  `0.04 s` at one prime (parity-block dimensions `40,40`); through level ten
  they take about `0.8--1.1 s` (dimensions `232,232`) on the profiling
  machine.

All high-level arithmetic is over `GF(p)`.  Repeating at primes
`p=1 mod 8`, separating the four choices of `sqrt(-1)` and `sqrt(2)`, and
CRT/rationally reconstructing only the final scalar is exact; no floating
point accuracy is lost.

There is an honest limitation.  If `d_l` is one Ramond parity-block
dimension, the explicit recurrence costs `O(d_l^3)` time and `O(d_l^2)`
memory.  The Ramond character gives `d_l=exp(O(sqrt(l)))`; since the branch
endpoint level is
`l_max=M(M+1)/2=O(n^2)`, an explicitly materialized reflection matrix takes
`exp(O(|n|))` time and memory up to polynomial powers.  A literal chi-path
expansion also has `2^(M+O(1))=exp(O(|n|))` terms.  It is therefore a fast
exact audit/fallback, not the final asymptotic production algorithm.

No polynomial shortcut around the explicit `R_l` is established here.  The
2013 equations quoted above intertwine the two free-field realizations of a
module; they do not by themselves give the action of reflection on the full
two-ground-index Ramond chiral vertex.  In particular, simply moving
`r(P)` through the vertex and cancelling a scalar primary reflection
amplitude would reduce the mixed-sheet calculation to the ordinary
covariance, which is known to miss the momentum-dependent polynomial `H`.
A chart-changing algorithm would therefore need a separately derived
matrix-valued vertex reflection law and must pass that hard channel before
it can replace the explicit recurrence.  At present this remains an open
acceleration, not part of the claimed result.
