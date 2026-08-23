# Contour and collision recurrences for the Ramond screening integral

## What the literature actually does

There are three logically distinct steps in the Selberg/parafermionic
derivation.

1.  Selberg's expansion method bounds the exponent support of the symmetric
    polynomial.  Those bounds determine every `A`- and `B`-dependent Gamma
    factor up to a function `C_m(g)`.
2.  A boundary-cluster residue lowers the number of screenings.  In the BFL
    notation, `N` screening coordinates collapse at an endpoint and give
    their equations (3.29)--(3.35), which determine `C_m(g)` recursively.
    For the superconformal `Z_2` case the cluster contains two screenings.
3.  A Dotsenko--Fateev contour deformation factorizes the complex-plane
    integral into real integrals on `[0,1]` and `[1,infinity)`.  The sine
    prefactor is insensitive to an inserted *single-valued polynomial*.

Thus the contour deformation is not the step that evaluates the unknown
real Selberg average.  It supplies analytic continuation and the connection
between chambers.  Polynomial support plus the collision residue perform
the actual evaluation.

## Scalar Dotsenko--Fateev chamber recurrence

For

```text
S_(N,p)(alpha,beta,g)
  = integral over [0,1]^p [1,infinity]^(N-p)
      product_i t_i^(alpha-1) |1-t_i|^(beta-1)
      product_(i<j) |t_i-t_j|^(2g),
```

deforming one `[0,1]` contour into a positively oriented indented
semicircle gives

```text
S_(N,p) = r_(N,p) S_(N,p-1),

r_(N,p) = p/(N-p+1)
  * sin(pi (N-p+1)g)
  * sin(pi (alpha+beta+(N+p-2)g))
  / (sin(pi p g) sin(pi (alpha+(p-1)g))).
```

The sine factors are discontinuities of the loaded contour.  In modern
language the integration chambers are twisted cycles.  Resonance makes the
map from compact to locally finite twisted homology non-injective, and the
boundary collision is the corresponding divisor contribution.

## Exact two-component Ramond collision

Let

```text
P_N^+(t) = 2^(-floor(N/2)) Delta(t)
           Pf((t_i+t_j)/(t_i-t_j))
```

with the standard bordered Pfaffian for odd `N`, and set

```text
P_N^-(t) = P_N^+(1-t).
```

The polynomial vector has the exact collision law

```text
(P_N^+, P_N^-)^T at (t_1,t_2)=(x,x)
 = product_(j=3)^N (x-t_j)^2
   diag(x,1-x)
   (P_(N-2)^+,P_(N-2)^-)^T.
```

For the vector Selberg integral

```text
I_N(A,B;g)
 = integral_[0,1]^N product_i t_i^A(1-t_i)^B
   product_(i<j)|t_i-t_j|^(2g) (P_N^+,P_N^-)^T,
```

this gives complementary endpoint residues

```text
lim_(A -> -g-1) (A+g+1) I_N(A,B;g)
 = kappa_N(g) Pi_0 I_(N-2)(1+3g,B;g),

lim_(B -> -g-1) (B+g+1) I_N(A,B;g)
 = kappa_N(g) Pi_1 I_(N-2)(A,1+3g;g),

Pi_0 = diag(0,1),       Pi_1 = diag(1,0),

kappa_N(g) = N(N-1)/2
  * Gamma(-g) Gamma(1+2g)/Gamma(1+g).
```

At `N=2`,

```text
I_+/S_2 = (A+1+g)/(A+B+2+2g),
I_-/S_2 = (B+1+g)/(A+B+2+2g),
```

so the projectors are immediate.  The naive radial candidate
`A=-g-3/2` for `P_N^+` is canceled by the angular factor
`Beta(g+1/2,-g-1/2)=0`; it is not a pole of the meromorphically continued
integral.

## Consequence for the branching coefficient

The rationalized primary two-spin Pfaffian is a single-valued polynomial.
Therefore its Dotsenko--Fateev sine factor is scalar and cannot by itself
produce the crossed numerator `H` or the unknown conformal factor `H/K`.
Those data live in the inserted descendant polynomial.

The useful generalization is a vector of descendant insertions

```text
F_N = (F_N^factorized,F_N^crossed)^T.
```

To repeat the BFL proof one needs three finite statements about `F_N`:

1. exponent-support bounds, to determine all `A,B` Gamma factors;
2. a two-screen collision matrix, to recurse in `N`;
3. an inversion/crossing law

   product_i t_i^d F_N(1/t_i) = U_N F_N(t_i),

   to close the `[1,infinity)` chamber after contour deformation.

For the primary spin polynomial, `U_N` is just the known permutation of
parafermionic labels.  For the double-Virasoro primary it is the full
reflected Ramond vertex, not merely a ground-space `Z` matrix.  The first
hard PBW result

```text
diag(1,H/K) * 1/2 [[-3,i],[-i,-3]]
```

is the first nontrivial value of precisely this connection problem.

## First crossed polynomial: finite contour orbit

The denominator-cleared, phase-free two-screening crossed insertion is

```text
F_bulk = (2 e_2-e_1)
  (e_2^2-e_1 e_2+e_1^2-3e_2).
```

Let

```text
R F(t_1,t_2) = F(1-t_1,1-t_2),
I F(t_1,t_2) = (t_1 t_2)^3 F(1/t_1,1/t_2).
```

The exact orbit contains three independent polynomials

```text
F_bulk,
F_zero = I F_bulk,
F_one  = R F_zero.
```

Their transformation law is

```text
I: F_bulk <-> F_zero,  F_one fixed,
R: F_zero <-> F_one,   F_bulk fixed.
```

Hence `I` and `R` generate the permutation representation of `S_3`.  The
endpoint radial orders in the displayed basis are

```text
at zero: (3,0,3),
at one:  (3,3,0).
```

Only `F_zero` contributes to the ordinary zero-endpoint Selberg residue and
only `F_one` to the one-endpoint residue.  `F_bulk`, whose integral gives the
crossed numerator `H`, is invisible to both of those residues.  This
explains why the scalar collision recursion alone cannot determine `H/K`.

The permutation module splits as `1 + 2`.  On the sum-zero basis

```text
u = F_bulk-F_zero,
v = F_zero-F_one,
```

the generators are

```text
I = [[-1,1],[0,1]],
R = [[1,0],[1,-1]].
```

They obey `I^2=R^2=(IR)^3=1`.  It is natural to conjecture that the physical
two-dimensional Ramond three-form is this standard doublet and that the
symmetric singlet is removed by the physical projection.  The polynomial
orbit proves the `S_3` structure; the identification of the physical
projection remains to be checked against the two zero-mode copies.

There is already a nontrivial match with the PBW transfer.  Conjugating the
doublet inversion matrix by

```text
S = [[1,0],[i,-i]]
```

gives the reflection in the physical `eta=(+,-)` frame,

```text
J_eta = S I S^(-1) = [[0,i],[-i,0]].
```

The momentum-independent matrix extracted independently from the ground to
hard PBW transfer is exactly

```text
C_PBW = 1/2 [[-3,i],[-i,-3]]
      = -3/2 identity + 1/2 J_eta.
```

Thus the universal copy mixing is an affine group-algebra element of the
same contour transposition.  This is strong evidence for the doublet
identification, although it does not by itself prove which singlet
combination is removed at every level.

For the actual two-screening Selberg integrals, put

```text
J = (J_bulk,J_zero,J_one)^T,
P_I = [[0,1,0],[1,0,0],[0,0,1]],
A_dual = -A-B-2g-5.
```

Direct evaluation of all three polynomial averages proves the exact vector
functional equation

```text
J(A,B;g) = R_2(A,B;g) P_I J(A_dual,B;g),

R_2(A,B;g)
 = sin(pi(A+B+2+g)) sin(pi(A+B+2+2g))
   / (sin(pi(A+1)) sin(pi(A+1+g))).
```

The shift by `-3` in `A_dual` relative to the ordinary Selberg transform is
the reciprocal degree of every orbit polynomial in each variable.  The
executable check reduces the quotient of the two sides to

```text
[A sin(pi A) Gamma(-A) Gamma(A)]
[(A+g) sin(pi(A+g)) Gamma(-A-g) Gamma(A+g)]
/
[(A+B+g) sin(pi(A+B+g)) Gamma(-A-B-g) Gamma(A+B+g)]
[(A+B+2g) sin(pi(A+B+2g)) Gamma(-A-B-2g) Gamma(A+B+2g)]
= 1
```

by Euler reflection.  Thus the contour connection is not merely a proposed
matrix pattern at this level; it is an exact identity of the integrated
functions.

The crossed normalized average itself is

```text
J_bulk/S_2 = -2
 (A+g+1)(A+g+2)(B+g+1)(B+g+2)
 [A B+4A g+3A+4B g+3B+4g^2+18g+9]
 /
 [(A+B+g+2)(A+B+g+3)(A+B+g+4)
  (A+B+2g+2)(A+B+2g+3)(A+B+2g+4)].
```

This is the generic `A,B,g` function whose Liouville specialization on the
two-screening neutrality plane gives the previously checked crossed factor
`H/(d_2 d_3)`.

At the adjacent natural three-screening node the native calculation
simplifies completely:

```text
F_3^(f=0,eta=+) = -sqrt(2)(1+i)/4 * Delta_3^2.
```

The odd-form result differs only by its fixed ground phase.  Since
`Delta_3^2` is invariant under both endpoint reflection and reciprocal
inversion

```text
(product_i t_i^4) Delta_3(1/t)^2 = Delta_3(t)^2,
```

this node is the `S_3` singlet and its integral is an ordinary Selberg
product with coupling `g+1`.  This explains structurally why the natural
`N=3` neutrality plane gives the factorized numerator `K`, whereas the
even `N=2` plane detects the nontrivial crossed orbit and produces `H`.

## Practical next test

The residue pair written above is not, by itself, a recurrence determining
the full function.  Each map has rank one.  In the scalar Selberg problem
this issue is absent because the collision closes on a one-dimensional
family: support bounds fix all Gamma factors, the residue fixes the one
remaining constant, and `S_0=1` supplies the base value.  Here uniqueness
requires a finite-dimensional replacement of that argument.

For each screening number, the next task is therefore to:

```text
1. determine the minimal contour-orbit space V_N of descendant polynomials;
2. prove degree and pole bounds for the normalized integral on V_N;
3. compute all independent cluster maps at 0, 1, and infinity;
4. compute the inversion/crossing connection matrix U_N;
5. check that these maps have no common invisible subspace;
6. fix the solution from the N=0 and N=1 base vectors.
```

The first crossed `N=2` insertion already has a three-dimensional contour
orbit (`F_bulk,F_zero,F_one`), even though the physical chiral-form matrix
has two rows.  Thus one must determine `dim V_N`; it should not be assumed
to be two.  If the combined residue and connection data have full rank on
`V_N`, the BFL uniqueness argument becomes a finite matrix Gamma-product
recurrence.  If they retain a common kernel, another boundary functional or
asymptotic condition is required.

This solves only the integral at screening neutrality.  The generic
branching kernel is a second reconstruction problem: after pole clearing,
its bounded-degree dependence on Liouville momentum is fixed from enough
same-parity screening nodes by finite interpolation.
