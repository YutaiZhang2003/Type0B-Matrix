# Ramond physical-matrix recurrence trial

## Definition before any recurrence

The branching coefficient itself is not the object on which the contour
recursion acts.  For each fixed ground-resolved component define the raw
double-Virasoro-primary matrix element

```text
M_(epsilon2,epsilon3;f)^eta
  = <W_(n1)(P1) | V_f^eta[W_(n2)^epsilon2(P2)](1)
                       | W_(n3)^epsilon3(P3)>.
```

If `N_i=||W_(ni)||^2`, then

```text
B_(epsilon2,epsilon3;f)^eta
  = M_(epsilon2,epsilon3;f)^eta
    /(||W_(n1)|| ||W_(n2)^epsilon2|| ||W_(n3)^epsilon3||),

(B_(epsilon2,epsilon3;f)^eta)^2
  = (M_(epsilon2,epsilon3;f)^eta)^2/(N_1 N_2 N_3).
```

At an `N`-screening neutrality point, the free-field claim is instead a
statement about the raw entry:

```text
M_(epsilon2,epsilon3;f)^eta |_N
  = g_(epsilon2,epsilon3;f)^eta * J_N[F_desc]/J_N[F_0],

J_N[F] = S_N[F; -b alpha1-u, -b alpha2-v, -b(b+1/b)/2].
```

Here `F_desc` is obtained by inserting the explicit three branch-state
strings into the two-spin Pfaffian, `F_0` is the primary two-spin
polynomial in the same ground component, and `g` is the raw ground-spin
matrix entry.  Thus the normalized Selberg average `J_N[F_desc]/J_N[F_0]`
is the descendant-to-primary Coulomb matrix element.  The ell products used
to clear poles and the three state norms are applied only afterwards.

For the checked `f=0`, `epsilon2=epsilon3=0` channels this identification is
especially transparent:

```text
J_2^-[F_desc]/J_2^-[F_0] = -H/(d_2 d_3),
J_3^+[F_desc]/J_3^+[F_0] = -K/(d_2 d_3).
```

The first line is the crossed Selberg channel and the second the factorized
one.  Neither is by itself the final branching coefficient.  A contour
orbit, a vector of integrals, or a projector belongs only to an efficient
algorithm for computing these scalar ratios; that algorithm is discussed
below, separately from this claim.

## Physical matrix packaging

For fixed `epsilon_3=f=0`, put the two physical NS--R--R chiral forms in
the rows and the two second-leg Ramond multiplicity copies in the columns:

```text
M(n_1,n_2,n_3) =
    [[M_(epsilon_2=0)^(eta=+), M_(epsilon_2=1)^(eta=+)],
     [M_(epsilon_2=0)^(eta=-), M_(epsilon_2=1)^(eta=-)]].
```

These are raw matrix elements.  A normalized branching coefficient is
obtained only afterwards as specified above.

At the Ramond ground point `(0,1/4,1/4)`,

```text
M_ground = [[1+i, -(1-i)/sqrt(2)],
            [1-i, -(1+i)/sqrt(2)]],

det(M_ground) = -2 sqrt(2) i.
```

Thus the two multiplicity copies really do supply two independent columns.

## First exact transfer

At `(0,3/4,3/4)`, let `K` be the factorized four-ell numerator, `H` the
irreducible crossed numerator, and `d_2 d_3` the two raw leg denominators.
The independently checked PBW matrix is

```text
M_hard = [[R_0^+,  i sqrt(2) R_0^+],
          [R_0^-, -i sqrt(2) R_0^-]],

R_0^+ = -(1+i) K/(d_2 d_3),
R_0^- = -(1-i) H/(d_2 d_3).
```

It follows exactly that

```text
M_hard = T_(1/4 -> 3/4) M_ground,

T_(1/4 -> 3/4)
  = 1/(2 d_2 d_3) [[-3K,  iK],
                    [-iH, -3H]]

  = K/(d_2 d_3) diag(1,H/K)
      * 1/2 [[-3,i],[-i,-3]].
```

The last constant matrix has eigenvalues `-1,-2`.  More precisely,

```text
C M_ground = M_ground diag(-1,-2),
C = 1/2 [[-3,i],[-i,-3]].
```

Thus the two ground multiplicity-copy columns are exactly the two
eigenvectors `(1,-i)` and `(1,i)` of the momentum-independent mixing.  This
is why both columns are needed: retaining only one copy projects out one
eigenchannel before the crossed numerator can be detected.

The transfer separates three
pieces which should not be conflated:

1. the scalar Hadasz--Jaskolski/factorized step `K/(d_2 d_3)`;
2. the genuinely Ramond crossed correction `H/K`;
3. a momentum-independent change between the multiplicity and chiral-form
   frames.

This is an exact first-step identity, but not yet an all-level recurrence:
the displayed transfer was reconstructed from the independently known
ground and hard matrices.

## Ramond collision recursion

For `N=2m` screenings define the radical-free two-spin polynomial

```text
P_hat_(2m)(t)
  = 2^(-m) Delta(t) Pf((t_i+t_j)/(t_i-t_j)).
```

The leading two-screening collision obeys

```text
P_hat_(2m)(x,x,t_3,...,t_(2m))
  = x product_(k=3)^(2m) (x-t_k)^2
      P_hat_(2m-2)(t_3,...,t_(2m)).
```

The endpoint-reflected partner

```text
P_hat_N^vee(t_1,...,t_N) = P_hat_N(1-t_1,...,1-t_N)
```

obeys the complementary rule

```text
P_hat_N^vee(x,x,t_3,...,t_N)
  = (1-x) product_(k=3)^N (x-t_k)^2
      P_hat_(N-2)^vee(t_3,...,t_N).
```

The bordered odd-screening Pfaffian obeys the same pair of identities, with
its normalization chosen so that `P_hat_1=1`.  The executable audit proves
the first nontrivial `N=4 -> N=2` and `N=3 -> N=1` instances.  In general
the identity follows from Pfaffian expansion: at `t_1=t_2`, only the
singular pairing of the colliding variables survives after multiplication
by the Vandermonde.

For a Selberg weight

```text
product_i t_i^A (1-t_i)^B |Delta|^(2g),
```

put

```text
I_N = integral Selberg_weight * (P_hat_N, P_hat_N^vee)^T.
```

At the zero endpoint the genuine Selberg pole is `A=-g-1`.  The first
component vanishes there and the second survives; at the one endpoint
`B=-g-1` the roles reverse.  More precisely,

```text
lim_(A -> -g-1) (A+g+1) I_N(A,B;g)
  = kappa_N(g) diag(0,1) I_(N-2)(1+3g,B;g),

lim_(B -> -g-1) (B+g+1) I_N(A,B;g)
  = kappa_N(g) diag(1,0) I_(N-2)(A,1+3g;g),

kappa_N(g)
  = N(N-1)/2 * Gamma(-g) Gamma(1+2g)/Gamma(1+g).
```

The projectors are already visible at `N=2`.  The normalized averages are

```text
I_+(A,B)/S_2 = (A+1+g)/(A+B+2+2g),
I_-(A,B)/S_2 = (B+1+g)/(A+B+2+2g),
```

and they sum to one.  At `A=-g-1` these ratios become `(0,1)`; at
`B=-g-1` they become `(1,0)`.

There is a tempting but incorrect extra pole.  Since `P_hat_N` contributes
one radial power when two variables approach zero, raw power counting gives
`tau^(2A+2g+2)` and suggests `A=-g-3/2`.  Its angular coefficient is
proportional to

```text
Beta(g+1/2,-g-1/2) = 0
```

by analytic continuation (`1/Gamma(0)=0`).  Therefore the would-be pole is
absent.  The actual meromorphic recurrence is the rank-one pair above, not
a half-shifted scalar recurrence.

## What remains to obtain generic labels

The useful final target is a recurrence for the physical matrix itself,

```text
M(n_1,n_2+1/2,n_3+1/2)
  = T_23(n_i;P_i,b) M(n_1,n_2,n_3),
```

derived from a degenerate insertion or from screening collision/loop
equations.  The presently proved endpoint collision equations do not yet
give it: their projectors are rank one, and a single residue discards a
component.  This is harmless in an ordinary scalar Selberg family but not
for the Ramond descendant integral.

The complete construction must determine the minimal contour-orbit space
`V_N`, its degree and pole bounds, all independent endpoint/cluster maps,
and the inversion/crossing matrix.  Together these data must have full rank
on `V_N`; otherwise a common kernel remains invisible to the residues.  The
base vectors at `N=0,1` then fix the overall normalization.  Only after this
vector Selberg problem is solved does one use same-parity screening nodes
to interpolate the pole-cleared matrix element to generic Liouville
momenta.

The scalar eigenchannel is already controlled by the ell/GKO recurrence.
The missing connection data must retain the reflected Ramond current piece;
otherwise the affine arrows see only the factorized channel.  The physical
matrix is obtained after solving the auxiliary contour-orbit problem; the
projectors are not extra branching coefficients.

Run the certificate with

```bash
PYTHONPATH='python 2' PYTHONDONTWRITEBYTECODE=1 \
python3 -m ramond_screening_algorithm.physical_matrix_recurrence_trial
```
