# Direct free-field matrix-element trial

This note corrects the status of equation (41) in
`r_branching_free_field.tex`.  That equation is a finite PBW/Ward
reconstruction of the three-form.  It is useful as an oracle, but it is not
the screening-charge derivation of the branching coefficient.

## The matrix element that must be evaluated

Keep the human-note order `(NS,R,R)` at `(infinity,1,0)`.  If
`W_n^epsilon(P)` denotes the explicit simultaneous `Vir x Vir` primary and
`V[W_n^epsilon(P)](1)` its chiral vertex operator, the raw branching form is

```text
M_f^eta = <W_(n1)(P1) | V[W_(n2)^epsilon2(P2)](1)
                         | W_(n3)^epsilon3(P3)>_f^eta.
```

The precise claim is

```text
B_f^eta = M_f^eta
          /(||W_(n1)|| ||W_(n2)^epsilon2|| ||W_(n3)^epsilon3||).
```

Let `g_(epsilon2,epsilon3;f)^eta` be the raw matrix of the two Ramond
ground spin fields in the selected component.  At a neutrality point with
`N` insertions of the `b`-screening current, let `I_N[F_desc]` be the
Coulomb integral after inserting all three double-Virasoro-primary strings,
and let `I_N[F_0]` be the same integral with the strings removed, but with
the same ground component.  Then

```text
M_f^eta |_N
  = g_(epsilon2,epsilon3;f)^eta * I_N[F_desc]/I_N[F_0]

  = g_(epsilon2,epsilon3;f)^eta
    * S_N[F_desc; -b alpha1-u, -b alpha2-v, g]
      /S_N[F_0; -b alpha1-u0, -b alpha2-v0, g],

g = -b(b+1/b)/2.
```

Thus the Selberg ratio computes the descendant matrix element relative to
the primary two-spin Coulomb integral.  Multiplication by the ground-spin
matrix restores the raw chiral three-form, and only division by the three
state norms produces the branching coefficient.  The pole-clearing
`delta_n(P)` or ell-product factors are not part of this matrix element;
they are introduced only to turn it into a polynomial for reconstruction.
Here `N` is the screening number (the number of integration variables), not
the branch level.

This claim concerns one scalar ground-resolved component.  A vector of
contour-orbit integrals and the corresponding projector may be introduced
to evaluate it efficiently, but they are auxiliary computational objects.
That evaluation problem is left to the later contour-recursion analysis.

In the free field, each `W` is the consecutive `chi=f-i psi` string, the
Ramond vertex contains a spin field, and a charge-neutral representative is

```text
<R_3| W_1^BPZ W_2(1) W_3(0)
       E_(alpha2)(1) Q_b^r Q_(1/b)^s |R_3>.
```

After separating the auxiliary and physical Majoranas, the fermionic part
is a zero-mode-resolved two-spin Pfaffian.  The bosonic part is the Coulomb
weight.  The quotient by the primary two-spin integral fixes the chosen
ground three-form.  This is the Ramond analogue of equations (3.12)--(3.14)
of Hadasz--Jaskolski.

## First crossed state

Take

```text
(n1,n2,n3)=(0,3/4,3/4), epsilon2=epsilon3=0.
```

Write `Q=b+1/b`.  At the two-`b`-screening neutrality plane

```text
P1=-Q/2-P2-P3-2b
```

the physical chiral form is `eta=-`.  With screening variables `t1,t2`, the
literal chi/spin-field calculation gives the denominator-cleared insertion

```text
F_2(t1,t2) = (1-i)/2 (2 e2-e1)
             (e2^2-e1 e2+e1^2-3 e2),
e1=t1+t2,  e2=t1*t2.
```

This is already a nontrivial result: no SCA three-form or PBW state appears.
Expanding in products of elementary symmetric functions and applying their
exact Selberg moments gives

```text
M_0^- = (-1+i) R_2,
M_1^- = -sqrt(2) i R_2,

R_2 = (13 b^4 + 10 b^3(P2+P3) + 4 b^2 P2 P3 + 8 b^2
       + 4 b(P2+P3) + 4)
      /[(b^2+2bP2+2)(b^2+2bP3+2)].
```

Independently organize the two Ramond boundary channels as

```text
E_j=Q+2P_j,
d_j=E_j^2+Q E_j+1,
L=(Q/2+P1+P2+P3)(-Q/2+P1-P2-P3),
H=L^2+2L(E2 E3+1)+d2*d3.
```

Direct symbolic reduction proves

```text
R_2 = [H/(d2*d3)]_(P1=-Q/2-P2-P3-2b).
```

Thus the irreducible crossed polynomial `H` is not merely a PBW artifact:
its restriction to this nontrivial screening hyperplane is obtained by an
explicit two-spin Pfaffian and Selberg integral.

At the natural three-screening plane

```text
P1=-Q/2-P2-P3-3b,
```

the genuine form is `eta=+`.  The determinant reduction gives

```text
M_0^+ = -4(1+i) R_3,
M_1^+ = -4 sqrt(2) R_3,

R_3 = b^2(3b^2-1)
      (2b^2+b(P2+P3)+1)(5b^2+2b(P2+P3)+1)
      /[(b^2+2bP2+2)(b^2+2bP3+2)
        (2b^2+2bP2+1)(2b^2+2bP3+1)].
```

If

```text
K=(x_++^2+Q x_+++1)(x_2^2+Q x_2+1),
x_++=Q/2+P1+P2+P3,
x_2 =Q/2-P1+P2+P3,
```

then the second exact identity is

```text
4 R_3 = [K/(d2*d3)]_(P1=-Q/2-P2-P3-3b).
```

Since the selected ground amplitudes are `g_00;0^- = 1-i` and
`g_00;0^+ = 1+i`, the two computations are equivalently the normalized
Selberg statements

```text
I_2^-[F_desc]/I_2^-[F_0] = -H/(d2*d3),
I_3^+[F_desc]/I_3^+[F_0] = -K/(d2*d3).
```

These are the crossed and factorized Coulomb ratios, respectively.  The raw
matrix element is obtained by restoring `g`, and the branching coefficient
is obtained only after the additional division by the three state norms.

## What this establishes

The direct free-field calculation now supplies two nontrivial exact
screening restrictions:

```text
N=2, eta=-  -> crossed H channel,
N=3, eta=+  -> factorized K channel.
```

This is stronger than the former PBW path-sum definition, but it is not yet
a generic-momentum formula.  To reconstruct each degree-four hard master
from one charge parity requires five same-parity screening nodes.  The next
even integrand (`N=4`) is explicitly computable (381 monomials, degree five
in each screening variable), but its generic symbolic Selberg reduction
still needs a more efficient symmetric-function implementation.  At general
branch level the complementary reflected vertex must also include the
bosonic-current terms generated by the Ramond reflection intertwiner.

Run the exact trial with

```sh
env PYTHONDONTWRITEBYTECODE=1 python3 \
  'Machine Notes/R branching coefficient/Code/direct_matrix_element_trial.py'
```
