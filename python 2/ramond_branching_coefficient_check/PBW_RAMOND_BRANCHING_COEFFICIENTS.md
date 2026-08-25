# Ramond branching coefficients from explicit PBW primaries

## Conventions and scope

This note uses the ordered trinion `(NS at infinity, R at one, R at zero)`
and the ground convention

```text
G_0 w^+ = i beta exp(-i pi/4) w^-,
G_0 w^- = i beta exp(+i pi/4) w^+.
```

The Ramond branch labels are

```text
n in Z/2 + 1/4,       epsilon=0,1,
```

where `epsilon` distinguishes the two copies of the same
`Vir x Vir` module.  The root-independent coefficient is

```text
(B_f^(eta))^2
  = rhohat_f^(eta)(v_1,W_2^epsilon2,W_3^epsilon3)^2
    /(N_1 N_2^epsilon2 N_3^epsilon3).
```

All displayed closed low-level formulas have an even NS state in the first
slot.  For an intrinsically odd NS state the generalized Ward identities use
`eta_eff=(-1)^p_phi eta`; the old fixed-even Ward implementation must not be
used without this replacement.

## Explicit low Ramond primaries

Write

```text
|a,g> = |a>_F x |P,g>_R,       a,g in {0,1},
D_sigma = 4P^2 + 6 sigma P Q + 2Q^2 + 1,
X_sigma = Q + 2 sigma P.
```

All `L` and `G` modes below act on the physical Ramond factor.  In the raw
2016 chi-string normalization, the onset states are

```text
W_(sigma/4)^0 = |0,0> + i sigma |1,1>,

W_(sigma/4)^1 = (|1,0> - i sigma |0,1>)/sqrt(2).
```

The first excited primaries are

```text
W_(3 sigma/4)^0
  = -f_-1 |1,0>/sqrt(2)
    +i sigma f_-1 |0,1>/sqrt(2)
    +2 L_-1 |0,0>/D_sigma
    -2 i sigma L_-1 |1,1>/D_sigma
    +sqrt(2) X_sigma G_-1 |1,0>/D_sigma
    +i sigma sqrt(2) X_sigma G_-1 |0,1>/D_sigma,

W_(3 sigma/4)^1
  = -f_-1 |0,0>
    -i sigma f_-1 |1,1>
    +2 sqrt(2) i sigma L_-1 |0,1>/D_sigma
    +2 sqrt(2) L_-1 |1,0>/D_sigma
    -2 X_sigma G_-1 |0,0>/D_sigma
    +2 i sigma X_sigma G_-1 |1,1>/D_sigma.
```

These expressions are simultaneous highest vectors of the two embedded
Virasoro algebras.  They contain two and six nonzero PBW components,
respectively.  At `|n|=5/4` the corresponding raw state has 30 nonzero
components per parity copy.

For positive `n`, put `M=2n-1/2`.  The low-level norm calculation gives

```text
N_n^0 =  2^(2 floor(M/2)+1)
         ell(2P,4n)/ell(Q+2P,4n),

N_n^1 = -2^(2 floor((M+1)/2))
         ell(2P,4n)/ell(Q+2P,4n).
```

This has been checked directly for `n=1/4,3/4,5/4`; its continuation to all
positive labels is the natural norm conjecture.  Negative labels follow by
the reflected realization.

## Exact finite PBW/path formula

Expand every chi-string primary in the common auxiliary-Majorana times SCA
PBW basis,

```text
v_1 = sum_a c_1(a) u_1(a) x X_1(a) phi_1,
W_2 = sum_b c_2(b) u_2(b) x X_2(b) w_2,
W_3 = sum_c c_3(c) u_3(c) x X_3(c) w_3.
```

Every chi mode has two colours: it acts either on the auxiliary fermion or
on the physical free-field fermion.  Therefore each sum is finite at fixed
branch label.  The branching numerator is exactly

```text
rhohat_f^eta(v_1,W_2,W_3)
 = sum_(a,b,c) (-1)^K c_1(a)c_2(b)c_3(c)
     rho_F(u_1(a),u_2(b),u_3(c))
     rho_f^eta(X_1(a)phi_1,X_2(b)w_2,X_3(c)w_3),

K = p_S(a)[p_F(b)+p_F(c)] + p_S(b)p_F(c)  (mod 2).
```

The auxiliary factor is the exact two-spin Ising Pfaffian and the physical
factor is reduced by the generalized NS--R--R Ward identities.  Division by
the three norms gives `(B_f^eta)^2`.  This is the constructive general
formula: it is finite at every fixed triple and it retains contractions
between the two Ramond strings.

## Ground coefficient

For `(n_1,n_2,n_3)=(0,1/4,1/4)`, the four raw master numerators with
`epsilon_3=f=0` are

```text
R_0^eta = 1 + i eta,
R_1^eta = -exp(-i eta pi/4).
```

Using `N_(1/4)^0=2` and `N_(1/4)^1=-1`, both second-leg copies give the same
normalized square:

```text
(B_f^eta[0,1/4,1/4;epsilon2,epsilon3])^2
  = (-1)^epsilon3 (-i)^f i eta/2.
```

The sign of an unsquared `B` remains sensitive to the independent sign
chosen for each branch highest vector.

## One excited Ramond leg: the scalar product survives

Let

```text
Pcal(n1,n2,n3;P1,P2,P3)
 = ell(Q/2+P1+P2+P3, 2(n1+n2+n3))
   ell(Q/2-P1+P2+P3, 2(-n1+n2+n3))
   ell(Q/2+P1-P2+P3, 2(n1-n2+n3))
   ell(Q/2+P1+P2-P3, 2(n1+n2-n3)),

D_j = ell(2P_j,4n_j) ell(Q+2P_j,4n_j).
```

For the boundary family `(n_1,n_2,n_3)=(0,1/4,n)`, `n>0`, put
`M=2n-1/2` and

```text
P2_eff = eta (-1)^M P2.
```

The explicit PBW calculation through `n=5/4` gives

```text
(B_f^eta)^2
 = eta (-1)^(epsilon3+M) i^(1-f)/2
   * Pcal(0,1/4,n;P1,P2_eff,P3)^2/(D_2 D_3).
```

It is symbolic at `n=1/4,3/4` and was checked at two independent exact
momentum samples at `n=5/4`.  This is the correct scalar-product conjecture
when only one Ramond string carries nonzero modes.

## First two-excited-leg coefficient

The first gatekeeper is

```text
(n_1,n_2,n_3) = (0,3/4,3/4).
```

Define

```text
x_pp = Q/2 + P1 + P2 + P3,
x_mm = Q/2 + P1 - P2 - P3,
x_2  = Q/2 - P1 + P2 + P3,

E_j = Q + 2P_j,
a_j = (2P_j)^2 + Q(2P_j) + 1,
d_j = E_j^2 + Q E_j + 1,

K = (x_pp^2 + Q x_pp + 1)(x_2^2 + Q x_2 + 1),
L = x_pp(x_mm-Q),
H = L^2 + 2L(E_2E_3+1) + d_2d_3.
```

For `epsilon_3=f=0`, the raw masters are

```text
R_0^+ = -(1+i) K/(d_2d_3),
R_1^+ =  i sqrt(2) R_0^+,

R_0^- = -(1-i) H/(d_2d_3),
R_1^- = -i sqrt(2) R_0^-.
```

After division by the branch norms, the two values of `epsilon_2` again
coincide and

```text
(B_f^eta)^2
 = (-1)^epsilon3 (-i)^f i eta/2
   * F_eta^2/(a_2 a_3 d_2 d_3),

F_+ = K,       F_- = H.
```

`K` is the expected scalar ell-product channel.  `H` is the crossed channel:
the terms `2L` and `d_2d_3` come from contractions joining the two excited
Ramond strings.  Exact factorization tests show that `H` is irreducible over
the rational coefficient field, while `H +/- iK` are irreducible over
`Q(i)`.  Hence no momentum-independent change of the four branch-copy/chiral
components can turn all low-level coefficients into individual NS-like ell
products.

## Screening-charge factorization and the stripped remainder

The directly relevant paper is Hadasz--Jaskolski, arXiv:1312.4520,
especially Sec. 3.2 and Appendix B.  In its NS free-field realization the
closed-contour screening charges are

```text
Q_b     = contour integral dz psi(z) E_b(z),
Q_(1/b) = contour integral dz psi(z) E_(1/b)(z).
```

Their commutators with all super-Virasoro modes vanish.  Charge neutrality
then gives eight screening representations of the same chiral matrix
element.  The vanishing of their fermionic correlators supplies four sets of
zeros.  A degree count proves that these zeros exhaust all momentum
dependence, leaving

```text
rho_NS^A(xi_3,xi_2,xi_1)
 = C_(j3,j2,j1)(b)
   * ell^sharp(X_1,j1+j2+j3)
   * ell^sharp(X_2,-j1+j2+j3)
   * ell^sharp(X_3,j1-j2+j3)
   * ell^sharp(X_4,j1+j2-j3).
```

Here `sharp=NS` or `R` is selected by the parity of `j1+j2+j3` and
`j_i=2 n_i` in the human-note convention.  At this stage the only unknown is
the momentum-independent factor `C_(j3,j2,j1)(b)`.  The paper evaluates a
special charge-neutral contour integral as a generalized Selberg integral
and obtains, in its normalization,

```text
C_(j3,j2,j1)(b)
 = (-1)^((j1+j2-j3)/2) * 2^((j1+j2+j3)/2).
```

Thus the precise scalar Ramond analogue to test is

```text
(B_(f,ell)^eta)^2
 = (-1)^epsilon3 (-i)^f i eta/2
   * Pcal(n1,n2,n3;P1,P2,P3)^2/(D_1 D_2 D_3),
```

with the reflected momentum sheet chosen consistently with the Ramond
structure.  This formula is the factor to divide out, rather than subtract:

```text
U_eta^2 = (B_f^eta)^2/(B_(f,ell)^eta)^2.
```

The low-level PBW result is

| labels `(n1,n2,n3)` | stripped `U_+^2` | stripped `U_-^2` |
| --- | --- | --- |
| `(0,1/4,1/4)` | `1` | `1` |
| `(0,1/4,3/4)` and the one-excited boundary family | `1` | `1` |
| `(0,3/4,3/4)` | `1` | `(H/K)^2` |

For the last row the literal four-ell numerator is

```text
Pcal(0,3/4,3/4) = ell(x_pp,3) ell(x_2,3) = 2^(1/4) K,
```

and the powers of `2` cancel against the two leg factors.  Thus, up to the
independent sign of each unsquared branch vector,

```text
U_+ = 1,                 U_- = H/K.
```

More precisely, if only the four-ell product and leg factors are removed,
the effective screening constant is

```text
C_eff^2
 = (B_f^eta)^2 * (D_1 D_2 D_3)/Pcal^2.
```

On the one-excited boundary this is the momentum-independent number

```text
C_eff^2 = eta (-1)^(epsilon3+M) i^(1-f)/2,
M=2n-1/2.
```

At `(0,3/4,3/4)` it is a constant in the `eta=+` channel, but in the
`eta=-` channel it contains `(H/K)^2`.  In fact `gcd(H,K)=1`; the scalar
ell polynomial `K` is not a factor of `H`.  Hence the Hadasz--Jaskolski
screening-factor argument does not extend to each fixed Ramond chiral
component as a scalar statement.

The unknown factor already has a useful two-state form.  With

```text
K23 = [[d_2 d_3, E_2 E_3+1],
       [E_2 E_3+1,       1]],
```

the crossed numerator is

```text
H = (1,L) K23 (1,L)^T.
```

Consequently the momentum-dependent remainder `H/K` is not the literature's
NS normalization constant `C_(j3,j2,j1)(b)`.  It is a chiral matrix element
produced by contractions between the two Ramond strings.  The natural
Ramond screening object is therefore a two-component polynomial or a
two-by-two spin-field kernel, with `(K,H)` as its first nontrivial example.

For comparison, Schomerus--Suchanek, arXiv:1210.1856, conjecture a different
local, nonchiral double-Liouville identity.  It contains a fourth root of
unity `varpi`; that local ambiguity should not be identified with the
momentum-dependent chiral ratio `H/K`.

This suggests the all-label organization

```text
(B_f^eta)^2
 = [universal zero-mode phase]
   * Pcal^2/(D_1 D_2 D_3)
   * U_eta^2,
```

where `U_eta` is a finite two-colour contraction/transfer-matrix element,
not generally a scalar phase.  It becomes `1` when at most one Ramond string
is excited and first becomes nontrivial at `(0,3/4,3/4)`.

## Conjectured general structure

The low-level calculation supports the following all-label structure.

1. At fixed positive labels, the sixteen choices
   `(epsilon_2,epsilon_3,f,eta)` reduce to four raw masters labelled by
   `(epsilon_2,eta)`.  For the normalized square,

   ```text
   (B_(epsilon2,epsilon3;f)^eta)^2
     = (-1)^epsilon3 (-i)^f
       (B_(epsilon2,0;0)^eta)^2.
   ```

2. Boundary sectors with at most one excited Ramond string are scalar
   four-ell products on a chiral-structure-dependent momentum sheet.

3. Generic sectors are two-channel, matrix-valued objects.  Their entries
   are finite two-colour path polynomials divided by the PBW leg factors.
   Cross-leg paths are essential and first produce `H` above.  Thus the
   natural general formula is the finite PBW/path sum, not one universal
   scalar blow-up product.

In the historical fixed-even Ward chart, the phase reduction was checked on
the full low grid

```text
n_1 = 0,1/2,1,
n_2,n_3 = 1/4,3/4,5/4,
```

for all 432 restrictions at two exact rational samples, with an additional
check at `(n_1,n_2,n_3)=(3/2,3/4,3/4)`.  The separate generalized-Ward audit
shows that an intrinsic odd NS primary replaces `eta` by `eta_eff`; this is
the required translation of the odd rows rather than a new scalar product.
In the scalar-product test, only 52/108 independent masters matched a single
product; 56/108 failed.

## Reproducible checks

```bash
python3 "python 2/ramond_three_point_grid/general_formula_search/path_sum_formula.py"

python3 "python 2/ramond_three_point_grid/certify_master_ell_ansatz.py"

python3 "python 2/nsrr_chi_branching/audit_stored_values.py"

python3 "python 2/ramond_branching_coefficient_check/strip_literature_ell_factor.py"
```

The first command checks twelve exact low master values, including the first
crossed channel.  The second certifies the `K,H` formulas and the failure of
the scalar ansatz.  The third checks the direct norms and the explicit hard
coefficient.  The exhaustive path/grid audit is available with `--full` or
`--audit` as documented in the corresponding scripts.
