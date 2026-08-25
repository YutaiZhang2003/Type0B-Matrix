# What arXiv:1510.01773 actually supplies

The primary source is Z. Jaskolski and P. Suchanek, *Non-rational
su(2)-hat cosets and Liouville field theory*, arXiv:1510.01773. It contains
neither the phrase “colored bifundamental” nor a Ramond-sector construction,
and it has no torus-holonomy labels `(0,1)` and `(1,0)`. The useful structure
is instead a two-dimensional space of chiral three-linear invariants.

The exact source equations are:

- (2.9): `S_epsilon = (-1)^(2 epsilon) S_A + S_B`, with
  `epsilon = 0, 1/2`. Thus

  ```text
  [S_0    ]   [ 1  1] [S_A]
  [S_{1/2}] = [-1  1] [S_B].
  ```

- (2.10): multiplication by the three pairwise powers shifts the invariant
  label by `n_123/2`:

  ```text
  (x_3-x_1)^(n_13^2) (x_2-x_3)^(n_23^1) (x_1-x_2)^(n_12^3)
      S_epsilon[j_i;epsilon_i]
    = S_{epsilon +dot n_123/2}[j_i+n_i; epsilon_i +dot n_i].
  ```

- (2.11): in the spin basis the two invariants occur as
  `g^31 + (-1)^(2 epsilon) g^13`. This confirms that (2.9) is a genuine
  two-channel Fourier combination, not just notation. Importantly, the
  transform is not constant after passage to the spin basis. In full,

  ```text
  g^epsilon = a_epsilon (g^31+(-1)^(2epsilon)g^13),

  a_epsilon = sin[pi(j_13^2/2-epsilon+epsilon_2)]
              sin[pi(j_1-epsilon_1)] sin[pi(j_3+epsilon_3)].
  ```

  Consequently

  ```text
  [g^0    ]   [a_0  0    ] [1  1] [g^31]
  [g^(1/2)] = [0    a_1/2] [1 -1] [g^13].
  ```

  This is the source's explicit momentum-dependent two-by-two channel
  matrix. A constant Hadamard transform is correct for the `x`-space
  invariant (2.9), but not for spin-basis matrix elements.

- (2.12) and (2.13): reflection of one representation shifts the same
  invariant label by that leg's `epsilon_i`. These equations are the source's
  matrix-valued reflection information. For reflection of leg 3, define

  ```text
  r_e^(3)=(-1)^(2 epsilon_3)
          Gamma(-j_23^1)/Gamma(1+j_13^2)
          sin pi[-1/2-j_23^1/2+e+epsilon_2]
          /sin pi[j_13^2/2-(e +dot epsilon_3)+epsilon_2].
  ```

  Equation (2.12) is

  ```text
  Gamma(-j_3+m_3+epsilon_3)/Gamma(1+j_3+m_3+epsilon_3)
      S_e[j_3,-1-j_2,-1-j_1]
    = r_e^(3) S_(e +dot epsilon_3)[-1-j_3,-1-j_2,-1-j_1].
  ```

  Therefore the reflected-sheet leg matrix is diagonal when
  `epsilon_3=0`, and for `epsilon_3=1/2` it is

  ```text
  [0       r_0^(3)]
  [r_1/2^(3)     0]
  ```

  between the ordered channels `(e=0,e=1/2)`. This is an exact source
  formula exhibiting both sheet reflection and momentum-dependent channel
  mixing.

- (3.20): the allowed shifts obey `n_1+n_2+n_3 = 0 mod 1`.

- (3.21) and (3.22): the normalized invariant has the same channel shift,
  up to a square-root sign denoted `(-1)^eta(n_3,n_2,n_1)`. The paper says
  explicitly that the general form of this sign is not known.

- (4.5): for the Liouville branch chosen in the paper,

  ```text
  kappa+2 = -1/(b Q),   kappa+3 = b/Q,   Q=b+b^(-1).
  ```

- (4.21): the finite products are

  ```text
  l(x,m) = product_{p=2..m} product_{q=1..p-1}
              [x-p(kappa+2)+q(kappa+3)]                    (m>1),
           1                                               (m=0,1),
           product_{p=0..|m|-1} product_{q=0..p}
              [x+p(kappa+2)-q(kappa+3)]                    (m<0),

  N(j,m)=(-1)^[m(2m-1)] [l(2j,2m)l(2j+1,2m)]^(-1/2).
  ```

- (4.23): apart from coordinate powers and the unresolved overall sign, the
  complete coset factor is

  ```text
  P(j,n) = product_i N(j_i,n_i)
           l(j_123+1,n_123)
           l(j_12^3,n_12^3)
           l(j_13^2,n_13^2)
           l(j_23^1,n_23^1).
  ```

## Translation to the present labels

Use the branch shifts

```text
k_i = 2 n_i(repo),       K = k_1+k_2+k_3 in Z.
```

For the grid in the current notes this is exactly the integral selection
rule: an NS branch gives integral `k_1`, while the two Ramond branches give
half-integral `k_2,k_3`, whose sum is integral. The paper's module parity
label is naturally `epsilon_i(paper)=epsilon_i(repo)/2`. Its invariant label
is closest to the present chiral sign through

```text
eta = (-1)^(2 epsilon),  so S_epsilon = S_B + eta S_A.
```

The paper's Liouville weight is `Delta_j=-Q^2 j(j+1)`. Hence one can set

```text
j_i = -1/2 + P_i/Q.
```

Changing this sign convention for `P_i` applies the paper's reflection
`j_i -> -1-j_i`, so it only exchanges the two momentum sheets.

For example, after this substitution the non-common diagonal entries in
the spin-basis matrix (2.11) become

```text
a_+ / common = sin pi[-1/4+(P_1+P_3-P_2)/(2Q)+epsilon_2(repo)/2],
a_- / common = sin pi[-3/4+(P_1+P_3-P_2)/(2Q)+epsilon_2(repo)/2].
```

This is direct evidence from the source that a reflected-sector formula
should use momentum-dependent leg matrices rather than a constant Fourier
matrix multiplying two scalar products.

Let `X=[[0,1],[1,0]]` and order the Fourier channels as `(eta=+,eta=-)`.
Equations (2.9), (2.10), and (4.23) then give the exact paper matrix

```text
T_eta(j,k) = P(j,k) X^K.
```

In the chamber basis `(A,B)` this is

```text
T_AB(j,k) = P(j,k) diag((-1)^K,1).
```

Identifying `(A,B)` with the two geometric holonomies `(0,1),(1,0)` is an
extra conjecture; the source does not make it. Either assignment differs
only by conjugation with `X`.

For the hard labels `(0,3/4,3/4)`, `k=(0,3/2,3/2)` and `K=3`. Therefore the
paper predicts an odd channel exchange,

```text
T_eta,hard = P_hard X,
T_AB,hard  = P_hard diag(-1,1),

P_hard = l(j_123+1,3) l(j_23^1,3)
         /sqrt[l(2j_2,3)l(2j_2+1,3)l(2j_3,3)l(2j_3+1,3)].
```

Here

```text
l(x,3)= product_(p,q) [x+(p/b+q b)/Q],
(p,q)=(2,1),(3,1),(3,2).
```

## Momentum-dependent lift required by the hard master

The scalar paper transfer cannot be the final Ramond answer. Its two chamber
eigenvalues have equal magnitude, whereas the direct hard calculation has
two independent polynomials. Strip the fixed phases by writing

```text
d_j = 2^(-1/8) ell(Q+2P_j,3),
E_j = ell(Q+2P_j,2),
L   = ell(x_++,2) ell(x_--,-2),

-d_2 d_3 R_0^(eta)/(1+i eta) = K  for eta=+,
                              = H  for eta=-.
```

The crossed polynomial has the exact two-state finite-product kernel

```text
H = [1,L]
    [ d_2 d_3      1+E_2 E_3 ] [1]
    [ 1+E_2 E_3             1 ] [L].
```

This is the minimal momentum-dependent lift of the scalar channel sign in
the source. If the two chamber sectors are identified with the two
holonomies, the exact phase-stripped hard matrix is the Hadamard transform

```text
M_hol = 1/2 [ K+H  K-H ]
              [ K-H  K+H ].
```

Its fixed-eta eigenvalues are exactly `K,H`. For parity copy `epsilon_2=1`,
the raw eigenvalues acquire the already verified factors `i sqrt(2) eta`;
equivalently multiply the fixed-eta diagonal matrix by
`diag(i sqrt(2),-i sqrt(2))` before the Hadamard transform.

More explicitly, at fixed parity copy `epsilon_2=0,1`, put

```text
a_epsilon=(1+i)(i sqrt(2))^epsilon,
b_epsilon=(1-i)(-i sqrt(2))^epsilon.
```

With the common denominator `d_2 d_3`, the exact raw hard matrix in the
candidate holonomy basis is

```text
R_hol,epsilon = -1/(2 d_2 d_3)
    [ a_epsilon K+b_epsilon H   a_epsilon K-b_epsilon H ]
    [ a_epsilon K-b_epsilon H   a_epsilon K+b_epsilon H ].
```

Hadamard diagonalization gives precisely

```text
R_epsilon^(+) = -(1+i)(i sqrt(2))^epsilon K/(d_2 d_3),
R_epsilon^(-) = -(1-i)(-i sqrt(2))^epsilon H/(d_2 d_3),
```

so this formula keeps both `eta` and `epsilon_2` explicit.

This hard matrix is exact, but its proposed interpretation as a holonomy
transfer is not proved by arXiv:1510.01773. The paper establishes the
two-channel Fourier algebra and the scalar finite product; the direct Ward
calculation establishes the momentum-dependent hard kernel.
