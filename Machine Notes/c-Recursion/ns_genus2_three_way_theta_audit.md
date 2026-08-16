# Genus-two all-NS theta block: computation and current check status

## 1. Checker frame and truncation

The edge order is

```text
(0,1,infinity) = (bra,middle,ket).
```

We use twice-levels

\[
 n=(n_0,n_1,n_\infty),\qquad n_e=2\,\text{level}_e,
\]

and the Liouville parametrization

\[
 Q=b+b^{-1},\qquad
 c=\frac32+3Q^2,\qquad
 h_e=\frac{Q^2}{8}-\frac{P_e^2}{2}.
\]

The fixed theta-frame orientation currently used by all three implementations
is

\[
 \varepsilon_\Theta(n)
 =(-1)^{n_0n_1+n_0n_\infty+n_1n_\infty+n_\infty},
 \tag{1.1}
\]

where the exponent only uses the twice-levels modulo two.  The last linear
term is the implemented choice of plumbing lift on the infinity edge.  The
human note's unpacked convention omits this linear term; Section 6.1 proves
that translating all three implementations to that convention cannot alter
the comparison status.

The generic calculation keeps every monomial with
\(n_0+n_1+n_\infty\le3\), i.e. total physical level at most \(3/2\).
The higher check keeps every monomial with total twice-level at most four and
evaluates it at exact rational values of \((b,P_0,P_1,P_\infty)\).

## 2. Method I: direct super-Virasoro sewing

At twice-level \(n_e\), let \(\{\mathbf L_{-A_e}\phi_e\}\) be the NS PBW
basis, ordered with decreasing Virasoro modes and strictly decreasing
fermionic modes.  The algebra used to construct the Gram matrices and the
three-point Ward identities is

\[
\begin{aligned}
[L_m,L_n]&=(m-n)L_{m+n}+\frac{c}{12}(m^3-m)\delta_{m+n,0},\\
[L_m,G_r]&=\left(\frac m2-r\right)G_{m+r},\\
\{G_r,G_s\}&=2L_{r+s}
 +\frac c3\left(r^2-\frac14\right)\delta_{r+s,0}.
\end{aligned}
\tag{2.1}
\]

Define

\[
 K_{h_e}(A_e,A'_e)
 =\langle\mathbf L_{-A_e}\phi_e\mid
          \mathbf L_{-A'_e}\phi_e\rangle,
 \qquad
 \mathbf G_{h_e}=K_{h_e}^{-1}.
 \tag{2.2}
\]

The normalized three-point form

\[
 \rho_a(A_0,A_1,A_\infty)
 =\rho_a(\mathbf L_{-A_0}\phi_0,
          \mathbf L_{-A_1}\phi_1,
          \mathbf L_{-A_\infty}\phi_\infty)
 \tag{2.3}
\]

is evaluated exactly by the NS Ward identities in the
\((\infty,1,0)\) frame.  Its primary normalization is

\[
 \rho_0(\phi_0,\phi_1,\phi_\infty)=1,
 \qquad
 \rho_1(\phi_0,G_{-1/2}\phi_1,\phi_\infty)=1.
 \tag{2.4}
\]

The direct coefficient is

\[
\boxed{
 D_a[n]=\delta_{a,\,n_0+n_1+n_\infty\ ({\rm mod}\ 2)}
 \varepsilon_\Theta(n)
 \sum_{A_e,A'_e}
 \rho_a(A_0,A_1,A_\infty)
 \prod_{e=0,1,\infty}\mathbf G_{h_e}^{A_eA'_e}
 \rho_a(A'_0,A'_1,A'_\infty).}
 \tag{2.5}
\]

All Gram matrices, their inverses, and the Ward-identity values are exact
SymPy rational functions.

## 3. Method II: NS fixed-weight \(c\)-recursion

Write \(R_a[n;c,h_0,h_1,h_\infty]\) for the recursive coefficient.  At the
orders used here, the regular term is the global
\(\mathfrak{osp}(1|2)\) theta coefficient.  Write

\[
 n_e=2m_e+\epsilon_e,\qquad m_e\in\mathbb Z_{\ge0},\quad
 \epsilon_e\in\{0,1\}.
\]

If \(\rho_{\mathfrak{osp}}(m_e,\epsilon_e;h_e)\) denotes the normalized
three-point form of

\[
 L_{-1}^{m_e}G_{-1/2}^{\epsilon_e}\phi_e,
\]

then the seed used in the computation is

\[
 R_a^{\rm reg}[n]
 =\delta_{a,\,\sum_e n_e\ ({\rm mod}\ 2)}
 \varepsilon_\Theta(n)
 \frac{\rho_{\mathfrak{osp}}(m_e,\epsilon_e;h_e)^2}
 {\displaystyle\prod_e m_e!\,(2h_e)_{m_e+\epsilon_e}}.
 \tag{3.1}
\]

Here \((x)_m\) is the rising Pochhammer symbol, and
\(\rho_{\mathfrak{osp}}\) is defined by restricting (2.1) to
\(L_{0,\pm1},G_{\pm1/2}\) and applying the same normalization (2.4).  The
implementation evaluates this form as a finite Pochhammer sum, not by a
numerical fit.

Let \(e_i\) be the unit vector on edge \(i\).  The coefficient recursion
actually used is

\[
\boxed{
\begin{aligned}
R_a[n;c,h]
={}&R_a^{\rm reg}[n;h]\\
&+\sum_{i=0,1,\infty}\ \sum_{(r,s)\in\{(3,1),(2,2)\}}
 \mathbf1_{n_i\ge rs}\,
 \frac{\varepsilon_\Theta(n)}
      {\varepsilon_\Theta(n-rs\,e_i)}
 \frac{\mathcal R_{a;rs}^{(i)}(h)}{c-c_{rs}(h_i)}\\
&\hspace{7em}\times
 R_{a+rs\ ({\rm mod}\ 2)}
 [n-rs\,e_i;c_{rs}(h_i),h_i\mapsto h_i+rs/2].
\end{aligned}}
 \tag{3.2}
\]

Only the \((3,1)\) pole contributes through total level \(3/2\); the
\((2,2)\) pole first contributes on the level-two shell.

The residue factors in (3.2) are evaluated directly as follows.  For the two
weights on the other edges, call them \(h_j,h_k\), and define

\[
 \Phi(u,v;t)=\frac{(u+t-v)^2-4tu}{64}.
 \tag{3.3}
\]

For \((r,s)=(3,1)\),

\[
\begin{gathered}
 x=-h_i-\frac12,\qquad
 c_{31}(h_i)=6-3h_i-\frac6{2h_i+1},\\
 Q_*^2=x+2+x^{-1},\qquad
 \lambda_j^2=Q_*^2-8h_j,\quad
 \lambda_k^2=Q_*^2-8h_k,\\
 \mathcal P_{31}^{0}=\Phi(\lambda_j^2,\lambda_k^2;4x),\qquad
 \mathcal P_{31}^{1}=\frac{\lambda_j^2-\lambda_k^2}{8},\\
 \mathcal R_{a;31}^{(i)}
 =\frac6{(2h_i+1)^2}\left(\mathcal P_{31}^{a}\right)^2.
\end{gathered}
 \tag{3.4}
\]

For \((r,s)=(2,2)\),

\[
\begin{gathered}
 c_{22}(h_i)=\frac32-8h_i,\qquad Q_*^2=-\frac83h_i,\\
 \lambda_j^2=Q_*^2-8h_j,\qquad
 \lambda_k^2=Q_*^2-8h_k,\\
 \mathcal P_{22}^{0}=\Phi(\lambda_j^2,\lambda_k^2;Q_*^2),\qquad
 \mathcal P_{22}^{1}=\Phi(\lambda_j^2,\lambda_k^2;Q_*^2-4),\\
 \mathcal R_{a;22}^{(i)}
 =\frac9{4h_i(2h_i+3)}\left(\mathcal P_{22}^{a}\right)^2.
\end{gathered}
 \tag{3.5}
\]

## 4. Method III: two Virasoro blocks and ordinary fermion inversion

### 4.1 Ordinary tensor-product factorization

The reviewed human convention is kept fixed:

\[
\boxed{\widehat{\mathbf F}_a(q)=
       \mathbf F_{\mathsf F}(q)\,\mathbf F_a(q).}
 \tag{4.1}
\]

This is ordinary multiplication.  Therefore, coefficientwise,

\[
 \widehat F_a[n]=\sum_{f+s=n}F_{\mathsf F}[f]F_a[s].
 \tag{4.2}
\]

Since \(F_{\mathsf F}[0]=1\), the SCA coefficient extracted from the third
method is

\[
\boxed{
 T_a[0]=\widehat F_a[0],\qquad
 T_a[n]=\widehat F_a[n]
 -\sum_{0\ne f\le n}F_{\mathsf F}[f]T_a[n-f].}
 \tag{4.3}
\]

### 4.2 Auxiliary Majorana factor

At twice-level \(f_e\), let \(\mathcal B_{\mathsf F}(f_e)\) be the NS Fock
basis

\[
 u_{\mathsf A}=\psi_{-r_1}\cdots\psi_{-r_m}\mathbf1_{\mathsf F},\qquad
 r_1<\cdots<r_m,\qquad \sum_jr_j=f_e/2.
\]

The BPZ-dual ordering is chosen so that this basis is orthonormal.  The
fermion three-point form \(\rho^{\mathsf F}\) is evaluated as the Pfaffian of
the pair contractions following from
\(\langle\psi(z)\psi(w)\rangle=(z-w)^{-1}\).  The coefficient used in (4.2)
is

\[
\boxed{
F_{\mathsf F}[f]
=\varepsilon_\Theta(f)
 \sum_{u_e\in\mathcal B_{\mathsf F}(f_e)}
 \rho^{\mathsf F}(u_0,u_1,u_\infty)^2.}
 \tag{4.4}
\]

### 4.3 Branching sum

Put \(k_e=2n_e^{\rm branch}\in\mathbb Z\).  The diagonal
\(\mathsf{Vir}^{(1)}\oplus\mathsf{Vir}^{(2)}\) expression implemented for
the enlarged block is

\[
\boxed{
\begin{aligned}
\widehat{\mathbf F}_a(q)
={}&\sum_{\substack{k_0,k_1,k_\infty\in\mathbb Z\\
                    k_0+k_1+k_\infty\equiv a\ ({\rm mod}\ 2)}}
 \varepsilon_\Theta(k_0^2,k_1^2,k_\infty^2)
 \prod_e q_e^{k_e^2/2}\,
 \mathcal B_a(P_e;k_e)\\
&\times
 F_{\rm Vir}^{(1)}(h_{0,k_0}^{(1)},h_{1,k_1}^{(1)},
                    h_{\infty,k_\infty}^{(1)};c^{(1)};q)
 F_{\rm Vir}^{(2)}(h_{0,k_0}^{(2)},h_{1,k_1}^{(2)},
                    h_{\infty,k_\infty}^{(2)};c^{(2)};q).
\end{aligned}}
 \tag{4.5}
\]

The two Virasoro parameters are

\[
\begin{gathered}
(b^{(1)})^2=\frac{2b^2}{1-b^2},\qquad
(b^{(2)})^{-2}=\frac2{b^2-1},\\
c^{(I)}=1+6(Q^{(I)})^2,\qquad Q^{(I)}=b^{(I)}+(b^{(I)})^{-1},\\
h_{e,k}^{(1)}=\frac{(Q^{(1)})^2}{4}
 -\frac{(P_e+kb)^2}{2-2b^2},\\
h_{e,k}^{(2)}=\frac{(Q^{(2)})^2}{4}
 -\frac{(P_e+k/b)^2}{2-2b^{-2}}.
\end{gathered}
 \tag{4.6}
\]

They obey

\[
 h_{e,k}^{(1)}+h_{e,k}^{(2)}=h_e+\frac{k^2}{2}.
 \tag{4.7}
\]

The squared normalized branching coefficient is

\[
\boxed{
\mathcal B_a(P_e;k_e)
=\frac{l(Q/2+P_1,k_1\mid P_0,k_0,P_\infty,k_\infty)^2}
       {N_{k_0}(P_0)N_{k_1}(P_1)N_{k_\infty}(P_\infty)}.}
 \tag{4.8}
\]

To define its factors, for \(r\ge0\) set

\[
\begin{aligned}
s_{\rm even}(x,r)
&=2^{-r^2/2}
 \prod_{\substack{i,j\ge1\\i+j\le2r\\i+j\ {\rm even}}}
 [x+(i-1)b+(j-1)b^{-1}],\\
s_{\rm odd}(x,r)
&=2^{-r(r+1)/2}
 \prod_{\substack{i,j\ge1\\i+j\le2r+1\\i+j\ {\rm odd}}}
 [x+(i-1)b+(j-1)b^{-1}],
\end{aligned}
 \tag{4.9}
\]

with negative-index continuation

\[
s_{\rm even}(x,r)=(-1)^r s_{\rm even}(Q-x,-r),\qquad
s_{\rm odd}(x,r)=s_{\rm odd}(Q-x,-r).
 \tag{4.10}
\]

Let \({\rm Int}(y)={\rm sgn}(y)\lfloor|y|\rfloor\).  For

\[
x_{\sigma\tau}=\alpha+\sigma P_0+\tau P_\infty,\qquad
r_{\sigma\tau}=\frac{k_1+\sigma k_0+\tau k_\infty}{2},
\]

the blow-up factor is

\[
l(\alpha,k_1\mid P_0,k_0,P_\infty,k_\infty)
=\begin{cases}
\displaystyle\prod_{\sigma,\tau=\pm1}
s_{\rm even}(x_{\sigma\tau},r_{\sigma\tau}),
&k_0+k_1+k_\infty\text{ even},\\[0.8em]
\displaystyle\prod_{\sigma,\tau=\pm1}
s_{\rm odd}(x_{\sigma\tau},{\rm Int}(r_{\sigma\tau})),
&k_0+k_1+k_\infty\text{ odd}.
\end{cases}
 \tag{4.11}
\]

Finally,

\[
 N_k(P)=s_{\rm even}(2P,k)s_{\rm even}(-2P,-k).
 \tag{4.12}
\]

### 4.4 Ordinary Virasoro blocks inside (4.5)

Each factor in (4.5) is computed independently by ordinary Virasoro
fixed-weight \(c\)-recursion.  For Virasoro levels
\(m=(m_0,m_1,m_\infty)\), the global seed is

\[
C_{\rm glob}^{\rm Vir}[m]
=\frac{\rho_{\mathfrak{sl}_2}(m;h)^2}
       {\prod_e m_e!(2h_e)_{m_e}}.
 \tag{4.13}
\]

In the edge order \((\text{bra},\text{middle},\text{ket})\), writing
\(m=(i,j,k)\),

\[
\begin{aligned}
\rho_{\mathfrak{sl}_2}(i,j,k)
={}&(h_0+i-h_1-j+1-h_\infty-k)_j\\
&\times\sum_{p=0}^{\min(i,k)}{i\choose p}
 (2h_\infty+k-1)^{\underline p}k^{\underline p}
 (h_\infty+h_1-h_0)_{k-p}\\
&\hspace{8em}\times
 (h_0+h_1-h_\infty+p-k)_{i-p}.
\end{aligned}
 \tag{4.14}
\]

At the present cutoff, only the ordinary Virasoro \((2,1)\) pole is needed.
For an edge of weight \(h_i\), define

\[
\begin{gathered}
x=-\frac23(2h_i+1),\qquad c_{21}(h_i)=13+6(x+x^{-1}),\\
Q_*^2=x+2+x^{-1},\qquad
\lambda_j^2=Q_*^2-4h_j,\quad \lambda_k^2=Q_*^2-4h_k,\\
\mathcal P_{21}
=\frac{(\lambda_j^2+\lambda_k^2-x)^2
       -4\lambda_j^2\lambda_k^2}{16},\\
\mathcal R_{21}^{(i)}=\frac2{x^2}\mathcal P_{21}^2.
\end{gathered}
 \tag{4.15}
\]

The implemented recursion is

\[
C_{\rm Vir}[m;c,h]
=C_{\rm glob}^{\rm Vir}[m;h]
 +\sum_i\mathbf1_{m_i\ge2}
 \frac{\mathcal R_{21}^{(i)}}{c-c_{21}(h_i)}
 C_{\rm Vir}[m-2e_i;c_{21}(h_i),h_i\mapsto h_i+2].
 \tag{4.16}
\]

The two Virasoro series are multiplied ordinarily, inserted into (4.5), and
then divided by (4.4) using (4.3).

## 5. Current check status

### 5.1 Generic symbolic check through total level \(3/2\)

| Comparison | Exact zero identities | Mismatches |
|---|---:|---:|
| Direct PBW (2.5) vs NS recursion (3.2) | 20/20 | 0 |
| Direct PBW (2.5) vs double-Virasoro result (4.3) | 14/20 | 6 |

The first failure is

\[
\left[D_a-T_a\right]_{(0,1,2)}
=-\frac{8b^2}{[(b^2+1)^2-4P_\infty^2b^2]}
=-\frac1{h_\infty}.
 \tag{5.1}
\]

The six mismatching twice-levels are

```text
(0,1,2), (0,2,1), (1,0,2),
(1,2,0), (2,0,1), (2,1,0).
```

### 5.2 Complete exact-rational check through total level two

The three samples are

```text
b=3/2, P=(2/7,3/11,5/13),
b=4/3, P=(1/5,2/9,4/11),
b=5/4, P=(-2/7,1/6,3/10).
```

All evaluations use exact rational arithmetic.

| Comparison | Exact zero identities | Mismatches |
|---|---:|---:|
| Direct PBW (2.5) vs NS recursion (3.2) | 105/105 | 0 |
| Direct PBW (2.5) vs double-Virasoro result (4.3) | 78/105 | 27 |

There are nine double-Virasoro mismatches per sample.  Six are inherited from
the level-\(3/2\) shell.  The three new level-two mismatches are

```text
(1,1,2), (1,2,1), (2,1,1).
```

The level-two calculation includes the NS \((2,2)\) residue, the ordinary
Virasoro \((2,1)\) residue in both Virasoro factors, and branch labels through
\(k=\pm2\).

Therefore the current status is:

```text
direct PBW = NS c-recursion,
direct PBW != current diagonal double-Virasoro implementation
             after ordinary fermion inversion.
```

## 6. Origin audit for the sign discrepancy

This section distinguishes an implementation frame sign from the sign that
actually causes the mismatch.  No sign is inserted into the product in this
audit.

### 6.1 The infinity-edge lift is not the cause

For a parity vector \(p=(p_0,p_1,p_\infty)\in(\mathbb Z_2)^3\), define

\[
 Q(p)=p_0p_1+p_0p_\infty+p_1p_\infty.
 \tag{6.1}
\]

The human note writes the theta-channel sewing sign as

\[
 \sigma_{\rm H}(p)=(-1)^{Q(p)}.
 \tag{6.2}
\]

The present checker instead uses

\[
 \sigma_{\rm impl}(p)=(-1)^{Q(p)+p_\infty}
 =\chi_\infty(p)\sigma_{\rm H}(p),
 \qquad \chi_\infty(p)=(-1)^{p_\infty}.
 \tag{6.3}
\]

This is the extra infinity-edge lift recorded in (1.1).  It is not written in
the human note's unpacked theta convention.  Nevertheless, it cannot produce
the observed failure: \(\chi_\infty\) is a character,

\[
 \chi_\infty(p+q)=\chi_\infty(p)\chi_\infty(q),
 \tag{6.4}
\]

so a coherent removal of it from the NS, Majorana, and diagonal-Virasoro
series commutes with ordinary series multiplication and division.  It changes
some displayed coefficient signs, but it leaves every equality or mismatch
in Section 5 unchanged.  Thus the checker should ultimately be presented in
the human frame (6.2), but the linear infinity sign is not the origin of
(5.1).

### 6.2 The non-multiplicative sign is the polarization of \(Q\)

Let \(s=(s_0,s_1,s_\infty)\) be the SCA descendant parities and
\(f=(f_0,f_1,f_\infty)\) the auxiliary-fermion parities.  The total parities
seen by the enlarged algebra are \(t=s+f\pmod 2\).  Expanding (6.1) gives the
identity

\[
 Q(s+f)=Q(s)+Q(f)+\Xi(s,f)\pmod2,
 \tag{6.5}
\]

where

\[
\begin{aligned}
 \Xi(s,f)={}&s_0f_1+f_0s_1+s_0f_\infty+f_0s_\infty\\
             &+s_1f_\infty+f_1s_\infty.
\end{aligned}
 \tag{6.6}
\]

Consequently,

\[
 \sigma_{\rm H}(s+f)
 =\sigma_{\rm H}(s)\sigma_{\rm H}(f)(-1)^{\Xi(s,f)}.
 \tag{6.7}
\]

The human note's product-basis definition and its ordinary factorization use
the separate sign \(\sigma_{\rm H}(s)\sigma_{\rm H}(f)\).  Its diagonal
Virasoro formula uses the total branch parity through
\((-1)^{4n_0n_1+4n_0n_\infty+4n_1n_\infty}\), namely
\(\sigma_{\rm H}(t)\) on the branch primaries.  Equations (6.5)--(6.7) show
that these two descriptions require a polarization conversion.  The
conversion is not a new multiplication law for
\(\mathbf F_{\mathsf F}\mathbf F_{\mathsf{SCA}}\); it belongs to the
product-basis-to-diagonal-basis sewing map if the ordinary factorization in
the human note is kept fixed.

More explicitly, if \(C\) is the change of basis from
\(\Psi_{-\mathsf A}\mathbf L_{-A}\phi\) states to diagonal-Virasoro states,
the orientation operator that must be transported is

\[
 C^{-1}\Omega_{\rm sep}C,\qquad
 \Omega_{\rm sep}|s,f\rangle
 =(-1)^{Q(s)+Q(f)}|s,f\rangle.
\]

Replacing this operator by the scalar total-parity ledger
\(\Omega_{\rm tot}=(-1)^{Q(s+f)}\) drops \((-1)^{\Xi(s,f)}\).  Since
\(\Xi\) depends on the separate product-basis components, its conjugate need
not be diagonal in the Virasoro descendant basis and cannot in general be
repaired by changing only a branch-primary coefficient.

The Koszul factor in the human note's tensor-product three-point function
does not remove (6.7) in the displayed block: the two three-point factors
have the same edge parities, so that factor occurs twice and squares to one.
The displayed product Gram factor supplies the separate fermion BPZ sign
\((-1)^{f_0+f_1+f_\infty}\), not the bilinear polarization (6.6).

### 6.3 The first mismatch isolates this polarization exactly

At total twice-level

\[
 \ell=(0,1,2),
 \tag{6.8}
\]

the only nonconstant auxiliary-fermion contribution in the ordinary product
has

\[
 f=(0,1,1),\qquad s=\ell-f=(0,0,1).
 \tag{6.9}
\]

In the human convention,

\[
 Q(f)=1,\qquad Q(s)=0,\qquad
 t=s+f=(0,1,0),\qquad Q(t)=0,
 \tag{6.10}
\]

and hence \(\Xi(s,f)=1\).  Directly from the level-\(1/2\) Gram matrix and the
human note's normalization
\(\rho_1(\phi_0,\phi_1,G_{-1/2}\phi_\infty)=1\),

\[
 [\mathbf F_{\mathsf{SCA}}]_{(0,0,1)}=\frac1{2h_\infty},
 \qquad
 [\mathbf F_{\mathsf F}]_{(0,1,1)}=-1.
 \tag{6.11}
\]

Thus ordinary factorization assigns this split the contribution
\(-1/(2h_\infty)\), whereas the total-parity diagonal sign assigns
\(+1/(2h_\infty)\).  Their difference is \(1/h_\infty\).  After ordinary
fermion division this is precisely

\[
 [D_a-T_a]_{(0,1,2)}=-\frac1{h_\infty},
 \tag{6.12}
\]

which is (5.1).  Therefore the proposed cross sign was not a numerical patch:
it detected the exact polarization gap between the two sewing descriptions.
It should, however, be withdrawn as a modification of the ordinary product.

This coefficient lies below both the first NS Kac contribution
\((r,s)=(3,1)\), which starts at twice-level \(3\) on one edge, and the first
ordinary Virasoro \((2,1)\) contribution, which starts at Virasoro level \(2\).
The failure is therefore independent of all recursion residues and fusion
polynomials.  In particular, the present evidence does **not** point to a
missing sign in the NS \(c\)-recursion.  It points to a sewing/order sign in
the passage from the \(\Psi_{-\mathsf A}\mathbf L_{-A}\phi\) product basis to
the diagonal \(\mathsf{Vir}\oplus\mathsf{Vir}\) basis.

### 6.4 First state at which the missing bookkeeping is visible

Let

\[
 X=L_{-1}(\mathbf1_{\mathsf F}\otimes\phi),\qquad
 Y=\psi_{-1/2}G_{-1/2}(\mathbf1_{\mathsf F}\otimes\phi).
 \tag{6.13}
\]

Using the generators stated in the human note,
\(L^{\mathsf F}_{-1}(\mathbf1_{\mathsf F}\otimes\phi)=0\) and
\(U_{-1}(\mathbf1_{\mathsf F}\otimes\phi)=Y\), so

\[
\begin{aligned}
 L_{-1}^{(1)}v_0
 &=\frac{b^{-1}X+Y}{b^{-1}-b},\\
 L_{-1}^{(2)}v_0
 &=\frac{bX+Y}{b-b^{-1}}.
\end{aligned}
 \tag{6.14}
\]

The mixed vector \(Y\) is simultaneously odd in the SCA and auxiliary
fermion factors while having even total parity.  Coupling it to the auxiliary
fermion component of the odd middle-edge branch primary gives exactly the
split (6.9).  A formula that remembers only total branch parity cannot see
this relative sign.  Equation (6.14) identifies the first finite-dimensional
change-of-basis matrix on which the missing Koszul polarization must be
derived explicitly.

The corrective task is therefore to compute the sewn bilinear form under the
level-one change of basis (6.14), with the operator order and BPZ rules of the
human note, and then extend that derived rule to higher descendants.  Until
that calculation is made, no extra sign is inserted into either the NS block,
the Majorana block, or their ordinary product.

## 7. Reproduction

```bash
PYTHONDONTWRITEBYTECODE=1 \
  python3 Code/ns_genus2_three_way_symbolic_check.py --max-twice-level 4

PYTHONPATH=Code PYTHONDONTWRITEBYTECODE=1 \
  python3 -m unittest Code/test_ns_genus2_three_way_symbolic_check.py
```

The human note is not modified by this computation.
