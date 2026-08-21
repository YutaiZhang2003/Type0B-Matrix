# Genus-two all-NS theta block: three-way audit

## 0. Certified current status (2026-08-21)

This section is the authoritative status ledger. It records only results with
an executable exact or numerical certificate. Older statements elsewhere in
this file are derivational background and do not supersede this section.

### 0.1 Completed all-NS block result

Within the finite ranges stated below, the current human-note NS PBW,
fusion-polynomial, Ward-identity, global-seed, and \(c\)-recursion conventions
are fully consistent for generic primary parity.  The diagonal
double-Virasoro comparison is separately certified for even primaries.  The
extension of the auxiliary-fermion/double-Virasoro factorization to generic
primary parity remains incomplete and is explicitly outside the certified
result.

| Certified comparison | Checked range | Result |
|---|---:|---:|
| Exact theta PBW sewing vs exact NS \(c\)-recursion, including all \((p_1,p_2,p_3)\in\mathbb Z_2^3\) | Every triple with total twice-level \(\le 6\) | 672/672 exact identities |
| Numerical PBW/Ward engine vs production coefficient recursion, all \(p_i\) and nontrivial lifts | Every triple with total twice-level \(\le 4\) | 280/280, maximum discrepancy below \(2\times10^{-12}\) |
| One-null three-point factorization into \(P_{r,s}^{a}\), all three slots and all \(p_i\) | \((1,1),(3,1),(2,2),(5,1)\), ground global descendants | 768/768 exact identities |
| Ordered two-null factorization and Appendix A.6 sign table | Same low null family in the checked configurations | 210/210 exact identities |
| Direct vs resummed global \(\mathfrak{osp}(1|2)\) graph | Theta and glasses, both relative labels and all \(p_i\) | 32/32 cases within \(2\times10^{-11}\) |
| Direct PBW vs NS recursion vs diagonal double Virasoro, \(p_i=0\) | Total level \(\le3/2\), then three exact samples through level 2 | 20/20 symbolic and 105/105 sampled identities |
| Provisional direct \(\mathsf{SCA}_{NS}\otimes\mathsf F_{NS}\) branching diagnostic, all \(p_i\) | All 216 cases with \(p\in\mathbb Z_2^3\), \(k_i\in\{0,1,2\}\) | Reproducible under the provisional matrix-element interpretation; 64/216 differ from the even-primary squared coefficient, but this is not a certified factorization formula |
| Original finite-\(c\) genus-two driver | Total twice-level \(\le12\) | 455 coefficients; worst relative error \(1.01\times10^{-12}\) |

The exact level-six result is the strongest current local certificate:

\[
 \boxed{D_a^{(p)}[n]=R_a^{(p)}[n]}
 \qquad
 \left(\sum_i n_i\le6,\quad p\in\mathbb Z_2^3\right).
\]

The earlier three-way result remains certified in its stated range:

\[
 \boxed{D_a[n]=R_a[n]=T_a[n]}.
\]

### 0.2 Incomplete: generic-parity auxiliary-fermion/double-Virasoro factorization

**Status: INCOMPLETE — deferred for further discussion.**

The following results remain certified and are not downgraded by this open
issue:

- direct NS PBW equals the NS \(c\)-recursion for every external parity in the
  ranges recorded above;
- the fusion-polynomial factorization, including its graded slot signs, passes
  the direct PBW checks;
- the diagonal double-Virasoro comparison passes for \(p_i=0\).

What remains open is the precise generic-parity definition of the
\(\mathsf F_{NS}\otimes\mathsf{SCA}_{NS}\) three-point block, the induced
Koszul rule in its sewing convolution, and consequently the correct
generic-parity branching coefficient.  The 64/216 count in the table is a
diagnostic under one provisional interpretation, not a final convention or a
certified obstruction.  It must not be used to modify the human note or the
production all-label double-Virasoro formula before this issue is resolved.

### 0.3 Definitive convention ledger

Let \((A,C,E)\) be descendant parities relative to the three highest-weight
states and \((p_1,p_2,p_3)\) their intrinsic parities.

1. The block, three-form, and fusion-polynomial label is always
   \[
     a=A+C+E\pmod2.
   \]
   It never contains \(p_1+p_2+p_3\).
2. The absolute trilinear parity is
   \[
     a+p_1+p_2+p_3\pmod2,
   \]
   and is used only for holomorphic/antiholomorphic parity matching.
3. Public Ward and global-\(\mathfrak{osp}(1|2)\) routines use
   \[
   \rho_a^{\rm H}(x_1,x_2,x_3)
   =(-1)^{a|x_3|+p_1|x_1|+p_2|x_3|}
     \rho_a^{\rm comp}(x_1,x_2,x_3).
   \]
4. The theta block contains the literal sewing factor
   \[
   (-1)^{Q(A+p_1,C+p_2,E+p_3)}
   \eta_1^{A+p_1}\eta_2^{C+p_2}\eta_3^{E+p_3},
   \qquad Q(x)=x_1x_2+x_1x_3+x_2x_3.
   \]
   There is no extra infinity-edge character and no additional
   \(\eta^{p_i}\) hidden in the definition.
5. The fusion polynomial is the parity-independent weight polynomial
   \(P_{r,s}^{a}\). Intrinsic primary parity enters only the graded local
   factorization signs. For \(\delta=rs\bmod2\), the one-null signs in slots
   \((1,2,3)=(\infty,1,0)\) are
   \[
   (-1)^{\delta(p_1+A)},\qquad
   1,\qquad
   (-1)^{\delta(1+p_2)}.
   \]
6. In the printed fusion-polynomial order of Appendix A.6, the two-null signs
   are
   \[
   \begin{array}{c|c}
   (1,2)&(-1)^{\delta(p_1+A)}\\
   (1,3)&(-1)^{\delta(1+p_1+p_2+A)}\\
   (2,3)&(-1)^{\delta p_2}.
   \end{array}
   \]
7. An odd null changes the chirality of the shifted internal primary. The
   child relative label is \(a\mapsto a+rs\pmod2\); its sewing character is
   transported with the adjacent lifts. This is not an additional
   parity-dependent factor inside \(P_{r,s}^{a}\).
8. The auxiliary-fermion/SCA module uses the ungraded tensor-product pairing.
   Its precise extension to generic-parity product-block three-point forms and
   double-Virasoro factorization is the incomplete issue recorded in 0.2; no
   generic-parity branching formula is certified here.

### 0.4 Resolved issues that must not be reopened

| Former concern | Certified resolution |
|---|---|
| Generic \(p_i\) might require changing the fusion polynomial | No. The polynomial remains \(P_{r,s}^{a}\); the graded Ward/factorization and sewing signs carry all \(p_i\)-dependence. |
| The block label might be the absolute parity | No. It is the relative \(a=A+C+E\); absolute parity is separate metadata. |
| Equation (3.30) or Appendix A.6 might require an untracked common sign | The slot-dependent one-null and ordered two-null signs are the formulas in 0.3 and have been obtained from direct PBW factorization. |
| Odd \(rs\) causes unexplained orbit mixing | The shifted null primary has opposite chirality, so its sewing character and adjacent lifts are transported. This accounts for the apparent \(\eta\)-flip. |
| The six failures beginning at \((0,1,2)\) disprove the block formula | They occur only in the deliberately naive ordinary-quotient control. The theta-polarized construction has no mismatch in the checked range. |
| A linear infinity-edge sign is part of the current convention | No. It was a historical backend frame character and has been removed; every current implementation uses the literal human-note lifts. |
| The \((1,1,1)\) branching computation remains inconsistent | No. The conflict came from mixing a superseded ordered-three-form convention with the current ungraded pairing. |

### 0.5 Certification boundary

The table above is a finite, reproducible certificate; it is not an all-level
proof.  In particular, a closed all-label blow-up formula for the
generic-primary-parity double-Virasoro branching coefficient has not yet been
derived.  The existing all-label diagonal double-Virasoro formula is
certified only for \(p_i=0\).  This does not alter the generic-parity NS
PBW/\(c\)-recursion certificate.  In particular, no final conclusion is drawn
from the provisional 64/216 diagnostic in 0.2.  The table also does not certify equality of
the fully integrated and
normalized theta- and glasses-channel genus-two partition functions. That
downstream question additionally requires controlled recursion-order,
quadrature, global-resummation, Schottky-product, and confluent-pole limits.
It must not be described as an incompleteness of the local NS block formula
certified here.

The ordinary quotient retained below is only a negative control. Its expected
failures must not be recycled as evidence against the current recursion or
branching formula.

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

Let \(d_e=n_e\bmod2\) be the descendant parity, let \(p_e\) be the intrinsic
primary parity, and let \(\eta_e=\pm1\) be the literal human-note lift. The
current theta sewing factor is

\[
 \varepsilon_\Theta(d,p;\eta)
 =(-1)^{Q(d+p)}\prod_{e=0,1,\infty}\eta_e^{d_e+p_e},
 \qquad
 Q(x)=x_0x_1+x_0x_\infty+x_1x_\infty,
 \tag{1.1}
\]

with all additions in \(Q\) understood modulo two. There is no extra linear
infinity-edge term. The original three-way audit is the specialization
\(p=(0,0,0)\); the later PBW/recursion certificate covers all eight \(p\).
In Sections 2--6, the shorthand \(\varepsilon_\Theta(n)\) means the unit-lift,
even-primary specialization \((-1)^{Q(n\bmod2)}\).

The generic calculation keeps every monomial with
\(n_0+n_1+n_\infty\le3\), i.e. total physical level at most \(3/2\).
The higher check keeps every monomial with total twice-level at most four and
evaluates it at exact rational values of \((b,P_0,P_1,P_\infty)\).
The newer exact two-way PBW/recursion audit keeps every monomial with total
twice-level at most six and checks every intrinsic-parity triple.

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

## 4. Method III: two Virasoro blocks and polarized fermion inversion

### 4.1 Theta-polarized convolution

The tensor product of the SCA and auxiliary Majorana theories is ordinary.
However, the theta sewing sign is quadratic in the three edge parities. Once
that sign is included in the coefficient series, tensor-product
factorization is represented coefficientwise by a polarized convolution.

For twice-level triples \(f,s\), define

\[
\begin{aligned}
 \Xi(f,s)={}&f_0s_1+s_0f_1+f_0s_\infty+s_0f_\infty\\
             &+f_1s_\infty+s_1f_\infty\pmod2,
\end{aligned}
 \tag{4.1}
\]

for the three ordered edges \((0,1,\infty)\). The convention used in the
human note and checker is

\[
\boxed{
 \widehat{\mathbf F}_a=\mathbf F_{\mathsf F}\star\mathbf F_a,
 \qquad
 \widehat F_a[n]
 =\sum_{f+s=n}(-1)^{\Xi(f,s)}F_{\mathsf F}[f]F_a[s].}
 \tag{4.2}
\]

Since \(F_{\mathsf F}[0]=1\), the SCA coefficient extracted from the third
method is

\[
\boxed{
 T_a[0]=\widehat F_a[0],\qquad
 T_a[n]=\widehat F_a[n]
 -\sum_{0\ne f\le n}(-1)^{\Xi(f,n-f)}
   F_{\mathsf F}[f]T_a[n-f].}
 \tag{4.3}
\]

Equation (4.3) is the triangular \(\star\)-inverse. Dropping its
\((-1)^{\Xi}\) factor defines the obsolete ordinary-quotient control discussed
in Sections 5 and 6; it is not Method III.

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

## 5. Verification record

### 5.1 Generic symbolic check through total level \(3/2\)

| Comparison | Exact zero identities | Mismatches |
|---|---:|---:|
| Direct PBW (2.5) vs NS recursion (3.2) | 20/20 | 0 |
| Direct PBW (2.5) vs double-Virasoro result using the theta-polarized star inverse | 20/20 | 0 |

Denote by \(T_a^{\rm ord}\) the naive ordinary-quotient control obtained by
dropping the polarization in (4.3). Its first expected failure is

\[
\left[D_a-T_a^{\rm ord}\right]_{(0,1,2)}
=-\frac{8b^2}{[(b^2+1)^2-4P_\infty^2b^2]}
=-\frac1{h_\infty}.
 \tag{5.1}
\]

The six control mismatches are

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
| Direct PBW (2.5) vs double-Virasoro result using the theta-polarized star inverse | 105/105 | 0 |

For comparison, the obsolete ordinary quotient has nine mismatches per
sample.  Six are inherited from the level-\(3/2\) shell, and its three
additional level-two control mismatches are

```text
(1,1,2), (1,2,1), (2,1,1).
```

The level-two calculation includes the NS \((2,2)\) residue, the ordinary
Virasoro \((2,1)\) residue in both Virasoro factors, and branch labels through
\(k=\pm2\).

### 5.3 Generic-primary-parity and fusion-factorization certificate

The later convention audit extends the direct PBW/recursion equality to all
intrinsic primary parities without changing the fusion polynomial. From the
repository root, the principal certificates are

```bash
python3 Code/check_ns_genus2_arbitrary_primary_parity.py \
  --max-twice-level 6 --json
python3 Code/check_ns_fusion_global_osp.py --max-occupation 0
python3 Code/check_ns_double_null_fusion.py --max-occupation 0
python3 Code/ns_genus12_finite_c_check.py
```

They certify, respectively:

```text
exact generic-p PBW/recursion coefficients          672/672
relative-label projectors                           672/672
null-orientation transports                         816/816
one-null PBW/global-osp factorizations               768/768
ordered two-null factorizations                      210/210
finite-c genus-two coefficients                      455 checked
finite-c worst relative error                        1.01e-12
```

The generic-parity production tests additionally compare direct and resummed
theta/glasses global blocks and the numerical PBW oracle against the
coefficient recursion. These tests live in
`Code/test_ns_genus2_partition.py` and
`Code/test_ns_genus2_arbitrary_primary_parity.py`.

Therefore the achieved finite-order result is:

```text
direct PBW = NS c-recursion,
direct PBW = diagonal double-Virasoro implementation
             after theta-polarized star inversion (p_i=0 only),
current p_i=0 double-Virasoro mismatches in the checked range = 0,
generic-p auxiliary-fermion/double-Virasoro factorization = INCOMPLETE.
```

## 6. Why the ordinary-quotient control fails

This section records the diagnosis so the resolved confusion is not reopened.
It distinguishes a removable implementation-frame sign from the polarization
that the naive ordinary quotient omits. The \(\star\)-sign is derived from the
theta parity ledger; it is not inserted coefficient by coefficient.

### 6.1 No extra infinity-edge character remains

For a parity vector \(x=(x_0,x_1,x_\infty)\in(\mathbb Z_2)^3\), define

\[
 Q(x)=x_0x_1+x_0x_\infty+x_1x_\infty.
 \tag{6.1}
\]

The human note and every current checker use the same theta-channel
orientation,

\[
 \sigma_{\rm H}(x)=(-1)^{Q(x)}.
 \tag{6.2}
\]

An older backend version used the rephased character

\[
 \sigma_{\rm old}(x)=(-1)^{Q(x)+x_\infty}
 =\chi_\infty(x)\sigma_{\rm H}(x),
 \qquad \chi_\infty(x)=(-1)^{x_\infty}.
 \tag{6.3}
\]

That backend rephasing has been removed. It was never part of the human-note
definition. It also could not have produced the ordinary-quotient control
failure because \(\chi_\infty\) is a character,

\[
 \chi_\infty(x+y)=\chi_\infty(x)\chi_\infty(y),
 \tag{6.4}
\]

so it commutes with the polarized convolution and its inverse. The current
code is already in the literal frame (6.2), and the linear infinity sign must
not be listed as an unfinished convention issue.

### 6.2 The required sign is the polarization of \(Q\)

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

The separately defined SCA and Majorana coefficient series carry
\(\sigma_{\rm H}(s)\) and \(\sigma_{\rm H}(f)\), whereas the enlarged sewn
series carries the total branch parity through
\((-1)^{4n_0n_1+4n_0n_\infty+4n_1n_\infty}\), namely
\(\sigma_{\rm H}(t)\) on the branch primaries.  Equations (6.5)--(6.7) show
that the coefficient series require the polarization conversion in (4.2).
This does not alter the ordinary tensor product of the two theories; it is how
that product is represented after the quadratic theta signs have been
absorbed into the sewn series.

No change-of-basis hypothesis is needed for this step: (6.7) is already the
coefficientwise definition of the \(\star\)-product. The ordered
tensor-three-form crossing in the current convention ledger is a different
sign and occurs twice in the genus-two block, so it cannot replace the
bilinear polarization (6.6).

### 6.3 The first mismatch isolates this polarization exactly

At total twice-level

\[
 \ell=(0,1,2),
 \tag{6.8}
\]

the only nonconstant auxiliary-fermion contribution in the naive ordinary
convolution has

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
\(\rho_1(\phi_0,\phi_1,G_{-1/2}\phi_\infty)=-1\),

\[
 [\mathbf F_{\mathsf{SCA}}]_{(0,0,1)}=-\frac1{2h_\infty},
 \qquad
 [\mathbf F_{\mathsf F}]_{(0,1,1)}=-1.
 \tag{6.11}
\]

Thus the naive ordinary convolution assigns this split the contribution
\(+1/(2h_\infty)\), whereas the polarized convolution assigns
\(-1/(2h_\infty)\). Their polarized-minus-naive difference is
\(-1/h_\infty\). After naive ordinary fermion division this is precisely

\[
 [D_a-T_a^{\rm ord}]_{(0,1,2)}=-\frac1{h_\infty},
 \tag{6.12}
\]

which is (5.1). Thus the cross sign is exactly the polarization gap between
the two parity ledgers and is the required \(\star\)-convolution sign. It is
not a modification of the underlying tensor product.

This coefficient lies below both the first NS Kac contribution
\((r,s)=(3,1)\), which starts at twice-level \(3\) on one edge, and the first
ordinary Virasoro \((2,1)\) contribution, which starts at Virasoro level \(2\).
The failure is therefore independent of all recursion residues and fusion
polynomials. It isolates the omitted quadratic-sign polarization in the naive
control and does **not** indicate a missing sign in the NS \(c\)-recursion or
branching coefficient. Restoring that polarization through (4.2)--(4.3)
removes all nine control failures in the checked range.

## 7. Reproduction

```bash
PYTHONDONTWRITEBYTECODE=1 \
  python3 Code/ns_genus2_three_way_symbolic_check.py --max-twice-level 4

PYTHONPATH=Code PYTHONDONTWRITEBYTECODE=1 \
  python3 -m unittest Code/test_ns_genus2_three_way_symbolic_check.py
```

The same polarized convolution and the exact $20/20$ and $105/105$ checks are
recorded in the human note. The full two-channel partition frontier stated in
Section 0 is tested by the separate `ns_genus2_partition.py` workflow.
