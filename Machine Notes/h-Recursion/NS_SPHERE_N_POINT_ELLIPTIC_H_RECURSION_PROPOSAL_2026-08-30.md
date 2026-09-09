# NS sphere multipoint elliptic h-recursion: working proposal

Date: 2026-08-30.

Status: a parity-resolved proposal, with an initial five-point numerical
coefficient check. The general multipoint large-weight seed is not proved.
The human note has not been edited.

The subsequent [upper-component extension](NS_UPPER_COMPONENT_ELLIPTIC_H_RECURSION_2026-08-30.md)
allows external `G_{-1/2} phi` states. It records the changed parity labels,
effective conformal weights, and component-dependent polynomial seeds,
without changing the bottom-component convention developed here.

## Scope and attribution

We start with ordinary central charge c, bottom components of intrinsically
even N=1 NS primaries, and NS internal modules in the sphere comb channel.
Ramond insertions, upper external components, and a full supermoduli-space
construction are not included in this first step.

The bosonic multipoint elliptic recursion and its sphere prefactor are prior
work of Artemev and Khromov, arXiv:2603.08194v2, equations (3.16) and
(5.14)-(5.18). We do not claim that bosonic result as new. The four-point NS
elliptic recursion is due to Hadasz, Jaskolski, and Suchanek (HJS),
arXiv:0711.1619, section 6. Multipoint NS c-recursion was given by Belavin and
Geiko, arXiv:1806.09563. The task here is to formulate and test the
fixed-external-weight, multipoint *elliptic* NS h-recursion. Novelty of that
specific extension has not been established by an exhaustive search.

## 1. Coordinates, components, and parity labels

Put n external bottom components of weights D_1,...,D_n at

\[
(0,z,t_1,\ldots,t_{n-4},1,\infty),\qquad
m=n-3.
\]

There are m internal weights h_1,...,h_m. Let epsilon_i=0 or 1 specify the
parity of the descendant level on edge i: an allowed level is in
\(\mathbb Z_{\ge0}+\epsilon_i/2\). The corresponding relative three-form labels are

\[
\alpha_1=\epsilon_1,\quad
\alpha_v=\epsilon_{v-1}\mathbin\oplus\epsilon_v
\quad(2\le v\le m),\quad
\alpha_{m+1}=\epsilon_m.
\]

Thus there are 2^(n-3) component blocks. Use the fixed-parity trilinear-form
convention of `Human Notes/SCblock.tex` and `ns_human_convention.py`.
Odd blocks in this convention can differ by an overall sign from HJS.

The ordinary elliptic map is unchanged. With conventional complete elliptic
integral K(z)=(pi/2) 2F1(1/2,1/2;1;z), define

\[
q=e^{i\pi\tau},\qquad \tau=i\frac{K(1-z)}{K(z)},\qquad
u_j=\frac{\pi}{4K(z)}
\int_0^{(t_j-z)/(z(t_j-1))}
\frac{ds}{\sqrt{s(1-s)(1-zs)}}.
\]

Here u is the Artemev-Khromov coordinate. On our aligned sheet w_j=pi+2u_j.
For n>=5 the effective propagation factors are

\[
\rho_1=4e^{2iu_1}=-4e^{iw_1},\qquad
\rho_i=e^{2i(u_i-u_{i-1})}=e^{i(w_i-w_{i-1})}
\quad(2\le i\le m-1),
\]
\[
\rho_m=4q e^{-2iu_{m-1}}=-4q e^{-iw_{m-1}},\qquad
\prod_{i=1}^{m}\rho_i=16q.
\]

For n=4 use rho_1=16q. Half-integer powers require a spin-lift convention:
take positive square roots of the positive rho_i on the ordered real sheet
0<z<t_1<...<t_(n-4)<1, then analytically continue coherently. Do not choose
the square roots separately at each recursive call.

## 2. Sphere-to-elliptic normalization

Write

\[
c=\frac32+3Q^2,\quad Q=b+b^{-1},\quad
\delta_{\rm NS}=\frac{c-3/2}{24}=\frac{Q^2}{8}.
\]

For any kappa define the scalar coordinate factor

\[
\begin{aligned}
\Lambda_n^{(\kappa)}={}&
z^{\kappa/24-D_1-D_2}
(1-z)^{\kappa/24-D_2-D_{n-1}}\\
&\times\theta_3(q)^{\kappa/2
-4(D_1+D_2+D_{n-1}+D_n)
-2\sum_{j=1}^{n-4}D_{j+2}}\\
&\times\prod_{j=1}^{n-4}
[t_j(1-t_j)(t_j-z)]^{-D_{j+2}/2}.
\end{aligned}
\]

The branch convention here is the aligned real convention, with local
phases incorporated in the cap and insertion states. Define the NS elliptic
components by the exact normalization identity

\[
\boxed{
\mathcal F_{n,\boldsymbol\epsilon}^{S^2}
=\Lambda_n^{(c-3/2)}
\prod_{i=1}^{m}\rho_i^{h_i-\delta_{\rm NS}}
H_{n,\boldsymbol\epsilon}.}
\]

This is a definition of H, not a proof of its large-weight behavior.
The scalar normalization reduces to HJS equation (6.1) at n=4. Ordinary
conformal covariance of the inserted bottom components supplies the same
mobile-insertion Jacobians as in the bosonic problem. The replacement
c-1 -> c-3/2 alone does not specify the superconformal block: the parity
components and their regular part are essential.

## 3. Large-weight proposal, including the oscillator product

For four bottom NS components, HJS equations (6.4)-(6.5) give

\[
H_{4,0}(h;q)\longrightarrow\theta_3(q^2),\qquad
H_{4,1}(h;q)\longrightarrow0\qquad(h\to\infty).
\]

For n points consider h_i=H+a_i with all external weights, c, and a_i fixed.
The candidate extension is

\[
\boxed{
\lim_{H\to\infty}H_{n,\boldsymbol\epsilon}
=\delta_{\boldsymbol\epsilon,\boldsymbol0}\,\theta_3(q^2).}
\tag{Seed proposal}
\]

Its motivation is that leading normalized even interior vertices should
transport the oscillator state as the identity; odd vertex transitions
should be suppressed by large-weight normalization. The cap contribution
would then be the known four-point one. This is a motivation, not the missing
Ward-identity/power-counting proof. In particular, it must not be extrapolated
to upper external components: already their four-point regular parts differ
and can contain polynomial dependence on h.

### 3.1 Product form with the original geometric prefactor

To state the proposal in the same normalization as the bosonic pillow
matrix element, remove only the geometric factor:

\[
G_{n,\boldsymbol\epsilon}
:=(\Lambda_n^{(c)})^{-1}
\mathcal F_{n,\boldsymbol\epsilon}^{S^2},\qquad
P(q):=\prod_{j=1}^{\infty}(1-q^{2j}).
\]

The theta identity

\[
z(1-z)\theta_3(q)^{12}=16q\,P(q)^{12}
\]

and \(\prod_i\rho_i=16q\) give, for any constant a,

\[
\frac{\Lambda_n^{(c-a)}
\prod_i\rho_i^{h_i-(c-a)/24}}
{\Lambda_n^{(c)}\prod_i\rho_i^{h_i-c/24}}
=P(q)^{-a/2}.
\]

Consequently the proposed large-weight formula, for h_i=H+a_i with
all a_i fixed, is

\[
\boxed{
\lim_{\substack{H\to\infty\\h_i=H+a_i}}
\frac{G_{n,\boldsymbol\epsilon}}
{\prod_{i=1}^{m}\rho_i^{h_i-c/24}}
=
\delta_{\boldsymbol\epsilon,\boldsymbol0}
\mathcal C_{\mathrm{NS,pill}}(q).}
\tag{Pillow large-weight proposal}
\]

where the universal product is explicitly

\[
\boxed{
\mathcal C_{\mathrm{NS,pill}}(q)
=\frac{\theta_3(q^2)}{P(q)^{3/4}}
=\prod_{j=1}^{\infty}
\frac{(1-q^{4j})(1+q^{4j-2})^2}
{(1-q^{2j})^{3/4}}.}
\tag{NS pillow product}
\]

The last equality is the Jacobi triple product. All fractional powers are
continued from q=0, where this product is one. Thus the proposed replacement
in the bosonic formula is precisely

\[
\underbrace{\prod_{j\ge1}(1-q^{2j})^{-1/2}}_{\text{bosonic pillow}}
\quad\longmapsto\quad
\underbrace{\mathcal C_{\mathrm{NS,pill}}(q)}_{\text{NS pillow, all-bottom even seed}}.
\]

The n-point extension introduces no new dependence on the individual
insertion positions into this seed: they enter through the propagation
factors rho_i, while the universal product depends only on
\(q=\prod_i\rho_i/16\). For n=4 the formula reads

\[
G_{4,0}(h;q)\sim(16q)^{h-c/24}
\mathcal C_{\mathrm{NS,pill}}(q).
\]

This product is fixed algebraically by the known four-point HJS seed and
our chosen geometric normalization; its interpretation directly in terms
of cap oscillator overlaps is not being assumed as a proved derivation.

**Distinction from the torus trace.** The ordinary NS descendant character
at torus nome \(Q_{\mathrm{tor}}=q^2\) is
\(\prod_{j\ge1}(1+q^{2j-1})/(1-q^{2j})\). That trace counts all NS states,
whereas G here is a parity-resolved cap matrix element. Substituting the
trace character without an accompanying change of normalization would
already fail the four-point check: it has a q term, while

\[
\mathcal C_{\mathrm{NS,pill}}(q)
=1+\frac{11}{4}q^2+\frac{93}{32}q^4+O(q^6).
\]

### 3.2 Fully stripped block with unit seed

Define

\[
\widehat H_{n,\boldsymbol\epsilon}
:=\frac{H_{n,\boldsymbol\epsilon}}{\theta_3(q^2)}.
\]

Then the complete proposed recipe can be written in bosonic-style form:

\[
\boxed{
\mathcal F_{n,\boldsymbol\epsilon}^{S^2}
=\Lambda_n^{(c)}
\left[\prod_i\rho_i^{h_i-c/24}\right]
\mathcal C_{\mathrm{NS,pill}}(q)
\widehat H_{n,\boldsymbol\epsilon},\qquad
\widehat H_{n,\boldsymbol\epsilon}
\longrightarrow\delta_{\boldsymbol\epsilon,\boldsymbol0}.}
\tag{Unit-seed normalization}
\]

The sphere coordinate map and geometric factor are unchanged. After the
local Virasoro residue data have been replaced by their NS counterparts
(null half-levels, fusion polynomials, and parity flips), the new
multipoint large-weight input is just this universal pillow product.

## 4. Conditional closed recursion

At fixed generic c the NS Kac poles and null levels are

\[
h_{rs}=\frac{Q^2-(rb+s/b)^2}{8},\qquad
\ell_{rs}=\frac{rs}{2},\qquad r,s\ge1,\quad r+s\in2\mathbb Z.
\]

The null vector has parity pi_rs=rs mod 2. In particular (r,s)=(1,1)
is an h-pole and must not be omitted using the r>=2 range of a c-recursion.

For a residue on edge k, put

\[
h_j^*=h_j-h_k+h_{rs},\qquad
\boldsymbol h'=\boldsymbol h^*+\ell_{rs}\boldsymbol e_k,\qquad
\boldsymbol\epsilon'=\boldsymbol\epsilon
\mathbin\oplus\pi_{rs}\boldsymbol e_k.
\]

Let R_(rs)^(k,epsilon) denote the ordinary plane-block residue, evaluated
at the pole weights h*. It contains the inverse null-norm slope and the
fusion factor at each end of edge k.

The regular/singular decomposition is most transparent after removing
the primary propagation powers, but retaining the oscillator product:

\[
\widetilde G_{n,\boldsymbol\epsilon}
:=\frac{G_{n,\boldsymbol\epsilon}}
{\prod_i\rho_i^{h_i-c/24}}
=\mathcal C_{\mathrm{NS,pill}}(q)\,
\widehat H_{n,\boldsymbol\epsilon}.
\]

With h_i=H+a_i and fixed a_i, the proposed partial-fraction expansion is

\[
\boxed{
\widetilde G_{n,\boldsymbol\epsilon}(\boldsymbol h)
=\underbrace{\delta_{\boldsymbol\epsilon,\boldsymbol0}
\mathcal C_{\mathrm{NS,pill}}(q)}_{\text{regular part in }H}
+\underbrace{\sum_{k=1}^{m}
\sum_{\substack{r,s\ge1\\r+s\in2\mathbb Z}}
\frac{\rho_k^{rs/2}R_{rs}^{(k,\boldsymbol\epsilon)}(\boldsymbol h^*)}
{h_k-h_{rs}}
\widetilde G_{n,\boldsymbol\epsilon'}(\boldsymbol h')}
_{\text{singular part in }H}.}
\tag{Regular part plus recursive pole terms}
\]

Here "regular" refers to dependence on the common internal-weight variable
H, not to order in q. At each fixed elliptic order, under coefficientwise
rationality and generic simple-pole factorization, subtracting these
principal parts leaves a polynomial in H. The proposed finite large-H
limit fixes that polynomial to the constant seed shown above. There is no
additional o(1) term in this recursion: the finite-weight corrections are
precisely the pole terms, whose residues factor through shifted blocks.
The all-even regular part is the infinite product; the other edge-parity
components have zero regular part in the present bottom-component scope.

Since the product is independent of internal weights, dividing it out
gives the unit-seed recursion below without changing the residue kernels.
Equivalently, in the HJS-style normalization the candidate formula is

\[
\boxed{
H_{n,\boldsymbol\epsilon}(\boldsymbol h)
=\delta_{\boldsymbol\epsilon,\boldsymbol0}\theta_3(q^2)
+\sum_{k=1}^{m}\sum_{\substack{r,s\ge1\\r+s\in2\mathbb Z}}
\frac{\rho_k^{rs/2} R_{rs}^{(k,\boldsymbol\epsilon)}(\boldsymbol h^*)}
{h_k-h_{rs}}
H_{n,\boldsymbol\epsilon'}(\boldsymbol h').}
\]

Equivalently, the fully stripped block obeys the particularly simple form

\[
\boxed{
\widehat H_{n,\boldsymbol\epsilon}(\boldsymbol h)
=\delta_{\boldsymbol\epsilon,\boldsymbol0}
+\sum_{k=1}^{m}\sum_{\substack{r,s\ge1\\r+s\in2\mathbb Z}}
\frac{\rho_k^{rs/2}R_{rs}^{(k,\boldsymbol\epsilon)}(\boldsymbol h^*)}
{h_k-h_{rs}}
\widehat H_{n,\boldsymbol\epsilon'}(\boldsymbol h').}
\tag{Unit-seed elliptic NS recursion}
\]

There is no additional theta-function ratio in a residue: theta_3(q^2)
is independent of all internal weights, and the coordinates are the same
in the parent and child blocks. Dividing the original recursion by this
common factor therefore leaves every kernel unchanged.

The coordinates and external weights are unchanged on the right-hand side.
For r,s both odd, the edge parity flips and the two adjacent vertex labels
alpha_k and alpha_(k+1) both flip. For r,s both even no parity label flips.

For the ordered bottom-component convention used by the existing sphere
code, the explicit scalar kernel is

\[
R_{rs}^{(k,\boldsymbol\epsilon)}
=(-1)^{rs} A_{rs}(b)
P_{rs}^{\alpha_k}(a_L,b_L)
P_{rs}^{\alpha_{k+1}}(a_R,b_R),
\]

where the weight pairs are

\[
(a_L,b_L)=
\begin{cases}(D_1,D_2),&k=1,\\
(h_{k-1}^*,D_{k+1}),&k>1,
\end{cases}
\quad
(a_R,b_R)=
\begin{cases}(D_n,D_{n-1}),&k=m,\\
(h_{k+1}^*,D_{k+2}),&k<m.
\end{cases}
\]

In the same normalization,

\[
A_{rs}=\frac12
\prod_{\substack{p=1-r,\ldots,r\\v=1-s,\ldots,s\\
p+v\in2\mathbb Z\\(p,v)\ne(0,0),(r,s)}}
\frac{\sqrt2}{pb+v/b},
\]

and, writing lambda_a^2=Q^2-8a,

\[
P_{rs}^{\alpha}(a,b_0)=
\prod_{\substack{p=1-r,3-r,\ldots,r-1\\
v=1-s,3-s,\ldots,s-1\\
p+v-r-s\equiv2(1-\alpha)\pmod4}}
\frac{(\lambda_a-\lambda_{b_0}+pb+v/b)
(\lambda_a+\lambda_{b_0}+pb+v/b)}{8}.
\]

The symbol b_0 in the fusion polynomial denotes a weight, not the
super-Liouville parameter b. These products are implemented in
`Code/c_Recursion/ns_recursion_recipe.py`. The sign above is convention
specific; it is not an extra universal sign to append to an HJS formula.

## 5. Initial checks performed

The script
`Code/sphere_five_kummer_h_recursion/check_ns_sphere_five_elliptic_h_recursion.py`
compares the proposed five-point elliptic recursion with the independent
Belavin-Geiko c-recursion implementation, pulled back using the existing
exact five-point pillow coordinate series. It checks all four edge-parity
channels. At total elliptic degree four there are 45 coefficients, counting
half-integer edge powers; this is not a rectangular level-four check.

The two generic test cases are:

| Case | c | External weights | Internal weights |
|---|---|---|---|
| A | 14.7 | (0.31,0.42,0.53,0.47,0.28) | (0.73,1.10) |
| B | 21.3 | (0.22,0.61,0.39,0.74,0.45) | (1.13,0.85) |

At 75-digit working precision, with 60-digit comparison data, the maximum
relative errors were approximately 9.1e-61 and 3.0e-60 respectively.
Relative errors use max(1,abs(left),abs(right)) in the denominator.

The same script now also compares the unit-seed recursion for H_hat against
the c-recursion pullback divided by theta_3(q^2). For the same 45
coefficients per case, the maximum relative errors were 2.7e-60 (A) and
1.9e-60 (B). This verifies that the normalization change leaves the residue
kernels unchanged in the tested coefficients.

The explicit numerator/denominator product in section 3.1 was checked
symbolically through q^10. The geometric conversion factor was separately
checked at q=0.01,0.1,0.3 with maximum relative discrepancy 2.3e-75 at
75-digit precision. The q^10 test is a product-identity check, not an
extension of the five-point block comparison to order ten.

A separate seed check used the same external data and c, with internal
weights (H,H+0.37) in case A and (H,H-0.28) in case B. In every parity sector
the coefficientwise deviation decreased approximately as 1/H for
H=50,200,800. For case A, the all-even coefficient of p_1^2 p_2^2 was
1.9028270749, 1.9746992975, and 1.9936084467, approaching the predicted 2.

These are preliminary five-point checks, not a proof of the seed, not a
direct NS PBW audit of the new recursion, and not a six-point test.

Run the reproducible finite-weight comparison from the repository root:

```sh
python3 Code/sphere_five_kummer_h_recursion/check_ns_sphere_five_elliptic_h_recursion.py
```

## 6. Next checks and derivation targets

1. Derive the fixed-difference seed using the super-Virasoro Ward identities
   and large-H normalization, including the suppression of odd vertices.
2. Independently compare against NS PBW sewing, first at low half-levels.
3. Test six points, where the interior propagation factor has no endpoint 4
   and all eight edge-parity channels must be included.
4. Increase depth and generic parameter coverage before using the prototype
   as a production evaluator. Treat confluent Kac poles separately.
5. Add upper components and Ramond sectors only after their seeds and spin
   conventions are separately established.

Sources:

- https://arxiv.org/abs/2603.08194
- https://arxiv.org/abs/0711.1619
- https://arxiv.org/abs/1806.09563
