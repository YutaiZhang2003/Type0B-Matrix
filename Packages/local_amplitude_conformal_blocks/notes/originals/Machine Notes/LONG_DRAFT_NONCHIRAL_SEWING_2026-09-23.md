# Nonchiral sewing in the long draft's convention

For the subsequent NSRR coefficient normalization result, use
[the 2026-09-24 identity and mixed sphere audit](Genus%202/NSRR_NORMALIZATION_RESOLUTION_2026-09-24.md):
`c=(E,O)` with equal NS/R metrics, applied once. That result supplements
the general tensor algebra below; it does not supply a complete proof of
the physical interacting Ramond state-space reduction.

Derived and checked locally on 2026-09-23. Scope: genus-two theta and sphere
four-point sewing, including internal and external fermion parity.

The reference is [the actual long draft](/Users/yutaizhang/Desktop/Type0B-Matrix/draft-long/scblock_long.tex:101),
in particular its normalized forms, graph block, and Ward appendix. The derivation
below supplies the nonchiral coefficient matrix that is not written out there.
It agrees with the general sign identity in the separate arbitrary-plumbing
notes, but that identity was independently proved and exhaustively checked here.

**Result.** For these two-vertex graphs, expand the full three-point tensors in
the draft's chiral form bases. If their coefficient matrices are \(C_L,C_R\),
the coefficient multiplying the two graph blocks is

\[
\boxed{
M_{(\lambda_L,\lambda_R),(\tilde\lambda_L,\tilde\lambda_R)}
=(-1)^{\tilde s_Ls_R+E_{\rm ext}}
 (C_L)_{\lambda_L\tilde\lambda_L}
 (C_R)_{\lambda_R\tilde\lambda_R},\qquad
E_{\rm ext}=\sum_{a<b}\tilde\epsilon_a\epsilon_b .
}
\tag{1}
\]

Here \(s_v\) is the **absolute parity of the chiral form**, not necessarily its
label \(a\) or \(f\). The order in the external sum is the prescribed order of
full external fields. Spectrum sums/integrals, inverse primary multiplicity
metrics, primary powers of the plumbing parameters, and the common anomaly
factor multiply (1). Ramond state-space reductions must be included in the
physical input as explained below.

## 1. States, parity, and frames

All parity equations and exponents of \((-1)\) are modulo two. Write \(d(A)\)
for the number of \(G\)-modes in a PBW word, modulo two. Then

\[
\epsilon(L_{-A}\phi)=p_\phi+d(A),\qquad
\epsilon(L_{-A}w^\alpha)=d(A)+\alpha,
\quad \alpha=0\ (w^+),\;1\ (w^-).
\tag{2}
\]

The antiholomorphic states have independent tilded parities. A full homogeneous
state \(X=x\otimes\tilde x\) has parity \(Q=\epsilon+\tilde\epsilon\).
Thus an odd full external operator does not imply that both chiral factors
are odd. General external fields are expanded into homogeneous components.

The ordered three-point slots are \((\infty,1,0)\). An NSRR vertex always has
its NS leg first. Keep the draft's seeds

\[
\rho_0(\phi_1,\phi_2,\phi_3)=1,\quad
\rho_1(\phi_1,G_{-1/2}\phi_2,\phi_3)=1,
\]
\[
\rho_0^\eta(\phi,w^+,w^+)=1,\quad
\rho_0^\eta(\phi,w^-,w^-)=\eta,
\quad
\rho_1^\eta(\phi,w^+,w^-)=1,\quad
\boxed{\rho_1^\eta(\phi,w^-,w^+)=-i\eta.}
\tag{3}
\]

Use \(\zeta_e\) for the tube's sign below, to distinguish it from the local
form label \(\eta_v\). The tube contributes \(\zeta_e^{\epsilon_e}\), including
the primary or Ramond ground parity.

The form degrees are

\[
s_v= a_v+p_{v1}+p_{v2}+p_{v3}\quad(\mathrm{NSNSNS}),\qquad
s_v=f_v+p_{v,\mathrm{NS}}\quad(\mathrm{NSRR}).
\tag{4}
\]

An even full three-point tensor requires

\[
(C_v)_{\lambda\tilde\lambda}=0\quad\hbox{unless}\quad
s_v(\lambda)=\tilde s_v(\tilde\lambda).
\tag{5}
\]

In particular, \(a=\tilde a\) or \(f=\tilde f\) is not a universal selection
rule when intrinsic primary parities differ between the two chiral factors.

There is an explicit extension of the all-NS Ward forms to odd primaries,
with the same mode-action matrices and the same two seed normalizations:

\[
\rho_a^{(p_1,p_2,p_3)}(A,B,C)
=(-1)^{p_2d(C)}\rho_a^{(0,0,0)}(A,B,C).
\tag{6}
\]

Indeed, every \(G\) on the third leg changes this factor by \((-1)^{p_2}\),
exactly supplying the primary contribution to the appendix's
\((-1)^{\epsilon_2}\). The first and second legs introduce no such change.
Virasoro identities and the middle-leg seed are preserved. In an NSRR vertex,
changing the NS primary parity changes (4), the graph sign, and tube parity;
it does not change the numerical Ward form, since the NS leg is first.
Both statements were checked against the appendix identities.

Use a consistent antiholomorphic frame. In the literal conjugate frame,
\(\tilde\rho=\rho^*\) at conjugate data; hence the anti odd seed is
\(+i\tilde\eta\), and the anti Ramond ground Gram is
\(\operatorname{diag}(1,-i)\). The holomorphic ground Gram in the draft's
\(w\) basis is \(\operatorname{diag}(1,i)\), as follows from its \(G_0\) action
and contravariant pairing. Using the same analytic \(-i\) frame on both sides
is another possible choice, but then the anti metric and coefficients must
be transported too, and the anti block is not simply the complex conjugate.

## 2. Converting physical three-point data to the coefficient matrix

Fix the graded tensor basis, with all holomorphic factors ordered before all
antiholomorphic factors. For three full states, define the coefficient matrix
by the unambiguous component equation

\[
T_v(X_1,X_2,X_3)
=(-1)^{\chi_v}
\sum_{\lambda,\tilde\lambda}
(C_v)_{\lambda\tilde\lambda}
\rho_\lambda(x_1,x_2,x_3)
\tilde\rho_{\tilde\lambda}(\tilde x_1,\tilde x_2,\tilde x_3),
\quad
\chi_v=\sum_{i<j}\tilde\epsilon_i\epsilon_j.
\tag{7}
\]

This defines the phases of \(C\); they should not be guessed from absolute
squares or from the names “even” and “odd”. These are coefficients of the
**full** three-point tensor, not additional holomorphic structure constants.

For all-NS vertices, evaluate (7) on the two holomorphic seeds and the two
anti seeds. With a \(G_{-1/2}\) in the middle slot for seed 1,

\[
C_{a\tilde a}
=(-1)^{\tilde p_1(p_2+a+p_3)+(\tilde p_2+\tilde a)p_3}
T_{a\tilde a}.
\tag{8}
\]

The seed states here are literal tensor products. If the supplied physical
component is written instead as \(G^a\tilde G^{\tilde a}\mathcal O_2\), use
the graded action first: it differs from
\(G^a\phi_2\otimes\tilde G^{\tilde a}\tilde\phi_2\) by
\((-1)^{p_2\tilde a}\). Normalize primaries consistently with the two-point
input as well.

For NSRR, order the ground pairs as \((++,--,+-,-+)\), and form labels as
\((0,+),(0,-),(1,+),(1,-)\). The draft gives the seed evaluation matrix

\[
A=\begin{pmatrix}
1&1&0&0\\1&-1&0&0\\0&0&1&1\\0&0&-i&i
\end{pmatrix},\qquad
A^{-1}=\frac12\begin{pmatrix}
1&1&0&0\\1&-1&0&0\\0&0&1&i\\0&0&1&-i
\end{pmatrix}.
\tag{9}
\]

Let \(T_{(\alpha\beta),(\tilde\alpha\tilde\beta)}\) be the full ground
three-point table in the chosen tensor basis. Remove its interleaving sign:

\[
H_{(\alpha\beta),(\tilde\alpha\tilde\beta)}
=(-1)^{\tilde p_{\mathrm{NS}}(\alpha+\beta)+\tilde\alpha\beta}
T_{(\alpha\beta),(\tilde\alpha\tilde\beta)}.
\]
\[
\boxed{C=A^{-1}H(\tilde A^{-1})^T.}\tag{10}
\]

Here \(\tilde A=A^*\) in the conjugate frame. For example the holomorphic
dual odd combination is \((H_{+-}+i\eta H_{-+})/2\). The plus sign in this
**dual** combination follows from the minus sign in the draft's seed.
Equation (10) fixes all factors of two associated with this basis change.
It does not license an extra normalization factor for the partition function.

## 3. Why the nonchiral sign is (1)

For an even chiral Gram matrix and its anti counterpart, the graded full
pairing is

\[
\mathcal B_{I\tilde I;J\tilde J}
=d_e(-1)^{\tilde\epsilon_I\epsilon_J}
G_{IJ}\tilde G_{\tilde I\tilde J},\qquad
\mathcal B^{I\tilde I;J\tilde J}
=d_e^{-1}(-1)^{\epsilon_I\tilde\epsilon_I}
G^{IJ}\tilde G^{\tilde I\tilde J}.
\tag{11}
\]

The scalar \(d_e\) denotes an optional tensor-basis/multiplicity normalization;
a matrix multiplicity metric is contracted instead if necessary. In particular,
an NS primary of full norm \(D_e\), directly identified with
\(\phi\otimes\tilde\phi\) with unit chiral norms, has
\(d_e=(-1)^{p_e\tilde p_e}D_e\). Alternatively rescale the full primaries and
their three-point tensors to the \(d_e=1\) tensor basis. Ramond ground norms
are already in \(G,\tilde G\); they must not be divided out again.

The direct full state sum has the permutation sign \(K(\epsilon+\tilde\epsilon)\).
Factorization of every inverse pairing supplies
\(\sum_e\epsilon_e\tilde\epsilon_e\), and (7) supplies \(\sum_v\chi_v\).
The separate chiral blocks already contain \(K(\epsilon)\) and
\(K(\tilde\epsilon)\). The remaining exponent is consequently

\[
S=K(\epsilon+\tilde\epsilon)+K(\epsilon)+K(\tilde\epsilon)
 +\sum_e\epsilon_e\tilde\epsilon_e+\chi_L+\chi_R.
\]

To evaluate it, compare regrouping holomorphic and anti factors before and
after the graph permutation. In vertex order, the regrouping exponent is
\(\chi_L+\chi_R+\tilde s_Ls_R\). In paired-edge order followed by the external
fields, it is \(\sum_e\epsilon_e\tilde\epsilon_e+E_{\rm ext}\): each internal
edge occurs twice, so exchanges of whole pairs have no remaining sign.
Subtracting these two descriptions proves

\[
\boxed{S=\tilde s_Ls_R+E_{\rm ext}.}\tag{12}
\]

This proof is independent of descendant level, weights, and NS/R labels.
It also proves that an even full theory has zero correlator for odd total
external parity: add the even-vertex selection rules, and all internal
parities appear twice.

## 4. Genus-two theta prescription

Order the vertices as \((1_L,2_L,3_L)\), \((1_R,2_R,3_R)\), starting from
paired order \((1_L,1_R,2_L,2_R,3_L,3_R)\). Then

\[
K_\Theta(\epsilon)=\epsilon_1\epsilon_2+
\epsilon_1\epsilon_3+\epsilon_2\epsilon_3.
\]

Let \(F_\Theta^{\lambda_L\lambda_R}\) be exactly the descendant graph block
in the draft, including this sign and \(\zeta_e^{\epsilon_e}\), but excluding
primary \(q_e^{h_e}\). Even inverse pairings require the same parity at both
ends of each edge, so \(s_L=s_R=s\). Equation (5) gives
\(\tilde s_L=\tilde s_R=s\). Therefore

\[
Z_\Theta=\mathcal A_\Theta
\sum_{\rm spectrum}\left(\prod_{e=1}^3
\frac{q_e^{h_e}\tilde q_e^{\tilde h_e}}{d_e}\right)
\sum_{\lambda_L,\lambda_R,\tilde\lambda_L,\tilde\lambda_R}
(-1)^s C_L^{\lambda_L\tilde\lambda_L}
C_R^{\lambda_R\tilde\lambda_R}
F_\Theta^{\lambda_L\lambda_R}
\tilde F_\Theta^{\tilde\lambda_L\tilde\lambda_R}.
\tag{13}
\]

For a continuous spectrum, use the specified spectral measures in place of
the sum. \(\mathcal A_\Theta\) is the common conformal-anomaly/frame factor.
The draft's blocks omit the primary powers; an additional cylinder Casimir
factor must only be inserted as part of a stated change of frame.

For all-NS tubes, put \(P=p_1+p_2+p_3\), \(\tilde P=\tilde p_1+\tilde p_2+\tilde p_3\).
The only nonzero labels have \(a_L=a_R=a\) and
\(\tilde a_L=\tilde a_R=a+P+\tilde P\). Their coefficient is

\[
(-1)^{a+P} C_L^{a,\tilde a}C_R^{a,\tilde a}.
\tag{14}
\]

For NSRR tubes, label the NS edge 1. Write its primary parities as \(p,\tilde p\).
Then \(f_L=f_R=f\), \(\tilde f_L=\tilde f_R=f+p+\tilde p\), and

\[
M_{(f,\eta_L,\eta_R),(\tilde f,\tilde\eta_L,\tilde\eta_R)}
=(-1)^{f+p}
C_L^{(f,\eta_L),(\tilde f,\tilde\eta_L)}
C_R^{(f,\eta_R),(\tilde f,\tilde\eta_R)}.
\tag{15}
\]

Keep all allowed off-diagonal entries in the \(\eta\) labels. Replacing (15)
by a sum of absolute squares is an additional physical assertion about \(C\),
not a consequence of the chiral block construction.

A direct ground-state check of the unprojected chiral NSRR theta block gives

\[
F_{0}^{\eta\eta'}[0,0,0]
=\zeta_1^p(1+\eta\eta'\zeta_2\zeta_3),
\qquad
F_{1}^{\eta\eta'}[0,0,0]
=i(-1)^p\zeta_1^p(\eta\eta'\zeta_2-\zeta_3).
\tag{16}
\]

Both the \(-i\eta\) seed and the Ramond inverse ground metrics are needed
for this result. It is a useful check before inserting a physical projector.

## 5. Sphere four-point prescription and the first coefficients

Prescribe the full external order \((4,3,2,1)\). For the standard NS-exchange
frame take \(v_L=(4,3,e_L)\), \(v_R=(e_R,2,1)\). Starting from
\((e_L,e_R,4,3,2,1)\), the chiral graph permutation has \(K=0\).
This includes four NS fields and two NS fields at 4,3 with two R fields at 2,1.

Use the draft's Ward forms to build

\[
F^{\lambda_L\lambda_R}(q)
=\sum_{I,J}q^{|I|}\zeta^{\epsilon_I}
\rho_{\lambda_L}(x_4,x_3,x_I)G^{IJ}
\rho_{\lambda_R}(x_J,x_2,x_1).
\tag{17}
\]

The nonchiral answer in the marked plumbing frame is

\[
\mathcal G_4^{\rm sew}=\mathcal A_4
\sum_{\rm spectrum}\frac{q^h\tilde q^{\tilde h}}{d_e}
\sum_{\lambda_L,\lambda_R,\tilde\lambda_L,\tilde\lambda_R}
(-1)^{\tilde s_Ls_R+E_{4321}}
C_L^{\lambda_L\tilde\lambda_L} C_R^{\lambda_R\tilde\lambda_R}
F^{\lambda_L\lambda_R}\tilde F^{\tilde\lambda_L\tilde\lambda_R},
\tag{18}
\]
\[
E_{4321}=\tilde\epsilon_4(\epsilon_3+\epsilon_2+\epsilon_1)
 +\tilde\epsilon_3(\epsilon_2+\epsilon_1)+\tilde\epsilon_2\epsilon_1.
\]

To express the standard NS-exchange correlator at \((\infty,1,z,0)\), set
\(q=z\) and include the external scaling factor
\(z^{-\Delta_2-\Delta_1}\tilde z^{-\tilde\Delta_2-\tilde\Delta_1}\), with
\(\Delta_a=h_a+|A_a|\) for the chosen external states. Thus the leading
internal factor is \(z^{h-\Delta_2-\Delta_1}\). For other slot arrangements,
transport the marked coordinates and spin lifts as well as their Koszul sign.

For four NS bottom components with even chiral primaries and \(\zeta=1\),
the draft's precise seeds imply

\[
F^{00}(q)=1+
\frac{(h+h_3-h_4)(h+h_2-h_1)}{2h}q+O(q^2),
\qquad
\boxed{F^{11}(q)=-\frac{q^{1/2}}{2h}+O(q^{3/2}).}
\tag{19}
\]

The minus sign is fixed by
\(\rho_1(\phi_4,\phi_3,G_{-1/2}\phi_h)=-1\),
\(\rho_1(G_{-1/2}\phi_h,\phi_2,\phi_1)=1\), and the norm \(2h\).
For an odd primary in the middle slot of the left vertex, the first value
is instead \(-(-1)^{p_3}\), by (6). A conventional sphere block with positive
odd leading term is a redefinition of this block; its coefficient matrix
must be transported with it.

For NS4, NS3, \(w_2^+,w_1^+\), still with even NS primaries,

\[
\rho_1^\eta(G_{-1/2}\phi_h,w_2^+,w_1^+)
=e^{3\pi i/4}(\beta_1-\eta\beta_2),
\]
\[
F^{1,(1,\eta)}(q)
=-\frac{e^{3\pi i/4}(\beta_1-\eta\beta_2)}{2h}q^{1/2}
 +O(q^{3/2}).
\tag{20}
\]

This follows directly from the long draft's \(G_0\) action and \(-i\eta\)
seed. If the external R grounds have chiral bits \(\alpha_2,\alpha_1\), the
right form instead obeys \(f_R=a_L+\alpha_2+\alpha_1\) for bottom NS3, NS4.
An external NS descendant also contributes its \(G\)-word parity. Equation
(18), with absolute form degrees, handles all these cases without a new rule.

The finite tests also cover four R external fields with NS exchange and
mixed external fields with R exchange. To keep NS first, their vertices are
respectively \((e_L,3,4),(e_R,2,1)\) and \((4,3,e_L),(1,2,e_R)\).
Their chiral Koszul exponents are respectively
\(\epsilon_e(\epsilon_3+\epsilon_4)+\epsilon_3\epsilon_4\) and
\(\epsilon_e(\epsilon_1+\epsilon_2)+\epsilon_1\epsilon_2\).
The nonchiral formula (18) is unchanged; the graph blocks include the
appropriate chiral exponent. These are marked-coordinate sewing checks,
not a comparison of s and t channels in a common global coordinate frame.

## 6. Physical state spaces and changes of convention

Equations (13) and (18) use the factorized completeness pairing (11).
The SCFT must supply the spectrum, measures, full three-point tensors and
actual Ramond multiplicities or state-space reductions. The tensor-product
Ramond ground space has four components, while a physical theory can use a
smaller representation. For example, the super-Liouville reduction is stated
in equation (7) of [Suchanek, arXiv:1012.2974](https://arxiv.org/pdf/1012.2974).
Its phases cannot be imported while keeping a different metric or vertex
frame fixed.

There is a direct prescription even when a physical edge is specified in a
smaller basis. Let \(J_L,J_R\) embed the chosen physical states at the two
ends into the respective tensor-product bases, and let \(D^{ab}_{\rm phys}\)
be their ordered inverse physical pairing. Replace the ambient completeness
kernel by

\[
\mathcal P^{I\tilde I;J\tilde J}
=\sum_{a,b}(J_L)^{I\tilde I}{}_a\,
D_{\rm phys}^{ab}\,(J_R)^{J\tilde J}{}_b .
\tag{21}
\]

Use the dual embedding appropriate to each end. Apply (21) in the full
graded state sum before factorizing; equivalently extend the physical local
tensors with these embeddings and inverse pairings consistently. Additional
edge intertwiners must be retained if the kernel does not reduce to the
chosen ordinary chiral-block basis. The data \(J,D_{\rm phys}\) are physical
input, not something fixed by the four chiral seeds alone.

A simpler, explicitly checkable projection is onto a specified total edge
parity \(\kappa=\pm1\):

\[
\Pi_\kappa=\tfrac12(1+\kappa(-1)^{F+\tilde F}).
\]

Its effect on an edge is the weighted sum of its existing spin lifts and
both reversed spin lifts, with coefficients \(1/2,\kappa/2\). On theta,

\[
Z_{\{\kappa_e\}}=2^{-3}\sum_{t_1,t_2,t_3=0}^1
\left(\prod_e\kappa_e^{t_e}\right)
Z(\{(-1)^{t_e}\zeta_e\},\{(-1)^{t_e}\tilde\zeta_e\}).
\tag{22}
\]

This parity projection is not the Ramond small-representation reduction.
Neither (9) nor (22) establishes a universal factor of four between the
earlier local NSRR and all-NS numerical partition functions.

Finally, if chiral form columns are changed by \(\rho'=U\rho\) and
\(\tilde\rho'=V\tilde\rho\), then

\[
C'=U^{-T}CV^{-1}.
\tag{23}
\]

Transport Gram matrices, state frames, and coordinate/spin lifts too if those
are changed. The full answer, including interference terms, is invariant.
At degenerate weights use nondegenerate quotient modules or justified
limits instead of inverting a singular Gram matrix.

The practical sequence is: fix the physical states and ordered local frames;
convert their full three-point tensors using (7)--(10); compute the draft's
chiral blocks with total parities (2)--(6); assemble (13) or (18), including
the actual edge completeness kernels; and then insert the spectral measure,
primary powers, and common geometric factor. A comparison between channels
must use the same physical data and transport both answers to the same
surface metric and external-coordinate frame.

## 7. Local verification and reproducibility

The [isolated check implementation](/Users/yutaizhang/Desktop/Type0B-Matrix/C++/experiments/long_draft_sewing_2026-09-23/check_sewing.py)
contains a reusable `coefficient_matrix` function implementing (1).
The [native exporter](/Users/yutaizhang/Desktop/Type0B-Matrix/C++/experiments/long_draft_sewing_2026-09-23/export_vertices.cpp)
uses the frozen GitHub SCA Ward/Gram code matching the long draft, rather than
the retained local NR vertex implementation. It works at 60 decimal digits.
Full tensor contractions and their independently assembled chiral
factorizations are compared using complex double precision.

The checks include:

- The sign identity for all 288 theta slot orders and 64 parity assignments,
  and all 432 sphere slot orders and 1,024 parity assignments: **460,800 exact
  identities**, with no numerical tolerance.
- **3,456 Ward-identity checks**, including all intrinsic all-NS primary
  parities and both NS primary parities at NSRR vertices; maximum scaled
  residual below \(4\times10^{-61}\).
- **68 theta full-tensor comparisons** for all-NS and NSRR, and **5,632 sphere
  comparisons** for all-NS, mixed NS exchange, four-R NS exchange, and mixed
  R exchange, including external NS superpartners and odd full operators.
  NS descendants reach level \(3/2\); R descendants reach level 1, including
  non-diagonal descendant Gram blocks. The maximum scaled discrepancy is
  below \(4\times10^{-15}\).
- Direct inversion of full graded Gram matrices, the Ramond ground tensor
  conversion (10), the analytic coefficients (16), (19), (20), independent
  holomorphic/anti basis changes, and all eight specified theta parity
  projections (22).
- Negative controls: omitting the residual sign, dropping the external-order
  sign, discarding Ramond off-diagonal couplings, or using the wrong odd ground
  frame each produces a substantial discrepancy.

The comparisons use general complex even three-point tensors, independent
holomorphic/anti primary parities, and complex plumbing values. They do not
assume a diagonal absolute-square decomposition. Numerical residuals and
source hashes are recorded in
[results.json](/Users/yutaizhang/Desktop/Type0B-Matrix/C++/experiments/long_draft_sewing_2026-09-23/results.json).

These tests validate the graded factorization and the draft's low-level
conventions. They do not establish crossing for arbitrary input constants,
a particular physical Ramond reduction, the super-Liouville factor-of-four
normalization, or equality of different genus-two channels. Those are
additional dynamical/state-space tests once that SCFT's input is fixed.

A subsequent [BRY NS sphere check](</Users/yutaizhang/Desktop/Type0B-Matrix/Machine Notes/LONG_DRAFT_BRY_SPHERE_CHECK_2026-09-23.md>)
supplies that physical input for NS super-Liouville at b=1. It verifies the
full G,H,J decompositions and numerical crossing, including external
superpartners. At the import boundary the normalized graded coefficients
are C_0=C_BRY and C_1=i*tilde_C_BRY; inserting the real odd BRY function
directly into the graded formula gives the wrong answer. This extends the
checks to that NS sphere specialization, without resolving the physical
Ramond reduction or the genus-two factor of four.

To regenerate from the repository root:

```sh
VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 C++/experiments/long_draft_sewing_2026-09-23/check_sewing.py --build
```

Omit `--build` to reuse the saved native data. The macOS build needs clang++,
the existing Homebrew MPC/MPFR/GMP libraries, and NumPy for the Python checks.
The long draft and production implementations have not been edited.
