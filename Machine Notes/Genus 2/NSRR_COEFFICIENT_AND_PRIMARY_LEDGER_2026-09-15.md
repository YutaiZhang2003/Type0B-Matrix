# NSRR coefficient matrix and primary-power ledger

Date: 2026-09-15. This follows the user's descendant-only definition of the
conformal block. It records the supplied three-point constants before any
sewing simplification, and keeps the primary propagation outside both the
block and the coefficient matrix.

## 1. Definition to use throughout

Let \(\gamma\) label the decomposition channel, including its plumbing
chart and edge sectors. Let \(A,B\) label chiral and antichiral blocks.
The definition is

\[
\boxed{
Z_\gamma=\int d\mu_\gamma\sum_{A,B}
M^{\gamma}_{AB}\,
P^{\gamma}_A(q)\,\widetilde P^{\gamma}_B(\tilde q)\,
F^{\gamma}_A(q)\,\widetilde F^{\gamma}_B(\tilde q),
\qquad
P^{\gamma}_A=\exp\!\left(\sum_i h^{\gamma}_{i,A}\ell_i^\gamma\right),
\quad \ell_i^\gamma=\Log q_i^\gamma .
}
\tag{1}
\]

Here \(F_A=\sum_{\boldsymbol N}F_A[\boldsymbol N]
\prod_i q_i^{N_i}\) contains **descendant levels only**. The three-point
constants, inverse pairings and sewing phases enter \(M\). The factor
\(P_A\) is neither part of \(F_A\) nor part of \(M\). Denote the
propagated amplitude by \(\mathcal B_A=P_AF_A\) when it is useful; never
silently rename it \(F_A\).

The source definitions are `NSblockThetaDefinition` at
[SCblock.tex:709](</Users/yutaizhang/Desktop/Type0B-Matrix/Human Notes/SCblock.tex:709>),
`NSblockGlassesDefinition` at line 901, and `Rblock` at line 1209.
The full all-NS decompositions at lines 715 and 907 explicitly place
\(\prod_i q_i^{h_i}\tilde q_i^{\tilde h_i}\) outside the blocks.
The prose before `Rblock` calls it glasses, but its displayed two
\(\rho(NS,R,R)\) tensors and three connecting edges have the theta
topology. This ledger follows the displayed definition and the implemented
theta contraction; it does not relabel it as the glasses formula.

For the current eight NSRR channels \(A=(f,\eta_L,\eta_R)\), all eight
share the same three SCA primary weights at fixed momenta. Therefore their
\(P_A\) can be factored out of the eight-channel sum. This does **not**
allow factoring a common \(P\) across different momentum triples,
decomposition channels, or branching primaries.

If a literature basis change is stated for propagated amplitudes,
\(\mathcal B'=U\mathcal B\), then the descendant blocks obey
\(F'=D_{P'}^{-1}U D_P F\), where \(D_P\) is diagonal with entries
\(P_A\). The ratio of the primary factors must accompany a weight-changing
basis conversion. For a sphere or elliptic block, its stated external-weight
and coordinate prefactors must also be matched before using it as this
plumbing block.

## 2. Weights, slots, and channel dependence

Human slots are \((1,2,3)=(\infty,1,0)\). The geometry arrays are ordered
\((0,1,\infty)\). Reverse **all** edge data together: \(q\), momenta,
sectors, lifts and logarithms.

For the positive-momentum Liouville convention used in the computation,

\[
Q=b+b^{-1},\qquad c=\tfrac32+3Q^2,
\qquad h_{\rm NS}(p)=\frac{Q^2}{8}+\frac{p^2}{2},
\qquad h_{\rm R}(p)=\frac{c}{24}+\frac{p^2}{2}
=h_{\rm NS}(p)+\frac1{16}.
\tag{2}
\]

| Channel | Sectors in geometry order | Primary propagation |
|---|---|---|
| Current NSRR source | R, R, NS | \(\exp[h_R(p_0)\ell_0+h_R(p_1)\ell_1+h_{NS}(p_\infty)\ell_\infty]\) |
| Current all-NS target | NS, NS, NS | \(\exp[\sum_{e=0,1,\infty}h_{NS}(p'_e)\ell'_e]\) |

The target \(q'_e,p'_e\) belong to its own decomposition. Crossing is an
identity between integrated decompositions; it is not a pointwise
identification of source and target momenta. Even at an artificial common
\(q,p\), the primary factors would differ by

\[
\frac{P_{RRNS}}{P_{NSNSNS}}
=\exp\!\left(\frac{\ell_0+\ell_1}{16}\right).
\tag{3}
\]

Equation (3) is a bookkeeping example, not the source-to-target geometry
map. Writing a principal power of \(q_0q_1\) could change its phase;
the sum of the selected logarithms is the definition.

For arbitrary complex weights and logarithms, track

\[
\log|P_A|=\Re\sum_i h_{i,A}\ell_i,
\qquad \arg P_A=\Im\sum_i h_{i,A}\ell_i\pmod{2\pi}.
\tag{4}
\]

When the weights are real and \(\tilde\ell_i=\bar\ell_i\), an individual
pair has

\[
P_A\widetilde P_B
=\prod_i |q_i|^{h_{i,A}+\tilde h_{i,B}}
\exp\!\left[i\sum_i(h_{i,A}-\tilde h_{i,B})\arg q_i\right].
\tag{5}
\]

Only matching left/right weights eliminate this phase. The present
pointwise adapter uses principal logarithms. A continued calculation must
transport the logarithms for **both** the primary and the fractional
descendant powers. A winding changes the primary by
\(\exp(2\pi i\sum_i n_i h_i)\); it is not an adjustment of \(M\).

## 3. Channel-dependent three-point products

With even NS primaries, the Human Note gives, suppressing only the
spectral integral,

\[
Z_\Theta=P_\Theta\widetilde P_\Theta
\sum_a(-1)^a C^{L,(a)}_{123}C^{R,(a)}_{123}
F^\Theta_a\widetilde F^\Theta_a,
\tag{6}
\]

whereas the glasses formula is

\[
Z_{\rm Gl}=P_{\rm Gl}\widetilde P_{\rm Gl}
\sum_a(-1)^a C^{L,(a)}_{131}C^{R,(a)}_{232}
F^{\rm Gl}_a\widetilde F^{\rm Gl}_a.
\tag{7}
\]

For nonzero intrinsic primary parities, retain the note's respective
signs \((-1)^{a+p_1+p_2+p_3}\) and \((-1)^{a+p_1}\). The repeated
indices in (7) are essential. The glasses coefficient cannot be replaced
by a square of a theta three-point constant.

The existing NS conversion \(C_{HN}^{(1)}=i\widetilde C_{BRY}\) yields
\(-C_{HN}^{L,(1)}C_{HN}^{R,(1)}
=+\widetilde C^L_{BRY}\widetilde C^R_{BRY}\).
This leaves the primary factor untouched. For an odd block, a first term
at level \(1/2\) belongs to \(F\); it does not change the supermultiplet's
primary weight to \(h+1/2\).

## 4. NSRR: start from the supplied constants

Keep the two pants distinct:

\[
E_v=C_{{\rm even},v},\quad O_v=C_{{\rm odd},v},\qquad
d_{v,+}=\frac{E_v+O_v}{2},\quad
d_{v,-}=\frac{E_v-O_v}{2},\qquad
c_{v,+}=\frac{E_v}{2},\quad c_{v,-}=\frac{O_v}{2}.
\tag{8}
\]

The \(d\)'s label physical equal-family \(R^\pm R^\pm NS\)
three-point functions; the \(c\)'s multiply the \(\eta=\pm\) chiral
three-form basis. These are different labels. The supplied functions
`rr_ns_structure_constants` and `GenericSuperLiouvilleConstants.rr_ns_constants`
return \((E,O)\). A helper's name containing `hjs` does not remove the
factors in (8).

For ordered NR vertex operators, the literature gives

\[
R_{v,\eta}^{+}=\frac{c_{v,\eta}}{\sqrt2}
 (V_0^{\eta}\otimes\bar V_0^{\eta}
  -i V_1^{\eta}\otimes\bar V_1^{\eta}),\qquad
R_{v,\eta}^{-}=\frac{\eta c_{v,\eta}}{\sqrt2}
 (V_0^{\eta}\otimes\bar V_1^{\eta}
  +V_1^{\eta}\otimes\bar V_0^{\eta}).
\tag{9}
\]

These fix the local phases before any genus-two reduction.
[HJS, (4.13)](https://arxiv.org/pdf/0810.1203#page=13);
use the corrected NR/RN conventions in
[Suchanek, (28)–(31)](https://arxiv.org/pdf/1012.2974#page=11).

### Keep the four parity indices

Let \(a,c\) be the operator parities at the left/right holomorphic
vertices and \(b,d\) their antiholomorphic counterparts. At the primary
index contraction on the middle R edge, the vertex product is

\[
W^{\eta\eta'}_{ac;bd}
=\sum_{\epsilon=\pm}
\ell^{\epsilon,\eta}_{ab}\ell^{\epsilon,\eta'}_{cd}.
\]

Here \(\ell\) includes \(1/\sqrt2\) and the phases in (9), while the
two \(c\)'s are kept outside. In row/column order \((00,01,10,11)\),

\[
\boxed{
c_{L,\eta}c_{R,\eta'} W^{\eta\eta'}
=\frac{B_{L,\eta}B_{R,\eta'}}8
\begin{pmatrix}
1&0&0&k\\
0&-i&k&0\\
0&k&-i&0\\
k&0&0&-1
\end{pmatrix},\qquad
k=\eta\eta',\quad B_+=E,\ B_-=O.
}
\tag{10}
\]

Equation (10) is an **unreduced vertex coefficient tensor**, not the
final \(M\) on the eight saved genus-two blocks. Matrix-element Koszul
signs, the remaining edges and the ordered-slot/BPZ conversion are still
to be contracted. In particular, the indices \((01),(10)\) cannot be
discarded by asserting that the final blocks have matching form parity.
Any such elimination must follow from the actual contraction and Ward
relations. A specified relative family weight on this edge would enter
as \(k\mapsto\chi k\); the global fixed-spin prescription is not inferred
from this observation.

The four constant products, before identifying the pants, are
\(E_LE_R,E_LO_R,O_LE_R,O_LO_R\). Every entry in (10) is recorded in
`ordered_vertex_coefficients.csv`, including zeros. In particular, the
\(-i\) and \(-1\) entries retain their complex phases; no absolute value
has been applied to the coefficient products.

### Independent local normalization check

Use normalized small-representation kets

\[
E_R=\frac1{\sqrt2}
\begin{pmatrix}1&0\\0&1\\0&1\\-i&0\end{pmatrix}.
\]

At a primary NS bra, the ordered matrix elements for a unit \(c_\eta\)
are the rows of

\[
V_\eta=\frac1{\sqrt2}
\begin{pmatrix}1&0&0&i\\0&\eta&\eta&0\end{pmatrix},
\qquad V_\eta E_R=\operatorname{diag}(1,\eta).
\tag{11}
\]

The \(+i\) in (11) is the \(-i\) operator coefficient times the
Koszul minus from crossing the odd ket factor. Summing \(c_\eta V_\eta\)
therefore reproduces exactly \(d_+,d_-\) from (8). This determines a
local relative phase from the provided three-point functions.

For unitary ground-frame normalization, the projector is
\(\Pi=E_RE_R^\dagger\). Its diagonal part alone supplies only half of
\(\operatorname{tr}(V_\eta\Pi V_{\eta'}^\dagger)=1+\eta\eta'\).
Thus the off-diagonal Ramond ground indices contribute even to this
elementary check. The resulting two-family coefficient is

\[
d_{L,+}d_{R,+}+d_{L,-}d_{R,-}
=\frac{E_LE_R+O_LO_R}{2}.
\tag{12}
\]

This local Hermitian-frame check does not select the genus-two BPZ
continuation, a spin lift or a GSO weight.

### What remains to obtain the saved-basis matrix

Once the remaining contractions have been expressed in the saved block
basis, their coefficients \(\mathcal S_{AB}^{\eta\eta'}\) give

\[
\boxed{
M_{AB}=\sum_{\eta,\eta'}c_{L,\eta}c_{R,\eta'}
\mathcal S_{AB}^{\eta\eta'}
=\frac14\left[
E_LE_R\mathcal S_{AB}^{++}+E_LO_R\mathcal S_{AB}^{+-}
+O_LE_R\mathcal S_{AB}^{-+}+O_LO_R\mathcal S_{AB}^{--}
\right].
}
\tag{13}
\]

The \(1/4\) in (13) is exclusively the two conversions in (8).
The \(1/2\) from the two physical-field factors in (9) remains in
\(\mathcal S\). Neither is a primary propagation factor.

On an **unrestricted** tensor product, with full-vertex coefficients
\(t^v_{f;\eta\zeta}\) already in the Human bilinear frame, the exact
grading reduction is

\[
M_{(f,\eta,\eta'),(g,\zeta,\zeta')}
=\delta_{fg}(-1)^f
t^L_{f;\eta\zeta}t^R_{f;\eta'\zeta'}.
\tag{14}
\]

A physical Ramond restriction must be applied before assuming this
factorization. The earlier candidate
\(B_{L,\eta}B_{R,\eta'}
\left(\begin{smallmatrix}1&-ik\\ik&1\end{smallmatrix}\right)/16\)
is exported separately in `legacy_candidate_M.csv`. It remains a candidate:
the missing reduction from (10) has not been proved by taking a Hadamard
product of two selected \(2\times2\) tables. The primary-power audit does
not settle this remaining coefficient problem.

## 5. Branching weights: relative powers versus full propagation

For the enlarged SCA plus auxiliary Majorana representation, the Human
Note has

\[
h_n^{(1)}+h_n^{(2)}=
\begin{cases}
h_{\rm SCA}+2n^2,&NS,\\
h_{\rm SCA}+2n^2-1/16,&R.
\end{cases}
\]

The enlarged primary has weight \(h_{\rm SCA}+h_{\rm aux}\), with
\(h_{\rm aux}=0\) in NS and \(1/16\) in R. Consequently the power in
the **reduced** enlarged block is

\[
\Delta_n=h_n^{(1)}+h_n^{(2)}-(h_{\rm SCA}+h_{\rm aux})
=\begin{cases}2n^2,&NS,\\2n^2-1/8,&R.\end{cases}
\tag{15}
\]

Thus
\(q^{h_{\rm SCA}+h_{\rm aux}}q^{\Delta_n}
=q^{h_n^{(1)}+h_n^{(2)}}\), on the same logarithm branch.
At \(n=\pm1/4\) in R, \(\Delta_n=0\), as required for a ground
contribution. Using \(2n^2-1/16\) as the relative exponent would count
the auxiliary ground weight again.

When recovering the SCA block, both the enlarged block and the auxiliary
Majorana block in the star convolution are descendant-only. At the
propagated level the outside factors are respectively
\(P_{\rm SCA}P_{\rm aux}\) and \(P_{\rm aux}\). The recovered outside
factor is \(P_{\rm SCA}\). This statement does not replace the star
convolution by ordinary pointwise multiplication.

Likewise a null-vector recursion term
\(q_i^{rs/2}F(h_i\mapsto h_i+rs/2)\) is inside the reduced original
block. Its original outside \(q_i^{h_i}\) supplies the full
\(q_i^{h_i+rs/2}\) once. Shifting the outside exponent *and* retaining
the same relative power would double-count the null level.

There is an additional NSRR recursion subtlety: at fixed \(\beta\),
the computational substitution \(c\mapsto c_{rs}\) also changes the
Ramond weight \(c/24-\beta^2\) inside the recursively evaluated block.
The outside primary factor remains the one of the original physical
channel. If a recursive term is rewritten using a propagated block at the
shifted parameters, the ratio of the original and shifted primary factors
must be included explicitly. Recomputing the outside prefactor at each
recursion pole would alter the defined descendant expansion.

## 6. Checks and computation output

The reproducible audit is
`Code/genus_2/audit_nsrr_coefficient_primary_ledger.py`; its results are in
`Data Set/nsrr_coefficient_primary_ledger_20260915/`.

- 64 entries of the unreduced ordered vertex tensor checked exactly.
- 128 ground/first-NS-half-level coefficients checked against direct Ward
  forms, for both form parities, all sign pairs and all eight lifts.
- 20 source/target chart primary records, with unequal test momenta to
  detect a slot mismatch. Maximum source-adapter relative error:
  `8.032e-16`.
- 14 branching-weight identities checked using the implemented two
  Virasoro weights, including positive and negative branching labels.
- 23 focused tests passed, including exact preservation of the wrapper's
  historical numerical result and the protected-kernel hash boundary.

`NSRRPlumbingInputs.weights_slots` now exposes the primary weights.
`resummed_integrand` now separately returns `descendant_blocks`,
`primary_prefactor`, `primary_weights_slots` and `log_q_slots`. Its old
`blocks` field retains its historical meaning \(P F\) for compatibility;
the newly named `descendant_blocks` field contains the same chosen lift
combination of descendant-only \(F\)'s.
The complex free-superfield convention audit also now records its
primary weights, primary factor and descendant-only values explicitly.

**Status:** primary-power bookkeeping and the local coefficient tensor
are verified. The complete interacting fixed-spin NSRR matrix in the
saved eight-block basis remains unproved; no corrected physical partition
function is asserted on the basis of this ledger alone.
