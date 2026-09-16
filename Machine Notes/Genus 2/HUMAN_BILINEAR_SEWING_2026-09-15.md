# Genus-two sewing in the Human Note bilinear convention

**NSRR update:** the local physical Ramond dual and matrix left open in
this initial audit are now derived in
[NSRR_BILINEAR_PAIRING_2026-09-15.md](NSRR_BILINEAR_PAIRING_2026-09-15.md).
That continuation includes all-level Clifford reduction, independent BPZ
two-point checks, 1,024 physical descendant comparisons, and the torus
normalization. Statements below that the descendant dual is still missing
describe the status of this earlier audit and are superseded by that result.
The marked cross-channel spin transport remains a distinct question.

## Result and scope

The all-NS assembler now explicitly implements the **complex bilinear**
prescription in `Human Notes/SCblock.tex:467–715`. Its two blocks and its
two pants' coefficients are independent inputs. The previous real-data
all-NS specialization is numerically unchanged. The new tests distinguish
this prescription from an implicit Hermitian pairing.

The all-NS grading reduction is exact. Direct finite-state checks cover
descendants and independent complex data. The analogous reduction for
unrestricted NSRR tensor-product modules also passes. The physical small
Ramond representation still requires a descendant-compatible ket/dual
embedding and its transport on the marked graph. **A complete physical
NSRR-to-all-NS partition comparison has not been established by this audit.**

The earlier reflected formula with coefficients `E_L E_R/4` and `O_L O_R/4`
has the scope stated in its reflected-state audit. It is not silently
identified with the Human Note bilinear prescription here. The old
interference formula, including the extra factor four in the resummed
scalar wrapper, is now explicitly labelled historical.

## 1. Pairing and the all-NS matrix

For homogeneous states the Human Note uses

\[
B(x\otimes\tilde x,y\otimes\tilde y)
=(-1)^{p(\tilde x)p(y)}B_h(x,y)B_{\tilde h}(\tilde x,\tilde y).
\]

Scalar coefficients are not conjugated. The reversal of a descendant word
under the dagger is retained. For example,
`(L_-1 G_-1 w)^dagger = w^dagger G_1 L_1`.

Let `p` and `ptilde` be the three intrinsic primary parities. With the
literal descendant-only theta blocks in the Human Note,

\[
Z_\Theta=\int d\mu\sum_{a,\tilde a}
P_a\,\widetilde P_{\tilde a}\,
F_a\,M^\Theta_{a\tilde a}\,\widetilde F_{\tilde a},
\]

\[
\boxed{M^\Theta_{a\tilde a}
=(-1)^{a+\sum_i p_i}C_L^{(a)}C_R^{(a)}
\,\delta_{\tilde a,\,a+\sum_i p_i+\sum_i\tilde p_i\ ({\rm mod}\ 2)}}.
\]

This is the product of the two normalized three-point coefficients, not
their absolute squares. Equal pants give the square written in the note.
For even primaries it reduces to

\[
M^\Theta=\operatorname{diag}(C_L^{(0)}C_R^{(0)},
                            -C_L^{(1)}C_R^{(1)}).
\]

Using the supplied convention `C_HN^(0)=C_BRY`,
`C_HN^(1)=i*Ctilde_BRY` on **both** pants gives

\[
M^\Theta=\operatorname{diag}(C_L C_R,\widetilde C_L\widetilde C_R).
\]

There is no additional factor four in this all-NS unit-primary convention.

### Why the sign holds beyond primaries

For each edge let `x_i` and `y_i` be the absolute chiral and antichiral
state parities. Put `K(x)=x_1 x_2+x_1 x_3+x_2 x_3`. The full theta
reordering contributes `K(x+y)`; inverse graded Grams contribute `x.y`.
At a nonzero full vertex, `sum(x)=sum(y)=a_abs mod 2`. Therefore

\[
K(x+y)+x\cdot y=K(x)+K(y)+a_{\rm abs}\pmod2.
\]

The first two terms are precisely the signs already inside the literal
chiral blocks. The remaining sign belongs to `M`. This argument uses
absolute parity, not a restriction to primary states or a choice of PBW
word order. The audit checks all 2,048 combinations of descendant and
primary parities that satisfy locality.

## 2. Basis transport and primary factors

If `F'=U F`, `Ftilde'=V Ftilde` with unchanged external primary powers,

\[
\boxed{M'=U^{-T} M V^{-1}}.
\]

`V` is independently specified. It is not automatically `conjugate(U)`.
For a map on propagated amplitudes `P'F'=U(PF)`, the descendant map is

\[
F'=D_{P'}^{-1} U D_P F.
\]

The code takes the primary factors as explicit inputs evaluated with the
chosen logarithms. A channel-dependent weight, or a continued logarithm,
must not be absorbed into `M` or dropped from this relation. This finite
basis rule by itself is not the continuous fusion kernel connecting
different Liouville pants decompositions.

For the real `c,h` all-NS evaluation only, the independently defined
antiholomorphic block at `qbar` equals `conjugate(F)`. The node evaluator
uses this specialization explicitly, without conjugating `C_R`.

## 3. What is established in NSRR

For a specified full, bosonic vertex tensor in the same bilinear frame,
write

\[
T_v=\sum_{f,\eta,\zeta}t^v_{f\eta\zeta}
\rho_f^\eta\widetilde\rho_f^\zeta
\]

with the usual local ordering signs understood. On the unrestricted
tensor-product edge modules, for an even NS primary, the reduction is

\[
M_{(f,\eta_L,\eta_R),(g,\zeta_L,\zeta_R)}
=\delta_{fg}(-1)^f
t^L_{f\eta_L\zeta_L}t^R_{f\eta_R\zeta_R}.
\]

The 36 direct Ramond descendant checks verify this identity with independent
complex vertex tensors. They do not infer these tensors from the physical
`(E,O)` constants. Applying a physical Ramond restriction is another step.

For declared ket and dual embeddings `E_R,E_L`, that step is

\[
G_{\rm phys}=E_L^T G_{\rm full}E_R,\qquad
K_{\rm phys}=E_R G_{\rm phys}^{-1}E_L^T.
\]

The restriction is made **before** inversion. A new helper implements this
operation and rejects singular supplied pairings.

### Concrete ground-state check

Starting from the small representation in [Suchanek, Eq. (7)](https://arxiv.org/pdf/1012.2974),
rephase the antiholomorphic odd ground state as
`wtilde^- = i*wbar^-`. Both chiral ground metrics are then `diag(1,i)`;
the canonical antiholomorphic beta is `-beta`. In product order
`(++,+-,-+,--)`, the Human Note graded product and ket embedding are

\[
D=\operatorname{diag}(1,i,i,1),\qquad
E_R=\frac1{\sqrt2}
\begin{pmatrix}1&0\\0&-i\\0&1\\-1&0\end{pmatrix}.
\]

Using the same embedding on the two sides gives

\[
E_R^TDE_R=\operatorname{diag}(1,0).
\]

One explicit ground dual is

\[
E_L=\frac1{\sqrt2}
\begin{pmatrix}1&0\\0&1\\0&-i\\-1&0\end{pmatrix},
\qquad E_L^TDE_R=I_2.
\]

This is not a failure of the physical representation: it shows why its
Hermitian ket embedding cannot also be used as a bilinear dual embedding.
Multiplying the singular matrix by four cannot repair it. The displayed
dual fixes this ground example; extending the dual to all descendants and
the two oriented vertex tensors is still required. Ground matching alone
does not determine the eight-channel physical matrix or its global spin
transport.

The primed pairing in the double-Virasoro construction at Human Note
lines 1423–1433 and 1747 is a different pairing: the SCA and auxiliary
fermion factors use an ungraded product. The physical holomorphic/
antiholomorphic crossing sign must not be inserted into that auxiliary
engine. Its chiral kernels were not changed by this work.

## 4. Verification and limits

- 122 all-NS full-state descendant comparisons: maximum scaled error
  `2.289e-16`. The fixtures include independent complex central charges,
  weights, and three-point constants, with NS states through level 3/2.
- 32 complex finite-series comparisons cover all four NS lift choices
  independently on both sides, with unequal continued primary factors.
  Maximum norm-ratio error `6.661e-16`; maximum phase difference
  `3.638e-16` radians.
- 2,048 exact grading identities and complex nonunitary basis checks pass.
- 36 unrestricted NSRR comparisons through level-one Ramond descendants:
  maximum scaled error `6.429e-16`.
- The corrected bra ordering and Ramond Ward routines pass their existing
  literature, torus, and genus-two regressions.

The finite-series checks compare the same retained state sets; their small
errors are not truncation-error estimates for the full partition function.
They also do not fix a global single-Majorana Pfaffian branch. No new
interacting cross-channel equality or momentum-integral convergence claim
is made here.

Machine-readable results and source hashes are in
`Data Set/human_bilinear_sewing_20260915/audit.json`.
