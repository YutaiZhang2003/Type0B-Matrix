# NSRR sewing: physical Ramond families, the reflected bra, and the coefficient matrix

Date: 2026-09-15.

## 1. Result and its scope

The legacy matrix fails an exact descendant test. After stripping primary propagation, its coefficient of \(q_{\rm NS}^{1/2}\) vanishes identically. Sewing the two physical Ramond families using their supplied three-point coefficients gives a generically nonzero coefficient.

For normalized physical Ramond families, a radial Hermitian reflected bra, and no inserted parity defect, the replacement checked here is

\[
Z_s^{\rm refl}
=\int_{\mathbb R_+^3}\frac{d^3p}{\pi^3}|P_s|^2
\left[
\frac{E_LE_R}{4}\left|\widehat F_0^{++}\right|^2+
\frac{O_LO_R}{4}\left|\widehat F_0^{--}\right|^2
\right].
\tag{1}
\]

The matrix is determined by the exact ground, half-level, and mixed half-level coefficients within the class diagonal in the pair \((\eta,\eta')\), with an arbitrary Hermitian two-by-two matrix in the three-form parity. Independent physical-state sewing checks additional descendants on both Ramond edges. This is a local reflection-frame result; it does not establish the interacting spin and local-coordinate transport to the saved all-NS channel, or exclude every possible matrix mixing different \((\eta,\eta')\) pairs.

The numerical recombination and complex free-field checks are in the [result report](</Users/yutaizhang/Desktop/Type0B-Matrix/Data Set/nsrr_reflected_state_sewing_20260915/README.md>). The checked assembly changes the source by a moduli-dependent amount relative to the legacy local matrix; its normalized ratio to the saved target is still near \(0.25\).

## 2. Match the user's descendant-only block definition

Use geometry order \((0,1,\infty)\), with sectors \((R,R,NS)\). The Human Note's three-form slots are \((\infty,1,0)\); both the momenta and plumbing data are reversed when passed to that ordering. Put

\[
Q=b+b^{-1},\qquad c=\frac32+3Q^2,\qquad
h_{\rm NS}(p)=\frac{Q^2}{8}+\frac{p^2}{2},\qquad
h_R(p)=h_{\rm NS}(p)+\frac1{16},\qquad
\beta_j=\frac{ip_j}{\sqrt2}.
\tag{2}
\]

The saved blocks are the literal descendant sums \(F_f^{\eta\eta'}\) in the [Human Note](</Users/yutaizhang/Desktop/Type0B-Matrix/Human Notes/SCblock.tex:1208>), with its bilinear Gram matrices and ordering signs. In the same projected basis used by the saved comparison, define

\[
\widehat F_f^{\eta\eta'}(q)
=\frac{
 F_f^{\eta\eta'}(q;+++)
+F_f^{\eta\eta'}(q;+-+)}{\sqrt2}.
\tag{3}
\]

The lift triples here are in geometry order. The parity \(f=0\) labels the even three-form; it does **not** restrict the NS edge to integer levels. A half-integer NS descendant can be accompanied by an odd Ramond ground state.

Every one of the eight \((f,\eta,\eta')\) blocks at fixed source momenta shares

\[
P_s=\exp\!\left[
h_R(p_0)\Log q_0+
h_R(p_1)\Log q_1+
h_{\rm NS}(p_\infty)\Log q_\infty
\right].
\tag{4}
\]

The all-NS target instead has its own \(q\)'s and momenta, and three weights \(h_{\rm NS}\). Neither \(P_s\) nor a cylinder Casimir factor is included in \(F\) or \(M\). A fractional descendant power uses the chosen plumbing square-root branch separately from the primary logarithm.

For \(A=(f,\eta,\eta')\), equation (1) is the contraction

\[
Z_s^{\rm refl}
=\int\frac{d^3p}{\pi^3}|P_s|^2
\widehat F^T M^{\rm refl}\overline{\widehat F},
\quad
M^{\rm refl}_{(0,+,+),(0,+,+)}=\frac{E_LE_R}{4},
\quad
M^{\rm refl}_{(0,-,-),(0,-,-)}=\frac{O_LO_R}{4},
\tag{5}
\]

with other entries zero in this representative. The order \(F^T M\bar F\) matters when comparing with the legacy off-diagonal matrix.

## 3. Start from the supplied three-point coefficients

Denote the supplied constants by \(E=C_{\rm even}\) and \(O=C_{\rm odd}\). The physical ground coefficients and chiral coefficients are

\[
d_+=\frac{E+O}{2},\qquad d_-=\frac{E-O}{2},
\qquad c_+=\frac E2,\qquad c_-=\frac O2.
\tag{6}
\]

The physical \(d_\pm\) relation is the convention of [BRY, equation (3.8)](https://arxiv.org/html/2201.05621#S3.SS1). The chiral change of basis follows [Suchanek, equations (28)–(31)](https://arxiv.org/pdf/1012.2974). The odd three-forms for incoming and outgoing Ramond states have different ordered definitions; the warning in footnote 5 concerns precisely this bra conversion.

Keep the two pants labelled throughout: \(E_LE_R\), \(E_LO_R\), \(O_LE_R\), \(O_LO_R\). For the reflected physical calculation these constants can first be taken real. The resulting polynomial then defines their analytic continuation as oriented left/right products. Do not replace them by numerical absolute squares. In the saved real-momentum theta calculation, the two supplied tuples coincide.

The unit physical ground metric gives

\[
\sum_{\epsilon=\pm}d_{L,\epsilon}d_{R,\epsilon}
=\frac{E_LE_R+O_LO_R}{2}.
\tag{7}
\]

Since \(\widehat F_0^{\eta\eta'}(0)=\sqrt2\), equation (1) has exactly this leading coefficient.

## 4. The physical Ramond bra is part of the calculation

In holomorphic/antiholomorphic product-ground order \(++,+-,-+,--\), the two normalized physical ket families have embedding

\[
E_0=\frac1{\sqrt2}
\begin{pmatrix}
1&0\\0&1\\0&1\\-i&0
\end{pmatrix}.
\tag{8}
\]

This is the small Ramond module of [Suchanek, equation (7)](https://arxiv.org/pdf/1012.2974). The corresponding invariant subspace and graded vertex construction are also described in [Hadasz–Jaskólski–Suchanek, section 4](https://arxiv.org/pdf/0810.1203).

The chiral bilinear ground Gram matrix is \(B_0=\operatorname{diag}(1,i)\), while the antiholomorphic one is \(\bar B_0=\operatorname{diag}(1,-i)\). Including the graded tensor sign gives

\[
D_0=\operatorname{diag}(1,-i,i,-1).
\tag{9}
\]

In the declared radial Hermitian reflection frame, conversion to the bilinear bra acts by

\[
S_0=D_0^{-1}=\operatorname{diag}(1,i,-i,-1),\qquad
E_L=S_0\overline{E_0},\qquad E_L^T D_0 E_0=I_2.
\tag{10}
\]

Inserting the same ket embedding on both sides of a bilinear pairing is not this operation. Restriction and inversion also do not commute: induce the physical Gram matrix first, then invert it.

At descendant level, order physical states as a holomorphic lowering word, an antiholomorphic lowering word, then a physical ground family. The product-space embedding contains

\[
(-1)^{\#\bar G\,\alpha}(E_0)_{\alpha\bar\alpha,\epsilon}.
\tag{11}
\]

For a holomorphic PBW state based on \(w^\alpha\), the Hermitian Gram is obtained from its bilinear Gram by multiplying the row by \((-i)^\alpha\). The antiholomorphic construction conjugates the corresponding expression at \(\beta\mapsto-\beta\). This gives the prescribed antiholomorphic zero modes with the opposite \(i\) phase.

### The intrinsic-parity correction in a full vertex

The independent oracle reduces the antiholomorphic words first. An antiholomorphic NS descendant with parity \(\delta_{\rm NS}\) is then a holomorphic highest state of intrinsic parity \(\delta_{\rm NS}\). Using intrinsic parity zero for this intermediate state gives a wrong mixed half-level coefficient.

For an odd antiholomorphic Ramond word \(\bar A\), canonical holomorphic ground states are instead

\[
w'_+=\bar A w^-,\qquad w'_-=-i\,\bar A w^+.
\tag{12}
\]

These follow from anticommutation of the two supercurrents and the stated zero-mode actions. The oracle applies these changes before the holomorphic Ward reduction. Its full physical vertex is consequently computed without assuming equation (5).

## 5. Exact descendant obstruction and determination of \(M\)

Let \(h=h_{\rm NS}(p_\infty)\), \(x=q_\infty^{1/2}\), and keep both Ramond descendant levels zero. The projected chiral blocks start as

\[
\widehat F_0^{\eta\eta'}=\sqrt2(1+\alpha_{\eta\eta'}x+\cdots),
\qquad
\widehat F_1^{\eta\eta'}=-i\sqrt2(1-\alpha_{\eta\eta'}x+\cdots),
\tag{13}
\]

\[
\alpha_{\eta\eta'}
=-\frac{(\beta_0-\eta\beta_1)(\beta_0-\eta'\beta_1)}{2h}
=\frac{(p_0-\eta p_1)(p_0-\eta'p_1)}{4h}.
\tag{14}
\]

For example, with \(u=e^{-i\pi/4}\), the holomorphic NS-fermion physical vertex has its two off-diagonal family entries

\[
T(G_{-1/2}V,R^+,R^-)
=-iu(\beta_0d_+-\beta_1d_-),
\qquad
T(G_{-1/2}V,R^-,R^+)
=u(\beta_0d_--\beta_1d_+).
\tag{15}
\]

Contracting the physical family indices and dividing by the NS descendant norm \(2h\) gives the following coefficients of \(Z/|P_s|^2\):

\[
[1]=\frac{E_LE_R+O_LO_R}{2},
\tag{16}
\]

\[
[x]=[\bar x]
=\frac{E_LE_R(p_0-p_1)^2+O_LO_R(p_0+p_1)^2}{8h},
\tag{17}
\]

\[
[x\bar x]
=\frac{E_LE_R(p_0-p_1)^4+O_LO_R(p_0+p_1)^4}{32h^2}.
\tag{18}
\]

Equation (18) is independently obtained from the two-algebra vertex reduction described above.

The legacy local matrix for a fixed pair was

\[
M_{\rm old}^{\eta\eta'}
=\frac{B_{L,\eta}B_{R,\eta'}}{16}
\begin{pmatrix}1&-i\eta\eta'\\i\eta\eta'&1\end{pmatrix},
\qquad B_+=E,\quad B_-=O.
\tag{19}
\]

For \(\eta\eta'=+1\), its combination \(\widehat F_0+i\widehat F_1\) cancels the term linear in \(x\). For \(\eta\eta'=-1\), its combination has zero ground term, so its modulus square again has no term linear in \(x\). Thus its complete answer has \([x]=0\), in contradiction with (17). Multiplication by four preserves this failure.

To determine the parity matrix independently of a numerical crossing ratio, write

\[
M^{\eta\eta'}
=\frac{B_{L,\eta}B_{R,\eta'}}{16}
\begin{pmatrix}a&r+iy\\r-iy&d\end{pmatrix}.
\tag{20}
\]

Using the vectors \(v=\sqrt2(1,-i)^T\) and \(w=\sqrt2\alpha(1,i)^T\), match \(v^TM\bar v\), \(w^TM\bar v\), its counterpart, and \(w^TM\bar w\) with (16)–(18), keeping all four constant products independent. The solution is

\[
\eta=\eta':\ (a,d,r,y)=(4,0,0,0),\qquad
\eta\ne\eta':\ (a,d,r,y)=(0,0,0,0).
\tag{21}
\]

This yields (5). It is an exact determination within the specified matrix class, followed by additional coefficient checks, not a fit to the integrated target.

## 6. Clifford identities and the family trace

The odd intertwiner on a chiral Ramond module can be written

\[
J_0=\begin{pmatrix}0&u\\\bar u&0\end{pmatrix},\quad J_0^2=1,
\qquad J(Aw)=(-1)^{\#G_A}A\,J_0w.
\tag{22}
\]

It commutes with \(L_n\) and anticommutes with \(G_n\). With \(P_h=\operatorname{diag}(1,-1)\), the physical ground projector is

\[
\Gamma=i(J_0P_h)\otimes\overline{J_0},\qquad
\Gamma^2=1,\qquad
\Gamma E_0=-E_0,\qquad
\Pi=\frac{1-\Gamma}{2}=E_0E_0^\dagger.
\tag{23}
\]

The relevant three-form identities in the ordered NS,R,R slots are

\[
\rho_f^\eta(x_1,Jx_2,x_3)
=i^f\eta\bar u\,(-1)^{p_2}\rho_{1-f}^\eta(x_1,x_2,x_3),
\tag{24}
\]

\[
\rho_f^\eta(x_1,x_2,Jx_3)
=i^fu\,\rho_{1-f}^\eta(x_1,x_2,x_3).
\tag{25}
\]

For real physical weights and imaginary \(\beta\), the outgoing-bra phase check is

\[
i^{\alpha+\gamma}
\overline{\rho_f^\eta(A,Bw^\alpha,Cw^\gamma)}
=(-i)^f(-1)^{p_3}
\rho_f^\eta(A,Bw^\alpha,Cw^\gamma).
\tag{26}
\]

Here \(p_2,p_3\) are full chiral Ramond state parities. The audit checks each identity exactly on 720 combinations, including NS descendants through level \(3/2\) and both Ramond descendants through level 1.

The reduced physical-family sign tables in the corresponding \(J\)-adapted bases are

\[
K_{00}^\eta=\begin{pmatrix}1&0\\0&\eta\end{pmatrix},\quad
K_{10}^\eta=\begin{pmatrix}0&-i\\\eta&0\end{pmatrix},\quad
K_{01}^\eta=\begin{pmatrix}0&i\\\eta&0\end{pmatrix},\quad
K_{11}^\eta=\begin{pmatrix}-i&0\\0&i\eta\end{pmatrix}.
\tag{27}
\]

Their two subscripts are the holomorphic and antiholomorphic NS parities. The trace identity

\[
\operatorname{Tr}\!\left(K_{p\bar p}^\eta
(K_{p\bar p}^{\eta'})^\dagger\right)
=1+\eta\eta'=2\delta_{\eta,\eta'}
\tag{28}
\]

explains the cancellation of the mixed \(EO,OE\) products in this reflection pairing. It is checked for all 16 combinations. These local Clifford and bra identities do not specify a modular transport or a defect insertion around a handle.

## 7. What the computations establish

The [independent state oracle](</Users/yutaizhang/Desktop/Type0B-Matrix/Code/genus_2/nsrr_reflected_state_sewing.py>) contracts full physical three-point vertices against physical inverse Gram matrices. It is separate from the chiral block assembly used for comparison. The [audit and recombination script](</Users/yutaizhang/Desktop/Type0B-Matrix/Code/genus_2/audit_nsrr_reflected_state_sewing.py>) verifies:

- 136 full physical-state coefficient comparisons at two generic parameter fixtures, including simultaneous NS and Ramond descendants and descendants on both Ramond edges; maximum scaled error \(1.71\times10^{-15}\).
- Exact symbolic coefficients (16)–(18), and the matrix solution (21).
- The 2,160 Ward identities and 16 trace identities above.
- Both free scalar charge sectors associated with \(\eta=\pm\), including the chiral single-Majorana phase. At plumbing scale \(0.02\), norm errors are below \(6.4\times10^{-12}\) and phase errors are below \(1.04\times10^{-10}\) radians at total descendant cutoff 3.
- Recombination of 243 saved momentum nodes, keeping the primary factor outside the block and matrix. The adapter reproduces every saved source primary exactly.

The free complex comparison uses a charged Heisenberg scalar and a bosonized Majorana with its branch continued from the positive Ramond degeneration. A real nonchiral result alone would not check that branch. The earlier legacy free combination had a phase error of about \(-0.0126\) radians at the same smaller plumbing scale.

## 8. The remaining cross-channel issue

At \(t=0.60\), the new source with L3/N5 is

\[
Z_s^{\rm refl}=2.0355505410\times10^{-10},\qquad
\mathcal Q_s^{\rm refl}=4.9500263314\times10^{-8},
\qquad
\frac{\mathcal Q_s^{\rm refl}}{\mathcal Q_t^{\rm saved}}
=0.2539222151.
\tag{29}
\]

Here \(\mathcal Q=Z/Z_{\rm free}^{\,1+2Q^2}\). The free factor removes the chart-dependent Weyl normalization only when numerator and denominator refer to the same spin prescription.

The saved all-NS target still uses its historical Human-Note assembly. Its independent free control is a nontrivial combination \(F=UD\), \(U=\mathbf 1_{4\times4}/2-I_4\), of four raw fixed-spin chiral Majorana blocks. This prevents certification of the numerator's spin from its lift label alone. The free identity does not yet derive the interacting bra and spin-basis matrix. The [saved five-point spin audit](</Users/yutaizhang/Desktop/Type0B-Matrix/Data Set/nsrr_spin_quadrature_t060_20260830/spin_basis_fivepoint.json>) is included in the new provenance manifest.

BRY normalize both NS and Ramond two-point functions by \(\pi\delta(P-P')\); therefore those metrics do not justify a factor two per Ramond edge. The relation between their local/defect operator conventions and the desired closed-surface spin prescription must also be respected. See [BRY, equations (3.1), (3.6) and §3.1](https://arxiv.org/html/2201.05621#S3.SS1).

Consequently, neither the old near-unit ratio after a factor four nor the new ratio near one quarter decides the global convention. The exact half-level and complex phase tests decide the local assembly in the stated frame. A complete crossing test still requires transporting that pairing, with the marked spin structure, to the all-NS block definition.

The order-eight saved files contain only the legacy-contracted scalar. They cannot recover the new selected complex channels, so this correction uses the complete lower-order complex data and preserves the historical high-order results as such.
