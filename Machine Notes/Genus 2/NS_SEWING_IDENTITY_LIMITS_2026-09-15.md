# The NSRR factor 1/4, the genus-one identity limit, and the all-NS decomposition

Date: 2026-09-15.

## 1. Where the NSRR factor comes from

The factor is fixed by the **three-point basis and block normalization**. It is not an extra factor in the two-point integration measure.

For a fixed normalized Ramond module, the supplied physical-family amplitudes are

\[
d_+=\frac{E+O}{2},\qquad d_-=\frac{E-O}{2}.
\]

As a matrix on the two ground families, the three-point vertex is

\[
T=\begin{pmatrix}d_+&0\\0&d_-\end{pmatrix}
=\frac E2 I+\frac O2\sigma_z.
\]

Consequently the two labelled pants give

\[
\operatorname{Tr}(T_LT_R)
=\frac14\left[
E_LE_R\operatorname{Tr}I+
O_LO_R\operatorname{Tr}\sigma_z^2+
(E_LO_R+O_LE_R)\operatorname{Tr}\sigma_z
\right]
=\frac{E_LE_R+O_LO_R}{2}.
\tag{1}
\]

These are oriented left/right coefficient products. In the physical reflection calculation the coefficients may first be taken real; the resulting expressions then continue analytically. Their products are not replaced by numerical absolute squares.

Our saved two-lift descendant blocks are

\[
\widehat F_f^{\eta\eta'}
=\frac{F_f^{\eta\eta'}(+++)+F_f^{\eta\eta'}(+-+)}{\sqrt2},
\qquad \widehat F_0^{++}(0)=\widehat F_0^{--}(0)=\sqrt2.
\tag{2}
\]

The lift triples are in geometry order \((0,1,\infty)\). The factor two in the physical-family trace is already present in \(|\widehat F(0)|^2\). Therefore

\[
\frac{E_LE_R}{4}|\widehat F_0^{++}|^2+
\frac{O_LO_R}{4}|\widehat F_0^{--}|^2
\tag{3}
\]

has the required leading coefficient (1). Equivalently, the two chiral coefficients \(c_+=E/2\), \(c_-=O/2\) supply one factor \(1/2\) at each vertex.

The numerical coefficient changes if the block is renormalized. With \(\mathcal B_\eta=\widehat F_0^{\eta\eta}/\sqrt2\), so that \(\mathcal B_\eta(0)=1\), the same expression is

\[
\frac{E_LE_R}{2}|\mathcal B_+|^2+
\frac{O_LO_R}{2}|\mathcal B_-|^2.
\]

Thus the invariant statement is the physical trace, together with a declared block basis, rather than the number \(1/4\) by itself. Equation (1) fixes the leading normalization; the additional descendant tests in the [reflected-state derivation](</Users/yutaizhang/Desktop/Type0B-Matrix/Machine Notes/Genus 2/NSRR_REFLECTED_STATE_SEWING_2026-09-15.md>) determine the local parity assembly.

BRY use the same \(\pi\delta(P-P')\) two-point normalization for NS and R families; their physical NSRR amplitudes use the \(d_\pm\) convention above. Those facts do not supply an additional edge multiplicity. [BRY, equations (3.1), (3.6), (3.8)](https://arxiv.org/html/2201.05621#S3.SS1).

## 2. Identity insertion gives a genus-one test

Insert the NS identity on the infinity edge and retain that edge at level zero. On the remaining two edges use the same Ramond module. For normalized states,

\[
\langle I\,R^\epsilon R^{\epsilon'}\rangle
=\delta_{\epsilon\epsilon'},\qquad
d_+=d_-=1,\qquad E=2,\quad O=0.
\tag{4}
\]

This is an identity-module sewing test. In noncompact Liouville theory the identity is not a normalizable member of the continuum being integrated. We work per normalized module, or with the continuum identity interpreted as an operator kernel composed using its inverse two-point metric. We do not multiply delta functions pointwise or claim that this is the leading asymptotic of the integrated Liouville partition function.

The identity is a quotient of the NS Verma module: \(G_{-1/2}|I\rangle=L_{-1}|I\rangle=0\). The diagnostic removes identity-edge descendants. Where the NS Ward implementation uses intermediate inverse Gram matrices, it keeps \(h_I\) symbolic until cancellation, then takes \(h_I=0\). Substituting zero before these cancellations would invert a singular Verma Gram matrix.

### The torus multiplier in the actual plumbing coordinates

With remaining punctures at zero and one, the two seam maps are

\[
z\longmapsto\frac{q_0}{z},\qquad
z\longmapsto1+\frac{q_1}{z-1}.
\]

Their composition has matrix

\[
A=\begin{pmatrix}q_1-1&q_0\\-1&q_0\end{pmatrix},
\qquad \det A=q_0q_1,\qquad\operatorname{Tr}A=q_0+q_1-1.
\]

The small eigenvalue ratio is

\[
B=\frac{1-q_0-q_1+
\sqrt{(1-q_0-q_1)^2-4q_0q_1}}2,\qquad
k=\frac{q_0q_1}{B^2},\qquad B\longrightarrow1.
\tag{5}
\]

Keep

\[
\Log k=\Log q_0+\Log q_1-2\Log B.
\tag{6}
\]

The leading approximation \(k\simeq q_0q_1\) is insufficient for a descendant-level test.

For a generic Ramond super-Virasoro module define the oscillator character per ground family

\[
\mathcal C_R(k)=
\prod_{n=1}^\infty\frac{1+k^n}{1-k^n}
=1+2k+4k^2+8k^3+\cdots.
\tag{7}
\]

In the identity limit the saved projected block obeys

\[
\widehat F_0^{++}
=\sqrt2\,B^{-2h_R}\mathcal C_R(k),\qquad
\widehat F_1^{++}=-i\widehat F_0^{++}.
\tag{8}
\]

The audit verifies these coefficients through total plumbing level three directly from the Ramond PBW Gram matrices and Ward forms, at \(c=81/5,\ \beta=2i/5,\ h_R=167/200\).

Primary propagation remains separate:

\[
P_2=(q_0q_1)^{h_R},\qquad
P_2\widehat F_0^{++}
=\sqrt2\,k^{h_R}\mathcal C_R(k).
\tag{9}
\]

The analytic factor \(B^{-2h_R}=1+\cdots\) belongs to the descendant expansion in the original two-edge coordinates. After reducing to the torus coordinate, the primary is \(k^{h_R}\), outside the torus descendant character.

Equations (3), (4), and (9) give

\[
Z_{R,\;q^{L_0}}^{(1)}
=\frac{2\cdot2}{4}|P_2\widehat F_0^{++}|^2
=2\,|k^{h_R}\mathcal C_R(k)|^2.
\tag{10}
\]

This is exactly the physical small-Ramond-module trace: two ground families with independent left/right oscillator descendants. A total fermion-parity insertion instead gives zero, because every oscillator configuration has two ground-family choices of opposite parity.

This count concerns the unprojected fixed-spin theory. A bosonic theory obtained by summing spin structures and applying a GSO projection has a different state-space prescription. BRY distinguish their local Ramond field from its disorder/defect partner; using their three-point basis does not by itself specify that global projection.

Omitting \(1/4\) from (3) would give \(8|k^{h_R}\mathcal C_R(k)|^2\), hence eight ground states. Multiplying the old local assembly by four has the same failure.

### What this limit does not distinguish

The old *local* matrix, before its additional factor four, also passes this identity test:

\[
\frac{E_LE_R}{16}
|\widehat F_0+i\widehat F_1|^2
\ \xrightarrow{I}\
\frac{E_LE_R}{4}|\widehat F_0|^2.
\tag{11}
\]

The NS descendant responsible for the earlier half-level mismatch has disappeared in the identity limit. Therefore genus one checks the normalization and limiting trace, but does not replace the genus-two descendant test that distinguishes the local matrices.

## 3. Single-Majorana torus norm and phase

The generic super-Virasoro module test is supplemented by an independent free single-Majorana test. With \(k=e^{2\pi i\tau}\), the standard cylinder characters are

\[
\chi_{\rm NS,+}
=k^{-1/48}\prod_{n\ge1}(1+k^{n-1/2})
=\sqrt{\frac{\theta_3}{\eta}},
\]

\[
\chi_{\rm NS,-}
=k^{-1/48}\prod_{n\ge1}(1-k^{n-1/2})
=\sqrt{\frac{\theta_4}{\eta}},
\]

\[
\chi_{\rm R,+}
=\sqrt2\,k^{1/24}\prod_{n\ge1}(1+k^n)
=\sqrt{\frac{\theta_2}{\eta}},
\qquad Z_{\rm R,-}=0.
\tag{12}
\]

The theta/eta formula for a complex fermion is the genus-one bosonization identity in [Tuite–Zuevsky, section 5.4](https://arxiv.org/pdf/1007.5203#page=28). For the three even characteristics used here, its additional characteristic phase is one. Equation (12) takes the single-Majorana square root with a specified continuation.

The two real Ramond zero modes, one from each chirality, form a two-state physical ground space. Its dimension is two, not four; the \(\sqrt2\) belongs to the chiral spin partition function whose modulus square gives that trace.

The audit evaluates theta functions by their lattice sums, compares them with fermion Fock products, and continues each square root from positive imaginary \(\tau\). It checks both norm and phase at seven complex moduli, for each of the three nonzero spin sectors. It also checks

\[
\chi_{\rm R,+}(\tau+1)=e^{i\pi/12}\chi_{\rm R,+}(\tau),\qquad
\chi_{\rm NS,+}(\tau+2)=e^{-i\pi/12}\chi_{\rm NS,+}(\tau).
\tag{13}
\]

In particular, the audit checks the longer continuations

\[
\chi_{\rm R,+}(\tau+12)=-\chi_{\rm R,+}(\tau),\qquad
\chi_{\rm NS,+}(\tau+24)=-\chi_{\rm NS,+}(\tau).
\]

Both squared amplitudes return to their initial values. Their single-Majorana square roots change sign. This directly demonstrates the information lost by keeping only the bosonized square or by resetting its square root independently at the endpoints. An absolute-value comparison would miss it as well.

Equation (12) uses \(k^{L_0-c/24}\). To compare with (9) and (10), multiply by \(k^{-c/24}\) externally on the chiral side. This factor is never incorporated into the descendant-only block.

## 4. Status of the all-NS decomposition

The written grading algebra is consistent. With full holomorphic and antiholomorphic parity triples \(x,y\),

\[
K(x+y)+x\cdot y
=K(x)+K(y)+\left(\sum_i x_i\right)\left(\sum_jy_j\right)
\pmod2,
\quad K(x)=\sum_{i<j}x_ix_j.
\tag{14}
\]

Matching total parities leaves the sign \((-1)^{a+p_1+p_2+p_3}\). For even NS primaries,

\[
Z_{\rm HN}=|P|^2
\left[C_LC_R|F_0|^2
-(i\widetilde C_L)(i\widetilde C_R)|F_1|^2\right].
\tag{15}
\]

Thus the complete odd coefficient is \(+\widetilde C_L\widetilde C_R\). This is consistent bookkeeping in the literal Human-Note basis; it does not establish the fixed-spin interpretation of a single lift.

### Reflection expressed in the unchanged Human-Note blocks

For real physical NS weights the local reflected pants norm contracts a vertex with its conjugate using the Hermitian Gram. Its chiral coefficient has no extra \(K\) sign. This conversion can be expressed using all four literal lift values, without redefining their descendant series.

In lift order \((+++),(+-+),(-++),(--+)\), put \(H_a=U_aF_a\). For even highest states,

\[
U_0=\frac12
\begin{pmatrix}
-1&1&1&1\\1&-1&1&1\\1&1&-1&1\\1&1&1&-1
\end{pmatrix},\qquad
U_1=\frac12
\begin{pmatrix}
1&1&1&-1\\1&1&-1&1\\1&-1&1&1\\-1&1&1&1
\end{pmatrix}.
\tag{16}
\]

These are finite parity-Fourier transforms: \(U_a=\frac14H\operatorname{diag}((-1)^K)H^T\) on the four parity triples of total parity \(a\). In particular \(U_a^2=I\). The odd and even transforms differ.

For a specified local reflected lift \(\ell\), the matrix in the unchanged literal block basis is

\[
M^{(a,\ell)}_{\sigma\tau}
=B_{L,a}B_{R,a}(U_a)_{\ell\sigma}(U_a)_{\ell\tau},
\qquad B_0=C,\quad B_1=\widetilde C,
\tag{17}
\]

with \(P=\exp(\sum h_{\rm NS}\Log q)\) outside. This is a coherent combination including off-diagonal lift terms, not a spin average.

The fresh single-Majorana genus-two test checks all four all-NS spin choices at the saved \(t=0.60\) target geometry. The transformed even block agrees with the branch-fixed bosonized Majorana. A literal single Human-Note lift fails that comparison. The free vacuum has no odd three-form coupling; the odd sector is checked by the algebra, Ward seeds, and the distinct matrix \(U_1\), rather than by that vacuum example.

The all-NS identity limit provides a further check: for two surviving NS edges,

\[
\mathcal C_{\rm NS,\epsilon}(k)
=\prod_{n\ge1}\frac{1+\epsilon k^{n-1/2}}{1-k^n}.
\tag{18}
\]

The literal block has \(\epsilon=-\ell_0\ell_1\), while the reflected combination has \(\epsilon=+\ell_0\ell_1\). The audit verifies the corresponding coefficients through total level three, treating the identity null states as above.

Therefore the all-NS **relative coefficient sign** is consistent, and its identity limit can be matched with explicit spin transport. The historical single-lift target is still not a certified genus-two fixed-spin numerator. The local conversion (16) does not by itself prove the full interacting NSRR-to-all-NS modular transport.

## 5. Reproducible files

- [Audit script](</Users/yutaizhang/Desktop/Type0B-Matrix/Code/genus_2/audit_ns_sewing_identity_limits.py>)
- [All-NS reflection adapter](</Users/yutaizhang/Desktop/Type0B-Matrix/Code/genus_2/all_ns_reflected_sewing.py>)
- [Numerical report and tables](</Users/yutaizhang/Desktop/Type0B-Matrix/Data Set/ns_sewing_identity_limits_20260915/README.md>)
- [Full audit data and source hashes](</Users/yutaizhang/Desktop/Type0B-Matrix/Data Set/ns_sewing_identity_limits_20260915/summary.json>)

All computations are diagnostic; the protected production kernels and historical integrated data are preserved.
