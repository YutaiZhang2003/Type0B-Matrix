# Physical NSRR sewing in the Human Note bilinear frame

## Provisional genus-two experiment (2026-09-16)

At the user's request, the subsequent numerical experiment uses
`M_trial = 4 M_local`, explicitly marked as an **assumption**. In the basis
below this replaces `B_L B_R/8` by `B_L B_R/2` on the same supported sectors.
It rescales the entire matrix, including interference, once. It does not
modify the supplied three-point coefficients, the descendant blocks, their
phases, or the channel-dependent primary powers outside the blocks.

This extra genus-two factor is not established by the local Clifford or
torus calculations below. Those remain factor-one local checks.
The implementation and convergence data are in
[the provisional normalization study](../../Data%20Set/nsrr_provisional_factor4_20260916/README.md).
The factor-one cross-channel results in the next section are historical.

## Cross-channel status

The subsequent [cross-channel test](../../Data%20Set/nsrr_bilinear_cross_channel_20260915/README.md)
**fails** for this local matrix together with the tested spin transport.
Fresh complex all-NS blocks give normalized NSRR/all-NS ratios
`0.251–0.253`, where agreement requires one. The sum of the two transported
spin contributions has the same discrepancy, so changing their relative
Majorana phase alone cannot remove it. The cutoff variations are much
smaller than this gap. The local BPZ, descendant, and torus checks below
remain valid checks of the local construction; they do not establish a
crossing-invariant genus-two partition function. The global pairing and
relative channel normalization remain unresolved. No fitted factor four
has been inserted.

The [higher-quadrature study](../../Data%20Set/nsrr_bilinear_quadrature_20260915/FINAL_REPORT.md)
supersedes the central low-N estimate: at `t=0.60`, source N7 and target N10,
the spin-sum ratio stabilizes at `0.249928019` with source L5 and target R12
fixed. The two sign ratios are `0.250052595` and `0.249813244`. Each integral
and the full complex spin matrices pass two successive relative changes
below 0.01%. Thus the earlier approximately 1% residual after a hypothetical
factor four was largely quadrature error. The remaining fixed-cutoff
spin-sum residual under that hypothetical rescaling is `-0.028792%`.
This does not derive the rescaling, bound the block truncation error, or
establish the interacting spin transport. Only the central surface has
received this quadrature refinement.

## Result

This completes the local physical Ramond dual and matrix reduction left
open in `HUMAN_BILINEAR_SEWING_2026-09-15.md`. It uses the two-family small
Ramond representation and the complex bilinear pairing of the Human Note.
It does not replace that pairing by a Hermitian Gram matrix.

The result applies to generic NS and Ramond modules with an even NS
primary. At degenerate weights the inverse Gram construction must be taken
on the appropriate quotient or by the usual meromorphic limit.

Write the supplied BRY constants as `B_+=E`, `B_-=O`, independently on the
left and right pants. The physical equal-family ground amplitudes are
`d_+=(E+O)/2`, `d_-=(E-O)/2`, and the chiral three-form coefficients are
`c_+=E/2`, `c_-=O/2`. These are different pairs of quantities.

In Human slot order `(NS at infinity, R at 1, R at 0)`, define

\[
\widehat F_f^{\eta\eta'}=
\frac{F_f^{\eta\eta'}(+,+,+)+F_f^{\eta\eta'}(+,-,+)}{\sqrt2}.
\]

This fixes a chiral parity basis: it is `sqrt(2)` times the components with
even absolute parity on R at 1. It does **not** by itself specify the
physical full-state tube characters. Denote those separate signs by
`omega=(omega_NS,omega_R1,omega_R0)` and put

\[
s=\omega_{\mathrm{NS}}\omega_{R0},\qquad r=\omega_{R1}\omega_{R0}.
\]

Simultaneously reversing all three signs has no effect. For
`A=(f,eta,eta')`, `B=(g,zeta,zeta')`, the matrix is

\[
\boxed{
M_{AB}^{(s,r)}=
\delta_{\eta\zeta}\delta_{\eta'\zeta'}
\delta_{\eta\eta',-r}\,
\frac{B_{L,\eta}B_{R,\eta'}}8
\begin{pmatrix}1&s\\s&1\end{pmatrix}_{fg}.}
\]

In particular, `r=-1` selects `EE,OO`; `r=+1` selects `EO,OE`. The
relative form-parity sign is the REAL sign `s`. It is neither the old
`+-i eta eta'` interference sign nor an adjustable common normalization.

With `G_s=Fhat_0+s Fhat_1` and an independently specified HJS antichiral
block, the decomposition is

\[
Z_\Theta^{\mathrm{NSRR}}(s,r)=
\int\frac{d^3p}{\pi^3}\,P\widetilde P
\sum_{\eta\eta'=-r}\frac{B_{L,\eta}B_{R,\eta'}}8
G_s^{\eta\eta'}\widetilde G_s^{\eta\eta'}.
\]

No scalar coefficient is conjugated by this formula. On the real physical
slice, at conjugate moduli and with the HJS anti basis, one may specialize
`Ftilde=conjugate(F)` and `Ptilde=conjugate(P)`. The general implementation
takes the two sides independently.

## Descendant-compatible physical BPZ dual

Put `u=exp(-i*pi/4)`. On the chiral ground doublet define

\[
J=\begin{pmatrix}0&u\\\bar u&0\end{pmatrix},\qquad
J(Ww)=(-1)^{\#G_W}WJw.
\]

Then `J^2=1`, it commutes with L and anticommutes with G, and
`B(Jx,Jy)=-B(x,y)`. At each level take an even PBW basis `v_i` and its odd
partners `Jv_i`. This separates the multiplicity Gram `B_0` from the
two-dimensional Clifford factor. The HJS antichiral construction uses
`Jbar` and ground metric `diag(1,-i)`; holomorphically it is `diag(1,i)`.

In product Clifford order `(00,01,10,11)`, the graded metric and physical
ket/dual embeddings are

\[
D_J=\operatorname{diag}(1,-1,-1,-1),\quad
E_R^J=\frac1{\sqrt2}
\begin{pmatrix}1&0\\0&\bar u\\0&u\\-i&0\end{pmatrix},\quad
E_L^J=\frac1{\sqrt2}
\begin{pmatrix}1&0\\0&-u\\0&-\bar u\\-i&0\end{pmatrix}.
\]

\[
(E_L^J)^T D_J E_R^J=I_2,\qquad
G_{\mathrm{phys},J}=B_{0,h}\otimes B_{0,\tilde h}\otimes I_2.
\]

The same constant Clifford embeddings extend to every descendant level;
all weight dependence is in the multiplicity Grams. If `U` maps this
basis to canonical physical states `W_h W_a |R^epsilon>`, then

\[
G_{\mathrm{phys}}^{-1}=U G_{\mathrm{phys},J}^{-1}U^T.
\]

There is no dagger on `U`. The implementation obtains it by comparing
coefficients in the ambient basis, not by taking a Hermitian inner product.
Both three-point tensors in the Human theta sum are evaluated on physical
**ket** states. `E_L` implements the covector inside the BPZ inverse Gram;
feeding it into a trinion as if it were another physical ket is incorrect.

As an independent check, use the identity NS insertion and the Möbius map
`f(z)=z/(1-z)` to compute the full two-point BPZ matrix from Ward identities:

\[
B(l,r)=(-1)^{N_l+\widetilde N_l}
T_{\mathbf1}(e^{L_1+\widetilde L_1}l,
             e^{-L_1-\widetilde L_1}r).
\]

The resulting 228 entries agree exactly with the above Gram construction
at R levels `(0,0),(1,0),(0,1),(1,1),(2,0),(0,2)`. Mixed words use the
reversed dagger order; e.g. `L_-1 G_-1` has bra `G_1 L_1`.

## Reduction to the matrix, including the factor 1/8

For NS chiral/antichiral descendant parities `a,b`, the local full vertex
factorization contains the ordering sign

\[
(-1)^{b(a+\alpha+\gamma)+\beta\gamma}.
\]

Here `(alpha,beta)` and `(gamma,delta)` are the chiral/antichiral parities
on the two Ramond legs in the J basis. The bra self-term `ab` is required
when grouping the two algebras at the outgoing NS insertion.

Restricting the vertex to the two physical families gives, after factoring
out the chiral multiplicity amplitudes,

| `(a,b)` | `K_eta` |
|---|---|
| `(0,0)` | `diag(1,eta)` |
| `(0,1)` | `[[0,ubar],[eta*u,0]]` |
| `(1,0)` | `[[0,u],[eta*ubar,0]]` |
| `(1,1)` | `diag(-i,i*eta)` |

The full theta sign is `(-1)^K(n,e,f)`, where
`K(n,e,f)=n*e+n*f+e*f`, `n=a+b mod 2`, and `e,f` are physical Ramond
family parities. The inverse graded NS Gram supplies `(-1)^(ab)`.
Summing the two Ramond families with the tube characters gives

\[
S_{\eta\eta'}(a,b)=0\quad(\eta\eta'\ne-r),
\]

\[
\big(S(0,0),S(0,1),S(1,0),S(1,1)\big)
=\big(2,-2is,2is,2\big)\quad(\eta\eta'=-r).
\]

This is an exact finite Clifford identity; it is not a numerical fit.
In the projected block basis, coefficient by coefficient,

\[
\widehat F_1=-i(-1)^a\widehat F_0,\qquad
\widetilde{\widehat F}_1=+i(-1)^b\widetilde{\widehat F}_0.
\]

Each projected `Fhat_0` contains a factor `sqrt(2)`. Expressing the four
values of `S` in these normalized blocks gives
`M/(c_L c_R)=[[1,s],[s,1]]/2`. Finally
`c_L c_R=B_L B_R/4`, hence `M=B_L B_R[[1,s],[s,1]]/8`.

The earlier `1/4` multiplying only `Fhat_0 Ftildehat_0` referred to a
different reflected gluing prescription. It is not the coefficient of
each entry of this bilinear 2-by-2 matrix.

For `r=-1`, two immediate checks are

\[
Z[0,0]=\frac{E_LE_R+O_LO_R}{2},
\]

\[
Z[\tfrac12,0]=is\,
\frac{E_LE_R(p_0-p_1)^2+O_LO_R(p_0+p_1)^2}{8h_{\mathrm{NS}}},
\]

where `beta_j=i*p_j/sqrt(2)` and the second formula specifies a
holomorphic NS half-level with every other descendant level zero. The
antiholomorphic half-level has the opposite phase. Their cancellation
on some real slices does not mean either individual coefficient vanishes.

## Genus-one identity limit

The identity NS insertion has `d_+=d_-=1`, so `E=2,O=0`. The matrix gives
2 for `r=-1` and zero for `r=+1`, independently of `s`. After the two-point
coordinate conversion this is the ordinary Ramond trace and supertrace.
The ordinary descendant character is

\[
2\prod_{n\ge1}\frac{1+q^n}{1-q^n}
 \prod_{n\ge1}\frac{1+\widetilde q^n}{1-\widetilde q^n}.
\]

The checked holomorphic multiplicities per family are `1,2,4,8` through
level three; the physical trace is twice their product with the independent
anti multiplicity. The supertrace is zero at each tested pair of levels.
This rejects another overall factor four. BRY's common `pi delta` two-point
normalization supplies no extra factor two per R edge.

The comparison is made after converting the two-point local coordinates.
It does not equate an arbitrary product of the original three plumbing
parameters with the torus nome at finite neck size.

## One-Majorana phase and the spin basis

At `c=3/2`, with charge conservation and the appropriate NS weight, let
`D_00,D_11` denote the two even RR free scalar-times-Majorana spin blocks
in the charge homology frame, each normalized by the positive Ramond
degeneration. Independent fermion-mode sewing gives

\[
P\widehat F_0=D_{00},\qquad
iP\widehat F_1=D_{11},\qquad
\boxed{P G_s=D_{00}-isD_{11}.}
\]

The two raw blocks differ by the sign of odd NS descendant parity.
The same relation follows algebraically from the coefficient identities
above. The audit compares the last combination to both direct Majorana
mode sewing and the bosonized square root, whose sign is continued from
degeneration independently of the block calculation.

With total descendant cutoff three at two complex plumbing scales and
both allowed charge choices, the maximum norm-ratio error is `3.646e-8`
and phase error `5.320e-9` radians. Direct Majorana mode sewing versus the
bosonized reference agrees to `1.888e-14`. The block truncation error
decreases as the plumbing parameters shrink.

Thus a literal Human tube-sign combination `G_s` is a specified linear
combination of two raw spin blocks. Comparing it directly with a single
theta square root is a basis mismatch. Taking an absolute value first
would hide part of that mismatch. This explicit free-theory relation is
not a derivation of an interacting modular fusion kernel to a different
pants decomposition.

## Primary weights, numerical audit, and saved data

The block contains only descendant powers. In geometry order `(R0,R1,NS)`,

\[
P=\exp\big[h_R(p_0)\Log q_0+h_R(p_1)\Log q_1+
                 h_{\mathrm{NS}}(p_\infty)\Log q_\infty\big],
\]

\[
h_{\mathrm{NS}}=Q^2/8+p^2/2,\quad h_R=h_{\mathrm{NS}}+1/16,
\quad Q=b+b^{-1}.
\]

The anti factor, primary weights in another decomposition, and continued
logarithms are separate inputs. No `q^h` or cylinder `q^(-c/24)` is hidden
in `M`.

- 1,024 direct full-state NSRR descendant coefficients pass across all
  four tube-sign choices and all four products of pants coefficients.
  Maximum scaled discrepancy: `1.477e-15`. Checks include NS level 5/2,
  R level 2, mixed descendants, and independent complex anti parameters.
- Twelve finite-series comparisons with independent complex moduli,
  unequal primary weights and continued logs give norm error at most
  `2.221e-16`, phase error at most `1.995e-16` radians.
- All 243 nodes in the complete complex-block L3/N5 and L5/N3 datasets
  were recombined for all four tube-sign choices. All saved primary
  prefactors agree exactly with the explicit weight formula.

For example, at saved `t=0.60`, L3/N5 gives

| `(s,r)` | `Z_NSrr` | Ratio using saved marked-spin denominators and all-NS target |
|---|---:|---:|
| `(+,-)` | `1.9275860966e-10` | `0.2404543251` |
| `(-,-)` | `2.0904726061e-10` | `0.2607733997` |
| `(+,+)` | `1.8536209564e-10` | `0.2312276359` |
| `(-,+)` | `1.8488920067e-10` | `0.2306377289` |

These ratios retain the old marked-spin denominator/target and are
**diagnostics**, not equal-spin cross-channel equality tests. The free
basis conversion above explains a concrete reason the raw labels cannot
be identified. No common factor four has been fitted to these numbers.
High-order saved scalar contractions cannot reconstruct the new matrix;
the recombination uses datasets retaining each complex block.

The new default resummed assembler uses this bilinear matrix and requires
explicit physical tube signs. Historical `legacy-times-four` selection is
available only for reproducing the old scalar. The frozen-grid worker
records the convention, descendant blocks, primary factors, and logs.
The chiral double-Virasoro recursion and its auxiliary ungraded pairing
are unchanged by this physical sewing correction.

## Files and reproduction

- `Code/genus_2/nsrr_bilinear_sewing.py`: production matrix and contraction.
- `Code/genus_2/nsrr_bilinear_state_oracle.py`: independent physical Ward
  and BPZ state sum, with no conformal-block matrix inside it.
- `Code/genus_2/audit_nsrr_bilinear_sewing.py`: exact Clifford reduction,
  independent two-point Gram check, descendant comparisons, torus limit,
  one-Majorana norm/phase comparison, and saved-data reduction.
- `Data Set/nsrr_bilinear_sewing_20260915/`: machine-readable results,
  coefficient tables, saved comparisons, and input SHA-256 manifest.

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  /private/tmp/type0b-nsrr-smoke-venv/bin/python \
  Code/genus_2/audit_nsrr_bilinear_sewing.py
```

Literature inputs: the two-family ground embedding and chiral/antichiral
three-form phases in [Suchanek, arXiv:1012.2974](https://arxiv.org/pdf/1012.2974),
and the supplied physical coefficients and two-point normalization in
[BRY, §3.1](https://arxiv.org/html/2201.05621#S3.SS1). The matrix reduction
above is derived here with the local Human Note convention, not quoted
as a matrix printed in either paper.
