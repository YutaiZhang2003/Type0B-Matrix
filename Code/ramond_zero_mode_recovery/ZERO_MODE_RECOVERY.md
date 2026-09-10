# Recovery of the physical block by saturating the auxiliary zero mode

The proposed modification works as a formal sewing construction. It supplies
the missing auxiliary sector and yields an auxiliary series with constant
**1 in the full star algebra**. The physical block is unchanged. The enlarged
numerator must be computed with the same insertion as the auxiliary denominator.

## 1. The zero mode and its sewing pairing

Use the auxiliary Ramond basis

\[
 |B,g\rangle=\psi_{-r_1}\cdots\psi_{-r_k}u^g,
 \qquad g=0,1,\quad r_j>0.
\]

In the code's convention,

\[
 \psi_0|B,g\rangle=\frac{(-1)^k}{\sqrt2}|B,1-g\rangle.
\]

The ground-state interchange \(S|B,g\rangle=|B,1-g\rangle\) is an odd
ordinary commuting intertwiner of the auxiliary Clifford action. Let
\(P_{\mathsf F}|B,g\rangle=(-1)^{k+g}|B,g\rangle\) and
\(\Theta=P_{\mathsf F}S\). Then

\[
 \Theta^2=-1,\qquad \{\Theta,\psi_r\}=0,
 \qquad D=\sqrt2\,\Theta\psi_0,
 \qquad D|B,g\rangle=(-1)^g|B,g\rangle.
\]

Thus the even, level-preserving operator

\[
 \boxed{\Pi_0=\frac{1+D}{2}}
\]

fixes the auxiliary ground label to \(g=0\) on a chosen Ramond cut.
Above level zero, **D is ground-label parity, not total fermion parity**.

A lone odd operator has zero ordinary trace and zero supertrace on the doubled
graded module. Zero-mode saturation therefore includes an odd closure/pairing.
Our convention is equivalent to the odd trace through
\(\operatorname{STr}(\Theta\psi_0q^{L_0-c/24})
=\operatorname{Tr}(S\psi_0q^{L_0-c/24})\).
For a free Majorana fermion,

\[
 \frac{1}{\sqrt2}\operatorname{Tr}_{\mathcal F_R}
 (S\psi_0q^{L_0-1/48})
 =q^{1/24}\prod_{n\ge1}(1-q^n).
\]

After stripping the leading power this has constant 1. This is the elementary
torus example of the desired saturation; see J. van Ekeren,
[Vertex Operator Superalgebras and Odd Trace Functions, §5](https://arxiv.org/abs/1307.4114).
The symbol for the odd intertwiner in that reference uses a different convention.

## 2. A unit auxiliary series in the manuscript's theta channel

Write \(A=\mathbf F_{\mathsf F}\) and \(x=\eta_2\eta_3\), so
\(x\star x=1\) and \(A_0=1+x\). Insert D on the first Ramond edge:
multiply each summand in the displayed auxiliary sewing formula by
\((-1)^{\mathsf b}\), where \(\mathsf b\) is that edge's ground label.
Call the resulting series \(A_D\). Its constant is

\[
 (A_D)_0=1-x.
\]

Consequently,

\[
 \boxed{A_{\rm reg}=\frac{A+A_D}{2}
       =\mathbf F_{\mathsf F}[\Pi_0],\qquad (A_{\rm reg})_0=1.}
\]

Equivalently, evaluate the existing auxiliary sum with \(\mathsf b=0\).
In the current code's q and spin-frame conventions,

\[
 A_{\rm reg}=1+\frac12\eta_1\eta_3q_1^{1/2}
 +\frac18\eta_2\eta_3q_2+\frac18q_3
 +O_{\rm total}(3/2).
\]

This is a modified sewing functional; it is not the unsaturated free-fermion
partition function for a fixed spin structure. It packages the ordinary and
zero-mode-saturated information into one invertible parity-valued series.
Spin structures and the vertex-form signs \(\eta,\eta'\) must still be
distinguished.

## 3. Recovering F, including the previously annihilated sector

Insert the same auxiliary \(\Pi_0\otimes1_{\rm SCA}\) in the enlarged
propagator, defining

\[
 H_{\rm reg}=\frac{\widehat{\mathbf F}+
                    \widehat{\mathbf F}[D]}{2}.
\]

Since D acts only on the auxiliary factor, preserves parity, and preserves
level, the same tensor-factorization and Koszul-sign calculation gives

\[
 H_{\rm reg}=A_{\rm reg}\star\mathbf F,
 \qquad
 \boxed{\mathbf F=A_{\rm reg}^{-1}\star H_{\rm reg}.}
\]

This argument does not require a sector identity. For coefficient multiindices,

\[
 \mathbf F_{\boldsymbol n}=(H_{\rm reg})_{\boldsymbol n}
 -\sum_{0<\boldsymbol m\le\boldsymbol n}
 (A_{\rm reg})_{\boldsymbol m}\star
 \mathbf F_{\boldsymbol n-\boldsymbol m}.
\]

The order is increasing total plumbing level; there is no division by 2 in
this recurrence because the auxiliary constant is exactly 1.

For comparison with the previous restricted-inverse construction,
\(x\star A=A\) and \(x\star A_D=-A_D\). For a physical block satisfying
\(x\star\mathbf F=s\mathbf F\), the ordinary numerator carries the
\(s=+1\) information and the D-inserted numerator carries the \(s=-1\)
information. The latter cannot be reconstructed from the former. In
particular, changing only the denominator of an existing output file does
not compute the missing physical block.

## 4. Computing the inserted numerator in the double-Virasoro basis

In the graded auxiliary-first tensor realization,
\(G_r=P_{\mathsf F}\otimes G_r^{\rm phys}\). Hence \(\Theta\)
anticommutes with both \(\psi_r\) and \(G_r\), and commutes with
\(L_n^{\mathsf F},L_n,U_n\), and therefore with both embedded Virasoro
algebras. However, \(\psi_0\), D, and \(\Pi_0\) do not commute with
the two Virasoro algebras separately. A diagonal reweighting of the original
branch sum would be incorrect.

There is a concrete finite-level prescription. Let \(T_\ell\) have the
double-Virasoro descendant states as columns in the auxiliary-times-SCA PBW
basis at total edge level \(\ell\). Then

\[
 D_\ell=T_\ell^{-1}\operatorname{diag}((-1)^g)T_\ell,
 \qquad
 K_\ell=\frac{I+D_\ell}{2}\,G_\ell^{-1}.
\]

Replace the selected edge's inverse Gram matrix by \(K_\ell\).
The vertex tensors are the branching primary coefficients times the two
ordinary Virasoro descendant three-point functions. This computes the
inserted numerator independently of the physical block.

At level zero the current unnormalized branch conventions give

\[
 Dv_{1/4}^{\alpha}=(-1)^\alpha v_{-1/4}^{\alpha},\qquad
 Dv_{-1/4}^{\alpha}=(-1)^\alpha v_{1/4}^{\alpha}.
\]

More generally, the insertion couples \(n\) to \(n\pm\tfrac12\), with
the descendant levels adjusted to preserve total level. Its origin is the
auxiliary field \(\psi\), which is a simultaneous Virasoro primary in
the **irreducible physical vacuum** tensored with the auxiliary NS module:

\[
 h_\psi^{(1)}=-\frac{1+2b^2}{2(1-b^2)},\qquad
 h_\psi^{(2)}=\frac{b^2+2}{2(1-b^2)},\qquad
 h_\psi^{(1)}+h_\psi^{(2)}=\frac12.
\]

Put \(t_1=2b^2/(1-b^2)\), \(t_2=2/(b^2-1)\). In each copy,

\[
 h_\psi^{(a)}=-\frac12-\frac34t_a,\qquad
 ((L_{-1}^{(a)})^2+t_aL_{-2}^{(a)})|\psi\rangle=0.
\]

Thus the extra matrix elements can alternatively be generated by Virasoro
Ward identities for a degenerate insertion, followed by taking its Ramond
zero mode. In the usual choice of parameters these are the (2,1) field of
the first copy and the (1,2) field of the second. This requires blocks with
an additional insertion; the existing unpunctured genus-two block routine
alone does not provide them.

This alternative is now implemented in `degenerate_kernel.py`. Set
\(\ell_n=2n^2-1/8\) and compute only the primary matrix elements

\[
 C_{n'n}^{(\alpha)}=\sqrt2\,
 \langle v_{n'}^\alpha|\Theta\psi_{\ell_n-\ell_{n'}}|v_n^\alpha\rangle.
\]

For descendants at matching total levels, the zero-mode matrix is

\[
 M_{(n',A'_1,A'_2),(n,A_1,A_2)}
 =C_{n'n}^{(\alpha)}\prod_{a=1}^2
 \rho_{\mathrm{Vir}}\!\left(
 L_{-A'_a}v_{h_{n'}^{(a)}},\psi^{(a)},
 L_{-A_a}v_{h_n^{(a)}}\right),
\]

with unit-normalized primary Virasoro forms. Only
\(n'=n\pm\tfrac12\) contribute. The inserted sewing kernel is

\[
 K_{\Pi_0}=\frac12\left(G^{-1}+G^{-1}MG^{-1}\right).
\]

The numerical algorithm therefore constructs oscillator states only for
these primary matrix elements and the branching primary coefficients.
It generates all remaining descendants by ordinary Virasoro Ward identities.
Inverting the two separate Virasoro Gram matrices keeps the matrix blocks
small. This avoids the full high-level change-of-basis construction in the
original low-level prototype.

## 5. Scope and numerical checks

The construction has now been extended through total level 10 using the
degenerate-field kernel. One complete physical block took **299.2 seconds**
at 384-bit complex precision. Auxiliary and insertion identities and a
full 32-case sewing comparison passed through level 10 in exact arithmetic
modulo 65521. The modular comparison finished before the user's instruction
to stop the direct physical route; the subsequent high-precision physical
comparison was cancelled. See [LEVEL10_VALIDATION.md](LEVEL10_VALIDATION.md)
for the completed checks, timing, and limitations. High-order runners now
disable the direct physical reference by default and instead compare the
two Ramond cuts when running all cases.

The following describes the original low-level prototype.

`zero_mode_recovery.py` implements the ordinary, D-inserted, and projected
auxiliary sums and the full-algebra recovery recurrence.
`check_zero_mode_sewing.py` is a finite-level prototype of the inserted
double-Virasoro numerator just described. It uses branch states and
Virasoro Ward forms to compute the numerator; physical PBW coefficients
enter only as an independent comparison after sewing.

The validation covers all f, intrinsic NS parities, both vertex signs at
each vertex, and either choice of Ramond cut, through total level 2.
At the benchmark \(b=7/5\), \((P_1,P_2,P_3)=(11/23,13/29,17/31)\),
all 32 cases pass: maximum recovery error is \(1.43\times10^{-13}\),
and maximum insertion matrix entry outside the predicted branch support
is \(5.35\times10^{-12}\).
The odd-form eta label of the archived branching oracle is transported as
\(\eta_{\rm native}=(-1)^f\eta_{\rm physical}\); the validation includes
both f values rather than extrapolating the even-form convention.
See `validation_level2.json` for the measured errors and the degenerate
primary/null checks. This is a low-level prototype, not a replacement for
the high-order production recursion.

Reproduce with:

```sh
python3 Code/ramond_zero_mode_recovery/check_zero_mode_sewing.py \
  --twice-level 4 \
  --json Code/ramond_zero_mode_recovery/validation_level2.json
```

For a closed plumbing graph with only NSNSNS and NSRR vertices, the Ramond
subgraph consists of disjoint cycles. Fix one auxiliary ground label on
one cut in each cycle. At zero plumbing level the NS states are vacuum
states, and each NSRR vertex propagates equality of the two auxiliary
ground labels. The projectors therefore leave the unique all-zero ground
configuration, with coefficient 1 in normalized sewing frames. This gives
the analogous local formal-series construction for arbitrary closed
plumbing channels.

This graph argument does not identify one insertion per Ramond cycle with
the number of global holomorphic spinor zero modes on every smooth
surface. For geometric odd-spin correlators one must saturate all actual
zero modes and account for the insertion's spinor dependence. Ramond
external legs likewise require specified endpoint pairings.
