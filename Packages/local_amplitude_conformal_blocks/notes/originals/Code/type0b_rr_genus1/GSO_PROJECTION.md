# Type 0B GSO projection in the RR sewing basis

The GSO choice is fixed by the theory. It is now implemented on the combined
time–Liouville states, with explicit superghost grading, and applied to the
finite-level NS–R sewing tensors. The four picture-changing endpoint
components and the BRY NSRR pants constants are included in this reference
calculation. This document describes the projected coefficient layer.
The subsequent [local OPE layer](LOCAL_DENSITY.md) now assembles a torus
density in a contractible puncture patch. **An independently integrated
RR reflection amplitude has not been obtained.**

The remaining spin-related implementation is transport between the
implemented OPE theta frame and the mixed bulk/cusp charts, including
the disorder-line winding. This is a convention/coordinate calculation,
not another GSO choice.

## Project the product theory

The defining retained sectors are

\[
(\mathrm{NS}+ ,\mathrm{NS}+ )\oplus(\mathrm{NS}- ,\mathrm{NS}- )
\oplus(\mathrm{R}+ ,\mathrm{R}+ )\oplus(\mathrm{R}- ,\mathrm{R}- ).
\]

The definition and the normalized axion vertices are given in
[BRY I, equations (2.1), (2.6)–(2.8)](https://arxiv.org/html/2201.05621v2#S2.SS1).
For matched closed-string sectors this means

\[
P_{0B}=\frac{1+\Gamma_h\Gamma_a}{2},
\qquad F_{\rm total}=F_{X^0,\psi^0}+F_{\rm SL}+F_{\beta\gamma}\pmod2.
\]

Mixed NS–R and R–NS closed-string sectors are excluded. On the underlying
spin-CFT states, 0A instead retains odd total parity in the R sector; its
NS projection is the same. `keeps_closed_sector` implements the entire
sector table.

In particular, neither the Liouville factor nor its two chiral Ramond
ground spaces should be projected separately. An odd Liouville state
paired with an odd spectator state is allowed. The saved checks exhibit
nonzero examples with an odd time descendant and with odd superghost
grading, respectively.

## Ground matrices and BRY states

In the ordered basis

\[
(\sigma^0 V^{R,+},\ \sigma^0 V^{R,-},\
  \mu^0 V^{R,+},\ \mu^0 V^{R,-}),
\]

the asymptotic chiral Clifford gradings are

\[
\Gamma_h=\begin{pmatrix}
0&0&0&1\\0&0&-i&0\\0&i&0&0\\1&0&0&0
\end{pmatrix},\qquad
\Gamma_a=\begin{pmatrix}
0&0&0&1\\0&0&i&0\\0&-i&0&0\\1&0&0&0
\end{pmatrix}.
\]

They follow from \(2\psi^0_0\psi^1_0\) using the timelike Clifford sign,
and satisfy

\[
\Gamma_h^2=\Gamma_a^2=1,\quad [\Gamma_h,\Gamma_a]=0,\quad
\Gamma_h\Gamma_a=\operatorname{diag}(1,-1,-1,1).
\]

Thus at the reference \((-1/2,-1/2)\) picture,

\[
P_{0B}=\operatorname{diag}(1,0,0,1),\qquad
W_s=\frac{1}{\sqrt2}(1,0,0,s)^T,\quad s=\pm1.
\]

Both \(W_s\) survive. In the asymptotic free spin basis they have chiralities
\((s,s)\). The opposite-chirality eigenvectors are
\((0,1,\pm i,0)^T/\sqrt2\), retained by the reference 0A R projector.
The exact diagonal grading survives the Liouville Yukawa interaction;
the two individual asymptotic chiral gradings need not separately commute
with a fixed interacting wall. No extra chiral symmetry is assumed here.

The existing physical Clifford matrices give
\(G_0^{\rm m}W_s=\widetilde G_0^{\rm m}W_s=0\) at \(k=sP\).
The opposite-chirality subspace has no simultaneous kernel for nonzero P.
This check concerns propagating states; it does not discard the separate
zero-energy 0A flux problem.

Regrouping the product-theory two-point function gives the bilinear metric

\[
D=\operatorname{diag}(1,1,1,-1),\qquad
W_+^TDW_-=1,\qquad W_\pm^TDW_\pm=0.
\]

Consequently `contract_bry_pair` uses

\[
\frac12\sum_{i,j=0}^1(-1)^{ij}s_1^i s_2^j
\langle t_i t_j\rangle\langle L_i L_j\rangle.
\]

The time and Liouville tensors must have the same marked frame. Their
spin/disorder phases are retained, and the second tensor is not conjugated
at analytically continued energy. This fixes the product-theory external
cocycle; it does not set the torus order/disorder correlators equal.

## Descendants, ghosts and picture changing

`GhostPicture` records shifts from \((-1,-1)\) for NS or
\((-1/2,-1/2)\) for R. The relative superghost GSO bit is

\[
g=\Delta q_h+\Delta q_a+N_{\beta\gamma}\pmod2.
\]

This is a GSO grading, not the Grassmann statistics of the commuting
beta/gamma system. It follows by requiring the BRST operator to be even:
the \(\gamma G^{\rm m}\) term pairs two odd GSO factors. In the bosonized
description take eta/xi GSO even and \(e^\varphi\) GSO odd. The bosonized
PCO terms \(c\partial\xi\), \(e^\varphi G^{\rm m}\), and the
\(e^{2\varphi}b\eta\) derivative terms are then all GSO even.
The PCO conventions are those already used in
[PICTURE_CHANGING.md](PICTURE_CHANGING.md).

For fixed descendant words containing \(N_h+N_a\) odd matter modes, apply

\[
P_{0B}(N_h,N_a,g)=\frac12
\left[1+(-1)^{N_h+N_a+g}\operatorname{diag}(1,-1,-1,1)\right]
\]

to their ground coefficients. A reduced \(G_0\) action changes the ground
vector; it must not also be counted as an unreduced odd word.

The exact check is the intertwining relation

\[
P(q_h+1,q_a)\,G_h^{\rm m}=G_h^{\rm m}\,P(q_h,q_a),
\]

and its antiholomorphic counterpart. Projecting with an unchanged matter
matrix after raising just one picture would incorrectly remove states.

`liouville_raised_external` supplies the four ordered
\(G_{-p}\widetilde G_{-q}\), \(p,q\in\{0,1\}\), endpoint states for
`raised_combination`. It reduces the zero modes in the existing physical
Ramond basis. In particular,
\(G_0\widetilde G_{-1}=-\widetilde G_{-1}G_0\); this minus sign is included.
Both insertions together preserve matter parity. The functions return
local descendants; the previously derived flat-coordinate connections
still have to be applied.

## Projection inside the sewn coefficient

`JointMixedPlumbing` uses the already checked `PhysicalMixedPlumbing`
vertices and BPZ Grams in both factors. For a product vertex with slots
\((N,R,E)\), grouping time entries before Liouville entries contributes

\[
(-1)^{p_L(N)[p_t(R)+p_t(E)]+p_L(R)p_t(E)}.
\]

The product BPZ inverse additionally has the sign
\((-1)^{p_t p_L}\) on each sewn edge. These signs are evaluated explicitly
in the joint basis. The marked theta-sewing crossing is applied to the
**combined** NS and R matter parities.

Projection can then be performed in three equivalent ways:

1. Replace each inverse Gram by \(P B^{-1}P^T\).
2. Restrict the basis to the allowed combined states before contraction.
3. Average marked parity insertions with the ghost characters:

\[
K^{0B}_{g_N,g_R}=\frac14\sum_{\ell_N,\ell_R=\pm1}
\ell_N^{g_N}\ell_R^{g_R}K(\ell_N,\ell_R).
\]

The four terms here implement two edge projectors. On a closed amplitude
with even external vertices, parity conservation makes the two edge
constraints redundant after including the ghost vertices. This is not an
additional factor of one quarter multiplying a torus spin sum.

The saved comparisons use imaginary and complex external Liouville
momenta, actual interacting \(c=27/2\) Liouville Ward tensors, and a free
\(c=3/2\) analytic spectator module. They also compare against a separate
route through the old two factor sewing routines. In that route the
internal product cocycles cancel to the external sign
\((-1)^{p_t(E)p_L(E)}\).

`coupled_coefficients` contracts the four independent pants choices with
\(c_+=C_{\rm even}(P_R,\omega,P_N)/2\) and
\(c_-=C_{\rm odd}(P_R,\omega,P_N)/2\). Both pants use analytic constants;
the right factor is not replaced by a complex conjugate. These are the
mixed-necklace constants, not the equal-external-momentum OPE constants.

## Global spin-character convention

The Hamiltonian table is fixed:

| Spatial sector | Temporal insertion | Theta label | 0B weight | Closed/capped 0A weight |
| --- | --- | --- | --- | --- |
| NS | ordinary trace | 3 | +1 | +1 |
| NS | supertrace | 4 | +1 | +1 |
| R | ordinary trace | 2 | +1 | +1 |
| R | supertrace | 1 | +1 | -1 |

Writing the two periodicity bits as \((a,b)\), these weights are
\((-1)^{n ab}\), with n=0 for 0B and n=1 for 0A. This is the Arf formulation
of the diagonal projection in
[Kaidi, Parra-Martinez and Tachikawa, “Classification of String Theories via Topological Phases”](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.124.121601).
The trace/supertrace conventions match
[Hadasz, Jaskólski and Suchanek](https://arxiv.org/pdf/1207.5740).
With Ramond punctures the endpoint projection must accompany the closed
spin weight. Reweighting a 0B axion correlator by the 0A Arf sign does not
produce a 0A propagating RR observable.

The implementation records S: \((2\ 4)\), T: \((3\ 4)\),
\(z\mapsto z+1\): \((1\ 2)(3\ 4)\), and
\(z\mapsto z+\tau\): \((1\ 4)(2\ 3)\).
These are characteristic permutations; coordinate and square-root phases
are not replaced by these permutations.

Once both factors are expressed in that frame, the 0B sum is
\(\tfrac12\sum_{\delta=1}^4 L_\delta S_\delta\), using the same delta
for Liouville and the time/ghost spectator. `diagonal_spin_sum` implements
this contraction. When its spectator uses the existing `free_outer_factor`,
set `gso_half_in_spectator=True`: that factor already contains one half.

## Saved evidence and remaining integration

The executable check and coefficient archive are reproduced with

```sh
python3.11 -B Code/type0b_rr_genus1/check_gso_projection.py \
  --output /tmp/type0b-rr/GSO_LAYER_CHECKS.json \
  --bank /tmp/type0b-rr/GSO_COEFFICIENT_FIXTURES.npz
```

The saved run has 278 algebra/convention checks, 264 projected-sewing
comparisons, 256 ground and 32 raised comparisons with the old factorized
sewing routines. Maximum projection residual is `4.27e-17`; maximum
product-sewing residual is `2.22e-16`. There are 72 unit-coupling fixtures
and 24 additional coefficients with freshly evaluated BRY pants constants.
The latter have maximum projection residual `9.70e-19`.

The archive preserves the two matter-parity channels of each edge and
the four possible ghost-edge gradings. A fixed ghost grading is a
coefficient of the eventual spectator bank, not a replacement for its
correlator. The level fixtures are low-order reference data, not a
convergence study.

The local OPE density now includes the time/ghost/Liouville contraction.
Its extension to the bulk and cusp charts is still required before the
global physical-spin-projection condition in `matched_integration.py`
can be certified. In particular, raw marked lifts must not simply be
renamed theta labels. The global overlap, boundary prescription and
spectral/moduli convergence checks remain required.
The matrix target stays independent, and all RR worldsheet-amplitude
entries remain null.
