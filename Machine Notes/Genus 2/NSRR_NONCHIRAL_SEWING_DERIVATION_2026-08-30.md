# NSRR theta sewing: a derivation with explicit grading and Ramond indices

Date: 2026-08-30. This is a separate machine working note. The Human Note
and the checked PBW, branching, and double-Virasoro kernels are not edited.

## 0. Result and status

The grading part of the sewing can be derived completely. On an
**unrestricted** holomorphic/antiholomorphic tensor product, it gives the
Human Note's quadratic signs inside the chiral blocks and the remaining
factor `(-1)^f` outside them. With a physical Ramond restriction, the
correct starting point is instead the restricted inverse pairing in
equations (12)–(13) below; it must be applied before simplifying signs.

This note also derives exact ground and first-NS-descendant coefficients
for every pair of chiral three-form signs and all eight plumbing lifts.
They show explicitly that the opposite-sign channels cannot be discarded
on the grounds of a vanishing even ground term.

**Not yet derived:** the complete local-coordinate/BPZ dictionary taking
the physical Ramond three-point tensors to the particular chiral basis
used by the genus-two adapter. Consequently this note does not yet give
a certified numerical matrix `M` for the physical partition function.
Sections 6–7 identify the missing conversion as a concrete two-sided
pairing problem, rather than an unspecified multiplicity. No numerical
partition or modular comparison is claimed here.

## 1. Labels, states, and local coordinates

Use Human-Note slots `(1,2,3)=(NS at infinity, R at 1, R at 0)`.
The geometry code orders its punctures oppositely, `(0,1,infinity)`.
Thus `q_slots = q_geometry[::-1]`, with the momenta and lifts reversed too.
Use the same local coordinates on the two pants as in the chiral block.

| Symbol | Meaning |
|---|---|
| `alpha,beta=0,1` | ground labels of the two **chiral** Ramond modules |
| `p_i`, `p'_i` | complete holomorphic state parities on the first/second pants |
| `tilde p_i`, `tilde p'_i` | corresponding antiholomorphic parities |
| `f,g` | total chiral three-form parity at the first/second pants |
| `eta,eta'` | signs labelling the two **holomorphic three-forms** |
| `zeta,zeta'` | signs labelling the antiholomorphic three-forms |
| `lambda_i`, `tilde lambda_i` | plumbing lifts, not three-form labels |
| `epsilon=+,-` | physical Ramond-family label, not `eta` |

For an even NS primary,

\[
x_1=\mathbf L_{-A}\phi,\qquad
x_2=\mathbb L_{-B}w^\alpha,\qquad
x_3=\mathbb L_{-C}w^\beta,
\quad
p=(\#G_A,\#G_B+\alpha,\#G_C+\beta)\pmod2.                 \tag{1}
\]

The exponent of a plumbing lift is the **whole state parity**, not just
the number of odd lowering operators. Define

\[
K(p)=p_1p_2+p_1p_3+p_2p_3\pmod2.
\]

For the generic-b continuum application, put `Q_L=b+1/b`,
`c=3/2+3 Q_L^2`, `h_1=Q_L^2/8+P_1^2/2`, and
`beta_i=i P_i/sqrt(2)` on the R edges. Then
`h_i=c/24-beta_i^2=1/16+Q_L^2/8+P_i^2/2` for `i=2,3`.
The algebraic checks below keep `h_1,beta_2,beta_3` symbolic.

Use `q^L0` propagation, consistently with the saved calculation. Primary
powers are kept outside the descendant blocks. Nothing here changes the
period matrix, spin marking, scalar zero-mode measure, or Weyl frame.

## 2. The physical pairing, not the auxiliary SCA+F pairing

For a homogeneous chiral basis with even bilinear BPZ Gram matrix `B`,
and an independently specified antichiral matrix `Btilde`, the Human Note
gives

\[
\mathcal G_{a\tilde a;b\tilde b}
=(-1)^{p_b\tilde p_{\tilde a}}
B_{ab}\widetilde B_{\tilde a\tilde b}.                    \tag{2}
\]

Since the chiral Gram matrices preserve parity, its inverse in this
product basis is

\[
\mathcal G^{-1}_{a\tilde a;b\tilde b}
=(-1)^{p_a\tilde p_{\tilde a}}
(B^{-1})_{ab}(\widetilde B^{-1})_{\tilde a\tilde b}.        \tag{3}
\]

Equation (3) uses the full unrestricted product space. It is not the
inverse on a selected Ramond subrepresentation; see section 7.

All these pairings are **bilinear**. A tilde is not an instruction to
complex-conjugate the holomorphic numerical answer at fixed momenta.
That identification, where valid, needs the antiholomorphic phase and
reflection dictionary. In particular, complex conjugation sends
`beta=i P/sqrt(2)` to `-beta`; simply conjugating the factors of `i`
while holding beta fixed is a different operation.

The “Ramond blocks from double Virasoro” section of the Human Note
explicitly uses an auxiliary `SCA tensor F` pairing without the exchange
sign, and an auxiliary odd ground norm of minus one. Those conventions
belong to the checked branching identity. They do not replace (2).
The physical free-Majorana denominator remains an independent computation.

## 3. Sign ledger on an unrestricted tensor product

Write `P_i=p_i+tilde p_i` modulo two for the full state parity. There are
three distinct signs before any block factorization:

| Origin | Exponent modulo two |
|---|---|
| Reordering the three full states in theta sewing | `K(P)` |
| Three inverse physical product Gram matrices | `sum_i p_i tilde p_i` |
| Grouping holomorphic operators before antiholomorphic operators on one pant | `tau=tilde p_1(p_2+p_3)+tilde p_2 p_3` |

The last line is the Koszul sign for the specified operator order. Any
additional Ramond-field cocycle or normalization phase must be kept
separately in the three-point tensor; it is not included by guessing a
value for `tau`.

In the unrestricted contraction, even chiral Gram matrices imply
`p_i=p'_i` and `tilde p_i=tilde p'_i`. The two occurrences of `tau` then
cancel. Expanding the other two exponents gives

\[
\begin{aligned}
K(p+\tilde p)+\sum_i p_i\tilde p_i
&=K(p)+K(\tilde p)
 +\sum_{i,j}p_i\tilde p_j\\
&=K(p)+K(\tilde p)
 +\left(\sum_i p_i\right)\left(\sum_j\tilde p_j\right)
 \pmod2.                                                \tag{4}
\end{aligned}
\]

For an even full three-point tensor, the two sums are equal to `f`.
Thus the remaining sign is

\[
(-1)^{K(p)}(-1)^{K(\tilde p)}(-1)^f.                     \tag{5}
\]

The first two factors are already in the Human-Note chiral blocks. They
must not be removed again by an “untwisting” at the physical boundary.
For all-NS even primaries, the additional coefficient conversion
`C_note^(1)=i C_top` gives `(-1)(i C_top)^2=C_top^2`.
The odd coefficient phase and the decomposition minus sign are two
separate ingredients; neither replaces the other.

## 4. A useful intermediate block-decomposition theorem

Suppose, for this section only, that we are sewing the unrestricted
product modules and that the ordered full vertex has been expressed as

\[
\mathcal T_v(x,\tilde x)
=(-1)^{\tau(p,\tilde p)}
\sum_{f,\eta,\zeta} t^{v}_{f;\eta\zeta}
\rho_f^{(\eta)}(x)\widetilde\rho_f^{(\zeta)}(\tilde x),
\qquad v=L,R.                                           \tag{6}
\]

Keep the left and right vertex coefficients distinct until the BPZ
identification has been made. Substitution of (6) and (3) into the
theta contraction, followed by (4), gives

\[
\begin{aligned}
Z_{\rm unrestricted}^{\Theta}
={}&\int\prod_{i=1}^3\frac{dP_i}{\pi}\,
 \prod_i q_i^{h_i}\tilde q_i^{\tilde h_i}\\
&\times\sum_{f=0,1}(-1)^f
 \sum_{\eta,\eta',\zeta,\zeta'}
 t^L_{f;\eta\zeta}\,t^R_{f;\eta'\zeta'}
 \mathbb F_f^{(\eta,\eta')}(\mathbf q;\boldsymbol\lambda)
 \widetilde{\mathbb F}_f^{(\zeta,\zeta')}
      (\tilde{\mathbf q};\tilde{\boldsymbol\lambda}).       \tag{7}
\end{aligned}
\]

This is a derived formula under its stated assumptions, **not yet the
physical NSRR answer**. It explains precisely what would be needed to
write the simple coefficient matrix `M=(-1)^f t^L tensor t^R`.

Even if each vertex coefficient is diagonal in `eta,zeta`, equation (7)
does not impose `eta=eta'`: the two pants choose their three-form signs
independently. In particular, a diagonal vertex is not a justification
for dropping opposite-sign chiral blocks.

## 5. Three-point normalizations: keep three bases separate

Let `E=C_even` and `O=C_odd` denote the outputs of the BRY-normalized
structure-constant function. Let `d_+,d_-` denote the actual equal-family
ground correlators `<V R^+ R^+>` and `<V R^- R^->`. BRY's equation (3.8)
gives

\[
d_+=\frac{E+O}{2},\qquad d_-=\frac{E-O}{2}.               \tag{8}
\]

In Suchanek's three-form convention, the vertex coefficients denoted
there by `C^(eta)` are `(d_+ + eta d_-)/2`. If the external field
normalizations are identified literally, therefore,

\[
c_+=\frac{E}{2},\qquad c_-=\frac{O}{2}.                  \tag{9}
\]

References: [BRY, section 3.1](https://arxiv.org/html/2201.05621#S3.SS1),
and [Suchanek, section 4](https://arxiv.org/html/0810.1203#S4), around
equations (84)–(86). These are primary normalization inputs, not factors
inferred by fitting genus-two modular agreement.

The current helper `hjs_rr_ns_constant` returns `(E,O)`, not the literal
`(c_+,c_-)` in (9). That naming does not by itself prove a checked sphere
implementation is wrong: its blocks and outer prefactors may absorb
factors of two. It does mean that the genus-two assembler cannot import
the helper name as a proof of the vertex normalization. No helper or
checked code is changed in this derivation.

For example, the algebraic sum of squared physical ground correlators is

\[
d_+^2+d_-^2=\frac{E^2+O^2}{2},
\qquad d_+^2-d_-^2=EO.                                  \tag{10}
\]

These are normalization checks, **not yet a fixed-spin theta ground
partition**. Choosing which sum, phases, and parity insertions occur
requires the physical sewing and its defect/spin prescription.
In particular, `2(E^2+O^2)` is not the sum in (10).

At generic `b` the same linear changes of basis apply to whatever
consistently normalized functions `E(b),O(b)` are supplied. This note
does not rederive the generic-b Upsilon leg normalization. A common
cosmological factor multiplies each full vertex and is squared in the
two-pants amplitude; it cannot repair a relative grading error.

## 6. Ramond ket states are not their BPZ duals

For a generic long Ramond module, Suchanek's small-representation ket
embedding can be written in product-basis order `(00,01,10,11)` as

\[
E_R=\frac1{\sqrt2}
\begin{pmatrix}1&0\\0&1\\0&1\\-i&0\end{pmatrix}.        \tag{11}
\]

Its columns are the even and odd physical ground kets. This is the
embedding of section 3, equation (16), of
[Suchanek](https://arxiv.org/html/0810.1203#S3). It is a representation
statement, not by itself a formula for the bra embedding in every BPZ
convention.

Here is a concrete warning against an inconsistent transplant. For
`G_0 w^+ = i beta exp(-i pi/4) w^-`, bilinear self-adjointness of `G_0`
and unit even ground norm give the chiral Gram `B=diag(1,i)`, as in the
protected PBW code. With Suchanek's opposite antichiral zero-mode phases,
the corresponding antichiral Gram is `Btilde=diag(1,-i)`. Equation (2)
then gives

\[
\mathcal D=\operatorname{diag}(1,-i,i,-1),\qquad
E_R^T\mathcal D E_R=\operatorname{diag}(1,0).
\]

The naive same-matrix pullback is singular. This calculation does **not**
contradict the physical Ramond two-point functions or the Human Note.
It says that combining this ket embedding, these chiral bilinear metrics,
and an unexamined identical bra embedding is not a valid BPZ dictionary.

An algebraic dual illustrating the distinction is

\[
E_L=\mathcal D^{-1}\overline{E_R}
=\frac1{\sqrt2}
\begin{pmatrix}1&0\\0&i\\0&-i\\-i&0\end{pmatrix},
\qquad E_L^T\mathcal D E_R=I_2.
\]

This example keeps the graded bilinear metric throughout. It does not
replace physical sewing with a Hermitian norm. Nor is this particular
`E_L` asserted to be the production BPZ continuation: the corresponding
left vertex and descendant action have to be transported with it. It
exhibits explicitly why a ket formula alone cannot determine the sewing.

The ket representation itself checks correctly with the graded action
`G=G_chiral tensor 1`, `Gtilde=(-1)^F_chiral tensor Gtilde_chiral`.
The singular pairing above is thus a dual-identification issue, not a
failure of the Ramond ket representation.

## 7. The exact restricted sewing formula

At every pair of descendant levels, let `E_R,E_L` embed the selected ket
and BPZ-dual spaces in the full product spaces. Let `D` be their full
graded pairing, and assume the restricted pairing is nondegenerate:

\[
g=E_L^T D E_R,\qquad
\mathcal C=E_R g^{-1}E_L^T,\qquad
\Pi=\mathcal C D,\qquad \Pi E_R=E_R.                    \tag{12}
\]

`C` is the contravariant sewing tensor, not the identity matrix and not
the ordinary inverse of `D` on all four ground states. `Pi` is the
associated projector. Formula (12) is invariant under independent
changes of physical bra and ket bases: `E_L -> E_L S_L`,
`E_R -> E_R S_R`, since `g -> S_L^T g S_R`.

Let `I_i` and `J_i` now be full nonchiral product-basis indices on edge
`i`, with total parity `P(I_i)`. The full ordered pants tensors are
`T_L(I_1,I_2,I_3)` and `T_R(J_1,J_2,J_3)`. Including the primary and
descendant propagation and the compatible spin transport in `W(I)`, the
physical sewing has the concrete form

\[
\boxed{
Z_{\rm NSRR}^{\Theta}
=\int\prod_i\frac{dP_i}{\pi}
 \sum_{I,J} W(I)\,(-1)^{K(P(I))}
 T_L(I_1,I_2,I_3)
 \left[\prod_{i=1}^3\mathcal C_i^{I_iJ_i}\right]
 T_R(J_1,J_2,J_3).
}                                                       \tag{13}
\]

The NS edge uses the usual full product inverse pairing; the two R
edges use (12). There is no auxiliary fermion in (13). All phases in the
ordered physical vertices stay explicit. If a propagation/spin operator
does not preserve the chosen subspace, it must be inserted in (12) with
the correct source and target embeddings, rather than pulled out as a
diagonal weight `W`.

Two consequences matter when reducing (13) to ordinary chiral blocks:

1. A physical Ramond projector can preserve `p_i+tilde p_i` while
   mixing `p_i` and `tilde p_i` separately. In the ground example of
   section 6, `Pi` commutes with total parity but not chiral parity.
   One cannot then assume that the two vertex-ordering signs cancel
   term by term as in section 3.
2. If a term flips the individual chiral parities on the R edges by
   `r_2,r_3`, it flips the antichiral parities by the same amounts, and
   the two pants satisfy `g=f+r_2+r_3` modulo two. In particular,
   intermediate contractions can involve different three-form parities
   at the two pants. Reducing these inserted contractions to the
   available `F_f^(eta,eta')` requires the Ramond Ward identities; it is
   not equivalent to keeping only diagonal absolute squares.

In the same ground example, simultaneous holomorphic and antiholomorphic
parity insertion preserves the small space, but a one-sided insertion
does not. Therefore the lift-to-spin dictionary must be established
together with this restriction. Arbitrary independent chiral lift
choices are not automatically a physical spin sewing.

Equation (13) is the starting point for completing the derivation. To
obtain the desired coefficient matrix `M`, expand the two restricted R
sewing tensors in chiral/antichiral operators, retain the two `tau`
signs, and reduce the resulting inserted chiral contractions by the
zero-mode/descendant Ward identities. Only then combine three-form labels.

## 8. Exact low-level chiral checkpoints

The following calculations use the Human-Note forms and the protected
PBW ground metric; they require no proposed nonchiral projector.
Let `k=eta eta'`, and write `F_f[0]` for the coefficient at zero
descendant level. Directly summing the two allowed R ground pairs gives

\[
\begin{aligned}
\mathbb F_0^{(\eta,\eta')}[0]&=1+k\lambda_2\lambda_3,\\
\mathbb F_1^{(\eta,\eta')}[0]&=-i(\lambda_3-k\lambda_2).
\end{aligned}                                           \tag{14}
\]

For clarity, the four individual ground contributions are

| `f` | R ground pair | inverse Gram | theta orientation | contribution |
|---|---|---|---|---|
| 0 | `00` | `1` | `+1` | `1` |
| 0 | `11` | `-1` | `-1` | `k lambda_2 lambda_3` |
| 1 | `01` | `-i` | `+1` | `-i lambda_3` |
| 1 | `10` | `-i` | `+1` | `+i k lambda_2` |

The last row uses `(i eta)(i eta')=-k` from the two odd forms; it is
not an optional sign in the Gram matrix.

Next let `u=exp(-i pi/4)` and `v_eta=u(beta_3-eta beta_2)`. Applying
the protected contour Ward rule to `G_{-1/2} phi` gives

\[
\begin{array}{c|cc}
 &\text{ground pair}&\rho_f^{(\eta)}(G_{-1/2}\phi,\cdot,\cdot)\\\hline
f=0&01&-i v_\eta\\
f=0&10&\eta v_\eta\\
f=1&00&v_\eta\\
f=1&11&\eta v_\eta
\end{array}                                               \tag{15}
\]

All other entries vanish by total parity. The NS norm is `2h_1`.
Since `u^2=-i`, inserting these entries into the same chiral sewing
gives, for `A=(beta_3-eta beta_2)(beta_3-eta' beta_2)`,

\[
\begin{aligned}
\mathbb F_0^{(\eta,\eta')}[\tfrac12,0,0]
 &=-\frac{\lambda_1 A}{2h_1}(\lambda_3-k\lambda_2),\\
\mathbb F_1^{(\eta,\eta')}[\tfrac12,0,0]
 &=-\frac{i\lambda_1 A}{2h_1}(1+k\lambda_2\lambda_3).
\end{aligned}                                             \tag{16}
\]

These are the coefficients of `q_1^(1/2)`, with both R descendant
levels zero. They are not full truncated partition functions.

For `lambda_2=lambda_3=+1`, display the chiral sign indices in order
`(+,-)`. Equations (14)–(16) become

\[
\begin{aligned}
\mathbb F_0[0]&=2I,&\qquad \mathbb F_1[0]&=-2i\sigma_x,\\
\mathbb F_0[\tfrac12,0,0]
 &=\frac{\lambda_1(\beta_2^2-\beta_3^2)}{h_1}\sigma_x,
&
\mathbb F_1[\tfrac12,0,0]
 &=-\frac{i\lambda_1}{h_1}
 \operatorname{diag}\bigl((\beta_3-\beta_2)^2,
                         (\beta_3+\beta_2)^2\bigr).
\end{aligned}                                             \tag{17}
\]

Thus an opposite-sign **odd** block is already nonzero at level zero;
an opposite-sign **even** block is generically nonzero at the first NS
descendant. Choosing equal R momenta hides the latter checkpoint and is
unsuitable for testing whether that component was accidentally omitted.

## 9. Reproducible checks and next derivation step

Companion test: `Code/genus_2/test_nsrr_sewing_derivation.py`.
Run from the repository root:

```sh
env PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  PYTHONPATH=Code:Code/genus_2:Code/c_Recursion:Code/full_ramond_block_runtime:Code/double_virasoro/nsrr \
  python3 -m unittest \
  Code/genus_2/test_nsrr_sewing_derivation.py \
  Code/genus_2/test_graded_sewing_audit.py \
  Code/genus_2/test_fixed_spin_free_plumbing.py \
  Code/genus_2/test_nsrr_checked_kernel_boundary.py
```

The new tests check (15) against the protected Ward implementation,
and (14)/(16) symbolically for all choices of `f,eta,eta',lambda_i`.
They also verify the three coefficient bases, graded zero-mode action on
the small ket representation, the singular naive BPZ pullback, the
algebraic two-sided dual example, parity mixing, and basis invariance.
The earlier audit exhausts all 64 parity pairs in (4), and checks full
finite tensor contractions against (7) with one fixed three-form label.
The independent free-factor and protected-hash tests are included too.

Validation result: **30 tests pass** (8 new derivation tests, 8 graded
sewing tests, 13 independent free-factor tests, and 1 protected-kernel
manifest test). The latter checks all eight protected source hashes.

Local source anchors in `Human Notes/SCblock.tex` are
`eq: Gram_Factorialization`, `NSblockThetaDefinition`, `Rblock`, and the
section “Ramond blocks from double Virasoro.” The symbolic Ward tests
call `Code/double_virasoro/nsrr/ramond_pbw_generalized_ward.py` read-only.

Next, compute the **ordered physical RRNS vertex together with its BPZ
dual in the same local frame**, including one NS odd descendant and one
Ramond odd descendant. Suchanek's component Ward identities and the
BRY ground correlators provide independent checks of that conversion.
Transport both vertices and both R sewing tensors in (13); then carry
out the finite parity reduction to chiral blocks. Ground normalization
alone does not certify this step.

No protected code, Human Note, saved plumbing, or archived numerical
partition has been modified. The test with a chosen dual is explicitly
an algebraic consistency example, not a newly certified physical
projector. The current diagonal-norm NSRR toy must remain labelled a
diagnostic until the last local vertex/dual conversion is established.
