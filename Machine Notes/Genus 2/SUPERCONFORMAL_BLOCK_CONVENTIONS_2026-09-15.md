# Literature conventions for the superconformal block comparison

Date: 2026-09-15.

## Primary powers and subsequent coefficient bookkeeping

The user's block convention is descendant-only. Every expression below
that suppresses propagation means
`Z=sum_AB M_AB P_A Ptilde_B F_A Ftilde_B`, with
`P_A=exp(sum_i h_i,A Log(q_i))` outside both `F` and `M`.
The numerical field `chiral_value` in the complex free-field comparison
denotes the propagated amplitude `P*F`; its primary factor, weights and
descendant-only values are now recorded separately.

The subsequent derivation, including the channel-dependent weights,
three-point products, unreduced Ramond vertex tensor and remaining
saved-basis matrix problem, is in
[NSRR coefficient and primary ledger](</Users/yutaizhang/Desktop/Type0B-Matrix/Machine Notes/Genus 2/NSRR_COEFFICIENT_AND_PRIMARY_LEDGER_2026-09-15.md>).

## Clarification: the sign of a term in the full decomposition

The question is to determine the coefficient of `F_A Ftilde_B` with the
block definitions held fixed. A difference between individual block phases
does not establish a difference between physical correlators. In particular,
the coefficientwise NS comparisons below are checks of a basis conversion,
not by themselves a derivation of the physical decomposition.

Two direct literature anchors concern the decomposition itself:

\[
 \langle VVVV\rangle
 =\int\frac{dP}{\pi}\left[
 C_L C_R\,\mathcal F_e(z)\mathcal F_e(\bar z)
 +\widetilde C_L\widetilde C_R\,
 \mathcal F_o(z)\mathcal F_o(\bar z)\right].
\]

BRY (A.1) has this relative **plus**. BRY (A.2) has an overall **minus**
for `VWWV`, together with the exchanged even/odd structure-constant products;
the paragraph after (A.3) explains the anti-Hermiticity of `W`.
These are observable-dependent signs, not a freedom to choose a preferred
block phase. [BRY, Appendix A](https://arxiv.org/html/2201.05621#A1).

For the theta blocks defined in the Human Note, the explicit grading sign
is `(-1)^(a+p_1+p_2+p_3)`. For even NS primaries this gives, with common
propagation and integration suppressed,

\[
 Z_\Theta=(C_{\rm HN}^{(0)})^2F_0\widetilde F_0
           -(C_{\rm HN}^{(1)})^2F_1\widetilde F_1.
\]

Thus the sign explicitly preceding the odd term in this formula is minus.
With the implemented `C_HN^(1)=i*Ctilde_BRY` dictionary, the complete
odd coefficient is `-(i*Ctilde_BRY)^2=+Ctilde_BRY^2`. One must keep both
ingredients. Replacing the coefficient product by its absolute square
would erase the phase needed for this cancellation.

The grading derivation is short. Define `K(p)=sum_(i<j) p_i p_j` for the
three complete chiral state parities. The unrestricted product pairing
and theta orientation give

\[
 K(p+\tilde p)+p\cdot\tilde p
 =K(p)+K(\tilde p)
  +\left(\sum_i p_i\right)\left(\sum_j\tilde p_j\right)
 \pmod2.
\]

An even full vertex has matching total chiral and antichiral parity `f`.
After the two `K` factors have been included in the chiral blocks, the
remaining factor is therefore `(-1)^f`. For separately specified ordered
vertex coefficients the unrestricted formula is

\[
 Z_{\rm unrestricted}
 =\sum_{f,\eta,\eta',\zeta,\zeta'}
 (-1)^f\,t^L_{f;\eta\zeta}\,t^R_{f;\eta'\zeta'}
 F_f^{\eta\eta'}\widetilde F_f^{\zeta\zeta'}.
\]

The physical Ramond restriction must be inserted before using this
simplification. The complete physical NSRR coefficient matrix has not
been derived here. In particular, the free-field phase correction and a
choice between `F_0+iF_1` and `F_0-iF_1` do not derive that matrix.
Suchanek (45) has a plus between the two absolute-square contributions
for four `R+` fields, while (46) explicitly has `-i*Ctilde_L*C_R^(eta)`
for the odd contribution to `phi phi R+ R+`. Their different vertices
and constants are part of those formulas.
[Suchanek, (45)-(49)](https://arxiv.org/pdf/1012.2974#page=16).

## Conclusion

The single-Majorana square-root phase and the superconformal sewing
convention are separate pieces of data. The earlier bosonization audit
checks the former. It does **not** validate the interacting NSRR
contraction.

The NS sphere conversion is explicit and passes a fresh coefficient test.
The Ramond comparison exposes an outgoing-bra convention that must be
transported before importing literature vertices into the bilinear
genus-two calculation. The existing legacy contraction also fails a
free-superfield check in both magnitude and phase. This audit does not
change the checked block kernels or infer a replacement interacting
kernel from that free example.

## 1. Literature anchors

- **BRY:** the scalar decomposition (3.13) involves absolute squares;
  footnote 8 explicitly leaves a holomorphic sign undetermined by that
  decomposition. Equations (3.14), (3.18) specify signs, including starred
  insertions. Equations (3.6)-(3.10) distinguish physical Ramond fields
  and their constants. [Balthazar, Rodriguez, Yin](https://arxiv.org/html/2201.05621).
- **Suchanek:** section 2 defines an outgoing-antilinear three-form;
  (29) and (30) distinguish NR and RN odd forms; (31), (37) include an
  explicit relative `-i` in local nonchiral vertices. Footnotes 5 and 6
  correct treating RN as the conjugate NR form. Section 3.1 gives the
  corresponding Ramond sphere sewing. [Suchanek](https://arxiv.org/pdf/1012.2974).
- **HJS:** (3.3) defines the algebraic Gram form in a basis containing
  `G_0 w^+`; (3.6)-(3.7) specify the small Ramond representation and
  the left/right zero-mode phases. Those prescriptions require care
  under a complex change of ground basis.
  [Hadasz, Jaskolski, Suchanek](https://arxiv.org/pdf/0810.1203).
- **Belavin-Geiko:** (2.1)-(2.4) use a central charge equal to `2c/3`
  in our notation; (2.11)-(2.17) fix ordered NS matrix elements.
  The paper addresses NS blocks and does not supply a fixed-spin
  interacting NSRR genus-two contraction.
  [Belavin, Geiko](https://arxiv.org/pdf/1806.09563).

The calculations below are our convention conversions and checks. A
ground-state check is distinguished from an all-descendant identification.

## 2. What must be compared

In a fixed chiral basis, write the nonchiral decomposition as

\[
 Z=\mathcal F^T M\overline{\mathcal F}.
\]

Here `M` includes the structure constants, the pairing and any specified
spin/defect contraction. If `F'=U F`, consistency requires

\[
 M'=U^{-T}M\overline U^{-1}.
\]

For independent antichiral blocks, use `F^T M Ftilde` and transform the
two bases independently. Identifying `Ftilde` with a numerical conjugate
is an additional convention, particularly when `beta` is imaginary.

A common unit phase drops out of an absolute square. Relative phases
affect off-diagonal terms in `M`. Therefore a modulus comparison of
individual blocks cannot validate a contraction that mixes them.

The geometry comparison also needs the same local coordinates, spin lifts,
primary powers and logarithm branches. The saved genus-two amplitudes use
`q^L0`; primary powers sit outside their descendant series. A sphere's
elliptic block `H(q)` contains different prefactors from a plumbing block.
Their raw numerical values should not be equated before restoring those
prefactors.

## 3. Explicit NS dictionary

Write `H_saved^p` for our sphere block, with internal parity `p=0,1`.
Let `s_2,s_3` specify the two middle external level-one-half descendants;
the states at zero and infinity are bottom components. In the ordering
`(h_4,*^s_3 h_3,*^s_2 h_2,h_1)`, comparison with the cited BRY definitions
gives

\[
 \boxed{\mathcal F_{\rm BRY}^{p}
   =(-1)^{p(1+s_3+s_2s_3)}H_{\rm saved}^{p}.}
\]

| `(s_2,s_3)` | Even multiplier | Odd multiplier |
|---|---:|---:|
| `(0,0)` | +1 | -1 |
| `(1,0)` | +1 | -1 |
| `(0,1)` | +1 | +1 |
| `(1,1)` | +1 | -1 |

For four bottom components, put `a=h+h_3-h_4`, `b=h+h_2-h_1`.
The first terms, with the common primary power displayed, are

\[
 H^0=z^{h-h_1-h_2}\left(1+\frac{ab}{2h}z+\cdots\right),\qquad
 H^1=z^{h-h_1-h_2}\left(-\frac{z^{1/2}}{2h}+\cdots\right).
\]

Thus the odd block differs by phase pi while its norm agrees. This sign
comes from the saved ordered three-forms:

\[
 \rho_1(G_{-1/2}\nu,\nu,\nu)=1,\quad
 \rho_1(\nu,G_{-1/2}\nu,\nu)=1,\quad
 \rho_1(\nu,\nu,G_{-1/2}\nu)=-1.
\]

It is already present in `superconformal_blocks.py`; its comments about
the signed seed must be read as a statement about the saved basis, not
as literal equality to every BRY block definition.

**Fresh verification:** the independent `LiteralBRY` reference was checked
against the saved implementation for all four star patterns, two sets
of asymmetric real/complex external weights, and levels 0 through 4.
All 72 coefficient comparisons agree after conversion, with maximum
scaled error `2.3336307292e-61` at 60-digit precision. Complex evaluations
at `z=0.23+0.17i` separately record the norm and phase.

The reference includes the endpoint `rs=2m`, since a null pole contributes
at its own level. BRY's displayed strict inequality omits that endpoint;
this detail is recorded in the preexisting independent implementation.
No pole, fusion or seed function is shared with the saved reference target.

### The all-NS nonchiral coefficient carries another phase

Our genus-two formula contains `(-1)^a (C_HN^(a))^2`. For an even
primary, the graded bilinear norm of the raw tensor of two odd level-one-half
states is `-(2h)^2`. Multiplication by `i` converts it to `+(2h)^2`.
The implemented all-NS coefficient dictionary is

\[
 C_{\rm HN}^{(0)}=C_{\rm BRY},\qquad
 C_{\rm HN}^{(1)}=i\widetilde C_{\rm BRY},\qquad
 (-1)(i\widetilde C_{\rm BRY})^2=\widetilde C_{\rm BRY}^{\,2}.
\]

Choosing `-i` gives the same two-pants product. This is a convention for
the bilinear factorization of the full top component. It does not follow
from the phase of the free Majorana and does not determine the NSRR kernel.
The implementation already applies it in `ns_genus2_partition._structure_weight`.

## 4. Ramond: the outgoing dual is essential

The local human-note NR forms reproduce the pattern

\[
 \rho_0^{\eta}=\rho^{++}+\eta\rho^{--},\qquad
 \rho_1^{\eta}=\rho^{+-}+i\eta\rho^{-+}.
\]

For an outgoing Ramond state the literature RN odd form instead orders
its components as `rho^{-+}+i eta rho^{+-}`. Relocating the NS puncture
therefore involves an ordered-form and dual-state conversion. Relabeling
the slots while retaining the same numerical tensor is insufficient.

### An exact one-edge comparison

Use the physical momentum line `beta=iP/sqrt(2)`, `P>0`, and define

\[
 u=G_0w^+=a w^-,\quad a=i\beta e^{-i\pi/4},\quad
 \langle u,u\rangle=-\beta^2=P^2/2.
\]

Normalize both ground NR/RN three-form coefficients in the `w^-` basis
to one in the outgoing-antilinear convention. In the `u` basis, the
incoming coefficient is `a` and the outgoing coefficient is `a^*`.
The sphere odd ground contribution is

\[
 \frac{a^*a}{-\beta^2}=1.
\]

In contrast, our **bilinear** ground metric is

\[
 B(w^-,w^-)=\frac{-\beta^2}{a^2}=i.
\]

This agrees exactly with `RamondPBWModule.ground_pairing`. Combining this
metric with two unconverted unit literature ground coefficients gives

\[
 \frac{1\cdot1}{i}=-i:
 \quad\text{norm ratio }1,\quad\text{phase error }-\pi/2.
\]

The outgoing odd coefficient in this bilinear realization needs the
factor `a^*/a=i`; then `1*i/i=1`. Equivalently, work throughout with
the outgoing-antilinear convention. These are consistent alternatives.
The calculation does not say that the saved bilinear Gram matrix is wrong.

This example fixes a **ground outgoing-dual conversion**. It does not
prove that multiplying a whole genus-two odd block by `i` implements
the full local-coordinate, descendant, reflection and spin dictionary.
The genus-two graph uses two pants, three pairings and an explicit
quadratic grading sign.

The conjugation in the example is used to identify a state dual on the
physical momentum line. It is not a prescription to conjugate `q` or
to make the holomorphic block nonholomorphic. After identifying the dual,
analytically continue the resulting chiral functions.

### Parameters and constants also need an explicit map

Our weight is `h_R=c/24-beta^2`. Suchanek's (3),(6) give
`a=Q/2+ip`, `beta=(Q/2-a)/sqrt(2)=-ip/sqrt(2)`.
Consequently our `beta=+iP/sqrt(2)` corresponds to `p=-P` in that
parametrization. Equal conformal weights alone do not fix reflected field
normalization or the signs labelling three-forms. This is a convention
choice to track, not by itself evidence of a wrong block.

Separate the physical Ramond field constants `d_+,d_-` from the
coefficients `c_+,c_-` of the chiral-form combinations. Once physical
field phases and reflection conventions have been matched, the algebra is

\[
 d_\pm=\frac{E\pm O}{2},\qquad
 c_\pm=\frac{d_+\pm d_-}{2},\qquad
 (c_+,c_-)=(E/2,O/2).
\]

This change of coordinates on the two structure constants does not derive
a nonchiral sewing tensor. Likewise, the `-i` in a local literature
vertex is not a license to infer `|F_0+iF_1|^2` for a closed genus-two
graph without transporting its pairings and spin restriction.

## 5. Fresh complex free-superfield check

Take `c=3/2`, scalar charges `(0.31,-0.47,0.16)` in geometry order
`(zero,one,infinity)`, with the first two edges Ramond. Use the saved
two-lift projection, chiral form signs `eta=eta'=+`, and the charge
characteristic `[11|00]`. The independent comparison is

\[
 A_{\rm free}=Z_{X,\mathrm{charged}}^{\mathrm{chiral}}
 Z_{\psi}^{\mathrm{chiral}}.
\]

The scalar is computed by Heisenberg resummation and the Majorana by the
phase-preserving bosonized evaluator, with its sign fixed by radial
continuation before inspecting the SC result. The entire calculation is
in the charge chart, so its period-translation matrix is zero. The saved
SC block is independently evaluated by low-order PBW sewing.

Strip the common primary power and set the Ramond descendant levels to
zero. With `x=sqrt(q_NS)`, the direct Ward calculation gives

\[
 F_0=\sqrt2(1+x/2+\cdots),\qquad
 F_1=-i\sqrt2(1-x/2+\cdots).
\]

The even projected block agrees with the free reference. The legacy
combination `A_legacy=(F_0+iF_1)/2` loses the linear term. Hence

\[
 |A_{\rm free}|^2=2+x+\bar x+O(|x|^2),\qquad
 |A_{\rm legacy}|^2=2+O(|x|^2).
\]

For `q=scale*q_generic04`, total PBW order 3 gives:

| Scale | Quantity divided by `A_free` | Magnitude | Phase in degrees |
|---:|---|---:|---:|
| 0.02 | Even projected SC block | 1.00000000000177 | +0.000000005905 |
| 0.02 | Legacy combination | 0.996122115518 | -0.722526569 |
| 0.10 | Even projected SC block | 0.999999997856 | +0.000001559288 |
| 0.10 | Legacy combination | 0.991157684425 | -1.597259737 |

Order 2 versus order 3 confirms improvement of the even projected block;
its complex relative errors at order 3 are `1.03e-10` and `2.73e-8`.
The legacy complex discrepancies remain `0.01317` and `0.02913`.
All underlying complex values and both orders are saved.

The old nonchiral diagnostic was also rerun: its local kernel/reference
ratios are `0.9922592690` and `0.9823935554`. The separate factor four
still used in `nsrr_resummed_sewing.py` changes those to `3.9690370761`
and `3.9295742216`. Neither a constant normalization nor a common phase
can repair the missing series coefficient.

The free test checks the complete candidate vertex/reflection/projection
prescription under the stated calibration. It does not uniquely locate
every error or determine the interacting Liouville replacement.

## 6. Relation to the earlier Majorana correction

In the saved source marking, the theta translation contributes `exp(i*pi/4)`
to the doubled chiral fermion and `exp(i*pi/8)` to its continued single
Majorana root. That factor converts marked theta data to the sewing
frame. It depends on the marking and should not be imported as a
universal superconformal block rephasing.

For a double-Virasoro identity `A_(SCA+F)=A_SCA*A_F`, a chiral quotient
requires all three factors in the same convention, including the
normalization of the auxiliary Majorana. Formal power-series division
already chooses a branch through its leading coefficient. A second
endpoint phase multiplication can double count that choice.

The verified result is therefore: a single Majorana requires its chiral
phase; the NS block basis has a verified literature conversion; the
interacting NSRR physical sewing still needs the complete ordered vertex
and outgoing-dual conversion before the comparison can be certified.

## 7. Reproduction and verification

New diagnostic:

```sh
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 \
/private/tmp/type0b-nsrr-smoke-venv/bin/python \
Code/genus_2/audit_superconformal_block_conventions.py
```

The existing free-limit diagnostic was rerun with output directed to the
same new dataset. Seventeen existing graded-sewing, NSRR derivation and
protected-kernel tests passed. The protected-kernel hashes agree with
the checked manifest.

- [Complex convention data](</Users/yutaizhang/Desktop/Type0B-Matrix/Data Set/superconformal_block_conventions_20260915/conventions.json>)
- [Rerun nonchiral free-limit check](</Users/yutaizhang/Desktop/Type0B-Matrix/Data Set/superconformal_block_conventions_20260915/free_limit.json>)
- [Diagnostic source](/Users/yutaizhang/Desktop/Type0B-Matrix/Code/genus_2/audit_superconformal_block_conventions.py)
- [Earlier explicit graded derivation](</Users/yutaizhang/Desktop/Type0B-Matrix/Machine Notes/Genus 2/NSRR_NONCHIRAL_SEWING_DERIVATION_2026-08-30.md>)
