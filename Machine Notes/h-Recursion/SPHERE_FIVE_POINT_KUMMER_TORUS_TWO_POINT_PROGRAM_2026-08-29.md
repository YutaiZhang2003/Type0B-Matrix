# Sphere five-point blocks from the Kummer torus: a Fourier--Jacobi \(h\)-recursion program

Date: 2026-08-29

> **Superseded conceptual route.** The direct five-point generalization is
> an open pillow-cylinder matrix element with one additional operator and two
> representation projectors, not an ordinary torus two-point trace. The
> corrected derivation is in
> SPHERE_FIVE_POINT_PILLOW_MATRIX_ELEMENT_H_RECURSION_2026-08-29.md.
> This file is retained as an exploration of why the ordinary CCY torus
> two-point block is an insufficient identification and as an auxiliary cusp
> calculation.

Status: exploratory machine note.  Sections 2--4 record established geometry
and the Cho--Collier--Yin (CCY) recursion.  Sections 5--10 formulate a
research program.  A proposed identity is not to be used as production input
until it passes the coefficient tests in Section 9.

## 1. Objective and present conclusion

The sphere four-point elliptic representation succeeds because the single
cross-ratio is uniformized by

\[
 z=\lambda(\tau),\qquad q_{\rm pil}=e^{\pi i\tau},
\]

and the known large-internal-weight prefactor leaves a reduced block with unit
seed.  For a sphere five-point block there are two complex moduli, so one
should not apply the inverse modular lambda function independently to the two
linear-channel plumbing variables.

The natural two-dimensional replacement is the Kummer presentation of
\(\mathcal M_{0,5}\).  Four punctures determine an elliptic curve with level-2
structure, and the fifth puncture determines a point on its Kummer quotient.
The resulting variables are

\[
 (\tau,u),\qquad
 q_{\rm pil}=e^{\pi i\tau},\qquad
 Q=e^{2\pi i\tau}=q_{\rm pil}^{\,2},\qquad
 \zeta=e^{2\pi i u}.
\]

On the smooth double cover, the mobile sphere puncture lifts to the pair
\(u,-u\).  The relevant genus-one object is therefore a once-punctured
*pillow*, or equivalently a \(\mathbb Z_2\)-symmetric torus two-point object.

CCY give a complete simultaneous-weight \(h\)-recursion for the ordinary
torus two-point necklace block.  This recursion is an essential computational
ingredient, but it does **not** by itself prove an equality between a generic
sphere five-point block and a generic ordinary torus two-point block.  The
four branch-point operators remain as corner/twisted data on the pillow.  The
generic problem is therefore

\[
 \boxed{
 \text{sphere five-point block}
 \longrightarrow
 \text{corner-decorated symmetric torus two-point block}
 +\text{covering prefactor}.}
\]

There are two viable programs:

1. identify and test a special Poghossian-type locus on which the decorated
   object collapses to the ordinary CCY torus two-point block;
2. for generic weights, use the Kummer variables to reorganize the existing
   sphere five-point block directly and derive the missing decorated
   Fourier--Jacobi recursion.

The first new structural observation is that the CCY torus two-point block,
when organized at the torus cusp, has a sphere four-point block as its entire
leading coefficient.  This provides a controlled first elliptic resummation
and a sharp test of any proposed two-variable construction.

## 2. Kummer coordinates on the five-punctured sphere

Fix five labeled points on the sphere.  Choose four of them as branch points
\(x_1,x_2,x_3,x_4\), and call the remaining point \(t\).  The double cover

\[
 E:\qquad y^2=\prod_{a=1}^{4}(x-x_a)
\]

has genus one.  Let

\[
 2\omega_A=\oint_A\frac{dx}{y},\qquad
 2\omega_B=\oint_B\frac{dx}{y},\qquad
 \tau=\frac{\omega_B}{\omega_A},
\]

and normalize the Abel coordinate to have periods \(1,\tau\):

\[
 u(t)=\frac{1}{2\omega_A}\int_{x_*}^{t}\frac{dx}{y}
 \quad\bmod (\mathbb Z+\tau\mathbb Z).
\]

The two preimages of \(t\) are represented by \(u\) and \(-u\).  The inverse
map is an elliptic function,

\[
 t=X(u\mid\tau),
\]

where \(X\) is a fractional-linear transform of \(\wp(u\mid\tau)\).  With a
Legendre choice of branch points, their cross-ratio is

\[
 \lambda(\tau)=\frac{\theta_2(0\mid\tau)^4}
                         {\theta_3(0\mid\tau)^4}.
\]

For labeled branch points the modular group is reduced to the appropriate
level-2 subgroup.  The remaining identifications have Jacobi form,

\[
 (\tau,u)\sim
 \left(\frac{a\tau+b}{c\tau+d},\frac{u}{c\tau+d}\right),
 \qquad
 u\sim u+m+n\tau,
 \qquad
 u\sim-u.
\]

Thus \(\mathcal M_{0,5}\) is represented by a Kummer universal elliptic
family with level-2 structure.  The four collisions of the mobile puncture
with the branch punctures occur at the four half-periods

\[
 \omega_a\in
 \left\{0,\frac12,\frac\tau2,\frac{1+\tau}{2}\right\}.
\]

The natural functions detecting these divisors are the torus prime forms

\[
 E(u-\omega_a\mid\tau)
 =\frac{\theta_1(u-\omega_a\mid\tau)}
        {\theta_1'(0\mid\tau)}.
\]

### 2.1 A concrete sphere gauge

For the project linear-channel convention

\[
 (z_1,z_2,z_3,z_4,z_5)
 =(0,x_1x_2,x_2,1,\infty),
\]

one convenient Kummer chart takes

\[
 (0,x_2,1,\infty)
\]

as the branch quartet and \(t=x_1x_2\) as the mobile point.  Then \(\tau\)
is determined by the cross-ratio \(x_2\), while \(u\) is the normalized Abel
image of \(x_1x_2\).  Other choices of branch quartet give the other channel
charts and must ultimately be related by the \(S_5\) action.

This distinction in notation is important:

- \(x_1,x_2\) are sphere linear-channel plumbing variables;
- \(q_{\rm pil},Q,\zeta\) are Kummer/Fourier--Jacobi variables;
- \(p_1,p_2\) below are CCY torus-necklace cylinder variables.

They should never all be denoted by \(q_i\) in the same calculation.

## 3. Ordinary Virasoro torus two-point necklace block

Use the flat coordinate

\[
 w\sim w+2\pi\sim w+2\pi\tau
\]

and place the two torus operators at \(0\) and \(z\).  In the project
convention,

\[
 p_1=e^{iz},\qquad
 p_2=e^{i(2\pi\tau-z)},\qquad
 p_1p_2=Q=e^{2\pi i\tau}.
\]

Let internal weights \(h_1,h_2\) propagate on the two cylinders, and let the
external primaries of weights \(d_1,d_2\) join the two edges cyclically.  The
chiral block is normalized as

\[
 \mathcal B_{\rm neck}
 =p_1^{h_1-c/24}p_2^{h_2-c/24}
 \sum_{n_1,n_2\geq0}
 F_{n_1,n_2}\,p_1^{n_1}p_2^{n_2}.
\]

Set

\[
 a=h_2-h_1,
 \qquad H=h_1,
 \qquad \chi(Q)=\prod_{m=1}^{\infty}\frac{1}{1-Q^m},
\]

and define the reduced block \(f\) by

\[
 \sum_{n_1,n_2\geq0}F_{n_1,n_2}p_1^{n_1}p_2^{n_2}
 =\chi(Q)f(H,a;p_1,p_2).
\]

The simultaneous limit used by CCY is

\[
 H\to\infty,\qquad a=h_2-h_1\ \text{fixed}.
\]

The unstripped descendant block tends to \(\chi(Q)\), while the reduced block
has unit boundary value.

## 4. Exact \(N=2\) specialization of the CCY \(h\)-recursion

Let \(h_{r,s}(c)\) be the Virasoro degenerate weight, let
\(\ell_{r,s}=rs\), and let \(A_{r,s}(c)\) be the inverse null-norm slope.
Write

\[
 \mathcal R_{r,s}(h_L,h_R;d_L,d_R;c)
 =A_{r,s}(c)
 P_{r,s}(h_L,d_L;c)P_{r,s}(h_R,d_R;c),
\]

with the understanding that the precise slot ordering follows the CCY fusion
polynomial convention.  In the two-edge case the same neighboring weight
appears on both sides of the singular edge, while the two external weights
remain distinct.

The reduced block obeys

\[
\begin{aligned}
 f(H,a;p_1,p_2)
={}&1\\
&+\sum_{r,s\geq1}
 \frac{p_1^{rs}\,
 \mathcal R_{r,s}
 \bigl(h_{r,s}+a,h_{r,s}+a;d_2,d_1;c\bigr)}
 {H-h_{r,s}}
 f\bigl(h_{r,s}+rs,a-rs;p_1,p_2\bigr)\\
&+\sum_{r,s\geq1}
 \frac{p_2^{rs}\,
 \mathcal R_{r,s}
 \bigl(h_{r,s}-a,h_{r,s}-a;d_1,d_2;c\bigr)}
 {H+a-h_{r,s}}
 f\bigl(h_{r,s}-a,a+rs;p_1,p_2\bigr).
\end{aligned}
\]

The shifts are easiest to check in absolute weights:

\[
\begin{array}{c|cc}
\text{pole edge}&h_1'&h_2'\\ \hline
1&h_{r,s}+rs&h_{r,s}+a\\
2&h_{r,s}-a&h_{r,s}+rs.
\end{array}
\]

Equivalently, at a pole on edge \(e\), translate both old internal weights by
\(h_{r,s}-h_e\), and then raise the singular edge by \(rs\).  This is the
rule implemented in
`Code/bosonic_c1_one_to_n_reference/reference_implementation/plumbing/virasoro_blocks.py`.

At fixed bidegree the recursion is triangular.  The project implementation
has already been compared with independent descendant sewing through
rectangular level \((3,3)\).

### 4.1 NS extension already present in the project

For an NS torus two-point block, \(rs\) is replaced by the physical null
level \(rs/2\), only \(r+s\) even contributes, and an odd null vector toggles
the parity routing.  The regular entry is the NS character

\[
 \chi_{\rm NS}^{(\varepsilon)}(Q)
 =\prod_{m=1}^{\infty}
 \frac{1+\varepsilon Q^{m-1/2}}{1-Q^m}.
\]

The generic and collision-aware implementations are in
`Code/h_recursion/superconformal_torus_two_point.py`, with an independent PBW
check in `Code/ns_torus_two_point_h_recursion_check/`.  The geometric program
below should first be tested in the bosonic Virasoro setting and then lifted
to this finite routing-vector recursion.

## 5. Exact Fourier--Jacobi rewriting of the CCY variables

For the Kummer lift, put the two mobile torus points at \(u\) and \(-u\),
where \(u\) has periods \(1,\tau\).  After translating one insertion to the
origin, their flat-coordinate separation is \(z=4\pi u\).  Hence

\[
 \boxed{
 p_1=e^{4\pi i u}=\zeta^2,
 \qquad
 p_2=e^{2\pi i(\tau-2u)}=Q\zeta^{-2},
 \qquad p_1p_2=Q.}
\]

Therefore each necklace monomial becomes

\[
 p_1^{n_1}p_2^{n_2}
 =Q^{n_2}\zeta^{2(n_1-n_2)},
\]

and the primary propagation factor becomes

\[
 p_1^{h_1-c/24}p_2^{h_2-c/24}
 =Q^{h_2-c/24}\zeta^{2(h_1-h_2)}.
\]

Thus the ordinary CCY block has the exact formal Fourier--Jacobi form

\[
\begin{aligned}
 \mathcal B_{\rm neck}
 ={}&Q^{h_2-c/24}\zeta^{2(h_1-h_2)}\chi(Q)\\
 &\times
 \sum_{n_1,n_2\geq0}
 f_{n_1,n_2}
 Q^{n_2}\zeta^{2(n_1-n_2)}.
\end{aligned}
\]

This formula exposes a truncation issue.  At fixed power \(Q^{n_2}\), the
sum over \(n_1\) is infinite.  A rectangular truncation in \((p_1,p_2)\) is
therefore not a controlled cusp expansion at fixed \((\tau,u)\) unless the
entire \(p_1\)-dependence has first been resummed.

## 6. Cusp factorization: the first coefficient is a sphere four-point block

Set \(p_2\to0\) while holding \(p_1\) fixed.  Only the primary state on edge
2 propagates.  Directly from the necklace descendant definition,

\[
 \Phi_0(p_1)
 :=\sum_{n\geq0}F_{n,0}p_1^n
\]

is the ordinary sphere four-point Virasoro block with internal weight \(h_1\)
and external weights

\[
 (h_2,d_1,d_2,h_2)
\]

in the slot ordering inherited from the necklace.  The level-one coefficient
already displays this identification:

\[
 F_{1,0}
 =\frac{(h_1+d_1-h_2)(h_1+d_2-h_2)}{2h_1}.
\]

Similarly,

\[
 F_{0,1}
 =\frac{(h_2+d_1-h_1)(h_2+d_2-h_1)}{2h_2}.
\]

Consequently the first justified elliptic improvement of the torus two-point
block is

\[
 p_1=\lambda(\widehat q_u),
 \qquad
 \widehat q_u=
 \exp\left[-\pi\frac{K(1-p_1)}{K(p_1)}\right],
\]

together with the *full sphere four-point elliptic prefactor* appropriate to
\((h_2,d_1,d_2,h_2;h_1)\).  A bare composition
\(p_1=\lambda(\widehat q_u)\) is exact as a formal change of variable, but it
does not by itself obtain Zamolodchikov's coefficient control.

At higher powers of \(Q\), the coefficients are sphere four-point blocks with
descendant data on the two \(h_2\) endpoint legs.  This suggests either:

1. a matrix-valued sphere elliptic recursion at every torus level; or
2. a direct two-variable recursion whose regular seed is the exact elliptic
   four-point block \(\Phi_0\), rather than the constant one.

This cusp factorization is the principal structural test for the proposed
program.

### 6.1 Completed low-level certificate

The identity has now been checked at two generic irrational parameter sets on
both cusp faces through level six.  The largest absolute coefficient residual
was

\[
 1.29\times10^{-16}.
\]

The two sides were computed independently:

- the torus faces used direct Virasoro descendant contractions;
- the sphere blocks used the CCY sphere four-point \(c\)-recursion.

The executable certificate is
`Code/sphere_five_kummer_h_recursion/check_ccy_cusp_factorization.py`.

## 7. Why the ordinary CCY torus two-point block is not yet the generic answer

The Kummer equivalence is first an equivalence of moduli descriptions, not an
automatic equality of conformal blocks.

Under the double cover:

- each of the four chosen branch-point operators remains at a ramification
  point, equivalently at a pillow corner;
- the fifth sphere operator lifts to a pair at \(u,-u\);
- the stress tensor transforms with a Schwarzian term,

\[
 T_u(u)=X'(u)^2T_x(X(u))+\frac{c}{12}\{X(u),u\}.
\]

An ordinary torus two-point block contains only two external weights
\(d_1,d_2\) and two internal weights \(h_1,h_2\).  A generic sphere
five-point block contains five external weights and two internal weights.
The three missing continuous external parameters cannot be supplied by a
coordinate Jacobian alone.  They reside in the four corner states or twisted
boundary data.

Equivalently, the smooth cover contains the four ramification-point
insertions plus the two lifts of the mobile insertion.  Calling the result an
ordinary torus two-point correlator silently discards the corner data.

This leads to the following distinction.

### 7.1 Special-locus program

There may exist a Poghossian-type locus of sphere weights for which the four
corner insertions combine with the covering anomaly into fixed data, leaving
an ordinary CCY torus two-point block with transformed parameters.  On such a
locus one should seek

\[
 \mathcal F_5^{\rm sphere}
 =\mathcal P_{\rm cov}(\tau,u)
 \mathcal B_{\rm CCY}^{(2)}(\tau,u)
\]

and prove it coefficientwise from the two recursions.

### 7.2 Generic-weight program

For arbitrary external weights, define a corner-decorated pillow block
\(\mathcal B_{\rm dec}^{(2)}\) and seek

\[
 \mathcal F_5^{\rm sphere}
 =\mathcal P_{\rm cov}(\tau,u;\boldsymbol d)
 \mathcal B_{\rm dec}^{(2)}
 (\tau,u;\boldsymbol d,h_1,h_2,c).
\]

The decorated object should retain a two-edge Kac-pole structure, but its
large-weight regular part and residue incidence factors can depend on all four
corner weights.  The CCY proof supplies the model for this derivation, not the
final formula.

## 8. Candidate elliptic normalization

A generic covering prefactor should be assembled from three types of factors:

\[
 \mathcal P_{\rm cov}
 =\mathcal J_{\rm branch}
  \mathcal P_{\rm corner}
  \mathcal P_{\rm mobile}.
\]

Here:

1. \(\mathcal J_{\rm branch}\) contains the Jacobians of the branched map and
   the Schwarzian/Weyl anomaly;
2. \(\mathcal P_{\rm corner}\) is a product of theta constants encoding the
   four fixed-point weights;
3. \(\mathcal P_{\rm mobile}\) contains prime forms

\[
 \prod_{a=1}^{4}
 E(u-\omega_a\mid\tau)^{\gamma_a}
\]

with exponents fixed by the four mobile-corner OPE limits.

The reduced block should then have the schematic form

\[
 \mathcal H_5(q_{\rm pil},u)
 =\mathcal P_{\rm cov}^{-1}\mathcal F_5.
\]

The desired analogue of the sphere four-point boundary condition is a simple
correlated large-weight limit of \(\mathcal H_5\), ideally one or an explicitly
known Fourier--Jacobi seed.  This must be derived; it should not be inferred
from the ordinary CCY character without keeping the corner data.

## 9. Falsifiable tests

The following tests are ordered so that an incorrect identification fails
before any expensive worldsheet integration.

### Test A: Kummer coordinate round trip

For generic complex \((x_1,x_2)\) away from all divisors:

1. construct \(\tau\) from the branch quartet;
2. compute \(u\) by the normalized Abel integral;
3. reconstruct the mobile point using \(X(u\mid\tau)\);
4. check all five labeled cross-ratios after the inverse Möbius map.

Target: relative error below \(10^{-30}\) at high precision.

### Test B: divisor dictionary

Verify numerically and symbolically that the ten boundary divisors of
\(\overline{\mathcal M}_{0,5}\) map to:

- the torus cusp or its modular images;
- \(u\) approaching a half-period;
- the corresponding transformed divisors in the other branch-quartet charts.

This test determines how many Kummer charts are required.

### Test C: CCY cusp factorization

Using direct Virasoro descendants, verify

\[
 [p_2^0]\mathcal B_{\rm neck}
 =\mathcal F_4^{\rm sphere}(p_1)
\]

through at least level six in \(p_1\), including the exact sphere elliptic
prefactor.  Repeat with \(1\leftrightarrow2\).

The coefficient identity through level six is complete; the explicit
elliptic-prefactor comparison is the remaining part of this test.

### Test D: special-locus identity search

Choose a low-order ansatz mapping the five sphere external weights to the two
ordinary torus external weights, the transformed central charge, and the
covering exponents.  Compare:

\[
 \mathcal P_{\rm cov}^{-1}\mathcal F_5^{\rm PBW}
 \quad\text{and}\quad
 \mathcal B_{\rm CCY}^{(2)}
\]

through rectangular level \((2,2)\).  If the coefficient equations force
three independent constraints among the sphere weights, that identifies the
candidate special locus.  If they are inconsistent, the ordinary-block route
is ruled out.

### Test E: generic decorated residues

Re-expand the direct sphere five-point block in \((\tau,u)\), strip the
candidate prime-form prefactor, and inspect its poles in \(h_1,h_2\).  Test
whether the residues equal the CCY null factors times a shifted decorated
block.  This determines exactly how the corner weights modify the two-edge
recursion.

### Test F: convergence comparison

At a grid of points in one fundamental domain compare equal-cost truncations
of:

1. the sphere linear \((x_1,x_2)\) series;
2. the raw necklace \((p_1,p_2)\) series;
3. the present independent \(\lambda^{-1}(p_i)\) re-expansion;
4. the cusp-resummed Fourier--Jacobi expansion;
5. the local OPE channel.

Use a higher-order direct PBW or \(c\)-recursive value as the reference.  An
elliptic proposal is accepted only if its error is controlled by the proposed
small parameters across the tested chart, rather than merely improved at a
few points.

### Test G: Jacobi covariance

After including the prefactor, check the transformations

\[
 u\mapsto u+1,\qquad
 u\mapsto u+\tau,\qquad
 u\mapsto-u,\qquad
 (\tau,u)\mapsto
 \left(-\frac1\tau,\frac{u}{\tau}\right).
\]

These transformations should act by the expected phases and fusion/modular
kernels, not change the numerical correlator.

## 10. Immediate implementation sequence

The safest next implementation is deliberately narrower than a production
five-point replacement.

1. Add a standalone Kummer-coordinate module with arbitrary precision:

   `Code/sphere_five_kummer_h_recursion/kummer_coordinates.py`

2. The direct CCY cusp-factorization check using the existing Virasoro PBW
   and torus-necklace code is now present:

   `Code/sphere_five_kummer_h_recursion/check_ccy_cusp_factorization.py`

3. Reuse the established sphere four-point elliptic recursion to resum
   \([p_2^0]\mathcal B_{\rm neck}\), and compare its convergence with the raw
   \(p_1\)-series.

4. Only after Test C passes, solve the level-\((1,1)\) and \((2,1)\)
   coefficient equations for a possible ordinary-torus special locus.

5. If no generic mapping exists, construct the first corner-decorated residue
   directly from the sphere five-point PBW block rather than adding more
   heuristic per-edge lambda substitutions.

6. Once the bosonic structure is fixed, repeat with the NS routing vector and
   the project physical five-tachyon component sum.

No production worldsheet code should be changed before Tests A--D decide
which object is actually being computed.

## 11. Existing project assets

- `Machine Notes/h-Recursion/h_recursion_review_private.tex`:
  fixed-difference large-weight proof and sphere/necklace recursions.
- `Code/bosonic_c1_one_to_n_reference/reference_implementation/plumbing/virasoro_blocks.py`:
  ordinary Virasoro CCY necklace \(h\)-recursion.
- `Code/bosonic_c1_one_to_n_reference/reference_implementation/plumbing/torus_two_point_blocks.py`:
  direct necklace and OPE descendant blocks, current formal per-edge lambda
  composition, and flat-coordinate conventions.
- `Code/h_recursion/superconformal_torus_two_point.py`:
  NS two-edge routing recursion.
- `Code/ns_torus_two_point_h_recursion_check/`:
  independent NS PBW/recursion certificate.
- `Machine Notes/c-Recursion/TYPE0B_NS_SPHERE_FIVE_TACHYON_1TO4_2026-08-25.md`:
  physical five-point sphere conventions and component routing.

## 12. References

1. M. Cho, S. Collier, and X. Yin,
   *Recursive Representations of Arbitrary Virasoro Conformal Blocks*,
   arXiv:1703.09805, especially Sections 3.1--3.5.
2. L. Hadasz, Z. Jaskolski, and P. Suchanek,
   *Recursive representation of the torus 1-point conformal block*,
   arXiv:0911.2353.
3. R. Poghossian,
   *Recursion relations in CFT and N=2 SYM theory*,
   arXiv:0909.3412.
4. The modular interpretation used here is the standard identification of
   four ordered points with an elliptic curve with level-2 structure, and of
   the fifth point with the Kummer universal family over that modular curve.

## 13. Bottom line

CCY already solve the ordinary two-edge Kac-pole reconstruction problem.  The
new work is not to rediscover that recursion.  It is to determine whether the
Kummer lift of a generic sphere five-point block lands on:

1. a special ordinary CCY torus two-point block after a covering prefactor; or
2. a genuinely corner-decorated extension of the CCY block.

The exact Fourier--Jacobi rewriting and the sphere-four-point cusp
factorization make this question testable at low level before any speculative
all-order formula is adopted.
