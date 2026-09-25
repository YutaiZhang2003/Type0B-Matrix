# RR pair on the torus: local picture-changing reduction

This derives the **free-field operator acting on a supplied, correctly
spin-projected Liouville correlator**. It does not provide that interacting
spin projection, its collision subtraction, or an integrated amplitude.
Conventions are periods `(1,tau)`, `alpha'=2`, and BRY's axion `A`.
The raised puncture is at zero and the other puncture is at z.

There is one PCO per chirality. With one bare `-1/2` vertex and one
raised vertex, the matter term has superghost charges `(+1/2,-1/2)`.
The additional eta terms in the raised vertex have phi charge `+3/2`,
while the `c partial xi` term retains `-1/2`. Their total phi charges
with the remaining bare vertex are respectively `+1` and `-1`.
They therefore fail the genus-one phi zero-mode neutrality condition.
For this two-point picture assignment they do not contribute to the
correlator. This charge argument does not remove collision boundary terms.

## Physical ground states and the graded pairing

Use family order `(sigma,mu)` for the time fermion and `(V_R+,V_R-)`
for Liouville. Write `u=exp(-i*pi/4)` and `ubar=exp(i*pi/4)`. The
Liouville zero modes in the physical family basis are

\[
L_0^{G}=\frac{iP\bar u}{\sqrt2}
 \begin{pmatrix}0&i\\1&0\end{pmatrix},\qquad
\widetilde L_0^{G}=\frac{iPu}{\sqrt2}
 \begin{pmatrix}0&-i\\1&0\end{pmatrix}.
\]

These follow by acting with the chiral and antichiral zero modes on the
two local embedded Ramond kets in the README. The timelike family phases
compatible with BRY's incoming/outgoing operators give

\[
T_0^{G}=-\frac{k\bar u}{\sqrt2}
 \begin{pmatrix}0&i\\1&0\end{pmatrix},\qquad
\widetilde T_0^{G}=\frac{ku}{\sqrt2}
 \begin{pmatrix}0&-i\\1&0\end{pmatrix}.
\]

In time-times-Liouville order, use
`G0_m=T0_G tensor 1 + (-1)^F_time tensor L0_G`, and likewise for the
antichiral operator. Both squares are `(P^2-k^2)/2`, and they anticommute.
The two BRY states

\[
\mathcal W_s=\frac{|\sigma,V_R^+\rangle
 +s|\mu,V_R^-\rangle}{\sqrt2},\qquad k=sP,
\]

are annihilated by both matter zero modes. This is checked exactly in
`check_picture_changed.py`, including the in/out choice `s=+1,-1`.

The graded product pairing is `diag(1,1,1,-1)` in this basis. It gives
`<W_+,W_->=1`, while each same-sign bilinear is zero. In particular,
factorizing the two `mu V_R-` terms into time and Liouville correlators
requires their Grassmann interchange sign. Omitting it would incorrectly
make the incoming/outgoing ground pairing vanish.

## Remove the timelike supercurrent from the PCO

For one Majorana spin field S,

\[
\psi_{-1}S=4L_{-1}^{\psi}\psi_0 S.
\]

One can see the coefficient directly from
`L_-1=epsilon*psi_-1*psi_0/2` and `psi_0^2=epsilon/2`, for either sign
`epsilon` of the fermion OPE. With `G_time=-i psi^0 partial X^0`,

\[
G_{-1}^{t}(e^{ikX^0}S)
=G_0^t\left[4e^{ikX^0}\partial S
 -\frac1{k^2}(\partial e^{ikX^0})S\right].
\]

The on-shell identity `G0_time W=-G0_L W` reduces the full matter
picture change to `G_-1^L + D_delta G0^L` after contracting the time
fields. Define

\[
\mathcal G_t=2\log|E(z)|-\frac{2\pi(\operatorname{Im}z)^2}{\tau_2},
\qquad B_t=e^{\omega^2\mathcal G_t},
\]
\[
S_\delta(z)=E(z)^{-1/8}
 \left[\frac{\theta_\delta(z/2)}{\eta}\right]^{1/2}.
\]

Constant spin-field phases do not affect the logarithmic derivatives.
Since `partial_0=-partial_z` on a two-point function,

\[
D_\delta=\frac1{\omega^2}\partial_0\log B_t
 -4\partial_0\log S_\delta
=-\frac32\frac{E'(z)}{E(z)}
 +\frac{\theta'_\delta(z/2)}{\theta_\delta(z/2)}
 -\frac{2\pi i\operatorname{Im}z}{\tau_2}.
\]

The prime on theta differentiates its full argument. For the separate
antiholomorphic time factor use `Dbar=conjugate(D)`; D contains no energy.
This does not conjugate the continued energy or a Liouville coefficient.

The local coordinate connection in D is
`-4 h_spin + h_time/k^2 = -1/4-1/2 = -3/4`. It cancels the
`+(3/4) f'' G0` term in the transformation of `G_-1 R`. Thus the complete
picture-changed matter state has the required primary transformation.

## Two chiralities and the finite Gaussian term

Let `C_pq`, with `p,q=0,1`, denote the Liouville correlator with
`(D_p Dbar_q R)(0)` and `R(z)`, where `D_1=G_-1` and `D_0=G_0`,
including the BRY external spin/disorder projection in the selected
global spin frame. The holomorphic operator is ordered first.
The reduced matter expression is

\[
\boxed{C_{11}+D_\delta C_{01}+\overline D_\delta C_{10}
+\left(D_\delta\overline D_\delta
 -\frac\pi{\omega^2\tau_2}\right)C_{00}.}
\]

The last contribution follows from

\[
\frac1{\omega^4}\partial_0\bar\partial_0\log B_t
=\frac1{\omega^2}\partial_z\bar\partial_z\mathcal G_t
=-\frac\pi{\omega^2\tau_2}
\]

away from the collision lattice. It is the finite torus Gaussian
contraction at the raised vertex. It is **not** a prescription for the
distributional collision terms or the moduli-space boundary subtraction.
The zero-energy limit must include `C_00`, whose two zero modes carry
energy factors; the isolated `1/omega^2` coefficient is not that limit.

## Free prefactor and transport checks

The raised superghost correlator times one time-Majorana spin correlator
gives, in the two chiralities, `|eta| |E(z)|^(1/4)/|theta_delta(z/2)|`.
Using the same stripped area measure as the earlier NSNS calculation,
the free prefactor is

\[
\mathcal B_\delta^{RR}
=\frac{\omega^2|\eta|^3|E(z)|^{1/4}
 e^{\omega^2\mathcal G_t}}
 {16\sqrt{8\pi^2\tau_2}|\theta_\delta(z/2)|}.
\]

It includes the external `omega^2`, PCO coefficient `1/4`, diagonal GSO
weight `1/2`, and the previous `bc`/automorphism factor `1/2`.
The Liouville momentum measure is still `dP_NS dP_R/pi^2`.
An overall conversion of the completed density into the BRY S-matrix
must be carried through with the same normalization audit as for NSNS.

The code checks the free factors and D under the puncture transports

| Transport | Theta permutation |
| --- | --- |
| `z -> z+1` | `(1 2)(3 4)` |
| `z -> z+tau` | `(1 4)(2 3)` |
| Modular S | `(2 4)` |
| Modular T | `(3 4)` |

Under `tau'=(a tau+b)/(c tau+d)`, `z'=z/(c tau+d)`, D has weight one.
The free prefactor has absolute weight `7/4-2 omega^2`; the Liouville
combination has weight `17/4+2 omega^2`. Their sum is six, canceled by
`d^2tau d^2z`. The numerical checks include a genuinely complex energy
and preserve its analytic square. All saved transport residuals are
below `9e-61` at 60-digit precision. This certifies these free factors,
not the interacting Liouville spin sum. The subsequent
[local density layer](LOCAL_DENSITY.md) supplies that contraction in the
contractible OPE patch and checks it separately.
