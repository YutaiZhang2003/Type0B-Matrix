# The middle Ramond branching coefficient from the physical (L_1) identity

This file derives the recurrence implemented in `middle_branching.py`. It uses
exactly the Ramond primary normalization of `Human Notes/SCblock.tex`. The
repeated parity superscript in the two displayed definitions of the Ramond
primaries in those notes is a typographical error: the string without the
additional rightmost zero mode has parity (M+1), whereas the string with it
has parity (M), where (M=2n-\tfrac12\).

The insertion is radially ordered as

\[
\langle v_{n'}^\alpha|\Theta V[v_{1/2}(Q/2)](1)|v_n^\alpha\rangle,
\qquad V[v_{1/2}(Q/2)](1)=Q\psi(1).
\]

The bra is at infinity, the external fermion is at one, and the ket is at
zero. Neither the phases for permuting punctures nor square roots of primary
norms enter the production calculation. The normalized branching coefficient
can be recovered afterwards.

## Why the physical Ward identity is particularly simple

The physical SCA stress tensor acts only on the physical tensor factor. The
external field acts only on the auxiliary factor. In addition, the odd map
(Theta) commutes with the even physical generators (L_m). Therefore

\[
[L_m,\Theta Q\psi(1)]=0.
\]

The physical BPZ pairing has (L_m^\dagger=L_{-m}), and is bilinear rather than
sesquilinear. In particular the two identities needed below are

\[
\begin{aligned}
\langle L_1v_{n'}^\alpha|\Theta Q\psi(1)|v_n^\alpha\rangle
 &=\langle v_{n'}^\alpha|\Theta Q\psi(1)|L_{-1}v_n^\alpha\rangle,\\
\langle L_{-1}v_{n'}^\alpha|\Theta Q\psi(1)|v_n^\alpha\rangle
 &=\langle v_{n'}^\alpha|\Theta Q\psi(1)|L_1v_n^\alpha\rangle.
\end{aligned}
\tag{1}
\]

These are identities for the physical (L_{\pm1}), not either one of the
embedded Virasoro generators. In each embedded Virasoro algebra the field has
nonzero weight,

\[
h_\psi^{(1)}=-\frac{1+2b^2}{2(1-b^2)},\qquad
h_\psi^{(2)}=\frac{b^2+2}{2(1-b^2)}.
\]

The two level-two null relations of this field require
(n'=n\pm\tfrac12). The map (Theta) commutes with both embedded Virasoro
algebras, so it does not change that fusion rule. It reverses the parity once
more, leaving equal parity labels on the two endpoints.

## Reduction when the outgoing endpoint has larger absolute label

First suppose (n'=n+\tfrac12>0\), (n\geq\tfrac14\). Insert the existing
Ramond action expansions

\[
\begin{aligned}
L_1v_{n'}^\alpha
 &=\sum_{|A|+|B|=4n'-3}\mathbb V_{1,n',\alpha}^{AB}
 L_{-A}^{(1)}L_{-B}^{(2)}v_{n'-1}^\alpha,\\
L_{-1}v_n^\alpha
 &=\mathbb V_{n,\alpha}^{(1)}L_{-1}^{(1)}v_n^\alpha
  +\mathbb V_{n,\alpha}^{(2)}L_{-1}^{(2)}v_n^\alpha
  +\text{descendants in the other branch}.
\end{aligned}
\]

For (n\geq\tfrac34\), the other branch is (n-1). At (n=\tfrac14\), it is
(-\tfrac34\). Its difference from (n'=n+\tfrac12\) is (\tfrac32\), so its
three-point function with the inserted field vanishes by the fusion rule.
No numerical cancellation or truncation is being assumed in dropping it.

For a unit-normalized ordinary Virasoro three-point form in the puncture order
((\infty,1,0)),

\[
\rho(\nu_{h'},\nu_{h_\psi},L_{-1}\nu_h)=h+h_\psi-h',
\qquad
\rho(L_{-1}\nu_{h'},\nu_{h_\psi},\nu_h)=h'+h_\psi-h.
\]

Consequently, the first identity in (1) gives the scalar recursion

\[
\begin{aligned}
&\left[\sum_{i=1}^2\mathbb V_{n,\alpha}^{(i)}
 \left(h_{n}^{(i)}+h_\psi^{(i)}-h_{n'}^{(i)}\right)\right]
 \langle v_{n'}^\alpha|\Theta Q\psi(1)|v_n^\alpha\rangle\\
&\quad=\sum_{|A|+|B|=4n'-3}\mathbb V_{1,n',\alpha}^{AB}
 \rho(L_{-A}\nu_{h_{n'-1}^{(1)}},\nu_{h_\psi^{(1)}},\nu_{h_n^{(1)}})
 \rho(L_{-B}\nu_{h_{n'-1}^{(2)}},\nu_{h_\psi^{(2)}},\nu_{h_n^{(2)}})
 \langle v_{n'-1}^\alpha|\Theta Q\psi(1)|v_n^\alpha\rangle.
\end{aligned}
\tag{2}
\]

The two ordinary forms in this equation contain only a finite set of
descendants belonging to the reusable (L_1) action. Their evaluation uses
the ordinary three-point Ward identity; it is not a conformal-block
calculation and does not replace the later CCY recursion for the block.

## Reduction when the incoming endpoint has larger absolute label

Suppose instead (n=n'+\tfrac12>0\), (n'\geq\tfrac14\). The second identity
in (1), with precisely the same fusion argument, gives

\[
\begin{aligned}
&\left[\sum_{i=1}^2\mathbb V_{n',\alpha}^{(i)}
 \left(h_{n'}^{(i)}+h_\psi^{(i)}-h_n^{(i)}\right)\right]
 \langle v_{n'}^\alpha|\Theta Q\psi(1)|v_n^\alpha\rangle\\
&\quad=\sum_{|A|+|B|=4n-3}\mathbb V_{1,n,\alpha}^{AB}
 \rho(\nu_{h_{n'}^{(1)}},\nu_{h_\psi^{(1)}},L_{-A}\nu_{h_{n-1}^{(1)}})
 \rho(\nu_{h_{n'}^{(2)}},\nu_{h_\psi^{(2)}},L_{-B}\nu_{h_{n-1}^{(2)}})
 \langle v_{n'}^\alpha|\Theta Q\psi(1)|v_{n-1}^\alpha\rangle.
\end{aligned}
\tag{3}
\]

For negative labels, apply the existing reflection rule (n\mapsto-n),
(P\mapsto-P) to the action coefficients. The physical (w^-) component
changes sign on both sides of the action identity, so no extra coefficient
is introduced. The code always applies (L_1) to the endpoint with larger
absolute label. Equations (2) and (3) therefore reduce that absolute label
by one, including the crossing (\tfrac34\to-\tfrac14\). Every allowed pair
terminates at the two oriented ground pairs
((n',n)=(\tfrac14,-\tfrac14)) and
((n',n)=(-\tfrac14,\tfrac14)).

At exceptional momenta the scalar multiplying the unknown in (2) or (3)
can vanish. The algorithm implements the generic-momentum recurrence and
raises an explicit error at a zero pivot. Such a point requires analytic
continuation of the combined expression; silently dividing by a small
number or substituting a direct PBW primary coefficient would not establish
the requested recursion.

## The ground anchors and normalization

The zero mode obeys

\[
Q\Theta\psi_0\bigl(\boldsymbol\Psi_{-\mathsf B}u^{\mathsf b}\bigr)
=\frac{Q}{\sqrt2}(-1)^{\mathsf b}
\boldsymbol\Psi_{-\mathsf B}u^{\mathsf b}.
\]

Evaluating this diagonal action on the two explicitly defined ground
primaries, with their BPZ norms (\|v_{\pm1/4}^0\|^2=2\) and
(\|v_{\pm1/4}^1\|^2=-1\), gives

\[
\begin{aligned}
\langle v_{1/4}^0|\Theta Q\psi(1)|v_{-1/4}^0\rangle
&=\langle v_{-1/4}^0|\Theta Q\psi(1)|v_{1/4}^0\rangle=\sqrt2Q,\\
\langle v_{1/4}^1|\Theta Q\psi(1)|v_{-1/4}^1\rangle
&=\langle v_{-1/4}^1|\Theta Q\psi(1)|v_{1/4}^1\rangle=\frac{Q}{\sqrt2}.
\end{aligned}
\tag{4}
\]

There are no further input primary matrix elements.

For (M=2|n|-\tfrac12\), the raw primary definitions give

\[
\Theta v_n^0=-2^{(-1)^M/2}v_n^1,
\qquad
\Theta v_n^1=2^{-(-1)^M/2}v_n^0.
\tag{5}
\]

The anticommutation of (Theta) and (V[v_{1/2}]) then determines the
parity-changing matrix element without an arbitrary square-root phase:

\[
\begin{aligned}
\langle v_{n'}^0|V[v_{1/2}](1)|v_n^1\rangle
 &=2^{-(-1)^M/2}
 \langle v_{n'}^0|\Theta V[v_{1/2}](1)|v_n^0\rangle,\\
\langle v_{n'}^1|V[v_{1/2}](1)|v_n^0\rangle
 &=-2^{(-1)^M/2}
 \langle v_{n'}^1|\Theta V[v_{1/2}](1)|v_n^1\rangle.
\end{aligned}
\tag{6}
\]

Finally, divide this result by
(\|v_{n'}^\alpha\|\|v_{1/2}\|\|v_n^{1-\alpha}\|\) to obtain the radially
ordered normalized branching coefficient
(mathbb B(n',\tfrac12,n;\alpha,1-\alpha)). The external norm is
(\|v_{1/2}(Q/2)\|^2=-Q^2\). Equation (6) and the raw contraction used in the
implementation avoid treating a choice of square roots as an extra physical
phase.

## What is precomputed, and what is recursive

`RamondActions.plus` computes the coefficients
(\mathbb V_{1,n,\alpha}^{AB}\) by expanding (L_1v_n^\alpha\) in the
finite double-Virasoro descendant span of (v_{n-1}^\alpha\).
`RamondActions.minus` uses the existing corresponding (L_{-1}) solver.
The inputs to these two solves are the sparse free-field states constructed
from the notes' chi strings. These reusable action coefficients are **not**
claimed to have an independent recursion. They depend only on (b,P,n,\alpha)
and can be cached between computations using the same Ramond module.

`MiddleBranching.raw` subsequently computes every requested primary matrix
element by (2)--(4). It never converts a high primary to a physical PBW basis
and never evaluates a sought high-primary insertion matrix element directly.
The ordinary descendant factors in (2) and (3) use the existing Virasoro
three-point Ward routine.

For the balanced split parameters
(\hat q_{2,1}=u\sqrt{q_2}\),
(\hat q_{2,2}=u^{-1}\sqrt{q_2}\), the two primary prefactors have total
(q_2\)-degree (n'^2+n^2-\tfrac18\). At physical total level ten every
contributing pair has \(|n|,|n'|\leq\tfrac94\). The largest required
(L_1\) descendant level is six, from (n=\tfrac94\); the largest
(L_{-1}\) solve needed on the smaller endpoint also involves other-branch
descendants only through level six, from (n=\tfrac74\). This finite
precomputation supplies all middle coefficients needed at level ten.

## Validation status

The implementation has been syntax checked. No standalone numerical oracle
comparison is claimed here. The user's authorized checks are the final
physical block against PBW through total level five and its independence of
the split parameter at fixed product; the integrated pipeline must establish
those checks before this component is described as numerically validated.
