# Central-charge recursion with independently regulated vertices

This derivation addresses the auxiliary Ising quotient investigation.
It does not modify the production algorithm, and no numerical checks
were performed. The user's subsequent authorization of a direct
free-fermion benchmark supersedes implementation work on this route.

## Result and qualification

A genuine central-charge recursion closes on **decorated sewing
blocks**: the propagators are ordinary Verma inverse metrics, while
each incident vertex carries a finite Virasoro enveloping-algebra
polynomial. It applies to any finite sewing graph, hence arbitrary
genus and channel. At any prescribed degree the recursion is finite.

It does not close on the existing family of ordinary scalar CCY blocks
labelled only by shifted internal weights. The additional polynomial
decorations are essential. Their number and the Ward work needed to
evaluate their seeds can grow substantially; this is not an efficiency
claim or a completed level-ten implementation.

This construction computes independently regulated sewing exactly.
Whether a particular limiting prescription equals the irreducible
minimal-model block is a **separate** question, especially at nested
null intersections. That final quotient assertion is not proved here.

## 1. Operator-valued inverse-metric recursion

Use the PBW basis \(L_{-I}|H\rangle\) at level \(k\). For generic \(H\),
the inverse Gram matrix is meromorphic in its central charge \(c_g\).
At \(c_g=c_{rs}(H)\), choose the singular vector

\[
 \chi_{rs}=S_{rs}(c_{rs}(H),H)|H\rangle,
 \qquad [L_{-1}^{rs}]S_{rs}=1.
\]

Let \(E_{rs,k}\) be the finite PBW embedding matrix whose columns are
the states \(L_{-I}\chi_{rs}\), \(|I|=k-rs\). The universal residue
factorization is

\[
 \underset{c_g=c_{rs}(H)}{\operatorname{Res}}G_k(c_g,H)^{-1}
 =R_{rs}(H)E_{rs,k}
   G_{k-rs}(c_{rs}(H),H+rs)^{-1}E_{rs,k}^{T},
 \qquad
 R_{rs}(H)=-\frac{\partial c_{rs}(H)}{\partial H}A_{rs}(c_{rs}(H)).
\]

This is the standard null-vector residue identity before contracting
the two vertices into fusion polynomials. At fixed \(H\), the
large-\(c_g\) inverse metric has only its \(L_{-1}^{k}\) entry:

\[
 \lim_{c_g\to\infty}G_k(c_g,H)^{-1}
 =\frac{e_{1^k}e_{1^k}^{T}}{k!(2H)_k}.
\]

Indeed, the global descendant has a finite norm and finite pairings
with every other PBW state. The orthogonalized non-global oscillator
directions have divergent norms. Thus the polynomial part is precisely
this rank-one projector, and

\[
 G_k(c_g,H)^{-1}
 =\frac{e_{1^k}e_{1^k}^{T}}{k!(2H)_k}
 +\sum_{rs\leq k}
 \frac{R_{rs}(H)}{c_g-c_{rs}(H)}
 E_{rs,k}G_{k-rs}(c_{rs}(H),H+rs)^{-1}E_{rs,k}^{T}.
\]

The sum uses the usual nonredundant CCY list of finite central-charge
poles. Exact collisions are treated by meromorphic continuation from
distinct generic internal weights, rather than by dropping terms.
The \(H=0\) null at level one exists at every central charge and must
be handled by a separate weight regulator or a vacuum-module quotient;
it is not a finite \(c_g\) pole.

Finite primitive singular vectors can be constructed from their
positive-mode Ward equations. This is permitted precomputation; the
formula does not require directly inverting the finite-\(c_g\) Gram
matrix. Storing all matrix entries would nevertheless retain PBW-sized
costs, so the operator formula is principally a rigorous fallback.

## 2. Decorated scalar blocks close without Gram matrices

Keep the vertex central charge \(c_v\) independent of \(c_g\).
On an edge let \(h_e\) denote the original vertex-module weight,
\(H_e\) the propagator-module weight, and \(S_e\) a homogeneous
negative-mode polynomial. At its endpoint a propagator basis vector
labelled by \(I\) is evaluated by the vertex as

\[
 L_{-I}S_e|h_e\rangle.
\]

Initially \(S_e=1\). All vertex tensors use the fixed \(c_v,h_e\),
while their pairings use \(G(c_g,H_e)^{-1}\). A graph coefficient with
remaining propagator levels \(\mathbf k\) is denoted temporarily by
\(D_{\mathbf k}(c_g;c_v,\mathbf H,\mathbf S)\). This is a definition
by sewing, not an assumption that the decorated state is primary.

At a pole on edge \(e\), the inverse-metric identity replaces its
two endpoint states by

\[
 L_{-I}S_{rs}(c_{rs}(H_e),H_e)S_e|h_e\rangle.
\]

Consequently the residue is another block in the same family, with

\[
 k_e\longmapsto k_e-rs,\qquad
 H_e\longmapsto H_e+rs,\qquad
 S_e\longmapsto S_{rs}(c_{rs}(H_e),H_e)S_e.
\]

The last replacement is made at both ends of the edge. All other
decorations remain fixed, and all propagators in the recursive block
are evaluated at \(c_{rs}(H_e)\). For generic, distinct edge poles:

\[
\begin{aligned}
 D_{\mathbf k}(c_g;c_v,\mathbf H,\mathbf S)
 ={}&D_{\mathbf k}^{\infty}(c_v,\mathbf H,\mathbf S)\\
 &+\sum_e\sum_{rs\leq k_e}
 \frac{R_{rs}(H_e)}{c_g-c_{rs}(H_e)}
 D_{\mathbf k-rs\mathbf e}
 \left(c_{rs}(H_e);c_v,
       \mathbf H+rs\mathbf e,\mathbf S\big|_{S_e\mapsto S_{rs}S_e}\right).
\end{aligned}
\]

No fusion polynomial has been pulled out. In particular, a zero of an
ordinary primary fusion polynomial does not erase this decorated
residue when \(c_v\neq c_g\).

Each call lowers at least one remaining level. On the target graph,
the initial indices obey
\(2k_1+k_L+k_R+2k_3\leq20\). Along any recursive path the decoration
degree added on an edge plus its remaining level stays equal to that
edge's initial level. Thus all singular polynomials and all vertex
states required through physical total degree ten lie in a finite,
explicitly bounded set. Interleavings of residues on different edges
commute at the level of decorations. Different ordered chains on the
same edge must in general remain distinct.

## 3. The large-propagator-charge seed

Here \(c_v\) is fixed while \(c_g\to\infty\). Every vertex tensor is
therefore bounded in \(c_g\). The seed is the product of the decorated
vertex values with only global descendants, divided by

\[
 \prod_e k_e!(2H_e)_{k_e}.
\]

For the punctured theta graph its numerator is

\[
\begin{aligned}
 &\rho_{c_v}(L_{-1}^{k_1}S_1|h_1\rangle,
              L_{-1}^{k_L}S_L|h_L\rangle,
              L_{-1}^{k_3}S_3|h_3\rangle)\\
 &\quad\times\rho_{c_v}(L_{-1}^{k_1}S_1|h_1\rangle,
              L_{-1}^{k_R}S_R|h_R\rangle,
              L_{-1}^{k_3}S_3|h_3\rangle)\\
 &\quad\times\rho_{c_v}(L_{-1}^{k_R}S_R|h_R\rangle,
              |h_\psi\rangle,L_{-1}^{k_L}S_L|h_L\rangle).
\end{aligned}
\]

These are finite three-point Ward evaluations, with no summation over
general propagating descendants in the seed. If the initial vertex
and propagator weights agree, every \(S_e|h_e\rangle\) remains
quasiprimary of weight \(H_e\): the equation \(L_1S_{rs}|H\rangle=0\)
is independent of central charge, and composition preserves it.
Then each seed vertex equals its decorated ground coupling times the
closed ordinary global three-point coefficient. If separate weight
regulators are used, this simplification need not hold, but the finite
Ward expression above remains valid.

There is **no Gaussian oscillator vacuum factor** in this asymptotic
problem. The usual CCY vacuum factor arises when vertex and propagator
charges grow together. One may evaluate the resulting two-variable
identity at \(c_v=c_g\) afterward, but must not use that diagonal
large-charge seed during the independent-variable recursion.

## 4. Why ordinary shifted blocks do not suffice

Already at level two, write

\[
 S_2=L_{-2}-\frac{3}{2(2H+1)}L_{-1}^{2},
 \qquad c_g=c_{21}(H).
\]

In the vertex representation at \(c_v\),

\[
 L_1S_2|H\rangle=0,\qquad
 L_2S_2|H\rangle=\frac{c_v-c_{21}(H)}2|H\rangle.
\]

The second identity is an inhomogeneous Ward source. It is absent for
an ordinary primary of weight \(H+2\). Parameter derivatives of
ordinary primary vertices do not automatically supply this source:
the highest-weight condition is homogeneous throughout those families.
The source is retained exactly by the polynomial decoration above.
After repeated residues it is still a finite enveloping-algebra
polynomial, establishing constructive closure rather than assuming a
scalar shifted-block factorization.

## 5. Remaining quotient issue

The Ares–Santachiara–Viti prescription, Eq. (39), evaluates vertices at
\(c_0+\varepsilon^2\) and propagators at \(c_0+\varepsilon\), with a
separate treatment of vacuum weights:
[primary source](https://arxiv.org/html/2107.13925#S3.SS4).
The recursion derived here can compute that independently regulated
quantity coefficient by coefficient. Its finite-limit identification
with the irreducible quotient must still be justified at every nested
null order reached.

The distinction matters algebraically. A general analytic metric can
have a twice-vanishing null eigenvalue and order-one mixing with its
non-null complement. For example,

\[
 G(t)=\begin{pmatrix}1&t\\t&2t^2\end{pmatrix},\qquad
 G(t)^{-1}=\begin{pmatrix}2&-t^{-1}\\-t^{-1}&t^{-2}\end{pmatrix}.
\]

The critical quotient has inverse metric \(1\), yet contraction of
\(G(t)^{-1}\) with the fixed null-annihilating vector \((1,0)\)
gives \(2\). This is not asserted to be a Virasoro counterexample;
it shows exactly why a general proof needs information about the
Jantzen filtration and its mixing, not just the vanishing of the
critical vertex on the null subspace. The decorated recursion solves
the meromorphic computation problem. It does not, by itself, replace
that missing quotient proof.

## 6. Implementation assessment

A direct implementation needs: primitive singular-polynomial
precomputation; canonical multiplication of negative-mode polynomials;
finite decorated three-point Ward values; the scalar residue factor
\(R_{rs}\); and memoization of the level, shifted weights and complete
edge decorations. The existing `PuncturedCCY` pole arithmetic and
global coefficient routine are reusable, but its fusion residues and
cache keys are insufficient. Alternatively, the operator-valued
recursion constructs inverse metrics without direct Gram inversion,
at the cost of PBW matrix sizes and the final tensor contraction.

Neither option has been implemented or timed here. Both should be
described as more expensive propagator recursions, not as the existing
fast scalar CCY block algorithm.
