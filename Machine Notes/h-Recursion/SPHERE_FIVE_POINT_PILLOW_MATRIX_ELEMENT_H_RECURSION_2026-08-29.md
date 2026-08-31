# Sphere five-point elliptic \(h\)-recursion as a pillow matrix element

Date: 2026-08-29

Status: working final formula. The Hilbert-space representation and Kac-pole
residues below are firm. The large-weight regular seed is adopted as an
explicit assumption, motivated by the \(\mathbb Z_2\)-twist-character
argument; its first middle-vertex check is recorded in Section 6.1.

## 1. Correction of the starting point

The useful analogy is not

\[
 \text{sphere five-point block}
 \stackrel{?}{=}
 \text{ordinary torus two-point block}.
\]

It is the direct extension of pillow quantization of the sphere four-point
block.

Choose four punctures as the branch points of the double cover. They define
the pillow

\[
 \mathcal P_\tau=T^2_\tau/\mathbb Z_2.
\]

The two lower corner operators prepare a state
\(|\psi_{12}\rangle\), and the two upper corner operators prepare the BPZ bra
\(\langle\psi_{34}|\). The string-note convention writes the latter as
\(\langle\psi_{43}|\), reflecting BPZ order. For a four-point block in the representation
\(\mathcal V_h\), the chiral pillow amplitude is

\[
 \widetilde{\mathcal F}_4(h;q)
 =
 \langle\psi_{34}|
 q^{L_0-c/24}\,\mathbb P_h
 |\psi_{12}\rangle,
 \qquad
 q=e^{i\pi\tau}.
\]

Here \(\mathbb P_h\) is the projector onto the Verma module
\(\mathcal V_h\), including its inverse Gram matrices. This equation explains
both the natural elliptic coordinate and the origin of the Kac poles.

The four-point relation, with operators ordered at
\((0,z,1,\infty)\), is

\[
\begin{aligned}
 \mathcal F_c(h_1,h_2,h_3,h_4,h;z)
 ={}&
 \theta_3(\tau)^{\,c/2-4(h_1+h_2+h_3+h_4)}
 z^{\,c/24-h_1-h_2}
 (1-z)^{\,c/24-h_2-h_3}\\
 &\times
 \langle\psi_{43}|
 q^{L_0-c/24}
 |\psi_{12}\rangle_h .
\end{aligned}
\tag{1.0}
\]

The current string-note PDF prints \(h_3+h_4\) in the exponent of
\(1-z\) in E.103. A direct PBW comparison shows that this is a labeling
error: the exponent pairs the operators at \(z\) and \(1\), hence it is
\(h_2+h_3\), as in E.106. A numerical PBW comparison through \(q^6\)
agrees coefficient by coefficient with the independent \(h\)-recursion in
this convention. Keeping the E.103 exponent instead produces the nonzero
first elliptic mismatch \(16(h_2-h_4)q\).

The subscript \(h\) means that both pillow states have been projected onto
\(\mathcal V_h\). Equivalently, one may leave the states unprojected and
insert \(\mathbb P_h\) explicitly. This is exactly the distinction in the
user's formulation.

For five points, the fifth operator is a genuine local insertion in the
interior of the pillow. In the comb channel

\[
 ((d_1d_2)\to h_1,\quad (h_1d_5)\to h_2,\quad
  (h_2d_3d_4)),
\]

the correct chiral object is therefore

\[
 \boxed{
 \mathcal M_5
 =
 \langle\psi_{43}|\,
 \mathbb P_{h_2}\,
 p_2^{L_0-c/24}\,
 V_{d_5}^{(w)}(0)\,
 p_1^{L_0-c/24}\,
 \mathbb P_{h_1}\,
 |\psi_{12}\rangle .}
 \tag{1.1}
\]

Projectors commute with propagation, so their placement next to the
propagators is conventional. The essential facts are:

1. there is one local operator between the two propagation segments;
2. there are two independently specified exchanged representations;
3. the four original branch-point operators remain encoded in the two
   pillow states.

Equation (1.1), rather than a torus trace, is the starting point for the
five-point elliptic \(h\)-recursion.

## 2. Pillow geometry and the two segment variables

Use the covering-torus coordinate

\[
 w\sim w+2\pi\sim w+2\pi\tau,\qquad w\sim-w,
\]

with branch corners at

\[
 0,\quad \pi,\quad \pi\tau,\quad \pi(1+\tau).
\]

Quantize with the \(A\)-cycle as the spatial circle. The pillow has half the
height of the covering torus, so propagation from the lower pair to the
upper pair has total nome

\[
 q=e^{i\pi\tau}.
\]

Let the fifth puncture have pillow coordinate \(w_5\), measured from the
lower seam. The raw cylinder propagation variables are

\[
 \widetilde p_1=e^{iw_5},\qquad
 \widetilde p_2=e^{i(\pi\tau-w_5)}.
\]

For the OPE-aligned local coordinates used in the recursion, rotate each
boundary state by half an \(A\)-cycle and define

\[
 p_1=-\widetilde p_1=-e^{iw_5},\qquad
 p_2=-\widetilde p_2=-e^{i(\pi\tau-w_5)},\qquad
 \boxed{p_1p_2=q.}
 \tag{2.1}
\]

This choice is fixed by the cusp expansions

\[
 \frac{z}{t}=4p_1+O(p^2),\qquad
 t=4p_2+O(p^2),
\]

and is therefore the convention in which the residue variables are
\(4p_1\) and \(4p_2\), rather than \(-4p_1\) and \(-4p_2\).

In a coordinate \(u=w/(2\pi)\) with periods \(1,\tau\),

\[
 \zeta=e^{2\pi i u},\qquad
 p_1=-\zeta,\qquad
 p_2=-q\,\zeta^{-1}.
 \tag{2.2}
\]

Half-period shifts and a reversal of the cylinder orientation can change
signs or exchange \(p_1,p_2\). Those are chart conventions. The invariant
statement is that the two segment lengths add to half the \(B\)-cycle, hence
their product is the pillow nome \(q\), not the full torus nome
\(Q=e^{2\pi i\tau}=q^2\).

This corrects the earlier ordinary-torus assignment
\(p_1=\zeta^2,\ p_2=Q\zeta^{-2}\). The squared assignment describes the
separation of the two lifts \(u,-u\) in a full torus two-point trace; it is
not the direct pillow-cylinder decomposition (1.1).

### 2.1 Exact five-point generalization of E.103

Fix the sphere positions to be

\[
 V_{d_1}(0),\qquad V_{d_2}(z),\qquad
 V_{d_3}(1),\qquad V_{d_4}(\infty),\qquad V_{d_5}(t),
\tag{2.3}
\]

where the first four points are the branch quartet and \(t\) is the mobile
puncture. The chiral comb block is defined by

\[
 \mathcal F_5^{S^2}(z,t)
 =
 \langle d_4|
 V_{d_3}(1)\mathbb P_{h_2}
 V_{d_5}(t)\mathbb P_{h_1}
 V_{d_2}(z)|d_1\rangle,
\tag{2.4}
\]

with the primary three-point constants stripped in the same convention as
the ordinary four-point block.

The string-note map is

\[
 w(x)
 =
 \frac{1}{\theta_3(\tau)^2}
 \int_0^x
 \frac{dy}{\sqrt{y(1-y)(z-y)}},
\qquad
 \tau=i\frac{K(1-z)}{K(z)}.
\tag{2.5}
\]

Therefore

\[
 w_5=w(t),\qquad
 \frac{dw_5}{dt}
 =
 \frac{1}{
 \theta_3(\tau)^2\sqrt{t(1-t)(z-t)}}.
\tag{2.6}
\]

Away from the branch points, \(V_{d_5}\) is an ordinary primary. Hence

\[
 V_{d_5}^{(x)}(t)
 =
 \left(\frac{dw_5}{dt}\right)^{d_5}
 V_{d_5}^{(w)}(w_5).
\tag{2.7}
\]

The Liouville/Weyl anomaly and the regularizations at the four corners are
unchanged by the extra insertion. Applying E.103 to the branch quartet and
then (2.7) gives the exact five-point relation

\[
\boxed{
\begin{aligned}
 \mathcal F_5^{S^2}(z,t)
 ={}&
 \theta_3(\tau)^{\,c/2-4(d_1+d_2+d_3+d_4)}
 z^{\,c/24-d_1-d_2}
 (1-z)^{\,c/24-d_2-d_3}\\
 &\times
 \left(\frac{dw_5}{dt}\right)^{d_5}
 \mathcal M_5(q,w_5).
\end{aligned}}
\tag{2.8}
\]

Here

\[
 \mathcal M_5(q,w_5)
 =
 \langle\psi_{43}|
 \mathbb P_{h_2}
 p_2^{L_0-c/24}
 V_{d_5}^{(w)}(0)
 p_1^{L_0-c/24}
 \mathbb P_{h_1}
 |\psi_{12}\rangle,
\tag{2.9}
\]

with

\[
 p_1=-e^{iw_5},\qquad
 p_2=-e^{i(\pi\tau-w_5)},\qquad p_1p_2=q.
\]

Before the two half-period rotations, the raw translated insertion obeys

\[
 \widetilde p_2^{L_0-c/24}V_{d_5}^{(w)}(0)
 \widetilde p_1^{L_0-c/24}
 =
 q^{L_0-c/24}V_{d_5}^{(w)}(w_5).
\tag{2.10}
\]

The associated half-period phases are absorbed into the definitions of the
aligned pillow states and insertion used in (2.9).

Equivalently, after inserting (2.6), define

\[
\boxed{
\begin{aligned}
 \Lambda_5^{(c)}(z,t)
 :={}&
 \theta_3(\tau)^{\,c/2-4(d_1+d_2+d_3+d_4)-2d_5}\\
 &\times
 z^{\,c/24-d_1-d_2}
 (1-z)^{\,c/24-d_2-d_3}
 [t(1-t)(z-t)]^{-d_5/2}.
\end{aligned}}
\tag{2.11}
\]

Then the compact E.103 generalization is simply

\[
 \boxed{\mathcal F_5^{S^2}(z,t)
 =\Lambda_5^{(c)}(z,t)\,\mathcal M_5(q,w_5).}
\tag{2.12}
\]

For the full nonchiral block, multiply (2.12) by its antiholomorphic
counterpart with
\((c,d_i,h_i,z,t,w_5)\to
(\bar c,\bar d_i,\bar h_i,\bar z,\bar t,\bar w_5)\).

The square-root and fractional-power branches in (2.5)--(2.11) must be
continued together. Changing them changes the conformal-block branch, not
the local formula.

For the project linear chart

\[
 (0,x_1x_2,x_2,1,\infty),
\]

take \(z=x_2\) and \(t=x_1x_2\). If the corresponding external weights are
\((D_1,D_2,D_3,D_4,D_5)\), the dictionary is

\[
 (d_1,d_2,d_3,d_4;d_5)
 =
 (D_1,D_3,D_4,D_5;D_2).
\tag{2.13}
\]

In that chart the external covering factor is therefore

\[
\begin{aligned}
 \Lambda_{5,\rm lin}^{(c)}
 ={}&
 \theta_3(\tau)^{\,c/2-4(D_1+D_3+D_4+D_5)-2D_2}\\
 &\times
 x_2^{\,c/24-D_1-D_3}
 (1-x_2)^{\,c/24-D_3-D_4}\\
 &\times
 [x_1x_2(1-x_1x_2)(x_2-x_1x_2)]^{-D_2/2},
\qquad
 \tau=i\frac{K(1-x_2)}{K(x_2)}.
\end{aligned}
\tag{2.14}
\]

## 3. Descendant expansion of the matrix element

Let \(A_i,B_i\) be partitions of a common level \(n_i\), and let
\(G_{h_i}^{(n_i)}\) be the Verma Gram matrix. Suppressing the covering
prefactor and primary propagation powers, (1.1) expands as

\[
\begin{aligned}
 \mathcal G_5(p_1,p_2)
 ={}&
 \sum_{n_1,n_2\geq0}
 p_1^{n_1}p_2^{n_2}
 \sum_{\substack{|A_i|=|B_i|=n_i}}
 \Psi_{34}(A_2)\,
 (G_{h_2}^{(n_2)})^{-1}_{A_2B_2}
 \\
 &\qquad\times
 \rho(B_2,d_5,A_1)\,
 (G_{h_1}^{(n_1)})^{-1}_{A_1B_1}
 \Psi_{12}(B_1).
\end{aligned}
\tag{3.1}
\]

The boundary wavefunctions are

\[
 \Psi_{12}(B_1)=
 \langle h_1,B_1|\psi_{12}\rangle,
 \qquad
 \Psi_{34}(A_2)=
 \langle\psi_{34}|h_2,A_2\rangle.
\]

They are not ordinary primary three-point tensors in the plane frame.
They include the non-Möbius map from the punctured sphere to the pillow and
the regularization at the conical corners. This is precisely where the
elliptic prefactor differs from the CCY linear-plumbing frame.

The conformal transformation has the schematic form

\[
 \mathcal F_5^{S^2}
 =
 \Lambda_5^{(c)}(z,t)\,
 \mathcal M_5(\tau,w_5),
 \tag{3.2}
\]

where \(\mathcal P_{\rm pill}\) contains:

- the Weyl anomaly of the branched map;
- the four regularized corner factors;
- the ordinary primary Jacobian
  \((dw/dt)^{d_5}\) at the mobile puncture;
- the chosen primary propagation normalization.

The first concrete geometric calculation is to derive (3.2) with all
holomorphic powers fixed.

## 4. Why the \(h\)-poles are controlled by the projectors

At level \(n\), the projector is

\[
 \mathbb P_h^{(n)}
 =
 \sum_{|A|=|B|=n}
 |h,A\rangle\,
 (G_h^{(n)})^{-1}_{AB}\,
 \langle h,B|.
 \tag{4.1}
\]

As a meromorphic function of \(h\), it has a simple pole whenever
\(h=h_{r,s}(c)\). The singular part projects onto the submodule generated by
the level-\(rs\) null vector. Null-vector factorization then turns the
residue into the same matrix element with

\[
 h_{r,s}\longrightarrow h_{r,s}+rs,
\]

multiplied by the inverse null-norm slope \(A_{r,s}(c)\) and the fusion
polynomials at the two vertices adjacent to the singular edge.

This argument is local. It does not care whether the remote ends of the
cylinder are closed into a trace or terminate on pillow states. What changes
between the ordinary torus necklace and the pillow amplitude is the
incidence data at the ends and the regular part, not the Kac-pole mechanism.

## 5. The two polar terms

Denote the external weights at the lower corners by \(d_1,d_2\), those at
the upper corners by \(d_3,d_4\), and the mobile weight by \(d_5\). Up to the
ordering convention for the fusion polynomial, the residues are

\[
\begin{aligned}
 \mathcal R^{(1)}_{r,s}
 &=
 A_{r,s}(c)\,
 P_{r,s}(d_1,d_2;c)\,
 P_{r,s}(h_2,d_5;c),\\
 \mathcal R^{(2)}_{r,s}
 &=
 A_{r,s}(c)\,
 P_{r,s}(d_4,d_3;c)\,
 P_{r,s}(h_1,d_5;c).
\end{aligned}
\tag{5.1}
\]

The project implementation orders the five comb weights as
\((d_1,d_2,d_{\rm middle},d_{\rm upper,1},d_{\rm upper,2})\).
Thus the dictionary to
Code/bosonic_c1_one_to_n_reference/reference_implementation/plumbing/ccy_sphere_five_point.py
is

\[
 (d_1,d_2,d_3,d_4,d_5)_{\rm code}
 =
 (d_1,d_2,d_5,d_3,d_4)_{\rm note}.
\tag{5.1a}
\]

Thus, before choosing a simultaneous large-weight parameterization, the
polar structure is

\[
\begin{aligned}
 \mathcal M_5(h_1,h_2)
 ={}&\mathcal U(h_1,h_2)\\
 &+\sum_{r,s\geq1}
 \frac{\mathfrak p_1^{\,rs}\mathcal R^{(1)}_{r,s}}
      {h_1-h_{r,s}}\,
 \mathcal M_5(h_{r,s}+rs,h_2)\\
 &+\sum_{r,s\geq1}
 \frac{\mathfrak p_2^{\,rs}\mathcal R^{(2)}_{r,s}}
      {h_2-h_{r,s}}\,
 \mathcal M_5(h_1,h_{r,s}+rs).
\end{aligned}
\tag{5.2}
\]

Here \(\mathfrak p_i\) includes the segment nome \(p_i\) and the
internal-weight-dependent local-coordinate normalization. In the ordinary
four-point convention the corresponding factor is \(16q\), not just \(q\).
A natural symmetric pillow convention would assign a factor \(4\) to each
boundary state, suggesting

\[
 \mathfrak p_1=4p_1,\qquad
 \mathfrak p_2=4p_2,\qquad
 \mathfrak p_1\mathfrak p_2=16q.
 \tag{5.3}
\]

Equation (5.3) is the symmetric normalization adopted in Section 6.2. A
derivation from the complete covering factor (3.2) remains desirable.

For a single meromorphic recursion, set

\[
 H=h_1,\qquad a=h_2-h_1
\]

and hold \(a\) fixed. At an edge-1 pole,

\[
 (h_1,h_2)
 =(h_{r,s},h_{r,s}+a)
 \longrightarrow
 (h_{r,s}+rs,h_{r,s}+a),
\]

so the shifted variables are

\[
 H'=h_{r,s}+rs,\qquad a'=a-rs.
\tag{5.4}
\]

At an edge-2 pole,

\[
 (h_1,h_2)
 =(h_{r,s}-a,h_{r,s})
 \longrightarrow
 (h_{r,s}-a,h_{r,s}+rs),
\]

and therefore

\[
 H'=h_{r,s}-a,\qquad a'=a+rs.
\tag{5.5}
\]

These are the same kinematic shifts that occur in a two-edge necklace
recursion. The residue factors are different because an end of each edge is
a pillow boundary state rather than another vertex in a cyclic trace.

Consequently, once a pillow normalization has been fixed, the proposed
fixed-difference recursion has the explicit form

\[
\begin{aligned}
 \mathcal H_5(H,a)
={}&\mathcal U_5\\
&+\sum_{r,s\geq1}
 \frac{\mathfrak p_1^{\,rs}A_{r,s}
 P_{r,s}(d_1,d_2)
 P_{r,s}(h_{r,s}+a,d_5)}
 {H-h_{r,s}}\,
 \mathcal H_5(h_{r,s}+rs,a-rs)\\
&+\sum_{r,s\geq1}
 \frac{\mathfrak p_2^{\,rs}A_{r,s}
 P_{r,s}(d_4,d_3)
 P_{r,s}(h_{r,s}-a,d_5)}
 {H+a-h_{r,s}}\,
 \mathcal H_5(h_{r,s}-a,a+rs).
\end{aligned}
\tag{5.6}
\]

The dependence on \(c\) and on the remaining moduli is suppressed. Equation
(5.6) states the polar reconstruction only; it becomes a complete recursion
only after \(\mathcal U_5\) and the precise \(\mathfrak p_i\) are derived.

## 6. Large-weight regular part

The pole argument alone never determines an \(h\)-recursion. One must also
determine the entire part \(\mathcal U\), equivalently the correlated
large-weight limit after the correct prefactor is removed.

For the four-point block, the pillow transformation gives

\[
 \widetilde{\mathcal F}_4(h;q)
 =
 (16q)^{h-c/24}
 \prod_{n=1}^{\infty}(1-q^{2n})^{-1/2}
 H_4(h;q),
\qquad
 H_4(h;q)\longrightarrow1
\tag{6.1}
\]

up to the standard convention-dependent shift between \(c\) and \(c-1\)
that is moved between the external prefactor and the reduced block.

For five points, the promising correlated limit is

\[
 h_1=H,\qquad h_2=H+a,\qquad H\to\infty,
\tag{6.2}
\]

with \(a,c,d_i,\tau,w_5\) fixed.

The middle primary matrix element has the same large-weight property used
in the CCY necklace proof: after orthogonalizing descendants,

\[
 \frac{
 \langle H+a,A|V_{d_5}|H,B\rangle
 }{
 \sqrt{\langle H+a,A|H+a,A\rangle
       \langle H,B|H,B\rangle}}
 \longrightarrow \delta_{AB}.
\tag{6.3}
\]

Off-diagonal levels and partitions are suppressed. Consequently, to leading
order, the insertion becomes the identity on the high-weight oscillator
labels. The two propagation factors combine as

\[
 p_1^{|A|}p_2^{|A|}
 =q^{|A|}.
\]

We now adopt the statement that the five-point regular oscillator factor is
the same pillow factor as in the four-point problem and is independent of
the position \(w_5\):

\[
 \mathcal U_5(q)
 =
 \prod_{n=1}^{\infty}(1-q^{2n})^{-1/2},
\tag{6.4}
\]

or simply \(1\) after (6.4) and the primary large-\(H\) exponential have
been stripped.

This is the explicit large-weight assumption underlying the final recursion.
Its explanation in terms of the internal-primary three-point function and
the \(\mathbb Z_2\)-twist vacuum block will be given in the human note. The
nontrivial point is that the pillow boundary wavefunctions in (3.1), rather
than plane-frame three-point tensors, must be used before taking the limit.

### 6.1 Completed level-one middle-vertex check

The first nontrivial part of (6.3) follows exactly from the Ward identities.
In the normalized level-zero/level-one bases, with
\(h_1=H,\ h_2=H+a\) and middle weight \(d=d_5\), the relevant matrix
elements are

\[
\begin{aligned}
 \rho(h_2,d,h_1)&=1,\\
 \frac{\rho(L_{-1}h_2,d,h_1)}{\sqrt{2h_2}}
 &=\frac{d+a}{\sqrt{2(H+a)}},\\
 \frac{\rho(h_2,d,L_{-1}h_1)}{\sqrt{2h_1}}
 &=\frac{d-a}{\sqrt{2H}},\\
 \frac{\rho(L_{-1}h_2,d,L_{-1}h_1)}
 {\sqrt{4h_1h_2}}
 &=
 \frac{2H+(d-a)(d+a-1)}
 {\sqrt{4H(H+a)}}.
\end{aligned}
\tag{6.5}
\]

Therefore the diagonal entries tend to one and the level-changing entries
vanish as \(H^{-1/2}\). This verifies the first nontrivial instance of the
claim that the mobile insertion becomes the identity on high-weight
oscillator labels. Higher-level orthogonalized checks remain necessary.

### 6.2 Adopted asymptotic and final recursion

Define the pillow oscillator character

\[
 \chi_{\rm pill}(q)
 =
 \prod_{n=1}^{\infty}(1-q^{2n})^{-1/2}.
\tag{6.6}
\]

We choose the symmetric internal local-coordinate normalization

\[
 \widehat p_1=4p_1,\qquad
 \widehat p_2=4p_2,\qquad
 \widehat p_1\widehat p_2=16q.
\]

The two factors of \(4\) split the four-point \(16^{h-c/24}\) evenly
between the lower and upper pillow states. With the reflection-symmetric
corner regulator used in E.103, the \(h_1\) component of
\(|\psi_{12}\rangle\) carries \(4^{h_1-c/24}\), while the \(h_2\) component
of \(\langle\psi_{43}|\) carries \(4^{h_2-c/24}\). A replacement
\((4,4)\to(\alpha,16/\alpha)\) is merely a rescaling of the two projected
state bases; the symmetric choice fixes that convention.

The adopted simultaneous large-weight asymptotic is

\[
 \mathcal M_5(h_1,h_2;p_1,p_2)
 \underset{\substack{h_1=H,\ h_2=H+a\\H\to\infty}}{\sim}
 \widehat p_1^{\,h_1-c/24}
 \widehat p_2^{\,h_2-c/24}
 \chi_{\rm pill}(q).
\tag{6.7}
\]

Equivalently, using \(p_1p_2=q\),

\[
 \mathcal M_5
 \sim
 (16q)^{H-c/24}(4p_2)^a\,
 \chi_{\rm pill}(q).
\]

The factor \((4p_2)^a\) is only the finite primary propagation associated
with \(h_2-h_1=a\); the oscillator regular part is exactly the same twisted
character as for four points. Define the reduced five-point pillow block by

\[
 \mathcal M_5
 =
 \widehat p_1^{\,h_1-c/24}
 \widehat p_2^{\,h_2-c/24}
 \chi_{\rm pill}(q)\,
 \mathcal H_5(H,a;p_1,p_2),
\qquad
 \mathcal H_5\longrightarrow1.
\tag{6.8}
\]

Combining (2.12) and (6.8) pins down every conformal factor in the
geometric \(c\)-normalization:

\[
\boxed{
 \mathcal F_5^{S^2}
 =
 \Lambda_5^{(c)}(z,t)\,
 (4p_1)^{h_1-c/24}
 (4p_2)^{h_2-c/24}
 \chi_{\rm pill}(q)\,
 \mathcal H_5(H,a;p_1,p_2).}
\tag{6.9}
\]

There is an exactly equivalent Zamolodchikov \(c-1\) normalization. The
theta-function and eta-function identities imply

\[
 \theta_3(\tau)^{1/2}
 [z(1-z)]^{1/24}
 (16q)^{-1/24}
 =
 \chi_{\rm pill}(q)^{-1}.
\tag{6.10}
\]

For any parameter \(\kappa\), define

\[
\begin{aligned}
 \Lambda_5^{(\kappa)}(z,t)
 :={}&
 \theta_3(\tau)^{\,\kappa/2-4(d_1+d_2+d_3+d_4)-2d_5}\\
 &\times z^{\,\kappa/24-d_1-d_2}
 (1-z)^{\,\kappa/24-d_2-d_3}
 [t(1-t)(z-t)]^{-d_5/2}.
\end{aligned}
\]

Equation (6.10) converts (6.9) into

\[
\boxed{
 \mathcal F_5^{S^2}
 =
 \Lambda_5^{(c-1)}(z,t)\,
 (4p_1)^{h_1-(c-1)/24}
 (4p_2)^{h_2-(c-1)/24}
 \mathcal H_5(H,a;p_1,p_2).}
\tag{6.11}
\]

The character occurs in (6.9) but not in (6.11). Keeping both the
\(c-1\) shifts and an explicit
\(\prod_{n\geq1}(1-q^{2n})^{-1/2}\) double-counts the same factor. In
particular, combining E.103 and E.105 algebraically gives (6.11); the
explicit product displayed in E.106 of the current string-note PDF appears
to be an extra factor.

The identity (6.10) and the equivalence of the complete five-point factors
in (6.9) and (6.11) have been checked at two generic complex moduli. The
largest residual was \(5.57\times10^{-16}\). The executable certificate is
Code/sphere_five_kummer_h_recursion/check_pillow_conformal_factors.py.

The final fixed-difference \(h\)-recursion is then

\[
\boxed{
\begin{aligned}
 \mathcal H_5(H,a)
={}&1\\
&+\sum_{r,s\geq1}
 \frac{(4p_1)^{rs}A_{r,s}^{c}}
      {H-h_{r,s}(c)}
 P_{r,s}^{c}\!\begin{bmatrix}d_1\\ d_2\end{bmatrix}
 P_{r,s}^{c}\!\begin{bmatrix}h_{r,s}(c)+a\\ d_5\end{bmatrix}
 \mathcal H_5\!\left(h_{r,s}(c)+rs,a-rs\right)\\
&+\sum_{r,s\geq1}
 \frac{(4p_2)^{rs}A_{r,s}^{c}}
      {H+a-h_{r,s}(c)}
 P_{r,s}^{c}\!\begin{bmatrix}d_4\\ d_3\end{bmatrix}
 P_{r,s}^{c}\!\begin{bmatrix}h_{r,s}(c)-a\\ d_5\end{bmatrix}
 \mathcal H_5\!\left(h_{r,s}(c)-a,a+rs\right).
\end{aligned}}
\tag{6.12}
\]

All arguments \(c,d_i,p_1,p_2\) not shifted by the recursion are suppressed
on the right-hand side. Iteration generates the mixed double-pole terms
automatically.

## 7. Identity-insertion consistency condition

Set \(d_5=0\) and choose the identity channel at the middle vertex. Then

\[
 h_2=h_1,\qquad
 V_{d_5}=\mathbf 1.
\]

The five-point matrix element must collapse exactly to the four-point pillow
matrix element:

\[
 \mathcal M_5
 \longrightarrow
 \langle\psi_{34}|
 (p_1p_2)^{L_0-c/24}
 \mathbb P_h
 |\psi_{12}\rangle
 =
 \widetilde{\mathcal F}_4(h;q).
\tag{7.1}
\]

This fixes several otherwise ambiguous pieces at once:

- \(p_1p_2=q\);
- the product of internal local-coordinate scales must reproduce \(16q\);
- the five-point regular seed must reduce to the four-point pillow seed;
- after setting \(V_{d_5}=\mathbf1\), the two adjacent projectors must be
  composed into a single projector before analytic continuation in \(h\);
  the resulting single pole family is the ordinary four-point one.

Any proposed elliptic five-point normalization must pass (7.1)
symbolically, not only numerically.

## 8. Relation to the two CCY recursions

CCY provide two distinct ingredients:

1. Their sphere linear-channel \(h\)-recursion already gives the exact
   Kac-pole residues of a five-point comb block in the plane plumbing
   variables. Its simple unit regular term comes from sending both internal
   weights and the two endpoint external weights to infinity with fixed
   differences. CCY explicitly emphasize that this is not the
   Zamolodchikov elliptic large-\(h\) limit.

2. Their torus necklace analysis proves the diagonal high-weight behavior
   (6.3) and supplies a useful model for simultaneous two-edge shifts.

Neither result by itself gives the five-point pillow recursion. The new
problem is to combine the already-known polar data with the fixed-external-
weight pillow asymptotics.

An ordinary CCY torus two-point block has a cyclic trace and two local
vertices. Equation (1.1) has an open cylinder, two nontrivial boundary
states, and one local middle vertex. Even though both objects contain two
internal modules, their sewing graphs and regular parts are different.

On the smooth double cover the mobile pillow insertion has two images
\(u,-u\). This does not turn (1.1) into an ordinary torus trace: the four
branch-point operators are still encoded in the boundary-state preparation.

### 8.1 Explicit comparison with the CCY plane \(h\)-recursion

Set

\[
 x=\frac{z}{t},\qquad H=h_1,\qquad a=h_2-h_1,
 \qquad e_L=d_1-H,\qquad e_R=d_4-H.
\]

If \(G(H,a,e_L,e_R;x,t)\) denotes the normalized plane descendant series,
the five-point specialization of CCY Eq. (3.26) is

\[
\begin{aligned}
G(H,a,e_L,e_R;x,t)={}&1\\
&+\sum_{r,s\geq1}\frac{x^{rs}A_{r,s}^{c}}{H-h_{r,s}}
 P_{r,s}^{c}\!\begin{bmatrix}h_{r,s}+e_L\\d_2\end{bmatrix}
 P_{r,s}^{c}\!\begin{bmatrix}h_{r,s}+a\\d_5\end{bmatrix}
 G(h_{r,s}+rs,a-rs,e_L-rs,e_R-rs;x,t)\\
&+\sum_{r,s\geq1}\frac{t^{rs}A_{r,s}^{c}}{H+a-h_{r,s}}
 P_{r,s}^{c}\!\begin{bmatrix}h_{r,s}-a\\d_5\end{bmatrix}
 P_{r,s}^{c}\!\begin{bmatrix}h_{r,s}-a+e_R\\d_3\end{bmatrix}
 G(h_{r,s}-a,a+rs,e_L,e_R;x,t).
\end{aligned}
\tag{8.1}
\]

The two denominators, the shifts of \((H,a)\), and the middle-insertion
fusion polynomials are identical to those in the pillow recursion (6.12).
The boundary fusion polynomials and expansion variables differ. CCY take
\(H,d_1,d_4\to\infty\) at fixed \((a,e_L,e_R)\), yielding a unit plane
regular seed in \((x,t)\). The pillow limit holds every external weight
fixed and expands in \((4p_1,4p_2)\), with the twisted character appearing
before it is stripped. Therefore the pillow recursion cannot be obtained by
only substituting \(x\to4p_1\), \(t\to4p_2\) into (8.1); the coordinate map
and full conformal prefactor must be transformed at the same time.

We implemented (8.1) at 80-digit precision. At the generic parameters used
in the value scan below, all 66 coefficients with \(n_1+n_2\leq10\) agree
with the exact rational PBW series. The largest relative discrepancy is
\(1.51\times10^{-69}\), at \((n_1,n_2)=(10,0)\). This is an independent
check of the CCY dictionary and implementation.

## 9. Concrete derivation and verification program

The direct PBW comparison has now been completed through total bidegree ten.
Writing the plane block as

\[
 \mathcal F_5^{\rm PBW}
 =z^{h_1-d_1-d_2}t^{h_2-h_1-d_5}
 \sum_{n_1,n_2\geq0}C_{n_1,n_2}(z/t)^{n_1}t^{n_2},
\]

we transformed every coefficient with the exact inverse covering products,
divided by (6.9), and compared with (6.12). Generic symbolic coefficients
agree exactly through total degree three.  The higher-order check computes
the PBW contraction and pillow transformation with exact rational arithmetic,
then compares against an independent 80-digit evaluation of the recursion.
All 66 coefficients with \(n_1+n_2\leq10\) agree at five fully asymmetric
points, for 330 comparisons in total.  All seven weights are pairwise
distinct in each case, both signs of \(a=h_2-h_1\) are included, all 65
nonconstant PBW coefficients per case are nonzero, and the smallest recursion
denominator actually encountered is \(1.232\times10^{-3}\).  The largest
relative discrepancy is \(4.54\times10^{-73}\).

The same calculation diagnoses the printed E.103 exponent. Using
\((1-z)^{c/24-d_3-d_4}\) instead of the corrected
\((1-z)^{c/24-d_2-d_3}\) leaves

\[
 [p_1p_2]\left(
 \mathcal H_5^{\rm literal\ E.103}-\mathcal H_5^{\rm recursion}
 \right)=16(d_2-d_4).
\]

Thus the direct block fixes both the \(1-z\) pairing and the aligned
half-period signs in (2.1).

### Value-level comparison of both \(h\)-recursions with the \(c\)-recursion

We also compared the complete plane block reconstructed from the pillow
recursion with the independent CCY fixed-weight (c)-recursion.  The test
uses

\[
 c=31.7,\qquad z=0.08,\qquad
 (d_1,d_2,d_3,d_4,d_5)=(0.21,0.34,0.63,0.79,0.49),\qquad
 (h_1,h_2)=(1.03,1.19),
\]

and scans 61 points over \(0.12\leq t\leq0.72\). At every point
\((p_1,p_2)\) are obtained from the exact product map
\(p_1p_2=q(z)\), \(t=4p_2Y(p_1,p_2)\). The pillow series is truncated at
total elliptic degree ten and restored with the full \(c-1\) conformal
prefactor. The CCY plane \(h\)-recursion is truncated at total degree ten
in \((z/t,t)\), and the plane \(c\)-recursion is truncated at total degree
22.

![Fixed-z pillow and CCY plane h-recursions versus c-recursion](<../../Data Set/h-Recursion/sphere_five_pillow_h_vs_c_fixed_z.png>)

For the pillow representation the median and maximum relative differences
from the order-22 \(c\)-recursion are \(1.22\times10^{-9}\) and
\(2.77\times10^{-5}\). For the CCY plane \(h\)-recursion they are
\(1.81\times10^{-5}\) and \(2.52\times10^{-3}\). At \(t=0.42\) the two
differences are respectively \(9.62\times10^{-11}\) and
\(6.33\times10^{-6}\). At \(t=0.72\), the \(8\to10\) order shifts are
\(2.77\times10^{-7}\) for the pillow recursion and
\(3.20\times10^{-3}\) for the CCY plane recursion, while the \(20\to22\)
\(c\)-recursion shift is \(2.98\times10^{-5}\). Every difference remains
below the corresponding sum of observed shifts. Thus both \(h\)-recursions
converge to the same block, but the pillow variables converge substantially
faster at equal total order on this scan.

### Test A: exact covering factor

Compute the regularized Weyl transformation for arbitrary corner weights and
one ordinary interior insertion. Derive
\(\mathcal P_{\rm pill}\), including the fifth-primary Jacobian and the
local-coordinate factors multiplying \(p_1^{h_1}p_2^{h_2}\).

### Test B: identity insertion

Set \(d_5=0\), \(h_1=h_2\), and verify (7.1) exactly. This is the fastest
check of all factors of \(2\), \(4\), \(16\), \(q\), and \(Q\).

### Test C: pole residues

Expand the pillow-transformed direct sphere five-point block to low level.
At \(h_1=h_{r,s}\) and \(h_2=h_{r,s}\), check that its residues are (5.1)
times the shifted pillow block. This tests whether stripping the proposed
prefactor introduces any additional residue ratios.

### Test D: large-weight middle vertex

At levels zero through four, orthogonalize the Verma basis and evaluate the
normalized matrix of \(V_{d_5}\) at
\((h_1,h_2)=(H,H+a)\). Verify diagonal entries approach one and
off-diagonal entries vanish with the predicted powers of \(H\).

### Test E: regular seed

Using the transformed boundary wavefunctions, compute the first several
coefficients at large \(H\). After stripping the proposed primary
exponential, test (6.4). Position independence at leading order is a sharp
prediction.

### Test F: convergence

Compare equal-cost truncations of:

- the plane comb series;
- the existing CCY plane \(h\)-recursion;
- the new pillow \((q,w_5)\) recursion.

The expected advantage is controlled by the half-torus nome and by choosing
a pillow chart in which the mobile insertion lies away from the two
quantization seams.

## 10. Sphere \(n\)-point open-necklace extension

Let

\[
 m=n-4,\qquad r=m+1=n-3,
\]

where \(m\) fixed-weight mobile primaries \(V_{\mu_j}\) are inserted during
pillow evolution and \(r\) internal modules propagate between the two pillow
boundary states. If \(w_0=0\), \(w_r=\pi\tau\), and \(w_j\) are the ordered
mobile positions, first define the raw segment parameters

\[
 \widetilde p_i=e^{i(w_i-w_{i-1})},
 \qquad \prod_i\widetilde p_i=q.
\]

The phases are fixed by the nested collision limit. For

\[
 0<z<t_1<\cdots<t_m<1,
 \qquad
 x_1=\frac{z}{t_1},\quad
 x_i=\frac{t_{i-1}}{t_i}\ (2\leq i\leq m),\quad
 x_r=t_m,
\]

the comb cusp is \(x_i\to0\). The pillow map gives

\[
 e^{iw(t_j)}=-\frac{z}{4t_j}[1+O(\boldsymbol x)],
 \qquad q=\frac{z}{16}[1+O(z)].
\]

Consequently,

\[
 \widetilde p_1=-\frac{x_1}{4}[1+O(\boldsymbol x)],
 \qquad
 \widetilde p_i=x_i[1+O(\boldsymbol x)]\quad(2\leq i\leq r-1),
 \qquad
 \widetilde p_r=-\frac{x_r}{4}[1+O(\boldsymbol x)].
\]

Thus, for \(r\ge2\), the aligned phases are uniquely fixed:

\[
 \boxed{
 \varepsilon_1=\varepsilon_r=-1,
 \qquad \varepsilon_i=+1\quad(2\leq i\leq r-1),
 \qquad p_i=\varepsilon_i\widetilde p_i.}
\]

Their product is one, so \(\prod_i p_i=q\). For other kinematic regions
these phases are analytically continued from the nested collision region;
they are not independent conventions.

The open-necklace matrix element is schematically

\[
 \mathcal M_n=
 \langle\psi_{43}|\mathbb P_{h_r}p_r^{D}V_{\mu_m}
 \mathbb P_{h_{r-1}}p_{r-1}^{D}\cdots
 V_{\mu_1}p_1^{D}\mathbb P_{h_1}|\psi_{12}\rangle,
 \qquad D=L_0-c/24.
\]

### 10.1 Exact sphere-to-pillow factor and computational extraction

This is the key conversion needed before applying the recursion. In the
present conventions,

\[
 w(t)=\theta_3(\tau)^{-2}\int_0^t
 \frac{dy}{\sqrt{y(1-y)(z-y)}},
 \qquad
 \frac{dw_j}{dt_j}=\frac{1}{\theta_3(\tau)^2
 \sqrt{t_j(1-t_j)(z-t_j)}}.
\]

The exact relation between the plane sphere block and the pillow matrix
element is

\[
\boxed{
 \begin{aligned}
 \mathcal F_n^{S^2}(z;\boldsymbol t)
 ={}&\theta_3(\tau)^{c/2-4\sum_{a=1}^4d_a}
 z^{c/24-d_1-d_2}(1-z)^{c/24-d_2-d_3}\\
 &\times\prod_{j=1}^{m}
 \left(\frac{dw_j}{dt_j}\right)^{\mu_j}
 \mathcal M_n(\tau;\boldsymbol w).
 \end{aligned}}
\tag{10.1}
\]

Every mobile primary is inserted at a nonsingular point, so it contributes
only its own primary Jacobian. There are no extra pairwise prime-form factors
in the coordinate transformation; all dependence coupling different mobile
positions remains inside \(\mathcal M_n\).

Define

\[
 \Lambda_n^{(\kappa)}=
 \theta_3^{\kappa/2-4\sum_{a=1}^{4}d_a-2\sum_j\mu_j}
 z^{\kappa/24-d_1-d_2}(1-z)^{\kappa/24-d_2-d_3}
 \prod_j[t_j(1-t_j)(z-t_j)]^{-\mu_j/2}.
\tag{10.2}
\]

Then \(\mathcal F_n^{S^2}=\Lambda_n^{(c)}\mathcal M_n\), and after
introducing the reduced elliptic block below one has the equivalent exact
forms

\[
 \boxed{
 \mathcal F_n^{S^2}=\Lambda_n^{(c)}
 \left[\prod_i\rho_i^{h_i-c/24}\right]
 \chi_{\rm pill}(q)\mathcal H_n
 =\Lambda_n^{(c-1)}
 \left[\prod_i\rho_i^{h_i-(c-1)/24}\right]\mathcal H_n.}
\tag{10.3}
\]

Most importantly, a direct PBW sphere block is converted into the elliptic
block by division:

\[
\boxed{
 \mathcal H_n=
 \frac{\mathcal F_n^{S^2}}
 {\Lambda_n^{(c-1)}\prod_i\rho_i^{h_i-(c-1)/24}}
 =
 \frac{\mathcal F_n^{S^2}}
 {\Lambda_n^{(c)}\prod_i\rho_i^{h_i-c/24}
  \chi_{\rm pill}(q)}.}
\tag{10.4}
\]

The two expressions agree because

\[
 \theta_3^{1/2}[z(1-z)]^{1/24}(16q)^{-1/24}
 =\chi_{\rm pill}(q)^{-1},
 \qquad \prod_i\rho_i=16q.
\tag{10.5}
\]

Thus the concrete algorithm is: choose the branch quartet and channel;
compute the \(w_j\); form the segment nomes and fix their phases from the
collision limit; form the endpoint-normalized \(\rho_i\); divide the PBW
plane block by the denominator in (10.4); re-expand in the aligned \(p_i\);
and apply the edge \(h\)-recursion to the resulting \(\mathcal H_n\).

### 10.2 Why there are only two factors of four

The factors of four arise from the two regulated pillow boundary states.
They are not attached to every new propagation segment. The effective
plumbing parameters are

\[
 \boxed{
 \rho_i=4^{\delta_{i1}+\delta_{ir}}p_i,
 \qquad \prod_{i=1}^{r}\rho_i=16q.}
\tag{10.6}
\]

More strongly, for \(r\ge2\) the collision limit fixes each effective
plumbing coordinate separately:

\[
\boxed{\rho_i=x_i[1+O(\boldsymbol x)]}.
\]

For \(r=1\), set \(x_1=z\); then
\(\rho_1=16q=x_1[1+O(x_1)]\).

Thus

- four points: \(r=1\), \(\rho_1=16q\);
- five points: \(r=2\), \((\rho_1,\rho_2)=(4p_1,4p_2)\);
- six points: \(r=3\), \((\rho_1,\rho_2,\rho_3)=(4p_1,p_2,4p_3)\).

Giving every segment a factor of four would produce \(4^rq\) and would fail
the reduction in which all mobile insertions become identities and the
projectors combine into the single four-point projector.

### 10.3 Diagonal large-weight limit

Set

\[
 h_i=H+a_i,\qquad a_1=0,\qquad H\to\infty,
\]

with all \(a_i\), external weights, and moduli fixed. At every middle vertex
the CCY large-weight argument gives

\[
 \frac{\langle H+a_{j+1},A|V_{\mu_j}|H+a_j,B\rangle}
 {\|H+a_{j+1},A\|\,\|H+a_j,B\|}
 \longrightarrow\delta_{AB}.
\]

All internal oscillator labels therefore coincide. Their descendant
propagation depends only on \(\prod_i p_i=q\), and the terminal pillow
wavefunctions give the same twisted character as at four and five points:

\[
\boxed{
 \mathcal M_n
 \sim
 \left[\prod_{i=1}^{r}\rho_i^{h_i-c/24}\right]
 \prod_{k\ge1}(1-q^{2k})^{-1/2}.}
\tag{10.7}
\]

Equivalently,

\[
 \mathcal M_n\sim
 (16q)^{H-c/24}
 \left[\prod_{i=2}^{r}\rho_i^{a_i}\right]
 \prod_{k\ge1}(1-q^{2k})^{-1/2}.
\]

The finite product containing \(a_i\) is primary propagation. Each new
middle insertion becomes the identity on the leading oscillator labels, so
it does not supply another character. Define the reduced block by

\[
 \mathcal M_n=
 \left[\prod_i\rho_i^{h_i-c/24}\right]
 \chi_{\rm pill}(q)\mathcal H_n,
 \qquad \mathcal H_n\to1.
\tag{10.8}
\]

### 10.4 General edge recursion

At the Kac pole on edge \(i\),

\[
 H+a_i=h_{\alpha,\beta}(c).
\]

The left and right fusion factors are

\[
 \mathcal L_{i;\alpha,\beta}=
 \begin{cases}
 P_{\alpha,\beta}^{c}\!\begin{bmatrix}d_1\\d_2\end{bmatrix},&i=1,\\
 P_{\alpha,\beta}^{c}\!\begin{bmatrix}
 h_{\alpha,\beta}+a_{i-1}-a_i\\\mu_{i-1}
 \end{bmatrix},&i>1,
 \end{cases}
\]

\[
 \mathcal R_{i;\alpha,\beta}=
 \begin{cases}
 P_{\alpha,\beta}^{c}\!\begin{bmatrix}
 h_{\alpha,\beta}+a_{i+1}-a_i\\\mu_i
 \end{bmatrix},&i<r,\\
 P_{\alpha,\beta}^{c}\!\begin{bmatrix}d_4\\d_3\end{bmatrix},&i=r.
 \end{cases}
\]

For the base edge, the null-module shift is

\[
 (H,a_2,\ldots,a_r)\mapsto
 (h_{\alpha,\beta}+\alpha\beta,
  a_2-\alpha\beta,\ldots,a_r-\alpha\beta).
\]

For \(i>1\), it is

\[
 (H,a_i)\mapsto
 (h_{\alpha,\beta}-a_i,a_i+\alpha\beta),
\]

with every other \(a_j\) fixed. Denoting this map by
\(\mathsf T_{i;\alpha,\beta}\), the recursion is

\[
\boxed{
 \mathcal H_n
 =1+\sum_{i=1}^{r}\sum_{\alpha,\beta\ge1}
 \frac{\rho_i^{\alpha\beta}A_{\alpha,\beta}^{c}
       \mathcal L_{i;\alpha,\beta}\mathcal R_{i;\alpha,\beta}}
      {H+a_i-h_{\alpha,\beta}(c)}
 \mathcal H_n\!\left(
 \mathsf T_{i;\alpha,\beta}(H,\boldsymbol a);\boldsymbol p\right).}
\tag{10.9}
\]

For \(r=2\), this is precisely the five-point recursion (6.12). The
extension from five to \(n\) points introduces no new regular function: the
only common regular term is the single pillow character in (10.7).

### 10.5 Six-point PBW certificate

For the six-point comb at \((0,z,t_1,t_2,1,\infty)\), define

\[
 x_1=z/t_1,\qquad x_2=t_1/t_2,\qquad x_3=t_2,
 \qquad q=p_1p_2p_3.
\]

Writing \(t_1=4p_2p_3Y_1\), \(t_2=4p_3Y_2\), and \(z=16qZ\), the exact
PBW-to-pillow variable map is

\[
 \boxed{x_1=4p_1Z/Y_1,\qquad x_2=p_2Y_1/Y_2,\qquad x_3=4p_3Y_2.}
\tag{10.10}
\]

Here \(Y_1\) is the five-point pillow product for aggregate nomes
\((p_1,p_2p_3)\), while \(Y_2\) uses \((p_1p_2,p_3)\). This directly tests
that the middle segment carries no factor of four.

The plane primary factor is

\[
 z^{h_1-d_1-d_2}t_1^{h_2-h_1-d_3}t_2^{h_3-h_2-d_4}.
\]

After division by the exact sphere-to-pillow factor, the remaining unit
series multiplying the transformed descendant series is

\[
 \begin{aligned}
 &Z^{h_1-c/24}Y_1^{a_2}Y_2^{a_3-a_2}
 [(1-t_1)(1-z/t_1)]^{d_3/2}
 [(1-t_2)(1-z/t_2)]^{d_4/2}\\
 &\quad\times
 \theta_3^{-c/2+4(d_1+d_2+d_5+d_6)+2(d_3+d_4)}
 (1-z)^{-c/24+d_2+d_5}\chi_{\rm pill}^{-1}.
 \end{aligned}
\tag{10.11}
\]

A PBW term of degree \((n_1,n_2,n_3)\) additionally receives
\(4^{n_1+n_3}\), with no \(4^{n_2}\).

The exact symbolic test compares every coefficient with total degree at
most three as a rational function of
\((H,a_2,a_3,d_1,\ldots,d_6,b)\). All 20 coefficients agree exactly.

The order-ten test compares 286 coefficients at each of ten asymmetric
generic internal-weight points. The ensemble includes ascending and
descending triples, central peaks and valleys, two near-equal triples, and
widely separated weights. The PBW and conformal-factor sides use exact
rational arithmetic, while the independent recursion uses 80 decimal
digits. Every recursively visited denominator is audited before the PBW
calculation.

| Case | \((h_1,h_2,h_3)\) | Minimum denominator | Maximum relative error |
|---:|:---|---:|---:|
| 1 | \((0.9371,1.0837,1.3321)\) | \(1.60\times10^{-3}\) | \(2.85\times10^{-71}\) |
| 2 | \((1.4193,0.6871,1.1098)\) | \(1.34\times10^{-3}\) | \(5.16\times10^{-71}\) |
| 3 | \((0.7517,1.5372,0.8894)\) | \(9.97\times10^{-3}\) | \(4.24\times10^{-70}\) |
| 4 | \((1.6423,1.2711,0.8237)\) | \(5.73\times10^{-3}\) | \(3.22\times10^{-71}\) |
| 5 | \((0.9043,0.9187,0.8871)\) | \(2.80\times10^{-3}\) | \(7.39\times10^{-71}\) |
| 6 | \((0.6189,1.2473,1.8891)\) | \(4.98\times10^{-3}\) | \(2.53\times10^{-70}\) |
| 7 | \((1.1267,1.8429,0.5679)\) | \(1.71\times10^{-3}\) | \(3.44\times10^{-69}\) |
| 8 | \((1.7731,0.5427,1.2863)\) | \(4.82\times10^{-3}\) | \(7.38\times10^{-73}\) |
| 9 | \((1.1047,1.1299,1.0873)\) | \(6.63\times10^{-3}\) | \(5.43\times10^{-72}\) |
| 10 | \((0.4831,1.9637,1.2249)\) | \(6.00\times10^{-4}\) | \(2.29\times10^{-69}\) |

All nine weights are distinct at every point, and no nonconstant PBW
coefficient vanishes. Thus all 2,860 numerical coefficients pass. The
global worst coefficient is \((0,10,0)\) in case 7, a pure middle-edge
excitation, with relative error \(3.44\times10^{-69}\).

## 11. Existing code relevant to the corrected program

- Code/bosonic_c1_one_to_n_reference/reference_implementation/plumbing/ccy_sphere_five_point.py
  contains the direct descendant block and the exact CCY plane-frame
  five-point \(h\)-recursion.
- Code/bosonic_c1_one_to_n_reference/reference_implementation/plumbing/virasoro_plumbing_graph.py
  contains the Gram matrices and descendant three-point tensors needed for
  Tests C and D.
- Code/sphere_five_kummer_h_recursion/check_ccy_cusp_factorization.py
  checks an ordinary torus two-point degeneration. It is now an auxiliary
  check, not evidence for identifying the sphere five-point block with that
  torus block.
- Code/sphere_five_kummer_h_recursion/check_pillow_conformal_factors.py
  checks the theta identity and the equality of the full \(c\) and \(c-1\)
  conformal factors with zero, one, and three mobile insertions.
- Code/sphere_five_kummer_h_recursion/check_pillow_h_recursion_symbolic_order4.py
  performs the exact generic symbolic PBW comparison (used through total
  degree three).
- Code/sphere_five_kummer_h_recursion/check_pillow_h_recursion_numerical_order10.py
  checks all 66 coefficients with \(n_1+n_2\leq10\) at five asymmetric
  generic points using exact rational PBW data and an 80-digit recursion,
  audits all weights and recursion denominators, and certifies both
  normalization obstructions.
- Code/sphere_five_kummer_h_recursion/compare_pillow_h_vs_c_recursion_fixed_z.py
  reconstructs the complete plane block from the elliptic recursion,
  evaluates the CCY plane \(h\)-recursion at 80-digit precision, checks its
  coefficients against exact PBW data through total order ten, compares both
  \(h\)-representations with the CCY \(c\)-recursion over a fixed-\(z\) scan,
  and writes the CSV, JSON, and vector plot.
- Code/sphere_five_kummer_h_recursion/render_fixed_z_comparison_png.py
  renders the high-resolution raster figure from the saved CSV.
- Code/sphere_five_kummer_h_recursion/check_sphere_n_pillow_recursion_structure.py
  checks the endpoint normalization and every fixed-difference residue shift
  for one through eight internal segments.
- Code/sphere_five_kummer_h_recursion/check_sphere_six_pillow_h_recursion_symbolic_order3.py
  proves all 20 six-point identities through total degree three exactly.
- Code/sphere_five_kummer_h_recursion/check_sphere_six_pillow_h_recursion_numerical_order10.py
  checks 286 coefficients at each of ten generic points through total
  degree ten, using exact rational PBW data and an 80-digit recursion.

## 12. References

1. M. Cho, S. Collier, and X. Yin,
   “Recursive Representations of Arbitrary Virasoro Conformal Blocks,”
   arXiv:1703.09805, especially Sections 2.3 and 3.3--3.5.
2. J. Maldacena, D. Simmons-Duffin, and A. Zhiboedov,
   “Looking for a bulk point,” arXiv:1509.03612, Section 7 and Appendix D
   for pillow quantization and regularization.
3. A. B. Zamolodchikov, the elliptic internal-weight recursion summarized
   in equation (2.19)--(2.20) of arXiv:1703.09805.
4. References/string notes-3.pdf, Appendix E.7, especially equations
   (E.100)--(E.107), for the conventions used in (1.0), (2.1), and (6.1).

## 13. Bottom line

The fifth puncture should be treated as an operator inserted during pillow
evolution:

\[
 \boxed{
 \langle\psi_{34}|
 \mathbb P_{h_2}p_2^{L_0-c/24}
 V_{d_5}
 p_1^{L_0-c/24}\mathbb P_{h_1}
 |\psi_{12}\rangle,
 \qquad p_1p_2=q.}
\]

The two projectors give the two Kac-pole families. The middle vertex gives
the fusion factors coupling \(h_1,d_5,h_2\). The pillow boundary states give
the other two fusion factors and, crucially, determine the elliptic
large-weight prefactor. Assuming that their large-weight limit is the same
\(\mathbb Z_2\)-twisted character as at four points fixes the regular part
and gives the closed recursion (6.12). The remaining conceptual task is the
human-note derivation of this assumption, not the computation of an ordinary
torus two-point trace.

For a sphere \(n\)-point open-necklace block, insert \(n-4\) fixed-weight
primaries during the same pillow evolution. There are \(r=n-3\) internal
segments, but still only two regulated pillow boundaries. Consequently
\(\rho_i=4^{\delta_{i1}+\delta_{ir}}p_i\) and
\(\prod_i\rho_i=16q\). In the diagonal limit
\(h_i=H+a_i\), every middle vertex becomes the identity on the leading
oscillator labels. The regular part is therefore the same single twisted
character, and the complete \(r\)-edge recursion is (10.9).
