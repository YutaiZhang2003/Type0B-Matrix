# Genus-Two Vacuum Virasoro Blocks

This note accompanies `genus2_vacuum_blocks.py`.  The implementation has three
layers:

1. the large-\(c\) genus-two vacuum block in Schottky coordinates for the
   glasses and sunrise plumbing channels;
2. a finite-\(c\) direct descendant sum in the theta pair-of-pants sewing frame
   of Cho-Collier-Yin, arXiv:1703.09805;
3. low-order Schottky-sewing comparison formulae from
   Headrick-Maloney-Perlmutter-Zadeh, arXiv:1503.07111.

The large-\(c\) Schottky product is the part that is presently implemented
directly for both requested plumbing channels.  The finite-\(c\) machinery is
kept in the same module because it is the substrate needed for the full
higher-genus central-charge recursion, but the current direct finite-\(c\)
series uses theta sewing variables rather than a glasses/sunrise coordinate
map.

## Large-\(c\) Schottky Product

For a genus-\(g\) Schottky group \(\Gamma\), the \(c\)-regular vacuum-block
seed used in arXiv:1703.09805 is

\[
Z_{\mathrm{vac}}^{(g)}(\Gamma)
=
\prod_{\gamma\in \mathcal{P}}
\prod_{n=2}^{\infty}
\left(1-q_\gamma^n\right)^{-1/2}.
\]

Here \(\mathcal{P}\) is the set of primitive conjugacy classes of the Schottky
group, and \(q_\gamma\) is the attracting multiplier of \(\gamma\).  In genus
two, \(\Gamma\) is a rank-two free group, so the code enumerates primitive
cyclically reduced words in two generators.  Cyclic rotations and word
inverses are identified, so each primitive conjugacy class contributes once.

For a Mobius matrix \(M_\gamma\) with trace \(t\), determinant \(\Delta\), and
eigenvalues \(\lambda_+\), \(\lambda_-\), the attracting multiplier is computed
as

\[
q_\gamma = \frac{\lambda_-}{\lambda_+},
\qquad
\lambda_+\lambda_-=\Delta,
\qquad
|\lambda_+|\ge |\lambda_-|.
\]

Numerically the code evaluates this as

\[
q_\gamma = \frac{\Delta}{\lambda_+^2},
\]

which avoids subtractive cancellation in the smaller eigenvalue.

The truncated logarithm is

\[
\log Z_{\mathrm{vac}}
=
-\frac12
\sum_{\gamma\in\mathcal{P}_{\le L}}
\sum_{n=2}^{N}
\log(1-q_\gamma^n),
\]

with `max_word_length` \(L\) and `max_mode` \(N\).  The reported
`omitted_estimate` bounds only the oscillator tail \(n>N\); it is not a bound
on the omitted primitive words of length \(>L\).

## Plumbing Channels

The module reuses the Schottky generators from `plumbing_algorithms.py`.

### Glasses/Sunglasses

`glasses_vacuum_block(q1, q2, q_bridge)` evaluates the Schottky product using
the glasses generators.  The parameters \(q_1\) and \(q_2\) are the two
self-plumbing multipliers, and \(q_b\) is the bridge parameter.

In the separating limit \(q_b\to 0\), the genus-two surface becomes two genus
one components.  The implemented check verifies

\[
Z_{\mathrm{vac}}^{\mathrm{glasses}}(q_1,q_2,q_b)
\longrightarrow
\chi_{\mathrm{vac}}(q_1)\,\chi_{\mathrm{vac}}(q_2),
\]

where

\[
\chi_{\mathrm{vac}}(q)
=
\prod_{n=2}^{\infty}
\left(1-q^n\right)^{-1/2}.
\]

### Sunrise

`sunrise_vacuum_block(q0, q1, q2)` evaluates the same Schottky product using
the sunrise generators.  The sunrise graph is symmetric under permutations of
its three edges, although the implementation chooses the \(q_0\) edge as a
spanning-tree edge when building Schottky generators.  The checks verify that
this auxiliary choice does not change the block:

\[
Z_{\mathrm{vac}}^{\mathrm{sunrise}}(q_0,q_1,q_2)
=
Z_{\mathrm{vac}}^{\mathrm{sunrise}}(q_{\sigma(0)},q_{\sigma(1)},q_{\sigma(2)})
\]

for the tested edge permutations \(\sigma\), up to primitive-word truncation
error.

The checks also include a sunrise edge-collapse limit.  When \(q_1\to0\), one
cycle collapses and the remaining rank-one Schottky multiplier is \(q_0q_2\).
The block approaches

\[
Z_{\mathrm{vac}}^{\mathrm{sunrise}}(q_0,q_1,q_2)
\longrightarrow
\chi_{\mathrm{vac}}(q_0q_2).
\]

The Schottky product itself is channel agnostic once the two generators have
been constructed.

## Finite-\(c\) Theta Sewing Substrate

The file also implements a direct finite-\(c\) descendant sum for the
pair-of-pants, or theta, decomposition used in equation (5.5) of
arXiv:1703.09805.  For three sewing parameters \(q_1,q_2,q_3\), the vacuum
specialization has the form

\[
\mathcal{F}_{\mathrm{vac}}(q_1,q_2,q_3)
=
\sum_{\substack{A_i,B_i\in\mathcal{H}_{\mathrm{vac}}\\ i=1,2,3}}
q_1^{|A_1|}
q_2^{|A_2|}
q_3^{|A_3|}
\rho(A_1,A_2,A_3)
\prod_{i=1}^{3} G^{A_iB_i}
\rho(B_1,B_2,B_3).
\]

The implementation truncates each tube to level at most `max_level`.
Descendants are represented in a PBW basis

\[
L_{-n_1}L_{-n_2}\cdots L_{-n_k}|0\rangle,
\qquad
n_1\ge n_2\ge\cdots\ge n_k\ge 2.
\]

The condition \(n_i\ge2\) is the vacuum quotient

\[
L_{-1}|0\rangle=0.
\]

The Gram matrix is computed directly from the Virasoro algebra,

\[
[L_m,L_n]
=
(m-n)L_{m+n}
\,+\,
\frac{c}{12}m(m^2-1)\delta_{m+n,0}.
\]

For example, the checks verify

\[
G_{(2),(2)}=\frac{c}{2},
\qquad
G_{(3),(3)}=2c,
\]

and at level four, in the basis \(\{L_{-4}|0\rangle,L_{-2}^2|0\rangle\}\),

\[
G_4=
\begin{pmatrix}
5c & 3c\\
3c & \frac12 c(c+8)
\end{pmatrix}.
\]

The three-point functional \(\rho\) is evaluated using the Ward identities from
Appendix A of arXiv:1703.09805, with insertions at \(0\), \(1\), and
\(\infty\).  The basic regression values are

\[
\rho(0,0,0)=1,
\qquad
\rho(0,L_{-2},0)=0,
\qquad
\rho(L_{-2},0,L_{-2})=\frac{c}{2},
\qquad
\rho(L_{-2},L_{-2},L_{-2})=c.
\]

Through level two the direct theta sewing block is checked against

\[
\mathcal{F}_{\mathrm{vac}}
=
1
+q_1^2q_2^2
+q_1^2q_3^2
+q_2^2q_3^2
+\frac{8}{c}q_1^2q_2^2q_3^2
+O(q^3).
\]

Use `theta_finite_vacuum_block` for this frame.  The compatibility wrappers
`glasses_finite_vacuum_block` and `sunrise_finite_vacuum_block` intentionally
emit warnings: they do not perform a Schottky-to-theta coordinate conversion.

## HMPZ Schottky-Sewing Cross-Check

Headrick-Maloney-Perlmutter-Zadeh compute the genus-two vacuum-channel
partition function in Schottky sewing variables \((p_1,p_2,x)\):

\[
Z_{\mathrm{vac}}
=
\sum_{h_1,h_2\ge0}
p_1^{h_1}p_2^{h_2}
C_{h_1,h_2}(x).
\]

The implemented HMPZ generator map fixes the first handle at \((0,\infty)\)
and the second at \((1,x)\).  With

\[
\gamma_{a,r}(z)=\frac{rz+a}{z+1},
\qquad
\delta_p(z)=pz,
\]

the Schottky generators are

\[
\Gamma_1(z)=p_1z,
\qquad
\Gamma_2
=
\gamma_{1,x}\circ\delta_{p_2}\circ\gamma_{1,x}^{-1}.
\]

The code includes the level-two coefficient

\[
C_{2,2}(x)
=
1+(x-1)^4+\frac{(x-1)^4}{x^4}
+\frac{8}{c}\,
\frac{(x-1)^2(1-x+x^2)}{x^2}.
\]

This gives the low-order Schottky-sewing block

\[
Z_{\mathrm{vac}}^{(\le2)}
=
1+p_1^2+p_2^2+p_1^2p_2^2C_{2,2}(x).
\]

The checks verify the crossing property

\[
C_{2,2}(x)=C_{2,2}(1/x),
\]

and the separating factorization

\[
C_{2,2}(1)=1,
\qquad
Z_{\mathrm{vac}}^{(\le2)}(p_1,p_2,1)
=(1+p_1^2)(1+p_2^2).
\]

Because the arXiv:1703.09805 seed is the square-root one-loop product, while
HMPZ use the unsquared holomorphic one-loop partition normalization, the
large-\(c\) comparison is made against the square of the Schottky product.
At order \(p_1^2p_2^2\), the HMPZ mixed large-\(c\) coefficient is

\[
(x-1)^4+\frac{(x-1)^4}{x^4}.
\]

The check extracts this coefficient directly from the length-two primitive
Schottky words \(AB\) and \(AB^{-1}\):

\[
\left(\frac{q_{AB}}{p_1p_2}\right)^2
+
\left(\frac{q_{AB^{-1}}}{p_1p_2}\right)^2
\longrightarrow
(x-1)^4+\frac{(x-1)^4}{x^4},
\qquad
p_1,p_2\to0.
\]

The file also includes the HMPZ level-four quasi-primary contribution
\(C_{4,4}|_{\Lambda}\):

\[
\begin{aligned}
C_{4,4}|_{\Lambda}(x)
&=
\frac{(1-x+x^2)^8}{x^8}
\,+\,
\left(\frac{32}{c}-8\right)
\frac{(x-1)^2(1-x+x^2)^5}{x^6}
\\
&\quad+
\frac{4(3704+590c+125c^2)}{5c(22+5c)}
\frac{(x-1)^4(1-x+x^2)^2}{x^4}.
\end{aligned}
\]

This is the first source of an all-orders \(1/c\) expansion in the genus-two
vacuum free energy.  The implemented leading three-loop term is

\[
F_{\mathrm{vac};3}
=
p_1^4p_2^4\,
\frac{13312 (x-1)^4(1-x+x^2)^2}{25x^4}
+O(p_1^4p_2^5).
\]

The check verifies that this term vanishes at \(x=1\), the separating
degeneration point, and is nonzero for a generic sample away from that point.

## Modularity Checks

A single handlebody block is not a scalar invariant under the full
\(\mathrm{Sp}(4,\mathbb{Z})\) modular group.  A full modular-invariant genus-two
partition function requires summing over the appropriate modular images, as in
genus-two modular-bootstrap applications.  What the code can check directly in
Schottky coordinates is invariance under handlebody-preserving changes of
Schottky marking.

The implemented checks include:

\[
A\leftrightarrow B,
\qquad
A\mapsto A^{-1},
\qquad
A\mapsto AB,
\qquad
B\mapsto BA.
\]

These are Nielsen transformations of the rank-two Schottky marking.  The
truncated product is unchanged to numerical precision for the tested samples.

## Command Line

Evaluate a glasses-channel large-\(c\) sample:

```bash
./.venv/bin/python plumbing/genus2_vacuum_blocks.py \
  --channel glasses \
  --q 0.04 0.03 0.05 \
  --max-word-length 8 \
  --max-mode 50
```

Evaluate a sunrise-channel large-\(c\) sample:

```bash
./.venv/bin/python plumbing/genus2_vacuum_blocks.py \
  --channel sunrise \
  --q 0.026+0.005j 0.031-0.011j 0.023+0.013j
```

Evaluate the direct finite-\(c\) theta descendant sum through level two:

```bash
./.venv/bin/python plumbing/genus2_vacuum_blocks.py \
  --finite-c \
  --central-charge 17 \
  --max-level 2 \
  --q 0.01+0.002j -0.015+0.001j 0.012-0.003j \
  --list-contributions
```

Run the built-in checks:

```bash
./.venv/bin/python plumbing/genus2_vacuum_blocks.py --check
```

## Current Status

Implemented:

- large-\(c\) Schottky vacuum product for glasses and sunrise plumbing
  generators;
- primitive conjugacy-class enumeration with cyclic/inverse identification;
- separating degeneration check for the glasses channel;
- sunrise edge-permutation and edge-collapse checks;
- Schottky marking checks via swap, inversion, and Nielsen moves;
- finite-\(c\) theta-frame descendant algebra, Gram matrices, Ward functional,
  and direct level-cutoff sewing sum;
- HMPZ low-order Schottky-sewing comparison formulae and checks.

Not yet implemented:

- the full central-charge recursion of arXiv:1703.09805;
- an explicit coordinate conversion from glasses/sunrise Schottky parameters to
  the finite-\(c\) theta sewing variables;
- a modular-image sum giving a scalar \(\mathrm{Sp}(4,\mathbb{Z})\)-invariant
  genus-two partition function.
