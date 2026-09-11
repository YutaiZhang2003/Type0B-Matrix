# Independent review of the Schottky vacuum factor

This review reads Section 5 of [Cho, Collier and Yin,
arXiv:1703.09805](https://arxiv.org/pdf/1703.09805) and inspects
`schottky_vacuum.py`, its import in `punctured_ccy.py`, and its use in
`pipeline.py`. It performs no numerical CFT calculation, comparison with the
old Gaussian coefficients, or component cross-check. The only numerical
validation of the final physical block remains the two checks requested by
the user.

## What is taken from CCY

Equation (5.3) separates the large-central-charge block into its global
SL(2) block and universal vacuum factor in the same plumbing frame.
Equations (5.4)–(5.5) express the latter through the Schottky group and the
product over primitive conjugacy classes. CCY's exponent is **minus one
half**, with oscillator powers starting at two. This formula supplies the
vacuum factor only; the global seed and finite-central-charge residue
recursion remain necessary.

The statements and implementation identities below derive the proposed
coefficient algorithm from that product, for the actual local coordinates
used in this pipeline.

## Sewing convention

The ordered theta vertices use punctures `(infinity, one, zero)` for edges
`(1,2,3)` on both spheres. The sewing conditions and their crossing maps are

\[
\begin{array}{c|c|c}
e&\text{sewing relation}&z_B\text{ in terms of }z_A\\\hline
1&(1/z_A)(1/z_B)=q_1&1/(q_1z_A)\\
2&(z_A-1)(z_B-1)=q_2&1+q_2/(z_A-1)\\
3&z_Az_B=q_3&q_3/z_A
\end{array}
\]

They give precisely the three implemented projective matrices

\[
\begin{pmatrix}0&1\\q_1&0\end{pmatrix},\qquad
\begin{pmatrix}1&q_2-1\\1&-1\end{pmatrix},\qquad
\begin{pmatrix}0&q_3\\1&0\end{pmatrix}.
\]

The matrix for edge `e` squares to `q_e` times the identity and has
determinant `-q_e`. It therefore represents the inverse crossing map too,
up to an irrelevant projective scalar. Using the same matrix in both
directions is valid for these three same-puncture gluings. It must not be
assumed for an arbitrary gluing between different punctures.

The implementation left-multiplies each successive crossing matrix. Thus a
closed walk has the correctly ordered composite map on its initial sphere.
Cyclic rotation changes the starting point and conjugates that map; reversing
the directed walk gives its inverse. No matrix normalization by a square root
of its determinant is necessary.

The split identity sphere in the universal factor can be removed using
`q2=qL*qR`, as already established for the specified plumbing. This gives
the implemented lift `(2*a,b,b,2*c)`. It does not impose equal split powers
on the numerator, auxiliary factor, or quotient. The direct-auxiliary
pipeline still restores two copies of this universal factor.

## Primitive classes and orientation

The sewing graph is a spine of the selected handlebody. Closed reduced
graph walks describe conjugacy classes of its free fundamental group; the
Möbius holonomy realizes the Schottky group near the plumbing degeneration.
“Primitive” here means **not a proper power of a shorter closed walk**, not
an element belonging to some free generating basis.

The code represents both orientations of every edge by separate arc IDs.
For theta these are paired as `2*i <-> 2*i+1`, with reversed source and
target. The inverse-arc consistency check is correct. The walk enumerator

- extends only along arcs leaving the current vertex;
- excludes immediate reversal of the preceding arc;
- excludes reversal across the cyclic join;
- tests proper powers using the full directed arc word;
- identifies all cyclic rotations and the inverse directed word.

Retaining the directed arcs matters beyond the first few lengths: an odd
period in theta's undirected edge labels changes the base vertex and is not
a shorter closed walk. The code does not make that erroneous identification.

A nontrivial element of a free group and its inverse are distinct conjugacy
classes, but have the same attracting multiplier. Pairing their two factors
with exponent `-1/2` therefore gives exponent `-1`. The code's product over
unoriented primitive classes has exactly this multiplicity. Oscillator
products generate the repeated powers; nonprimitive graph walks must not
also be included as independent factors.

## Why the word cutoff is complete

At `q=0`, each theta crossing matrix has rank one. The constant trace of a
cyclic product is a product of pairings between successive punctures. Each
pairing vanishes only when the walk immediately returns through the same
puncture. A cyclically reduced theta walk has no such turn, so its constant
trace is `+1` or `-1`.

For a walk of length `l`, its determinant is the product of the `l` edge
determinants and has total plumbing degree `l`. Since the trace is a unit,

\[
t=\frac{\det M}{(\operatorname{tr}M)^2}
\]

has the same leading degree. If `k` is the small-to-large eigenvalue ratio,
then

\[
t=\frac{k}{(1+k)^2},\qquad
k=t(1+k)^2=\sum_{j\geq1}\frac1{j+1}\binom{2j}{j}t^j.
\]

Thus the multiplier also begins at degree `l`. The oscillator product starts
at `k^2`, so walks with `2*l>N` cannot contribute through total degree `N`.
Theta is bipartite and has only even closed-walk lengths. At `N=10`, lengths
two and four suffice. This is an exact degree argument, not a guessed
word-length cap. In particular the theta wrapper correctly returns one
below degree four.

For the generic graph helper, the same argument uses the sum of the positive
determinant orders along the walk instead of its unweighted length. Its
stated hypotheses are essential: inverse arcs must be inverse Möbius maps,
every edge determinant must have positive order, and every reduced cycle
must have nonzero constant trace in the supplied projective representatives.
The helper rejects nonpositive determinant order and a nonunit cycle trace;
it does not itself establish arbitrary supplied maps' inverse relation.

## Exact series operations

`inverse_unit` first factors out the nonzero constant and uses the finite
geometric expansion of the remaining positive-order series. Matrix products,
determinants, inverse traces, and multipliers are truncated only by the total
degree bound. The theta coefficients use rational arithmetic throughout.

For each unoriented primitive class the implemented logarithm is

\[
-\sum_{m\geq2}\log(1-k^m)
=\sum_{s\geq2}\frac1s
 \left(\sum_{\substack{m\mid s\\m\geq2}}m\right)k^s.
\]

This is the correct coefficient because a divisor `m` contributes the term
with logarithmic power `s/m`. The code advances multiplier powers in order
and uses this exact rational divisor sum. After adding the logarithms for
all classes, its exponential recurrence multiplies the previous power by
the logarithm and divides by the new index, yielding the terms `log^j/j!`.
All sums terminate because their inputs have strictly positive order. No
numeric smallness criterion or coefficient rounding determines support.

## Scope and conclusion

The graph traversal and series engine do not depend on genus two. Other
pants graphs can supply their own directed sewing maps and positive degree
costs; the present wrapper supplies only the authoritative theta
coordinates stated above. Older repository helpers with different edge
orders or local coordinates cannot be substituted without the corresponding
coordinate transformation.

The static mathematical and implementation review passes. This is a review
of the vacuum-factor construction and its integration, not a new numerical
comparison or an independently established level-ten physical runtime.
Those outcomes must be read from the final production and authorized
validation artifacts.
