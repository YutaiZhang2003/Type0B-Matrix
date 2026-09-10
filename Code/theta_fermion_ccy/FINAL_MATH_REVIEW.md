# Final mathematical review of the selected fermion-insertion algorithm

This review concerns the currently selected implementation: the enlarged
four-edge block uses two ordinary Virasoro blocks computed by CCY
central-charge recursion, and the auxiliary block uses its exact fermion
definition. It supersedes the obsolete overall-incompleteness conclusion
in `REVIEW_MATH_CCY.md`, which concerned a proposed Ising-recursion backend.
That backend is not required by the selected algorithm.

Reviewed sources: `Machine Notes/theta_fermion_ramond_recovery.tex`,
`theta_fermion_direct_definition.tex`, `theta_fermion_ising_component.tex`,
and the production `pipeline.py`, `direct_fermion.py`, `series_algebra.py`,
`outer_branching.py`, and CCY implementation. The review used definitions,
algebra, and source inspection. No additional numerical checks were run.

## Conclusion and scope

I found no remaining mathematical contradiction in the stated theta-channel
construction at generic parameters. The domain hypotheses, the corrected
contour transport, the ordinary-block reduction, the direct auxiliary sum,
and the sector division now agree. The arbitrary-genus discussion correctly
claims that the local auxiliary sewing ingredients extend to other pants
graphs. It does not establish a complete arbitrary-graph sector inverse.

This is a mathematical/source review, not a proof that every numerical
branch is well conditioned, an independent level-ten validation, or a
runtime result. The two authorized level-five end-to-end checks are
reported separately by the parent. The physical level-ten calculation was
still running when this review was completed.

## Attempts to falsify the construction

### The universal factor must be restored twice

Each CCY result used in the branch sum is the ordinary block divided by
the same universal large-central-charge factor. Consequently the assembled
numerator is the enlarged block divided by the square of that factor.
The direct auxiliary is the full fermion block. Dividing these two inputs
therefore gives the physical block divided by the universal factor
squared. Restoring one copy would be an error. `pipeline.py` restores two
copies for the direct backend, in agreement with both prose derivations.

The universal factor is independent of the branch weights and the external
light weight. Forgetting the added external puncture reduces its graph to
the original theta graph; the sewing product is exactly
`q2 = qL*qR`. No non-Mobius coordinate change or Schwarzian contribution is
introduced. The scalar factor has trivial parity, so taking it outside
the convolution is legitimate. Its lifted support `(2*a,b,b,2*c)` agrees
with this pullback.

### The truncation does not discard information required by division

The root exponent is `(a,l,r,d)` with balanced degree
`(a+l+r+d)/2`. All exponents are nonnegative and the cutoff is downward
closed. Every positive-degree convolution term therefore uses a previously
available coefficient. The four internal primary shifts in `pipeline.py`
are exactly `(4*n1^2, 2*n^2-1/8, 2*nprime^2-1/8, 4*n3^2-1/4)`;
the remaining descendant degree has weights `(2,1,1,2)`. These weights
are also used when multiplying the two reduced Virasoro series.

Neither assembly nor division imposes `l=r`. All split-edge powers remain
available for the requested fixed-product test. The quotient routine
tests membership in the minus ideal but retains the computed coefficients;
it does not project away a failing component.

### The direct auxiliary retains the full field

The diagonal term is the zero-mode action. Every off-diagonal term changes
one occupied positive Ramond mode and reverses the ground-state bit, so
the total Ramond parity is preserved. Creation and removal have the same
action coefficient in the stated ordered basis. The implementation
enumerates each unordered pair once by creation and explicitly supplies
both orientations, with transposed split exponents. It neither substitutes
the zero mode for the field nor assumes equal split levels.

The four inverse Fock norms combine with the outgoing norm already
contained in a raw middle matrix element. Using the action coefficient
instead leaves precisely the sign shown in the direct-sum formula. The
rational Ramond rescaling is undone explicitly: the diagonal term has
the denominator `2^(b+c)`, whereas the toggle term has `2^c` after the
external `sqrt(2)` normalization is accounted for. The code matches both
cases. The state enumeration has exactly the balanced cutoff, including
split-edge states of individual Ramond level as high as `2*N`.

The Ward recursion uses state-dependent finite mode support. There is no
fixed mode cutoff inherited from an earlier implementation. Its recursive
terms either annihilate an occupied mode or create a smaller mode than the
removed target, so the recursion lowers nonzero-mode level.

### The enlargement still reduces to ordinary Virasoro blocks

The outgoing Ramond intertwiner commutes with both embedded Virasoro
algebras. The combined field therefore has the ordinary primary Ward
identities, and its scalar primary matrix element fixes all descendant
matrix elements. The compatible double-degenerate fusion rules leave
`nprime=n+1/2` or `nprime=n-1/2`. The middle scalar can be computed by the
stated reusable action coefficients; it is not a substitute for computing
the full descendants by CCY.

The implementation uses raw middle coefficients and all four internal
primary norm factors, avoiding independent square-root choices. The
normalized branching formula is consistent provided the correlated norm
roots described in the notes are used. The external weight remains fixed
at recursive central-charge poles, as required for the ordinary block.

### The sector inverse follows from definitions

The physical Ramond ground flip, extended with the supercurrent-counting
sign, is a BPZ isometry and a graded SCA intertwiner. Applying it at both
Ramond slots gives the displayed sector identity to every level; this does
not depend on the level-five comparison. The corrected native-to-human
outer-vertex transport supplies the otherwise missing physical NS sign
in the tensor-product sewing. The remaining sign is exactly the stated
theta convolution cocycle.

The auxiliary constant is `Q/sqrt(2)*(1-eta2*eta3)`, which acts by the
nonzero scalar `sqrt(2)*Q` in the required minus ideal. The notes now state
`b != 0`, `b^2 != 1`, and `Q != 0` near the first parameter assumptions.
Thus the chosen insertion and division are not silently used at `b=+/-i`,
where this normalization vanishes. Degenerate momentum limits are also
distinguished from direct evaluation of separately singular summands.

### Reflection and the action-level bounds are now consistent

Negative Ramond labels are obtained by simultaneous momentum/label
reflection and reversal of the labels in the positive-chart action
expansion. The physical ground map is an isometry and maps the defining
fermion strings without an extra normalization factor. This agrees with
the revised outer action construction.

The notes distinguish the maximum descendant level six needed by the new
middle recurrence at total level ten from the level-eight action data
constructed by the current outer Ward grid. The former is not presented
as a bound on all preparatory work.

## Generalization boundary

Fock bases, diagonal pairings, fermion Ward forms, and sparse insertions
are local data and can be sewn on an arbitrary compatible pants graph.
Other graphs need their own spin transports and graded sewing signs;
the theta cocycle cannot be copied unchanged. A full recovery procedure
on another graph additionally needs the appropriate insertions on its
Ramond components and a proof of the resulting sector identity and
invertible constant there. The current notes make the narrower,
established auxiliary-sewing claim, and the current implementation and
runtime concern the specified theta graph only.

This conclusion was discussed with the middle-branching reviewer, who
independently checked the raw/normalized middle relations, reflected outer
normalization, and the same generalization boundary.
