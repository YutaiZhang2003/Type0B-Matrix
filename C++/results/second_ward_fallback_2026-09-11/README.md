# Second conformal Ward identity for outer branching

The first outer Ward identity is supplemented by the second identity in
Human Notes/SCblock.tex, with the physical L1 operator on the Ramond state
in the second slot. In the notation of those notes,

\[
\hat\rho(v_1,L_1v_2,v_3)
=\hat\rho(L_{-1}v_1,v_2,v_3)-\hat\rho(v_1,L_{-1}v_2,v_3)
-2\hat\rho(v_1,L_0v_2,v_3)-\hat\rho(v_1,v_2,L_1v_3).
\]

The implementation tests the full coefficient of the unknown in the first
equation. A coefficient below \(10^{-11}\) of the row norm is treated as an
unusable numerical pivot; this includes exact zeros but is not a test of
exact algebraic vanishing. For these rows, it adds the second equation.
If the matrix still fails the existing numerical rank check, it adds further
second equations in increasing order of required descendant degree.
All original equations remain in the final residual check.

NS L-1 and Ramond L0/L1 decompositions are computed only when needed and cached
across triples, Ramond parities, and vertex signs. The physical L0 acts with
the physical oscillator level; it is not replaced by the enlarged conformal
weight. The reflected NS half-primary is constructed in the same physical
module as G(-1/2)v0 + (Q/2-P)psi(-1/2)v0.

## Directed validation

Run from the repository root:

    make -C C++ all check-branching

Only mode-action and outer-branching tests are run. No CCY blocks, physical
PBW blocks, or full level-20 pipelines are computed.

- Additional NS L-1 actions at |n| = 1/2, 1, 3/2 and Ramond L0/L1 actions at
  |n| = 3/4, 5/4, 7/4 pass free-field reconstruction checks at 40 digits.
  Both momentum charts are included; maximum relative residual is 3.43e-26.
- At the default parameters through total level 5, 608 second-identity checks
  against coefficients determined by the first identity have maximum normalized
  residual 4.15e-26. These include both Ramond parities and both vertex signs.
- Setting P3=P2 explicitly produces the vanishing V-difference. The solver adds
  32 second equations. The same 608 checks have maximum normalized residual
  3.15e-25 at 40 digits and 1.02e-11 at machine precision.

For the explicit identity checks, the normalization is
abs(sum of terms)/(1 + sum of absolute values of the grouped terms).
The linear solver separately checks all matrix rows using its usual tolerance.
These are residual checks, not claims of 40 accurate output digits.

## Original level-20 failure

The diagnostic rerun with the first identity alone completed the mode actions
in 148.566 seconds and failed in the alpha=gamma=0 outer Ward system:
572 rows, 556 columns, numerical rank 550, smallest singular value 4.71485e-15.
This localizes the failure to the outer Ward system, rather than the
mode-action decompositions. See level20_first_identity_failure.log.

## Level-20 result with the second identity

The branching-only test at 40 digits completed the original action preparation
in 150.749 seconds, but the augmented alpha=gamma=0 system still failed the
unchanged double-precision rank guard: 1112 rows, 556 columns, numerical rank
551, smallest singular value 2.9421e-14. This matrix contains all 540 second
identities for unanchored unknowns, as well as the first identities and anchors.
See level20_second_identity_failure.log.

Thus the second identity is implemented and validated at explicit vanishing
pivots, but it has not resolved the level-20 numerical failure. The reported
numerical ranks do not establish an exact algebraic rank deficiency or a
physical null state. No level-20 branching solution or block was produced.
Both full pipelines remain stopped. No rank tolerance was relaxed.
