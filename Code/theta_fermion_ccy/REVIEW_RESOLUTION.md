# Historical review resolution: bounded Ising backend

This record predates the user's selection of direct fermion sewing for
the production auxiliary factor. Its backend limitations and remaining-work
statements describe that earlier stage. See `COMPLETION_AUDIT.md`,
`FINAL_MATH_REVIEW.md`, and `FINAL_PRESENTATION_REVIEW.md` for the selected
algorithm and current evidence. In particular, division by the full direct
auxiliary requires restoring **two** universal vacuum factors.

The initial LaTeX writeup was reviewed independently by the CCY and Ising
agents. They challenged one another's sign, null-vector and seed arguments.
The middle agent supplied additional independent symbolic and presentation
reviews. The root revised the notes after these reviews.

## Mathematical corrections incorporated

- The human notes' literal tensor definition of the hatted three-point form
  misses the physical NS contour transport. The separate notes now explicitly
  correct that convention, retain the original enlarged NS sign, and show
  the cancellation of its physical part. Original human notes and draft were
  not edited. See `REVIEW_MATH_CCY.md` for the all-three-contour derivation.
- The production adapter transports the native odd-form label as
  `eta_native=(-1)^f eta_physical`. The saved numerical benchmark has `f=0`,
  so this subsequent correction does not change its computed coefficients.
- The physical metric contraction `G^-1 G G^-1=G^-1`, the linear NS sign,
  the quadratic parity cocycle and the all-order physical sector identity
  now appear explicitly in the writeup.
- The raw four-norm contraction, radial normalized middle coefficient and
  coherent norm-root phase are distinguished. Explicit ground primaries,
  norms and zero-mode actions derive the middle anchors.
- The large-c vacuum seed has its oscillator norm and all three Wick
  contractions specified. The text explains why one factor remains after
  dividing the two-copy numerator by the one-copy auxiliary block.
- The descendant-level-six action bound is restricted to the new middle
  recurrence. The current outer grid also needs an action through level eight
  for a future physical level-ten computation.
- The first-cycle Ising subtraction is stated as a bounded result through
  physical level five, with the mixed-null selection argument supplied.

## Presentation revisions incorporated

The exact auxiliary decomposition is followed immediately by the sector
inverse. The longer irreducible-Ising analysis follows the generic CCY
derivation. Physical parameter conventions, parity labels and generic-domain
restrictions are defined. The notes distinguish the nonzero ordinary
auxiliary block from its annihilation of the physical minus sector.
Measured accuracy and level-five runtime are reported separately from
arithmetic precision and the unfinished level-ten computation.

The final LaTeX compiles to 15 pages with no warnings or overfull boxes.
This is a syntax/layout-log check, not a numerical conformal-block comparison.

## Numerical evidence and unchanged limitations

Only the two requested end-to-end numerical comparisons were run. The saved
level-five physical series agrees with independent PBW with maximum scaled
coefficient error `7.04785e-16`, and fixed-product split variation is
`1.26621e-19`. Full level-five elapsed algorithm time was `99.3412 s`.
Production `IsingFermion` now exposes this bounded implementation and rejects
higher levels. Subsequent edits were the odd-form convention correction,
default precision settings, bounded API wrapper, diagnostics and documentation;
none changes the saved `f=0` calculation.

The full Ising quotient through level ten remains unresolved. A fixed-weight
generic Verma limit has an interacting-null pole at physical level 13/2.
An improved exact degenerate-field family avoids some of that behavior but
still retains secondary null cycles, including an isolated coefficient 3/8
at physical level 11/2. Finiteness of a regulated network alone does not
justify replacing its higher-order null vertices by ordinary shifted-primary
blocks: their Ward identities can have finite inhomogeneous terms.
No general impossibility theorem is claimed. No level-ten runtime is claimed,
and the overall implementation objective is not complete.
