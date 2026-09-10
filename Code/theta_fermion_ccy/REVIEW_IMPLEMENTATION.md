# Preliminary independent review of the signed assembly

Scope: static mathematical and implementation review of `pipeline.py`,
`series_algebra.py`, `outer_branching.py`, and `middle_branching.py` against the
radial sewing definition and the conventions in `Human Notes/SCblock.tex`.
This is not the requested later adversarial review of the complete LaTeX notes.
No separate numerical component comparison was run.

## Signs and contractions

The production numerator uses two raw outer trilinear forms, the raw radial
matrix element of (Q\Theta\psi(1)), and one inverse primary norm on each of
the four internal edges. In particular both split Ramond norms occur. This is
the raw version of the normalized three-branching-coefficient formula and
avoids a choice of square roots of the middle Ramond norms.

The second original vertex keeps the same relative physical form label
(f). The insertion (\Theta\psi) is even, so its incoming and outgoing
Ramond parity labels coincide. The chosen third-edge parity
(gamma=f-2n_1-\alpha\pmod2) therefore gives the same allowed trilinear form
at both original vertices. Changing the second (f) to (1-f) would be an
error for this even insertion.

The intrinsic NS-primary parity (p) enters the spin monomial through
(eta_1^{2n_1+p}) and the total tensor reordering sign. It is correctly not
added to the relative (f)-selection condition. The raw outer trilinear forms
already include the intrinsic-primary signs of the human notes' factorized
three-point definition; another (f\)-dependent transport sign should not be
added at the middle sphere.

`theta_sign(index)` is exactly
((-1)^{\epsilon_1\epsilon_2+\epsilon_1\epsilon_3+\epsilon_2\epsilon_3}).
The additional ((-1)^{2n_1}) in `pipeline.py` is the auxiliary-first BPZ
convention retained from the original enlarged theta formula. `STAR_SIGN`
is the polarization of that same quadratic sign, hence equals the manuscript's
convolution cocycle. This review found no justified discrete-sign change.

The projector in `series_algebra.minus_projection` uses left multiplication
by the index-six monomial (eta_2\eta_3). Its target ideal is indeed the
minus ideal. The ground auxiliary coefficient is proportional to
(1-\eta_2\eta_3\), so division by twice its scalar coefficient is the correct
triangular inverse on that ideal. The implementation retains the residual
instead of silently projecting the numerator into the required sector.

## Split levels and the universal factor

The stored exponents are
((2\ell_1,\ell_{2,1},\ell_{2,2},2\ell_3)). Their sum divided by two is the
physical total degree after
(hat q_{2,1}=u\sqrt{q_2}),
(hat q_{2,2}=u^{-1}\sqrt{q_2}). The primary shift used by `pipeline.py`,

\[
(4n_1^2,\ 2n^2-\tfrac18,\ 2n'^2-\tfrac18,\ 4n_3^2-\tfrac14),
\]

is therefore correct. It permits unequal split exponents throughout division;
it does not build independence of (u) into the answer. That independence
must be established by the requested final comparison.

Each large-central-charge vacuum factor depends on the composite second
edge (\hat q_{2,1}\hat q_{2,2}\). The enlarged numerator has two such factors,
and the auxiliary Ising block has one. When both inputs are represented with
these factors removed, their quotient needs one ordinary theta vacuum factor
restored, exactly as in the final assembly. This statement presumes the
punctured CCY and Ising implementations use this same definition of their
reduced series; the individual implementations must establish that convention.

## Concrete precision defects found and corrected

The first integration failure reported by the root agent was a minus-ideal
residual of approximately (1.04\times10^{-8}) in a level-five coefficient.
That report alone does not identify its origin. Static inspection did identify
three avoidable precision losses in the outer coefficient path:

1. `OuterBranching` and the legacy `BranchingGrid` converted exact (b,P_i)
   inputs to binary64 before constructing multiprecision action coefficients.
2. The legacy grid intentionally evaluated its low-primary anchors in
   binary64, even when its Ward-system solve used arbitrary precision.
3. The legacy multiprecision Ward solver cast its returned coefficients back
   to Python `complex` after finishing refinement.

`outer_branching.py` now supplies a separate MP adaptation for `dps>=30`. It
converts rational parameters directly to MP, evaluates only the same allowed
low-primary anchors using MP change-of-basis solves, replaces the boundary
phase (exp(3\pi i/4)) by ((-1+i)/\sqrt2) at MP precision, and keeps the
returned Ward solution in MP. The anchors are restricted explicitly to NS
(0,\pm\tfrac12) and R (\pm\tfrac14,\pm\tfrac34). Thus this change does not
replace the branching recurrence with direct evaluation of high primaries.

The MP Ward solve uses the existing pivoted iterative-refinement routine and
checks its residual against **all** Ward and anchor equations. Failure of
that guard must be diagnosed; lowering the guard solely to obtain a physical
block would hide evidence. Syntax compilation passed. Whether these changes
resolve the reported residual remains to be determined by the user's permitted
end-to-end tests.
