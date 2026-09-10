# Level-10 numerator: support and implementation review

This is a static code and algebra review, plus exact enumeration of branch
labels. No block coefficients or numerical comparison were computed for this
report. The physical cutoff is the balanced split cutoff used by `pipeline.py`:

\[
4n_1^2+\left(2n_{2,1}^2-\tfrac18\right)
+\left(2n_{2,2}^2-\tfrac18\right)
+2\left(2n_3^2-\tfrac18\right)\leq20,
\qquad |n_{2,1}-n_{2,2}|=\tfrac12.
\]

## Exact support

The necessary NS labels are
\(0,\pm\tfrac12,\pm1,\pm\tfrac32,\pm2\).
The necessary Ramond labels on either split edge and on edge 3 are
\(\pm\tfrac14,\pm\tfrac34,\pm\tfrac54,\pm\tfrac74,\pm\tfrac94\),
whose levels are respectively \(0,1,3,6,10\).
There are 18 ordered middle pairs and 804 admissible primary quadruples.
Both alpha values share the same Virasoro blocks, so there are at most
1608 single-copy CCY series, not twice that number.

The largest required reusable actions are:

| Action | Largest absolute label | Descendant degree on the neighboring primary | Number of span columns |
| --- | ---: | ---: | ---: |
| Outer NS \(L_1\) | \(2\) | \(5\) | \(36\) |
| Outer Ramond \(L_{-1}\) | \(9/4\) | \(8\) | \(185+2=187\) |
| Middle Ramond \(L_1\) | \(9/4\) | \(6\) | \(65\) |
| Middle Ramond \(L_{-1}\) | \(7/4\) | \(6\) | \(65+2=67\) |

The outer code requests six nontrivial NS actions and 40 Ramond actions
(two momenta, two parities, ten signed labels). The middle recurrence needs
16 nontrivial plus actions and 16 minus actions. Its minus actions are a
subset of the outer edge-2 actions if evaluated at the same precision.
The ground middle pairs use the explicit anchors and require no action.
No action-degree cap in `compute_target.py` prevents these computations.
The largest outer Ramond action has total oscillator level 11, even though
its neighboring-primary descendant degree is only 8.

The full split descendant expansion still needs individual split levels up
to 20. Primary-label bounds and descendant-degree bounds must not be used
as a cap on those plumbing exponents.

## Smaller Ward tables

Only 464 distinct outer triples are queried by the 804 primary quadruples:
240 with even NS label parity and 224 with odd NS label parity. Close this
set under all three existing label moves, retaining the same-branch terms:

* NS: \(n\mapsto n-1\) for \(n\geq1\), and \(n\mapsto n+1\) for \(n\leq-1\).
* Ramond: \(n\mapsto n-1\) for \(n\geq3/4\), and \(n\mapsto n+1\) for \(n\leq-3/4\).
* Ramond ground boundary: \(1/4\mapsto-3/4\) and \(-1/4\mapsto3/4\).

This adds only 20 triples, giving 484 total: 260 even and 224 odd. The
current Cartesian grid instead uses 500 or 400 unknowns in each parity
system. A replacement should supply this action-closed list to the existing
Ward equation builder, retain the existing low anchors, and keep its full
column-rank and all-row residual guards. Action closure establishes that
no required row references an omitted coefficient; it is not a substitute
for the rank guard. This reduction does not reduce the action-label support
listed above.

The Ward matrix, including normalized anchor rows, is independent of eta.
For each `(alpha, gamma, f)` one can build and factor it once and solve the
two eta right-hand sides using the same factorization and independent
residual guards. The current `prepare` rebuilds and refactors each matrix
twice. The number of retained table entries becomes 1936 for both eta values
and all alpha/gamma choices, rather than 3600.

## Avoid high-level reflection conversion

The inherited `BranchingGrid.build_actions` calls
`solve_ramond_lminus(module, negative_n, parity)` in the native positive
chart. Consequently `r_branch` converts the negative primary through the
physical PBW basis. At level 10 that conversion can involve a 464 by 464
multiprecision matrix. Its transition matrices are cached, but its LU solve
is repeated for each auxiliary component and each repeated primary request.
This is avoidable in computing reusable action coefficients.

The existing `middle_branching.RamondActions.minus` already evaluates
negative labels at positive label and reflected momentum, then reverses all
output labels. This has exactly the same normalization as the inherited
conversion, by the following algebraic argument.

On the physical free-field basis multiply a vector by minus one precisely
when its ground label is odd. Comparing `(P, realization=+1)` with
`(-P, realization=-1)`, the product `realization * P` is unchanged. Only
the physical fermion zero-mode sign differs. This diagonal map conjugates
that zero mode and leaves nonzero fermion modes, bosons and auxiliary modes
unchanged. Inspection of the displayed free-field formulas therefore shows
that it intertwines physical \(L,G\) and both embedded Virasoro algebras.
It maps each positive chi string, including the optional opposite-chart
zero mode, to the raw negative chi string with no branch-dependent scalar.
Composing with the native-to-target PBW conversion gives precisely the
Human-Note reflection: the component ending in \(w^+\) is unchanged and
the component ending in \(w^-\) changes sign. The same intertwiner acts on
both sides of every descendant identity. Thus its coefficients are
unchanged after \((n,P)\mapsto(-n,-P)\), with all neighboring labels
reversed as well. This also covers the chart-crossing ground identities.

A minimal outer override can leave the NS action code unchanged, create
one `RamondActions` provider for each of `P2` and `P3`, and populate
`r_actions[(slot, parity, label)]` from its `minus(label, parity)`. Preserve
the returned fit diagnostics. Only the positive-ground identity
\(1/4\to-3/4\) then needs a reflected primary conversion, at level 1.
No primary normalization, eta convention, or scalar contraction changes.

## Other safe reuse

1. `ordinary_factor` currently constructs two new `VirasoroThreePoint`
   objects for every action term. Cache these evaluators by the changed
   label triple and Virasoro copy within one fixed weight/precision context;
   their descendant values are reusable across terms and eta systems.
2. Preserve action providers between outer and middle computations only
   when the same `(b, P, parity, arithmetic precision)` applies. In the
   current pipeline outer and middle precision can differ, and changing the
   shared MP context does not upgrade cached lower-precision numbers.
3. Cache primary expressions and descendant-prefix states inside an action
   provider if action-span assembly remains significant. These are reusable
   sparse state expressions, not new direct evaluations of the sought
   primary matrix elements. Avoid retaining all large span matrices after
   extracting their small action coefficient tables.
4. A persistent cache must record exact input parameters and precision.
   Existing `lru_cache` keys attached to module instances do not include the
   mutable global precision. Do not reuse a module after increasing that
   precision and assume its old cached states were recomputed.

The direct-auxiliary CLI integration now allows level 10; the level-5 cap
remains appropriate only for the optional corrected Ising CCY backend.
The complete runtime must include action construction, outer Ward solves,
middle recursion, both CCY factors, convolution/division and restoration of
the two universal vacuum factors. The standalone direct auxiliary timing
does not measure any of these numerator costs.
