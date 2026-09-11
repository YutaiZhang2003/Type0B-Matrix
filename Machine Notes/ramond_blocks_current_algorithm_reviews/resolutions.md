# Resolution of the adversarial reviews

The review concerns the new ramond_blocks_current_algorithm.tex and the current native C++ implementation. Production source, the manuscript, and earlier notes were not changed for this task. The separate branching and CCY reviewers challenged each other's normalization, sector, and truncation arguments; a third reviewer audited the PBW count model and saved timing evidence.

## Mathematical revisions

- Recorded the actual C++ NS primary order and its conversion phase to the reversed human-note order. The phase cancels in physical sewing, but cannot be ignored in raw action and branching conventions.
- Separated the componentwise two-intertwiner sign from its graded tensor sign in the sector proof.
- Supplied physical ground three-point values, both ground metrics, zero-mode actions, internal physical ground coordinates, and their contour transport.
- Supplied the oscillator algebra and the physical free-field realization used to build action-system columns.
- Restricted the Ramond positive-mode expansion and middle recurrence to their actual domains, keeping the ground pair as an anchor.
- Distinguished the ordinary inverse-Fock-metric cancellation from the extra middle-cut cancellation in inserted sewing.
- Explained why the physical identity contraction permits diagonal recovery, while unequal middle levels inside the individual Virasoro factors remain necessary.
- Added the exact maximum-level criterion proving the two-inequality truncation domain.
- Explained stable Kac-root coverage, fixed the edge/null summation notation, and identified the inserted field's degenerate Kac labels without colliding with edge/branch weight notation.
- Retained the distinction between assumed structural action supports, finite numerical consistency checks, and a proof or certified accuracy bound.

Both mathematical reviewers checked the revised text against the production source in a second pass. Their reports record no remaining blocking mathematical inconsistency in the arguments they reviewed; this is not a claim of certified high-level coefficient accuracy.

## Presentation revisions and debate

The reviewers independently requested an upfront total-level definition and an execution-order overview. Both were added. After discussing whether to reorder the full notes, they agreed that a preview of diagonal recovery plus local definitions gave the clearest account without duplicating the derivation.

The final text also identifies the physical Gram matrix, states immediately after the inserted sewing formula what production actually computes, avoids reusing the coupling symbol for a supercurrent count, and locates every numerical cache by its lifetime. It distinguishes the candidate branching support from the stricter diagonal assembly domain.

## Timing and PBW interpretation

The timing wrapper launches each requested C++ level/sector in a new process, sequentially, without numerical input caches. Internal reuse is enabled. It saves full coefficient arrays, component timers, parent process wall times, diagnostics, parameters, and source/binary hashes. The timing-table renderer reads those saved measurements and checks that the production source hashes still match; it does not rerun a block.

The PBW reviewer executed only integer workload counts and fits to saved logs. There were no new PBW, Gram-matrix, or Ward-block calculations. The notes distinguish native 53-bit and FLINT 384-bit historical calibration from the fresh 136-bit C++ runs; they make no matched-40-digit speedup claim. Equal-sign estimates discount only the model component that can benefit from reusing the same cached vertex form. Alternative cost models and their substantial level-15 spread are disclosed before and after the main estimate table.

The individual reports are branching.md, ccy.md, and pbw.md. The exact time observations and reproducible estimate inputs are under C++/results/current_algorithm_2026-09-11/.

## Final saved-result audit

Both level-15 runs completed, as did both fresh level-10 runs. Full-process wall times were 38.352 s and 110.415 s at level 10, and 471.398 s and 693.836 s at level 15, for positive and negative sectors respectively. Source and executable hashes remained unchanged. The CCY reviewer independently checked the generated timing, count, and residual tables against both summary and individual result files.

That final audit identified one diagnostic-label refinement: the parity residual is measured on the recovered coefficients before restoring the vacuum square. The notes now identify that stage explicitly and do not present it as a recomputed residual of the final physical arrays.
