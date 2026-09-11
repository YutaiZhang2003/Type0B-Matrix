# PBW comparison: adversarial audit

This review inspected sources and saved timing records, and executed only integer counting and fitting. It did not construct PBW states, Gram matrices, Ward entries, or branching coefficients.

## Claims challenged and resolution

1. **A fair 40-digit speedup cannot be quoted from the existing PBW logs.** The new C++ runs use 136 bits. `pbw_machine_level10.log` is native 53-bit complex arithmetic; `pbw_level10.log` is a 384-bit FLINT midpoint implementation. These are neither matched precision nor matched language/backend. The notes explicitly identify both arithmetic systems and decline to interpolate a 136-bit PBW time. The higher-precision table is a comparison proxy, not a 40-digit prediction.
2. **Equal- and opposite-sign PBW do not have identical Ward costs.** `Check.physical_coefficient` (`Code/ramond_zero_mode_recovery/check_level10_modular.py`, line 248) obtains forms through cached `human_form(p,f,eta)`. `GeneralizedNRRWard.value` (`Code/double_virasoro/nsrr/ramond_pbw_generalized_ward.py`, line 622) is also cached. Equal signs thus reuse the same form at the second vertex. However the code still reconstructs the array and phases, and contracts both tensors. The notes halve only a tensor-associated model component as a scenario, never the total runtime.
3. **`sum(d^3)` is not a measured inversion cost.** Per-monomial saved logs combine newly required Grams, inverses, three-point entries, contractions, and serialization. Their features are correlated. We therefore call `sum(d^3)` a work proxy, keep fitted stage components out of the timing table, and report fit residuals and an alternate model to expose extrapolation uncertainty.
4. **The earlier 18–40 minute machine level-15 estimate is not a confidence interval.** Its endpoints came from different feature models, not repeated runs or statistical sampling. The notes give the full operation-feature estimate first and discuss the lean model separately. The numerical spread is model sensitivity only; memory and internal Ward recursion can invalidate either model.
5. **Serialized output affects the old PBW timer.** `pbw_reference.py` rewrites the complete accumulated JSON after each monomial. The old internal timer includes those writes. C++'s internal timer excludes its final JSON write. The notes say to compare parent process wall times for end-to-end claims; the script retains both old internal and parent wall times.
6. **Level-15 memory growth matters.** Native Gram and inverse arrays alone rise from 14.4 MB at level 10 to 653.8 MB at level 15. This excludes Python object/cache memory and is not the memory estimate for FLINT. Counts and dimensions are exact; seconds are extrapolated under constant cost per feature.
7. **Source-hash provenance is not a proof of historical source identity.** The new estimate file hashes the currently inspected counting/source files and the saved logs. Some saved timing summaries also record historical source hashes. A current source hash documents the audit's inputs; it does not assert the old runs used today's source bit for bit. The math-loop structure relevant to counts was inspected directly, and both 506-row logs match its monomial order.

## Quantitative evidence

- Saved negative-sector level 10: 31.402591 s native internal / 33.550881 s full process; 110.189838 s FLINT384 internal / 112.828280 s full process.
- Exact level 15: 1,496 monomials; 11,968 parity output slots; largest NS/R-per-parity Gram dimensions 1,016/1,472; 20,431,839 Gram entries; 22,116,449,163 sum of dimension cubes; 9,895,892 requested form entries (two signs); 734,995,948 dense contraction multiplications.
- Full-feature level-15 fit: negative 2,373.1 s native or 15,144.3 s FLINT384; positive 2,047.4–2,373.1 s or 14,487.8–15,144.3 s, depending on cache-overhead scenario.
- Lean level-15 fit: negative 1,106.1 s native or 4,530.4 s FLINT384; positive 719.9–1,106.1 s or 3,673.5–4,530.4 s. This model omits the explicit cubic and contraction terms and is included as sensitivity evidence, not as an equally established asymptotic model.

## Initial-main-notes presentation review

The existing timing paragraphs correctly distinguish internal and parent wall timers and state that completion is not a numerical accuracy certificate. Preserve those sentences. The initial wording "possible reuse of identical vertices" is weaker than the source evidence: the second vertex uses the same cached form exactly for equal signs, though array construction is repeated. The include now makes that distinction explicit. Avoid one headline “C++ is X times faster than PBW” because precision, backend, historical scheduling, and estimate-versus-measurement differ. Show fresh measured values, saved values, and estimates in separately labeled columns/rows.

The notes should say the requested timing comparison covers the one explicit generic parameter point and both signs, not all momentum configurations. The count model has no pole-conditioning input, and the C++ runtime may also depend on numerical rank and refinement convergence. All numerical claims should use sensible significant figures: seconds to 2–3 significant digits and level-15 estimates rounded to minutes.

## Reproduction

`C++/results/current_algorithm_2026-09-11/estimate_pbw.py` imports only the repository's integer counting utility, NumPy, and SciPy. It derives both sectors from exact character dimensions, fits the two existing 506-row logs, and writes `pbw_estimates.json`. It asserts matching monomial order through the counting utility's calibration function. No numerical PBW backend is imported or executed.
