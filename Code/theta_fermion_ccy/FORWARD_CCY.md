# Forward CCY with equal-level product targets

The production pipeline now defaults to the forward recurrence and diagonal physical recovery. It is entirely Python. The previous recurrence and full four-variable output remain available with `--ccy-method backward --full-split`.

## Rearranging the recurrence

The existing CCY relation, with the universal large-c vacuum factor removed, is

\[
F_{\boldsymbol\ell}(c;\mathbf h)
=G_{\boldsymbol\ell}(\mathbf h)
+\sum_{e,r,s}\frac{R_{rs}^{(e)}(\mathbf h)}{c-c_{rs}(h_e)}
F_{\boldsymbol\ell-rs\mathbf e_e}
\bigl(c_{rs}(h_e);\mathbf h+rs\mathbf e_e\bigr).
\]

The residue, its denominator and the next central charge depend on the accumulated internal-weight shifts and the current central charge. They do not depend on the final descendant coefficient being requested. The previous implementation nevertheless traversed these transitions separately at each remaining-level vector.

The new implementation propagates path amplitudes once through the directed graph of **(accumulated weight shifts, last pole)**. Initially the amplitude at zero shifts and the original central charge is one. For each allowed null shift on an edge, multiply the current amplitude by the same CCY residue-over-pole-denominator and add it to the destination amplitude. Shift degree strictly increases, so this is a finite topological pass through the chosen truncation.

Keep the last-pole label during this propagation: different current central charges have different outgoing denominators. Only after propagation may amplitudes with the same shift be summed for multiplying a global seed.

If K denotes this summed weight for an accumulated shift, the final assembly is

\[
F_{\boldsymbol\ell}(c;\mathbf h)
=\sum_{\boldsymbol\sigma\le\boldsymbol\ell}
K_{\boldsymbol\sigma}(c;\mathbf h)\,
G_{\boldsymbol\ell-\boldsymbol\sigma}(\mathbf h+\boldsymbol\sigma).
\]

This identity follows by fully expanding the original finite recursion and grouping paths by their total shift and terminal global seed. It uses exactly the same residues and closed SL(2) seeds. No PBW block, new recursion identity, or approximation is used. The Schottky factor is still restored separately.

Each coordinate of an accumulated shift is zero or at least two. The possible (shift, remaining-level) splits of a requested coefficient depend only on integer levels and are cached across all branching terms. Global-seed values themselves are evaluated once per required split; there is no remaining-level pole-expansion cache in the forward pass.

## Equal middle powers

The equal-level condition belongs to the full physical block and to the product target, not separately to each Virasoro factor. For each branching tuple, compute the downward closure

\[
a+d+\max\left(2n_2^2-\tfrac18+b,\ 2(n_2')^2-\tfrac18+c\right)
\le N-2n_1^2-\left(2n_3^2-\tfrac18\right).
\]

The forward propagation uses this same domain for accumulated shifts. Individual Virasoro inputs retain the required unequal powers. Products are assembled only at targets with equal middle powers after adding the branching shift.

Since the physical block depends on the two split parameters through their product, its convolution closes on equal middle powers. The direct auxiliary provider therefore uses the zero-mode contribution only, and the division iterates over physical level triples. The existing parity algebra and constant inverse on the minus ideal are unchanged. The code still records numerical departures from that ideal; it does not project the parity coefficients.

## Implementation

- `forward_ccy.py`: forward null-shift weights and final global-seed sums. Uses the original `PuncturedCCY` residue, fusion, A/derivative, norm and global-seed routines. A single-coefficient call inherited from the base class remains available; the pipeline uses the forward `reduced_series` method.
- `series_algebra.py`: downward-closed Virasoro budgets, direct target products, and equal-power convolution iteration.
- `direct_fermion.py`: optional `diagonal_only` mode. Nonzero insertion modes create unequal middle levels and are not evaluated in this mode.
- `pipeline.py`: selects the forward engine and equal-power path by default, retaining within-run branching reuse and Schottky restoration.

All pole and norm guards are retained. A summed amplitude of zero is not used to discard outgoing transitions with different last-pole labels. The algorithm applies to downward-closed coefficient sets; the two-budget set is the specialization to this physical target.

## Validation and timing

Four symmetric/asymmetric ordinary-block cases, both Virasoro copies, agree with the original recurrence at 40 digits to a maximum scaled difference of approximately 4.74e-37. The complete level-five physical calculation at 40 digits differs from the saved full-split calculation by at most 5.32e-25. Repeating diagonal recovery on the old numerator shows almost all of this difference comes from the old numerical off-diagonal residuals: the new result differs from diagonal recovery of the old inputs by only 4.54e-28. The diagonal auxiliary coefficients match exactly.

Machine arithmetic is sensitive to summation order. Across all 168,616 level-ten ordinary coefficients, the maximum backward/forward scaled difference is 5.12e-7. The 229,360 level-fifteen sampled ordinary coefficients have a maximum scaled difference of 1.51e-3. Targeted 40-digit evaluation of the largest differences shows that either order can be closer; the faster method is not an accuracy guarantee. No new PBW run was performed.

The new full level-ten machine pipeline completed from scratch in **16.9148 seconds**, including all imports and output writes. CCY took 4.5632 seconds, outer branching 10.4596 seconds, middle branching 1.0260 seconds, and Virasoro products 0.1909 seconds. All intermediate branching coefficients and the physical result were stored. The 4,048 physical coefficient slots differ from the former full-split machine result by a maximum absolute 0.01465 and maximum scaled 0.000971; the latter compares to max(1,abs(old coefficient)). These are differences between numerical calculations, not errors against an exact reference.

| Main multiply-add workload | Level 10 | Level 15 |
| --- | ---: | ---: |
| Previous recursion, diagonal-target domain: pole additions | 17,977,704 | 1,608,644,832 |
| Forward residue-weight propagation | 1,221,008 | 41,107,800 |
| Forward global-seed assembly | 2,332,072 | 57,031,560 |

These counts omit arithmetic inside the global/residue formulas and simple amplitude merges. The main workload decreases by factors about 5.1 and 16.4 respectively, while measured CCY-stage speedups are smaller because those other costs remain.

The complete level-ten series benchmark decreased from 8.58 to 4.33 seconds. Repeating the 212-series level-fifteen sample (both copies for each of 106 budgets) decreases the weighted CCY-stage estimate from 391.61 to **85.36 seconds**. The same approximately 305-second branching estimate gives a full level-fifteen working estimate of **about 7 minutes**, not a completed full run.

## Reproduction

```sh
/private/tmp/theta_fermion_ccy_env/bin/python Code/theta_fermion_ccy/verify_forward_recurrence.py --json Code/theta_fermion_ccy/results/validation_forward_ccy_40digits_reproducible.json
/private/tmp/theta_fermion_ccy_env/bin/python Code/theta_fermion_ccy/run_timed_pipeline.py --timing-json Code/theta_fermion_ccy/results/ccy_forward_diagonal_machine_level10_walltime.json --level 10 --machine --auxiliary direct --sector-policy record --json Code/theta_fermion_ccy/results/ccy_forward_diagonal_machine_level10.json
```

Saved evidence is in `results/timing_forward_ccy_level10.json`, `results/timing_forward_ccy_level15_sample.json`, `results/validation_forward_ccy_worst_coefficients_40digits.json`, `results/validation_forward_diagonal_pipeline_40digits_level5.json`, and `results/comparison_forward_diagonal_machine_level10.json`. Ordinary series, sampled action coefficients, and full-pipeline branching coefficients are also retained in the result directory.
