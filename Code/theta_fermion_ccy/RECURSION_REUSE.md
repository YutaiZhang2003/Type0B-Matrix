# Reusing CCY recursion bookkeeping

The optimization preserves the existing CCY recurrence and its term order. The universal Schottky factor, branch-dependent truncation, branching coefficients, direct auxiliary fermion factor and convolution recovery are unchanged.

## Reuse across branching labels

For the benchmark b=7/5 and the fixed three momenta, each Virasoro copy has 402 computed base-weight tuples that are distinct modulo integer shifts. Since c-recursion shifts internal weights by integers, these base tuples cannot yield identical complete shifted blocks across different n branches. The exact rational inspection is saved in results/recursion_reuse_weight_overlap.json. Exchange of the two split edges still reuses complete products as before.

The level-lowering rules do not depend on weights, c or precision. `_level_steps` constructs the original ordered list of (edge,r,s,lower_levels,terminal_flag) once for each level tuple, instead of allocating a new lower-level tuple and evaluating max(lower) on every recursive visit. The terminal flag is the existing condition that every lower level is zero or one, hence no r>=2 residue is possible.

## Integer keys

A shared table assigns an integer ID to each exact four-integer vector. Zero has ID zero; distinct vectors have distinct IDs. Both shifts and remaining levels use the same bijective table. Coefficient and global caches use IDs, and transition-cache misses decode the shift vector before calling the unchanged pole/residue formulas. The returned shifted vector is then interned. No floating-point weight or central charge is rounded or merged.

The indexed level-lowering lists are also shared across branches and Virasoro copies. Only integer combinatorial data is global. Every numerical cache remains local to its engine and is cleared at the existing branch boundary. The integer table must remain alive while any engine exists, so IDs are never recycled by clear_caches().

## Transition retention

The native transition cache is enlarged from 32,768 to 131,072 entries. For the measured maximum-cutoff branch, this reduces transition misses from 230,786 to 113,742. The multiprecision capacity remains 32,768. Eviction only recomputes an unchanged transition; neither capacity affects numerical arithmetic or cutoff.

## Verification and timing

Syntax compilation passed. The maximum-cutoff first-copy branch (0,-1/4,1/4,-1/4) with 3,146 output coefficients agreed exactly with the archived implementation after the level-step and integer-key changes. Its paired timing fell from 2.319040125 s to 1.323774875 s. Increasing transition retention then took 1.128556208 s in a separate timing-only experiment. These individual timings are not full-pipeline measurements.

The complete fresh-process level-ten timing, coefficient comparison and final segment measurements are documented in Machine Notes/level10_recursion_reuse_timing.md after the full run. The previous production sources are preserved under profiling_baseline/pre_recursion_steps with hashes.
