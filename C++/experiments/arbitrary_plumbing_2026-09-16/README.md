# Validation of the arbitrary-plumbing notes

The deliverable is `Machine Notes/arbitrary_plumbing_superconformal_blocks.tex`.
This isolated experiment folder contains the directed checks supporting its
theta and glasses specializations. No draft file or production header was
changed. The formulas use generic weights, bilinear BPZ forms, and the explicit
transported enlarged-vertex convention stated in the note.

## Requirements and evidence

| Requested ingredient | Note labels | Evidence |
|---|---|---|
| Physical block on an ordered plumbing graph | `physical-states`, `graph-sign`, `physical-block`, `physical-vertices` | Independent physical PBW references in `theta_validate.cpp` and the glasses drivers; exact half-edge permutations in `graph_sign_checks.py`. |
| Correlator decomposition with fermion signs | `correlator`, `correlator-sign` | Algebraic graded-tensor derivation; all 64 holomorphic/antiholomorphic parity pairs in each of theta and glasses, plus 4096 pairs on a graph with external legs. |
| Auxiliary and enlarged block | `aux-states`, `aux-ground`, `ramond-vertex`, `ns-vertex`, `enlarged-block`, `fermion-block` | Direct auxiliary and enlarged PBW contractions in theta; independently sewn auxiliary tensors and physical PBW factors in glasses. Both are compared with the double-Virasoro numerator. |
| Explicit graph convolution | `transport`, `kernel`, `star`, `convolution` | Exact local phase separation and associative-kernel tests; coefficient comparisons in `theta_summary.json`, `glasses_results.json`, and `glasses_ns_assignments_results.json`. |
| Double-Virasoro reduction | `branches`, `weights`, `double-virasoro` | Independently generated branching/Virasoro numerators versus PBW; theta NSRR, theta all-NS, and glasses R/R, NS/R, NS/NS. |
| External-state expansion | `external-expansion`, `external-descendants` | `external_state_results.json`: all 21 tensor-PBW states at NS levels 0, 1/2, 1 and R levels 0, 1; 169 branch-Gram entries. |
| Arbitrary Ramond loops and selected sector | `loop-element`, `loop-centrality`, `physical-sector`, `transported-sectors`, `odd-cycle`, `constant-projectors` | Local odd-ground identity checked in 864 cases; exact loop algebra on theta, glasses and a three-vertex cycle; uninserted cancellation and sector identities in the numerical block suites. |
| Split-edge PBW and convolution | `split-map`, `theta`, `split-pbw`, `split-convolution` | Full independent middle-level powers in `theta_literal_split.hpp` and `glasses_full_split.cpp`, including nonzero fermion modes; explicit convolution comparisons. |
| Split-edge double-Virasoro reduction | `split-double-virasoro`, `middle-branching` | `theta_full_split_dv.hpp` and `glasses_full_split.cpp`; 8320 theta components and 1728 glasses components, including 960 glasses components with unequal split powers. |
| Recovery and diagonal cutoff | `zero-mode`, `diagonal-cutoff`, `recovery` | Diagonal versus zero-mode identity; recovered physical coefficients compared with independent PBW. Original branch labels and total segment powers are retained separately. |

The numerical evidence is low-level, as requested. The arbitrary-genus result
is the local-to-global contraction argument in the note, not a claim of running
every genus numerically. The correlator checks establish the sewing identity
and its signs, not a chosen SCFT's spectral sum or moduli integral. External
Ramond ground and descendant projections are checked locally; a full correlator
with external Ramond fields is not claimed to have been evaluated.

## Authoritative results and precision

- `theta_summary.json`: 136 passing comparisons at tolerance `1e-20`, using
  40-digit arithmetic, both NS primary parities, both form parities, all four
  vertex-sign pairs, and all eight spin components through physical total
  level 3. Full split powers have the explicitly documented cutoff.
- `theta_all_ns_results.json`: 64 exact local vertex comparisons, 384 exact
  Virasoro Ward identities, and an all-NS theta block through total level 2.
  The block-series backend is machine precision; only branching-polynomial
  evaluation uses 40 digits.
- `glasses_results.json`, `glasses_full_split_results.json`,
  `glasses_ns_assignments_results.json`: machine precision, individual level 1,
  even intrinsic primary parities, tolerance `1e-9`. The R/NS assignment is the
  handle-exchanged version of NS/R. At this cutoff each Virasoro edge has level
  at most one; these checks do not test higher CCY residues or nonconstant
  Schottky terms.
- `theta_ramond_local_results.json`, `external_state_results.json`, and
  `graph_sign_results.json`: local Ward, BPZ projection, and exact graph checks.

The physical PBW references do not use double-Virasoro branching coefficients.
The double-Virasoro numerators do not use the physical PBW answers. The full
theta split PBW numerator is directly contracted. The glasses numerator is
compared with the independent physical-PBW/free-fermion convolution. No fitted
result or already known physical answer is used to generate either numerator.

## Reproduction

See `theta_README.md` and `glasses_README.md` for commands, parameters, cutoffs,
normalizations and error measures. Run the exact finite graph checks with:

```sh
python3 C++/experiments/arbitrary_plumbing_2026-09-16/graph_sign_checks.py
```

Build and run the external-state check from this folder:

```sh
make -f external_state_makefile
./external_state external_state_results.json
```

The numerical result manifests record source hashes and commands. The final
completion audit matched the current sources to all 20 theta-manifest hashes,
22 glasses-manifest hashes, five external-projection hashes, and two local
Ramond-test hashes. Numerical source files were unchanged during the final
documentation pass, so the saved directed computations were not rerun merely
for prose or equation-layout changes.

## Adversarial review and resolutions

Independent sign-derivation, theta-validation, and glasses-validation reviewers
challenged both the formulas and the independence of the comparisons. They
cross-examined each other's sign conventions and coverage. The final review
resolved the following concrete issues:

1. A graph-permutation sign alone misses the mixed local term at a self-loop.
   The note includes the explicit graph kernel and physical parity transport.
2. A single vertex weight is not identified with a geometric spin lift. Its
   algebraic definition and compatibility with both Virasoro algebras are given.
3. The loop obstruction includes vertex-form and intrinsic NS parity factors;
   the bare product of vertex signs is only the theta specialization. Glasses
   with odd form parity supplies an explicit distinguishing example.
4. External Ramond ground tensors require a normalized sum over branches.
   External descendants require BPZ overlaps and inverse Virasoro Grams.
5. The split middle physical contraction uses two inverse Grams and one Gram.
   All contracted primed indices are now explicit.
6. The zero-mode sign counts the auxiliary ground parity. Equal total split
   powers do not mean equal branch labels or equal levels in each Virasoro copy.
7. Recovery is restricted to the matching loop sector. No unrestricted formal
   inverse is claimed.
8. The physical ground normalizations and the local odd-Ramond factor are now
   stated explicitly so the loop proof can be followed within the note.
9. The validation table distinguishes arithmetic precision, exact finite sign
   checks, physical-block checks, and their scope. A stale README numerator
   error bound was corrected to `7.02e-13`; the source results were unchanged.

The final reviewers found no remaining mathematical inconsistency within the
stated generic-weight scope after these corrections. The compiled note was
also inspected visually; the manuscript itself was left unchanged.
