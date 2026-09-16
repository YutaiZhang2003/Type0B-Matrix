# NSRR/all-NS momentum quadrature refinement

This file records the initial matched-grid stage through N=7. The completed
independent refinement, with source N7 and target N10, is in
[FINAL_REPORT.md](FINAL_REPORT.md). Both integrals and the full complex
matrices meet the two-step 0.01% criterion. The partial source N8 grid is
unused and excluded from every result.

Status: **refinement_in_progress**. Both channels use the same number N of quadrature nodes per direction.

The pairing, spin transport, BRY constants, and primary factors are unchanged from the preceding cross-channel test. Source total descendant cutoff L=5 and target recursion twice-level R=12 are fixed. The central surface is t=0.60 and b=1.4. Agreement requires normalized ratio one.

| N | Nodes/channel | Source spin sum | Target spin sum | Ratio s=+ | Ratio s=- | Spin-sum ratio | Max relative step |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 3 | 27 | 3.981886076202e-10 | 1.287925734756e-09 | 0.2529257455 | 0.2528363803 | 0.2528792247 | — |
| 4 | 64 | 4.016308467205e-10 | 1.302470221297e-09 | 0.2523066185 | 0.2521344756 | 0.2522170258 | 1.153e-02 |
| 5 | 125 | 4.018835780568e-10 | 1.311343175668e-09 | 0.2507826956 | 0.2505624815 | 0.2506680841 | 6.814e-03 |
| 6 | 216 | 4.018832588080e-10 | 1.314138177988e-09 | 0.2502558009 | 0.2500232116 | 0.2501347467 | 2.152e-03 |
| 7 | 343 | 4.018798024166e-10 | 1.314914298254e-09 | 0.2501081704 | 0.2498714330 | 0.2499849562 | 6.071e-04 |

The maximum step checks **each channel integral and its ratio**, for s=±1, each resolved spin, and their sum. Stabilization requires two successive steps below 1.0e-04 relative. This is not a rigorous error bound.

The existing N3 source and N3/N4 target are reused with verified input hashes. All new source nodes retain complex projected blocks from the current 40-digit native engine, at both L3 and L5. All new target nodes retain four complex lift blocks in each form sector. Each node independently checks the momentum measure. Only complete grids enter this table.

The target global series retains its 2e-9 tolerance; its allowed occupation ceiling is increased from 36 to 60 to accommodate larger tail momenta. The recursion order remains R12. No unconverged global sum is accepted.

See `convergence.csv` for separate signs and resolved spins; `config.json` for frozen conventions and implementation hashes; `source/`, `target/`, and `provenance-N*.json` for complete node data.

Reproduce with `python Code/genus_2/refine_nsrr_bilinear_quadrature.py --workers 6`.
