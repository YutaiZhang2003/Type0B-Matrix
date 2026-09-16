# Convergence of the NSRR/all-NS momentum integrals

Status: **quadrature_stabilized**. Surface t=0.60, b=1.4, source L=5, target R=12.

The source meets the two-step convergence criterion at N=7 and is then held fixed while the target is refined further. The two quadrature orders are stated separately. The pairing, spin transport, three-point constants, and channel primary factors are unchanged. Agreement requires normalized ratio one.

| Source N | Target N | Ratio s=+ | Ratio s=- | Spin-sum ratio | Max target relative step |
|---:|---:|---:|---:|---:|---:|
| 3 | 3 | 0.2529257455 | 0.2528363803 | 0.2528792247 | — |
| 4 | 4 | 0.2523066185 | 0.2521344756 | 0.2522170258 | 1.153e-02 |
| 5 | 5 | 0.2507826956 | 0.2505624815 | 0.2506680841 | 6.814e-03 |
| 6 | 6 | 0.2502558009 | 0.2500232116 | 0.2501347467 | 2.148e-03 |
| 7 | 7 | 0.2501081704 | 0.2498714330 | 0.2499849562 | 5.994e-04 |
| 7 | 8 | 0.2500681878 | 0.2498296784 | 0.2499440509 | 1.671e-04 |
| 7 | 9 | 0.2500564176 | 0.2498172799 | 0.2499319535 | 4.963e-05 |
| 7 | 10 | 0.2500525946 | 0.2498132435 | 0.2499280195 | 1.616e-05 |

The stabilization criterion is two successive relative changes below 1e-4 (0.01%) for each integral, each sign, each resolved spin, and their sum. The last two source refinements give 9.121e-06, 9.178e-06.

Only complete grids enter the table. N means nodes per momentum direction; the target uses N^3 nodes. An unused partial source N8 grid is excluded. The unchanged target worker is reused, including its global-series convergence requirement and independent momentum-measure check.

See `target_refinement.json` and `target_refinement.csv` for both raw integrals, resolved spin ratios, and successive changes. `target_refinement_design.json` records the source stabilization and frozen inputs. The initial matched-grid comparison is in `README.md` and `summary.json`.

Reproduce the remaining refinement with `python Code/genus_2/continue_nsrr_target_quadrature.py --workers 6`. The worker program and original configuration remain unchanged.
