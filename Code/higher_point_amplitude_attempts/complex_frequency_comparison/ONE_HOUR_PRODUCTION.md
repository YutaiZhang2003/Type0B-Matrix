# One-hour preliminary five-point production

At the user's request, the long benchmark and its dependent production,
reduction and comparison jobs (43245958, 43245961, 43245965, 43251834) were
canceled. Their files and coefficient stores remain unchanged.

The replacement uses `Code/config/type0b_ns_five_tachyon_one_hour_preliminary.json`
and `Code/cluster/type0b_ns_fivepoint_one_hour.slurm`. Four workers, reduction
and the fixed-complex-frequency comparison share one four-CPU allocation
with a one-hour limit. There is no benchmark dependency or separate reducer
queue. Queue waiting is outside the computation time limit.

| Setting | Previous run | Preliminary run |
| --- | --- | --- |
| Outgoing energies | 0.25 + 0.02i each | unchanged |
| Incoming energy | 1 + 0.08i | unchanged |
| Edge twice-level cutoffs | (8,8), physical levels (4,4) | (4,4), physical levels (2,2) |
| Momentum orders | (6,7) | (2,3) |
| Momentum refinement shells | 4 | 1 |
| Momentum maximum | 2 | 2 |
| Collar radii | 0.01, 0.005, 0.0025 | 0.01 only |
| Total bulk / face samples per radius | 512 / 1024 | 32 / 64 |
| Independent RQMC replicates | 16 per worker requested | 2 per worker, 8 total |
| Collar / block convergence certificate | diagnostic | omitted |
| Arithmetic digits, block / structure | 45 / 22 | unchanged |

The c-recursion backend, all 120 oriented charts, all boundary faces and
corners, momentum singularity subtraction and analytic forest finite parts
are retained. Arithmetic precision stays unchanged because reducing the
quadrature and block depth is a more direct way to reduce work.

The driver now applies the configured seed stride so replicate seeds do not
overlap between workers. Omitted certificates are reported as `null`, not
as passed. The preliminary profile has a separate schema; the standard
profile still requires its original cutoff and diagnostic settings.

This is a coarse first estimate, not a converged amplitude. Its reported
QMC errors do not include block or momentum truncation errors. The matrix
prediction is evaluated only after worldsheet reduction, at the same complex
energies, without fitting or an epsilon-to-zero extrapolation. The one-hour
limit is a resource bound, not by itself evidence of completion; worker timing
and completion records provide that evidence.
