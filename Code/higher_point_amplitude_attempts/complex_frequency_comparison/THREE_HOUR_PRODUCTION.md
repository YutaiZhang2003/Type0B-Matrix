# Three-hour accuracy refinement

The target remains 10% total numerical accuracy for the reduced all-tachyon
1-to-4 amplitude at outgoing energies `0.25+0.02i` and incoming energy
`1+0.08i`. No delta function is evaluated at complex energies and no
epsilon-to-zero limit is taken. The matrix prediction is used only after
the worldsheet computation, never to choose weights or corrections.

The campaign manifest is
`Code/config/type0b_ns_five_tachyon_three_hour_campaign.json` and the Slurm
script is `Code/cluster/type0b_ns_fivepoint_three_hour.slurm`.
One allocation uses four CPUs, 8 GiB, and at most three hours including
reduction, comparisons, and cleanup. The larger time budget supersedes the
earlier one-hour limit for this campaign.

## Primary estimate

* 16 independent randomized Sobol replicates, four per worker.
* 4096 bulk and 4096 face evaluations per replicate: **65536 of each per
  radius**, at both rho=0.01 and rho=0.005. These are respectively 128 and 64
  times the completed v7 counts. The two radii share random numbers.
* c-recursion only, twice-levels (4,4), total 8; momentum orders (2,3),
  Pmax=2, one threshold refinement shell. The primary estimate keeps these
  controls fixed so the improvement in sampling can be identified.
* Coefficients retain 45-digit construction precision; structure constants
  retain 22 digits. Tensor evaluation and its existing scalar fallback are
  unchanged.
* Nested sample prefixes, starting at 32 evaluations per component per
  replicate, report both estimates and paired changes as sampling grows.

The face sampler divides the exact four-point cell into four radial bands:
(0,rho), (rho,4rho), (4rho,1/2), and (1/2,1). The inner band samples squared
radius uniformly; the others sample log radius uniformly. The polar angular
bounds and area Jacobians are exact. All four bands receive equal numbers
of points, with the corresponding weights. No region is discarded.
At very small tangential radii, the auxiliary projection radius is reduced
to remain inside the normal collar; the same radius is used for the face
and its corner counterterm. This finite projection is a separate numerical
approximation, so a matched projection-radius check is included.

A local pilot at the old sample count reduced the combined real/imaginary
face standard error from about 157 to 80 at rho=.01 and from 1016 to 219 at
rho=.005, compared with the completed Cannon v7 run. This is not a matched
hardware timing benchmark or a convergence certificate. One replicate still
has a large face weight. An angular-pairing probe did not consistently
improve the largest bulk weights, so production retains the original bulk
proposal and increases its sample count instead.

## Separate numerical controls

After saving the primary result, the same allocation attempts four smaller
phases, each using all 16 replicate seeds and 32 bulk/32 face evaluations
per replicate at rho=.01. Each is compared with the identical primary
sample prefix, including the change in the deterministic corner.

| Phase | Only numerical control changed |
| --- | --- |
| projection_radius | normal projection radius from 1e-5 to 1e-6, with the same tangential safety cap |
| block_depth | twice-levels (6,6), total 12: physical level 3 per edge |
| momentum_order | momentum quadrature orders (3,4) |
| momentum_tail | Pmax from 2 to 3 |

These coarse paired differences are diagnostics, not corrections silently
added to the main amplitude. A small statistical error alone cannot certify
10% total accuracy. Large or unresolved control shifts will be reported.
Control phases require sufficient time remaining to start; a missing phase
is explicitly marked incomplete. The primary result is saved before any
control phase begins, so it remains usable if a later phase reaches the
time limit.

## Runtime and recovery

Completed coefficient stores are copied from v7 without changing that run.
Each worker stages its own store on node-local disk and copies committed
tables back atomically. Sample checkpoints use an append-only journal on
persistent storage instead of rewriting the whole sample history each time.
The journal fsyncs every 16 new samples and on normal or handled exceptional
exit. A hard kill/node loss can require recomputing at most 16 unsynced
samples. Truncated last records are recovered; complete-record corruption,
wrong signatures, and conflicting writers fail closed.

Slurm gives a two-minute warning. The campaign stops workers gracefully so
journals flush before coefficient copy-back. No old sample checkpoints are
relabelled or merged into this changed sampler. `campaign_status.json`,
`summary.json`, `comparison.json`, and `refinement_comparison.json` identify
completed work and remaining uncertainty separately.
