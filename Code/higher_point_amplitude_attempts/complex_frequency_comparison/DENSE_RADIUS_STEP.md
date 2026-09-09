# Denser sampling and smaller-radius comparison toward 10%

The target is 10% numerical accuracy for the analytically continued reduced
all-tachyon amplitude, at the unchanged complex energies. There is no delta
function evaluated at complex arguments. Coupling normalization remains
`A_T=i I5/pi^2` after removing `mu_F^-3`.

The next configuration is
`Code/config/type0b_ns_five_tachyon_dense_paired_radius.json`.
It uses 512 bulk and 1024 face samples per radius (16 times the baseline),
with the same eight independent replicate seeds, and compares rho=0.01
against the requested smaller rho=0.005. Both radii use the same Sobol points.
Block depth, momentum quadrature, projection radius and arithmetic precision
are unchanged so sampling and radius effects can be identified separately.

A deterministic coarse-quadrature probe found

| rho | magnitude of the complete corner finite part |
| --- | ---: |
| 0.01 | 309.435674 |
| 0.005 | 1399.480976 |
| 0.0025 | 6806.184948 |

The probe is recorded in
`Data Set/type0b_ns_fivepoint_corner_radius_probe_20260831.json`.
It uses central-charge shift zero, exactly as the production config. These
are individual subtraction terms, not complete amplitudes or a convergence
certificate. Shrinking the collar does not guarantee smaller subtraction
terms: analytically continued radial finite parts scale as
`rho^(beta+2)/(beta+2)` and can grow when `Re(beta+2)<0`. The full sum and its
paired-radius difference must be measured before selecting a radius.

To remove the measured shared-storage bottleneck, the run stages copies of
existing coefficient databases on compute-node-local storage. SQLite keys,
coefficient precision, integration formulas and sampling remain unchanged.
The persistent cache is locked during ownership. Committed local tables are
copied back with SQLite backup and atomic replacement on normal completion,
Python exceptions, or SIGTERM; Slurm gives a two-minute warning for this.
Sample checkpoints stay on persistent storage throughout. SIGKILL/node loss
may require regenerating newly computed coefficients. A failed copy-back
preserves the previous persistent database and leaves a local recovery copy.

The report distinguishes a 10% *sampling* target from 10% *total numerical
accuracy*. Block, momentum and cutoff errors have not yet been bounded.
Meeting the sampling threshold, or approaching the matrix prediction, does
not by itself establish the requested accuracy. Subsequent controlled
refinements remain necessary. Each production job retains a one-hour limit.
