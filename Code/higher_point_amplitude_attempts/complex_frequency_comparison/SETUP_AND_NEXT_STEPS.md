# Five-point Type-0B setup and gradual optimization

Baseline: Cannon job 43256507, source c738e39, completed in 17 min 46 s.
The baseline configuration is
`Code/config/type0b_ns_five_tachyon_one_hour_preliminary.json`; results are in
`Data Set/type0b_ns_fivepoint_one_hour_20260831/`.
The following steps are proposals, not newly submitted jobs.

## Observable and conventions

Compute the genus-zero all-NS, all-tachyon 1->4 amplitude at
`omega_a = 0.25 + 0.02i` for all four outgoing particles, and
`Omega = sum_a omega_a = 1 + 0.08i`.
These fixed complex energies define the target. The imaginary part is not
being extrapolated away. We evaluate the analytic continuation of the reduced
amplitude on `Omega=sum_a omega_a`, with the coupling power `mu_F^-3` removed.
No delta function of a complex argument is present in the numerical quantity.
The delta function belongs only to the original real-energy S matrix.

The implemented BRY normalization and matrix prediction are

```
A_T,WS = i I5 / pi^2
A_T,MM = i Omega prod_a(omega_a) (1 + 2i Omega)(2 + 2i Omega).
```

This compares the same all-tachyon basis on both sides. It does not assume
equality of the other worldsheet NS/R amplitudes. The BRY coupling dictionary
and examples of finite-complex-energy comparison are in
[arXiv:2201.05621, (4.14) and section 4.2.2](https://arxiv.org/pdf/2201.05621).

## Integral and conformal blocks

Fix three punctures on the sphere, leaving two complex moduli. A five-point
comb channel contains two internal Liouville momenta. Thus the unfactorized
integration has four real moduli coordinates and two real spectral variables.
The worldsheet integrand includes the super-Liouville structure constants,
NS blocks and required picture-changing and timelike factors.

The code compares 120 oriented charts representing 15 trivalent trees and
chooses the chart minimizing `max(|q1|,|q2|)`. These q variables are ordinary
linear-channel plumbing coordinates, not elliptic nomes.

Fixed-weight c-recursion constructs the reduced polynomial coefficients for
each required set of external weights, internal momenta, descendant labels
and parity sectors. The tables are reused over moduli points. A deeper block
cutoff can extend an existing table; changing momentum nodes generally needs
additional tables. The h-recursion is not used in this production.

At complex external energies, the antiholomorphic block retains the same
analytically continued external parameters. One must not replace the block
product by an absolute-value square that conjugates those parameters.

## Boundary treatment and numerical controls

The numerical bulk remainder is schematically

```
F - chi_1 P_1 - chi_2 P_2 + chi_1 chi_2 P_12.
```

The subtracted terms are restored by analytic radial finite parts on the
faces and compatible corners. The face contribution also has its tangential
corner subtraction. All ten boundary faces and fifteen compatible corner
intersections are included, using symmetry orbits where appropriate.
The final integral is `I5 = B + D + C` (bulk, face and corner terms).
These are pieces of one subtraction prescription, not separate physical
amplitudes. Their individual magnitudes do not diagnose a normalization error.

| Control | Baseline |
| --- | --- |
| c-block edge twice-levels | (4,4), physical descendant levels (2,2) |
| Total twice-level cutoff | 8, full rectangle |
| Spectral quadrature | composite Gauss orders (2,3), staggered node sets |
| Spectral domain | 0 < P1,P2 < 2 |
| Threshold refinement | one shell, constant singularity subtraction enabled |
| Collar radius rho | 0.01 |
| Primary projection radius | 1e-5 |
| Randomized Sobol integration | 8 independent replicates; 4 bulk and 8 face samples per replicate |
| Total moduli samples | 32 bulk, 64 face |
| Arithmetic precision | coefficients 45 digits; structure constants 22 digits |
| Fast evaluation | complex128 tensors with scalar fallback for flagged cancellation |
| Execution | four workers, reduction and comparison in one one-hour allocation |

The collar radius rho is a numerical subtraction parameter, distinct from
the physical imaginary frequency epsilon=0.02. Neither 45-digit coefficient
construction nor a deterministic corner calculation guarantees a precise
final integral. The corner still has quadrature and finite-projection errors.

## What the baseline tells us

| Raw integral component | Real | Imaginary | Sampling SE, real / imaginary |
| --- | ---: | ---: | ---: |
| Bulk B | 138.5135 | 145.0312 | 72.5132 / 75.4635 |
| Face D | -142.5682 | -117.9603 | 44.5808 / 38.4748 |
| Corner C | 243.7647 | 190.6023 | not sampled; quadrature error not estimated |
| Sum I5 | 239.7100 | 217.6732 | 92.3150 / 90.9072 |

The total error is computed from replicate sums; component errors are not
assumed independent. The normalized result is
`(-22.0549 +/- 9.2108) + i(24.2877 +/- 9.3535)`, whereas the matrix prediction
is `-0.01581584458368 - 0.017243732128i`.
Sampling errors are much larger than the target scale, and systematic errors
are unbounded. The baseline does not isolate which approximation causes the
large discrepancy. In particular, it does not establish a physical mismatch.

There is also a runtime issue separate from accuracy. Worker 0 completed its
first bulk batch at 209.897 s and its first face sample at 989.487 s; the
following seven face samples ended at 990.463 s. The first face therefore
took 779.591 s, versus 0.975 s for the next seven together. Slurm reports
487.409 CPU seconds over 1066 wall seconds with four CPUs, about 11.4% use
of the allocated CPU capacity. The cache performs a synchronous SQLite
transaction for every table write, on shared storage. I/O waiting is a strong
hypothesis, not yet an isolated measurement of every second of that delay.

## Proposed sequence, with at most one hour per run

1. **Remove avoidable runtime overhead without changing the calculation.**
   Reuse the completed coefficient stores; test compute-node-local SQLite
   storage with durable copies/checkpoints. Consider batching writes with an
   explicit flush policy. Repeat the same seeds and numerical settings and
   compare outputs with the baseline. This is a runtime check, not an accuracy
   claim, and must preserve completed data on normal termination.
2. **Check deterministic momentum integration on a small fixed set.**
   Hold frequencies, rho, block depth and threshold shells fixed. Compare
   orders (2,3), (3,4), (4,5) for the complete corner term and selected face
   and bulk points. Vary the shell count separately if needed. This can expose
   bias without burying it in Monte Carlo noise.
3. **Check block depth at those same points.**
   Increase twice-levels (4,4) to (6,6), with the full rectangular total
   cutoff. Keep the quadrature fixed. The corner primary projection alone is
   not a test of higher descendant contributions.
4. **Improve the moduli estimator.**
   Inspect the largest sample weights and their charts. Increase bulk samples
   first or improve importance sampling/control variates where supported by
   the measured variance. For a simple first comparison, raise total bulk
   samples from 32 to 128 while holding face samples and other controls fixed;
   use nested Sobol points. Evaluate face refinement separately. Any new
   control variate must have a known integral or an independently accounted
   contribution; no matrix value is used to tune the estimator.
5. **Check the remaining cutoff dependence on the improved estimator.**
   Vary Pmax from 2 to 3 while preserving the old unit panels, then compare
   rho=0.01 and rho=0.005 with common random samples. Do not combine these
   changes into a single unexplained shift or perform an epsilon extrapolation.

The user has now set a target of 10% numerical accuracy. This is an accuracy
objective, not permission to label an unresolved result converged.
Report sampling error and the shifts under controlled refinements separately;
agreement with the matrix prediction is not itself an error estimate.
