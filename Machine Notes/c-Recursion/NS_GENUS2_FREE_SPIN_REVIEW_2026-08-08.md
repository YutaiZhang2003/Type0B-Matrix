# Genus-two NS free-superfield and spin-transport review map

This bundle records the code used to form

\[
Q_L^{\rm pl}=\frac{Z_L^{\rm pl}}
{\left(Z_{X+\psi}^{\rm pl}\right)^9}
\]

in the theta and glasses plumbing frames, together with the completed
five-point \(R=24,N=10\) result and independent regression checks.  No
theta/glasses agreement is imposed.

## Certification status

The fixed-cutoff calculation is now reproducible from the exact production
source and all 10,000 immutable Cannon shards.  It is **not** yet certified as
a channel-independent genus-two partition function: the five ratios
\(Q_L^{\rm theta}/Q_L^{\rm glasses}\) range from 1.02108 to 1.02795.  This
2.1--2.8 percent discrepancy is much larger than the measured period-map and
free-denominator errors.  The supplied run contains only \(R=24,N=10\), so it
does not establish integrated recursion- or momentum-order convergence.

The exact production bundle is
`ns-genus2-fivepoint-r24-n10-production-repro.tar.gz`, with SHA-256
`8236a9ff096138ec8a4f067e6778851018a8f2a060975066b92770b65c9cfd70`.
It contains the original unmodified summary, source snapshot, configuration,
and all shards.  Its production fingerprint is
`c241dee82bcf535da8034200a753152a4df2d417980ccead3c2e62839518ed6e`.
The independent shard audit validates each filename, task and node index,
decoded channel/design, quadrature indices, momenta, measure, edge ordering,
config digest, and fingerprint before recomputing all ten Liouville numerator
sums.  The ordered shard-set SHA-256 is
`9f0f990bdfaae203fb1034cb5cecb2cf1a0902dbd2d61444908adbefd280000c`.

## Free-superfield formula

For the primitive Schottky multipliers \(k_\gamma\) in the selected plumbing
marking,

\[
P_X^{\rm pl}(\mathbf q)
=\prod_{[\gamma]\in\mathcal P_{\rm prim}}
 \prod_{n=1}^{\infty}(1-k_\gamma^n)^{-1}.
\]

With unit connected target-space zero-mode volume, one real scalar and one
Majorana fermion contribute

\[
Z_{X+\psi}^{\rm pl}
=(\det\operatorname{Im}\Omega)^{-1/2}
\left|\vartheta[\delta](0|\Omega)\right|
\left|P_X^{\rm pl}\right|^3.
\]

The factor \((\det\operatorname{Im}\Omega)^{-1/2}\) is the two-handle scalar
momentum Gaussian.  The Majorana factor is evaluated by exact bosonization,
\(Z_{\psi,L}^{\rm pl}=(\vartheta[\delta]P_X^{\rm pl})^{1/2}\).  The common
edge Casimir powers are stripped in both numerator and denominator.

## Spin marking fixed in this snapshot

The complete branch-composed glasses-to-theta matrix is

\[
M=\begin{pmatrix}
0&0&-1&-1\\
0&0&0&-1\\
1&0&0&0\\
-1&1&0&0
\end{pmatrix}.
\]

For \(\Omega'=(A\Omega+B)(C\Omega+D)^{-1}\), characteristics are transported
by

\[
\alpha'=D\alpha-C\beta+\operatorname{diag}(CD^T),\qquad
\beta'=-B\alpha+A\beta+\operatorname{diag}(AB^T)\pmod2.
\]

This maps \([00|00]\) to \([00|00]\).  Both channel configurations therefore
use physical plumbing lifts \((+,+,+)\).  The older pre-branch theta marking
\((+,+,-)\), which realizes \([00|11]\), is not used after the final integer
branch transformation.  A Cannon configuration that declares
`expected_spin_characteristics` now fails before shard evaluation if its
lifts do not realize the declared characteristic.

## Primary review files

- `Code/ns_genus2_partition.py`: Liouville integrand, recursion, global-block
  resummations, free-superfield denominator, spin characteristic and analytic
  checks.
- `Code/ns_genus2_cannon.py`: array design, deterministic reduction,
  fail-closed shard/configuration validation, runtime fingerprints, and
  formation of \(Q_L\).
- `Code/audit_ns_genus2_production_bundle.py`: independent identity and
  deterministic-reduction audit for the exact archived production run.
- `Code/test_ns_genus2_partition.py`: twenty-one genus-two regressions.
- `Code/python/free_boson_pair_of_pants.py`: independent direct Heisenberg
  descendant sewing, which does not use the primitive-word product.
- `Code/python/free_boson_plumbing.py`: scalar zero-mode conventions and
  Riemann theta constants.
- `Code/python/plumbing_algorithms.py`: theta/glasses plumbing maps, Schottky
  generators, and period maps.
- `Code/python/genus2_vacuum_blocks.py`: primitive conjugacy words and word
  multipliers.
- `Code/ns_vacuum_schottky.py`: lifted NS Schottky product.
- `Data Set/ns_genus2_fivepoint_r24_n10_affine_summary.json`: complete
  five-point configuration, provenance, spin ledgers, free denominators,
  Liouville numerators, and channel ratios.  Its stale branch-provenance sign
  is corrected with an explicit post-run correction record; numerical values
  are unchanged.
- `Machine Notes/c-Recursion/ns-genus2-fivepoint-r24-n10-production-repro.tar.gz`:
  exact original production snapshot and all 10,000 shards.

The remaining files in the archive are the recursion, structure-constant,
regular-block, and cluster-entry dependencies imported by these primary files.

## Reproduction checks

From the repository root:

```bash
PYTHONPATH=Code:Code/python python3 -m unittest Code.test_ns_genus2_partition
PYTHONPATH=Code:Code/python python3 Code/python/free_boson_plumbing_checks.py
```

The first command passes 21/21 tests in this snapshot.  The free-boson command
also runs from the isolated ZIP; its two formatting helpers no longer import
an omitted workspace module.

To audit the exact production run without recomputing its CFT blocks:

```bash
mkdir -p /tmp/ns-genus2-production-audit
tar -xzf \
  'Machine Notes/c-Recursion/ns-genus2-fivepoint-r24-n10-production-repro.tar.gz' \
  -C /tmp/ns-genus2-production-audit
python3 Code/audit_ns_genus2_production_bundle.py \
  /tmp/ns-genus2-production-audit
```

This audit certifies fixed-cutoff provenance and reduction only; its output
therefore records `convergence_certified: false`.

To extract the exact completed-run configuration for a new run:

```bash
jq '.config' 'Data Set/ns_genus2_fivepoint_r24_n10_affine_summary.json' \
  > /tmp/ns_genus2_fivepoint_r24_n10_affine_config.json
```

Then inspect the planned designs without launching work:

```bash
PYTHONPATH=Code:Code/python python3 Code/ns_genus2_cannon.py \
  --config /tmp/ns_genus2_fivepoint_r24_n10_affine_config.json plan
```

## Disjoint-annulus certificate

The formal inequalities \(|q_e|<1\) are not by themselves enough to justify
analytic sewing.  In the standard trinion coordinates
\((u_0,u_1,u_\infty)=(z,z-1,1/z)\), choose balanced incidence radii
\(\sqrt{|q_e|}\) and enlarge every radius by a common factor
\(\lambda=1.1\).  The enlarged disks remain pairwise disjoint on every
trinion at all five points.  Since

\[
 r_{e,L}r_{e,R}=\lambda^2|q_e|=1.21|q_e|>|q_e|,
\]

each tube has a nonempty sewing annulus.  The minimum disk clearance is
0.14722 in the glasses charts and 0.45180 in the theta charts.  The smallest
limiting enlargement over all points and channels is still 1.28990.  Thus the
plumbing decompositions used here lie strictly inside the disjoint-annulus
domain, rather than on its boundary.  The complete certificate is generated
by `Code/audit_ns_genus2_plumbing_annuli.py` and stored in
`Data Set/ns_genus2_fivepoint_plumbing_annuli_audit.json`.

## Plumbing-coordinate stability

Across the five points, \(\max|q_e|=0.2479\).  Finite-difference Jacobians of
the period maps in real and imaginary \(\log q_e\) coordinates have condition
numbers between 3.08 and 3.68.  Treating the word-length-eight to nine period
movement as an effective coordinate error gives
\(\max|\delta q_e/q_e|=3.13\times10^{-9}\).  The largest resulting relative
change of one free-superfield denominator is below \(2.7\times10^{-9}\), or
\(2.4\times10^{-8}\) after the ninth power.  Increasing the genus-two theta
lattice cutoff to 16 changes the theta constants by at most
\(2.3\times10^{-16}\).

At `o0239`, direct scalar descendant sewing approaches the resummed primitive
product as follows:

| total sewing level | theta relative error | glasses relative error |
|---:|---:|---:|
| 4 | \(1.80\times10^{-9}\) | \(1.41\times10^{-3}\) |
| 6 | \(8.92\times10^{-11}\) | \(2.97\times10^{-4}\) |
| 8 | \(3.96\times10^{-12}\) | \(6.91\times10^{-5}\) |

These errors are truncation errors of the independent direct sum; the
primitive product itself is carried to word length 14 and mode 70 in the
completed calculation.

## Scope note

The completed numerical summary already used the branch-composed
\([00|00]\) configuration.  The code correction in this snapshot repairs the
stale pre-branch analytic assertion and makes future configuration checking
fail closed; it does not retroactively alter the stored five-point numerical
values.

The summary/source fingerprint mismatch in the first review bundle was a
packaging mistake: it combined the archived summary with a later checkout.
Future shards record Python, mpmath, NumPy, and SciPy versions; the reducer now
rejects mixed implementations, wrong task/node identities, missing or extra
filenames, quadrature-data mismatches, and edge-order mismatches.

The next physics-level checks remain an independent finite-\(c\) direct sewing
oracle for the glasses graph and same-point integrated sweeps at
\(R=20,22,24\) and \(N=8,10,12\).  Until those are complete, the five-point
result must be described as an unresolved locality discrepancy rather than a
successful crossing check.
