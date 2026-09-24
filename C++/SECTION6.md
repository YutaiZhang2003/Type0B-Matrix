# Section 6: superconformal blocks and numerical comparisons

Build from the repository root with `make -C C++ section6`. Requirements are a
C++17 compiler, GMP, MPFR, MPC and LAPACK/BLAS (Accelerate on macOS). The test
launcher requires Python 3.10 or later and only its standard library. Optional
archival Python partition-function tools additionally use NumPy, SciPy and mpmath;
the native `bin/partition` does not.

The numerical block computations are C++. Python launches them and compares
decimal output. Each process computes its own intermediate data and reuses them
in memory; it loads no numerical coefficient cache. Numerical conventions are
`product_bpz_residue_2026-09-22`, as in the current implementation. Neither
manuscript is edited by these commands.

## Coefficient tests

The cutoff policy agreed for the release is:

| Comparison | Domain | Choices covered |
| --- | --- | --- |
| NS-R-R theta versus physical SCA PBW | Each of three edges ≤5 | Both primary parities, both form parities, all vertex signs and all eight spin-lift evaluations |
| All-NS theta and glasses versus physical SCA PBW | Each of three edges ≤5 | Both form parities and all spin lifts |
| NS/R and R/R glasses versus physical SCA PBW | Each of three edges ≤5 | All allowed form/sign choices; zero, one and two insertions |
| Tetrahedron with a triangular R face versus physical SCA PBW | Sum of six edge levels ≤5 | All 64 allowed form/sign choices and all 64 spin-lift evaluations |
| All-NS theta versus independent SCA c-recursion | Each of three edges ≤10 | Both form parities and all eight spin-lift evaluations |
| All-NS tetrahedron versus independent SCA c-recursion | Sum of six edge levels ≤10 | All eight allowed form-parity choices and all 64 spin-lift evaluations |

For an NS edge, levels are half integers; for an R edge they are integers.
Thus independent level 10 on six NS edges would contain `21^6` multidegrees.
It is **not** the tetrahedron test above. R/NS glasses is obtained by exchanging
the two named handles. The tetrahedron's inequivalent four-edge R cycle is not
part of the stated R-face test.

Run a group independently, or use `--suite paper` for all groups:

```sh
python3 C++/tools/section6.py run --suite theta-pbw --output /tmp/section6-theta-pbw
python3 C++/tools/section6.py run --suite graph-pbw --output /tmp/section6-graph-pbw
python3 C++/tools/section6.py run --suite theta-ns --output /tmp/section6-theta-ns
python3 C++/tools/section6.py run --suite tetrahedron-ns --output /tmp/section6-tetrahedron-ns
```

These are full calculations, not smoke tests. Each group defaults to 40 decimal
digits. `--dps D`, for D≥30, controls the working precision. Use a fresh output
directory: successful results are not overwritten. The launcher retains logs,
individual coefficients, source hashes, process times and comparison summaries.
An exception, incomplete coefficient table or numerical disagreement gives a
nonzero exit status. If a graph PBW comparison misses its numerical threshold,
the launcher retains that result and repeats only the failing form/sign cases
with 20 additional digits. Both methods use the increased precision. The report
identifies each escalation; it never relabels the initial result as passed.
A completed computation is not itself a passed comparison.

For the theta NS-R-R executable alone, `--dps 0` selects machine precision;
both machine and MPC paths are documented in [README.md](README.md). The new
general-graph frontend currently uses MPC; it does not offer a machine-precision
graph path.

The comparison criterion is

```
abs(result - reference) / max(1, abs(result), abs(reference)) <= 1e-18.
```

Reports also retain the maximum absolute discrepancy. All formal edge-parity
components are checked, including zero components. Spin lifts evaluate the
finite parity polynomial by `(-1)^popcount(lift & parity)`; checking every
component therefore checks every lift. Graph PBW runs additionally evaluate
every lift explicitly. All-NS output has only the parity mask determined by
the multidegree; the other components vanish by the NS parity constraint.

## Individual blocks

The optimized NS-R-R theta interface remains:

```sh
C++/bin/ramond --mode ordinary --level 5 --truncation per-edge --dps 40 --json /tmp/theta-ordinary.json
C++/bin/ramond --mode inserted --level 5 --truncation per-edge --dps 40 --json /tmp/theta-inserted.json
```

`ordinary` means equal endpoint eta signs; `inserted` recovers the physical
block with opposite endpoint eta signs using the auxiliary `Theta psi`
insertion. `--p`, `--f`, `--eta`, `--b` and `--P1` through `--P3` choose the
parameters. `C++/bin/pbw` independently computes the physical block.

The common graph frontend accepts `theta-ns`, `glasses-ns`, `glasses-ns-r`,
`glasses-r`, `tetrahedron-r` and `tetrahedron-ns`:

```sh
C++/bin/graph_blocks --channel tetrahedron-r --method pbw-check --level 5 --truncation total --dps 40 --output /tmp/tetrahedron-pbw
C++/bin/graph_blocks --channel theta-ns --method dv --level 10 --truncation per-edge --dps 40 --output /tmp/theta-dv
C++/bin/graph_blocks --channel theta-ns --method ns --level 10 --truncation per-edge --dps 40 --output /tmp/theta-ns
python3 C++/tools/section6.py compare-ns --dv /tmp/theta-dv --ns /tmp/theta-ns --output /tmp/theta-comparison.json
```

`pbw-check` recovers the block from the enlarged double-Virasoro numerator and
compares it with independent physical SCA sewing. It also checks forward
convolution, unequal split powers, vanishing uninserted odd-loop numerators
and all lift evaluations. `dv` writes physical coefficients from double
Virasoro for every listed graph, without a physical-block PBW reference.
`ns` supplies the independent all-NS SCA c-recursion for theta and tetrahedron.
`--b` and `--P1` through `--P6`
override the default generic benchmark parameters. `--case-index K` selects
one form/sign case for `pbw-check`; the output metadata lists its actual forms
and signs, and its output case index is then zero.

`metadata.json` records the parameter values, graph, vertex forms, precision,
conventions and cutoff. In graph `coefficients.jsonl`, `level2[e]` is twice the
physical exponent of edge e; `case` indexes the forms in the metadata. Each
row is `[parity_mask, real_string, imaginary_string]`. Omitted row components
are zero. `summary.json` is published only after all output is written.

## Numerical implementation

The common graph core is in `include/scblocks/`; the optimized NS-R-R theta
core remains in `include/ramond/`. Both use cached forward CCY c-recursion,
pole prefactors, fusion polynomials, local global vertices and inverse
central-charge differences. A branch receives only its remaining descendant
domain. Split edges retain the unequal intermediate levels required for
the physical diagonal. The general graph verifier also computes the unequal
split outputs needed by its forward-convolution test.

The Virasoro seed combines weight-dependent global blocks with the exact
primitive Schottky vacuum product of the ordered graph. The two vacuum
factors are restored after the branch sum. Independent SCA c-recursion uses
its own superglobal coefficients, NS poles and fusion polynomials, together
with the bosonic and fermionic Schottky products. Neither recursion uses a
finite-c Virasoro Gram sum as its seed.

Each primitive cycle's trace is expanded only to the order still available
after its first possible oscillator contribution. The bosonic product starts
with two copies of the cycle; the NS fermionic product starts with three copies
in half-level units. Keeping that leading monomial separate substantially
reduces exact rational work at independent-edge cutoffs. The remaining domain
is allowed to have different cutoffs on different edges. The directed exact
check is `make -C C++ check-graph-schottky`.

The independent SCA reference also batches its recursion. For a fixed output
edge-parity mask, residues depend on the accumulated null shifts, rather than
the remaining even levels. Paths with the same shift and incoming pole are
therefore merged before attaching global coefficients. The Schottky factor is
restored once: its monomials have even incidence at every vertex, and the
resulting null-transport sign is precisely the star-product sign. The original
target-by-target recursion remains available as `NSRecursion::coefficient` for
directed verification. `make -C C++ check-ns-forward` compares both evaluations
at independent level 3. The batch reference is also compared with the archived
tetrahedron coefficients through total level 6.

NS-R-R branching uses the existing cached boundary recurrence, including
the second Ward identity at a vanishing first pivot. The two identical NS
theta vertices use the existing closed four-product branching formula.
Other all-NS graph vertices use cached local free-field/PBW branching data.
These local data do not use the final physical-block reference. Free-fermion
factors are computed directly by Fock-state sewing and Ward/Pfaffian forms.

The physical PBW reference contains full inverse SCA Gram matrices and
independent Ward vertices. At each trivalent vertex, slot order is
`(infinity, 1, 0)`. For a marked R edge the expanded domain uses the maximum
of its left and right levels as its original-edge level. Recovery takes
place in the appropriate R-cycle parity sector. Degenerate weights and
coincident recursion poles require a limiting prescription not supplied by
these generic-weight commands.

## Partition-function experiment

The native entry point `bin/partition` retains the paper's literal `F`,
`C_a`, and `C_f_eta` in edge order `(infinity,1,0)`. For the physical
partition comparison, it multiplies each block coefficient by the local
geometric BPZ phase and then evaluates it at one fixed tube-sign triple.
This is done independently in the NS-R-R and all-NS channels. The two
marked spin surfaces and their raw plumbing signs are recorded in
[PARTITION.md](PARTITION.md). No two-sign block average or all-NS sewing
matrix is used. The unconverted paper blocks remain in the node output
for auditing. All cutoffs are total descendant levels, both Ramond `f`
sectors are computed directly, and the identity-residue normalization is
in [PARTITION_NORMALIZATION.md](PARTITION_NORMALIZATION.md).

The directed geometry check builds 228 Majorana Wick monomials through
level 8 and compares the literal block's diagonal square with
bosonization for all four marked NS spins under the assumed map. That
map fails by 5.53% to 31.86%; the SCblock sign reduction does not.
Literal NS and Ramond chiral blocks pass direct PBW comparisons through
total level 3. The full Ramond nonchiral source contraction still needs
an independent test in the paper's ground-state basis.

`tools/section6_partition.py` delegates to this native executable.
`tools/prepare_partition_inputs.py` converts historical geometry-order
inputs once, discards all saved block fields, and supplies paper-order
inputs and coefficients. The production source archive excludes the
obsolete Python sewing adapters. The older adjusted-block data remain
as provenance, not as a fixed-spin check of the paper's literal block.

## Evidence and timing

Fresh results from this consolidation are stored in
`results/section6_2026-09-23/`. The previously completed total-level-10
tetrahedron comparison is in
`experiments/mercedes_all_ns_level10_2026-09-16/`: 230,230 multidegrees,
maximum scaled discrepancy `2.054e-25`, about 16 minutes for the two methods
at 40 digits. Its saved coefficient files and manifests remain unchanged.

The original approximately 7-minute ordinary and 27-minute inserted theta
benchmarks at independent level 10 are in
`results/per_edge_level10_2026-09-11/optimized_full_pipeline_40dps/`.
They are archived timings, not fresh timing claims for this release.
The direct-PBW level-10 estimates and measured level-5 benchmarks are in
`results/direct_pbw_cpp_2026-09-12/`. Level-10 physical PBW is not run by
the Section 6 suite.

`tools/estimate_tetrahedron_box.py` counts the rejected six-edge independent
cutoff option and scales saved timings. It performs no block computation;
its runtime scenarios are estimates, not measured bounds.
Its SCA-reference estimate describes the earlier target-by-target evaluation;
it does not estimate the new batched SCA reference.

## Source archive

```sh
python3 C++/tools/export_section6.py --output /tmp/section6-superconformal-blocks.tar.gz
```

The archive contains the numerical cores, drivers, comparison tools, convention
documentation and compact result summaries. It includes no manuscript files,
binaries or numerical caches. The SHA-256 manifest records the packaged source.
Build the extracted archive with `make -C C++ section6`; partition-function
momentum bundles are supplied separately.
