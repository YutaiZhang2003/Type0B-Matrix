# C++ PBW versus C++ double Virasoro

Fresh sequential processes at 40 decimal digits / 136 bits, with each of
q1, q2, q3 independently cut off at level 5. No numerical caches were loaded;
within-run caches were retained. Compilation is excluded; wall time includes
startup, serialization, cache destruction, and exit. Both numerical algorithms
are C++17/MPC. Python only launches processes and compares output.

Parameters: b=7/5, P=(11/23,13/29,17/31), p=f=0, eta=+1.

| Sector | C++ PBW wall | C++ double Virasoro wall | PBW/DV |
|---|---:|---:|---:|
| eta eta'=+1 | 4.771 s | 3.097 s | 1.54 |
| eta eta'=-1 | 8.609 s | 7.322 s | 1.18 |

Each output contains 396 monomials and 3,168 parity slots, including the
q1^5 q2^5 q3^5 corner. All components were compared. Maximum scaled differences
PBW versus double Virasoro are 8.598e-21 and 1.395e-20; the scale is
max(1,abs(a),abs(b)). Absolute maxima are 3.951e-16 and 1.724e-16.
The C++ port agrees with saved Python PBW to 2.817e-38 and 1.732e-36 on the
same scale. No Python block was rerun. The directed level-2 port check is
saved in `validation/`.

The double-Virasoro executable is the isolated optimized build in
`C++/experiments/pbw_vs_double_virasoro_level5_2026-09-12/`, including grouped
inserted assembly and dense vertex caches. Its build recipe is `build_ccy.py`
in that directory. The C++ PBW source is `C++/include/ramond/direct_pbw.hpp`,
with benchmark driver `C++/drivers/pbw_main.cpp` and target `C++/bin/pbw`.
Source and binary hashes, compiler version, commands, stages, counts, and
directed comparison records are in `timings.json`.

The PBW `inserted` mode selects opposite vertex signs for direct physical
sewing; it introduces no auxiliary fermion insertion. The driver has the
fixed benchmark b and momenta above. Its `--level` always means an independent
cutoff on every edge. Both machine (`--dps 0`) and MPC (`--dps >=30`) numeric
templates compile; the measured and validated runs here use 40 digits.

## Level-10 estimate, without running PBW

`level10_estimate.json` contains exact basis/tensor/contraction counts and
stage-by-stage scaling of these C++ timings. At independent level 10:

| Sector | Constant unit costs | Longer-Ward sensitivity scenario |
|---|---:|---:|
| eta eta'=+1 | 6.55 h | 8.71 h |
| eta eta'=-1 | 8.54 h | 12.90 h |

Vertex requests grow by 1,862.04 and contraction products by 16,605.39.
The baseline holds their measured unit costs fixed. It predicts 1.90/3.83 h
of vertex work and 4.61 h of contraction work in each sector. The second
scenario multiplies only vertex time by 2.137, the exact ratio of entry-weighted
mean total levels. It illustrates sensitivity to longer Ward recursion; it
does not count level-10 internal Ward operations. Neither scenario is a bound
or confidence interval.

The measured C++ double-Virasoro production level-10 times remain 416.619 s
and 1,655.360 s. The approximately 19-minute negative-sector figure with the
latest CCY changes is a projection, not a completed fresh full run.

Both PBW scenarios assume enough RAM without paging. Persistent caches store
at least 659,037,706 / 1,318,075,412 requested Ward entries. At the measured
96-byte key/value object size plus at least 48 bytes of complex mantissa
payload, these alone require approximately 95 / 190 GB (decimal). Hash-table
overhead, internal Ward entries, action caches and tensors need additional
memory. No tensor-permutation tables are stored. No level-10 PBW run, peak
memory measurement, or accuracy validation was performed.

## Reproduction

From the repository root:

```sh
make -C 'C++' bin/pbw
python3 'C++/tools/benchmark_direct_pbw.py'
python3 'C++/tools/estimate_direct_pbw.py'
python3 'C++/tools/render_per_edge_notes.py'
```

The benchmark performs only fresh C++ level-5 runs. The latter two commands
analyze saved records and update LaTeX tables without evaluating blocks.
