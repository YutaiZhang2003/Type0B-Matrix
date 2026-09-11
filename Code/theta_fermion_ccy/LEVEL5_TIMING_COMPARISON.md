# Level-five timing: current CCY pipeline and direct physical PBW

After the subsequent reuse of zero and tolerance constants, the current
complete CCY/branching/fermion-recovery pipeline took **12.6937 s**, compared
with the direct physical PBW measurement of **1.4991 s**. PBW remains about
**8.5 times faster** for this sample. The new run and its validation are
recorded in the final section below.

The earlier CCY implementation took **89.8903 s**, about 60 times the PBW
process time. That measurement is retained below as a **prior implementation**
record. These are measurements of existing implementations on one specified
physical block, not asymptotic complexity results.

The user requested this comparison before continuing the level-ten work.
The PBW calculation was rerun solely to measure its runtime. No new numerical
comparison of its coefficients with CCY coefficients, auxiliary coefficients,
or any other block was performed for this timing report.

## Earlier comparison: before constant reuse

| Measurement | Prior CCY pipeline | Direct physical PBW |
| --- | ---: | ---: |
| Internal recorded time | 88.26416633400004 s | 0.711694791996706 s |
| Fresh child-process elapsed time | 89.89025191700057 s | 1.4991441250022035 s |
| Process exit code | 0 | 0 |
| Saved coefficient vectors | 581 four-variable vectors | 91 three-variable vectors |

The internal-time ratio is about 124, but the child-process ratio of about
60 is the preferable headline because the internal clocks exclude different
pieces of setup and output. Both complete process times include imports,
computation, output writes, and process exit.

The parameters and physical truncation agree:

\[
b=\frac75,\qquad
(P_1,P_2,P_3)=\left(\frac{11}{23},\frac{13}{29},\frac{17}{31}\right),
\qquad p_1=f=0,\qquad(\eta,\eta')=(+,-),
\]

with all physical terms through total \(q\)-level five and the same eight
parity components. The CCY route retains the four plumbing variables of the
subdivided theta graph, uses the complete direct fermion auxiliary, and
recovers the physical block by restricted convolution division. The direct
PBW route constructs that physical block directly in the three original
plumbing variables. The differing vector counts reflect this intermediate
representation, not differing physical cutoffs. No unequal split powers are
deleted from the CCY output to reduce its cost.

## What each timer includes

The CCY pipeline's internal timer begins after its module imports, argument
parsing and rational-parameter setup. It includes action construction, outer
Ward solves, middle recurrence, both Virasoro factors, branch assembly, direct
auxiliary calculation, restricted division, restoration of two universal
vacuum factors, and intermediate serialization. It stops before the final
physical-series encoding and final JSON save. Its separately saved wrapper
time covers the entire child process from launch through exit.

The PBW internal timer starts after `load_runner()` and its imports/dynamic
source loading, immediately before constructing the physical oracle. It
includes physical Gram/Ward calculations, coefficient encoding and earlier
checkpoint writes. The recorded last timestamp precedes the last checkpoint
write/progress print and the final status save. Its separate child-process
time includes all those exclusions as well as imports and setup.

Both runs launched fresh Python processes. Neither loads previously computed
block coefficients or a precomputed coefficient cache as input. The PBW loader
reads implementation source files to construct its arithmetic-specialized
backend; those reads are source loading, not saved-coefficient reuse. Normal
within-run caches for repeated Gram, Ward, action or CCY data are part of each
implementation. These are fresh-process measurements, not a claim that the
operating system's filesystem caches were flushed.

## Arithmetic and implementation scope

The current CCY route uses 70 decimal digits for the outer Ward/anchor system,
100 decimal digits for the middle recurrence and CCY engines, and exact
rational auxiliary sewing before restoring its common normalization. Most
branching and CCY arithmetic is implemented in Python with mpmath; the Ward
refinement uses numerical linear-algebra preconditioners and multiprecision
residuals.

The direct PBW implementation uses FLINT complex matrices at 384 bits of
arithmetic precision, approximately 116 decimal digits. Its scalar wrapper
retains complex midpoints and discards scalar terms below `1e-80`; encoded
outputs contain up to 105 decimal digits. Thus the two runs are neither
precision-matched nor arithmetic-backend-matched. The speed difference cannot
be assigned solely to the abstract recursion versus PBW algorithms. It does
establish that the present PBW code computes this low-level physical block
much faster, even though its nominal arithmetic precision is higher.

The rerun followed the actual direct-physical call path:
`pbw_reference.py` calls `Check.physical_coefficient`, which contracts the
physical SCA Gram inverses with the two physical HumanForm tensors. It does
not call the enlarged-coefficient routine, the zero-mode-recovery routine,
or the CCY pipeline. Loading a module that also contains those routines does
not execute them on this path.

For the preceding CCY N5 run, outer branching took 56.9850 s and the middle
recurrence 9.2499 s, together about 75% of its internal time. Its Virasoro
block/product stage took 18.2149 s. These saved stage measurements explain why
optimizing only the CCY seed cannot remove the entire low-level overhead.
They do not provide a level-ten runtime estimate.

The PBW rerun was performed while the level-ten CCY process remained active.
Its wall-time record states that concurrency explicitly. Resource contention
can affect this single measurement, and the runs were not a controlled
isolated-machine benchmark. No claim about performance at level ten or
arbitrary genus follows from this level-five comparison.

## Saved evidence and static audit

- `results/ccy_cached_actions_level5.json` records the preceding CCY
  production, parameters, source manifest, all coefficients and internal
  timing.
- `results/ccy_cached_actions_level5_walltime.json` records its successful
  fresh-process command and complete elapsed time.
- `results/pbw_level5_retimed.json` records the newly timed direct physical
  PBW calculation with the same parameters and cutoff.
- `results/pbw_level5_retimed_walltime.json` records its command, return code,
  complete elapsed time, timer scope and concurrent level-ten process.
- The already authorized coefficient and split-position validations remain
  in `results/validation_ccy_cached_actions_level5.json`. They used the
  independently stored reference `pbw_level5_p0_f0_eta_plus_minus.json`.
  The newly retimed PBW coefficients were not compared again.

An independent static inspection read the timer boundaries, the PBW loader,
its actual `physical_coefficient` call path, the 384-bit arithmetic setting,
the scalar threshold, the current CCY source and saved metadata. It confirmed
the parameter/cutoff agreement and the distinctions stated above. It imported
no production module, ran no calculation and performed no block comparison.
That production and validation recorded the same 17-file source manifest.
After constant reuse, their only difference from current sources is
`compute_target.py`; the matching old file is archived. They no longer serve
as the current-source validation.

## Later run: reuse of zero and tolerance constants

The user next requested reuse of redundant zero and tolerance values,
followed by a fresh complete N5 timing. The new run is
`results/ccy_reused_constants_level5.json`, with complete child-process timing
in `results/ccy_reused_constants_level5_walltime.json`. Both files record
successful completion. Its parameters, 70/100-digit settings, full
four-variable cutoff, direct auxiliary backend and restored seed power two
are unchanged.

| Measurement | CCY with constant reuse |
| --- | ---: |
| Internal computation time | 12.246520542001235 s |
| Complete child-process elapsed time | 12.693745042000955 s |
| Outer branching | 5.71879374999844 s |
| Action preparation, included in outer branching | 5.292387083001813 s |
| Middle recurrence | 1.0367864490181091 s |
| Virasoro block products | 4.577204588022141 s |
| Direct auxiliary | 0.11444854100045632 s |
| Division and two restored vacuum factors | 0.3801997089976794 s |
| Peak resident memory | 115.5 MiB |

A nearby rerun of the archived pre-constant implementation gives the more
useful comparison: **19.7228 s to 12.6937 s**, about **1.55 times faster**
for the complete process. The outer stage changed from **11.7306 s to
5.7188 s**, about **2.05 times faster**. Its CCY stage remained close,
4.5469 s before versus 4.5772 s after. The earlier 89.8903 s measurement
is therefore not used to attribute a speedup to this small change.

Both nearby retimings ran while the old N10 process remained active, so they
are still single measurements under concurrent load, not an isolated-machine
scaling study. The earlier PBW arithmetic/backend and timer-scope
qualifications continue to apply; no precision-matched scaling between PBW
and CCY is established.

The edit is confined to `compute_target.py`: reuse exact immutable complex
zero in sparse accumulation, and cache the unchanged tolerance formula by
target decimal precision, active binary precision and rounding. It preserves
the summation order and pruning rule. `CONSTANT_REUSE_REVIEW.md` independently
audits those semantics. The archived source
`profiling_baseline/compute_target_pre_constants.py` has SHA-256
`46ebb1a8bd10b356ee8697637f61ffb62b680258858e95aabbb69ad68b6568ca`, matching
the prior N5 manifest. The new source has SHA-256
`3368f697c713860f9c2a8619a29c8a5e1113da54240337dc1ebc65a846390c42`.

The parent repeated only the same two prescribed final-output validations.
`results/validation_ccy_reused_constants_level5.json` records both as passed,
with 4,648 parity slots, maximum scaled physical-PBW error
`3.836489189392700749077260789112036774263177627230062719485550661212873e-52`,
and maximum split-position variation
`1.665633005810068725586493084353646290474732375332339043314027019095599e-58`.
All 17 hashes in the new production manifest and copied validation manifest
match current sources. The output retains 581 four-variable coefficient
vectors and all 24 split evaluations. This report inspected those saved
metadata and hashes without importing production or performing a new
coefficient calculation/comparison.

The `ccy_cached_actions_level10` process imported the old source before
this edit and captured its old manifest. It subsequently completed in
1189.8001477500002 s internally and 1191.6387245000005 s from launch through
exit. This is a prior-implementation result, not a benchmark of the new
constant-reuse source. A fresh `ccy_reused_constants_level10` run is now
active; its runtime remains unmeasured.

## Nearby archived baseline and its provenance

`results/ccy_pre_constants_retimed_level5.json` records the archived baseline;
`results/ccy_pre_constants_retimed_level5_walltime.json` records its successful
fresh-process timing. Its internal computation took
`19.32923983300134` s and complete process `19.722815040997375` s. The outer,
middle and CCY stages were respectively `11.730626958997163`,
`2.1348760460205085`, and `4.546881951991963` s. It retains the same parameters,
precision, 581 vectors, 296 branch cases and seed power two. This was a timing
run only; its coefficients were not compared to another result.

The driver `profiling_baseline/run_pre_constants_level5.py` registers the
archived file in `sys.modules` under the canonical name `compute_target`
**before** importing the unchanged production pipeline. All imports of that
module therefore receive the archived definitions. The only pipeline override
is its provenance function: replace the current inherited-source path in the
manifest with the archive's actual path and add the driver itself. The
pipeline's mathematical entry point and CLI arguments are unchanged.

Static inspection also checked the archive's changed `__file__` location.
Its derived `HERE` appears only in unused standalone CLI output defaults,
including the imported default in `direct_state_check`; it does not alter the
production calculation. All 18 manifest hashes match their named files. The
manifest includes the archive and driver, and excludes the current
`Code/ramond_branching_recursion/compute_target.py`, correctly identifying
which implementation was timed. This provenance review imported no module
and performed no numerical calculation or coefficient comparison.
