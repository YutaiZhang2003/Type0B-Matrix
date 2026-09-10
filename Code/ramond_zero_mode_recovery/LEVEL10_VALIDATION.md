# Level-10 zero-mode recovery: results and runtime

The subsequently requested restricted-inverse benchmark is recorded
separately in [RESTRICTED_LEVEL10_RUNTIME.md](RESTRICTED_LEVEL10_RUNTIME.md).
It computed the level-10 physical block at 384-bit precision in 116.4794
seconds, enforcing the inverse's domain requirement and running no
independent cross-checks.

A later, explicitly requested comparison of the two saved complex physical
blocks passed through level 10 across all 4,048 coefficient slots. The
maximum absolute difference is 2.29682e-106 and the maximum scaled difference
is 3.1067634e-108. See
[RESTRICTED_VS_ZERO_MODE_LEVEL10.md](RESTRICTED_VS_ZERO_MODE_LEVEL10.md).

The zero-mode/Virasoro construction computed one complete physical block
through total level 10 at 384-bit complex precision in **299.1915 seconds**
(4 minutes 59.2 seconds). The subsequent high-precision direct physical PBW
comparison was **cancelled at the user's request**. No level-10 direct
physical PBW complex comparison was completed; the later comparison above
uses the saved restricted-inverse result instead.

## Computation and timing

The benchmark is
\(b=7/5\), \((P_1,P_2,P_3)=(11/23,13/29,17/31)\).
The timed block has \(p=f=0\), \((\eta,\eta')=(+,+)\), with the projector
on the first Ramond cut. It contains all 506 plumbing monomials through
total level 10, retaining eight parity components per monomial.

| Stage | Wall time (seconds) |
| --- | ---: |
| Projected auxiliary block | 3.1191 |
| Inserted enlarged numerator, including setup | 290.0748 |
| Recovery of the physical block | 5.9953 |
| **Total, including remaining setup overhead** | **299.1915** |

The numerator is computed from branching primary coefficients, ordinary
Virasoro descendant Ward identities, and the degenerate-field insertion
kernel. It is not constructed by multiplying the auxiliary block by a
previously known physical block. This timing excludes imports and the
subsequent validation suite. Other validation work overlapped part of the
run, so it is an observed runtime on this machine, not an uncontended
benchmark. Coefficients and timing are in
[complex_level10.cold.json](complex_level10.cold.json).

## Completed checks

| Check | Scope and result |
| --- | --- |
| Auxiliary inverse | Both Ramond cuts; all 506 monomials through level 10; exact residual zero modulo 65521. |
| Zero-mode identities | 1,806 anticommutator checks through level 10; ground-label action of D and the odd intertwiner identities pass. |
| Degenerate field | Both Virasoro level-two null identities pass exactly modulo 65521. |
| Insertion kernel | All 44 Ramond edge level/parity/cut matrices through level 10 obey BPZ symmetry and D squared equal to the identity, exactly modulo 65521. |
| Independent kernel construction | Degenerate Ward kernel equals the explicit oscillator change-of-basis kernel through level 3, on both cuts and in both parities. |
| Archived full modular sewing comparison | All 32 cases (16 physical choices, each computed on two cuts) pass coefficientwise through level 10, with zero convolution and recovery discrepancies. This run finished **before** the stop instruction. |

The archived modular comparison also implies agreement between the two
cut choices: both recover the same reference coefficients. Its total
runtime, including one cold modular block and all comparisons, was
944.5482 seconds. The separate auxiliary/zero-mode/null suite took
23.4169 seconds. These are validation runtimes, not the complex block's
computation time.

Results:
[modular_degenerate_level10.json](modular_degenerate_level10.json),
[zero_mode_identities_level10.json](zero_mode_identities_level10.json).

## Scope after stopping the direct physical route

No further direct physical sewing was run. Both high-order runners now
disable that comparison by default and reject direct-reference cutoffs
above level 3. Their high-order comparison is recovery using the two
different Ramond cuts. Optional low-level physical checks require an
explicit `--physical-reference-level` argument.

The changed default paths were checked at level zero with the direct
physical routine replaced by a function that raises if called. Both passed
all 32 cases and the 16 cut comparisons. The revised runners have not been
rerun through level 10; the completed level-10 evidence above comes from
the preserved pre-stop modular run.

To compute one complex block without a direct physical reference:

```sh
python3 Code/ramond_zero_mode_recovery/check_level10_complex.py \
  --level 10 --json /tmp/zero_mode_block_level10.json
```

Add `--all-cases` to compare the two cuts for all parity/sign choices.
The complex runner needs the dependencies in
[requirements-validation.txt](requirements-validation.txt), in addition
to the workspace's NumPy, SciPy, SymPy, and mpmath dependencies.

These checks concern the manuscript's theta channel at one generic
rational parameter point. Exact arithmetic at one prime is not a symbolic
proof for arbitrary parameters. Complex arithmetic uses 384-bit midpoints
and drops terms below 1e-80; it is not a certified interval computation.
The cancelled complex comparison supplies no final level-10 error bound.
The scripts remain a separate implementation and do not validate the older
binary64 production output or all plumbing channels.
