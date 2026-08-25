# Code layout

The research code is grouped by its primary purpose:

- `c_Recursion/`: NS and Ramond central-charge recursion, its regular/global
  seeds, sphere and genus-two consumers, numerical drivers, and tests.
- `h_recursion/`: fixed-weight NS/R recursion, torus one- and two-point
  blocks, Ramond sphere blocks, correlator assembly, and tests.
- `full_ramond_block_runtime/`: certified NS--R--R double-Virasoro q-expansion.
- `ramond_branching_recursion/`: Yuchen's branching actions, Ward grid, and
  direct boundary implementation.
- `double_virasoro/`: active verification code, split into `all_ns/`, `nsrr/`,
  and `audits/`.
- `genus_2/`: parity-correct nonchiral Type-0B theta-channel assembly,
  including the sector-pairing sign and odd-null lift transport.
- `unused_human_note_computation/`: provenance-only legacy and exploratory
  computations excluded from the active import and test graph.
- `genus_2_cross_channel/`: the frozen earlier StringMC/SCFT genus-two
  cross-channel snapshot, including its own docs, configs, and cluster files.
- `cluster/` and `config/`: launchers and configurations shared by the active
  Type 0B calculations.

`baseline.json`, `benchmark_genus_one.py`, and
`test_benchmark_genus_one.py` stay at this level because they are common
project benchmarks rather than one of the five recursion/code families.

## Running code

Run from the repository root with `Code` on `PYTHONPATH`; this preserves the
historical basename imports across the purpose folders:

    PYTHONPATH=Code python3 Code/c_Recursion/ns_genus_c_recursion_checks.py
    PYTHONPATH=Code python3 Code/h_recursion/evaluate_superconformal_torus_block.py --help
    python3 Code/full_ramond_block_runtime/compute_q_expansion.py --cutoff 6 --direct-pbw-check
    PYTHONPATH=Code/double_virasoro/all_ns python3 -m unittest discover -s Code/double_virasoro/all_ns -p 'test_*.py'
    PYTHONPATH=Code/double_virasoro/nsrr python3 -m unittest discover -s Code/double_virasoro/nsrr -p 'test_*.py'
    python3 Code/double_virasoro/audits/audit_nsrr_boundary_star.py

The frozen genus-two snapshot retains same-directory imports and can also be
run from within `Code/genus_2_cross_channel/`, as described in its README.
