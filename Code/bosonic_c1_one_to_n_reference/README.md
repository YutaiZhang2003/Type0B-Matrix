# Bosonic c=1 `1 -> n` reference implementation

This subfolder is a frozen, independently runnable reference copy of the
bosonic `c=1` genus-zero and genus-one `1 -> n` worldsheet computations from
StringMC.  It is included in Type0B-Matrix as a numerical and architectural
reference for the superconformal calculation; it is **not** itself a Type 0B
amplitude implementation.

The copied snapshot was assembled on 25 August 2026 from StringMC source
revision `cfc786ea3c077407a336373d6990c2a17474499a`.  Its own
`MANIFEST.json` provides file-by-file SHA-256 integrity records.

## What this reference contains

The nested `reference_implementation/` directory contains the complete
standalone snapshot:

- sphere genus-zero `1 -> 2`, `1 -> 3`, `1 -> 4`, and `1 -> 5` code;
- torus genus-one `1 -> 1` and `1 -> 2` code;
- bosonic Virasoro `c`- and `h`-recursion implementations and checks;
- Liouville structure constants and momentum quadrature;
- necklace/OPE channel atlases, Weyl/frame factors, RQMC moduli integration,
  and direct large-`tau_2` tail integration;
- frozen target-blind datasets and separate post-freeze comparisons;
- a reproduction runbook, dependency list, check record, and integrity
  verifier.

Start with:

- `reference_implementation/README.md` for installation and scope;
- `reference_implementation/RUNBOOK.md` for the process-by-process commands;
- `reference_implementation/plumbing/sphere_one_to_n_amplitudes_machine_note.tex`;
- `reference_implementation/plumbing/torus_one_to_n_amplitudes_note.tex`.

## How to use it for the superconformal computation

The most reusable pieces are the numerical organization rather than the
bosonic CFT ingredients:

1. preserve the separation between target-blind worldsheet production and
   later matrix-model comparison;
2. reuse the adaptive channel-selection, atlas-overlap, RQMC, checkpoint,
   direct-tail, and hash-freeze patterns;
3. use the bosonic torus two- and three-point blocks as regression templates
   for normalization, nome conventions, conformal-frame factors, collision
   handling, and channel matching;
4. replace every bosonic Virasoro block by the appropriate NS/R
   superconformal block, with the correct spin-structure and parity sums;
5. replace the bosonic DOZZ/Yin constants by the Type 0B super-Liouville
   constants (`C`, `C_tilde`, `C_even`, and `C_odd`) in the convention ledger;
6. rederive the matter, ghost/superghost, PCO, zero-mode, and coupling factors
   in Type 0B conventions rather than copying the bosonic normalization.

In particular, the BRY dictionary stored in this reference is the bosonic
`c=1` dictionary.  It must not be imported as the Type 0B normalization.

## Particularly useful source maps

For torus block and integration architecture, inspect:

- `plumbing/torus_two_point_blocks.py`;
- `plumbing/genus1_two_point_worldsheet.py`;
- `plumbing/torus_three_point_blocks.py`;
- `plumbing/genus1_three_point_worldsheet.py`;
- `plumbing/genus1_three_point_channel_atlas.py`;
- `plumbing/smoke_genus1_three_point_channel_atlas.py`.

For sphere multi-point atlases and momentum contraction, inspect:

- `plumbing/ccy_sphere_four_point.py`;
- `plumbing/ccy_sphere_five_point.py`;
- `plumbing/ccy_sphere_six_point.py`;
- `plumbing/ccy_sphere_six_point_star.py`;
- `plumbing/sphere_six_point_atlas.py`;
- `plumbing/sphere_six_point_equal_energy.py`.

## Verification

Use Python 3.10 or newer:

```bash
cd reference_implementation
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python verify_manifest.py
bash run_core_checks.sh
```

The snapshot was independently extracted and its integrity verifier and core
numerical suite passed before transfer into Type0B-Matrix.
