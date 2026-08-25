# Standalone c=1 string 1->n amplitude computations

This directory is a self-contained copy of the current genus-zero and
genus-one `1 -> n` worldsheet computations.  It preserves the target-blind
worldsheet/post-freeze comparison boundary used in the machine notes.

It contains 101 process-specific Python files and 169
Python files after following every local import transitively.  Nothing in the
bundle imports code from the parent StringMC checkout.

## Current calculations

| Genus | Process | Current packaged result |
|---|---|---|
| 0 | sphere `1 -> 2` | normalization anchor and topology audit |
| 0 | sphere `1 -> 3` | blind thirty-point imaginary-ray scan and affine fit |
| 0 | sphere `1 -> 4` | blind thirty-point scan, quadratic fit, and residue continuation |
| 0 | sphere `1 -> 5` | blind thirty-point paired order-eight estimator and cubic fit |
| 1 | torus `1 -> 1` | frozen fifty-point hybrid-recursion scan and BRY fit |
| 1 | torus `1 -> 2` | frozen ten-point h-recursion-dominant channel-atlas scan |

## Install and verify

Use Python 3.10 or newer (Python 3.11 or 3.12 is recommended).

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
bash run_core_checks.sh
python verify_manifest.py
```

`run_core_checks.sh` can use another interpreter through
`PYTHON_BIN=/path/to/python`.  The checks run from the bundle root with
`PYTHONPATH` restricted to this bundle's `plumbing/` directory.

## Reproduction

See [RUNBOOK.md](RUNBOOK.md) for the worldsheet-only stages, frozen fits,
post-freeze comparisons, representative smoke commands, and cluster entry
points.  Production scans are intentionally not launched by the check suite:
their sample counts range from hours to a cluster campaign.

## Integrity

`MANIFEST.json` gives a SHA-256 digest and category for every packaged file.
`verify_manifest.py` verifies those hashes.  `CHECK_RESULTS.txt` records the
exact core checks run while this snapshot was built.
