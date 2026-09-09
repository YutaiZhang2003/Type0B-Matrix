#!/usr/bin/env python3
"""Create/verify an immutable local cluster bundle; never submits a job."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(root):
    manifest = json.loads((root/"bundle_manifest.json").read_text())
    for relative, expected in manifest["sha256"].items():
        path = root/relative
        if not path.is_file() or sha(path) != expected:
            raise ValueError(f"frozen bundle mismatch: {relative}")
    return {"verified_files": len(manifest["sha256"]), "submitted": False}


def build(destination):
    import sympy
    import nsrr_trial_cluster as runtime
    config_path = ROOT/"Code/config/nsrr_trial_L5_N4_cluster_20260830.json"
    config = runtime.trial.load(config_path)
    runtime.preflight(config)
    if destination.exists():
        raise FileExistsError("choose a new bundle directory; snapshots are not overwritten")
    destination.mkdir(parents=True)
    files = set()
    for directory in ("c_Recursion", "genus_2", "full_ramond_block_runtime", "ramond_branching_recursion",
                      "double_virasoro/nsrr", "genus_2_cross_channel", "h_recursion"):
        files.update(p for p in (ROOT/"Code"/directory).rglob("*.py") if "__pycache__" not in p.parts)
    files.update([ROOT/"Code/__init__.py", config_path,
                  ROOT/"Code/genus_2/nsrr_checked_kernel_manifest.json", Path(__file__),
                  ROOT/"Code/cluster/nsrr_trial_L5_3h.slurm"])
    for source in sorted(files):
        target = destination/source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    stringmc = Path(runtime.trial.dv.reduced_virasoro_series.__globals__["ccy"].__file__).resolve().parent
    for name in ("ccy_genus2_block.py", "genus2_vacuum_blocks.py", "virasoro_plumbing_graph.py",
                 "plumbing_algorithms.py", "virasoro_blocks.py", "virasoro_descendant_algebra.py"):
        target = destination/"StringMC/plumbing"/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(stringmc/name, target)
    shutil.copytree(Path(sympy.__file__).resolve().parent, destination/"vendor/sympy",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    hashes = {str(p.relative_to(destination)): sha(p) for p in sorted(destination.rglob("*")) if p.is_file()}
    manifest = {"schema": "nsrr-trial-frozen-bundle-v1", "config_digest": runtime.trial.digest(config),
                "sympy_version": sympy.__version__, "sha256": hashes,
                "status": "prepared_not_submitted", "queue_time_included_in_three_hours": False}
    runtime.trial.save(destination/"bundle_manifest.json", manifest)
    return verify(destination)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "verify"))
    parser.add_argument("--bundle-root", type=Path, required=True)
    args = parser.parse_args()
    print(build(args.bundle_root.resolve()) if args.command == "build" else verify(args.bundle_root.resolve()))
