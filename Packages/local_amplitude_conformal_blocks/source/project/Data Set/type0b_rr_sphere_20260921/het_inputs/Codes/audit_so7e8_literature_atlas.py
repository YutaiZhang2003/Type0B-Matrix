#!/usr/bin/env python3
"""Recheck reusable CCY banks on their assigned moduli patches.

No recursion, submission or moduli integration. Compatible native tables
are reused by family, momenta and P, independently of their old labels.
Missing third-channel banks are reported rather than replaced by s blocks.
All variants and both analytic chiralities are checked at every assigned
grid point. The OPE-disk boundary probes do not certify the disk integral.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

from so7e8_literature_atlas import (
    CHARTS, BlockGeometry, atlas_point_sets, block_grid_values,
    pointwise_adjacent_change,
)
from so7e8_literature_campaign import (
    ROOT, FrozenBlocks, _matching_record, atomic_json, digest, pairs,
    read_manifest, unpairs,
)


def native_key(family, momenta, P):
    return digest(dict(family=family, momenta=momenta, P=P))


def atlas_tasks(parent):
    available = {}
    for index, task in enumerate(parent["bank_tasks"]):
        available.setdefault(native_key(task["family"], task["momenta"], task["P"]), index)
    tasks = []
    for energy, momenta in parent["energies"].items():
        for observable in CHARTS:
            base = [momenta[j] for j in (0, 3, 2, 1)] if observable == "mixed" else momenta
            for channel, (family, sector, perm) in CHARTS[observable].items():
                pp = [base[j] for j in perm]
                rule = parent["momentum_rules"][sector]
                for j, (P, weight) in enumerate(zip(rule["momenta"], rule["dP_weights"])):
                    source = available.get(native_key(family, pp, P))
                    tasks.append(dict(energy=energy, observable=observable, channel=channel,
                        family=family, sector=sector, momenta=pp, momentum_index=j, P=P,
                        dP_weight=weight, point_set=observable+"/"+channel,
                        parent_bank=source))
    return tasks


def recheck_table(task, payload, geometry, settings):
    """Use only actual retained adjacent orders; a cap never means pass."""
    table = unpairs(payload["coefficients"])
    available_order = int(payload["order"])
    if table.ndim != 5 or table.shape[0] != 2 or table.shape[-1] != 2*available_order+1:
        raise ValueError("bank order does not match its complete two-chirality table")
    p = tuple(unpairs(task["momenta"]))
    previous = None
    history = []
    for order in range(settings["baseline_order"], min(available_order, settings["maximum_order"])+1):
        current = np.array([block_grid_values(
            FrozenBlocks(task["family"], pp, task["P"], tab[..., :2*order+1]), geometry)
            for pp, tab in zip((p, tuple(x.conjugate() for x in p)), table)])
        if previous is not None:
            changes = pointwise_adjacent_change(previous, current)
            worst = int(np.argmax(changes))
            failed = np.flatnonzero(changes > settings["relative_tolerance"])
            history.append(dict(order=order, maximum_relative_change=float(changes[worst]),
                worst_point_index=worst, worst_native_coordinate=pairs(geometry.points[worst]),
                points_tested=len(changes), points_failing=len(failed),
                failing_point_indices=failed.tolist()))
            if len(failed) == 0:
                return dict(status="order_resolved_on_assigned_points", block_converged=True,
                            order=order, history=history)
        previous = current
    return dict(status="order_cap_unconverged" if available_order >= settings["maximum_order"]
                else "higher_order_required", block_converged=False,
                order=available_order, history=history)


def audit(run, output):
    started = time.monotonic()
    run, output = Path(run).resolve(), Path(output).resolve()
    if run == output or run in output.parents:
        raise ValueError("write the atlas audit outside the frozen input run")
    # The parent archive is frozen; other research in the live workspace
    # can change unrelated modules. Check every imported parent source used
    # in this recheck rather than pretending the entire workspace is frozen.
    parent = read_manifest(run, check_sources=False)
    runtime_parent_hashes = {}
    for module in tuple(sys.modules.values()):
        path = getattr(module, "__file__", None)
        if not path:
            continue
        path = Path(path).resolve()
        if not path.is_relative_to(ROOT):
            continue
        name = str(path.relative_to(ROOT))
        if name in parent["source_sha256"]:
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual != parent["source_sha256"][name]:
                raise ValueError(f"recheck dependency changed since the parent release: {name}")
            runtime_parent_hashes[name] = actual
    if parent["config"]["momentum"]["nodes_per_channel"] != 32:
        raise ValueError("the current atlas requires the agreed 32-node parent")
    point_sets, geometry_receipt = atlas_point_sets(parent["config"]["moduli"]["settings"])
    geometries = {name: BlockGeometry.build(points) for name, points in point_sets.items()}
    tasks = atlas_tasks(parent)
    sources = [ROOT/"Codes"/name for name in (
        "so7e8_literature_atlas.py", "audit_so7e8_literature_atlas.py",
        "so7e8_literature_amplitude.py")]
    source_hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    input_signature = digest(dict(parent=parent["manifest_sha256"], sources=source_hashes,
        points={name: digest(pairs(points)) for name, points in point_sets.items()}))
    cache, rows = {}, []
    for index, task in enumerate(tasks):
        row = dict(task_index=index, task=task)
        source = task["parent_bank"]
        if source is None:
            row.update(status="missing_native_bank", block_converged=False)
        else:
            record_path = run/"banks"/f"{source:04d}.json"
            payload = _matching_record(record_path, parent, parent["bank_tasks"][source])
            record_sha = hashlib.sha256(record_path.read_bytes()).hexdigest()
            key = digest(dict(source=source, source_sha256=record_sha, points=task["point_set"], audit=input_signature))
            checkpoint = output/"rechecks"/(key+".json")
            if key not in cache:
                if checkpoint.exists():
                    saved = json.loads(checkpoint.read_text())
                    if saved["input_signature"] != key or saved["payload_sha256"] != digest(saved["payload"]):
                        raise ValueError("atlas recheck checkpoint integrity failure")
                    cache[key] = saved["payload"]
                else:
                    cache[key] = recheck_table(task, payload, geometries[task["point_set"]], parent["config"]["blocks"])
                    atomic_json(checkpoint, dict(input_signature=key, payload=cache[key], payload_sha256=digest(cache[key])))
            row.update(cache[key], parent_bank_sha256=record_sha)
        rows.append(row)
        if (index+1) % 128 == 0:
            print(f"atlas assignments checked: {index+1}/{len(tasks)}", flush=True)
    tested = [r for r in rows if r.get("history")]
    worst = max(tested, key=lambda r: r["history"][-1]["maximum_relative_change"])
    unique_missing = {native_key(t["family"], t["momenta"], t["P"]) for t in tasks if t["parent_bank"] is None}
    report = dict(schema="so7e8-literature-regional-atlas-audit-v1",
        checked_at=datetime.now(timezone.utc).isoformat(), input_run=str(run),
        parent_manifest_sha256=parent["manifest_sha256"], source_sha256=source_hashes,
        verified_runtime_parent_source_sha256=runtime_parent_hashes,
        input_signature=input_signature, geometry=geometry_receipt,
        point_set_sha256={name: digest(pairs(p)) for name, p in point_sets.items()},
        scientific_settings=parent["config"], rows=rows, summary=dict(
            total_channel_node_assignments=len(rows), reused_assignments=sum(t["parent_bank"] is not None for t in tasks),
            unique_rechecks=len(cache), unique_missing_banks=len(unique_missing),
            order_resolved_assignments=sum(r["block_converged"] for r in rows),
            higher_order_required_assignments=sum(r["status"] == "higher_order_required" for r in rows),
            order_cap_unconverged_assignments=sum(r["status"] == "order_cap_unconverged" for r in rows),
            missing_assignments=sum(r["status"] == "missing_native_bank" for r in rows),
            maximum_reused_adjacent_change=worst["history"][-1]["maximum_relative_change"], worst_task=worst["task_index"]),
        policy="prefer smallest |q| among locally order-resolved charts; unused unresolved channels do not veto the moduli integral; retain resolved overlap diagnostics",
        assembled_overlap_checked=False, analytic_disk_integrals_checked=False,
        moduli_integration_performed=False, new_jobs_submitted=False,
        elapsed_seconds=time.monotonic()-started)
    atomic_json(output/"atlas_audit.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.run, args.output)
    print(json.dumps(result["summary"], indent=2))
