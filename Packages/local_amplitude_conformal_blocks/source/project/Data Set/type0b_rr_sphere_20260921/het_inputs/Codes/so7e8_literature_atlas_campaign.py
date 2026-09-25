#!/usr/bin/env python3
"""Complete only missing or unresolved banks for the regional channel atlas.

The output certifies block orders on assigned bulk/lens points and labelled
disk-boundary probes. It does not certify momentum-quadrature errors, the
analytic disk integrals, physical GSO normalization, or an amplitude value.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from literature_self_dual_blocks import ExtrapolatedLiteratureBlocks, ExtrapolationOptions
from so7e8_literature_atlas import BlockGeometry, block_grid_values, pointwise_adjacent_change
from so7e8_literature_campaign import (
    FrozenBlocks, _matching_record, _save_record, coefficient_table,
    pairs, unpairs, read_manifest,
)


def load_geometry(run, manifest, task):
    path = Path(run)/manifest["point_sets_file"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["point_sets_sha256"]:
        raise ValueError("atlas point-set integrity failure")
    return BlockGeometry.build(unpairs(json.loads(path.read_text())[task["point_set"]]))


def load_seed(run, task):
    if task.get("seed_file") is None:
        return None
    path = Path(run)/task["seed_file"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != task["seed_sha256"]:
        raise ValueError("parent seed integrity failure")
    data = json.loads(path.read_text())
    parent_task = data["task"]
    if any(parent_task[k] != task[k] for k in ("family", "momenta", "P", "dP_weight")):
        raise ValueError("parent seed has incompatible native parameters")
    from so7e8_literature_campaign import digest
    if data["payload_sha256"] != digest(data["payload"]):
        raise ValueError("parent seed payload integrity failure")
    return data["payload"]


def build_atlas_node(run, index, *, manifest=None, refine=False):
    run = Path(run)
    manifest = manifest or read_manifest(run)
    task = manifest["bank_tasks"][index]
    final = run/"banks"/f"{index:04d}.json"
    if final.exists():
        existing = _matching_record(final, manifest, task)
        if not refine or existing["block_converged"]:
            return existing
    geometry = load_geometry(run, manifest, task)
    seed = load_seed(run, task)
    config = manifest["config"]
    settings = config["blocks"]
    p = tuple(unpairs(task["momenta"]))
    pp = (p, tuple(x.conjugate() for x in p))
    options = ExtrapolationOptions(config["extrapolation"]["epsilon"])
    history, previous = [], None
    started = time.monotonic()
    for order in range(settings["baseline_order"], settings["maximum_order"]+1):
        checkpoint = run/"checkpoints"/f"{index:04d}_q{order:02d}.json"
        if checkpoint.exists():
            table = unpairs(_matching_record(checkpoint, manifest, task)["coefficients"])
        elif seed is not None and seed["order"] >= order:
            table = unpairs(seed["coefficients"])[..., :2*order+1]
        else:
            table = np.array([coefficient_table(ExtrapolatedLiteratureBlocks(
                task["family"], momenta, task["P"], 2*order, options=options)) for momenta in pp])
        if table.shape[0] != 2 or table.shape[-1] != 2*order+1 or not np.isfinite(table).all():
            raise ValueError("invalid complete native table at this order")
        payload = dict(order=order, maximum_twice_level=2*order, coefficients=pairs(table))
        if not checkpoint.exists():
            _save_record(checkpoint, manifest, task, payload)
        current = np.array([block_grid_values(
            FrozenBlocks(task["family"], momenta, task["P"], tab), geometry)
            for momenta, tab in zip(pp, table)])
        change, resolved = None, False
        if previous is not None:
            errors = pointwise_adjacent_change(previous, current)
            worst = int(np.argmax(errors))
            failing = np.flatnonzero(errors > settings["relative_tolerance"])
            change = dict(maximum_relative_change=float(errors[worst]), worst_point_index=worst,
                worst_native_coordinate=pairs(geometry.points[worst]), points_tested=len(errors),
                points_failing=len(failing), failing_point_indices=failing.tolist())
            resolved = len(failing) == 0
        history.append(dict(order=order, adjacent=change))
        # Try a better channel for the whole correlator before advancing an
        # unstable unused channel. This first pass does not change the cap.
        selection_pending = not refine and order == settings["baseline_order"]+1
        if resolved or order == settings["maximum_order"] or selection_pending:
            status = ("order_resolved_on_assigned_points" if resolved else
                      "order_cap_unconverged" if order == settings["maximum_order"] else
                      "awaiting_regional_channel_selection")
            payload.update(status=status,
                block_converged=resolved, history=history, seconds=time.monotonic()-started,
                point_set=task["point_set"], points_checked=len(geometry.points),
                convergence_domain="assigned bulk/lens points and disk-boundary probes only",
                analytic_disk_integral_checked=False, extrapolation=config["extrapolation"])
            _save_record(final, manifest, task, payload)
            return payload
        previous = current
        print(f"atlas node {index}: physical order {order} complete", flush=True)
    raise AssertionError("the adaptive loop must terminate")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("bank", "refine", "reduce"))
    parser.add_argument("--run", type=Path, default=Path("run"))
    parser.add_argument("--task", type=int)
    args = parser.parse_args()
    if args.stage in ("bank", "refine"):
        if args.task is None:
            parser.error("bank requires --task")
        result = build_atlas_node(args.run, args.task, refine=args.stage == "refine")
        print(json.dumps({k: result[k] for k in ("status", "order", "history", "seconds")}, indent=2))
    else:
        from resolve_so7e8_literature_atlas import reduce_regional_atlas
        result = reduce_regional_atlas(args.run)
        print(json.dumps({k: result[k] for k in ("status", "total_channel_node_assignments", "unresolved_count")}, indent=2))
        if not result["moduli_grid_adjacent_orders_pass"]:
            raise SystemExit(2)
