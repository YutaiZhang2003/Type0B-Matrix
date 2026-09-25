#!/usr/bin/env python3
"""Cannon coefficient banks and fixed-z crossing gate for SO(7) x E8.

This campaign deliberately stops at the crossing gate. A successful
super-Liouville gate is necessary, but does not itself certify the old
heterotic GSO/PCO assembler. Requested amplitude tasks are recorded in the
manifest and cannot be dispatched through this verification driver.
"""
import argparse
from dataclasses import asdict
from functools import lru_cache
import hashlib
from itertools import product
import json
import math
import os
from pathlib import Path
import time

import numpy as np

from heterotic_so23_1to3_vvvv_fit_bundle.liouville_momentum_quadrature import threshold_weighted_rule
from literature_component_blocks import SECTORS, crossing_phase, sewn_integrand
from literature_double_virasoro import LiteratureDoubleVirasoroBlocks
from literature_self_dual_blocks import (
    ExtrapolatedLiteratureBlocks, ExtrapolationOptions, analytic_antiholomorphic,
)
from literature_self_dual_correlator import SelfDualConstants


ROOT = Path(__file__).resolve().parents[1]


def pairs(value):
    a = np.asarray(value, complex)
    return np.stack((a.real, a.imag), axis=-1).tolist()


def unpairs(value):
    a = np.asarray(value, float)
    return a[..., 0]+1j*a[..., 1]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")
    os.replace(temporary, path)


def chiral_components(family):
    # Physical middle R+/- fields are vertex-operator labels, not w+/- bits.
    return tuple(e for e in product((0, 1), repeat=4)
                 if e[1] == 0 and (family == "mixed_ns" or e[2] == 0))


def sign_pairs(family):
    return tuple(product((1,) if family == "mixed_ns" else (1, -1), (1, -1)))


def physical_components(family):
    if family == "rrrr":
        return tuple(product((0, 1), repeat=4))
    return tuple((r1, r2, (a, ab), (d, db))
                 for r1, r2, a, ab, d, db in product((0, 1), repeat=6))


class FrozenBlocks(ExtrapolatedLiteratureBlocks):
    """Read a complete coefficient table; no recursion or inferred entries."""
    def __init__(self, family, momenta, P, table):
        table = np.asarray(table, complex)
        expected = (len(chiral_components(family)), 2, len(sign_pairs(family)))
        if table.ndim != 4 or table.shape[:3] != expected or not np.isfinite(table).all():
            raise ValueError("invalid parity/sign coefficient bank")
        self.table = table
        self.indices = {e: j for j, e in enumerate(chiral_components(family))}
        super().__init__(family, momenta, P, table.shape[-1]-1)

    def _native_components(self, external, parity):
        return self.table[self.indices[external], parity]


def coefficient_table(blocks):
    return np.array([[blocks._native_components(e, k) for k in (0, 1)]
                     for e in chiral_components(blocks.family)])


def block_values(blocks, points):
    return np.array([[[[blocks.value(complex(z), e, k, sl, sr) for z in points]
                       for sl, sr in sign_pairs(blocks.family)] for k in (0, 1)]
                     for e in chiral_components(blocks.family)])


def adjacent_change(previous, current, *, zero_floor=1e-12):
    """Componentwise relative change; retain a documented numerical-zero floor."""
    previous, current = np.asarray(previous, complex), np.asarray(current, complex)
    if previous.shape != current.shape or not np.isfinite([previous, current]).all():
        raise ValueError("matching finite block values are required")
    scale = max(float(np.max(abs(previous))), float(np.max(abs(current))), 1e-300)
    denominator = np.maximum(np.maximum(abs(previous), abs(current)), zero_floor*scale)
    changes = abs(current-previous)/denominator
    worst = np.unravel_index(np.argmax(changes), changes.shape)
    return dict(maximum_relative_change=float(changes[worst]), worst_index=list(map(int, worst)),
                zero_floor_relative_to_largest_block=zero_floor,
                nearly_zero_entries=int(np.sum(np.maximum(abs(previous), abs(current)) < zero_floor*scale)))


def read_manifest(run, *, check_sources=True):
    run = Path(run)
    manifest = json.loads((run/"manifest.json").read_text())
    signature = manifest.pop("manifest_sha256")
    if digest(manifest) != signature:
        raise ValueError("manifest integrity failure")
    manifest["manifest_sha256"] = signature
    if check_sources:
        for name, expected in manifest["source_sha256"].items():
            if hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != expected:
                raise ValueError(f"source changed since preparation: {name}")
    return manifest


def _matching_record(path, manifest, task):
    data = json.loads(Path(path).read_text())
    if data["manifest_sha256"] != manifest["manifest_sha256"] or data["task"] != task:
        raise ValueError(f"checkpoint belongs to different sources/settings: {path}")
    if digest(data["payload"]) != data["payload_sha256"]:
        raise ValueError(f"checkpoint payload integrity failure: {path}")
    return data["payload"]


def _save_record(path, manifest, task, payload):
    atomic_json(path, dict(manifest_sha256=manifest["manifest_sha256"], task=task,
                           payload=payload, payload_sha256=digest(payload)))


def build_node(run, task_id, *, manifest=None):
    run = Path(run)
    manifest = manifest or read_manifest(run)
    task = manifest["bank_tasks"][task_id]
    final_path = run/"banks"/f"{task_id:04d}.json"
    if final_path.exists():
        return _matching_record(final_path, manifest, task)
    config = manifest["config"]
    family, P = task["family"], task["P"]
    momenta = tuple(unpairs(task["momenta"]))
    points = unpairs(task["points"])
    options = ExtrapolationOptions(config["extrapolation"]["epsilon"])
    previous = None
    history = []
    started = time.monotonic()
    for order in range(config["blocks"]["baseline_order"], config["blocks"]["maximum_order"]+1):
        checkpoint = run/"checkpoints"/f"{task_id:04d}_q{order:02d}.json"
        if checkpoint.exists():
            payload = _matching_record(checkpoint, manifest, task)
            table = unpairs(payload["coefficients"])
        else:
            tables = []
            for pp in (momenta, tuple(p.conjugate() for p in momenta)):
                blocks = ExtrapolatedLiteratureBlocks(family, pp, P, 2*order, options=options)
                tables.append(coefficient_table(blocks))
            table = np.array(tables)
            payload = dict(order=order, maximum_twice_level=2*order,
                           coefficients=pairs(table), seconds=time.monotonic()-started)
            _save_record(checkpoint, manifest, task, payload)
        current = np.array([block_values(FrozenBlocks(family, pp, P, tab), points)
                            for pp, tab in zip((momenta, tuple(p.conjugate() for p in momenta)), table)])
        change = None if previous is None else adjacent_change(previous, current)
        history.append(dict(order=order, adjacent=change))
        converged = change is not None and change["maximum_relative_change"] <= config["blocks"]["relative_tolerance"]
        if converged or order == config["blocks"]["maximum_order"]:
            payload.update(status="converged_on_crossing_points" if converged else "order_cap_unconverged",
                           history=history, seconds=time.monotonic()-started,
                           convergence_domain="fixed crossing points only; not the moduli integration grid",
                           external_chiral_components=chiral_components(family),
                           structure_sign_pairs=sign_pairs(family),
                           extrapolation=config["extrapolation"])
            _save_record(final_path, manifest, task, payload)
            return payload
        previous = current
        print(f"node {task_id}: order {order} complete", flush=True)
    raise AssertionError("adaptive order loop must terminate")


def _correlator_rows(manifest, run, energy, observable, channel):
    tasks = [(j, t) for j, t in enumerate(manifest["bank_tasks"])
             if (t["energy"], t["observable"], t["channel"]) == (energy, observable, channel)]
    if len(tasks) != manifest["config"]["momentum"]["nodes_per_channel"]:
        raise ValueError("incomplete or duplicate momentum rule")
    exts = physical_components(observable)
    if channel == "t":
        exts = tuple((e[2], e[1], e[0], e[3]) for e in exts)
    points = unpairs(tasks[0][1]["points"])
    sums = np.zeros((len(exts), len(points)), complex)
    constants = SelfDualConstants(manifest["config"]["precision"])
    statuses, orders = [], []
    for j, task in tasks:
        payload = _matching_record(Path(run)/"banks"/f"{j:04d}.json", manifest, task)
        table = unpairs(payload["coefficients"])
        p = tuple(unpairs(task["momenta"]))
        blocks = FrozenBlocks(task["family"], p, task["P"], table[0])
        dual = FrozenBlocks(task["family"], tuple(x.conjugate() for x in p), task["P"], table[1])
        for a, external in enumerate(exts):
            for b, z in enumerate(points):
                sums[a, b] += task["dP_weight"]*sewn_integrand(
                    blocks, constants, task["P"], complex(z), external,
                    antiholomorphic_value=analytic_antiholomorphic(dual))
        statuses.append(payload["status"])
        orders.append(payload["order"])
    return sums, statuses, orders


def reduce_crossing(run, *, energies=None):
    manifest = read_manifest(run)
    rows, all_statuses = [], []
    names = energies or list(manifest["energies"])
    for energy, observable in product(names, ("mixed", "rrrr")):
        lhs, s_status, s_orders = _correlator_rows(manifest, run, energy, observable, "s")
        rhs, t_status, t_orders = _correlator_rows(manifest, run, energy, observable, "t")
        all_statuses += s_status+t_status
        sectors = SECTORS["rrrr" if observable == "rrrr" else "mixed_ns"]
        scale = max(float(np.max(abs(lhs))), float(np.max(abs(rhs))), 1e-300)
        for j, e in enumerate(physical_components(observable)):
            phase = crossing_phase(sectors, e)
            parity = sum(x if s == "R" else sum(x) for s, x in zip(sectors, e)) % 2
            for k, z in enumerate(manifest["config"]["crossing"]["points"]):
                x, y = lhs[j, k], phase*rhs[j, k]
                error = abs(x-y)/max(abs(x), abs(y), scale*1e-12)
                rows.append(dict(energy=energy, observable=observable, external=e, z=z,
                                 phase=pairs(phase), lhs=pairs(x), transported_rhs=pairs(y),
                                 relative_error=float(error), vanishes_by_parity=bool(parity),
                                 zero_residual=float(max(abs(x), abs(y))/scale) if parity else None,
                                 s_orders=s_orders, t_orders=t_orders))
    nonzero = [row for row in rows if not row["vanishes_by_parity"]]
    worst = max(nonzero, key=lambda row: row["relative_error"])
    zeros_pass = all(row["zero_residual"] <= 1e-12 for row in rows if row["vanishes_by_parity"])
    blocks_pass = all(s == "converged_on_crossing_points" for s in all_statuses)
    crossing_pass = worst["relative_error"] <= manifest["config"]["crossing"]["relative_tolerance"] and zeros_pass
    complete = set(names) == set(manifest["energies"])
    report = dict(manifest_sha256=manifest["manifest_sha256"], energies=names,
                  status="passed" if complete and crossing_pass and blocks_pass else "not_verified",
                  complete_energy_scan=complete, crossing_pass=crossing_pass,
                  adjacent_block_orders_pass=blocks_pass, parity_zeros_pass=zeros_pass,
                  maximum_relative_error=worst["relative_error"], worst=worst, rows=rows,
                  integration="internal momentum at fixed z; NO moduli integration",
                  normalization="common external literature Upsilon factor stripped on both sides",
                  heterotic_amplitudes_enabled=False,
                  remaining_amplitude_gate="Transport and verify the heterotic GSO/PCO/free-field coefficient tensors in this same convention; pure SL crossing does not certify them.")
    name = "crossing.json" if complete else "crossing_"+"_".join(names)+".json"
    atomic_json(Path(run)/name, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("bank", "crossing"))
    parser.add_argument("--run", type=Path, default=Path("run"))
    parser.add_argument("--task", type=int)
    parser.add_argument("--energy", action="append")
    args = parser.parse_args()
    if args.stage == "bank":
        if args.task is None:
            parser.error("bank requires --task")
        result = build_node(args.run, args.task)
        print(json.dumps({k: result[k] for k in ("status", "order", "history", "seconds")}, indent=2))
    else:
        result = reduce_crossing(args.run, energies=args.energy)
        print(json.dumps({k: result[k] for k in ("status", "maximum_relative_error", "adjacent_block_orders_pass")}, indent=2))
        if result["status"] != "passed":
            raise SystemExit(2)


if __name__ == "__main__":
    main()
