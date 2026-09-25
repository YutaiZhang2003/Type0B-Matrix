"""Choose one order-resolved channel at each prepared moduli query.

Selection is made for the entire 32-node correlator, never independently
per momentum node. All chiral variants and both analytic chiralities enter
the order diagnostic. Smaller |q| ranks candidates but cannot certify one.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

from so7e8_literature_atlas import (
    BlockGeometry, block_grid_values, channel_coordinates, pointwise_adjacent_change,
)
from so7e8_literature_campaign import (
    FrozenBlocks, _matching_record, atomic_json, pairs, read_manifest, unpairs,
)
from sphere_block_uniformization import elliptic_nome


def inverse_chart(w, observable, channel):
    if channel == "s":
        return w
    if channel == "t":
        return 1-w
    return 1-1/w if observable == "mixed" else 1/w


def channel_order_errors(records, points, *, baseline=4):
    """Maximum adjacent change across every node/component at each point."""
    if len(records) != 32 or {t["momentum_index"] for t, p in records} != set(range(32)):
        raise ValueError("exactly 32 distinct momentum nodes required")
    geometry = BlockGeometry.build(points)
    worst = np.zeros(len(geometry.points))
    orders = []
    for task, payload in records:
        table = unpairs(payload["coefficients"])
        order = int(payload["order"])
        if order <= baseline or table.shape[0] != 2 or table.shape[-1] != 2*order+1:
            raise ValueError("a complete table with two actual adjacent orders required")
        p = tuple(unpairs(task["momenta"]))
        values = []
        for n in (order-1, order):
            values.append(np.array([block_grid_values(
                FrozenBlocks(task["family"], pp, task["P"], tab[..., :2*n+1]), geometry)
                for pp, tab in zip((p, tuple(x.conjugate() for x in p)), table)]))
        worst = np.maximum(worst, pointwise_adjacent_change(*values))
        orders.append(order)
    return worst, orders


def resolve_energy(observable, point_sets, records, *, tolerance=.02):
    """Try alternatives only for points whose preferred chart fails.

    An override records its index in the original chart point set, retaining
    the original integration point and its measure. No point is discarded.
    """
    overrides, unresolved, channels = [], [], {}
    maximum_selected_change, maximum_selected_nome = 0., 0.
    for channel in ("s", "t", "u"):
        key = observable+"/"+channel
        native = np.asarray(point_sets[key], complex)
        errors, orders = channel_order_errors(records[channel], native)
        passing = errors <= tolerance
        if np.any(passing):
            maximum_selected_change = max(maximum_selected_change, float(max(errors[passing])))
            maximum_selected_nome = max(maximum_selected_nome, float(max(abs(elliptic_nome(native[passing])))))
        bad = np.flatnonzero(~passing)
        channels[channel] = dict(points=len(native), initially_resolved=int(sum(passing)),
            initially_unresolved=len(bad), orders=orders,
            maximum_preferred_change=float(max(errors)))
        if not len(bad):
            continue
        canonical = inverse_chart(native[bad], observable, channel)
        coordinates = channel_coordinates(canonical, observable)
        alternatives = {}
        for other in ("s", "t", "u"):
            if other != channel:
                change, other_orders = channel_order_errors(records[other], coordinates[other])
                alternatives[other] = (change, other_orders)
        for j, index in enumerate(bad):
            candidates = [c for c, (change, _) in alternatives.items() if change[j] <= tolerance]
            entry = dict(original_channel=channel, original_point_index=int(index),
                original_native_coordinate=pairs(native[index]), canonical_z=pairs(canonical[j]),
                preferred_change=float(errors[index]),
                alternative_changes={c: float(v[0][j]) for c, v in alternatives.items()})
            if candidates:
                chosen = min(candidates, key=lambda c: (abs(elliptic_nome(coordinates[c][j])), c))
                change, other_orders = alternatives[chosen]
                q = float(abs(elliptic_nome(coordinates[chosen][j])))
                maximum_selected_change = max(maximum_selected_change, float(change[j]))
                maximum_selected_nome = max(maximum_selected_nome, q)
                entry.update(selected_channel=chosen, selected_native_coordinate=pairs(coordinates[chosen][j]),
                             selected_change=float(change[j]), selected_nome=q, orders=other_orders)
                overrides.append(entry)
            else:
                entry.update(status="higher_orders_required_before_production")
                unresolved.append(entry)
    return dict(channels=channels, overrides=overrides, unresolved=unresolved,
        maximum_selected_change=maximum_selected_change, maximum_selected_nome=maximum_selected_nome,
        complete_order_resolved_coverage=not unresolved)


def reduce_regional_atlas(run):
    run = Path(run)
    manifest = read_manifest(run)
    parent = read_manifest(run/"parent", check_sources=False)
    if parent["manifest_sha256"] != manifest["parent_manifest_sha256"]:
        raise ValueError("wrong parent manifest")
    points_path = run/manifest["point_sets_file"]
    if hashlib.sha256(points_path.read_bytes()).hexdigest() != manifest["point_sets_sha256"]:
        raise ValueError("point-set integrity failure")
    point_sets = {key: unpairs(value) for key, value in json.loads(points_path.read_text()).items()}
    records = {}
    for index, task in enumerate(parent["bank_tasks"]):
        path = run/"parent/banks"/f"{index:04d}.json"
        if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["parent_bank_sha256"][str(index)]:
            raise ValueError("parent bank changed since preparation")
        payload = _matching_record(path, parent, task)
        records.setdefault((task["energy"], task["observable"], task["channel"]), []).append((task, payload))
    for index, task in enumerate(manifest["bank_tasks"]):
        payload = _matching_record(run/"banks"/f"{index:04d}.json", manifest, task)
        records.setdefault((task["energy"], task["observable"], task["channel"]), []).append((task, payload))
    rows = []
    for energy in manifest["energies"]:
        for observable in ("mixed", "rrrr"):
            row = resolve_energy(observable, point_sets,
                {c: records[energy, observable, c] for c in ("s", "t", "u")},
                tolerance=manifest["config"]["blocks"]["relative_tolerance"])
            row.update(energy=energy, observable=observable)
            rows.append(row)
            print(f"regional selection {energy}/{observable}: {len(row['overrides'])} switches, {len(row['unresolved'])} unresolved points", flush=True)
    complete = all(row["complete_order_resolved_coverage"] for row in rows)
    result = dict(schema="so7e8-literature-regional-selection-v1", manifest_sha256=manifest["manifest_sha256"],
        status="selected_blocks_order_resolved" if complete else "higher_orders_required",
        rows=rows, moduli_grid_adjacent_orders_pass=complete,
        complete_channel_node_coverage=True,
        total_channel_node_assignments=len(parent["bank_tasks"])+len(manifest["bank_tasks"]),
        maximum_selected_change=max(row["maximum_selected_change"] for row in rows),
        maximum_selected_nome=max(row["maximum_selected_nome"] for row in rows),
        override_count=sum(len(row["overrides"]) for row in rows),
        unresolved_count=sum(len(row["unresolved"]) for row in rows),
        assembled_overlap_checked=False, analytic_disks_checked=False,
        physical_amplitude_certified=False, moduli_integration_performed=False,
        scientific_settings=manifest["config"], geometry=manifest["geometry"],
        interpretation="choose a single resolved channel for all 32 nodes at a point; retain all branches and variants; unused unstable channels do not veto that choice")
    atomic_json(run/"atlas_completion.json", result)
    if not complete:
        atomic_json(run/"refinement_plan.json", dict(manifest_sha256=manifest["manifest_sha256"],
            maximum_order=manifest["config"]["blocks"]["maximum_order"],
            required=[dict(energy=row["energy"], observable=row["observable"], point=point)
                      for row in rows for point in row["unresolved"]]))
    return result
