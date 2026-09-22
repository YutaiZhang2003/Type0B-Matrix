#!/usr/bin/env python3
"""Recombine saved genus-two data with explicit coefficients and primary powers.

This is a conditional comparison of recorded sewing prescriptions, not a
derivation/certification of the interacting fixed-spin Ramond matrix. No
conformal-block node, structure constant, or free determinant is recomputed.
"""
from __future__ import annotations

import argparse
import ast
import cmath
import csv
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
from pathlib import Path

from nsrr_plumbing_adapter import GEOMETRY_SECTORS, NSRRPlumbingInputs
from physical_nsrr_sewing import CHANNELS, SOURCE_FIXED_SPIN_LIFTS, contract_physical_blocks


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "Data Set"
BASE = DATA / "nsrr_double_virasoro_N7_L5_20260911"
SCAN = DATA / "nsrr_threshold_order_scan_20260911"
FINE = DATA / "nsrr_momentum_integration_20260911/half_N12"
RAW = DATA / "nsrr_factorized_sign_trial_L3_N5_20260830"
LEDGER = DATA / "nsrr_coefficient_primary_ledger_20260915"
DEFAULT_OUTPUT = DATA / "genus2_saved_coefficient_comparison_20260915"


def object_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def csum(values):
    values = list(values)
    return complex(math.fsum(z.real for z in values), math.fsum(z.imag for z in values))


def decode(value):
    return complex(value) if isinstance(value, str) else complex(float(value[0]), float(value[1]))


def encode(value):
    return [float(value.real), float(value.imag)]


def relative(a, b):
    return abs(a - b) / max(abs(a), abs(b), 1e-300)


def scalar(value):
    return math.fsum(value) if isinstance(value, list) else float(value)


class Inputs:
    def __init__(self):
        self.files = {}

    def bytes(self, path):
        path = Path(path)
        content = path.read_bytes()
        key = str(path.relative_to(ROOT))
        digest = hashlib.sha256(content).hexdigest()
        assert self.files.get(key, digest) == digest, f"input changed during reduction: {path}"
        self.files[key] = digest
        return content

    def read(self, path):
        return json.loads(self.bytes(path))


def primary_record(b, q_values, momenta, channel):
    q = tuple(complex(v) for v in q_values)
    bg = b + 1 / b
    h = tuple(bg * bg / 8 + p * p / 2 + (1 / 16 if channel == "source" and e < 2 else 0)
              for e, p in enumerate(momenta))
    logs = tuple(cmath.log(v) for v in q)
    log_primary = sum(w * ell for w, ell in zip(h, logs))
    result = dict(weights_geometry=list(h), log_q_geometry=[encode(v) for v in logs],
                  log_abs_primary=log_primary.real,
                  unwrapped_primary_phase_rad=log_primary.imag,
                  primary=encode(cmath.exp(log_primary)))
    if channel == "source":
        plumbing = NSRRPlumbingInputs(q, (1, 1, 1), GEOMETRY_SECTORS)
        assert relative(decode(result["primary"]), plumbing.primary(b, momenta)) < 2e-13
    return result


def read_grid(inputs, directory, config):
    manifest = inputs.read(directory / "manifest.json")
    summary = inputs.read(directory / "summary.json")
    assert summary["manifest"] == manifest
    assert manifest["base_config_digest"] == object_digest(config)
    pids = [p["point_id"] for p in config["points"]]
    assert manifest["point_ids"] == pids
    digest = object_digest(manifest)
    n = manifest["order"]
    sums = defaultdict(list)
    nodes = {}
    primary_examples = []
    for channel in ("source", "target"):
        files = sorted((directory / channel).glob("node-*.json"))
        assert [p.name for p in files] == [f"node-{i:05d}.json" for i in range(n ** 3)]
        levels = manifest["source_levels" if channel == "source" else "target_twice_levels"]
        rules = manifest["nodes_and_weights"][channel]
        for i, path in enumerate(files):
            node = inputs.read(path)
            assert (node["manifest_digest"], node["channel"], node["index"]) == (digest, channel, i)
            indices = (i // n ** 2, (i // n) % n, i % n)
            assert node["momenta"] == [rules[e]["nodes"][j] for e, j in enumerate(indices)]
            assert relative(node["measure"], math.prod(rules[e]["weights"][j] for e, j in enumerate(indices))) < 2e-15
            assert [r["point_id"] for r in node["rows"]] == pids
            for point, row in zip(config["points"], node["rows"]):
                assert set(row["values"]) == set(map(str, levels))
                for level in levels:
                    value = scalar(row["values"][str(level)])
                    assert math.isfinite(value)
                    sums[channel, point["point_id"], level].append(node["measure"] * value)
                # These are expected primary factors, not recovered chiral phases.
                if i in (0, n ** 3 // 2, n ** 3 - 1):
                    primary_examples.append(dict(grid=directory.name, channel=channel, index=i,
                        point_id=point["point_id"], momenta_geometry=node["momenta"],
                        **primary_record(config["b"], point[channel]["q_values"], node["momenta"], channel)))
            nodes[channel, i] = node
    totals = {key: math.fsum(values) for key, values in sums.items()}
    assert len(summary["rows"]) == len(totals)
    error = max(relative(row["Z"], totals[row["channel"], row["point_id"], row["level"]])
                for row in summary["rows"])
    assert error < 2e-14
    return totals, nodes, dict(directory=str(directory.relative_to(ROOT)), nodes_per_channel=n ** 3,
                              manifest_digest=digest, max_saved_grid_relative_error=error,
                              primary_examples=primary_examples)


def high_order_comparison(inputs, config):
    fine, _, fine_check = read_grid(inputs, FINE, config)
    coarse, nodes, coarse_check = read_grid(inputs, SCAN / "grid_N7", config)
    saved = inputs.read(SCAN / "summary.json")
    reference = {p["point_id"]: p for p in saved["points"]}
    rows = []
    max_reproduction_error = 0.0
    for point in config["points"]:
        pid = point["point_id"]
        for order in (5, 6, 7, 8):
            z = {}
            for channel, control in (("source", 3), ("target", 0)):
                level = order if channel == "source" else 2 * order
                terms = []
                j = [p["point_id"] for p in config["points"]].index(pid)
                for i in range(7 ** 3):
                    node = nodes[channel, i]
                    values = node["rows"][j]["values"]
                    terms.append(node["measure"] * (scalar(values[str(level)]) - scalar(values[str(control)])))
                z[channel] = fine[channel, pid, control] + math.fsum(terms)
                assert relative(z[channel], fine[channel, pid, control] + coarse[channel, pid, level]
                                - coarse[channel, pid, control]) < 2e-14
                error = relative(z[channel], reference[pid][channel][str(order)]["Z"])
                max_reproduction_error = max(max_reproduction_error, error)
                assert error < 2e-14
            sq = z["source"] / point["source"]["Z_free"] ** config["kappa"]
            tq = z["target"] / point["target"]["Z_free"] ** config["kappa"]
            rows.append(dict(point_id=pid, source_total_level=order, target_null_twice_level=2 * order,
                source_Z_saved=z["source"], source_Z_local_candidate=z["source"] / 4,
                target_Z=z["target"], source_free=point["source"]["Z_free"],
                target_free=point["target"]["Z_free"], source_Q_saved=sq,
                source_Q_local_candidate=sq / 4, target_Q=tq,
                ratio_saved=sq / tq, ratio_local_candidate=sq / (4 * tq),
                discrepancy_saved_percent=100 * (sq / tq - 1),
                discrepancy_local_candidate_percent=100 * (sq / (4 * tq) - 1)))
    steps = []
    for point in config["points"]:
        lookup = {r["source_total_level"]: r for r in rows if r["point_id"] == point["point_id"]}
        steps.append(dict(point_id=point["point_id"],
            source_L7_to_L8_relative=lookup[8]["source_Z_saved"] / lookup[7]["source_Z_saved"] - 1,
            target_null7_to_null8_relative=lookup[8]["target_Z"] / lookup[7]["target_Z"] - 1))
    return dict(rows=rows, order_steps=steps, max_saved_summary_relative_error=max_reproduction_error,
                grids=[fine_check, coarse_check],
                estimator="I_N12[f_control]+I_N7[f_order-f_control]; source control L3, target control null0",
                phase_status="Complex block vectors were not retained here; a chiral phase cannot be reconstructed.")


def interference_comparison(inputs, config):
    digest = object_digest(config)
    actual = inputs.read(BASE / "depth5_config.json")
    saved = inputs.read(BASE / "depth5_summary.json")
    reference = {r["point_id"]: r for r in saved["comparisons"] if r["momentum_order"] == 7}
    sums = defaultdict(list)
    check_error = 0.0
    for i in range(7 ** 3):
        source = inputs.read(BASE / "source/shards" / f"node-{i:03d}.json")
        target = inputs.read(BASE / "target_depth5/shards" / f"node-{i:03d}.json")
        assert source["config_digest"] == target["config_digest"] == digest
        assert source["task_index"] == source["node"] == target["node"] == i
        assert source["channel"] == "source" and source["quadrature_order"] == 7
        assert target["audit_implementation_sha256"] == actual["audit_implementation_sha256"]
        pids = [p["point_id"] for p in config["points"]]
        assert [r["point_id"] for r in source["values"]] == pids
        design = {(r["point_id"], r["null_twice_level"], r["endpoint_cap"]): r for r in target["rows"]}
        assert len(design) == len(target["rows"]) == 20
        assert set(design) == {(pid, 10, cap) for pid in pids for cap in (8, 10)}
        for row in source["values"]:
            pid = row["point_id"]
            d, interference = row["diagonal_Z_unscaled_node"], row["interference_Z_unscaled_node"]
            check_error = max(check_error, relative(d + interference, row["source_Z_unscaled_M_node"]),
                              relative((d + interference) / 4, row["source_Z_local_M_over_4_node"]))
            sums[pid, "diagonal"].append(source["measure"] * d / 4)
            sums[pid, "interference"].append(source["measure"] * interference / 4)
            for cap in (8, 10):
                # These endpoint-audit shards already include the momentum measure.
                sums[pid, f"target{cap}"].append(math.fsum(design[pid, 10, cap]["sector_contributions"]))
    assert check_error < 5e-14
    rows = []
    for point in config["points"]:
        pid = point["point_id"]
        d, interference, target, target10 = [math.fsum(sums[pid, k])
                                            for k in ("diagonal", "interference", "target8", "target10")]
        sf = point["source"]["Z_free"] ** config["kappa"]
        tf = point["target"]["Z_free"] ** config["kappa"]
        denominator = sf * target / tf
        assert relative((d + interference) * 4 / sf, reference[pid]["source_Q"]) < 2e-14
        assert relative(target / tf, reference[pid]["target_Q"]) < 2e-14
        rows.append(dict(point_id=pid, source_total_level=5, target_null_twice_level=10,
                         momentum_order=7, diagonal_Z_local=d, interference_Z_local=interference,
                         source_Z_local_candidate=d + interference, source_Z_saved=4 * (d + interference),
                         target_Z=target, interference_fraction=interference / (d + interference),
                         ratio_local_candidate=(d + interference) / denominator,
                         ratio_opposite_interference=(d - interference) / denominator,
                         ratio_diagonal_only=d / denominator,
                         ratio_saved=4 * (d + interference) / denominator,
                         target_endpoint8_to_10_relative=target10 / target - 1))
    return dict(rows=rows, max_node_sum_relative_error=check_error,
                sign_variants_status="Diagnostics at fixed coefficients; no variant chosen by agreement.")


def matrix_entries(inputs):
    content = inputs.bytes(LEDGER / "legacy_candidate_M.csv").decode()
    factors = {"1/16": 1 / 16, "I/16": 1j / 16, "-I/16": -1j / 16}
    entries = []
    for row in csv.DictReader(content.splitlines()):
        a, b = ast.literal_eval(row["A"]), ast.literal_eval(row["B"])
        assert row["includes_primary_powers"] == "False"
        assert a[1:] == b[1:]
        factor = factors[row["coefficient_of_constant_product"]]
        k = a[1] * a[2]
        assert factor == ((1 if a[0] == b[0] else (-1j * k if a[0] == 0 else 1j * k)) / 16)
        entries.append((CHANNELS.index(a), CHANNELS.index(b), row["constant_product"], factor))
    assert len(entries) == len({(a, b) for a, b, _, _ in entries}) == 16
    return entries


def contract_entries(entries, vector, constants):
    # The two equal theta pants have the same supplied (E,O); retain E_L*O_R
    # and O_L*E_R separately and do not replace analytic products by |C|^2.
    e, o = constants
    products = {"E_L*E_R": e * e, "E_L*O_R": e * o, "O_L*E_R": o * e, "O_L*O_R": o * o}
    groups = defaultdict(list)
    diagonal, interference = [], []
    for a, b, label, factor in entries:
        value = vector[a] * products[label] * factor * vector[b].conjugate()
        groups[label].append(value)
        (diagonal if a == b else interference).append(value)
    return ({k: csum(v) for k, v in groups.items()}, csum(diagonal), csum(interference))


def raw_block_comparison(inputs, config):
    summary = inputs.read(RAW / "summary.json")
    raw_config = summary["config"]
    assert raw_config["b"] == config["b"]
    assert raw_config["channels"] == [list(v) for v in CHANNELS]
    digest = object_digest(raw_config)
    free_path = DATA / "fixed_spin_free_NSrr_20260830/summary.json"
    target_path = DATA / "nsrr_nsnsns_target_R8_R12_R16_N5_20260830/summary.json"
    free = {p["t"]: p for p in inputs.read(free_path)["points"]}
    target_summary = inputs.read(target_path)
    target = {r["t"]: r for r in target_summary["rows"]
              if r["quadrature_order"] == 5 and r["recursion_order"] == 16}
    target_charts = {p["t"]: p["charts"]["target_nsnsns"]
                     for p in target_summary["config"]["baseline_config"]["points"]}
    old = inputs.read(DATA / "nsrr_nsnsns_normalized_sewing_recheck_20260903/summary.json")
    old_rows = {r["t"]: r for r in old["comparisons"]}
    entries = matrix_entries(inputs)
    files = sorted((RAW / "shards").glob("node-*.json"))
    tasks = [(n, i) for n in raw_config["quadrature_orders"] for i in range(n ** 3)]
    assert len(files) == len(tasks)
    expected_rows = set(itertools.product((p["t"] for p in raw_config["points"]),
                                          raw_config["levels"], map(tuple, raw_config["lifts_geometry"])))
    sums = defaultdict(list)
    primary_rows = []
    checks = dict(selected_nodes=0, complex_matrix_contractions=0, max_primary_relative_error=0.0,
                  max_primary_phase_error_rad=0.0, max_primary_inside_outside_relative_error=0.0,
                  max_matrix_vs_modulus_relative_error=0.0, max_matrix_vs_legacy_relative_error=0.0,
                  max_analytic_constant_product_imaginary=0.0)
    for index, (path, task) in enumerate(zip(files, tasks)):
        shard = inputs.read(path)
        assert shard["index"] == index and shard["config_digest"] == digest
        assert (shard["quadrature_order"], shard["node"]) == task
        assert shard["momenta_slots"] == shard["momenta_geometry"][::-1]
        if task[0] != 5:
            continue
        checks["selected_nodes"] += 1
        by_row = {(r["t"], r["level"], tuple(r["lifts_geometry"])): r for r in shard["rows"]}
        assert len(by_row) == len(shard["rows"]) and set(by_row) == expected_rows
        constants = tuple(decode(v) for v in shard["C_BRY"])
        for point in raw_config["points"]:
            t = point["t"]
            assert point["q_geometry"] == free[t]["source_NSrr"]["q_values"]
            assert target_charts[t]["q_values"] == free[t]["target_NSnsns"]["q_values"]
            selected = [by_row[t, 3.0, lift] for lift in SOURCE_FIXED_SPIN_LIFTS]
            primaries = [decode(r["primary"]) for r in selected]
            assert primaries[0] == primaries[1]
            record = primary_record(config["b"], point["q_geometry"], shard["momenta_geometry"], "source")
            primary = decode(record["primary"])
            p_error = relative(primary, primaries[0])
            phase_error = abs(cmath.phase(primary / primaries[0]))
            checks["max_primary_relative_error"] = max(checks["max_primary_relative_error"], p_error)
            checks["max_primary_phase_error_rad"] = max(checks["max_primary_phase_error_rad"], phase_error)
            # F remains descendant-only, including the two-lift projection.
            vector = [csum(decode(row["blocks"][a]) for row in selected) / math.sqrt(2)
                      for a in range(len(CHANNELS))]
            groups, d, interference = contract_entries(entries, vector, constants)
            propagated = [csum(decode(row["primary"]) * decode(row["blocks"][a]) for row in selected)
                          / math.sqrt(2) for a in range(len(CHANNELS))]
            inside_groups, _, _ = contract_entries(entries, propagated, constants)
            outside = abs(primary) ** 2 * csum(groups.values())
            inside = csum(inside_groups.values())
            coeff = {1: constants[0], -1: constants[1]}
            modulus = abs(primary) ** 2 * csum(coeff[eta] * coeff[etap] / 16 *
                abs(vector[CHANNELS.index((0, eta, etap))] + 1j * eta * etap *
                    vector[CHANNELS.index((1, eta, etap))]) ** 2
                for eta in (1, -1) for etap in (1, -1))
            legacy = contract_physical_blocks(dict(zip(CHANNELS, propagated)), constants)["total"]
            checks["complex_matrix_contractions"] += 1
            for key, error in (("max_primary_inside_outside_relative_error", relative(outside, inside)),
                               ("max_matrix_vs_modulus_relative_error", relative(outside, modulus)),
                               ("max_matrix_vs_legacy_relative_error", relative(outside, legacy))):
                checks[key] = max(checks[key], error)
            checks["max_analytic_constant_product_imaginary"] = max(
                checks["max_analytic_constant_product_imaginary"],
                *(abs((a * b).imag) for a in constants for b in constants))
            scale = shard["measure"] * abs(primary) ** 2
            for label, value in groups.items():
                sums[t, label].append(scale * value)
            sums[t, "diagonal"].append(scale * d)
            sums[t, "interference"].append(scale * interference)
            primary_rows.append(dict(t=t, node=shard["node"], momenta_geometry=shard["momenta_geometry"],
                weights_geometry=record["weights_geometry"], primary_real=primary.real,
                primary_imag=primary.imag, primary_abs=abs(primary), primary_phase_rad=cmath.phase(primary),
                saved_primary_relative_error=p_error, saved_primary_phase_error_rad=phase_error))
    assert checks["selected_nodes"] == 125 and checks["complex_matrix_contractions"] == 625
    assert all(value < 5e-13 for key, value in checks.items() if key.startswith("max_") and "imaginary" not in key)
    rows = []
    products = ["E_L*E_R", "E_L*O_R", "O_L*E_R", "O_L*O_R"]
    for point in raw_config["points"]:
        t = point["t"]
        groups = {label: csum(sums[t, label]) for label in products}
        z = csum(groups.values())
        d, interference = csum(sums[t, "diagonal"]), csum(sums[t, "interference"])
        assert relative(z, d + interference) < 2e-14
        assert relative(z, old_rows[t]["source_Z_L3_N5"]) < 5e-13
        sf, tf = free[t]["source_NSrr"]["Z_free"], free[t]["target_NSnsns"]["Z_free"]
        tq = target[t]["target_Z"] / tf ** config["kappa"]
        sq = z / sf ** config["kappa"]
        row = dict(t=t, source_level=3, source_momentum_order=5, target_null_twice_level=16,
                   target_momentum_order=5, source_Z_local_real=z.real, source_Z_local_imag=z.imag,
                   source_Z_saved_candidate_real=4 * z.real, target_Z=target[t]["target_Z"],
                   source_Q_local_real=sq.real, source_Q_local_imag=sq.imag, target_Q=tq,
                   ratio_norm_local=abs(sq / tq), ratio_norm_saved=4 * abs(sq / tq),
                   nonchiral_ratio_phase_rad=cmath.phase(sq / tq),
                   interference_fraction=(interference / z).real,
                   ratio_norm_opposite_interference=abs((d - interference) / sf ** config["kappa"] / tq),
                   source_free=sf, target_free=tf)
        for label, value in groups.items():
            row[label + "_Z_real"] = value.real
            row[label + "_Z_imag"] = value.imag
        rows.append(row)
    return dict(rows=rows, primary_rows=primary_rows, checks=checks,
                phase_status="Complex arithmetic retained through integration. Near-zero nonchiral phase is not a chiral Majorana phase test.")


def free_phase_control(inputs):
    audit = inputs.read(DATA / "superconformal_block_conventions_20260915/conventions.json")
    rows = []
    for row in audit["free_superfield_complex"]["rows"]:
        reference = decode(row["free_chiral"])
        primary = decode(row["primary_prefactor"])
        f0, f1 = [decode(row["descendant_blocks"][str(f)]) for f in (0, 1)]
        even, combination = primary * f0, primary * (f0 + 1j * f1) / 2
        for label, value in (("even_projected", even), ("legacy_combination", combination)):
            assert relative(value, decode(row[label]["chiral_value"])) < 2e-14
            ratio = value / reference
            rows.append(dict(q_scale=row["q_scale"], total_PBW_order=row["total_PBW_order"],
                block=label, norm_ratio=abs(ratio), phase_difference_degrees=math.degrees(cmath.phase(ratio)),
                complex_relative_error=abs(ratio - 1)))
    return dict(rows=rows, theory=audit["free_superfield_complex"]["theory"],
                scope="Recomputed ratios from saved complex free-field amplitudes. This is a separate phase calibration, not an interacting kernel derivation.")


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def report(summary):
    high = summary["high_order"]
    rows = [r for r in high["rows"] if r["source_total_level"] == 8]
    worst = max(abs(r["ratio_saved"] - 1) for r in rows)
    lines = ["# Genus-two partition functions recombined from saved data", "", "Date: 2026-09-15.", "",
        "## Result", "",
        f"The saved order-eight prescription reproduces the ten cross-channel ratios with a maximum discrepancy of **{100 * worst:.6f}%**. "
        "Applying the coefficient ledger's local candidate matrix gives exactly one quarter of the saved NSRR source. "
        "Its ratio to the all-NS target is approximately 0.25. This factor is a coefficient normalization, not a missing primary power or a Majorana square-root sign.", "",
        "Both source prescriptions remain conditional: the ordered three-point vertex tensor has not yet been contracted with a verified interacting fixed-spin Ramond pairing to establish the final M_AB. Numerical agreement does not select that pairing. The target retains its recorded interacting spin prescription as well.", "",
        "No new block nodes were evaluated, and no normalization was fitted.", "",
        "## Coefficients and conformal weights used", "",
        r"The block is descendant-only: $F_A=\sum_N F_A[N]q^N$. We use", "",
        r"$$Z_\gamma=\int\frac{d^3p}{\pi^3}\,|P_\gamma|^2 F_\gamma^T M_\gamma\overline{F_\gamma},\qquad P_\gamma=\exp\!\left(\sum_e h_e^\gamma\Log q_e^\gamma\right).$$", "",
        r"At fixed NSRR momenta all eight $(f,\eta,\eta')$ blocks share these primary weights. In geometry order $(0,1,\infty)$, the source weights are $(h_R(p_0),h_R(p_1),h_{NS}(p_\infty))$ and the target weights are all NS, at the target's own momenta and plumbing parameters. Here $h_{NS}=Q^2/8+p^2/2$, $h_R=h_{NS}+1/16$, and $Q=b+b^{-1}$.", "",
        r"For each $(\eta,\eta')$, with $k=\eta\eta'$, the explicit candidate from the ledger is", "",
        r"$$M_{\rm local}^{(\eta,\eta')}=\frac{B_{L,\eta}B_{R,\eta'}}{16}\begin{pmatrix}1&-ik\\ik&1\end{pmatrix},\quad B_+=E,\quad B_-=O,\quad c_+=E/2,\quad c_-=O/2.$$",
        "", r"The saved source uses $M_{\rm saved}=4M_{\rm local}$. The two-lift combination is the recorded $(F_{+++}+F_{+-+})/\sqrt2$ in geometry order. The CSV matrix is contracted as $F^T M\bar F$; interchanging this with $F^\dagger M F$ without transposing M would reverse the interference sign.", "",
        "The EE, EO, OE and OO products are retained separately; the analytic three-point products are not replaced by absolute squares. The all-NS target uses the recorded conversion C_HN^(1)=i Ctilde_BRY, whose explicit sewing minus sign combines with i² to give the positive BRY odd coefficient product.", "",
        "## Order-eight comparison on ten surfaces", "",
        r"The comparison quantity is $\mathcal Q_\gamma=Z_\gamma/(Z_{\mathrm{free},\gamma})^\kappa$, with the same saved single-Majorana free superfield in each local chart, $b=1.4$ and $\kappa=9.940408163265307$. The raw Z values belong to different plumbing Weyl frames and are listed separately in the CSV.", "",
        "| Surface | Q source, saved M | Q target | Saved ratio | Local candidate ratio |", "| --- | ---: | ---: | ---: | ---: |"]
    for r in rows:
        lines.append(f'| {r["point_id"]} | {r["source_Q_saved"]:.10e} | {r["target_Q"]:.10e} | {r["ratio_saved"]:.9f} | {r["ratio_local_candidate"]:.9f} |')
    lines += ["", "The integral is reconstructed from 1,728 control nodes per channel and 343 correction nodes per channel:", "",
        r"$$Z_s(L)=I_{12}[f_{s,3}]+I_7[f_{s,L}-f_{s,3}],\qquad Z_t(L)=I_{12}[f_{t,0}]+I_7[f_{t,2L}-f_{t,0}].$$", "",
        "Here L is the source total descendant cutoff; 2L is the target's twice-null-level cutoff, with endpoint cap 8 and a resummed middle chain. These are different truncations. The scalar node values already contain primary propagation; no extra q^h is applied during this reduction.", "",
        f'The largest saved L7-to-L8 change is {max(abs(r["source_L7_to_L8_relative"]) for r in high["order_steps"]):.3e} relative in the source and {max(abs(r["target_null7_to_null8_relative"]) for r in high["order_steps"]):.3e} in the target. These changes do not bound the momentum-integration error or establish the spin prescription.', "",
        "## Interference-sign comparison", "",
        "The complete L5/N7 run retains diagonal D and interference I separately. All entries below use the same local coefficient normalization. Flipping I is a diagnostic, not a choice made by closeness to one.", "",
        "| Surface | I/(D+I) | Ratio (D+I) | Ratio (D−I) |", "| --- | ---: | ---: | ---: |"]
    for r in summary["interference"]["rows"]:
        lines.append(f'| {r["point_id"]} | {r["interference_fraction"]:.7f} | {r["ratio_local_candidate"]:.9f} | {r["ratio_opposite_interference"]:.9f} |')
    lines += ["", "## Direct complex block recombination", "",
        "A separate five-surface L3/N5 run retains all eight complex descendant blocks, both required lifts, the complex primary and both supplied constants. We reconstruct the 16 nonzero matrix entries directly from legacy_candidate_M.csv. The target is the saved N5/twice-null-level-16 result; its raw numerator is divided by the corrected saved fixed-spin free factor, not its obsolete stored Q.", "",
        "| t | Candidate source Z | Target Z | Norm ratio of Q | Phase of nonchiral ratio (rad) |", "| ---: | ---: | ---: | ---: | ---: |"]
    for r in summary["raw_blocks"]["rows"]:
        lines.append(f'| {r["t"]:.2f} | {r["source_Z_local_real"]:.10e} | {r["target_Z"]:.10e} | {r["ratio_norm_local"]:.9f} | {r["nonchiral_ratio_phase_rad"]:.3e} |')
    checks = summary["raw_blocks"]["checks"]
    lines += ["", f'Across {checks["complex_matrix_contractions"]} complex contractions, the maximum relative difference between primary-outside and propagated-amplitude evaluation is **{checks["max_primary_inside_outside_relative_error"]:.3e}**. Reconstructed primary phases agree with the saved phases within **{checks["max_primary_phase_error_rad"]:.3e} radians**. The matrix and explicit modulus formulas agree within **{checks["max_matrix_vs_modulus_relative_error"]:.3e} relative**.', "",
        "The integrated nonchiral phase is numerically zero. This tests the contraction's reality, not the phase of a chiral single-Majorana partition function: a common chiral phase cancels against its conjugate. Relative block phases remain in I and are retained above. The order-eight scalar files cannot recover discarded chiral phases.", "",
        "## Independent saved free-field phase control", "",
        "For the saved c=3/2 scalar-plus-one-Majorana calibration, multiply the descendant block by its own primary before comparing it with the branch-fixed free chiral answer. The highest saved PBW order at q scale 0.02 gives:", "",
        "| Combination | Chiral norm ratio | Phase difference (degrees) |", "| --- | ---: | ---: |"]
    control = [r for r in summary["free_phase_control"]["rows"] if r["q_scale"] == 0.02]
    for r in control:
        if r["total_PBW_order"] == max(v["total_PBW_order"] for v in control):
            lines.append(f'| {r["block"]} | {r["norm_ratio"]:.12f} | {r["phase_difference_degrees"]:.9g} |')
    lines += ["", "This exposes a chiral phase mismatch of the legacy combination even though its nonchiral contraction is real. It does not determine the missing interacting sewing tensor.", "",
        "## Files and reproduction", "",
        "- [Order-five through order-eight values](high_order_comparison.csv)",
        "- [Diagonal and interference sums](interference_comparison.csv)",
        "- [Complex source values and EE/EO/OE/OO contributions](raw_block_comparison.csv)",
        "- [625 primary norm and phase checks](raw_primary_checks.csv)",
        "- [Saved free-field phase comparisons](free_phase_control.csv)",
        "- [Full result and checks](summary.json)",
        "- [Input SHA-256 manifest](provenance.json)", "",
        "```sh", "OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \\",
        "  /private/tmp/type0b-nsrr-smoke-venv/bin/python \\",
        "  Code/genus_2/recombine_saved_genus2_coefficient_ledger.py", "```", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    inputs = Inputs()
    inputs.bytes(Path(__file__))
    for path in ("Code/genus_2/nsrr_plumbing_adapter.py", "Code/genus_2/physical_nsrr_sewing.py",
                 "Code/genus_2/compare_nsrr_nsnsns_theta.py",
                 "Data Set/nsrr_momentum_integration_20260911/evaluator.py",
                 "Data Set/nsrr_double_virasoro_N7_L5_20260911/summarize_depth5.py"):
        inputs.bytes(ROOT / path)
    config = inputs.read(BASE / "config.json")
    assert relative(config["kappa"], 1 + 2 * (config["b"] + 1 / config["b"]) ** 2) < 1e-14
    inputs.read(LEDGER / "ledger.json")
    result = dict(schema="genus2-saved-coefficient-comparison-v1", status="complete_conditional_comparison",
                  completed_at_utc=datetime.now(timezone.utc).isoformat(), b=config["b"], kappa=config["kappa"],
                  physical_interacting_M_certified=False, normalization_fitted=False, block_nodes_evaluated=0)
    result["high_order"] = high_order_comparison(inputs, config)
    print("Reconstructed both high-order quadratures and all ten order-eight comparisons.", flush=True)
    result["interference"] = interference_comparison(inputs, config)
    result["raw_blocks"] = raw_block_comparison(inputs, config)
    result["free_phase_control"] = free_phase_control(inputs)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    for name, rows in (("high_order_comparison", result["high_order"]["rows"]),
                       ("interference_comparison", result["interference"]["rows"]),
                       ("raw_block_comparison", result["raw_blocks"]["rows"]),
                       ("raw_primary_checks", result["raw_blocks"].pop("primary_rows")),
                       ("free_phase_control", result["free_phase_control"]["rows"])):
        write_csv(output / f"{name}.csv", rows)
    (output / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    provenance = dict(input_file_count=len(inputs.files), sha256=dict(sorted(inputs.files.items())),
                      scope="Exact saved files used by this reduction; historical generator hashes are not certifications of their physics.")
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    (output / "README.md").write_text(report(result))
    rows = [r for r in result["high_order"]["rows"] if r["source_total_level"] == 8]
    print(json.dumps(dict(output=str(output), input_files=len(inputs.files),
        max_saved_reproduction_error=result["high_order"]["max_saved_summary_relative_error"],
        order8_saved_ratio_range=[min(r["ratio_saved"] for r in rows), max(r["ratio_saved"] for r in rows)],
        order8_local_ratio_range=[min(r["ratio_local_candidate"] for r in rows), max(r["ratio_local_candidate"] for r in rows)],
        complex_checks=result["raw_blocks"]["checks"]), indent=2))


if __name__ == "__main__":
    main()
