#!/usr/bin/env python3
"""Evaluate and audit NSRR / all-NS free factors on the saved five surfaces.

Only geometry is read from the old run. No Liouville value, Q, or fitted
normalization enters this calculation. Archived outputs are not modified.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path
import time

import numpy as np

from fixed_spin_free_plumbing import (
    charged_frame, charge_lattice_sum, characteristic_in_charge_frame,
    fixed_spin_partition, direct_charged_fock_sum,
)
from physical_free_plumbing_resummation import theta_physical_fermion_fredholm
from recompute_all_ns_reference import protected_hashes


ROOT = Path(__file__).resolve().parents[2]
SOURCE_BRANCH = ((0, 0), (0, 1))
TARGET_BRANCH = ((-1, -1), (-1, 0))


def serializable(value):
    if isinstance(value, complex):
        return str(value)
    if isinstance(value, dict):
        return {str(k): serializable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [serializable(v) for v in value]
    return value


def ns_identity_audit(q, branch):
    frame = charged_frame(q, max_mode=32)
    rows = []
    for e0, e1 in itertools.product((1, -1), repeat=2):
        lifts = (e0, e1, 1)
        value = theta_physical_fermion_fredholm(q, lifts, max_mode=32)
        raw = value.determinant_values[0]
        charge_spin = ((0, 0), (int(e0 < 0), int(e1 < 0)))
        marked_spin = characteristic_in_charge_frame(charge_spin, -np.asarray(branch))
        lattice = charge_lattice_sum(frame.omega_charge, charge_spin)
        error = abs(raw**2/(frame.boson_chiral*lattice)-1)
        rows.append({"lifts_geometry": lifts, "characteristic_charge": charge_spin,
                     "characteristic_marked": marked_spin,
                     "unfiltered_fredholm": raw, "filtered_legacy_block": value.chiral_value,
                     "complex_bosonization_relative_error": error})
    return {"rows": rows, "maximum_relative_error": max(r["complex_bosonization_relative_error"] for r in rows),
            "legacy_filter_identity": "(-D+++ + D-++ + D+-+ + D++-)/2",
            "conclusion": "The unfiltered determinants are fixed-spin Majorana blocks; their legacy filtered linear combination is not one fixed-spin partition."}


def run(geometry_path, output, *, b=1.4):
    start = time.monotonic()
    if not np.isfinite(b) or b <= 0:
        raise ValueError("b must be finite and positive")
    before = protected_hashes()
    payload = geometry_path.read_bytes()
    geometry = json.loads(payload)
    matrix = np.asarray(geometry["source_to_target"], dtype=int)
    source_spin = ((1, 1), (0, 0))
    target_spin = ((0, 0), (0, 0))
    from ns_genus2_partition import _transport_spin_characteristic
    if _transport_spin_characteristic(matrix, source_spin) != target_spin:
        raise ArithmeticError("wrong modular spin transport")
    rows = []
    for point in geometry["points"]:
        evaluations = []
        for key, spin, branch in (("source_chart", source_spin, SOURCE_BRANCH),
                                  ("target_chart", target_spin, TARGET_BRANCH)):
            chart = point[key]
            q = tuple(complex(x) for x in chart["q_values"])
            omega = np.array([[complex(z) for z in row] for row in chart["omega"]])
            values = [fixed_spin_partition(q, omega, spin, period_branch=branch, max_mode=n)
                      for n in (16, 24, 32)]
            result = values[-1]
            result["mode_sweep"] = [{"max_mode": n, "Z_free": v["Z_free"]}
                                    for n, v in zip((16, 24, 32), values)]
            result["mode_relative_change_24_to_32"] = abs(values[1]["Z_free"]/result["Z_free"]-1)
            result["all_NS_bosonization_audit"] = ns_identity_audit(q, branch)
            if result["mode_relative_change_24_to_32"] > 1e-10:
                raise ArithmeticError("free factor not mode converged")
            if result["all_NS_bosonization_audit"]["maximum_relative_error"] > 1e-10:
                raise ArithmeticError("independent NS Fredholm check failed")
            evaluations.append(result)
        source, target = evaluations
        direct_rows = []
        exact_dirac = source["boson_chiral"]*source["dirac_charge_sum"]
        for level in (6, 10, 14):
            direct = direct_charged_fock_sum(source["q_values"], source["characteristic_charge"],
                                             total_level=level, lattice_cutoff=4)
            direct["relative_error_against_resummation"] = abs(direct["dirac_chiral"]/exact_dirac-1)
            direct_rows.append(direct)
        if direct_rows[-1]["relative_error_against_resummation"] > 1e-10:
            raise ArithmeticError("independent Ramond Fock sewing failed")
        source["direct_Ramond_Fock_sweep"] = direct_rows
        # A scalar has c=1; the superfield has c=3/2. This tests the
        # plumbing-frame anomaly without assuming raw Z_source=Z_target.
        scalar_ratio = source["Z_boson"]/target["Z_boson"]
        free_ratio = source["Z_free"]/target["Z_free"]
        anomaly_error = abs(free_ratio/scalar_ratio**1.5-1)
        if anomaly_error > 1e-8:
            raise ArithmeticError("free modular/frame check failed")
        row = {"t": point["t"], "source_NSrr": source, "target_NSnsns": target,
               "free_frame_ratio": free_ratio, "scalar_frame_ratio": scalar_ratio,
               "free_modular_frame_relative_error": anomaly_error,
               "Q_NSrr": None, "Q_status": "Liouville nonchiral Ramond assembly remains separate; no Q computed here"}
        rows.append(row)
        print(f"t={point['t']:.2f}: free NSRR={source['Z_free']:.12f}, "
              f"free NSNSNS={target['Z_free']:.12f}, frame error={anomaly_error:.3e}", flush=True)
    if protected_hashes() != before:
        raise ArithmeticError("protected kernels changed")
    dependencies = [Path(__file__), Path(__file__).with_name("fixed_spin_free_plumbing.py"),
                    Path(__file__).with_name("physical_free_plumbing_resummation.py"),
                    ROOT/"Code/genus_2_cross_channel/free_boson_pair_of_pants.py",
                    ROOT/"Code/genus_2_cross_channel/free_boson_plumbing.py"]
    result = {"schema": "fixed-spin-free-NSRR-plumbing-v1", "b": b,
              "c_SL": 1.5+3*(b+1/b)**2, "kappa": 1+2*(b+1/b)**2,
              "geometry_path": str(geometry_path.resolve()),
              "geometry_sha256": hashlib.sha256(payload).hexdigest(),
              "protected_kernel_sha256": before,
              "implementation_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in dependencies},
              "conventions": {"edge_order": ["zero", "one", "infinity"],
                              "source_sectors": ["R", "R", "NS"],
                              "human_slot_sectors": ["NS", "R", "R"],
                              "scalar_measure": "da_zero da_one, h(a)=a^2/2; unit connected zero-mode volume",
                              "propagator": "q^L0; common edge Casimir powers stripped",
                              "free_formula": "det(2 Im Omega_charge)^(-1/2) |P(q)|^3 |theta_delta|",
                              "Ramond_ground": "2 |q_zero q_one|^(1/8) for the nonchiral Majorana in the joint small-q limit",
                              "physical_spin": "explicit marked characteristic; no unproved identification of package Ramond parity lifts",
                              "legacy_outputs": "unchanged; legacy filtered NS denominator must not be reused as a fixed-spin determinant"},
              "source_to_target": matrix.tolist(), "points": rows,
              "elapsed_seconds": time.monotonic()-start}
    output.mkdir(parents=True, exist_ok=True)
    (output/"summary.json").write_text(json.dumps(serializable(result), indent=2, allow_nan=False)+"\n")
    with (output/"free_factors.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("Re_original_Omega12", "Z_free_NSrr_source_frame", "Z_free_NSnsns_target_frame", "free_frame_relative_error"))
        for row in rows:
            writer.writerow((row["t"], row["source_NSrr"]["Z_free"], row["target_NSnsns"]["Z_free"],
                             row["free_modular_frame_relative_error"]))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry", type=Path, default=ROOT/"Data Set/nsnsns_recompute_fivepoint_R16_N5_20260830/source_geometry_audit.json")
    parser.add_argument("--output", type=Path, default=ROOT/"Data Set/fixed_spin_free_NSrr_20260830")
    parser.add_argument("--b", type=float, default=1.4)
    args = parser.parse_args()
    run(args.geometry, args.output, b=args.b)


if __name__ == "__main__":
    main()
