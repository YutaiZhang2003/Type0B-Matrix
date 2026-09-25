"""Isolate momentum-quadrature convergence at FIXED sewing/moduli cutoffs.

No amplitude proposal is imported. The logarithmic-map control is independent
of the weighted endpoint/tail coordinate maps. These are numerical changes,
not rigorous error bounds or a certification of the complete string amplitude.
"""

import argparse
from dataclasses import replace
import json
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np

import spin23_singlet_amplitudes as s


FIELDS = ("ssvv_raw", "ssss_raw", "v_to_vss_raw")
CHANNELS = ("original_s", "original_t", "swapped_s", "swapped_t")


def pair(value):
    return [float(value.real), float(value.imag)]


def evaluate(atlas, energies, order):
    return s._evaluate_at_order(atlas, energies, order=order, epsilon0=0.30, epsilon1=0.24,
                               theta_orders=(8, 8, 24), radial_order=12,
                               disk_total_order=20, crossed_disk_total_order=20,
                               include_v_to_vss=True)


def select_region(atlas, lower, upper):
    return replace(atlas, **{
        name: replace(getattr(atlas, name), kernels=tuple(
            k for k in getattr(atlas, name).kernels if lower < k.momentum < upper))
        for name in CHANNELS
    })


def check(order=5):
    omega = 1/9 + 0.2j
    energies = (omega, omega, omega, 3*omega)
    rows, saved = [], {}
    cases = (
        ("coarse", (8, 32, 16), {}),
        ("medium", (12, 64, 24), {}),
        ("fine", (16, 96, 32), {}),
        ("endpoint_half", (16, 96, 32), {"endpoint": 0.09}),
        ("tail_earlier", (16, 96, 32), {"tail": 2.1}),
        ("tail_later", (16, 96, 32), {"tail": 4.5}),
        ("envelope_changed", (16, 96, 32), {"a": 1.6, "s": 0.9}),
        ("endpoint_unfactored_control", (24, 96, 32), {"beta": 0.0}),
        ("logarithmic_control", 128, None),
    )
    fine_atlas = None
    for label, counts, options in cases:
        started = time.perf_counter()
        scheme = "infinite_gauss" if options is None else "threshold_weighted"
        atlas = s._build_atlas(energies, q_order=order, p_nodes=counts, p_max=0, p_cut=0,
                               gram_condition_limit=1e13, momentum_scheme=scheme,
                               momentum_threshold_options=options, infinite_gauss_scale=0.5)
        values = evaluate(atlas, energies, order)
        saved[label] = {name: complex(getattr(values, name)) for name in FIELDS}
        row = {"label": label, "quadrature": atlas.momentum_quadrature,
               "amplitudes": {name: pair(value) for name, value in saved[label].items()},
               "elapsed_seconds": time.perf_counter()-started}
        if options is not None and options.get("tail", 3.2) == 3.2:
            tail = evaluate(select_region(atlas, 3.2, np.inf), energies, order)
            row["tail_above_3p2"] = {name: pair(getattr(tail, name)) for name in FIELDS}
        rows.append(row)
        print(json.dumps({"completed": label, "elapsed_seconds": row["elapsed_seconds"]}), flush=True)
        if label == "fine":
            fine_atlas = atlas
    for row in rows:
        row["relative_change_from_fine"] = {
            name: abs(value-saved["fine"][name])/max(abs(saved["fine"][name]), 1e-300)
            for name, value in saved[row["label"]].items()
        }

    region_rows = []
    for label, lo, hi in (("endpoint", 0, .18), ("bulk", .18, 3.2), ("tail", 3.2, np.inf)):
        values = evaluate(select_region(fine_atlas, lo, hi), energies, order)
        region_rows.append({"region": label, **{name: pair(getattr(values, name)) for name in FIELDS}})
    region_sum_error = {name: abs(sum(complex(*row[name]) for row in region_rows)-saved["fine"][name])
                        for name in FIELDS}
    assert max(region_sum_error.values()) < 1e-11

    # Independent tail-only map. Filtering nodes of a GLOBAL quadrature at
    # P=3.2 would NOT give a valid tail quadrature. Instead shift the entire
    # logarithmic half-line map before constructing the block atlas.
    tail_p, tail_w = s._momentum_quadrature(96, 0, 0, scheme="infinite_gauss", infinite_gauss_scale=0.5)
    with patch.object(s, "_momentum_quadrature", return_value=(3.2+tail_p, tail_w)):
        tail_atlas = s._build_atlas(energies, q_order=order, p_nodes=96, p_max=0, p_cut=0,
                                   gram_condition_limit=1e13, momentum_scheme="infinite_gauss")
    tail_control_values = evaluate(tail_atlas, energies, order)
    fine_tail = next(row for row in rows if row["label"] == "fine")["tail_above_3p2"]
    for row in rows:
        if "tail_above_3p2" in row:
            row["tail_relative_change_from_fine"] = {
                name: abs(complex(*row["tail_above_3p2"][name])-complex(*fine_tail[name]))/max(abs(complex(*fine_tail[name])), 1e-300)
                for name in FIELDS
            }
    tail_control = {"scheme": "P=3.2-0.5*log((1-x)/2), Gauss-Legendre", "nodes": 96,
                    "amplitudes": {name: pair(getattr(tail_control_values, name)) for name in FIELDS},
                    "relative_change_from_fine_tail": {
                        name: abs(getattr(tail_control_values, name)-complex(*fine_tail[name]))/max(abs(complex(*fine_tail[name])), 1e-300)
                        for name in FIELDS}}

    # Inspect endpoint power in the complete fixed-z spectral density, not in
    # an isolated block or in one structure constant. No threshold is fitted.
    endpoint_rows = []
    z = np.asarray([.42+.13j])
    for kernel in fine_atlas.original_s.kernels[:4]:
        data = replace(fine_atlas.original_s, kernels=(replace(kernel, quadrature_weight=1.0),))
        row = {"P": kernel.momentum}
        for process in ("ssvv", "ssss", "v_to_vss"):
            value = s._grid_integral(data, z, np.ones(1), original_z=z,
                                     q=None, theta3=None, order=order, process=process)
            row[process] = pair(value)
            row[process+"_over_P2"] = pair(value/kernel.momentum**2)
        endpoint_rows.append(row)
    powers = {process: float(np.polyfit(np.log([r["P"] for r in endpoint_rows[:3]]),
                          np.log([abs(complex(*r[process])) for r in endpoint_rows[:3]]), 1)[0])
              for process in ("ssvv", "ssss", "v_to_vss")}
    return {
        "status": "fixed_sewing_and_moduli_momentum_diagnostic_only",
        "series_parameter": "sewing", "sewing_order": order, "omega": pair(omega),
        "fixed_moduli_options": {"epsilon0": .30, "epsilon1": .24, "theta_orders": [8, 8, 24],
                                 "radial_order": 12, "disk_total_order": 20, "crossed_disk_total_order": 20},
        "measure": "dP/pi, unchanged", "rows": rows, "momentum_region_contributions": region_rows,
        "region_sum_absolute_errors": region_sum_error,
        "independent_tail_control": tail_control,
        "endpoint_fixed_z": pair(z[0]), "endpoint_probe": endpoint_rows,
        "observed_endpoint_log_slopes_first_three_nodes": powers,
        "warning": "The thresholds and envelope are numerical controls, not a contour continuation. Passing momentum refinement does not establish sewing-level or moduli convergence, nor validity at a pinched threshold.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sewing-order", type=int, default=5)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("choose a fresh report path")
    report = check(args.sewing_order)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"output": str(args.output), "status": report["status"],
                      "relative_changes": {r["label"]: r["relative_change_from_fine"] for r in report["rows"]}}, indent=2))


if __name__ == "__main__":
    main()
