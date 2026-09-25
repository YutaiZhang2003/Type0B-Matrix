"""Finite-order sewing diagnostics; NOT a precision-certified amplitude scan."""

import argparse
import json
from pathlib import Path

import spin23_singlet_amplitudes as s
from validate_spin23_c_recursion import crossing_check


def check():
    omega = 1/9 + 0.2j
    energies = (omega, omega, omega, 3*omega)
    atlas = s._build_atlas(energies, q_order=9, p_nodes=32, p_max=0, p_cut=0,
                           momentum_scheme="infinite_gauss", infinite_gauss_scale=0.5,
                           gram_condition_limit=1e13)
    rows = []
    previous = None
    for order in (5, 7, 9):
        values = s._evaluate_at_order(atlas, energies, order=order, epsilon0=0.30, epsilon1=0.24,
                                     theta_orders=(12, 12, 48), radial_order=20,
                                     disk_total_order=24, crossed_disk_total_order=24,
                                     include_v_to_vss=True)
        row = {"sewing_order": order}
        for key in ("ssvv_raw", "ssss_raw", "v_to_vss_raw"):
            value = getattr(values, key)
            row[key] = [value.real, value.imag]
            if previous is not None:
                row[key + "_relative_order_change"] = abs(value - getattr(previous, key)) / max(abs(value), 1e-300)
        rows.append(row)
        previous = values
    return {
        "series_parameter": "sewing", "q_definition": "q_s=z; q_t=1-z",
        "bulk_chart_selection": "s for abs(z)<=abs(1-z); t otherwise",
        "status": "diagnostic_only_not_precision_certified",
        "omega": [omega.real, omega.imag],
        "momentum_nodes": 32, "momentum_scheme": "infinite_gauss", "infinite_gauss_scale": 0.5,
        "epsilon0": 0.30, "epsilon1": 0.24, "theta_orders": [12, 12, 48], "radial_order": 20,
        "disk_total_order": 24, "crossed_disk_total_order": 24,
        "amplitude_order_sweep": rows,
        "spectral_crossing": crossing_check(q_order=7),
        "warning": "The sewing cutoff is not an elliptic-nome cutoff. No old q8 accuracy claim is inherited; momentum/moduli refinement and larger-level or additional sewing charts may be needed.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("choose a fresh report path")
    report = check()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
