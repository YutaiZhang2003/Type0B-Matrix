"""Reproducible coefficient timings and comparison with a frozen full amplitude.

Run from the repository root.  The optional amplitude report must be fresh
output of run_spin23_ssvv_equal_blind.py at precision_infinite_q8, y=0.2.
No amplitude proposal is imported or used to compute block coefficients.
Cold and warm Gram timings are kept separate; recursion need not outperform
an already cached low-order Gram calculation at every batch size.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

import spin23_singlet_amplitudes as singlet


def crossing_check(q_order=7):
    """Report sewing-series channel convergence; do not reuse nome error bounds."""
    energies = (0.11 + 0.15j, 0.17 + 0.16j, 0.23 + 0.18j, 0.51 + 0.49j)
    atlas = singlet._build_atlas(
        energies, q_order=q_order, p_nodes=48, p_max=0, p_cut=0,
        momentum_scheme="infinite_gauss", infinite_gauss_scale=0.5,
        gram_condition_limit=1e13,
    )
    rows = []
    for z in (0.42 + 0.13j, 0.53 + 0.10j):
        for process in ("ssvv", "ssss", "v_to_vss"):
            row = {"z": [z.real, z.imag], "process": process, "relative_differences": {}}
            for order in (q_order - 2, q_order):
                values = []
                for channel, coordinate in ((atlas.original_s, z), (atlas.original_t, 1 - z)):
                    coord = np.asarray([coordinate])
                    values.append(singlet._grid_integral(
                        channel, coord, np.ones(1), original_z=np.asarray([z]),
                        q=None, theta3=None, order=order, process=process,
                    ))
                row["relative_differences"][str(order)] = abs(values[0] - values[1]) / abs(values[0])
            rows.append(row)
    return {"momentum_nodes": 48, "momentum_scheme": "infinite_gauss", "scale": 0.5,
            "series_parameter": "sewing", "q_definition": "q_s=z; q_t=1-z",
            "improves_with_order": all(r["relative_differences"][str(q_order)] < r["relative_differences"][str(q_order - 2)] for r in rows),
            "within_2e_minus4": all(r["relative_differences"][str(q_order)] < 2e-4 for r in rows),
            "external_energies": [[w.real, w.imag] for w in energies], "rows": rows}


def validate(q_order=7, amplitude_report=None, include_crossing=False):
    weights = tuple(singlet.fast.h_of_p(w) for w in
                    (0.11 + 0.15j, 0.17 + 0.16j, 0.23 + 0.18j, 0.51 + 0.49j))
    momenta = np.asarray([0.006, 0.73, 2.1])
    names = tuple(singlet.WORD_PATTERNS)
    start = time.perf_counter()
    recursive = singlet._recursive_coefficient_table(weights, momenta, q_order, names)
    recursion_seconds = time.perf_counter() - start
    start = time.perf_counter()
    solvers, _ = singlet._gram_solvers(momenta, 2 * q_order + 1, condition_limit=1e13)

    def gram():
        return singlet._coefficient_table(
            weights, momenta, solvers, q_order, names,
            high_precision_condition=1e8, high_precision_digits=70,
            high_precision_max_momentum=0.18,
        )

    direct = gram()
    gram_cold_seconds = time.perf_counter() - start
    start = time.perf_counter()
    gram()
    gram_warm_seconds = time.perf_counter() - start
    start = time.perf_counter()
    singlet._recursive_coefficient_table(weights, momenta, q_order, names)
    recursion_warm_seconds = time.perf_counter() - start
    residuals = {
        name: float(max(np.max(abs(recursive.coefficients[name][b] - direct.coefficients[name][b])
                               / np.maximum(1, abs(direct.coefficients[name][b]))) for b in (0, 1)))
        for name in names
    }
    if max(residuals.values()) > 2e-10:
        raise AssertionError(residuals)
    report = {
        "series_parameter": "sewing",
        "q_order": q_order,
        "maximum_twice_level": 2 * q_order + 1,
        "external_weights": [[w.real, w.imag] for w in weights],
        "momenta": momenta.tolist(),
        "coefficient_error_scale": "abs(recursion-Gram)/max(1,abs(Gram))",
        "maximum_coefficient_errors_by_pattern": residuals,
        "timings_seconds": {
            "recursion_first": recursion_seconds, "recursion_repeated": recursion_warm_seconds,
            "gram_including_cold_templates": gram_cold_seconds,
            "gram_reusing_templates_and_solvers": gram_warm_seconds,
        },
        "recursion_digits": 70,
        "recursion_reference_p_max": 0.18,
        "recursion_cancellation_limit": 1e6,
        "high_precision_recursion_nodes": recursive.high_precision_recursion_node_count,
        "maximum_recursion_cancellation": recursive.maximum_recursion_cancellation,
        "high_precision_gram_solves": direct.high_precision_solve_count,
        "scope": "three-node coefficient batch, not the full momentum or moduli integral",
    }
    if amplitude_report is not None:
        frozen_path = Path(__file__).resolve().parent.parent / "Data Set/high_accuracy/spin23_ssvv_equal_blind.json"
        frozen = json.loads(frozen_path.read_text())
        fresh = json.loads(Path(amplitude_report).read_text())

        def row(source):
            profile = source["profiles"]["precision_infinite_q8"]
            return profile["settings"], next(r for r in profile["rows"] if r["im_omega"] == 0.2)

        old_settings, old = row(frozen)
        settings, new = row(fresh)
        numerical_keys = ("q_order", "lower_q_order", "p_nodes", "p_max", "p_cut",
                          "momentum_scheme", "infinite_gauss_scale", "epsilon0", "epsilon1",
                          "theta_orders", "radial_order", "disk_total_order", "crossed_disk_total_order")
        assert all(settings[k] == old_settings[k] for k in numerical_keys)
        assert new["block_backend"] == "c_recursion"
        error = abs(complex(*new["ssvv_raw"]) - complex(*old["ssvv_raw"]))
        relative = error / abs(complex(*old["ssvv_raw"]))
        same_series = new.get("series_parameter", "elliptic_nome") == old.get("series_parameter", "elliptic_nome")
        same_ope = new.get("local_ope_policy", "historical_hybrid") == old.get("local_ope_policy", "historical_hybrid")
        if same_series and same_ope:
            assert relative < 1e-9
        report["full_q8_ssvv_check"] = {
            "omega": new["omega"], "new_settings": settings,
            "frozen_reference": str(frozen_path.relative_to(frozen_path.parent.parent.parent)),
            "old_gram_amplitude": old["ssvv_raw"], "new_recursion_amplitude": new["ssvv_raw"],
            "absolute_difference": error, "relative_difference": relative,
            "same_series_parameter": same_series,
            "same_local_ope_policy": same_ope,
            "comparison_status": "backend check" if same_series and same_ope else "different finite expansions/OPE policies; independent convergence required",
            "new_diagnostics": new,
            "historical_block_build_seconds": old["build_seconds"],
            "timing_caveat": "historical and new full-amplitude runs had different machine loads",
        }
    if include_crossing:
        report["spectral_crossing_check"] = crossing_check(q_order)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--q-order", type=int, default=7)
    parser.add_argument("--amplitude-report", type=Path)
    parser.add_argument("--crossing", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = validate(args.q_order, args.amplitude_report, args.crossing)
    encoded = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        if args.output.exists():
            parser.error("output exists; choose a new validation report path")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
    print(encoded)


if __name__ == "__main__":
    main()
