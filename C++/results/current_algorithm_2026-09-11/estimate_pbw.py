#!/usr/bin/env python3
"""Count PBW work and fit SAVED logs; never import or run the PBW implementation.

Run with the existing numerical Python environment (NumPy/SciPy required).
The only imported repository module is count_level_work, a counting utility.
The eta*eta'=+1 estimate discounts a tensor-associated cost, not Gram/inverse
or contraction costs.  Its range describes two cost assumptions and is not a
confidence interval.  No estimate at a new arithmetic precision is invented.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parents[3]
SAVED = ROOT / "Code/theta_fermion_ccy/results"
COUNT_SOURCE = ROOT / "Code/theta_fermion_ccy/count_level_work.py"


def main():
    started = time.perf_counter()
    spec = importlib.util.spec_from_file_location("integer_work_counter", COUNT_SOURCE)
    counter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(counter)
    levels = {str(n): {"pbw": counter.pbw_counts(n)} for n in (10, 15)}
    report = {
        "scope": "Exact integer counts and nonnegative fits to saved timing logs only",
        "new_pbw_gram_ward_or_branching_evaluations": 0,
        "parameters": {"b": "7/5", "momenta": ["11/23", "13/29", "17/31"],
                       "p": 0, "f": 0},
        "levels": levels,
        "calibrations": {},
        "equal_sign_reuse": {
            "source": "Code/ramond_zero_mode_recovery/check_level10_modular.py:248",
            "form_cache_source": "Code/double_virasoro/nsrr/ramond_pbw_generalized_ward.py:622",
            "reason": "For eta'=eta both vertices use the same cached HumanForm and cached value. Tensor arrays, phases and contractions are still evaluated twice/as usual.",
            "distinct_requested_top_level_ward_entries": {
                n: {"minus": v["pbw"]["requested_three_point_tensor_entries"],
                    "plus": v["pbw"]["requested_three_point_tensor_entries"] // 2}
                for n, v in levels.items()},
            "model_rule": "Keep all other fitted features fixed; multiply the tensor-associated fitted contribution by a factor between 1/2 and 1. Internal Ward cache misses and cache-hit overhead are not measured separately.",
        },
        "precision_comparison": {
            "fresh_cpp_requested_bits": 136,
            "pbw_calibrated_bits": [53, 384],
            "precision_matched": False,
            "pbw_136_bit_estimate_seconds": None,
            "explanation": "No saved 136-bit physical PBW timing exists. Native and 384-bit timings use different backends; interpolation by bit count is not justified. The 384-bit estimates can be shown as a higher-precision comparison proxy, never as a measured or fitted 40-digit time.",
        },
        "limitations": [
            "These predict the saved Python physical SCA PBW implementation, not a C++ PBW port.",
            "Gram and Ward internal mode recursions are not independently counted or timed.",
            "The sum of dimension cubes is a dense-inversion work proxy, not an exact FLOP count or a measured inversion speed.",
            "The lean model omits explicit cubic/inverse and contraction features; it is included to expose model sensitivity, not as an equally established asymptotic model.",
            "The full model fits per-monomial cost with correlated features, so fitted stage contributions are not measured stage times.",
            "The models assume unchanged cost per feature; larger basis memory pressure, cache behavior and backend crossover are not captured.",
            "Saved internal PBW timers include repeatedly rewriting the accumulated JSON result. Fresh C++ internal timers exclude final serialization; parent wall times should be compared where available.",
            "The spread of models is not a confidence interval or a guaranteed upper/lower runtime bound.",
        ],
    }
    sources = [COUNT_SOURCE, Path(__file__),
               ROOT / "Code/ramond_zero_mode_recovery/check_level10_modular.py",
               ROOT / "Code/double_virasoro/nsrr/ramond_pbw_generalized_ward.py",
               ROOT / "Code/theta_fermion_ccy/pbw_reference.py"]
    for label, bits, stem in (("native", 53, "pbw_machine_level10"),
                              ("flint_high_precision", 384, "pbw_level10")):
        log = SAVED / (stem + ".log")
        records = [json.loads(line) for line in log.read_text().splitlines()
                   if line.startswith("{")]
        wall = json.loads((SAVED / (stem + "_walltime.json")).read_text())
        fitted = counter.calibrate_pbw({"levels": levels}, log)
        for model in fitted["models"]:
            index = model["features"].index("requested_tensor_entries")
            rate = model["fitted_seconds_per_feature_unit"][index]
            model["predicted_equal_sign_seconds"] = {}
            for n, value in levels.items():
                total = model["predicted_seconds"][n]
                vertex = rate * value["pbw"]["requested_three_point_tensor_entries"]
                model["predicted_equal_sign_seconds"][n] = {
                    "half_tensor_associated_cost": total - vertex / 2,
                    "no_discount": total,
                }
        report["calibrations"][label] = {
            "precision_bits": bits,
            "saved_signs": [1, -1],
            "saved_level10_internal_seconds": records[-1]["elapsed_seconds"],
            "saved_level10_process_wall_seconds": wall["process_wall_seconds"],
            "log_rows": len(records),
            **fitted,
        }
        sources.extend((log, SAVED / (stem + "_walltime.json")))
    report["source_sha256"] = {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sources
    }
    report["counting_and_fit_seconds"] = time.perf_counter() - started
    output = Path(__file__).with_name("pbw_estimates.json")
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(output)
    print(json.dumps({name: {"saved_level10_seconds": value["saved_level10_internal_seconds"],
                            "minus_level15_seconds": [m["predicted_seconds"]["15"] for m in value["models"]],
                            "plus_level15_seconds": [m["predicted_equal_sign_seconds"]["15"] for m in value["models"]]}
                      for name, value in report["calibrations"].items()}, indent=2))


if __name__ == "__main__":
    main()
