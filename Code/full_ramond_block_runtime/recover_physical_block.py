#!/usr/bin/env python3
"""Recover the equal-structure physical NSRR block from an enlarged q-series."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "double_virasoro" / "nsrr"))

from nsrr_genus2_block import (  # noqa: E402
    ZERO_VECTOR,
    auxiliary_majorana_nsrr_series,
    ramond_sector_residual,
    recover_same_structure_nsrr_series,
    star_convolve_series,
)


def components_from_record(record, cutoff):
    result = {}
    for item in record["coefficients"]:
        levels = tuple(item["twice_levels"])
        if sum(levels) > 2 * cutoff:
            continue
        parity = item["eta_parity"]
        if len(parity) != 3 or any(bit not in (0, 1) for bit in parity):
            raise ValueError("eta_parity must contain three bits")
        index = sum(bit << edge for edge, bit in enumerate(parity))
        values = result.setdefault(levels, [0j] * 8)
        value = item["coefficient"]
        values[index] += complex(value["real"], value["imag"])
    return result


def series_discrepancy(left, right):
    absolute = scaled = 0.0
    for levels in set(left) | set(right):
        for a, b in zip(left.get(levels, ZERO_VECTOR), right.get(levels, ZERO_VECTOR)):
            difference = abs(a - b)
            absolute = max(absolute, difference)
            scaled = max(scaled, difference / max(1.0, abs(a), abs(b)))
    return {"maximum_absolute_error": absolute, "maximum_scaled_error": scaled}


def recover_record(record, *, cutoff=None, sector_tolerance=1e-8, check_pbw=False):
    """Return physical coefficients; direct PBW is optional validation only."""

    if not record.get("block", "").startswith("widehat "):
        raise ValueError("input must identify an enlarged block")
    available = record["total_level_cutoff"]
    cutoff = available if cutoff is None else cutoff
    if int(cutoff) != cutoff or not 0 <= cutoff <= available:
        raise ValueError("requested cutoff must be a nonnegative integer within the input cutoff")
    cutoff = int(cutoff)
    parameters = dict(record["parameters"])
    etas = parameters["three_point_eta"]
    enlarged = components_from_record(record, cutoff)
    started = time.perf_counter()
    auxiliary = auxiliary_majorana_nsrr_series(maximum_total_twice_level=2 * cutoff)
    auxiliary_seconds = time.perf_counter() - started
    started = time.perf_counter()
    physical = recover_same_structure_nsrr_series(
        enlarged,
        auxiliary,
        maximum_total_twice_level=2 * cutoff,
        etas=etas,
        sector_tolerance=sector_tolerance,
    )
    recovery_seconds = time.perf_counter() - started
    factorized = star_convolve_series(
        auxiliary, physical, maximum_total_twice_level=2 * cutoff
    )
    diagnostics = {
        "input_sector_scaled_residual": ramond_sector_residual(enlarged),
        "auxiliary_sector_scaled_residual": ramond_sector_residual(auxiliary),
        "sector_tolerance": sector_tolerance,
        "forward_reconstruction": series_discrepancy(factorized, enlarged),
    }
    if check_pbw:
        from nsrr_genus2_block import direct_pbw_nsrr_series

        reference = direct_pbw_nsrr_series(
            b=parameters["b"],
            momenta=parameters["P"],
            form_parity=parameters["fermion_parity"],
            primary_parity=parameters.get("primary_parity", 0),
            etas=etas,
            maximum_total_twice_level=2 * cutoff,
        )
        diagnostics["direct_physical_pbw_comparison"] = series_discrepancy(physical, reference)
    coefficients = []
    for levels, vector in sorted(physical.items(), key=lambda item: (sum(item[0]), item[0])):
        for index, value in enumerate(vector):
            if value == 0:
                continue
            coefficients.append({
                "twice_levels": list(levels),
                "levels": [level / 2 for level in levels],
                "eta_parity": [(index >> edge) & 1 for edge in range(3)],
                "coefficient": {"real": value.real, "imag": value.imag},
            })
    result = {
        "calculation": "physical NSRR q-expansion recovered by inversion in the equal-structure ideal",
        "block": record["block"].removeprefix("widehat "),
        "total_level_cutoff": cutoff,
        "parameters": parameters,
        "method": {
            "reconstruction": "coefficientwise restricted convolution inverse, auxiliary constant = 2 e_+",
            "physical_pbw_used_for_reconstruction": False,
            "supported_vertex_signs": "eta = eta_prime",
            "all_eight_tube_signs_retained": True,
        },
        "timing_seconds": {"auxiliary": auxiliary_seconds, "recovery": recovery_seconds},
        "diagnostics": diagnostics,
        "coefficients": coefficients,
    }
    if "evaluation_check" in record:
        q = record["evaluation_check"]["q"]
        tube_signs = record["evaluation_check"]["eta_tubes"]
        value = 0j
        for item in coefficients:
            term = complex(item["coefficient"]["real"], item["coefficient"]["imag"])
            for edge in range(3):
                term *= q[edge] ** (item["twice_levels"][edge] / 2) * tube_signs[edge] ** item["eta_parity"][edge]
            value += term
        result["evaluation_check"] = {"q": q, "eta_tubes": tube_signs, "value": {"real": value.real, "imag": value.imag}}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("enlarged_json", type=Path)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--cutoff", type=int)
    parser.add_argument("--sector-tolerance", type=float, default=1e-8)
    parser.add_argument("--direct-pbw-check", action="store_true")
    args = parser.parse_args()
    if args.enlarged_json.resolve() == args.json.resolve():
        raise ValueError("physical output must differ from the enlarged input")
    result = recover_record(
        json.loads(args.enlarged_json.read_text()),
        cutoff=args.cutoff,
        sector_tolerance=args.sector_tolerance,
        check_pbw=args.direct_pbw_check,
    )
    result["enlarged_input"] = str(args.enlarged_json.resolve())
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"block": result["block"], "timing_seconds": result["timing_seconds"], "diagnostics": result["diagnostics"], "output": str(args.json)}, indent=2))


if __name__ == "__main__":
    main()
