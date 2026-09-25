"""Double-Virasoro-only order/contour stability probe; no PBW comparison.

Run from the repository: python Codes/benchmark_so7e8_mixed_double_virasoro.py
The fixed complex-momentum point is a CFT diagnostic, not an integrated
physical scattering amplitude. Increase orders only after the contour check
passes. JSON output reports actual block values as well as coefficient drift.
"""
from __future__ import annotations

import argparse
import json
import time

from ramond_sphere_uniformization import elliptically_prefactor_two_ramond_internal_ns_series
from so7e8_mixed_double_virasoro import self_dual_mixed_double_virasoro_coefficients


def run(*, orders=(3, 5, 7), samples=16, component="odd"):
    points = (.15 + .05j, .35 + .1j, .55 - .08j)
    rows = []
    previous = None
    previous_values = None
    for order in orders:
        started = time.perf_counter()
        result = self_dual_mixed_double_virasoro_coefficients(
            internal_momentum=.52 + .1j,
            external_momenta=(.23 + .08j, .37 - .12j, .41 + .03j, .61 - .05j),
            maximum_twice_level=order, component=component,
            rr_structure_sign=-1, ns_stars=(True, True), samples=samples)
        block = elliptically_prefactor_two_ramond_internal_ns_series(result.series)
        values = [complex(block.value(point)) for point in points]
        overlap = None if previous is None else max(
            abs(value - result.series.coefficients[k]) / max(1, abs(value))
            for k, value in previous.items())
        change = None if previous_values is None else [
            abs(a - d) / max(abs(a), 1e-30) for a, d in zip(values, previous_values)]
        rows.append(dict(
            maximum_twice_level=order, samples=samples,
            maximum_radius_discrepancy=max(d.absolute_error for d in result.diagnostics.values()),
            maximum_known_coefficient_drift=overlap,
            values=[[v.real, v.imag] for v in values], relative_order_change=change,
            seconds=time.perf_counter() - started))
        previous, previous_values = result.series.coefficients, values
    return dict(backend="double_virasoro", full_pbw_block_used=False,
                branching_source="finite PBW embedding vectors", component=component,
                stars=[1, 1], b=1, moduli=[[z.real, z.imag] for z in points], rows=rows,
                integrated_amplitude_certified=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--orders", type=int, nargs="+", default=[3, 5, 7])
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--component", choices=("even", "odd"), default="odd")
    args = parser.parse_args()
    print(json.dumps(run(orders=args.orders, samples=args.samples, component=args.component), indent=2))
