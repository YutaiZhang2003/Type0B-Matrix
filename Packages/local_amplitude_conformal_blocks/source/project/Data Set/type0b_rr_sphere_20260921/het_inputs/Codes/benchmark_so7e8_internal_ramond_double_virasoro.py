"""Internal-R double-Virasoro-only truncation probe; no full PBW blocks."""
from __future__ import annotations

import argparse
import json
import time

from ramond_sphere_uniformization import elliptically_prefactor_internal_ramond_series
from so7e8_internal_ramond_double_virasoro import self_dual_internal_ramond_double_virasoro_coefficients


def run(*, orders=(0, 2, 4, 6), samples=16):
    points = (.08 + .02j, .25 + .06j, .5 - .05j)
    rows, previous, previous_values = [], None, None
    for order in orders:
        started = time.perf_counter()
        result = self_dual_internal_ramond_double_virasoro_coefficients(
            internal_momentum=.52 + .1j,
            external_momenta=(.23 + .08j, .37 - .12j, .41 + .03j, .61 - .05j),
            maximum_twice_level=order, ns_stars=(1, 1), form_parities=(1, 1),
            left_structure_sign=-1, right_structure_sign=1, samples=samples)
        block = elliptically_prefactor_internal_ramond_series(result.series)
        values = [complex(block.value(z)) for z in points]
        rows.append(dict(
            maximum_twice_level=order, samples=samples,
            maximum_radius_discrepancy=max(d.absolute_error for d in result.diagnostics.values()),
            maximum_known_coefficient_drift=None if previous is None else max(
                abs(value - result.series.coefficients[k]) / max(1, abs(value))
                for k, value in previous.items()),
            values=[[value.real, value.imag] for value in values],
            relative_order_change=None if previous_values is None else [
                abs(value - old) / max(abs(value), 1e-30) for value, old in zip(values, previous_values)],
            seconds=time.perf_counter() - started))
        previous, previous_values = result.series.coefficients, values
    return dict(backend="internal_R_double_virasoro", full_pbw_block_used=False,
                branching_source="finite PBW embedding vectors", b=1,
                moduli=[[z.real, z.imag] for z in points], rows=rows,
                nonchiral_projector_certified=False, integrated_amplitude_certified=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--orders", type=int, nargs="+", default=[0, 2, 4, 6])
    parser.add_argument("--samples", type=int, default=16)
    args = parser.parse_args()
    print(json.dumps(run(orders=args.orders, samples=args.samples), indent=2))
