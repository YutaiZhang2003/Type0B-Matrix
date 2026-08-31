#!/usr/bin/env python3
"""Coefficient and five-chart certificates, independent of Liouville Z.

Supported star channels compare branching-recursion/double-Virasoro results
with direct PBW. Nullspace completion itself is PBW data, not a second
independent double-Virasoro determination. No partition normalization or
modular agreement is fitted here.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import time

import sympy as sp

import nsrr_nsnsns_theta_omega_scan as scan
from nsrr_genus2_block import HumanNSRRThetaOracle, ZERO_VECTOR, star_convolve_series
from theta_star_algebra import fwht, star_spectrum
from nsrr_plumbing_adapter import NSRRPlumbingInputs, GEOMETRY_SECTORS


def certify(cutoff, geometry):
    started = time.perf_counter()
    fingerprint_started = scan.fingerprint()
    b = sp.Rational(7, 5)
    bg = b + 1/b
    momenta = (sp.Rational(21, 100), sp.Rational(37, 100), sp.Rational(52, 100))
    rows = []
    for p1 in (0, 1):
        runtime = scan.NSRRDoubleVirasoroTheta(
            b=float(b), physical_momenta=momenta, cutoff=cutoff,
            primary_parity=p1, completion="pbw_diagnostic",
            pbw_completion_max_level=cutoff)
        for f in (0, 1):
            for left, right in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
                oracle = HumanNSRRThetaOracle(
                    central_charge=sp.Rational(3, 2)+3*bg**2,
                    h_ns=bg**2/8+momenta[0]**2/2,
                    beta_r1=sp.I*momenta[1]/sp.sqrt(2),
                    beta_r2=sp.I*momenta[2]/sp.sqrt(2),
                    form_parity=f, primary_parity=p1, etas=(left, right))
                actual = runtime.physical_components(f, left, right)
                expected = {e: oracle.coefficient_components(e[0], e[1]//2, e[2]//2)
                            for e in actual}
                component_error = max(abs(x-y) for e in actual
                                      for x, y in zip(actual[e], expected[e]))
                supported_error = max(abs(star_spectrum(actual[e])[k]
                                          -star_spectrum(expected[e])[k])
                                      for e in actual for k in (2, 3, 4, 5))
                physical_scale = max(1., *(abs(v) for vector in expected.values() for v in vector))
                sewn = star_convolve_series(runtime.auxiliary, expected,
                                            maximum_total_twice_level=2*cutoff)
                enlarged = {}
                for key, value in runtime.enlarged_series(f, left, right).items():
                    enlarged.setdefault(key[:3], [0j]*8)[key[3]+2*key[4]+4*key[5]] += value
                forward_error = max(abs(x-y) for e in sewn.keys() | enlarged.keys()
                                    for x, y in zip(sewn.get(e, ZERO_VECTOR), enlarged.get(e, ZERO_VECTOR)))
                points = []
                for point in geometry["points"]:
                    errors = []
                    for character in range(8):
                        lifts_slots=tuple(-1 if character & (1 << edge) else 1 for edge in range(3))
                        plumbing=NSRRPlumbingInputs(
                            tuple(complex(x) for x in point["source_chart"]["q_values"]),
                            lifts_slots[::-1], GEOMETRY_SECTORS)
                        q=plumbing.q_slots
                        if q != tuple(complex(x) for x in point["q_in_human_nsrr_slot_order"]):
                            raise ArithmeticError("saved q slot order disagrees with the checked boundary adapter")
                        if plumbing.momenta_slots(tuple(reversed(momenta))) != tuple(float(p) for p in momenta):
                            raise ArithmeticError("momentum edge labels were not transported with q")
                        reference = scan.evaluate_twice_level_series(
                            {e: fwht(v)[character] for e, v in expected.items()}, q)
                        result = scan.evaluate_twice_level_series(
                            runtime.physical_series(f, left, right, character), q)
                        errors.append(abs(result-reference))
                    points.append({"t": point["t"], "maximum_absolute_block_error_all_eight_lifts": max(errors)})
                row = {"primary_parity": p1, "form_parity": f,
                       "eta_left": left, "eta_right": right,
                       "parity_coefficient_count": 8*len(actual),
                       "maximum_component_absolute_error": component_error,
                       "maximum_component_scaled_error": component_error/physical_scale,
                       "maximum_supported_spectrum_error": supported_error,
                       "maximum_forward_star_identity_error": forward_error,
                       "ward_residual": runtime.ward_residual_maximum,
                       "nullspace_source": "equal-sign Ward support" if left == right else "direct PBW",
                       "independent_double_virasoro_physical_coefficient_count": (
                           8*len(actual) if left == right else 0),
                       "points": points}
                if component_error/physical_scale > 2e-8 or forward_error/max(1., physical_scale) > 2e-8:
                    raise ArithmeticError(f"certificate failed: {row}")
                rows.append(row)
                print(f"p1={p1} f={f} eta=({left:+d},{right:+d}): "
                      f"PBW error {component_error:.3e}; forward error {forward_error:.3e}", flush=True)
    if scan.fingerprint() != fingerprint_started:
        raise RuntimeError("implementation files changed during the certificate run; rerun on one frozen snapshot")
    return {"schema": "nsrr-human-note-block-certificate-v1", "cutoff": cutoff,
            "b": float(b), "physical_momenta": [float(x) for x in momenta],
            "numerical_kernel_fingerprint": fingerprint_started,
            "geometry_digest": scan._digest(geometry), "rows": rows,
            "parity_coefficients_checked": sum(r["parity_coefficient_count"] for r in rows),
            "independent_double_virasoro_physical_coefficients_checked": sum(
                r["independent_double_virasoro_physical_coefficient_count"] for r in rows),
            "spin_projected_point_values_checked": 8*len(rows)*len(geometry["points"]),
            "plumbing_boundary": "q, literal lifts and momenta transported together by NSRRPlumbingInputs",
            "runtime_seconds": time.perf_counter()-started,
            "scope": "Chiral Human-Note blocks and forward auxiliary-star identity only. Not a nonchiral partition-function or modular-invariance certificate."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cutoff", type=int, default=3)
    parser.add_argument("--geometry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    scan.write_json(args.output, certify(args.cutoff, scan._load(args.geometry)))


if __name__ == "__main__":
    main()
