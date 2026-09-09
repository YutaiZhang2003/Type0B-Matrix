#!/usr/bin/env python3
"""Marked theta charts with NS at infinity, as required by the PBW package.

Geometry APIs order punctures as (zero,one,infinity); the Human-Note NSRR
trinion orders slots as (infinity,one,zero). Merely reversing the old q's
does not transform the local coordinates at the middle puncture. We first
re-mark the surface and solve the inverse period problem in the new chart.
The source spin and its NS reference denominator are kept distinct from
the as-yet-uncertified nonchiral Ramond sewing projector.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import nsrr_nsnsns_theta_omega_scan as scan
from nsrr_plumbing_adapter import to_human_slots


# A'_0=-A_0-A_1, A'_1=A_1, B'=T^{-T} B. This exchanges the zero
# and infinity cutting curves while preserving a symplectic marking.
T = np.array([[-1, -1], [0, 1]], dtype=int)
ZERO = np.zeros((2, 2), dtype=int)
IDENTITY = np.eye(2, dtype=int)
RELABEL = np.block([[T.T, ZERO], [ZERO, T]])
# Fixed B-path branch of the collocation chart. This is explicit, is applied
# to the characteristic too, and is not re-rounded separately at each point.
K = np.array([[0, 1], [1, 1]], dtype=int)
SOURCE_REMARKING = np.block([[IDENTITY, K], [ZERO, IDENTITY]]) @ RELABEL
SOURCE_SPIN = ((1, 1), (0, 0))
NS_REFERENCE_SPIN = ((0, 0), (0, 0))
NS_REFERENCE_LIFTS = (1, -1, 1)


def action(matrix, omega):
    return ((matrix[:2, :2] @ omega + matrix[:2, 2:])
            @ np.linalg.inv(matrix[2:, :2] @ omega + matrix[2:, 2:]))


def geometry_to_human_slots(values):
    """Relabel data in an ALREADY matched NS-at-infinity theta chart."""
    return to_human_slots(values)


def build_geometry(baseline_config):
    transported = scan._transport_spin_characteristic(SOURCE_REMARKING, scan.SOURCE_SPIN)
    if transported != SOURCE_SPIN:
        raise ArithmeticError("source characteristic transport failed")
    inverse_remarking = np.rint(np.linalg.inv(SOURCE_REMARKING)).astype(int)
    to_target = scan.MATRIX @ inverse_remarking
    if scan._transport_spin_characteristic(to_target, SOURCE_SPIN) != scan.TARGET_SPIN:
        raise ArithmeticError("re-marked source does not map to the target spin")
    rows = []
    for point in baseline_config["points"]:
        old_chart = point["charts"]["source_nsrr"]
        old_omega = scan.complex_matrix(old_chart["omega"])
        desired = action(SOURCE_REMARKING, old_omega)
        # Only a seed: exact re-plumbing changes the collar coordinates.
        seed = geometry_to_human_slots(tuple(complex(q) for q in old_chart["q_values"]))
        chart = scan.inverse_chart(desired, seed)
        q = tuple(complex(x) for x in chart["q_values"])
        if scan._spin_characteristic_from_lifts("theta", q, NS_REFERENCE_LIFTS) != NS_REFERENCE_SPIN:
            raise ArithmeticError("the directly sewn NS reference has the wrong characteristic")
        backward = scan.solve_theta_collocation(*q, basis_order=32, samples_per_seam=160)
        residual = float(np.max(np.abs(backward.omega-desired)))
        if residual > 1e-8:
            raise ArithmeticError("independent higher-order forward period map failed")
        free_values = [float(scan.physical_superfield_plumbing_partition(
            "theta", q, NS_REFERENCE_LIFTS, max_mode=mode).one_superfield_value)
                       for mode in (36, 44)]
        free_reference = free_values[1]
        free_mode_change = abs(free_values[0]/free_values[1]-1)
        if free_mode_change > 1e-9:
            raise ArithmeticError("same-frame free determinant is not converged")
        spin_ratio = abs(scan.riemann_theta_constant_genus2(desired, SOURCE_SPIN, tol=1e-15)
                         / scan.riemann_theta_constant_genus2(desired, NS_REFERENCE_SPIN, tol=1e-15))
        target = scan.complex_matrix(point["charts"]["target_nsnsns"]["omega"])
        if np.max(np.abs(action(to_target, desired)-target)) > 1e-12:
            raise ArithmeticError("marked period transformation to target failed")
        from audit_nsrr_free_spin_conversion import audit as audit_spin_conversion
        spin_conversion = audit_spin_conversion(q, desired)
        rows.append({"t": point["t"], "source_chart": chart,
                     "q_in_human_nsrr_slot_order": [str(x) for x in geometry_to_human_slots(q)],
                     "high_order_forward_period_residual": residual,
                     "candidate_theta_ratio_free_superfield": free_reference*float(spin_ratio),
                     "theta_ratio_free_conversion_audit": spin_conversion,
                     "physical_free_superfield_status": (
                         "spin-ratio necessary check passed; Ramond sewing still required"
                         if spin_conversion["compatible"] else
                         "unavailable: the theta-ratio adapter fails its all-NS compatibility test"),
                     "free_ns_reference": free_reference, "free_majorana_spin_ratio": float(spin_ratio),
                     "free_mode_relative_change_36_to_44": free_mode_change,
                     "target_chart": point["charts"]["target_nsnsns"]})
        print(f"t={point['t']:.2f}: source re-plumbed, forward residual {residual:.3e}", flush=True)
    return {"schema": "nsrr-human-note-marked-geometry-v2",
            "geometry_edge_order": ["zero", "one", "infinity"],
            "geometry_edge_sectors": ["R", "R", "NS"],
            "human_slot_order": ["infinity", "one", "zero"],
            "human_slot_sectors": ["NS", "R", "R"],
            "source_remarking": SOURCE_REMARKING.tolist(),
            "source_to_target": to_target.tolist(),
            "source_characteristic": SOURCE_SPIN,
            "free_ns_reference_characteristic": NS_REFERENCE_SPIN,
            "free_ns_reference_lifts_geometry_order": NS_REFERENCE_LIFTS,
            "physical_ramond_lift_dictionary_status": "not inferred from the all-NS reference lifts",
            "baseline_digest": scan._digest(baseline_config), "points": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    scan.write_json(args.output, build_geometry(scan._load(args.baseline)))


if __name__ == "__main__":
    main()
