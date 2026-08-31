#!/usr/bin/env python3
"""Read-only kernel-order diagnostic; this does NOT relabel a physical run.

The numerical example isolates the ordinary Virasoro global seed, not the
full NSRR partition. Any repair additionally needs consistent NS/R placement,
spin transport, primary factors and local-frame normalization.
"""
import argparse
from pathlib import Path

import nsrr_nsnsns_theta_omega_scan as scan
from compute_q_expansion import global_series
from ns_genus2_partition import THETA_GEOMETRY_EDGE_ORDER, THETA_CCY_DESCENDANT_EDGE_ORDER


def audit():
    weights_geometry = (.7, 1.1, 1.9)
    passed_unchanged = dict(global_series(weights_geometry, 1))
    slot_series = dict(global_series(weights_geometry[::-1], 1))
    converted_to_geometry = {exponent[::-1]: value for exponent, value in slot_series.items()}
    middle = (0, 1, 0)
    unchanged = passed_unchanged[middle]
    geometric = converted_to_geometry[middle]
    assert abs(unchanged-(.7-1.1-1.9)**2/(2*1.1)) < 1e-12
    assert abs(geometric-(1.9-1.1-.7)**2/(2*1.1)) < 1e-12
    return {
        "schema": "nsrr-nsnsns-edge-order-diagnostic-v1",
        "numerical_kernel_fingerprint": scan.fingerprint(),
        "geometry_order": list(THETA_GEOMETRY_EDGE_ORDER),
        "ccy_tensor_slot_order": list(THETA_CCY_DESCENDANT_EDGE_ORDER),
        "example_weights_in_geometric_order": list(weights_geometry),
        "example_q_one_global_coefficient_without_boundary_conversion": [unchanged.real, unchanged.imag],
        "example_q_one_global_coefficient_with_boundary_conversion": [geometric.real, geometric.imag],
        "observed_interface": "source_values passes saved geometric q and lifts unchanged to the NSRR slot-ordered series; target global block and physical Majorana explicitly reverse geometric labels at their trinion boundaries.",
        "scope": "A concrete order/frame inconsistency to investigate, not a repaired NSRR calculation or proof of the entire numerical discrepancy's origin. Do not reverse q alone: NS/R slots, lifts, primary factors, marking and free denominator must be treated together.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit()
    scan.write_json(args.output, result)
    for key, value in result.items():
        print(key, value)


if __name__ == "__main__":
    main()
