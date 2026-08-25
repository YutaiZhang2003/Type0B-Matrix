#!/usr/bin/env python3
"""Audit NS plumbing sectors, lifts, and modular spin transport at genus two."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from typing import Sequence

import numpy as np

import ns_genus2_cannon as cannon
from ns_genus2_partition import (
    GLASSES_GEOMETRY_EDGE_ORDER,
    NSGenus2CRecursion,
    THETA_GEOMETRY_EDGE_ORDER,
    _q_power,
    _spin_characteristic_from_lifts,
    _transport_spin_characteristic,
    free_superfield_partition,
    ns_weight,
)
from ns_genus_c_recursion_checks import ns_c_pole
from free_majorana_pair_of_pants import (
    ns_fermion_states_at_twice_level,
    ns_fermion_twice_level,
)


EXPECTED_LIFTS = {"glasses": (1, 1, 1), "theta": (-1, 1, 1)}
EXPECTED_EDGES = {
    "glasses": tuple(GLASSES_GEOMETRY_EDGE_ORDER),
    "theta": tuple(THETA_GEOMETRY_EDGE_ORDER),
}


def _omega(point: dict, channel: str) -> np.ndarray:
    return np.asarray(
        [[complex(value) for value in row] for row in point["omega"][channel]],
        dtype=np.complex128,
    )


def _characteristic_json(characteristic) -> dict[str, list[int]]:
    return {
        "alpha": [int(value) for value in characteristic[0]],
        "beta": [int(value) for value in characteristic[1]],
    }


def audit(config_path: Path, summary_path: Path | None = None) -> dict:
    config = json.loads(config_path.read_text())
    ledger = cannon._validate_config_spin_characteristics(config)
    lifts = {
        channel: tuple(int(value) for value in config["physical_lifts"][channel])
        for channel in ("glasses", "theta")
    }
    if lifts != EXPECTED_LIFTS:
        raise AssertionError(f"unexpected production lifts: {lifts}")

    # The propagator probe distinguishes NS half-integer levels from a
    # Ramond integer-moded module and verifies that xi acts only on parity.
    q_probe = 0.17 + 0.03j
    for occupation in range(4):
        if _q_power(q_probe, occupation, 0) != q_probe**occupation:
            raise AssertionError("even NS propagation is not q^n")
        expected_odd = q_probe**occupation * np.sqrt(q_probe)
        if abs(_q_power(q_probe, occupation, 1) - expected_odd) > 1.0e-15:
            raise AssertionError("odd NS propagation is not q^(n+1/2)")

    # The independent fermion oracle enumerates psi_{-(k-1/2)} modes: a
    # one-particle state at twice-level 1 is the NS mode psi_{-1/2}.
    if (1,) not in ns_fermion_states_at_twice_level(1):
        raise AssertionError("free Majorana sewing lacks psi_-1/2")
    if ns_fermion_twice_level((1,)) != 1:
        raise AssertionError("free Majorana twice-level convention changed")

    # The c-recursion kernel must accept only the NS Kac lattice r+s even.
    ns_c_pole(2, 2, ns_weight(0.31))
    try:
        ns_c_pole(2, 1, ns_weight(0.31))
    except ValueError:
        pass
    else:
        raise AssertionError("Ramond Kac label entered the NS recursion")

    provenance = config["provenance"]
    matrix = np.asarray(
        provenance["symplectic_matrix_glasses_to_theta_after_branch"],
        dtype=int,
    )
    A, B = matrix[:2, :2], matrix[:2, 2:]
    C, D = matrix[2:, :2], matrix[2:, 2:]
    point_rows = []
    for point in config["points"]:
        point_id = str(point["id"])
        characteristics = {
            channel: _spin_characteristic_from_lifts(
                channel,
                tuple(complex(value) for value in point["q_values"][channel]),
                lifts[channel],
            )
            for channel in ("glasses", "theta")
        }
        transported = _transport_spin_characteristic(
            matrix, characteristics["glasses"]
        )
        if transported != characteristics["theta"]:
            raise AssertionError(f"spin transport failed at {point_id}")

        omega_glasses = _omega(point, "glasses")
        omega_theta = _omega(point, "theta")
        mapped_omega = (A @ omega_glasses + B) @ np.linalg.inv(
            C @ omega_glasses + D
        )
        period_residual = float(np.max(np.abs(mapped_omega - omega_theta)))

        free_rows = {}
        for channel in ("glasses", "theta"):
            free = free_superfield_partition(
                channel=channel,
                q_values=tuple(
                    complex(value) for value in point["q_values"][channel]
                ),
                omega=_omega(point, channel),
                physical_lifts=lifts[channel],
                max_word_length=int(config["numerics"]["free_word_length"]),
                max_mode=int(config["numerics"]["free_max_mode"]),
            )
            free_characteristic = (
                tuple(free.characteristic_alpha),
                tuple(free.characteristic_beta),
            )
            if free_characteristic != characteristics[channel]:
                raise AssertionError(
                    f"block/free spin mismatch at {point_id}/{channel}"
                )
            free_rows[channel] = {
                "lifts": list(lifts[channel]),
                "characteristic": _characteristic_json(free_characteristic),
                "fermion_method": free.fermion_method,
            }

        point_rows.append(
            {
                "point_id": point_id,
                "glasses_characteristic": _characteristic_json(
                    characteristics["glasses"]
                ),
                "theta_characteristic": _characteristic_json(
                    characteristics["theta"]
                ),
                "transported_glasses_characteristic": _characteristic_json(
                    transported
                ),
                "period_map_max_residual": period_residual,
                "block_and_free": free_rows,
            }
        )

    summary_checks = None
    if summary_path is not None:
        summary = json.loads(summary_path.read_text())
        if summary.get("config_digest") != cannon._digest(config):
            raise AssertionError("summary/config digest mismatch")
        if summary.get("spin_characteristics") != ledger:
            raise AssertionError("summary spin ledger mismatch")
        if any(row.get("global_method") != "resummed" for row in summary["rows"]):
            raise AssertionError("a reduced row did not use the resummed global block")
        if any(int(row.get("global_nonconverged_calls", 0)) for row in summary["rows"]):
            raise AssertionError("a reduced row contains nonconverged global calls")
        summary_checks = {
            "path": str(summary_path),
            "config_digest_matches": True,
            "spin_ledger_matches": True,
            "all_rows_resummed": True,
            "all_global_calls_converged": True,
        }

    return {
        "schema": "ns-genus2-sewing-spin-audit-v1",
        "config": str(config_path),
        "representation_sector": {
            "all_internal_edges": "NS",
            "supercurrent_modes": "r in Z+1/2",
            "primary_weight_formula": "h(P)=1/2+P^2/2 at b=1",
            "propagator": "q^(n+epsilon/2) xi^epsilon",
            "kac_lattice": "r+s even",
            "component_sector_labels": {
                "0": "even NS three-form",
                "1": "odd NS three-form",
                "not_meaning": "Ramond representation",
            },
        },
        "plumbing_graphs": {
            "glasses": {
                "edge_order": list(EXPECTED_EDGES["glasses"]),
                "puncture_types": ["NS", "NS", "NS"],
                "lifts": list(lifts["glasses"]),
            },
            "theta": {
                "edge_order": list(EXPECTED_EDGES["theta"]),
                "puncture_types": ["NS", "NS", "NS"],
                "lifts": list(lifts["theta"]),
            },
        },
        "spin_transport": {
            "source_channel": "glasses",
            "target_channel": "theta",
            "symplectic_matrix": matrix.tolist(),
            "affine_action_checked": True,
        },
        "points": point_rows,
        "summary_checks": summary_checks,
        "passed": True,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = audit(args.config, args.summary)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "passed": True}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
