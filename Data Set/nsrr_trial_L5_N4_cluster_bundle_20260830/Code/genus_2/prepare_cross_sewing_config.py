#!/usr/bin/env python3
"""Prepare a spin-certified theta/glasses cross-sewing production config."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
CODE_ROOT = SCRIPT_DIR.parent
C_RECURSION_DIR = CODE_ROOT / "c_Recursion"
for dependency in (CODE_ROOT, C_RECURSION_DIR):
    if str(dependency) not in sys.path:
        sys.path.insert(0, str(dependency))

from ns_genus2_cannon import _validate_config_spin_characteristics  # noqa: E402


BRANCH_COMPOSED_GLASSES_TO_THETA = [
    [0, 0, -1, -1],
    [0, 0, 0, -1],
    [1, 0, 0, 0],
    [-1, 1, 0, 0],
]
SPIN_ZERO = {"alpha": [0, 0], "beta": [0, 0]}


def prepare_config(
    source: Path,
    *,
    recursion_order: int = 24,
    quadrature_order: int = 10,
    designs: Sequence[tuple[int, int]] | None = None,
    point_ids: Sequence[str] | None = None,
) -> dict:
    """Select convergence designs and points, then certify transported spin."""

    config = json.loads(source.read_text(encoding="utf-8"))
    selected_pairs = (
        ((int(recursion_order), int(quadrature_order)),)
        if designs is None
        else tuple((int(r), int(n)) for r, n in designs)
    )
    if not selected_pairs or len(set(selected_pairs)) != len(selected_pairs):
        raise ValueError("designs must be a nonempty unique sequence")
    selected_designs = [
        {"recursion_order": r, "quadrature_order": n}
        for r, n in selected_pairs
    ]
    available = config.get("convergence_designs", ())
    missing = [design for design in selected_designs if design not in available]
    if missing:
        raise ValueError(f"requested designs {missing!r} are absent from {source}")

    if point_ids is not None:
        requested_points = tuple(str(value) for value in point_ids)
        if not requested_points or len(set(requested_points)) != len(requested_points):
            raise ValueError("point_ids must be a nonempty unique sequence")
        by_id = {str(point["id"]): point for point in config["points"]}
        missing_points = [value for value in requested_points if value not in by_id]
        if missing_points:
            raise ValueError(f"unknown point ids: {missing_points!r}")
        config["points"] = [by_id[value] for value in requested_points]

    config["description"] = (
        "Fresh theta/glasses cross-sewing check with the parity-correct "
        "glasses c-recursion, ordinary glasses vacuum/global seed, and "
        "branch-composed [00|00] spin transport"
    )
    config["convergence_designs"] = selected_designs
    config["physical_lifts"] = {
        # In Human Note edge order (zero, one, infinity), direct physical
        # Majorana sewing maps (+,-,+) to [00|00].  This affine map is
        # unrelated to the auxiliary fermion in double Virasoro.
        "theta": [1, -1, 1],
        "glasses": [1, 1, 1],
    }
    config["expected_spin_characteristics"] = {
        "theta": dict(SPIN_ZERO),
        "glasses": dict(SPIN_ZERO),
    }
    provenance = config.setdefault("provenance", {})
    provenance.update(
        {
            "symplectic_matrix_glasses_to_theta_after_branch": (
                BRANCH_COMPOSED_GLASSES_TO_THETA
            ),
            "spin_transport": (
                "full affine genus-two characteristic action; both marked "
                "sewings select [00|00]"
            ),
            "spin_transport_source_channel": "glasses",
            "spin_transport_target_channel": "theta",
            "spin_transport_period_tolerance": 5.0e-10,
            "glasses_recursion_revision": (
                "ordinary vacuum/global product; self-loop Koszul sign "
                "included once; parity-correct odd-null transport"
            ),
            "cross_sewing_baseline": (
                "Data Set/ns_genus2_fivepoint_r24_n10_theta_parity_corrected.json"
            ),
        }
    )
    ledger = _validate_config_spin_characteristics(config)
    config["spin_transport_ledger"] = ledger
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--recursion-order", type=int, default=24)
    parser.add_argument("--quadrature-order", type=int, default=10)
    parser.add_argument(
        "--design",
        action="append",
        help="repeatable R:N design; overrides --recursion-order/--quadrature-order",
    )
    parser.add_argument("--point-id", action="append")
    args = parser.parse_args()
    designs = None
    if args.design:
        designs = []
        for value in args.design:
            recursion_text, quadrature_text = value.split(":", 1)
            designs.append((int(recursion_text), int(quadrature_text)))
    config = prepare_config(
        args.source,
        recursion_order=args.recursion_order,
        quadrature_order=args.quadrature_order,
        designs=designs,
        point_ids=args.point_id,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "point_count": len(config["points"]),
                "designs": config["convergence_designs"],
                "physical_lifts": config["physical_lifts"],
                "spin_characteristics": config["expected_spin_characteristics"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
