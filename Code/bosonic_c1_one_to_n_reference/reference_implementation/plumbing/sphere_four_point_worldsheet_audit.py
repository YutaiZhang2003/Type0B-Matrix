#!/usr/bin/env python3
"""Blind convergence audit for a frozen sphere ``1->3`` worldsheet scan."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import date
from pathlib import Path

try:
    from sphere_four_point_worldsheet_scan import scan_point
except ImportError:  # pragma: no cover - package-style execution
    from plumbing.sphere_four_point_worldsheet_scan import scan_point


AUDIT_T_VALUES = (0.16, 0.30, 0.34, 0.46)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _settings(**updates: int | float) -> dict[str, int | float]:
    values: dict[str, int | float] = {
        "block_order": 10,
        "momentum_order": 24,
        "momentum_maximum": 8.0,
        "momentum_panels": 2,
        "sobol_power": 11,
        "replicates": 6,
    }
    values.update(updates)
    return values


def _evaluate(t: float, settings: dict[str, int | float], seed: int) -> dict[str, object]:
    return scan_point(
        t,
        block_order=int(settings["block_order"]),
        momentum_order=int(settings["momentum_order"]),
        momentum_maximum=float(settings["momentum_maximum"]),
        momentum_panels=int(settings["momentum_panels"]),
        sobol_power=int(settings["sobol_power"]),
        replicates=int(settings["replicates"]),
        seed=seed,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    base = Path(__file__).parent / "results" / "sphere_four_point_1to3"
    parser.add_argument("--scan", type=Path, default=base / "worldsheet_scan.json")
    parser.add_argument(
        "--scan-manifest",
        type=Path,
        default=base / "worldsheet_scan_frozen.json",
    )
    parser.add_argument("--t", nargs="+", type=float, default=AUDIT_T_VALUES)
    parser.add_argument("--output", type=Path, default=base / "worldsheet_audit.json")
    parser.add_argument(
        "--freeze-manifest",
        type=Path,
        default=base / "worldsheet_audit_frozen.json",
    )
    arguments = parser.parse_args()

    scan_manifest = json.loads(arguments.scan_manifest.read_text())
    scan_hash = _sha256(arguments.scan)
    if scan_hash != scan_manifest["sha256"]:
        raise RuntimeError("the frozen worldsheet scan hash does not match")

    variants = {
        "reference": _settings(),
        "block_order_8": _settings(block_order=8),
        "momentum_order_28": _settings(momentum_order=28),
        "momentum_maximum_7": _settings(momentum_maximum=7.0),
        "deep_rqmc": _settings(sobol_power=14, replicates=12),
    }
    records = []
    t_values = tuple(float(value) for value in arguments.t)
    if len(t_values) != len(set(t_values)) or any(not 0.0 < value < 0.5 for value in t_values):
        parser.error("--t values must be distinct and lie in 0<t<1/2")
    for index, t in enumerate(t_values):
        seed = 781237 + 1009 * index
        evaluations = {}
        for name, settings in variants.items():
            evaluations[name] = _evaluate(t, settings, seed)
            print(
                json.dumps(
                    {
                        "t": t,
                        "variant": name,
                        "Q3": evaluations[name]["Q3"],
                        "Q3_standard_error": evaluations[name]["Q3_standard_error"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        reference = float(evaluations["reference"]["Q3"]["real"])
        deterministic_shifts = {
            name: float(evaluations[name]["Q3"]["real"]) - reference
            for name in ("block_order_8", "momentum_order_28", "momentum_maximum_7")
        }
        records.append(
            {
                "t": t,
                "evaluations": evaluations,
                "deterministic_shifts_from_reference": deterministic_shifts,
                "maximum_absolute_deterministic_shift": max(
                    abs(value) for value in deterministic_shifts.values()
                ),
            }
        )

    payload = {
        "status": "worldsheet_only_audit_frozen_before_external_comparison",
        "matrix_model_information_used": False,
        "verified_scan_sha256": scan_hash,
        "variants": variants,
        "points": records,
        "maximum_absolute_deterministic_shift": max(
            float(record["maximum_absolute_deterministic_shift"]) for record in records
        ),
        "maximum_deep_rqmc_standard_error": max(
            float(record["evaluations"]["deep_rqmc"]["Q3_standard_error"]["real"])
            for record in records
        ),
    }
    arguments.output.write_text(json.dumps(payload, indent=2) + "\n")
    manifest = {
        "status": "worldsheet_only_audit_frozen_before_external_comparison",
        "artifact": str(arguments.output.resolve()),
        "sha256": _sha256(arguments.output),
        "frozen_on": date.today().isoformat(),
        "matrix_model_information_used": False,
    }
    arguments.freeze_manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
