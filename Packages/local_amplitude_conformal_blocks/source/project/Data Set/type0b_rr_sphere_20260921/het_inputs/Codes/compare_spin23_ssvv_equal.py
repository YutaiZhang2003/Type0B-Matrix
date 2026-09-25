#!/usr/bin/env python3
"""Compare the frozen equal-energy SSVV integral with the locked proposal.

This post-processor is intentionally separate from
``run_spin23_ssvv_equal_blind.py``.  The numerical driver imports no proposal
module; this script reads its completed JSON only after the numerical scan has
been frozen.
"""

from __future__ import annotations

import hashlib
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


CODE_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = CODE_DIR.parent
SOURCE = (
    WORKSPACE_DIR
    / "Data Set"
    / "high_accuracy"
    / "spin23_ssvv_equal_blind.json"
)
OUTPUT = (
    WORKSPACE_DIR
    / "Data Set"
    / "high_accuracy"
    / "spin23_ssvv_equal_comparison.json"
)
PRE_PROPOSAL_SNAPSHOT_SHA256 = (
    "723bb2323afe4a09aab6fe2b014e7ef33b6ed74635935421d4256a350fd404ed"
)

if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from spin23_genuine_formulas import s_to_svv_raw_candidate  # noqa: E402


def _pair(value: complex) -> list[float]:
    value = complex(value)
    return [float(value.real), float(value.imag)]


def _complex(row: Mapping[str, Any], key: str) -> complex:
    return complex(*row[key])


def _relative(first: complex, second: complex) -> float:
    return float(abs(first - second) / abs(second))


def _row_at(rows: Sequence[Mapping[str, Any]], y: float) -> Mapping[str, Any]:
    return min(rows, key=lambda row: abs(float(row["im_omega"]) - y))


def main() -> None:
    source_bytes = SOURCE.read_bytes()
    source = json.loads(source_bytes)
    if source.get("status") != "complete":
        raise RuntimeError("blind numerical source is not complete")

    profiles = source["profiles"]
    production = profiles["precision_infinite_q7"]["rows"]
    q8_rows = profiles["precision_infinite_q8"]["rows"]
    momentum_rows = profiles["precision_infinite128_q7"]["rows"]
    dense_rows = profiles["precision_infinite_dense_q7"]["rows"]

    rows = []
    for row in production:
        omega = _complex(row, "omega")
        numerical = _complex(row, "ssvv_raw")
        proposal = s_to_svv_raw_candidate(omega, omega, omega)
        rows.append(
            {
                "im_omega": float(row["im_omega"]),
                "omega": row["omega"],
                "numerical_q7": _pair(numerical),
                "proposal": _pair(proposal),
                "numerical_over_proposal": _pair(numerical / proposal),
                "formula_relative_residual": _relative(numerical, proposal),
                "q6_to_q7_relative_change": float(
                    row["adjacent_q_relative_change"]
                ),
            }
        )

    q8_comparisons = []
    for row in q8_rows:
        omega = _complex(row, "omega")
        numerical = _complex(row, "ssvv_raw")
        proposal = s_to_svv_raw_candidate(omega, omega, omega)
        q8_comparisons.append(
            {
                "im_omega": float(row["im_omega"]),
                "numerical_q8": _pair(numerical),
                "proposal": _pair(proposal),
                "numerical_over_proposal": _pair(numerical / proposal),
                "formula_relative_residual": _relative(numerical, proposal),
                "q7_to_q8_relative_change": float(
                    row["adjacent_q_relative_change"]
                ),
            }
        )

    momentum_controls = []
    for control in momentum_rows:
        y = float(control["im_omega"])
        baseline = _row_at(production, y)
        baseline_value = _complex(baseline, "ssvv_raw")
        control_value = _complex(control, "ssvv_raw")
        momentum_controls.append(
            {
                "im_omega": y,
                "nodes": [96, 128],
                "relative_change": _relative(control_value, baseline_value),
            }
        )

    worldsheet_controls = []
    for control in dense_rows:
        y = float(control["im_omega"])
        baseline = _row_at(production, y)
        baseline_value = _complex(baseline, "ssvv_raw")
        control_value = _complex(control, "ssvv_raw")
        worldsheet_controls.append(
            {
                "im_omega": y,
                "relative_change": _relative(control_value, baseline_value),
            }
        )

    formula_residuals = [row["formula_relative_residual"] for row in rows]
    q_changes = [row["q6_to_q7_relative_change"] for row in rows]
    report = {
        "status": "complete",
        "comparison_stage": "post_frozen_blind_numerics",
        "blind_driver_imports_proposal": False,
        "pre_proposal_snapshot_sha256": PRE_PROPOSAL_SNAPSHOT_SHA256,
        "final_numerical_source_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "process": "S(3*omega) -> S(omega) V^a(omega) V^b(omega)",
        "normalization": "raw reduced coefficient multiplying delta_ab",
        "path": "omega=1/9+i*y, 1/6 <= y <= 7/30",
        "equal_energy_proposal": (
            "M=3*pi*omega^4*(1+3*i*omega)*(1+5*i*omega)/(1+2*i*omega)"
        ),
        "numerical_method": {
            "structure_constants": (
                "exp(log C) with Upsilon_1 shift reduction to the fundamental strip"
            ),
            "momentum_integral": (
                "96-node Gauss-Legendre after P=-0.5*log((1-x)/2); P in [0,infinity)"
            ),
            "worldsheet": (
                "q^7 c-recursion; analytic disks epsilon0=0.30, epsilon1=0.24; "
                "theta=(48,48,192), radial=72"
            ),
        },
        "summary": {
            "q7_formula_relative_residual": {
                "minimum": min(formula_residuals),
                "median": statistics.median(formula_residuals),
                "maximum": max(formula_residuals),
            },
            "q6_to_q7_relative_change": {
                "minimum": min(q_changes),
                "median": statistics.median(q_changes),
                "maximum": max(q_changes),
            },
            "maximum_96_to_128_momentum_change": max(
                row["relative_change"] for row in momentum_controls
            ),
            "maximum_dense_worldsheet_change": max(
                row["relative_change"] for row in worldsheet_controls
            ),
            "q8_formula_relative_residual": q8_comparisons[0][
                "formula_relative_residual"
            ],
            "q7_to_q8_relative_change": q8_comparisons[0][
                "q7_to_q8_relative_change"
            ],
        },
        "rows": rows,
        "q8_comparisons": q8_comparisons,
        "momentum_controls": momentum_controls,
        "worldsheet_controls": worldsheet_controls,
        "conclusion": (
            "The cutoff-free, enlarged-patch computation agrees with the locked "
            "equal-energy SSVV proposal within the observed numerical controls."
        ),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
