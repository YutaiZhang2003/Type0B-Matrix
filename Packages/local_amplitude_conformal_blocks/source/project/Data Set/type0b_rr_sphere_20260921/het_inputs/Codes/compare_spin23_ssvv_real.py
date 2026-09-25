#!/usr/bin/env python3
"""Compare frozen real-energy SSVV numerics with the locked proposal.

This post-processor is intentionally separate from
``run_spin23_ssvv_real_blind.py``.  It imports the proposed closed form only
after verifying the SHA-256 digest of the completed formula-blind dataset.
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
    / "spin23_ssvv_real_blind.json"
)
OUTPUT = (
    WORKSPACE_DIR
    / "Data Set"
    / "high_accuracy"
    / "spin23_ssvv_real_comparison.json"
)
FROZEN_BLIND_SHA256 = (
    "6b932f514211bd792c124164c334c446bea8d003b189a522518b6f0cc634bed3"
)
SUPPLEMENTAL_SOURCES = {
    "q8_endpoint": (
        WORKSPACE_DIR
        / "Data Set/high_accuracy/spin23_ssvv_real_q8_blind.json",
        "a80792045b77f6d535e8b911a1454c3654c2eed36069428d0b41f82e89a4019c",
    ),
    "q9_endpoint": (
        WORKSPACE_DIR
        / "Data Set/high_accuracy/spin23_ssvv_real_q9_blind.json",
        "550b45bb73f49b6167689b34d7c7de3c4e8ecc187183c11859398af5630825cb",
    ),
    "q8_mid": (
        WORKSPACE_DIR
        / "Data Set/high_accuracy/spin23_ssvv_real_q8_mid_blind.json",
        "1990c1bd8b07c5a7d28372ce67b87ed143f837582d09d3321c764f6e55e633db",
    ),
    "q8_p525": (
        WORKSPACE_DIR
        / "Data Set/high_accuracy/spin23_ssvv_real_q8_p525_blind.json",
        "48058fbeb5df1887a5d2bf615d27cd976ac352c96bc7f921347a5f59d6d534dd",
    ),
}

if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from spin23_genuine_formulas import s_to_svv_raw_candidate  # noqa: E402


def _pair(value: complex) -> list[float]:
    value = complex(value)
    return [float(value.real), float(value.imag)]


def _complex(row: Mapping[str, Any], key: str) -> complex:
    return complex(*row[key])


def _relative(first: complex, second: complex) -> float:
    return float(abs(first - second) / max(abs(second), sys.float_info.min))


def _row_at(
    rows: Sequence[Mapping[str, Any]], energy: float, epsilon: float
) -> Mapping[str, Any]:
    for row in rows:
        if (
            float(row["real_energy"]) == float(energy)
            and float(row["regulator_epsilon"]) == float(epsilon)
        ):
            return row
    raise KeyError(f"no row at E={energy}, epsilon={epsilon}")


def _read_frozen(path: Path, expected_hash: str) -> Mapping[str, Any]:
    source_bytes = path.read_bytes()
    actual_hash = hashlib.sha256(source_bytes).hexdigest()
    if actual_hash != expected_hash:
        raise RuntimeError(
            f"supplemental blind source hash mismatch for {path}: "
            f"{actual_hash} != {expected_hash}"
        )
    source = json.loads(source_bytes)
    if source.get("status") != "complete":
        raise RuntimeError(f"supplemental blind source is incomplete: {path}")
    if source.get("proposal_imported_or_evaluated") is not False:
        raise RuntimeError(f"supplemental source is not formula-blind: {path}")
    return source


def main() -> None:
    source_bytes = SOURCE.read_bytes()
    actual_hash = hashlib.sha256(source_bytes).hexdigest()
    if actual_hash != FROZEN_BLIND_SHA256:
        raise RuntimeError(
            "blind source hash differs from the frozen pre-proposal artifact: "
            f"{actual_hash} != {FROZEN_BLIND_SHA256}"
        )
    source = json.loads(source_bytes)
    if source.get("status") != "complete":
        raise RuntimeError("blind numerical source is not complete")
    if source.get("proposal_imported_or_evaluated") is not False:
        raise RuntimeError("blind numerical source does not certify proposal separation")

    profiles = source["profiles"]
    production = profiles["production_extended_q7"]["rows"]
    ultradense = profiles["momentum_ultradense_extended_q7"]["rows"]
    superdense = profiles["momentum_superdense_extended_q7"]["rows"]
    worldsheet_dense = profiles["worldsheet_dense_extended_q7"]["rows"]
    supplemental = {
        name: _read_frozen(path, expected_hash)
        for name, (path, expected_hash) in SUPPLEMENTAL_SOURCES.items()
    }
    q8_mid = supplemental["q8_mid"]["profiles"]["block_q8_extended"]["rows"]
    q8_p525 = supplemental["q8_p525"]["profiles"][
        "block_q8_momentum_superdense"
    ]["rows"]
    q9_endpoint = supplemental["q9_endpoint"]["profiles"][
        "block_q9_extended"
    ]["rows"]

    rows = []
    for production_row in production:
        energy = float(production_row["real_energy"])
        epsilon = float(production_row["regulator_epsilon"])
        selected = production_row
        selected_profile = "production_extended_q7"
        selected_block_order = 7
        selected_momentum_nodes = 375
        momentum_control = None
        higher_order_momentum_control = None

        try:
            control = _row_at(ultradense, energy, epsilon)
        except KeyError:
            pass
        else:
            momentum_control = {
                "node_sequence": [375, 450],
                "relative_changes": [
                    _relative(
                        _complex(control, "ssvv_raw"),
                        _complex(production_row, "ssvv_raw"),
                    )
                ],
            }
            selected = control
            selected_profile = "momentum_ultradense_extended_q7"
            selected_momentum_nodes = 450

        try:
            control = _row_at(superdense, energy, epsilon)
        except KeyError:
            pass
        else:
            if momentum_control is None:
                raise RuntimeError("superdense row has no ultradense predecessor")
            momentum_control["node_sequence"].append(525)
            momentum_control["relative_changes"].append(
                _relative(
                    _complex(control, "ssvv_raw"),
                    _complex(selected, "ssvv_raw"),
                )
            )
            selected = control
            selected_profile = "momentum_superdense_extended_q7"
            selected_momentum_nodes = 525

        if energy == 0.5:
            selected = _row_at(q8_mid, energy, epsilon)
            selected_profile = "block_q8_extended"
            selected_block_order = 8
            selected_momentum_nodes = 375
        elif energy == 0.75:
            q8_at_375 = _row_at(q8_mid, energy, epsilon)
            selected = _row_at(q8_p525, energy, epsilon)
            selected_profile = "block_q8_momentum_superdense"
            selected_block_order = 8
            selected_momentum_nodes = 525
            higher_order_momentum_control = {
                "block_order": 8,
                "node_sequence": [375, 525],
                "relative_change": _relative(
                    _complex(selected, "ssvv_raw"),
                    _complex(q8_at_375, "ssvv_raw"),
                ),
            }
        elif energy == 1.0:
            selected = _row_at(q9_endpoint, energy, epsilon)
            selected_profile = "block_q9_extended"
            selected_block_order = 9
            selected_momentum_nodes = 375

        try:
            dense = _row_at(worldsheet_dense, energy, epsilon)
        except KeyError:
            worldsheet_change = None
        else:
            worldsheet_change = _relative(
                _complex(dense, "ssvv_raw"),
                _complex(production_row, "ssvv_raw"),
            )

        omega = _complex(selected, "omega")
        numerical = _complex(selected, "ssvv_raw")
        proposal = s_to_svv_raw_candidate(omega, omega, omega)
        rows.append(
            {
                "real_energy": energy,
                "regulator_epsilon": epsilon,
                "omega": selected["omega"],
                "selected_numerical_profile": selected_profile,
                "selected_block_order": selected_block_order,
                "selected_momentum_nodes": selected_momentum_nodes,
                "selected_numerical": _pair(numerical),
                "proposal": _pair(proposal),
                "numerical_over_proposal": _pair(numerical / proposal),
                "formula_relative_residual": _relative(numerical, proposal),
                "adjacent_block_order_relative_change": float(
                    selected["adjacent_q_relative_change"]
                ),
                "momentum_control": momentum_control,
                "higher_order_momentum_control": higher_order_momentum_control,
                "worldsheet_dense_relative_change": worldsheet_change,
                "piece_cancellation_ratio": float(
                    selected["piece_cancellation_ratio"]
                ),
            }
        )

    residuals = [row["formula_relative_residual"] for row in rows]
    controlled = [row for row in rows if row["momentum_control"] is not None]
    final_momentum_changes = [
        row["momentum_control"]["relative_changes"][-1] for row in controlled
    ]
    worldsheet_changes = [
        row["worldsheet_dense_relative_change"]
        for row in rows
        if row["worldsheet_dense_relative_change"] is not None
    ]
    report = {
        "status": "complete",
        "comparison_stage": "post_frozen_blind_numerics",
        "proposal_evaluated_after_hash_freeze": True,
        "blind_source_sha256": actual_hash,
        "supplemental_blind_source_sha256": {
            name: expected_hash
            for name, (_, expected_hash) in SUPPLEMENTAL_SOURCES.items()
        },
        "chronology_note": (
            "The eight-point q7 source was frozen before the first proposal comparison. "
            "The q8/q9 convergence escalations use the same formula-blind driver but "
            "were adaptively requested after the q7 comparison exposed slow convergence."
        ),
        "process": "S(3*omega) -> S(omega) V^a(omega) V^b(omega)",
        "normalization": "raw reduced coefficient multiplying delta_ab",
        "path": "omega=E+i*epsilon, E in [0.10,1.00], epsilon=0.02",
        "physical_status": (
            "finite-epsilon regulated real-energy data; epsilon->0+ contour "
            "continuation was not performed"
        ),
        "fit_parameters": 0,
        "general_proposal": (
            "M_raw^ab=delta_ab*pi*W*(1+2*i*omega0-"
            "omega0*omega1/(1+i*(omega2+omega3))), "
            "W=omega0*omega1*omega2*omega3, omega0=omega1+omega2+omega3"
        ),
        "factorized_proposal": (
            "M_raw^ab=delta_ab*pi*W*(1+i*omega0)*"
            "(1+i*(2*omega0-omega1))/(1+i*(omega0-omega1))"
        ),
        "equal_energy_proposal": (
            "M_raw^ab=delta_ab*3*pi*omega^4*(1+3*i*omega)*"
            "(1+5*i*omega)/(1+2*i*omega)"
        ),
        "summary": {
            "formula_relative_residual": {
                "minimum": min(residuals),
                "median": statistics.median(residuals),
                "maximum": max(residuals),
            },
            "controlled_anchor_formula_relative_residual_maximum": max(
                row["formula_relative_residual"] for row in controlled
            ),
            "final_momentum_relative_change_maximum": max(final_momentum_changes),
            "worldsheet_dense_relative_change_maximum": max(worldsheet_changes),
            "selected_adjacent_block_order_change_maximum": max(
                row["adjacent_block_order_relative_change"] for row in rows
            ),
            "higher_order_momentum_relative_change_at_E_0p75": next(
                row["higher_order_momentum_control"]["relative_change"]
                for row in rows
                if row["higher_order_momentum_control"] is not None
            ),
            "highest_resolution_formula_relative_residual_at_E_0p5": next(
                row["formula_relative_residual"]
                for row in rows
                if row["real_energy"] == 0.5
            ),
            "highest_resolution_formula_relative_residual_at_E_0p75": next(
                row["formula_relative_residual"]
                for row in rows
                if row["real_energy"] == 0.75
            ),
            "highest_resolution_formula_relative_residual_at_E_1p0": next(
                row["formula_relative_residual"]
                for row in rows
                if row["real_energy"] == 1.0
            ),
        },
        "rows": rows,
        "interpretation": (
            "Agreement is assessed only after the blind numerical source was frozen. "
            "It tests the proposal on the equal-outgoing-energy, finite-epsilon slice; "
            "it is not an epsilon->0+ continuation or a generic-energy proof."
        ),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
