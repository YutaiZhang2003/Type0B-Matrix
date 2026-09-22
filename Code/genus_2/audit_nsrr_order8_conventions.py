#!/usr/bin/env python3
"""Audit convention counterfactuals for the historical order-eight NSRR run.

This does not manufacture a corrected NSRR partition function.  It separates
three questions that can be decided with existing data:

* whether the legacy theta-ratio free-spin adapter represents one fixed spin;
* how BRY ``(C_even,C_odd)`` differ from the HJS three-form coefficients; and
* what isolated parity-sign changes would do to the completed sector sums.

The physical Ramond projector and its two-sided BPZ dictionary remain inputs
that cannot be inferred by fitting modular agreement.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np

import audit_nsrr_free_spin_conversion as legacy_free
from fixed_spin_free_plumbing import charged_frame, fixed_spin_partition


SCHEMA = "nsrr-order8-convention-troubleshooting-v1"
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
DEFAULT_SUMMARY = (
    ROOT
    / "Data Set"
    / "nsrr_nsnsns_theta_order8_cannon_resume_summary_20260902.json"
)
DEFAULT_OUTPUT = (
    ROOT
    / "Data Set"
    / "nsrr_nsnsns_theta_order8_convention_audit_20260902.json"
)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _complex_triple(values: Sequence[str]) -> tuple[complex, complex, complex]:
    result = tuple(complex(value) for value in values)
    if len(result) != 3:
        raise ValueError("expected three plumbing parameters")
    return result  # type: ignore[return-value]


def _complex_matrix(values: Sequence[Sequence[str]]) -> np.ndarray:
    result = np.asarray(
        [[complex(value) for value in row] for row in values],
        dtype=np.complex128,
    )
    if result.shape != (2, 2):
        raise ValueError("expected a two-by-two period matrix")
    return result


def _characteristic(chart: dict) -> tuple[tuple[int, int], tuple[int, int]]:
    value = chart["characteristic"]
    return tuple(value["alpha"]), tuple(value["beta"])  # type: ignore[return-value]


def grouped_rows(summary: dict) -> dict[int, dict[str, dict]]:
    grouped: dict[int, dict[str, dict]] = {}
    for row in summary["rows"]:
        if int(row["block_order"]) != 8:
            continue
        order = int(row["quadrature_order"])
        channel = str(row["channel"])
        if channel in grouped.setdefault(order, {}):
            raise ValueError(f"duplicate order-{order} row for {channel}")
        grouped[order][channel] = row
    for order, rows in grouped.items():
        if set(rows) != {"source_nsrr", "target_nsnsns"}:
            raise ValueError(f"incomplete order-{order} comparison")
    if not grouped:
        raise ValueError("no order-eight rows found")
    return grouped


def _declared_fixed_spin_free(chart: dict, *, max_mode: int) -> dict:
    q_values = _complex_triple(chart["q_values"])
    omega = _complex_matrix(chart["omega"])
    frame = charged_frame(q_values, max_mode=max_mode)
    delta = frame.omega_charge - omega
    branch = np.rint(delta.real).astype(int)
    branch_residual = float(np.max(np.abs(delta - branch)))
    if branch_residual > 1.0e-8:
        raise ArithmeticError(
            f"cannot infer an integral charged-period branch: {branch_residual:.3e}"
        )
    result = fixed_spin_partition(
        q_values,
        omega,
        _characteristic(chart),
        period_branch=branch,
        max_mode=max_mode,
    )
    return {
        "declared_marked_characteristic": chart["characteristic"],
        "inferred_charge_period_branch": branch.tolist(),
        "period_residual": branch_residual,
        "charge_characteristic": [
            list(result["characteristic_charge"][0]),
            list(result["characteristic_charge"][1]),
        ],
        "z_free": float(result["Z_free"]),
        "z_boson": float(result["Z_boson"]),
        "z_majorana": float(result["Z_majorana"]),
        "has_fermion_zero_mode": bool(result["has_fermion_zero_mode"]),
    }


def audit(summary: dict, *, free_max_mode: int = 32) -> dict:
    if summary.get("schema") != "nsrr-nsnsns-theta-cannon-v1":
        raise ValueError("not an NSRR/NSNSNS Cannon summary")
    grouped = grouped_rows(summary)
    config = summary["config"]
    charts = config["marked_surface"]["charts"]
    source_chart = charts["source_nsrr"]
    source_q = _complex_triple(source_chart["q_values"])
    source_omega = _complex_matrix(source_chart["omega"])

    legacy_ratio_test = legacy_free.audit(
        source_q,
        source_omega,
        max_mode=free_max_mode,
        tolerance=1.0e-8,
    )
    fixed_free = {
        channel: _declared_fixed_spin_free(charts[channel], max_mode=free_max_mode)
        for channel in ("source_nsrr", "target_nsnsns")
    }
    historical_free = summary["free_superfield_same_local_frame"]
    old_source_free = float(historical_free["source_nsrr"])
    old_target_free = float(historical_free["target_nsnsns"])
    fixed_source_free = fixed_free["source_nsrr"]["z_free"]
    fixed_target_free = fixed_free["target_nsnsns"]["z_free"]
    kappa = float(historical_free["power_kappa"])

    comparisons = []
    for order, pair in sorted(grouped.items()):
        source = pair["source_nsrr"]
        target = pair["target_nsnsns"]
        source_even, source_odd = map(float, source["sector_values"])
        target_even, target_odd = map(float, target["sector_values"])
        source_total = source_even + source_odd
        target_total = target_even + target_odd
        ratio = float(source["q_observable"]) / float(target["q_observable"])
        comparisons.append(
            {
                "quadrature_order": order,
                "reported_source_over_target": ratio,
                "uniform_source_multiplier_needed_for_one": 1.0 / ratio,
                "source_form_fractions": [
                    source_even / source_total,
                    source_odd / source_total,
                ],
                "source_flip_only_f1": ratio
                * (source_even - source_odd)
                / source_total,
                "source_decomposition_minus_with_two_odd_vertex_i_phases": ratio,
                "source_replace_E_O_by_E_over_2_O_over_2_only": ratio / 4.0,
                "source_unrestricted_trial_normalization_same_archived_blocks": (
                    ratio / 8.0
                ),
                "target_odd_fraction": target_odd / target_total,
                "target_flip_only_odd": ratio
                * target_total
                / (target_even - target_odd),
                "target_omit_both_odd_i_phase_and_decomposition_minus": ratio,
                "declared_fixed_free_source_only": ratio
                * (old_source_free / fixed_source_free) ** kappa,
                "declared_fixed_free_target_only": ratio
                * (fixed_target_free / old_target_free) ** kappa,
                "declared_fixed_free_both": ratio
                * (
                    old_source_free
                    * fixed_target_free
                    / (fixed_source_free * old_target_free)
                )
                ** kappa,
            }
        )

    return {
        "schema": SCHEMA,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_summary_status": summary["status"],
        "task_count": int(summary["task_count"]),
        "diagnosis": {
            "free_theory": (
                "FAIL: the historical filtered Fredholm block is a linear "
                "combination of fixed-spin determinants, so its theta-ratio "
                "conversion is not a one-spin physical denominator. This is "
                "a spin-basis/assembly error, not a lone fermion sign."
            ),
            "three_point_coefficients": (
                "FAIL at the frozen nonchiral boundary: it inserts BRY "
                "(E,O) directly where the derived HJS three-form coefficients "
                "are (E/2,O/2), and combines this with an unproved factor-four "
                "Ramond-ground multiplicity. The pieces cannot be repaired "
                "independently of the physical projector."
            ),
            "superconformal_block_sign": (
                "PASS for the checked chiral quadratic signs and the universal "
                "physical decomposition factor (-1)^f. The frozen assembler "
                "omits both that outer sign and the two i^f vertex phases; their "
                "numerical cancellation in a restricted ansatz is not a "
                "convention derivation. Do not flip signs inside the checked block."
            ),
            "overall": (
                "The 3.1% near-match is not evidence for a small correction. "
                "The archived observable mixes an invalid free-spin adapter, "
                "an unproved Ramond multiplicity/projector, and an inconsistent "
                "three-point basis. A corrected run requires a two-sided physical "
                "Ramond projector and spin dictionary before recomputation."
            ),
        },
        "free_theory": {
            "legacy_theta_ratio_test": legacy_ratio_test,
            "historical_denominators": {
                "source_nsrr": old_source_free,
                "target_nsnsns": old_target_free,
            },
            "fixed_free_for_declared_marked_spins": fixed_free,
            "warning": (
                "The fixed denominators are decisive tests of the free theory, "
                "but the counterfactual ratios are not corrected physical Qs: "
                "the archived Liouville numerator lacks a certified matching "
                "spin projection."
            ),
        },
        "three_point_conventions": {
            "BRY_basis": "(E,O)=(C_even,C_odd)",
            "physical_equal_family_basis": "d_+=(E+O)/2, d_-=(E-O)/2",
            "HJS_three_form_basis": "c_+=E/2, c_-=O/2",
            "frozen_assembler": (
                "uses (E,O) directly and multiplies by "
                "RAMOND_GROUND_COMPLETENESS(4) * HJS_COMPLETENESS(1/2)"
            ),
            "normalization_warning": (
                "Replacing coefficients alone scales the archived source by "
                "1/4. Comparing the frozen 2*E_eta*E_eta' weight with the "
                "unrestricted trial E_eta*E_eta'/4 gives a factor eight. Neither "
                "is a physical prescription until the Ramond projector is fixed."
            ),
        },
        "sign_conventions": {
            "theta_quadratic_sign": (
                "(-1)^K remains inside each checked chiral block"
            ),
            "physical_decomposition_sign": "(-1)^f",
            "odd_vertex_phase_hypothesis": "i^f at each of the two pants",
            "net_in_diagonal_trial": "(-1)^f * i^f * i^f = +1",
            "warning": (
                "The cancellation is algebraic; both ingredients must remain "
                "explicit so a different physical BPZ/vertex dictionary cannot "
                "silently change the answer."
            ),
        },
        "counterfactual_comparisons": comparisons,
        "scientific_scope": (
            "Troubleshooting of the historical diagnostic only; no corrected "
            "physical NSRR partition function is claimed."
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--free-max-mode", type=int, default=32)
    args = parser.parse_args(argv)
    if args.free_max_mode <= 0:
        parser.error("--free-max-mode must be positive")
    report = audit(load_json(args.summary), free_max_mode=args.free_max_mode)
    write_json(args.output, report)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
