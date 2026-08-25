#!/usr/bin/env python3
"""Compare threshold quadrature with the torus two-point cusp Laplace series."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

try:
    from genus1_two_point_adaptive_momentum import AuditPoint, evaluate_point
    from genus1_two_point_cusp_laplace import necklace_laplace_estimate
except ImportError:  # pragma: no cover
    from plumbing.genus1_two_point_adaptive_momentum import AuditPoint, evaluate_point
    from plumbing.genus1_two_point_cusp_laplace import necklace_laplace_estimate


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = (
    ROOT
    / "results/genus1_two_point_worldsheet/threshold_vs_laplace_multi_point_v1"
)


def _point(name: str, *, x: float, tau2: float, fraction: float) -> dict[str, object]:
    tau = 0.13 + 1.0j * float(tau2)
    z = 2.0 * math.pi * (0.31 + float(fraction) * tau)
    return {
        "name": name,
        "x": float(x),
        "tau": tau,
        "z": z,
        "fraction": float(fraction),
    }


def comparison_points() -> list[dict[str, object]]:
    points: list[dict[str, object]] = []
    for tau2 in (8.0, 10.0, 12.0, 16.0, 20.0, 24.0):
        points.append(
            _point(f"tau{tau2:g}_v025_x04", x=0.4, tau2=tau2, fraction=0.25)
        )
    for fraction in (0.10, 0.15, 0.40, 0.50):
        points.append(
            _point(
                f"tau16_v{int(100*fraction):03d}_x04",
                x=0.4,
                tau2=16.0,
                fraction=fraction,
            )
        )
    for tau2 in (12.0, 20.0):
        for x in (0.2, 0.6, 0.8):
            points.append(
                _point(
                    f"tau{tau2:g}_v025_x{int(10*x):02d}",
                    x=x,
                    tau2=tau2,
                    fraction=0.25,
                )
            )
    return points


def _relative_change(first: complex, second: complex) -> float:
    return float(abs(second - first) / max(abs(first), abs(second), 1.0e-300))


def _shanks(first: complex, second: complex, third: complex) -> complex:
    """Accelerate three successive alternating Watson truncations."""

    denominator = third - 2.0 * second + first
    if abs(denominator) <= 1.0e-300:
        return third
    return third - (third - second) ** 2 / denominator


def compare_point(
    specification: dict[str, object],
    *,
    threshold_orders: tuple[int, ...] = (8, 10, 12),
    threshold_tolerance: float = 1.0e-7,
    maximum_laplace_degree: int = 10,
) -> dict[str, object]:
    name = str(specification["name"])
    x = float(specification["x"])
    tau = complex(specification["tau"])
    z = complex(specification["z"])
    point = AuditPoint(name, "necklace", z, tau)
    threshold_attempts: list[dict[str, object]] = []
    previous_threshold: complex | None = None
    threshold_value: complex | None = None
    for order in threshold_orders:
        value, _ = evaluate_point(
            point,
            x=x,
            order=order,
            necklace_orders=(6, 3),
            ope_orders=(3, 8),
            dps=32,
        )
        drift = (
            None
            if previous_threshold is None
            else _relative_change(previous_threshold, value)
        )
        threshold_attempts.append(
            {
                "order": int(order),
                "value_real": float(value.real),
                "value_imag": float(value.imag),
                "relative_step": drift,
            }
        )
        threshold_value = value
        if drift is not None and drift < threshold_tolerance:
            break
        previous_threshold = value
    assert threshold_value is not None

    laplace_attempts: list[dict[str, object]] = []
    for degree in range(maximum_laplace_degree + 1):
        estimate = necklace_laplace_estimate(
            x,
            z,
            tau,
            max_x_degree=degree,
            dps=50,
            cauchy_nodes=128,
        )
        laplace_attempts.append(
            {
                "degree": degree,
                "value_real": float(estimate.value.real),
                "value_imag": float(estimate.value.imag),
                "relative_step": estimate.relative_step_from_previous,
                "q_max": float(estimate.maximum_primary_q_abs),
                "decay_first": float(estimate.decay_coefficients[0]),
                "decay_second": float(estimate.decay_coefficients[1]),
            }
        )
    candidates = [
        attempt
        for attempt in laplace_attempts[1:]
        if attempt["relative_step"] is not None
        and math.isfinite(float(attempt["relative_step"]))
    ]
    selected = min(candidates, key=lambda attempt: float(attempt["relative_step"]))
    raw_laplace_value = complex(
        float(selected["value_real"]),
        float(selected["value_imag"]),
    )
    final_values = [
        complex(float(attempt["value_real"]), float(attempt["value_imag"]))
        for attempt in laplace_attempts[-4:]
    ]
    previous_accelerated = _shanks(*final_values[:3])
    accelerated = _shanks(*final_values[1:])
    accelerated_step = _relative_change(previous_accelerated, accelerated)
    relative_difference = _relative_change(threshold_value, accelerated)
    threshold_last_step = threshold_attempts[-1]["relative_step"]
    return {
        "name": name,
        "x": x,
        "omega_imaginary": x,
        "tau_real": float(tau.real),
        "tau_imag": float(tau.imag),
        "z_real": float(z.real),
        "z_imag": float(z.imag),
        "necklace_fraction": float(specification["fraction"]),
        "threshold_attempts": threshold_attempts,
        "threshold_selected_order": int(threshold_attempts[-1]["order"]),
        "threshold_selected_value_real": float(threshold_value.real),
        "threshold_last_step": threshold_last_step,
        "laplace_attempts": laplace_attempts,
        "laplace_selected_degree": int(selected["degree"]),
        "laplace_selected_value_real": float(raw_laplace_value.real),
        "laplace_smallest_step": float(selected["relative_step"]),
        "laplace_selection_at_maximum_degree": int(selected["degree"])
        == maximum_laplace_degree,
        "maximum_primary_q_abs": float(selected["q_max"]),
        "minimum_decay_coefficient": min(
            float(selected["decay_first"]),
            float(selected["decay_second"]),
        ),
        "laplace_shanks_value_real": float(accelerated.real),
        "laplace_shanks_value_imag": float(accelerated.imag),
        "laplace_shanks_previous_value_real": float(previous_accelerated.real),
        "laplace_shanks_relative_step": accelerated_step,
        "threshold_laplace_raw_relative_difference": _relative_change(
            threshold_value, raw_laplace_value
        ),
        "threshold_laplace_relative_difference": relative_difference,
        "laplace_self_convergence_pass": accelerated_step < 5.0e-5,
        "passes_5e_5_comparison": relative_difference < 5.0e-5,
    }


def run(output: Path = DEFAULT_OUTPUT) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    for specification in comparison_points():
        row = compare_point(specification)
        rows.append(row)
        print(
            f"{row['name']:22s} Q={row['threshold_selected_order']:2d} "
            f"D={row['laplace_selected_degree']:2d} "
            f"qmax={row['maximum_primary_q_abs']:.2e} "
            f"difference={row['threshold_laplace_relative_difference']:.3e} "
            f"self={row['laplace_shanks_relative_step']:.3e}",
            flush=True,
        )
    output.mkdir(parents=True, exist_ok=True)
    summary = {
        "calculation": "genus-one two-point threshold versus Watson/Laplace cusp expansion",
        "threshold_orders": [8, 10, 12],
        "threshold_relative_tolerance": 1.0e-7,
        "laplace_maximum_x_degree": 10,
        "block_orders": {"necklace": [6, 3]},
        "comparison_tolerance": 5.0e-5,
        "point_count": len(rows),
        "pass_count": sum(bool(row["passes_5e_5_comparison"]) for row in rows),
        "self_convergence_pass_count": sum(
            bool(row["laplace_self_convergence_pass"]) for row in rows
        ),
        "rows": rows,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    scalar_fields = [
        key
        for key, value in rows[0].items()
        if not isinstance(value, (list, dict))
    ]
    with (output / "comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=scalar_fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(
            {key: row[key] for key in scalar_fields}
            for row in rows
        )
    print(f"wrote {output}", flush=True)
    return summary


if __name__ == "__main__":
    run()
