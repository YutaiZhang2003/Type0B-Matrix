#!/usr/bin/env python3
"""Assemble completed worldsheet integrals and only then compare conventions."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
RESULT_DIR = ROOT / "results" / "genus1_two_point_worldsheet"
INPUTS = {
    0.2: RESULT_DIR / "x02_production.json",
    0.4: RESULT_DIR / "x04_production.json",
    0.6: RESULT_DIR / "x06_production.json",
    0.8: RESULT_DIR / "x08_production.json",
}


def matrix_model_minus_i_amplitude(x: float) -> float:
    return (-x * x + 2.0 * x**4 - x**5) / 24.0


def run() -> None:
    rows: list[dict[str, float]] = []
    for x, path in INPUTS.items():
        record = json.loads(path.read_text(encoding="utf-8"))
        final = record["cusp_fit"]["final_I"]
        error = record["cusp_fit"]["final_rqmc_standard_error"]
        reduced = float(final["real"])
        reduced_error = abs(float(error["real"]))
        # This transformation is deliberately performed only after the native
        # worldsheet integrations have been completed and serialized.
        bry_value = 0.5 * reduced
        bry_error = 0.5 * reduced_error
        target = matrix_model_minus_i_amplitude(x)
        rows.append(
            {
                "x": x,
                "native_reduced_I1": reduced,
                "native_rqmc_error": reduced_error,
                "native_A_over_i_gs2": 8.0 * np.pi**2 * reduced,
                "bry_minus_i_A": bry_value,
                "bry_rqmc_error": bry_error,
                "matrix_model_minus_i_A": target,
                "worldsheet_over_matrix": bry_value / target,
            }
        )

    x_values = np.asarray([row["x"] for row in rows])
    y_values = np.asarray([row["bry_minus_i_A"] for row in rows])
    errors = np.asarray([row["bry_rqmc_error"] for row in rows])
    design = np.column_stack([-x_values**2, 2.0 * x_values**4, -x_values**5]) / 24.0
    weighted_design = design / errors[:, None]
    weighted_values = y_values / errors
    fit, *_ = np.linalg.lstsq(weighted_design, weighted_values, rcond=None)
    covariance = np.linalg.inv(weighted_design.T @ weighted_design)
    fit_errors = np.sqrt(np.diag(covariance))

    csv_path = RESULT_DIR / "worldsheet_amplitude_table.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "ordering_statement": (
            "All native reduced worldsheet integrals were completed before "
            "the BRY/matrix-model notation transformation in this assembly step."
        ),
        "native_relation": "A_1^ws = 8*pi^2*i*g_s^2*I_1",
        "notation_transform": "-i A_BRY^(1) = I_1/2, using g=2*pi*g_s",
        "rows": rows,
        "weighted_fit": {
            "form": "(-a*x^2+2*b*x^4-c*x^5)/24",
            "a": float(fit[0]),
            "b": float(fit[1]),
            "c": float(fit[2]),
            "rqmc_only_errors": {
                "a": float(fit_errors[0]),
                "b": float(fit_errors[1]),
                "c": float(fit_errors[2]),
            },
        },
    }
    json_path = RESULT_DIR / "worldsheet_amplitude_summary.json"
    json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    plot_x = np.linspace(0.0, 1.0, 500)
    target_y = np.asarray([matrix_model_minus_i_amplitude(value) for value in plot_x])
    fit_y = (
        -fit[0] * plot_x**2 + 2.0 * fit[1] * plot_x**4 - fit[2] * plot_x**5
    ) / 24.0
    width, height = 720.0, 470.0
    left, right, top, bottom = 90.0, 25.0, 30.0, 70.0
    y_min, y_max = -0.0082, 0.0002

    def map_x(value: float) -> float:
        return left + value * (width - left - right)

    def map_y(value: float) -> float:
        return top + (y_max - value) / (y_max - y_min) * (height - top - bottom)

    def polyline(values_x: np.ndarray, values_y: np.ndarray) -> str:
        return " ".join(
            f"{map_x(float(x)):.2f},{map_y(float(y)):.2f}"
            for x, y in zip(values_x, values_y)
        )

    svg: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width:.0f}" height="{height:.0f}" viewBox="0 0 {width:.0f} {height:.0f}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<line x1="{left}" y1="{map_y(0):.2f}" x2="{width-right}" y2="{map_y(0):.2f}" stroke="#444"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#444"/>',
    ]
    for tick in np.linspace(0.0, 1.0, 6):
        x_pixel = map_x(float(tick))
        svg.append(f'<line x1="{x_pixel:.2f}" y1="{top}" x2="{x_pixel:.2f}" y2="{height-bottom}" stroke="#ddd"/>')
        svg.append(f'<text x="{x_pixel:.2f}" y="{height-bottom+25:.2f}" text-anchor="middle" font-size="14">{tick:.1f}</text>')
    for tick in (-0.008, -0.006, -0.004, -0.002, 0.0):
        y_pixel = map_y(tick)
        svg.append(f'<line x1="{left}" y1="{y_pixel:.2f}" x2="{width-right}" y2="{y_pixel:.2f}" stroke="#ddd"/>')
        svg.append(f'<text x="{left-10:.2f}" y="{y_pixel+5:.2f}" text-anchor="end" font-size="14">{tick:.3f}</text>')
    svg.extend(
        [
            f'<polyline points="{polyline(plot_x,target_y)}" fill="none" stroke="#315b9d" stroke-width="2" stroke-dasharray="7 5"/>',
            f'<polyline points="{polyline(plot_x,fit_y)}" fill="none" stroke="#d36c45" stroke-width="2"/>',
        ]
    )
    for x, y, error in zip(x_values, y_values, errors):
        x_pixel, y_pixel = map_x(float(x)), map_y(float(y))
        y_low, y_high = map_y(float(y - error)), map_y(float(y + error))
        svg.append(f'<line x1="{x_pixel:.2f}" y1="{y_low:.2f}" x2="{x_pixel:.2f}" y2="{y_high:.2f}" stroke="#b8322a" stroke-width="2"/>')
        svg.append(f'<circle cx="{x_pixel:.2f}" cy="{y_pixel:.2f}" r="5" fill="#b8322a"/>')
    svg.extend(
        [
            f'<text x="{(left+width-right)/2:.2f}" y="{height-20:.2f}" text-anchor="middle" font-size="17">x = -i omega</text>',
            f'<text x="22" y="{(top+height-bottom)/2:.2f}" text-anchor="middle" font-size="17" transform="rotate(-90 22 {(top+height-bottom)/2:.2f})">-i A^(1)_(1 to 1)</text>',
            f'<line x1="{width-220}" y1="42" x2="{width-180}" y2="42" stroke="#315b9d" stroke-width="2" stroke-dasharray="7 5"/><text x="{width-170}" y="47" font-size="14">matrix model</text>',
            f'<line x1="{width-220}" y1="64" x2="{width-180}" y2="64" stroke="#d36c45" stroke-width="2"/><text x="{width-170}" y="69" font-size="14">worldsheet fit</text>',
            f'<circle cx="{width-200}" cy="86" r="5" fill="#b8322a"/><text x="{width-170}" y="91" font-size="14">direct worldsheet</text>',
            '</svg>',
        ]
    )
    (RESULT_DIR / "worldsheet_amplitude_comparison.svg").write_text(
        "\n".join(svg) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2))
    print(f"wrote {csv_path}")
    print(f"wrote {json_path}")


if __name__ == "__main__":
    run()
