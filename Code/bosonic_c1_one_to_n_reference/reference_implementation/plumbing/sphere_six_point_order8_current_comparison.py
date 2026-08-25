#!/usr/bin/env python3
"""Compare the fixed paired order-8 sphere 1->5 fit with the matrix model."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from sphere_six_point_order8_current_fit import STATUS, design


MATRIX_COEFFICIENTS_IN_T = np.asarray([6.0, -55.0, 150.0, -125.0])


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def amplitude_imaginary(t: np.ndarray, q5: np.ndarray) -> np.ndarray:
    return -5.0 * t**6 * q5


def coefficient_diagnostics(
    fit_result: dict[str, Any], analytic: np.ndarray
) -> dict[str, Any]:
    names = ("a", "b", "c", "d")
    coefficients = np.asarray(fit_result["coefficients_in_t"], dtype=float)
    errors = np.asarray(fit_result["coefficient_standard_errors"], dtype=float)
    covariance = np.asarray(fit_result["coefficient_covariance"], dtype=float)
    differences = coefficients - analytic
    scales = np.sqrt(np.diag(covariance))
    correlation = covariance / np.outer(scales, scales)
    return {
        "coefficient_rows": [
            {
                "name": name,
                "worldsheet_fit": float(value),
                "marginal_standard_error": float(error),
                "matrix_model": float(target),
                "worldsheet_minus_matrix": float(difference),
                "relative_difference_percent": float(100.0 * difference / target),
                "marginal_pull": float(difference / error),
            }
            for name, value, error, target, difference in zip(
                names,
                coefficients,
                errors,
                analytic,
                differences,
                strict=True,
            )
        ],
        "coefficient_correlation_matrix": [
            [float(value) for value in row] for row in correlation
        ],
        "joint_coefficient_quadratic_form": float(
            differences @ np.linalg.inv(covariance) @ differences
        ),
    }


def root_diagnostics(
    fit_result: dict[str, Any], analytic: np.ndarray, sampled_t: np.ndarray
) -> list[dict[str, Any]]:
    coefficients = np.asarray(fit_result["coefficients_in_t"], dtype=float)
    covariance = np.asarray(fit_result["coefficient_covariance"], dtype=float)
    numerical_roots = np.sort_complex(np.roots(coefficients[::-1]))
    analytic_roots = np.sort_complex(np.roots(analytic[::-1]))
    if np.max(np.abs(numerical_roots.imag)) > 1.0e-10:
        raise ValueError("the fitted cubic does not have three real roots")
    if np.max(np.abs(analytic_roots.imag)) > 1.0e-10:
        raise ValueError("the comparison cubic does not have three real roots")
    rows: list[dict[str, Any]] = []
    for numerical, target in zip(
        numerical_roots.real, analytic_roots.real, strict=True
    ):
        basis = np.asarray([1.0, numerical, numerical**2, numerical**3])
        derivative = (
            coefficients[1]
            + 2.0 * coefficients[2] * numerical
            + 3.0 * coefficients[3] * numerical**2
        )
        gradient = -basis / derivative
        error = float(np.sqrt(gradient @ covariance @ gradient))
        rows.append(
            {
                "worldsheet_fit_root": float(numerical),
                "formal_standard_error": error,
                "matrix_model_root": float(target),
                "worldsheet_minus_matrix": float(numerical - target),
                "formal_pull": float((numerical - target) / error),
                "inside_sampled_interval": bool(
                    sampled_t.min() <= numerical <= sampled_t.max()
                ),
            }
        )
    return rows


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "Arial Bold.ttf" if bold else "Arial.ttf"
    return ImageFont.truetype(f"/System/Library/Fonts/Supplemental/{name}", size)


def draw_figure(fit: dict[str, Any], figure_path: Path) -> None:
    points = fit["points"]
    t = np.asarray([point["t"] for point in points], dtype=float)
    q8 = np.asarray([point["Q5_order8_estimate"] for point in points], dtype=float)
    envelope = np.asarray(
        [point["available_numerical_envelope_proxy_Q5"] for point in points],
        dtype=float,
    )
    primary = fit["primary_fit"]
    coefficients = np.asarray(primary["coefficients_in_t"], dtype=float)
    covariance = np.asarray(primary["coefficient_covariance"], dtype=float)
    grid = np.linspace(t.min(), 1.0 / 3.0, 700, endpoint=False)
    grid_design = design(grid)
    q_fit = grid_design @ coefficients
    q_matrix = grid_design @ MATRIX_COEFFICIENTS_IN_T
    q_fit_error = np.sqrt(
        np.einsum("ij,jk,ik->i", grid_design, covariance, grid_design)
    )

    # Draw directly with Pillow so the postprocessor has no plotting-library
    # dependency on either the laptop or Cannon.
    width, height = 2496, 1968
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image, "RGBA")
    title_font = _font(47, bold=True)
    label_font = _font(34)
    tick_font = _font(27)
    legend_font = _font(27)
    small_font = _font(25)
    left, right = 235, 2420
    panels = [(205, 915), (1110, 1820)]
    blue = (0, 114, 178, 255)
    orange = (213, 94, 0, 255)
    charcoal = (34, 34, 34, 255)
    grey = (92, 92, 92, 255)
    grid_colour = (205, 205, 205, 255)
    band_colour = (119, 119, 119, 42)
    t_min, t_max = float(t.min()), 1.0 / 3.0

    title = "Sphere 1→5: 30-point paired order-8 result below the first residue wall"
    title_box = draw.textbbox((0, 0), title, font=title_font)
    draw.text(((width - (title_box[2] - title_box[0])) / 2, 55), title,
              fill=charcoal, font=title_font)

    panel_data = [
        (q_fit, q_matrix, q_fit_error, q8, envelope, "Q5(i t)"),
        (
            amplitude_imaginary(grid, q_fit),
            amplitude_imaginary(grid, q_matrix),
            np.abs(5.0 * grid**6 * q_fit_error),
            amplitude_imaginary(t, q8),
            5.0 * t**6 * envelope,
            "Im[mu^4 A_tree 1->5(i t)]",
        ),
    ]

    for panel_index, ((top, bottom), values) in enumerate(zip(panels, panel_data)):
        fit_values, matrix_values, band_errors, point_values, point_errors, ylabel = values
        all_low = np.concatenate((fit_values - band_errors, point_values - point_errors))
        all_high = np.concatenate((fit_values + band_errors, point_values + point_errors))
        y_min = float(np.min(all_low))
        y_max = float(np.max(all_high))
        padding = 0.10 * max(y_max - y_min, 1.0e-12)
        y_min -= padding
        y_max += padding

        def xpixel(value: float) -> float:
            return left + (value - t_min) * (right - left) / (t_max - t_min)

        def ypixel(value: float) -> float:
            return bottom - (value - y_min) * (bottom - top) / (y_max - y_min)

        x_ticks = np.linspace(t_min, t_max, 7)
        y_ticks = np.linspace(y_min, y_max, 6)
        for value in x_ticks:
            x = xpixel(float(value))
            draw.line((x, top, x, bottom), fill=grid_colour, width=2)
            if panel_index == 1:
                text_value = f"{value:.3f}"
                box = draw.textbbox((0, 0), text_value, font=tick_font)
                draw.text((x - (box[2] - box[0]) / 2, bottom + 17), text_value,
                          fill=charcoal, font=tick_font)
        for value in y_ticks:
            y = ypixel(float(value))
            draw.line((left, y, right, y), fill=grid_colour, width=2)
            text_value = f"{value:.4f}" if abs(value) < 0.1 else f"{value:.3f}"
            box = draw.textbbox((0, 0), text_value, font=tick_font)
            draw.text((left - (box[2] - box[0]) - 22, y - 15), text_value,
                      fill=charcoal, font=tick_font)

        band_upper = [(xpixel(float(x)), ypixel(float(y))) for x, y in zip(grid, fit_values + band_errors)]
        band_lower = [(xpixel(float(x)), ypixel(float(y))) for x, y in zip(grid[::-1], (fit_values - band_errors)[::-1])]
        draw.polygon(band_upper + band_lower, fill=band_colour)
        fit_line = [(xpixel(float(x)), ypixel(float(y))) for x, y in zip(grid, fit_values)]
        matrix_line = [(xpixel(float(x)), ypixel(float(y))) for x, y in zip(grid, matrix_values)]
        draw.line(fit_line, fill=charcoal, width=5)
        # White dashes make the target-free fit visually distinct from the matrix curve.
        for index in range(0, len(fit_line) - 8, 18):
            draw.line(fit_line[index:index + 9], fill=(255, 255, 255, 255), width=2)
        draw.line(matrix_line, fill=orange, width=7)
        for x_value, y_value, error in zip(t, point_values, point_errors):
            x = xpixel(float(x_value))
            y_low = ypixel(float(y_value - error))
            y_high = ypixel(float(y_value + error))
            y = ypixel(float(y_value))
            draw.line((x, y_low, x, y_high), fill=blue, width=3)
            draw.line((x - 7, y_low, x + 7, y_low), fill=blue, width=3)
            draw.line((x - 7, y_high, x + 7, y_high), fill=blue, width=3)
            draw.ellipse((x - 7, y - 7, x + 7, y + 7), fill=blue, outline="white", width=2)

        wall_x = xpixel(1.0 / 3.0)
        for y in range(top, bottom, 18):
            draw.line((wall_x, y, wall_x, min(y + 9, bottom)), fill=grey, width=4)
        draw.rectangle((left, top, right, bottom), outline=charcoal, width=3)
        draw.text((left, top - 54), ylabel, fill=charcoal, font=label_font)
        if panel_index == 0:
            label = "first residue wall"
            box = draw.textbbox((0, 0), label, font=small_font)
            draw.text((wall_x - (box[2] - box[0]) - 16, top + 12), label,
                      fill=grey, font=small_font)

    x_label = "t  in  ω = i t"
    x_label_box = draw.textbbox((0, 0), x_label, font=label_font)
    draw.text(((left + right - (x_label_box[2] - x_label_box[0])) / 2, 1905),
              x_label, fill=charcoal, font=label_font)

    legend_y = 950
    legend_items = [
        (blue, "paired order-8 estimates", "point"),
        (charcoal, "target-free cubic fit", "line"),
        (orange, "matrix model", "line"),
        ((119, 119, 119, 55), "fit statistical 1σ band", "band"),
    ]
    legend_x = left
    for colour, label, kind in legend_items:
        if kind == "point":
            draw.line((legend_x, legend_y + 15, legend_x + 48, legend_y + 15), fill=colour, width=3)
            draw.ellipse((legend_x + 17, legend_y + 7, legend_x + 33, legend_y + 23), fill=colour)
        elif kind == "band":
            draw.rectangle((legend_x, legend_y + 4, legend_x + 48, legend_y + 26), fill=colour)
        else:
            draw.line((legend_x, legend_y + 15, legend_x + 48, legend_y + 15), fill=colour, width=6)
        draw.text((legend_x + 62, legend_y), label, fill=charcoal, font=legend_font)
        legend_x += 535 if kind != "band" else 0

    figure_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(figure_path, format="PNG", optimize=True)


def compare(
    fit_path: Path,
    output_path: Path,
    csv_path: Path,
    figure_path: Path,
) -> dict[str, Any]:
    fit = json.loads(fit_path.read_text())
    if fit.get("status") != STATUS:
        raise ValueError("comparison requires the target-free paired order-8 fit")
    if fit.get("target_information_used") is not False:
        raise ValueError("target information entered the order-8 fit")
    points = fit["points"]
    t = np.asarray([point["t"] for point in points], dtype=float)
    q8 = np.asarray([point["Q5_order8_estimate"] for point in points], dtype=float)
    statistical = np.asarray(
        [point["Q5_order8_propagated_statistical_error"] for point in points],
        dtype=float,
    )
    envelope = np.asarray(
        [point["available_numerical_envelope_proxy_Q5"] for point in points],
        dtype=float,
    )
    matrix_values = design(t) @ MATRIX_COEFFICIENTS_IN_T
    residuals = q8 - matrix_values
    primary = fit["primary_fit"]
    coefficients = np.asarray(primary["coefficients_in_t"], dtype=float)
    coefficient_errors = np.asarray(primary["coefficient_standard_errors"], dtype=float)
    coefficient_difference = coefficients - MATRIX_COEFFICIENTS_IN_T
    primary_coefficient_diagnostics = coefficient_diagnostics(
        primary, MATRIX_COEFFICIENTS_IN_T
    )
    envelope_fit = fit["available_envelope_weighted_sensitivity_fit"]
    envelope_coefficient_diagnostics = coefficient_diagnostics(
        envelope_fit, MATRIX_COEFFICIENTS_IN_T
    )

    point_rows = [
        {
            "t": float(value),
            "Q5_order8_estimate": float(worldsheet),
            "Q5_propagated_statistical_error": float(stat_error),
            "Q5_available_numerical_envelope_proxy": float(proxy),
            "Q5_matrix_model": float(target),
            "Q5_worldsheet_minus_matrix": float(residual),
            "conservative_proxy_pull": float(residual / proxy),
        }
        for value, worldsheet, stat_error, proxy, target, residual in zip(
            t, q8, statistical, envelope, matrix_values, residuals, strict=True
        )
    ]
    result: dict[str, Any] = {
        "status": "sphere_1to5_order8_30point_compared_after_target_free_fit",
        "target_free_fit": str(fit_path),
        "target_free_fit_sha256": sha256(fit_path),
        "comparison_domain": "omega=i*t with 0<t<1/3",
        "matrix_model_coefficients_in_t": [
            float(value) for value in MATRIX_COEFFICIENTS_IN_T
        ],
        "matrix_model_reduced_amplitude": "Q_5(i*t)=6-55*t+150*t^2-125*t^3",
        "worldsheet_fit_coefficients_in_t": [float(value) for value in coefficients],
        "worldsheet_fit_coefficient_standard_errors": [
            float(value) for value in coefficient_errors
        ],
        "coefficient_marginal_pulls": [
            float(value) for value in coefficient_difference / coefficient_errors
        ],
        "maximum_absolute_coefficient_relative_difference": float(
            np.max(np.abs(coefficients / MATRIX_COEFFICIENTS_IN_T - 1.0))
        ),
        "detailed_coefficient_comparison": {
            "propagated_statistical_fit": primary_coefficient_diagnostics,
            "available_envelope_weighted_sensitivity_fit": (
                envelope_coefficient_diagnostics
            ),
            "interpretation": (
                "The statistical covariance is strongly correlated and excludes "
                "higher-order numerical truncation. The envelope fit is a sensitivity "
                "diagnostic, not a formal likelihood or convergence certificate."
            ),
        },
        "root_comparison": {
            "propagated_statistical_fit": root_diagnostics(
                primary, MATRIX_COEFFICIENTS_IN_T, t
            ),
            "available_envelope_weighted_sensitivity_fit": root_diagnostics(
                envelope_fit, MATRIX_COEFFICIENTS_IN_T, t
            ),
            "interpretation": (
                "Only the root near t=0.2 lies inside the sampled interval; the "
                "other two are cubic extrapolations. Root errors are delta-method "
                "errors within the indicated fit covariance."
            ),
        },
        "pointwise_comparison": {
            "available_envelope_proxy_chi_squared": float(
                np.sum((residuals / envelope) ** 2)
            ),
            "degrees_of_freedom": len(t),
            "maximum_absolute_available_envelope_proxy_pull": float(
                np.max(np.abs(residuals / envelope))
            ),
            "statistical_only_chi_squared": float(
                np.sum((residuals / statistical) ** 2)
            ),
            "maximum_absolute_statistical_only_pull": float(
                np.max(np.abs(residuals / statistical))
            ),
            "rms_Q5_residual": float(np.sqrt(np.mean(residuals**2))),
        },
        "formal_order8_to_higher_convergence_certificate": False,
        "figure": str(figure_path),
        "points": point_rows,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    with csv_path.open("w", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(point_rows[0]))
        writer.writeheader()
        writer.writerows(point_rows)
    draw_figure(fit, figure_path)
    return result


def main() -> None:
    base = Path(__file__).parent / "results" / "sphere_six_point_1to5" / "order8_30point_current"
    parser = argparse.ArgumentParser()
    parser.add_argument("--fit", type=Path, default=base / "order8_target_free_fit.json")
    parser.add_argument("--output", type=Path, default=base / "order8_matrix_comparison.json")
    parser.add_argument("--csv", type=Path, default=base / "order8_matrix_comparison.csv")
    parser.add_argument(
        "--figure", type=Path, default=base / "sphere_one_to_five_order8_30point_comparison.png"
    )
    arguments = parser.parse_args()
    result = compare(arguments.fit, arguments.output, arguments.csv, arguments.figure)
    print(json.dumps(result["pointwise_comparison"], indent=2, sort_keys=True))
    print(arguments.output)
    print(arguments.csv)
    print(arguments.figure)


if __name__ == "__main__":
    main()
