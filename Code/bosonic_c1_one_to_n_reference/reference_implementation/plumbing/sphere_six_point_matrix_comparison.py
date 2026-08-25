#!/usr/bin/env python3
"""Compare the frozen sphere 1->5 worldsheet scan to the matrix model.

This script is intentionally downstream of ``worldsheet_freeze_manifest.json``.
It verifies the frozen scan checksum before evaluating the matrix-model formula.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def q5_matrix_model(t: float | np.ndarray) -> float | np.ndarray:
    """Q5(i t)=(1-5t)(2-5t)(3-5t)."""

    return (1.0 - 5.0 * t) * (2.0 - 5.0 * t) * (3.0 - 5.0 * t)


def amplitude_imaginary_matrix_model(t: float | np.ndarray) -> float | np.ndarray:
    """Imaginary part of mu^4 A_tree(i t)=-5 i t^6 Q5(i t)."""

    return -5.0 * t**6 * q5_matrix_model(t)


def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = (
        Path("/System/Library/Fonts/HelveticaNeue.ttc"),
        Path("/System/Library/Fonts/Helvetica.ttc"),
        Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
    )
    for candidate in candidates:
        if candidate.exists():
            try:
                return ImageFont.truetype(str(candidate), size=size, index=1 if bold else 0)
            except OSError:
                continue
    return ImageFont.load_default()


def render_comparison_png(
    output: Path,
    *,
    t_points: np.ndarray,
    amplitude_points: np.ndarray,
    amplitude_errors: np.ndarray,
    q5_points: np.ndarray,
    q5_errors: np.ndarray,
    first_wall: float,
) -> None:
    """Render a publication-readable two-panel PNG using Pillow."""

    width, height = 1800, 1900
    background = (250, 250, 249)
    foreground = (32, 34, 38)
    border = (118, 122, 128)
    grid = (219, 221, 224)
    matrix_color = (213, 94, 0)
    worldsheet_color = (0, 114, 178)
    wall_color = (90, 94, 100)
    image = Image.new("RGB", (width, height), background)
    draw = ImageDraw.Draw(image)
    title_font = _font(48, bold=True)
    axis_font = _font(31)
    tick_font = _font(25)
    legend_font = _font(27)
    draw.text(
        (width / 2, 54),
        "Sphere 1 -> 5 amplitude below the first residue wall",
        fill=foreground,
        font=title_font,
        anchor="ma",
    )

    x_min, x_max = 0.13, 0.338
    t_curve = np.linspace(x_min, first_wall, 900, endpoint=False)
    panels = (
        {
            "top": 175,
            "bottom": 870,
            "curve": np.asarray(amplitude_imaginary_matrix_model(t_curve)),
            "points": amplitude_points,
            "errors": amplitude_errors,
            "ylabel": "Im[mu^4 A_1->5(i t)]",
        },
        {
            "top": 1030,
            "bottom": 1725,
            "curve": np.asarray(q5_matrix_model(t_curve)),
            "points": q5_points,
            "errors": q5_errors,
            "ylabel": "Q_5(i t)",
        },
    )
    left, right = 245, 1690
    x_ticks = (0.14, 0.18, 0.22, 0.26, 0.30, first_wall)

    def x_pixel(value: float) -> float:
        return left + (value - x_min) / (x_max - x_min) * (right - left)

    for panel_index, panel in enumerate(panels):
        top = int(panel["top"])
        bottom = int(panel["bottom"])
        curve = np.asarray(panel["curve"], dtype=float)
        observations = np.asarray(panel["points"], dtype=float)
        errors = np.asarray(panel["errors"], dtype=float)
        all_values = np.concatenate((curve, observations - errors, observations + errors))
        y_min = float(np.min(all_values))
        y_max = float(np.max(all_values))
        padding = 0.10 * max(y_max - y_min, 1.0e-12)
        y_min -= padding
        y_max += padding

        def y_pixel(value: float) -> float:
            return bottom - (value - y_min) / (y_max - y_min) * (bottom - top)

        y_ticks = np.linspace(y_min, y_max, 6)
        for value in y_ticks:
            py = y_pixel(float(value))
            draw.line((left, py, right, py), fill=grid, width=2)
            label = f"{value:.4g}"
            draw.text(
                (left - 20, py),
                label,
                fill=foreground,
                font=tick_font,
                anchor="rm",
            )
        for value in x_ticks:
            px = x_pixel(value)
            draw.line((px, top, px, bottom), fill=grid, width=2)
            label = "1/3" if abs(value - first_wall) < 1.0e-12 else f"{value:.2f}"
            draw.text(
                (px, bottom + 18),
                label,
                fill=foreground,
                font=tick_font,
                anchor="ma",
            )
        draw.rectangle((left, top, right, bottom), outline=border, width=3)
        if y_min <= 0.0 <= y_max:
            draw.line((left, y_pixel(0.0), right, y_pixel(0.0)), fill=border, width=3)

        curve_pixels = [
            (x_pixel(float(t)), y_pixel(float(value)))
            for t, value in zip(t_curve, curve)
        ]
        draw.line(curve_pixels, fill=matrix_color, width=6, joint="curve")
        wall_x = x_pixel(first_wall)
        for segment_top in range(top, bottom, 24):
            draw.line(
                (wall_x, segment_top, wall_x, min(segment_top + 13, bottom)),
                fill=wall_color,
                width=4,
            )
        for t, value, error in zip(t_points, observations, errors):
            px = x_pixel(float(t))
            py = y_pixel(float(value))
            py_low = y_pixel(float(value - error))
            py_high = y_pixel(float(value + error))
            draw.line((px, py_low, px, py_high), fill=worldsheet_color, width=5)
            draw.line((px - 10, py_low, px + 10, py_low), fill=worldsheet_color, width=4)
            draw.line((px - 10, py_high, px + 10, py_high), fill=worldsheet_color, width=4)
            draw.ellipse(
                (px - 10, py - 10, px + 10, py + 10),
                fill=worldsheet_color,
                outline=background,
                width=3,
            )

        ylabel_layer = Image.new("RGBA", (800, 80), (0, 0, 0, 0))
        ylabel_draw = ImageDraw.Draw(ylabel_layer)
        ylabel_draw.text(
            (400, 40),
            str(panel["ylabel"]),
            fill=foreground,
            font=axis_font,
            anchor="mm",
        )
        ylabel_layer = ylabel_layer.rotate(90, expand=True)
        image.paste(
            ylabel_layer,
            (55, int((top + bottom - ylabel_layer.height) / 2)),
            ylabel_layer,
        )
        draw = ImageDraw.Draw(image)

        if panel_index == 0:
            legend_y = top + 38
            draw.line((left + 35, legend_y, left + 115, legend_y), fill=matrix_color, width=6)
            draw.text(
                (left + 132, legend_y),
                "matrix model",
                fill=foreground,
                font=legend_font,
                anchor="lm",
            )
            draw.line(
                (left + 420, legend_y - 17, left + 420, legend_y + 17),
                fill=worldsheet_color,
                width=5,
            )
            draw.ellipse(
                (left + 410, legend_y - 10, left + 430, legend_y + 10),
                fill=worldsheet_color,
            )
            draw.text(
                (left + 442, legend_y),
                "worldsheet (QMC + discretization)",
                fill=foreground,
                font=legend_font,
                anchor="lm",
            )

    draw.text(
        ((left + right) / 2, 1815),
        "t in omega = i t",
        fill=foreground,
        font=axis_font,
        anchor="ma",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output, format="PNG", optimize=True)


def main() -> None:
    base = Path(__file__).parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scan",
        type=Path,
        default=base
        / "results"
        / "sphere_six_point_1to5"
        / "worldsheet_convergent_scan.json",
    )
    parser.add_argument(
        "--freeze-manifest",
        type=Path,
        default=base
        / "results"
        / "sphere_six_point_1to5"
        / "worldsheet_freeze_manifest.json",
    )
    parser.add_argument(
        "--audit",
        type=Path,
        default=base
        / "results"
        / "sphere_six_point_1to5"
        / "worldsheet_numerical_audit.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=base
        / "results"
        / "sphere_six_point_1to5"
        / "matrix_model_comparison.json",
    )
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=base
        / "results"
        / "sphere_six_point_1to5"
        / "matrix_model_comparison.csv",
    )
    parser.add_argument(
        "--figure",
        type=Path,
        default=base
        / "figures"
        / "sphere_one_to_five_amplitude_residue_free.png",
    )
    arguments = parser.parse_args()

    manifest = json.loads(arguments.freeze_manifest.read_text())
    actual_checksum = sha256(arguments.scan)
    expected_checksum = str(manifest["sha256"])
    if actual_checksum != expected_checksum:
        raise RuntimeError(
            "the worldsheet scan changed after freezing: "
            f"expected {expected_checksum}, found {actual_checksum}"
        )
    scan = json.loads(arguments.scan.read_text())
    if scan["status"] != "worldsheet_only_no_matrix_model_imported":
        raise RuntimeError("input scan is not labelled worldsheet-only")
    audit = json.loads(arguments.audit.read_text())
    discretization_q5 = float(audit["diagnostics"]["combined_discretization_Q5"])

    points: list[dict[str, float]] = []
    for source in scan["points"]:
        t = float(source["t"])
        q5_worldsheet = float(source["Q5_worldsheet"]["real"])
        q5_qmc = float(source["Q5_worldsheet_standard_error"]["real"])
        q5_combined = math.hypot(q5_qmc, discretization_q5)
        amplitude_worldsheet = float(source["mu4_A_tree_worldsheet"]["imag"])
        amplitude_qmc = float(
            source["mu4_A_tree_worldsheet_standard_error"]["imag"]
        )
        amplitude_discretization = 5.0 * t**6 * discretization_q5
        amplitude_combined = math.hypot(amplitude_qmc, amplitude_discretization)
        q5_target = float(q5_matrix_model(t))
        amplitude_target = float(amplitude_imaginary_matrix_model(t))
        points.append(
            {
                "t": t,
                "q5_worldsheet": q5_worldsheet,
                "q5_qmc_standard_error": q5_qmc,
                "q5_discretization_error": discretization_q5,
                "q5_combined_error": q5_combined,
                "q5_matrix_model": q5_target,
                "q5_residual": q5_worldsheet - q5_target,
                "q5_pull": (q5_worldsheet - q5_target) / q5_combined,
                "amplitude_imaginary_worldsheet": amplitude_worldsheet,
                "amplitude_imaginary_qmc_standard_error": amplitude_qmc,
                "amplitude_imaginary_discretization_error": amplitude_discretization,
                "amplitude_imaginary_combined_error": amplitude_combined,
                "amplitude_imaginary_matrix_model": amplitude_target,
                "amplitude_imaginary_residual": amplitude_worldsheet
                - amplitude_target,
                "amplitude_imaginary_pull": (
                    amplitude_worldsheet - amplitude_target
                )
                / amplitude_combined,
            }
        )

    pulls = np.asarray([point["q5_pull"] for point in points])
    summary = {
        "input_worldsheet_sha256": actual_checksum,
        "comparison_performed_after_freeze": True,
        "first_residue_wall": float(scan["kinematic_domain"]["first_residue_wall"]),
        "matrix_model": {
            "Q5_omega": "(1+5 i omega)(2+5 i omega)(3+5 i omega)",
            "Q5_it": "(1-5 t)(2-5 t)(3-5 t)",
            "mu4_A_tree_it": "-5 i t^6 Q5(i t)",
        },
        "error_model": {
            "qmc": "standard error over independently scrambled Sobol replicates",
            "discretization_Q5": discretization_q5,
            "discretization_source": "quadrature sum of momentum order, block order, and momentum cutoff shifts at t=0.18",
            "combined": "quadrature sum of QMC and discretization errors",
        },
        "goodness": {
            "maximum_absolute_pull": float(np.max(np.abs(pulls))),
            "chi_squared": float(np.sum(pulls**2)),
            "degrees_of_freedom": len(points),
            "rms_pull": float(np.sqrt(np.mean(pulls**2))),
        },
        "points": points,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(summary, indent=2) + "\n")
    with arguments.csv_output.open("w", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(points[0]))
        writer.writeheader()
        writer.writerows(points)

    t_points = np.asarray([point["t"] for point in points])
    amplitude_points = np.asarray(
        [point["amplitude_imaginary_worldsheet"] for point in points]
    )
    amplitude_errors = np.asarray(
        [point["amplitude_imaginary_combined_error"] for point in points]
    )
    q5_points = np.asarray([point["q5_worldsheet"] for point in points])
    q5_errors = np.asarray([point["q5_combined_error"] for point in points])
    first_wall = float(scan["kinematic_domain"]["first_residue_wall"])
    t_curve = np.linspace(0.13, first_wall, 600, endpoint=False)

    render_comparison_png(
        arguments.figure,
        t_points=t_points,
        amplitude_points=amplitude_points,
        amplitude_errors=amplitude_errors,
        q5_points=q5_points,
        q5_errors=q5_errors,
        first_wall=first_wall,
    )

    print(json.dumps(summary["goodness"], indent=2))
    print(f"wrote {arguments.output}")
    print(f"wrote {arguments.csv_output}")
    print(f"wrote {arguments.figure}")


if __name__ == "__main__":
    main()
