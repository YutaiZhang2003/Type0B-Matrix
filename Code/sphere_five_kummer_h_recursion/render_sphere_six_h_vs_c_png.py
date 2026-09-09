#!/usr/bin/env python3
"""Render the fixed-(t1,t2) sphere-six h/c comparison as a PNG."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DEFAULT_DATA = ROOT / "Data Set" / "h-Recursion"
FONT_REGULAR = "/System/Library/Fonts/Supplemental/Arial.ttf"
FONT_BOLD = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"


def font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT_BOLD if bold else FONT_REGULAR, size)


def dashed_line(
    draw: ImageDraw.ImageDraw,
    points: list[tuple[float, float]],
    *,
    fill: str,
    width: int,
    dash: float,
    gap: float,
) -> None:
    for start, end in zip(points, points[1:]):
        x0, y0 = start
        x1, y1 = end
        distance = math.hypot(x1 - x0, y1 - y0)
        if distance == 0:
            continue
        position = 0.0
        while position < distance:
            finish = min(position + dash, distance)
            a = position / distance
            b = finish / distance
            draw.line(
                (
                    x0 + a * (x1 - x0),
                    y0 + a * (y1 - y0),
                    x0 + b * (x1 - x0),
                    y0 + b * (y1 - y0),
                ),
                fill=fill,
                width=width,
            )
            position += dash + gap


def render(csv_path: Path, output_path: Path) -> None:
    with csv_path.open(newline="", encoding="utf-8") as stream:
        rows = [
            {key: float(value) for key, value in row.items()}
            for row in csv.DictReader(stream)
        ]
    width, height = 1800, 1350
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    left, right = 175.0, 55.0
    top_y0, top_y1 = 215.0, 760.0
    bottom_y0, bottom_y1 = 915.0, 1220.0
    x0, x1 = left, width - right

    z_values = [row["z"] for row in rows]
    h_values = [row["pillow_h_order10"] for row in rows]
    c_values = [row["c_recursion_order20"] for row in rows]
    comparison = [row["relative_h10_vs_c20"] for row in rows]
    h_shift = [row["relative_h_truncation_shift"] for row in rows]
    c_shift = [row["relative_c_truncation_shift"] for row in rows]
    combined_shift = [a + b for a, b in zip(h_shift, c_shift)]
    z_low, z_high = min(z_values), max(z_values)
    value_low = min(h_values + c_values)
    value_high = max(h_values + c_values)
    value_pad = 0.08 * max(value_high - value_low, abs(value_high), 1.0e-8)
    value_low -= value_pad
    value_high += value_pad
    error_floor = 1.0e-18
    all_errors = [
        max(value, error_floor)
        for value in comparison + h_shift + c_shift + combined_shift
    ]
    log_low = min(-8.0, math.floor(math.log10(min(all_errors))))
    log_high = max(-2.0, math.ceil(math.log10(max(all_errors))))
    if log_high - log_low < 5:
        log_low = log_high - 5

    def x_coordinate(value: float) -> float:
        return x0 + (value - z_low) / (z_high - z_low) * (x1 - x0)

    def top_coordinate(value: float) -> float:
        return top_y1 - (value - value_low) / (value_high - value_low) * (top_y1 - top_y0)

    def bottom_coordinate(value: float) -> float:
        logarithm = math.log10(max(value, error_floor))
        return bottom_y1 - (logarithm - log_low) / (log_high - log_low) * (
            bottom_y1 - bottom_y0
        )

    title_font = font(38, bold=True)
    subtitle_font = font(23)
    label_font = font(29)
    tick_font = font(23)
    legend_font = font(23)
    axis_color, grid_color = "#202124", "#d9dde1"
    draw.text(
        (width / 2, 48),
        "Sphere six-point block: pillow h-recursion vs. c-recursion",
        fill=axis_color,
        font=title_font,
        anchor="ma",
    )
    draw.text(
        (width / 2, 112),
        "t1=0.32, t2=0.62, c=26.215; all weights fixed, z scanned in 0<z<t1",
        fill=axis_color,
        font=subtitle_font,
        anchor="ma",
    )

    for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
        value = value_low + fraction * (value_high - value_low)
        y = top_coordinate(value)
        draw.line((x0, y, x1, y), fill=grid_color, width=2)
        draw.text((x0 - 16, y), f"{value:.5f}", fill=axis_color, font=tick_font, anchor="rm")
    for exponent in range(int(log_low), int(log_high) + 1):
        y = bottom_coordinate(10.0**exponent)
        draw.line((x0, y, x1, y), fill=grid_color, width=2)
        draw.text((x0 - 16, y), f"10^{exponent}", fill=axis_color, font=tick_font, anchor="rm")
    for index in range(7):
        z_value = z_low + index * (z_high - z_low) / 6
        x = x_coordinate(z_value)
        draw.line((x, top_y0, x, top_y1), fill=grid_color, width=2)
        draw.line((x, bottom_y0, x, bottom_y1), fill=grid_color, width=2)
        draw.text((x, bottom_y1 + 20), f"{z_value:.3f}", fill=axis_color, font=tick_font, anchor="ma")
    for y_start, y_end in ((top_y0, top_y1), (bottom_y0, bottom_y1)):
        draw.line((x0, y_start, x0, y_end), fill=axis_color, width=3)
        draw.line((x0, y_end, x1, y_end), fill=axis_color, width=3)

    top_c = list(zip(map(x_coordinate, z_values), map(top_coordinate, c_values)))
    top_h = list(zip(map(x_coordinate, z_values), map(top_coordinate, h_values)))
    compare_points = list(
        zip(map(x_coordinate, z_values), map(bottom_coordinate, comparison))
    )
    h_shift_points = list(
        zip(map(x_coordinate, z_values), map(bottom_coordinate, h_shift))
    )
    c_shift_points = list(
        zip(map(x_coordinate, z_values), map(bottom_coordinate, c_shift))
    )
    combined_points = list(
        zip(map(x_coordinate, z_values), map(bottom_coordinate, combined_shift))
    )
    draw.line(top_c, fill="#1f4e79", width=7, joint="curve")
    dashed_line(draw, top_h, fill="#d95f02", width=5, dash=18, gap=12)
    draw.line(compare_points, fill="#7b3294", width=7, joint="curve")
    dashed_line(draw, h_shift_points, fill="#d95f02", width=4, dash=5, gap=10)
    dashed_line(draw, c_shift_points, fill="#1f4e79", width=4, dash=18, gap=10)
    dashed_line(draw, combined_points, fill="#777777", width=4, dash=10, gap=9)

    def vertical_label(text: str, center_y: float, canvas_width: int) -> None:
        label = Image.new("RGBA", (canvas_width, 60), (255, 255, 255, 0))
        label_draw = ImageDraw.Draw(label)
        label_draw.text(
            (canvas_width / 2, 30), text, fill=axis_color, font=label_font, anchor="mm"
        )
        rotated = label.rotate(90, expand=True)
        image.paste(rotated, (25, int(center_y - canvas_width / 2)), rotated)

    vertical_label("chiral block F6(z,t1,t2)", (top_y0 + top_y1) / 2, 650)
    vertical_label("relative difference", (bottom_y0 + bottom_y1) / 2, 520)
    draw.text(
        ((x0 + x1) / 2, height - 43),
        "first mobile insertion z  (0 < z < t1)",
        fill=axis_color,
        font=label_font,
        anchor="ms",
    )

    legend_x, legend_y = x1 - 515, top_y0 + 42
    top_legend = (
        ("#1f4e79", False, "c-recursion, total degree N=20"),
        ("#d95f02", True, "pillow h-recursion, total degree N=10"),
    )
    for index, (color, dashed, text) in enumerate(top_legend):
        y = legend_y + 44 * index
        if dashed:
            dashed_line(draw, [(legend_x, y), (legend_x + 78, y)], fill=color, width=6, dash=16, gap=10)
        else:
            draw.line((legend_x, y, legend_x + 78, y), fill=color, width=7)
        draw.text((legend_x + 95, y), text, fill=axis_color, font=legend_font, anchor="lm")

    legend_x, legend_y = x0 + 30, bottom_y0 + 37
    bottom_legend = (
        ("#7b3294", False, "|h10-c20| / |c20|"),
        ("#d95f02", True, "pillow shift N=8 to 10"),
        ("#1f4e79", True, "c shift N=18 to 20"),
        ("#777777", True, "sum of observed shifts"),
    )
    draw.rounded_rectangle(
        (x0 + 15, bottom_y0 + 14, x0 + 1055, bottom_y0 + 120),
        radius=10,
        fill="white",
        outline="#c8cdd2",
        width=2,
    )
    for index, (color, dashed, text) in enumerate(bottom_legend):
        column, row = index // 2, index % 2
        item_x = legend_x + 525 * column
        y = legend_y + 46 * row
        if dashed:
            dashed_line(draw, [(item_x, y), (item_x + 72, y)], fill=color, width=4, dash=12, gap=8)
        else:
            draw.line((item_x, y, item_x + 72, y), fill=color, width=7)
        draw.text((item_x + 87, y), text, fill=axis_color, font=legend_font, anchor="lm")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, format="PNG", optimize=True)
    print(f"wrote {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_DATA / "sphere_six_pillow_h_vs_c_fixed_t1_t2.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_DATA / "sphere_six_pillow_h_vs_c_fixed_t1_t2.png",
    )
    arguments = parser.parse_args()
    render(arguments.csv.resolve(), arguments.output.resolve())


if __name__ == "__main__":
    main()
