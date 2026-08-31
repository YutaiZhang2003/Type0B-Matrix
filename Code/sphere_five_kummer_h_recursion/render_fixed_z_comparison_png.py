#!/usr/bin/env python3
"""Render the fixed-z h/c-recursion CSV as a high-resolution PNG."""

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

    t_values = [row["t"] for row in rows]
    h_values = [row["pillow_h_order10"] for row in rows]
    c_values = [row["c_recursion_order22"] for row in rows]
    ccy_h_values = [row["ccy_plane_h_order10"] for row in rows]
    comparison = [row["relative_h10_vs_c22"] for row in rows]
    h_shift = [row["relative_h_truncation_shift"] for row in rows]
    c_shift = [row["relative_c_truncation_shift"] for row in rows]
    ccy_comparison = [row["relative_ccy_h10_vs_c22"] for row in rows]
    ccy_h_shift = [row["relative_ccy_h_truncation_shift"] for row in rows]
    t_low, t_high = min(t_values), max(t_values)
    value_low = min(h_values + c_values + ccy_h_values)
    value_high = max(h_values + c_values + ccy_h_values)
    value_pad = 0.08 * max(value_high - value_low, abs(value_high), 1.0e-8)
    value_low -= value_pad
    value_high += value_pad
    error_floor = 1.0e-18
    all_errors = [
        max(value, error_floor)
        for value in comparison + h_shift + c_shift + ccy_comparison + ccy_h_shift
    ]
    log_low = min(-8.0, math.floor(math.log10(min(all_errors))))
    log_high = max(-2.0, math.ceil(math.log10(max(all_errors))))
    if log_high - log_low < 5:
        log_low = log_high - 5

    def x_coordinate(value: float) -> float:
        return x0 + (value - t_low) / (t_high - t_low) * (x1 - x0)

    def top_coordinate(value: float) -> float:
        return top_y1 - (value - value_low) / (value_high - value_low) * (top_y1 - top_y0)

    def bottom_coordinate(value: float) -> float:
        logarithm = math.log10(max(value, error_floor))
        return bottom_y1 - (logarithm - log_low) / (log_high - log_low) * (
            bottom_y1 - bottom_y0
        )

    title_font = font(38, bold=True)
    subtitle_font = font(24)
    label_font = font(29)
    tick_font = font(23)
    legend_font = font(23)
    title = "Sphere five-point block: pillow h-recursion vs. CCY recursions"
    subtitle = (
        "z=0.08, c=31.7, (d1,d2,d3,d4,d5)=(0.21,0.34,0.63,0.79,0.49), "
        "(h1,h2)=(1.03,1.19)"
    )
    draw.text((width / 2, 48), title, fill="#202124", font=title_font, anchor="ma")
    draw.text((width / 2, 112), subtitle, fill="#202124", font=subtitle_font, anchor="ma")

    grid_color = "#d9dde1"
    axis_color = "#202124"
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
        t_value = t_low + index * (t_high - t_low) / 6
        x = x_coordinate(t_value)
        draw.line((x, top_y0, x, top_y1), fill=grid_color, width=2)
        draw.line((x, bottom_y0, x, bottom_y1), fill=grid_color, width=2)
        draw.text((x, bottom_y1 + 20), f"{t_value:.2f}", fill=axis_color, font=tick_font, anchor="ma")

    for y_start, y_end in ((top_y0, top_y1), (bottom_y0, bottom_y1)):
        draw.line((x0, y_start, x0, y_end), fill=axis_color, width=3)
        draw.line((x0, y_end, x1, y_end), fill=axis_color, width=3)

    top_c_points = list(zip(map(x_coordinate, t_values), map(top_coordinate, c_values)))
    top_h_points = list(zip(map(x_coordinate, t_values), map(top_coordinate, h_values)))
    top_ccy_h_points = list(
        zip(map(x_coordinate, t_values), map(top_coordinate, ccy_h_values))
    )
    comparison_points = list(
        zip(map(x_coordinate, t_values), map(bottom_coordinate, comparison))
    )
    h_shift_points = list(zip(map(x_coordinate, t_values), map(bottom_coordinate, h_shift)))
    c_shift_points = list(zip(map(x_coordinate, t_values), map(bottom_coordinate, c_shift)))
    ccy_comparison_points = list(
        zip(map(x_coordinate, t_values), map(bottom_coordinate, ccy_comparison))
    )
    ccy_h_shift_points = list(
        zip(map(x_coordinate, t_values), map(bottom_coordinate, ccy_h_shift))
    )
    draw.line(top_c_points, fill="#1f4e79", width=7, joint="curve")
    dashed_line(draw, top_ccy_h_points, fill="#1b9e77", width=5, dash=5, gap=10)
    dashed_line(draw, top_h_points, fill="#d95f02", width=5, dash=18, gap=12)
    draw.line(comparison_points, fill="#7b3294", width=7, joint="curve")
    draw.line(ccy_comparison_points, fill="#1b9e77", width=6, joint="curve")
    dashed_line(draw, h_shift_points, fill="#d95f02", width=4, dash=5, gap=10)
    dashed_line(draw, ccy_h_shift_points, fill="#1b9e77", width=4, dash=5, gap=10)
    dashed_line(draw, c_shift_points, fill="#1f4e79", width=4, dash=18, gap=10)

    y_label = Image.new("RGBA", (620, 60), (255, 255, 255, 0))
    y_draw = ImageDraw.Draw(y_label)
    y_draw.text((310, 30), "chiral block F5(z,t)", fill=axis_color, font=label_font, anchor="mm")
    image.paste(y_label.rotate(90, expand=True), (25, int((top_y0 + top_y1) / 2 - 310)), y_label.rotate(90, expand=True))
    y_label2 = Image.new("RGBA", (520, 60), (255, 255, 255, 0))
    y_draw2 = ImageDraw.Draw(y_label2)
    y_draw2.text((260, 30), "relative difference", fill=axis_color, font=label_font, anchor="mm")
    rotated2 = y_label2.rotate(90, expand=True)
    image.paste(rotated2, (25, int((bottom_y0 + bottom_y1) / 2 - 260)), rotated2)
    draw.text(
        ((x0 + x1) / 2, height - 43),
        "mobile insertion t  (z < t < 1)",
        fill=axis_color,
        font=label_font,
        anchor="ms",
    )

    legend_x, legend_y = x1 - 465, top_y0 + 38
    top_legend = (
        ("#1f4e79", False, "CCY c-recursion, N=22"),
        ("#1b9e77", True, "CCY plane h-recursion, N=10"),
        ("#d95f02", True, "pillow h-recursion, N=10"),
    )
    for index, (color, dashed, text) in enumerate(top_legend):
        y = legend_y + 44 * index
        if dashed:
            dashed_line(draw, [(legend_x, y), (legend_x + 78, y)], fill=color, width=6, dash=16, gap=10)
        else:
            draw.line((legend_x, y, legend_x + 78, y), fill=color, width=7)
        draw.text((legend_x + 95, y), text, fill=axis_color, font=legend_font, anchor="lm")

    legend_x, legend_y = x0 + 30, bottom_y0 + 34
    bottom_legend = (
        ("#7b3294", False, "pillow h10 vs. c22"),
        ("#1b9e77", False, "CCY plane h10 vs. c22"),
        ("#d95f02", True, "pillow shift N=8 to 10"),
        ("#1b9e77", True, "CCY h shift N=8 to 10"),
        ("#1f4e79", True, "c shift N=20 to 22"),
    )
    draw.rounded_rectangle(
        (x0 + 15, bottom_y0 + 14, x0 + 975, bottom_y0 + 137),
        radius=10,
        fill="white",
        outline="#c8cdd2",
        width=2,
    )
    for index, (color, dashed, text) in enumerate(bottom_legend):
        column = 0 if index < 3 else 1
        row = index if index < 3 else index - 3
        item_x = legend_x + 485 * column
        y = legend_y + 39 * row
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
        default=DEFAULT_DATA / "sphere_five_pillow_h_vs_c_fixed_z.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_DATA / "sphere_five_pillow_h_vs_c_fixed_z.png",
    )
    arguments = parser.parse_args()
    render(arguments.csv.resolve(), arguments.output.resolve())


if __name__ == "__main__":
    main()
