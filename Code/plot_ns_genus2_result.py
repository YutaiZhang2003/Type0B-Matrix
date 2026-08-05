#!/usr/bin/env python3
"""Plot the recorded first theta/glasses genus-two NS result with Pillow."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "Data Set" / "ns_genus2_theta_glasses_hatc9.json"
OUTPUT = ROOT / "Data Set" / "ns_genus2_theta_glasses_hatc9.png"


def _font(size: int, *, bold: bool = False):
    candidates = (
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
        if bold
        else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    )
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            pass
    return ImageFont.load_default()


def _map(value: float, start: float, stop: float, pixel_start: float, pixel_stop: float) -> float:
    return pixel_start + (value - start) * (pixel_stop - pixel_start) / (stop - start)


def _vertical_label(image: Image.Image, text: str, center: tuple[int, int], font) -> None:
    layer = Image.new("RGBA", (420, 52), (0, 0, 0, 0))
    layer_draw = ImageDraw.Draw(layer)
    layer_draw.text((210, 26), text, font=font, fill="#222", anchor="mm")
    rotated = layer.rotate(90, expand=True)
    image.paste(
        rotated,
        (center[0] - rotated.width // 2, center[1] - rotated.height // 2),
        rotated,
    )


def main() -> None:
    data = json.loads(DATA.read_text())
    scan = data["common_quadrature_N4_recursion_scan"]
    n6 = data["common_quadrature_N6_order4"]
    n8_theta = data["theta_quadrature_N8_order4"]

    image = Image.new("RGB", (1900, 820), "#fbfaf7")
    draw = ImageDraw.Draw(image)
    title = _font(34, bold=True)
    label = _font(24)
    small = _font(20)
    legend = _font(19)
    draw.text(
        (950, 28),
        "Genus-two NS super-Liouville, hat c=9, overlap point o0026",
        font=title,
        fill="#242424",
        anchor="ma",
    )

    panels = ((100, 125, 920, 700), (1030, 125, 1850, 700))
    theta_color = "#176b87"
    glasses_color = "#c65d37"
    ratio_color = "#75539b"
    grid_color = "#ddd9d0"

    # Left panel: the two absolute normalized partitions.
    x0, y0, x1, y1 = panels[0]
    q_min, q_max = 4.0e-19, 8.0e-19
    for tick in (4.0, 5.0, 6.0, 7.0, 8.0):
        y = _map(tick * 1.0e-19, q_min, q_max, y1, y0)
        draw.line((x0, y, x1, y), fill=grid_color, width=2)
        draw.text((x0 - 14, y), f"{tick:.0f}", font=small, fill="#444", anchor="rm")
    for order in (0, 3, 4):
        x = _map(order, 0, 4, x0, x1)
        draw.line((x, y0, x, y1), fill=grid_color, width=2)
        draw.text((x, y1 + 18), str(order), font=small, fill="#444", anchor="ma")
    draw.rectangle(panels[0], outline="#555", width=2)
    draw.text(((x0 + x1) / 2, y0 - 28), "Independent normalized partitions", font=label, fill="#222", anchor="ms")
    draw.text(((x0 + x1) / 2, y1 + 62), "c-recursion order", font=label, fill="#222", anchor="ma")
    _vertical_label(image, "Q_L  (x 10^-19)", (34, (y0 + y1) // 2), label)

    for channel, color, key, shape in (
        ("theta", theta_color, "theta_Q_L", "circle"),
        ("glasses", glasses_color, "glasses_Q_L", "square"),
    ):
        points = [
            (
                _map(row["recursion_order"], 0, 4, x0, x1),
                _map(row[key], q_min, q_max, y1, y0),
            )
            for row in scan
        ]
        draw.line(points, fill=color, width=5)
        for x, y in points:
            if shape == "circle":
                draw.ellipse((x - 8, y - 8, x + 8, y + 8), fill=color)
            else:
                draw.rectangle((x - 8, y - 8, x + 8, y + 8), fill=color)

    x_order4 = x1
    for value, color, offset in (
        (n6["theta_Q_L"], theta_color, -10),
        (n6["glasses_Q_L"], glasses_color, 10),
        (n8_theta["theta_Q_L"], "#123f52", -24),
    ):
        y = _map(value, q_min, q_max, y1, y0)
        draw.polygon(
            [(x_order4 + offset, y - 11), (x_order4 + offset + 11, y), (x_order4 + offset, y + 11), (x_order4 + offset - 11, y)],
            fill=color,
        )
    draw.text((x0 + 22, y0 + 205), "theta N=4", font=legend, fill=theta_color)
    draw.text((x0 + 22, y0 + 235), "glasses N=4", font=legend, fill=glasses_color)
    draw.text((x0 + 22, y0 + 265), "diamonds at R=4: N=6 (plus theta N=8)", font=legend, fill="#333")

    # Right panel: the ratio, with the equality line kept visible.
    x0, y0, x1, y1 = panels[1]
    r_min, r_max = 0.54, 1.03
    for tick in (0.6, 0.7, 0.8, 0.9, 1.0):
        y = _map(tick, r_min, r_max, y1, y0)
        draw.line((x0, y, x1, y), fill=grid_color, width=2)
        draw.text((x0 - 14, y), f"{tick:.1f}", font=small, fill="#444", anchor="rm")
    for order in (0, 3, 4):
        x = _map(order, 0, 4, x0, x1)
        draw.line((x, y0, x, y1), fill=grid_color, width=2)
        draw.text((x, y1 + 18), str(order), font=small, fill="#444", anchor="ma")
    equality_y = _map(1.0, r_min, r_max, y1, y0)
    draw.line((x0, equality_y, x1, equality_y), fill="#555", width=3)
    draw.text((x1 - 8, equality_y - 10), "matching", font=legend, fill="#555", anchor="rs")
    draw.rectangle(panels[1], outline="#555", width=2)
    draw.text(((x0 + x1) / 2, y0 - 28), "No matching imposed", font=label, fill="#222", anchor="ms")
    draw.text(((x0 + x1) / 2, y1 + 62), "c-recursion order", font=label, fill="#222", anchor="ma")
    _vertical_label(image, "Q_theta / Q_glasses", (965, (y0 + y1) // 2), label)
    ratio_points = [
        (
            _map(row["recursion_order"], 0, 4, x0, x1),
            _map(row["theta_over_glasses"], r_min, r_max, y1, y0),
        )
        for row in scan
    ]
    draw.line(ratio_points, fill=ratio_color, width=5)
    for x, y in ratio_points:
        draw.ellipse((x - 8, y - 8, x + 8, y + 8), fill=ratio_color)
    common_n6_y = _map(n6["theta_over_glasses"], r_min, r_max, y1, y0)
    draw.polygon(
        [(x1, common_n6_y - 12), (x1 + 12, common_n6_y), (x1, common_n6_y + 12), (x1 - 12, common_n6_y)],
        fill="#4b8f4b",
    )
    draw.text((x0 + 22, y0 + 22), "circles: common N=4", font=legend, fill=ratio_color)
    draw.text((x0 + 22, y0 + 52), "diamond: common N=6", font=legend, fill="#4b8f4b")

    draw.text(
        (950, 786),
        "Order-four common-N=6 ratio = 0.579555; the observed 42% difference is reported, not tuned away.",
        font=small,
        fill="#333",
        anchor="ma",
    )
    image.save(OUTPUT)


if __name__ == "__main__":
    main()
