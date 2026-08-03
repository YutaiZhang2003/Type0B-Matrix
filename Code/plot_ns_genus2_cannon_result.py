#!/usr/bin/env python3
"""Plot the order-eight five-point genus-two Cannon result as standalone SVG."""

from __future__ import annotations

import argparse
from html import escape
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "Data Set" / "ns_genus2_cannon_fivepoint_order8_summary.json"
OUTPUT = ROOT / "Data Set" / "ns_genus2_cannon_fivepoint_order8.svg"


def _map(value: float, start: float, stop: float, pixel_start: float, pixel_stop: float) -> float:
    return pixel_start + (value - start) * (pixel_stop - pixel_start) / (stop - start)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DATA)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    summary = json.loads(args.input.read_text())
    primary_radius = float(summary["config"]["finite_part_radii"][0])
    point_ids = [point["id"] for point in summary["config"]["points"]]
    rows = {
        (row["point_id"], row["channel"]): row
        for row in summary["rows"]
        if float(row["finite_part_radius"]) == primary_radius
    }
    crossings = {
        row["point_id"]: row
        for row in summary["crossing"]
        if float(row["finite_part_radius"]) == primary_radius
    }

    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1800" height="1120" viewBox="0 0 1800 1120" role="img">',
        '<title>Order-eight five-point genus-two Liouville quotient in theta and glasses channels</title>',
        '<desc>The channels differ by three to four orders of magnitude, but the glasses global seed did not converge at occupation ceiling 22.</desc>',
        '<rect width="1800" height="1120" fill="#fbfaf7"/>',
        '<style>text{font-family:Arial,Helvetica,sans-serif;fill:#242424}.title{font-size:34px;font-weight:700}.label{font-size:24px}.small{font-size:19px}.tiny{font-size:17px}.grid{stroke:#ddd9d0;stroke-width:2}.axis{stroke:#555;stroke-width:2;fill:none}.theta{fill:#176b87}.glasses{fill:#c65d37}.warning{fill:#a23a2a}</style>',
        '<text x="900" y="48" text-anchor="middle" class="title">Genus-two NS super-Liouville, hat c=9: order-8 five-point Cannon run</text>',
        '<text x="900" y="82" text-anchor="middle" class="small">No channel matching imposed; Q_L = Z_Liouville / Z_free-superfield^9</text>',
    ]
    if summary.get("free_rerun"):
        parts.append(
            '<text x="900" y="112" text-anchor="middle" class="tiny">Free denominator rerun with the period-matched theta Schottky marking; stored Liouville numerators preserved</text>'
        )

    x0, y0, x1, y1 = 130, 145, 1690, 650
    y_min, y_max = -19.5, -14.0
    for exponent in range(-19, -13):
        yy = _map(exponent, y_min, y_max, y1, y0)
        parts.append(f'<line x1="{x0}" y1="{yy:.2f}" x2="{x1}" y2="{yy:.2f}" class="grid"/>')
        parts.append(f'<text x="{x0-16}" y="{yy+6:.2f}" text-anchor="end" class="small">10^{exponent}</text>')
    parts.extend(
        [
            f'<rect x="{x0}" y="{y0}" width="{x1-x0}" height="{y1-y0}" class="axis"/>',
            f'<text x="{(x0+x1)/2}" y="{y0-28}" text-anchor="middle" class="label">Normalized partition in both channels</text>',
            f'<text x="42" y="{(y0+y1)/2}" text-anchor="middle" class="label" transform="rotate(-90 42 {(y0+y1)/2})">Q_L (log scale)</text>',
            f'<circle cx="{x0+26}" cy="{y0+26}" r="8" class="theta"/><text x="{x0+44}" y="{y0+33}" class="small">theta</text>',
            f'<rect x="{x0+142}" y="{y0+18}" width="16" height="16" class="glasses"/><text x="{x0+168}" y="{y0+33}" class="small">glasses</text>',
        ]
    )
    spacing = (x1 - x0) / len(point_ids)
    for index, point_id in enumerate(point_ids):
        xx = x0 + spacing * (index + 0.5)
        theta = float(rows[(point_id, "theta")]["q_l"])
        glasses = float(rows[(point_id, "glasses")]["q_l"])
        theta_y = _map(math.log10(theta), y_min, y_max, y1, y0)
        glasses_y = _map(math.log10(glasses), y_min, y_max, y1, y0)
        ratio = float(crossings[point_id]["theta_over_glasses"])
        parts.extend(
            [
                f'<line x1="{xx:.2f}" y1="{theta_y:.2f}" x2="{xx:.2f}" y2="{glasses_y:.2f}" stroke="#aaa69d" stroke-width="2"/>',
                f'<circle cx="{xx:.2f}" cy="{theta_y:.2f}" r="9" class="theta"/>',
                f'<rect x="{xx-9:.2f}" y="{glasses_y-9:.2f}" width="18" height="18" class="glasses"/>',
                f'<text x="{xx:.2f}" y="{y1+28}" text-anchor="middle" class="small">{escape(point_id)}</text>',
                f'<text x="{xx:.2f}" y="{theta_y+25:.2f}" text-anchor="middle" class="tiny theta">{theta:.2e}</text>',
                f'<text x="{xx:.2f}" y="{glasses_y-14:.2f}" text-anchor="middle" class="tiny glasses">{glasses:.2e}</text>',
                f'<text x="{xx:.2f}" y="{y1+58}" text-anchor="middle" class="tiny">theta/glasses = {ratio:.2e}</text>',
            ]
        )

    x0, y0, x1, y1 = 130, 780, 1690, 1030
    d_min, d_max = -8.5, -1.0
    for exponent in (-8, -6, -4, -2):
        yy = _map(exponent, d_min, d_max, y1, y0)
        parts.append(f'<line x1="{x0}" y1="{yy:.2f}" x2="{x1}" y2="{yy:.2f}" class="grid"/>')
        parts.append(f'<text x="{x0-16}" y="{yy+6:.2f}" text-anchor="end" class="small">10^{exponent}</text>')
    parts.extend(
        [
            f'<rect x="{x0}" y="{y0}" width="{x1-x0}" height="{y1-y0}" class="axis"/>',
            f'<text x="{(x0+x1)/2}" y="{y0-28}" text-anchor="middle" class="label">Glasses global-seed last-shell diagnostic</text>',
            f'<text x="42" y="{(y0+y1)/2}" text-anchor="middle" class="label" transform="rotate(-90 42 {(y0+y1)/2})">relative last shell (log scale)</text>',
        ]
    )
    tolerance = float(summary["config"]["numerics"]["global_tolerance"])
    tolerance_y = _map(math.log10(tolerance), d_min, d_max, y1, y0)
    parts.append(f'<line x1="{x0}" y1="{tolerance_y:.2f}" x2="{x1}" y2="{tolerance_y:.2f}" stroke="#4b8f4b" stroke-width="3"/>')
    parts.append(f'<text x="{x1-8}" y="{tolerance_y-8:.2f}" text-anchor="end" class="tiny" fill="#356a35">requested tolerance 2e-8</text>')
    for index, point_id in enumerate(point_ids):
        xx = x0 + spacing * (index + 0.5)
        diagnostic = float(rows[(point_id, "glasses")]["global_worst_last_shell_relative"])
        yy = _map(math.log10(diagnostic), d_min, d_max, y1, y0)
        parts.extend(
            [
                f'<line x1="{xx:.2f}" y1="{yy:.2f}" x2="{xx:.2f}" y2="{y1}" stroke="#c65d37" stroke-width="3"/>',
                f'<rect x="{xx-9:.2f}" y="{yy-9:.2f}" width="18" height="18" class="glasses"/>',
                f'<text x="{xx:.2f}" y="{yy-14:.2f}" text-anchor="middle" class="tiny glasses">{100*diagnostic:.2f}%</text>',
                f'<text x="{xx:.2f}" y="{y1+28}" text-anchor="middle" class="small">{escape(point_id)}</text>',
            ]
        )
    parts.extend(
        [
            '<text x="900" y="1090" text-anchor="middle" class="small warning">The glasses values are not converged at occupation ceiling 22; the channel discrepancy is not a physical locality conclusion.</text>',
            '</svg>',
        ]
    )
    args.output.write_text("\n".join(parts) + "\n")


if __name__ == "__main__":
    main()
