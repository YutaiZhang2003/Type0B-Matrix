#!/usr/bin/env python3
"""Render an exploratory worldsheet amplitude-versus-t SVG."""

from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path
from typing import Sequence


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", default="Type 0B sphere five-tachyon amplitude")
    return parser


def _ticks(lower: float, upper: float, count: int = 5) -> list[float]:
    if lower == upper:
        scale = max(abs(lower), 1.0)
        lower -= 0.1 * scale
        upper += 0.1 * scale
    return [lower + index * (upper - lower) / (count - 1) for index in range(count)]


def _format_tick(value: float) -> str:
    if value == 0.0:
        return "0"
    if abs(value) >= 1.0e4 or abs(value) < 1.0e-3:
        return f"{value:.2e}"
    return f"{value:.4g}"


def render(payload: dict[str, object], title: str) -> str:
    if payload.get("matrix_model_used") is not False:
        raise ValueError("the input is not marked as a matrix-model-blind scan")
    points = list(payload.get("points", []))
    if len(points) < 2:
        raise ValueError("the plot requires at least two completed path points")
    points.sort(key=lambda item: float(item["t"]))

    width, height = 980.0, 650.0
    left, right = 105.0, 945.0
    panels = ((90.0, 305.0, "Re Aₜ⁵"), (385.0, 600.0, "Im Aₜ⁵"))
    t_values = [float(item["t"]) for item in points]
    t_min, t_max = min(t_values), max(t_values)
    t_padding = 0.04 * (t_max - t_min)
    t_min -= t_padding
    t_max += t_padding

    pieces = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width:.0f}" height="{height:.0f}" viewBox="0 0 {width:.0f} {height:.0f}">',
        "<style>text{font-family:Helvetica,Arial,sans-serif;fill:#222}.axis{stroke:#333;stroke-width:1}.grid{stroke:#ddd;stroke-width:1}.curve{fill:none;stroke:#1769aa;stroke-width:2}.mark{fill:#1769aa}.error{stroke:#1769aa;stroke-width:1.2}.note{font-size:13px;fill:#555}</style>",
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2:.1f}" y="34" text-anchor="middle" font-size="21">{html.escape(title)}</text>',
        '<text x="490" y="58" text-anchor="middle" class="note">Worldsheet-only, literal all-NS amplitude Aₜ⁵/[gₛ⁵ Cₛ² δ(E)] = i ∫I_NS/64</text>',
    ]

    for component_index, (top, bottom, label) in enumerate(panels):
        values = [
            float(item["literal_all_ns_stripped_amplitude"][("real", "imag")[component_index]])
            for item in points
        ]
        errors = [
            float(item["literal_all_ns_stripped_standard_error"][("real", "imag")[component_index]])
            for item in points
        ]
        y_min = min(value - error for value, error in zip(values, errors))
        y_max = max(value + error for value, error in zip(values, errors))
        padding = 0.12 * max(y_max - y_min, max(abs(y_min), abs(y_max), 1.0) * 0.02)
        y_min -= padding
        y_max += padding

        def xmap(value: float) -> float:
            return left + (value - t_min) * (right - left) / (t_max - t_min)

        def ymap(value: float) -> float:
            return bottom - (value - y_min) * (bottom - top) / (y_max - y_min)

        pieces.append(f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" class="axis"/>')
        pieces.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" class="axis"/>')
        pieces.append(f'<text x="25" y="{(top+bottom)/2:.1f}" font-size="15" transform="rotate(-90 25 {(top+bottom)/2:.1f})" text-anchor="middle">{label}</text>')
        for tick in _ticks(y_min, y_max):
            y = ymap(tick)
            pieces.append(f'<line x1="{left}" y1="{y:.2f}" x2="{right}" y2="{y:.2f}" class="grid"/>')
            pieces.append(f'<text x="{left-9}" y="{y+4:.2f}" text-anchor="end" font-size="12">{html.escape(_format_tick(tick))}</text>')
        path = " ".join(
            ("M" if index == 0 else "L") + f" {xmap(t):.2f} {ymap(value):.2f}"
            for index, (t, value) in enumerate(zip(t_values, values))
        )
        pieces.append(f'<path d="{path}" class="curve"/>')
        for t, value, error in zip(t_values, values, errors):
            x = xmap(t)
            y = ymap(value)
            y_low = ymap(value - error)
            y_high = ymap(value + error)
            pieces.append(f'<line x1="{x:.2f}" y1="{y_low:.2f}" x2="{x:.2f}" y2="{y_high:.2f}" class="error"/>')
            pieces.append(f'<line x1="{x-4:.2f}" y1="{y_low:.2f}" x2="{x+4:.2f}" y2="{y_low:.2f}" class="error"/>')
            pieces.append(f'<line x1="{x-4:.2f}" y1="{y_high:.2f}" x2="{x+4:.2f}" y2="{y_high:.2f}" class="error"/>')
            pieces.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3.3" class="mark"/>')

    for tick in _ticks(min(t_values), max(t_values)):
        x = left + (tick - t_min) * (right - left) / (t_max - t_min)
        pieces.append(f'<line x1="{x:.2f}" y1="600" x2="{x:.2f}" y2="606" class="axis"/>')
        pieces.append(f'<text x="{x:.2f}" y="623" text-anchor="middle" font-size="12">{tick:.4f}</text>')
    pieces.append('<text x="525" y="646" text-anchor="middle" font-size="15">t along ω(t)=√(t²−t/2+1/20)+it</text>')
    settings = payload.get("settings", {})
    pieces.append(
        f'<text x="{right}" y="74" text-anchor="end" class="note">Sobol 2^{html.escape(str(settings.get("sobol_power", "?")))} × {html.escape(str(settings.get("replicates", "?")))} replicates; error bars are replicate standard errors</text>'
    )
    pieces.append("</svg>")
    return "\n".join(pieces) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    payload = json.loads(args.input.read_text())
    svg = render(payload, args.title)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(svg)
    temporary.replace(output)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
