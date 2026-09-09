"""Reproduce the BRY Figure 4 VWWV crossing benchmark.

The direct curve is -H(4321|z) at the BRY reference order q^8.  The crossed
curve is -|z|^alpha H(4231|1/z) at a selectable order q^L (q^12 by default),
with the momenta and normalizations of arXiv:2201.05621, Figure 4.  A tiny
positive imaginary part on 1/z fixes the conjugate boundary values across
the real branch cut.

The script has no plotting dependency: it writes a self-contained SVG.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import html
import json
import math
from pathlib import Path
from typing import Iterable, Sequence

from sphere_four_point import BRYNSFourPointCorrelator


DATA_ROOT = Path(__file__).resolve().parents[2] / "Data Set"
MOMENTA = (0.5, 1.0 / 3.0, 0.25, 0.6)
BRY_REFERENCE_ORDER = 8
DEFAULT_CROSSING_ORDER = 12
DEFAULT_QUADRATURE_ORDER = 24
_SUPERSCRIPT_DIGITS = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")


def _q_label(order: int) -> str:
    return "q" + str(order).translate(_SUPERSCRIPT_DIGITS)


@dataclass(frozen=True)
class CrossingData:
    z: tuple[float, ...]
    direct: tuple[float, ...]
    crossed: tuple[float, ...]
    crossing_order: int


class BRYFigure4Benchmark:
    """Evaluate the direct and crossed BRY VWWV correlators."""

    def __init__(
        self,
        *,
        p_max: float = 5.0,
        quadrature_order: int = DEFAULT_QUADRATURE_ORDER,
        branch_epsilon: float = 1.0e-8,
    ) -> None:
        p1, p2, p3, p4 = MOMENTA
        self.p_max = float(p_max)
        self.quadrature_order = int(quadrature_order)
        self.branch_epsilon = float(branch_epsilon)
        self.direct = BRYNSFourPointCorrelator(
            p1=p1,
            p2=p2,
            p3=p3,
            p4=p4,
            bry_q_order=BRY_REFERENCE_ORDER,
        )
        self.crossed = BRYNSFourPointCorrelator(
            p1=p1,
            p2=p3,
            p3=p2,
            p4=p4,
            bry_q_order=DEFAULT_CROSSING_ORDER,
        )
        h1, h2, h3, h4 = [
            self.direct.block_weight(momentum) for momentum in MOMENTA
        ]
        self.crossing_exponent = 2.0 * (
            h4 - (h3 + 0.5) - (h2 + 0.5) - h1
        )

    def _integrate_h(self, correlator: BRYNSFourPointCorrelator, z: complex) -> complex:
        return correlator.evaluate_h(
            z, p_max=self.p_max, quadrature_order=self.quadrature_order
        )

    def direct_value(self, z: float) -> float:
        return -self._integrate_h(self.direct, z).real

    def crossed_value(self, z: float, order: int) -> float:
        if not isinstance(order, int) or order < 1:
            raise ValueError("order must be a positive integer")
        self.crossed.bry_q_order = order
        inverse_z = 1.0 / z + 1j * self.branch_epsilon
        transformed = z**self.crossing_exponent * self._integrate_h(
            self.crossed, inverse_z
        )
        return -transformed.real

    def sample(
        self,
        z_values: Iterable[float],
        crossing_order: int = DEFAULT_CROSSING_ORDER,
    ) -> CrossingData:
        z = tuple(float(value) for value in z_values)
        if any(value <= 0.0 or value >= 1.0 for value in z):
            raise ValueError("Figure 4 sampling requires 0 < z < 1")
        direct = tuple(self.direct_value(value) for value in z)
        crossed = tuple(
            self.crossed_value(value, crossing_order) for value in z
        )
        return CrossingData(
            z=z,
            direct=direct,
            crossed=crossed,
            crossing_order=crossing_order,
        )


def _linspace(start: float, stop: float, count: int) -> tuple[float, ...]:
    if count < 2:
        raise ValueError("a plot grid needs at least two points")
    step = (stop - start) / (count - 1)
    return tuple(start + index * step for index in range(count))


def _path(
    x_values: Sequence[float],
    y_values: Sequence[float],
    x_map,
    y_map,
) -> str:
    points = [
        (x_map(x_value), y_map(y_value))
        for x_value, y_value in zip(x_values, y_values)
    ]
    return " ".join(
        ("M" if index == 0 else "L") + f"{x:.3f},{y:.3f}"
        for index, (x, y) in enumerate(points)
    )


def _nice_ticks(low: float, high: float, count: int = 5) -> tuple[float, ...]:
    span = high - low
    raw = span / max(1, count - 1)
    exponent = 10.0 ** math.floor(math.log10(raw))
    fraction = raw / exponent
    step = (1.0 if fraction <= 1.0 else 2.0 if fraction <= 2.0 else 5.0) * exponent
    first = math.ceil(low / step) * step
    ticks = []
    value = first
    while value <= high + 1.0e-12:
        ticks.append(value)
        value += step
    return tuple(ticks)


def render_svg(data: CrossingData, output: Path) -> None:
    """Render the BRY full-range and zoomed crossing panels."""

    width, height = 1240, 520
    panel_width, panel_height = 500, 360
    top, left_full, left_zoom = 90, 90, 700
    crossed_color = "#2748a8"

    all_values = list(data.direct) + list(data.crossed)
    y_full_low = 0.0
    y_full_high = max(all_values) * 1.06
    zoom_indices = [
        index for index, value in enumerate(data.z) if 0.01 <= value <= 0.011
    ]
    zoom_values = [data.direct[index] for index in zoom_indices]
    zoom_values.extend(data.crossed[index] for index in zoom_indices)
    zoom_low, zoom_high = min(zoom_values), max(zoom_values)
    zoom_pad = 0.08 * (zoom_high - zoom_low)
    zoom_low -= zoom_pad
    zoom_high += zoom_pad

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#1d1d1f}'
        '.axis{stroke:#333;stroke-width:1}.grid{stroke:#d7d7d7;stroke-width:1}'
        '.series{fill:none;stroke-width:2}.direct{fill:none;stroke:#111;stroke-width:3}'
        '.tick{font-size:13px}.label{font-size:16px}.legend{font-size:13px}</style>',
        '<text x="620" y="28" text-anchor="middle" font-size="20">'
        'BRY Figure 4: VWWV crossing convergence</text>',
        '<text x="620" y="54" text-anchor="middle" font-size="14">'
        f'P₁=1/2, P₂=1/3, P₃=1/4, P₄=3/5; direct q⁸ versus crossed '
        f'{_q_label(data.crossing_order)}</text>',
    ]

    def draw_panel(
        left: float,
        x_low: float,
        x_high: float,
        y_low: float,
        y_high: float,
        indices: Sequence[int],
        title: str,
    ) -> None:
        x_map = lambda value: left + panel_width * (value - x_low) / (x_high - x_low)
        y_map = lambda value: top + panel_height * (y_high - value) / (y_high - y_low)
        parts.append(
            f'<text x="{left + panel_width / 2}" y="{top - 18}" '
            f'text-anchor="middle" class="label">{html.escape(title)}</text>'
        )
        for tick in _nice_ticks(y_low, y_high, 6):
            y = y_map(tick)
            parts.append(
                f'<line x1="{left}" y1="{y:.3f}" x2="{left + panel_width}" '
                f'y2="{y:.3f}" class="grid"/>'
            )
            parts.append(
                f'<text x="{left - 10}" y="{y + 4:.3f}" text-anchor="end" '
                f'class="tick">{tick:.3g}</text>'
            )
        for tick in _nice_ticks(x_low, x_high, 6):
            x = x_map(tick)
            parts.append(
                f'<line x1="{x:.3f}" y1="{top}" x2="{x:.3f}" '
                f'y2="{top + panel_height}" class="grid"/>'
            )
            label = f"{tick:.4f}" if x_high <= 0.02 else f"{tick:.2f}"
            parts.append(
                f'<text x="{x:.3f}" y="{top + panel_height + 22}" '
                f'text-anchor="middle" class="tick">{label}</text>'
            )
        parts.append(
            f'<line x1="{left}" y1="{top + panel_height}" '
            f'x2="{left + panel_width}" y2="{top + panel_height}" class="axis"/>'
        )
        parts.append(
            f'<line x1="{left}" y1="{top}" x2="{left}" '
            f'y2="{top + panel_height}" class="axis"/>'
        )
        x_values = [data.z[index] for index in indices]
        direct_values = [data.direct[index] for index in indices]
        direct_path = _path(x_values, direct_values, x_map, y_map)
        parts.append(f'<path d="{direct_path}" class="direct"/>')
        values = [data.crossed[index] for index in indices]
        path = _path(x_values, values, x_map, y_map)
        parts.append(
            f'<path d="{path}" class="series" stroke="{crossed_color}" '
            'stroke-dasharray="7 3" stroke-width="2.5"/>'
        )
        parts.append(
            f'<text x="{left + panel_width / 2}" y="{top + panel_height + 52}" '
            'text-anchor="middle" class="label">z</text>'
        )

    full_indices = tuple(range(len(data.z)))
    draw_panel(
        left_full,
        min(data.z),
        max(data.z),
        y_full_low,
        y_full_high,
        full_indices,
        "Full range",
    )
    draw_panel(
        left_zoom,
        0.01,
        0.011,
        zoom_low,
        zoom_high,
        zoom_indices,
        "BRY zoom window",
    )
    legend_y = 505
    legend_items = [
        ("direct q⁸", "#111"),
        (f"crossed {_q_label(data.crossing_order)}", crossed_color),
    ]
    for index, (label, color) in enumerate(legend_items):
        x = 440 + index * 190
        parts.append(
            f'<line x1="{x}" y1="{legend_y - 5}" x2="{x + 22}" '
            f'y2="{legend_y - 5}" stroke="{color}" stroke-width="3"/>'
        )
        parts.append(
            f'<text x="{x + 28}" y="{legend_y}" class="legend">'
            f'{html.escape(label)}</text>'
        )
    parts.append("</svg>")
    output.write_text("\n".join(parts), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DATA_ROOT / "bry_figure4_crossing.svg",
    )
    parser.add_argument("--p-max", type=float, default=5.0)
    parser.add_argument(
        "--quadrature-order", type=int, default=DEFAULT_QUADRATURE_ORDER
    )
    parser.add_argument("--full-points", type=int, default=25)
    parser.add_argument("--zoom-points", type=int, default=13)
    parser.add_argument("--branch-epsilon", type=float, default=1.0e-8)
    parser.add_argument(
        "--crossing-order", type=int, default=DEFAULT_CROSSING_ORDER
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    full = _linspace(0.01, 0.1, args.full_points)
    zoom = _linspace(0.01, 0.011, args.zoom_points)
    z_values = tuple(sorted(set(full + zoom)))
    benchmark = BRYFigure4Benchmark(
        p_max=args.p_max,
        quadrature_order=args.quadrature_order,
        branch_epsilon=args.branch_epsilon,
    )
    data = benchmark.sample(z_values, crossing_order=args.crossing_order)
    render_svg(data, args.output)
    index = min(range(len(data.z)), key=lambda item: abs(data.z[item] - 0.01))
    direct = data.direct[index]
    crossed = data.crossed[index]
    relative_error = abs(crossed - direct) / abs(direct)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "crossing_exponent": benchmark.crossing_exponent,
                "z": data.z[index],
                "direct": direct,
                "crossing_order": data.crossing_order,
                "quadrature_order": benchmark.quadrature_order,
                "p_max": benchmark.p_max,
                "crossed": crossed,
                "relative_error": relative_error,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
