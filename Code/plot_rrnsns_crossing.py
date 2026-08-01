"""Plot the NS- and R-exchange channels of the BRY RRNSNS correlator.

The insertion ordering is

    <NS(P4, infinity) NS(P3, 1) R(P2, z) R(P1, 0)>.

The direct curve uses NS exchange at ``z``.  The crossed curve uses R
exchange at ``1-z``.  The default momenta and numerical cutoffs are the
crossing benchmark recorded in ``super_zamolodchikov_recursion.tex``.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Iterable, Sequence

from ramond_sphere_correlators import (
    RRNNMixedChannelCorrelator,
    SymmetricRRNNMixedChannelCorrelator,
)


DATA_ROOT = Path(__file__).resolve().parent.parent / "Data Set"
DEFAULT_MOMENTA = (0.20, 0.40, 0.30, 0.30)


@dataclass(frozen=True)
class CrossingSample:
    z: tuple[float, ...]
    ns_channel: tuple[float, ...]
    r_channel: tuple[float, ...]
    signed_difference_ppm: tuple[float, ...]
    p1_r: float
    p2_r: float
    p3_ns: float
    p4_ns: float
    block_order: int
    quadrature_order: int
    p_max: float
    central_charge_shift: float
    symmetric_regulator: bool


class RRNSNSCrossingBenchmark:
    """Sample the two factorizations of one mixed sphere correlator."""

    def __init__(
        self,
        *,
        p1_r: float = DEFAULT_MOMENTA[0],
        p2_r: float = DEFAULT_MOMENTA[1],
        p3_ns: float = DEFAULT_MOMENTA[2],
        p4_ns: float = DEFAULT_MOMENTA[3],
        block_order: int = 8,
        structure_precision: int = 30,
        central_charge_shift: float = 3.0e-5,
        symmetric_regulator: bool = False,
        p_max: float = 5.0,
        quadrature_order: int = 24,
    ) -> None:
        self.p1_r = float(p1_r)
        self.p2_r = float(p2_r)
        self.p3_ns = float(p3_ns)
        self.p4_ns = float(p4_ns)
        self.block_order = int(block_order)
        self.central_charge_shift = float(central_charge_shift)
        self.symmetric_regulator = bool(symmetric_regulator)
        self.p_max = float(p_max)
        self.quadrature_order = int(quadrature_order)
        correlator_type = (
            SymmetricRRNNMixedChannelCorrelator
            if self.symmetric_regulator
            else RRNNMixedChannelCorrelator
        )
        self.correlator = correlator_type(
            p1_r=self.p1_r,
            p2_r=self.p2_r,
            p3_ns=self.p3_ns,
            p4_ns=self.p4_ns,
            block_order=self.block_order,
            structure_precision=structure_precision,
            central_charge_shift=self.central_charge_shift,
        )

    def sample(self, z_values: Iterable[float]) -> CrossingSample:
        z = tuple(float(value) for value in z_values)
        if any(value <= 0.0 or value >= 1.0 for value in z):
            raise ValueError("sampling requires 0 < z < 1")
        ns_values = []
        r_values = []
        signed_difference_ppm = []
        for value in z:
            ns_value = self.correlator.evaluate_ns_channel(
                value,
                p_max=self.p_max,
                quadrature_order=self.quadrature_order,
            ).real
            r_value = self.correlator.evaluate_crossed_r_channel(
                value,
                p_max=self.p_max,
                quadrature_order=self.quadrature_order,
            ).real
            ns_values.append(ns_value)
            r_values.append(r_value)
            signed_difference_ppm.append(
                1.0e6 * (r_value - ns_value) / ns_value
            )
        return CrossingSample(
            z=z,
            ns_channel=tuple(ns_values),
            r_channel=tuple(r_values),
            signed_difference_ppm=tuple(signed_difference_ppm),
            p1_r=self.p1_r,
            p2_r=self.p2_r,
            p3_ns=self.p3_ns,
            p4_ns=self.p4_ns,
            block_order=self.block_order,
            quadrature_order=self.quadrature_order,
            p_max=self.p_max,
            central_charge_shift=self.central_charge_shift,
            symmetric_regulator=self.symmetric_regulator,
        )


def _linspace(start: float, stop: float, count: int) -> tuple[float, ...]:
    if count < 2:
        raise ValueError("a plot grid needs at least two points")
    step = (stop - start) / (count - 1)
    return tuple(start + index * step for index in range(count))


def _polyline(
    x_values: Sequence[float],
    y_values: Sequence[float],
    x_map,
    y_map,
) -> str:
    return " ".join(
        f"{x_map(x_value):.3f},{y_map(y_value):.3f}"
        for x_value, y_value in zip(x_values, y_values)
    )


def render_svg(sample: CrossingSample, output: Path) -> None:
    """Write a two-panel SVG with the correlators and their signed mismatch."""

    width, height = 1120, 720
    left, right = 95.0, 1060.0
    top, upper_bottom = 105.0, 465.0
    lower_top, lower_bottom = 525.0, 650.0
    x_low, x_high = min(sample.z), max(sample.z)
    y_low = min(min(sample.ns_channel), min(sample.r_channel))
    y_high = max(max(sample.ns_channel), max(sample.r_channel))
    y_pad = 0.04 * (y_high - y_low)
    y_low -= y_pad
    y_high += y_pad
    d_bound = 1.08 * max(abs(value) for value in sample.signed_difference_ppm)

    x_map = lambda value: left + (right - left) * (value - x_low) / (x_high - x_low)
    y_map = lambda value: top + (upper_bottom - top) * (y_high - value) / (y_high - y_low)
    d_map = lambda value: lower_top + (lower_bottom - lower_top) * (d_bound - value) / (2.0 * d_bound)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#1d1d1f}'
        '.grid{stroke:#dedede;stroke-width:1}.axis{stroke:#333;stroke-width:1.2}'
        '.ns{fill:none;stroke:#171717;stroke-width:3}'
        '.rr{fill:none;stroke:#3159bd;stroke-width:2.2;stroke-dasharray:8 5}'
        '.diff{fill:none;stroke:#9b3b73;stroke-width:2}'
        '.tick{font-size:13px}.label{font-size:15px}.legend{font-size:14px}</style>',
        '<text x="560" y="30" text-anchor="middle" font-size="20">'
        'NSNSRR sphere correlator in two crossing channels</text>',
        '<text x="560" y="56" text-anchor="middle" font-size="14">'
        f'P₁ᴿ={sample.p1_r:g}, P₂ᴿ={sample.p2_r:g}, '
        f'P₃ᴺˢ={sample.p3_ns:g}, P₄ᴺˢ={sample.p4_ns:g}; '
        f'recursion order {sample.block_order}, Nₚ={sample.quadrature_order}, '
        f'P≤{sample.p_max:g}; '
        + (
            f'symmetric ε=±{sample.central_charge_shift:g}</text>'
            if sample.symmetric_regulator
            else f'ε={sample.central_charge_shift:g}</text>'
        ),
    ]

    x_ticks = (0.1, 0.3, 0.5, 0.7, 0.9)
    y_ticks = tuple(
        y_low + index * (y_high - y_low) / 4.0 for index in range(5)
    )
    for value in x_ticks:
        x = x_map(value)
        parts.append(
            f'<line x1="{x:.3f}" y1="{top}" x2="{x:.3f}" '
            f'y2="{lower_bottom}" class="grid"/>'
        )
        parts.append(
            f'<text x="{x:.3f}" y="{lower_bottom + 24}" '
            f'text-anchor="middle" class="tick">{value:.1f}</text>'
        )
    for value in y_ticks:
        y = y_map(value)
        parts.append(
            f'<line x1="{left}" y1="{y:.3f}" x2="{right}" '
            f'y2="{y:.3f}" class="grid"/>'
        )
        parts.append(
            f'<text x="{left - 12}" y="{y + 4:.3f}" text-anchor="end" '
            f'class="tick">{value:.3f}</text>'
        )
    for value in (-d_bound, 0.0, d_bound):
        y = d_map(value)
        parts.append(
            f'<line x1="{left}" y1="{y:.3f}" x2="{right}" '
            f'y2="{y:.3f}" class="grid"/>'
        )
        parts.append(
            f'<text x="{left - 12}" y="{y + 4:.3f}" text-anchor="end" '
            f'class="tick">{value:.2f}</text>'
        )

    parts.extend(
        [
            f'<polyline points="{_polyline(sample.z, sample.ns_channel, x_map, y_map)}" class="ns"/>',
            f'<polyline points="{_polyline(sample.z, sample.r_channel, x_map, y_map)}" class="rr"/>',
            f'<polyline points="{_polyline(sample.z, sample.signed_difference_ppm, x_map, d_map)}" class="diff"/>',
            f'<line x1="{left}" y1="{upper_bottom}" x2="{right}" y2="{upper_bottom}" class="axis"/>',
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{upper_bottom}" class="axis"/>',
            f'<line x1="{left}" y1="{lower_bottom}" x2="{right}" y2="{lower_bottom}" class="axis"/>',
            f'<line x1="{left}" y1="{lower_top}" x2="{left}" y2="{lower_bottom}" class="axis"/>',
            '<line x1="740" y1="81" x2="780" y2="81" class="ns"/>',
            '<text x="790" y="86" class="legend">NS exchange at z</text>',
            '<line x1="895" y1="81" x2="935" y2="81" class="rr"/>',
            '<text x="945" y="86" class="legend">R exchange at 1−z</text>',
            f'<text x="28" y="{(top + upper_bottom) / 2:.1f}" '
            'transform="rotate(-90 28 285)" text-anchor="middle" class="label">𝒢(z)</text>',
            f'<text x="28" y="{(lower_top + lower_bottom) / 2:.1f}" '
            'transform="rotate(-90 28 587.5)" text-anchor="middle" class="label">'
            '10⁶(𝒢ᴿ−𝒢ᴺˢ)/𝒢ᴺˢ</text>',
            f'<text x="{(left + right) / 2:.1f}" y="700" '
            'text-anchor="middle" class="label">z</text>',
            '</svg>',
        ]
    )
    output.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=DATA_ROOT / "rrnsns_crossing.svg"
    )
    parser.add_argument(
        "--data", type=Path, default=DATA_ROOT / "rrnsns_crossing.json"
    )
    parser.add_argument("--z-min", type=float, default=0.05)
    parser.add_argument("--z-max", type=float, default=0.95)
    parser.add_argument("--points", type=int, default=37)
    parser.add_argument("--block-order", type=int, default=8)
    parser.add_argument("--quadrature-order", type=int, default=24)
    parser.add_argument("--central-charge-shift", type=float, default=3.0e-5)
    parser.add_argument("--symmetric-regulator", action="store_true")
    args = parser.parse_args()

    benchmark = RRNSNSCrossingBenchmark(
        block_order=args.block_order,
        quadrature_order=args.quadrature_order,
        central_charge_shift=args.central_charge_shift,
        symmetric_regulator=args.symmetric_regulator,
    )
    sample = benchmark.sample(_linspace(args.z_min, args.z_max, args.points))
    render_svg(sample, args.output)
    args.data.write_text(
        json.dumps(asdict(sample), indent=2) + "\n",
        encoding="utf-8",
    )
    max_error = max(abs(value) for value in sample.signed_difference_ppm)
    print(f"wrote {args.output} and {args.data}")
    print(f"maximum signed crossing mismatch: {max_error:.6g} ppm")


if __name__ == "__main__":
    main()
