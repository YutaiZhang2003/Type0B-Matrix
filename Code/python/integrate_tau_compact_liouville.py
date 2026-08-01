#!/usr/bin/env python3
"""Direct genus-one vacuum integrals over the tau fundamental domain.

The ``string_note_*`` entry points use the physical Liouville momentum and the
ordinary zero-mode volume ``V_phi`` of the string note.  The older entry points
are retained for BRY/Xi-normalized plumbing comparisons.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from xml.sax.saxutils import escape

import numpy as np

if sys.version_info < (3, 10):
    raise RuntimeError("Use Python 3.10+ for this script, e.g. .venv/bin/python")

THIS_FILE = Path(__file__).resolve()
ROOT = THIS_FILE.parents[1]
sys.path.insert(0, str(THIS_FILE.parent))

from liouville_partition import continuum_measure_factor  # noqa: E402


@dataclass(frozen=True)
class CutoffIntegral:
    tau2_max: float
    value: float
    zero_mode_tail: float
    tail_corrected_value: float


def zero_mode_tail_estimate(tau2_max: float, *, liouville_measure: str) -> float:
    """Large-cusp tail in the BRY/Xi ``q**P**2`` momentum convention."""

    return float(continuum_measure_factor(liouville_measure) / (4.0 * math.sqrt(float(tau2_max))))


def compact_theta_poisson(tau: complex, radius: float, cutoff: int) -> float:
    r"""Return sum exp(-pi R^2/tau2 |m tau - n|^2), using Poisson in n."""

    tau = complex(tau)
    tau2 = tau.imag
    if tau2 <= 0.0:
        raise ValueError("tau must be in the upper half-plane")
    radius = float(radius)
    values = np.arange(-int(cutoff), int(cutoff) + 1, dtype=float)
    m, k = np.meshgrid(values, values, indexing="ij")
    exponent = -math.pi * tau2 * ((radius * radius) * m * m + (k * k) / (radius * radius))
    phase = 2.0 * math.pi * k * m * tau.real
    return float((math.sqrt(tau2) / radius) * np.sum(np.exp(exponent) * np.cos(phase)))


def string_note_compact_theta_poisson(
    tau: complex,
    radius: float,
    cutoff: int,
    *,
    alpha_prime: float = 1.0,
) -> float:
    r"""Return the string-note Lagrangian lattice sum ``Theta_R(tau)``.

    The returned object is

    ``sum_{m,n} exp[-pi R^2 |m tau-n|^2/(alpha' tau2)]``.

    It is evaluated after Poisson resummation in ``n`` for numerical stability.
    Here ``R`` is the physical radius in ``X ~ X+2*pi*R``.
    """

    tau = complex(tau)
    tau2 = tau.imag
    radius = float(radius)
    alpha_prime = float(alpha_prime)
    if tau2 <= 0.0:
        raise ValueError("tau must be in the upper half-plane")
    if not math.isfinite(radius) or radius <= 0.0:
        raise ValueError("radius must be positive and finite")
    if not math.isfinite(alpha_prime) or alpha_prime <= 0.0:
        raise ValueError("alpha_prime must be positive and finite")
    values = np.arange(-int(cutoff), int(cutoff) + 1, dtype=float)
    m, k = np.meshgrid(values, values, indexing="ij")
    exponent = -math.pi * tau2 * (
        (radius * radius / alpha_prime) * m * m
        + (alpha_prime / (radius * radius)) * k * k
    )
    phase = 2.0 * math.pi * k * m * tau.real
    prefactor = math.sqrt(alpha_prime * tau2) / radius
    return float(prefactor * np.sum(np.exp(exponent) * np.cos(phase)))


def string_note_genus1_integrand_per_liouville_volume(
    tau: complex,
    *,
    radius: float,
    theta_cutoff: int,
    alpha_prime: float = 1.0,
) -> float:
    r"""Return the complete string-note genus-one density per ``V_phi``.

    In the convention ``z ~ z+2*pi ~ z+2*pi*tau`` and
    ``X ~ X+2*pi*R``, equations (4.90)--(4.94) of the string note give

    .. math::

       \frac{1}{V_\phi}\frac{dA_1}{d^2\tau}
       =\frac{R}{4\pi\alpha'\tau_2^2}
        \sum_{m,n\in\mathbb Z}
        e^{-\pi R^2|m\tau-n|^2/(\alpha'\tau_2)}.

    The coefficient includes both the torus ``1/2`` and the translation-CKV
    ``1/tau2``.  It does not replace ``V_phi`` by a logarithm of ``mu``.
    """

    tau = complex(tau)
    radius = float(radius)
    alpha_prime = float(alpha_prime)
    theta = string_note_compact_theta_poisson(
        tau,
        radius,
        theta_cutoff,
        alpha_prime=alpha_prime,
    )
    return float(radius * theta / (4.0 * math.pi * alpha_prime * tau.imag**2))


def string_note_zero_mode_tail_estimate(
    tau2_max: float,
    *,
    alpha_prime: float = 1.0,
) -> float:
    """Large-cusp tail of the string-note density per ``V_phi``."""

    tau2_max = float(tau2_max)
    alpha_prime = float(alpha_prime)
    if tau2_max <= 0.0:
        raise ValueError("tau2_max must be positive")
    if not math.isfinite(alpha_prime) or alpha_prime <= 0.0:
        raise ValueError("alpha_prime must be positive and finite")
    return 1.0 / (2.0 * math.pi * math.sqrt(alpha_prime * tau2_max))


def integrand_half_eta4_compact_liouville(
    tau: complex,
    *,
    radius: float,
    theta_cutoff: int,
    liouville_measure: str,
) -> float:
    r"""Compute the BRY/Xi-normalized genus-one density in tau coordinates.

    The factor ``1/2`` divides the residual reflection automorphism of the
    torus, while ``1/tau2`` divides its translation CKV volume.  The eta
    factors then cancel analytically:

        (1/2) |eta|^4 Z_compact Z_L / tau2
        = c_P * R * Theta_R(tau) / (8 tau2^2),

    where ``c_P`` is ``1/pi`` for the BRY/Xi measure ``dP_bry/pi``.  It is
    not the Liouville cosmological constant.
    """

    tau = complex(tau)
    theta = compact_theta_poisson(tau, radius=radius, cutoff=theta_cutoff)
    return float(
        continuum_measure_factor(liouville_measure)
        * float(radius)
        * theta
        / (8.0 * tau.imag * tau.imag)
    )


def integrate_string_note_fundamental_domain_cutoff(
    tau2_max: float,
    *,
    radius: float,
    theta_cutoff: int,
    alpha_prime: float,
    x_order: int,
    y_order: int,
) -> float:
    """Integrate the string-note density over a cutoff fundamental domain."""

    tau2_max = float(tau2_max)
    if tau2_max <= math.sqrt(3.0) / 2.0:
        return 0.0
    x_nodes, x_weights = np.polynomial.legendre.leggauss(int(x_order))
    y_nodes, y_weights = np.polynomial.legendre.leggauss(int(y_order))
    total = 0.0
    for x_node, x_weight in zip(x_nodes, x_weights):
        x = 0.5 * float(x_node)
        wx = 0.5 * float(x_weight)
        y_lower = math.sqrt(max(0.0, 1.0 - x * x))
        if tau2_max <= y_lower:
            continue
        midpoint = 0.5 * (tau2_max + y_lower)
        half_width = 0.5 * (tau2_max - y_lower)
        inner = 0.0
        for y_node, y_weight in zip(y_nodes, y_weights):
            y = midpoint + half_width * float(y_node)
            inner += float(y_weight) * string_note_genus1_integrand_per_liouville_volume(
                complex(x, y),
                radius=radius,
                theta_cutoff=theta_cutoff,
                alpha_prime=alpha_prime,
            )
        total += wx * half_width * inner
    return float(total)


def integrate_fundamental_domain_cutoff(
    tau2_max: float,
    *,
    radius: float,
    theta_cutoff: int,
    liouville_measure: str,
    x_order: int,
    y_order: int,
) -> float:
    """Integrate over |tau1| <= 1/2, |tau| >= 1, tau2 <= tau2_max."""

    tau2_max = float(tau2_max)
    if tau2_max <= math.sqrt(3.0) / 2.0:
        return 0.0
    x_nodes, x_weights = np.polynomial.legendre.leggauss(int(x_order))
    y_nodes, y_weights = np.polynomial.legendre.leggauss(int(y_order))
    total = 0.0
    for x_node, x_weight in zip(x_nodes, x_weights):
        x = 0.5 * float(x_node)
        wx = 0.5 * float(x_weight)
        y_lower = math.sqrt(max(0.0, 1.0 - x * x))
        if tau2_max <= y_lower:
            continue
        midpoint = 0.5 * (tau2_max + y_lower)
        half_width = 0.5 * (tau2_max - y_lower)
        inner = 0.0
        for y_node, y_weight in zip(y_nodes, y_weights):
            y = midpoint + half_width * float(y_node)
            inner += float(y_weight) * integrand_half_eta4_compact_liouville(
                complex(x, y),
                radius=radius,
                theta_cutoff=theta_cutoff,
                liouville_measure=liouville_measure,
            )
        total += wx * half_width * inner
    return float(total)


def parse_cutoffs(value: str) -> list[float]:
    cutoffs = [float(piece.strip()) for piece in value.split(",") if piece.strip()]
    if not cutoffs:
        raise ValueError("Need at least one cutoff")
    return sorted(set(cutoffs))


def write_csv(path: Path, rows: list[CutoffIntegral]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "tau2_max",
                "cutoff_integral",
                "zero_mode_tail_estimate",
                "tail_corrected_integral",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    f"{row.tau2_max:.16g}",
                    f"{row.value:.16g}",
                    f"{row.zero_mode_tail:.16g}",
                    f"{row.tail_corrected_value:.16g}",
                ]
            )


def _fmt(value: float) -> str:
    if value == 0.0:
        return "0"
    if abs(value) >= 1.0e4 or abs(value) < 1.0e-3:
        return f"{value:.2e}"
    return f"{value:.4g}"


def _points(xs, ys) -> str:
    return " ".join(f"{x:.3f},{y:.3f}" for x, y in zip(xs, ys))


def write_svg(path: Path, rows: list[CutoffIntegral], *, title: str, subtitle: str) -> None:
    width, height = 1000, 640
    left, right = 104, 46
    top, plot_h = 88, 420
    bottom = top + plot_h
    plot_w = width - left - right
    xs = [row.tau2_max for row in rows]
    ys = [row.value for row in rows]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = 0.0, max(ys) * 1.12

    def sx(value: float) -> float:
        return left + (value - x_min) / (x_max - x_min) * plot_w if x_max != x_min else left

    def sy(value: float) -> float:
        return bottom - (value - y_min) / (y_max - y_min) * plot_h if y_max != y_min else bottom

    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}">',
        "<style>",
        "text { font-family: Helvetica, Arial, sans-serif; fill: #202124; }",
        ".subtitle { font-size: 18px; fill: #4f5661; } .tick { font-size: 14px; fill: #4f5661; }",
        ".grid { stroke: #d7dce2; stroke-width: 1; } .axis { stroke: #202124; stroke-width: 1.5; }",
        ".curve { fill: none; stroke: #176f7a; stroke-width: 3; stroke-linecap: round; stroke-linejoin: round; }",
        ".dot { fill: #c2402f; }",
        "</style>",
        '<rect x="0" y="0" width="1000" height="640" fill="#ffffff"/>',
        f'<text x="{left}" y="40" font-size="25" font-weight="700">{escape(title)}</text>',
        f'<text x="{left}" y="66" class="subtitle">{escape(subtitle)}</text>',
    ]
    for tick in np.linspace(y_min, y_max, 6):
        y = sy(float(tick))
        elements.append(f'<line x1="{left}" y1="{y:.3f}" x2="{width - right}" y2="{y:.3f}" class="grid"/>')
        elements.append(f'<text x="{left - 10}" y="{y + 4:.3f}" class="tick" text-anchor="end">{_fmt(float(tick))}</text>')
    for tick in np.linspace(x_min, x_max, min(7, len(xs))):
        x = sx(float(tick))
        elements.append(f'<line x1="{x:.3f}" y1="{top}" x2="{x:.3f}" y2="{bottom}" class="grid"/>')
        elements.append(f'<text x="{x:.3f}" y="{bottom + 25}" class="tick" text-anchor="middle">{_fmt(float(tick))}</text>')
    elements.append(f'<line x1="{left}" y1="{bottom}" x2="{width - right}" y2="{bottom}" class="axis"/>')
    elements.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" class="axis"/>')
    elements.append(f'<polyline points="{_points((sx(x) for x in xs), (sy(y) for y in ys))}" class="curve"/>')
    for x, y in zip(xs, ys):
        elements.append(f'<circle cx="{sx(x):.3f}" cy="{sy(y):.3f}" r="4" class="dot"/>')
    elements.append(f'<text x="{left + plot_w / 2:.3f}" y="{height - 36}" font-size="18" text-anchor="middle">tau2 cutoff</text>')
    elements.append(f'<text x="30" y="{top + plot_h / 2:.3f}" font-size="18" text-anchor="middle" transform="rotate(-90 30 {top + plot_h / 2:.3f})">direct tau integral</text>')
    elements.append(f'<text x="{left}" y="{height - 76}" class="subtitle">last tail-corrected value: {_fmt(rows[-1].tail_corrected_value)}</text>')
    elements.append("</svg>")
    path.write_text("\n".join(elements))


def run(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Integrate the genus-one compact-boson/Liouville density."
    )
    parser.add_argument(
        "--tau2-cutoffs",
        default="2,3,4.873733715435174,5,10,20",
        help="Comma-separated tau2 upper cutoffs.",
    )
    parser.add_argument("--radius", type=float, default=1.0)
    parser.add_argument("--theta-cutoff", type=int, default=12)
    parser.add_argument(
        "--convention",
        choices=["string-note", "bry"],
        default="string-note",
        help="string-note returns A1/V_phi; bry retains the older q**P**2 density",
    )
    parser.add_argument("--alpha-prime", type=float, default=1.0)
    parser.add_argument("--liouville-measure", choices=["dP/pi", "2dP/pi", "dP"], default="dP/pi")
    parser.add_argument("--x-order", type=int, default=96)
    parser.add_argument("--y-order", type=int, default=96)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/tau_moduli_integral"))
    parser.add_argument("--prefix")
    args = parser.parse_args(argv)

    rows: list[CutoffIntegral] = []
    for tau2_max in parse_cutoffs(args.tau2_cutoffs):
        if args.convention == "string-note":
            value = integrate_string_note_fundamental_domain_cutoff(
                tau2_max,
                radius=args.radius,
                theta_cutoff=args.theta_cutoff,
                alpha_prime=args.alpha_prime,
                x_order=args.x_order,
                y_order=args.y_order,
            )
            tail = string_note_zero_mode_tail_estimate(
                tau2_max,
                alpha_prime=args.alpha_prime,
            )
        else:
            value = integrate_fundamental_domain_cutoff(
                tau2_max,
                radius=args.radius,
                theta_cutoff=args.theta_cutoff,
                liouville_measure=args.liouville_measure,
                x_order=args.x_order,
                y_order=args.y_order,
            )
            tail = zero_mode_tail_estimate(tau2_max, liouville_measure=args.liouville_measure)
        rows.append(
            CutoffIntegral(
                tau2_max=float(tau2_max),
                value=value,
                zero_mode_tail=tail,
                tail_corrected_value=value + tail,
            )
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.prefix or (
        f"tau_direct_{args.convention}_R{args.radius:g}_x{args.x_order}_y{args.y_order}"
    )
    csv_path = args.out_dir / f"{prefix}.csv"
    svg_path = args.out_dir / f"{prefix}.svg"
    write_csv(csv_path, rows)
    write_svg(
        svg_path,
        rows,
        title="Direct tau Integral",
        subtitle=(
            f"convention={args.convention}, R={args.radius:g}, "
            f"alpha'={args.alpha_prime:g}, measure={args.liouville_measure}"
        ),
    )

    print("direct tau moduli integral")
    print(f"  csv: {csv_path}")
    print(f"  svg: {svg_path}")
    print(f"  radius={args.radius:g}")
    print(f"  convention={args.convention}")
    print(f"  x_order={args.x_order}, y_order={args.y_order}, theta_cutoff={args.theta_cutoff}")
    for row in rows:
        print(
            f"  tau2_max={row.tau2_max:.12g} "
            f"cutoff_integral={row.value:.12e} "
            f"zero_mode_tail={row.zero_mode_tail:.12e} "
            f"tail_corrected={row.tail_corrected_value:.12e}"
        )


if __name__ == "__main__":
    run()
