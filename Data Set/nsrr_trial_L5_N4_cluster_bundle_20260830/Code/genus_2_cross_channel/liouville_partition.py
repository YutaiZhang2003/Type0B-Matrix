#!/usr/bin/env python3
"""Volume-normalized genus-one partition functions for c=25 Liouville.

Two momentum conventions occur in this project and must not be conflated:

* the BRY/Xi CFT convention uses ``q**(P_bry**2)`` and ``dP_bry/pi``;
* the string-note target-space convention uses a physical momentum
  ``P_note`` with ``exp(-pi*alpha_prime*tau2*P_note**2) dP_note/pi``.

They are related by ``P_note = 2*P_bry/sqrt(alpha_prime)``.  Consequently the
string-note torus density per ordinary Liouville zero-mode volume is
``2/sqrt(alpha_prime)`` times the BRY/Xi density.
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from xml.sax.saxutils import escape

import numpy as np

try:
    from virasoro_blocks import dedekind_eta
except ImportError:  # pragma: no cover - supports package-style execution
    from plumbing.virasoro_blocks import dedekind_eta


def q_grid(q_min: float, q_max: float, points: int, spacing: str) -> list[float]:
    if points < 2:
        raise ValueError("points must be at least 2")
    if not 0.0 < q_min < q_max < 1.0:
        raise ValueError("expected 0 < q_min < q_max < 1")
    if spacing == "linear":
        return [float(value) for value in np.linspace(q_min, q_max, points)]
    if spacing == "geometric":
        return [float(value) for value in np.geomspace(q_min, q_max, points)]
    raise ValueError("spacing must be linear or geometric")


def tau_imag_from_real_q(q: float) -> float:
    if not 0.0 < q < 1.0:
        raise ValueError("real plumbing q must satisfy 0 < q < 1")
    return -math.log(q) / (2.0 * math.pi)


def tau_imag_from_q(q: complex) -> float:
    q = complex(q)
    if not 0.0 < abs(q) < 1.0:
        raise ValueError("plumbing q must satisfy 0 < |q| < 1")
    return -math.log(abs(q)) / (2.0 * math.pi)


def continuum_measure_factor(measure: str) -> float:
    if measure == "dP/pi":
        return 1.0 / math.pi
    if measure == "2dP/pi":
        return 2.0 / math.pi
    if measure == "dP":
        return 1.0
    raise ValueError("measure must be dP/pi, 2dP/pi, or dP")


def string_note_momentum_from_bry(momentum: float, *, alpha_prime: float = 1.0) -> float:
    r"""Convert BRY/Xi momentum to the physical momentum in the string note.

    The BRY/Xi character contributes ``exp(-4*pi*tau2*P_bry**2)``.  The
    string note instead writes ``exp(-pi*alpha_prime*tau2*P_note**2)``.
    """

    alpha_prime = float(alpha_prime)
    if not math.isfinite(alpha_prime) or alpha_prime <= 0.0:
        raise ValueError("alpha_prime must be positive and finite")
    return 2.0 * float(momentum) / math.sqrt(alpha_prime)


def string_note_liouville_c25_partition_density_from_tau(
    tau: complex,
    *,
    alpha_prime: float = 1.0,
) -> float:
    r"""Return ``Z_L/V_phi`` in the momentum convention of the string note.

    With ``P_note >= 0`` and completeness measure ``dP_note/pi``, this is

    .. math::

       \int_0^\infty \frac{dP_{\rm note}}{\pi}
       \frac{e^{-\pi\alpha'\tau_2P_{\rm note}^2}}{|\eta(\tau)|^2}
       =\frac{1}{2\pi\sqrt{\alpha'\tau_2}|\eta(\tau)|^2}.

    ``V_phi`` is the ordinary coordinate length of the canonically normalized
    Liouville zero mode.  No ``log(mu)`` identification is made here.
    """

    tau = complex(tau)
    alpha_prime = float(alpha_prime)
    if tau.imag <= 0.0:
        raise ValueError("tau must have positive imaginary part")
    if not math.isfinite(alpha_prime) or alpha_prime <= 0.0:
        raise ValueError("alpha_prime must be positive and finite")
    q = np.exp(2.0j * np.pi * tau)
    eta_abs_squared = abs(dedekind_eta(q)) ** 2
    return 1.0 / (
        2.0 * math.pi * math.sqrt(alpha_prime * tau.imag) * eta_abs_squared
    )


def liouville_c25_partition_density(q: float, *, measure: str = "dP/pi") -> float:
    """Return int_0^infty measure |q^(P^2)/eta(q)|^2 for real positive q.

    This is the BRY/Xi non-compact, volume-normalized continuum character
    trace.  The absolute Liouville torus partition function carries a
    conventional infinite zero-mode volume factor; this routine omits that
    overall volume.  Use ``string_note_liouville_c25_partition_density_from_tau``
    for the physical-momentum convention of the string note.
    """
    tau_imag = tau_imag_from_real_q(q)
    eta_abs_squared = abs(dedekind_eta(q)) ** 2
    gaussian_integral = 1.0 / (4.0 * math.sqrt(tau_imag))
    return continuum_measure_factor(measure) * gaussian_integral / eta_abs_squared


def liouville_c25_partition_density_from_q(q: complex, *, measure: str = "dP/pi") -> float:
    """Return the volume-normalized c=25 Liouville partition density for complex q."""
    q = complex(q)
    tau_imag = tau_imag_from_q(q)
    eta_abs_squared = abs(dedekind_eta(q)) ** 2
    gaussian_integral = 1.0 / (4.0 * math.sqrt(tau_imag))
    return continuum_measure_factor(measure) * gaussian_integral / eta_abs_squared


def liouville_c25_partition_density_from_tau(tau: complex, *, measure: str = "dP/pi") -> float:
    tau = complex(tau)
    if tau.imag <= 0:
        raise ValueError("tau must have positive imaginary part")
    q = np.exp(2.0j * np.pi * tau)
    return liouville_c25_partition_density_from_q(q, measure=measure)


def estimate_p_max_for_partition(q: float, tail_tolerance: float = 1.0e-14, safety_margin: float = 0.5) -> float:
    tau_imag = tau_imag_from_real_q(q)
    return math.sqrt(max(1.0, -math.log(tail_tolerance)) / (4.0 * math.pi * tau_imag)) + safety_margin


def liouville_c25_partition_density_numeric(
    q: float,
    *,
    measure: str = "dP/pi",
    p_max: float | None = None,
    quadrature_order: int = 64,
) -> float:
    """Numerical P-integral version, useful as a check of the analytic Gaussian."""
    if quadrature_order <= 0:
        raise ValueError("quadrature_order must be positive")
    if p_max is None:
        p_max = estimate_p_max_for_partition(q)
    tau_imag = tau_imag_from_real_q(q)
    eta_abs_squared = abs(dedekind_eta(q)) ** 2
    nodes, weights = np.polynomial.legendre.leggauss(quadrature_order)
    midpoint = 0.5 * p_max
    total = 0.0
    for node, weight in zip(nodes, weights):
        p = midpoint * (float(node) + 1.0)
        total += float(weight) * math.exp(-4.0 * math.pi * tau_imag * p * p)
    return continuum_measure_factor(measure) * midpoint * total / eta_abs_squared


@dataclass(frozen=True)
class PartitionSample:
    q: float
    tau_imag: float
    eta_abs_squared: float
    value: float


def scan_partition(q_values: Iterable[float], *, measure: str = "dP/pi") -> list[PartitionSample]:
    samples: list[PartitionSample] = []
    for q in q_values:
        samples.append(
            PartitionSample(
                q=q,
                tau_imag=tau_imag_from_real_q(q),
                eta_abs_squared=abs(dedekind_eta(q)) ** 2,
                value=liouville_c25_partition_density(q, measure=measure),
            )
        )
    return samples


def write_csv(path: Path, samples: list[PartitionSample]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["q", "tau_imag", "eta_abs_squared", "partition_density"])
        for sample in samples:
            writer.writerow(
                [
                    f"{sample.q:.16g}",
                    f"{sample.tau_imag:.16g}",
                    f"{sample.eta_abs_squared:.16g}",
                    f"{sample.value:.16g}",
                ]
            )


def _nice_ticks(vmin: float, vmax: float, count: int = 5) -> list[float]:
    if vmin == vmax:
        return [vmin]
    raw_step = (vmax - vmin) / max(1, count - 1)
    magnitude = 10 ** math.floor(math.log10(abs(raw_step)))
    normalized = raw_step / magnitude
    if normalized <= 1.5:
        nice = 1.0
    elif normalized <= 3.5:
        nice = 2.0
    elif normalized <= 7.5:
        nice = 5.0
    else:
        nice = 10.0
    step = nice * magnitude
    start = math.ceil(vmin / step) * step
    ticks = []
    value = start
    while value <= vmax + 0.5 * step:
        ticks.append(value)
        value += step
    return ticks


def write_svg(path: Path, samples: list[PartitionSample], *, title: str, subtitle: str) -> None:
    x_values = [sample.q for sample in samples]
    y_values = [sample.value for sample in samples]
    x_min, x_max = min(x_values), max(x_values)
    y_min, y_max = min(y_values), max(y_values)
    padding = max(1.0e-12, 0.08 * (y_max - y_min))
    y_min -= padding
    y_max += padding

    width, height = 960, 620
    left, right, top, bottom = 108, 40, 76, 82
    plot_width = width - left - right
    plot_height = height - top - bottom

    def sx(x: float) -> float:
        return left + (x - x_min) / (x_max - x_min) * plot_width

    def sy(y: float) -> float:
        return top + (y_max - y) / (y_max - y_min) * plot_height

    points = " ".join(f"{sx(x):.3f},{sy(y):.3f}" for x, y in zip(x_values, y_values))
    x_ticks = _nice_ticks(x_min, x_max)
    y_ticks = _nice_ticks(y_min, y_max)

    elements: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}">',
        "<style>",
        "text { font-family: Helvetica, Arial, sans-serif; fill: #202124; }",
        ".small { font-size: 18px; }",
        ".tick { font-size: 14px; fill: #4f5661; }",
        ".grid { stroke: #d7dce2; stroke-width: 1; }",
        ".axis { stroke: #202124; stroke-width: 1.5; }",
        ".curve { fill: none; stroke: #176f7a; stroke-width: 3.2; stroke-linejoin: round; stroke-linecap: round; }",
        ".dot { fill: #c2402f; }",
        "</style>",
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"/>',
        f'<text x="{left}" y="38" font-size="26" font-weight="700">{escape(title)}</text>',
        f'<text x="{left}" y="64" class="small">{escape(subtitle)}</text>',
    ]
    for tick in y_ticks:
        y = sy(tick)
        elements.append(f'<line x1="{left}" y1="{y:.3f}" x2="{width - right}" y2="{y:.3f}" class="grid"/>')
        elements.append(f'<text x="{left - 12}" y="{y + 5:.3f}" class="tick" text-anchor="end">{tick:.3g}</text>')
    for tick in x_ticks:
        x = sx(tick)
        elements.append(f'<line x1="{x:.3f}" y1="{top}" x2="{x:.3f}" y2="{height - bottom}" class="grid"/>')
        elements.append(f'<text x="{x:.3f}" y="{height - bottom + 28}" class="tick" text-anchor="middle">{tick:.3g}</text>')

    elements.extend(
        [
            f'<line x1="{left}" y1="{height - bottom}" x2="{width - right}" y2="{height - bottom}" class="axis"/>',
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}" class="axis"/>',
            f'<polyline points="{points}" class="curve"/>',
        ]
    )
    for x, y in zip(x_values, y_values):
        elements.append(f'<circle cx="{sx(x):.3f}" cy="{sy(y):.3f}" r="3.3" class="dot"/>')

    elements.extend(
        [
            f'<text x="{left + plot_width / 2:.3f}" y="{height - 24}" font-size="18" text-anchor="middle">plumbing parameter q</text>',
            (
                f'<text x="28" y="{top + plot_height / 2:.3f}" font-size="18" '
                f'text-anchor="middle" transform="rotate(-90 28 {top + plot_height / 2:.3f})">partition density</text>'
            ),
            "</svg>",
        ]
    )
    path.write_text("\n".join(elements))


def safe_token(value: object) -> str:
    return (
        str(value)
        .replace("+", "p")
        .replace("-", "m")
        .replace(".", "p")
        .replace("/", "_over_")
        .replace(" ", "")
    )


def run(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Plot the c=25 Liouville genus-one partition density.")
    parser.add_argument("--q-min", type=float, default=0.005)
    parser.add_argument("--q-max", type=float, default=0.3)
    parser.add_argument("--points", type=int, default=40)
    parser.add_argument("--spacing", choices=["linear", "geometric"], default="linear")
    parser.add_argument("--measure", choices=["dP/pi", "2dP/pi", "dP"], default="dP/pi")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/liouville_partition"))
    parser.add_argument("--prefix")
    parser.add_argument("--check-numeric", action="store_true")
    args = parser.parse_args(argv)

    q_values = q_grid(args.q_min, args.q_max, args.points, args.spacing)
    samples = scan_partition(q_values, measure=args.measure)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.prefix or f"c25_liouville_partition_{safe_token(args.measure)}_q{safe_token(args.q_min)}-{safe_token(args.q_max)}"
    csv_path = args.out_dir / f"{prefix}.csv"
    svg_path = args.out_dir / f"{prefix}.svg"
    write_csv(csv_path, samples)
    write_svg(
        svg_path,
        samples,
        title="c=25 Liouville genus-one partition density",
        subtitle=f"real q, continuum measure {args.measure}, zero-mode volume omitted",
    )

    print("c=25 Liouville genus-one partition density")
    print(f"  svg: {svg_path}")
    print(f"  csv: {csv_path}")
    print(f"  q range=[{args.q_min:g}, {args.q_max:g}], points={args.points}, spacing={args.spacing}")
    print(f"  measure={args.measure}")
    print(f"  first value={samples[0].value:.12e}")
    print(f"  last value={samples[-1].value:.12e}")

    if args.check_numeric:
        midpoint = samples[len(samples) // 2].q
        analytic = liouville_c25_partition_density(midpoint, measure=args.measure)
        numeric = liouville_c25_partition_density_numeric(midpoint, measure=args.measure)
        relative = abs(numeric - analytic) / max(1.0, abs(analytic))
        print(f"  numeric check q={midpoint:.12g}: relative difference={relative:.6e}")


if __name__ == "__main__":
    run()
