#!/usr/bin/env python3
"""Compare the sphere-six pillow h-recursion with the fixed-weight c-recursion.

The plane frame is

    (0, z, t1, t2, 1, infinity),       0 < z < t1 < t2 < 1,

and the first mobile coordinate z is scanned while t1 and t2 are fixed.  The
reduced pillow series H6(p1,p2,p3) is converted back to the same
plane-normalized chiral block used by the independent CCY c-recursion.  Both
methods are evaluated at two total-degree truncations, so that the numerical
disagreement can be compared with an observed truncation scale.

The aligned-cover inversion uses two nested five-point maps.  If
q=q(z), P23=p2*p3, then

    t1 = T(P23,q),  t2 = T(p3,q),
    p1 = q/P23,     p2 = P23/p3.

Here T is the exact infinite product entering the five-point pillow map.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import sys
import time
from pathlib import Path

import mpmath as mp


WORKING_DIGITS = 80
mp.mp.dps = WORKING_DIGITS

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CCY_DIR = (
    ROOT
    / "Code"
    / "bosonic_c1_one_to_n_reference"
    / "reference_implementation"
    / "plumbing"
)
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(CCY_DIR))

from check_sphere_six_pillow_h_recursion_numerical_order10 import (  # noqa: E402
    proposed_coefficients,
)
from ccy_sphere_six_point import sphere_six_point_c_coefficients  # noqa: E402


CENTRAL_CHARGE = mp.mpf("26.215")
# Weights at (0,z,t1,t2,1,infinity).
EXTERNAL = tuple(map(mp.mpf, ("0.17", "0.29", "0.43", "0.58", "0.71", "0.86")))
INTERNAL = tuple(map(mp.mpf, ("0.9371", "1.0837", "1.3321")))
FIXED_T1 = mp.mpf("0.32")
FIXED_T2 = mp.mpf("0.62")
Z_MIN = mp.mpf("0.015")
Z_MAX = mp.mpf("0.20")
POINT_COUNT = 61
H_ORDER = 10
H_LOWER_ORDER = 8
C_ORDER = 20
C_LOWER_ORDER = 18


def elliptic_nome(z: mp.mpf) -> mp.mpf:
    """Return q=exp[-pi K(1-z)/K(z)] on the real pillow branch."""

    return mp.exp(-mp.pi * mp.ellipk(1 - z) / mp.ellipk(z))


def theta3_from_nome(q: mp.mpf) -> mp.mpf:
    return mp.jtheta(3, mp.mpf(0), q)


def t_from_segment_nome(p_right: mp.mpf, q: mp.mpf) -> mp.mpf:
    """Evaluate T(p_right,q)=4 p_right Y(q/p_right,p_right)."""

    p_left = q / p_right
    y = (1 + p_left) ** 2
    n = 1
    while True:
        q_odd = q ** (2 * n - 1)
        q_even = q ** (2 * n)
        y *= ((1 + q_even) / (1 + q_odd)) ** 4
        y *= (
            (1 + p_left ** (2 * n + 1) * p_right ** (2 * n))
            * (1 + p_left ** (2 * n - 1) * p_right ** (2 * n))
            / (
                (1 + p_left ** (2 * n) * p_right ** (2 * n - 1))
                * (1 + p_left ** (2 * n - 2) * p_right ** (2 * n - 1))
            )
        ) ** 2
        if max(
            abs(q_even), abs(p_left) ** (2 * n), abs(p_right) ** (2 * n)
        ) < mp.mpf("1e-75"):
            break
        n += 1
        if n > 10000:
            raise RuntimeError("aligned-cover product did not converge")
    return 4 * p_right * y


def invert_mobile_position(t: mp.mpf, q: mp.mpf) -> tuple[mp.mpf, mp.mpf]:
    """Solve T(p_right,q)=t by bisection on q<p_right<1."""

    lower = q * (1 + mp.mpf("1e-40"))
    upper = 1 - mp.mpf("1e-40")
    for _ in range(220):
        midpoint = (lower + upper) / 2
        if t_from_segment_nome(midpoint, q) < t:
            lower = midpoint
        else:
            upper = midpoint
    p_right = (lower + upper) / 2
    return p_right, abs(t_from_segment_nome(p_right, q) - t)


def segment_nomes(
    z: mp.mpf, t1: mp.mpf, t2: mp.mpf
) -> tuple[mp.mpf, mp.mpf, mp.mpf, mp.mpf, mp.mpf]:
    """Invert the six-point aligned cover on 0<z<t1<t2<1."""

    if not 0 < z < t1 < t2 < 1:
        raise ValueError("the real aligned branch requires 0 < z < t1 < t2 < 1")
    q = elliptic_nome(z)
    p23, residual1 = invert_mobile_position(t1, q)
    p3, residual2 = invert_mobile_position(t2, q)
    p1 = q / p23
    p2 = p23 / p3
    if not 0 < p1 < 1 or not 0 < p2 < 1 or not 0 < p3 < 1:
        raise AssertionError("inverse cover left the real plumbing polydisc")
    product_residual = abs(p1 * p2 * p3 - q)
    return p1, p2, p3, max(residual1, residual2), product_residual


def triangular_value(
    coefficients: dict[tuple[int, int, int], complex | mp.mpf | mp.mpc],
    x1: mp.mpf,
    x2: mp.mpf,
    x3: mp.mpf,
    order: int,
) -> mp.mpc:
    total = mp.mpc(0)
    for (n1, n2, n3), coefficient in coefficients.items():
        if n1 + n2 + n3 <= order:
            total += mp.mpc(coefficient) * x1**n1 * x2**n2 * x3**n3
    return total


def pillow_plane_prefactor(
    z: mp.mpf,
    t1: mp.mpf,
    t2: mp.mpf,
    p1: mp.mpf,
    p2: mp.mpf,
    p3: mp.mpf,
    theta3: mp.mpf,
) -> mp.mpf:
    """Return Lambda_6^(c-1) times the three propagation powers."""

    d1, d2, d3, d4, d5, d6 = EXTERNAL
    h1, h2, h3 = INTERNAL
    delta = (CENTRAL_CHARGE - 1) / 24
    theta_exponent = (
        (CENTRAL_CHARGE - 1) / 2
        - 4 * (d1 + d2 + d5 + d6)
        - 2 * (d3 + d4)
    )
    lambda_factor = (
        theta3**theta_exponent
        * z ** (delta - d1 - d2)
        * (1 - z) ** (delta - d2 - d5)
        * (t1 * (1 - t1) * (t1 - z)) ** (-d3 / 2)
        * (t2 * (1 - t2) * (t2 - z)) ** (-d4 / 2)
    )
    propagation = (
        (4 * p1) ** (h1 - delta)
        * p2 ** (h2 - delta)
        * (4 * p3) ** (h3 - delta)
    )
    return lambda_factor * propagation


def plane_primary_factor(z: mp.mpf, t1: mp.mpf, t2: mp.mpf) -> mp.mpf:
    d1, d2, d3, d4, _, _ = EXTERNAL
    h1, h2, h3 = INTERNAL
    return (
        z ** (h1 - d1 - d2)
        * t1 ** (h2 - h1 - d3)
        * t2 ** (h3 - h2 - d4)
    )


def relative_shift(value: mp.mpc, reference: mp.mpc) -> mp.mpf:
    return abs(value - reference) / max(abs(reference), mp.mpf("1e-70"))


def real_part(value: mp.mpc, label: str) -> float:
    if abs(mp.im(value)) > mp.mpf("1e-35") * max(abs(value), mp.mpf(1)):
        raise AssertionError(f"unexpected imaginary part in {label}: {mp.nstr(value, 20)}")
    return float(mp.re(value))


def write_svg_plot(rows: list[dict[str, float]], output_path: Path) -> None:
    """Write a dependency-free two-panel vector plot."""

    width, height = 1260, 900
    left, right = 120.0, 45.0
    top_y0, top_y1 = 145.0, 505.0
    bottom_y0, bottom_y1 = 605.0, 815.0
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

    def path(values: list[float], y_function) -> str:
        return " ".join(
            ("M" if index == 0 else "L")
            + f" {x_coordinate(z_value):.2f} {y_function(value):.2f}"
            for index, (z_value, value) in enumerate(zip(z_values, values))
        )

    pieces = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Helvetica,Arial,sans-serif;fill:#202124}.axis{stroke:#202124;stroke-width:1.3}.grid{stroke:#c8cdd2;stroke-width:1;opacity:.45}.tick{font-size:16px}.label{font-size:19px}.legend{font-size:16px}.title{font-size:23px;font-weight:600}.subtitle{font-size:15px}</style>',
        f'<text class="title" x="{width/2:.0f}" y="38" text-anchor="middle">Sphere six-point block: pillow h-recursion vs. c-recursion</text>',
        f'<text class="subtitle" x="{width/2:.0f}" y="68" text-anchor="middle">t1=0.32, t2=0.62, c=26.215; weights fixed and z scanned in the ordered real channel</text>',
    ]

    for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
        value = value_low + fraction * (value_high - value_low)
        y = top_coordinate(value)
        pieces.append(f'<line class="grid" x1="{x0}" y1="{y:.2f}" x2="{x1}" y2="{y:.2f}"/>')
        pieces.append(f'<text class="tick" x="{x0-12}" y="{y+5:.2f}" text-anchor="end">{value:.5f}</text>')
    for exponent in range(int(log_low), int(log_high) + 1):
        y = bottom_coordinate(10.0**exponent)
        pieces.append(f'<line class="grid" x1="{x0}" y1="{y:.2f}" x2="{x1}" y2="{y:.2f}"/>')
        pieces.append(f'<text class="tick" x="{x0-12}" y="{y+5:.2f}" text-anchor="end">10^{exponent}</text>')
    for index in range(7):
        z_value = z_low + index * (z_high - z_low) / 6
        x = x_coordinate(z_value)
        pieces.append(f'<line class="grid" x1="{x:.2f}" y1="{top_y0}" x2="{x:.2f}" y2="{top_y1}"/>')
        pieces.append(f'<line class="grid" x1="{x:.2f}" y1="{bottom_y0}" x2="{x:.2f}" y2="{bottom_y1}"/>')
        pieces.append(f'<text class="tick" x="{x:.2f}" y="{bottom_y1+27}" text-anchor="middle">{z_value:.3f}</text>')

    pieces.extend(
        [
            f'<line class="axis" x1="{x0}" y1="{top_y0}" x2="{x0}" y2="{top_y1}"/>',
            f'<line class="axis" x1="{x0}" y1="{top_y1}" x2="{x1}" y2="{top_y1}"/>',
            f'<line class="axis" x1="{x0}" y1="{bottom_y0}" x2="{x0}" y2="{bottom_y1}"/>',
            f'<line class="axis" x1="{x0}" y1="{bottom_y1}" x2="{x1}" y2="{bottom_y1}"/>',
            f'<path d="{path(c_values, top_coordinate)}" fill="none" stroke="#1f4e79" stroke-width="4"/>',
            f'<path d="{path(h_values, top_coordinate)}" fill="none" stroke="#d95f02" stroke-width="3" stroke-dasharray="10 7"/>',
            f'<path d="{path(comparison, bottom_coordinate)}" fill="none" stroke="#7b3294" stroke-width="4"/>',
            f'<path d="{path(h_shift, bottom_coordinate)}" fill="none" stroke="#d95f02" stroke-width="2.5" stroke-dasharray="3 6"/>',
            f'<path d="{path(c_shift, bottom_coordinate)}" fill="none" stroke="#1f4e79" stroke-width="2.5" stroke-dasharray="11 5 2 5"/>',
            f'<path d="{path(combined_shift, bottom_coordinate)}" fill="none" stroke="#777777" stroke-width="2" stroke-dasharray="5 5"/>',
            f'<text class="label" transform="translate(31 {(top_y0+top_y1)/2:.0f}) rotate(-90)" text-anchor="middle">chiral block F6(z,t1,t2)</text>',
            f'<text class="label" transform="translate(31 {(bottom_y0+bottom_y1)/2:.0f}) rotate(-90)" text-anchor="middle">relative difference</text>',
            f'<text class="label" x="{(x0+x1)/2:.0f}" y="{height-20}" text-anchor="middle">first mobile insertion z (0 &lt; z &lt; t1)</text>',
        ]
    )

    top_legend = (
        ("#1f4e79", "", "c-recursion, total degree N=20"),
        ("#d95f02", "10 7", "pillow h-recursion, total degree N=10"),
    )
    legend_x, legend_y = x1 - 355, top_y0 + 24
    for index, (color, dash, label) in enumerate(top_legend):
        y = legend_y + 28 * index
        dash_attribute = f' stroke-dasharray="{dash}"' if dash else ""
        pieces.append(f'<line x1="{legend_x}" y1="{y}" x2="{legend_x+52}" y2="{y}" stroke="{color}" stroke-width="4"{dash_attribute}/>')
        pieces.append(f'<text class="legend" x="{legend_x+64}" y="{y+5}">{html.escape(label)}</text>')

    bottom_legend = (
        ("#7b3294", "", "|h10-c20| / |c20|"),
        ("#d95f02", "3 6", "pillow shift N=8 to 10"),
        ("#1f4e79", "11 5 2 5", "c shift N=18 to 20"),
        ("#777777", "5 5", "sum of observed shifts"),
    )
    legend_x, legend_y = x0 + 22, bottom_y0 + 24
    for index, (color, dash, label) in enumerate(bottom_legend):
        y = legend_y + 27 * index
        dash_attribute = f' stroke-dasharray="{dash}"' if dash else ""
        pieces.append(f'<line x1="{legend_x}" y1="{y}" x2="{legend_x+48}" y2="{y}" stroke="{color}" stroke-width="3"{dash_attribute}/>')
        pieces.append(f'<text class="legend" x="{legend_x+60}" y="{y+5}">{html.escape(label)}</text>')

    pieces.append("</svg>")
    output_path.write_text("\n".join(pieces) + "\n", encoding="utf-8")


def run(output_directory: Path) -> dict[str, object]:
    mp.mp.dps = WORKING_DIGITS
    output_directory.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()
    h_coefficients = proposed_coefficients(
        H_ORDER,
        central_charge=CENTRAL_CHARGE,
        external_weights=EXTERNAL,
        internal_weights=INTERNAL,
    )
    h_seconds = time.perf_counter() - start
    start = time.perf_counter()
    c_coefficients = sphere_six_point_c_coefficients(
        central_charge=float(CENTRAL_CHARGE),
        external_weights=tuple(map(float, EXTERNAL)),
        internal_weights=tuple(map(float, INTERNAL)),
        order1=C_ORDER,
        order2=C_ORDER,
        order3=C_ORDER,
        max_total_order=C_ORDER,
    )
    c_seconds = time.perf_counter() - start

    rows: list[dict[str, float]] = []
    max_cover_residual = mp.mpf(0)
    max_product_residual = mp.mpf(0)
    for index in range(POINT_COUNT):
        z = Z_MIN + (Z_MAX - Z_MIN) * index / (POINT_COUNT - 1)
        q = elliptic_nome(z)
        p1, p2, p3, cover_residual, product_residual = segment_nomes(
            z, FIXED_T1, FIXED_T2
        )
        max_cover_residual = max(max_cover_residual, cover_residual)
        max_product_residual = max(max_product_residual, product_residual)
        theta3 = theta3_from_nome(q)
        pillow_prefactor = pillow_plane_prefactor(
            z, FIXED_T1, FIXED_T2, p1, p2, p3, theta3
        )
        h_high = pillow_prefactor * triangular_value(
            h_coefficients, p1, p2, p3, H_ORDER
        )
        h_low = pillow_prefactor * triangular_value(
            h_coefficients, p1, p2, p3, H_LOWER_ORDER
        )

        x1, x2, x3 = z / FIXED_T1, FIXED_T1 / FIXED_T2, FIXED_T2
        primary = plane_primary_factor(z, FIXED_T1, FIXED_T2)
        c_high = primary * triangular_value(c_coefficients, x1, x2, x3, C_ORDER)
        c_low = primary * triangular_value(
            c_coefficients, x1, x2, x3, C_LOWER_ORDER
        )
        comparison = relative_shift(h_high, c_high)
        h_shift = relative_shift(h_high, h_low)
        c_shift = relative_shift(c_high, c_low)
        rows.append(
            {
                "z": float(z),
                "q": float(q),
                "p1": float(p1),
                "p2": float(p2),
                "p3": float(p3),
                "pillow_h_order10": real_part(h_high, "pillow block"),
                "pillow_h_order8": real_part(h_low, "lower pillow block"),
                "c_recursion_order20": real_part(c_high, "c-recursion block"),
                "c_recursion_order18": real_part(c_low, "lower c-recursion block"),
                "relative_h10_vs_c20": float(comparison),
                "relative_h_truncation_shift": float(h_shift),
                "relative_c_truncation_shift": float(c_shift),
                "difference_over_shift_sum": float(
                    comparison / max(h_shift + c_shift, mp.mpf("1e-70"))
                ),
                "cover_residual": float(cover_residual),
                "nome_product_residual": float(product_residual),
            }
        )

    csv_path = output_directory / "sphere_six_pillow_h_vs_c_fixed_t1_t2.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    svg_path = output_directory / "sphere_six_pillow_h_vs_c_fixed_t1_t2.svg"
    write_svg_plot(rows, svg_path)

    max_difference_row = max(rows, key=lambda row: row["relative_h10_vs_c20"])
    max_ratio_row = max(rows, key=lambda row: row["difference_over_shift_sum"])
    summary: dict[str, object] = {
        "central_charge": str(CENTRAL_CHARGE),
        "external_weights_at_0_z_t1_t2_1_infinity": [str(x) for x in EXTERNAL],
        "internal_weights": [str(x) for x in INTERNAL],
        "fixed_t1": str(FIXED_T1),
        "fixed_t2": str(FIXED_T2),
        "z_range": [str(Z_MIN), str(Z_MAX)],
        "point_count": POINT_COUNT,
        "pillow_orders": [H_LOWER_ORDER, H_ORDER],
        "c_recursion_orders": [C_LOWER_ORDER, C_ORDER],
        "h_coefficient_seconds": h_seconds,
        "c_coefficient_seconds": c_seconds,
        "max_relative_h_vs_c": max_difference_row["relative_h10_vs_c20"],
        "z_at_max_relative_h_vs_c": max_difference_row["z"],
        "max_difference_over_shift_sum": max_ratio_row["difference_over_shift_sum"],
        "z_at_max_difference_over_shift_sum": max_ratio_row["z"],
        "max_cover_residual": mp.nstr(max_cover_residual, 12),
        "max_nome_product_residual": mp.nstr(max_product_residual, 12),
        "csv": str(csv_path.resolve()),
        "svg": str(svg_path.resolve()),
    }
    json_path = output_directory / "sphere_six_pillow_h_vs_c_fixed_t1_t2.json"
    json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    summary["json"] = str(json_path.resolve())
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=ROOT / "Data Set" / "h-Recursion",
    )
    return parser.parse_args()


if __name__ == "__main__":
    result = run(parse_args().output_directory)
    print(json.dumps(result, indent=2))
