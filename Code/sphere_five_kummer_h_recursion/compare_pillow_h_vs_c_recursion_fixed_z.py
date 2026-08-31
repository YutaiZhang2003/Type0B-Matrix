#!/usr/bin/env python3
"""Compare the pillow and CCY plane h-recursions with the CCY c-recursion.

The pillow recursion returns the reduced elliptic block H5(p1,p2).  This
script restores the complete c-1 conformal prefactor, so that its output is
the same plane-normalized chiral sphere block produced by the independent
fixed-weight central-charge recursion.  It also evaluates the CCY plane
h-recursion at high precision and audits its coefficients against exact PBW
data through total degree ten.

Conventions in the machine note are

    (d1,d2,d3,d4,d5) at (0,z,1,infinity,t),

whereas ``ccy_sphere_five_point.py`` orders the mobile insertion third:

    (d1,d2,d5,d3,d4) at (0,z,t,1,infinity).

The default plot fixes z=0.08 and scans z<t<1.  Both h-recursions are
truncated at total degree ten; the c-recursion is evaluated through plane
degree 22, with order shifts shown as convergence diagnostics.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import sys
from functools import lru_cache
from pathlib import Path

import mpmath as mp
import sympy as sp

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

from check_pillow_h_recursion_numerical_order10 import (  # noqa: E402
    a_rs,
    background_data,
    degenerate_weight,
    direct_pbw_coefficients,
    fusion_polynomial,
    proposed_coefficients,
)
from ccy_sphere_five_point import (  # noqa: E402
    sphere_five_point_c_coefficients,
)


CENTRAL_CHARGE = mp.mpf("31.7")
# Machine-note order: (0,z,1,infinity,t).
EXTERNAL_NOTE = tuple(map(mp.mpf, ("0.21", "0.34", "0.63", "0.79", "0.49")))
INTERNAL = tuple(map(mp.mpf, ("1.03", "1.19")))
FIXED_Z = mp.mpf("0.08")
T_MIN = mp.mpf("0.12")
T_MAX = mp.mpf("0.72")
POINT_COUNT = 61
H_ORDER = 10
H_LOWER_ORDER = 8
C_ORDER = 22
C_LOWER_ORDER = 20


def ccy_external_order(weights: tuple[mp.mpf, ...]) -> tuple[float, ...]:
    """Convert (d1,d2,d3,d4,d5) to CCY's (d1,d2,d5,d3,d4)."""

    d1, d2, d3, d4, d5 = weights
    return tuple(map(float, (d1, d2, d5, d3, d4)))


def pillow_external_order(weights: tuple[mp.mpf, ...]) -> tuple[mp.mpf, ...]:
    """Convert note notation to the ordering used by the pillow checker."""

    d1, d2, d3, d4, d5 = weights
    return d1, d2, d5, d3, d4


def elliptic_nome(z: mp.mpf) -> mp.mpf:
    return mp.exp(-mp.pi * mp.ellipk(1 - z) / mp.ellipk(z))


def theta3_from_nome(q: mp.mpf) -> mp.mpf:
    return mp.jtheta(3, mp.mpf(0), q)


def t_from_segment_nome(p2: mp.mpf, q: mp.mpf) -> mp.mpf:
    """Evaluate the exact inverse-covering product t=4 p2 Y(q/p2,p2)."""

    p1 = q / p2
    y = (1 + p1) ** 2
    n = 1
    while True:
        q_odd = q ** (2 * n - 1)
        q_even = q ** (2 * n)
        y *= ((1 + q_even) / (1 + q_odd)) ** 4
        y *= (
            (1 + p1 ** (2 * n + 1) * p2 ** (2 * n))
            * (1 + p1 ** (2 * n - 1) * p2 ** (2 * n))
            / (
                (1 + p1 ** (2 * n) * p2 ** (2 * n - 1))
                * (1 + p1 ** (2 * n - 2) * p2 ** (2 * n - 1))
            )
        ) ** 2
        if max(abs(q_even), abs(p1) ** (2 * n), abs(p2) ** (2 * n)) < mp.mpf(
            "1e-75"
        ):
            break
        n += 1
        if n > 10000:
            raise RuntimeError("inverse-covering product did not converge")
    return 4 * p2 * y


def segment_nomes(z: mp.mpf, t: mp.mpf) -> tuple[mp.mpf, mp.mpf, mp.mpf]:
    """Invert the aligned product map on the real comb region z<t<1."""

    if not 0 < z < t < 1:
        raise ValueError("the real aligned branch requires 0 < z < t < 1")
    q = elliptic_nome(z)
    lower = q * (1 + mp.mpf("1e-40"))
    upper = 1 - mp.mpf("1e-40")
    # 220 bisections resolve the map to roughly 66 decimal digits, far beyond
    # the series truncation uncertainty displayed in the comparison.
    for _ in range(220):
        midpoint = (lower + upper) / 2
        if t_from_segment_nome(midpoint, q) < t:
            lower = midpoint
        else:
            upper = midpoint
    p2 = (lower + upper) / 2
    p1 = q / p2
    residual = abs(t_from_segment_nome(p2, q) - t)
    return p1, p2, residual


def triangular_value(
    coefficients: dict[tuple[int, int], complex | mp.mpf | mp.mpc],
    x1: mp.mpf,
    x2: mp.mpf,
    order: int,
) -> mp.mpc:
    total = mp.mpc(0)
    for (n1, n2), coefficient in coefficients.items():
        if n1 + n2 <= order:
            total += mp.mpc(coefficient) * x1**n1 * x2**n2
    return total


def pillow_plane_prefactor(
    z: mp.mpf,
    t: mp.mpf,
    p1: mp.mpf,
    p2: mp.mpf,
    theta3: mp.mpf,
) -> mp.mpf:
    """Return the character-absorbed c-1 prefactor on the real branch."""

    d1, d2, d3, d4, d5 = EXTERNAL_NOTE
    h1, h2 = INTERNAL
    delta = (CENTRAL_CHARGE - 1) / 24
    anomaly = (CENTRAL_CHARGE - 1) / 2 - 4 * (d1 + d2 + d3 + d4) - 2 * d5
    lambda_factor = (
        theta3**anomaly
        * z ** (delta - d1 - d2)
        * (1 - z) ** (delta - d2 - d3)
        * (t * (1 - t) * (t - z)) ** (-d5 / 2)
    )
    propagation = (4 * p1) ** (h1 - delta) * (4 * p2) ** (h2 - delta)
    return lambda_factor * propagation


def plane_primary_factor(z: mp.mpf, t: mp.mpf) -> mp.mpf:
    d1, d2, _, _, d5 = EXTERNAL_NOTE
    h1, h2 = INTERNAL
    return z ** (h1 - d1 - d2) * t ** (h2 - h1 - d5)


def relative_shift(value: mp.mpc, reference: mp.mpc) -> mp.mpf:
    return abs(value - reference) / max(abs(reference), mp.mpf("1e-70"))


def ccy_plane_h_coefficients(
    order: int,
) -> dict[tuple[int, int], mp.mpf | mp.mpc]:
    """High-precision specialization of CCY (3.26) to the five-point comb.

    CCY hold ``a=h2-h1``, ``e_left=d1-h1``, and
    ``e_right=d4-h1`` fixed in the plane-frame large-h limit.  Here d4 is
    the operator at infinity in the machine-note ordering.
    """

    d1, d2, d3, d4, d5 = EXTERNAL_NOTE
    initial_h, h2 = INTERNAL
    initial_a = h2 - initial_h
    initial_e_left = d1 - initial_h
    initial_e_right = d4 - initial_h
    q_background, b = background_data(CENTRAL_CHARGE)

    @lru_cache(maxsize=None)
    def coefficient(
        n1: int,
        n2: int,
        current_h: mp.mpf | mp.mpc,
        current_a: mp.mpf | mp.mpc,
        current_e_left: mp.mpf | mp.mpc,
        current_e_right: mp.mpf | mp.mpc,
    ) -> mp.mpf | mp.mpc:
        total: mp.mpf | mp.mpc = mp.mpf(1) if (n1, n2) == (0, 0) else mp.mpf(0)
        for r in range(1, n1 + 1):
            for s in range(1, n1 // r + 1):
                level = r * s
                pole = degenerate_weight(r, s, q_background, b)
                residue = (
                    a_rs(r, s, b)
                    * fusion_polynomial(
                        r,
                        s,
                        top=pole + current_e_left,
                        bottom=d2,
                        q_background=q_background,
                        b=b,
                    )
                    * fusion_polynomial(
                        r,
                        s,
                        top=pole + current_a,
                        bottom=d5,
                        q_background=q_background,
                        b=b,
                    )
                    / (current_h - pole)
                )
                total += residue * coefficient(
                    n1 - level,
                    n2,
                    pole + level,
                    current_a - level,
                    current_e_left - level,
                    current_e_right - level,
                )
        for r in range(1, n2 + 1):
            for s in range(1, n2 // r + 1):
                level = r * s
                pole = degenerate_weight(r, s, q_background, b)
                residue = (
                    a_rs(r, s, b)
                    * fusion_polynomial(
                        r,
                        s,
                        top=pole - current_a,
                        bottom=d5,
                        q_background=q_background,
                        b=b,
                    )
                    * fusion_polynomial(
                        r,
                        s,
                        top=pole - current_a + current_e_right,
                        bottom=d3,
                        q_background=q_background,
                        b=b,
                    )
                    / (current_h + current_a - pole)
                )
                total += residue * coefficient(
                    n1,
                    n2 - level,
                    pole - current_a,
                    current_a + level,
                    current_e_left,
                    current_e_right,
                )
        return total

    return {
        (n1, n2): coefficient(
            n1,
            n2,
            initial_h,
            initial_a,
            initial_e_left,
            initial_e_right,
        )
        for n1 in range(order + 1)
        for n2 in range(order + 1 - n1)
    }


def write_svg_plot(rows: list[dict[str, float]], output_path: Path) -> None:
    """Write a dependency-free two-panel scientific plot as vector SVG."""

    width, height = 1200, 900
    left, right = 118.0, 35.0
    top_y0, top_y1 = 145.0, 505.0
    bottom_y0, bottom_y1 = 605.0, 815.0
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
    uncertainty_ratios = [
        difference / max(higher_shift + plane_shift, 1.0e-300)
        for difference, higher_shift, plane_shift in zip(comparison, h_shift, c_shift)
    ]
    within_observed_shifts = all(ratio <= 1.0 for ratio in uncertainty_ratios)
    if not within_observed_shifts:
        raise AssertionError("h/c disagreement exceeds the observed truncation shifts")
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

    def path(values: list[float], y_function) -> str:
        return " ".join(
            ("M" if index == 0 else "L")
            + f" {x_coordinate(t_value):.2f} {y_function(value):.2f}"
            for index, (t_value, value) in enumerate(zip(t_values, values))
        )

    pieces = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Helvetica,Arial,sans-serif;fill:#202124}.axis{stroke:#202124;stroke-width:1.3}.grid{stroke:#c8cdd2;stroke-width:1;opacity:.45}.tick{font-size:16px}.label{font-size:19px}.legend{font-size:16px}.title{font-size:23px;font-weight:600}.subtitle{font-size:15px}</style>',
        f'<text class="title" x="{width/2:.0f}" y="38" text-anchor="middle">Sphere five-point block: pillow h-recursion vs. CCY recursions</text>',
        f'<text class="subtitle" x="{width/2:.0f}" y="68" text-anchor="middle">z=0.08, c=31.7, (d1,d2,d3,d4,d5)=(0.21,0.34,0.63,0.79,0.49), (h1,h2)=(1.03,1.19)</text>',
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
        t_value = t_low + index * (t_high - t_low) / 6
        x = x_coordinate(t_value)
        pieces.append(f'<line class="grid" x1="{x:.2f}" y1="{top_y0}" x2="{x:.2f}" y2="{top_y1}"/>')
        pieces.append(f'<line class="grid" x1="{x:.2f}" y1="{bottom_y0}" x2="{x:.2f}" y2="{bottom_y1}"/>')
        pieces.append(f'<text class="tick" x="{x:.2f}" y="{bottom_y1+27}" text-anchor="middle">{t_value:.2f}</text>')

    pieces.extend(
        [
            f'<line class="axis" x1="{x0}" y1="{top_y0}" x2="{x0}" y2="{top_y1}"/>',
            f'<line class="axis" x1="{x0}" y1="{top_y1}" x2="{x1}" y2="{top_y1}"/>',
            f'<line class="axis" x1="{x0}" y1="{bottom_y0}" x2="{x0}" y2="{bottom_y1}"/>',
            f'<line class="axis" x1="{x0}" y1="{bottom_y1}" x2="{x1}" y2="{bottom_y1}"/>',
            f'<path d="{path(c_values, top_coordinate)}" fill="none" stroke="#1f4e79" stroke-width="4"/>',
            f'<path d="{path(ccy_h_values, top_coordinate)}" fill="none" stroke="#1b9e77" stroke-width="3" stroke-dasharray="3 6"/>',
            f'<path d="{path(h_values, top_coordinate)}" fill="none" stroke="#d95f02" stroke-width="3" stroke-dasharray="10 7"/>',
            f'<path d="{path(comparison, bottom_coordinate)}" fill="none" stroke="#7b3294" stroke-width="4"/>',
            f'<path d="{path(ccy_comparison, bottom_coordinate)}" fill="none" stroke="#1b9e77" stroke-width="3.5"/>',
            f'<path d="{path(h_shift, bottom_coordinate)}" fill="none" stroke="#d95f02" stroke-width="2.5" stroke-dasharray="3 6"/>',
            f'<path d="{path(ccy_h_shift, bottom_coordinate)}" fill="none" stroke="#1b9e77" stroke-width="2.5" stroke-dasharray="3 6"/>',
            f'<path d="{path(c_shift, bottom_coordinate)}" fill="none" stroke="#1f4e79" stroke-width="2.5" stroke-dasharray="11 5 2 5"/>',
            f'<text class="label" transform="translate(31 {(top_y0+top_y1)/2:.0f}) rotate(-90)" text-anchor="middle">chiral block F5(z,t)</text>',
            f'<text class="label" transform="translate(31 {(bottom_y0+bottom_y1)/2:.0f}) rotate(-90)" text-anchor="middle">relative difference</text>',
            f'<text class="label" x="{(x0+x1)/2:.0f}" y="{height-20}" text-anchor="middle">mobile insertion t (z &lt; t &lt; 1)</text>',
        ]
    )

    legend_x, legend_y = x1 - 330, top_y0 + 22
    top_legend = (
        ("#1f4e79", "", "CCY c-recursion, N=22"),
        ("#1b9e77", "3 6", "CCY plane h-recursion, N=10"),
        ("#d95f02", "10 7", "pillow h-recursion, N=10"),
    )
    for index, (color, dash, label) in enumerate(top_legend):
        y = legend_y + 27 * index
        dash_attribute = f' stroke-dasharray="{dash}"' if dash else ""
        pieces.append(f'<line x1="{legend_x}" y1="{y}" x2="{legend_x+52}" y2="{y}" stroke="{color}" stroke-width="4"{dash_attribute}/>')
        pieces.append(f'<text class="legend" x="{legend_x+64}" y="{y+5}">{html.escape(label)}</text>')

    bottom_legend = (
        ("#7b3294", "", "pillow h10 vs. c22"),
        ("#1b9e77", "", "CCY plane h10 vs. c22"),
        ("#d95f02", "3 6", "pillow shift N=8 to 10"),
        ("#1b9e77", "3 6", "CCY h shift N=8 to 10"),
        ("#1f4e79", "11 5 2 5", "c shift N=20 to 22"),
    )
    legend_x, legend_y = x0 + 22, bottom_y0 + 25
    for index, (color, dash, label) in enumerate(bottom_legend):
        y = legend_y + 25 * index
        dash_attribute = f' stroke-dasharray="{dash}"' if dash else ""
        pieces.append(f'<line x1="{legend_x}" y1="{y}" x2="{legend_x+48}" y2="{y}" stroke="{color}" stroke-width="3"{dash_attribute}/>')
        pieces.append(f'<text class="legend" x="{legend_x+60}" y="{y+5}">{html.escape(label)}</text>')

    pieces.append("</svg>")
    output_path.write_text("\n".join(pieces) + "\n", encoding="utf-8")


def run(output_directory: Path) -> dict[str, object]:
    mp.mp.dps = WORKING_DIGITS
    output_directory.mkdir(parents=True, exist_ok=True)

    pillow_coefficients = proposed_coefficients(
        H_ORDER,
        central_charge=CENTRAL_CHARGE,
        external_weights=pillow_external_order(EXTERNAL_NOTE),
        internal_weights=INTERNAL,
    )
    ccy_h_coefficients = ccy_plane_h_coefficients(H_ORDER)
    exact_plane = direct_pbw_coefficients(
        H_ORDER,
        c=sp.Rational(str(CENTRAL_CHARGE)),
        h1=sp.Rational(str(INTERNAL[0])),
        h2=sp.Rational(str(INTERNAL[1])),
        weights=tuple(
            sp.Rational(str(value)) for value in pillow_external_order(EXTERNAL_NOTE)
        ),
    )
    ccy_coefficient_errors = {
        key: relative_shift(
            ccy_h_coefficients[key],
            mp.mpf(str(sp.N(exact_plane[key], WORKING_DIGITS))),
        )
        for key in ccy_h_coefficients
    }
    worst_ccy_coefficient = max(ccy_coefficient_errors, key=ccy_coefficient_errors.get)
    if ccy_coefficient_errors[worst_ccy_coefficient] > mp.mpf("1e-60"):
        raise AssertionError("high-precision CCY h-recursion failed the exact PBW audit")
    c_coefficients = sphere_five_point_c_coefficients(
        central_charge=float(CENTRAL_CHARGE),
        external_weights=ccy_external_order(EXTERNAL_NOTE),
        internal_weights=tuple(map(float, INTERNAL)),
        order1=C_ORDER,
        order2=C_ORDER,
        max_total_order=C_ORDER,
    )

    z = FIXED_Z
    q = elliptic_nome(z)
    theta3 = theta3_from_nome(q)
    rows: list[dict[str, float]] = []
    maximum_map_residual = mp.mpf(0)
    for index in range(POINT_COUNT):
        t = T_MIN + (T_MAX - T_MIN) * index / (POINT_COUNT - 1)
        p1, p2, residual = segment_nomes(z, t)
        maximum_map_residual = max(maximum_map_residual, residual)

        prefactor = pillow_plane_prefactor(z, t, p1, p2, theta3)
        h10 = prefactor * triangular_value(pillow_coefficients, p1, p2, H_ORDER)
        h8 = prefactor * triangular_value(
            pillow_coefficients, p1, p2, H_LOWER_ORDER
        )

        q1 = z / t
        primary = plane_primary_factor(z, t)
        c22 = primary * triangular_value(c_coefficients, q1, t, C_ORDER)
        c20 = primary * triangular_value(c_coefficients, q1, t, C_LOWER_ORDER)
        ccy_h10 = primary * triangular_value(ccy_h_coefficients, q1, t, H_ORDER)
        ccy_h8 = primary * triangular_value(
            ccy_h_coefficients, q1, t, H_LOWER_ORDER
        )
        if max(abs(mp.im(c22)), abs(mp.im(c20))) > mp.mpf("1e-12"):
            raise AssertionError("the real-branch c-recursion acquired an imaginary part")

        rows.append(
            {
                "t": float(t),
                "p1": float(p1),
                "p2": float(p2),
                "p1_p2_minus_q": float(p1 * p2 - q),
                "pillow_h_order10": float(mp.re(h10)),
                "pillow_h_order8": float(mp.re(h8)),
                "c_recursion_order22": float(mp.re(c22)),
                "c_recursion_order20": float(mp.re(c20)),
                "ccy_plane_h_order10": float(mp.re(ccy_h10)),
                "ccy_plane_h_order8": float(mp.re(ccy_h8)),
                "relative_h10_vs_c22": float(relative_shift(h10, c22)),
                "relative_h_truncation_shift": float(relative_shift(h10, h8)),
                "relative_c_truncation_shift": float(relative_shift(c22, c20)),
                "relative_ccy_h10_vs_c22": float(relative_shift(ccy_h10, c22)),
                "relative_ccy_h_truncation_shift": float(
                    relative_shift(ccy_h10, ccy_h8)
                ),
                "map_residual": float(residual),
            }
        )

    csv_path = output_directory / "sphere_five_pillow_h_vs_c_fixed_z.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    t_values = [row["t"] for row in rows]
    h_values = [row["pillow_h_order10"] for row in rows]
    c_values = [row["c_recursion_order22"] for row in rows]
    comparison = [row["relative_h10_vs_c22"] for row in rows]
    h_shift = [row["relative_h_truncation_shift"] for row in rows]
    c_shift = [row["relative_c_truncation_shift"] for row in rows]
    ccy_comparison = [row["relative_ccy_h10_vs_c22"] for row in rows]
    ccy_h_shift = [row["relative_ccy_h_truncation_shift"] for row in rows]

    uncertainty_ratios = [
        difference / max(higher_shift + plane_shift, 1.0e-300)
        for difference, higher_shift, plane_shift in zip(comparison, h_shift, c_shift)
    ]
    within_observed_shifts = all(ratio <= 1.0 for ratio in uncertainty_ratios)
    if not within_observed_shifts:
        raise AssertionError("h/c disagreement exceeds the observed truncation shifts")
    ccy_uncertainty_ratios = [
        difference / max(higher_shift + plane_shift, 1.0e-300)
        for difference, higher_shift, plane_shift in zip(
            ccy_comparison, ccy_h_shift, c_shift
        )
    ]
    ccy_within_observed_shifts = all(
        ratio <= 1.0 for ratio in ccy_uncertainty_ratios
    )
    if not ccy_within_observed_shifts:
        raise AssertionError("CCY h/c disagreement exceeds the observed order shifts")

    svg_path = output_directory / "sphere_five_pillow_h_vs_c_fixed_z.svg"
    write_svg_plot(rows, svg_path)

    summary = {
        "description": (
            "sphere five-point pillow and CCY plane h-recursions "
            "versus CCY c-recursion"
        ),
        "central_charge": float(CENTRAL_CHARGE),
        "external_weights_note_order": [float(value) for value in EXTERNAL_NOTE],
        "internal_weights": [float(value) for value in INTERNAL],
        "fixed_z": float(FIXED_Z),
        "t_range": [float(T_MIN), float(T_MAX)],
        "point_count": POINT_COUNT,
        "elliptic_nome_q": float(q),
        "h_total_order": H_ORDER,
        "c_total_order": C_ORDER,
        "maximum_relative_h10_vs_c22": max(comparison),
        "median_relative_h10_vs_c22": sorted(comparison)[len(comparison) // 2],
        "maximum_relative_h_order_shift": max(h_shift),
        "maximum_relative_c_order_shift": max(c_shift),
        "maximum_ratio_to_combined_order_shifts": max(uncertainty_ratios),
        "all_points_within_combined_order_shifts": within_observed_shifts,
        "ccy_plane_h_total_order": H_ORDER,
        "maximum_relative_ccy_h10_vs_c22": max(ccy_comparison),
        "median_relative_ccy_h10_vs_c22": sorted(ccy_comparison)[
            len(ccy_comparison) // 2
        ],
        "maximum_relative_ccy_h_order_shift": max(ccy_h_shift),
        "maximum_ccy_ratio_to_combined_order_shifts": max(
            ccy_uncertainty_ratios
        ),
        "all_ccy_points_within_combined_order_shifts": ccy_within_observed_shifts,
        "maximum_ccy_coefficient_vs_exact_pbw_error": float(
            ccy_coefficient_errors[worst_ccy_coefficient]
        ),
        "worst_ccy_coefficient": list(worst_ccy_coefficient),
        "maximum_inverse_map_residual": float(maximum_map_residual),
        "anchors": [rows[0], rows[len(rows) // 2], rows[-1]],
    }
    json_path = output_directory / "sphere_five_pillow_h_vs_c_fixed_z.json"
    with json_path.open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)
        stream.write("\n")

    print(json.dumps(summary, indent=2))
    print(f"wrote {svg_path}")
    print(f"wrote {csv_path}")
    print(f"wrote {json_path}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=ROOT / "Data Set" / "h-Recursion",
    )
    arguments = parser.parse_args()
    run(arguments.output_directory.resolve())


if __name__ == "__main__":
    main()
