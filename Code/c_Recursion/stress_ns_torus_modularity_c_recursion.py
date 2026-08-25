#!/usr/bin/env python3
"""Test ordinary-NS torus one-point modularity by direct c-recursion.

For a bottom-component NS primary, the assembled Type-0B correlator is

    G_NS(tau) = integral_0^infinity dP/pi C(P,Pext,P)
                |q^(h(P)-c/24) F_P(q)|^2.

The ordinary NS spin structure is fixed by modular S, so the independently
evaluated frames must obey

    G_NS(-1/tau) = |tau|^(2 d_ext) G_NS(tau).

Every chiral block is evaluated from the functional Zamolodchikov
central-charge recursion.  Its terminal nodes are the exact hypergeometric
osp(1|2) torus block times the converged non-global NS vacuum product.  There
is no local-q or elliptic-q series cutoff; ``recursion_order`` limits only the
accumulated physical Kac level in nested residue paths.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import html
import json
import math
from pathlib import Path
from typing import Iterable, Sequence

import mpmath

from compare_ns_torus_c_h_recursion import (
    TorusCRecursion,
    _central_charge,
    _ns_weight,
)
from super_liouville_structure_constants import ns_structure_constant
from super_liouville_torus_one_point import (
    ns_lift_sign_from_tau,
    type0b_ns_gauss_legendre_rule,
)


DATA_ROOT = Path(__file__).resolve().parents[2] / "Data Set"
DEFAULT_IMAGINARY_PARTS = (0.25,)


@dataclass(frozen=True)
class DirectCModularResult:
    recursion_order: int
    tau: complex
    s_tau: complex
    lift_sign: int
    s_lift_sign: int
    value_tau: complex
    value_s_tau: complex
    external_weight: float

    @property
    def expected_ratio(self) -> float:
        return abs(self.tau) ** (2.0 * self.external_weight)

    @property
    def transformed_back(self) -> complex:
        return self.value_s_tau / self.expected_ratio

    @property
    def relative_residual(self) -> float:
        return abs(self.value_tau - self.transformed_back) / max(
            abs(self.value_tau), abs(self.transformed_back), 1.0e-300
        )

    @property
    def signed_relative_residual(self) -> float:
        denominator = max(
            abs(self.value_tau), abs(self.transformed_back), 1.0e-300
        )
        return (self.value_tau.real - self.transformed_back.real) / denominator

    def json_record(self) -> dict:
        return {
            "recursion_order": self.recursion_order,
            "tau": [self.tau.real, self.tau.imag],
            "s_tau": [self.s_tau.real, self.s_tau.imag],
            "lift_sign": self.lift_sign,
            "s_lift_sign": self.s_lift_sign,
            "value_tau": [self.value_tau.real, self.value_tau.imag],
            "value_s_tau": [self.value_s_tau.real, self.value_s_tau.imag],
            "expected_ratio": self.expected_ratio,
            "transformed_back": [
                self.transformed_back.real,
                self.transformed_back.imag,
            ],
            "relative_residual": self.relative_residual,
            "signed_relative_residual": self.signed_relative_residual,
        }


def _parse_csv(text: str, caster) -> tuple:
    return tuple(caster(item.strip()) for item in text.split(",") if item.strip())


def _mp_complex(value: complex):
    """Lift a Python complex into the current mpmath precision context."""

    value = complex(value)
    return mpmath.mpc(str(value.real), str(value.imag))


def direct_c_modularity_scan(
    *,
    taus: Sequence[complex],
    recursion_orders: Iterable[int] = (12, 14, 16, 18, 20),
    external_momentum: float = 0.33,
    p_max: float = 5.5,
    quadrature_order: int = 96,
    structure_precision: int = 45,
    working_precision: int = 110,
) -> list[DirectCModularResult]:
    """Return independent direct- and S-frame continuum integrals."""

    points = tuple(complex(tau) for tau in taus)
    if not points or any(tau.imag <= 0.0 for tau in points):
        raise ValueError("taus must be a nonempty upper-half-plane sequence")
    orders = tuple(int(order) for order in recursion_orders)
    if not orders or any(order < 0 for order in orders):
        raise ValueError("recursion_orders must contain nonnegative integers")

    s_points = tuple(-1.0 / tau for tau in points)
    lift_signs = tuple(ns_lift_sign_from_tau(tau) for tau in points)
    s_lift_signs = tuple(ns_lift_sign_from_tau(s_tau) for s_tau in s_points)

    with mpmath.workdps(working_precision):
        tau_values_mp = tuple(
            mpmath.mpc(str(tau.real), str(tau.imag)) for tau in points
        )
        s_tau_values_mp = tuple(-1 / tau for tau in tau_values_mp)
        q_values = tuple(
            mpmath.exp(2.0j * mpmath.pi * tau) for tau in tau_values_mp
        )
        s_q_values = tuple(
            mpmath.exp(2.0j * mpmath.pi * s_tau)
            for s_tau in s_tau_values_mp
        )
        all_q_values = q_values + s_q_values
        all_lift_signs = lift_signs + s_lift_signs
        b = mpmath.mpf(1)
        central = _central_charge(b)
        external_weight_mp = _ns_weight(
            mpmath.mpf(str(external_momentum)), b
        )
        totals = {
            order: [mpmath.mpc(0) for _ in all_q_values]
            for order in orders
        }

        for momentum, spectral_weight in type0b_ns_gauss_legendre_rule(
            p_max, quadrature_order
        ):
            structure = ns_structure_constant(
                momentum,
                external_momentum,
                momentum,
                structure_precision,
            )
            weighted_structure = (
                mpmath.mpf(str(spectral_weight))
                * _mp_complex(structure)
                / mpmath.pi
            )
            internal_weight = _ns_weight(
                mpmath.mpf(str(momentum)), b
            )
            block = TorusCRecursion(
                c=central,
                internal_weight=internal_weight,
                external_weight=external_weight_mp,
            )
            leading_powers = tuple(
                q ** (internal_weight - central / 24)
                for q in all_q_values
            )
            for order in orders:
                descendant_blocks = block.recursive_blocks(
                    all_q_values,
                    order,
                    all_lift_signs,
                )
                for index, (leading, descendant) in enumerate(
                    zip(leading_powers, descendant_blocks)
                ):
                    chiral = leading * descendant
                    totals[order][index] += (
                        weighted_structure * abs(chiral) ** 2
                    )

        external_weight = float(mpmath.re(external_weight_mp))
        results = []
        point_count = len(points)
        for order in orders:
            values = totals[order]
            for index, (tau, s_tau) in enumerate(zip(points, s_points)):
                results.append(
                    DirectCModularResult(
                        recursion_order=order,
                        tau=tau,
                        s_tau=s_tau,
                        lift_sign=lift_signs[index],
                        s_lift_sign=s_lift_signs[index],
                        value_tau=complex(values[index]),
                        value_s_tau=complex(values[point_count + index]),
                        external_weight=external_weight,
                    )
                )
    return results


def build_ledger(
    *,
    taus: Sequence[complex],
    recursion_orders: Iterable[int] = (12, 14, 16, 18, 20),
    external_momentum: float = 0.33,
    p_max: float = 5.5,
    quadrature_order: int = 96,
    structure_precision: int = 45,
    working_precision: int = 110,
) -> dict:
    orders = tuple(recursion_orders)
    results = direct_c_modularity_scan(
        taus=taus,
        recursion_orders=orders,
        external_momentum=external_momentum,
        p_max=p_max,
        quadrature_order=quadrature_order,
        structure_precision=structure_precision,
        working_precision=working_precision,
    )
    maxima = {
        str(order): max(
            result.relative_residual
            for result in results
            if result.recursion_order == order
        )
        for order in orders
    }
    drift = None
    if len(orders) >= 2:
        previous, final = orders[-2:]
        previous_rows = sorted(
            (row for row in results if row.recursion_order == previous),
            key=lambda row: (row.tau.real, row.tau.imag),
        )
        final_rows = sorted(
            (row for row in results if row.recursion_order == final),
            key=lambda row: (row.tau.real, row.tau.imag),
        )
        drift = max(
            max(
                abs(old.value_tau - new.value_tau)
                / max(abs(old.value_tau), abs(new.value_tau), 1.0e-300),
                abs(old.value_s_tau - new.value_s_tau)
                / max(abs(old.value_s_tau), abs(new.value_s_tau), 1.0e-300),
            )
            for old, new in zip(previous_rows, final_rows)
        )
    ledger = {
        "parameters": {
            "hat_c": 9.0,
            "c": 13.5,
            "b": 1.0,
            "external_momentum": external_momentum,
            "external_weight": results[0].external_weight,
            "taus": [[complex(tau).real, complex(tau).imag] for tau in taus],
            "recursion_orders": list(orders),
            "p_max": p_max,
            "quadrature_order": quadrature_order,
            "structure_precision": structure_precision,
            "working_precision": working_precision,
            "spectral_measure": "dP/pi",
            "block_evaluation": (
                "functional-c-recursion-with-exact-global-and-vacuum-product-leaves"
            ),
            "series_truncation": None,
        },
        "max_relative_residual_by_recursion_order": maxima,
        "max_individual_frame_drift_last_two_orders": drift,
        "rows": [result.json_record() for result in results],
    }
    if len(taus) == 1:
        tau = complex(taus[0])
        s_tau = -1 / tau
        ledger["hard_point"] = {
            "tau": [tau.real, tau.imag],
            "s_tau": [s_tau.real, s_tau.imag],
            "q_abs": math.exp(-2 * math.pi * tau.imag),
            "q_tilde_abs": math.exp(-2 * math.pi * s_tau.imag),
        }
    return ledger


def _nice_ticks(low: float, high: float, count: int = 5) -> tuple[float, ...]:
    if high <= low:
        return (low,)
    raw = (high - low) / max(1, count - 1)
    exponent = 10.0 ** math.floor(math.log10(raw))
    fraction = raw / exponent
    step = (
        1.0 if fraction <= 1.0 else 2.0 if fraction <= 2.0 else 5.0
    ) * exponent
    first = math.ceil(low / step) * step
    values = []
    value = first
    tolerance = 1.0e-10 * max(
        abs(low), abs(high), abs(step), 1.0e-300
    )
    while value <= high + tolerance:
        values.append(value)
        value += step
    return tuple(values)


def _svg_path(x_values, y_values, x_map, y_map) -> str:
    return " ".join(
        ("M" if index == 0 else "L")
        + f"{x_map(x_value):.3f},{y_map(y_value):.3f}"
        for index, (x_value, y_value) in enumerate(zip(x_values, y_values))
    )


def render_svg(ledger: dict, output: Path) -> None:
    """Plot both modular frames and their signed relative residual."""

    final_order = max(ledger["parameters"]["recursion_orders"])
    rows = sorted(
        (
            row
            for row in ledger["rows"]
            if row["recursion_order"] == final_order
        ),
        key=lambda row: row["tau"][1],
    )
    if len(rows) < 2:
        raise ValueError("the modularity plot requires at least two tau values")
    y_tau = [float(row["tau"][1]) for row in rows]
    direct = [float(row["value_tau"][0]) for row in rows]
    transformed = [float(row["transformed_back"][0]) for row in rows]
    residual = [float(row["signed_relative_residual"]) for row in rows]

    width, height = 1240, 590
    panel_width, panel_height = 500, 350
    top, left_main, left_residual = 108, 90, 700
    x_low, x_high = min(y_tau), max(y_tau)
    value_low, value_high = min(direct + transformed), max(direct + transformed)
    padding = 0.08 * max(value_high - value_low, abs(value_high), 1.0e-12)
    value_low -= padding
    value_high += padding
    residual_limit = 1.12 * max(max(map(abs, residual)), 1.0e-16)

    tau_real = rows[0]["tau"][0]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#191919}'
        '.axis{stroke:#333;stroke-width:1}.grid{stroke:#ddd;stroke-width:1}'
        '.tick{font-size:13px}.label{font-size:16px}.legend{font-size:13px}'
        '.direct{fill:none;stroke:#111;stroke-width:3}'
        '.modular{fill:none;stroke:#2754b8;stroke-width:2.5;stroke-dasharray:8 4}'
        '.residual{fill:none;stroke:#b23a48;stroke-width:2.5}</style>',
        '<text x="620" y="29" text-anchor="middle" font-size="21">'
        'NS–NS torus one-point modularity at ĉ=9</text>',
        '<text x="620" y="55" text-anchor="middle" font-size="14">'
        f'direct functional c-recursion, order {final_order}; '
        f'τ={tau_real:.3g}+iy; no q-series cutoff</text>',
    ]

    def draw_axes(left, vertical_low, vertical_high, title, scientific=False):
        x_map = lambda value: left + panel_width * (value - x_low) / (x_high - x_low)
        y_map = lambda value: top + panel_height * (
            vertical_high - value
        ) / (vertical_high - vertical_low)
        parts.append(
            f'<text x="{left + panel_width / 2}" y="{top - 18}" '
            f'text-anchor="middle" class="label">{html.escape(title)}</text>'
        )
        for tick in _nice_ticks(vertical_low, vertical_high, 6):
            y = y_map(tick)
            label = f"{tick:.1e}" if scientific else f"{tick:.3g}"
            parts.append(
                f'<line x1="{left}" y1="{y:.3f}" x2="{left + panel_width}" '
                f'y2="{y:.3f}" class="grid"/>'
            )
            parts.append(
                f'<text x="{left - 10}" y="{y + 4:.3f}" text-anchor="end" '
                f'class="tick">{label}</text>'
            )
        for tick in _nice_ticks(x_low, x_high, 7):
            x = x_map(tick)
            parts.append(
                f'<line x1="{x:.3f}" y1="{top}" x2="{x:.3f}" '
                f'y2="{top + panel_height}" class="grid"/>'
            )
            parts.append(
                f'<text x="{x:.3f}" y="{top + panel_height + 23}" '
                f'text-anchor="middle" class="tick">{tick:.2f}</text>'
            )
        parts.append(
            f'<line x1="{left}" y1="{top + panel_height}" '
            f'x2="{left + panel_width}" y2="{top + panel_height}" class="axis"/>'
        )
        parts.append(
            f'<line x1="{left}" y1="{top}" x2="{left}" '
            f'y2="{top + panel_height}" class="axis"/>'
        )
        parts.append(
            f'<text x="{left + panel_width / 2}" y="{top + panel_height + 53}" '
            'text-anchor="middle" class="label">Im τ</text>'
        )
        return x_map, y_map

    x_main, y_main = draw_axes(
        left_main,
        value_low,
        value_high,
        "Independent modular frames",
    )
    parts.append(
        f'<path d="{_svg_path(y_tau, direct, x_main, y_main)}" class="direct"/>'
    )
    parts.append(
        f'<path d="{_svg_path(y_tau, transformed, x_main, y_main)}" class="modular"/>'
    )

    x_residual, y_residual = draw_axes(
        left_residual,
        -residual_limit,
        residual_limit,
        "Signed relative modular residual",
        scientific=True,
    )
    parts.append(
        f'<line x1="{left_residual}" y1="{y_residual(0):.3f}" '
        f'x2="{left_residual + panel_width}" y2="{y_residual(0):.3f}" class="axis"/>'
    )
    parts.append(
        f'<path d="{_svg_path(y_tau, residual, x_residual, y_residual)}" '
        'class="residual"/>'
    )

    legend_y = 568
    legend = (
        ("G_NS(τ)", "#111", False),
        ("|τ|⁻²ᵈ G_NS(−1/τ)", "#2754b8", True),
        ("signed residual", "#b23a48", False),
    )
    for index, (label, color, dashed) in enumerate(legend):
        x = 260 + index * 285
        dash = ' stroke-dasharray="8 4"' if dashed else ""
        parts.append(
            f'<line x1="{x}" y1="{legend_y - 5}" x2="{x + 24}" '
            f'y2="{legend_y - 5}" stroke="{color}" stroke-width="3"{dash}/>'
        )
        parts.append(
            f'<text x="{x + 31}" y="{legend_y}" class="legend">'
            f'{html.escape(label)}</text>'
        )
    parts.append("</svg>")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(parts) + "\n", encoding="utf-8")


def render_order_convergence_svg(ledger: dict, output: Path) -> None:
    """Plot modular-frame and residual convergence at one hard tau value."""

    rows = sorted(ledger["rows"], key=lambda row: row["recursion_order"])
    if len(rows) < 2 or len({tuple(row["tau"]) for row in rows}) != 1:
        raise ValueError("order convergence requires one tau and at least two orders")
    orders = [int(row["recursion_order"]) for row in rows]
    direct = [float(row["value_tau"][0]) for row in rows]
    transformed = [float(row["transformed_back"][0]) for row in rows]
    residuals = [max(float(row["relative_residual"]), 1.0e-300) for row in rows]

    width, height = 1240, 590
    panel_width, panel_height = 500, 350
    top, left_main, left_residual = 108, 90, 700
    order_low, order_high = min(orders), max(orders)
    value_low, value_high = min(direct + transformed), max(direct + transformed)
    padding = 0.10 * max(value_high - value_low, 1.0e-15)
    value_low -= padding
    value_high += padding
    log_residuals = [math.log10(value) for value in residuals]
    log_low = math.floor(min(log_residuals))
    log_high = math.ceil(max(log_residuals))
    tau = complex(*rows[0]["tau"])
    q_abs = ledger.get("hard_point", {}).get(
        "q_abs", math.exp(-2 * math.pi * tau.imag)
    )
    q_tilde_abs = ledger.get("hard_point", {}).get(
        "q_tilde_abs", math.exp(-2 * math.pi * (-1 / tau).imag)
    )

    x_map = lambda value: left_main + panel_width * (value - order_low) / (order_high - order_low)
    x_map_residual = lambda value: left_residual + panel_width * (value - order_low) / (order_high - order_low)
    y_value = lambda value: top + panel_height * (value_high - value) / (value_high - value_low)
    y_log = lambda value: top + panel_height * (log_high - math.log10(value)) / (log_high - log_low)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#191919}'
        '.axis{stroke:#333;stroke-width:1}.grid{stroke:#ddd;stroke-width:1}'
        '.tick{font-size:13px}.label{font-size:16px}.legend{font-size:13px}'
        '.direct{fill:none;stroke:#111;stroke-width:3}'
        '.modular{fill:none;stroke:#2754b8;stroke-width:2.5;stroke-dasharray:8 4}'
        '.residual{fill:none;stroke:#b23a48;stroke-width:2.5}'
        '.mark-direct{fill:#111}.mark-modular{fill:#2754b8}.mark-residual{fill:#b23a48}</style>',
        '<text x="620" y="29" text-anchor="middle" font-size="21">Hard NS–NS torus modularity test at ĉ=9</text>',
        '<text x="620" y="55" text-anchor="middle" font-size="14">'
        f'τ={tau.imag:.2f}i, |q|={q_abs:.6g}, |q̃|={q_tilde_abs:.6g}; '
        'functional c-recursion with no q-series cutoff</text>',
        f'<text x="{left_main + panel_width / 2}" y="{top - 18}" text-anchor="middle" class="label">Independent modular frames</text>',
        f'<text x="{left_residual + panel_width / 2}" y="{top - 18}" text-anchor="middle" class="label">Absolute modular residual</text>',
    ]
    for tick in _nice_ticks(value_low, value_high, 6):
        y = y_value(tick)
        parts.extend((
            f'<line x1="{left_main}" y1="{y:.3f}" x2="{left_main + panel_width}" y2="{y:.3f}" class="grid"/>',
            f'<text x="{left_main - 10}" y="{y + 4:.3f}" text-anchor="end" class="tick">{tick:.7f}</text>',
        ))
    for exponent in range(log_low, log_high + 1):
        value = 10.0**exponent
        y = y_log(value)
        parts.extend((
            f'<line x1="{left_residual}" y1="{y:.3f}" x2="{left_residual + panel_width}" y2="{y:.3f}" class="grid"/>',
            f'<text x="{left_residual - 10}" y="{y + 4:.3f}" text-anchor="end" class="tick">1e{exponent}</text>',
        ))
    for order in orders:
        for left, mapping in ((left_main, x_map), (left_residual, x_map_residual)):
            x = mapping(order)
            parts.append(f'<line x1="{x:.3f}" y1="{top}" x2="{x:.3f}" y2="{top + panel_height}" class="grid"/>')
            parts.append(f'<text x="{x:.3f}" y="{top + panel_height + 23}" text-anchor="middle" class="tick">{order}</text>')
    for left in (left_main, left_residual):
        parts.append(f'<line x1="{left}" y1="{top + panel_height}" x2="{left + panel_width}" y2="{top + panel_height}" class="axis"/>')
        parts.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + panel_height}" class="axis"/>')
        parts.append(f'<text x="{left + panel_width / 2}" y="{top + panel_height + 53}" text-anchor="middle" class="label">recursion order N</text>')
    parts.append(f'<path d="{_svg_path(orders, direct, x_map, y_value)}" class="direct"/>')
    parts.append(f'<path d="{_svg_path(orders, transformed, x_map, y_value)}" class="modular"/>')
    parts.append(f'<path d="{_svg_path(orders, residuals, x_map_residual, y_log)}" class="residual"/>')
    for order, value in zip(orders, direct):
        parts.append(f'<circle cx="{x_map(order):.3f}" cy="{y_value(value):.3f}" r="4" class="mark-direct"/>')
    for order, value in zip(orders, transformed):
        parts.append(f'<circle cx="{x_map(order):.3f}" cy="{y_value(value):.3f}" r="3.5" class="mark-modular"/>')
    for order, value in zip(orders, residuals):
        parts.append(f'<circle cx="{x_map_residual(order):.3f}" cy="{y_log(value):.3f}" r="4" class="mark-residual"/>')
    legend_y = 568
    for index, (label, color, dashed) in enumerate((
        ("G_NS(τ)", "#111", False),
        ("|τ|⁻²ᵈ G_NS(−1/τ)", "#2754b8", True),
        ("absolute residual", "#b23a48", False),
    )):
        x = 260 + index * 285
        dash = ' stroke-dasharray="8 4"' if dashed else ""
        parts.append(f'<line x1="{x}" y1="{legend_y - 5}" x2="{x + 24}" y2="{legend_y - 5}" stroke="{color}" stroke-width="3"{dash}/>')
        parts.append(f'<text x="{x + 31}" y="{legend_y}" class="legend">{html.escape(label)}</text>')
    parts.append("</svg>")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(parts) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tau-real", type=float, default=0.0)
    parser.add_argument(
        "--tau-imag",
        default=",".join(f"{value:.2f}" for value in DEFAULT_IMAGINARY_PARTS),
    )
    parser.add_argument("--orders", default="12,14,16,18,20")
    parser.add_argument("--external-momentum", type=float, default=0.33)
    parser.add_argument("--p-max", type=float, default=5.5)
    parser.add_argument("--quadrature-order", type=int, default=96)
    parser.add_argument("--structure-precision", type=int, default=45)
    parser.add_argument("--working-precision", type=int, default=110)
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            DATA_ROOT
            / "ns_torus_modularity_direct_c_recursion_hard_tau025.json"
        ),
    )
    parser.add_argument(
        "--plot-output",
        type=Path,
        default=DATA_ROOT / "ns_torus_modularity_direct_c_recursion_hard_tau025.svg",
    )
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    taus = tuple(
        complex(args.tau_real, imaginary)
        for imaginary in _parse_csv(args.tau_imag, float)
    )
    ledger = build_ledger(
        taus=taus,
        recursion_orders=_parse_csv(args.orders, int),
        external_momentum=args.external_momentum,
        p_max=args.p_max,
        quadrature_order=args.quadrature_order,
        structure_precision=args.structure_precision,
        working_precision=args.working_precision,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(ledger, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(args.output)
    for order, residual in ledger[
        "max_relative_residual_by_recursion_order"
    ].items():
        print(f"order {order}: max modular residual {residual:.8e}")
    if not args.no_plot:
        if len(taus) == 1:
            render_order_convergence_svg(ledger, args.plot_output)
        else:
            render_svg(ledger, args.plot_output)
        print(args.plot_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
