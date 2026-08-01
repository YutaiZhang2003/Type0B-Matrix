#!/usr/bin/env python3
"""Plot the highest-cutoff torus modularity scan from a reduced cluster run.

The input is the deterministic ``summary.json`` written by
``super_liouville_torus_modular_cluster.py reduce`` together with the
configuration snapshot used for that run.  By default, only the largest
configured twice-level is plotted.
"""

from __future__ import annotations

import argparse
import cmath
import hashlib
import html
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from super_liouville_torus_modular_cluster import config_sha256

DEFAULT_RUN = (
    Path(__file__).resolve().parent.parent
    / "Data Set"
    / "results"
    / "type0b_torus_modular_cluster"
    / "cannon_qscan_20260723_v1"
)


def _complex_pair(value: Sequence[float]) -> complex:
    if len(value) != 2:
        raise ValueError("a complex pair must have exactly two entries")
    return complex(float(value[0]), float(value[1]))


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _repository_relative(path: Path) -> str:
    root = Path(__file__).resolve().parent
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(root))
    except ValueError:
        return str(resolved)


def validate_summary_config(
    summary: Mapping[str, Any],
    config: Mapping[str, Any],
) -> str:
    """Require the reduced summary to match the supplied configuration."""

    expected = config_sha256(config)
    observed = str(summary.get("config_sha256", ""))
    if observed != expected:
        raise ValueError(
            "summary/configuration digest mismatch: "
            f"summary={observed or '<missing>'}, config={expected}"
        )
    return expected


def _maximum_nome(tau: complex) -> float:
    q = cmath.exp(2.0j * math.pi * tau)
    q_tilde = cmath.exp(2.0j * math.pi * (-1.0 / tau))
    return max(abs(q), abs(q_tilde))


def _scan_rows(
    summary: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    study: str,
    level: int,
) -> dict[str, list[dict[str, Any]]]:
    study_data = summary["studies"][study]["taus"]
    rows: dict[str, list[dict[str, Any]]] = {}
    for entry in config["taus"]:
        name = str(entry["name"])
        family = str(entry.get("family", "scan"))
        tau = _complex_pair(entry["tau"])
        reduced = study_data[name]
        reduced_level = reduced["levels"][str(level)]
        raw_residual = float(reduced_level["relative_error_abs"])
        rows.setdefault(family, []).append(
            {
                "name": name,
                "tau": tau,
                "scan_value": float(entry.get("scan_value", tau.imag)),
                "q_max": _maximum_nome(tau),
                "raw_residual": raw_residual,
                "residual": max(raw_residual, 1.0e-16),
                "lift_sign": int(reduced["lift_sign"]),
                "lift_sign_tilde": int(reduced["lift_sign_tilde"]),
            }
        )
    return rows


def fit_power_law(
    rows: Sequence[Mapping[str, Any]],
    *,
    residual_floor: float,
) -> dict[str, Any]:
    """Fit ``residual = normalization * q_max**exponent`` above a floor."""

    residual_floor = float(residual_floor)
    if residual_floor <= 0.0:
        raise ValueError("residual_floor must be positive")
    included = [
        row
        for row in rows
        if float(row["raw_residual"]) > residual_floor
    ]
    if len(included) < 2:
        raise ValueError("at least two residuals above the floor are required")
    log_q = [math.log(float(row["q_max"])) for row in included]
    log_residual = [
        math.log(float(row["raw_residual"])) for row in included
    ]
    mean_q = math.fsum(log_q) / len(log_q)
    mean_residual = math.fsum(log_residual) / len(log_residual)
    denominator = math.fsum((value - mean_q) ** 2 for value in log_q)
    if denominator == 0.0:
        raise ValueError("the included q_max values are not distinct")
    exponent = (
        math.fsum(
            (q_value - mean_q) * (residual - mean_residual)
            for q_value, residual in zip(log_q, log_residual)
        )
        / denominator
    )
    log_normalization = mean_residual - exponent * mean_q
    return {
        "residual_floor_exclusive": residual_floor,
        "included_point_count": len(included),
        "included_points": [str(row["name"]) for row in included],
        "exponent": exponent,
        "normalization": math.exp(log_normalization),
        "minimum_q_max": min(float(row["q_max"]) for row in included),
        "maximum_q_max": max(float(row["q_max"]) for row in included),
    }


def _superscript_integer(value: int) -> str:
    digits = str(value).translate(
        str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")
    )
    return f"10{digits}"


class _SVG:
    """Small dependency-free SVG writer for this fixed diagnostic."""

    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self.items = [
            (
                f'<svg xmlns="http://www.w3.org/2000/svg" '
                f'width="{width}" height="{height}" '
                f'viewBox="0 0 {width} {height}">'
            ),
            "<style>"
            "text{font-family:-apple-system,BlinkMacSystemFont,"
            "'Segoe UI',sans-serif;fill:#222}"
            ".small{font-size:12px}.label{font-size:14px}"
            ".grid{stroke:#d9dde3;stroke-width:1;opacity:.55}"
            ".axis{stroke:#aeb5bf;stroke-width:1}"
            ".data{fill:none;stroke:#2f91f8;stroke-width:2}"
            ".guide{fill:none;stroke:#777;stroke-width:1.2;"
            "stroke-dasharray:3 4}"
            ".muted{fill:#737982}"
            "</style>",
            '<rect width="100%" height="100%" fill="white"/>',
        ]

    def line(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        *,
        css_class: str,
        dash: str | None = None,
    ) -> None:
        dash_attribute = (
            f' stroke-dasharray="{html.escape(dash)}"' if dash else ""
        )
        self.items.append(
            f'<line x1="{x1:.3f}" y1="{y1:.3f}" '
            f'x2="{x2:.3f}" y2="{y2:.3f}" '
            f'class="{css_class}"{dash_attribute}/>'
        )

    def polyline(
        self,
        points: Sequence[tuple[float, float]],
        *,
        css_class: str,
        dash: str | None = None,
    ) -> None:
        point_text = " ".join(f"{x:.3f},{y:.3f}" for x, y in points)
        dash_attribute = (
            f' stroke-dasharray="{html.escape(dash)}"' if dash else ""
        )
        self.items.append(
            f'<polyline points="{point_text}" class="{css_class}"'
            f'{dash_attribute}/>'
        )

    def circle(self, x: float, y: float, radius: float = 4.0) -> None:
        self.items.append(
            f'<circle cx="{x:.3f}" cy="{y:.3f}" r="{radius:.2f}" '
            'fill="#2f91f8" stroke="white" stroke-width="1"/>'
        )

    def diamond(self, x: float, y: float, radius: float = 5.0) -> None:
        points = (
            f"{x:.3f},{y-radius:.3f} {x+radius:.3f},{y:.3f} "
            f"{x:.3f},{y+radius:.3f} {x-radius:.3f},{y:.3f}"
        )
        self.items.append(
            f'<polygon points="{points}" fill="#2f91f8" '
            'stroke="white" stroke-width="1"/>'
        )

    def text(
        self,
        x: float,
        y: float,
        value: str,
        *,
        css_class: str = "small",
        anchor: str = "start",
        rotate: float | None = None,
        weight: int | None = None,
    ) -> None:
        transform = (
            f' transform="rotate({rotate:g} {x:.3f} {y:.3f})"'
            if rotate is not None
            else ""
        )
        weight_attribute = (
            f' font-weight="{int(weight)}"' if weight is not None else ""
        )
        self.items.append(
            f'<text x="{x:.3f}" y="{y:.3f}" '
            f'text-anchor="{anchor}" class="{css_class}"'
            f'{transform}{weight_attribute}>'
            f"{html.escape(value)}</text>"
        )

    def finish(self) -> str:
        return "\n".join((*self.items, "</svg>", ""))


def _linear_scale(
    domain_minimum: float,
    domain_maximum: float,
    range_minimum: float,
    range_maximum: float,
):
    return lambda value: (
        range_minimum
        + (value - domain_minimum)
        * (range_maximum - range_minimum)
        / (domain_maximum - domain_minimum)
    )


def _draw_log_axes(
    svg: _SVG,
    *,
    left: float,
    right: float,
    top: float,
    bottom: float,
    x_ticks: Sequence[float],
    y_ticks: Sequence[int],
    x_scale,
    y_scale,
    x_label: str,
    y_label: str,
) -> None:
    for tick in x_ticks:
        x = x_scale(tick)
        svg.line(x, top, x, bottom, css_class="grid")
        svg.text(
            x,
            bottom + 20,
            f"10^{tick:g}".replace("-", "−"),
            anchor="middle",
            css_class="small muted",
        )
    for tick in y_ticks:
        y = y_scale(tick)
        svg.line(left, y, right, y, css_class="grid")
        svg.text(
            left - 10,
            y + 4,
            _superscript_integer(tick),
            anchor="end",
            css_class="small muted",
        )
    svg.line(left, bottom, right, bottom, css_class="axis")
    svg.line(left, top, left, bottom, css_class="axis")
    svg.text(
        (left + right) / 2,
        bottom + 46,
        x_label,
        anchor="middle",
        css_class="label",
    )
    svg.text(
        28,
        (top + bottom) / 2,
        y_label,
        anchor="middle",
        css_class="label",
        rotate=-90,
    )


def plot_scan(
    *,
    summary_path: Path,
    config_path: Path,
    output_path: Path,
    study: str = "production",
    level: int | None = None,
    lift_family: str = "lift_x045",
    fit_family: str = "radial_x020",
    fit_residual_floor: float = 1.0e-15,
) -> Path:
    """Render the two-panel highest-cutoff modularity diagnostic as SVG."""

    summary = _load_json(summary_path)
    config = _load_json(config_path)
    digest = validate_summary_config(summary, config)
    configured_levels = tuple(int(value) for value in config["levels"])
    selected_level = max(configured_levels) if level is None else int(level)
    if selected_level not in configured_levels:
        raise ValueError(
            f"twice-level {selected_level} is absent from the configuration"
        )
    if study not in summary["studies"]:
        raise ValueError(f"study {study!r} is absent from the summary")

    families = _scan_rows(
        summary,
        config,
        study=study,
        level=selected_level,
    )
    if lift_family not in families:
        raise ValueError(f"lift family {lift_family!r} is absent")
    if fit_family not in families:
        raise ValueError(f"fit family {fit_family!r} is absent")
    fit = fit_power_law(
        families[fit_family],
        residual_floor=fit_residual_floor,
    )

    if output_path.suffix.lower() != ".svg":
        raise ValueError("the dependency-free plotter writes an .svg file")

    svg = _SVG(900, 780)
    svg.text(82, 27, f"twice-level {selected_level}", weight=600)
    svg.line(178, 23, 198, 23, css_class="data")
    svg.circle(207, 23, 4)
    svg.text(218, 27, "Re τ = 0.20")
    svg.line(310, 23, 330, 23, css_class="data", dash="5 4")
    svg.diamond(339, 23, 5)
    svg.text(350, 27, "Re τ = 0.45")

    left, right = 105.0, 855.0
    top_one, bottom_one = 54.0, 365.0
    x_one = _linear_scale(-2.65, -1.15, left, right)
    y_one = _linear_scale(-3.0, -16.5, top_one, bottom_one)
    _draw_log_axes(
        svg,
        left=left,
        right=right,
        top=top_one,
        bottom=bottom_one,
        x_ticks=(-2.5, -2.25, -2.0, -1.75, -1.5, -1.25),
        y_ticks=(-4, -6, -8, -10, -12, -14, -16),
        x_scale=x_one,
        y_scale=y_one,
        x_label="larger nome qₘₐₓ = max(|q|, |q̃|)",
        y_label="|G(q̃) / (|τ|²ᵈ G(q)) − 1|",
    )
    for family, family_rows in families.items():
        ordered = sorted(family_rows, key=lambda row: row["q_max"])
        points = [
            (
                x_one(math.log10(row["q_max"])),
                y_one(math.log10(row["residual"])),
            )
            for row in ordered
        ]
        svg.polyline(
            points,
            css_class="data",
            dash="5 4" if family == lift_family else None,
        )
        for x, y in points:
            if family == lift_family:
                svg.diamond(x, y)
            else:
                svg.circle(x, y)

    fit_log_q_min = math.log10(float(fit["minimum_q_max"]))
    fit_log_q_max = math.log10(float(fit["maximum_q_max"]))
    fit_log_q = tuple(
        fit_log_q_min
        + index * (fit_log_q_max - fit_log_q_min) / 5.0
        for index in range(6)
    )
    fit_line = [
        (
            x_one(log_q),
            y_one(
                math.log10(float(fit["normalization"]))
                + float(fit["exponent"]) * log_q
            ),
        )
        for log_q in fit_log_q
    ]
    svg.polyline(fit_line, css_class="guide")
    svg.text(
        x_one(-2.58),
        y_one(-3.45),
        (
            f"fit {float(fit['exponent']):.4f} "
            "(δ > 10⁻¹⁵); expected 6.5"
        ),
        css_class="small muted",
    )

    lift_rows = sorted(
        families[lift_family], key=lambda row: row["scan_value"]
    )
    top_two, bottom_two = 440.0, 700.0
    x_two = _linear_scale(0.53, 1.22, left, right)
    y_two = _linear_scale(-3.0, -16.5, top_two, bottom_two)
    for tick in (0.55, 0.65, 0.75, 0.85, 0.95, 1.05, 1.15):
        x = x_two(tick)
        svg.line(x, top_two, x, bottom_two, css_class="grid")
        svg.text(
            x,
            bottom_two + 20,
            f"{tick:.2f}",
            anchor="middle",
            css_class="small muted",
        )
    for tick in (-4, -6, -8, -10, -12, -14, -16):
        y = y_two(tick)
        svg.line(left, y, right, y, css_class="grid")
        svg.text(
            left - 10,
            y + 4,
            _superscript_integer(tick),
            anchor="end",
            css_class="small muted",
        )
    svg.line(left, bottom_two, right, bottom_two, css_class="axis")
    svg.line(left, top_two, left, bottom_two, css_class="axis")
    lift_points = [
        (
            x_two(row["scan_value"]),
            y_two(math.log10(row["residual"])),
        )
        for row in lift_rows
    ]
    svg.polyline(lift_points, css_class="data")
    for x, y in lift_points:
        svg.circle(x, y)

    lift_real_part = lift_rows[0]["tau"].real
    boundary_squared = 2.0 * lift_real_part - lift_real_part**2
    if boundary_squared > 0.0:
        boundary = math.sqrt(boundary_squared)
        boundary_x = x_two(boundary)
        svg.line(
            boundary_x,
            top_two,
            boundary_x,
            bottom_two,
            css_class="guide",
            dash="4 4",
        )
        svg.text(
            boundary_x - 8,
            top_two + 16,
            "q̃ lift −1",
            anchor="end",
            css_class="small muted",
        )
        svg.text(
            boundary_x + 8,
            top_two + 16,
            "q̃ lift +1",
            css_class="small muted",
        )
    svg.text(
        (left + right) / 2,
        bottom_two + 46,
        f"Im τ at fixed Re τ = {lift_real_part:.2f}",
        anchor="middle",
        css_class="label",
    )
    svg.text(
        28,
        (top_two + bottom_two) / 2,
        "modular residual",
        anchor="middle",
        css_class="label",
        rotate=-90,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(svg.finish())
    fit_record = {
        "schema_version": 1,
        "summary": _repository_relative(summary_path),
        "summary_sha256": hashlib.sha256(
            summary_path.read_bytes()
        ).hexdigest(),
        "configuration": _repository_relative(config_path),
        "config_sha256": digest,
        "study": study,
        "twice_level": selected_level,
        "family": fit_family,
        **fit,
        "expected_first_omitted_exponent": 0.5 * (selected_level + 1),
    }
    fit_path = output_path.with_name(f"{output_path.stem}.fit.json")
    fit_path.write_text(
        json.dumps(fit_record, indent=2, sort_keys=True) + "\n"
    )
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary",
        type=Path,
        default=DEFAULT_RUN / "summary.json",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_RUN / "config.snapshot.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_RUN / "torus_modularity_q_scan.svg",
    )
    parser.add_argument("--study", default="production")
    parser.add_argument("--level", type=int)
    parser.add_argument("--lift-family", default="lift_x045")
    parser.add_argument("--fit-family", default="radial_x020")
    parser.add_argument("--fit-residual-floor", type=float, default=1.0e-15)
    arguments = parser.parse_args()
    output = plot_scan(
        summary_path=arguments.summary,
        config_path=arguments.config,
        output_path=arguments.output,
        study=arguments.study,
        level=arguments.level,
        lift_family=arguments.lift_family,
        fit_family=arguments.fit_family,
        fit_residual_floor=arguments.fit_residual_floor,
    )
    fit_path = output.with_name(f"{output.stem}.fit.json")
    print(output)
    print(fit_path)
    fit = _load_json(fit_path)
    print(
        f"fit_exponent={fit['exponent']:.12g} "
        f"(residual>{fit['residual_floor_exclusive']:.1e}, "
        f"points={fit['included_point_count']})"
    )


if __name__ == "__main__":
    main()
