#!/usr/bin/env python3
"""Plot the two channels of the NS-tilde/R torus modularity test."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT.parent / "Data Set"
DEFAULT_RUN = (
    DATA_ROOT
    / "results"
    / "type0b_torus_modular_cluster"
    / "cannon_ns_tilde_r_qscan_20260723_v1"
)
DEFAULT_SUMMARY = DEFAULT_RUN / "summary.json"
DEFAULT_CONFIG = (
    ROOT / "config" / "type0b_torus_ns_tilde_r_q_scan_cluster.json"
)
DEFAULT_OUTPUT = DEFAULT_RUN / "torus_ns_tilde_r_tau_scan.svg"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def config_sha256(payload: Mapping[str, Any]) -> str:
    """Match the cluster driver's canonical configuration digest."""

    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def validate_summary_config(
    summary: Mapping[str, Any],
    config: Mapping[str, Any],
) -> str:
    expected = config_sha256(config)
    observed = str(summary.get("config_sha256", ""))
    if observed != expected:
        raise ValueError(
            "summary/configuration digest mismatch: "
            f"summary={observed or '<missing>'}, config={expected}"
        )
    return expected


def scan_rows(
    summary: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    study: str,
    level: int,
) -> list[dict[str, float]]:
    reduced_taus = summary["studies"][study]["taus"]
    rows: list[dict[str, float]] = []
    for entry in config["taus"]:
        name = str(entry["name"])
        reduced = reduced_taus[name]
        reduced_level = reduced["levels"][str(level)]
        expected_ratio = float(reduced_level["expected_ratio"])
        direct = float(reduced_level["value_q"][0])
        transformed = float(reduced_level["value_q_tilde"][0])
        rows.append(
            {
                "re_tau": float(reduced["tau"][0]),
                "im_tau": float(reduced["tau"][1]),
                "direct": direct,
                "transformed": transformed / expected_ratio,
                "residual": float(reduced_level["relative_error_abs"]),
            }
        )
    return sorted(rows, key=lambda row: row["im_tau"])


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


def _polyline(points: list[tuple[float, float]], css_class: str) -> str:
    coordinates = " ".join(f"{x:.3f},{y:.3f}" for x, y in points)
    return f'<polyline points="{coordinates}" class="{css_class}"/>'


def _text(
    x: float,
    y: float,
    value: str,
    *,
    css_class: str = "tick",
    anchor: str = "start",
    rotate: float | None = None,
) -> str:
    transform = (
        f' transform="rotate({rotate:g} {x:.3f} {y:.3f})"'
        if rotate is not None
        else ""
    )
    return (
        f'<text x="{x:.3f}" y="{y:.3f}" text-anchor="{anchor}" '
        f'class="{css_class}"{transform}>{html.escape(value)}</text>'
    )


def plot_scan(
    *,
    summary_path: Path,
    config_path: Path,
    output_path: Path,
    study: str = "production",
    level: int | None = None,
) -> Path:
    summary = _load_json(summary_path)
    config = _load_json(config_path)
    validate_summary_config(summary, config)

    configured_levels = tuple(int(value) for value in config["levels"])
    selected_level = max(configured_levels) if level is None else int(level)
    if selected_level not in configured_levels:
        raise ValueError(
            f"twice-level {selected_level} is absent from the configuration"
        )
    if study not in summary["studies"]:
        raise ValueError(f"study {study!r} is absent from the summary")

    rows = scan_rows(
        summary,
        config,
        study=study,
        level=selected_level,
    )
    re_tau_values = {round(row["re_tau"], 14) for row in rows}
    if len(re_tau_values) != 1:
        raise ValueError("the tau scan must keep Re(tau) fixed")

    im_tau = [row["im_tau"] for row in rows]
    direct = [row["direct"] for row in rows]
    transformed = [row["transformed"] for row in rows]
    residual = [max(row["residual"], 1.0e-16) for row in rows]
    if output_path.suffix.lower() != ".svg":
        raise ValueError("output path must end in .svg")

    width, height = 960, 720
    left, right = 112.0, 925.0
    upper_top, upper_bottom = 72.0, 408.0
    lower_top, lower_bottom = 474.0, 655.0
    x_scale = _linear_scale(min(im_tau), max(im_tau), left, right)
    value_min = min(direct + transformed)
    value_max = max(direct + transformed)
    value_padding = 0.06 * (value_max - value_min)
    y_value = _linear_scale(
        value_min - value_padding,
        value_max + value_padding,
        upper_bottom,
        upper_top,
    )
    log_residual = [math.log10(value) for value in residual]
    y_error = _linear_scale(-16.0, -7.0, lower_bottom, lower_top)

    items = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
            f'height="{height}" viewBox="0 0 {width} {height}">'
        ),
        "<title>NS-tilde/R torus modularity scan</title>",
        (
            "<desc>The direct NS-tilde torus one-point function and the "
            "weight-corrected modular R-channel result overlap over the tau "
            "scan. The lower panel shows their relative residual.</desc>"
        ),
        (
            "<style>"
            "text{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;"
            "fill:#222}.tick{font-size:13px}.label{font-size:15px}.note{font-size:14px;"
            "fill:#59616b}.grid{stroke:#d9dde3;stroke-width:1}.axis{stroke:#8d96a2;"
            "stroke-width:1}.direct{fill:none;stroke:#1769aa;stroke-width:2.2}"
            ".transformed{fill:none;stroke:#d1495b;stroke-width:1.8;"
            "stroke-dasharray:7 5}.residual{fill:none;stroke:#5f6368;"
            "stroke-width:1.8}.direct-mark{fill:#1769aa;stroke:white;stroke-width:1}"
            ".transformed-mark{stroke:#d1495b;stroke-width:1.8}.residual-mark{"
            "fill:#5f6368;stroke:white;stroke-width:1}</style>"
        ),
        '<rect width="100%" height="100%" fill="white"/>',
    ]

    x_ticks = [0.45, 0.60, 0.75, 0.90, 1.05, 1.20, 1.40]
    value_ticks = [0.045, 0.050, 0.055, 0.060, 0.065]
    error_ticks = [-16, -14, -12, -10, -8]
    for tick in x_ticks:
        x = x_scale(tick)
        items.append(
            f'<line x1="{x:.3f}" y1="{upper_top}" x2="{x:.3f}" '
            f'y2="{upper_bottom}" class="grid"/>'
        )
        items.append(
            f'<line x1="{x:.3f}" y1="{lower_top}" x2="{x:.3f}" '
            f'y2="{lower_bottom}" class="grid"/>'
        )
        items.append(
            _text(x, lower_bottom + 23, f"{tick:.2f}", anchor="middle")
        )
    for tick in value_ticks:
        y = y_value(tick)
        items.append(
            f'<line x1="{left}" y1="{y:.3f}" x2="{right}" '
            f'y2="{y:.3f}" class="grid"/>'
        )
        items.append(_text(left - 12, y + 4, f"{tick:.3f}", anchor="end"))
    for tick in error_ticks:
        y = y_error(float(tick))
        items.append(
            f'<line x1="{left}" y1="{y:.3f}" x2="{right}" '
            f'y2="{y:.3f}" class="grid"/>'
        )
        items.append(_text(left - 12, y + 4, f"10^{tick}", anchor="end"))

    for top, bottom in ((upper_top, upper_bottom), (lower_top, lower_bottom)):
        items.append(
            f'<line x1="{left}" y1="{top}" x2="{left}" '
            f'y2="{bottom}" class="axis"/>'
        )
        items.append(
            f'<line x1="{left}" y1="{bottom}" x2="{right}" '
            f'y2="{bottom}" class="axis"/>'
        )

    direct_points = [
        (x_scale(x), y_value(y)) for x, y in zip(im_tau, direct)
    ]
    transformed_points = [
        (x_scale(x), y_value(y)) for x, y in zip(im_tau, transformed)
    ]
    residual_points = [
        (x_scale(x), y_error(y)) for x, y in zip(im_tau, log_residual)
    ]
    items.extend(
        (
            _polyline(direct_points, "direct"),
            _polyline(transformed_points, "transformed"),
            _polyline(residual_points, "residual"),
        )
    )
    for x, y in direct_points:
        items.append(
            f'<circle cx="{x:.3f}" cy="{y:.3f}" r="4" class="direct-mark"/>'
        )
    for x, y in transformed_points:
        items.append(
            f'<line x1="{x-4:.3f}" y1="{y-4:.3f}" x2="{x+4:.3f}" '
            f'y2="{y+4:.3f}" class="transformed-mark"/>'
        )
        items.append(
            f'<line x1="{x-4:.3f}" y1="{y+4:.3f}" x2="{x+4:.3f}" '
            f'y2="{y-4:.3f}" class="transformed-mark"/>'
        )
    for x, y in residual_points:
        items.append(
            f'<circle cx="{x:.3f}" cy="{y:.3f}" r="3.5" '
            'class="residual-mark"/>'
        )

    items.extend(
        (
            '<line x1="125" y1="28" x2="157" y2="28" class="direct"/>',
            '<circle cx="141" cy="28" r="4" class="direct-mark"/>',
            _text(168, 33, "direct G_NS-tilde(tau)", css_class="note"),
            (
                '<line x1="410" y1="28" x2="442" y2="28" '
                'class="transformed"/>'
            ),
            (
                '<line x1="422" y1="24" x2="430" y2="32" '
                'class="transformed-mark"/>'
            ),
            (
                '<line x1="422" y1="32" x2="430" y2="24" '
                'class="transformed-mark"/>'
            ),
            _text(
                453,
                33,
                "weight-corrected G_R(-1/tau)",
                css_class="note",
            ),
            _text(
                126,
                upper_bottom - 15,
                (
                    f"Re(tau) = {next(iter(re_tau_values)):.2f}; "
                    f"twice-level {selected_level}"
                ),
                css_class="note",
            ),
            _text(
                28,
                (upper_top + upper_bottom) / 2,
                "torus one-point function",
                css_class="label",
                anchor="middle",
                rotate=-90,
            ),
            _text(
                28,
                (lower_top + lower_bottom) / 2,
                "relative residual",
                css_class="label",
                anchor="middle",
                rotate=-90,
            ),
            _text(
                (left + right) / 2,
                height - 18,
                "Im(tau)",
                css_class="label",
                anchor="middle",
            ),
            "</svg>",
            "",
        )
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(items))
    return output_path


def plot_scan_pdf(
    *,
    summary_path: Path,
    config_path: Path,
    output_path: Path,
    study: str = "production",
    level: int | None = None,
) -> Path:
    """Render the same diagnostic as a vector PDF for LaTeX inclusion."""

    from reportlab.lib.colors import HexColor, white
    from reportlab.pdfgen import canvas

    summary = _load_json(summary_path)
    config = _load_json(config_path)
    validate_summary_config(summary, config)
    configured_levels = tuple(int(value) for value in config["levels"])
    selected_level = max(configured_levels) if level is None else int(level)
    if selected_level not in configured_levels:
        raise ValueError(
            f"twice-level {selected_level} is absent from the configuration"
        )
    if study not in summary["studies"]:
        raise ValueError(f"study {study!r} is absent from the summary")

    rows = scan_rows(
        summary,
        config,
        study=study,
        level=selected_level,
    )
    re_tau_values = {round(row["re_tau"], 14) for row in rows}
    if len(re_tau_values) != 1:
        raise ValueError("the tau scan must keep Re(tau) fixed")
    if output_path.suffix.lower() != ".pdf":
        raise ValueError("PDF output path must end in .pdf")

    im_tau = [row["im_tau"] for row in rows]
    direct = [row["direct"] for row in rows]
    transformed = [row["transformed"] for row in rows]
    residual = [max(row["residual"], 1.0e-16) for row in rows]

    width, height = 720.0, 540.0
    left, right = 78.0, 704.0
    upper_bottom, upper_top = 242.0, 468.0
    lower_bottom, lower_top = 48.0, 174.0
    x_scale = _linear_scale(min(im_tau), max(im_tau), left, right)
    value_min = min(direct + transformed)
    value_max = max(direct + transformed)
    value_padding = 0.06 * (value_max - value_min)
    y_value = _linear_scale(
        value_min - value_padding,
        value_max + value_padding,
        upper_bottom,
        upper_top,
    )
    y_error = _linear_scale(-16.0, -7.0, lower_bottom, lower_top)

    direct_color = HexColor("#1769aa")
    transformed_color = HexColor("#d1495b")
    residual_color = HexColor("#5f6368")
    grid_color = HexColor("#d9dde3")
    axis_color = HexColor("#8d96a2")
    text_color = HexColor("#222222")
    muted_color = HexColor("#59616b")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pdf = canvas.Canvas(str(output_path), pagesize=(width, height))
    pdf.setTitle("NS-tilde/R torus modularity scan")
    pdf.setAuthor("StringMC project")
    pdf.setFillColor(white)
    pdf.rect(0, 0, width, height, stroke=0, fill=1)

    def draw_text(
        x: float,
        y: float,
        value: str,
        *,
        size: float = 10.5,
        color=muted_color,
        align: str = "left",
    ) -> None:
        pdf.setFont("Helvetica", size)
        pdf.setFillColor(color)
        if align == "right":
            pdf.drawRightString(x, y, value)
        elif align == "center":
            pdf.drawCentredString(x, y, value)
        else:
            pdf.drawString(x, y, value)

    x_ticks = [0.45, 0.60, 0.75, 0.90, 1.05, 1.20, 1.40]
    value_ticks = [0.045, 0.050, 0.055, 0.060, 0.065]
    error_ticks = [-16, -14, -12, -10, -8]
    pdf.setLineWidth(0.65)
    pdf.setStrokeColor(grid_color)
    for tick in x_ticks:
        x = x_scale(tick)
        pdf.line(x, upper_bottom, x, upper_top)
        pdf.line(x, lower_bottom, x, lower_top)
        draw_text(x, lower_bottom - 17, f"{tick:.2f}", align="center")
    for tick in value_ticks:
        y = y_value(tick)
        pdf.line(left, y, right, y)
        draw_text(left - 8, y - 3, f"{tick:.3f}", align="right")
    for tick in error_ticks:
        y = y_error(float(tick))
        pdf.line(left, y, right, y)
        draw_text(left - 8, y - 3, f"1e{tick}", align="right")

    pdf.setStrokeColor(axis_color)
    pdf.setLineWidth(0.8)
    for bottom, top in (
        (upper_bottom, upper_top),
        (lower_bottom, lower_top),
    ):
        pdf.line(left, bottom, right, bottom)
        pdf.line(left, bottom, left, top)

    def draw_polyline(
        points: list[tuple[float, float]],
        *,
        color,
        width_value: float,
        dash: tuple[float, ...] | None = None,
    ) -> None:
        pdf.setStrokeColor(color)
        pdf.setLineWidth(width_value)
        pdf.setDash(dash or ())
        path = pdf.beginPath()
        path.moveTo(*points[0])
        for point in points[1:]:
            path.lineTo(*point)
        pdf.drawPath(path, stroke=1, fill=0)
        pdf.setDash()

    direct_points = [
        (x_scale(x), y_value(y)) for x, y in zip(im_tau, direct)
    ]
    transformed_points = [
        (x_scale(x), y_value(y)) for x, y in zip(im_tau, transformed)
    ]
    residual_points = [
        (x_scale(x), y_error(math.log10(y)))
        for x, y in zip(im_tau, residual)
    ]
    draw_polyline(
        direct_points,
        color=direct_color,
        width_value=1.5,
    )
    pdf.setFillColor(direct_color)
    pdf.setStrokeColor(white)
    pdf.setLineWidth(0.65)
    for x, y in direct_points:
        pdf.circle(x, y, 2.7, stroke=1, fill=1)

    draw_polyline(
        transformed_points,
        color=transformed_color,
        width_value=1.25,
        dash=(5.0, 3.5),
    )
    pdf.setStrokeColor(transformed_color)
    pdf.setLineWidth(1.25)
    for x, y in transformed_points:
        pdf.line(x - 2.8, y - 2.8, x + 2.8, y + 2.8)
        pdf.line(x - 2.8, y + 2.8, x + 2.8, y - 2.8)

    draw_polyline(
        residual_points,
        color=residual_color,
        width_value=1.25,
    )
    pdf.setFillColor(residual_color)
    pdf.setStrokeColor(white)
    pdf.setLineWidth(0.65)
    for x, y in residual_points:
        pdf.circle(x, y, 2.5, stroke=1, fill=1)

    pdf.setStrokeColor(direct_color)
    pdf.setLineWidth(1.5)
    pdf.line(92, 513, 120, 513)
    pdf.setFillColor(direct_color)
    pdf.setStrokeColor(white)
    pdf.circle(106, 513, 2.7, stroke=1, fill=1)
    draw_text(
        130,
        509.5,
        "direct G_NStilde(tau)",
        size=12.0,
        color=muted_color,
    )

    pdf.setStrokeColor(transformed_color)
    pdf.setLineWidth(1.25)
    pdf.setDash(5.0, 3.5)
    pdf.line(350, 513, 378, 513)
    pdf.setDash()
    pdf.line(361.2, 509.2, 368.8, 516.8)
    pdf.line(361.2, 516.8, 368.8, 509.2)
    draw_text(
        388,
        509.5,
        "|tau|^(-2d) G_R(-1/tau)",
        size=12.0,
        color=muted_color,
    )

    draw_text(
        left + 9,
        upper_bottom + 10,
        (
            f"Re(tau) = {next(iter(re_tau_values)):.2f}; "
            f"twice-level {selected_level}"
        ),
        size=11.0,
        color=muted_color,
    )
    draw_text(
        (left + right) / 2,
        11,
        "Im(tau)",
        size=12.5,
        color=text_color,
        align="center",
    )
    pdf.saveState()
    pdf.translate(19, (upper_bottom + upper_top) / 2)
    pdf.rotate(90)
    draw_text(
        0,
        0,
        "torus one-point function",
        size=12.5,
        color=text_color,
        align="center",
    )
    pdf.restoreState()
    pdf.saveState()
    pdf.translate(19, (lower_bottom + lower_top) / 2)
    pdf.rotate(90)
    draw_text(
        0,
        0,
        "relative residual",
        size=12.5,
        color=text_color,
        align="center",
    )
    pdf.restoreState()
    pdf.showPage()
    pdf.save()
    return output_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--pdf-output",
        type=Path,
        default=None,
        help="also render a vector PDF at this path",
    )
    parser.add_argument("--study", default="production")
    parser.add_argument("--level", type=int)
    return parser


def main() -> None:
    args = _parser().parse_args()
    output_path = plot_scan(
        summary_path=args.summary,
        config_path=args.config,
        output_path=args.output,
        study=args.study,
        level=args.level,
    )
    print(output_path)
    if args.pdf_output is not None:
        pdf_path = plot_scan_pdf(
            summary_path=args.summary,
            config_path=args.config,
            output_path=args.pdf_output,
            study=args.study,
            level=args.level,
        )
        print(pdf_path)


if __name__ == "__main__":
    main()
