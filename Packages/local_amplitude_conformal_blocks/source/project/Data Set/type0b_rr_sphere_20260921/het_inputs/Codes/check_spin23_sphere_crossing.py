#!/usr/bin/env python3
"""Numerically test crossing of the momentum-integrated NS sphere correlator.

The comparison is deliberately made only after the internal Liouville-momentum
integral.  A block at fixed internal momentum is not crossing invariant.

With the external ordering ``(4,3,2,1) = (infinity,1,z,0)``, this script tests

    G_s(z,zbar; 1,2,3,4) = G_t(1-z,1-zbar; 3,2,1,4),

where both sides include the even and odd NS structure-constant products and
the spectral measure ``dP/pi``.  The central-charge recursion implementation
under review is imported unchanged from the production four-vector bundle.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np


CODE_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = CODE_DIR.parent
FIT_BUNDLE_DIR = CODE_DIR / "heterotic_so23_1to3_vvvv_fit_bundle"
DATA_DIR = WORKSPACE_DIR / "Data Set" / "crossing_equation"
NOTE_DIR = WORKSPACE_DIR / "Machine Note"

if str(FIT_BUNDLE_DIR) not in sys.path:
    sys.path.insert(0, str(FIT_BUNDLE_DIR))

import heterotic_so23_1to3 as reference  # noqa: E402
import heterotic_so23_1to3_fast as fast  # noqa: E402


DEFAULT_MOMENTA = (0.20, 0.35, 0.50, 0.65)
DEFAULT_Q_ORDERS = (3, 5, 7, 9)
DEFAULT_DENSE_SEGMENTS = (20, 23, 30, 23, 20, 18, 18)


def external_weight(momentum: complex) -> complex:
    """Return the b=1 NS continuum weight h=(1+p^2)/2."""
    return (1.0 + complex(momentum) ** 2) / 2.0


def evaluate_channel(data: Sequence[Any], z: np.ndarray) -> np.ndarray:
    """Evaluate one already assembled spectral channel at ``(z, zbar)``.

    The external weights are intentionally unchanged in the antiholomorphic
    factor.  Only the coordinate is conjugated, as required by the convention
    in the implementation note.
    """
    z = np.asarray(z, dtype=complex)
    zbar = np.conjugate(z)
    result = np.zeros_like(z, dtype=complex)

    for kernel in data:
        even = kernel.primary.value(z, "e")
        odd = kernel.primary.value(z, "o")
        evenbar = kernel.primary.value(zbar, "e")
        oddbar = kernel.primary.value(zbar, "o")
        result += kernel.weight * (
            kernel.even_structure * even * evenbar
            + kernel.odd_structure * odd * oddbar
        )
    return result


def assemble_channels(
    momenta: Sequence[complex],
    q_order: int,
    p_nodes: int | str | Sequence[int],
    p_max: float,
    caches: dict[str, dict[Any, Any]],
    *, series_parameter: str = "elliptic_nome",
) -> tuple[Sequence[Any], Sequence[Any], dict[str, Any]]:
    """Assemble direct and crossed spectral data with production routines."""
    diagnostics: dict[str, Any] = {}
    common = dict(
        series_parameter=series_parameter,
        q_order=int(q_order),
        p_nodes=p_nodes,
        p_max=float(p_max),
        diagnostics=diagnostics,
        structure_cache=caches["structure"],
        block_cache=caches["block"],
        elliptic_cache=caches["elliptic"],
    )
    s_channel = fast.build_s_channel_data(momenta, **common)

    # s-channel builder slots are (1,2,3,4).  The t channel uses
    # x=1-z and the standardized ordering (1',2',3',4')=(3,2,1,4).
    crossed_momenta = (momenta[2], momenta[1], momenta[0], momenta[3])
    t_channel = fast.build_s_channel_data(crossed_momenta, **common)
    return s_channel, t_channel, diagnostics


def crossing_rows(
    *,
    label: str,
    q_order: int,
    z: np.ndarray,
    s_channel: Sequence[Any],
    t_channel: Sequence[Any],
) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate one crossing profile and return serializable rows."""
    gs = evaluate_channel(s_channel, z)
    gt = evaluate_channel(t_channel, 1.0 - z)
    difference = np.abs(gs - gt)
    scale = np.maximum(np.maximum(np.abs(gs), np.abs(gt)), 1.0e-300)
    relative = difference / scale
    note_scaled = difference / np.maximum(1.0, scale)
    q_s = np.asarray(z, dtype=complex)
    q_t = 1.0 - q_s
    representation=s_channel[0].primary.series_parameter
    if representation == "elliptic_nome":
        q_s=reference.elliptic_conversion.nome_geometry(q_s)[0]
        q_t=reference.elliptic_conversion.nome_geometry(q_t)[0]

    rows: list[dict[str, Any]] = []
    for index, value in enumerate(z):
        rows.append(
            {
                "path": label,
                "q_order": int(q_order),
                "series_parameter": representation,
                "z_real": float(value.real),
                "z_imag": float(value.imag),
                "q_s_abs": float(abs(q_s[index])),
                "q_t_abs": float(abs(q_t[index])),
                "g_s_real": float(gs[index].real),
                "g_s_imag": float(gs[index].imag),
                "g_t_real": float(gt[index].real),
                "g_t_imag": float(gt[index].imag),
                "absolute_difference": float(difference[index]),
                "relative_crossing_residual": float(relative[index]),
                "note_scaled_residual": float(note_scaled[index]),
            }
        )
    return rows, gs, gt, relative


def profile_summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    relative = np.asarray(
        [row["relative_crossing_residual"] for row in rows], dtype=float
    )
    absolute = np.asarray([row["absolute_difference"] for row in rows], dtype=float)
    worst = int(np.argmax(relative))
    return {
        "points": int(len(rows)),
        "maximum_relative_residual": float(relative[worst]),
        "median_relative_residual": float(np.median(relative)),
        "maximum_absolute_difference": float(np.max(absolute)),
        "worst_z": {
            "real": float(rows[worst]["z_real"]),
            "imag": float(rows[worst]["z_imag"]),
        },
    }


def complex_control_summary(
    baseline: np.ndarray, control: np.ndarray
) -> dict[str, float]:
    delta = np.abs(control - baseline)
    scale = np.maximum(np.maximum(np.abs(control), np.abs(baseline)), 1.0e-300)
    relative = delta / scale
    return {
        "maximum_relative_change": float(np.max(relative)),
        "median_relative_change": float(np.median(relative)),
        "maximum_absolute_change": float(np.max(delta)),
    }


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_figure(
    path_base: Path,
    real_profiles: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]],
    real_z: np.ndarray,
    complex_profile: tuple[np.ndarray, np.ndarray, np.ndarray],
    complex_z: np.ndarray,
) -> None:
    """Write a dependency-free SVG of the crossing profiles versus z."""
    highest_order = max(real_profiles)
    gs, gt, _ = real_profiles[highest_order]
    width, height = 900.0, 800.0
    left, right = 102.0, 858.0
    top_y0, top_y1 = 110.0, 365.0
    bottom_y0, bottom_y1 = 475.0, 730.0
    x_min, x_max = float(real_z.real.min()), float(real_z.real.max())

    def x_map(value: float) -> float:
        return left + (float(value) - x_min) * (right - left) / (x_max - x_min)

    correlator_values = np.concatenate([gs.real, gt.real])
    g_min, g_max = map(float, (correlator_values.min(), correlator_values.max()))
    g_padding = 0.08 * (g_max - g_min)
    g_min -= g_padding
    g_max += g_padding

    def g_map(value: float) -> float:
        return top_y1 - (float(value) - g_min) * (top_y1 - top_y0) / (g_max - g_min)

    all_residuals = [profile[2] for profile in real_profiles.values()]
    all_residuals.append(complex_profile[2])
    positive_residuals = np.concatenate(all_residuals)
    positive_residuals = positive_residuals[positive_residuals > 0]
    log_min = min(-12.5, math.floor(float(np.log10(positive_residuals.min()))))
    log_max = max(-5.0, math.ceil(float(np.log10(positive_residuals.max()))))

    def residual_map(value: float) -> float:
        exponent = math.log10(max(float(value), 10.0 ** log_min))
        return bottom_y1 - (exponent - log_min) * (bottom_y1 - bottom_y0) / (
            log_max - log_min
        )

    def polyline(x_values: np.ndarray, y_values: np.ndarray, mapper: Any) -> str:
        return " ".join(
            f"{x_map(float(x)):.2f},{mapper(float(y)):.2f}"
            for x, y in zip(x_values, y_values)
        )

    elements: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width:.0f} {height:.0f}" '
        'role="img" aria-labelledby="title description">',
        '<title id="title">Spin(23) NS sphere crossing test</title>',
        '<desc id="description">Direct and crossed momentum-integrated correlators, '
        'with logarithmic crossing residuals at several recursion orders.</desc>',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;fill:#202124}'
        '.axis{stroke:#4b4b4b;stroke-width:1}.grid{stroke:#d9dde3;stroke-width:1}'
        '.tick{font-size:12px}.label{font-size:14px}.title{font-size:18px;font-weight:600}'
        '.legend{font-size:12px}</style>',
        f'<text class="title" x="{width/2:.1f}" y="30" text-anchor="middle">'
        f'Spin(23) NS sphere crossing at c=27/2 (recursion order {highest_order})</text>',
        f'<text class="label" x="{width/2:.1f}" y="54" text-anchor="middle">'
        'elliptic nome |qhat(z)| (top labels)</text>',
    ]

    x_ticks = np.linspace(x_min, x_max, 5)
    q_tick_values = np.abs(reference.elliptic_conversion.nome_geometry(x_ticks)[0])
    for x_tick, q_tick in zip(x_ticks, q_tick_values):
        x_coord = x_map(float(x_tick))
        elements.extend(
            [
                f'<line class="grid" x1="{x_coord:.2f}" y1="{top_y0}" '
                f'x2="{x_coord:.2f}" y2="{top_y1}"/>',
                f'<line class="grid" x1="{x_coord:.2f}" y1="{bottom_y0}" '
                f'x2="{x_coord:.2f}" y2="{bottom_y1}"/>',
                f'<text class="tick" x="{x_coord:.2f}" y="88" text-anchor="middle">'
                f'{abs(q_tick):.4f}</text>',
                f'<text class="tick" x="{x_coord:.2f}" y="750" text-anchor="middle">'
                f'{x_tick:.2f}</text>',
            ]
        )

    for g_tick in np.linspace(g_min, g_max, 5):
        y_coord = g_map(float(g_tick))
        elements.extend(
            [
                f'<line class="grid" x1="{left}" y1="{y_coord:.2f}" '
                f'x2="{right}" y2="{y_coord:.2f}"/>',
                f'<text class="tick" x="{left-10}" y="{y_coord+4:.2f}" '
                f'text-anchor="end">{g_tick:.4g}</text>',
            ]
        )

    for exponent in range(int(log_min), int(log_max) + 1):
        y_coord = residual_map(10.0 ** exponent)
        elements.extend(
            [
                f'<line class="grid" x1="{left}" y1="{y_coord:.2f}" '
                f'x2="{right}" y2="{y_coord:.2f}"/>',
                f'<text class="tick" x="{left-10}" y="{y_coord+4:.2f}" '
                f'text-anchor="end">10^{exponent}</text>',
            ]
        )

    elements.extend(
        [
            f'<line class="axis" x1="{left}" y1="{top_y1}" x2="{right}" y2="{top_y1}"/>',
            f'<line class="axis" x1="{left}" y1="{top_y0}" x2="{left}" y2="{top_y1}"/>',
            f'<line class="axis" x1="{left}" y1="{bottom_y1}" x2="{right}" y2="{bottom_y1}"/>',
            f'<line class="axis" x1="{left}" y1="{bottom_y0}" x2="{left}" y2="{bottom_y1}"/>',
            f'<polyline fill="none" stroke="#1f5aa6" stroke-width="3" points="'
            f'{polyline(real_z.real, gs.real, g_map)}"/>',
            f'<polyline fill="none" stroke="#d55e00" stroke-width="2" '
            f'stroke-dasharray="8 5" points="{polyline(real_z.real, gt.real, g_map)}"/>',
            f'<text class="label" transform="translate(28 {(top_y0+top_y1)/2:.1f}) rotate(-90)" '
            'text-anchor="middle">integrated correlator Re G</text>',
            f'<text class="label" transform="translate(28 {(bottom_y0+bottom_y1)/2:.1f}) rotate(-90)" '
            'text-anchor="middle">|Gs-Gt| / max(|Gs|,|Gt|)</text>',
            f'<text class="label" x="{width/2:.1f}" y="782" text-anchor="middle">'
            'cross-ratio x = Re z</text>',
            '<line x1="650" y1="120" x2="690" y2="120" stroke="#1f5aa6" stroke-width="3"/>',
            '<text class="legend" x="698" y="124">Gs(z)</text>',
            '<line x1="750" y1="120" x2="790" y2="120" stroke="#d55e00" '
            'stroke-width="2" stroke-dasharray="8 5"/>',
            '<text class="legend" x="798" y="124">Gt(1-z)</text>',
        ]
    )

    colors = ["#888888", "#56b4e9", "#009e73", "#cc79a7", "#e69f00"]
    legend_x, legend_y = 118.0, 450.0
    for index, (color, q_order) in enumerate(zip(colors, sorted(real_profiles))):
        residual = real_profiles[q_order][2]
        dash = "" if index else ' stroke-dasharray="4 3"'
        elements.append(
            f'<polyline fill="none" stroke="{color}" stroke-width="2"{dash} '
            f'points="{polyline(real_z.real, residual, residual_map)}"/>'
        )
        item_x = legend_x + index * 145.0
        elements.extend(
            [
                f'<line x1="{item_x}" y1="{legend_y}" x2="{item_x+25}" y2="{legend_y}" '
                f'stroke="{color}" stroke-width="2"{dash}/>',
                f'<text class="legend" x="{item_x+31}" y="{legend_y+4}">order {q_order}</text>',
            ]
        )

    elements.extend(
        [
            f'<polyline fill="none" stroke="#0072b2" stroke-width="2.3" '
            f'stroke-dasharray="3 4" points="'
            f'{polyline(complex_z.real, complex_profile[2], residual_map)}"/>',
            f'<line x1="{left}" y1="{residual_map(1.0e-7):.2f}" x2="{right}" '
            f'y2="{residual_map(1.0e-7):.2f}" stroke="#303030" stroke-width="1.2" '
            'stroke-dasharray="7 5"/>',
            f'<text class="legend" x="{right-8}" y="{residual_map(1.0e-7)-6:.2f}" '
            'text-anchor="end">review target 10^-7</text>',
            f'<line x1="{right-188}" y1="{legend_y}" x2="{right-158}" y2="{legend_y}" '
            'stroke="#0072b2" stroke-width="2.3" stroke-dasharray="3 4"/>',
            f'<text class="legend" x="{right-152}" y="{legend_y+4}">z=x+0.12i</text>',
            '</svg>',
        ]
    )

    path_base.parent.mkdir(parents=True, exist_ok=True)
    path_base.with_suffix(".svg").write_text("\n".join(elements), encoding="utf-8")


def write_note(path: Path, summary: dict[str, Any]) -> None:
    real_results = summary["crossing_profiles"]["real_axis"]
    complex_result = summary["crossing_profiles"]["complex_path"]
    controls = summary["momentum_controls"]
    highest = str(max(map(int, real_results)))
    weight_text = ", ".join(
        f"{weight['real']:.8g}" for weight in summary["external_weights"]
    )
    if controls:
        control_text = f"""Increasing the cutoff from `P_max=6` to `P_max=8` changes either channel by at
most `{max(controls['p_max_8_segmented']['s_channel_change_from_baseline']['maximum_relative_change'], controls['p_max_8_segmented']['t_channel_change_from_baseline']['maximum_relative_change']):.6e}`.
Increasing the segmented momentum rule from 124 to 156 nodes changes either
channel by at most
`{max(controls['p_max_6_dense_segmented']['s_channel_change_from_baseline']['maximum_relative_change'], controls['p_max_6_dense_segmented']['t_channel_change_from_baseline']['maximum_relative_change']):.6e}`."""
    else:
        control_text = "Momentum-cutoff and dense-quadrature controls were skipped for this profile."
    q_lines = "\n".join(
        f"- order {order}: maximum relative residual "
        f"`{values['maximum_relative_residual']:.6e}`"
        for order, values in sorted(real_results.items(), key=lambda item: int(item[0]))
    )
    text = f"""# Spin(23) sphere crossing check

This is a machine-generated reviewer check of the central-charge-recursion
implementation used for the genus-zero four-point function.  It compares the
**complete internal-momentum spectral integral**, not an individual fixed-P
block.

## Equation checked

For the note convention `(4,3,2,1)=(infinity,1,z,0)`, the test is

```text
G_s(z,zbar; 1,2,3,4) = G_t(1-z,1-zbar; 3,2,1,4).
```

Each side contains both NS parities, their appropriate products of three-point
structure constants, and `dP/pi`.  The external Liouville momenta are
`{summary['external_momenta']}`; hence the weights are
`[{weight_text}]`.  The unequal values make the channel-routing
test nontrivial.

## Results

{q_lines}

At the highest recursion order ({highest}), the complex path `z=x+0.12 i`
has maximum relative residual
`{complex_result['maximum_relative_residual']:.6e}`.  The stated relative
residual is `|Gs-Gt|/max(|Gs|,|Gt|)`, which is stricter here than the manual's
`max(1,|Gs|,|Gt|)` normalization.

{control_text}

Passing this check validates the primary NS block recursion,
the even/odd structure-constant assembly, and the s/t external routing for the
tested real continuum weights.  It does not by itself validate the external-
descendant Ward-identity assembly or analytic continuation across Liouville
structure-constant poles.

## Important correction to the old pre-flight wording

A fixed-P block or fixed-P kernel is not expected to be crossing invariant.
Crossing is a fusion transformation that mixes the internal spectrum.  The
meaningful numerical equality is the spectral integral above.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--momenta", nargs=4, type=float, default=DEFAULT_MOMENTA)
    parser.add_argument("--q-orders", nargs="+", type=int, default=DEFAULT_Q_ORDERS)
    parser.add_argument("--p-max", type=float, default=6.0)
    parser.add_argument("--z-points", type=int, default=65)
    parser.add_argument("--z-min", type=float, default=0.10)
    parser.add_argument("--z-max", type=float, default=0.90)
    parser.add_argument("--complex-height", type=float, default=0.12)
    parser.add_argument(
        "--skip-controls",
        action="store_true",
        help="Skip p_max=8 and denser segmented-quadrature controls.",
    )
    parser.add_argument(
        "--output-stem", default="spin23_sphere_c_recursion_elliptic_crossing", help="Output basename."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    momenta = tuple(map(float, args.momenta))
    q_orders = tuple(sorted(set(map(int, args.q_orders))))
    real_z = np.linspace(args.z_min, args.z_max, args.z_points).astype(complex)
    complex_z = real_z + 1j * float(args.complex_height)
    caches: dict[str, dict[Any, Any]] = {
        "structure": {},
        "block": {},
        "elliptic": {},
    }

    all_rows: list[dict[str, Any]] = []
    real_profiles: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    profile_summaries: dict[str, Any] = {}
    diagnostics: dict[str, Any] = {}
    highest_channels: tuple[Sequence[Any], Sequence[Any]] | None = None

    for q_order in q_orders:
        print(f"assembling q-order {q_order}", flush=True)
        s_channel, t_channel, order_diagnostics = assemble_channels(
            momenta, q_order, "segmented", args.p_max, caches
        )
        rows, gs, gt, residual = crossing_rows(
            label="real_axis",
            q_order=q_order,
            z=real_z,
            s_channel=s_channel,
            t_channel=t_channel,
        )
        all_rows.extend(rows)
        real_profiles[q_order] = (gs, gt, residual)
        profile_summaries[str(q_order)] = profile_summary(rows)
        diagnostics[str(q_order)] = order_diagnostics
        highest_channels = (s_channel, t_channel)
        print(
            f"  max real-axis residual: {np.max(residual):.6e}", flush=True
        )

    assert highest_channels is not None
    highest_order = max(q_orders)
    complex_rows, complex_gs, complex_gt, complex_residual = crossing_rows(
        label="complex_x_plus_i_height",
        q_order=highest_order,
        z=complex_z,
        s_channel=highest_channels[0],
        t_channel=highest_channels[1],
    )
    all_rows.extend(complex_rows)

    controls: dict[str, Any] = {}
    baseline_gs, baseline_gt, _ = real_profiles[highest_order]
    if not args.skip_controls:
        for label, p_nodes, p_max in (
            ("p_max_8_segmented", "segmented", 8.0),
            ("p_max_6_dense_segmented", DEFAULT_DENSE_SEGMENTS, args.p_max),
        ):
            print(f"assembling control {label}", flush=True)
            control_s, control_t, control_diagnostics = assemble_channels(
                momenta, highest_order, p_nodes, p_max, caches
            )
            control_gs = evaluate_channel(control_s, real_z)
            control_gt = evaluate_channel(control_t, 1.0 - real_z)
            control_residual = np.abs(control_gs - control_gt) / np.maximum(
                np.maximum(np.abs(control_gs), np.abs(control_gt)), 1.0e-300
            )
            controls[label] = {
                "p_max": float(p_max),
                "spectral_nodes": int(len(control_s)),
                "maximum_crossing_residual": float(np.max(control_residual)),
                "s_channel_change_from_baseline": complex_control_summary(
                    baseline_gs, control_gs
                ),
                "t_channel_change_from_baseline": complex_control_summary(
                    baseline_gt, control_gt
                ),
                "backend_diagnostics": control_diagnostics,
            }

    csv_path = DATA_DIR / f"{args.output_stem}.csv"
    json_path = DATA_DIR / f"{args.output_stem}_summary.json"
    figure_base = NOTE_DIR / args.output_stem
    note_path = NOTE_DIR / (
        "SPIN23_SPHERE_CROSSING_CHECK.md"
        if args.output_stem == "spin23_sphere_crossing"
        else f"{args.output_stem.upper()}_CHECK.md"
    )
    write_csv(csv_path, all_rows)

    summary: dict[str, Any] = {
        "status": "pass"
        if max(profile_summaries[str(highest_order)]["maximum_relative_residual"],
               float(np.max(complex_residual))) < 1.0e-7
        else "fail",
        "central_charge": 27.0 / 2.0,
        **reference.elliptic_conversion.representation_metadata("elliptic_nome",highest_order),
        "external_momenta": list(momenta),
        "external_weights": [
            {"real": external_weight(p).real, "imag": external_weight(p).imag}
            for p in momenta
        ],
        "channel_convention": {
            "insertions": "(4,3,2,1)=(infinity,1,z,0)",
            "s_order": [1, 2, 3, 4],
            "t_coordinate": "1-z",
            "t_order": [3, 2, 1, 4],
            "antiholomorphic_rule": "conjugate coordinate, unchanged external weights",
        },
        "residual_definition": "abs(Gs-Gt)/max(abs(Gs),abs(Gt))",
        "review_target": 1.0e-7,
        "baseline_controls": {
            "q_orders": list(q_orders),
            "p_max": float(args.p_max),
            "p_nodes": "segmented (124 total nodes including endpoint rule)",
            "real_z_interval": [float(args.z_min), float(args.z_max)],
            "real_z_points": int(args.z_points),
            "complex_path": f"z=x+{args.complex_height}i",
        },
        "crossing_profiles": {
            "real_axis": profile_summaries,
            "complex_path": profile_summary(complex_rows),
        },
        "momentum_controls": controls,
        "backend_diagnostics": diagnostics,
        "scope": {
            "tested": [
                "primary NS c-recursion",
                "even/odd structure-constant assembly",
                "s/t external-state routing",
                "complete spectral-integral crossing at tested real weights",
            ],
            "not_validated": [
                "external-descendant Ward-identity assembly",
                "analytic continuation across structure-constant pole crossings",
                "the full heterotic moduli integral",
            ],
        },
    }
    json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    make_figure(
        figure_base,
        real_profiles,
        real_z,
        (complex_gs, complex_gt, complex_residual),
        complex_z,
    )
    write_note(note_path, summary)

    print(json.dumps({
        "status": summary["status"],
        "maximum_real_residual": profile_summaries[str(highest_order)][
            "maximum_relative_residual"
        ],
        "maximum_complex_residual": float(np.max(complex_residual)),
        "csv": str(csv_path),
        "summary": str(json_path),
        "figure": str(figure_base.with_suffix('.svg')),
        "note": str(note_path),
    }, indent=2))


if __name__ == "__main__":
    main()
