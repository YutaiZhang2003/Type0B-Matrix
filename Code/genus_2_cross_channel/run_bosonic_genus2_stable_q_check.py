#!/usr/bin/env python3
"""Run the five-point bosonic genus-two locality test at better-conditioned q.

The geometry is selected without looking at any CFT value.  For every point
the script compares

    Q_L = Z_L / Z_X**25

in period-matched theta and glasses plumbing frames.  The default order ladder
separates the Zamolodchikov-recursion and momentum-quadrature movements.  No
agreement is assumed or used as an acceptance criterion.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import os
import time
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DESIGN = ROOT / "Data Set" / "genus2_bosonic_stable_q_fivepoint.json"
DEFAULT_OUT_DIR = ROOT / "Data Set" / "genus2_bosonic_stable_q_check"


def _parse_order(value: str) -> tuple[int, int]:
    pieces = value.split(":")
    if len(pieces) != 2:
        raise argparse.ArgumentTypeError("orders must have RECURSION:MOMENTUM form")
    recursion, momentum = (int(piece) for piece in pieces)
    if recursion < 0 or momentum < 1:
        raise argparse.ArgumentTypeError("recursion must be nonnegative and momentum positive")
    return recursion, momentum


def _matrix(values: Sequence[Sequence[str]]) -> np.ndarray:
    result = np.asarray(
        [[complex(value) for value in row] for row in values],
        dtype=np.complex128,
    )
    if result.shape != (2, 2):
        raise ValueError("period matrices must be 2x2")
    return result


def _symplectic_transform(matrix: np.ndarray, omega: np.ndarray) -> np.ndarray:
    a, b = matrix[:2, :2], matrix[:2, 2:]
    c, d = matrix[2:, :2], matrix[2:, 2:]
    return (a @ omega + b) @ np.linalg.inv(c @ omega + d)


def _transport_characteristic(
    matrix: np.ndarray,
    characteristic: tuple[tuple[int, int], tuple[int, int]],
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Apply the affine Sp(4,Z) action to a binary characteristic."""

    a, b = matrix[:2, :2], matrix[:2, 2:]
    c, d = matrix[2:, :2], matrix[2:, 2:]
    alpha = np.asarray(characteristic[0], dtype=int)
    beta = np.asarray(characteristic[1], dtype=int)
    transported_alpha = (d @ alpha - c @ beta + np.diag(c @ d.T)) % 2
    transported_beta = (-b @ alpha + a @ beta + np.diag(a @ b.T)) % 2
    return (
        tuple(int(value) for value in transported_alpha),
        tuple(int(value) for value in transported_beta),
    )  # type: ignore[return-value]


def _load_design(path: Path) -> dict[str, object]:
    design = json.loads(path.read_text())
    if int(design.get("schema_version", 0)) != 1:
        raise ValueError("unsupported stable-q design schema")
    return design


def preflight_design(design: dict[str, object]) -> list[dict[str, object]]:
    """Validate markings, q conditioning, annuli, and the saved spin ledger."""

    from genus2_hybrid_period_map import plumbing_geometry
    from plumbing_algorithms import schottky_theta_period_matrix_cross_ratio

    policy = dict(design["selection_policy"])  # type: ignore[arg-type]
    minimum_q = float(policy["minimum_abs_q"])
    maximum_q = float(policy["maximum_abs_q"])
    maximum_validation = float(policy["maximum_theta_validation_residual"])
    maximum_stability = float(policy["maximum_theta_map_stability"])
    minimum_margin = float(policy["minimum_disjoint_annulus_margin"])

    unbranched = np.asarray(design["unbranched_symplectic_matrix"], dtype=int)
    complete = np.asarray(design["branch_composed_symplectic_matrix"], dtype=int)
    branch = np.asarray(design["theta_integer_branch"], dtype=int)
    identity = np.eye(2, dtype=int)
    zero = np.zeros((2, 2), dtype=int)
    branch_transform = np.block([[identity, branch], [zero, identity]])
    if not np.array_equal(branch_transform @ unbranched, complete):
        raise AssertionError("integer branch does not compose the saved modular word")
    symplectic_form = np.block([[zero, identity], [-identity, zero]])
    if not np.array_equal(complete.T @ symplectic_form @ complete, symplectic_form):
        raise AssertionError("branch-composed modular matrix is not symplectic")
    transported = _transport_characteristic(complete, ((0, 0), (0, 0)))
    if transported != ((0, 0), (0, 0)):
        raise AssertionError("saved modular map does not preserve the selected NS characteristic")
    spin_contract = dict(design["future_super_reuse_contract"])  # type: ignore[arg-type]
    glasses_lifts = tuple(int(value) for value in spin_contract["glasses_edge_lifts"])
    theta_lifts = tuple(int(value) for value in spin_contract["theta_edge_lifts"])
    if glasses_lifts != (1, 1, 1) or theta_lifts != (-1, 1, 1):
        raise AssertionError("saved same-spin lift ledger changed")
    glasses_beta = tuple(int(sign < 0) for sign in glasses_lifts[:2])
    theta_generator_signs = (
        theta_lifts[0] * theta_lifts[2],
        -theta_lifts[1] * theta_lifts[2],
    )
    theta_beta = tuple(int(sign > 0) for sign in theta_generator_signs)
    if glasses_beta != (0, 0) or theta_beta != (0, 0):
        raise AssertionError("saved edge lifts do not realize [00|00] in both markings")

    reports: list[dict[str, object]] = []
    points = list(design["points"])  # type: ignore[arg-type]
    selected_ids = list(design["selected_ids"])  # type: ignore[arg-type]
    if [point["id"] for point in points] != selected_ids:
        raise AssertionError("point order differs from selected_ids")
    for point in points:
        point_id = str(point["id"])
        glasses_q = tuple(complex(value) for value in point["glasses_q"])
        theta_q = tuple(complex(value) for value in point["theta_q"])
        all_abs_q = [abs(value) for value in (*glasses_q, *theta_q)]
        q_min = min(all_abs_q)
        q_max = max(all_abs_q)
        if q_min < minimum_q or q_max > maximum_q:
            raise AssertionError(
                f"{point_id}: q conditioning gate failed: min={q_min:.3e}, max={q_max:.3e}"
            )
        if not math.isclose(q_min, float(point["q_min_abs"]), rel_tol=2.0e-13):
            raise AssertionError(f"{point_id}: saved q_min_abs changed")
        if not math.isclose(q_max, float(point["q_max_abs"]), rel_tol=2.0e-13):
            raise AssertionError(f"{point_id}: saved q_max_abs changed")

        glasses_omega = _matrix(point["glasses_omega"])
        theta_unbranched = _matrix(point["theta_omega_unbranched"])
        modular_residual = float(
            np.max(np.abs(_symplectic_transform(unbranched, glasses_omega) - theta_unbranched))
        )
        if modular_residual > 5.0e-14:
            raise AssertionError(f"{point_id}: saved modular period relation failed")
        for name, omega in (("glasses", glasses_omega), ("theta", theta_unbranched)):
            if np.max(np.abs(omega - omega.T)) > 5.0e-14:
                raise AssertionError(f"{point_id}: {name} period matrix is not symmetric")
            if np.min(np.linalg.eigvalsh(omega.imag)) <= 0.0:
                raise AssertionError(f"{point_id}: {name} period matrix is not in Siegel space")

        theta_plumbing = schottky_theta_period_matrix_cross_ratio(
            *theta_q,
            max_word_len=6,
        )
        measured_branch = np.rint((theta_plumbing - theta_unbranched).real).astype(int)
        measured_branch = np.rint(0.5 * (measured_branch + measured_branch.T)).astype(int)
        theta_forward_residual = float(
            np.max(np.abs(theta_plumbing - theta_unbranched - measured_branch))
        )
        if not np.array_equal(measured_branch, branch):
            raise AssertionError(f"{point_id}: theta q landed on an unexpected integer branch")
        if theta_forward_residual > 5.0e-9:
            raise AssertionError(f"{point_id}: theta Schottky forward check failed")

        geometry = dict(point["geometry"])
        if float(geometry["theta_validation_residual"]) > maximum_validation:
            raise AssertionError(f"{point_id}: theta validation residual is too large")
        if float(geometry["theta_map_stability"]) > maximum_stability:
            raise AssertionError(f"{point_id}: theta period map is insufficiently stable")
        glasses_annuli = plumbing_geometry("glasses", glasses_q)
        theta_annuli = plumbing_geometry("theta", theta_q)
        if (
            not glasses_annuli.valid
            or not theta_annuli.valid
            or glasses_annuli.minimum_margin < minimum_margin
            or theta_annuli.minimum_margin < minimum_margin
        ):
            raise AssertionError(f"{point_id}: standard plumbing annuli are not safely disjoint")

        reports.append(
            {
                "point_id": point_id,
                "q_min_abs": q_min,
                "q_max_abs": q_max,
                "q_spread": q_max / q_min,
                "modular_period_residual": modular_residual,
                "theta_schottky_forward_residual": theta_forward_residual,
                "theta_integer_branch": measured_branch.tolist(),
                "transported_characteristic": {
                    "alpha": list(transported[0]),
                    "beta": list(transported[1]),
                },
                "glasses_annulus_margin": glasses_annuli.minimum_margin,
                "theta_annulus_margin": theta_annuli.minimum_margin,
            }
        )
    return reports


def _write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _plot(path: Path, rows: Sequence[dict[str, object]]) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/type0b-matplotlib")
    try:
        import matplotlib
    except ModuleNotFoundError:
        _plot_svg_fallback(path.with_suffix(".svg"), rows)
        print("matplotlib is unavailable; wrote the SVG plot only", flush=True)
        return

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(10.0, 6.0), constrained_layout=True)
    order_keys = list(
        dict.fromkeys(
            (int(row["recursion_order"]), int(row["momentum_order"]))
            for row in rows
            if row["status"] == "ok"
        )
    )
    point_ids = list(dict.fromkeys(str(row["point_id"]) for row in rows))
    x_values = np.arange(len(point_ids), dtype=float)
    for recursion, momentum in order_keys:
        by_id = {
            str(row["point_id"]): row
            for row in rows
            if row["status"] == "ok"
            and int(row["recursion_order"]) == recursion
            and int(row["momentum_order"]) == momentum
        }
        axis.plot(
            x_values,
            [
                100.0 * float(by_id[point_id]["relative_difference"])
                if point_id in by_id
                else math.nan
                for point_id in point_ids
            ],
            marker="o",
            linewidth=1.4,
            label=f"R={recursion}, N={momentum}",
        )
    axis.axhline(0.0, color="#555555", linewidth=1.0)
    axis.set_xticks(x_values, point_ids)
    axis.set(
        xlabel="period-matched point",
        ylabel=r"$100\,[Q_L^\theta/Q_L^{\rm gl}-1]$",
        title="Bosonic genus-two locality at better-conditioned plumbing points",
    )
    axis.grid(alpha=0.22)
    axis.legend(frameon=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    fig.savefig(path.with_suffix(".svg"))
    plt.close(fig)


def _plot_svg_fallback(path: Path, rows: Sequence[dict[str, object]]) -> None:
    """Write a compact dependency-free plot when matplotlib is unavailable."""

    point_ids = list(dict.fromkeys(str(row["point_id"]) for row in rows))
    order_keys = list(
        dict.fromkeys(
            (int(row["recursion_order"]), int(row["momentum_order"]))
            for row in rows
            if row["status"] == "ok"
        )
    )
    series: list[tuple[tuple[int, int], list[float | None]]] = []
    finite_values: list[float] = []
    for key in order_keys:
        by_id = {
            str(row["point_id"]): 100.0 * float(row["relative_difference"])
            for row in rows
            if row["status"] == "ok"
            and (int(row["recursion_order"]), int(row["momentum_order"])) == key
        }
        values = [by_id.get(point_id) for point_id in point_ids]
        finite_values.extend(value for value in values if value is not None)
        series.append((key, values))
    if not finite_values:
        return
    width, height = 960.0, 560.0
    left, right, top, bottom = 90.0, 220.0, 60.0, 80.0
    plot_width = width - left - right
    plot_height = height - top - bottom
    y_min = min(0.0, min(finite_values))
    y_max = max(0.0, max(finite_values))
    padding = max(0.05, 0.08 * max(y_max - y_min, 1.0e-12))
    y_min -= padding
    y_max += padding

    def x_coord(index: int) -> float:
        return left + (index + 0.5) * plot_width / max(len(point_ids), 1)

    def y_coord(value: float) -> float:
        return top + (y_max - value) * plot_height / (y_max - y_min)

    colors = ("#176b87", "#c65d37", "#4b8f4b", "#75539b", "#b07b24")
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width:.0f}" height="{height:.0f}" viewBox="0 0 {width:.0f} {height:.0f}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#222}.axis{stroke:#444;stroke-width:1}.grid{stroke:#bbb;stroke-width:1;stroke-dasharray:4 4}</style>',
        f'<text x="{width/2:.1f}" y="28" font-size="18" text-anchor="middle">Bosonic genus-two locality at better-conditioned plumbing points</text>',
        f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_height}"/>',
        f'<line class="axis" x1="{left}" y1="{top+plot_height}" x2="{left+plot_width}" y2="{top+plot_height}"/>',
    ]
    if y_min <= 0.0 <= y_max:
        zero_y = y_coord(0.0)
        elements.append(
            f'<line class="grid" x1="{left}" y1="{zero_y:.2f}" x2="{left+plot_width}" y2="{zero_y:.2f}"/>'
        )
    for tick in range(6):
        value = y_min + tick * (y_max - y_min) / 5.0
        y = y_coord(value)
        elements.extend(
            [
                f'<line class="grid" x1="{left}" y1="{y:.2f}" x2="{left+plot_width}" y2="{y:.2f}"/>',
                f'<text x="{left-10}" y="{y+4:.2f}" font-size="12" text-anchor="end">{value:.3g}</text>',
            ]
        )
    for index, point_id in enumerate(point_ids):
        elements.append(
            f'<text x="{x_coord(index):.2f}" y="{top+plot_height+25:.2f}" font-size="12" text-anchor="middle">{html.escape(point_id)}</text>'
        )
    elements.append(
        f'<text x="20" y="{top+plot_height/2:.2f}" font-size="13" text-anchor="middle" transform="rotate(-90 20 {top+plot_height/2:.2f})">100 [Q_theta/Q_glasses - 1]</text>'
    )
    for series_index, (key, values) in enumerate(series):
        color = colors[series_index % len(colors)]
        points = [
            (x_coord(index), y_coord(value))
            for index, value in enumerate(values)
            if value is not None
        ]
        if points:
            polyline = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
            elements.append(
                f'<polyline points="{polyline}" fill="none" stroke="{color}" stroke-width="2"/>'
            )
            elements.extend(
                f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4" fill="{color}"/>'
                for x, y in points
            )
        legend_y = top + 22.0 * series_index
        elements.extend(
            [
                f'<line x1="{left+plot_width+20}" y1="{legend_y}" x2="{left+plot_width+50}" y2="{legend_y}" stroke="{color}" stroke-width="2"/>',
                f'<text x="{left+plot_width+60}" y="{legend_y+4}" font-size="12">R={key[0]}, N={key[1]}</text>',
            ]
        )
    elements.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(elements) + "\n")


def _evaluate(
    design: dict[str, object],
    *,
    point_ids: set[str] | None,
    orders: Sequence[tuple[int, int]],
    dps: int,
    vacuum_word_length: int,
    vacuum_oscillator_level: int,
    scalar_word_length: int,
    scalar_max_mode: int,
    scalar_tolerance: float,
) -> list[dict[str, object]]:
    from monte_carlo_integrate_genus2_c1 import (
        evaluate_liouville_rescaled,
        evaluate_noncompact_scalar,
    )

    branch = np.asarray(design["theta_integer_branch"], dtype=int)
    rows: list[dict[str, object]] = []
    for point in design["points"]:  # type: ignore[index]
        point_id = str(point["id"])
        if point_ids is not None and point_id not in point_ids:
            continue
        glasses_q = tuple(complex(value) for value in point["glasses_q"])
        theta_q = tuple(complex(value) for value in point["theta_q"])
        glasses_omega = _matrix(point["glasses_omega"])
        theta_omega = _matrix(point["theta_omega_unbranched"]) + branch

        scalar_started = time.time()
        glasses_scalar, glasses_scalar_tail, glasses_primitives = evaluate_noncompact_scalar(
            "glasses",
            glasses_q,
            glasses_omega,
            word_length=scalar_word_length,
            max_mode=scalar_max_mode,
            tolerance=scalar_tolerance,
        )
        theta_scalar, theta_scalar_tail, theta_primitives = evaluate_noncompact_scalar(
            "theta",
            theta_q,
            theta_omega,
            word_length=scalar_word_length,
            max_mode=scalar_max_mode,
            tolerance=scalar_tolerance,
        )
        scalar_seconds = time.time() - scalar_started
        log_zx_glasses = math.log(glasses_scalar)
        log_zx_theta = math.log(theta_scalar)

        for recursion_order, momentum_order in orders:
            started = time.time()
            row: dict[str, object] = {
                "point_id": point_id,
                "status": "failed",
                "error": "",
                "recursion_order": recursion_order,
                "momentum_order": momentum_order,
                "q_min_abs": float(point["q_min_abs"]),
                "q_max_abs": float(point["q_max_abs"]),
                "scalar_seconds": scalar_seconds,
                "glasses_scalar_tail": glasses_scalar_tail,
                "theta_scalar_tail": theta_scalar_tail,
                "glasses_scalar_primitives": glasses_primitives,
                "theta_scalar_primitives": theta_primitives,
            }
            try:
                glasses_liouville = evaluate_liouville_rescaled(
                    "glasses",
                    glasses_q,
                    block_order=recursion_order,
                    quadrature_order=momentum_order,
                    quadrature_scheme="primary-gaussian",
                    dps=dps,
                    vacuum_word_length=vacuum_word_length,
                    vacuum_oscillator_level=vacuum_oscillator_level,
                )
                theta_liouville = evaluate_liouville_rescaled(
                    "theta",
                    theta_q,
                    block_order=recursion_order,
                    quadrature_order=momentum_order,
                    quadrature_scheme="primary-gaussian",
                    dps=dps,
                    vacuum_word_length=vacuum_word_length,
                    vacuum_oscillator_level=vacuum_oscillator_level,
                )
                log_ql_glasses = glasses_liouville.log_partition - 25.0 * log_zx_glasses
                log_ql_theta = theta_liouville.log_partition - 25.0 * log_zx_theta
                log_ratio = log_ql_theta - log_ql_glasses
                ratio = math.exp(log_ratio)
                row.update(
                    {
                        "status": "ok",
                        "log_zl_glasses": glasses_liouville.log_partition,
                        "log_zl_theta": theta_liouville.log_partition,
                        "log_zx_glasses": log_zx_glasses,
                        "log_zx_theta": log_zx_theta,
                        "log_ql_glasses": log_ql_glasses,
                        "log_ql_theta": log_ql_theta,
                        "log_theta_over_glasses": log_ratio,
                        "theta_over_glasses": ratio,
                        "relative_difference": ratio - 1.0,
                    }
                )
            except Exception as exc:  # noqa: BLE001 - retain failed points in the audit.
                row["error"] = f"{type(exc).__name__}: {exc}"
            row["runtime_seconds"] = time.time() - started
            rows.append(row)
            print(
                f"{point_id} R={recursion_order} N={momentum_order} "
                f"status={row['status']} ratio={row.get('theta_over_glasses', math.nan):.10g}",
                flush=True,
            )
    return rows


def _convergence_summary(rows: Sequence[dict[str, object]]) -> dict[str, object]:
    by_point: dict[str, dict[tuple[int, int], dict[str, object]]] = {}
    for row in rows:
        if row["status"] != "ok":
            continue
        by_point.setdefault(str(row["point_id"]), {})[
            (int(row["recursion_order"]), int(row["momentum_order"]))
        ] = row
    output: dict[str, object] = {}
    for point_id, values in by_point.items():
        point: dict[str, object] = {}
        if all(key in values for key in ((8, 10), (10, 10), (12, 10))):
            point["recursion_axis_at_N10"] = {
                "ratios": {
                    str(order): float(values[(order, 10)]["theta_over_glasses"])
                    for order in (8, 10, 12)
                },
                "log_step_R8_to_R10": float(values[(10, 10)]["log_theta_over_glasses"])
                - float(values[(8, 10)]["log_theta_over_glasses"]),
                "log_step_R10_to_R12": float(values[(12, 10)]["log_theta_over_glasses"])
                - float(values[(10, 10)]["log_theta_over_glasses"]),
            }
        if all(key in values for key in ((12, 10), (12, 12))):
            point["momentum_axis_at_R12"] = {
                "ratios": {
                    str(order): float(values[(12, order)]["theta_over_glasses"])
                    for order in (10, 12)
                },
                "log_step_N10_to_N12": float(values[(12, 12)]["log_theta_over_glasses"])
                - float(values[(12, 10)]["log_theta_over_glasses"]),
            }
        output[point_id] = point
    return output


def run(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design", type=Path, default=DEFAULT_DESIGN)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--point-id", action="append")
    parser.add_argument(
        "--orders",
        type=_parse_order,
        nargs="+",
        default=((8, 10), (10, 10), (12, 10), (12, 12)),
    )
    parser.add_argument("--dps", type=int, default=40)
    parser.add_argument("--vacuum-word-length", type=int, default=8)
    parser.add_argument("--vacuum-oscillator-level", type=int, default=32)
    parser.add_argument("--scalar-word-length", type=int, default=12)
    parser.add_argument("--scalar-max-mode", type=int, default=120)
    parser.add_argument("--scalar-tolerance", type=float, default=1.0e-15)
    parser.add_argument("--geometry-only", action="store_true")
    parser.add_argument("--skip-plot", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)

    design = _load_design(args.design)
    preflight = preflight_design(design)
    requested = set(args.point_id) if args.point_id else None
    available = {str(point["id"]) for point in design["points"]}  # type: ignore[index]
    if requested is not None and not requested <= available:
        raise ValueError(f"unknown point IDs: {sorted(requested - available)}")

    rows: list[dict[str, object]] = []
    if not args.geometry_only:
        rows = _evaluate(
            design,
            point_ids=requested,
            orders=args.orders,
            dps=args.dps,
            vacuum_word_length=args.vacuum_word_length,
            vacuum_oscillator_level=args.vacuum_oscillator_level,
            scalar_word_length=args.scalar_word_length,
            scalar_max_mode=args.scalar_max_mode,
            scalar_tolerance=args.scalar_tolerance,
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "scope": "Bosonic genus-two theta/glasses locality at better-conditioned q.",
        "observable": design["observable"],
        "matching_assumed": False,
        "design": str(args.design),
        "selected_ids": [report["point_id"] for report in preflight],
        "geometry_preflight": preflight,
        "orders": [list(order) for order in args.orders],
        "settings": {
            "dps": args.dps,
            "vacuum_word_length": args.vacuum_word_length,
            "vacuum_oscillator_level": args.vacuum_oscillator_level,
            "scalar_word_length": args.scalar_word_length,
            "scalar_max_mode": args.scalar_max_mode,
            "scalar_tolerance": args.scalar_tolerance,
            "global_seed": "resummed genus-two global block in both channels",
            "free_scalar_zero_mode": "included as det(Im Omega)^(-1/2)",
        },
        "geometry_only": args.geometry_only,
        "successful_evaluations": sum(row["status"] == "ok" for row in rows),
        "failed_evaluations": sum(row["status"] != "ok" for row in rows),
        "convergence": _convergence_summary(rows),
        "rows": rows,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if rows:
        _write_csv(args.out_dir / "frame_comparison.csv", rows)
        if not args.skip_plot:
            _plot(args.out_dir / "bosonic_stable_q_locality.png", rows)
    print(f"wrote {args.out_dir / 'summary.json'}")


if __name__ == "__main__":
    run()
