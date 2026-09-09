#!/usr/bin/env python3
"""Attach plumbing difficulty diagnostics to a genus-two RQMC design."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

THIS_FILE = Path(__file__).resolve()
sys.path.insert(0, str(THIS_FILE.parent))

from monte_carlo_integrate_genus2_c1 import omega_from_csv_row  # noqa: E402
from scan_genus2_moduli_plumbing_coverage import leading_scan  # noqa: E402


DEFAULT_INPUT = Path(
    "plumbing/results/genus2_c1_moduli_mc/"
    "rqmc_design_R8_M64/domain_nodes.csv"
)
DEFAULT_OUTPUT = Path(
    "plumbing/results/genus2_c1_moduli_mc/"
    "rqmc_design_R8_M64"
)


def difficulty_tier(leading_q_max: float) -> str:
    if leading_q_max <= 0.16:
        return "reference"
    if leading_q_max <= 0.25:
        return "moderate"
    return "hard"


def prepare_rows(
    source_rows: Sequence[dict[str, str]],
    *,
    search_depth: int,
) -> list[dict[str, object]]:
    omega = np.asarray([omega_from_csv_row(row) for row in source_rows])
    leading = leading_scan(omega, search_depth=search_depth)
    prepared: list[dict[str, object]] = []
    for source, diagnostic in zip(source_rows, leading):
        best_q = float(diagnostic["best_leading_q_max"])
        row: dict[str, object] = dict(source)
        row.update(
            {
                "theta_leading_q_max": diagnostic["theta_leading_q_max"],
                "glasses_leading_q_max": diagnostic["glasses_leading_q_max"],
                "best_leading_q_max": best_q,
                "best_leading_topology": diagnostic["best_leading_topology"],
                "theta_leading_symplectic_word": diagnostic["theta_symplectic_word"],
                "glasses_leading_symplectic_word": diagnostic["glasses_symplectic_word"],
                "plumbing_difficulty_tier": difficulty_tier(best_q),
                "atlas_policy": (
                    "base-depth-3-then-refine-depth-4"
                    if best_q > 0.16
                    else "base-depth-3"
                ),
                "cft_order_policy": (
                    "one fixed production block/quadrature order at every node; "
                    "lower-order convergence runs are separate diagnostics"
                ),
                "period_map_policy": (
                    "adaptive hybrid certificate: normalized holomorphic one-forms "
                    "in the bulk, rescaled multiprecision holomorphic forms in mixed "
                    "cusps, Schottky words only when every q is small, and both "
                    "methods with an explicit agreement bar in their overlap"
                ),
            }
        )
        prepared.append(row)
    return prepared


def _quantiles(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "minimum": float(np.min(array)),
        "median": float(np.quantile(array, 0.5)),
        "q75": float(np.quantile(array, 0.75)),
        "q90": float(np.quantile(array, 0.90)),
        "q95": float(np.quantile(array, 0.95)),
        "q99": float(np.quantile(array, 0.99)),
        "maximum": float(np.max(array)),
    }


def summarize(rows: Sequence[dict[str, object]]) -> dict[str, object]:
    best_q = [float(row["best_leading_q_max"]) for row in rows]
    tier_counts = {
        tier: sum(row["plumbing_difficulty_tier"] == tier for row in rows)
        for tier in ("reference", "moderate", "hard")
    }
    replicate_rows = []
    for replicate in sorted({int(row["rqmc_replicate"]) for row in rows}):
        group = [row for row in rows if int(row["rqmc_replicate"]) == replicate]
        replicate_rows.append(
            {
                "replicate": replicate,
                "domain_count": len(group),
                "reference_count": sum(
                    row["plumbing_difficulty_tier"] == "reference" for row in group
                ),
                "moderate_count": sum(
                    row["plumbing_difficulty_tier"] == "moderate" for row in group
                ),
                "hard_count": sum(
                    row["plumbing_difficulty_tier"] == "hard" for row in group
                ),
                "maximum_leading_q": max(float(row["best_leading_q_max"]) for row in group),
            }
        )
    return {
        "domain_node_count": len(rows),
        "leading_q_quantiles": _quantiles(best_q),
        "tier_counts": tier_counts,
        "inside_q_0p16_fraction": sum(value <= 0.16 for value in best_q) / len(best_q),
        "inside_q_0p25_fraction": sum(value <= 0.25 for value in best_q) / len(best_q),
        "replicates": replicate_rows,
        "policy": {
            "sampling": "no node is removed or reweighted by plumbing difficulty",
            "atlas": "depth three for all nodes; depth-four refinement when needed",
            "cft": (
                "all nodes use one fixed production order; recursion-order scans are "
                "kept outside the sampling design"
            ),
            "period_map": (
                "validate the selected q with two holomorphic-form basis orders at "
                "tolerance 1e-6 and re-invert directly when needed; use Schottky "
                "only in the deep cusp and retain logarithmic sewing coordinates"
            ),
            "failure": "one failed node makes its entire scramble incomplete",
        },
    }


def _write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_plot(path: Path, rows: Sequence[dict[str, object]]) -> None:
    os.environ.setdefault(
        "MPLCONFIGDIR",
        str(Path(tempfile.gettempdir()) / "stringmc-matplotlib"),
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    q_values = np.sort(np.asarray([float(row["best_leading_q_max"]) for row in rows]))
    cdf = np.arange(1, len(q_values) + 1) / len(q_values)
    replicates = sorted({int(row["rqmc_replicate"]) for row in rows})
    tier_names = ("reference", "moderate", "hard")
    colors = ("#16717c", "#c28c2c", "#a34a34")
    counts = np.array(
        [
            [
                sum(
                    int(row["rqmc_replicate"]) == replicate
                    and row["plumbing_difficulty_tier"] == tier
                    for row in rows
                )
                for replicate in replicates
            ]
            for tier in tier_names
        ]
    )

    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    axes[0].plot(q_values, cdf, color="#16717c", linewidth=2.2)
    axes[0].axvline(0.16, color="#4b5056", linestyle="--", label="q=0.16 reference")
    axes[0].axvline(0.25, color="#a34a34", linestyle=":", label="q=0.25 scheduler")
    axes[0].set_xlabel("best leading plumbing q")
    axes[0].set_ylabel("fraction of in-domain nodes")
    axes[0].set_title("Leading plumbing difficulty")
    axes[0].legend(frameon=False)

    bottom = np.zeros(len(replicates))
    for name, color, values in zip(tier_names, colors, counts):
        axes[1].bar(replicates, values, bottom=bottom, color=color, label=name)
        bottom += values
    axes[1].set_xlabel("independent Sobol scramble")
    axes[1].set_ylabel("in-domain CFT nodes")
    axes[1].set_title("No hard node is dropped")
    axes[1].legend(frameon=False, ncol=3)
    for axis in axes:
        axis.grid(True, axis="y", color="#d9d9d6", alpha=0.8)
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle("Genus-two RQMC production preflight", fontsize=15, fontweight="semibold")
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def run(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Prepare a genus-two RQMC CFT manifest.")
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--search-depth", type=int, default=3)
    parser.add_argument(
        "--previous-design-csv",
        type=Path,
        help="older nested design whose node ids already have reusable CFT values",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--skip-plot", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)

    source_rows = list(csv.DictReader(args.input_csv.open()))
    rows = prepare_rows(source_rows, search_depth=args.search_depth)
    previous_ids = (
        set()
        if args.previous_design_csv is None
        else {
            row["rqmc_node_id"]
            for row in csv.DictReader(args.previous_design_csv.open())
        }
    )
    for row in rows:
        row["nested_cft_status"] = (
            "reuse-existing" if row["rqmc_node_id"] in previous_ids else "evaluate-new"
        )
    new_rows = [row for row in rows if row["nested_cft_status"] == "evaluate-new"]
    summary = summarize(rows)
    summary["previous_design_csv"] = (
        None if args.previous_design_csv is None else str(args.previous_design_csv)
    )
    summary["reusable_existing_node_count"] = len(rows) - len(new_rows)
    summary["new_cft_node_count"] = len(new_rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "production_nodes.csv"
    new_csv_path = args.out_dir / "new_production_nodes.csv"
    json_path = args.out_dir / "production_summary.json"
    png_path = args.out_dir / "production_preflight.png"
    _write_csv(csv_path, rows)
    _write_csv(new_csv_path, new_rows)
    json_path.write_text(
        json.dumps(
            {
                "scope": (
                    "Leading plumbing scheduler for a complete weighted RQMC design; "
                    "leading q is not a finite-q chart certification."
                ),
                "input_csv": str(args.input_csv),
                "search_depth": args.search_depth,
                **summary,
            },
            indent=2,
        )
        + "\n"
    )
    if not args.skip_plot:
        _write_plot(png_path, rows)
    print("Genus-two RQMC production preflight")
    print(f"  domain nodes={len(rows)}, tiers={summary['tier_counts']}")
    print(
        f"  reusable CFT nodes={summary['reusable_existing_node_count']}, "
        f"new evaluations={summary['new_cft_node_count']}"
    )
    print(
        f"  inside leading q<=0.16: {float(summary['inside_q_0p16_fraction']):.3%}; "
        f"q<=0.25: {float(summary['inside_q_0p25_fraction']):.3%}"
    )
    print(f"  wrote {csv_path}")
    print(f"  wrote {new_csv_path}")
    print(f"  wrote {json_path}")
    if not args.skip_plot:
        print(f"  wrote {png_path}")


if __name__ == "__main__":
    run()
