#!/usr/bin/env python3
"""Measure theta/glasses plumbing coverage on the full genus-two moduli space.

The sampled period matrices are unweighted draws from the invariant Siegel
measure on Gottschling's six-real-dimensional fundamental domain.  The scan
has two deliberately separate stages:

1. a finite symplectic-marking search ranked by leading sewing coordinates;
2. a finite-q inverse-period and Schottky word-stability check on an iid
   subsample, plus a separate sample from the hard leading-q tail.

Neither ``q <= 0.16`` nor a leading-q score is called a convergence theorem.
The former is only the range reached by the saved c=25 order-12 benchmark;
actual integration nodes still require recursion-order and momentum-integral
doubling checks.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

try:
    from genus2_plumbing_atlas import build_plumbing_atlas, shortlist_markings
    from genus2_siegel_fundamental_domain import (
        SIEGEL_VOLUME_G2,
        estimate_invariant_volume,
        sample_invariant_domain,
    )
except ImportError:  # pragma: no cover
    from plumbing.genus2_plumbing_atlas import build_plumbing_atlas, shortlist_markings
    from plumbing.genus2_siegel_fundamental_domain import (
        SIEGEL_VOLUME_G2,
        estimate_invariant_volume,
        sample_invariant_domain,
    )


DEFAULT_THRESHOLDS = (0.05, 0.10, 0.16, 0.20, 0.25, 0.30, 0.40, 0.50)


def _wilson_interval(successes: int, count: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if count <= 0:
        return math.nan, math.nan
    p = successes / count
    denominator = 1.0 + z * z / count
    center = (p + z * z / (2.0 * count)) / denominator
    half_width = z * math.sqrt(p * (1.0 - p) / count + z * z / (4.0 * count * count)) / denominator
    return center - half_width, center + half_width


def _fraction_summary(mask: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(mask, dtype=bool)
    successes = int(np.sum(values))
    count = int(values.size)
    low, high = _wilson_interval(successes, count)
    return {
        "successes": successes,
        "count": count,
        "fraction": successes / count if count else math.nan,
        "wilson_95_low": low,
        "wilson_95_high": high,
    }


def _quantiles(values: Sequence[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=float)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return {"finite_count": 0, "nonfinite_count": int(array.size)}
    return {
        "finite_count": int(finite.size),
        "nonfinite_count": int(array.size - finite.size),
        "q05": float(np.quantile(finite, 0.05)),
        "q25": float(np.quantile(finite, 0.25)),
        "q50": float(np.quantile(finite, 0.50)),
        "q75": float(np.quantile(finite, 0.75)),
        "q90": float(np.quantile(finite, 0.90)),
        "q95": float(np.quantile(finite, 0.95)),
        "q99": float(np.quantile(finite, 0.99)),
        "max": float(np.max(finite)),
    }


def _period_coordinates(omega: np.ndarray) -> dict[str, float]:
    x = omega.real
    y = omega.imag
    return {
        "x11": float(x[0, 0]),
        "x12": float(x[0, 1]),
        "x22": float(x[1, 1]),
        "y11": float(y[0, 0]),
        "y12": float(y[0, 1]),
        "y22": float(y[1, 1]),
        "det_y": float(np.linalg.det(y)),
        "cusp_scale": float(math.log(y[0, 0] / (math.sqrt(3.0) / 2.0))),
        "anisotropy": float(math.log(y[1, 1] / y[0, 0])),
        "minkowski_mix": float(2.0 * y[0, 1] / y[0, 0]),
    }


def leading_scan(
    omega_values: np.ndarray,
    *,
    search_depth: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index, omega in enumerate(omega_values):
        theta = shortlist_markings(omega, "theta", search_depth=search_depth, count=1)
        glasses = shortlist_markings(omega, "glasses", search_depth=search_depth, count=1)
        theta_best = theta[0] if theta else None
        glasses_best = glasses[0] if glasses else None
        theta_score = math.inf if theta_best is None else theta_best.leading_q_max
        glasses_score = math.inf if glasses_best is None else glasses_best.leading_q_max
        preferred = "theta" if theta_score <= glasses_score else "glasses"
        best_score = min(theta_score, glasses_score)
        row: dict[str, object] = {
            "sample_index": index,
            **_period_coordinates(omega),
            "theta_leading_q_max": theta_score,
            "glasses_leading_q_max": glasses_score,
            "best_leading_q_max": best_score,
            "best_leading_topology": preferred,
            "theta_symplectic_word": "" if theta_best is None else theta_best.word,
            "glasses_symplectic_word": "" if glasses_best is None else glasses_best.word,
            "exact_sample_kind": "",
            "exact_coverage_status": "",
            "exact_best_topology": "",
            "exact_best_q_max": math.nan,
            "exact_period_max_residual": math.nan,
            "exact_period_map_stability": math.nan,
            "exact_symplectic_word": "",
        }
        rows.append(row)
    return rows


def _usable_best_chart(result: object) -> object | None:
    return next(
        (
            chart
            for chart in result.charts
            if chart.status in {"reference-q-envelope", "requires-recursion-order-study"}
        ),
        None,
    )


def exact_scan(
    omega_values: np.ndarray,
    rows: list[dict[str, object]],
    indices: Sequence[int],
    *,
    sample_kind: str,
    search_depth: int,
    prefilter_count: int,
    word_length: int,
    stability_step: int,
    max_nfev: int,
    q_reference_max: float,
    period_tolerance: float,
    stability_tolerance: float,
) -> list[dict[str, object]]:
    exact_rows: list[dict[str, object]] = []
    for offset, index in enumerate(indices, start=1):
        result = build_plumbing_atlas(
            omega_values[index],
            search_depth=search_depth,
            prefilter_count=prefilter_count,
            word_length=word_length,
            stability_step=stability_step,
            max_nfev=max_nfev,
            q_reference_max=q_reference_max,
            period_tolerance=period_tolerance,
            stability_tolerance=stability_tolerance,
        )
        chart = _usable_best_chart(result)
        row = rows[index]
        row.update(
            {
                "exact_sample_kind": sample_kind,
                "exact_coverage_status": result.coverage_status,
                "exact_best_topology": result.best_topology or "none",
                "exact_best_q_max": math.nan if result.best_q_max is None else result.best_q_max,
                "exact_period_max_residual": math.nan if chart is None else chart.period_max_residual,
                "exact_period_map_stability": math.nan if chart is None else chart.period_map_stability,
                "exact_symplectic_word": "" if chart is None else chart.word,
            }
        )
        exact_rows.append(dict(row))
        if offset % 16 == 0 or offset == len(indices):
            print(f"  finite-q {sample_kind}: {offset}/{len(indices)}")
    return exact_rows


def summarize_leading(rows: list[dict[str, object]], thresholds: Sequence[float]) -> dict[str, object]:
    theta = np.asarray([row["theta_leading_q_max"] for row in rows], dtype=float)
    glasses = np.asarray([row["glasses_leading_q_max"] for row in rows], dtype=float)
    best = np.minimum(theta, glasses)
    preferred_theta = theta <= glasses
    return {
        "sample_count": len(rows),
        "best_q_quantiles": _quantiles(best),
        "theta_q_quantiles": _quantiles(theta),
        "glasses_q_quantiles": _quantiles(glasses),
        "preferred_theta": _fraction_summary(preferred_theta),
        "threshold_cdf": {
            f"{threshold:.6g}": _fraction_summary(best <= threshold) for threshold in thresholds
        },
        "warning": (
            "These are leading degeneration coordinates after a finite symplectic search. "
            "They rank candidate charts but do not certify the finite-q period map or block convergence."
        ),
    }


def summarize_exact(rows: list[dict[str, object]], *, q_reference_max: float) -> dict[str, object]:
    statuses = Counter(str(row["exact_coverage_status"]) for row in rows)
    chart_found = np.asarray(
        [
            row["exact_coverage_status"]
            in {
                "period-chart-inside-reference-q-envelope",
                "period-chart-found-but-block-order-unvalidated",
            }
            for row in rows
        ],
        dtype=bool,
    )
    inside = np.asarray(
        [row["exact_coverage_status"] == "period-chart-inside-reference-q-envelope" for row in rows],
        dtype=bool,
    )
    valid_q = np.asarray(
        [float(row["exact_best_q_max"]) for row in rows if math.isfinite(float(row["exact_best_q_max"]))],
        dtype=float,
    )
    topology = Counter(
        str(row["exact_best_topology"])
        for row in rows
        if str(row["exact_best_topology"]) in {"theta", "glasses"}
    )
    return {
        "sample_count": len(rows),
        "status_counts": dict(statuses),
        "certified_period_chart": _fraction_summary(chart_found),
        "inside_reference_q_envelope": _fraction_summary(inside),
        "valid_q_quantiles": {} if valid_q.size == 0 else _quantiles(valid_q),
        "best_topology_counts": dict(topology),
        "q_reference_max": q_reference_max,
        "warning": (
            "Inside the reference q envelope means only q_max <= the saved c=25 order-12 benchmark edge. "
            "It is not a pointwise recursion-error certificate."
        ),
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = list(rows[0])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_plot(
    path: Path,
    rows: list[dict[str, object]],
    primary_exact: list[dict[str, object]],
    *,
    q_reference_max: float,
    search_depth: int | str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from matplotlib.lines import Line2D

    theta = np.asarray([row["theta_leading_q_max"] for row in rows], dtype=float)
    glasses = np.asarray([row["glasses_leading_q_max"] for row in rows], dtype=float)
    best = np.minimum(theta, glasses)
    cusp = np.asarray([row["cusp_scale"] for row in rows], dtype=float)
    anisotropy = np.asarray([row["anisotropy"] for row in rows], dtype=float)
    mix = np.asarray([row["minkowski_mix"] for row in rows], dtype=float)

    fig, axes = plt.subplots(1, 3, figsize=(16.8, 5.2), constrained_layout=True)

    for values, color, label, width in (
        (theta, "#246A73", "theta", 1.8),
        (glasses, "#D17A22", "glasses", 1.8),
        (best, "#202124", "best atlas chart", 2.5),
    ):
        ordered = np.sort(values[np.isfinite(values)])
        cdf = np.arange(1, len(ordered) + 1) / len(values)
        axes[0].step(ordered, cdf, where="post", color=color, linewidth=width, label=label)
    axes[0].axvline(q_reference_max, color="#B23A48", linestyle="--", linewidth=1.5)
    axes[0].set_xscale("log")
    axes[0].set_xlim(max(1.0e-8, float(np.min(best)) * 0.7), 1.0)
    axes[0].set_ylim(0.0, 1.01)
    axes[0].set_xlabel(r"leading score $\max_e |q_e|$")
    axes[0].set_ylabel("invariant-volume CDF")
    axes[0].set_title("Finite marking search (leading q only)")
    axes[0].legend(frameon=False, loc="lower right")
    axes[0].grid(alpha=0.18)

    scatter = axes[1].scatter(
        cusp,
        anisotropy,
        c=best,
        s=22.0 + 26.0 * mix,
        cmap="viridis",
        norm=LogNorm(vmin=max(1.0e-8, float(np.min(best))), vmax=min(1.0, float(np.max(best)))),
        alpha=0.76,
        linewidths=0.0,
    )
    colorbar = fig.colorbar(scatter, ax=axes[1], pad=0.02)
    colorbar.set_label(r"best leading $\max_e|q_e|$")
    axes[1].set_xlabel(r"cusp scale $\log(Y_{11}/(\sqrt{3}/2))$")
    axes[1].set_ylabel(r"anisotropy $\log(Y_{22}/Y_{11})$")
    axes[1].set_title("Projection of the full 6D domain")
    axes[1].grid(alpha=0.15)

    status_style = {
        "period-chart-inside-reference-q-envelope": ("#2A9D8F", "o", "inside q reference"),
        "period-chart-found-but-block-order-unvalidated": ("#E9C46A", "D", "higher block order needed"),
        "uncovered-at-current-search-settings": ("#B23A48", "x", "period chart not certified"),
    }
    for status, (color, marker, label) in status_style.items():
        subset = [row for row in primary_exact if row["exact_coverage_status"] == status]
        if not subset:
            continue
        leading = np.asarray([row["best_leading_q_max"] for row in subset], dtype=float)
        finite = np.asarray([row["exact_best_q_max"] for row in subset], dtype=float)
        if marker == "x":
            # Uncertified points have no finite-q ordinate; place them on the
            # upper rim without pretending that this is a measured q value.
            finite = np.full_like(leading, 0.93)
        axes[2].scatter(
            leading,
            finite,
            color=color,
            marker=marker,
            s=38,
            linewidths=1.0,
            alpha=0.85,
            label=label,
        )
    diagonal = np.geomspace(max(1.0e-7, float(np.min(best))), 1.0, 200)
    axes[2].plot(diagonal, diagonal, color="#777777", linestyle=":", linewidth=1.2)
    axes[2].axhline(q_reference_max, color="#B23A48", linestyle="--", linewidth=1.3)
    axes[2].set_xscale("log")
    axes[2].set_yscale("log")
    axes[2].set_xlim(max(1.0e-7, float(np.min(best)) * 0.7), 1.0)
    axes[2].set_ylim(max(1.0e-7, float(np.min(best)) * 0.7), 1.0)
    axes[2].set_xlabel(r"leading best $\max_e|q_e|$")
    axes[2].set_ylabel(r"finite-q certified $\max_e|q_e|$")
    axes[2].set_title("IID finite-q certification sample")
    axes[2].grid(alpha=0.18)
    axes[2].legend(frameon=False, loc="lower right")

    fig.suptitle(
        f"Genus-two plumbing atlas on invariant Siegel-domain samples (Sp search depth {search_depth})",
        fontsize=13,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    if path.suffix.lower() != ".svg":
        fig.savefig(path.with_suffix(".svg"))
    plt.close(fig)


def run(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Scan plumbing coverage over the full genus-two moduli space.")
    parser.add_argument("--volume-proposals", type=int, default=250_000)
    parser.add_argument("--sample-count", type=int, default=768)
    parser.add_argument("--exact-count", type=int, default=96)
    parser.add_argument("--tail-count", type=int, default=24)
    parser.add_argument("--seed", type=int, default=20260711)
    parser.add_argument("--search-depth", type=int, default=3)
    parser.add_argument("--prefilter-count", type=int, default=2)
    parser.add_argument("--word-length", type=int, default=4)
    parser.add_argument("--stability-step", type=int, default=1)
    parser.add_argument("--max-nfev", type=int, default=120)
    parser.add_argument("--q-reference-max", type=float, default=0.16)
    parser.add_argument("--period-tolerance", type=float, default=2.0e-5)
    parser.add_argument("--stability-tolerance", type=float, default=2.0e-5)
    parser.add_argument("--out-dir", type=Path, default=Path("plumbing/results/genus2_full_moduli_coverage"))
    args = parser.parse_args(argv)

    if not 0 < args.exact_count <= args.sample_count:
        raise ValueError("exact-count must be in [1, sample-count]")
    if not 0 <= args.tail_count <= args.sample_count - args.exact_count:
        raise ValueError("tail-count must fit outside the iid exact subsample")

    print("Checking the invariant-domain normalization...")
    volume = estimate_invariant_volume(proposal_count=args.volume_proposals, seed=args.seed)
    if abs(volume.importance_z_score) > 5.0:
        raise RuntimeError("Gottschling-domain volume check failed; refusing to report atlas coverage")
    print(
        f"  volume={volume.importance_estimate:.9f} +/- {volume.importance_standard_error:.2g}; "
        f"exact={SIEGEL_VOLUME_G2:.9f}"
    )

    print(f"Drawing {args.sample_count} invariant-volume period matrices...")
    sample = sample_invariant_domain(args.sample_count, seed=args.seed + 1)
    print("Running the leading-q marking census...")
    rows = leading_scan(sample.omega, search_depth=args.search_depth)

    primary_indices = list(range(args.exact_count))
    remaining = list(range(args.exact_count, args.sample_count))
    tail_indices = sorted(
        remaining,
        key=lambda index: float(rows[index]["best_leading_q_max"]),
        reverse=True,
    )[: args.tail_count]

    print(f"Certifying {len(primary_indices)} iid points with the finite-q period map...")
    exact_kwargs = {
        "search_depth": args.search_depth,
        "prefilter_count": args.prefilter_count,
        "word_length": args.word_length,
        "stability_step": args.stability_step,
        "max_nfev": args.max_nfev,
        "q_reference_max": args.q_reference_max,
        "period_tolerance": args.period_tolerance,
        "stability_tolerance": args.stability_tolerance,
    }
    primary_exact = exact_scan(
        sample.omega,
        rows,
        primary_indices,
        sample_kind="iid",
        **exact_kwargs,
    )
    print(f"Auditing {len(tail_indices)} additional hard-tail points...")
    tail_exact = exact_scan(
        sample.omega,
        rows,
        tail_indices,
        sample_kind="hard-tail",
        **exact_kwargs,
    )

    leading_summary = summarize_leading(rows, DEFAULT_THRESHOLDS)
    primary_summary = summarize_exact(primary_exact, q_reference_max=args.q_reference_max)
    tail_summary = summarize_exact(tail_exact, q_reference_max=args.q_reference_max)
    payload = {
        "scope": (
            "Full six-real-dimensional genus-two Gottschling fundamental domain, sampled with "
            "the invariant Siegel measure. The decomposable locus has measure zero."
        ),
        "measure": "d^3 X d^3 Y / det(Im Omega)^3",
        "volume_check": asdict(volume),
        "sampling": {
            "sample_count": args.sample_count,
            "sample_seed": args.seed + 1,
            "sampler_proposal_count": sample.proposal_count,
            "iid_exact_count": args.exact_count,
            "hard_tail_count": args.tail_count,
        },
        "atlas_settings": {
            "search_depth": args.search_depth,
            "prefilter_count": args.prefilter_count,
            "word_length": args.word_length,
            "stability_step": args.stability_step,
            "max_nfev": args.max_nfev,
            "q_reference_max": args.q_reference_max,
            "period_tolerance": args.period_tolerance,
            "stability_tolerance": args.stability_tolerance,
        },
        "leading_census": leading_summary,
        "finite_q_iid": primary_summary,
        "finite_q_hard_tail": tail_summary,
        "interpretation": {
            "geometric_fraction": (
                "All reported fractions are with respect to invariant Siegel volume, not the c=1 string integrand."
            ),
            "period_chart": (
                "A certified period chart passed the nonlinear inverse-period residual and a Schottky word-cutoff step."
            ),
            "recursion": (
                "q<=0.16 is a reference envelope only. Pointwise Virasoro recursion and momentum quadrature "
                "must be order-doubled before the physical moduli integral is evaluated."
            ),
            "atlas_completeness": (
                "The current search includes theta and glasses graphs and finite Sp(4,Z) homology markings. "
                "It does not yet enumerate Torelli-distinct pants decompositions or multiple inverse roots."
            ),
        },
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "full_moduli_depth3.csv"
    json_path = args.out_dir / "full_moduli_depth3.json"
    figure_path = args.out_dir / "full_moduli_depth3.png"
    write_csv(csv_path, rows)
    json_path.write_text(json.dumps(payload, indent=2) + "\n")
    write_plot(
        figure_path,
        rows,
        primary_exact,
        q_reference_max=args.q_reference_max,
        search_depth=args.search_depth,
    )

    inside = primary_summary["inside_reference_q_envelope"]
    chart = primary_summary["certified_period_chart"]
    print("Full-domain coverage summary")
    print(
        f"  leading q<={args.q_reference_max:g}: "
        f"{leading_summary['threshold_cdf'][f'{args.q_reference_max:.6g}']['fraction']:.3f}"
    )
    print(f"  finite-q period chart certified: {chart['successes']}/{chart['count']} ({chart['fraction']:.3f})")
    print(
        f"  finite-q chart inside q reference: {inside['successes']}/{inside['count']} "
        f"({inside['fraction']:.3f})"
    )
    print(f"  wrote {csv_path}")
    print(f"  wrote {json_path}")
    print(f"  wrote {figure_path} and {figure_path.with_suffix('.svg')}")


if __name__ == "__main__":
    run()
