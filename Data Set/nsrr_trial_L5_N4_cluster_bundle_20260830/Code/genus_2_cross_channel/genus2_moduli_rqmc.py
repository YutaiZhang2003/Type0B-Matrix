#!/usr/bin/env python3
"""Coverage-controlled randomized QMC designs for the genus-two moduli integral.

The design integrates over the six independent variables of the existing
Minkowski-cone proposal.  It does not accept/reject down to a fixed number of
points.  Instead, every scrambled Sobol proposal retains its exact importance
weight and the Gottschling indicator.  CFT data are needed only for proposals
inside the domain; outside proposals contribute known zeros.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from scipy.stats import qmc

THIS_FILE = Path(__file__).resolve()
sys.path.insert(0, str(THIS_FILE.parent))

from genus2_integrand_normalization import GENUS2_GENERIC_STACK_WEIGHT  # noqa: E402
from genus2_siegel_fundamental_domain import (  # noqa: E402
    SIEGEL_VOLUME_G2,
    gottschling_min_margin,
    in_gottschling_domain,
    minkowski_proposals_from_unit_cube,
)


DEFAULT_OUTPUT = Path(
    "plumbing/results/genus2_c1_moduli_mc/rqmc_design_R8_M64"
)


@dataclass(frozen=True)
class RQMCReplicateSummary:
    replicate: int
    scramble_seed: int
    power: int
    proposal_count: int
    domain_count: int
    domain_fraction: float
    coarse_domain_volume_estimate: float
    relative_volume_error: float
    proposal_discrepancy: float
    marginal_bin_count_min: int
    marginal_bin_count_max: int
    maximum_t1: float
    maximum_t3: float
    maximum_t1_tail_level: int
    maximum_t3_tail_level: int


@dataclass(frozen=True)
class RQMCIntegralEstimate:
    replicate_count: int
    cft_node_count: int
    raw_replicate_estimates: tuple[float, ...]
    raw_estimate: float
    raw_scramble_standard_error: float
    volume_calibrated_replicate_estimates: tuple[float, ...]
    volume_calibrated_estimate: float
    volume_calibrated_scramble_standard_error: float


def _tail_level(unit_coordinate: np.ndarray) -> np.ndarray:
    """Return dyadic tail levels with mass ``2^(-level-1)``."""

    return np.floor(-np.log2(1.0 - unit_coordinate)).astype(int)


def _marginal_balance(points: np.ndarray, bins: int) -> tuple[int, int]:
    counts = []
    for column in range(points.shape[1]):
        histogram, _ = np.histogram(points[:, column], bins=bins, range=(0.0, 1.0))
        counts.extend(int(value) for value in histogram)
    return min(counts), max(counts)


def generate_rqmc_replicate(
    *,
    replicate: int,
    power: int,
    scramble_seed: int,
    marginal_bins: int = 8,
) -> tuple[list[dict[str, object]], RQMCReplicateSummary]:
    """Generate one independently scrambled direct-importance replicate."""

    if power < 1:
        raise ValueError("power must be positive")
    proposal_count = 2**int(power)
    if marginal_bins < 1 or proposal_count % marginal_bins != 0:
        raise ValueError("marginal_bins must divide the proposal count")

    engine = qmc.Sobol(d=6, scramble=True, seed=int(scramble_seed))
    points = engine.random_base2(m=int(power))
    omega, invariant_weight, coordinates = minkowski_proposals_from_unit_cube(points)
    domain = np.asarray(in_gottschling_domain(omega), dtype=bool)
    margins = np.asarray(gottschling_min_margin(omega), dtype=float)
    domain_indices = np.flatnonzero(domain)
    domain_count = int(domain_indices.size)
    volume_estimate = float(np.mean(invariant_weight * domain))
    t1_tail = _tail_level(points[:, 3])
    t3_tail = _tail_level(points[:, 4])
    bin_min, bin_max = _marginal_balance(points, marginal_bins)

    rows: list[dict[str, object]] = []
    for proposal_index in domain_indices:
        value = omega[proposal_index]
        weight = float(invariant_weight[proposal_index])
        rows.append(
            {
                "sampling_scheme": "scrambled_sobol_minkowski_importance",
                "rqmc_node_id": f"r{int(replicate):03d}-p{int(proposal_index):08d}",
                "rqmc_replicate": int(replicate),
                "rqmc_scramble_seed": int(scramble_seed),
                "rqmc_power": int(power),
                "rqmc_proposal_count": proposal_count,
                "rqmc_proposal_index": int(proposal_index),
                "rqmc_domain_count": domain_count,
                "rqmc_invariant_weight": weight,
                "rqmc_coarse_volume_weight": weight / proposal_count,
                "rqmc_stack_integration_weight": (
                    GENUS2_GENERIC_STACK_WEIGHT * weight / proposal_count
                ),
                "rqmc_u_x11": float(points[proposal_index, 0]),
                "rqmc_u_x12": float(points[proposal_index, 1]),
                "rqmc_u_x22": float(points[proposal_index, 2]),
                "rqmc_u_t1": float(points[proposal_index, 3]),
                "rqmc_u_t3": float(points[proposal_index, 4]),
                "rqmc_u_r": float(points[proposal_index, 5]),
                "rqmc_t1": float(coordinates[proposal_index, 0]),
                "rqmc_t3": float(coordinates[proposal_index, 1]),
                "rqmc_r": float(coordinates[proposal_index, 2]),
                "rqmc_t1_tail_level": int(t1_tail[proposal_index]),
                "rqmc_t3_tail_level": int(t3_tail[proposal_index]),
                "gottschling_margin": float(margins[proposal_index]),
                "det_im_omega": float(np.linalg.det(value.imag)),
                "x11": float(value[0, 0].real),
                "x12": float(value[0, 1].real),
                "x22": float(value[1, 1].real),
                "y11": float(value[0, 0].imag),
                "y12": float(value[0, 1].imag),
                "y22": float(value[1, 1].imag),
            }
        )

    summary = RQMCReplicateSummary(
        replicate=int(replicate),
        scramble_seed=int(scramble_seed),
        power=int(power),
        proposal_count=proposal_count,
        domain_count=domain_count,
        domain_fraction=domain_count / proposal_count,
        coarse_domain_volume_estimate=volume_estimate,
        relative_volume_error=volume_estimate / SIEGEL_VOLUME_G2 - 1.0,
        proposal_discrepancy=float(qmc.discrepancy(points)),
        marginal_bin_count_min=bin_min,
        marginal_bin_count_max=bin_max,
        maximum_t1=float(np.max(coordinates[:, 0])),
        maximum_t3=float(np.max(coordinates[:, 1])),
        maximum_t1_tail_level=int(np.max(t1_tail)),
        maximum_t3_tail_level=int(np.max(t3_tail)),
    )
    return rows, summary


def generate_rqmc_design(
    *,
    replicate_count: int,
    power: int,
    base_seed: int,
    marginal_bins: int = 8,
) -> tuple[list[dict[str, object]], list[RQMCReplicateSummary]]:
    """Generate all domain nodes for independent scrambled replicates."""

    if replicate_count < 2:
        raise ValueError("at least two replicates are required for an error estimate")
    rows: list[dict[str, object]] = []
    summaries: list[RQMCReplicateSummary] = []
    sample_index = 0
    for replicate in range(int(replicate_count)):
        replicate_rows, summary = generate_rqmc_replicate(
            replicate=replicate,
            power=power,
            scramble_seed=int(base_seed) + replicate,
            marginal_bins=marginal_bins,
        )
        for row in replicate_rows:
            row["sample_index"] = sample_index
            sample_index += 1
        rows.extend(replicate_rows)
        summaries.append(summary)
    return rows, summaries


def _mean_and_scramble_se(values: Sequence[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or array.size < 2 or not np.all(np.isfinite(array)):
        raise ValueError("need at least two finite replicate values")
    return float(np.mean(array)), float(np.std(array, ddof=1) / math.sqrt(array.size))


def estimate_rqmc_integral(
    rows: Sequence[dict[str, object]],
    transformed_values: Sequence[float],
) -> RQMCIntegralEstimate:
    """Assemble raw and exact-volume-calibrated estimates by scramble."""

    if len(rows) != len(transformed_values) or not rows:
        raise ValueError("rows and transformed_values must have the same nonzero length")
    values = np.asarray(transformed_values, dtype=float)
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("transformed values must be finite and nonnegative")

    replicates = sorted({int(row["rqmc_replicate"]) for row in rows})
    raw: list[float] = []
    calibrated: list[float] = []
    for replicate in replicates:
        indices = [
            index
            for index, row in enumerate(rows)
            if int(row["rqmc_replicate"]) == replicate
        ]
        expected = int(rows[indices[0]]["rqmc_domain_count"])
        if len(indices) != expected:
            raise ValueError(
                f"replicate {replicate} is incomplete: {len(indices)} of {expected} domain nodes"
            )
        weights = np.asarray(
            [float(rows[index]["rqmc_invariant_weight"]) for index in indices]
        )
        proposal_count = int(rows[indices[0]]["rqmc_proposal_count"])
        if any(int(rows[index]["rqmc_proposal_count"]) != proposal_count for index in indices):
            raise ValueError(f"replicate {replicate} mixes proposal counts")
        replicate_values = values[indices]
        raw.append(
            float(
                GENUS2_GENERIC_STACK_WEIGHT
                * np.sum(weights * replicate_values)
                / proposal_count
            )
        )
        calibrated.append(
            float(
                GENUS2_GENERIC_STACK_WEIGHT
                * SIEGEL_VOLUME_G2
                * np.sum(weights * replicate_values)
                / np.sum(weights)
            )
        )

    raw_mean, raw_se = _mean_and_scramble_se(raw)
    calibrated_mean, calibrated_se = _mean_and_scramble_se(calibrated)
    return RQMCIntegralEstimate(
        replicate_count=len(replicates),
        cft_node_count=len(rows),
        raw_replicate_estimates=tuple(raw),
        raw_estimate=raw_mean,
        raw_scramble_standard_error=raw_se,
        volume_calibrated_replicate_estimates=tuple(calibrated),
        volume_calibrated_estimate=calibrated_mean,
        volume_calibrated_scramble_standard_error=calibrated_se,
    )


def _write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate a genus-two scrambled-Sobol design.")
    parser.add_argument("--replicates", type=int, default=8)
    parser.add_argument("--power", type=int, default=6)
    parser.add_argument("--base-seed", type=int, default=20260712)
    parser.add_argument("--marginal-bins", type=int, default=8)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(list(argv) if argv is not None else None)

    rows, summaries = generate_rqmc_design(
        replicate_count=args.replicates,
        power=args.power,
        base_seed=args.base_seed,
        marginal_bins=args.marginal_bins,
    )
    volume_estimates = [summary.coarse_domain_volume_estimate for summary in summaries]
    volume_mean, volume_se = _mean_and_scramble_se(volume_estimates)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    nodes_path = args.out_dir / "domain_nodes.csv"
    replicates_path = args.out_dir / "replicate_summary.csv"
    summary_path = args.out_dir / "summary.json"
    _write_csv(nodes_path, rows)
    _write_csv(replicates_path, [asdict(summary) for summary in summaries])
    payload = {
        "scope": (
            "Direct randomized-QMC importance design on the complete six-dimensional "
            "Minkowski proposal; only in-domain nodes require CFT evaluation."
        ),
        "estimator": (
            "J2=(1/2)*mean_over_all_proposals[1_F2*w*det(Im Omega)^3*I2]"
        ),
        "replicate_count": args.replicates,
        "power": args.power,
        "proposal_count_per_replicate": 2**args.power,
        "domain_cft_node_count": len(rows),
        "base_seed": args.base_seed,
        "marginal_bins": args.marginal_bins,
        "exact_coarse_domain_volume": SIEGEL_VOLUME_G2,
        "rqmc_volume_estimate": volume_mean,
        "rqmc_volume_scramble_standard_error": volume_se,
        "rqmc_volume_z_score": (volume_mean - SIEGEL_VOLUME_G2) / volume_se,
        "replicates": [asdict(summary) for summary in summaries],
        "notes": [
            "Every finite design has full proposal support in distribution; Sobol balance controls all resolved quantiles.",
            "Independent scrambles, rather than individual domain nodes, are the units used for the standard error.",
            "The raw direct-importance estimator is unbiased over the random scrambles.",
            "Exact-volume calibration is a lower-variance but finite-sample biased diagnostic and is not the primary estimator.",
            "No failed in-domain CFT node may be dropped from a replicate.",
            "The stable rqmc_node_id permits nested power-of-two extension without recomputing old CFT nodes.",
        ],
    }
    summary_path.write_text(json.dumps(payload, indent=2) + "\n")

    print("Genus-two randomized-QMC moduli design")
    print(
        f"  replicates={args.replicates}, proposals/replicate={2**args.power}, "
        f"CFT domain nodes={len(rows)}"
    )
    print(
        f"  volume={volume_mean:.12g} +/- {volume_se:.3g}; "
        f"exact={SIEGEL_VOLUME_G2:.12g}; "
        f"z={(volume_mean - SIEGEL_VOLUME_G2) / volume_se:.3g}"
    )
    print(
        "  marginal proposal-bin counts="
        f"{min(summary.marginal_bin_count_min for summary in summaries)}.."
        f"{max(summary.marginal_bin_count_max for summary in summaries)}"
    )
    print(f"  wrote {nodes_path}")
    print(f"  wrote {replicates_path}")
    print(f"  wrote {summary_path}")


if __name__ == "__main__":
    run()
