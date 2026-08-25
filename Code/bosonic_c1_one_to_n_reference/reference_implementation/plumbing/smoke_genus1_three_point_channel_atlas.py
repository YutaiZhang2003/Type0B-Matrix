#!/usr/bin/env python3
"""Small blind RQMC smoke run for the torus three-point channel atlas."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

try:
    from genus1_three_point_channel_atlas import (
        LiouvilleTorusThreePointAtlas,
        reduced_worldsheet_integrand_three_point_patched,
    )
    from genus1_three_point_worldsheet import LiouvilleTorusThreePointNecklace
    from genus1_two_point_worldsheet import MomentumRule
    from refine_genus1_three_point_worldsheet import (
        TAIL_SEED_OFFSET,
        mean_and_standard_error,
        ordered_gap_importance,
        sobol_points,
        tail_height_and_jacobian,
    )
except ImportError:  # pragma: no cover
    from plumbing.genus1_three_point_channel_atlas import (
        LiouvilleTorusThreePointAtlas,
        reduced_worldsheet_integrand_three_point_patched,
    )
    from plumbing.genus1_three_point_worldsheet import LiouvilleTorusThreePointNecklace
    from plumbing.genus1_two_point_worldsheet import MomentumRule
    from plumbing.refine_genus1_three_point_worldsheet import (
        TAIL_SEED_OFFSET,
        mean_and_standard_error,
        ordered_gap_importance,
        sobol_points,
        tail_height_and_jacobian,
    )


def _complex_record(value: complex) -> dict[str, float]:
    return {"real": float(complex(value).real), "imag": float(complex(value).imag)}


def _positions(point: np.ndarray, tau: complex, alpha: float) -> tuple[list[tuple[complex, complex]], float]:
    first_x, second_x, weight_x = ordered_gap_importance(
        float(point[2]),
        float(point[3]),
        alpha=alpha,
    )
    first_y, second_y, weight_y = ordered_gap_importance(
        float(point[4]),
        float(point[5]),
        alpha=alpha,
    )
    pairs = [
        (
            2.0 * math.pi * (first_x + first_y * tau),
            2.0 * math.pi * (second_x + second_y * tau),
        ),
        (
            2.0 * math.pi * (second_x + first_y * tau),
            2.0 * math.pi * (first_x + second_y * tau),
        ),
    ]
    return pairs, 0.5 * weight_x * weight_y


def evaluate_bulk(
    atlas: LiouvilleTorusThreePointAtlas,
    points: np.ndarray,
    *,
    cutoff: float,
    alpha: float,
    target_channel: str | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    values = np.zeros(len(points), dtype=np.complex128)
    components = {
        channel: np.zeros(len(points), dtype=np.complex128)
        for channel in ("necklace", "pair_ope", "comb_ope")
    }
    for index, point in enumerate(points):
        tau1 = float(point[0]) - 0.5
        tau2_min = math.sqrt(1.0 - tau1 * tau1)
        tau2 = tau2_min + float(point[1]) * (float(cutoff) - tau2_min)
        tau = tau1 + 1.0j * tau2
        pairs, position_weight = _positions(point, tau, alpha)
        reduced_rows = []
        for first, second in pairs:
            choice = atlas.choose_channel(first, second, tau)
            if target_channel is not None and choice.channel != target_channel:
                reduced_rows.append((0.0j, choice))
            else:
                reduced_rows.append(
                    reduced_worldsheet_integrand_three_point_patched(
                        atlas,
                        first,
                        second,
                        tau,
                    )
                )
        reduced = sum(row[0] for row in reduced_rows)
        jacobian = (
            position_weight
            * (float(cutoff) - tau2_min)
            * (2.0 * math.pi) ** 4
            * tau2**2
        )
        values[index] = jacobian * reduced
        for reduced_value, choice in reduced_rows:
            components[choice.channel][index] += jacobian * reduced_value
    return values, components


def evaluate_tail(
    atlas: LiouvilleTorusThreePointAtlas,
    points: np.ndarray,
    *,
    alpha: float,
    proposal_exponent: float,
    target_channel: str | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    values = np.zeros(len(points), dtype=np.complex128)
    components = {
        channel: np.zeros(len(points), dtype=np.complex128)
        for channel in ("necklace", "pair_ope", "comb_ope")
    }
    for index, point in enumerate(points):
        tau2, tau_jacobian = tail_height_and_jacobian(
            float(point[0]),
            tail_start=8.0,
            proposal_exponent=proposal_exponent,
        )
        tau = (float(point[1]) - 0.5) + 1.0j * tau2
        pairs, position_weight = _positions(point, tau, alpha)
        reduced_rows = []
        for first, second in pairs:
            choice = atlas.choose_channel(first, second, tau)
            if target_channel is not None and choice.channel != target_channel:
                reduced_rows.append((0.0j, choice))
            else:
                reduced_rows.append(
                    reduced_worldsheet_integrand_three_point_patched(
                        atlas,
                        first,
                        second,
                        tau,
                    )
                )
        common_jacobian = (
            tau_jacobian
            * position_weight
            * (2.0 * math.pi) ** 4
            * tau2**2
        )
        values[index] = common_jacobian * sum(row[0] for row in reduced_rows)
        for reduced_value, choice in reduced_rows:
            components[choice.channel][index] += common_jacobian * reduced_value
    return values, components


def run(args: argparse.Namespace) -> dict[str, object]:
    rules = tuple(
        MomentumRule.power_legendre(5.0, args.momentum_order, 2.0 + 0.137 * edge)
        for edge in range(3)
    )
    necklace = LiouvilleTorusThreePointNecklace(
        args.t,
        momentum_rules=rules,
        high_order=args.necklace_order,
        low_order=args.necklace_low_order,
        block_backend=args.necklace_backend,
        c_regulator=args.c_regulator,
        special_dps=args.dps,
        coefficient_workers=args.coefficient_workers,
    )
    if args.necklace_bank_cache is not None and args.necklace_bank_cache.is_file():
        necklace.load_banks(args.necklace_bank_cache)
        print(f"loaded necklace banks from {args.necklace_bank_cache}", flush=True)
    if args.prepare_necklace_first:
        necklace.prepare(checkpoint_path=args.necklace_bank_cache)
    atlas = LiouvilleTorusThreePointAtlas(
        necklace,
        patch_epsilon=args.patch_epsilon,
        triple_patch_epsilon=args.triple_patch_epsilon,
        ope_order=args.ope_order,
        total_ope_order=args.ope_order,
        high_loop_order=args.ope_loop_order,
        low_loop_order=args.ope_low_loop_order,
        comb_loop_order=min(3, args.ope_order),
        special_dps=args.dps,
        bank_cache_path=args.bank_cache,
        evaluation_order_cap=args.evaluation_order_cap,
        necklace_qhat_threshold=args.necklace_qhat_threshold,
        necklace_second_qhat_threshold=args.necklace_second_qhat_threshold,
    )
    replicate_values: list[complex] = []
    component_replicates = {
        region: {channel: [] for channel in ("necklace", "pair_ope", "comb_ope")}
        for region in ("bulk", "tail")
    }
    stratum_alphas = {"necklace": 1.0, "pair_ope": args.alpha, "comb_ope": 0.2}
    for replicate in range(args.replicates):
        bulk = 0.0j
        tail = 0.0j
        for channel_index, channel in enumerate(component_replicates["bulk"]):
            seed_offset = 100000 * channel_index
            bulk_points = sobol_points(
                6,
                args.sobol_power,
                args.seed + seed_offset + replicate,
            )
            tail_points = sobol_points(
                6,
                args.sobol_power,
                args.seed + seed_offset + TAIL_SEED_OFFSET + replicate,
            )
            bulk_values, _ = evaluate_bulk(
                atlas,
                bulk_points,
                cutoff=8.0,
                alpha=stratum_alphas[channel],
                target_channel=channel,
            )
            tail_values, _ = evaluate_tail(
                atlas,
                tail_points,
                alpha=stratum_alphas[channel],
                proposal_exponent=args.tail_proposal_exponent,
                target_channel=channel,
            )
            channel_bulk = complex(np.mean(bulk_values))
            channel_tail = complex(np.mean(tail_values))
            component_replicates["bulk"][channel].append(channel_bulk)
            component_replicates["tail"][channel].append(channel_tail)
            bulk += channel_bulk
            tail += channel_tail
        replicate_values.append(bulk + tail)
        print(
            f"replicate {replicate + 1}/{args.replicates}: "
            f"bulk={bulk.imag:+.7e}j tail={tail.imag:+.7e}j "
            f"total={(bulk + tail).imag:+.7e}j",
            flush=True,
        )
    mean, standard_error = mean_and_standard_error(replicate_values)
    payload: dict[str, object] = {
        "calculation": "blind torus-three-point channel-atlas smoke run",
        "t": float(args.t),
        "matrix_model_present": False,
        "design": {
            "momentum_order": int(args.momentum_order),
            "necklace_order": int(args.necklace_order),
            "necklace_low_order": int(args.necklace_low_order),
            "necklace_backend": args.necklace_backend,
            "necklace_c_regulator": float(args.c_regulator),
            "necklace_qhat_threshold": args.necklace_qhat_threshold,
            "necklace_second_qhat_threshold": args.necklace_second_qhat_threshold,
            "ope_order": int(args.ope_order),
            "ope_loop_order": int(args.ope_loop_order),
            "ope_low_loop_order": int(args.ope_low_loop_order),
            "evaluation_order_cap": args.evaluation_order_cap,
            "sobol_power": int(args.sobol_power),
            "replicates": int(args.replicates),
            "position_alpha": float(args.alpha),
            "stratified_position_alphas": stratum_alphas,
            "patch_epsilon": float(args.patch_epsilon),
            "triple_patch_epsilon": float(args.triple_patch_epsilon),
            "tail_integrated_directly_to_infinity": True,
        },
        "replicate_values": [_complex_record(value) for value in replicate_values],
        "mean": _complex_record(mean),
        "rqmc_standard_error": _complex_record(standard_error),
        "channel_components": {
            region: {
                channel: {
                    "replicate_values": [
                        _complex_record(value) for value in values
                    ],
                    "mean": _complex_record(mean_and_standard_error(values)[0]),
                    "rqmc_standard_error": _complex_record(
                        mean_and_standard_error(values)[1]
                    ),
                }
                for channel, values in channels.items()
            }
            for region, channels in component_replicates.items()
        },
        "atlas_diagnostics": atlas.diagnostics(),
        "necklace_diagnostics": necklace.diagnostics(),
        "status": "smoke test only; not a frozen amplitude",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if args.necklace_bank_cache is not None:
        necklace.save_banks(args.necklace_bank_cache)
    print(f"wrote {args.output}", flush=True)
    return payload


def parser() -> argparse.ArgumentParser:
    out = argparse.ArgumentParser(description=__doc__)
    out.add_argument("--t", type=float, default=0.75)
    out.add_argument("--momentum-order", type=int, default=4)
    out.add_argument("--necklace-order", type=int, default=4)
    out.add_argument("--necklace-low-order", type=int, default=2)
    out.add_argument(
        "--necklace-backend",
        choices=("regulated-h-recursion", "exact-c25-descendants"),
        default="exact-c25-descendants",
    )
    out.add_argument("--c-regulator", type=float, default=0.05)
    out.add_argument("--coefficient-workers", type=int, default=1)
    out.add_argument("--necklace-qhat-threshold", type=float)
    out.add_argument("--necklace-second-qhat-threshold", type=float)
    out.add_argument("--ope-order", type=int, default=6)
    out.add_argument("--ope-loop-order", type=int, default=4)
    out.add_argument("--ope-low-loop-order", type=int, default=2)
    out.add_argument("--evaluation-order-cap", type=int)
    out.add_argument("--sobol-power", type=int, default=5)
    out.add_argument("--replicates", type=int, default=2)
    out.add_argument("--seed", type=int, default=17051301)
    out.add_argument("--alpha", type=float, default=0.3)
    out.add_argument("--patch-epsilon", type=float, default=0.15)
    out.add_argument("--triple-patch-epsilon", type=float, default=0.10)
    out.add_argument("--tail-proposal-exponent", type=float, default=1.5)
    out.add_argument("--dps", type=int, default=24)
    out.add_argument("--bank-cache", type=Path)
    out.add_argument("--necklace-bank-cache", type=Path)
    out.add_argument("--prepare-necklace-first", action="store_true")
    out.add_argument(
        "--output",
        type=Path,
        default=Path(
            "plumbing/results/genus1_three_point_worldsheet/"
            "channel_atlas_smoke_t075_p4_n32_r2_v1.json"
        ),
    )
    return out


if __name__ == "__main__":
    run(parser().parse_args())
