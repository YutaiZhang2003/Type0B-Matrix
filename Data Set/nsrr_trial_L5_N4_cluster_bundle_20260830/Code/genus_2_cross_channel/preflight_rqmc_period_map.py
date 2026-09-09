#!/usr/bin/env python3
"""Preflight exact plumbing period maps on difficult nodes of an RQMC design."""

from __future__ import annotations

import argparse
import csv
import json
import math
import multiprocessing as mp
import queue
import sys
import time
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

THIS_FILE = Path(__file__).resolve()
sys.path.insert(0, str(THIS_FILE.parent))

from audit_q_to_omega_accuracy import validate_or_refine_period_map  # noqa: E402
from monte_carlo_integrate_genus2_c1 import (  # noqa: E402
    _matrix_from_strings,
    omega_from_csv_row,
    select_plumbing_chart,
)
from genus2_period_table import Genus2PeriodMapTable  # noqa: E402
from genus2_plumbing_atlas import symplectic_matrix_csv_fields  # noqa: E402


DEFAULT_INPUT = Path(
    "plumbing/results/genus2_c1_moduli_mc/"
    "rqmc_design_R8_M64/production_nodes.csv"
)
DEFAULT_OUTPUT = Path(
    "plumbing/results/genus2_c1_moduli_mc/"
    "rqmc_design_R8_M64/period_map_preflight"
)
STRESS_METADATA_KEYS = (
    "rqmc_stratum_label",
    "rqmc_component_name",
    "rqmc_t1",
    "rqmc_t3_tail_level",
    "rqmc_t3",
    "det_im_omega",
)


def select_tier_stress_rows(
    rows: Sequence[dict[str, str]],
) -> list[dict[str, str]]:
    """Select the largest leading-q node in every nonempty tier and scramble."""

    selected: list[dict[str, str]] = []
    replicates = sorted({int(row["rqmc_replicate"]) for row in rows})
    for replicate in replicates:
        for tier in ("reference", "moderate", "hard"):
            candidates = [
                row
                for row in rows
                if int(row["rqmc_replicate"]) == replicate
                and row["plumbing_difficulty_tier"] == tier
            ]
            if candidates:
                selected.append(
                    max(candidates, key=lambda row: float(row["best_leading_q_max"]))
                )
    return selected


def select_tail_stress_rows(
    rows: Sequence[dict[str, str]],
) -> list[dict[str, str]]:
    """Select the deepest realized t3-tail node in every scramble."""

    selected: list[dict[str, str]] = []
    replicates = sorted({int(row["rqmc_replicate"]) for row in rows})
    for replicate in replicates:
        candidates = [row for row in rows if int(row["rqmc_replicate"]) == replicate]
        if candidates:
            selected.append(
                max(
                    candidates,
                    key=lambda row: (
                        int(row.get("rqmc_t3_tail_level", "0")),
                        float(row["rqmc_t3"]),
                        float(row["best_leading_q_max"]),
                    ),
                )
            )
    return selected


def select_stress_rows(
    rows: Sequence[dict[str, str]],
    *,
    mode: str,
) -> list[dict[str, str]]:
    if mode == "tiers":
        candidates = select_tier_stress_rows(rows)
    elif mode == "tail":
        candidates = select_tail_stress_rows(rows)
    elif mode == "tiers-and-tail":
        candidates = select_tier_stress_rows(rows) + select_tail_stress_rows(rows)
    else:
        raise ValueError(f"unknown stress selection mode: {mode}")
    selected: list[dict[str, str]] = []
    seen: set[str] = set()
    for row in candidates:
        node_id = row["rqmc_node_id"]
        if node_id not in seen:
            seen.add(node_id)
            selected.append(row)
    return selected


def audit_node(row: dict[str, str], args: argparse.Namespace) -> dict[str, object]:
    started = time.time()
    result: dict[str, object] = {
        "rqmc_node_id": row["rqmc_node_id"],
        "sample_index": int(row["sample_index"]),
        "rqmc_replicate": int(row["rqmc_replicate"]),
        "plumbing_difficulty_tier": row["plumbing_difficulty_tier"],
        "best_leading_q_max": float(row["best_leading_q_max"]),
        "status": "failed",
        "error": "",
    }
    for key in STRESS_METADATA_KEYS:
        if row.get(key, "") != "":
            result[key] = row[key]
    try:
        omega = omega_from_csv_row(row)
        period_table = (
            None
            if getattr(args, "period_table_index", None) is None
            else Genus2PeriodMapTable.from_portable_index(
                args.period_table_index,
                verify_table_path=getattr(args, "period_table_csv", None),
            )
        )
        chart, search_stage = select_plumbing_chart(
            omega,
            q_reference_max=args.q_reference_max,
            refine_above_q=args.refine_above_q,
            base_search_depth=args.base_search_depth,
            base_prefilter_count=args.base_prefilter_count,
            base_word_length=args.base_word_length,
            base_period_tolerance=args.base_period_tolerance,
            base_stability_tolerance=args.base_stability_tolerance,
            refined_search_depth=args.refined_search_depth,
            refined_prefilter_count=args.refined_prefilter_count,
            refined_word_length=args.refined_word_length,
            refined_period_tolerance=args.refined_period_tolerance,
            refined_stability_tolerance=args.refined_stability_tolerance,
            period_table=period_table,
        )
        atlas_q = tuple(complex(value) for value in chart.q)
        atlas_log_q = tuple(complex(value) for value in chart.log_q)
        target = _matrix_from_strings(chart.omega_chart)
        result.update(
            {
                "search_stage": search_stage,
                "topology": chart.topology,
                "symplectic_word": chart.word,
                **symplectic_matrix_csv_fields(chart.matrix),
                "chart_status": chart.status,
                "atlas_inverse_seed_source": chart.inverse_seed_source,
                "atlas_q_max": chart.q_max,
                "atlas_period_algorithm": chart.period_algorithm,
                "atlas_period_map_region": chart.period_map_region,
                "atlas_plumbing_geometry_margin": chart.plumbing_geometry_margin,
                "atlas_period_overlap_residual": (
                    ""
                    if chart.period_overlap_residual is None
                    else chart.period_overlap_residual
                ),
                "atlas_period_residual": chart.period_max_residual,
                "atlas_period_map_stability": chart.period_map_stability,
                **{
                    f"atlas_q{index}": value
                    for index, value in enumerate(chart.q, start=1)
                },
                **{
                    f"atlas_log_q{index}": value
                    for index, value in enumerate(chart.log_q, start=1)
                },
            }
        )
        certificate = validate_or_refine_period_map(
            chart.topology,
            target,
            atlas_q,
            word_length=args.validation_word_length,
            word_step=args.validation_word_step,
            tolerance=args.validation_tolerance,
            reinverse_validation_word_length=args.reinverse_validation_word_length,
            reinverse_max_nfev=args.reinverse_max_nfev,
            initial_log_q=atlas_log_q,
        )
        final_log_q = certificate.log_q or tuple(complex(np.log(value)) for value in certificate.q)
        result.update(
            {
                "status": "ok",
                "final_q_max": math.exp(max(value.real for value in final_log_q)),
                **{
                    f"final_q{index}": format(complex(value), ".17e")
                    for index, value in enumerate(certificate.q, start=1)
                },
                **{
                    f"final_log_q{index}": format(complex(value), ".17e")
                    for index, value in enumerate(final_log_q, start=1)
                },
                "period_algorithm": certificate.period_algorithm,
                "period_map_region": certificate.period_map_region or "",
                "period_overlap_residual": (
                    ""
                    if certificate.overlap_residual is None
                    else certificate.overlap_residual
                ),
                "period_agreement_tolerance": (
                    ""
                    if certificate.agreement_tolerance is None
                    else certificate.agreement_tolerance
                ),
                "period_validity_cell_id": certificate.validity_cell_id or "",
                "period_validity_reference_table_sha256": (
                    certificate.validity_reference_table_sha256 or ""
                ),
                "period_certified_error_bound": (
                    ""
                    if certificate.certified_error_bound is None
                    else certificate.certified_error_bound
                ),
                "fixed_q_period_residual": certificate.fixed_q_residual,
                "fixed_q_period_map_step": certificate.fixed_q_stability,
                "q_refined": certificate.refined,
                "final_period_residual": certificate.final_residual,
                "final_period_map_step": certificate.final_stability,
                "period_map_low_order": certificate.low_order,
                "period_map_high_order": certificate.high_order,
                "period_map_validation_order": certificate.validation_order,
                "period_seam_residual": certificate.seam_residual,
                "period_symmetry_error": certificate.symmetry_error,
                "reinverse_nfev": certificate.reinverse_nfev,
                "max_tau_shift": certificate.max_tau_shift,
            }
        )
    except Exception as exc:  # noqa: BLE001 - every preflight failure is retained.
        result["error"] = f"{type(exc).__name__}: {exc}"
    result["runtime_seconds"] = time.time() - started
    return result


def audit_node_with_timeout(
    row: dict[str, str],
    args: argparse.Namespace,
) -> dict[str, object]:
    """Audit one node behind a process-level wall-clock budget."""

    timeout = float(args.node_timeout_seconds)
    if timeout <= 0.0:
        return audit_node(row, args)
    context = mp.get_context("spawn")
    output: mp.Queue[dict[str, object]] = context.Queue(maxsize=1)
    process = context.Process(
        target=_audit_worker,
        args=(row, vars(args), output),
    )
    started = time.time()
    process.start()
    process.join(timeout)
    if process.is_alive():
        process.terminate()
        process.join()
        return {
            "rqmc_node_id": row["rqmc_node_id"],
            "sample_index": int(row["sample_index"]),
            "rqmc_replicate": int(row["rqmc_replicate"]),
            "plumbing_difficulty_tier": row["plumbing_difficulty_tier"],
            "best_leading_q_max": float(row["best_leading_q_max"]),
            "status": "failed",
            "error": f"TimeoutError: period-map preflight exceeded {timeout:g} seconds",
            "runtime_seconds": time.time() - started,
        }
    try:
        return output.get(timeout=2.0)
    except queue.Empty:
        return {
            "rqmc_node_id": row["rqmc_node_id"],
            "sample_index": int(row["sample_index"]),
            "rqmc_replicate": int(row["rqmc_replicate"]),
            "plumbing_difficulty_tier": row["plumbing_difficulty_tier"],
            "best_leading_q_max": float(row["best_leading_q_max"]),
            "status": "failed",
            "error": f"WorkerExitError: period-map worker exited with code {process.exitcode}",
            "runtime_seconds": time.time() - started,
        }


def _audit_worker(
    row: dict[str, str],
    args_values: dict[str, object],
    output: mp.Queue[dict[str, object]],
) -> None:
    output.put(audit_node(row, argparse.Namespace(**args_values)))


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


def _checkpoint_settings(args: argparse.Namespace) -> dict[str, object]:
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
        if key not in {"input_csv", "out_dir", "resume"}
    }


def run(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Preflight RQMC plumbing period maps.")
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--q-reference-max", type=float, default=0.16)
    parser.add_argument("--refine-above-q", type=float, default=0.16)
    parser.add_argument("--base-search-depth", type=int, default=3)
    parser.add_argument("--base-prefilter-count", type=int, default=2)
    parser.add_argument("--base-word-length", type=int, default=4)
    parser.add_argument("--base-period-tolerance", type=float, default=2.0e-5)
    parser.add_argument("--base-stability-tolerance", type=float, default=2.0e-5)
    parser.add_argument("--refined-search-depth", type=int, default=4)
    parser.add_argument("--refined-prefilter-count", type=int, default=6)
    parser.add_argument("--refined-word-length", type=int, default=6)
    parser.add_argument("--refined-period-tolerance", type=float, default=5.0e-6)
    parser.add_argument("--refined-stability-tolerance", type=float, default=5.0e-6)
    parser.add_argument("--validation-word-length", type=int, default=8)
    parser.add_argument("--validation-word-step", type=int, default=1)
    parser.add_argument("--validation-tolerance", type=float, default=1.0e-6)
    parser.add_argument("--reinverse-validation-word-length", type=int, default=10)
    parser.add_argument("--reinverse-max-nfev", type=int, default=80)
    parser.add_argument(
        "--period-table-index",
        type=Path,
        help="optional portable q-to-Omega table used for certified inverse seeds",
    )
    parser.add_argument(
        "--period-table-csv",
        type=Path,
        help="optional canonical table CSV used to verify the portable index hash",
    )
    parser.add_argument("--node-timeout-seconds", type=float, default=300.0)
    parser.add_argument(
        "--selection-mode",
        choices=("tiers", "tail", "tiers-and-tail"),
        default="tiers-and-tail",
    )
    parser.add_argument(
        "--node-id",
        action="append",
        help="audit an explicit node id; repeat for multiple nodes instead of stress selection",
    )
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.period_table_csv is not None and args.period_table_index is None:
        raise ValueError("--period-table-csv requires --period-table-index")
    if args.period_table_index is not None:
        period_table = Genus2PeriodMapTable.from_portable_index(
            args.period_table_index,
            verify_table_path=args.period_table_csv,
        )
        if not period_table.has_fundamental_index:
            raise ValueError(
                "--period-table-index must use schema v3 with fundamental-domain "
                "Omega coordinates and exact Sp(4,Z) markings"
            )

    source_rows = list(csv.DictReader(args.input_csv.open()))
    if args.node_id:
        source_by_id = {row["rqmc_node_id"]: row for row in source_rows}
        missing = [node_id for node_id in args.node_id if node_id not in source_by_id]
        if missing:
            raise ValueError(f"requested node ids are absent from the input: {missing}")
        selected = [source_by_id[node_id] for node_id in dict.fromkeys(args.node_id)]
    else:
        selected = select_stress_rows(source_rows, mode=args.selection_mode)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "period_map_preflight.csv"
    json_path = args.out_dir / "summary.json"
    checkpoint_path = args.out_dir / "checkpoint.json"
    selected_node_ids = [item["rqmc_node_id"] for item in selected]
    settings = _checkpoint_settings(args)
    rows: list[dict[str, object]] = []
    if args.resume and checkpoint_path.is_file():
        checkpoint = json.loads(checkpoint_path.read_text())
        if (
            checkpoint.get("input_csv") == str(args.input_csv)
            and checkpoint.get("selected_node_ids") == selected_node_ids
            and checkpoint.get("settings") == settings
        ):
            rows = list(checkpoint.get("rows", []))
    completed = {str(row["rqmc_node_id"]) for row in rows}
    for offset, row in enumerate(selected, start=1):
        if row["rqmc_node_id"] in completed:
            print(
                f"  {offset}/{len(selected)} {row['rqmc_node_id']} "
                "status=resume"
            )
            continue
        result = audit_node_with_timeout(row, args)
        rows.append(result)
        _write_csv(csv_path, rows)
        checkpoint_path.write_text(
            json.dumps(
                {
                    "input_csv": str(args.input_csv),
                    "selected_node_ids": selected_node_ids,
                    "settings": settings,
                    "rows": rows,
                },
                indent=2,
            )
            + "\n"
        )
        print(
            f"  {offset}/{len(selected)} {result['rqmc_node_id']} "
            f"tier={result['plumbing_difficulty_tier']} status={result['status']}"
        )

    source_by_id = {row["rqmc_node_id"]: row for row in source_rows}
    for result in rows:
        source = source_by_id[str(result["rqmc_node_id"])]
        for key in STRESS_METADATA_KEYS:
            if source.get(key, "") != "":
                result[key] = source[key]
    checkpoint_path.write_text(
        json.dumps(
            {
                "input_csv": str(args.input_csv),
                "selected_node_ids": selected_node_ids,
                "settings": settings,
                "rows": rows,
            },
            indent=2,
        )
        + "\n"
    )

    successful = [row for row in rows if row["status"] == "ok"]
    payload = {
        "scope": (
            "Period-map stress preflight selected by the requested leading-q "
            "tier and/or deepest-t3-tail policy in each independent scramble."
        ),
        "selection_mode": args.selection_mode,
        "explicit_node_ids": args.node_id,
        "input_csv": str(args.input_csv),
        "selected_count": len(rows),
        "successful_count": len(successful),
        "failed_node_ids": [row["rqmc_node_id"] for row in rows if row["status"] != "ok"],
        "refined_q_count": sum(bool(row["q_refined"]) for row in successful),
        "maximum_leading_q": max(float(row["best_leading_q_max"]) for row in rows),
        "maximum_final_q": max(
            (float(row["final_q_max"]) for row in successful),
            default=None,
        ),
        "maximum_final_period_residual": max(
            (float(row["final_period_residual"]) for row in successful),
            default=None,
        ),
        "maximum_final_map_step": max(
            (float(row["final_period_map_step"]) for row in successful),
            default=None,
        ),
        "runtime_seconds": sum(float(row["runtime_seconds"]) for row in rows),
        "settings": settings,
        "rows": rows,
    }
    _write_csv(csv_path, rows)
    json_path.write_text(json.dumps(payload, indent=2) + "\n")
    print("Genus-two RQMC period-map preflight")
    residual_text = (
        "n/a"
        if payload["maximum_final_period_residual"] is None
        else f"{payload['maximum_final_period_residual']:.3e}"
    )
    print(
        f"  passed={len(successful)}/{len(rows)}, refined={payload['refined_q_count']}, "
        f"max final residual={residual_text}"
    )
    print(f"  wrote {csv_path}")
    print(f"  wrote {json_path}")


if __name__ == "__main__":
    run()
