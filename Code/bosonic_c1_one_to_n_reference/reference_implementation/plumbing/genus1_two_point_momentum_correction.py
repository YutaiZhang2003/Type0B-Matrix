#!/usr/bin/env python3
"""Paired RQMC correction for the genus-one c=1 two-point amplitude.

The saved production calculation supplies a high-statistics fixed-rule
control variate.  At a smaller nested scrambled-Sobol design we evaluate both
that old rule and a locally scaled correlated polar rule, then integrate their
difference.  The new estimator is

``saved old production + mean(new local polar - old fixed rule)``.

The ordinary momentum measure is matched to
``P1^2 P2^2 exp(-a1 P1^2-a2 P2^2)``.  The analytic collision disc cancels the
second threshold factor and is matched to
``P_loop^2 exp(-a_loop P_loop^2-a_ope P_ope^2)``.  Every proposal weight is
undone exactly.  Adjacent radial-angular orders are promoted node by node.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
from scipy.stats import qmc

try:
    from genus1_two_point_adaptive_momentum import (
        AuditPoint,
        _relative_change,
        evaluate_point_polar,
    )
    from genus1_two_point_worldsheet import (
        LiouvilleTorusTwoPoint,
        MomentumRule,
        dedekind_eta,
        reduced_worldsheet_integrand,
        torus_prime_form_norm,
    )
    from integrate_genus1_two_point_worldsheet import (
        fit_tau_integrand_tail,
        integrated_fitted_tail,
    )
    from torus_two_point_blocks import _basis_and_inverse_gram
except ImportError:  # pragma: no cover - package-style execution
    from plumbing.genus1_two_point_adaptive_momentum import (
        AuditPoint,
        _relative_change,
        evaluate_point_polar,
    )
    from plumbing.genus1_two_point_worldsheet import (
        LiouvilleTorusTwoPoint,
        MomentumRule,
        dedekind_eta,
        reduced_worldsheet_integrand,
        torus_prime_form_norm,
    )
    from plumbing.integrate_genus1_two_point_worldsheet import (
        fit_tau_integrand_tail,
        integrated_fitted_tail,
    )
    from plumbing.torus_two_point_blocks import _basis_and_inverse_gram


DEFAULT_X_VALUES = (0.2, 0.4, 0.6, 0.8)
DEFAULT_POLAR_ORDERS = ((8, 10), (12, 14), (16, 18), (20, 22), (24, 26))
DEFAULT_DISC_ORDERS = ((8, 10), (12, 14), (16, 18))
DEFAULT_TAIL_SLICES = (8.0, 10.0, 12.0, 16.0, 20.0)
FORMAT_VERSION = 1


def _complex_record(value: complex) -> dict[str, float]:
    number = complex(value)
    return {"real": float(number.real), "imag": float(number.imag)}


def _record_complex(record: Mapping[str, object]) -> complex:
    return complex(float(record["real"]), float(record["imag"]))


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def _parse_pairs(values: Sequence[str]) -> tuple[tuple[int, int], ...]:
    pairs: list[tuple[int, int]] = []
    for value in values:
        pieces = tuple(int(piece) for piece in value.lower().split("x"))
        if len(pieces) != 2 or any(piece <= 0 for piece in pieces):
            raise ValueError(f"invalid polar order {value!r}")
        pairs.append((pieces[0], pieces[1]))
    if len(pairs) < 2:
        raise ValueError("at least two polar orders are required")
    if any(a[0] >= b[0] or a[1] >= b[1] for a, b in zip(pairs, pairs[1:])):
        raise ValueError("polar orders must increase in both directions")
    return tuple(pairs)


def prepare_manifest(
    *,
    path: Path,
    summary_path: Path,
    x_values: Sequence[float],
    replicates: int,
    bulk_sobol_power: int,
    tail_sobol_power: int,
    seed: int,
    cutoff: float,
    tail_slices: Sequence[float],
) -> dict[str, object]:
    if replicates < 2:
        raise ValueError("at least two scrambled Sobol replicates are required")
    rows: list[dict[str, object]] = []
    for x in x_values:
        if not 0.0 < float(x) < 1.0:
            raise ValueError("the real-contour calculation requires 0<x<1")
        for replicate in range(replicates):
            bulk_points = qmc.Sobol(
                d=4, scramble=True, seed=seed + replicate
            ).random_base2(bulk_sobol_power)
            for sample_index, point in enumerate(bulk_points):
                tau1 = float(point[0]) - 0.5
                tau2_min = math.sqrt(1.0 - tau1 * tau1)
                tau2 = tau2_min + float(point[1]) * (cutoff - tau2_min)
                tau = tau1 + 1.0j * tau2
                z = 2.0 * math.pi * (
                    float(point[2]) + 0.5 * float(point[3]) * tau
                )
                rows.append(
                    {
                        "target_index": len(rows),
                        "x": float(x),
                        "replicate": replicate,
                        "kind": "bulk-cutoff",
                        "sample_index": sample_index,
                        "cutoff": float(cutoff),
                        "tail_tau2": "",
                        "tau_real": tau.real,
                        "tau_imag": tau.imag,
                        "z_real": z.real,
                        "z_imag": z.imag,
                        "tau_jacobian": float(cutoff - tau2_min),
                    }
                )
            for slice_index, tau2 in enumerate(tail_slices):
                tail_points = qmc.Sobol(
                    d=3,
                    scramble=True,
                    seed=seed + 10000 + 97 * replicate + slice_index,
                ).random_base2(tail_sobol_power)
                for sample_index, point in enumerate(tail_points):
                    tau = float(point[0]) - 0.5 + 1.0j * float(tau2)
                    z = 2.0 * math.pi * (
                        float(point[1]) + 0.5 * float(point[2]) * tau
                    )
                    rows.append(
                        {
                            "target_index": len(rows),
                            "x": float(x),
                            "replicate": replicate,
                            "kind": "tail-slice",
                            "sample_index": sample_index,
                            "cutoff": "",
                            "tail_tau2": float(tau2),
                            "tau_real": tau.real,
                            "tau_imag": tau.imag,
                            "z_real": z.real,
                            "z_imag": z.imag,
                            "tau_jacobian": 1.0,
                        }
                    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "format_version": FORMAT_VERSION,
        "target_count": len(rows),
        "x_values": [float(value) for value in x_values],
        "replicates": replicates,
        "bulk_sobol_power": bulk_sobol_power,
        "bulk_points_per_replicate": 2**bulk_sobol_power,
        "tail_sobol_power": tail_sobol_power,
        "tail_points_per_slice_replicate": 2**tail_sobol_power,
        "seed": seed,
        "cutoff": float(cutoff),
        "tail_slices": [float(value) for value in tail_slices],
    }
    _atomic_json(summary_path, summary)
    return summary


def _old_correlator(
    x: float,
    *,
    old_order: int,
    p_max: float,
    power: float,
    necklace_orders: tuple[int, int],
    ope_orders: tuple[int, int],
    dps: int,
) -> LiouvilleTorusTwoPoint:
    return LiouvilleTorusTwoPoint(
        1.0j * x,
        momentum_rule=MomentumRule.power_legendre(p_max, old_order, power),
        necklace_orders=necklace_orders,
        ope_orders=ope_orders,
        special_dps=dps,
    )


def _channel_point(z: complex, tau: complex, epsilon: float) -> tuple[AuditPoint, str]:
    candidates = (
        z,
        z - 2.0 * math.pi,
        z - 2.0 * math.pi * tau,
        z - 2.0 * math.pi * (1.0 + tau),
    )
    local_z = min(candidates, key=abs)
    if abs(local_z) < 2.0 * math.pi * epsilon:
        v = cmath_exp_minus_i(local_z) - 1.0
        if 0.0 < abs(v) < 1.0:
            return AuditPoint("worldsheet-node", "ope", local_z, tau), "ope"
        return (
            AuditPoint("worldsheet-node", "necklace", z, tau),
            "necklace-fallback-|v|>=1",
        )
    return AuditPoint("worldsheet-node", "necklace", z, tau), "necklace"


def cmath_exp_minus_i(z: complex) -> complex:
    # Kept separate so the representation-domain gate is easy to unit test.
    return complex(np.exp(-1.0j * complex(z)))


def _adaptive_polar_value(
    point: AuditPoint,
    *,
    x: float,
    orders: Sequence[tuple[int, int]],
    tolerance: float,
    necklace_orders: tuple[int, int],
    ope_orders: tuple[int, int],
    dps: int,
) -> tuple[complex, float, bool, list[dict[str, object]]]:
    attempts: list[dict[str, object]] = []
    previous: complex | None = None
    selected: complex | None = None
    converged = False
    for radial_order, angular_order in orders:
        started = time.perf_counter()
        value, _ = evaluate_point_polar(
            point,
            x=x,
            radial_order=radial_order,
            angular_order=angular_order,
            necklace_orders=necklace_orders,
            ope_orders=ope_orders,
            dps=dps,
        )
        drift = None if previous is None else _relative_change(previous, value)
        attempts.append(
            {
                "radial_order": radial_order,
                "angular_order": angular_order,
                "node_count": radial_order * angular_order,
                "value": _complex_record(value),
                "relative_change": drift,
                "runtime_seconds": time.perf_counter() - started,
            }
        )
        selected = value
        if drift is not None and drift <= tolerance:
            converged = True
            break
        previous = value
    assert selected is not None
    last_drift = attempts[-1]["relative_change"]
    if last_drift is None:
        relative_envelope = 1.0
    else:
        relative_envelope = max(
            tolerance if converged else 0.0,
            (2.0 if converged else 4.0) * float(last_drift),
        )
    return selected, relative_envelope, converged, attempts


def evaluate_target(
    target: Mapping[str, str],
    *,
    output_dir: Path,
    old: LiouvilleTorusTwoPoint,
    polar_orders: Sequence[tuple[int, int]],
    disc_orders: Sequence[tuple[int, int]],
    tolerance: float,
    epsilon: float,
    collision_radius: float,
    necklace_orders: tuple[int, int],
    ope_orders: tuple[int, int],
    dps: int,
) -> dict[str, object]:
    target_index = int(target["target_index"])
    output = output_dir / f"target_{target_index:08d}.json"
    if output.exists():
        cached = json.loads(output.read_text())
        if cached.get("status") == "ok" and cached.get("target_index") == target_index:
            return cached
    x = float(target["x"])
    tau = complex(float(target["tau_real"]), float(target["tau_imag"]))
    z = complex(float(target["z_real"]), float(target["z_imag"]))
    tau_jacobian = float(target["tau_jacobian"])
    started = time.perf_counter()
    local_distance = min(abs(z), abs(z - 2.0 * math.pi))
    if local_distance >= collision_radius:
        point, channel_used = _channel_point(z, tau, epsilon)
        liouville, bulk_relative_error, bulk_converged, bulk_attempts = (
            _adaptive_polar_value(
                point,
                x=x,
                orders=polar_orders,
                tolerance=tolerance,
                necklace_orders=necklace_orders,
                ope_orders=ope_orders,
                dps=dps,
            )
        )
        prime_norm = torus_prime_form_norm(z, tau)
        timelike = complex(np.exp((1.0j * x) ** 2 * math.log(prime_norm)))
        common = abs(dedekind_eta(tau)) ** 2 * timelike / math.sqrt(tau.imag)
        new_bulk = 4.0 * math.pi**2 * tau.imag * common * liouville
        new_bulk_error = abs(new_bulk) * bulk_relative_error
        old_bulk = (
            4.0
            * math.pi**2
            * tau.imag
            * reduced_worldsheet_integrand(old, z, tau, epsilon=epsilon)
        )
    else:
        channel_used = "excised-collision-disc"
        new_bulk = 0.0 + 0.0j
        old_bulk = 0.0 + 0.0j
        new_bulk_error = 0.0
        bulk_converged = True
        bulk_attempts = []

    disc_point = AuditPoint(
        "worldsheet-disc",
        "collision-disc",
        0.0 + 0.0j,
        tau,
        collision_radius=collision_radius,
    )
    new_disc, disc_relative_error, disc_converged, disc_attempts = (
        _adaptive_polar_value(
            disc_point,
            x=x,
            orders=disc_orders,
            tolerance=tolerance,
            necklace_orders=necklace_orders,
            ope_orders=ope_orders,
            dps=dps,
        )
    )
    old_disc = complex(old.leading_collision_disc(tau, collision_radius))
    new_disc_error = abs(new_disc) * disc_relative_error
    new_sample = tau_jacobian * (new_bulk + new_disc)
    old_sample = tau_jacobian * (old_bulk + old_disc)
    payload: dict[str, object] = {
        "format_version": FORMAT_VERSION,
        "status": "ok",
        "target_index": target_index,
        "x": x,
        "replicate": int(target["replicate"]),
        "kind": target["kind"],
        "sample_index": int(target["sample_index"]),
        "cutoff": None if not target["cutoff"] else float(target["cutoff"]),
        "tail_tau2": (
            None if not target["tail_tau2"] else float(target["tail_tau2"])
        ),
        "tau": _complex_record(tau),
        "z": _complex_record(z),
        "tau_jacobian": tau_jacobian,
        "channel_used": channel_used,
        "bulk_converged": bulk_converged,
        "disc_converged": disc_converged,
        "bulk_attempts": bulk_attempts,
        "disc_attempts": disc_attempts,
        "new_bulk": _complex_record(new_bulk),
        "old_bulk": _complex_record(old_bulk),
        "new_disc": _complex_record(new_disc),
        "old_disc": _complex_record(old_disc),
        "new_sample": _complex_record(new_sample),
        "old_sample": _complex_record(old_sample),
        "delta_sample": _complex_record(new_sample - old_sample),
        "new_sample_absolute_momentum_error": float(
            tau_jacobian * (new_bulk_error + new_disc_error)
        ),
        "runtime_seconds": time.perf_counter() - started,
    }
    _atomic_json(output, payload)
    _basis_and_inverse_gram.cache_clear()
    return payload


def evaluate_range(
    *,
    manifest_path: Path,
    output_dir: Path,
    start: int,
    end: int,
    polar_orders: Sequence[tuple[int, int]],
    disc_orders: Sequence[tuple[int, int]],
    tolerance: float,
    epsilon: float,
    collision_radius: float,
    old_order: int,
    p_max: float,
    power: float,
    necklace_orders: tuple[int, int],
    ope_orders: tuple[int, int],
    dps: int,
) -> None:
    with manifest_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not 0 <= start <= end <= len(rows):
        raise ValueError("invalid target range")
    old_cache: dict[float, LiouvilleTorusTwoPoint] = {}
    for target in rows[start:end]:
        x = float(target["x"])
        if x not in old_cache:
            old_cache[x] = _old_correlator(
                x,
                old_order=old_order,
                p_max=p_max,
                power=power,
                necklace_orders=necklace_orders,
                ope_orders=ope_orders,
                dps=dps,
            )
        result = evaluate_target(
            target,
            output_dir=output_dir,
            old=old_cache[x],
            polar_orders=polar_orders,
            disc_orders=disc_orders,
            tolerance=tolerance,
            epsilon=epsilon,
            collision_radius=collision_radius,
            necklace_orders=necklace_orders,
            ope_orders=ope_orders,
            dps=dps,
        )
        print(
            f"target={result['target_index']} x={x:g} "
            f"kind={result['kind']} channel={result['channel_used']} "
            f"bulk_ok={result['bulk_converged']} disc_ok={result['disc_converged']} "
            f"runtime={float(result['runtime_seconds']):.1f}s",
            flush=True,
        )


def _mean_complex(records: Sequence[Mapping[str, object]], key: str) -> complex:
    return complex(np.mean([_record_complex(record[key]) for record in records]))  # type: ignore[arg-type]


def _tail_linear_weights(tau2_values: np.ndarray, tail_start: float) -> np.ndarray:
    design = np.column_stack(
        [tau2_values**-2.0, tau2_values ** (-5.0 / 3.0), tau2_values**-3.0]
    )
    coefficient_map = np.linalg.pinv(design)
    integral_vector = np.asarray(
        [1.0 / tail_start, 1.5 * tail_start ** (-2.0 / 3.0), 0.5 / tail_start**2]
    )
    return integral_vector @ coefficient_map


def finalize(
    *,
    manifest_path: Path,
    evaluation_dir: Path,
    saved_result_dir: Path,
    out_dir: Path,
) -> dict[str, object]:
    with manifest_path.open(newline="") as handle:
        targets = list(csv.DictReader(handle))
    records: list[dict[str, object]] = []
    missing: list[int] = []
    for target in targets:
        index = int(target["target_index"])
        path = evaluation_dir / f"target_{index:08d}.json"
        if not path.exists():
            missing.append(index)
            continue
        record = json.loads(path.read_text())
        if record.get("status") != "ok":
            missing.append(index)
            continue
        records.append(record)
    if missing:
        raise RuntimeError(
            f"cannot finalize: {len(missing)} missing/failed records; first={missing[:10]}"
        )

    grouped: dict[tuple[float, int, str, float | None], list[dict[str, object]]] = defaultdict(list)
    for record in records:
        key = (
            float(record["x"]),
            int(record["replicate"]),
            str(record["kind"]),
            None if record["tail_tau2"] is None else float(record["tail_tau2"]),
        )
        grouped[key].append(record)

    rows: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    for x in sorted({float(record["x"]) for record in records}):
        saved_path = saved_result_dir / f"x{int(round(10*x)):02d}_production.json"
        saved = json.loads(saved_path.read_text())
        saved_replicates = [
            _record_complex(value) for value in saved["cusp_fit"]["replicate_finals"]
        ]
        replicate_count = len(saved_replicates)
        tail_start = float(saved["cusp_fit"]["tail_start"])
        tail_slices = np.asarray(saved["cusp_fit"]["tau2_slices"], dtype=float)
        tail_error_weights = _tail_linear_weights(tail_slices, tail_start)
        correction_replicates: list[complex] = []
        corrected_replicates: list[complex] = []
        momentum_error_replicates: list[float] = []
        for replicate in range(replicate_count):
            bulk_records = grouped[(x, replicate, "bulk-cutoff", None)]
            bulk_delta = _mean_complex(bulk_records, "delta_sample")
            bulk_error = float(
                np.mean(
                    [record["new_sample_absolute_momentum_error"] for record in bulk_records]
                )
            )
            slice_deltas: list[complex] = []
            slice_errors: list[float] = []
            for tau2 in tail_slices:
                slice_records = grouped[(x, replicate, "tail-slice", float(tau2))]
                slice_deltas.append(_mean_complex(slice_records, "delta_sample"))
                slice_errors.append(
                    float(
                        np.mean(
                            [
                                record["new_sample_absolute_momentum_error"]
                                for record in slice_records
                            ]
                        )
                    )
                )
            coefficients = fit_tau_integrand_tail(
                tail_slices, np.asarray(slice_deltas)
            )
            tail_delta = integrated_fitted_tail(tail_start, coefficients)
            correction = bulk_delta + tail_delta
            corrected = saved_replicates[replicate] + correction
            tail_error = float(
                np.dot(np.abs(tail_error_weights), np.asarray(slice_errors))
            )
            correction_replicates.append(correction)
            corrected_replicates.append(corrected)
            momentum_error_replicates.append(bulk_error + tail_error)
            rows.append(
                {
                    "x": x,
                    "replicate": replicate,
                    "saved_old_final_real": saved_replicates[replicate].real,
                    "momentum_correction_real": correction.real,
                    "corrected_final_real": corrected.real,
                    "momentum_systematic_bound": bulk_error + tail_error,
                }
            )
        corrected_array = np.asarray(corrected_replicates)
        correction_array = np.asarray(correction_replicates)
        corrected_mean = complex(np.mean(corrected_array))
        corrected_se = float(
            np.std(corrected_array.real, ddof=1) / math.sqrt(replicate_count)
        )
        correction_mean = complex(np.mean(correction_array))
        correction_se = float(
            np.std(correction_array.real, ddof=1) / math.sqrt(replicate_count)
        )
        saved_mean = _record_complex(saved["cusp_fit"]["final_I"])
        summaries.append(
            {
                "x": x,
                "saved_old_I1": _complex_record(saved_mean),
                "mean_momentum_correction": _complex_record(correction_mean),
                "momentum_correction_rqmc_se": correction_se,
                "corrected_I1": _complex_record(corrected_mean),
                "corrected_rqmc_se": corrected_se,
                "momentum_systematic_bound": float(
                    max(momentum_error_replicates)
                ),
                "relative_shift": corrected_mean.real / saved_mean.real - 1.0,
                "replicate_corrections": [
                    _complex_record(value) for value in correction_replicates
                ],
                "replicate_corrected_values": [
                    _complex_record(value) for value in corrected_replicates
                ],
            }
        )

    channel_counts: dict[str, int] = defaultdict(int)
    unconverged_bulk = 0
    unconverged_disc = 0
    for record in records:
        channel_counts[str(record["channel_used"])] += 1
        unconverged_bulk += not bool(record["bulk_converged"])
        unconverged_disc += not bool(record["disc_converged"])
    payload = {
        "calculation": "paired asymptotic-momentum correction of genus-one two-point amplitude",
        "blind_freeze": True,
        "blind_freeze_statement": (
            "This artifact contains worldsheet estimates and intrinsic numerical "
            "uncertainties only. No matrix-model or literature target is evaluated "
            "by the finalizer."
        ),
        "native_normalization": "A_1^ws(omega)=8*pi^2*i*g_s^2*I_1(omega)",
        "domain": "omega=i*x with x in {0.2,0.4,0.6,0.8} and 0<x<1",
        "target_count": len(targets),
        "channel_counts": dict(channel_counts),
        "unconverged_bulk_node_count": int(unconverged_bulk),
        "unconverged_disc_node_count": int(unconverged_disc),
        "results": summaries,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(out_dir / "summary.json", payload)
    with (out_dir / "replicate_table.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    _atomic_json(out_dir / "RUN_COMPLETE.json", {"status": "complete"})
    return payload


def parser() -> argparse.ArgumentParser:
    out = argparse.ArgumentParser(description=__doc__)
    subparsers = out.add_subparsers(dest="mode", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--manifest", type=Path, required=True)
    prepare.add_argument("--summary", type=Path, required=True)
    prepare.add_argument("--x-values", type=float, nargs="+", default=DEFAULT_X_VALUES)
    prepare.add_argument("--replicates", type=int, default=8)
    prepare.add_argument("--bulk-sobol-power", type=int, default=8)
    prepare.add_argument("--tail-sobol-power", type=int, default=8)
    prepare.add_argument("--seed", type=int, default=170507151)
    prepare.add_argument("--cutoff", type=float, default=8.0)
    prepare.add_argument("--tail-slices", type=float, nargs="+", default=DEFAULT_TAIL_SLICES)

    evaluate = subparsers.add_parser("evaluate-range")
    evaluate.add_argument("--manifest", type=Path, required=True)
    evaluate.add_argument("--output-dir", type=Path, required=True)
    evaluate.add_argument("--start", type=int, required=True)
    evaluate.add_argument("--end", type=int, required=True)
    evaluate.add_argument(
        "--polar-orders",
        nargs="+",
        default=[f"{a}x{b}" for a, b in DEFAULT_POLAR_ORDERS],
    )
    evaluate.add_argument(
        "--disc-orders",
        nargs="+",
        default=[f"{a}x{b}" for a, b in DEFAULT_DISC_ORDERS],
    )
    evaluate.add_argument("--tolerance", type=float, default=5.0e-5)
    evaluate.add_argument("--epsilon", type=float, default=0.15)
    evaluate.add_argument("--collision-radius", type=float, default=0.10)
    evaluate.add_argument("--old-order", type=int, default=16)
    evaluate.add_argument("--p-max", type=float, default=6.0)
    evaluate.add_argument("--power", type=float, default=2.0)
    evaluate.add_argument("--necklace-orders", default="6,3")
    evaluate.add_argument("--ope-orders", default="3,8")
    evaluate.add_argument("--dps", type=int, default=28)

    final = subparsers.add_parser("finalize")
    final.add_argument("--manifest", type=Path, required=True)
    final.add_argument("--evaluation-dir", type=Path, required=True)
    final.add_argument("--saved-result-dir", type=Path, required=True)
    final.add_argument("--out-dir", type=Path, required=True)
    return out


def main(argv: Iterable[str] | None = None) -> None:
    args = parser().parse_args(list(argv) if argv is not None else None)
    if args.mode == "prepare":
        result = prepare_manifest(
            path=args.manifest,
            summary_path=args.summary,
            x_values=args.x_values,
            replicates=args.replicates,
            bulk_sobol_power=args.bulk_sobol_power,
            tail_sobol_power=args.tail_sobol_power,
            seed=args.seed,
            cutoff=args.cutoff,
            tail_slices=args.tail_slices,
        )
        print(json.dumps(result, indent=2))
    elif args.mode == "evaluate-range":
        evaluate_range(
            manifest_path=args.manifest,
            output_dir=args.output_dir,
            start=args.start,
            end=args.end,
            polar_orders=_parse_pairs(args.polar_orders),
            disc_orders=_parse_pairs(args.disc_orders),
            tolerance=args.tolerance,
            epsilon=args.epsilon,
            collision_radius=args.collision_radius,
            old_order=args.old_order,
            p_max=args.p_max,
            power=args.power,
            necklace_orders=tuple(int(value) for value in args.necklace_orders.split(",")),  # type: ignore[arg-type]
            ope_orders=tuple(int(value) for value in args.ope_orders.split(",")),  # type: ignore[arg-type]
            dps=args.dps,
        )
    else:
        result = finalize(
            manifest_path=args.manifest,
            evaluation_dir=args.evaluation_dir,
            saved_result_dir=args.saved_result_dir,
            out_dir=args.out_dir,
        )
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
