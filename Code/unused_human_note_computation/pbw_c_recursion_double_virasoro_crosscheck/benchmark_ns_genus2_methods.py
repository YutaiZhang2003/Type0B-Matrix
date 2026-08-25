#!/usr/bin/env python3
"""Benchmark convergence of NS c-recursion and double Virasoro.

Each timing sample is taken in a fresh Python process so memoized recursion
data from one cutoff cannot make a later cutoff artificially cheap.  On the
c-recursion side the numerical ``q`` values, lift, and sector are inserted
before the functional recursion; no coefficient table is built.  On the
double-Virasoro side the present implementation constructs the truncated
series needed for the Majorana ``star`` inverse.

Convergence at cutoff N means that every mutually validated truncation from N
through physical level four agrees with an independent NS c-recursion
reference at physical level eight to the requested normalized tolerance.  The
level-eight NS vacuum seed is independently extracted in
``ns_genus12_finite_c_check.py``; it is not built from the double-Virasoro
answer.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from itertools import product
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import time
from typing import Mapping, Sequence

from compare_ns_genus2_double_virasoro import (
    DEFAULT_SAMPLES,
    Sample,
    auxiliary_majorana_series,
    divide_theta_star_series,
    double_virasoro_enlarged_series,
    evaluated_sector,
)
from ns_genus2_pointwise_c_recursion import PointwiseHumanThetaCRecursion


LIFTS = tuple(product((-1, 1), repeat=3))
SECTORS = (0, 1)
ValueKey = tuple[int, int, int, int]


def _background_data(sample: Sample) -> tuple[float, tuple[float, float, float]]:
    q_background = sample.b + 1.0 / sample.b
    central_charge = 1.5 + 3.0 * q_background * q_background
    weights = tuple(
        q_background * q_background / 8.0 - momentum * momentum / 2.0
        for momentum in sample.momenta
    )
    return central_charge, weights


def compute_method(
    method: str,
    *,
    sample: Sample,
    maximum_total_twice_level: int,
    q_values: Sequence[float],
    precision: int,
) -> tuple[float, float, dict[ValueKey, complex]]:
    """Evaluate one method and return cold-single and full-batch timings."""

    cutoff = int(maximum_total_twice_level)
    central_charge, weights = _background_data(sample)
    canonical_key = (1, 1, 1, 0)
    if method == "c_recursion":
        evaluator = PointwiseHumanThetaCRecursion(
            q_values=q_values,
            global_tolerance=1.0e-14,
            global_max_total_occupation=24,
            vacuum_word_length=8,
            vacuum_max_mode=50,
        )
        start = time.perf_counter()
        canonical_value = evaluator.block(
            central_charge=central_charge,
            weights=weights,
            sector=canonical_key[3],
            recursion_order=cutoff,
            lifts=canonical_key[:3],
        )
        single_elapsed = time.perf_counter() - start
        values = {canonical_key: canonical_value}
        for lifts in LIFTS:
            for sector in SECTORS:
                key = (*lifts, sector)
                if key == canonical_key:
                    continue
                values[key] = evaluator.block(
                    central_charge=central_charge,
                    weights=weights,
                    sector=sector,
                    recursion_order=cutoff,
                    lifts=lifts,
                )
    elif method == "double_virasoro":
        start = time.perf_counter()
        hatted = double_virasoro_enlarged_series(
            b=sample.b,
            momenta=sample.momenta,
            cutoff=cutoff,
            precision=precision,
        )
        majorana = auxiliary_majorana_series(cutoff=cutoff)
        series = divide_theta_star_series(hatted, majorana, cutoff=cutoff)
        canonical_value = evaluated_sector(
            series,
            q_values=q_values,
            lifts=canonical_key[:3],
            sector=canonical_key[3],
        )
        single_elapsed = time.perf_counter() - start
        values = {canonical_key: canonical_value}
        for lifts in LIFTS:
            for sector in SECTORS:
                key = (*lifts, sector)
                if key == canonical_key:
                    continue
                values[key] = evaluated_sector(
                    series,
                    q_values=q_values,
                    lifts=lifts,
                    sector=sector,
                )
    else:
        raise ValueError(f"unknown method {method!r}")
    batch_elapsed = time.perf_counter() - start
    return single_elapsed, batch_elapsed, values


def normalized_error(left: complex, right: complex) -> float:
    return float(abs(left - right) / max(1.0, abs(left), abs(right)))


def maximum_value_error(
    left: Mapping[ValueKey, complex], right: Mapping[ValueKey, complex]
) -> float:
    return max(normalized_error(left[key], right[key]) for key in left)


def first_stable_cutoff(
    errors_to_reference: Sequence[float], *, tolerance: float
) -> int | None:
    """Return the first cutoff whose whole remaining tail is within tolerance."""

    suffix_maximum = 0.0
    stable: int | None = None
    for cutoff in range(len(errors_to_reference) - 1, -1, -1):
        suffix_maximum = max(suffix_maximum, errors_to_reference[cutoff])
        if suffix_maximum <= tolerance:
            stable = cutoff
    return stable


def _encode_values(values: Mapping[ValueKey, complex]) -> list[dict[str, object]]:
    return [
        {
            "lifts": list(key[:3]),
            "sector": key[3],
            "real": value.real,
            "imag": value.imag,
        }
        for key, value in values.items()
    ]


def _decode_values(payload: Sequence[Mapping[str, object]]) -> dict[ValueKey, complex]:
    result: dict[ValueKey, complex] = {}
    for item in payload:
        lifts = tuple(int(value) for value in item["lifts"])
        key = (*lifts, int(item["sector"]))
        result[key] = complex(float(item["real"]), float(item["imag"]))
    return result


def _fresh_worker(
    method: str,
    *,
    sample: Sample,
    cutoff: int,
    q_values: Sequence[float],
    precision: int,
) -> tuple[float, float, dict[ValueKey, complex]]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        method,
        "--worker-cutoff",
        str(cutoff),
        "--worker-b",
        repr(sample.b),
        "--worker-momenta",
        *(repr(value) for value in sample.momenta),
        "--q-values",
        *(repr(value) for value in q_values),
        "--precision",
        str(precision),
    ]
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    return (
        float(payload["single_seconds"]),
        float(payload["batch_seconds"]),
        _decode_values(payload["values"]),
    )


@dataclass(frozen=True)
class OrderResult:
    maximum_total_twice_level: int
    maximum_total_physical_level: float
    double_virasoro_coefficient_count: int
    c_recursion_single_value_median_seconds: float
    double_virasoro_single_value_median_seconds: float
    c_recursion_all_lifts_sectors_median_seconds: float
    double_virasoro_all_lifts_sectors_median_seconds: float
    c_recursion_error_to_reference: float
    double_virasoro_error_to_reference: float
    cross_method_error: float


@dataclass(frozen=True)
class EfficiencySampleResult:
    b: float
    momenta: tuple[float, float, float]
    c_recursion_first_stable_twice_level: int | None
    double_virasoro_first_stable_twice_level: int | None
    common_first_stable_twice_level: int | None
    c_recursion_seconds_at_first_stable: float | None
    double_virasoro_seconds_at_first_stable: float | None
    speed_ratio_double_virasoro_over_c_recursion: float | None
    c_recursion_batch_seconds_at_first_stable: float | None
    double_virasoro_batch_seconds_at_first_stable: float | None
    batch_speed_ratio_double_virasoro_over_c_recursion: float | None
    cross_method_error_at_respective_stable_orders: float | None
    cross_method_error_at_common_stable_order: float | None
    c_recursion_reference_seconds: float
    plus_plus_plus_even_c_recursion: tuple[float, float] | None
    plus_plus_plus_even_double_virasoro: tuple[float, float] | None
    plus_plus_plus_odd_c_recursion: tuple[float, float] | None
    plus_plus_plus_odd_double_virasoro: tuple[float, float] | None
    orders: tuple[OrderResult, ...]


@dataclass(frozen=True)
class EfficiencyResult:
    tolerance: float
    comparison_maximum_total_twice_level: int
    comparison_maximum_total_physical_level: float
    reference_maximum_total_twice_level: int
    reference_maximum_total_physical_level: float
    q_values: tuple[float, float, float]
    timing_repeats: int
    timing_protocol: str
    samples: tuple[EfficiencySampleResult, ...]


def benchmark_sample(
    sample: Sample,
    *,
    comparison_cutoff: int,
    reference_cutoff: int,
    q_values: Sequence[float],
    tolerance: float,
    timing_repeats: int,
    precision: int,
) -> EfficiencySampleResult:
    method_values: dict[str, list[dict[ValueKey, complex]]] = {
        "c_recursion": [],
        "double_virasoro": [],
    }
    method_single_times: dict[str, list[float]] = {
        "c_recursion": [],
        "double_virasoro": [],
    }
    method_batch_times: dict[str, list[float]] = {
        "c_recursion": [],
        "double_virasoro": [],
    }
    _, reference_seconds, reference_values = _fresh_worker(
        "c_recursion",
        sample=sample,
        cutoff=reference_cutoff,
        q_values=q_values,
        precision=precision,
    )
    for cutoff in range(comparison_cutoff + 1):
        for method in ("c_recursion", "double_virasoro"):
            single_timings: list[float] = []
            batch_timings: list[float] = []
            canonical_values: dict[ValueKey, complex] | None = None
            for _ in range(timing_repeats):
                single_elapsed, batch_elapsed, values = _fresh_worker(
                    method,
                    sample=sample,
                    cutoff=cutoff,
                    q_values=q_values,
                    precision=precision,
                )
                single_timings.append(single_elapsed)
                batch_timings.append(batch_elapsed)
                if canonical_values is None:
                    canonical_values = values
                elif maximum_value_error(canonical_values, values) > 1.0e-13:
                    raise AssertionError("fresh-process evaluations were not reproducible")
            method_single_times[method].append(
                statistics.median(single_timings)
            )
            method_batch_times[method].append(
                statistics.median(batch_timings)
            )
            assert canonical_values is not None
            method_values[method].append(canonical_values)

    errors: dict[str, list[float]] = {}
    stable_cutoffs: dict[str, int | None] = {}
    for method in ("c_recursion", "double_virasoro"):
        errors[method] = [
            maximum_value_error(values, reference_values)
            for values in method_values[method]
        ]
        stable_cutoffs[method] = first_stable_cutoff(
            errors[method], tolerance=tolerance
        )

    cross_errors = [
        maximum_value_error(
            method_values["c_recursion"][cutoff],
            method_values["double_virasoro"][cutoff],
        )
        for cutoff in range(comparison_cutoff + 1)
    ]
    orders = tuple(
        OrderResult(
            maximum_total_twice_level=cutoff,
            maximum_total_physical_level=cutoff / 2.0,
            double_virasoro_coefficient_count=math.comb(cutoff + 3, 3),
            c_recursion_single_value_median_seconds=(
                method_single_times["c_recursion"][cutoff]
            ),
            double_virasoro_single_value_median_seconds=(
                method_single_times["double_virasoro"][cutoff]
            ),
            c_recursion_all_lifts_sectors_median_seconds=(
                method_batch_times["c_recursion"][cutoff]
            ),
            double_virasoro_all_lifts_sectors_median_seconds=(
                method_batch_times["double_virasoro"][cutoff]
            ),
            c_recursion_error_to_reference=errors["c_recursion"][cutoff],
            double_virasoro_error_to_reference=errors["double_virasoro"][cutoff],
            cross_method_error=cross_errors[cutoff],
        )
        for cutoff in range(comparison_cutoff + 1)
    )

    c_stable = stable_cutoffs["c_recursion"]
    d_stable = stable_cutoffs["double_virasoro"]
    common = None if c_stable is None or d_stable is None else max(c_stable, d_stable)
    c_seconds = (
        None
        if c_stable is None
        else method_single_times["c_recursion"][c_stable]
    )
    d_seconds = (
        None
        if d_stable is None
        else method_single_times["double_virasoro"][d_stable]
    )
    ratio = (
        None
        if c_seconds is None or d_seconds is None or c_seconds == 0
        else d_seconds / c_seconds
    )
    c_batch_seconds = (
        None
        if c_stable is None
        else method_batch_times["c_recursion"][c_stable]
    )
    d_batch_seconds = (
        None
        if d_stable is None
        else method_batch_times["double_virasoro"][d_stable]
    )
    batch_ratio = (
        None
        if c_batch_seconds is None
        or d_batch_seconds is None
        or c_batch_seconds == 0
        else d_batch_seconds / c_batch_seconds
    )

    def displayed(method: str, sector: int) -> tuple[float, float] | None:
        if common is None:
            return None
        value = method_values[method][common][(1, 1, 1, sector)]
        return value.real, value.imag

    return EfficiencySampleResult(
        b=sample.b,
        momenta=sample.momenta,
        c_recursion_first_stable_twice_level=c_stable,
        double_virasoro_first_stable_twice_level=d_stable,
        common_first_stable_twice_level=common,
        c_recursion_seconds_at_first_stable=c_seconds,
        double_virasoro_seconds_at_first_stable=d_seconds,
        speed_ratio_double_virasoro_over_c_recursion=ratio,
        c_recursion_batch_seconds_at_first_stable=c_batch_seconds,
        double_virasoro_batch_seconds_at_first_stable=d_batch_seconds,
        batch_speed_ratio_double_virasoro_over_c_recursion=batch_ratio,
        cross_method_error_at_respective_stable_orders=(
            None
            if c_stable is None or d_stable is None
            else maximum_value_error(
                method_values["c_recursion"][c_stable],
                method_values["double_virasoro"][d_stable],
            )
        ),
        cross_method_error_at_common_stable_order=(
            None if common is None else cross_errors[common]
        ),
        c_recursion_reference_seconds=reference_seconds,
        plus_plus_plus_even_c_recursion=displayed("c_recursion", 0),
        plus_plus_plus_even_double_virasoro=displayed("double_virasoro", 0),
        plus_plus_plus_odd_c_recursion=displayed("c_recursion", 1),
        plus_plus_plus_odd_double_virasoro=displayed("double_virasoro", 1),
        orders=orders,
    )


def run_benchmark(
    *,
    samples: Sequence[Sample] = DEFAULT_SAMPLES,
    q_values: Sequence[float] = (0.013, 0.017, 0.011),
    tolerance: float = 1.0e-6,
    comparison_maximum_total_physical_level: int = 4,
    reference_maximum_total_physical_level: int = 8,
    timing_repeats: int = 3,
    precision: int = 70,
) -> EfficiencyResult:
    comparison_cutoff = 2 * int(comparison_maximum_total_physical_level)
    reference_cutoff = 2 * int(reference_maximum_total_physical_level)
    if comparison_cutoff < 0 or comparison_cutoff > 8:
        raise ValueError("the mutually validated comparison level is at most four")
    if reference_cutoff < comparison_cutoff or reference_cutoff > 16:
        raise ValueError(
            "the independent NS reference must lie between the comparison "
            "cutoff and physical level eight"
        )
    if timing_repeats < 1:
        raise ValueError("at least one timing repeat is required")
    if tolerance <= 0:
        raise ValueError("the convergence tolerance must be positive")
    results = tuple(
        benchmark_sample(
            sample,
            comparison_cutoff=comparison_cutoff,
            reference_cutoff=reference_cutoff,
            q_values=q_values,
            tolerance=tolerance,
            timing_repeats=timing_repeats,
            precision=precision,
        )
        for sample in samples
    )
    return EfficiencyResult(
        tolerance=tolerance,
        comparison_maximum_total_twice_level=comparison_cutoff,
        comparison_maximum_total_physical_level=comparison_cutoff / 2.0,
        reference_maximum_total_twice_level=reference_cutoff,
        reference_maximum_total_physical_level=reference_cutoff / 2.0,
        q_values=tuple(float(value) for value in q_values),
        timing_repeats=timing_repeats,
        timing_protocol=(
            "median fresh-process wall time; cold single value is (+++, even); "
            "batch timing evaluates 8 lifts x 2 sectors with within-process caches"
        ),
        samples=results,
    )


def _worker_main(args: argparse.Namespace) -> None:
    sample = Sample(
        b=float(args.worker_b),
        momenta=tuple(float(value) for value in args.worker_momenta),
    )
    single_elapsed, batch_elapsed, values = compute_method(
        args.worker,
        sample=sample,
        maximum_total_twice_level=args.worker_cutoff,
        q_values=args.q_values,
        precision=args.precision,
    )
    print(
        json.dumps(
            {
                "single_seconds": single_elapsed,
                "batch_seconds": batch_elapsed,
                "values": _encode_values(values),
            }
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tolerance", type=float, default=1.0e-6)
    parser.add_argument("--comparison-level", type=int, default=4)
    parser.add_argument("--reference-level", type=int, default=8)
    parser.add_argument("--timing-repeats", type=int, default=3)
    parser.add_argument("--precision", type=int, default=70)
    parser.add_argument(
        "--q-values", type=float, nargs=3, default=(0.013, 0.017, 0.011)
    )
    parser.add_argument("--sample", type=int, choices=(1, 2), action="append")
    parser.add_argument("--json", action="store_true")
    parser.add_argument(
        "--worker", choices=("c_recursion", "double_virasoro"), help=argparse.SUPPRESS
    )
    parser.add_argument("--worker-cutoff", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--worker-b", type=float, help=argparse.SUPPRESS)
    parser.add_argument(
        "--worker-momenta", type=float, nargs=3, help=argparse.SUPPRESS
    )
    args = parser.parse_args()
    if args.worker is not None:
        _worker_main(args)
        return

    selected = DEFAULT_SAMPLES
    if args.sample:
        selected = tuple(DEFAULT_SAMPLES[index - 1] for index in args.sample)
    result = run_benchmark(
        samples=selected,
        q_values=args.q_values,
        tolerance=args.tolerance,
        comparison_maximum_total_physical_level=args.comparison_level,
        reference_maximum_total_physical_level=args.reference_level,
        timing_repeats=args.timing_repeats,
        precision=args.precision,
    )
    if args.json:
        print(json.dumps(asdict(result), indent=2))
        return

    print("NS genus-two convergence and efficiency benchmark")
    print(
        f"  tolerance={result.tolerance:.1e}, q={result.q_values}, "
        f"comparison/reference levels="
        f"{result.comparison_maximum_total_physical_level:g}/"
        f"{result.reference_maximum_total_physical_level:g}"
    )
    for index, sample in enumerate(result.samples, start=1):
        print(f"  sample {index}: b={sample.b}, P={sample.momenta}")
        print(
            "    first stable twice-level (c-recursion / double Virasoro): "
            f"{sample.c_recursion_first_stable_twice_level}/"
            f"{sample.double_virasoro_first_stable_twice_level}"
        )
        print(
            "    cold (+++,even) time (c-recursion / double Virasoro): "
            f"{sample.c_recursion_seconds_at_first_stable:.6f}s/"
            f"{sample.double_virasoro_seconds_at_first_stable:.6f}s"
        )
        print(
            "    double-Virasoro / c-recursion time ratio: "
            f"{sample.speed_ratio_double_virasoro_over_c_recursion:.3f}"
        )
        print(
            "    all-lift/sector batch time (c-recursion / double Virasoro): "
            f"{sample.c_recursion_batch_seconds_at_first_stable:.6f}s/"
            f"{sample.double_virasoro_batch_seconds_at_first_stable:.6f}s"
        )
        print(
            "    cross-method error at common stable order: "
            f"{sample.cross_method_error_at_common_stable_order:.3e}"
        )


if __name__ == "__main__":
    main()
