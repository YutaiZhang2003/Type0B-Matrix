#!/usr/bin/env python3
"""Blind RQMC integration of the genus-one c=1 1->2 worldsheet amplitude.

This first local design follows the equal-split imaginary ray

    omega_in = i*t,   omega_out,1 = omega_out,2 = i*t/2,   0<t<1.

The estimator uses only the three-edge Liouville necklace block.  Matrix-model
and literature amplitudes are intentionally absent; comparison belongs in a
separate post-freeze program.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import qmc

try:
    from genus1_three_point_worldsheet import (
        LiouvilleTorusThreePointNecklace,
        reduced_worldsheet_integrand_three_point,
    )
    from genus1_two_point_worldsheet import MomentumRule
except ImportError:  # pragma: no cover
    from plumbing.genus1_three_point_worldsheet import (
        LiouvilleTorusThreePointNecklace,
        reduced_worldsheet_integrand_three_point,
    )
    from plumbing.genus1_two_point_worldsheet import MomentumRule


def _complex_record(value: complex) -> dict[str, float]:
    value = complex(value)
    return {"real": float(value.real), "imag": float(value.imag)}


def fit_cusp_limit(
    cutoffs: np.ndarray,
    estimates: np.ndarray,
) -> tuple[complex, complex, complex]:
    """Exploratory target-free fit ``I(T)=I_inf+a/T+b/T^2``."""
    design = np.column_stack(
        [np.ones_like(cutoffs), 1.0 / cutoffs, 1.0 / (cutoffs * cutoffs)]
    )
    real_fit, *_ = np.linalg.lstsq(design, estimates.real, rcond=None)
    imag_fit, *_ = np.linalg.lstsq(design, estimates.imag, rcond=None)
    combined = real_fit + 1.0j * imag_fit
    return complex(combined[0]), complex(combined[1]), complex(combined[2])


def integrate_cutoffs_one_replicate(
    correlator: LiouvilleTorusThreePointNecklace,
    *,
    cutoffs: tuple[float, ...],
    sobol_power: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Integrate tau and both labeled outgoing punctures for one scramble."""
    sampler = qmc.Sobol(d=6, scramble=True, seed=int(seed))
    points = sampler.random_base2(int(sobol_power))
    estimates = np.zeros(len(cutoffs), dtype=np.complex128)
    lower_order_estimates = np.zeros(len(cutoffs), dtype=np.complex128)
    maximum_absolute_sample = 0.0

    for cutoff_index, cutoff in enumerate(cutoffs):
        values = np.zeros(len(points), dtype=np.complex128)
        lower_values = np.zeros(len(points), dtype=np.complex128)
        for point_index, point in enumerate(points):
            tau1 = float(point[0]) - 0.5
            tau2_min = math.sqrt(1.0 - tau1 * tau1)
            tau2 = tau2_min + float(point[1]) * (float(cutoff) - tau2_min)
            tau = tau1 + 1.0j * tau2
            w1 = 2.0 * math.pi * (float(point[2]) + float(point[3]) * tau)
            w2 = 2.0 * math.pi * (float(point[4]) + float(point[5]) * tau)
            position_jacobian = (2.0 * math.pi) ** 4 * tau2 * tau2
            tau_jacobian = float(cutoff) - tau2_min
            common_jacobian = tau_jacobian * position_jacobian

            main = reduced_worldsheet_integrand_three_point(
                correlator,
                w1,
                w2,
                tau,
            )
            lower = reduced_worldsheet_integrand_three_point(
                correlator,
                w1,
                w2,
                tau,
                order_cap=correlator.high_order - 1,
                record_diagnostics=False,
            )
            values[point_index] = common_jacobian * main
            lower_values[point_index] = common_jacobian * lower

        estimates[cutoff_index] = np.mean(values)
        lower_order_estimates[cutoff_index] = np.mean(lower_values)
        maximum_absolute_sample = max(
            maximum_absolute_sample,
            float(np.max(np.abs(values))),
        )

    diagnostics = {
        "maximum_absolute_sample": maximum_absolute_sample,
        "points_per_cutoff": float(len(points)),
    }
    return estimates, lower_order_estimates, diagnostics


def integrate_tau_slice_one_replicate(
    correlator: LiouvilleTorusThreePointNecklace,
    *,
    tau2: float,
    sobol_power: int,
    seed: int,
) -> complex:
    """Integrate tau1 and both punctures at one fixed cusp height."""
    sampler = qmc.Sobol(d=5, scramble=True, seed=int(seed))
    points = sampler.random_base2(int(sobol_power))
    values = np.zeros(len(points), dtype=np.complex128)
    tau2 = float(tau2)
    for point_index, point in enumerate(points):
        tau = (float(point[0]) - 0.5) + 1.0j * tau2
        w1 = 2.0 * math.pi * (float(point[1]) + float(point[2]) * tau)
        w2 = 2.0 * math.pi * (float(point[3]) + float(point[4]) * tau)
        position_jacobian = (2.0 * math.pi) ** 4 * tau2 * tau2
        values[point_index] = position_jacobian * reduced_worldsheet_integrand_three_point(
            correlator,
            w1,
            w2,
            tau,
        )
    return complex(np.mean(values))


def fit_tail_power(
    tau2_values: np.ndarray,
    slice_values: np.ndarray,
    *,
    exponent: float | None = None,
) -> tuple[float, complex, complex, float]:
    r"""Fit ``A*t^-p+B*t^(-p-1)`` with a common real exponent ``p>1``."""
    tau2_values = np.asarray(tau2_values, dtype=float)
    slice_values = np.asarray(slice_values, dtype=complex)

    def coefficients_at(power: float) -> tuple[complex, complex, float]:
        design = np.column_stack(
            [tau2_values ** (-power), tau2_values ** (-power - 1.0)]
        )
        real_fit, *_ = np.linalg.lstsq(design, slice_values.real, rcond=None)
        imag_fit, *_ = np.linalg.lstsq(design, slice_values.imag, rcond=None)
        coefficients = real_fit + 1.0j * imag_fit
        residual = slice_values - design @ coefficients
        relative_residual = float(
            np.linalg.norm(residual) / max(np.linalg.norm(slice_values), 1.0e-300)
        )
        return complex(coefficients[0]), complex(coefficients[1]), relative_residual

    if exponent is None:
        optimization = minimize_scalar(
            lambda power: coefficients_at(float(power))[2],
            bounds=(1.05, 5.0),
            method="bounded",
            options={"xatol": 1.0e-6},
        )
        exponent = float(optimization.x)
    exponent = float(exponent)
    first, second, residual = coefficients_at(exponent)
    return exponent, first, second, residual


def integrated_power_tail(
    tail_start: float,
    exponent: float,
    first: complex,
    second: complex,
) -> complex:
    """Integrate ``A*t^-p+B*t^(-p-1)`` from ``tail_start`` to infinity."""
    if exponent <= 1.0:
        raise ValueError("the fitted cusp exponent must exceed one")
    tail_start = float(tail_start)
    return (
        first * tail_start ** (1.0 - exponent) / (exponent - 1.0)
        + second * tail_start ** (-exponent) / exponent
    )


def run(args: argparse.Namespace) -> dict[str, object]:
    if not 0.0 < args.t < 1.0:
        raise ValueError("the direct equal-split real-contour computation requires 0<t<1")
    cutoffs = tuple(float(piece) for piece in args.cutoffs.split(",") if piece.strip())
    if len(cutoffs) < 3 or tuple(sorted(cutoffs)) != cutoffs:
        raise ValueError("provide at least three increasing cusp cutoffs")
    if cutoffs[0] <= 1.0:
        raise ValueError("the smallest cusp cutoff must exceed one")

    momentum_powers: tuple[float, ...] | None = None
    reference_log_q_abs: tuple[float, ...] | None = None
    if args.momentum_kind == "power-legendre":
        momentum_powers = tuple(
            float(args.momentum_power + edge * args.momentum_power_step)
            for edge in range(3)
        )
        momentum_rules = tuple(
            MomentumRule.power_legendre(
                args.p_max,
                args.momentum_order,
                power,
            )
            for power in momentum_powers
        )
    elif args.momentum_kind == "threshold-gaussian":
        reference_log_q_abs = tuple(
            float(args.reference_log_q_abs - edge * args.reference_log_q_step)
            for edge in range(3)
        )
        if any(value >= 0.0 for value in reference_log_q_abs):
            raise ValueError("reference log|q| values must be negative")
        momentum_rules = tuple(
            MomentumRule.threshold_gaussian(
                math.exp(log_q_abs),
                args.momentum_order,
                log_q_abs=log_q_abs,
            )
            for log_q_abs in reference_log_q_abs
        )
    else:  # pragma: no cover - guarded by argparse
        raise ValueError(f"unknown momentum quadrature kind {args.momentum_kind!r}")
    correlator = LiouvilleTorusThreePointNecklace(
        args.t,
        momentum_rules=momentum_rules,
        high_order=args.high_order,
        low_order=args.low_order,
        adaptive_tolerance=args.block_tolerance,
        c_regulator=args.c_regulator,
        block_backend=args.block_backend,
        special_dps=args.dps,
    )
    correlator.prepare()

    replicate_values: list[np.ndarray] = []
    replicate_lower_values: list[np.ndarray] = []
    replicate_diagnostics: list[dict[str, float]] = []
    for replicate in range(args.replicates):
        values, lower_values, diagnostics = integrate_cutoffs_one_replicate(
            correlator,
            cutoffs=cutoffs,
            sobol_power=args.sobol_power,
            seed=args.seed + replicate,
        )
        replicate_values.append(values)
        replicate_lower_values.append(lower_values)
        replicate_diagnostics.append(diagnostics)
        print(
            f"replicate {replicate + 1}/{args.replicates}: "
            + ", ".join(
                f"T={cutoff:g}: {value.real:+.8e}{value.imag:+.2e}j"
                for cutoff, value in zip(cutoffs, values)
            ),
            flush=True,
        )

    tail_slices = tuple(
        float(piece) for piece in args.tail_slices.split(",") if piece.strip()
    )
    if len(tail_slices) < 4 or min(tail_slices) < cutoffs[-1]:
        raise ValueError(
            "tail slices must contain at least four values at or above the last cutoff"
        )
    replicate_slice_values: list[np.ndarray] = []
    for replicate in range(args.replicates):
        slices = np.asarray(
            [
                integrate_tau_slice_one_replicate(
                    correlator,
                    tau2=tau2,
                    sobol_power=args.tail_sobol_power,
                    seed=args.seed + 10000 + 97 * replicate + slice_index,
                )
                for slice_index, tau2 in enumerate(tail_slices)
            ]
        )
        replicate_slice_values.append(slices)
        print(
            f"tail replicate {replicate + 1}/{args.replicates}: "
            + ", ".join(
                f"tau2={tau2:g}: {value.real:+.3e}{value.imag:+.3e}j"
                for tau2, value in zip(tail_slices, slices)
            ),
            flush=True,
        )

    replicate_array = np.asarray(replicate_values)
    replicate_lower_array = np.asarray(replicate_lower_values)
    means = np.mean(replicate_array, axis=0)
    lower_means = np.mean(replicate_lower_array, axis=0)
    if args.replicates > 1:
        standard_errors = (
            np.std(replicate_array.real, axis=0, ddof=1)
            + 1.0j * np.std(replicate_array.imag, axis=0, ddof=1)
        ) / math.sqrt(args.replicates)
    else:
        standard_errors = np.full(len(cutoffs), np.nan + 1.0j * np.nan)

    fitted_replicates = np.asarray(
        [fit_cusp_limit(np.asarray(cutoffs), row)[0] for row in replicate_array]
    )
    lower_fitted_replicates = np.asarray(
        [fit_cusp_limit(np.asarray(cutoffs), row)[0] for row in replicate_lower_array]
    )
    fitted_limit, fitted_a, fitted_b = fit_cusp_limit(np.asarray(cutoffs), means)
    lower_fitted_limit, _, _ = fit_cusp_limit(np.asarray(cutoffs), lower_means)
    if args.replicates > 1:
        fitted_standard_error = complex(
            float(np.std(fitted_replicates.real, ddof=1) / math.sqrt(args.replicates)),
            float(np.std(fitted_replicates.imag, ddof=1) / math.sqrt(args.replicates)),
        )
    else:
        fitted_standard_error = complex(float("nan"), float("nan"))

    slice_array = np.asarray(replicate_slice_values)
    slice_means = np.mean(slice_array, axis=0)
    tail_exponent, tail_first, tail_second, tail_relative_residual = fit_tail_power(
        np.asarray(tail_slices),
        slice_means,
    )
    replicate_tail_integrals: list[complex] = []
    replicate_final_values: list[complex] = []
    replicate_tail_residuals: list[float] = []
    for values, slices in zip(replicate_array, slice_array):
        _, first, second, residual = fit_tail_power(
            np.asarray(tail_slices),
            slices,
            exponent=tail_exponent,
        )
        tail_integral = integrated_power_tail(
            cutoffs[-1],
            tail_exponent,
            first,
            second,
        )
        replicate_tail_integrals.append(tail_integral)
        replicate_final_values.append(complex(values[-1] + tail_integral))
        replicate_tail_residuals.append(residual)
    final_mean = complex(np.mean(replicate_final_values))
    tail_integral_mean = complex(np.mean(replicate_tail_integrals))
    if args.replicates > 1:
        final_standard_error = complex(
            float(
                np.std(np.asarray(replicate_final_values).real, ddof=1)
                / math.sqrt(args.replicates)
            ),
            float(
                np.std(np.asarray(replicate_final_values).imag, ddof=1)
                / math.sqrt(args.replicates)
            ),
        )
    else:
        final_standard_error = complex(float("nan"), float("nan"))

    result: dict[str, object] = {
        "calculation": "direct c=1 genus-one three-point worldsheet integral",
        "blind_freeze": True,
        "blind_freeze_statement": (
            "The estimator, h-recursion order selection, momentum quadrature, and "
            "cusp extrapolation contain no matrix-model or literature target."
        ),
        "native_convention": True,
        "native_normalization": (
            "A_1,3^ws(omega1,omega2)=16*pi^2*i*(g_s^Xi)^3*I_1,3(omega1,omega2)"
        ),
        "kinematics": {
            "slice": "equal-split imaginary ray",
            "omega_in": _complex_record(correlator.omega_in),
            "omega_out_1": _complex_record(correlator.omega_out),
            "omega_out_2": _complex_record(correlator.omega_out),
            "t": float(args.t),
            "domain": "0<t<1; real positive Liouville contours, necklace channel only",
        },
        "moduli_measure": (
            "d2tau d2w1 d2w2; w_a=2*pi*(r_a+s_a*tau), "
            "Jacobian=(2*pi)^4*tau2^2"
        ),
        "momentum_rule": {
            "kind": str(args.momentum_kind),
            "p_max": (
                float(args.p_max) if args.momentum_kind == "power-legendre" else None
            ),
            "order_per_edge": int(args.momentum_order),
            "powers_by_edge": (
                None if momentum_powers is None else list(momentum_powers)
            ),
            "reference_log_q_abs_by_edge": (
                None if reference_log_q_abs is None else list(reference_log_q_abs)
            ),
            "distinct_rule_reason": (
                "avoids coincident internal weights in intermediate simultaneous "
                "h-recursion residues; every rule independently integrates [0,p_max]"
            ),
            "measure": "dQ0*dQ1*dQ2/pi^3",
        },
        "block_design": {
            "channel": "single cyclic three-point necklace",
            "backend": str(args.block_backend),
            "formula": (
                "2*F(c=25+epsilon)-F(c=25+2*epsilon)"
                if args.block_backend == "regulated-h-recursion"
                else "finite-level Gram-matrix descendant sewing at c=25"
            ),
            "c_regulator": float(args.c_regulator),
            "elliptic_nomes": "hat_q_i=E(q_i), q_i=lambda(hat_q_i)",
            "high_edge_max_order": int(args.high_order),
            "other_edge_order": int(args.low_order),
            "adaptive_tail_proxy": float(args.block_tolerance),
            "lower_order_comparison_cap": int(args.high_order - 1),
        },
        "rqmc": {
            "dimension": 6,
            "sobol_power": int(args.sobol_power),
            "points_per_cutoff_replicate": int(2**args.sobol_power),
            "replicates": int(args.replicates),
            "seed": int(args.seed),
        },
        "tail_rqmc": {
            "dimension": 5,
            "sobol_power": int(args.tail_sobol_power),
            "points_per_slice_replicate": int(2**args.tail_sobol_power),
            "replicates": int(args.replicates),
            "seed_scheme": "seed+10000+97*replicate+slice_index",
        },
        "special_dps": int(args.dps),
        "cutoffs": list(cutoffs),
        "cutoff_means": [_complex_record(value) for value in means],
        "cutoff_standard_errors": [_complex_record(value) for value in standard_errors],
        "lower_order_cutoff_means": [_complex_record(value) for value in lower_means],
        "replicate_values": [
            [_complex_record(value) for value in row] for row in replicate_array
        ],
        "replicate_diagnostics": replicate_diagnostics,
        "block_diagnostics": correlator.diagnostics(),
        "cusp_fit": {
            "status": "exploratory local-smoke extrapolation",
            "ansatz": "I(T)=I_infinity+a/T+b/T^2",
            "I_infinity": _complex_record(fitted_limit),
            "a": _complex_record(fitted_a),
            "b": _complex_record(fitted_b),
            "rqmc_standard_error": _complex_record(fitted_standard_error),
            "replicate_limits": [_complex_record(value) for value in fitted_replicates],
            "lower_order_I_infinity": _complex_record(lower_fitted_limit),
            "lower_order_replicate_limits": [
                _complex_record(value) for value in lower_fitted_replicates
            ],
            "last_retained_order_shift": _complex_record(
                fitted_limit - lower_fitted_limit
            ),
        },
        "tail_completion": {
            "status": "primary local-smoke estimator",
            "slice_ansatz": "f(tau2)=A*tau2^(-p)+B*tau2^(-p-1), p>1",
            "fit_is_target_free": True,
            "tail_start": float(cutoffs[-1]),
            "tau2_slices": list(tail_slices),
            "slice_means": [_complex_record(value) for value in slice_means],
            "replicate_slice_values": [
                [_complex_record(value) for value in row] for row in slice_array
            ],
            "fitted_exponent": float(tail_exponent),
            "mean_coefficients": {
                "A": _complex_record(tail_first),
                "B": _complex_record(tail_second),
            },
            "mean_relative_fit_residual": float(tail_relative_residual),
            "replicate_relative_fit_residuals": replicate_tail_residuals,
            "mean_integrated_tail": _complex_record(tail_integral_mean),
            "replicate_integrated_tails": [
                _complex_record(value) for value in replicate_tail_integrals
            ],
            "final_I": _complex_record(final_mean),
            "final_rqmc_standard_error": _complex_record(final_standard_error),
            "replicate_finals": [
                _complex_record(value) for value in replicate_final_values
            ],
        },
        "native_amplitude": {
            "A_over_i_gs3": _complex_record(16.0 * math.pi**2 * final_mean),
            "rqmc_standard_error": _complex_record(
                16.0 * math.pi**2 * final_standard_error
            ),
        },
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"wrote {output}")
    print(
        f"I_1,3={final_mean.real:+.12e}{final_mean.imag:+.3e}j "
        f"(RQMC SE {final_standard_error.real:.3e}+{final_standard_error.imag:.3e}j)",
        flush=True,
    )
    return result


def parser() -> argparse.ArgumentParser:
    out = argparse.ArgumentParser()
    out.add_argument("--t", type=float, default=0.6)
    out.add_argument("--p-max", type=float, default=5.0)
    out.add_argument("--momentum-order", type=int, default=4)
    out.add_argument(
        "--momentum-kind",
        choices=("power-legendre", "threshold-gaussian"),
        default="power-legendre",
    )
    out.add_argument("--momentum-power", type=float, default=2.0)
    out.add_argument("--momentum-power-step", type=float, default=0.137)
    out.add_argument("--reference-log-q-abs", type=float, default=-1.9)
    out.add_argument("--reference-log-q-step", type=float, default=0.137)
    out.add_argument("--high-order", type=int, default=6)
    out.add_argument("--low-order", type=int, default=2)
    out.add_argument(
        "--block-backend",
        choices=("regulated-h-recursion", "exact-c25-descendants"),
        default="regulated-h-recursion",
    )
    out.add_argument("--block-tolerance", type=float, default=5.0e-5)
    out.add_argument("--c-regulator", type=float, default=0.05)
    out.add_argument("--cutoffs", default="3,4,6,8")
    out.add_argument("--sobol-power", type=int, default=8)
    out.add_argument("--replicates", type=int, default=4)
    out.add_argument("--seed", type=int, default=17051301)
    out.add_argument("--tail-slices", default="8,10,12,16,20")
    out.add_argument("--tail-sobol-power", type=int, default=8)
    out.add_argument("--dps", type=int, default=28)
    out.add_argument(
        "--output",
        default=(
            "plumbing/results/genus1_three_point_worldsheet/"
            "equal_split_t060_local_smoke_n256_v1/worldsheet_blind.json"
        ),
    )
    return out


if __name__ == "__main__":
    run(parser().parse_args())
