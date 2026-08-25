#!/usr/bin/env python3
"""Torus S/T diagnostics using the plumbing period solver and Liouville one-point data."""

from __future__ import annotations

import argparse
import cmath
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

try:
    from liouville_torus import (
        LiouvilleTorusOnePointQuadrature,
        estimate_p_max,
        format_complex,
        parse_complex,
        q_from_tau,
    )
    from plumbing_algorithms import solve_torus_collocation, tau_from_q
except ImportError:  # pragma: no cover - supports package-style execution
    from plumbing.liouville_torus import (
        LiouvilleTorusOnePointQuadrature,
        estimate_p_max,
        format_complex,
        parse_complex,
        q_from_tau,
    )
    from plumbing.plumbing_algorithms import solve_torus_collocation, tau_from_q


def modular_s(tau: complex) -> complex:
    return -1.0 / tau


def modular_t(tau: complex) -> complex:
    return tau + 1.0


def principal_period_difference(left: complex, right: complex) -> complex:
    """Return left-right modulo integer real shifts, with the nearest lift."""
    diff = complex(left - right)
    return diff - round(diff.real)


@dataclass(frozen=True)
class TorusPlumbingModularResult:
    tau_input: complex
    q_input: complex
    tau_plumbing: complex
    q_plumbing_reconstructed: complex
    plumbing_tau_error_mod_z: float
    s_tau_target: complex
    s_q: complex
    s_tau_plumbing: complex
    s_tau_error_mod_z: float
    t_tau_target: complex
    t_q: complex
    t_tau_plumbing: complex
    t_tau_lift_difference: complex
    t_tau_error_mod_z: float
    g_tau: complex
    g_s_tau: complex
    s_numeric_ratio: complex
    s_analytic_ratio: complex
    s_ratio_over_analytic: complex
    s_expected: complex
    s_relative_error: complex
    g_t_tau: complex
    t_numeric_ratio: complex
    t_ratio_over_analytic: complex
    t_relative_error: complex
    external_weight: complex
    covariance_power: float


def run_torus_plumbing_modular_check(
    *,
    tau: complex,
    b: float,
    external_momentum: complex,
    block_order: int,
    quadrature_order: int,
    p_max: float | None,
    dps: int,
    form: str,
    collocation_order: int,
    collocation_samples: int,
) -> TorusPlumbingModularResult:
    if tau.imag <= 0:
        raise ValueError("tau must lie in the upper half-plane")

    q = q_from_tau(tau)
    s_tau = modular_s(tau)
    s_q = q_from_tau(s_tau)
    t_tau = modular_t(tau)
    t_q = q_from_tau(t_tau)

    torus = solve_torus_collocation(q, order=collocation_order, samples=collocation_samples)
    s_torus = solve_torus_collocation(s_q, order=collocation_order, samples=collocation_samples)
    t_torus = solve_torus_collocation(t_q, order=collocation_order, samples=collocation_samples)

    tau_plumbing = torus.b_period
    s_tau_plumbing = s_torus.b_period
    t_tau_plumbing = t_torus.b_period
    q_values = [q, s_q, t_q]
    if p_max is None:
        p_max = max(estimate_p_max(value, tail_tolerance=1.0e-12, safety_margin=0.6) for value in q_values)

    quadrature = LiouvilleTorusOnePointQuadrature(
        b=b,
        external_momentum=external_momentum,
        block_order=block_order,
        p_max=p_max,
        quadrature_order=quadrature_order,
        dps=dps,
    )
    if form == "full":
        value_at_q = quadrature.full_one_point
        covariance_power = 2.0 * quadrature.external_weight.real
    elif form == "hjs-stripped":
        value_at_q = quadrature.hjs_stripped_integral
        covariance_power = 2.0 * quadrature.external_weight.real + 1.0
    else:
        raise ValueError("form must be full or hjs-stripped")

    g_tau = value_at_q(q_from_tau(tau_plumbing))
    g_s_tau = value_at_q(q_from_tau(s_tau_plumbing))
    s_analytic_ratio = complex(abs(tau) ** covariance_power)
    s_expected = s_analytic_ratio * g_tau
    s_numeric_ratio = g_s_tau / g_tau
    g_t_tau = value_at_q(q_from_tau(t_tau_plumbing))
    t_numeric_ratio = g_t_tau / g_tau

    return TorusPlumbingModularResult(
        tau_input=tau,
        q_input=q,
        tau_plumbing=tau_plumbing,
        q_plumbing_reconstructed=q_from_tau(tau_plumbing),
        plumbing_tau_error_mod_z=abs(principal_period_difference(tau_plumbing, tau)),
        s_tau_target=s_tau,
        s_q=s_q,
        s_tau_plumbing=s_tau_plumbing,
        s_tau_error_mod_z=abs(principal_period_difference(s_tau_plumbing, s_tau)),
        t_tau_target=t_tau,
        t_q=t_q,
        t_tau_plumbing=t_tau_plumbing,
        t_tau_lift_difference=t_tau_plumbing - t_tau,
        t_tau_error_mod_z=abs(principal_period_difference(t_tau_plumbing, t_tau)),
        g_tau=g_tau,
        g_s_tau=g_s_tau,
        s_numeric_ratio=s_numeric_ratio,
        s_analytic_ratio=s_analytic_ratio,
        s_ratio_over_analytic=s_numeric_ratio / s_analytic_ratio,
        s_expected=s_expected,
        s_relative_error=(g_s_tau - s_expected) / s_expected,
        g_t_tau=g_t_tau,
        t_numeric_ratio=t_numeric_ratio,
        t_ratio_over_analytic=t_numeric_ratio,
        t_relative_error=(g_t_tau - g_tau) / g_tau,
        external_weight=quadrature.external_weight,
        covariance_power=float(covariance_power),
    )


def write_csv(path: Path, result: TorusPlumbingModularResult) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["quantity", "real", "imag", "abs"])
        rows = [
            ("tau_input", result.tau_input),
            ("tau_plumbing", result.tau_plumbing),
            ("S_tau_target", result.s_tau_target),
            ("S_tau_plumbing", result.s_tau_plumbing),
            ("T_tau_target", result.t_tau_target),
            ("T_tau_plumbing", result.t_tau_plumbing),
            ("T_tau_lift_difference", result.t_tau_lift_difference),
            ("G_tau", result.g_tau),
            ("G_S_tau", result.g_s_tau),
            ("S_numeric_ratio_GS_over_G", result.s_numeric_ratio),
            ("S_analytic_ratio_abs_tau_power", result.s_analytic_ratio),
            ("S_numeric_over_analytic", result.s_ratio_over_analytic),
            ("S_expected", result.s_expected),
            ("S_relative_error", result.s_relative_error),
            ("G_T_tau", result.g_t_tau),
            ("T_numeric_ratio_GT_over_G", result.t_numeric_ratio),
            ("T_numeric_over_analytic", result.t_ratio_over_analytic),
            ("T_relative_error", result.t_relative_error),
        ]
        for name, value in rows:
            writer.writerow([name, f"{value.real:.16g}", f"{value.imag:.16g}", f"{abs(value):.16g}"])


def safe_token(value: object) -> str:
    return str(value).replace("+", "p").replace("-", "m").replace(".", "p").replace("j", "i").replace(" ", "")


def run(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Check torus S/T transformations using plumbing collocation and Liouville one-point data."
    )
    parser.add_argument("--tau", type=parse_complex, default=0.2 + 0.9j)
    parser.add_argument("--b", type=float, default=0.8)
    parser.add_argument("--external-momentum", type=parse_complex, default=0.2 + 0.0j)
    parser.add_argument("--block-order", type=int, default=2)
    parser.add_argument("--quadrature-order", type=int, default=14)
    parser.add_argument("--p-max", type=float)
    parser.add_argument("--dps", type=int, default=22)
    parser.add_argument("--form", choices=["full", "hjs-stripped"], default="full")
    parser.add_argument("--collocation-order", type=int, default=10)
    parser.add_argument("--collocation-samples", type=int, default=96)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/liouville_torus_plumbing_modular_check"))
    parser.add_argument("--prefix")
    args = parser.parse_args(argv)

    result = run_torus_plumbing_modular_check(
        tau=args.tau,
        b=args.b,
        external_momentum=args.external_momentum,
        block_order=args.block_order,
        quadrature_order=args.quadrature_order,
        p_max=args.p_max,
        dps=args.dps,
        form=args.form,
        collocation_order=args.collocation_order,
        collocation_samples=args.collocation_samples,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.prefix or (
        f"torus_ST_tau{safe_token(args.tau)}_b{safe_token(args.b)}_P{safe_token(args.external_momentum.real)}"
        f"_order{args.block_order}"
    )
    csv_path = args.out_dir / f"{prefix}.csv"
    write_csv(csv_path, result)

    print("Liouville torus plumbing S/T check")
    print(f"  tau input={format_complex(result.tau_input)}")
    print(f"  q input={format_complex(result.q_input)}")
    print(f"  plumbing tau={format_complex(result.tau_plumbing)}")
    print(f"  plumbing tau error mod Z={result.plumbing_tau_error_mod_z:.6e}")
    print("")
    print("  S transformation")
    print(f"    target S tau={format_complex(result.s_tau_target)}")
    print(f"    q_S={format_complex(result.s_q)}")
    print(f"    plumbing S tau={format_complex(result.s_tau_plumbing)}")
    print(f"    S tau error mod Z={result.s_tau_error_mod_z:.6e}")
    print(f"    covariance power={result.covariance_power:.12g}")
    print(f"    G(S tau)={format_complex(result.g_s_tau)}")
    print(f"    numeric ratio G(S tau)/G(tau)={format_complex(result.s_numeric_ratio)}")
    print(f"    analytic ratio |tau|^power={format_complex(result.s_analytic_ratio)}")
    print(f"    numeric/analytic={format_complex(result.s_ratio_over_analytic)}")
    print(f"    |tau|^power G(tau)={format_complex(result.s_expected)}")
    print(f"    S relative error={format_complex(result.s_relative_error)}")
    print("")
    print("  T transformation")
    print(f"    target T tau={format_complex(result.t_tau_target)}")
    print(f"    q_T={format_complex(result.t_q)}")
    print(f"    plumbing T tau principal={format_complex(result.t_tau_plumbing)}")
    print(f"    T lift difference tau_plumbing-(tau+1)={format_complex(result.t_tau_lift_difference)}")
    print(f"    T tau error mod Z={result.t_tau_error_mod_z:.6e}")
    print(f"    G(T tau principal)={format_complex(result.g_t_tau)}")
    print(f"    numeric ratio G(T tau)/G(tau)={format_complex(result.t_numeric_ratio)}")
    print(f"    numeric/analytic={format_complex(result.t_ratio_over_analytic)}")
    print(f"    T relative error={format_complex(result.t_relative_error)}")
    print(f"  csv={csv_path}")


if __name__ == "__main__":
    run()
