#!/usr/bin/env python3
"""Test the full local c=1 separating residue.

The physical genus-two integrand is represented as a critical ``26 X+bc``
Mumford density times the replacement quotient

    Z_compact(R) Z_L / Z_X^26.

At a separating node the auxiliary noncompact-scalar sewing coefficients are
one, so the full local residue can be tested by multiplying three independent
ratios:

1. the normalized critical Mumford residue;
2. the compact winding-lattice residue, including the inverse vacuum metric
   ``<0|0>_R^{-1}=1/R``;
3. the Liouville once-punctured-torus sewing residue, including
   ``<V_P V_P'>^{-1}=dP/pi``.

For Liouville, the primary bridge coefficient is evaluated independently by
``liouville_genus2_pair_of_tori`` and by the level-zero shell of the Zhu
descendant construction.  Higher Zhu shells provide a second check: their
effect must vanish as the bridge plumbing parameter tends to zero.
"""

from __future__ import annotations

import argparse
import cmath
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np

try:
    from audit_genus2_from_genus1_sewing import critical_separating_residue
    from genus2_c1_string_integrand import compact_boson_winding_sum_genus2
    from liouville_genus2 import liouville_genus2_pair_of_tori
    from liouville_genus2_glasses import (
        liouville_genus2_glasses_partition,
        liouville_weight_from_momentum,
    )
    from liouville_genus2_separating_zhu import liouville_genus2_separating_zhu
    from virasoro_blocks import TorusOnePointVirasoroBlock
except ImportError:  # pragma: no cover - supports package-style execution
    from plumbing.audit_genus2_from_genus1_sewing import critical_separating_residue
    from plumbing.genus2_c1_string_integrand import compact_boson_winding_sum_genus2
    from plumbing.liouville_genus2 import liouville_genus2_pair_of_tori
    from plumbing.liouville_genus2_glasses import (
        liouville_genus2_glasses_partition,
        liouville_weight_from_momentum,
    )
    from plumbing.liouville_genus2_separating_zhu import liouville_genus2_separating_zhu
    from plumbing.virasoro_blocks import TorusOnePointVirasoroBlock


DEFAULT_OUTPUT = Path(
    "plumbing/results/genus2_c1_moduli_mc/"
    "full_c1_separating_factorization_audit.json"
)


def _relative_error(value: float | complex, target: float | complex = 1.0) -> float:
    return float(abs(complex(value) / complex(target) - 1.0))


def _tau_from_nome(q: complex) -> complex:
    q = complex(q)
    if not 0.0 < abs(q) < 1.0:
        raise ValueError("handle nome must have modulus between zero and one")
    return cmath.log(q) / (2j * math.pi)


def _compact_winding_sum_genus1(tau: complex, radius: float, nmax: int) -> float:
    return float(
        sum(
            math.exp(
                -math.pi
                * radius**2
                * abs(m + complex(tau) * n) ** 2
                / complex(tau).imag
            )
            for m in range(-int(nmax), int(nmax) + 1)
            for n in range(-int(nmax), int(nmax) + 1)
        )
    )


def _liouville_comparison(
    *,
    q_left: complex,
    q_right: complex,
    q_bridge: complex,
    b_regulator: float,
    block_order: int,
    bridge_level: int,
    bridge_p_max: float,
    handle_p_max: float,
    bridge_quadrature_order: int,
    handle_quadrature_order: int,
    dps: int,
) -> dict[str, float]:
    common = {
        "b": float(b_regulator),
        "q1": complex(q_left),
        "q2": complex(q_right),
        "q_bridge": complex(q_bridge),
        "block_order": int(block_order),
        "bridge_p_max": float(bridge_p_max),
        "handle_p_max": float(handle_p_max),
        "bridge_quadrature_order": int(bridge_quadrature_order),
        "handle_quadrature_order": int(handle_quadrature_order),
        "dps": int(dps),
        "include_cosmological_prefactor": False,
    }
    primary = liouville_genus2_pair_of_tori(**common)
    zhu = liouville_genus2_separating_zhu(
        **common,
        bridge_level=int(bridge_level),
        torus_series_method="recursion",
        descendant_basis="ordinary",
    )
    primary_value = float(primary.value.real)
    zhu_total = float(zhu.value.real)
    zhu_primary_shell = float(
        sum(sample.shell_contributions[0] for sample in zhu.samples).real
    )
    if min(primary_value, zhu_total, zhu_primary_shell) <= 0.0:
        raise ValueError("Liouville comparison produced a nonpositive partition")
    return {
        "direct_pair_of_tori_primary": primary_value,
        "zhu_primary_shell": zhu_primary_shell,
        "zhu_through_bridge_level": zhu_total,
        "primary_normalization_ratio": zhu_primary_shell / primary_value,
        "descendant_complete_ratio_to_primary": zhu_total / primary_value,
        "relative_descendant_correction": zhu_total / zhu_primary_shell - 1.0,
    }


def _production_glasses_comparison(
    *,
    q_left: complex,
    q_right: complex,
    q_bridge: complex,
    b_regulator: float,
    block_order: int,
    p_max: float,
    quadrature_order: int,
    dps: int,
) -> dict[str, float]:
    """Compare the production CCY block with two torus one-point blocks."""

    production = liouville_genus2_glasses_partition(
        b=float(b_regulator),
        q_left=complex(q_left),
        q_right=complex(q_right),
        q_bridge=complex(q_bridge),
        block_order=int(block_order),
        p_max=float(p_max),
        quadrature_order=int(quadrature_order),
        quadrature_scheme="uniform",
        dps=int(dps),
        propagator_shift=0.0,
        include_vacuum_seed=True,
        vacuum_word_len=5,
        vacuum_oscillator_level_max=30,
        include_cosmological_prefactor=False,
        store_samples=True,
    )
    independently_factorized = 0.0 + 0.0j
    for sample in production.samples:
        h_left = liouville_weight_from_momentum(b_regulator, sample.p_left)
        h_right = liouville_weight_from_momentum(b_regulator, sample.p_right)
        h_bridge = liouville_weight_from_momentum(b_regulator, sample.p_bridge)
        left = TorusOnePointVirasoroBlock(
            production.central_charge,
            h_left,
            h_bridge,
        ).chiral_block(q_left, block_order, include_prefactor=False)
        right = TorusOnePointVirasoroBlock(
            production.central_charge,
            h_right,
            h_bridge,
        ).chiral_block(q_right, block_order, include_prefactor=False)
        independently_factorized += (
            sample.measure_weight
            * sample.structure_left
            * sample.structure_right
            * abs(sample.propagator * left * right) ** 2
        )
    production_value = float(production.value.real)
    factorized_value = float(independently_factorized.real)
    if min(production_value, factorized_value) <= 0.0:
        raise ValueError("production Liouville comparison is nonpositive")
    return {
        "production_ccy_glasses": production_value,
        "independent_two_torus_blocks": factorized_value,
        "ratio": production_value / factorized_value,
    }


def build_audit(
    *,
    q_left: complex,
    q_right: complex,
    bridge_values: tuple[float, ...],
    radius: float,
    b_regulator: float,
    block_order: int,
    bridge_level: int,
    theta_nmax: int,
    lattice_nmax: int,
    bridge_p_max: float,
    handle_p_max: float,
    bridge_quadrature_order: int,
    handle_quadrature_order: int,
    production_quadrature_order: int,
    dps: int,
) -> dict[str, object]:
    tau_left = _tau_from_nome(q_left)
    tau_right = _tau_from_nome(q_right)
    theta_left = _compact_winding_sum_genus1(tau_left, radius, lattice_nmax)
    theta_right = _compact_winding_sum_genus1(tau_right, radius, lattice_nmax)

    rows: list[dict[str, object]] = []
    for q_bridge_abs in bridge_values:
        q_bridge = complex(float(q_bridge_abs), 0.0)
        epsilon = q_bridge / (2j * math.pi)
        omega = np.asarray(
            [[tau_left, epsilon], [epsilon, tau_right]],
            dtype=np.complex128,
        )
        critical = critical_separating_residue(
            tau_left,
            tau_right,
            epsilon,
            theta_nmax=theta_nmax,
        )
        compact_genus2 = compact_boson_winding_sum_genus2(
            omega,
            radius,
            lattice_nmax=lattice_nmax,
        )
        compact_ratio = compact_genus2 / (theta_left * theta_right)
        liouville = _liouville_comparison(
            q_left=q_left,
            q_right=q_right,
            q_bridge=q_bridge,
            b_regulator=b_regulator,
            block_order=block_order,
            bridge_level=bridge_level,
            bridge_p_max=bridge_p_max,
            handle_p_max=handle_p_max,
            bridge_quadrature_order=bridge_quadrature_order,
            handle_quadrature_order=handle_quadrature_order,
            dps=dps,
        )
        production_liouville = _production_glasses_comparison(
            q_left=q_left,
            q_right=q_right,
            q_bridge=q_bridge,
            b_regulator=b_regulator,
            block_order=block_order,
            p_max=bridge_p_max,
            quadrature_order=production_quadrature_order,
            dps=dps,
        )
        critical_ratio = float(critical["ratio_to_punctured_tori"])
        liouville_primary_ratio = float(liouville["primary_normalization_ratio"])
        liouville_full_ratio = float(liouville["descendant_complete_ratio_to_primary"])
        rows.append(
            {
                "q_bridge": q_bridge_abs,
                "epsilon_abs": abs(epsilon),
                "critical_26X_plus_bc_ratio": critical_ratio,
                "compact_winding_ratio": compact_ratio,
                "liouville": liouville,
                "production_liouville": production_liouville,
                "combined_primary_residue_ratio": (
                    critical_ratio * compact_ratio * liouville_primary_ratio
                ),
                "combined_with_bridge_descendants_ratio": (
                    critical_ratio * compact_ratio * liouville_full_ratio
                ),
                "production_combined_residue_ratio": (
                    critical_ratio
                    * compact_ratio
                    * float(production_liouville["ratio"])
                ),
            }
        )

    final = rows[-1]
    exact_b1_crosscheck = _liouville_comparison(
        q_left=q_left,
        q_right=q_right,
        q_bridge=complex(float(bridge_values[-1]), 0.0),
        b_regulator=1.0,
        block_order=1,
        bridge_level=0,
        bridge_p_max=bridge_p_max,
        handle_p_max=handle_p_max,
        bridge_quadrature_order=bridge_quadrature_order,
        handle_quadrature_order=handle_quadrature_order,
        dps=dps,
    )
    primary_errors = [
        abs(float(row["combined_primary_residue_ratio"]) - 1.0) for row in rows
    ]
    descendant_errors = [
        abs(float(row["combined_with_bridge_descendants_ratio"]) - 1.0)
        for row in rows
    ]
    production_errors = [
        abs(float(row["production_combined_residue_ratio"]) - 1.0)
        for row in rows
    ]
    checks = {
        "critical_bc_residue_is_unit": _relative_error(
            float(final["critical_26X_plus_bc_ratio"])
        )
        < 2.0e-6,
        "compact_inverse_metric_residue_is_unit": _relative_error(
            float(final["compact_winding_ratio"])
        )
        < 2.0e-6,
        "liouville_primary_inverse_metric_is_unit": _relative_error(
            float(final["liouville"]["primary_normalization_ratio"])
        )
        < 2.0e-6,
        "liouville_exact_b1_order1_crosscheck": _relative_error(
            exact_b1_crosscheck["primary_normalization_ratio"]
        )
        < 1.0e-4,
        "combined_primary_residue_is_unit": primary_errors[-1] < 5.0e-6,
        "bridge_descendants_vanish_toward_node": (
            descendant_errors[-1] < descendant_errors[0]
            and descendant_errors[-1] < 2.0e-3
        ),
        "production_ccy_path_factorizes": (
            production_errors[-1] < production_errors[0]
            and production_errors[-1] < 2.0e-3
        ),
    }
    return {
        "scope": (
            "Local separating factorization of the physical bc + compact-boson "
            "+ c=25 Liouville genus-two integrand. No matrix-model input is used."
        ),
        "formula": (
            "R_full(q)=R_(26X+bc)(q) R_compact(q) R_Liouville(q), "
            "with the auxiliary scalar sewing coefficients canceled against "
            "Z_compact Z_L/Z_X^26"
        ),
        "state_metrics": {
            "compact_vacuum": "<0|0>_R=R; inverse metric 1/R",
            "liouville": "<V_P V_P'>=pi delta(P-P'); inverse metric dP/pi",
            "ghost": (
                "translation c zero modes are saturated at each node; the b zero "
                "mode for q supplies the plumbing differential"
            ),
        },
        "liouville_regulator": {
            "physical_limit": "b -> 1",
            "value_used": b_regulator,
            "reason": (
                "The generic torus-block recursion has a colliding pole at exactly "
                "b=1 from level two onward. A nearby-b regulator tests the finite "
                "collision limit without changing the state normalization."
            ),
        },
        "liouville_exact_b1_crosscheck": {
            "block_order": 1,
            "bridge_level": 0,
            "q_bridge": bridge_values[-1],
            **exact_b1_crosscheck,
            "interpretation": (
                "This is evaluated at the physical b=1 point. Order one is the "
                "highest generic recursion order before the level-two colliding "
                "pole requires the regulated collision limit."
            ),
        },
        "parameters": {
            "q_left": {"real": complex(q_left).real, "imag": complex(q_left).imag},
            "q_right": {"real": complex(q_right).real, "imag": complex(q_right).imag},
            "tau_left": {"real": tau_left.real, "imag": tau_left.imag},
            "tau_right": {"real": tau_right.real, "imag": tau_right.imag},
            "radius": radius,
            "block_order": block_order,
            "bridge_level": bridge_level,
            "theta_nmax": theta_nmax,
            "lattice_nmax": lattice_nmax,
            "bridge_p_max": bridge_p_max,
            "handle_p_max": handle_p_max,
            "bridge_quadrature_order": bridge_quadrature_order,
            "handle_quadrature_order": handle_quadrature_order,
            "production_quadrature_order": production_quadrature_order,
            "dps": dps,
        },
        "rows": rows,
        "checks": checks,
        "all_checks_pass": all(checks.values()),
        "result": {
            "final_combined_primary_residue_ratio": float(
                final["combined_primary_residue_ratio"]
            ),
            "final_combined_with_descendants_ratio": float(
                final["combined_with_bridge_descendants_ratio"]
            ),
            "final_production_combined_residue_ratio": float(
                final["production_combined_residue_ratio"]
            ),
            "local_full_c1_separating_normalization": 1.0,
            "extra_moduli_independent_factor_derived": False,
        },
    }


def run(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Audit the full local c=1 separating factorization residue."
    )
    parser.add_argument("--q-left", type=complex, default=0.003 + 0.0j)
    parser.add_argument("--q-right", type=complex, default=0.0025 + 0.0j)
    parser.add_argument(
        "--bridge-values",
        default="0.02,0.01,0.005,0.0025,0.00125",
    )
    parser.add_argument("--radius", type=float, default=1.31)
    parser.add_argument("--b-regulator", type=float, default=0.999)
    parser.add_argument("--block-order", type=int, default=2)
    parser.add_argument("--bridge-level", type=int, default=2)
    parser.add_argument("--theta-nmax", type=int, default=9)
    parser.add_argument("--lattice-nmax", type=int, default=8)
    parser.add_argument("--bridge-p-max", type=float, default=1.5)
    parser.add_argument("--handle-p-max", type=float, default=1.5)
    parser.add_argument("--bridge-quadrature-order", type=int, default=4)
    parser.add_argument("--handle-quadrature-order", type=int, default=6)
    parser.add_argument("--production-quadrature-order", type=int, default=3)
    parser.add_argument("--dps", type=int, default=24)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(list(argv) if argv is not None else None)

    bridge_values = tuple(
        sorted(
            (float(piece.strip()) for piece in args.bridge_values.split(",") if piece.strip()),
            reverse=True,
        )
    )
    if not bridge_values or min(bridge_values) <= 0.0 or max(bridge_values) >= 1.0:
        raise ValueError("bridge values must lie strictly between zero and one")

    audit = build_audit(
        q_left=args.q_left,
        q_right=args.q_right,
        bridge_values=bridge_values,
        radius=args.radius,
        b_regulator=args.b_regulator,
        block_order=args.block_order,
        bridge_level=args.bridge_level,
        theta_nmax=args.theta_nmax,
        lattice_nmax=args.lattice_nmax,
        bridge_p_max=args.bridge_p_max,
        handle_p_max=args.handle_p_max,
        bridge_quadrature_order=args.bridge_quadrature_order,
        handle_quadrature_order=args.handle_quadrature_order,
        production_quadrature_order=args.production_quadrature_order,
        dps=args.dps,
    )
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(audit, indent=2) + "\n")

    print("Full local c=1 separating-factorization audit")
    for row in audit["rows"]:
        print(
            f"  q={float(row['q_bridge']):.6g}: "
            f"critical={float(row['critical_26X_plus_bc_ratio']):.12g}, "
            f"compact={float(row['compact_winding_ratio']):.12g}, "
            f"Liouville-primary={float(row['liouville']['primary_normalization_ratio']):.12g}, "
            f"combined-primary={float(row['combined_primary_residue_ratio']):.12g}, "
            f"combined-full={float(row['combined_with_bridge_descendants_ratio']):.12g}, "
            f"production={float(row['production_combined_residue_ratio']):.12g}"
        )
    print(f"  all checks pass={audit['all_checks_pass']}")
    print(f"  output={args.out_json}")
    if not audit["all_checks_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    run()
