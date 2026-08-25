#!/usr/bin/env python3
"""Measure the joint momentum geometry of the genus-one two-point integrand.

The analysis starts from the exact threshold factors and primary Gaussian
propagation, evaluates the complete production-order block on a joint
radial-angular rule, and moment-matches the remaining radial suppression and
angular DOZZ ridge.  The fitted sampler parameters are diagnostics: every
quadrature weight explicitly undoes the proposal, so they cannot change the
target integral.
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
    from genus1_two_point_adaptive_momentum import (
        _complex_record,
        default_points,
        local_polar_momentum_rule,
    )
    from genus1_two_point_worldsheet import (
        C_LIOUVILLE,
        LiouvilleTorusTwoPoint,
        dedekind_eta,
    )
    from torus_two_point_blocks import elliptic_nome
except ImportError:  # pragma: no cover - package-style execution
    from plumbing.genus1_two_point_adaptive_momentum import (
        _complex_record,
        default_points,
        local_polar_momentum_rule,
    )
    from plumbing.genus1_two_point_worldsheet import (
        C_LIOUVILLE,
        LiouvilleTorusTwoPoint,
        dedekind_eta,
    )
    from plumbing.torus_two_point_blocks import elliptic_nome


def _term_contributions(
    correlator: LiouvilleTorusTwoPoint,
    *,
    channel: str,
    z: complex,
    tau: complex,
    collision_radius: float | None,
) -> tuple[complex, np.ndarray]:
    """Return the correlator and its individual joint-rule contributions."""

    z = complex(z)
    tau = complex(tau)
    if channel == "necklace":
        total = complex(correlator.correlator_necklace(z, tau))
        assert correlator._necklace_arrays is not None
        h1, h2, weights, coefficients = correlator._necklace_arrays
        log_q1 = 1.0j * z
        log_q2 = 1.0j * (2.0 * math.pi * tau - z)
        hat_q1 = elliptic_nome(cmath.exp(log_q1))
        hat_q2 = elliptic_nome(cmath.exp(log_q2))
        descendants = np.einsum(
            "tij,i,j->t",
            coefficients,
            hat_q1 ** np.arange(coefficients.shape[1]),
            hat_q2 ** np.arange(coefficients.shape[2]),
            optimize=True,
        )
        primary = np.exp(
            2.0
            * (
                (h1 - C_LIOUVILLE / 24.0) * log_q1.real
                + (h2 - C_LIOUVILLE / 24.0) * log_q2.real
            )
        )
        contributions = weights * primary * np.abs(descendants) ** 2
    elif channel == "ope":
        total = complex(correlator.correlator_ope(z, tau))
        assert correlator._ope_arrays is not None
        h_loop, h_ope, weights, coefficients = correlator._ope_arrays
        q = cmath.exp(2.0j * math.pi * tau)
        v = cmath.exp(-1.0j * z) - 1.0
        flat_frame = cmath.exp(
            -2.0
            * correlator.external_weight
            * cmath.log(2.0 * cmath.sin(z / 2.0))
        )
        descendants = np.einsum(
            "tij,i,j->t",
            coefficients,
            q ** np.arange(coefficients.shape[1]),
            z ** np.arange(coefficients.shape[2]),
            optimize=True,
        )
        primary = np.exp(
            -4.0 * math.pi * tau.imag * (h_loop - C_LIOUVILLE / 24.0)
            + 2.0 * h_ope * math.log(abs(v))
        )
        contributions = (
            weights * abs(flat_frame) ** 2 * primary * np.abs(descendants) ** 2
        )
    elif channel == "collision-disc":
        if collision_radius is None:
            raise ValueError("collision-disc analysis requires its radius")
        total = complex(
            correlator.leading_collision_disc(tau, collision_radius)
        )
        assert correlator._ope_arrays is not None
        h_loop, h_ope, weights, coefficients = correlator._ope_arrays
        q = cmath.exp(2.0j * math.pi * tau)
        descendants = np.einsum(
            "tij,i,j->t",
            coefficients,
            q ** np.arange(coefficients.shape[1]),
            np.r_[
                1.0 + 0.0j,
                np.zeros(coefficients.shape[2] - 1, dtype=complex),
            ],
            optimize=True,
        )
        loop_primary = np.exp(
            -4.0 * math.pi * tau.imag * (h_loop - C_LIOUVILLE / 24.0)
        )
        p_ope_squared = h_ope - 1.0
        radial = (
            math.pi
            * np.exp(2.0 * p_ope_squared * math.log(collision_radius))
            / p_ope_squared
        )
        common = abs(dedekind_eta(tau)) ** 2 / math.sqrt(tau.imag)
        contributions = (
            common * weights * loop_primary * np.abs(descendants) ** 2 * radial
        )
    else:
        raise ValueError(f"unsupported channel {channel!r}")
    reconstruction = complex(np.sum(contributions))
    if abs(reconstruction - total) > 2.0e-12 * max(abs(total), 1.0e-300):
        raise RuntimeError("pointwise contribution decomposition did not close")
    return total, np.asarray(contributions, dtype=np.complex128)


def _fit_jacobi(mean_y: float, variance_y: float) -> tuple[float, float, float]:
    mean_t = min(1.0 - 1.0e-6, max(1.0e-6, 0.5 * (mean_y + 1.0)))
    variance_t = max(1.0e-8, 0.25 * variance_y)
    concentration = mean_t * (1.0 - mean_t) / variance_t - 1.0
    concentration = max(0.2, concentration)
    # roots_jacobi(alpha,beta) has density
    # (1-y)^alpha (1+y)^beta.  Thus beta controls t=(1+y)/2.
    beta = mean_t * concentration - 1.0
    alpha = (1.0 - mean_t) * concentration - 1.0
    return max(-0.9, alpha), max(-0.9, beta), concentration


def analyze_point(
    point: object,
    *,
    x: float,
    radial_order: int,
    angular_order: int,
    necklace_orders: tuple[int, int],
    ope_orders: tuple[int, int],
    dps: int,
) -> dict[str, object]:
    rule = local_polar_momentum_rule(point, radial_order, angular_order)
    correlator = LiouvilleTorusTwoPoint(
        1.0j * x,
        momentum_pair_rule=rule,
        necklace_orders=necklace_orders,
        ope_orders=ope_orders,
        special_dps=dps,
    )
    total, contributions = _term_contributions(
        correlator,
        channel=point.channel,
        z=point.z,
        tau=point.tau,
        collision_radius=point.collision_radius,
    )
    first_decay = 1.0 / rule.first_gaussian_width**2
    second_decay = 1.0 / rule.second_gaussian_width**2
    first_u = first_decay * rule.first_nodes**2
    second_u = second_decay * rule.second_nodes**2
    radial_v = first_u + second_u
    angular_y = (first_u - second_u) / radial_v
    absolute_weights = np.abs(contributions)
    normalization = float(np.sum(absolute_weights))
    probabilities = absolute_weights / normalization
    mean_v = float(np.dot(probabilities, radial_v))
    variance_v = float(np.dot(probabilities, (radial_v - mean_v) ** 2))
    mean_y = float(np.dot(probabilities, angular_y))
    variance_y = float(np.dot(probabilities, (angular_y - mean_y) ** 2))
    jacobi_alpha, jacobi_beta, concentration = _fit_jacobi(mean_y, variance_y)
    reference_radial_mean = rule.radial_laguerre_alpha + 1.0
    decay_scale = reference_radial_mean / mean_v
    decay_scale = min(4.0, max(0.5, decay_scale))
    cancellation_ratio = abs(total) / max(normalization, 1.0e-300)
    return {
        "name": point.name,
        "channel": point.channel,
        "z": _complex_record(point.z),
        "tau": _complex_record(point.tau),
        "value": _complex_record(total),
        "base_rule": {
            "radial_order": radial_order,
            "angular_order": angular_order,
            "node_count": radial_order * angular_order,
            "radial_laguerre_alpha": rule.radial_laguerre_alpha,
            "angular_jacobi_alpha": rule.angular_jacobi_alpha,
            "angular_jacobi_beta": rule.angular_jacobi_beta,
            "first_decay_coefficient": first_decay,
            "second_decay_coefficient": second_decay,
        },
        "measured_full_integrand_geometry": {
            "mean_v": mean_v,
            "variance_v": variance_v,
            "mean_y": mean_y,
            "variance_y": variance_y,
            "absolute_cancellation_ratio": cancellation_ratio,
        },
        "recommended_exact_proposal": {
            "common_decay_scale": decay_scale,
            "angular_jacobi_alpha": jacobi_alpha,
            "angular_jacobi_beta": jacobi_beta,
            "beta_concentration": concentration,
            "note": (
                "The proposal is moment-matched to the complete fixed-point "
                "integrand. Its weights must be undone exactly."
            ),
        },
    }


def run(argv: Iterable[str] | None = None) -> dict[str, object]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--x", type=float, default=0.4)
    parser.add_argument("--points", default="moderate-bulk,moderate-collision,moderate-disc,cusp-bulk,cusp-collision,cusp-disc")
    parser.add_argument("--radial-order", type=int, default=12)
    parser.add_argument("--angular-order", type=int, default=24)
    parser.add_argument("--necklace-orders", default="6,3")
    parser.add_argument("--ope-orders", default="3,8")
    parser.add_argument("--dps", type=int, default=28)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "plumbing/results/genus1_two_point_worldsheet/"
            "momentum_asymptotics_x04.json"
        ),
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    point_map = {point.name: point for point in default_points()}
    names = tuple(value for value in args.points.split(",") if value)
    if any(name not in point_map for name in names):
        raise ValueError("unknown asymptotic audit point")
    necklace_orders = tuple(int(value) for value in args.necklace_orders.split(","))
    ope_orders = tuple(int(value) for value in args.ope_orders.split(","))
    results = [
        analyze_point(
            point_map[name],
            x=args.x,
            radial_order=args.radial_order,
            angular_order=args.angular_order,
            necklace_orders=necklace_orders,  # type: ignore[arg-type]
            ope_orders=ope_orders,  # type: ignore[arg-type]
            dps=args.dps,
        )
        for name in names
    ]
    payload: dict[str, object] = {
        "calculation": "genus-one two-point complete-integrand momentum geometry",
        "x": float(args.x),
        "exact_asymptotic_factorization": {
            "ordinary_edges": "P1^2 P2^2 exp(-a1 P1^2-a2 P2^2)",
            "collision_disc": "P_loop^2 exp(-a_loop P_loop^2-a_ope P_ope^2)",
            "decay_coefficients": "a_i=-2 log|q_i|",
        },
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    for row in results:
        geometry = row["measured_full_integrand_geometry"]
        proposal = row["recommended_exact_proposal"]
        print(
            f"{row['name']:22s} mean(v)={geometry['mean_v']:.4f} "
            f"mean(y)={geometry['mean_y']:+.4f} "
            f"scale={proposal['common_decay_scale']:.4f} "
            f"Jacobi=({proposal['angular_jacobi_alpha']:.3f},"
            f"{proposal['angular_jacobi_beta']:.3f}) "
            f"sign_ratio={geometry['absolute_cancellation_ratio']:.6f}"
        )
    print(f"wrote {args.output}")
    return payload


if __name__ == "__main__":
    run()
