#!/usr/bin/env python3
"""Audit genus-one two-point Liouville momentum integration point by point.

The old worldsheet calculation uses one fixed finite rule ``P=p_max*u^2``
over the complete ``(z,tau)`` domain.  This diagnostic instead constructs
local generalized-Laguerre rules from the two actual primary propagation
coordinates and promotes a genuinely distinct sequence of momentum orders.

This module deliberately audits fixed worldsheet points before attempting the
full modular integral.  A local rule changes its nodes with ``(z,tau)`` and
therefore cannot reuse the global block table which made the old integration
cheap.  The pointwise audit measures both the numerical shift and that cost.
"""

from __future__ import annotations

import argparse
import cmath
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

try:
    from genus1_two_point_worldsheet import (
        LiouvilleTorusTwoPoint,
        MomentumPairRule,
        MomentumRule,
    )
except ImportError:  # pragma: no cover - package-style execution
    from plumbing.genus1_two_point_worldsheet import (
        LiouvilleTorusTwoPoint,
        MomentumPairRule,
        MomentumRule,
    )


DEFAULT_ORDERS = (4, 6, 8, 10, 12)


@dataclass(frozen=True)
class AuditPoint:
    name: str
    channel: str
    z: complex
    tau: complex
    collision_radius: float | None = None


def _complex_record(value: complex) -> dict[str, float]:
    number = complex(value)
    return {"real": float(number.real), "imag": float(number.imag)}


def _parse_orders(text: str) -> tuple[int, ...]:
    orders = tuple(int(item) for item in text.split(",") if item.strip())
    if len(orders) < 2 or tuple(sorted(set(orders))) != orders:
        raise ValueError("orders must contain at least two strictly increasing integers")
    if any(order <= 0 for order in orders):
        raise ValueError("momentum orders must be positive")
    return orders


def _relative_change(previous: complex, current: complex) -> float:
    scale = max(abs(complex(previous)), abs(complex(current)), 1.0e-300)
    return float(abs(complex(current) - complex(previous)) / scale)


def _rule_record(rule: MomentumRule) -> dict[str, object]:
    return {
        "kind": rule.kind,
        "order": int(rule.order),
        "q_abs": None if rule.q_abs is None else float(rule.q_abs),
        "gaussian_width": (
            None if rule.gaussian_width is None else float(rule.gaussian_width)
        ),
        "minimum_node": float(min(rule.nodes)),
        "maximum_node": float(max(rule.nodes)),
    }


def local_momentum_rules(
    point: AuditPoint,
    order: int,
) -> tuple[MomentumRule, MomentumRule]:
    """Return the two exact local Gaussian transformations for one channel."""

    z = complex(point.z)
    tau = complex(point.tau)
    if tau.imag <= 0.0:
        raise ValueError("tau must lie in the upper half-plane")
    if point.channel == "necklace":
        log_q_first = 1.0j * z
        log_q_second = 1.0j * (2.0 * math.pi * tau - z)
        if log_q_first.real >= 0.0 or log_q_second.real >= 0.0:
            raise ValueError("necklace propagation requires both |q_i|<1")
        return (
            MomentumRule.threshold_gaussian(
                cmath.exp(log_q_first), order, log_q_abs=log_q_first.real
            ),
            MomentumRule.threshold_gaussian(
                cmath.exp(log_q_second), order, log_q_abs=log_q_second.real
            ),
        )
    if point.channel in {"ope", "collision-disc"}:
        log_q_loop = 2.0j * math.pi * tau
        if point.channel == "collision-disc":
            if point.collision_radius is None or point.collision_radius <= 0.0:
                raise ValueError("collision-disc points require a positive radius")
            log_q_ope = math.log(float(point.collision_radius)) + 0.0j
        else:
            v = cmath.exp(-1.0j * z) - 1.0
            if not 0.0 < abs(v) < 1.0:
                raise ValueError(
                    "the local OPE primary coordinate must obey 0<|v|<1; "
                    "use the necklace channel outside this representation domain"
                )
            log_q_ope = cmath.log(v)
        loop_rule = MomentumRule.threshold_gaussian(
            cmath.exp(log_q_loop), order, log_q_abs=log_q_loop.real
        )
        # The analytic disc contributes 1/P_ope^2, cancelling its exact
        # threshold factor.  Ordinary OPE points retain both threshold zeros.
        ope_constructor = (
            MomentumRule.primary_gaussian
            if point.channel == "collision-disc"
            else MomentumRule.threshold_gaussian
        )
        ope_rule = ope_constructor(
            cmath.exp(log_q_ope), order, log_q_abs=log_q_ope.real
        )
        return loop_rule, ope_rule
    raise ValueError(f"unsupported channel {point.channel!r}")


def local_polar_momentum_rule(
    point: AuditPoint,
    radial_order: int,
    angular_order: int,
    *,
    angular_jacobi_alpha: float | None = None,
    angular_jacobi_beta: float | None = None,
    decay_scale: float = 1.0,
) -> MomentumPairRule:
    r"""Return an exact joint radial-angular Gaussian transformation.

    With ``x_i=sqrt(a_i) P_i``, set

    ``v=x_1^2+x_2^2`` and ``y=(x_1^2-x_2^2)/v``.

    Ordinary necklace/OPE edges have threshold powers ``(P_1^2,P_2^2)``,
    giving a ``v^2 exp(-v) sqrt(1-y^2)`` reference measure.  The collision
    disc has powers ``(P_loop^2,P_ope^0)`` and therefore uses
    ``v^1 exp(-v) (1+y)^(1/2) (1-y)^(-1/2)``.  Alternate Jacobi exponents
    change only the proposal; the returned weights undo them exactly.
    """

    radial_order = int(radial_order)
    angular_order = int(angular_order)
    if radial_order <= 0 or angular_order <= 0:
        raise ValueError("polar orders must be positive")
    # The tensor helper already validates the channel coordinates.  Only its
    # q values and Gaussian widths are needed to define the scaled plane.
    tensor_rules = local_momentum_rules(point, max(2, radial_order))
    decay_scale = float(decay_scale)
    if not math.isfinite(decay_scale) or decay_scale <= 0.0:
        raise ValueError("decay scale must be positive and finite")
    first_decay = decay_scale / float(tensor_rules[0].gaussian_width) ** 2
    second_decay = decay_scale / float(tensor_rules[1].gaussian_width) ** 2
    first_power = 2.0
    second_power = 0.0 if point.channel == "collision-disc" else 2.0
    radial_alpha = 0.5 * (first_power + second_power)
    default_alpha = 0.5 * (second_power - 1.0)
    default_beta = 0.5 * (first_power - 1.0)
    jacobi_alpha = (
        default_alpha
        if angular_jacobi_alpha is None
        else float(angular_jacobi_alpha)
    )
    jacobi_beta = (
        default_beta
        if angular_jacobi_beta is None
        else float(angular_jacobi_beta)
    )
    if jacobi_alpha <= -1.0 or jacobi_beta <= -1.0:
        raise ValueError("Jacobi exponents must exceed -1")
    try:
        from scipy.special import roots_genlaguerre, roots_jacobi
    except ImportError as exc:  # pragma: no cover - SciPy is a project dependency
        raise ImportError("polar momentum sampling requires SciPy") from exc

    radial_nodes, radial_weights = roots_genlaguerre(radial_order, radial_alpha)
    angular_nodes, angular_weights = roots_jacobi(
        angular_order, jacobi_alpha, jacobi_beta
    )
    first_nodes: list[float] = []
    second_nodes: list[float] = []
    weights: list[float] = []
    decay_jacobian = math.sqrt(first_decay * second_decay)
    for radial_node, radial_weight in zip(radial_nodes, radial_weights):
        v = float(radial_node)
        radial_undo = float(radial_weight) * math.exp(v) / v**radial_alpha
        for angular_node, angular_weight in zip(angular_nodes, angular_weights):
            y = float(angular_node)
            first_nodes.append(
                math.sqrt(0.5 * v * (1.0 + y) / first_decay)
            )
            second_nodes.append(
                math.sqrt(0.5 * v * (1.0 - y) / second_decay)
            )
            weights.append(
                radial_undo
                * float(angular_weight)
                / (
                    4.0
                    * decay_jacobian
                    * (1.0 - y) ** (jacobi_alpha + 0.5)
                    * (1.0 + y) ** (jacobi_beta + 0.5)
                )
            )
    return MomentumPairRule(
        first_nodes=np.asarray(first_nodes, dtype=float),
        second_nodes=np.asarray(second_nodes, dtype=float),
        weights=np.asarray(weights, dtype=float),
        radial_order=radial_order,
        angular_order=angular_order,
        kind=(
            "threshold-polar"
            if point.channel != "collision-disc"
            else "threshold-primary-polar"
        ),
        first_q_abs=float(tensor_rules[0].q_abs),
        second_q_abs=float(tensor_rules[1].q_abs),
        first_gaussian_width=1.0 / math.sqrt(first_decay),
        second_gaussian_width=1.0 / math.sqrt(second_decay),
        radial_laguerre_alpha=float(radial_alpha),
        angular_jacobi_alpha=float(jacobi_alpha),
        angular_jacobi_beta=float(jacobi_beta),
    )


def evaluate_point_polar(
    point: AuditPoint,
    *,
    x: float,
    radial_order: int,
    angular_order: int,
    necklace_orders: tuple[int, int],
    ope_orders: tuple[int, int],
    dps: int,
    angular_jacobi_alpha: float | None = None,
    angular_jacobi_beta: float | None = None,
    decay_scale: float = 1.0,
) -> tuple[complex, MomentumPairRule]:
    rule = local_polar_momentum_rule(
        point,
        radial_order,
        angular_order,
        angular_jacobi_alpha=angular_jacobi_alpha,
        angular_jacobi_beta=angular_jacobi_beta,
        decay_scale=decay_scale,
    )
    correlator = LiouvilleTorusTwoPoint(
        1.0j * float(x),
        momentum_pair_rule=rule,
        necklace_orders=necklace_orders,
        ope_orders=ope_orders,
        special_dps=dps,
    )
    if point.channel == "necklace":
        value = correlator.correlator_necklace(point.z, point.tau)
    elif point.channel == "ope":
        value = correlator.correlator_ope(point.z, point.tau)
    else:
        assert point.collision_radius is not None
        value = correlator.leading_collision_disc(
            point.tau, point.collision_radius
        )
    return complex(value), rule


def evaluate_point(
    point: AuditPoint,
    *,
    x: float,
    order: int,
    necklace_orders: tuple[int, int],
    ope_orders: tuple[int, int],
    dps: int,
) -> tuple[complex, tuple[MomentumRule, MomentumRule]]:
    rules = local_momentum_rules(point, order)
    correlator = LiouvilleTorusTwoPoint(
        1.0j * float(x),
        momentum_rules=rules,
        necklace_orders=necklace_orders,
        ope_orders=ope_orders,
        special_dps=dps,
    )
    if point.channel == "necklace":
        value = correlator.correlator_necklace(point.z, point.tau)
    elif point.channel == "ope":
        value = correlator.correlator_ope(point.z, point.tau)
    else:
        assert point.collision_radius is not None
        value = correlator.leading_collision_disc(
            point.tau, point.collision_radius
        )
    return complex(value), rules


def evaluate_old_anchor(
    point: AuditPoint,
    *,
    x: float,
    order: int,
    p_max: float,
    power: float,
    necklace_orders: tuple[int, int],
    ope_orders: tuple[int, int],
    dps: int,
) -> complex:
    rule = MomentumRule.power_legendre(p_max, order, power)
    correlator = LiouvilleTorusTwoPoint(
        1.0j * float(x),
        momentum_rule=rule,
        necklace_orders=necklace_orders,
        ope_orders=ope_orders,
        special_dps=dps,
    )
    if point.channel == "necklace":
        value = correlator.correlator_necklace(point.z, point.tau)
    elif point.channel == "ope":
        value = correlator.correlator_ope(point.z, point.tau)
    else:
        assert point.collision_radius is not None
        value = correlator.leading_collision_disc(
            point.tau, point.collision_radius
        )
    return complex(value)


def audit_point(
    point: AuditPoint,
    *,
    x: float,
    orders: Sequence[int],
    tolerance: float,
    old_order: int,
    p_max: float,
    power: float,
    necklace_orders: tuple[int, int],
    ope_orders: tuple[int, int],
    dps: int,
) -> dict[str, object]:
    attempts: list[dict[str, object]] = []
    previous: complex | None = None
    converged = False
    selected: complex | None = None
    for order in orders:
        started = time.perf_counter()
        value, rules = evaluate_point(
            point,
            x=x,
            order=int(order),
            necklace_orders=necklace_orders,
            ope_orders=ope_orders,
            dps=dps,
        )
        drift = None if previous is None else _relative_change(previous, value)
        attempts.append(
            {
                "order": int(order),
                "value": _complex_record(value),
                "relative_change": drift,
                "runtime_seconds": float(time.perf_counter() - started),
                "rules": [_rule_record(rule) for rule in rules],
            }
        )
        print(
            f"{point.name:22s} {point.channel:14s} Q={order:2d} "
            f"value={value.real:+.12e}{value.imag:+.2e}j "
            f"step={float('nan') if drift is None else drift:.3e} "
            f"time={attempts[-1]['runtime_seconds']:.1f}s",
            flush=True,
        )
        selected = value
        if drift is not None and drift <= tolerance:
            converged = True
            break
        previous = value

    anchor_started = time.perf_counter()
    old_anchor = evaluate_old_anchor(
        point,
        x=x,
        order=old_order,
        p_max=p_max,
        power=power,
        necklace_orders=necklace_orders,
        ope_orders=ope_orders,
        dps=dps,
    )
    anchor_runtime = time.perf_counter() - anchor_started
    assert selected is not None
    shift = (selected - old_anchor) / old_anchor if old_anchor != 0.0 else complex("nan")
    print(
        f"{point.name:22s} old Q={old_order:2d} value={old_anchor.real:+.12e} "
        f"adaptive shift={shift.real:+.3%} runtime={anchor_runtime:.1f}s",
        flush=True,
    )
    return {
        "point": {
            "name": point.name,
            "channel": point.channel,
            "z": _complex_record(point.z),
            "tau": _complex_record(point.tau),
            "collision_radius": point.collision_radius,
        },
        "adaptive": {
            "scheme": "local threshold-Gaussian; primary-Gaussian on disc OPE edge",
            "orders_requested": [int(order) for order in orders],
            "relative_tolerance": float(tolerance),
            "converged": bool(converged),
            "selected_order": int(attempts[-1]["order"]),
            "selected_value": _complex_record(selected),
            "attempts": attempts,
        },
        "old_anchor": {
            "scheme": f"P={p_max:g}*u^{power:g}",
            "order": int(old_order),
            "value": _complex_record(old_anchor),
            "runtime_seconds": float(anchor_runtime),
        },
        "adaptive_relative_shift_from_old": _complex_record(shift),
    }


def default_points() -> tuple[AuditPoint, ...]:
    moderate_tau = 0.17 + 1.08j
    cusp_tau = 0.13 + 8.0j
    return (
        AuditPoint("moderate-bulk", "necklace", 0.80 + 0.50j, moderate_tau),
        AuditPoint("moderate-collision", "ope", 0.35 + 0.15j, moderate_tau),
        AuditPoint(
            "moderate-disc",
            "collision-disc",
            0.0 + 0.0j,
            moderate_tau,
            collision_radius=0.10,
        ),
        AuditPoint(
            "cusp-bulk",
            "necklace",
            2.0 * math.pi * (0.31 + 0.25 * cusp_tau),
            cusp_tau,
        ),
        AuditPoint("cusp-collision", "ope", 0.25 + 0.08j, cusp_tau),
        AuditPoint(
            "cusp-disc",
            "collision-disc",
            0.0 + 0.0j,
            cusp_tau,
            collision_radius=0.10,
        ),
    )


def run(argv: Iterable[str] | None = None) -> dict[str, object]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--x", type=float, default=0.4)
    parser.add_argument("--orders", default=",".join(str(v) for v in DEFAULT_ORDERS))
    parser.add_argument("--tolerance", type=float, default=5.0e-5)
    parser.add_argument("--old-order", type=int, default=16)
    parser.add_argument("--p-max", type=float, default=6.0)
    parser.add_argument("--power", type=float, default=2.0)
    parser.add_argument("--necklace-orders", default="6,3")
    parser.add_argument("--ope-orders", default="3,8")
    parser.add_argument("--dps", type=int, default=28)
    parser.add_argument(
        "--points",
        default="moderate-bulk,moderate-collision,moderate-disc,cusp-bulk,cusp-collision,cusp-disc",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "plumbing/results/genus1_two_point_worldsheet/"
            "adaptive_momentum_point_audit_x04.json"
        ),
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    if not 0.0 < args.x < 1.0:
        raise ValueError("the real-contour audit requires 0<x<1")
    if args.tolerance <= 0.0:
        raise ValueError("tolerance must be positive")
    orders = _parse_orders(args.orders)
    necklace_orders = tuple(int(v) for v in args.necklace_orders.split(","))
    ope_orders = tuple(int(v) for v in args.ope_orders.split(","))
    if len(necklace_orders) != 2 or len(ope_orders) != 2:
        raise ValueError("block-order pairs must contain exactly two integers")
    points_by_name = {point.name: point for point in default_points()}
    requested_names = tuple(name for name in args.points.split(",") if name)
    unknown = sorted(set(requested_names) - set(points_by_name))
    if unknown:
        raise ValueError(f"unknown audit points: {unknown}")

    started = time.perf_counter()
    results = [
        audit_point(
            points_by_name[name],
            x=args.x,
            orders=orders,
            tolerance=args.tolerance,
            old_order=args.old_order,
            p_max=args.p_max,
            power=args.power,
            necklace_orders=necklace_orders,  # type: ignore[arg-type]
            ope_orders=ope_orders,  # type: ignore[arg-type]
            dps=args.dps,
        )
        for name in requested_names
    ]
    payload: dict[str, object] = {
        "calculation": "genus-one two-point pointwise adaptive momentum audit",
        "x": float(args.x),
        "omega": _complex_record(1.0j * args.x),
        "threshold_justification": (
            "Each ordinary internal edge has the exact b=1 DOZZ P^2 zero. "
            "The analytic collision-disc 1/P_ope^2 factor cancels that zero "
            "on the OPE edge, so that edge uses alpha=-1/2."
        ),
        "block_orders": {
            "necklace": list(necklace_orders),
            "ope": list(ope_orders),
        },
        "total_runtime_seconds": float(time.perf_counter() - started),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output}", flush=True)
    return payload


if __name__ == "__main__":
    run()
