#!/usr/bin/env python3
"""Separating-channel genus-two Liouville sewing with Zhu descendants.

This module is an experimental bridge between the existing primary
pair-of-tori approximation and a full descendant sewing.  It keeps the primary
torus one-point block from Zamolodchikov recursion and uses Zhu recursion to
generate bridge-descendant torus one-point blocks in one separating plumbing
frame.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np

try:
    from liouville_genus2 import format_complex, parse_complex
    from liouville_torus import (
        UpsilonB,
        estimate_p_max,
        lambda_from_yin_momentum,
        liouville_weight_from_lambda,
        q_from_tau,
        validate_nonresonant_b_for_block,
        yin_structure_constant_momentum,
    )
    from torus_descendant_blocks import (
        State,
        gram_matrix,
        integer_partitions,
        torus_one_point_descendant_coefficients,
    )
    from zhu_torus_descendants import (
        ZhuSeries,
        primary_torus_zhu_series,
        zhu_descendant_series,
    )
except ImportError:  # pragma: no cover - supports package-style execution
    from plumbing.liouville_genus2 import format_complex, parse_complex
    from plumbing.liouville_torus import (
        UpsilonB,
        estimate_p_max,
        lambda_from_yin_momentum,
        liouville_weight_from_lambda,
        q_from_tau,
        validate_nonresonant_b_for_block,
        yin_structure_constant_momentum,
    )
    from plumbing.torus_descendant_blocks import (
        State,
        gram_matrix,
        integer_partitions,
        torus_one_point_descendant_coefficients,
    )
    from plumbing.zhu_torus_descendants import (
        ZhuSeries,
        primary_torus_zhu_series,
        zhu_descendant_series,
    )


def _validate_plumbing_q(name: str, value: complex) -> complex:
    value = complex(value)
    if not 0 < abs(value) < 1:
        raise ValueError(f"{name} must satisfy 0 < |{name}| < 1")
    return value


def liouville_central_charge(b: float) -> float:
    q_background = b + 1.0 / b
    return 1.0 + 6.0 * q_background * q_background


def liouville_weight_from_yin_momentum(b: float, momentum: float | complex) -> complex:
    q_background = b + 1.0 / b
    momentum = complex(momentum)
    return 0.25 * q_background * q_background + momentum * momentum


@dataclass(frozen=True)
class TorusZhuDescendantMatrices:
    max_level: int
    bases_by_level: tuple[tuple[State, ...], ...]
    matrices: tuple[tuple[np.ndarray, ...], ...]


@dataclass(frozen=True)
class LiouvilleGenus2SeparatingZhuSample:
    bridge_momentum: float
    bridge_measure_weight: float
    bridge_prefactor: float
    shell_contributions: tuple[complex, ...]
    contribution: complex


@dataclass(frozen=True)
class LiouvilleGenus2SeparatingZhuResult:
    value: complex
    q1: complex
    q2: complex
    q_bridge: complex
    b: float
    central_charge: float
    block_order: int
    bridge_level: int
    bridge_p_max: float
    handle_p_max_left: float
    handle_p_max_right: float
    bridge_quadrature_order: int
    handle_quadrature_order: int
    dps: int
    torus_series_method: str
    descendant_basis: str
    include_bridge_vacuum_energy: bool
    include_cosmological_prefactor: bool
    samples: tuple[LiouvilleGenus2SeparatingZhuSample, ...]


class _TorusZhuWorkspace:
    def __init__(
        self,
        *,
        b: float,
        q: complex,
        block_order: int,
        p_max: float,
        quadrature_order: int,
        special: UpsilonB,
        mu: complex,
        include_cosmological_prefactor: bool,
        torus_series_method: str,
        descendant_basis: str,
    ) -> None:
        if p_max <= 0:
            raise ValueError("handle p_max must be positive")
        if quadrature_order <= 0:
            raise ValueError("handle quadrature_order must be positive")

        self.b = float(b)
        self.q = _validate_plumbing_q("q", q)
        self.block_order = int(block_order)
        self.p_max = float(p_max)
        self.quadrature_order = int(quadrature_order)
        self.special = special
        self.mu = complex(mu)
        self.include_cosmological_prefactor = bool(include_cosmological_prefactor)
        if torus_series_method not in {"recursion", "direct"}:
            raise ValueError("torus_series_method must be 'recursion' or 'direct'")
        self.torus_series_method = torus_series_method
        if descendant_basis not in {"zhu", "ordinary"}:
            raise ValueError("descendant_basis must be 'zhu' or 'ordinary'")
        self.descendant_basis = descendant_basis
        self.central_charge = liouville_central_charge(self.b)

        nodes, weights = np.polynomial.legendre.leggauss(self.quadrature_order)
        midpoint = 0.5 * self.p_max
        self._nodes = tuple(float(midpoint * (node + 1.0)) for node in nodes)
        self._measure_weights = tuple(float(midpoint * weight / math.pi) for weight in weights)

    def matrices(self, external_momentum: float, max_level: int) -> TorusZhuDescendantMatrices:
        if max_level < 0:
            raise ValueError("max_level must be non-negative")
        external_lambda = lambda_from_yin_momentum(external_momentum)
        external_weight = liouville_weight_from_lambda(self.b, external_lambda)
        bases_by_level = tuple(integer_partitions(level) for level in range(max_level + 1))
        matrices = [
            [
                np.zeros((len(bases_by_level[left_level]), len(bases_by_level[right_level])), dtype=np.complex128)
                for right_level in range(max_level + 1)
            ]
            for left_level in range(max_level + 1)
        ]
        for p, measure_weight in zip(self._nodes, self._measure_weights):
            internal_weight = liouville_weight_from_yin_momentum(self.b, p)
            structure_constant = yin_structure_constant_momentum(
                self.special,
                p,
                external_momentum,
                p,
                mu=self.mu,
                include_cosmological_prefactor=self.include_cosmological_prefactor,
            )
            primary: ZhuSeries = primary_torus_zhu_series(
                self.central_charge,
                internal_weight,
                external_weight,
                self.block_order,
                b=self.b,
                external_lambda=external_lambda,
                method=self.torus_series_method,
            )
            values_by_level: list[np.ndarray] = []
            for level, basis in enumerate(bases_by_level):
                values = np.zeros(len(basis), dtype=np.complex128)
                for state_index, state in enumerate(basis):
                    if self.descendant_basis == "zhu":
                        descendant = zhu_descendant_series(
                            primary,
                            state,
                            external_weight=external_weight,
                            central_charge=self.central_charge,
                        )
                        values[state_index] = descendant.value(self.q)
                    else:
                        coeffs = torus_one_point_descendant_coefficients(
                            self.central_charge,
                            internal_weight,
                            external_weight,
                            state,
                            self.block_order,
                        )
                        values[state_index] = (self.q ** (internal_weight - self.central_charge / 24.0)) * sum(
                            coeff * (self.q**idx) for idx, coeff in enumerate(coeffs)
                        )
                values_by_level.append(values)
            weighted_structure = measure_weight * structure_constant
            for hol_level in range(max_level + 1):
                hol_values = values_by_level[hol_level]
                for anti_level in range(max_level + 1):
                    matrices[hol_level][anti_level] += weighted_structure * np.outer(
                        hol_values,
                        values_by_level[anti_level].conjugate(),
                    )

        return TorusZhuDescendantMatrices(
            max_level=max_level,
            bases_by_level=bases_by_level,
            matrices=tuple(tuple(matrix for matrix in row) for row in matrices),
        )


def bridge_prefactor(
    *,
    q_bridge: complex,
    b: float,
    bridge_momentum: float,
    include_bridge_vacuum_energy: bool,
) -> float:
    h = liouville_weight_from_yin_momentum(b, bridge_momentum)
    c = liouville_central_charge(b)
    exponent = h - c / 24.0 if include_bridge_vacuum_energy else h
    if abs(exponent.imag) > 1.0e-13:
        raise ValueError("bridge exponent has unexpected imaginary part")
    return abs(q_bridge) ** (2.0 * float(exponent.real))


def bridge_inverse_gram_matrices(
    *,
    bridge_weight: complex,
    central_charge: complex,
    max_level: int,
) -> tuple[tuple[State, ...], tuple[np.ndarray, ...]]:
    bases: list[tuple[State, ...]] = []
    inverse_grams: list[np.ndarray] = []
    for level in range(max_level + 1):
        basis, gram = gram_matrix(bridge_weight, central_charge, level)
        bases.append(basis)
        inverse_grams.append(np.linalg.inv(gram))
    return tuple(bases), tuple(inverse_grams)


def separating_zhu_bridge_expression(
    *,
    q_bridge: complex,
    max_level: int,
    inverse_grams: tuple[np.ndarray, ...],
    left: TorusZhuDescendantMatrices,
    right: TorusZhuDescendantMatrices,
) -> tuple[complex, tuple[complex, ...]]:
    if max_level < 0:
        raise ValueError("max_level must be non-negative")
    if left.bases_by_level != right.bases_by_level:
        raise ValueError("left and right torus matrices use different descendant bases")
    shell_contributions = [0.0 + 0.0j for _ in range(max_level + 1)]
    total = 0.0 + 0.0j
    for hol_level in range(max_level + 1):
        q_hol = q_bridge**hol_level
        inverse_hol = inverse_grams[hol_level]
        for anti_level in range(max_level + 1):
            q_anti = q_bridge.conjugate() ** anti_level
            inverse_anti = inverse_grams[anti_level].conjugate()
            term = np.einsum(
                "ab,ij,ai,bj->",
                inverse_hol,
                inverse_anti,
                left.matrices[hol_level][anti_level],
                right.matrices[hol_level][anti_level],
                optimize=True,
            )
            weighted_term = q_hol * q_anti * term
            shell_contributions[max(hol_level, anti_level)] += weighted_term
            total += weighted_term
    return total, tuple(shell_contributions)


def liouville_genus2_separating_zhu(
    *,
    b: float,
    q1: complex,
    q2: complex,
    q_bridge: complex,
    block_order: int,
    bridge_level: int = 2,
    bridge_p_max: float | None = None,
    handle_p_max: float | None = None,
    handle_p_max_left: float | None = None,
    handle_p_max_right: float | None = None,
    bridge_quadrature_order: int = 6,
    handle_quadrature_order: int = 8,
    dps: int = 35,
    mu: complex = 1.0,
    torus_series_method: str = "recursion",
    descendant_basis: str = "zhu",
    include_bridge_vacuum_energy: bool = True,
    include_cosmological_prefactor: bool = False,
    tail_tolerance: float = 1.0e-12,
    safety_margin: float = 1.0,
) -> LiouvilleGenus2SeparatingZhuResult:
    if b <= 0:
        raise ValueError("b must be positive")
    if block_order < 0:
        raise ValueError("block_order must be non-negative")
    if bridge_level < 0:
        raise ValueError("bridge_level must be non-negative")
    if torus_series_method not in {"recursion", "direct"}:
        raise ValueError("torus_series_method must be 'recursion' or 'direct'")
    if descendant_basis not in {"zhu", "ordinary"}:
        raise ValueError("descendant_basis must be 'zhu' or 'ordinary'")

    q1 = _validate_plumbing_q("q1", q1)
    q2 = _validate_plumbing_q("q2", q2)
    q_bridge = _validate_plumbing_q("q_bridge", q_bridge)
    if torus_series_method == "recursion":
        validate_nonresonant_b_for_block(b, block_order)

    if bridge_p_max is None:
        bridge_p_max = estimate_p_max(q_bridge, tail_tolerance=tail_tolerance, safety_margin=safety_margin)
    if handle_p_max is not None and (handle_p_max_left is not None or handle_p_max_right is not None):
        raise ValueError("use either handle_p_max or left/right cutoffs, not both")
    if handle_p_max is None:
        if handle_p_max_left is None:
            handle_p_max_left = estimate_p_max(q1, tail_tolerance=tail_tolerance, safety_margin=safety_margin)
        if handle_p_max_right is None:
            handle_p_max_right = estimate_p_max(q2, tail_tolerance=tail_tolerance, safety_margin=safety_margin)
    else:
        handle_p_max_left = handle_p_max
        handle_p_max_right = handle_p_max

    special = UpsilonB(b=b, dps=dps)
    left_workspace = _TorusZhuWorkspace(
        b=b,
        q=q1,
        block_order=block_order,
        p_max=float(handle_p_max_left),
        quadrature_order=handle_quadrature_order,
        special=special,
        mu=mu,
        include_cosmological_prefactor=include_cosmological_prefactor,
        torus_series_method=torus_series_method,
        descendant_basis=descendant_basis,
    )
    right_workspace = _TorusZhuWorkspace(
        b=b,
        q=q2,
        block_order=block_order,
        p_max=float(handle_p_max_right),
        quadrature_order=handle_quadrature_order,
        special=special,
        mu=mu,
        include_cosmological_prefactor=include_cosmological_prefactor,
        torus_series_method=torus_series_method,
        descendant_basis=descendant_basis,
    )

    nodes, weights = np.polynomial.legendre.leggauss(bridge_quadrature_order)
    midpoint = 0.5 * float(bridge_p_max)
    samples: list[LiouvilleGenus2SeparatingZhuSample] = []
    total = 0.0 + 0.0j
    for node, weight in zip(nodes, weights):
        bridge_momentum = float(midpoint * (node + 1.0))
        bridge_measure_weight = float(midpoint * weight / math.pi)
        h_bridge = liouville_weight_from_yin_momentum(b, bridge_momentum)
        inverse_bases, inverse_grams = bridge_inverse_gram_matrices(
            bridge_weight=h_bridge,
            central_charge=liouville_central_charge(b),
            max_level=bridge_level,
        )
        prefactor = bridge_prefactor(
            q_bridge=q_bridge,
            b=b,
            bridge_momentum=bridge_momentum,
            include_bridge_vacuum_energy=include_bridge_vacuum_energy,
        )
        left = left_workspace.matrices(bridge_momentum, bridge_level)
        right = right_workspace.matrices(bridge_momentum, bridge_level)
        if inverse_bases != left.bases_by_level:
            raise ValueError("bridge inverse Gram matrices use a different descendant basis")
        bridge_expression, shell_contributions = separating_zhu_bridge_expression(
            q_bridge=q_bridge,
            max_level=bridge_level,
            inverse_grams=inverse_grams,
            left=left,
            right=right,
        )
        contribution = bridge_measure_weight * prefactor * bridge_expression
        samples.append(
            LiouvilleGenus2SeparatingZhuSample(
                bridge_momentum=bridge_momentum,
                bridge_measure_weight=bridge_measure_weight,
                bridge_prefactor=prefactor,
                shell_contributions=tuple(bridge_measure_weight * prefactor * value for value in shell_contributions),
                contribution=contribution,
            )
        )
        total += contribution

    central_charge = liouville_central_charge(b)
    return LiouvilleGenus2SeparatingZhuResult(
        value=total,
        q1=q1,
        q2=q2,
        q_bridge=q_bridge,
        b=float(b),
        central_charge=float(central_charge),
        block_order=int(block_order),
        bridge_level=int(bridge_level),
        bridge_p_max=float(bridge_p_max),
        handle_p_max_left=float(handle_p_max_left),
        handle_p_max_right=float(handle_p_max_right),
        bridge_quadrature_order=int(bridge_quadrature_order),
        handle_quadrature_order=int(handle_quadrature_order),
        dps=int(dps),
        torus_series_method=torus_series_method,
        descendant_basis=descendant_basis,
        include_bridge_vacuum_energy=bool(include_bridge_vacuum_energy),
        include_cosmological_prefactor=bool(include_cosmological_prefactor),
        samples=tuple(samples),
    )


def _resolve_q(name: str, q_value: complex | None, tau_value: complex | None) -> complex:
    if (q_value is None) == (tau_value is None):
        raise ValueError(f"provide exactly one of --{name} or --tau-{name[-1]}")
    return q_value if q_value is not None else q_from_tau(tau_value)


def handle_s1_transformed_q(tau1: complex, q_bridge: complex) -> tuple[complex, complex]:
    """Leading separating-channel handle-S transform for the first torus."""
    tau1_s = -1.0 / tau1
    return q_from_tau(tau1_s), q_bridge / tau1


def run(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Zhu separating-channel genus-two Liouville sewing.")
    parser.add_argument("--b", type=float, required=True)
    parser.add_argument("--q1", type=parse_complex)
    parser.add_argument("--q2", type=parse_complex)
    parser.add_argument("--tau-1", type=parse_complex)
    parser.add_argument("--tau-2", type=parse_complex)
    parser.add_argument("--q-bridge", type=parse_complex, required=True)
    parser.add_argument("--block-order", type=int, default=3)
    parser.add_argument("--bridge-level", type=int, default=5)
    parser.add_argument("--bridge-p-max", type=float)
    parser.add_argument("--handle-p-max", type=float)
    parser.add_argument("--handle-p-max-left", type=float)
    parser.add_argument("--handle-p-max-right", type=float)
    parser.add_argument("--bridge-quadrature-order", type=int, default=6)
    parser.add_argument("--handle-quadrature-order", type=int, default=8)
    parser.add_argument("--dps", type=int, default=35)
    parser.add_argument("--mu", type=parse_complex, default=1.0 + 0.0j)
    parser.add_argument("--torus-series-method", choices=["recursion", "direct"], default="recursion")
    parser.add_argument("--descendant-basis", choices=["zhu", "ordinary"], default="zhu")
    parser.add_argument("--include-cosmological-prefactor", action="store_true")
    parser.add_argument("--handle-s1-check", action="store_true")
    args = parser.parse_args(argv)

    q1 = _resolve_q("q1", args.q1, args.tau_1)
    q2 = _resolve_q("q2", args.q2, args.tau_2)
    result = liouville_genus2_separating_zhu(
        b=args.b,
        q1=q1,
        q2=q2,
        q_bridge=args.q_bridge,
        block_order=args.block_order,
        bridge_level=args.bridge_level,
        bridge_p_max=args.bridge_p_max,
        handle_p_max=args.handle_p_max,
        handle_p_max_left=args.handle_p_max_left,
        handle_p_max_right=args.handle_p_max_right,
        bridge_quadrature_order=args.bridge_quadrature_order,
        handle_quadrature_order=args.handle_quadrature_order,
        dps=args.dps,
        mu=args.mu,
        torus_series_method=args.torus_series_method,
        descendant_basis=args.descendant_basis,
        include_cosmological_prefactor=args.include_cosmological_prefactor,
    )

    print("Liouville genus-two separating Zhu sewing")
    print(f"  b={result.b:.12g}")
    print(f"  c={result.central_charge:.12g}")
    print(f"  q1={format_complex(result.q1)}")
    print(f"  q2={format_complex(result.q2)}")
    print(f"  q_bridge={format_complex(result.q_bridge)}")
    print(f"  bridge max level={result.bridge_level}")
    print(f"  block order={result.block_order}")
    print(f"  torus series method={result.torus_series_method}")
    print(f"  descendant basis={result.descendant_basis}")
    print(f"  bridge P cutoff={result.bridge_p_max:.12g}")
    print(f"  handle P cutoffs=({result.handle_p_max_left:.12g}, {result.handle_p_max_right:.12g})")
    print(f"  quadrature orders=({result.bridge_quadrature_order}, {result.handle_quadrature_order})")
    print(f"  value={format_complex(result.value)}")
    shell_totals = [sum(sample.shell_contributions[level] for sample in result.samples) for level in range(result.bridge_level + 1)]
    print("  bridge shell contributions by max(hol,anti) level:")
    for level, contribution in enumerate(shell_totals):
        print(f"    N={level}: {format_complex(contribution)}")

    if args.handle_s1_check:
        if args.tau_1 is None:
            parser.error("--handle-s1-check requires --tau-1 so the analytic factor is unambiguous")
        q1_s, q_bridge_s = handle_s1_transformed_q(args.tau_1, args.q_bridge)
        transformed = liouville_genus2_separating_zhu(
            b=args.b,
            q1=q1_s,
            q2=q2,
            q_bridge=q_bridge_s,
            block_order=args.block_order,
            bridge_level=args.bridge_level,
            bridge_p_max=args.bridge_p_max,
            handle_p_max=args.handle_p_max,
            handle_p_max_left=args.handle_p_max_left,
            handle_p_max_right=args.handle_p_max_right,
            bridge_quadrature_order=args.bridge_quadrature_order,
            handle_quadrature_order=args.handle_quadrature_order,
            dps=args.dps,
            mu=args.mu,
            torus_series_method=args.torus_series_method,
            descendant_basis=args.descendant_basis,
            include_cosmological_prefactor=args.include_cosmological_prefactor,
        )
        observed_ratio = transformed.value / result.value
        expected_ratio = abs(args.tau_1) ** (-result.central_charge)
        print("  handle S1 leading-chart check:")
        print(f"    q1_S={format_complex(q1_s)}")
        print(f"    q_bridge_S={format_complex(q_bridge_s)}")
        print(f"    transformed value={format_complex(transformed.value)}")
        print(f"    observed ratio={format_complex(observed_ratio)}")
        print(f"    expected |tau1|^-c={expected_ratio:.12e}")
        print(f"    observed/expected={format_complex(observed_ratio / expected_ratio)}")


if __name__ == "__main__":
    run()
