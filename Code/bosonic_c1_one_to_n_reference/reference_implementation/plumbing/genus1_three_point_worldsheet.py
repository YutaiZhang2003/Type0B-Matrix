#!/usr/bin/env python3
"""Blind worldsheet kernel for the c=1 genus-one 1->2 amplitude.

The incoming energy is ``omega_in=i*t`` and, in the first symmetric slice,
the two outgoing energies are ``omega_out=i*t/2``. Only the torus necklace
channel is used. The Liouville block can be evaluated either with the
three-edge h-recursion at two nearby central charges and linearly extrapolated
to c=25, or by the equivalent finite-level Gram-matrix sewing at exactly
c=25. The latter avoids resonant intermediate h-recursion residues at b=1 and
is used for the higher momentum-order smoke run. No matrix-model expression
is imported or evaluated here.
"""

from __future__ import annotations

import cmath
import concurrent.futures
import json
import math
import os
from pathlib import Path
from dataclasses import dataclass
from itertools import product

import numpy as np

try:
    from genus1_two_point_worldsheet import (
        MomentumRule,
        dedekind_eta,
        torus_prime_form_norm,
    )
    from liouville_torus import UpsilonB, yin_structure_constant_momentum
    from torus_two_point_blocks import (
        elliptic_nome,
        necklace_coefficients_in_elliptic_nomes_nd,
    )
    from torus_three_point_blocks import necklace_descendant_coefficients_three_point
    from virasoro_blocks import TorusThreePointVirasoroBlock
except ImportError:  # pragma: no cover - package-style execution
    from plumbing.genus1_two_point_worldsheet import (
        MomentumRule,
        dedekind_eta,
        torus_prime_form_norm,
    )
    from plumbing.liouville_torus import UpsilonB, yin_structure_constant_momentum
    from plumbing.torus_two_point_blocks import (
        elliptic_nome,
        necklace_coefficients_in_elliptic_nomes_nd,
    )
    from plumbing.torus_three_point_blocks import necklace_descendant_coefficients_three_point
    from plumbing.virasoro_blocks import TorusThreePointVirasoroBlock


C_LIOUVILLE = 25.0


@dataclass(frozen=True)
class _ThreePointBlockBank:
    """Vectorized data for one rectangular edge-order assignment."""

    orders: tuple[int, int, int]
    internal_weights: tuple[np.ndarray, np.ndarray, np.ndarray]
    weighted_structure_constants: np.ndarray
    coefficients: np.ndarray


def _regulated_h_recursion_coefficients(
    internal_weights: tuple[float, float, float],
    external_weights: tuple[float, float, float],
    orders: tuple[int, int, int],
    *,
    c_regulator: float,
) -> np.ndarray:
    r"""Return the finite c->25 block by a two-point linear extrapolation.

    At b=1, separate h-recursion residues contain resonant zero denominators.
    The summed necklace block is regular.  Evaluating at ``25+eps`` and
    ``25+2*eps`` and forming ``2 F(eps)-F(2 eps)`` removes the leading
    regulator dependence while avoiding the singular intermediate terms.
    """

    epsilon = float(c_regulator)
    if epsilon <= 0.0:
        raise ValueError("the c=25 h-recursion regulator must be positive")

    def coefficients_at(central_charge: float) -> np.ndarray:
        block = TorusThreePointVirasoroBlock(
            central_charge,
            *internal_weights,
            *external_weights,
        )
        try:
            return block.descendant_coefficients(orders)
        finally:
            # ``_reduced_coefficient`` is decorated on the class with an
            # unbounded lru_cache.  Its keys include ``self``, so without an
            # explicit clear the class-level wrapper retains every completed
            # block and all of its recursive states.  A bank worker evaluates
            # hundreds of independent momentum triples; retaining those
            # states made its RSS grow until Slurm killed the process.  Each
            # regulated evaluation is independent, and one worker evaluates
            # only one block at a time, so the cache must end with that block.
            block._reduced_coefficient.cache_clear()

    first = coefficients_at(C_LIOUVILLE + epsilon)
    second = coefficients_at(C_LIOUVILLE + 2.0 * epsilon)
    return 2.0 * first - second


def _regulated_h_recursion_task(
    task: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[int, int, int],
        float,
    ],
) -> np.ndarray:
    """Pickle-friendly worker entry point for one regulated block tensor."""
    internal_weights, external_weights, orders, c_regulator = task
    return _regulated_h_recursion_coefficients(
        internal_weights,
        external_weights,
        orders,
        c_regulator=c_regulator,
    )


def evaluate_pade_rows(
    coefficients: np.ndarray,
    value: complex,
    numerator_order: int,
    denominator_order: int,
) -> np.ndarray:
    """Evaluate one Padé approximant for every row of series coefficients."""
    coefficients = np.asarray(coefficients, dtype=np.complex128)
    if coefficients.ndim != 2:
        raise ValueError("Padé row coefficients must form a matrix")
    numerator_order = int(numerator_order)
    denominator_order = int(denominator_order)
    required_order = numerator_order + denominator_order
    if numerator_order < 0 or denominator_order <= 0:
        raise ValueError("Padé orders must satisfy L>=0 and M>0")
    if coefficients.shape[1] <= required_order:
        raise ValueError("not enough series coefficients for the requested Padé order")

    row_count = coefficients.shape[0]
    # A descendant series can span hundreds of orders of magnitude across
    # momentum nodes.  Padé is invariant under an overall rescaling of one
    # row, so normalize before forming its linear system.  This prevents an
    # algebraically harmless large common factor from overflowing either the
    # solve or the numerator convolution.
    row_scales = np.max(np.abs(coefficients[:, : required_order + 1]), axis=1)
    row_scales = np.where(row_scales > 0.0, row_scales, 1.0)
    series = coefficients[:, : required_order + 1] / row_scales[:, np.newaxis]
    matrices = np.empty(
        (row_count, denominator_order, denominator_order),
        dtype=np.complex128,
    )
    right_hand_sides = np.empty((row_count, denominator_order), dtype=np.complex128)
    for row in range(denominator_order):
        series_index = numerator_order + 1 + row
        right_hand_sides[:, row] = -series[:, series_index]
        for column in range(denominator_order):
            matrices[:, row, column] = series[
                :,
                series_index - (column + 1),
            ]
    try:
        denominator_tail = np.linalg.solve(
            matrices,
            right_hand_sides[..., np.newaxis],
        )[..., 0]
    except np.linalg.LinAlgError:
        denominator_tail = np.empty_like(right_hand_sides)
        for row in range(row_count):
            denominator_tail[row], *_ = np.linalg.lstsq(
                matrices[row],
                right_hand_sides[row],
                rcond=None,
            )
    denominator = np.column_stack(
        [np.ones(row_count, dtype=np.complex128), denominator_tail]
    )
    numerator = np.zeros((row_count, numerator_order + 1), dtype=np.complex128)
    for order in range(numerator_order + 1):
        for denominator_index in range(min(order, denominator_order) + 1):
            numerator[:, order] += (
                denominator[:, denominator_index]
                * series[:, order - denominator_index]
            )
    rational_scales = np.maximum(
        np.max(np.abs(numerator), axis=1),
        np.max(np.abs(denominator), axis=1),
    )
    rational_scales = np.where(rational_scales > 0.0, rational_scales, 1.0)
    numerator /= rational_scales[:, np.newaxis]
    denominator /= rational_scales[:, np.newaxis]
    powers_numerator = complex(value) ** np.arange(numerator_order + 1)
    powers_denominator = complex(value) ** np.arange(denominator_order + 1)
    numerator_value = numerator @ powers_numerator
    denominator_value = denominator @ powers_denominator
    result = row_scales * numerator_value / denominator_value
    if not np.all(np.isfinite(result)):
        raise FloatingPointError("non-finite row-wise Padé approximant")
    return result


class LiouvilleTorusThreePointNecklace:
    """Three-momentum necklace representation of the torus 1->2 correlator."""

    def __init__(
        self,
        t: float,
        *,
        momentum_rule: MomentumRule | None = None,
        momentum_rules: tuple[MomentumRule, MomentumRule, MomentumRule] | None = None,
        high_order: int = 6,
        low_order: int = 2,
        adaptive_tolerance: float = 5.0e-5,
        c_regulator: float = 0.05,
        block_backend: str = "regulated-h-recursion",
        special_dps: int = 28,
        coefficient_workers: int = 1,
    ) -> None:
        t = float(t)
        if not 0.0 < t < 1.0:
            raise ValueError("the direct equal-split real-contour slice requires 0<t<1")
        if high_order < low_order:
            raise ValueError("high_order must be at least low_order")
        if low_order < 0:
            raise ValueError("block orders must be non-negative")
        if not 0.0 < adaptive_tolerance < 1.0:
            raise ValueError("adaptive_tolerance must lie between zero and one")

        self.t = t
        self.omega_in = 1.0j * t
        self.omega_out = 0.5j * t
        self.external_momenta = (
            self.omega_out / 2.0,
            self.omega_out / 2.0,
            self.omega_in / 2.0,
        )
        self.external_weights = tuple(
            float((1.0 + omega * omega / 4.0).real)
            for omega in (self.omega_out, self.omega_out, self.omega_in)
        )
        self.signed_energies = (self.omega_in, -self.omega_out, -self.omega_out)
        if (momentum_rule is None) == (momentum_rules is None):
            raise ValueError("provide exactly one of momentum_rule or momentum_rules")
        if momentum_rules is None:
            assert momentum_rule is not None
            momentum_rules = (momentum_rule, momentum_rule, momentum_rule)
        if len(momentum_rules) != 3:
            raise ValueError("three internal momentum rules are required")
        self.momentum_rules = tuple(momentum_rules)
        self.momentum_rule = momentum_rule
        self.high_order = int(high_order)
        self.low_order = int(low_order)
        self.adaptive_tolerance = float(adaptive_tolerance)
        self.c_regulator = float(c_regulator)
        if block_backend not in {"regulated-h-recursion", "exact-c25-descendants"}:
            raise ValueError("unknown torus three-point block backend")
        self.block_backend = str(block_backend)
        self.special = UpsilonB(1.0, dps=int(special_dps))
        self.special_dps = int(special_dps)
        self.coefficient_workers = int(coefficient_workers)
        if self.coefficient_workers < 1:
            raise ValueError("coefficient_workers must be positive")
        self._banks: dict[int, _ThreePointBlockBank] = {}
        self._cached_structure_matrices: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
        self.order_histogram = {order: 0 for order in range(self.low_order, self.high_order + 1)}
        self.high_edge_histogram = {edge: 0 for edge in range(3)}
        self.maximum_hat_q_seen = 0.0
        self.maximum_second_hat_q_seen = 0.0

    def _structure_matrices(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self._cached_structure_matrices is not None:
            return self._cached_structure_matrices
        matrices: list[np.ndarray] = []
        for edge, external_momentum in enumerate(self.external_momenta):
            first_nodes = np.asarray(self.momentum_rules[edge].nodes, dtype=float)
            second_nodes = np.asarray(
                self.momentum_rules[(edge + 1) % 3].nodes,
                dtype=float,
            )
            matrix = np.empty((len(first_nodes), len(second_nodes)), dtype=np.complex128)
            for first_index, first in enumerate(first_nodes):
                for second_index, second in enumerate(second_nodes):
                    matrix[first_index, second_index] = yin_structure_constant_momentum(
                        self.special,
                        external_momentum,
                        float(first),
                        float(second),
                    )
            matrices.append(matrix)
        self._cached_structure_matrices = tuple(matrices)  # type: ignore[assignment]
        return self._cached_structure_matrices

    def _build_bank(self, high_edge: int) -> _ThreePointBlockBank:
        if high_edge not in (0, 1, 2):
            raise ValueError("high_edge must be 0, 1, or 2")
        orders_list = [self.low_order] * 3
        orders_list[high_edge] = self.high_order
        orders = tuple(orders_list)
        nodes = tuple(
            np.asarray(rule.nodes, dtype=float) for rule in self.momentum_rules
        )
        quadrature_weights = tuple(
            np.asarray(rule.weights, dtype=float) for rule in self.momentum_rules
        )
        structures = self._structure_matrices()

        h_arrays = [[], [], []]
        weighted_structures: list[complex] = []
        coefficient_tensors: list[np.ndarray] = []
        index_rows = tuple(product(
            range(len(nodes[0])),
            range(len(nodes[1])),
            range(len(nodes[2])),
        ))
        coefficient_rows: list[np.ndarray] | None = None
        if self.block_backend == "regulated-h-recursion" and self.coefficient_workers > 1:
            tasks = []
            for indices in index_rows:
                momenta = tuple(
                    float(nodes[edge][index]) for edge, index in enumerate(indices)
                )
                internal_weights = tuple(1.0 + momentum * momentum for momentum in momenta)
                tasks.append(
                    (internal_weights, self.external_weights, orders, self.c_regulator)
                )
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=self.coefficient_workers
            ) as executor:
                coefficient_rows = list(executor.map(_regulated_h_recursion_task, tasks))

        for flat_index, (first_index, second_index, third_index) in enumerate(index_rows):
            indices = (first_index, second_index, third_index)
            momenta = tuple(float(nodes[edge][index]) for edge, index in enumerate(indices))
            internal_weights = tuple(1.0 + momentum * momentum for momentum in momenta)
            structure_product = (
                structures[0][first_index, second_index]
                * structures[1][second_index, third_index]
                * structures[2][third_index, first_index]
            )
            quadrature_weight = (
                quadrature_weights[0][first_index]
                * quadrature_weights[1][second_index]
                * quadrature_weights[2][third_index]
                / math.pi**3
            )
            if self.block_backend == "regulated-h-recursion":
                if coefficient_rows is None:
                    raw_coefficients = _regulated_h_recursion_coefficients(
                        internal_weights,
                        self.external_weights,
                        orders,
                        c_regulator=self.c_regulator,
                    )
                else:
                    raw_coefficients = coefficient_rows[flat_index]
            else:
                raw_coefficients = necklace_descendant_coefficients_three_point(
                    C_LIOUVILLE,
                    internal_weights,
                    self.external_weights,
                    orders,
                )
            coefficients = necklace_coefficients_in_elliptic_nomes_nd(
                raw_coefficients,
                orders,
            )
            for edge in range(3):
                h_arrays[edge].append(internal_weights[edge])
            weighted_structures.append(quadrature_weight * structure_product)
            coefficient_tensors.append(coefficients)
            if (flat_index + 1) % 32 == 0 or flat_index + 1 == len(index_rows):
                print(
                    f"three-point {self.block_backend} bank progress "
                    f"{flat_index + 1}/{len(index_rows)} for high edge {high_edge}",
                    flush=True,
                )

        return _ThreePointBlockBank(
            orders=orders,
            internal_weights=tuple(
                np.asarray(values, dtype=float) for values in h_arrays
            ),  # type: ignore[arg-type]
            weighted_structure_constants=np.asarray(weighted_structures, dtype=complex),
            coefficients=np.stack(coefficient_tensors),
        )

    def prepare(self, *, checkpoint_path: str | Path | None = None) -> None:
        """Build all three anisotropic banks before starting moduli sampling."""
        for edge in range(3):
            if edge not in self._banks:
                self._banks[edge] = self._build_bank(edge)
                print(
                    f"prepared three-point {self.block_backend} bank for high edge {edge} "
                    f"with orders {self._banks[edge].orders}",
                    flush=True,
                )
                if checkpoint_path is not None:
                    self.save_banks(checkpoint_path, prepare_missing=False)
                    print(
                        f"checkpointed necklace banks through high edge {edge} "
                        f"to {checkpoint_path}",
                        flush=True,
                    )

    def save_banks(
        self,
        path: str | Path,
        *,
        prepare_missing: bool = True,
    ) -> None:
        """Persist prepared numerical banks for restartable high-order runs."""
        if prepare_missing:
            self.prepare()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "format": "torus-three-point-bank-v1",
            "t": self.t,
            "external_weights": list(self.external_weights),
            "high_order": self.high_order,
            "low_order": self.low_order,
            "adaptive_tolerance": self.adaptive_tolerance,
            "block_backend": self.block_backend,
            "momentum_nodes": [rule.nodes.tolist() for rule in self.momentum_rules],
            "momentum_weights": [rule.weights.tolist() for rule in self.momentum_rules],
        }
        if self.block_backend == "regulated-h-recursion":
            metadata["c_regulator"] = self.c_regulator
        arrays: dict[str, np.ndarray] = {
            "metadata": np.asarray(json.dumps(metadata)),
            "edge_indices": np.asarray(sorted(self._banks), dtype=int),
        }
        for edge, bank in self._banks.items():
            arrays[f"edge{edge}_orders"] = np.asarray(bank.orders, dtype=int)
            arrays[f"edge{edge}_weighted"] = bank.weighted_structure_constants
            arrays[f"edge{edge}_coefficients"] = bank.coefficients
            for internal_edge, values in enumerate(bank.internal_weights):
                arrays[f"edge{edge}_internal{internal_edge}"] = values
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp.npz")
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)

    def load_banks(self, path: str | Path) -> None:
        """Load banks saved by :meth:`save_banks`, failing closed on mismatch."""
        path = Path(path)
        with np.load(path, allow_pickle=False) as payload:
            metadata = json.loads(str(payload["metadata"].item()))
            expected = {
                "format": "torus-three-point-bank-v1",
                "t": self.t,
                "external_weights": list(self.external_weights),
                "high_order": self.high_order,
                "low_order": self.low_order,
                "adaptive_tolerance": self.adaptive_tolerance,
                "block_backend": self.block_backend,
                "momentum_nodes": [rule.nodes.tolist() for rule in self.momentum_rules],
                "momentum_weights": [rule.weights.tolist() for rule in self.momentum_rules],
            }
            if self.block_backend == "regulated-h-recursion":
                expected["c_regulator"] = self.c_regulator
            if metadata != expected:
                raise ValueError(f"cached block bank metadata mismatch in {path}")
            banks: dict[int, _ThreePointBlockBank] = {}
            edges = (
                tuple(int(edge) for edge in payload["edge_indices"])
                if "edge_indices" in payload
                else (0, 1, 2)
            )
            for edge in edges:
                banks[edge] = _ThreePointBlockBank(
                    orders=tuple(int(value) for value in payload[f"edge{edge}_orders"]),
                    internal_weights=tuple(
                        np.asarray(payload[f"edge{edge}_internal{internal_edge}"], dtype=float)
                        for internal_edge in range(3)
                    ),
                    weighted_structure_constants=np.asarray(
                        payload[f"edge{edge}_weighted"],
                        dtype=np.complex128,
                    ),
                    coefficients=np.asarray(
                        payload[f"edge{edge}_coefficients"],
                        dtype=np.complex128,
                    ),
                )
        self._banks = banks

    def adaptive_order(self, q_abs: float) -> int:
        """Smallest retained order with ``|hat_q|**(N+1) <= tolerance``."""
        q_abs = float(q_abs)
        if q_abs <= 0.0:
            return self.low_order
        if q_abs >= 1.0:
            return self.high_order
        estimate = int(math.ceil(math.log(self.adaptive_tolerance) / math.log(q_abs) - 1.0))
        return min(self.high_order, max(self.low_order, estimate))

    def correlator_from_logs(
        self,
        log_q_values: tuple[complex, complex, complex],
        *,
        order_cap: int | None = None,
        record_diagnostics: bool = True,
        cancel_eta_vacuum: bool = False,
        pade_orders: tuple[int, int] | None = None,
    ) -> complex:
        """Evaluate the nonchiral Liouville correlator for three cylinder logs.

        When ``cancel_eta_vacuum`` is true, use
        ``h-c/24=P^2-1/24 -> P^2`` in the primary propagation exponent.  The
        caller then supplies only the eta oscillator product.  Since the three
        necklace logs sum to ``2*pi*i*tau``, this exactly cancels the common
        Liouville ``exp(+pi*tau2/6)`` against the eta
        ``exp(-pi*tau2/6)`` before either exponential is formed.
        """
        if len(log_q_values) != 3:
            raise ValueError("three necklace cylinder logarithms are required")
        hat_q_values = tuple(elliptic_nome(cmath.exp(log_q)) for log_q in log_q_values)
        magnitudes = np.asarray([abs(value) for value in hat_q_values], dtype=float)
        descending = np.sort(magnitudes)[::-1]
        high_edge = int(np.argmax(magnitudes))
        selected_order = self.adaptive_order(float(magnitudes[high_edge]))
        if pade_orders is not None:
            selected_order = int(pade_orders[0]) + int(pade_orders[1])
            if selected_order > self.high_order:
                raise ValueError("Padé order exceeds the prepared block order")
        if order_cap is not None:
            selected_order = min(selected_order, max(self.low_order, int(order_cap)))
        if record_diagnostics:
            self.order_histogram[selected_order] += 1
            self.high_edge_histogram[high_edge] += 1
            self.maximum_hat_q_seen = max(self.maximum_hat_q_seen, float(descending[0]))
            self.maximum_second_hat_q_seen = max(
                self.maximum_second_hat_q_seen,
                float(descending[1]),
            )

        if high_edge not in self._banks:
            self._banks[high_edge] = self._build_bank(high_edge)
        bank = self._banks[high_edge]
        slices = [slice(None)] * 4
        slices[1 + high_edge] = slice(0, selected_order + 1)
        coefficients = bank.coefficients[tuple(slices)]
        powers = []
        for edge, hat_q in enumerate(hat_q_values):
            order = selected_order if edge == high_edge else self.low_order
            powers.append(hat_q ** np.arange(order + 1))
        if pade_orders is None:
            descendants = np.einsum(
                "tijk,i,j,k->t",
                coefficients,
                powers[0],
                powers[1],
                powers[2],
                optimize=True,
            )
        else:
            moved = np.moveaxis(coefficients, 1 + high_edge, 1)
            other_edges = [edge for edge in range(3) if edge != high_edge]
            series_rows = np.einsum(
                "tnij,i,j->tn",
                moved,
                powers[other_edges[0]],
                powers[other_edges[1]],
                optimize=True,
            )
            descendants = evaluate_pade_rows(
                series_rows,
                hat_q_values[high_edge],
                int(pade_orders[0]),
                int(pade_orders[1]),
            )
        primary_exponent = np.zeros(len(descendants), dtype=float)
        reference_weight = 1.0 if cancel_eta_vacuum else C_LIOUVILLE / 24.0
        for internal_weights, log_q in zip(bank.internal_weights, log_q_values):
            primary_exponent += 2.0 * (
                internal_weights - reference_weight
            ) * float(complex(log_q).real)
        primary_norm_squared = np.exp(primary_exponent)
        return complex(
            np.dot(
                bank.weighted_structure_constants,
                primary_norm_squared * np.abs(descendants) ** 2,
            )
        )

    def diagnostics(self) -> dict[str, object]:
        return {
            "adaptive_order_histogram": {
                str(order): int(count) for order, count in self.order_histogram.items()
            },
            "largest_hat_q_seen": float(self.maximum_hat_q_seen),
            "largest_second_hat_q_seen": float(self.maximum_second_hat_q_seen),
            "high_edge_histogram": {
                str(edge): int(count) for edge, count in self.high_edge_histogram.items()
            },
            "upsilon_cache_size": int(len(self.special._log_cache)),
        }


def ordered_necklace_data(
    w1: complex,
    w2: complex,
    tau: complex,
) -> tuple[tuple[complex, complex], tuple[complex, complex, complex]]:
    """Sort the two outgoing punctures and return the three positive-height gaps."""
    tau = complex(tau)
    if tau.imag <= 0.0:
        raise ValueError("tau must lie in the upper half-plane")
    ordered = tuple(sorted((complex(w1), complex(w2)), key=lambda value: value.imag))
    first, second = ordered
    logs = (
        1.0j * first,
        1.0j * (second - first),
        1.0j * (2.0 * math.pi * tau - second),
    )
    if any(log_q.real >= 1.0e-13 for log_q in logs):
        raise ValueError("puncture sorting did not produce contracting necklace cylinders")
    return ordered, logs


def dedekind_eta_oscillator_abs_squared(
    tau: complex,
    tolerance: float = 1.0e-16,
) -> float:
    r"""Return ``|prod_(n>=1) (1-q^n)|^2`` for ``q=exp(2*pi*i*tau)``.

    This is the Dedekind eta norm with its ``|q|^(1/12)`` vacuum factor
    removed.  It is paired with ``cancel_eta_vacuum=True`` above.
    """
    tau = complex(tau)
    q = cmath.exp(2.0j * math.pi * tau)
    value = 1.0 + 0.0j
    q_power = q
    n = 1
    while abs(q_power) > tolerance and n < 10000:
        value *= 1.0 - q_power
        n += 1
        q_power *= q
    return float(abs(value) ** 2)


def dedekind_eta_log_abs_squared(
    tau: complex,
    tolerance: float = 1.0e-16,
) -> float:
    r"""Return ``log(|eta(tau)|^2)`` without forming an exponentially small eta."""
    tau = complex(tau)
    q = cmath.exp(2.0j * math.pi * tau)
    log_norm = -math.pi * tau.imag / 6.0
    q_power = q
    n = 1
    while abs(q_power) > tolerance and n < 10000:
        log_norm += 2.0 * math.log(abs(1.0 - q_power))
        n += 1
        q_power *= q
    return float(log_norm)


def _log_abs_sine(value: complex) -> float:
    """Stable ``log(abs(sin(value)))`` for large imaginary part."""
    value = complex(value)
    magnitude = abs(value.imag)
    if magnitude < 20.0:
        return math.log(abs(cmath.sin(value)))
    correction = 1.0 + math.exp(-4.0 * magnitude)
    correction -= 2.0 * math.cos(2.0 * value.real) * math.exp(-2.0 * magnitude)
    return magnitude - math.log(2.0) + 0.5 * math.log(correction)


def torus_prime_form_log_norm(
    z: complex,
    tau: complex,
    tolerance: float = 1.0e-16,
) -> float:
    """Return the logarithm of the single-valued torus prime-form norm.

    The ordinary evaluator is used throughout the compact domain.  Above
    ``tau2=40`` we first reduce the point to the centered torus cell and use
    the leading theta term.  The omitted relative correction is bounded by
    ``O(exp(-pi*tau2))`` there, while evaluating the sine in logarithmic form
    avoids overflow.
    """
    z = complex(z)
    tau = complex(tau)
    if tau.imag <= 0.0:
        raise ValueError("tau must lie in the upper half-plane")
    if tau.imag <= 40.0:
        norm = torus_prime_form_norm(z, tau, tolerance=tolerance)
        if norm <= 0.0 or not math.isfinite(norm):
            raise FloatingPointError("prime-form norm is not positive and finite")
        return math.log(norm)

    vertical_coordinate = z.imag / (2.0 * math.pi * tau.imag)
    vertical_shift = math.floor(vertical_coordinate + 0.5)
    centered = z - 2.0 * math.pi * vertical_shift * tau
    horizontal_coordinate = centered.real / (2.0 * math.pi)
    horizontal_shift = math.floor(horizontal_coordinate + 0.5)
    centered -= 2.0 * math.pi * horizontal_shift
    w = centered / (2.0 * math.pi)
    log_holomorphic_norm = math.log(2.0) + _log_abs_sine(math.pi * w)
    log_gaussian = -(centered.imag**2) / (4.0 * math.pi * tau.imag)
    return float(log_holomorphic_norm + log_gaussian)


def reduced_worldsheet_integrand_three_point(
    correlator: LiouvilleTorusThreePointNecklace,
    w1: complex,
    w2: complex,
    tau: complex,
    *,
    order_cap: int | None = None,
    record_diagnostics: bool = True,
    pade_orders: tuple[int, int] | None = None,
) -> complex:
    """Return ``|eta|^2 K_X0 G_L/sqrt(tau2)`` in native normalization."""
    tau = complex(tau)
    ordered, logs = ordered_necklace_data(w1, w2, tau)
    first, second = ordered
    log_prime_first = torus_prime_form_log_norm(first, tau)
    log_prime_second = torus_prime_form_log_norm(second, tau)
    log_prime_pair = torus_prime_form_log_norm(first - second, tau)

    omega_in = correlator.omega_in
    omega_out = correlator.omega_out
    koba_nielsen = cmath.exp(
        omega_in * omega_out * log_prime_first
        + omega_in * omega_out * log_prime_second
        - omega_out * omega_out * log_prime_pair
    )
    liouville_without_vacuum = correlator.correlator_from_logs(
        logs,
        order_cap=order_cap,
        record_diagnostics=record_diagnostics,
        cancel_eta_vacuum=True,
        pade_orders=pade_orders,
    )
    return (
        dedekind_eta_oscillator_abs_squared(tau)
        * koba_nielsen
        * liouville_without_vacuum
        / math.sqrt(tau.imag)
    )
