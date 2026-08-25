#!/usr/bin/env python3
"""Channel atlas for the c=1 genus-one three-point worldsheet integrand.

The compact part of moduli space is divided by the nearest puncture-pair
distance.  Away from pair collisions the existing necklace h-recursion (or
its exact c=25 descendant baseline) is used.  Inside a collision patch the
pair-OPE block is evaluated by fixed-weight c-recursion.  Both answers are
returned in the same flat-torus conformal frame before the common timelike,
eta, and moduli-measure factors are applied.
"""

from __future__ import annotations

import cmath
import json
import math
import os
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import numpy as np

try:
    from genus1_three_point_worldsheet import (
        LiouvilleTorusThreePointNecklace,
        dedekind_eta_oscillator_abs_squared,
        ordered_necklace_data,
        torus_prime_form_log_norm,
    )
    from genus1_two_point_worldsheet import MomentumRule
    from liouville_torus import UpsilonB, yin_structure_constant_momentum
    from torus_three_point_ope_blocks import (
        comb_ope_c_recursion_coefficients,
        pair_ope_c_recursion_coefficients,
        pair_ope_coefficients_in_elliptic_loop_nomes,
        pair_ope_coefficients_in_local_delta,
    )
    from torus_two_point_blocks import elliptic_nome
except ImportError:  # pragma: no cover - package-style execution
    from plumbing.genus1_three_point_worldsheet import (
        LiouvilleTorusThreePointNecklace,
        dedekind_eta_oscillator_abs_squared,
        ordered_necklace_data,
        torus_prime_form_log_norm,
    )
    from plumbing.genus1_two_point_worldsheet import MomentumRule
    from plumbing.liouville_torus import UpsilonB, yin_structure_constant_momentum
    from plumbing.torus_three_point_ope_blocks import (
        comb_ope_c_recursion_coefficients,
        pair_ope_c_recursion_coefficients,
        pair_ope_coefficients_in_elliptic_loop_nomes,
        pair_ope_coefficients_in_local_delta,
    )
    from plumbing.torus_two_point_blocks import elliptic_nome


C_LIOUVILLE = 25.0


@dataclass(frozen=True)
class ChannelChoice:
    channel: str
    pair: tuple[int, int] | None
    local_displacement: complex | None
    nearest_pair_distance: float


@dataclass(frozen=True)
class _PairOPEBank:
    external_indices: tuple[int, int, int]
    high_loop_edge: int
    orders: tuple[int, int, int]
    internal_weights: tuple[np.ndarray, np.ndarray, np.ndarray]
    weighted_structure_constants: np.ndarray
    coefficients: np.ndarray


@dataclass(frozen=True)
class _CombOPEBank:
    external_indices: tuple[int, int, int]
    orders: tuple[int, int, int]
    internal_weights: tuple[np.ndarray, np.ndarray, np.ndarray]
    weighted_structure_constants: np.ndarray
    coefficients: np.ndarray


def _torus_fractional_coordinates(z: complex, tau: complex) -> tuple[float, float]:
    tau = complex(tau)
    z = complex(z)
    if tau.imag <= 0.0:
        raise ValueError("tau must lie in the upper half-plane")
    vertical = z.imag / (2.0 * math.pi * tau.imag)
    horizontal = (z.real - 2.0 * math.pi * tau.real * vertical) / (2.0 * math.pi)
    return float(horizontal), float(vertical)


def nearest_torus_displacement(z: complex, tau: complex) -> complex:
    """Return the shortest representative of ``z`` modulo the torus lattice."""
    horizontal, vertical = _torus_fractional_coordinates(z, tau)
    center_horizontal = int(round(horizontal))
    center_vertical = int(round(vertical))
    candidates = (
        complex(z) - 2.0 * math.pi * (shift_x + shift_y * complex(tau))
        for shift_x in range(center_horizontal - 1, center_horizontal + 2)
        for shift_y in range(center_vertical - 1, center_vertical + 2)
    )
    return min(candidates, key=abs)


def canonical_loop_displacement(z: complex, tau: complex) -> complex:
    """Choose ``z=2*pi*(x+y*tau)`` with ``0<=x,y<1``."""
    horizontal, vertical = _torus_fractional_coordinates(z, tau)
    horizontal -= math.floor(horizontal)
    vertical -= math.floor(vertical)
    return complex(2.0 * math.pi * (horizontal + vertical * complex(tau)))


def pair_disc_to_flat_log_factor(
    delta: complex,
    d_a: complex,
    d_b: complex,
) -> complex:
    r"""Return the logarithm of the chiral pair disc-to-flat factor.

    With ``v=exp(-i*delta)-1``, the cylinder-to-plane derivatives give

    ``exp(i*(d_b-d_a)*delta/2) [2 sin(delta/2)]^(-d_a-d_b)``.
    """
    delta = complex(delta)
    if delta == 0.0j:
        raise ValueError("the pair frame factor is singular at exact coincidence")
    return complex(
        0.5j * (complex(d_b) - complex(d_a)) * delta
        - (complex(d_a) + complex(d_b))
        * cmath.log(2.0 * cmath.sin(delta / 2.0))
    )


class LiouvilleTorusThreePointAtlas:
    """Necklace/OPE channel atlas in a common flat-cylinder frame."""

    def __init__(
        self,
        necklace: LiouvilleTorusThreePointNecklace,
        *,
        patch_epsilon: float = 0.15,
        ope_order: int = 6,
        high_loop_order: int = 6,
        low_loop_order: int = 2,
        triple_patch_epsilon: float = 0.10,
        total_ope_order: int | None = None,
        comb_loop_order: int = 3,
        special_dps: int = 28,
        bank_cache_path: str | Path | None = None,
        evaluation_order_cap: int | None = None,
        necklace_qhat_threshold: float | None = None,
        necklace_second_qhat_threshold: float | None = None,
    ) -> None:
        if not 0.0 < float(patch_epsilon) < 0.5:
            raise ValueError("patch_epsilon must lie in (0,1/2)")
        if min(int(ope_order), int(high_loop_order), int(low_loop_order)) < 0:
            raise ValueError("OPE block orders must be non-negative")
        if int(high_loop_order) < int(low_loop_order):
            raise ValueError("high_loop_order must be at least low_loop_order")
        if not 0.0 < float(triple_patch_epsilon) <= float(patch_epsilon):
            raise ValueError("triple_patch_epsilon must lie in (0,patch_epsilon]")
        self.necklace = necklace
        self.patch_epsilon = float(patch_epsilon)
        self.ope_order = int(ope_order)
        self.high_loop_order = int(high_loop_order)
        self.low_loop_order = int(low_loop_order)
        self.triple_patch_epsilon = float(triple_patch_epsilon)
        self.total_ope_order = (
            int(ope_order) if total_ope_order is None else int(total_ope_order)
        )
        self.comb_loop_order = int(comb_loop_order)
        self.evaluation_order_cap = (
            None if evaluation_order_cap is None else int(evaluation_order_cap)
        )
        self.necklace_qhat_threshold = (
            None if necklace_qhat_threshold is None else float(necklace_qhat_threshold)
        )
        self.necklace_second_qhat_threshold = (
            None
            if necklace_second_qhat_threshold is None
            else float(necklace_second_qhat_threshold)
        )
        if self.necklace_qhat_threshold is not None and not (
            0.0 < self.necklace_qhat_threshold < 1.0
        ):
            raise ValueError("necklace_qhat_threshold must lie in (0,1)")
        if self.necklace_second_qhat_threshold is not None and not (
            0.0 < self.necklace_second_qhat_threshold < 1.0
        ):
            raise ValueError("necklace_second_qhat_threshold must lie in (0,1)")
        if (
            self.necklace_second_qhat_threshold is not None
            and self.necklace_qhat_threshold is None
        ):
            raise ValueError(
                "necklace_second_qhat_threshold requires necklace_qhat_threshold"
            )
        if self.evaluation_order_cap is not None and self.evaluation_order_cap < 0:
            raise ValueError("evaluation_order_cap must be non-negative")
        if min(self.total_ope_order, self.comb_loop_order) < 0:
            raise ValueError("comb-OPE orders must be non-negative")
        self.special = UpsilonB(1.0, dps=int(special_dps))
        self._banks: dict[tuple[tuple[int, int, int], int], _PairOPEBank] = {}
        self._comb_banks: dict[tuple[int, int, int], _CombOPEBank] = {}
        self.channel_counts = {"necklace": 0, "pair_ope": 0, "comb_ope": 0}
        self.pair_counts = {"0-1": 0, "0-2": 0, "1-2": 0}
        self.maximum_ope_v = 0.0
        self.maximum_ope_delta = 0.0
        self.maximum_ope_hat_q = 0.0
        self.c_recursion_collision_count = 0
        self.maximum_c_recursion_regulator_error = 0.0
        self.bank_cache_path = (
            None if bank_cache_path is None else Path(bank_cache_path)
        )
        if self.bank_cache_path is not None and self.bank_cache_path.is_file():
            self.load_banks(self.bank_cache_path)
            print(f"loaded channel-atlas banks from {self.bank_cache_path}", flush=True)

    @property
    def external_weights(self) -> tuple[float, float, float]:
        return self.necklace.external_weights

    @property
    def external_momenta(self) -> tuple[complex, complex, complex]:
        return self.necklace.external_momenta

    def choose_channel(
        self,
        w1: complex,
        w2: complex,
        tau: complex,
        *,
        patch_epsilon: float | None = None,
    ) -> ChannelChoice:
        points = (complex(w1), complex(w2), 0.0j)
        necklace_magnitudes: tuple[float, float, float] | None = None
        if self.necklace_qhat_threshold is not None:
            _, logs = ordered_necklace_data(w1, w2, tau)
            necklace_magnitudes = tuple(
                sorted(
                    (abs(elliptic_nome(cmath.exp(log_q))) for log_q in logs),
                    reverse=True,
                )
            )
            largest_is_safe = (
                necklace_magnitudes[0] <= self.necklace_qhat_threshold
            )
            second_is_safe = (
                self.necklace_second_qhat_threshold is None
                or necklace_magnitudes[1] <= self.necklace_second_qhat_threshold
            )
            if largest_is_safe and second_is_safe:
                pair_distances = (
                    abs(nearest_torus_displacement(points[first] - points[second], tau))
                    for first, second in ((0, 1), (0, 2), (1, 2))
                )
                return ChannelChoice(
                    "necklace",
                    None,
                    None,
                    float(min(pair_distances)),
                )
        pair_rows = []
        for first, second in ((0, 1), (0, 2), (1, 2)):
            delta = nearest_torus_displacement(points[first] - points[second], tau)
            pair_rows.append((abs(delta), first, second, delta))
        distance, first, second, delta = min(pair_rows, key=lambda row: row[0])
        epsilon = self.patch_epsilon if patch_epsilon is None else float(patch_epsilon)
        use_ope = distance < 2.0 * math.pi * epsilon
        if necklace_magnitudes is not None:
            use_ope = True
        if use_ope:
            spectator = next(index for index in range(3) if index not in (first, second))
            second_delta = nearest_torus_displacement(
                points[second] - points[spectator],
                tau,
            )
            use_comb = (
                max(distance, abs(second_delta))
                < 2.0 * math.pi * self.triple_patch_epsilon
            )
            if (
                necklace_magnitudes is not None
                and self.necklace_second_qhat_threshold is not None
                and necklace_magnitudes[1] > self.necklace_second_qhat_threshold
            ):
                use_comb = True
            if use_comb:
                return ChannelChoice("comb_ope", (first, second), delta, float(distance))
            return ChannelChoice("pair_ope", (first, second), delta, float(distance))
        return ChannelChoice("necklace", None, None, float(distance))

    def _bank_key(self, pair: tuple[int, int], high_loop_edge: int) -> tuple[tuple[int, int, int], int]:
        spectator = next(index for index in range(3) if index not in pair)
        signature = (pair[0], pair[1], spectator)
        # The two outgoing operators are identical, so the two out-in charts
        # share one numerical bank.
        external_signature = tuple(self.external_weights[index] for index in signature)
        for key in self._banks:
            stored_signature, stored_high = key
            if stored_high == high_loop_edge and tuple(
                self.external_weights[index] for index in stored_signature
            ) == external_signature:
                return key
        return signature, int(high_loop_edge)

    def _build_bank(
        self,
        external_indices: tuple[int, int, int],
        high_loop_edge: int,
    ) -> _PairOPEBank:
        if high_loop_edge not in (1, 2):
            raise ValueError("the high pair-OPE loop edge must be 1 or 2")
        orders = [self.ope_order, self.low_loop_order, self.low_loop_order]
        orders[high_loop_edge] = self.high_loop_order
        orders_tuple = tuple(orders)
        rules = self.necklace.momentum_rules
        external_weights = tuple(
            self.external_weights[index] for index in external_indices
        )
        external_momenta = tuple(
            self.external_momenta[index] for index in external_indices
        )
        h_arrays = ([], [], [])
        weighted_structures: list[complex] = []
        coefficient_tensors: list[np.ndarray] = []
        node_product = tuple(product(*(range(len(rule.nodes)) for rule in rules)))
        for flat_index, node_indices in enumerate(node_product, start=1):
            momenta = tuple(
                float(rules[edge].nodes[node_indices[edge]]) for edge in range(3)
            )
            internal_weights = tuple(1.0 + momentum * momentum for momentum in momenta)
            quadrature_weight = math.prod(
                float(rules[edge].weights[node_indices[edge]]) for edge in range(3)
            ) / math.pi**3
            structure = yin_structure_constant_momentum(
                self.special,
                external_momenta[0],
                external_momenta[1],
                momenta[0],
            )
            structure *= yin_structure_constant_momentum(
                self.special,
                momenta[1],
                momenta[0],
                momenta[2],
            )
            structure *= yin_structure_constant_momentum(
                self.special,
                momenta[2],
                external_momenta[2],
                momenta[1],
            )
            coefficients = self._collision_safe_c_coefficients(
                pair_ope_c_recursion_coefficients,
                external_weights=external_weights,
                internal_weights=internal_weights,
                orders=orders_tuple,
            )
            coefficients = pair_ope_coefficients_in_elliptic_loop_nomes(
                coefficients,
                first_loop_order=orders_tuple[1],
                second_loop_order=orders_tuple[2],
            )
            coefficients = pair_ope_coefficients_in_local_delta(
                coefficients,
                delta_order=orders_tuple[0],
            )
            for edge in range(3):
                h_arrays[edge].append(internal_weights[edge])
            weighted_structures.append(quadrature_weight * structure)
            coefficient_tensors.append(coefficients)
            if flat_index % 128 == 0:
                print(
                    f"pair-OPE bank progress {flat_index}/{len(node_product)} "
                    f"for {external_indices} orders {orders_tuple}",
                    flush=True,
                )
        bank = _PairOPEBank(
            external_indices=external_indices,
            high_loop_edge=high_loop_edge,
            orders=orders_tuple,
            internal_weights=tuple(
                np.asarray(values, dtype=float) for values in h_arrays
            ),  # type: ignore[arg-type]
            weighted_structure_constants=np.asarray(weighted_structures, dtype=np.complex128),
            coefficients=np.stack(coefficient_tensors),
        )
        print(
            "prepared torus three-point pair-OPE c-recursion bank "
            f"for external indices {external_indices} and orders {orders_tuple}",
            flush=True,
        )
        return bank

    def _collision_safe_c_coefficients(
        self,
        builder: object,
        *,
        external_weights: tuple[complex, complex, complex],
        internal_weights: tuple[complex, complex, complex],
        orders: tuple[int, int, int],
    ) -> np.ndarray:
        """Resolve a confluent intermediate c-pole by a generic-weight limit."""
        try:
            return builder(  # type: ignore[operator]
                C_LIOUVILLE,
                external_weights=external_weights,
                internal_weights=internal_weights,
                orders=orders,
            )
        except ZeroDivisionError:
            scales = np.asarray([2.0e-4, 1.0e-4, 5.0e-5, 2.5e-5], dtype=float)
            directions = np.asarray([0.173, 0.311, 0.487], dtype=float)
            samples = []
            for scale in scales:
                shifted = tuple(
                    complex(weight) + float(scale * direction)
                    for weight, direction in zip(internal_weights, directions)
                )
                samples.append(
                    builder(  # type: ignore[operator]
                        C_LIOUVILLE,
                        external_weights=external_weights,
                        internal_weights=shifted,
                        orders=orders,
                        pole_tolerance=1.0e-14,
                    )
                )

            def extrapolate(count: int) -> np.ndarray:
                selected = scales[:count]
                moment_matrix = np.vstack(
                    [selected**degree for degree in range(count)]
                )
                target = np.zeros(count, dtype=float)
                target[0] = 1.0
                weights = np.linalg.solve(moment_matrix, target)
                return np.tensordot(
                    weights,
                    np.stack(samples[:count]),
                    axes=(0, 0),
                )

            quadratic = extrapolate(3)
            cubic = extrapolate(4)
            relative_error = float(
                np.max(np.abs(cubic - quadratic))
                / max(float(np.max(np.abs(cubic))), 1.0e-300)
            )
            if not np.all(np.isfinite(cubic)):
                raise FloatingPointError("non-finite collision-regulated c-recursion")
            self.c_recursion_collision_count += 1
            self.maximum_c_recursion_regulator_error = max(
                self.maximum_c_recursion_regulator_error,
                relative_error,
            )
            return np.asarray(cubic, dtype=np.complex128)

    def _cache_metadata(self) -> dict[str, object]:
        return {
            "format": "torus-three-point-channel-atlas-banks-v1",
            "t": self.necklace.t,
            "external_weights": list(self.external_weights),
            "ope_order": self.ope_order,
            "high_loop_order": self.high_loop_order,
            "low_loop_order": self.low_loop_order,
            "total_ope_order": self.total_ope_order,
            "comb_loop_order": self.comb_loop_order,
            "momentum_nodes": [
                rule.nodes.tolist() for rule in self.necklace.momentum_rules
            ],
            "momentum_weights": [
                rule.weights.tolist() for rule in self.necklace.momentum_rules
            ],
        }

    def save_banks(self, path: str | Path | None = None) -> None:
        """Persist every channel bank currently built by the lazy atlas."""
        target = self.bank_cache_path if path is None else Path(path)
        if target is None:
            raise ValueError("no channel-atlas bank cache path was supplied")
        target.parent.mkdir(parents=True, exist_ok=True)
        arrays: dict[str, np.ndarray] = {
            "metadata": np.asarray(json.dumps(self._cache_metadata())),
            "pair_count": np.asarray(len(self._banks), dtype=int),
            "comb_count": np.asarray(len(self._comb_banks), dtype=int),
        }
        for index, ((external_indices, high_loop_edge), bank) in enumerate(
            sorted(self._banks.items())
        ):
            prefix = f"pair{index}"
            arrays[f"{prefix}_external"] = np.asarray(external_indices, dtype=int)
            arrays[f"{prefix}_high"] = np.asarray(high_loop_edge, dtype=int)
            arrays[f"{prefix}_orders"] = np.asarray(bank.orders, dtype=int)
            arrays[f"{prefix}_weighted"] = bank.weighted_structure_constants
            arrays[f"{prefix}_coefficients"] = bank.coefficients
            for edge, values in enumerate(bank.internal_weights):
                arrays[f"{prefix}_internal{edge}"] = values
        for index, (external_indices, bank) in enumerate(sorted(self._comb_banks.items())):
            prefix = f"comb{index}"
            arrays[f"{prefix}_external"] = np.asarray(external_indices, dtype=int)
            arrays[f"{prefix}_orders"] = np.asarray(bank.orders, dtype=int)
            arrays[f"{prefix}_weighted"] = bank.weighted_structure_constants
            arrays[f"{prefix}_coefficients"] = bank.coefficients
            for edge, values in enumerate(bank.internal_weights):
                arrays[f"{prefix}_internal{edge}"] = values
        temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp.npz")
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, target)

    def load_banks(self, path: str | Path) -> None:
        """Load cached banks, failing closed if any design input changed."""
        path = Path(path)
        with np.load(path, allow_pickle=False) as payload:
            metadata = json.loads(str(payload["metadata"].item()))
            # v1 caches written before patch-radius audits included the two
            # evaluation boundaries even though no bank array depends on
            # them.  Drop those legacy fields when validating the numerical
            # block design.
            metadata.pop("patch_epsilon", None)
            metadata.pop("triple_patch_epsilon", None)
            if metadata != self._cache_metadata():
                raise ValueError(f"channel-atlas cache metadata mismatch in {path}")
            pair_banks: dict[tuple[tuple[int, int, int], int], _PairOPEBank] = {}
            for index in range(int(payload["pair_count"].item())):
                prefix = f"pair{index}"
                external = tuple(int(value) for value in payload[f"{prefix}_external"])
                high = int(payload[f"{prefix}_high"].item())
                pair_banks[(external, high)] = _PairOPEBank(
                    external_indices=external,  # type: ignore[arg-type]
                    high_loop_edge=high,
                    orders=tuple(
                        int(value) for value in payload[f"{prefix}_orders"]
                    ),  # type: ignore[arg-type]
                    internal_weights=tuple(
                        np.asarray(payload[f"{prefix}_internal{edge}"], dtype=float)
                        for edge in range(3)
                    ),  # type: ignore[arg-type]
                    weighted_structure_constants=np.asarray(
                        payload[f"{prefix}_weighted"],
                        dtype=np.complex128,
                    ),
                    coefficients=np.asarray(
                        payload[f"{prefix}_coefficients"],
                        dtype=np.complex128,
                    ),
                )
            comb_banks: dict[tuple[int, int, int], _CombOPEBank] = {}
            for index in range(int(payload["comb_count"].item())):
                prefix = f"comb{index}"
                external = tuple(int(value) for value in payload[f"{prefix}_external"])
                comb_banks[external] = _CombOPEBank(
                    external_indices=external,  # type: ignore[arg-type]
                    orders=tuple(
                        int(value) for value in payload[f"{prefix}_orders"]
                    ),  # type: ignore[arg-type]
                    internal_weights=tuple(
                        np.asarray(payload[f"{prefix}_internal{edge}"], dtype=float)
                        for edge in range(3)
                    ),  # type: ignore[arg-type]
                    weighted_structure_constants=np.asarray(
                        payload[f"{prefix}_weighted"],
                        dtype=np.complex128,
                    ),
                    coefficients=np.asarray(
                        payload[f"{prefix}_coefficients"],
                        dtype=np.complex128,
                    ),
                )
        self._banks = pair_banks
        self._comb_banks = comb_banks

    def _save_cache_if_requested(self) -> None:
        if self.bank_cache_path is not None:
            self.save_banks(self.bank_cache_path)

    def _comb_bank_key(self, external_indices: tuple[int, int, int]) -> tuple[int, int, int]:
        external_signature = tuple(
            self.external_weights[index] for index in external_indices
        )
        for stored_indices in self._comb_banks:
            if tuple(self.external_weights[index] for index in stored_indices) == external_signature:
                return stored_indices
        return external_indices

    def _build_comb_bank(
        self,
        external_indices: tuple[int, int, int],
    ) -> _CombOPEBank:
        orders = (self.ope_order, self.total_ope_order, self.comb_loop_order)
        rules = self.necklace.momentum_rules
        external_weights = tuple(
            self.external_weights[index] for index in external_indices
        )
        external_momenta = tuple(
            self.external_momenta[index] for index in external_indices
        )
        h_arrays = ([], [], [])
        weighted_structures: list[complex] = []
        coefficient_tensors: list[np.ndarray] = []
        node_product = tuple(product(*(range(len(rule.nodes)) for rule in rules)))
        for flat_index, node_indices in enumerate(node_product, start=1):
            momenta = tuple(
                float(rules[edge].nodes[node_indices[edge]]) for edge in range(3)
            )
            internal_weights = tuple(1.0 + momentum * momentum for momentum in momenta)
            quadrature_weight = math.prod(
                float(rules[edge].weights[node_indices[edge]]) for edge in range(3)
            ) / math.pi**3
            structure = yin_structure_constant_momentum(
                self.special,
                external_momenta[0],
                external_momenta[1],
                momenta[0],
            )
            structure *= yin_structure_constant_momentum(
                self.special,
                momenta[1],
                external_momenta[2],
                momenta[0],
            )
            structure *= yin_structure_constant_momentum(
                self.special,
                momenta[2],
                momenta[1],
                momenta[2],
            )
            coefficients = self._collision_safe_c_coefficients(
                comb_ope_c_recursion_coefficients,
                external_weights=external_weights,
                internal_weights=internal_weights,
                orders=orders,
            )
            coefficients = pair_ope_coefficients_in_local_delta(
                coefficients,
                delta_order=orders[0],
            )
            coefficients = np.moveaxis(
                pair_ope_coefficients_in_local_delta(
                    np.moveaxis(coefficients, 1, 0),
                    delta_order=orders[1],
                ),
                0,
                1,
            )
            for edge in range(3):
                h_arrays[edge].append(internal_weights[edge])
            weighted_structures.append(quadrature_weight * structure)
            coefficient_tensors.append(coefficients)
            if flat_index % 64 == 0:
                print(
                    f"comb-OPE bank progress {flat_index}/{len(node_product)} "
                    f"for {external_indices} orders {orders}",
                    flush=True,
                )
        bank = _CombOPEBank(
            external_indices=external_indices,
            orders=orders,
            internal_weights=tuple(
                np.asarray(values, dtype=float) for values in h_arrays
            ),  # type: ignore[arg-type]
            weighted_structure_constants=np.asarray(weighted_structures, dtype=np.complex128),
            coefficients=np.stack(coefficient_tensors),
        )
        print(
            "prepared torus three-point comb-OPE c-recursion bank "
            f"for external indices {external_indices} and orders {orders}",
            flush=True,
        )
        return bank

    def correlator_ope(
        self,
        w1: complex,
        w2: complex,
        tau: complex,
        pair: tuple[int, int],
        *,
        record_diagnostics: bool = True,
    ) -> complex:
        points = (complex(w1), complex(w2), 0.0j)
        first, second = pair
        spectator = next(index for index in range(3) if index not in pair)
        delta = nearest_torus_displacement(points[first] - points[second], tau)
        fused_to_spectator = canonical_loop_displacement(
            points[second] - points[spectator],
            tau,
        )
        log_q_first = 1.0j * fused_to_spectator
        log_q_second = 1.0j * (2.0 * math.pi * complex(tau) - fused_to_spectator)
        q_first = cmath.exp(log_q_first)
        q_second = cmath.exp(log_q_second)
        hat_q = (elliptic_nome(q_first), elliptic_nome(q_second))
        high_loop_edge = 1 + int(abs(hat_q[1]) > abs(hat_q[0]))
        external_indices = (first, second, spectator)
        key = self._bank_key(pair, high_loop_edge)
        if key not in self._banks:
            self._banks[key] = self._build_bank(key[0], high_loop_edge)
            self._save_cache_if_requested()
        bank = self._banks[key]
        v = cmath.exp(-1.0j * delta) - 1.0
        selected_orders = list(bank.orders)
        if self.evaluation_order_cap is not None:
            selected_orders[0] = min(selected_orders[0], self.evaluation_order_cap)
        powers = (
            delta ** np.arange(selected_orders[0] + 1),
            hat_q[0] ** np.arange(selected_orders[1] + 1),
            hat_q[1] ** np.arange(selected_orders[2] + 1),
        )
        descendants = np.einsum(
            "tijk,i,j,k->t",
            bank.coefficients[
                :,
                : selected_orders[0] + 1,
                : selected_orders[1] + 1,
                : selected_orders[2] + 1,
            ],
            powers[0],
            powers[1],
            powers[2],
            optimize=True,
        )
        h_ope, h_first, h_second = bank.internal_weights
        primary_exponent = (
            2.0 * h_ope * math.log(abs(v))
            + 2.0 * (h_first - 1.0) * log_q_first.real
            + 2.0 * (h_second - 1.0) * log_q_second.real
        )
        d_a = self.external_weights[first]
        d_b = self.external_weights[second]
        flat_log = pair_disc_to_flat_log_factor(delta, d_a, d_b)
        block_norm_squared = np.exp(primary_exponent + 2.0 * flat_log.real)
        if record_diagnostics:
            self.channel_counts["pair_ope"] += 1
            self.pair_counts[f"{min(pair)}-{max(pair)}"] += 1
            self.maximum_ope_v = max(self.maximum_ope_v, abs(v))
            self.maximum_ope_delta = max(self.maximum_ope_delta, abs(delta))
            self.maximum_ope_hat_q = max(
                self.maximum_ope_hat_q,
                abs(hat_q[0]),
                abs(hat_q[1]),
            )
        return complex(
            np.dot(
                bank.weighted_structure_constants,
                block_norm_squared * np.abs(descendants) ** 2,
            )
        )

    def correlator_comb_ope(
        self,
        w1: complex,
        w2: complex,
        tau: complex,
        pair: tuple[int, int],
        *,
        record_diagnostics: bool = True,
    ) -> complex:
        points = (complex(w1), complex(w2), 0.0j)
        first, second = pair
        spectator = next(index for index in range(3) if index not in pair)
        delta_pair = nearest_torus_displacement(points[first] - points[second], tau)
        # The comb tensor convention has the pair primary at local zero and
        # the spectator at the second OPE coordinate.
        delta_total = nearest_torus_displacement(
            points[spectator] - points[second],
            tau,
        )
        v_pair = cmath.exp(-1.0j * delta_pair) - 1.0
        v_total = cmath.exp(-1.0j * delta_total) - 1.0
        log_q = 2.0j * math.pi * complex(tau)
        q = cmath.exp(log_q)
        external_indices = (first, second, spectator)
        key = self._comb_bank_key(external_indices)
        if key not in self._comb_banks:
            self._comb_banks[key] = self._build_comb_bank(key)
            self._save_cache_if_requested()
        bank = self._comb_banks[key]
        selected_orders = list(bank.orders)
        if self.evaluation_order_cap is not None:
            selected_orders[0] = min(selected_orders[0], self.evaluation_order_cap)
            selected_orders[1] = min(selected_orders[1], self.evaluation_order_cap)
        powers = (
            delta_pair ** np.arange(selected_orders[0] + 1),
            delta_total ** np.arange(selected_orders[1] + 1),
            q ** np.arange(selected_orders[2] + 1),
        )
        descendants = np.einsum(
            "tijk,i,j,k->t",
            bank.coefficients[
                :,
                : selected_orders[0] + 1,
                : selected_orders[1] + 1,
                : selected_orders[2] + 1,
            ],
            powers[0],
            powers[1],
            powers[2],
            optimize=True,
        )
        h_pair, h_total, h_loop = bank.internal_weights
        first_flat = pair_disc_to_flat_log_factor(
            delta_pair,
            self.external_weights[first],
            self.external_weights[second],
        )
        log_sine_total = cmath.log(2.0 * cmath.sin(delta_total / 2.0))
        d_spectator = self.external_weights[spectator]
        second_flat = (
            0.5j * (h_pair - d_spectator) * delta_total
            - (h_pair + d_spectator) * log_sine_total
        )
        exponent = (
            2.0 * h_pair * math.log(abs(v_pair))
            + 2.0 * h_total * math.log(abs(v_total))
            + 2.0 * (h_loop - 1.0) * log_q.real
            + 2.0 * first_flat.real
            + 2.0 * second_flat.real
        )
        if record_diagnostics:
            self.channel_counts["comb_ope"] += 1
            self.pair_counts[f"{min(pair)}-{max(pair)}"] += 1
            self.maximum_ope_v = max(self.maximum_ope_v, abs(v_pair), abs(v_total))
            self.maximum_ope_delta = max(
                self.maximum_ope_delta,
                abs(delta_pair),
                abs(delta_total),
            )
        return complex(
            np.dot(
                bank.weighted_structure_constants,
                np.exp(exponent) * np.abs(descendants) ** 2,
            )
        )

    def correlator_patched_without_eta_vacuum(
        self,
        w1: complex,
        w2: complex,
        tau: complex,
        *,
        patch_epsilon: float | None = None,
        record_diagnostics: bool = True,
        necklace_pade_orders: tuple[int, int] | None = None,
    ) -> tuple[complex, ChannelChoice]:
        choice = self.choose_channel(w1, w2, tau, patch_epsilon=patch_epsilon)
        if choice.channel == "pair_ope":
            assert choice.pair is not None
            return (
                self.correlator_ope(
                    w1,
                    w2,
                    tau,
                    choice.pair,
                    record_diagnostics=record_diagnostics,
                ),
                choice,
            )
        if choice.channel == "comb_ope":
            assert choice.pair is not None
            return (
                self.correlator_comb_ope(
                    w1,
                    w2,
                    tau,
                    choice.pair,
                    record_diagnostics=record_diagnostics,
                ),
                choice,
            )
        _, logs = ordered_necklace_data(w1, w2, tau)
        if record_diagnostics:
            self.channel_counts["necklace"] += 1
        return (
            self.necklace.correlator_from_logs(
                logs,
                record_diagnostics=record_diagnostics,
                cancel_eta_vacuum=True,
                pade_orders=necklace_pade_orders,
            ),
            choice,
        )

    def diagnostics(self) -> dict[str, object]:
        return {
            "patch_epsilon": self.patch_epsilon,
            "triple_patch_epsilon": self.triple_patch_epsilon,
            "necklace_qhat_threshold": self.necklace_qhat_threshold,
            "necklace_second_qhat_threshold": self.necklace_second_qhat_threshold,
            "channel_counts": dict(self.channel_counts),
            "pair_counts": dict(self.pair_counts),
            "maximum_ope_abs_v": float(self.maximum_ope_v),
            "maximum_ope_abs_delta": float(self.maximum_ope_delta),
            "maximum_ope_abs_hat_q": float(self.maximum_ope_hat_q),
            "c_recursion_collision_count": int(self.c_recursion_collision_count),
            "maximum_c_recursion_regulator_error": float(
                self.maximum_c_recursion_regulator_error
            ),
            "ope_orders": {
                "local": self.ope_order,
                "high_loop": self.high_loop_order,
                "low_loop": self.low_loop_order,
                "comb_total": self.total_ope_order,
                "comb_loop": self.comb_loop_order,
                "evaluation_local_order_cap": self.evaluation_order_cap,
            },
        }


def reduced_worldsheet_integrand_three_point_patched(
    atlas: LiouvilleTorusThreePointAtlas,
    w1: complex,
    w2: complex,
    tau: complex,
    *,
    patch_epsilon: float | None = None,
    record_diagnostics: bool = True,
    necklace_pade_orders: tuple[int, int] | None = None,
) -> tuple[complex, ChannelChoice]:
    """Return the patched reduced integrand and the selected channel."""
    tau = complex(tau)
    first, second = complex(w1), complex(w2)
    log_prime_first = torus_prime_form_log_norm(first, tau)
    log_prime_second = torus_prime_form_log_norm(second, tau)
    log_prime_pair = torus_prime_form_log_norm(first - second, tau)
    omega_in = atlas.necklace.omega_in
    omega_out = atlas.necklace.omega_out
    koba_nielsen = cmath.exp(
        omega_in * omega_out * log_prime_first
        + omega_in * omega_out * log_prime_second
        - omega_out * omega_out * log_prime_pair
    )
    liouville, choice = atlas.correlator_patched_without_eta_vacuum(
        w1,
        w2,
        tau,
        patch_epsilon=patch_epsilon,
        record_diagnostics=record_diagnostics,
        necklace_pade_orders=necklace_pade_orders,
    )
    return (
        dedekind_eta_oscillator_abs_squared(tau)
        * koba_nielsen
        * liouville
        / math.sqrt(tau.imag),
        choice,
    )


__all__ = [
    "ChannelChoice",
    "LiouvilleTorusThreePointAtlas",
    "canonical_loop_displacement",
    "nearest_torus_displacement",
    "pair_disc_to_flat_log_factor",
    "reduced_worldsheet_integrand_three_point_patched",
]
