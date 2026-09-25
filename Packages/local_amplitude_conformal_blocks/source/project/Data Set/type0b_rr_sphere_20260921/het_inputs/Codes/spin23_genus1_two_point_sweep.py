#!/usr/bin/env python3
r"""Two-stage fixed-geometry genus-one vector two-point energy sweep.

Stage 1 constructs an order-``n`` tensor generalized-Laguerre rule and, at
each of its ``n**2`` internal-momentum nodes, stores the exact finite-level
NS/R block response as a polynomial in the common external weight

.. math::

   h_\omega=\frac{1+\omega^2}{2}.

Stage 2 evaluates those saved polynomials, the exact super-Liouville
structure constants, and the quadrature sum for a requested energy grid.
The final reducer performs only deterministic shard summation and the
already-tested free-field, PCO, and GSO assembly.  No interpolation in
internal momentum or external energy is introduced.

The implementation is intentionally specialized to the equal-energy vector
two-point function.  General external kinematics require multivariate
external-weight polynomials and are not silently coerced into this format.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time
from typing import Mapping, Sequence

import numpy as np
from scipy.special import roots_genlaguerre

from ns_algebra.ns_sca import G, Word
from spin23_genus1_amplitude import (
    GenusOneFixedGeometryWorkspace,
    GenusOneNSState,
)
from spin23_genus1_spectral import SpectralIntegralDiagnostics
from spin23_ns_fast import (
    _fast_edge_factor as _ns_fast_edge_factor,
    _prewhitened_vertex_coefficients,
    direct_fast_b1_ns_two_point_polynomial_series,
)
from spin23_ramond_fast import (
    _fast_edge_factor as _ramond_fast_edge_factor,
    _prewhitened_vertex_sign_coefficients,
    direct_fast_b1_ramond_two_sign_polynomial_series,
)
from spin23_super_liouville_data import (
    ns_structure_constants,
    ns_weight,
    rr_ns_structure_constants,
)


SCHEMA_VERSION = 1
EMPTY_WORD: Word = ()
G_MINUS_HALF: Word = (G(Fraction(-1, 2)),)


@dataclass(frozen=True)
class ResponseFamily:
    """One spin/PCO spectral integral represented in every cache shard."""

    spin_label: str
    sector: str
    temporal_lift_sign: int
    component_index: int
    holomorphic_words: tuple[Word, Word]
    antiholomorphic_words: tuple[Word, Word]
    endpoint_powers: tuple[int, int]

    @property
    def name(self) -> str:
        """Return the stable artifact label for this response family."""

        return f"{self.spin_label}_component{self.component_index}"


_FAMILIES = (
    ResponseFamily(
        "NS",
        "NS",
        1,
        0,
        (G_MINUS_HALF, G_MINUS_HALF),
        (EMPTY_WORD, EMPTY_WORD),
        (0, 0),
    ),
    ResponseFamily(
        "NS",
        "NS",
        1,
        1,
        (EMPTY_WORD, EMPTY_WORD),
        (EMPTY_WORD, EMPTY_WORD),
        (0, 0),
    ),
    ResponseFamily(
        "NS_tilde",
        "NS",
        -1,
        0,
        (G_MINUS_HALF, G_MINUS_HALF),
        (EMPTY_WORD, EMPTY_WORD),
        (0, 0),
    ),
    ResponseFamily(
        "NS_tilde",
        "NS",
        -1,
        1,
        (EMPTY_WORD, EMPTY_WORD),
        (EMPTY_WORD, EMPTY_WORD),
        (0, 0),
    ),
    ResponseFamily(
        "R",
        "R",
        1,
        0,
        (G_MINUS_HALF, G_MINUS_HALF),
        (EMPTY_WORD, EMPTY_WORD),
        (0, 0),
    ),
    ResponseFamily(
        "R",
        "R",
        1,
        1,
        (EMPTY_WORD, EMPTY_WORD),
        (EMPTY_WORD, EMPTY_WORD),
        (1, 1),
    ),
)


def _encode_complex(value: complex) -> list[float]:
    number = complex(value)
    return [float(number.real), float(number.imag)]


def _decode_complex(value: Sequence[float]) -> complex:
    if len(value) != 2:
        raise ValueError("a complex value must contain real and imaginary parts")
    return complex(float(value[0]), float(value[1]))


def _word_label(word: Word) -> str:
    if word == EMPTY_WORD:
        return "P"
    if word == G_MINUS_HALF:
        return "G"
    raise ValueError("the two-point sweep supports only P and G_-1/2 words")


def _family_metadata(family: ResponseFamily) -> dict[str, object]:
    return {
        "name": family.name,
        "spin_label": family.spin_label,
        "sector": family.sector,
        "temporal_lift_sign": family.temporal_lift_sign,
        "component_index": family.component_index,
        "holomorphic_words": [
            _word_label(word) for word in family.holomorphic_words
        ],
        "antiholomorphic_words": [
            _word_label(word) for word in family.antiholomorphic_words
        ],
        "endpoint_powers": list(family.endpoint_powers),
    }


def _validate_partition(task_index: int, task_count: int) -> None:
    if isinstance(task_index, bool) or not isinstance(task_index, int):
        raise TypeError("task_index must be an integer")
    if isinstance(task_count, bool) or not isinstance(task_count, int):
        raise TypeError("task_count must be an integer")
    if task_count < 1 or not 0 <= task_index < task_count:
        raise ValueError("task partition must satisfy 0 <= index < count")


def _cutoff_pair(value: int | Sequence[int]) -> tuple[int, int]:
    if isinstance(value, bool):
        raise TypeError("maximum_twice_levels must be an integer or pair")
    if isinstance(value, int):
        result = (value, value)
    else:
        result = tuple(int(entry) for entry in value)
    if len(result) != 2 or any(entry < 0 for entry in result):
        raise ValueError("maximum_twice_levels must be a nonnegative pair")
    return result


def _laguerre_channels(
    *,
    quadrature_order: int,
    scales: tuple[float, float],
    endpoint_powers: tuple[int, int],
) -> tuple[tuple[tuple[float, float], ...], ...]:
    """Return momentum and transformed-weight pairs on each tensor axis."""

    if quadrature_order < 2:
        raise ValueError("quadrature_order must be at least two")
    channels = []
    for scale, power in zip(scales, endpoint_powers):
        nodes, weights = roots_genlaguerre(
            quadrature_order,
            (power - 1.0) / 2.0,
        )
        channels.append(
            tuple(
                (
                    math.sqrt(float(node) / scale),
                    float(weight)
                    / (2.0 * math.pi * scale ** ((power + 1.0) / 2.0)),
                )
                for node, weight in zip(nodes, weights)
            )
        )
    return tuple(channels)


def _laguerre_node(
    channels: tuple[tuple[tuple[float, float], ...], ...],
    flat_index: int,
) -> tuple[tuple[float, float], float]:
    """Return one row-major tensor node and endpoint-normalized weight."""

    order = len(channels[0])
    first_index, second_index = divmod(flat_index, order)
    selected = (channels[0][first_index], channels[1][second_index])
    momenta = (selected[0][0], selected[1][0])
    return momenta, selected[0][1] * selected[1][1]


def _pad_polynomials(
    values: Sequence[Sequence[Sequence[Sequence[np.ndarray]]]],
) -> np.ndarray:
    width = max(
        len(polynomial)
        for node in values
        for family in node
        for branch in family
        for polynomial in branch
    )
    output = np.zeros(
        (len(values), len(_FAMILIES), 2, 2, width),
        dtype=np.complex128,
    )
    for node_index, node in enumerate(values):
        for family_index, family in enumerate(node):
            for branch_index, branch in enumerate(family):
                for chirality_index, polynomial in enumerate(branch):
                    output[
                        node_index,
                        family_index,
                        branch_index,
                        chirality_index,
                        : len(polynomial),
                    ] = polynomial
    return output


def _build_family_response(
    family: ResponseFamily,
    internal_momenta: tuple[float, float],
    plumbing: tuple[complex, complex],
    *,
    maximum_twice_levels: tuple[int, int],
    condition_limit: float,
    evaluation_twice_levels: tuple[int, int] | None = None,
) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """Return two branch responses, optionally at a lower level rectangle."""

    lifts = (family.temporal_lift_sign, 1)
    conjugate_plumbing = tuple(value.conjugate() for value in plumbing)
    if family.sector == "NS":
        branches = []
        for parity in (0, 1):
            form_weight = (1.0, 0.0) if parity == 0 else (0.0, 1.0)
            common = {
                "internal_momenta": internal_momenta,
                "maximum_twice_levels": maximum_twice_levels,
                "edge_lift_signs": lifts,
                "form_weights": (form_weight, form_weight),
                "condition_limit": condition_limit,
            }
            holomorphic = direct_fast_b1_ns_two_point_polynomial_series(
                external_words=family.holomorphic_words,
                **common,
            )
            antiholomorphic = direct_fast_b1_ns_two_point_polynomial_series(
                external_words=family.antiholomorphic_words,
                **common,
            )
            branches.append(
                (
                    holomorphic.descendant_polynomial(
                        plumbing,
                        maximum_twice_levels=evaluation_twice_levels,
                    ),
                    antiholomorphic.descendant_polynomial(
                        conjugate_plumbing,
                        maximum_twice_levels=evaluation_twice_levels,
                    ),
                )
            )
        return tuple(branches)  # type: ignore[return-value]

    common = {
        "internal_momenta": internal_momenta,
        "temporal_lift_sign": family.temporal_lift_sign,
        "maximum_twice_levels": maximum_twice_levels,
        "edge_lift_signs": lifts,
        "condition_limit": condition_limit,
    }
    holomorphic = direct_fast_b1_ramond_two_sign_polynomial_series(
        external_words=family.holomorphic_words,
        **common,
    )
    antiholomorphic = direct_fast_b1_ramond_two_sign_polynomial_series(
        external_words=family.antiholomorphic_words,
        **common,
    )
    return tuple(
        (
            holomorphic[first_sign].descendant_polynomial(
                plumbing,
                maximum_twice_levels=evaluation_twice_levels,
            ),
            antiholomorphic[first_sign].descendant_polynomial(
                conjugate_plumbing,
                maximum_twice_levels=evaluation_twice_levels,
            ),
        )
        for first_sign in (-1, 1)
    )  # type: ignore[return-value]


def _clear_node_caches() -> None:
    """Bound stage-1 memory after all families at one node are complete."""

    _prewhitened_vertex_coefficients.cache_clear()
    _prewhitened_vertex_sign_coefficients.cache_clear()
    _ns_fast_edge_factor.cache_clear()
    _ramond_fast_edge_factor.cache_clear()


def _atomic_savez(path: Path, **arrays: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def prepare_response_shard(
    *,
    output_path: Path,
    tau: complex,
    points: Sequence[complex],
    quadrature_order: int,
    maximum_twice_levels: int | Sequence[int],
    condition_limit: float,
    task_index: int,
    task_count: int,
) -> Path:
    """Build one persistent shard of the fixed-geometry block response."""

    _validate_partition(task_index, task_count)
    cutoffs = _cutoff_pair(maximum_twice_levels)
    workspace = GenusOneFixedGeometryWorkspace.from_points(tau, points)
    plumbing = workspace.coordinates.plumbing_parameters
    if len(plumbing) != 2:
        raise ValueError("the reusable sweep currently requires two punctures")
    scales = tuple(-math.log(abs(value)) for value in plumbing)
    channel_tables = {
        powers: _laguerre_channels(
            quadrature_order=quadrature_order,
            scales=scales,
            endpoint_powers=powers,
        )
        for powers in {family.endpoint_powers for family in _FAMILIES}
    }
    total_nodes = quadrature_order**2
    node_indices = tuple(range(task_index, total_nodes, task_count))
    started = time.perf_counter()
    momenta = np.empty((len(node_indices), len(_FAMILIES), 2), dtype=np.float64)
    normalized_weights = np.empty(
        (len(node_indices), len(_FAMILIES)),
        dtype=np.float64,
    )
    responses = []
    for local_index, flat_index in enumerate(node_indices):
        node_responses = []
        for family_index, family in enumerate(_FAMILIES):
            node, transformed_weight = _laguerre_node(
                channel_tables[family.endpoint_powers],
                flat_index,
            )
            endpoint_factor = math.prod(
                momentum**power
                for momentum, power in zip(node, family.endpoint_powers)
            )
            momenta[local_index, family_index] = node
            normalized_weights[local_index, family_index] = (
                transformed_weight / endpoint_factor
            )
            node_responses.append(
                _build_family_response(
                    family,
                    node,
                    plumbing,
                    maximum_twice_levels=cutoffs,
                    condition_limit=condition_limit,
                )
            )
        responses.append(node_responses)
        _clear_node_caches()

    metadata = {
        "schema_version": SCHEMA_VERSION,
        "kind": "spin23_genus1_two_point_response",
        "tau": _encode_complex(tau),
        "points": [_encode_complex(value) for value in points],
        "plumbing_parameters": [_encode_complex(value) for value in plumbing],
        "quadrature_order": quadrature_order,
        "total_nodes": total_nodes,
        "maximum_twice_levels": list(cutoffs),
        "condition_limit": float(condition_limit),
        "task_index": task_index,
        "task_count": task_count,
        "families": [_family_metadata(family) for family in _FAMILIES],
        "runtime_seconds": time.perf_counter() - started,
    }
    _atomic_savez(
        output_path,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
        node_indices=np.asarray(node_indices, dtype=np.int64),
        momenta=momenta,
        normalized_weights=normalized_weights,
        response_polynomials=_pad_polynomials(responses),
    )
    return output_path


def _load_npz(path: Path) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata_json"].item()))
        arrays = {
            name: np.array(archive[name], copy=True)
            for name in archive.files
            if name != "metadata_json"
        }
    if metadata.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported response schema in {path}")
    return metadata, arrays


def _artifact_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evaluate_response_shard(
    *,
    response_path: Path,
    output_path: Path,
    omega_values: Sequence[complex],
    structure_precision: int,
) -> Path:
    """Evaluate one saved response shard for every requested energy."""

    metadata, arrays = _load_npz(response_path)
    if metadata.get("kind") != "spin23_genus1_two_point_response":
        raise ValueError("input is not a two-point response artifact")
    energies = np.asarray(tuple(complex(value) for value in omega_values))
    if energies.ndim != 1 or not len(energies):
        raise ValueError("omega_values must be a nonempty one-dimensional grid")
    if not np.all(np.isfinite(energies)):
        raise ValueError("omega_values must be finite")

    node_indices = arrays["node_indices"]
    momenta = arrays["momenta"]
    normalized_weights = arrays["normalized_weights"]
    polynomials = arrays["response_polynomials"]
    if momenta.shape[:2] != normalized_weights.shape or (
        polynomials.shape[:2] != normalized_weights.shape
    ):
        raise ValueError("response artifact arrays have incompatible shapes")

    plumbing = tuple(
        _decode_complex(value) for value in metadata["plumbing_parameters"]
    )
    ns_primary = math.prod(abs(value) ** (-1.0 / 8.0) for value in plumbing)
    partials = np.zeros((len(energies), len(_FAMILIES)), dtype=np.complex128)
    started = time.perf_counter()
    for energy_index, omega in enumerate(energies):
        external_weight = ns_weight(complex(omega))
        for local_index in range(len(node_indices)):
            ns_constants: tuple[complex, complex] | None = None
            for family_index, family in enumerate(_FAMILIES):
                internal = tuple(float(value) for value in momenta[local_index, family_index])
                if family.sector == "NS":
                    if ns_constants is None:
                        ns_constants = ns_structure_constants(
                            internal[1],
                            complex(omega),
                            internal[0],
                            precision=structure_precision,
                        )
                    branch_coefficients = (
                        ns_constants[0] ** 2,
                        ns_constants[1] ** 2,
                    )
                    primary = ns_primary
                else:
                    rr_constants = rr_ns_structure_constants(
                        internal[1],
                        internal[0],
                        complex(omega),
                        precision=structure_precision,
                    )
                    # Stored branch order is first_sign=(-1,+1), while the
                    # HJS structure table order is (even, odd).
                    branch_coefficients = (
                        2.0 * rr_constants[1] ** 2,
                        2.0 * rr_constants[0] ** 2,
                    )
                    primary = 1.0

                node_value = 0.0j
                for branch_index, coefficient in enumerate(branch_coefficients):
                    holomorphic = np.polynomial.polynomial.polyval(
                        external_weight,
                        polynomials[
                            local_index,
                            family_index,
                            branch_index,
                            0,
                        ],
                    )
                    antiholomorphic = np.polynomial.polynomial.polyval(
                        external_weight,
                        polynomials[
                            local_index,
                            family_index,
                            branch_index,
                            1,
                        ],
                    )
                    node_value += coefficient * holomorphic * antiholomorphic
                partials[energy_index, family_index] += (
                    normalized_weights[local_index, family_index]
                    * primary
                    * node_value
                )

    output_metadata = {
        "schema_version": SCHEMA_VERSION,
        "kind": "spin23_genus1_two_point_partial_integrals",
        "response_sha256": _artifact_sha256(response_path),
        "source_metadata": metadata,
        "structure_precision": int(structure_precision),
        "runtime_seconds": time.perf_counter() - started,
    }
    _atomic_savez(
        output_path,
        metadata_json=np.asarray(json.dumps(output_metadata, sort_keys=True)),
        node_indices=node_indices,
        omega_values=energies,
        partial_liouville_values=partials,
    )
    return output_path


def _stable_complex_sum(values: Sequence[complex]) -> complex:
    normalized = tuple(complex(value) for value in values)
    return complex(
        math.fsum(value.real for value in normalized),
        math.fsum(value.imag for value in normalized),
    )


def reduce_partial_integrals(
    *,
    partial_paths: Sequence[Path],
    output_csv: Path,
    flavor: int,
    free_field_precision: int,
    include_string_phase: bool,
) -> Path:
    """Reduce all node shards and assemble the physical fixed-moduli result."""

    if not partial_paths:
        raise ValueError("at least one partial integral is required")
    loaded = tuple(_load_npz(path) for path in partial_paths)
    reference_metadata = loaded[0][0]
    reference_source = reference_metadata["source_metadata"]
    omega_values = loaded[0][1]["omega_values"]
    for metadata, arrays in loaded[1:]:
        source = metadata["source_metadata"]
        for key in (
            "tau",
            "points",
            "plumbing_parameters",
            "quadrature_order",
            "total_nodes",
            "maximum_twice_levels",
            "condition_limit",
            "task_count",
            "families",
        ):
            if source[key] != reference_source[key]:
                raise ValueError(f"partial shards disagree on {key}")
        if not np.array_equal(arrays["omega_values"], omega_values):
            raise ValueError("partial shards use different energy grids")

    all_indices = np.concatenate(
        [arrays["node_indices"] for _, arrays in loaded]
    )
    total_nodes = int(reference_source["total_nodes"])
    if not np.array_equal(np.sort(all_indices), np.arange(total_nodes)):
        raise ValueError("partial shards do not cover every quadrature node once")

    partial_arrays = [
        arrays["partial_liouville_values"] for _, arrays in loaded
    ]
    liouville_values = np.empty_like(partial_arrays[0])
    for energy_index in range(len(omega_values)):
        for family_index in range(len(_FAMILIES)):
            liouville_values[energy_index, family_index] = _stable_complex_sum(
                values[energy_index, family_index] for values in partial_arrays
            )

    family_lookup = {
        (
            family.sector,
            family.temporal_lift_sign,
            family.holomorphic_words,
            family.antiholomorphic_words,
        ): index
        for index, family in enumerate(_FAMILIES)
    }

    def saved_liouville_batch(**options: object) -> tuple[SpectralIntegralDiagnostics, ...]:
        key = (
            options["sector"],
            int(options["temporal_lift_sign"]),
            tuple(tuple(word) for word in options["holomorphic_words"]),
            tuple(tuple(word) for word in options["antiholomorphic_words"]),
        )
        family_index = family_lookup[key]
        family = _FAMILIES[family_index]
        return tuple(
            SpectralIntegralDiagnostics(
                value=complex(value),
                refined_value=None,
                extended_value=None,
                quadrature_absolute_error=None,
                tail_absolute_error=None,
                estimated_absolute_error=None,
                dimension=2,
                p_max=None,
                quadrature_order=int(reference_source["quadrature_order"]),
                refined_order=None,
                extended_p_max=None,
                function_evaluations=total_nodes,
                quadrature_method="persistent_gauss_laguerre_response",
                orders_evaluated=(int(reference_source["quadrature_order"]),),
                endpoint_powers=family.endpoint_powers,
                block_backend_requested="direct_fast",
                worker_processes=int(reference_source["task_count"]),
            )
            for value in liouville_values[:, family_index]
        )

    tau = _decode_complex(reference_source["tau"])
    points = tuple(_decode_complex(value) for value in reference_source["points"])
    state_sets = tuple(
        (
            GenusOneNSState.vector(complex(omega), -complex(omega), flavor),
            GenusOneNSState.vector(complex(omega), complex(omega), flavor),
        )
        for omega in omega_values
    )
    workspace = GenusOneFixedGeometryWorkspace.from_points(tau, points)
    evaluations = workspace.evaluate(
        state_sets,
        maximum_twice_levels=tuple(reference_source["maximum_twice_levels"]),
        p_max=None,
        quadrature_order=int(reference_source["quadrature_order"]),
        refined_order=None,
        extended_p_max=None,
        structure_precision=int(reference_metadata["structure_precision"]),
        block_digits=50,
        condition_limit=float(reference_source["condition_limit"]),
        free_field_precision=free_field_precision,
        spectral_method="gauss_laguerre",
        spectral_relative_tolerance=None,
        block_backend="direct_fast",
        spectral_workers=1,
        spin_workers=1,
        include_string_phase=include_string_phase,
        liouville_batch_evaluator=saved_liouville_batch,
    )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "omega_re",
                "omega_im",
                "value_re",
                "value_im",
                "NS_re",
                "NS_im",
                "NS_tilde_re",
                "NS_tilde_im",
                "R_re",
                "R_im",
                "quadrature_order",
                "function_evaluations_per_component",
                "maximum_twice_level_0",
                "maximum_twice_level_1",
            ),
        )
        writer.writeheader()
        for omega, evaluation in zip(omega_values, evaluations):
            row: dict[str, object] = {
                "omega_re": float(complex(omega).real),
                "omega_im": float(complex(omega).imag),
                "value_re": float(evaluation.value.real),
                "value_im": float(evaluation.value.imag),
                "quadrature_order": int(reference_source["quadrature_order"]),
                "function_evaluations_per_component": total_nodes,
                "maximum_twice_level_0": int(
                    reference_source["maximum_twice_levels"][0]
                ),
                "maximum_twice_level_1": int(
                    reference_source["maximum_twice_levels"][1]
                ),
            }
            for label in ("NS", "NS_tilde", "R"):
                value = evaluation.fixed_spin[label].value
                row[f"{label}_re"] = float(value.real)
                row[f"{label}_im"] = float(value.imag)
            writer.writerow(row)
    return output_csv


def _complex_argument(values: Sequence[str], name: str) -> complex:
    if len(values) != 2:
        raise ValueError(f"{name} requires real and imaginary parts")
    return complex(float(values[0]), float(values[1]))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--tau", nargs=2, required=True)
    prepare.add_argument("--point", nargs=2, action="append", required=True)
    prepare.add_argument("--quadrature-order", type=int, default=100)
    prepare.add_argument("--maximum-twice-level", type=int, default=8)
    prepare.add_argument("--condition-limit", type=float, default=1.0e11)
    prepare.add_argument("--task-index", type=int, required=True)
    prepare.add_argument("--task-count", type=int, default=200)

    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--response", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--omega-count", type=int, default=1000)
    evaluate.add_argument("--omega-min", type=float, default=0.05)
    evaluate.add_argument("--omega-max", type=float, default=1.0)
    evaluate.add_argument("--omega-imag", type=float, default=0.0)
    evaluate.add_argument("--structure-precision", type=int, default=24)

    reduce = subparsers.add_parser("reduce")
    reduce.add_argument("--partials", type=Path, nargs="+", required=True)
    reduce.add_argument("--output-csv", type=Path, required=True)
    reduce.add_argument("--flavor", type=int, default=4)
    reduce.add_argument("--free-field-precision", type=int, default=40)
    reduce.add_argument("--include-string-phase", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "prepare":
        prepare_response_shard(
            output_path=args.output,
            tau=_complex_argument(args.tau, "tau"),
            points=tuple(
                _complex_argument(value, "point") for value in args.point
            ),
            quadrature_order=args.quadrature_order,
            maximum_twice_levels=args.maximum_twice_level,
            condition_limit=args.condition_limit,
            task_index=args.task_index,
            task_count=args.task_count,
        )
        return 0
    if args.command == "evaluate":
        if args.omega_count < 1 or args.omega_max < args.omega_min:
            raise ValueError("the energy grid must be nonempty and ordered")
        energies = (
            np.linspace(args.omega_min, args.omega_max, args.omega_count)
            + 1j * args.omega_imag
        )
        evaluate_response_shard(
            response_path=args.response,
            output_path=args.output,
            omega_values=energies,
            structure_precision=args.structure_precision,
        )
        return 0
    reduce_partial_integrals(
        partial_paths=args.partials,
        output_csv=args.output_csv,
        flavor=args.flavor,
        free_field_precision=args.free_field_precision,
        include_string_phase=args.include_string_phase,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "evaluate_response_shard",
    "prepare_response_shard",
    "reduce_partial_integrals",
]
