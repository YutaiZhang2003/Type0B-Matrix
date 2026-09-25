#!/usr/bin/env python3
r"""Evaluate auditable fixed-moduli genus-one Spin(23) validation samples.

This runner deliberately stops before the integrations over ``tau`` and the
puncture positions.  Each manifest row specifies one ordered necklace chart,
external on-shell NS states, and a numerical profile.  The output retains the
three even spin structures, every total-supercurrent PCO component, and the
independent spectral diagnostics needed to audit the assembled value.

The optional lower block order probes finite conformal-block truncation.  The
quadrature-order, momentum-cutoff, and block-order differences are convergence
diagnostics; none is a rigorous error bound.

Distributed runs may save complete spin shards or disjoint Ramond PCO
components.  The reducer restores canonical component and spin order before
the diagonal GSO sum.  For the fixed-four-core scheduler, the reported wall
time is the sum of the two concurrent-wave maxima rather than the maximum of
all shards.  A profile with ``subtract_maximum_twice_levels`` returns only the
shared-table truncation correction and is labeled by that profile in the
output JSON.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import tempfile
import time
import traceback
from typing import Any, Mapping, Sequence

from spin23_genus1_amplitude import (
    GenusOneIntegrandEvaluation,
    GenusOneNSState,
    evaluate_genus_one_integrand,
    ordered_necklace_coordinates,
    validate_external_states,
)
from spin23_genus1_gso import diagonal_gso_table
from spin23_genus1_spectral import SpectralIntegralDiagnostics


_TINY = float.fromhex("0x1.0p-1022")
_MAX_FINITE_FLOAT = float.fromhex("0x1.fffffffffffffp+1023")
_ASSEMBLER_KEYS = (
    "maximum_twice_levels",
    "p_max",
    "quadrature_order",
    "refined_order",
    "extended_p_max",
    "structure_precision",
    "block_digits",
    "condition_limit",
    "free_field_precision",
    "spectral_method",
    "spectral_relative_tolerance",
    "include_string_phase",
)
_OPTIONAL_ASSEMBLER_DEFAULTS = {
    "block_backend": "direct",
    "recursion_radius": 0.04,
    "recursion_check_radius": 0.05,
    "recursion_samples": 24,
    "recursion_finite_part_tolerance": 1.0e-7,
    "spectral_workers": 1,
    "spin_workers": 1,
    "subtract_maximum_twice_levels": None,
}


def _decode_complex(value: Any, *, name: str) -> complex:
    """Decode a manifest complex number stored as ``[real, imaginary]``."""

    if not isinstance(value, list) or len(value) != 2:
        raise TypeError(f"{name} must be a two-entry [real, imaginary] list")
    result = complex(float(value[0]), float(value[1]))
    if not math.isfinite(result.real) or not math.isfinite(result.imag):
        raise ValueError(f"{name} must be finite")
    return result


def _encode_complex(value: complex | None) -> dict[str, float] | None:
    """Encode a finite complex number in a JSON-portable representation."""

    if value is None:
        return None
    result = complex(value)
    if not math.isfinite(result.real) or not math.isfinite(result.imag):
        raise ArithmeticError("cannot serialize a non-finite complex value")
    return {"re": float(result.real), "im": float(result.imag)}


def _relative_change(change: float | None, value: complex) -> float | None:
    if change is None:
        return None
    return float(change / max(abs(value), _TINY))


def load_validation_manifest(path: Path) -> dict[str, Any]:
    """Load and structurally validate a genus-one validation manifest."""

    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError("the validation manifest must use schema_version 1")
    profiles = manifest.get("profiles")
    samples = manifest.get("samples")
    if not isinstance(profiles, dict) or not profiles:
        raise ValueError("the validation manifest must define profiles")
    if not isinstance(samples, list) or not samples:
        raise ValueError("the validation manifest must define samples")
    identifiers = [sample.get("id") for sample in samples]
    if any(not isinstance(identifier, str) or not identifier for identifier in identifiers):
        raise ValueError("every sample must have a nonempty string id")
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("sample ids must be unique")
    for profile_name, profile in profiles.items():
        if not isinstance(profile, dict):
            raise TypeError(f"profile {profile_name!r} must be an object")
        missing = set(_ASSEMBLER_KEYS) - set(profile)
        if missing:
            raise ValueError(
                f"profile {profile_name!r} is missing {sorted(missing)!r}"
            )
    return manifest


def states_from_sample(sample: Mapping[str, Any]) -> tuple[GenusOneNSState, ...]:
    """Construct and validate physical NS states from one manifest sample."""

    encoded_states = sample.get("states")
    if not isinstance(encoded_states, list):
        raise TypeError("sample states must be a list")
    states: list[GenusOneNSState] = []
    for index, encoded in enumerate(encoded_states):
        if not isinstance(encoded, dict):
            raise TypeError(f"state {index} must be an object")
        kind = encoded.get("kind")
        p = _decode_complex(
            encoded.get("liouville_momentum"),
            name=f"state {index} liouville_momentum",
        )
        k = _decode_complex(
            encoded.get("time_momentum"),
            name=f"state {index} time_momentum",
        )
        if kind == "vector":
            states.append(GenusOneNSState.vector(p, k, int(encoded["flavor"])))
        elif kind == "singlet":
            states.append(GenusOneNSState.singlet(p, k))
        else:
            raise ValueError(f"state {index} has unknown kind {kind!r}")
    return validate_external_states(states)


def coordinates_from_sample(
    sample: Mapping[str, Any],
) -> tuple[complex, tuple[complex, ...]]:
    """Decode and validate the ordered additive torus coordinates."""

    tau = _decode_complex(sample.get("tau"), name="tau")
    encoded_points = sample.get("points")
    if not isinstance(encoded_points, list):
        raise TypeError("sample points must be a list")
    points = tuple(
        _decode_complex(value, name=f"point {index}")
        for index, value in enumerate(encoded_points)
    )
    ordered_necklace_coordinates(tau, points)
    return tau, points


def _assembler_options(profile: Mapping[str, Any]) -> dict[str, Any]:
    options = {name: profile[name] for name in _ASSEMBLER_KEYS}
    options.update(
        {
            name: profile.get(name, default)
            for name, default in _OPTIONAL_ASSEMBLER_DEFAULTS.items()
        }
    )
    return options


def _lower_block_options(profile: Mapping[str, Any]) -> dict[str, Any] | None:
    """Return the base-rule controls for an optional lower block order."""

    lower_level = profile.get("lower_maximum_twice_levels")
    if lower_level is None:
        return None
    options = _assembler_options(profile)
    if isinstance(lower_level, int):
        normalized_lower_level: int | tuple[int, ...] = lower_level
    elif isinstance(lower_level, (list, tuple)):
        normalized_lower_level = tuple(int(value) for value in lower_level)
    else:
        raise TypeError(
            "lower_maximum_twice_levels must be an integer or a sequence"
        )
    options["maximum_twice_levels"] = normalized_lower_level
    # Only the selected value enters the block-order comparison.  For the
    # legacy rules, repeating separate refinement diagnostics would leave that
    # value unchanged while adding work.  The adaptive Laguerre method uses
    # refined_order as its maximum allowed order and must retain it.
    if options["spectral_method"] != "gauss_laguerre_adaptive":
        options["refined_order"] = None
    options["extended_p_max"] = None
    if options["spectral_method"] == "gauss_laguerre_7_15":
        # The selected rule is already order 15.  Reuse that order for the
        # lower-block comparison without repeating its order-7 diagnostic.
        options["spectral_method"] = "gauss_laguerre"
        options["quadrature_order"] = 15
    return options


def _serialize_spectral(
    diagnostic: SpectralIntegralDiagnostics,
) -> dict[str, Any]:
    return {
        "value": _encode_complex(diagnostic.value),
        "refined_value": _encode_complex(diagnostic.refined_value),
        "extended_value": _encode_complex(diagnostic.extended_value),
        "quadrature_absolute_error": diagnostic.quadrature_absolute_error,
        "tail_absolute_error": diagnostic.tail_absolute_error,
        "estimated_absolute_error": diagnostic.estimated_absolute_error,
        "estimated_relative_error": diagnostic.estimated_relative_error,
        "dimension": diagnostic.dimension,
        "p_max": diagnostic.p_max,
        "quadrature_order": diagnostic.quadrature_order,
        "refined_order": diagnostic.refined_order,
        "extended_p_max": diagnostic.extended_p_max,
        "function_evaluations": diagnostic.function_evaluations,
        "quadrature_method": diagnostic.quadrature_method,
        "coarse_value": _encode_complex(diagnostic.coarse_value),
        "coarse_order": diagnostic.coarse_order,
        "target_relative_tolerance": diagnostic.target_relative_tolerance,
        "converged": diagnostic.converged,
        "orders_evaluated": list(diagnostic.orders_evaluated),
        "endpoint_powers": list(diagnostic.endpoint_powers),
        "block_backend_requested": diagnostic.block_backend_requested,
        "recursive_block_evaluations": (
            diagnostic.recursive_block_evaluations
        ),
        "direct_block_evaluations": diagnostic.direct_block_evaluations,
        "fast_direct_block_evaluations": (
            diagnostic.fast_direct_block_evaluations
        ),
        "recursion_fallbacks": diagnostic.recursion_fallbacks,
        "worker_processes": diagnostic.worker_processes,
        "runtime_seconds": diagnostic.runtime_seconds,
    }


def _serialize_evaluation(
    evaluation: GenusOneIntegrandEvaluation,
) -> dict[str, Any]:
    fixed_spin: dict[str, Any] = {}
    for label, spin in evaluation.fixed_spin.items():
        components = []
        for component in spin.components:
            components.append(
                {
                    "time_fermion_indices": list(
                        component.component.time_fermion_indices
                    ),
                    "momentum_coefficient": _encode_complex(
                        component.component.momentum_coefficient
                    ),
                    "holomorphic_liouville_words": [
                        [repr(operator) for operator in word]
                        for word in component.component.holomorphic_liouville_words
                    ],
                    "value": _encode_complex(component.value),
                    "multiplier": _encode_complex(component.multiplier),
                    "liouville": _serialize_spectral(component.liouville),
                }
            )
        fixed_spin[label] = {
            "value": _encode_complex(spin.value),
            "absolute_component_scale": spin.absolute_component_scale,
            "cancellation_factor": spin.cancellation_factor,
            "estimated_absolute_error": spin.estimated_absolute_error,
            "estimated_relative_error": _relative_change(
                spin.estimated_absolute_error,
                spin.value,
            ),
            "estimated_error_over_absolute_scale": (
                None
                if spin.estimated_absolute_error is None
                else spin.estimated_absolute_error
                / max(spin.absolute_component_scale, _TINY)
            ),
            "common_free_field_factor": _encode_complex(
                spin.common_free_field_factor
            ),
            "components": components,
        }
    return {
        "value": _encode_complex(evaluation.value),
        "absolute_spin_scale": evaluation.absolute_spin_scale,
        "cancellation_factor": evaluation.cancellation_factor,
        "estimated_absolute_error": evaluation.estimated_absolute_error,
        "estimated_relative_error": _relative_change(
            evaluation.estimated_absolute_error,
            evaluation.value,
        ),
        "estimated_error_over_absolute_scale": (
            None
            if evaluation.estimated_absolute_error is None
            else evaluation.estimated_absolute_error
            / max(evaluation.absolute_spin_scale, _TINY)
        ),
        "picture_raising_factor": evaluation.picture_raising_factor,
        "includes_string_phase": evaluation.includes_string_phase,
        "coordinates": {
            "tau": _encode_complex(evaluation.coordinates.tau),
            "additive_points": [
                _encode_complex(value)
                for value in evaluation.coordinates.additive_points
            ],
            "multiplicative_points": [
                _encode_complex(value)
                for value in evaluation.coordinates.multiplicative_points
            ],
            "plumbing_parameters": [
                _encode_complex(value)
                for value in evaluation.coordinates.plumbing_parameters
            ],
        },
        "fixed_spin": fixed_spin,
    }


def _block_comparison(
    current: GenusOneIntegrandEvaluation,
    lower: GenusOneIntegrandEvaluation | None,
) -> dict[str, Any]:
    block_change = None if lower is None else float(abs(current.value - lower.value))
    spin_changes: dict[str, float] = {}
    conservative_block_change = None
    if lower is not None:
        spin_changes = {
            label: float(abs(spin.value - lower.fixed_spin[label].value))
            for label, spin in current.fixed_spin.items()
        }
        projector_weights = {
            entry.spin_structure.label: abs(entry.projector_coefficient)
            for entry in diagonal_gso_table()
        }
        conservative_block_change = sum(
            projector_weights[label] * change
            for label, change in spin_changes.items()
        )
    available = tuple(
        value
        for value in (
            current.estimated_absolute_error,
            conservative_block_change,
        )
        if value is not None
    )
    combined = max(available) if available else None
    return {
        "block_truncation_absolute_change": block_change,
        "block_truncation_relative_change": _relative_change(
            block_change,
            current.value,
        ),
        "block_truncation_change_over_absolute_scale": (
            None
            if block_change is None
            else block_change / max(current.absolute_spin_scale, _TINY)
        ),
        "block_truncation_conservative_absolute_change": (
            conservative_block_change
        ),
        "block_truncation_conservative_relative_change": _relative_change(
            conservative_block_change,
            current.value,
        ),
        "block_truncation_conservative_change_over_absolute_scale": (
            None
            if conservative_block_change is None
            else conservative_block_change
            / max(current.absolute_spin_scale, _TINY)
        ),
        "spin_block_truncation_absolute_changes": spin_changes,
        "combined_diagnostic_absolute_error": combined,
        "combined_diagnostic_relative_error": _relative_change(
            combined,
            current.value,
        ),
        "combined_diagnostic_error_over_absolute_scale": (
            None
            if combined is None
            else combined / max(current.absolute_spin_scale, _TINY)
        ),
        "is_rigorous_error_bound": False,
    }


def evaluate_sample(
    sample: Mapping[str, Any],
    profile: Mapping[str, Any],
    *,
    spin_labels: Sequence[str] | None = None,
    component_indices: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Evaluate one fixed-moduli sample and all requested diagnostics."""

    states = states_from_sample(sample)
    tau, points = coordinates_from_sample(sample)
    options = _assembler_options(profile)
    if spin_labels is not None:
        options["spin_labels"] = tuple(spin_labels)
    if component_indices is not None:
        options["component_indices"] = tuple(component_indices)
    current_started = time.perf_counter()
    current = evaluate_genus_one_integrand(states, tau, points, **options)
    current_runtime_seconds = time.perf_counter() - current_started

    lower = None
    lower_runtime_seconds = None
    lower_options = _lower_block_options(profile)
    if lower_options is not None:
        if spin_labels is not None:
            lower_options["spin_labels"] = tuple(spin_labels)
        if component_indices is not None:
            lower_options["component_indices"] = tuple(component_indices)
        lower_started = time.perf_counter()
        lower = evaluate_genus_one_integrand(
            states,
            tau,
            points,
            **lower_options,
        )
        lower_runtime_seconds = time.perf_counter() - lower_started
    return {
        "current": _serialize_evaluation(current),
        "lower_block_order": (
            None if lower is None else _serialize_evaluation(lower)
        ),
        "lower_block_order_controls": lower_options,
        "convergence": _block_comparison(current, lower),
        "timing": {
            "current_runtime_seconds": current_runtime_seconds,
            "lower_runtime_seconds": lower_runtime_seconds,
        },
    }


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        text=True,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _completed_successfully(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return payload.get("status") == "ok"


def _complex_from_json(value: Mapping[str, Any]) -> complex:
    """Decode this runner's serialized finite complex representation."""

    return complex(float(value["re"]), float(value["im"]))


def _combine_serialized_spin_evaluations(
    evaluations: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Combine disjoint fixed-spin shards in deterministic GSO order."""

    if not evaluations:
        raise ValueError("at least one serialized spin evaluation is required")
    reference = evaluations[0]
    fixed_spin: dict[str, Any] = {}
    for evaluation in evaluations:
        if evaluation["coordinates"] != reference["coordinates"]:
            raise ValueError("spin shards use different necklace coordinates")
        if evaluation["picture_raising_factor"] != reference["picture_raising_factor"]:
            raise ValueError("spin shards use different picture factors")
        if evaluation["includes_string_phase"] != reference["includes_string_phase"]:
            raise ValueError("spin shards use different string-phase choices")
        for label, spin in evaluation["fixed_spin"].items():
            if label not in fixed_spin:
                fixed_spin[label] = dict(spin)
                continue
            previous = fixed_spin[label]
            if previous.get("common_free_field_factor") != spin.get(
                "common_free_field_factor"
            ):
                raise ValueError(
                    f"component shards for {label!r} use different free-field factors"
                )
            previous_components = tuple(previous.get("components", ()))
            new_components = tuple(spin.get("components", ()))
            component_keys = [
                tuple(component["time_fermion_indices"])
                for component in (*previous_components, *new_components)
            ]
            if len(set(component_keys)) != len(component_keys):
                raise ValueError(
                    f"duplicate PCO component in fixed-spin shard {label!r}"
                )
            raw_value = _complex_from_json(previous["value"]) + _complex_from_json(
                spin["value"]
            )
            partial_errors = tuple(
                float(value)
                for value in (
                    previous.get("estimated_absolute_error"),
                    spin.get("estimated_absolute_error"),
                )
                if value is not None
            )
            raw_error = sum(partial_errors) if partial_errors else None
            combined_components = sorted(
                (*previous_components, *new_components),
                key=lambda component: (
                    len(component["time_fermion_indices"]),
                    tuple(component["time_fermion_indices"]),
                ),
            )
            component_scale = sum(
                abs(_complex_from_json(component["value"]))
                for component in combined_components
            )
            fixed_spin[label] = {
                **previous,
                "value": _encode_complex(raw_value),
                "absolute_component_scale": component_scale,
                "cancellation_factor": (
                    min(
                        component_scale / max(abs(raw_value), _TINY),
                        _MAX_FINITE_FLOAT,
                    )
                    if raw_value != 0
                    else (_MAX_FINITE_FLOAT if component_scale else 1.0)
                ),
                "estimated_absolute_error": raw_error,
                "estimated_relative_error": _relative_change(
                    raw_error,
                    raw_value,
                ),
                "estimated_error_over_absolute_scale": (
                    None
                    if raw_error is None
                    else raw_error / max(component_scale, _TINY)
                ),
                "components": combined_components,
            }

    expected_order = tuple(
        entry.spin_structure.label
        for entry in diagonal_gso_table()
        if not entry.spin_structure.arf_invariant
    )
    if set(fixed_spin) != set(expected_order):
        raise ValueError(
            "spin shards must contain exactly NS, NS_tilde, and R"
        )
    projector_entries = tuple(
        entry for entry in diagonal_gso_table()
        if not entry.spin_structure.arf_invariant
    )
    total = sum(
        (
            entry.projector_coefficient
            * _complex_from_json(
                fixed_spin[entry.spin_structure.label]["value"]
            )
            for entry in projector_entries
        ),
        0.0j,
    )
    error_terms = tuple(
        abs(entry.projector_coefficient)
        * float(fixed_spin[entry.spin_structure.label]["estimated_absolute_error"])
        for entry in projector_entries
        if fixed_spin[entry.spin_structure.label].get(
            "estimated_absolute_error"
        ) is not None
    )
    combined_error = sum(error_terms) if error_terms else None
    absolute_spin_scale = sum(
        abs(
            entry.projector_coefficient
            * _complex_from_json(
                fixed_spin[entry.spin_structure.label]["value"]
            )
        )
        for entry in projector_entries
    )
    if reference["includes_string_phase"]:
        point_count = len(reference["coordinates"]["additive_points"])
        phase = 1j**point_count
        total *= phase
    return {
        **reference,
        "value": _encode_complex(total),
        "absolute_spin_scale": absolute_spin_scale,
        "cancellation_factor": (
            min(
                absolute_spin_scale / max(abs(total), _TINY),
                _MAX_FINITE_FLOAT,
            )
            if total != 0
            else (_MAX_FINITE_FLOAT if absolute_spin_scale else 1.0)
        ),
        "estimated_absolute_error": combined_error,
        "estimated_relative_error": _relative_change(combined_error, total),
        "estimated_error_over_absolute_scale": (
            None
            if combined_error is None
            else combined_error / max(absolute_spin_scale, _TINY)
        ),
        "fixed_spin": {
            label: fixed_spin[label] for label in expected_order
        },
    }


def _serialized_block_comparison(
    current: Mapping[str, Any],
    lower: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Reconstruct the full block-order diagnostic from spin shards."""

    current_value = _complex_from_json(current["value"])
    block_change = None
    conservative_change = None
    spin_changes: dict[str, float] = {}
    if lower is not None:
        lower_value = _complex_from_json(lower["value"])
        block_change = float(abs(current_value - lower_value))
        projector_weights = {
            entry.spin_structure.label: abs(entry.projector_coefficient)
            for entry in diagonal_gso_table()
            if not entry.spin_structure.arf_invariant
        }
        spin_changes = {
            label: float(
                abs(
                    _complex_from_json(spin["value"])
                    - _complex_from_json(lower["fixed_spin"][label]["value"])
                )
            )
            for label, spin in current["fixed_spin"].items()
        }
        conservative_change = sum(
            projector_weights[label] * change
            for label, change in spin_changes.items()
        )
    available = tuple(
        float(value)
        for value in (
            current.get("estimated_absolute_error"),
            conservative_change,
        )
        if value is not None
    )
    combined = max(available) if available else None
    absolute_spin_scale = float(current["absolute_spin_scale"])
    return {
        "block_truncation_absolute_change": block_change,
        "block_truncation_relative_change": _relative_change(
            block_change,
            current_value,
        ),
        "block_truncation_change_over_absolute_scale": (
            None
            if block_change is None
            else block_change / max(absolute_spin_scale, _TINY)
        ),
        "block_truncation_conservative_absolute_change": conservative_change,
        "block_truncation_conservative_relative_change": _relative_change(
            conservative_change,
            current_value,
        ),
        "block_truncation_conservative_change_over_absolute_scale": (
            None
            if conservative_change is None
            else conservative_change / max(absolute_spin_scale, _TINY)
        ),
        "spin_block_truncation_absolute_changes": spin_changes,
        "combined_diagnostic_absolute_error": combined,
        "combined_diagnostic_relative_error": _relative_change(
            combined,
            current_value,
        ),
        "combined_diagnostic_error_over_absolute_scale": (
            None
            if combined is None
            else combined / max(absolute_spin_scale, _TINY)
        ),
        "is_rigorous_error_bound": False,
    }


def _spin_shard_schedule_wall_seconds(
    runtimes: Mapping[str, float],
    *,
    sequential_spin_waves: bool,
) -> float:
    """Return wall time for concurrent shards or the balanced two-wave plan."""

    if not runtimes:
        raise ValueError("at least one shard runtime is required")
    if not sequential_spin_waves:
        return max(float(value) for value in runtimes.values())
    required = {"NS", "NS_tilde", "R_component_0", "R_component_1"}
    if set(runtimes) != required:
        raise ValueError(
            "the sequential spin-wave schedule requires split Ramond "
            "component shards"
        )
    return max(
        float(runtimes["R_component_0"]),
        float(runtimes["R_component_1"]),
    ) + max(
        float(runtimes["NS"]),
        float(runtimes["NS_tilde"]),
    )


def run_manifest_sample(
    manifest_path: Path,
    index: int,
    profile_name: str,
    output_dir: Path,
    *,
    overwrite: bool = False,
    spin_label: str | None = None,
    component_index: int | None = None,
) -> Path:
    """Evaluate one manifest sample and atomically write its JSON record."""

    manifest = load_validation_manifest(manifest_path)
    samples = manifest["samples"]
    if not 0 <= index < len(samples):
        raise IndexError(f"sample index must lie in [0, {len(samples)})")
    if profile_name not in manifest["profiles"]:
        raise KeyError(f"unknown numerical profile {profile_name!r}")
    sample = samples[index]
    if component_index is not None and spin_label is None:
        raise ValueError("component_index requires a single spin_label")
    component_suffix = (
        "" if component_index is None else f"_component_{component_index}"
    )
    spin_suffix = (
        "" if spin_label is None else f"_spin_{spin_label}{component_suffix}"
    )
    output = output_dir / (
        f"result_{index:03d}_{sample['id']}{spin_suffix}.json"
    )
    if not overwrite and _completed_successfully(output):
        print(f"skipping completed sample {sample['id']}: {output}", flush=True)
        return output

    started = time.perf_counter()
    base_payload = {
        "schema_version": 1,
        "sample_index": index,
        "sample_id": sample["id"],
        "profile_name": profile_name,
        "manifest": str(manifest_path),
        "sample": sample,
        "profile": manifest["profiles"][profile_name],
        "scope": (
            "fixed_moduli_integrand_not_moduli_space_integral"
            if spin_label is None
            else "fixed_moduli_integrand_single_spin_shard"
        ),
        "spin_label": spin_label,
        "component_index": component_index,
    }
    print(
        f"starting sample {index}: {sample['id']} with profile {profile_name}",
        flush=True,
    )
    try:
        result = evaluate_sample(
            sample,
            manifest["profiles"][profile_name],
            spin_labels=(spin_label,) if spin_label is not None else None,
            component_indices=(
                (component_index,) if component_index is not None else None
            ),
        )
        payload = {
            **base_payload,
            "status": "ok",
            "runtime_seconds": time.perf_counter() - started,
            "result": result,
        }
        _atomic_write_json(output, payload)
        print(
            f"completed {sample['id']} in {payload['runtime_seconds']:.1f} s: "
            f"{output}",
            flush=True,
        )
        return output
    except Exception as error:
        failure = {
            **base_payload,
            "status": "error",
            "runtime_seconds": time.perf_counter() - started,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "traceback": traceback.format_exc(),
        }
        _atomic_write_json(output, failure)
        raise


def reduce_manifest_spin_shards(
    manifest_path: Path,
    index: int,
    profile_name: str,
    output_dir: Path,
    *,
    overwrite: bool = False,
    split_ramond_components: bool = False,
    sequential_spin_waves: bool = False,
) -> Path:
    """Merge three successful fixed-spin shards into one standard result."""

    manifest = load_validation_manifest(manifest_path)
    samples = manifest["samples"]
    if not 0 <= index < len(samples):
        raise IndexError(f"sample index must lie in [0, {len(samples)})")
    if profile_name not in manifest["profiles"]:
        raise KeyError(f"unknown numerical profile {profile_name!r}")
    sample = samples[index]
    output = output_dir / f"result_{index:03d}_{sample['id']}.json"
    if not overwrite and _completed_successfully(output):
        print(f"skipping completed reduction {sample['id']}: {output}", flush=True)
        return output

    labels = tuple(
        entry.spin_structure.label
        for entry in diagonal_gso_table()
        if not entry.spin_structure.arf_invariant
    )
    shard_specs: tuple[tuple[str, int | None], ...] = (
        (
            ("NS", None),
            ("NS_tilde", None),
            ("R", 0),
            ("R", 1),
        )
        if split_ramond_components
        else tuple((label, None) for label in labels)
    )
    shard_paths = tuple(
        output_dir
        / (
            f"result_{index:03d}_{sample['id']}_spin_{label}"
            + ("" if component is None else f"_component_{component}")
            + ".json"
        )
        for label, component in shard_specs
    )
    shards = []
    for (label, component), path in zip(shard_specs, shard_paths):
        if not _completed_successfully(path):
            raise FileNotFoundError(
                f"missing successful {label} spin shard: {path}"
            )
        shard = json.loads(path.read_text(encoding="utf-8"))
        if shard.get("sample_id") != sample["id"]:
            raise ValueError(f"spin shard {path} has the wrong sample id")
        if shard.get("profile_name") != profile_name:
            raise ValueError(f"spin shard {path} has the wrong profile")
        if shard.get("spin_label") != label:
            raise ValueError(f"spin shard {path} has the wrong spin label")
        if shard.get("component_index") != component:
            raise ValueError(f"spin shard {path} has the wrong component index")
        shards.append(shard)

    current = _combine_serialized_spin_evaluations(
        tuple(shard["result"]["current"] for shard in shards)
    )
    lower_rows = tuple(
        shard["result"]["lower_block_order"] for shard in shards
    )
    if all(row is None for row in lower_rows):
        lower = None
    elif any(row is None for row in lower_rows):
        raise ValueError("spin shards disagree on the lower block-order probe")
    else:
        lower = _combine_serialized_spin_evaluations(lower_rows)  # type: ignore[arg-type]

    shard_names = tuple(
        label if component is None else f"{label}_component_{component}"
        for label, component in shard_specs
    )
    shard_runtimes = {
        name: float(shard["runtime_seconds"])
        for name, shard in zip(shard_names, shards)
    }
    current_runtimes = {
        name: float(shard["result"]["timing"]["current_runtime_seconds"])
        for name, shard in zip(shard_names, shards)
    }
    lower_runtimes = {
        name: shard["result"]["timing"]["lower_runtime_seconds"]
        for name, shard in zip(shard_names, shards)
    }
    if sequential_spin_waves and not split_ramond_components:
        raise ValueError(
            "sequential spin waves require split Ramond components"
        )
    current_schedule_wall = _spin_shard_schedule_wall_seconds(
        current_runtimes,
        sequential_spin_waves=sequential_spin_waves,
    )
    lower_schedule_wall = (
        None
        if all(value is None for value in lower_runtimes.values())
        else _spin_shard_schedule_wall_seconds(
            {
                name: float(value)
                for name, value in lower_runtimes.items()
                if value is not None
            },
            sequential_spin_waves=sequential_spin_waves,
        )
    )
    schedule_wall = _spin_shard_schedule_wall_seconds(
        shard_runtimes,
        sequential_spin_waves=sequential_spin_waves,
    )
    result = {
        "current": current,
        "lower_block_order": lower,
        "lower_block_order_controls": shards[0]["result"][
            "lower_block_order_controls"
        ],
        "convergence": _serialized_block_comparison(current, lower),
        "timing": {
            "current_runtime_seconds": current_schedule_wall,
            "lower_runtime_seconds": lower_schedule_wall,
            "distributed_spin_runtime_seconds": shard_runtimes,
            "distributed_current_runtime_seconds": current_runtimes,
            "distributed_lower_runtime_seconds": lower_runtimes,
            "execution_schedule": (
                "ramond_then_ns_two_waves"
                if sequential_spin_waves
                else "all_shards_concurrent"
            ),
        },
    }
    payload = {
        "schema_version": 1,
        "sample_index": index,
        "sample_id": sample["id"],
        "profile_name": profile_name,
        "manifest": str(manifest_path),
        "sample": sample,
        "profile": manifest["profiles"][profile_name],
        "scope": "fixed_moduli_integrand_not_moduli_space_integral",
        "spin_label": None,
        "component_index": None,
        "status": "ok",
        "runtime_seconds": schedule_wall,
        "result": result,
    }
    _atomic_write_json(output, payload)
    print(
        f"reduced {sample['id']} in {payload['runtime_seconds']:.1f} s: "
        f"{output}",
        flush=True,
    )
    return output


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--task-index", "--index", dest="index", type=int, required=True)
    parser.add_argument("--profile", default="validation")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--spin-label",
        choices=("NS", "NS_tilde", "R"),
        default=None,
        help="evaluate only one even spin structure and write a shard",
    )
    parser.add_argument(
        "--component-index",
        type=int,
        default=None,
        help="evaluate one PCO component of the selected spin structure",
    )
    parser.add_argument(
        "--reduce-spin-shards",
        action="store_true",
        help="merge completed NS, NS_tilde, and R shards",
    )
    parser.add_argument(
        "--split-ramond-components",
        action="store_true",
        help="reduce NS, NS_tilde, R-component-0, and R-component-1 shards",
    )
    parser.add_argument(
        "--sequential-spin-waves",
        action="store_true",
        help="report a Ramond wave followed by an NS wave",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run one manifest shard/sample or reduce completed distributed shards."""

    args = _parse_args(argv)
    if args.reduce_spin_shards:
        if args.spin_label is not None or args.component_index is not None:
            raise ValueError(
                "--reduce-spin-shards cannot select a spin or component"
            )
        reduce_manifest_spin_shards(
            args.manifest,
            args.index,
            args.profile,
            args.output_dir,
            overwrite=args.overwrite,
            split_ramond_components=args.split_ramond_components,
            sequential_spin_waves=args.sequential_spin_waves,
        )
    else:
        if args.split_ramond_components or args.sequential_spin_waves:
            raise ValueError(
                "shard-reducer options require --reduce-spin-shards"
            )
        run_manifest_sample(
            args.manifest,
            args.index,
            args.profile,
            args.output_dir,
            overwrite=args.overwrite,
            spin_label=args.spin_label,
            component_index=args.component_index,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
