#!/usr/bin/env python3
"""Evaluate one all-singlet point on the first nonzero five-point resonance."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import json
import math
from pathlib import Path
import time
from typing import Any, Sequence

from spin23_sphere_fivepoint_resonance import (
    ResonanceResidueSettings,
    all_singlet_first_resonance_states,
    build_assembled_resonant_fivepoint_workspace,
    integrate_full_sphere_resonance,
    singlet_resonance_collision_margin,
)


def _complex_json(value: complex) -> dict[str, float]:
    number = complex(value)
    return {"real": float(number.real), "imag": float(number.imag)}


def _json_ready(value: Any) -> Any:
    if isinstance(value, complex):
        return _complex_json(value)
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    return value


def _parse_numbers(text: str, *, count: int, name: str) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in text.split(",") if item.strip())
    if len(values) != count or any(not math.isfinite(value) for value in values):
        raise ValueError(f"{name} requires {count} finite comma-separated values")
    return values


def _parse_ints(text: str, *, name: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in text.split(",") if item.strip())
    if not values:
        raise ValueError(f"{name} cannot be empty")
    return values


def _manifest_partition(
    path: Path, index: int
) -> tuple[tuple[float, float, float, float], dict[str, Any]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not 0 <= index < len(rows):
        raise IndexError(f"manifest index {index} is outside 0..{len(rows)-1}")
    row = rows[index]
    partition = tuple(float(row[f"t{leg}"]) for leg in range(1, 5))
    metadata = {
        "manifest": str(path),
        "manifest_index": index,
        "manifest_kind": row.get("kind", "unspecified"),
        "manifest_collision_margin": (
            float(row["collision_margin"])
            if row.get("collision_margin") not in (None, "")
            else None
        ),
    }
    return partition, metadata  # type: ignore[return-value]


def _settings(args: argparse.Namespace) -> ResonanceResidueSettings:
    return ResonanceResidueSettings(
        regulator_radii=tuple(
            float(value) for value in args.regulator_radii.split(",")
        ),
        regulator_direction=complex(args.regulator_direction),
        cauchy_order=args.cauchy_order,
        outer_radius_ratio=args.outer_radius_ratio,
        inner_radius_ratio=args.inner_radius_ratio,
        extrapolation_degree=args.extrapolation_degree,
        structure_precision=args.structure_precision,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--partition",
        help="positive imaginary coefficients t1,t2,t3,t4 with sum two",
    )
    source.add_argument("--manifest", type=Path)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--pointwise", action="store_true")
    parser.add_argument("--q1", type=complex, default=0.18 + 0.03j)
    parser.add_argument("--q2", type=complex, default=0.22 - 0.02j)
    parser.add_argument("--maximum-twice-level", type=int, default=6)
    parser.add_argument("--lower-maximum-twice-level", type=int, default=4)
    parser.add_argument("--sample-powers", default="8,10,12")
    parser.add_argument("--replicates", type=int, default=8)
    parser.add_argument("--radial-scale", type=float, default=1.0)
    parser.add_argument(
        "--moduli-sampler",
        choices=("radial_charts", "global_planes"),
        default="radial_charts",
    )
    parser.add_argument("--seed", type=int, default=230824)
    parser.add_argument("--regulator-radii", default=".01,.007,.0045,.0028")
    parser.add_argument("--regulator-direction", default="1+.37j")
    parser.add_argument("--cauchy-order", type=int, default=10)
    parser.add_argument("--outer-radius-ratio", type=float, default=0.08)
    parser.add_argument("--inner-radius-ratio", type=float, default=0.025)
    parser.add_argument("--extrapolation-degree", type=int, default=3)
    parser.add_argument("--structure-precision", type=int, default=55)
    parser.add_argument("--block-digits", type=int, default=55)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.partition is not None:
        partition = _parse_numbers(
            args.partition, count=4, name="--partition"
        )
        source_metadata: dict[str, Any] = {"input_kind": "direct_partition"}
    else:
        partition, source_metadata = _manifest_partition(args.manifest, args.index)
    if abs(sum(partition) - 2.0) > 1.0e-10:
        raise ValueError("partition coefficients must sum to two")
    outgoing = tuple(1j * value for value in partition)
    states = all_singlet_first_resonance_states(outgoing)
    settings = _settings(args)
    started = time.time()

    common: dict[str, Any] = {
        "schema": "spin23-sphere-fivepoint-first-resonance-v1",
        "channel": "S_to_SSSS_raw",
        "screening_number": 1,
        "incoming_liouville_momentum": _complex_json(2j),
        "partition": list(partition),
        "collision_margin": singlet_resonance_collision_margin(partition),
        "input": source_metadata,
        "outgoing_momenta": [_complex_json(value) for value in outgoing],
        "residue_settings": _json_ready(asdict(settings)),
        "maximum_twice_level": args.maximum_twice_level,
        "lower_maximum_twice_level": args.lower_maximum_twice_level,
    }
    if args.pointwise:
        workspace = build_assembled_resonant_fivepoint_workspace(
            states,
            maximum_twice_levels=args.maximum_twice_level,
            residue_settings=settings,
            block_digits=args.block_digits,
        )
        levels = sorted(
            set(
                (
                    args.lower_maximum_twice_level,
                    args.maximum_twice_level,
                )
            )
        )
        values = {}
        for level in levels:
            value, spread = workspace.evaluate_with_spread(
                args.q1 * args.q2,
                args.q2,
                maximum_twice_levels=level,
            )
            values[str(level)] = {
                "value": _complex_json(value),
                "regulator_fit_spread": spread,
            }
        result = {
            **common,
            "mode": "pointwise",
            "q1": _complex_json(args.q1),
            "q2": _complex_json(args.q2),
            "levels": values,
            "maximum_gram_condition": workspace.maximum_gram_condition,
        }
    else:
        diagnostics = integrate_full_sphere_resonance(
            states,
            maximum_twice_levels=args.maximum_twice_level,
            lower_maximum_twice_levels=args.lower_maximum_twice_level,
            sample_powers=_parse_ints(args.sample_powers, name="--sample-powers"),
            replicates=args.replicates,
            radial_scale=args.radial_scale,
            base_seed=args.seed,
            moduli_sampler=args.moduli_sampler,
            residue_settings=settings,
            block_digits=args.block_digits,
        )
        result = {
            **common,
            "mode": "full_sphere_qmc",
            "diagnostics": _json_ready(asdict(diagnostics)),
        }
    result["runtime_seconds"] = time.time() - started
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
