#!/usr/bin/env python3
"""Combine independent sphere-five-point chamber probe JSON files."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np


def _decode_complex(value: dict[str, float]) -> complex:
    return complex(float(value["real"]), float(value["imag"]))


def _encode_complex(value: complex) -> dict[str, float]:
    number = complex(value)
    return {
        "real": float(number.real),
        "imag": float(number.imag),
        "abs": float(abs(number)),
    }


def reduce_chamber_files(paths: Sequence[Path]) -> dict[str, Any]:
    """Pool independent scrambled-Sobol replicate estimates."""

    input_paths = tuple(Path(path) for path in paths)
    if not input_paths:
        raise ValueError("at least one chamber file is required")
    payloads = [
        json.loads(path.read_text(encoding="utf-8")) for path in input_paths
    ]
    chambers = [payload.get("linear_chamber_profile") for payload in payloads]
    if any(chamber is None for chamber in chambers):
        raise ValueError("every input must contain a linear_chamber_profile")
    typed_chambers: list[dict[str, Any]] = chambers  # type: ignore[assignment]
    signature_fields = (
        "scope",
        "q_maxima",
        "sample_power",
        "samples_per_replicate",
        "maximum_twice_levels",
        "spectral_order",
        "spectral_p_max",
    )
    signature = {
        field: typed_chambers[0][field] for field in signature_fields
    }
    for chamber in typed_chambers[1:]:
        if any(chamber[field] != signature[field] for field in signature_fields):
            raise ValueError("chamber discretizations do not match")
    t = float(payloads[0]["equal_imaginary_t"])
    if any(float(payload["equal_imaginary_t"]) != t for payload in payloads[1:]):
        raise ValueError("input imaginary frequencies do not match")

    values = np.asarray(
        [
            _decode_complex(value)
            for chamber in typed_chambers
            for value in chamber["replicate_values"]
        ],
        dtype=np.complex128,
    )
    if len(values) < 2:
        raise ValueError("at least two independent replicates are required")
    mean = complex(np.mean(values))
    real_error = float(np.std(values.real, ddof=1) / math.sqrt(len(values)))
    imaginary_error = float(
        np.std(values.imag, ddof=1) / math.sqrt(len(values))
    )
    combined_error = math.hypot(real_error, imaginary_error)
    return {
        "scope": signature["scope"],
        "equal_imaginary_t": t,
        "value": _encode_complex(mean),
        "real_standard_error": real_error,
        "imaginary_standard_error": imaginary_error,
        "combined_standard_error": combined_error,
        "relative_standard_error": combined_error / max(abs(mean), 1.0e-300),
        "independent_replicates": len(values),
        "task_files": [str(path) for path in input_paths],
        "discretization": signature,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    result = reduce_chamber_files(args.inputs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
