#!/usr/bin/env python3
"""Fast structural checks for the genus-one momentum correction campaign."""

from __future__ import annotations

import csv
import math
import tempfile
from pathlib import Path

import numpy as np

try:
    from genus1_two_point_momentum_correction import (
        _channel_point,
        _parse_pairs,
        _tail_linear_weights,
        prepare_manifest,
    )
except ImportError:  # pragma: no cover - package-style execution
    from plumbing.genus1_two_point_momentum_correction import (
        _channel_point,
        _parse_pairs,
        _tail_linear_weights,
        prepare_manifest,
    )


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    require(
        _parse_pairs(("8x10", "12x14")) == ((8, 10), (12, 14)),
        "polar order parsing failed",
    )
    tau = 0.17 + 1.08j
    valid, label = _channel_point(0.25 + 0.1j, tau, 0.15)
    require(valid.channel == "ope" and label == "ope", "valid OPE point rejected")
    invalid, label = _channel_point(0.0 + 0.9j, tau, 0.15)
    require(
        invalid.channel == "necklace" and "fallback" in label,
        "invalid |v|>=1 OPE representation was not gated",
    )

    with tempfile.TemporaryDirectory(prefix="g1-correction-check-") as temporary:
        root = Path(temporary)
        summary = prepare_manifest(
            path=root / "manifest.csv",
            summary_path=root / "summary.json",
            x_values=(0.4,),
            replicates=2,
            bulk_sobol_power=2,
            tail_sobol_power=1,
            seed=19,
            cutoff=8.0,
            tail_slices=(8.0, 10.0, 12.0),
        )
        # 2 replicates * (4 bulk + 3 slices * 2 tail points).
        require(summary["target_count"] == 20, "manifest target count is wrong")
        with (root / "manifest.csv").open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        require(
            [int(row["target_index"]) for row in rows] == list(range(20)),
            "manifest indices are not contiguous",
        )

    tau2 = np.asarray([8.0, 10.0, 12.0, 16.0, 20.0])
    weights = _tail_linear_weights(tau2, 8.0)
    coefficients = np.asarray([1.3, -0.7, 0.2])
    values = (
        coefficients[0] * tau2**-2.0
        + coefficients[1] * tau2 ** (-5.0 / 3.0)
        + coefficients[2] * tau2**-3.0
    )
    exact_integral = (
        coefficients[0] / 8.0
        + 1.5 * coefficients[1] * 8.0 ** (-2.0 / 3.0)
        + 0.5 * coefficients[2] / 8.0**2
    )
    require(
        abs(float(np.dot(weights, values)) - exact_integral) < 2.0e-14,
        "tail correction linear weights do not reproduce the fit",
    )
    print("genus-one two-point momentum correction checks: PASS")


if __name__ == "__main__":
    main()
