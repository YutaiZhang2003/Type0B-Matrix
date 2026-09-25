#!/usr/bin/env python3
"""Fast consistency checks for the stored VVVV numerical-fit data."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from heterotic_so23_1to3_fast import pair_channel_ansatz

CODE_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = CODE_DIR.parents[1]
DATA_DIR = WORKSPACE_DIR / "Data Set" / CODE_DIR.name


def main() -> None:
    data = json.loads((DATA_DIR / "heterotic_vvvv_fit_data.json").read_text())
    errors = []
    for row in data["rows"]:
        energies = [complex(*z) for z in row["energies"]]
        values = np.array([complex(*z) for z in row["value"]])
        prediction = pair_channel_ansatz(energies)
        errors.extend(np.abs((values - prediction) / prediction))
    errors = np.asarray(errors)
    assert errors.size == 30
    assert errors.max() < 3e-6, errors.max()

    # Resonance reduction at omega0=i for an arbitrary complex partition.
    w1 = 0.17 + 0.08j
    w2 = -0.06 + 0.21j
    w3 = 1j - w1 - w2
    pred = pair_channel_ansatz([w1, w2, w3, 1j])
    exact = math.pi * np.array([w1 * w2, w1 * w3, w2 * w3])
    assert np.max(np.abs(pred - exact)) < 2e-14

    print(
        json.dumps(
            {
                "stored_coefficients": int(errors.size),
                "max_relative_error": float(errors.max()),
                "rms_relative_error": float(np.sqrt(np.mean(errors**2))),
                "resonance_check": "passed",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
