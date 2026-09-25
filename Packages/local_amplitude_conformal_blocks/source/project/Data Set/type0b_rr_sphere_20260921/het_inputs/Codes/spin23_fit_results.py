#!/usr/bin/env python3
"""Fit the Spin(23) 1->3 channel data to a holomorphic polynomial
R_i = F(S,p,v), where R_i=M_i/(pi*omega_j*omega_k).

Typical usage after collecting one CSV per array task:

  python spin23_fit_results.py '../Data Set/results/*.csv' --degree 6 \
      --output fit_degree6.json

Rows with family=core and setting=production are used for training.  Rows with
family=structured and setting=production are used as the default independent
test set.  All three tensor channels are pooled into one Bose-symmetric fit.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


CODE_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = CODE_DIR.parent / "Data Set" / "spin23_fit.json"


def load_rows(patterns: Sequence[str]) -> list[dict[str, str]]:
    paths: list[str] = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        paths.extend(matches if matches else [pattern])
    rows: list[dict[str, str]] = []
    for path in paths:
        with open(path, newline="") as handle:
            rows.extend(csv.DictReader(handle))
    return rows


def cvalue(row: dict[str, str], prefix: str) -> complex:
    return complex(float(row[f"{prefix}_re"]), float(row[f"{prefix}_im"]))


def channel_samples(rows: Iterable[dict[str, str]], family: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    features: list[tuple[complex, complex, complex]] = []
    targets: list[complex] = []
    labels: list[str] = []
    for row in rows:
        if row.get("family") != family or row.get("setting") != "production":
            continue
        if row.get("status", "ok") not in ("ok", "", "success"):
            continue
        try:
            ws = [cvalue(row, f"omega{i}") for i in (1, 2, 3)]
            total = sum(ws)
            ms = [cvalue(row, f"M{i}") for i in (1, 2, 3)]
        except (KeyError, ValueError):
            continue
        for i in range(3):
            rest = [ws[j] for j in range(3) if j != i]
            v = rest[0] * rest[1]
            if abs(v) < 1e-14:
                continue
            features.append((total, ws[i], v))
            targets.append(ms[i] / (math.pi * v))
            labels.append(f"{row.get('point_id','')}::M{i+1}")
    return np.asarray(features, dtype=complex), np.asarray(targets, dtype=complex), labels


def exponents(degree: int) -> list[tuple[int, int, int]]:
    return [
        (a, b, c)
        for a in range(degree + 1)
        for b in range(degree + 1 - a)
        for c in range(degree + 1 - a - b)
    ]


def scale_features(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    center = x.mean(axis=0)
    scale = np.sqrt(np.mean(np.abs(x - center) ** 2, axis=0))
    scale[scale == 0] = 1
    return (x - center) / scale, center, scale


def design_matrix(x: np.ndarray, powers: Sequence[tuple[int, int, int]]) -> np.ndarray:
    columns = []
    for a, b, c in powers:
        columns.append((x[:, 0] ** a) * (x[:, 1] ** b) * (x[:, 2] ** c))
    return np.column_stack(columns)


def metrics(y: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    residual = prediction - y
    denom = np.maximum(np.abs(y), 1e-14)
    return {
        "count": int(len(y)),
        "rms_abs": float(np.sqrt(np.mean(np.abs(residual) ** 2))),
        "max_abs": float(np.max(np.abs(residual))),
        "median_relative": float(np.median(np.abs(residual) / denom)),
        "max_relative": float(np.max(np.abs(residual) / denom)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", help="CSV files or glob patterns")
    parser.add_argument("--degree", type=int, default=6)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--train-family", default="core")
    parser.add_argument("--test-family", default="structured")
    args = parser.parse_args()

    rows = load_rows(args.inputs)
    x_train, y_train, labels_train = channel_samples(rows, args.train_family)
    x_test, y_test, labels_test = channel_samples(rows, args.test_family)
    if len(y_train) == 0:
        raise SystemExit("no usable training samples found")

    x_scaled, center, scale = scale_features(x_train)
    powers = exponents(args.degree)
    matrix = design_matrix(x_scaled, powers)
    coefficients, *_ = np.linalg.lstsq(matrix, y_train, rcond=None)
    train_prediction = matrix @ coefficients

    output = {
        "degree": args.degree,
        "number_of_coefficients": len(powers),
        "variables": ["S", "p", "v"],
        "scaling": {
            "center": [[float(z.real), float(z.imag)] for z in center],
            "scale": [float(s) for s in scale],
        },
        "coefficients": [
            {
                "powers": list(power),
                "real": float(coef.real),
                "imag": float(coef.imag),
            }
            for power, coef in zip(powers, coefficients)
        ],
        "train": metrics(y_train, train_prediction),
    }

    if len(y_test):
        x_test_scaled = (x_test - center) / scale
        test_prediction = design_matrix(x_test_scaled, powers) @ coefficients
        output["test"] = metrics(y_test, test_prediction)
    else:
        output["test"] = {"count": 0}

    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"train": output["train"], "test": output["test"]}, indent=2))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
