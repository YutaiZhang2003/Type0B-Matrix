#!/usr/bin/env python3
"""Compare the direct free-Majorana theta factor with a local theta oracle."""

from __future__ import annotations

from fractions import Fraction
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
CPP = ROOT / "C++"
CODE = ROOT / "Packages/local_amplitude_conformal_blocks/source/project/Code"
sys.path[:0] = [str(CODE / "double_virasoro/nsrr"),
                str(CODE / "c_Recursion")]
from nsrr_genus2_block import auxiliary_majorana_nsrr_series  # noqa: E402


def main() -> None:
    subprocess.run(["make", "-C", str(CPP), "bin/auxiliary_dump"],
                   capture_output=True, check=True)
    output = subprocess.run([str(CPP / "bin/auxiliary_dump")],
                            capture_output=True, text=True, check=True).stdout
    native = {}
    for line in output.splitlines():
        a, b, c, parity, value = line.split()
        native[(int(a), int(b), int(c), int(parity))] = complex(Fraction(value))
    copied = {}
    for levels, row in auxiliary_majorana_nsrr_series(
            maximum_total_twice_level=4).items():
        for parity, value in enumerate(row):
            if abs(value) > 1e-14:
                copied[(*levels, parity)] = complex(value)
    maximum = max(abs(native.get(key, 0) - copied.get(key, 0))
                  for key in native.keys() | copied.keys())
    result = {
        "nonzero_native_components": len(native),
        "nonzero_oracle_components": len(copied),
        "maximum_absolute_error": maximum,
    }
    print(json.dumps(result, indent=2))
    if len(native) != len(copied) or maximum >= 1e-12:
        raise AssertionError("auxiliary Majorana theta factors disagree")


if __name__ == "__main__":
    main()
