#!/usr/bin/env python3
"""Check BPZ-native open-edge torus coefficients against Yutai's RR block.

The torus code uses the normalized local Ward and contravariant metric
frame. The BPZ sewing of the two remaining edges gives i^(k-r_parity),
where k is the NS twice-level parity. No theta-channel closure is used.
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import sympy as sp


ROOT = Path(__file__).resolve().parents[2]
CODE = ROOT / "Packages/local_amplitude_conformal_blocks/source/project/Code"
sys.path.insert(0, str(CODE / "type0b_rr_genus1"))
from mixed_blocks import MixedNSRamondPlumbingBlock, RamondState  # noqa: E402


def main() -> None:
    subprocess.run(["make", "-C", str(ROOT / "C++"),
                    "bin/torus_open_bpz_probe"],
                   capture_output=True, check=True)
    rows = subprocess.run([str(ROOT / "C++/bin/torus_open_bpz_probe")],
                          capture_output=True, text=True, check=True).stdout
    block = MixedNSRamondPlumbingBlock(
        p_ns=sp.Rational(22, 100), p_r=sp.Rational(27, 100),
        omega=sp.Rational(31, 100), b=sp.Integer(1))
    maximum = 0.0
    maximum_raw = 0.0
    count = 0
    for line in rows.splitlines():
        fields = line.split()
        n, r, parity, form, eta, gl, gr, external_mode = map(int, fields[:8])
        if n > 2 or r > 1:
            continue
        left = RamondState((("G", -1),) if external_mode == 1 else (), gl)
        right = RamondState((("G", -1),) if external_mode == 2 else (), gr)
        reference = block.coefficient(
            n, r, left, right, forms=(form, form), etas=(eta, eta),
            r_parity=parity)
        actual = complex(float(fields[8]), float(fields[9]))
        expected = 1j ** (n % 2 - parity) * reference
        maximum = max(maximum, abs(actual - expected))
        maximum_raw = max(maximum_raw, abs(actual - reference))
        count += 1
    result = {
        "coefficients": count,
        "NS_twice_level_maximum": 2,
        "R_level_maximum": 1,
        "external_states": "ground or one G_{-1} insertion",
        "maximum_absolute_error_in_BPZ_sewing_frame": maximum,
        "maximum_raw_difference_between_frames": maximum_raw,
    }
    print(json.dumps(result, indent=2))
    if count != 576 or maximum >= 1e-10:
        raise AssertionError("BPZ-native torus open block disagrees")


if __name__ == "__main__":
    main()
