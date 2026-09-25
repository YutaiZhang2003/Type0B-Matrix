#!/usr/bin/env python3
"""Compare all three Type 0B sphere block families in their local form frames.

The C++ probes sew their own SCA Ward forms and inverse PBW Gram matrices.
Their ordered form frames are converted to Yutai's native sphere form frame
using phases fixed by the local form and Ramond ground-metric definitions.
The separate Yutai PBW and double-Virasoro implementations are also compared.
"""

from __future__ import annotations

from itertools import product
import json
from pathlib import Path
import subprocess
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
CPP = ROOT / "C++"
SPHERE = (ROOT / "Packages/local_amplitude_conformal_blocks/source/project/"
          "Data Set/type0b_rr_sphere_20260921/het_inputs/Codes")
sys.path.insert(0, str(SPHERE))
from literature_component_blocks import ComponentBlocks  # noqa: E402
from literature_double_virasoro import LiteratureDoubleVirasoroBlocks  # noqa: E402


def compare_probe(family: str, name: str) -> dict[str, float | int]:
    output = subprocess.run(
        [str(CPP / "bin" / name)], capture_output=True, text=True,
        check=True).stdout
    reference = ComponentBlocks(
        family, (0.22, 0.27, 0.31, 0.37), 0.43,
        maximum_twice_level=4,
    )
    maximum = 0.0
    maximum_raw = 0.0
    count = 0
    for line in output.splitlines():
        fields = line.split()
        if family == "mixed_ns":
            level, a1, a2, a3, a4, right, real, imaginary = fields
            level, a1, a2, a3, a4, right = map(
                int, (level, a1, a2, a3, a4, right))
            k, left, exponent = level % 2, 1, level
            # NSWard::phase for the left all-NS vertex.
            phase = (-1) ** (k * (1 + a3 + a4))
        elif family == "mixed_r":
            level, k, a1, a2, a3, a4, left, right, real, imaginary = fields
            level, k, a1, a2, a3, a4, left, right = map(
                int, (level, k, a1, a2, a3, a4, left, right))
            exponent = 2 * level
            # R--R--NS inversion and the Ramond bra-metric conversion.
            phase = 1j ** (k - a1) * (-1) ** (k * a1)
        else:
            level, a1, a2, a3, a4, left, right, real, imaginary = fields
            level, a1, a2, a3, a4, left, right = map(
                int, (level, a1, a2, a3, a4, left, right))
            k, exponent = level % 2, level
            # R--R--NS inversion with an NS internal edge.
            phase = (-1j) ** k * (-1) ** (a4 * k)
        ordered = complex(float(real), float(imaginary))
        expected = reference.coefficients(
            (a1, a2, a3, a4), k, left, right)[exponent]
        maximum = max(maximum, float(abs(phase * ordered - expected)))
        maximum_raw = max(maximum_raw, float(abs(ordered - expected)))
        count += 1
    return {
        "components": count,
        "maximum_absolute_error_in_common_form_frame": maximum,
        "maximum_raw_difference_between_form_frames": maximum_raw,
    }


def compare_yutai_methods(family: str) -> dict[str, float | int]:
    kwargs = {
        "family": family,
        "momenta": (0.22, 0.27, 0.31, 0.37),
        "P": 0.43,
        "maximum_twice_level": 4,
    }
    pbw = ComponentBlocks(**kwargs)
    double_virasoro = LiteratureDoubleVirasoroBlocks(**kwargs)
    count = 0
    maximum = 0.0
    for external in product((0, 1), repeat=4):
        for parity in (0, 1):
            for left in ((1,) if family == "mixed_ns" else (-1, 1)):
                for right in (-1, 1):
                    a = pbw.coefficients(external, parity, left, right)
                    b = double_virasoro.coefficients(
                        external, parity, left, right)
                    for value, reference in zip(a, b):
                        maximum = max(maximum, float(abs(value - reference)))
                        count += 1
    return {"components": count, "maximum_absolute_error": maximum}


def compare_frozen_bank(family: str, name: str) -> dict[str, float | int]:
    """Compare one physical c=27/2 production node, before amplitude sewing."""
    base = SPHERE.parent
    tasks = json.loads((base / "native/manifest.json").read_text())["bank_tasks"]
    index, task = next(
        (index, task) for index, task in enumerate(tasks)
        if task["energy"] == "t0250" and task["family"] == family
        and task["momentum_index"] == 16
    )
    record = json.loads((base / "native/banks" / f"{index:04d}.json").read_text())
    pairs = np.asarray(record["payload"]["coefficients"], float)
    bank = pairs[..., 0] + 1j * pairs[..., 1]
    external_bits = tuple(
        bits for bits in product((0, 1), repeat=4)
        if bits[1] == 0 and (family == "mixed_ns" or bits[2] == 0)
    )
    signs = tuple(product(
        (1,) if family == "mixed_ns" else (1, -1), (1, -1)
    ))
    momenta = [complex(*pair) for pair in task["momenta"]]
    momenta.append(complex(task["P"]))
    args = [str(CPP / "bin" / name), "13.5"]
    for momentum in momenta:
        args.extend((str(momentum.real), str(momentum.imag)))
    output = subprocess.run(args, capture_output=True, text=True, check=True).stdout
    maximum = 0.0
    count = 0
    for line in output.splitlines():
        fields = line.split()
        if family == "mixed_ns":
            level, a1, a2, a3, a4, right = map(int, fields[:6])
            k, left, exponent = level % 2, 1, level
            phase = (-1) ** (k * (1 + a3 + a4))
            real, imaginary = map(float, fields[6:])
        elif family == "mixed_r":
            level, k, a1, a2, a3, a4, left, right = map(int, fields[:8])
            exponent = 2 * level
            phase = 1j ** (k - a1) * (-1) ** (k * a1)
            real, imaginary = map(float, fields[8:])
        else:
            level, a1, a2, a3, a4, left, right = map(int, fields[:7])
            k, exponent = level % 2, level
            phase = (-1j) ** k * (-1) ** (a4 * k)
            real, imaginary = map(float, fields[7:])
        external = (a1, a2, a3, a4)
        if external not in external_bits:
            continue
        expected = bank[0, external_bits.index(external), k,
                        signs.index((left, right)), exponent]
        maximum = max(maximum, float(abs(phase * complex(real, imaginary)
                                     - expected)))
        count += 1
    return {"bank_task": index, "components": count,
            "maximum_absolute_error": maximum}


def main() -> None:
    names = {
        "mixed_ns": "sphere_mixed_ns_probe",
        "mixed_r": "sphere_mixed_r_probe",
        "rrrr": "sphere_rrrr_probe",
    }
    subprocess.run(
        ["make", "-C", str(CPP), *(f"bin/{name}" for name in names.values())],
        capture_output=True, check=True)
    result = {
        family: {
            "ours_vs_yutai_pbw": compare_probe(family, name),
            "yutai_pbw_vs_double_virasoro": compare_yutai_methods(family),
            "ours_vs_physical_frozen_bank": compare_frozen_bank(family, name),
        }
        for family, name in names.items()
    }
    print(json.dumps(result, indent=2))
    for family, row in result.items():
        direct = row["ours_vs_yutai_pbw"]
        production = row["yutai_pbw_vs_double_virasoro"]
        frozen = row["ours_vs_physical_frozen_bank"]
        expected = {"mixed_ns": (160, 320), "mixed_r": (384, 640),
                    "rrrr": (320, 640)}[family]
        frozen_count = {"mixed_ns": 80, "mixed_r": 96, "rrrr": 80}[family]
        if ((direct["components"], production["components"]) != expected
                or frozen["components"] != frozen_count
                or direct["maximum_absolute_error_in_common_form_frame"] >= 1e-12
                or direct["maximum_raw_difference_between_form_frames"] <= 1e-3
                or production["maximum_absolute_error"] >= 1e-12
                or frozen["maximum_absolute_error"] >= 1e-6):
            raise AssertionError(f"{family}: Type 0B sphere block mismatch")


if __name__ == "__main__":
    main()
