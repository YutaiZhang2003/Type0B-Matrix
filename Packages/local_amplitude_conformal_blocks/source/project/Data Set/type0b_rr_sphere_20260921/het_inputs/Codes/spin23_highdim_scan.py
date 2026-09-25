#!/usr/bin/env python3
"""High-dimensional scan design and SLURM-array runner for the Spin(23)
heterotic tree-level 1->3 continuum amplitude.

The script does two jobs:

1. ``generate`` writes deterministic CSV manifests for the scan proposed in
   the accompanying answer.  The primary scan has independent complex
   outgoing energies, not only equal-energy/common-phase kinematics.
2. ``run`` evaluates one manifest row by importing a user-supplied evaluator
   module.  The evaluator must expose

       evaluate_point(omega1, omega2, omega3, settings) -> mapping

   where the mapping contains complex channel coefficients ``M1``, ``M2``,
   and ``M3`` (or pairs ``M1_re``, ``M1_im``, etc.).  Channel Mi multiplies
   delta^(a4 ai) delta^(aj ak), with {i,j,k}={1,2,3}.

The manifests are designed to stay in a simple no-pole-crossing domain for
an undeformed real internal-Liouville-momentum contour.  The scan generator
uses SciPy's scrambled Sobol sequence with fixed seeds, so every point is
exactly reproducible.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import math
import os
import shlex
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from itertools import permutations
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy.stats import qmc


SEED = 220105621
CODE_DIR = Path(__file__).resolve().parent
DATA_DIR = CODE_DIR.parent / "Data Set"


@dataclass(frozen=True)
class Point:
    point_id: str
    family: str
    setting: str
    omega1: complex
    omega2: complex
    omega3: complex
    parent_id: str = ""
    permutation: str = "012"
    note: str = ""

    @property
    def omega(self) -> complex:
        return self.omega1 + self.omega2 + self.omega3

    def row(self) -> dict[str, Any]:
        total = self.omega
        return {
            "point_id": self.point_id,
            "family": self.family,
            "setting": self.setting,
            "parent_id": self.parent_id,
            "permutation": self.permutation,
            "note": self.note,
            "omega1_re": format(self.omega1.real, ".17g"),
            "omega1_im": format(self.omega1.imag, ".17g"),
            "omega2_re": format(self.omega2.real, ".17g"),
            "omega2_im": format(self.omega2.imag, ".17g"),
            "omega3_re": format(self.omega3.real, ".17g"),
            "omega3_im": format(self.omega3.imag, ".17g"),
            "omega_re": format(total.real, ".17g"),
            "omega_im": format(total.imag, ".17g"),
            "pole_margin": format(pole_margin(self.omega1, self.omega2, self.omega3), ".17g"),
            "max_pair_re_square": format(max_pair_re_square(self.omega1, self.omega2, self.omega3), ".17g"),
        }


CSV_FIELDS = [
    "point_id",
    "family",
    "setting",
    "parent_id",
    "permutation",
    "note",
    "omega1_re",
    "omega1_im",
    "omega2_re",
    "omega2_im",
    "omega3_re",
    "omega3_im",
    "omega_re",
    "omega_im",
    "pole_margin",
    "max_pair_re_square",
]


def pole_margin(w1: complex, w2: complex, w3: complex) -> float:
    """Distance to the nearest *first* C-structure-constant crossing estimate.

    For positive imaginary parts the relevant sufficient condition is

        Im(omega1+omega2+omega3) + max_i Im(omegai) < 1.

    A positive return value is the margin to one.  This is deliberately a
    conservative diagnostic, not a proof that no more distant singularity is
    relevant.
    """
    ims = np.array([w1.imag, w2.imag, w3.imag], dtype=float)
    return float(1.0 - (ims.sum() + ims.max()))


def max_pair_re_square(w1: complex, w2: complex, w3: complex) -> float:
    values = [((w1 + w2) ** 2).real, ((w1 + w3) ** 2).real, ((w2 + w3) ** 2).real]
    return float(max(values))


def point_id(family: str, index: int, ws: Sequence[complex], setting: str) -> str:
    payload = family + setting + "|" + "|".join(f"{w.real:.17g},{w.imag:.17g}" for w in ws)
    digest = hashlib.blake2b(payload.encode(), digest_size=5).hexdigest()
    return f"{family}-{index:05d}-{digest}"


def sobol_points(d: int, m: int, seed: int) -> np.ndarray:
    return qmc.Sobol(d=d, scramble=True, seed=seed).random_base2(m=m)


def core_points() -> list[Point]:
    """4096 generic points in three independent complex energies."""
    unit = sobol_points(d=6, m=12, seed=SEED)
    points: list[Point] = []
    for i, u in enumerate(unit):
        re = 0.05 + 0.25 * u[:3]
        im = 0.12 + 0.10 * u[3:]
        ws = tuple(complex(re[j], im[j]) for j in range(3))
        points.append(
            Point(
                point_id("core", i, ws, "production"),
                "core",
                "production",
                ws[0], ws[1], ws[2],
                note="6D scrambled Sobol core",
            )
        )
    return points


def chebyshev_lobatto(n: int, lower: float, upper: float) -> np.ndarray:
    if n < 2:
        raise ValueError("n must be at least 2")
    k = np.arange(n)
    raw = np.cos(np.pi * k / (n - 1))
    values = (lower + upper) / 2 + (upper - lower) * raw / 2
    return np.sort(values)


def ordered_barycentric_lattice(total: int, minimum: int) -> list[tuple[int, int, int]]:
    triples: list[tuple[int, int, int]] = []
    for n1 in range(minimum, total + 1):
        for n2 in range(n1, total + 1):
            n3 = total - n1 - n2
            if n3 < n2 or n3 < minimum:
                continue
            triples.append((n1, n2, n3))
    return triples


def structured_points() -> list[Point]:
    """1480 common-phase points resolving total energy and simplex shape."""
    total_re = chebyshev_lobatto(8, 0.18, 0.90)
    total_im = chebyshev_lobatto(5, 0.34, 0.48)
    shapes = ordered_barycentric_lattice(total=30, minimum=4)
    assert len(shapes) == 37

    points: list[Point] = []
    i = 0
    for er in total_re:
        for ai in total_im:
            total_energy = complex(float(er), float(ai))
            for n1, n2, n3 in shapes:
                fractions = (n1 / 30, n2 / 30, n3 / 30)
                ws = tuple(x * total_energy for x in fractions)
                points.append(
                    Point(
                        point_id("structured", i, ws, "production"),
                        "structured",
                        "production",
                        ws[0], ws[1], ws[2],
                        note=f"common phase; fractions={n1}/30,{n2}/30,{n3}/30",
                    )
                )
                i += 1
    assert len(points) == 1480
    return points


def soft_points() -> list[Point]:
    """288 dedicated soft-limit points."""
    unit = sobol_points(d=4, m=4, seed=SEED + 1)  # 16 hard pairs
    scales = [1 / 2, 1 / 4, 1 / 8, 1 / 16, 1 / 32, 1 / 64]
    soft_direction = complex(0.10, 0.14)
    points: list[Point] = []
    i = 0
    for hard_index, u in enumerate(unit):
        hard_re = 0.10 + 0.15 * u[:2]
        hard_im = 0.14 + 0.06 * u[2:]
        hard = [complex(hard_re[0], hard_im[0]), complex(hard_re[1], hard_im[1])]
        for soft_leg in range(3):
            for scale in scales:
                ws: list[complex] = []
                hard_cursor = 0
                for leg in range(3):
                    if leg == soft_leg:
                        ws.append(scale * soft_direction)
                    else:
                        ws.append(hard[hard_cursor])
                        hard_cursor += 1
                points.append(
                    Point(
                        point_id("soft", i, ws, "high"),
                        "soft",
                        "high",
                        ws[0], ws[1], ws[2],
                        note=f"hard_pair={hard_index}; soft_leg={soft_leg + 1}; scale={scale:.8g}",
                    )
                )
                i += 1
    assert len(points) == 288
    return points


def near_real_points() -> list[Point]:
    """320 points approaching the positive-real energy region from +i epsilon."""
    unit = sobol_points(d=3, m=6, seed=SEED + 2)  # 64 real triples
    epsilons = [0.10, 0.075, 0.05, 0.03, 0.02]
    points: list[Point] = []
    i = 0
    for ray_index, u in enumerate(unit):
        re = 0.05 + 0.25 * u
        for eps in epsilons:
            ws = tuple(complex(float(re[j]), eps) for j in range(3))
            points.append(
                Point(
                    point_id("near-real", i, ws, "high"),
                    "near-real",
                    "high",
                    ws[0], ws[1], ws[2],
                    note=f"ray={ray_index}; common_epsilon={eps:.8g}",
                )
            )
            i += 1
    assert len(points) == 320
    return points


def symmetry_points(core: Sequence[Point]) -> list[Point]:
    """192 explicit permutation reruns for an implementation-level check."""
    indices = np.linspace(0, len(core) - 1, 32, dtype=int)
    perms = list(permutations(range(3)))
    points: list[Point] = []
    i = 0
    for idx in indices:
        parent = core[int(idx)]
        ws0 = [parent.omega1, parent.omega2, parent.omega3]
        for perm in perms:
            ws = tuple(ws0[j] for j in perm)
            tag = "".join(str(j) for j in perm)
            points.append(
                Point(
                    point_id("symmetry", i, ws, "production"),
                    "symmetry",
                    "production",
                    ws[0], ws[1], ws[2],
                    parent_id=parent.point_id,
                    permutation=tag,
                    note="explicit outgoing-leg permutation",
                )
            )
            i += 1
    assert len(points) == 192
    return points


def convergence_points(core: Sequence[Point]) -> list[Point]:
    """512 repeated jobs probing four independent numerical controls."""
    indices = np.linspace(0, len(core) - 1, 128, dtype=int)
    variants = ["q_plus2", "p_extended", "ope_radius_low", "quadrature_double"]
    points: list[Point] = []
    i = 0
    for idx in indices:
        parent = core[int(idx)]
        ws = (parent.omega1, parent.omega2, parent.omega3)
        for variant in variants:
            points.append(
                Point(
                    point_id("convergence", i, ws, variant),
                    "convergence",
                    variant,
                    ws[0], ws[1], ws[2],
                    parent_id=parent.point_id,
                    note=f"single-control variation: {variant}",
                )
            )
            i += 1
    assert len(points) == 512
    return points


def settings_table() -> dict[str, dict[str, Any]]:
    """Fresh weighted-threshold controls; targets are not accuracy certificates."""
    table = {
        "production": {
            "mp_dps": 80,
            "q_order": 10,
            "p_max": 6.0,
            "p_nodes": 96,
            "z_radial_nodes": 48,
            "z_angular_nodes": 96,
            "ope_radius": 0.02,
            "relative_target": 3e-5,
        },
        "high": {
            "mp_dps": 100,
            "q_order": 12,
            "p_max": 8.0,
            "p_nodes": 144,
            "z_radial_nodes": 72,
            "z_angular_nodes": 144,
            "ope_radius": 0.015,
            "relative_target": 1e-5,
        },
        "q_plus2": {
            "mp_dps": 90,
            "q_order": 12,
            "p_max": 6.0,
            "p_nodes": 96,
            "z_radial_nodes": 48,
            "z_angular_nodes": 96,
            "ope_radius": 0.02,
        },
        "p_extended": {
            "mp_dps": 90,
            "q_order": 10,
            "p_max": 8.0,
            "p_nodes": 144,
            "z_radial_nodes": 48,
            "z_angular_nodes": 96,
            "ope_radius": 0.02,
        },
        "ope_radius_low": {
            "mp_dps": 90,
            "q_order": 10,
            "p_max": 6.0,
            "p_nodes": 96,
            "z_radial_nodes": 56,
            "z_angular_nodes": 112,
            "ope_radius": 0.015,
        },
        "quadrature_double": {
            "mp_dps": 90,
            "q_order": 10,
            "p_max": 6.0,
            "p_nodes": 192,
            "z_radial_nodes": 96,
            "z_angular_nodes": 192,
            "ope_radius": 0.02,
        },
    }
    for name, controls in table.items():
        controls.pop("p_max")
        controls["series_parameter"] = "elliptic_nome"
        controls["momentum_scheme"] = "threshold_weighted"
        controls["momentum_threshold_options"] = {
            "endpoint": 0.18, "tail": 4.5 if name == "p_extended" else 3.2,
            "beta": 2.0, "a": 1.0, "s": 0.0,
        }
    return table


def write_manifest(path: Path, points: Sequence[Point]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for point in points:
            writer.writerow(point.row())


def generate(outdir: Path) -> dict[str, int]:
    outdir.mkdir(parents=True, exist_ok=True)
    core = core_points()
    families = {
        "core": core,
        "structured": structured_points(),
        "soft": soft_points(),
        "near_real": near_real_points(),
        "symmetry": symmetry_points(core),
        "convergence": convergence_points(core),
    }

    counts: dict[str, int] = {}
    all_points: list[Point] = []
    for name, points in families.items():
        path = outdir / f"spin23_scan_{name}_{len(points)}.csv"
        write_manifest(path, points)
        counts[name] = len(points)
        all_points.extend(points)

    primary_points = [
        point
        for name, points in families.items()
        if name != "convergence"
        for point in points
    ]
    primary_path = outdir / f"spin23_scan_primary_{len(primary_points)}.csv"
    write_manifest(primary_path, primary_points)
    counts["primary"] = len(primary_points)

    all_path = outdir / f"spin23_scan_all_{len(all_points)}.csv"
    write_manifest(all_path, all_points)
    counts["all"] = len(all_points)

    plan = {
        "seed": SEED,
        "counts": counts,
        "core_domain": {
            "Re_omega_i": [0.05, 0.30],
            "Im_omega_i": [0.12, 0.22],
            "dimension_real": 6,
            "sampling": "4096-point scrambled Sobol sequence",
        },
        "structured_domain": {
            "Re_total": [0.18, 0.90],
            "Im_total": [0.34, 0.48],
            "total_nodes": "8x5 Chebyshev-Lobatto",
            "shape_nodes": "ordered n_i/30, n_i>=4, sum n_i=30 (37 points)",
        },
        "settings": settings_table(),
        "channel_convention": {
            "M1": "delta^(a4 a1) delta^(a2 a3)",
            "M2": "delta^(a4 a2) delta^(a1 a3)",
            "M3": "delta^(a4 a3) delta^(a1 a2)",
            "reduced_Ri": "Mi/(pi*omega_j*omega_k)",
        },
        "required_outputs": [
            "M1_re", "M1_im", "M2_re", "M2_im", "M3_re", "M3_im",
            "estimated_abs_error_M1", "estimated_abs_error_M2", "estimated_abs_error_M3",
            "runtime_seconds", "status",
        ],
    }
    (outdir / "spin23_scan_plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    return counts


def read_row(path: Path, index: int) -> dict[str, str]:
    if index < 0:
        raise ValueError("index must be nonnegative")
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for i, row in enumerate(reader):
            if i == index:
                return row
    raise IndexError(f"manifest index {index} is out of range")


def row_energies(row: Mapping[str, str]) -> tuple[complex, complex, complex]:
    return tuple(
        complex(float(row[f"omega{i}_re"]), float(row[f"omega{i}_im"]))
        for i in (1, 2, 3)
    )  # type: ignore[return-value]


def normalize_result(result: Mapping[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for name in ("M1", "M2", "M3"):
        if name in result:
            value = complex(result[name])
            output[f"{name}_re"] = value.real
            output[f"{name}_im"] = value.imag
        elif f"{name}_re" in result and f"{name}_im" in result:
            output[f"{name}_re"] = float(result[f"{name}_re"])
            output[f"{name}_im"] = float(result[f"{name}_im"])
        else:
            raise KeyError(f"evaluator result is missing {name}")
    for key, value in result.items():
        if key not in output and key not in ("M1", "M2", "M3"):
            output[key] = value
    return output


def append_result(path: Path, row: Mapping[str, str], result: Mapping[str, Any]) -> None:
    normalized = normalize_result(result)
    combined = dict(row)
    combined.update({key: str(value) for key, value in normalized.items()})
    fields = list(combined)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if write_header:
            writer.writeheader()
        writer.writerow(combined)


def run_one(manifest: Path, index: int, evaluator_module: str, output: Path, settings_path: Path | None) -> None:
    row = read_row(manifest, index)
    w1, w2, w3 = row_energies(row)

    if settings_path is None:
        settings_all = settings_table()
    else:
        settings_all = json.loads(settings_path.read_text())
    setting_name = row["setting"]
    if setting_name not in settings_all:
        raise KeyError(f"unknown setting {setting_name!r}")
    settings = dict(settings_all[setting_name])

    module = importlib.import_module(evaluator_module)
    evaluator = getattr(module, "evaluate_point", None)
    if evaluator is None:
        raise AttributeError(
            f"{evaluator_module!r} must define evaluate_point(omega1, omega2, omega3, settings)"
        )

    started = time.perf_counter()
    result = dict(evaluator(w1, w2, w3, settings))
    result.setdefault("runtime_seconds", time.perf_counter() - started)
    result.setdefault("status", "ok")
    append_result(output, row, result)


def print_slurm(manifest: Path, evaluator_module: str, output_dir: Path) -> None:
    with manifest.open() as handle:
        count = sum(1 for _ in handle) - 1
    max_index = count - 1
    print("#!/bin/bash")
    print(f"#SBATCH --array=0-{max_index}")
    print("#SBATCH --cpus-per-task=1")
    print("#SBATCH --time=24:00:00")
    print("#SBATCH --mem=8G")
    print("set -euo pipefail")
    print(f"mkdir -p {shlex.quote(str(output_dir))}")
    print(
        "python spin23_highdim_scan.py run "
        f"--manifest {shlex.quote(str(manifest))} --index $SLURM_ARRAY_TASK_ID "
        f"--evaluator {shlex.quote(evaluator_module)} "
        f"--output {shlex.quote(str(output_dir))}/result_${{SLURM_ARRAY_TASK_ID}}.csv"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_generate = sub.add_parser("generate", help="write all deterministic scan manifests")
    p_generate.add_argument(
        "--outdir",
        type=Path,
        default=DATA_DIR / "spin23_scan_manifests",
    )

    p_run = sub.add_parser("run", help="evaluate one manifest row")
    p_run.add_argument("--manifest", type=Path, required=True)
    p_run.add_argument("--index", type=int, required=True)
    p_run.add_argument("--evaluator", required=True, help="Python module containing evaluate_point")
    p_run.add_argument("--output", type=Path, required=True)
    p_run.add_argument("--settings-json", type=Path, default=None)

    p_slurm = sub.add_parser("slurm", help="print a SLURM array script")
    p_slurm.add_argument("--manifest", type=Path, required=True)
    p_slurm.add_argument("--evaluator", required=True)
    p_slurm.add_argument("--output-dir", type=Path, default=DATA_DIR / "results")

    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "generate":
        counts = generate(args.outdir)
        print(json.dumps(counts, indent=2))
    elif args.command == "run":
        run_one(args.manifest, args.index, args.evaluator, args.output, args.settings_json)
    elif args.command == "slurm":
        print_slurm(args.manifest, args.evaluator, args.output_dir)
    else:
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
