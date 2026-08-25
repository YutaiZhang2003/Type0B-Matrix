#!/usr/bin/env python3
"""Run a blind ten-point torus three-point scan on the imaginary-energy ray.

The equal-split kinematics are ``omega=i*t`` and
``omega_1=omega_2=i*t/2``.  Every point lies strictly in ``0<t<1`` and uses
the same randomized Sobol scrambles.  This module contains no comparison
curve or target coefficient; it only drives and validates the worldsheet
integrator and freezes a manifest after all points have completed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

try:
    from integrate_genus1_three_point_worldsheet import run as run_one_point
except ImportError:  # pragma: no cover
    from plumbing.integrate_genus1_three_point_worldsheet import run as run_one_point


DEFAULT_T_VALUES = (0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95)
DEFAULT_OUTPUT_DIR = Path(
    "plumbing/results/genus1_three_point_worldsheet/"
    "equal_split_imaginary_t_scan10_p12_n256_v1"
)


@dataclass(frozen=True)
class ScanDesign:
    """Worldsheet-only numerical design fixed before starting the scan."""

    p_max: float = 5.0
    momentum_order: int = 12
    momentum_kind: str = "power-legendre"
    momentum_power: float = 2.0
    momentum_power_step: float = 0.137
    reference_log_q_abs: float = -1.9
    reference_log_q_step: float = 0.137
    high_order: int = 4
    low_order: int = 2
    block_backend: str = "exact-c25-descendants"
    block_tolerance: float = 5.0e-5
    c_regulator: float = 0.05
    cutoffs: str = "3,4,6,8"
    sobol_power: int = 8
    replicates: int = 4
    seed: int = 17051301
    tail_slices: str = "8,10,12,16,20"
    tail_sobol_power: int = 8
    dps: int = 28


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _complex(record: Mapping[str, object]) -> complex:
    return complex(float(record["real"]), float(record["imag"]))


def parse_t_values(text: str) -> tuple[float, ...]:
    values = tuple(float(piece) for piece in text.split(",") if piece.strip())
    if not values:
        raise ValueError("at least one value of t is required")
    if any(not 0.0 < value < 1.0 for value in values):
        raise ValueError("the direct pole-free equal-split scan requires 0<t<1")
    if any(right <= left for left, right in zip(values, values[1:])):
        raise ValueError("t values must be strictly increasing")
    return values


def point_tag(t: float) -> str:
    hundredths = int(round(100.0 * float(t)))
    if not math.isclose(float(t), hundredths / 100.0, abs_tol=1.0e-12):
        raise ValueError("the scan driver expects t values on a 0.01 grid")
    return f"t{hundredths:03d}"


def stored_point_path(output_dir: Path, t: float) -> Path:
    return output_dir / point_tag(t) / "worldsheet_blind.json"


def point_namespace(t: float, output: Path, design: ScanDesign) -> argparse.Namespace:
    return argparse.Namespace(t=float(t), output=str(output), **asdict(design))


def validate_frozen_record(
    record: Mapping[str, object],
    *,
    t: float,
    design: ScanDesign,
) -> None:
    """Fail closed if a point differs from the predeclared blind design."""
    if record.get("blind_freeze") is not True:
        raise ValueError(f"t={t:g}: record is not marked blind_freeze=true")
    if record.get("calculation") != "direct c=1 genus-one three-point worldsheet integral":
        raise ValueError(f"t={t:g}: wrong worldsheet calculation type")

    kinematics = record["kinematics"]
    if not math.isclose(float(kinematics["t"]), float(t), abs_tol=1.0e-14):
        raise ValueError(f"t={t:g}: stored kinematics do not match")
    if "0<t<1" not in str(kinematics["domain"]):
        raise ValueError(f"t={t:g}: point is not in the declared pole-free strip")
    expected_energies = (t, 0.5 * t, 0.5 * t)
    for name, expected in zip(
        ("omega_in", "omega_out_1", "omega_out_2"),
        expected_energies,
    ):
        energy = _complex(kinematics[name])
        if abs(energy - 1.0j * expected) > 1.0e-14:
            raise ValueError(f"t={t:g}: {name} changed")

    momentum = record["momentum_rule"]
    if momentum["kind"] != design.momentum_kind:
        raise ValueError(f"t={t:g}: momentum-rule kind changed")
    if int(momentum["order_per_edge"]) != design.momentum_order:
        raise ValueError(f"t={t:g}: momentum order changed")
    if not math.isclose(float(momentum["p_max"]), design.p_max, abs_tol=1.0e-14):
        raise ValueError(f"t={t:g}: momentum cutoff changed")
    expected_powers = [
        design.momentum_power + edge * design.momentum_power_step
        for edge in range(3)
    ]
    if any(
        not math.isclose(float(value), expected, abs_tol=1.0e-14)
        for value, expected in zip(momentum["powers_by_edge"], expected_powers)
    ):
        raise ValueError(f"t={t:g}: momentum maps changed")

    blocks = record["block_design"]
    expected_blocks = {
        "backend": design.block_backend,
        "high_edge_max_order": design.high_order,
        "other_edge_order": design.low_order,
    }
    for key, expected in expected_blocks.items():
        if blocks[key] != expected:
            raise ValueError(f"t={t:g}: block field {key!r} changed")
    if not math.isclose(
        float(blocks["adaptive_tail_proxy"]),
        design.block_tolerance,
        abs_tol=1.0e-16,
    ):
        raise ValueError(f"t={t:g}: adaptive block tolerance changed")

    rqmc = record["rqmc"]
    tail_rqmc = record["tail_rqmc"]
    expected_rqmc = {
        "sobol_power": design.sobol_power,
        "points_per_cutoff_replicate": 2**design.sobol_power,
        "replicates": design.replicates,
        "seed": design.seed,
    }
    for key, expected in expected_rqmc.items():
        if int(rqmc[key]) != int(expected):
            raise ValueError(f"t={t:g}: RQMC field {key!r} changed")
    expected_tail = {
        "sobol_power": design.tail_sobol_power,
        "points_per_slice_replicate": 2**design.tail_sobol_power,
        "replicates": design.replicates,
    }
    for key, expected in expected_tail.items():
        if int(tail_rqmc[key]) != int(expected):
            raise ValueError(f"t={t:g}: tail-RQMC field {key!r} changed")

    expected_cutoffs = [float(value) for value in design.cutoffs.split(",")]
    expected_slices = [float(value) for value in design.tail_slices.split(",")]
    if [float(value) for value in record["cutoffs"]] != expected_cutoffs:
        raise ValueError(f"t={t:g}: cutoff list changed")
    tail = record["tail_completion"]
    if [float(value) for value in tail["tau2_slices"]] != expected_slices:
        raise ValueError(f"t={t:g}: tail-slice list changed")
    if tail.get("fit_is_target_free") is not True:
        raise ValueError(f"t={t:g}: tail fit is not marked target-free")


def inspect_existing_scan(
    output_dir: Path,
    t_values: Sequence[float],
    design: ScanDesign,
) -> dict[str, object]:
    points: list[dict[str, object]] = []
    for t in t_values:
        path = stored_point_path(output_dir, t)
        if not path.is_file():
            raise FileNotFoundError(path)
        record = json.loads(path.read_text(encoding="utf-8"))
        validate_frozen_record(record, t=t, design=design)
        final_i = _complex(record["tail_completion"]["final_I"])
        final_se = _complex(record["tail_completion"]["final_rqmc_standard_error"])
        points.append(
            {
                "t": float(t),
                "path": str(path),
                "sha256": _sha256(path),
                "I_1,3": {"real": final_i.real, "imag": final_i.imag},
                "rqmc_standard_error": {
                    "real": abs(final_se.real),
                    "imag": abs(final_se.imag),
                },
                "replicate_finals": record["tail_completion"]["replicate_finals"],
                "tail_fraction": float(
                    abs(_complex(record["tail_completion"]["mean_integrated_tail"]))
                    / max(abs(final_i), 1.0e-300)
                ),
                "tail_fit_relative_residual": float(
                    record["tail_completion"]["mean_relative_fit_residual"]
                ),
                "fitted_tail_exponent": float(
                    record["tail_completion"]["fitted_exponent"]
                ),
                "largest_hat_q_seen": float(
                    record["block_diagnostics"]["largest_hat_q_seen"]
                ),
                "adaptive_order_histogram": record["block_diagnostics"][
                    "adaptive_order_histogram"
                ],
                "bulk_block_order_shift": record["cusp_fit"][
                    "last_retained_order_shift"
                ],
            }
        )
    return {
        "calculation": "blind genus-one three-point equal-split imaginary-energy scan",
        "comparison_stage_present": False,
        "pole_free_domain": "0<t<1 on the undeformed positive Liouville contours",
        "t_values": [float(value) for value in t_values],
        "common_scrambles_across_t": True,
        "design": asdict(design),
        "points": points,
    }


def run_scan(
    output_dir: Path,
    t_values: Sequence[float],
    design: ScanDesign,
) -> dict[str, object]:
    for index, t in enumerate(t_values):
        output = stored_point_path(output_dir, t)
        if output.exists():
            raise FileExistsError(
                f"refusing to overwrite frozen point {output}; use --check-existing "
                "or choose a new --output-dir"
            )
        print(f"starting blind point {index + 1}/{len(t_values)} at t={t:.2f}", flush=True)
        result = run_one_point(point_namespace(t, output, design))
        validate_frozen_record(result, t=t, design=design)

    manifest = inspect_existing_scan(output_dir, t_values, design)
    manifest_path = output_dir / "worldsheet_scan_manifest.json"
    _atomic_json(manifest_path, manifest)
    print(f"wrote {manifest_path}")
    return manifest


def parser() -> argparse.ArgumentParser:
    out = argparse.ArgumentParser(
        description="Run or validate the blind torus three-point imaginary-energy scan."
    )
    out.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    out.add_argument(
        "--t-values",
        default=",".join(f"{value:.2f}" for value in DEFAULT_T_VALUES),
    )
    out.add_argument("--check-existing", action="store_true")
    return out


def main() -> None:
    args = parser().parse_args()
    design = ScanDesign()
    t_values = parse_t_values(args.t_values)
    if args.check_existing:
        print(
            json.dumps(
                inspect_existing_scan(args.output_dir, t_values, design),
                indent=2,
                allow_nan=False,
            )
        )
        return
    run_scan(args.output_dir, t_values, design)


if __name__ == "__main__":
    main()
