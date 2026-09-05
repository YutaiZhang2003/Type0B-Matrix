#!/usr/bin/env python3
"""Prepare ten separate nonsymmetric genus-two NSRR/NSNSNS surfaces."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import nsrr_human_note_geometry as human_geometry
import nsrr_nsnsns_theta_omega_scan as scan
import recompute_all_ns_reference as all_ns
import run_nsrr_nsnsns_offaxis_constant_scan as runner
from fixed_spin_free_plumbing import fixed_spin_partition


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "Data Set" / "nsrr_nsnsns_generic_10point_N3_20260904"

# Entries are (Re Omega11, Re Omega22, Re Omega12,
#              Im Omega11, Im Omega22, Im Omega12).
# They were selected deterministically from two six-dimensional Latin-hypercube
# designs after imposing Im Omega>0, successful inverse plumbing in both
# markings, max|q_e|<0.12, and a common source plumbing sheet.
GENERIC_SURFACES = (
    ("generic_01", (-0.078132, -0.003029, 0.614087, 0.977692, 0.970738, 0.566950)),
    ("generic_02", (0.021921, -0.050840, 0.553118, 0.945604, 1.070333, 0.502228)),
    ("generic_03", (0.033931, -0.033277, 0.585867, 1.060259, 0.911846, 0.410825)),
    ("generic_04", (0.000618, -0.011005, 0.593715, 0.967258, 1.143518, 0.433776)),
    ("generic_05", (-0.012873, -0.013559, 0.580176, 0.913420, 0.921231, 0.497233)),
    ("generic_06", (-0.075882, -0.062788, 0.560592, 1.030649, 0.991406, 0.417189)),
    ("generic_07", (-0.000510, -0.039681, 0.540569, 0.950665, 1.089626, 0.477065)),
    ("generic_08", (-0.066200, -0.067798, 0.519777, 0.943646, 1.042490, 0.484838)),
    ("generic_09", (-0.024405, 0.008553, 0.599199, 0.934096, 1.011286, 0.477544)),
    ("generic_10", (-0.004924, 0.009837, 0.609197, 1.049956, 1.097666, 0.459401)),
)


def period_matrix(values) -> np.ndarray:
    re11, re22, re12, im11, im22, im12 = values
    return np.asarray(
        [
            [re11 + 1j * im11, re12 + 1j * im12],
            [re12 + 1j * im12, re22 + 1j * im22],
        ],
        dtype=complex,
    )


def prepare(
    geometry_path: Path,
    orders: tuple[int, ...],
    *,
    point_ids: tuple[str, ...] | None = None,
    q_envelope_config: Path | None = None,
) -> dict:
    baseline = runner.load(geometry_path)
    center = next(point for point in baseline["points"] if float(point["t"]) == 0.6)
    source_seed = tuple(complex(value) for value in center["source_chart"]["q_values"])
    target_seed = tuple(complex(value) for value in center["target_chart"]["q_values"])
    omega_center = np.asarray([[1j, 0.6 + 0.5j], [0.6 + 0.5j, 1j]], dtype=complex)
    source_center_omega = human_geometry.action(human_geometry.SOURCE_REMARKING, omega_center)
    target_center_omega = scan.omega_action(omega_center)
    source_center = scan.inverse_chart(source_center_omega, source_seed)
    target_center = scan.inverse_chart(target_center_omega, target_seed)
    reference_source_branch = runner.charge_period_branch(
        tuple(complex(value) for value in source_center["q_values"]), source_center_omega
    )
    reference_target_branch = runner.charge_period_branch(
        tuple(complex(value) for value in target_center["q_values"]), target_center_omega
    )

    surfaces = GENERIC_SURFACES
    if point_ids is not None:
        known = {point_id for point_id, _ in GENERIC_SURFACES}
        if len(set(point_ids)) != len(point_ids) or not set(point_ids) <= known:
            raise ValueError("point_ids must be unique members of the generic-ten design")
        selected = set(point_ids)
        surfaces = tuple(item for item in GENERIC_SURFACES if item[0] in selected)
    points = []
    for point_id, coordinates in surfaces:
        omega = period_matrix(coordinates)
        if np.linalg.eigvalsh(omega.imag)[0] <= 0:
            raise ValueError(f"{point_id}: period matrix is outside Siegel space")
        source_omega = human_geometry.action(human_geometry.SOURCE_REMARKING, omega)
        target_omega = scan.omega_action(omega)
        source_chart = scan.inverse_chart(source_omega, source_seed)
        target_chart = scan.inverse_chart(target_omega, target_seed)
        source_q = tuple(complex(value) for value in source_chart["q_values"])
        target_q = tuple(complex(value) for value in target_chart["q_values"])
        source_branch = runner.charge_period_branch(source_q, source_omega)
        target_branch = runner.charge_period_branch(target_q, target_omega)
        if source_branch != reference_source_branch:
            raise ArithmeticError(f"{point_id}: source plumbing sheet changed")
        if max(map(abs, source_q)) >= 0.12 or max(map(abs, target_q)) >= 0.12:
            raise ArithmeticError(f"{point_id}: plumbing multiplier exceeds 0.12")
        target_lifts = runner.continued_target_lifts(
            target_q, reference_target_branch, target_branch
        )
        source_free = fixed_spin_partition(
            source_q,
            source_omega,
            runner.SOURCE_SPIN,
            period_branch=source_branch,
            max_mode=32,
        )
        target_free = fixed_spin_partition(
            target_q,
            target_omega,
            runner.TARGET_SPIN,
            period_branch=target_branch,
            max_mode=32,
        )
        points.append(
            {
                "point_id": point_id,
                "omega_coordinates": {
                    name: float(value)
                    for name, value in zip(
                        ("re11", "re22", "re12", "im11", "im22", "im12"),
                        coordinates,
                    )
                },
                "omega_reference": runner.encoded_matrix(omega),
                "minimum_imaginary_eigenvalue": float(np.linalg.eigvalsh(omega.imag)[0]),
                "source": {
                    **source_chart,
                    "characteristic": runner.SOURCE_SPIN,
                    "period_branch": source_branch,
                    "Z_free": source_free["Z_free"],
                },
                "target": {
                    **target_chart,
                    "characteristic": runner.TARGET_SPIN,
                    "period_branch": target_branch,
                    "principal_lifts_at_reference": runner.TARGET_LIFTS,
                    "lifts": target_lifts,
                    "Z_free": target_free["Z_free"],
                },
            }
        )
        print(
            f"prepared {point_id}: min eig(Im Omega)={points[-1]['minimum_imaginary_eigenvalue']:.4f}, "
            f"max|q| source={max(map(abs, source_q)):.5f}, target={max(map(abs, target_q)):.5f}, "
            f"target lifts={target_lifts}",
            flush=True,
        )

    b = 1.4
    config = {
        "schema": runner.SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "point_design": {
            "name": "generic-ten-v1" if point_ids is None else "generic-ten-selected-refinement-v1",
            "point_ids": [point_id for point_id, _ in surfaces],
            "selection": (
                "deterministic six-dimensional Latin-hypercube candidates, filtered by "
                "Im Omega>0, inverse-chart validation, max|q_e|<0.12, and common source sheet"
            ),
        },
        "b": b,
        "kappa": 1.0 + 2.0 * (b + 1.0 / b) ** 2,
        "orders": list(orders),
        "source_level": 3.0,
        "target_recursion_twice_level": 16,
        "source_contraction": (
            "amplitude-level [11|00] projection followed by the unscaled Human M kernel"
        ),
        "normalization_policy": (
            "fixed by sewing: test the raw ratio source_Q/target_Q against exactly one; "
            "do not fit or apply an overall constant"
        ),
        "analytic_continuation": {
            "reference_point": "Omega_*=[[i,0.6+0.5i],[0.6+0.5i,i]]",
            "reference_target_period_branch": reference_target_branch,
            "reference_target_lifts": runner.TARGET_LIFTS,
            "rule": "principal-frame beta shifts by diag(B-B_reference) mod 2",
        },
        "source_to_target": (
            scan.MATRIX
            @ np.rint(np.linalg.inv(human_geometry.SOURCE_REMARKING)).astype(int)
        ).tolist(),
        "points": points,
        "q_envelope": {
            channel: [
                max(abs(complex(point[channel]["q_values"][edge])) for point in points)
                for edge in range(3)
            ]
            for channel in ("source", "target")
        },
        "protected_kernel_hashes": all_ns.protected_hashes(),
    }
    if q_envelope_config is not None:
        envelope_parent = runner.load(q_envelope_config)
        config["q_envelope"] = envelope_parent["q_envelope"]
        config["q_envelope_parent"] = str(q_envelope_config.resolve())
    runner.validate_config(config)
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry", type=Path, default=runner.DEFAULT_GEOMETRY)
    parser.add_argument("--orders", type=int, nargs="+", default=[3])
    parser.add_argument("--point-ids", nargs="+")
    parser.add_argument("--q-envelope-config", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    config = prepare(
        args.geometry,
        tuple(sorted(set(args.orders))),
        point_ids=tuple(args.point_ids) if args.point_ids else None,
        q_envelope_config=args.q_envelope_config,
    )
    runner.save(args.output_dir / "config.json", config)


if __name__ == "__main__":
    main()
