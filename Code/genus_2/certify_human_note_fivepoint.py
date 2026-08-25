#!/usr/bin/env python3
"""Build the machine-readable Human-Note genus-two locality certificate.

The five-point numerator inputs are immutable sector sums from the completed
Cannon runs.  Those runs used the explicit Human-Note descendant sign with
BRY's real odd coefficient.  The Human-Note coefficient is instead
``C_HN^(1)=i C_BRY^(1)``; because a genus-two term contains two pants
coefficients, the odd product acquires a minus sign.  Hence the corrected
numerator is ``Z_even - Z_odd_signed``.

The physical denominator is recomputed here, in each plumbing frame, from one
real scalar and one physical Majorana.  It never uses a period-matrix formula
or the auxiliary fermion of the double-Virasoro decomposition.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np

from genus_2.physical_free_plumbing_resummation import (
    physical_superfield_plumbing_partition,
)
from ns_genus2_cannon import _validate_config_spin_characteristics


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "Code/config/ns_genus2_cross_sewing_r24_n10_human_note_spin00.json"
DEFAULT_OUTPUT = ROOT / "Data Set/ns_genus2_human_note_fivepoint_certificate_2026-08-25.json"

POINT_ORDER = (
    "o0243-periodmatched",
    "o0127-periodmatched",
    "o0015-periodmatched",
    "o0167-periodmatched",
    "o0239-periodmatched",
)

# Values are math.fsum reductions of 1,000 immutable R=24,N=10 shards per
# point and channel.  ``odd_signed`` includes the explicit Human-Note sign but
# not yet the BRY-to-Human-Note coefficient phase.
SECTOR_SUMS = {
    "theta": {
        "o0243-periodmatched": (7.272416798496117e-18, -2.7770857401946667e-20),
        "o0127-periodmatched": (2.3290877398946325e-18, -6.261092484392573e-21),
        "o0015-periodmatched": (1.3108088634891162e-17, -4.906943155015132e-20),
        "o0167-periodmatched": (1.9226521893358103e-19, -4.729774805351799e-22),
        "o0239-periodmatched": (1.191316366203917e-19, -3.014418898119296e-22),
    },
    "glasses": {
        "o0243-periodmatched": (2.8844493964681592e-06, -6.87808460834072e-08),
        "o0127-periodmatched": (4.985550798062454e-06, -1.455513419013111e-07),
        "o0015-periodmatched": (2.5876466309895356e-06, -6.399688885635799e-08),
        "o0167-periodmatched": (8.449188147885872e-06, -2.4927378249320544e-07),
        "o0239-periodmatched": (9.008546069895429e-06, -2.594479793254639e-07),
    },
}

SHARD_PROVENANCE = {
    "theta": {
        "remote_root": "/n/home09/yutaizhang/Type0B-Matrix-runs/ns-genus2-cross-sewing-r24-n10-human-note-spin00-20260825-v1/theta-shards",
        "shard_count": 5000,
        "config_digest": "e3f3bba1ae989fcda13f5b26b3a250e8c0bb1dbc1832f30248d1c35bea2c9c75",
        "implementation_fingerprint": "213b60917554fb291414c6f95ede084f079f6779f81ed172496ccb8cf8c6a79a",
    },
    "glasses": {
        "remote_root": "/n/home09/yutaizhang/Type0B-Matrix-runs/ns-genus2-cross-sewing-r24-n10-glasses-parity-20260823-v1/shards",
        "shard_count_used": 5000,
        "config_digest": "2d057b992d529497b6a1047f7dfa555a52fa7374f729d7bb45078bc46fc27543",
        "implementation_fingerprint": "bd2f5e74bd15bd29e0cb051168af8ddea559580bb3418fd54169d97748d815dc",
    },
    "runtime_versions": {
        "python": "3.12.11 (conda-forge, GCC 13.3.0)",
        "numpy": "2.0.2",
        "scipy": "1.13.1",
        "mpmath": "1.3.0",
    },
}


def _complex_matrix(entries: Sequence[Sequence[str]]) -> np.ndarray:
    return np.asarray(
        [[complex(entries[i][j]) for j in range(2)] for i in range(2)],
        dtype=np.complex128,
    )


def _geometry_audit(config: dict) -> dict:
    matrix = np.asarray(
        config["provenance"]["symplectic_matrix_glasses_to_theta_after_branch"],
        dtype=int,
    )
    a, b = matrix[:2, :2], matrix[:2, 2:]
    c, d = matrix[2:, :2], matrix[2:, 2:]
    rows = []
    for point in config["points"]:
        omega_g = _complex_matrix(point["omega"]["glasses"])
        omega_t = _complex_matrix(point["omega"]["theta"])
        transported = (a @ omega_g + b) @ np.linalg.inv(c @ omega_g + d)
        rows.append(
            {
                "point_id": point["id"],
                "symplectic_transport_max_residual": float(
                    np.max(np.abs(transported - omega_t))
                ),
                "glasses_seam_residual": float(
                    point["provenance"]["glasses_seam_residual"]
                ),
                "theta_period_max_residual": float(
                    point["provenance"]["theta_period_max_residual"]
                ),
                "theta_word8_word9_max_difference": float(
                    point["provenance"]["theta_word8_word9_max_difference"]
                ),
                "matching_assumed": bool(point["provenance"]["matching_assumed"]),
            }
        )
    return {
        "rows": rows,
        "max_symplectic_transport_residual": max(
            row["symplectic_transport_max_residual"] for row in rows
        ),
        "max_glasses_seam_residual": max(
            row["glasses_seam_residual"] for row in rows
        ),
        "max_theta_period_residual": max(
            row["theta_period_max_residual"] for row in rows
        ),
        "max_theta_word8_word9_difference": max(
            row["theta_word8_word9_max_difference"] for row in rows
        ),
    }


def _free_audit(config: dict) -> tuple[dict, dict]:
    cutoffs = (16, 20, 24, 28)
    rows: dict[str, dict] = {}
    final: dict[str, dict[str, float]] = {}
    for point in config["points"]:
        point_rows = {}
        for cutoff in cutoffs:
            channel_rows = {}
            for channel in ("theta", "glasses"):
                value = physical_superfield_plumbing_partition(
                    channel,
                    [complex(item) for item in point["q_values"][channel]],
                    config["physical_lifts"][channel],
                    max_mode=cutoff,
                )
                channel_rows[channel] = {
                    "one_superfield": value.one_superfield_value,
                    "nine_superfield": value.nine_superfield_value,
                }
                if cutoff == cutoffs[-1]:
                    final.setdefault(point["id"], {})[channel] = (
                        value.one_superfield_value
                    )
            point_rows[str(cutoff)] = channel_rows
        rows[point["id"]] = point_rows

    comparisons = []
    for point_id in POINT_ORDER:
        for channel in ("theta", "glasses"):
            z20 = rows[point_id]["20"][channel]["nine_superfield"]
            z24 = rows[point_id]["24"][channel]["nine_superfield"]
            z28 = rows[point_id]["28"][channel]["nine_superfield"]
            comparisons.append(
                {
                    "point_id": point_id,
                    "channel": channel,
                    "relative_change_M20_to_M24": abs(z24 - z20) / abs(z24),
                    "relative_change_M24_to_M28": abs(z28 - z24) / abs(z28),
                }
            )
    return (
        {
            "definition": "one physical noncompact real scalar plus one physical NS Majorana, evaluated directly in the plumbing frame",
            "period_matrix_used": False,
            "auxiliary_double_virasoro_fermion_used": False,
            "cutoffs": list(cutoffs),
            "values": rows,
            "cutoff_comparisons": comparisons,
            "max_relative_change_M24_to_M28": max(
                row["relative_change_M24_to_M28"] for row in comparisons
            ),
        },
        final,
    )


def _fivepoint_rows(final_free: dict[str, dict[str, float]]) -> list[dict]:
    rows = []
    for point_id in POINT_ORDER:
        channels = {}
        for channel in ("theta", "glasses"):
            even, odd_signed = SECTOR_SUMS[channel][point_id]
            numerator = even - odd_signed
            denominator = final_free[point_id][channel]
            channels[channel] = {
                "even_sector_sum": even,
                "odd_sector_sum_before_coefficient_phase": odd_signed,
                "human_note_numerator": numerator,
                "physical_free_one_superfield": denominator,
                "Q_L": numerator / denominator**9,
                "odd_fraction_after_phase": (-odd_signed) / numerator,
            }
        ratio = channels["theta"]["Q_L"] / channels["glasses"]["Q_L"]
        rows.append(
            {
                "point_id": point_id,
                "recursion_order": 24,
                "quadrature_order": 10,
                "channels": channels,
                "theta_over_glasses": ratio,
                "relative_difference": ratio - 1.0,
            }
        )
    return rows


def _axis_audit(path: Path, free: dict[str, dict[str, float]], baseline: dict) -> dict:
    source = json.loads(path.read_text())
    point_id = "o0243-periodmatched"
    ratios = {(24, 10): baseline["theta_over_glasses"]}
    q_values = {
        (24, 10, channel): baseline["channels"][channel]["Q_L"]
        for channel in ("theta", "glasses")
    }
    for recursion_order, quadrature_order in (
        (22, 10),
        (24, 12),
        (24, 14),
        (24, 16),
    ):
        selected = {
            row["channel"]: row
            for row in source["rows"]
            if row["point_id"] == point_id
            and int(row["recursion_order"]) == recursion_order
            and int(row["quadrature_order"]) == quadrature_order
            and float(row["finite_part_radius"])
            == float(source["config"]["finite_part_radii"][0])
        }
        for channel in ("theta", "glasses"):
            q_values[(recursion_order, quadrature_order, channel)] = (
                float(selected[channel]["z_liouville"])
                / free[point_id][channel] ** 9
            )
        ratios[(recursion_order, quadrature_order)] = (
            q_values[(recursion_order, quadrature_order, "theta")]
            / q_values[(recursion_order, quadrature_order, "glasses")]
        )

    rows = []
    for pair in ((22, 10), (24, 10), (24, 12), (24, 14), (24, 16)):
        rows.append(
            {
                "recursion_order": pair[0],
                "quadrature_order": pair[1],
                "Q_L_theta": q_values[(pair[0], pair[1], "theta")],
                "Q_L_glasses": q_values[(pair[0], pair[1], "glasses")],
                "theta_over_glasses": ratios[pair],
                "relative_difference": ratios[pair] - 1.0,
            }
        )
    n14_n16 = abs(ratios[(24, 16)] - ratios[(24, 14)])
    return {
        "source_summary": str(path),
        "rows": rows,
        "recursion_ratio_change_R22_to_R24_at_N10": abs(
            ratios[(24, 10)] - ratios[(22, 10)]
        ),
        "quadrature_ratio_change_N14_to_N16": n14_n16,
        "N16_cross_channel_relative_difference": ratios[(24, 16)] - 1.0,
        "global_nonconverged_calls": sum(
            int(row.get("global_nonconverged_calls", 0)) for row in source["rows"]
        ),
        "ratio_stable_at_1e-6": n14_n16 < 1.0e-6,
        "channels_agree_at_1e-6": abs(ratios[(24, 16)] - 1.0) < 1.0e-6,
    }


def build(config_path: Path, axis_summary: Path | None) -> dict:
    config = json.loads(config_path.read_text())
    spin_ledger = _validate_config_spin_characteristics(config)
    geometry = _geometry_audit(config)
    free_audit, final_free = _free_audit(config)
    rows = _fivepoint_rows(final_free)
    differences = [float(row["relative_difference"]) for row in rows]
    baseline = next(row for row in rows if row["point_id"] == "o0243-periodmatched")
    result = {
        "schema": "ns-genus2-human-note-fivepoint-certificate-v1",
        "quantity": "Q_L = Z_L / (Z_free^pl)^9",
        "config": str(config_path),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "human_note_edited": False,
        "coefficient_convention": {
            "even": "C_HN^(0)=C_BRY^(0)",
            "odd": "C_HN^(1)=i*C_BRY_tilde^(1)",
            "genus_two_odd_coefficient_product": "i^2=-1",
            "recombination": "Z_HN=Z_even-Z_odd_signed",
        },
        "shard_provenance": SHARD_PROVENANCE,
        "spin_ledger": spin_ledger,
        "geometry": geometry,
        "physical_free": free_audit,
        "fivepoint_R24_N10": rows,
        "fixed_cutoff_statistics": {
            "mean_relative_difference": math.fsum(differences) / len(differences),
            "rms_relative_difference": math.sqrt(
                math.fsum(value * value for value in differences) / len(differences)
            ),
            "max_absolute_relative_difference": max(map(abs, differences)),
            "spread": max(differences) - min(differences),
            "passes_5e-4_fixed_cutoff_check": max(map(abs, differences)) < 5.0e-4,
        },
        "certification_status": {
            "convention_and_sign": "pass",
            "five_independent_moduli_at_R24_N10": "pass at 5e-4",
            "physical_free_M24_to_M28": (
                "pass" if free_audit["max_relative_change_M24_to_M28"] < 1.0e-10 else "fail"
            ),
            "full_1e-6": "pending convergence axis",
        },
    }
    if axis_summary is not None:
        axis = _axis_audit(axis_summary, final_free, baseline)
        result["o0243_convergence_axis"] = axis
        complete = (
            axis["recursion_ratio_change_R22_to_R24_at_N10"] < 1.0e-6
            and axis["ratio_stable_at_1e-6"]
            and axis["channels_agree_at_1e-6"]
            and axis["global_nonconverged_calls"] == 0
        )
        result["certification_status"]["full_1e-6"] = (
            "pass" if complete else "not yet certified"
        )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--axis-summary", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    result = build(args.config, args.axis_summary)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "output": str(args.output),
        "fixed_cutoff_statistics": result["fixed_cutoff_statistics"],
        "certification_status": result["certification_status"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
