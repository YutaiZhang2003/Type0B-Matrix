#!/usr/bin/env python3
"""Recombine saved NSRR chiral shards with the physical fixed-spin sewing.

No conformal-block node is recomputed.  The program verifies the saved shard
configuration digests, projects the two source lifts selecting [11|00],
applies the physical even/odd interference, and compares to the already
computed all-NS numerator using the independently fixed-spin free factors.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Iterable

from physical_nsrr_sewing import (
    CHANNELS,
    SOURCE_FIXED_SPIN_LIFTS,
    contract_physical_blocks,
    project_source_fixed_spin,
)


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
CODE_ROOT = HERE.parent
for directory in (CODE_ROOT / "c_Recursion", CODE_ROOT / "full_ramond_block_runtime"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from generic_super_liouville_structure_constants import (  # noqa: E402
    GenericSuperLiouvilleConstants,
)
from nsrr_plumbing_adapter import GEOMETRY_SECTORS, NSRRPlumbingInputs  # noqa: E402

SCHEMA = "nsrr-nsnsns-normalized-sewing-recheck-v4"
DEFAULT_L3 = ROOT / "Data Set" / "nsrr_factorized_sign_trial_L3_N5_20260830"
DEFAULT_L5 = ROOT / "Data Set" / "nsrr_trial_L5_N3_local_20260830"
DEFAULT_TARGET = ROOT / "Data Set" / "nsrr_nsnsns_target_R8_R12_R16_N5_20260830" / "summary.json"
DEFAULT_FREE = ROOT / "Data Set" / "fixed_spin_free_NSrr_20260830" / "summary.json"
DEFAULT_CENTRAL_REFINEMENT = ROOT / "Data Set" / "nsrr_spin_quadrature_t060_20260830"
DEFAULT_OUTPUT = ROOT / "Data Set" / "nsrr_nsnsns_normalized_sewing_recheck_20260903"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def digest_object(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def digest_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def encode(value: complex) -> list[float]:
    value = complex(value)
    return [float(value.real), float(value.imag)]


def decode(value) -> complex:
    if isinstance(value, str):
        return complex(value)
    if len(value) != 2:
        raise ValueError("encoded complex value must have two entries")
    return complex(value[0], value[1])


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _expected_node_count(config: dict) -> int:
    return sum(int(order) ** 3 for order in config["quadrature_orders"])


def _validate_source_directory(directory: Path) -> tuple[dict, list[dict], dict]:
    summary_path = directory / "summary.json"
    summary = load(summary_path)
    config = summary["config"]
    if config["channels"] != [list(channel) for channel in CHANNELS]:
        raise ValueError(f"{directory}: saved channel order has changed")
    required_lifts = {list(lift).__repr__() for lift in SOURCE_FIXED_SPIN_LIFTS}
    saved_lifts = {list(lift).__repr__() for lift in config["lifts_geometry"]}
    if not required_lifts <= saved_lifts:
        raise ValueError(f"{directory}: fixed-spin source lifts are missing")
    config_digest = digest_object(config)
    shard_paths = sorted((directory / "shards").glob("node-*.json"))
    if len(shard_paths) != _expected_node_count(config):
        raise ValueError(f"{directory}: incomplete saved shard set")
    shards = []
    manifest = []
    for expected_index, path in enumerate(shard_paths):
        shard = load(path)
        if shard["index"] != expected_index or shard["config_digest"] != config_digest:
            raise ValueError(f"{path}: shard index or configuration digest mismatch")
        if len(shard["rows"]) != len(config["points"]) * len(config["lifts_geometry"]) * len(config["levels"]):
            raise ValueError(f"{path}: incomplete row design")
        shards.append(shard)
        manifest.append({"path": str(path.resolve()), "sha256": digest_file(path)})
    provenance = {
        "directory": str(directory.resolve()),
        "summary_path": str(summary_path.resolve()),
        "summary_sha256": digest_file(summary_path),
        "config_digest": config_digest,
        "shard_count": len(shards),
        "shards": manifest,
    }
    return config, shards, provenance


def _row_index(shard: dict) -> dict[tuple[float, float, tuple[int, int, int]], dict]:
    result = {}
    for row in shard["rows"]:
        key = (float(row["t"]), float(row["level"]), tuple(row["lifts_geometry"]))
        if key in result:
            raise ValueError("duplicate t/level/lift row in source shard")
        result[key] = row
    return result


def _projected_node(shard: dict, t: float, level: float) -> dict:
    rows = _row_index(shard)
    amplitudes = {}
    primaries = {}
    for lift in SOURCE_FIXED_SPIN_LIFTS:
        row = rows[t, level, lift]
        primary = decode(row["primary"])
        primaries[lift] = primary
        amplitudes[lift] = {
            channel: primary * decode(row["blocks"][index])
            for index, channel in enumerate(CHANNELS)
        }
    primary_spread = abs(primaries[SOURCE_FIXED_SPIN_LIFTS[0]] - primaries[SOURCE_FIXED_SPIN_LIFTS[1]])
    projected = project_source_fixed_spin(amplitudes)
    contracted = contract_physical_blocks(projected, tuple(decode(value) for value in shard["C_BRY"]))
    contracted["primary_lift_spread"] = primary_spread
    return contracted


def reduce_source(config: dict, shards: Iterable[dict]) -> list[dict]:
    shards = list(shards)
    rows = []
    for order in config["quadrature_orders"]:
        selected = [shard for shard in shards if int(shard["quadrature_order"]) == int(order)]
        if len(selected) != int(order) ** 3:
            raise ValueError(f"quadrature order {order}: missing source nodes")
        for point in config["points"]:
            t = float(point["t"])
            for level in config["levels"]:
                values = [_projected_node(shard, t, float(level)) for shard in selected]
                total = math.fsum(shard["measure"] * value["total"] for shard, value in zip(selected, values))
                diagonal = math.fsum(shard["measure"] * value["diagonal"] for shard, value in zip(selected, values))
                interference = math.fsum(shard["measure"] * value["interference"] for shard, value in zip(selected, values))
                rows.append(
                    {
                        "t": t,
                        "level": float(level),
                        "quadrature_order": int(order),
                        "source_Z": total,
                        "diagonal_Z": diagonal,
                        "interference_Z": interference,
                        "opposite_interference_sign_Z": diagonal - interference,
                        "interference_fraction": interference / total,
                        "maximum_primary_lift_spread": max(value["primary_lift_spread"] for value in values),
                        "maximum_structure_product_imaginary_part": max(
                            value["maximum_coefficient_imaginary_part"] for value in values
                        ),
                    }
                )
    return rows


def _free_lookup(free_summary: dict) -> dict[float, dict[str, float]]:
    return {
        float(point["t"]): {
            "source": float(point["source_NSrr"]["Z_free"]),
            "target": float(point["target_NSnsns"]["Z_free"]),
        }
        for point in free_summary["points"]
    }


def _target_lookup(target_summary: dict) -> dict[float, dict]:
    rows = [
        row for row in target_summary["rows"]
        if int(row["quadrature_order"]) == 5 and int(row["recursion_order"]) == 16
    ]
    result = {float(row["t"]): row for row in rows}
    if len(result) != 5:
        raise ValueError("the target R16/N5 reference is incomplete")
    return result


def _select(rows: list[dict], *, order: int, level: float) -> dict[float, dict]:
    selected = {
        row["t"]: row for row in rows
        if row["quadrature_order"] == order and row["level"] == level
    }
    if len(selected) != 5:
        raise ValueError(f"source L{level:g}/N{order} selection is incomplete")
    return selected


def reduce_central_refinement(
    directory: Path,
    *,
    free: dict[float, dict[str, float]],
    kappa: float,
) -> dict:
    """Reuse the saved source N6 blocks and target N7 integral at t=0.60."""

    config_path = directory / "config.json"
    quadrature_path = directory / "quadrature_summary.json"
    config = load(config_path)
    quadrature = load(quadrature_path)
    config_digest = digest_object(config)
    source_config = config["source"]
    point = next(point for point in source_config["points"] if float(point["t"]) == 0.6)
    q_geometry = tuple(complex(value) for value in point["q_geometry"])
    b = float(source_config["b"])
    constants = GenericSuperLiouvilleConstants(b, dps=30)
    paths = sorted((directory / "shards").glob("source-N6-node-*.json"))
    if len(paths) != 6**3:
        raise ValueError("the saved source N6 refinement is incomplete")
    weighted = []
    archived_diagonal = []
    maximum_primary_spread = 0.0
    for path in paths:
        shard = load(path)
        if shard["config_digest"] != config_digest or shard["quadrature_order"] != 6:
            raise ValueError(f"{path}: central-refinement provenance mismatch")
        momenta = tuple(float(value) for value in shard["momenta"])
        bry = constants.rr_ns_constants(momenta[1], momenta[0], momenta[2])
        plumbing = NSRRPlumbingInputs(q_geometry, (1, 1, 1), GEOMETRY_SECTORS)
        primary = plumbing.primary(b, momenta)
        amplitudes = {
            tuple(row["lifts"]): {
                channel: primary * decode(row["blocks"][index])
                for index, channel in enumerate(CHANNELS)
            }
            for row in shard["rows"]
        }
        projected = project_source_fixed_spin(amplitudes)
        contraction = contract_physical_blocks(projected, bry)
        weighted.append(float(shard["measure"]) * contraction["total"])
        archived_diagonal.append(float(shard["rows"][0]["Z_weighted"]))
    source_z = math.fsum(weighted)
    archived_z = math.fsum(archived_diagonal)
    archived_reference = next(
        row for row in quadrature["rows"]
        if row["channel"] == "source" and int(row["N"]) == 6
    )
    target_reference = next(
        row for row in quadrature["rows"]
        if row["channel"] == "target" and int(row["N"]) == 7
    )
    reproduction_error = archived_z / float(archived_reference["Z"]) - 1.0
    if abs(reproduction_error) > 2.0e-14:
        raise ArithmeticError("the saved source N6 diagonal integral was not reproduced")
    source_q = source_z / free[0.6]["source"] ** kappa
    target_z = float(target_reference["Z"])
    target_q = target_z / free[0.6]["target"] ** kappa
    return {
        "t": 0.6,
        "source_quadrature_order": 6,
        "source_level": 3.0,
        "source_Z_physical": source_z,
        "source_Q_physical": source_q,
        "target_quadrature_order": 7,
        "target_recursion_order": 16,
        "target_Z_reused": target_z,
        "target_Q_fixed_free": target_q,
        "source_over_target": source_q / target_q,
        "relative_difference": source_q / target_q - 1.0,
        "archived_diagonal_Z_reproduced": archived_z,
        "archived_diagonal_relative_reproduction_error": reproduction_error,
        "source_shards_reused": len(paths),
        "new_block_nodes": 0,
        "provenance": {
            "directory": str(directory.resolve()),
            "config_path": str(config_path.resolve()),
            "config_sha256": digest_file(config_path),
            "config_digest": config_digest,
            "quadrature_summary_path": str(quadrature_path.resolve()),
            "quadrature_summary_sha256": digest_file(quadrature_path),
        },
    }


def recombine(
    l3_dir: Path,
    l5_dir: Path,
    target_path: Path,
    free_path: Path,
    central_refinement_dir: Path | None = None,
) -> dict:
    l3_config, l3_shards, l3_provenance = _validate_source_directory(l3_dir)
    l5_config, l5_shards, l5_provenance = _validate_source_directory(l5_dir)
    if float(l3_config["b"]) != float(l5_config["b"]):
        raise ValueError("source datasets use different Liouville couplings")
    l3_rows = reduce_source(l3_config, l3_shards)
    l5_rows = reduce_source(l5_config, l5_shards)
    target_summary = load(target_path)
    free_summary = load(free_path)
    target = _target_lookup(target_summary)
    free = _free_lookup(free_summary)
    kappa = 1.0 + 2.0 * (float(l3_config["b"]) + 1.0 / float(l3_config["b"])) ** 2
    l3n5 = _select(l3_rows, order=5, level=3.0)
    l3n4 = _select(l3_rows, order=4, level=3.0)
    l2n5 = _select(l3_rows, order=5, level=2.0)
    l5n3 = _select(l5_rows, order=3, level=5.0)
    l4n3 = _select(l5_rows, order=3, level=4.0)
    comparisons = []
    for t in sorted(target):
        source_best = l3n5[t]
        source_q = source_best["source_Z"] / free[t]["source"] ** kappa
        source_l5n3_q = l5n3[t]["source_Z"] / free[t]["source"] ** kappa
        target_z = float(target[t]["target_Z"])
        target_q = target_z / free[t]["target"] ** kappa
        comparisons.append(
            {
                "t": t,
                "source_Z_L3_N5": source_best["source_Z"],
                "source_Q_L3_N5": source_q,
                "source_Z_L5_N3": l5n3[t]["source_Z"],
                "source_Q_L5_N3": source_l5n3_q,
                "target_Z_R16_N5_reused": target_z,
                "target_Q_R16_N5_fixed_free": target_q,
                "source_over_target_L3_N5": source_q / target_q,
                "relative_difference_L3_N5": source_q / target_q - 1.0,
                "source_over_target_L5_N3": source_l5n3_q / target_q,
                "relative_difference_L5_N3": source_l5n3_q / target_q - 1.0,
                "source_interference_fraction_L3_N5": source_best["interference_fraction"],
                "diagonal_only_relative_difference_L3_N5": (
                    source_best["diagonal_Z"] / free[t]["source"] ** kappa / target_q - 1.0
                ),
                "opposite_interference_sign_relative_difference_L3_N5": (
                    source_best["opposite_interference_sign_Z"]
                    / free[t]["source"] ** kappa
                    / target_q
                    - 1.0
                ),
                "wrong_direct_BRY_as_HJS_relative_difference_L3_N5": (
                    4.0 * source_q / target_q - 1.0
                ),
                "unscaled_kernel_counterfactual_relative_difference_L3_N5": (
                    4.0 * source_q / target_q - 1.0
                ),
                "source_level_L2_to_L3_relative_change_N5": source_best["source_Z"] / l2n5[t]["source_Z"] - 1.0,
                "source_level_L4_to_L5_relative_change_N3": l5n3[t]["source_Z"] / l4n3[t]["source_Z"] - 1.0,
                "source_quadrature_N4_to_N5_relative_change_L3": source_best["source_Z"] / l3n4[t]["source_Z"] - 1.0,
            }
        )
    result = {
        "schema": SCHEMA,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete_recombination_of_saved_nodes",
        "b": float(l3_config["b"]),
        "kappa": kappa,
        "physical_conventions": {
            "source_marked_spin": "[11|00]",
            "source_fixed_spin_projection_geometry_lifts": "(F_[+,+,+]+F_[+,-,+])/sqrt(2)",
            "BRY_to_HJS": "c_+=C_even/2, c_-=C_odd/2",
            "form_parity_kernel": "K_etaeta'=(1/4)[[1,-i eta eta'],[i eta eta',1]]",
            "physical_chiral_combination": "(1/4)|F_0+i eta_left eta_right F_1|^2",
            "normalization": (
                "fixed by normalized physical Ramond completeness and by the "
                "identity-trinion ground sewing 1^2+1^2=2"
            ),
        },
        "reuse": {
            "source_L3": l3_provenance,
            "source_L5": l5_provenance,
            "target": {"path": str(target_path.resolve()), "sha256": digest_file(target_path)},
            "fixed_spin_free": {"path": str(free_path.resolve()), "sha256": digest_file(free_path)},
            "new_block_nodes": 0,
        },
        "source_rows_L3": l3_rows,
        "source_rows_L5": l5_rows,
        "comparisons": comparisons,
        "checks": {
            "maximum_primary_lift_spread": max(
                row["maximum_primary_lift_spread"] for row in l3_rows + l5_rows
            ),
            "maximum_structure_product_imaginary_part": max(
                row["maximum_structure_product_imaginary_part"] for row in l3_rows + l5_rows
            ),
            "target_R16_N5_rows": 5,
        },
        "interpretation": (
            "This follows the Human Note's Ramond block and three-point "
            "factorization while retaining the normalized nonchiral Ramond "
            "completeness kernel.  The relative +i interference and its 1/4 "
            "normalization pass the independent identity-trinion check.  Every "
            "available chiral node is reused."
        ),
    }
    if central_refinement_dir is not None:
        result["central_refinement"] = reduce_central_refinement(
            central_refinement_dir, free=free, kappa=kappa
        )
        result["reuse"]["central_refinement"] = result["central_refinement"]["provenance"]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-l3", type=Path, default=DEFAULT_L3)
    parser.add_argument("--source-l5", type=Path, default=DEFAULT_L5)
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    parser.add_argument("--fixed-free", type=Path, default=DEFAULT_FREE)
    parser.add_argument(
        "--central-refinement",
        type=Path,
        help="optional saved t=0.60 source-N6/target-N7 directory",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = recombine(
        args.source_l3,
        args.source_l5,
        args.target,
        args.fixed_free,
        args.central_refinement,
    )
    write_json(args.output_dir / "summary.json", result)
    with (args.output_dir / "comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(result["comparisons"][0]))
        writer.writeheader()
        writer.writerows(result["comparisons"])
    for row in result["comparisons"]:
        print(
            f"t={row['t']:.2f}  source/target(L3,N5)={row['source_over_target_L3_N5']:.12f} "
            f"relative={row['relative_difference_L3_N5']:+.3e}"
        )
    if "central_refinement" in result:
        row = result["central_refinement"]
        print(
            f"t=0.60  refined source-N6/target-N7={row['source_over_target']:.12f} "
            f"relative={row['relative_difference']:+.3e}"
        )


if __name__ == "__main__":
    main()
