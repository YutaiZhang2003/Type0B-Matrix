#!/usr/bin/env python3
r"""Recompute genus-two ``Q_L`` with the direct plumbing free denominator.

The super-Liouville numerator is copied from a completed Human-Note-sign
rerun because changing the physical free-theory denominator does not change
any conformal-block or momentum-quadrature node.  The denominator is freshly
evaluated at every requested mode cutoff from charged-boson and physical
Majorana pants sewing.  No period matrix, theta constant, or auxiliary
double-Virasoro fermion is used.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from typing import Sequence

from genus_2.physical_free_plumbing_resummation import (
    physical_superfield_plumbing_partition,
)


DEFAULT_SOURCE = Path(
    "Data Set/ns_genus2_theta_glasses_hatc9_human_note_2026-08-25.json"
)
DEFAULT_OUTPUT = Path(
    "Data Set/ns_genus2_theta_glasses_hatc9_plumbing_free_2026-08-25.json"
)


def _json_default(value: object) -> object:
    if isinstance(value, complex):
        return [float(value.real), float(value.imag)]
    raise TypeError(f"cannot serialize {type(value).__name__}")


def run(argv: Sequence[str] | None = None) -> dict[str, object]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--mode-cutoffs",
        type=int,
        nargs="+",
        default=(4, 6, 8, 10, 12, 16, 20, 24),
    )
    args = parser.parse_args(argv)

    cutoffs = tuple(sorted(set(int(value) for value in args.mode_cutoffs)))
    if not cutoffs or cutoffs[0] < 1:
        raise ValueError("positive mode cutoffs are required")
    source = json.loads(args.source.read_text())
    config_path = Path(source["config"])
    config = json.loads(config_path.read_text())
    point_id = str(source["point_id"])
    point = next(
        (row for row in config["points"] if str(row["id"]) == point_id),
        None,
    )
    if point is None:
        raise ValueError(f"point {point_id!r} is absent from {config_path}")

    q_values = {
        channel: tuple(complex(value) for value in point["q_values"][channel])
        for channel in ("theta", "glasses")
    }
    lifts = {
        channel: tuple(int(value) for value in config["physical_lifts"][channel])
        for channel in ("theta", "glasses")
    }
    convergence = []
    final_free = {}
    for cutoff in cutoffs:
        row = {"max_mode": cutoff, "channels": {}}
        for channel in ("theta", "glasses"):
            partition = physical_superfield_plumbing_partition(
                channel,
                q_values[channel],
                lifts[channel],
                max_mode=cutoff,
            )
            row["channels"][channel] = asdict(partition)
            if cutoff == cutoffs[-1]:
                final_free[channel] = partition
        convergence.append(row)

    results = []
    for old in source["results"]:
        channel = old["channel"]
        free_value = final_free[channel].one_superfield_value
        q_l = float(old["value"] / free_value**9)
        row = dict(old)
        row.update(
            {
                "free_superfield": free_value,
                "q_l": q_l,
                "previous_period_matrix_free_superfield": old["free_superfield"],
                "previous_period_matrix_q_l": old["q_l"],
                "new_over_previous_q_l": q_l / old["q_l"],
            }
        )
        results.append(row)

    comparisons = []
    settings = sorted(
        {
            (int(row["recursion_order"]), int(row["quadrature_order"]))
            for row in results
        }
    )
    for recursion_order, quadrature_order in settings:
        pair = {
            row["channel"]: row
            for row in results
            if row["recursion_order"] == recursion_order
            and row["quadrature_order"] == quadrature_order
        }
        ratio = pair["theta"]["q_l"] / pair["glasses"]["q_l"]
        comparisons.append(
            {
                "recursion_order": recursion_order,
                "quadrature_order": quadrature_order,
                "theta_over_glasses": ratio,
                "relative_difference": ratio - 1.0,
            }
        )

    denominator_comparison = {}
    for channel in ("theta", "glasses"):
        old_free = next(
            float(row["free_superfield"])
            for row in source["results"]
            if row["channel"] == channel
        )
        new_free = final_free[channel].one_superfield_value
        denominator_comparison[channel] = {
            "direct_plumbing_one_superfield": new_free,
            "previous_period_matrix_one_superfield": old_free,
            "direct_over_previous": new_free / old_free,
            "ql_multiplier": (old_free / new_free) ** 9,
        }

    output = {
        "schema": "ns-genus2-ql-direct-plumbing-free-v1",
        "quantity": "Q_L = Z_L / (Z_(X+psi)^pl)^9",
        "source_numerator_result": str(args.source),
        "source_config": str(config_path),
        "point_id": point_id,
        "numerator_recomputed": False,
        "numerator_reuse_reason": (
            "The physical free denominator is independent of the "
            "super-Liouville conformal-block evaluation."
        ),
        "free_denominator": {
            "formula": "G_X^pl |P_X^pl|^2 |F_psi^pl|^2",
            "physical_fields": (
                "one noncompact real scalar plus its physical NS Majorana"
            ),
            "loop_measure": "d alpha_1 d alpha_2, h(alpha)=alpha^2/2",
            "human_note_descendant_sign_resummed": True,
            "period_matrix_used": False,
            "riemann_theta_constant_used": False,
            "auxiliary_double_virasoro_fermion_used": False,
            "mode_cutoffs": list(cutoffs),
        },
        "physical_lifts": {
            channel: list(value) for channel, value in lifts.items()
        },
        "convergence": convergence,
        "denominator_comparison": denominator_comparison,
        "results": results,
        "comparisons": comparisons,
        "interpretation": (
            "The direct genus-two charge Gaussian supplies the explicit "
            "2^(-g/2)=1/2 factor appropriate to h(alpha)=alpha^2/2 at g=2. "
            "The previous det(Im Omega)^(-1/2) shortcut omitted that measure "
            "factor in the displayed convention, so absolute Q_L increases "
            "by approximately 2^9 while the channel ratio is nearly unchanged."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, default=_json_default) + "\n"
    )
    print(json.dumps(output, indent=2, default=_json_default))
    return output


if __name__ == "__main__":
    run()
