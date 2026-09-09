#!/usr/bin/env python3
r"""Rerun the current genus-two computation in the Human Note convention.

This driver is deliberately config-based.  A theta plumbing lift does not
determine a period-matrix characteristic without the integer-branch marking
of that period matrix, so historical overlap CSVs are not accepted here.

The numerator follows ``Human Notes/SCblock.tex``:

* theta: ``sum_a (-1)^(a+p1+p2+p3) C_a^2 |q^h F_a|^2``;
* glasses: ``sum_a (-1)^(a+p_bridge) C_LBL^a C_RBR^a |q^h F_a|^2``.

The denominator of ``Q_L`` is the physical free theory evaluated directly
in the plumbing frame,

``Z_(X+psi) = G_X^pl |P_X^pl|^2 |F_psi^pl|^2``

for one noncompact real scalar and one physical NS Majorana.  The auxiliary
Majorana used only for the double-Virasoro star quotient never enters this
driver or this denominator.  In particular, no period matrix or theta
constant is used to evaluate the free denominator.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

import numpy as np

from c_Recursion.ns_genus2_cannon import _validate_config_spin_characteristics
from c_Recursion.ns_genus2_partition import (
    C_ORDINARY_AT_HAT_C_9,
    HAT_C_TARGET,
    TYPE0B_NS_PRIMARY_PARITIES,
    _spin_characteristic_from_lifts,
    evaluate_channel,
    run_internal_checks,
)
from genus_2.physical_free_plumbing_resummation import (
    physical_superfield_plumbing_partition,
)


DEFAULT_CONFIG = Path(
    "Code/config/ns_genus2_cross_sewing_r24_n10_human_note_spin00.json"
)
DEFAULT_OUTPUT = Path(
    "Data Set/ns_genus2_theta_glasses_hatc9_human_note_plumbing_free_2026-08-25.json"
)


def _complex_matrix(rows: Sequence[Sequence[object]]) -> np.ndarray:
    return np.asarray(
        [[complex(value) for value in row] for row in rows],
        dtype=np.complex128,
    )


def _json_default(value: object) -> object:
    if isinstance(value, complex):
        return [float(value.real), float(value.imag)]
    raise TypeError(f"cannot serialize {type(value).__name__}")


def run(argv: Sequence[str] | None = None) -> dict[str, object]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--point-id")
    parser.add_argument(
        "--settings",
        nargs="+",
        default=("0:4", "3:4", "4:4", "4:6"),
        help="recursion:quadrature pairs",
    )
    parser.add_argument("--structure-precision", type=int, default=30)
    parser.add_argument("--global-tolerance", type=float, default=2.0e-9)
    parser.add_argument("--global-max-occupation", type=int, default=18)
    parser.add_argument("--vacuum-word-length", type=int, default=6)
    parser.add_argument("--vacuum-max-mode", type=int, default=36)
    parser.add_argument("--free-mode-cutoff", type=int, default=24)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    config = json.loads(args.config.read_text())
    spin_ledger = _validate_config_spin_characteristics(config)
    points = {str(point["id"]): point for point in config["points"]}
    point_id = args.point_id or next(iter(points))
    if point_id not in points:
        raise ValueError(f"unknown point id {point_id!r}")
    point = points[point_id]

    q_values = {
        channel: tuple(complex(value) for value in point["q_values"][channel])
        for channel in ("theta", "glasses")
    }
    omegas = {
        channel: _complex_matrix(point["omega"][channel])
        for channel in ("theta", "glasses")
    }
    lifts = {
        channel: tuple(int(value) for value in config["physical_lifts"][channel])
        for channel in ("theta", "glasses")
    }
    for channel in ("theta", "glasses"):
        characteristic = _spin_characteristic_from_lifts(
            channel, q_values[channel], lifts[channel]
        )
        expected = config["expected_spin_characteristics"][channel]
        observed = {
            "alpha": list(characteristic[0]),
            "beta": list(characteristic[1]),
        }
        if observed != expected or spin_ledger[point_id][channel] != expected:
            raise RuntimeError(
                f"physical spin mismatch at {point_id}/{channel}: "
                f"observed={observed}, expected={expected}"
            )

    settings = []
    for value in args.settings:
        recursion, quadrature = value.split(":", 1)
        settings.append((int(recursion), int(quadrature)))

    free_partitions = {
        channel: physical_superfield_plumbing_partition(
            channel,
            q_values[channel],
            lifts[channel],
            max_mode=args.free_mode_cutoff,
        )
        for channel in ("theta", "glasses")
    }
    for channel, free in free_partitions.items():
        print(
            f"{channel}: direct plumbing Z_(X+psi)="
            f"{free.one_superfield_value:.12e} at M={free.max_mode}",
            flush=True,
        )

    results = []
    free_diagnostics = {
        channel: asdict(value) for channel, value in free_partitions.items()
    }
    for recursion_order, quadrature_order in settings:
        for channel in ("theta", "glasses"):
            print(
                f"{channel}: recursion={recursion_order} "
                f"quadrature={quadrature_order}",
                flush=True,
            )
            result, legacy_free = evaluate_channel(
                channel=channel,
                q_values=q_values[channel],
                omega=omegas[channel],
                physical_lifts=lifts[channel],
                recursion_order=recursion_order,
                quadrature_order=quadrature_order,
                structure_precision=args.structure_precision,
                global_tolerance=args.global_tolerance,
                global_max_total_occupation=args.global_max_occupation,
                vacuum_word_length=args.vacuum_word_length,
                vacuum_max_mode=args.vacuum_max_mode,
                free_word_length=1,
                free_max_mode=1,
                free_superfield_override=(
                    free_partitions[channel].one_superfield_value
                ),
            )
            if legacy_free is not None:
                raise AssertionError(
                    "legacy period-matrix denominator was unexpectedly evaluated"
                )
            row = asdict(result)
            results.append(row)
            print(
                f"  ZL={result.value:.10e} Zfree={result.free_superfield:.10e} "
                f"QL={result.q_l:.10e} time={result.runtime_seconds:.1f}s",
                flush=True,
            )

    comparisons = []
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

    output = {
        "schema": "ns-genus2-human-note-current-config-rerun-v2",
        "scope": (
            "genus-two NS super-Liouville theta/glasses c-recursion with "
            "Human Note nonchiral sewing signs"
        ),
        "quantity": "Q_L = Z_L / Z_(X+psi)^9",
        "theta_partition_formula": (
            "sum_a (-1)^(a+p1+p2+p3) C_a^2 "
            "|prod_i(q_i^h_i) F_a|^2"
        ),
        "glasses_partition_formula": (
            "sum_a (-1)^(a+p_bridge) C_LBL^a C_RBR^a "
            "|prod_i(q_i^h_i) F_a|^2"
        ),
        "structure_constant_convention": {
            "source": "real BRY b=1 delta-normalized C and tilde C",
            "human_note_boundary": (
                "C_HN^(0)=C_BRY; C_HN^(1)=i*tilde_C_BRY"
            ),
            "odd_two_pants_phase": -1,
            "human_note_sewing_sign_retained": True,
        },
        "free_denominator_scope": {
            "role": "physical free-theory denominator of Q_L",
            "one_superfield": (
                "one noncompact real scalar plus one physical NS Majorana"
            ),
            "formula": (
                "G_X^pl |P_X^pl|^2 |F_psi^pl|^2, with every factor "
                "derived from charged-boson and physical-Majorana pants sewing"
            ),
            "loop_measure": (
                "d alpha_1 d alpha_2 with h(alpha)=alpha^2/2; "
                "G_X^pl=det(A_pl)^(-1/2)"
            ),
            "period_matrix_used": False,
            "riemann_theta_constant_used": False,
            "nonchiral_sign": (
                "Human Note sewing signs are already resummed into the "
                "fixed-spin Majorana determinant; no extra sign multiplies "
                "its absolute value"
            ),
            "auxiliary_double_virasoro_fermion_used": False,
        },
        "type0b_ns_primary_parities": list(TYPE0B_NS_PRIMARY_PARITIES),
        "central_charge": C_ORDINARY_AT_HAT_C_9,
        "hat_c": HAT_C_TARGET,
        "config": str(args.config),
        "point_id": point_id,
        "point_provenance": point.get("provenance", {}),
        "physical_lifts": {
            channel: list(values) for channel, values in lifts.items()
        },
        "spin_characteristics": spin_ledger[point_id],
        "checks": run_internal_checks(),
        "numerics": {
            "settings": list(args.settings),
            "structure_precision": args.structure_precision,
            "global_tolerance": args.global_tolerance,
            "global_max_total_occupation": args.global_max_occupation,
            "vacuum_word_length": args.vacuum_word_length,
            "vacuum_max_mode": args.vacuum_max_mode,
            "free_mode_cutoff": args.free_mode_cutoff,
        },
        "free_superfield_diagnostics": free_diagnostics,
        "results": results,
        "comparisons": comparisons,
        "limitations": (
            "This local rerun reaches recursion order 4.  The current "
            "order-24 five-point result requires fresh theta shards after "
            "the physical-lift correction."
        ),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(output, indent=2, default=_json_default) + "\n"
    )
    return output


if __name__ == "__main__":
    run()
