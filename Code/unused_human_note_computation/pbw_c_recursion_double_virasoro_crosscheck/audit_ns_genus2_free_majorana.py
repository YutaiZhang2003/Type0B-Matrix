#!/usr/bin/env python3
"""Audit genus-two bosonization against direct NS Majorana pants sewing.

The production denominator obtains the Majorana factor from

    Z_psi,L = exp(chiral_log_superfield - chiral_log_scalar)
            = sqrt(theta[delta](0|Omega) * P_X).

This script compares that value with a direct Fock/Pfaffian sewing sum in the
literal theta and glasses plumbing frames.  Agreement is never imposed by a
relative normalization.
"""

from __future__ import annotations

import argparse
import cmath
import itertools
import json
from pathlib import Path
from typing import Iterable

import numpy as np

from free_boson_plumbing import riemann_theta_constant_genus2
from free_majorana_pair_of_pants import (
    glasses_majorana_plumbing_partition,
    theta_majorana_plumbing_partition,
)


DEFAULT_SUMMARY = Path("Data Set/ns_genus2_fivepoint_r20_24_n8_12_axis_summary.json")


def _parse_complex(value: str | complex | float) -> complex:
    return complex(value)


def _stored_bosonized_majorana(free_diagnostics: dict) -> complex:
    stored_chiral_log = free_diagnostics["chiral_log"]
    chiral_log = (
        complex(stored_chiral_log)
        if isinstance(stored_chiral_log, str)
        else complex(*stored_chiral_log)
    )
    scalar_log = complex(
        free_diagnostics["scalar_chiral_log_real"],
        free_diagnostics["scalar_chiral_log_imag"],
    )
    return cmath.exp(chiral_log - scalar_log)


def _stored_scalar_oscillator(free_diagnostics: dict) -> complex:
    return cmath.exp(
        complex(
            free_diagnostics["scalar_chiral_log_real"],
            free_diagnostics["scalar_chiral_log_imag"],
        )
    )


def _even_characteristics():
    for alpha in itertools.product((0, 1), repeat=2):
        for beta in itertools.product((0, 1), repeat=2):
            if sum(left * right for left, right in zip(alpha, beta)) % 2 == 0:
                yield alpha, beta


def _characteristic_scan(omega, target_theta: complex) -> list[dict]:
    values = []
    for alpha, beta in _even_characteristics():
        theta = riemann_theta_constant_genus2(
            omega,
            (alpha, beta),
            tol=1.0e-14,
        )
        values.append(
            {
                "alpha": list(alpha),
                "beta": list(beta),
                "theta": [theta.real, theta.imag],
                "absolute_error": abs(theta - target_theta),
            }
        )
    values.sort(key=lambda row: row["absolute_error"])
    return values


def run(argv: Iterable[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument(
        "--twice-levels",
        type=int,
        nargs="+",
        default=(12, 16, 20, 24),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(list(argv) if argv is not None else None)

    source = json.loads(args.summary.read_text())
    # The focused current-config rerun stores the config path rather than a
    # second embedded copy.  Normalize it to the production-summary shape so
    # the same physical-Majorana oracle audits both artifacts.
    if isinstance(source.get("config"), str):
        config = json.loads(Path(source["config"]).read_text())
        point_id = str(source["point_id"])
        config["points"] = [
            point for point in config["points"] if str(point["id"]) == point_id
        ]
        if len(config["points"]) != 1:
            raise ValueError(f"rerun point {point_id!r} is absent from its config")
        source["config"] = config
        source["free_superfield"] = {
            point_id: source["free_superfield_diagnostics"]
        }
    twice_levels = tuple(sorted(set(int(value) for value in args.twice_levels)))
    if not twice_levels or twice_levels[0] < 0:
        raise ValueError("non-negative twice-level cutoffs are required")

    source_lifts = {
        channel: tuple(
            int(value)
            for value in source["config"]["physical_lifts"][channel]
        )
        for channel in ("theta", "glasses")
    }
    rows = []
    characteristic_matches = []
    for point in source["config"]["points"]:
        point_id = point["id"]
        for channel in ("theta", "glasses"):
            q_values = tuple(
                _parse_complex(value) for value in point["q_values"][channel]
            )
            expected = _stored_bosonized_majorana(
                source["free_superfield"][point_id][channel]
            )
            cutoff_rows = []
            for cutoff in twice_levels:
                if channel == "theta":
                    direct = theta_majorana_plumbing_partition(
                        *q_values,
                        max_total_twice_level=cutoff,
                        lifts=source_lifts[channel],
                    )
                else:
                    direct = glasses_majorana_plumbing_partition(
                        *q_values,
                        max_total_twice_level=cutoff,
                        lifts=source_lifts[channel],
                    )
                value_ratio = direct.chiral_value / expected
                square_ratio = direct.chiral_value**2 / expected**2
                cutoff_rows.append(
                    {
                        "max_total_twice_level": cutoff,
                        "direct_chiral": [
                            direct.chiral_value.real,
                            direct.chiral_value.imag,
                        ],
                        "last_shell": [
                            direct.last_shell.real,
                            direct.last_shell.imag,
                        ],
                        "last_shell_relative": abs(direct.last_shell)
                        / max(abs(direct.chiral_value), 1.0e-300),
                        "direct_over_bosonized": [
                            value_ratio.real,
                            value_ratio.imag,
                        ],
                        "square_ratio": [square_ratio.real, square_ratio.imag],
                        "relative_difference": abs(value_ratio - 1.0),
                        "nonzero_level_triples": direct.nonzero_level_triples,
                    }
                )
            rows.append(
                {
                    "point_id": point_id,
                    "channel": channel,
                    "physical_lifts": list(source_lifts[channel]),
                    "q_values": [[value.real, value.imag] for value in q_values],
                    "bosonized_chiral": [expected.real, expected.imag],
                    "cutoffs": cutoff_rows,
                }
            )

            omega = np.asarray(
                [
                    [_parse_complex(value) for value in omega_row]
                    for omega_row in point["omega"][channel]
                ],
                dtype=np.complex128,
            )
            scalar = _stored_scalar_oscillator(
                source["free_superfield"][point_id][channel]
            )
            final_direct = complex(*cutoff_rows[-1]["direct_chiral"])
            target_theta = final_direct**2 / scalar
            scan = _characteristic_scan(omega, target_theta)
            characteristic_matches.append(
                {
                    "point_id": point_id,
                    "channel": channel,
                    "physical_lifts": list(source_lifts[channel]),
                    "direct_implied_theta": [
                        target_theta.real,
                        target_theta.imag,
                    ],
                    "best_characteristic": scan[0],
                    "runner_up_characteristic": scan[1],
                }
            )

    reference_point = source["config"]["points"][0]
    reference_id = reference_point["id"]
    reference_q = tuple(
        _parse_complex(value) for value in reference_point["q_values"]["theta"]
    )
    reference_omega = np.asarray(
        [
            [_parse_complex(value) for value in omega_row]
            for omega_row in reference_point["omega"]["theta"]
        ],
        dtype=np.complex128,
    )
    reference_scalar = _stored_scalar_oscillator(
        source["free_superfield"][reference_id]["theta"]
    )
    theta_lift_characteristic_scan = []
    for lifts in itertools.product((-1, 1), repeat=3):
        direct = theta_majorana_plumbing_partition(
            *reference_q,
            max_total_twice_level=twice_levels[-1],
            lifts=lifts,
        )
        target_theta = direct.chiral_value**2 / reference_scalar
        scan = _characteristic_scan(reference_omega, target_theta)
        theta_lift_characteristic_scan.append(
            {
                "lifts": list(lifts),
                "best_alpha": scan[0]["alpha"],
                "best_beta": scan[0]["beta"],
                "absolute_error": scan[0]["absolute_error"],
            }
        )

    reference_glasses_characteristic = next(
        row["best_characteristic"]
        for row in characteristic_matches
        if row["point_id"] == reference_id and row["channel"] == "glasses"
    )
    reference_glasses_spin = {
        "alpha": reference_glasses_characteristic["alpha"],
        "beta": reference_glasses_characteristic["beta"],
    }
    same_spin_theta_lifts = [
        row["lifts"]
        for row in theta_lift_characteristic_scan
        if row["best_alpha"] == reference_glasses_spin["alpha"]
        and row["best_beta"] == reference_glasses_spin["beta"]
    ]
    observed_spins = {
        (row["point_id"], row["channel"]): {
            "alpha": row["best_characteristic"]["alpha"],
            "beta": row["best_characteristic"]["beta"],
        }
        for row in characteristic_matches
    }
    comparison_has_same_spin = all(
        observed_spins[(point["id"], "theta")]
        == observed_spins[(point["id"], "glasses")]
        for point in source["config"]["points"]
    )

    report = {
        "schema": "ns-genus2-free-majorana-direct-sewing-audit-v1",
        "source_summary": str(args.summary),
        "quantity": "Casimir-stripped chiral physical Majorana vacuum amplitude",
        "denominator_role": "physical free-theory denominator of Q_L",
        "auxiliary_double_virasoro_fermion_used": False,
        "normalization_fitted": False,
        "physical_lifts": {
            channel: list(values) for channel, values in source_lifts.items()
        },
        "production_declared_spin_characteristics": source["config"].get(
            "expected_spin_characteristics"
        ),
        "twice_level_cutoffs": list(twice_levels),
        "rows": rows,
        "characteristic_matches": characteristic_matches,
        "theta_lift_characteristic_scan": {
            "reference_point": reference_id,
            "rows": theta_lift_characteristic_scan,
        },
        "inference": {
            "reference_glasses_characteristic": reference_glasses_spin,
            "theta_lifts_for_same_spin_as_glasses": same_spin_theta_lifts,
            "production_channel_comparison_has_same_spin": comparison_has_same_spin,
        },
    }
    rendered = json.dumps(report, indent=2)
    if args.output is not None:
        args.output.write_text(rendered + "\n")
    print(rendered)
    return report


if __name__ == "__main__":
    run()
