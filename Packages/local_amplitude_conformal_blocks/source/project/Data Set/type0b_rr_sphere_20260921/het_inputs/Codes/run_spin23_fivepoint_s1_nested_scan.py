#!/usr/bin/env python3
r"""Run the ordered two-regulator limit for all first-resonance S/V channels.

The auxiliary KLT twist is removed at each fixed nonzero soft momentum.  Only
those inner intercepts are subsequently extrapolated to zero soft momentum.
All 26 scalar projections are reconstructed exactly from the 15 four-vector
seeds before either fit.  The output is the bare screened wall integral; the
audited delta-normalized and repository-reduced conventions are recorded as
separate derived columns.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from time import perf_counter
from typing import Mapping, Sequence

from spin23_fivepoint_resonance_klt import (
    evaluate_regulated_klt_projections,
    fit_laurent_series,
    s1_outgoing_leg_product,
)
from spin23_fivepoint_s1_reduction import (
    four_vector_seed_projections,
    reconstruct_all_s1_values,
)


DEFAULT_OUTGOING = (0.31j, 0.53j, 0.56j, 0.60j)
DEFAULT_SOFT_RADII = (0.014, 0.010, 0.007, 0.005, 0.0035)
DEFAULT_TWIST_RATIOS = (0.60, 0.40, 0.27, 0.18, 0.12)
DEFAULT_SOFT_DIRECTION = 0.41 + 0.23j
DEFAULT_TWIST_DIRECTION = 1.0 + 0.37j


def _complex_pair(value: complex) -> list[float]:
    normalized = complex(value)
    return [float(normalized.real), float(normalized.imag)]


def _parse_float_tuple(raw: str) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in raw.split(",") if item.strip())
    if not values or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("expected positive comma-separated values")
    return values


def _parse_complex_tuple(raw: str) -> tuple[complex, ...]:
    values = tuple(complex(item.strip()) for item in raw.split(",") if item.strip())
    if len(values) != 4:
        raise argparse.ArgumentTypeError("expected four comma-separated complex values")
    return values


def _fit_record(
    radii: Sequence[float],
    values: Sequence[complex],
    *,
    maximum_degree: int,
) -> dict[str, object]:
    degree = min(int(maximum_degree), len(radii) - 1)
    primary = fit_laurent_series(
        radii,
        values,
        powers=tuple(range(degree + 1)),
    )
    comparison_degree = max(1, degree - 1)
    comparison = fit_laurent_series(
        radii,
        values,
        powers=tuple(range(comparison_degree + 1)),
    )
    return {
        "powers": list(primary.powers),
        "coefficients": [_complex_pair(value) for value in primary.coefficients],
        "intercept": _complex_pair(primary.finite_part),
        "residual_norm": primary.residual_norm,
        "condition_number": primary.condition_number,
        "lower_degree": comparison_degree,
        "lower_degree_intercept": _complex_pair(comparison.finite_part),
        "degree_spread": abs(primary.finite_part - comparison.finite_part),
    }


def run_nested_scan(
    *,
    outgoing_momenta: Sequence[complex] = DEFAULT_OUTGOING,
    soft_radii: Sequence[float] = DEFAULT_SOFT_RADII,
    twist_ratios: Sequence[float] = DEFAULT_TWIST_RATIOS,
    soft_direction: complex = DEFAULT_SOFT_DIRECTION,
    twist_direction: complex = DEFAULT_TWIST_DIRECTION,
    rule: str = "gauss-jacobi-8",
    rtol: float = 2.0e-4,
    atol: float = 2.0e-7,
) -> dict[str, object]:
    resonant = tuple(complex(value) for value in outgoing_momenta)
    if len(resonant) != 4 or abs(sum(resonant) - 2j) > 1.0e-10:
        raise ValueError("the four base outgoing momenta must sum to 2i")
    q_radii = tuple(float(value) for value in soft_radii)
    ratios = tuple(float(value) for value in twist_ratios)
    if len(q_radii) < 3 or len(ratios) < 3:
        raise ValueError("at least three soft radii and three twist ratios are required")

    seeds = four_vector_seed_projections()
    labels = tuple(
        reconstruct_all_s1_values(
            {seed.label: 0.0j for seed in seeds},
            resonant,
        )
    )
    inner_by_label: dict[str, list[complex]] = {label: [] for label in labels}
    soft_rows: list[dict[str, object]] = []
    total_start = perf_counter()

    for soft_radius in q_radii:
        soft = soft_radius * complex(soft_direction)
        physical_outgoing = (resonant[0] - soft,) + resonant[1:]
        finite_values: dict[str, list[complex]] = {label: [] for label in labels}
        regulator_rows: list[dict[str, object]] = []
        twist_radii = tuple(soft_radius * ratio for ratio in ratios)
        for twist_radius in twist_radii:
            started = perf_counter()
            results = evaluate_regulated_klt_projections(
                seeds,
                resonant,
                soft_momentum=soft,
                twist_regulator=twist_radius * complex(twist_direction),
                rule=rule,
                rtol=rtol,
                atol=atol,
            )
            seed_values = {item.projection_label: item.wall_value for item in results}
            reconstructed = reconstruct_all_s1_values(
                seed_values,
                physical_outgoing,
                soft_momentum=soft,
            )
            for label, value in reconstructed.items():
                finite_values[label].append(value)
            regulator_rows.append(
                {
                    "twist_radius": twist_radius,
                    "elapsed_seconds": perf_counter() - started,
                    "all_periods_converged": all(
                        item.periods_converged for item in results
                    ),
                    "maximum_seed_error_bound": max(
                        item.absolute_error_bound for item in results
                    ),
                    "seed_wall_values": {
                        label: _complex_pair(value)
                        for label, value in seed_values.items()
                    },
                }
            )
            print(
                f"soft={soft_radius:.6g} twist={twist_radius:.6g} "
                f"elapsed={regulator_rows[-1]['elapsed_seconds']:.2f}s",
                flush=True,
            )

        inner_fits = {
            label: _fit_record(
                twist_radii,
                finite_values[label],
                maximum_degree=3,
            )
            for label in labels
        }
        for label in labels:
            inner_by_label[label].append(complex(*inner_fits[label]["intercept"]))
        soft_rows.append(
            {
                "soft_radius": soft_radius,
                "soft_momentum": _complex_pair(soft),
                "physical_outgoing_momenta": [
                    _complex_pair(value) for value in physical_outgoing
                ],
                "regulator_rows": regulator_rows,
                "twist_limit_fits": inner_fits,
            }
        )

    outer_fits = {
        label: _fit_record(q_radii, values, maximum_degree=3)
        for label, values in inner_by_label.items()
    }
    leg_product = s1_outgoing_leg_product(resonant)
    final_values = {}
    for label, fit in outer_fits.items():
        wall_value = complex(*fit["intercept"])
        final_values[label] = {
            "wall_value": _complex_pair(wall_value),
            "canonical_delta_normalized_screen_stripped": _complex_pair(
                -2j * leg_product * wall_value
            ),
            "repo_reduced_pco_stripped": _complex_pair(
                -16j * leg_product * wall_value
            ),
            "outer_degree_spread": fit["degree_spread"],
            "outer_residual_norm": fit["residual_norm"],
        }

    return {
        "schema": "spin23_fivepoint_s1_nested_limit_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "separated_soft_representative_not_contact_complete_physical_residue",
        "base_resonant_outgoing_momenta": [_complex_pair(value) for value in resonant],
        "soft_radii": list(q_radii),
        "twist_ratios": list(ratios),
        "soft_direction": _complex_pair(soft_direction),
        "twist_scalar_direction": _complex_pair(twist_direction),
        "quadrature_rule": rule,
        "rtol": rtol,
        "atol": atol,
        "four_vector_seed_labels": [seed.label for seed in seeds],
        "outgoing_leg_product_G4": _complex_pair(leg_product),
        "normalization_note": (
            "wall_value includes 1/8 from three physical PCOs; absolute sphere, "
            "wall, mu, and orientation constants remain conventional. This "
            "separated soft construction omits the correlated vertical/fusion "
            "completion and must not be used as a physical resonance datum."
        ),
        "soft_rows": soft_rows,
        "outer_soft_limit_fits": outer_fits,
        "final_values": final_values,
        "total_elapsed_seconds": perf_counter() - total_start,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outgoing", type=_parse_complex_tuple, default=DEFAULT_OUTGOING)
    parser.add_argument(
        "--soft-radii",
        type=_parse_float_tuple,
        default=DEFAULT_SOFT_RADII,
    )
    parser.add_argument(
        "--twist-ratios",
        type=_parse_float_tuple,
        default=DEFAULT_TWIST_RATIOS,
    )
    parser.add_argument("--soft-direction", type=complex, default=DEFAULT_SOFT_DIRECTION)
    parser.add_argument("--twist-direction", type=complex, default=DEFAULT_TWIST_DIRECTION)
    parser.add_argument("--rule", default="gauss-jacobi-8")
    parser.add_argument("--rtol", type=float, default=2.0e-4)
    parser.add_argument("--atol", type=float, default=2.0e-7)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = run_nested_scan(
        outgoing_momenta=args.outgoing,
        soft_radii=args.soft_radii,
        twist_ratios=args.twist_ratios,
        soft_direction=args.soft_direction,
        twist_direction=args.twist_direction,
        rule=args.rule,
        rtol=args.rtol,
        atol=args.atol,
    )
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(encoded, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
