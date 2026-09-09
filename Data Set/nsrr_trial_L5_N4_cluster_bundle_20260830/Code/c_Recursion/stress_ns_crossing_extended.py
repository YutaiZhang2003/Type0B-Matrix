"""Fresh multi-case crossing and negative-control audit for the NS sphere.

The two channels are integrated independently.  No channel averaging,
rescaling, or use of crossing symmetry enters the evaluation.  Besides the
physical even-plus-odd answer, the ledger records even-only, odd-only, and
even-minus-odd combinations.  The last combination is a deliberately wrong
*nonchiral* relative-sign control; it is not the overall chiral odd-block
phase, which cancels identically between holomorphic and antiholomorphic
blocks.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Sequence

from sphere_four_point import BRYNSFourPointCorrelator, GChannelComponents


DEFAULT_CASES = (
    ("BRY-generic", (0.5, 1.0 / 3.0, 0.25, 0.6)),
    ("generic-B", (0.2, 0.45, 0.7, 0.35)),
    ("generic-C", (0.8, 0.15, 0.55, 0.4)),
)
DEFAULT_Z = (
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.3 + 0.2j,
    0.5 + 0.25j,
    0.7 + 0.2j,
)


def _complex_record(value: complex) -> list[float]:
    value = complex(value)
    return [value.real, value.imag]


def _z_record(value: complex) -> float | list[float]:
    value = complex(value)
    if value.imag == 0.0:
        return value.real
    return _complex_record(value)


def _relative_residual(left: complex, right: complex) -> float:
    return abs(left - right) / max(abs(left), abs(right), 1.0e-300)


def _combinations(value: GChannelComponents) -> dict[str, complex]:
    return {
        "physical_even_plus_odd": value.total,
        "even_only": value.even,
        "odd_only": value.odd,
        "wrong_even_minus_odd": value.wrong_relative_sign,
    }


def _parse_orders(text: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in text.split(",") if item.strip())
    if not values or any(value < 0 for value in values):
        raise argparse.ArgumentTypeError("orders must be nonnegative integers")
    return values


def _parse_z(text: str) -> tuple[complex, ...]:
    try:
        values = tuple(
            complex(item.strip().replace("i", "j"))
            for item in text.split(",")
            if item.strip()
        )
    except ValueError as error:
        raise argparse.ArgumentTypeError("invalid complex z grid") from error
    if not values or any(value in (0, 1) for value in values):
        raise argparse.ArgumentTypeError("z grid must be nonempty and avoid 0,1")
    return values


def _parse_cases(text: str) -> tuple[tuple[str, tuple[float, ...]], ...]:
    cases = []
    try:
        for index, raw_case in enumerate(text.split(";"), start=1):
            raw_case = raw_case.strip()
            if not raw_case:
                continue
            if ":" in raw_case:
                name, raw_momenta = raw_case.split(":", 1)
            else:
                name, raw_momenta = f"case-{index}", raw_case
            momenta = tuple(float(item.strip()) for item in raw_momenta.split(","))
            if len(momenta) != 4:
                raise ValueError
            cases.append((name.strip(), momenta))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "cases must have the form name:p1,p2,p3,p4;name:..."
        ) from error
    if not cases:
        raise argparse.ArgumentTypeError("at least one case is required")
    return tuple(cases)


def _case_text(cases: Sequence[tuple[str, Sequence[float]]]) -> str:
    return ";".join(
        f"{name}:" + ",".join(str(value) for value in momenta)
        for name, momenta in cases
    )


def _z_text(points: Iterable[complex]) -> str:
    return ",".join(str(value).replace("j", "i") for value in points)


def evaluate_case(
    *,
    name: str,
    momenta: Sequence[float],
    orders: Sequence[int],
    z_values: Sequence[complex],
    p_max: float,
    quadrature_order: int,
    structure_precision: int,
    block_working_precision: int,
) -> dict:
    p1, p2, p3, p4 = momenta
    common = dict(
        c_recursion_order=orders[0],
        structure_precision=structure_precision,
        central_charge_shift=0.0,
        block_working_precision=block_working_precision,
    )
    direct = BRYNSFourPointCorrelator(
        p1=p1, p2=p2, p3=p3, p4=p4, **common
    )
    crossed = BRYNSFourPointCorrelator(
        p1=p3, p2=p2, p3=p1, p4=p4, **common
    )
    points = tuple(complex(value) for value in z_values)
    crossed_points = tuple(1.0 - value for value in points)
    rows = []
    values_by_order: dict[int, tuple[tuple[complex, ...], tuple[complex, ...]]] = {}
    for order in orders:
        print(
            f"case={name} order={order} p_max={p_max} quad={quadrature_order}",
            flush=True,
        )
        direct.c_recursion_order = order
        crossed.c_recursion_order = order
        left_components = direct.evaluate_g_components_grid(
            points, p_max=p_max, quadrature_order=quadrature_order
        )
        right_components = crossed.evaluate_g_components_grid(
            crossed_points, p_max=p_max, quadrature_order=quadrature_order
        )
        physical_left = tuple(value.total for value in left_components)
        physical_right = tuple(value.total for value in right_components)
        values_by_order[order] = (physical_left, physical_right)
        for z, left_component, right_component in zip(
            points, left_components, right_components
        ):
            left = _combinations(left_component)
            right = _combinations(right_component)
            comparisons = {}
            for key in left:
                comparisons[key] = {
                    "left": _complex_record(left[key]),
                    "right": _complex_record(right[key]),
                    "absolute_residual": abs(left[key] - right[key]),
                    "relative_residual": _relative_residual(left[key], right[key]),
                }
            rows.append(
                {
                    "order": order,
                    "twice_level": 2 * order,
                    "z": _z_record(z),
                    "comparisons": comparisons,
                }
            )

    maxima = {}
    for order in orders:
        order_rows = [row for row in rows if row["order"] == order]
        maxima[str(order)] = {
            key: max(
                row["comparisons"][key]["relative_residual"]
                for row in order_rows
            )
            for key in order_rows[0]["comparisons"]
        }

    drifts = {}
    for previous, current in zip(orders, orders[1:]):
        previous_left, previous_right = values_by_order[previous]
        current_left, current_right = values_by_order[current]
        drifts[f"{previous}->{current}"] = {
            "left": max(
                _relative_residual(old, new)
                for old, new in zip(previous_left, current_left)
            ),
            "right": max(
                _relative_residual(old, new)
                for old, new in zip(previous_right, current_right)
            ),
        }

    return {
        "name": name,
        "momenta": list(momenta),
        "max_relative_crossing_residual": maxima,
        "max_relative_channel_drift": drifts,
        "rows": rows,
    }


def build_ledger(
    *,
    cases: Sequence[tuple[str, Sequence[float]]] = DEFAULT_CASES,
    orders: Sequence[int] = (8, 10, 12),
    z_values: Sequence[complex] = DEFAULT_Z,
    p_max: float = 5.0,
    quadrature_order: int = 32,
    structure_precision: int = 35,
    block_working_precision: int = 80,
) -> dict:
    results = [
        evaluate_case(
            name=name,
            momenta=momenta,
            orders=orders,
            z_values=z_values,
            p_max=p_max,
            quadrature_order=quadrature_order,
            structure_precision=structure_precision,
            block_working_precision=block_working_precision,
        )
        for name, momenta in cases
    ]
    return {
        "method": {
            "theory": "self-dual b=1 NS super-Liouville",
            "hat_c": 9.0,
            "c": 13.5,
            "spectral_measure": "dP/pi",
            "channel_evaluation": "independent, no averaging or rescaling",
            "block_evaluation": "direct c-recursion with exact global osp(1|2) leaves",
            "central_charge_shift": 0.0,
            "series_truncation": None,
            "chiral_sign_limitation": (
                "The overall odd chiral phase cancels identically in "
                "F_1(z)F_1(zbar), so crossing cannot test it."
            ),
            "negative_control": (
                "wrong_even_minus_odd reverses the physical relative sign of "
                "the two nonchiral families; it is not a chiral phase flip."
            ),
        },
        "parameters": {
            "orders": list(orders),
            "twice_levels": [2 * order for order in orders],
            "z_values": [_z_record(value) for value in z_values],
            "p_max": p_max,
            "quadrature_order": quadrature_order,
            "structure_precision": structure_precision,
            "block_working_precision": block_working_precision,
        },
        "cases": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=_parse_cases, default=DEFAULT_CASES)
    parser.add_argument("--orders", type=_parse_orders, default=(8, 10, 12))
    parser.add_argument("--z", type=_parse_z, default=DEFAULT_Z)
    parser.add_argument("--p-max", type=float, default=5.0)
    parser.add_argument("--quadrature-order", type=int, default=32)
    parser.add_argument("--structure-precision", type=int, default=35)
    parser.add_argument("--block-working-precision", type=int, default=80)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    ledger = build_ledger(
        cases=args.cases,
        orders=args.orders,
        z_values=args.z,
        p_max=args.p_max,
        quadrature_order=args.quadrature_order,
        structure_precision=args.structure_precision,
        block_working_precision=args.block_working_precision,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(ledger, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
