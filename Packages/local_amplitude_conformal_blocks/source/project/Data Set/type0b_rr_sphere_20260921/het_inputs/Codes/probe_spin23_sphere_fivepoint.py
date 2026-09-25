#!/usr/bin/env python3
r"""Fixed-momentum convergence probe for the new sphere five-point blocks.

This is deliberately not a five-point amplitude integrator.  It tests the
new ingredient that such an integrator would repeatedly consume: the
structure-weighted ``b=1`` NS comb block multiplied by a normalized ordinary
Liouville Virasoro comb block at fixed internal momenta and moduli.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from spin23_liouville_data import liouville_fivepoint_structure_product
from spin23_sphere_fivepoint import (
    SelfDualNSSphereFivePointHRecursion,
    SelfDualVirasoroSphereFivePointHRecursion,
    direct_ns_sphere_fivepoint_series,
    direct_virasoro_sphere_fivepoint_series,
    ns_component_structure_product,
    ns_fivepoint_structure_weights,
    ns_liouville_weight,
    virasoro_central_charge,
    virasoro_liouville_weight,
)


def _complex_data(value: complex) -> dict[str, float]:
    value = complex(value)
    return {"real": value.real, "imag": value.imag, "abs": abs(value)}


def _relative_change(current: complex, previous: complex) -> float:
    return abs(current - previous) / max(abs(current), abs(previous), 1.0e-300)


def run_probe(
    *,
    t: float = 0.12,
    internal_momenta: Sequence[complex] = (0.37, 0.61),
    q_1: complex = 0.18 + 0.05j,
    q_2: complex = 0.22 - 0.04j,
    maximum_order: int = 2,
    radius: float = 0.025,
    check_radius: float = 0.035,
    samples: int = 12,
    structure_precision: int = 40,
) -> dict[str, Any]:
    r"""Run one equal-imaginary-energy fixed-block convergence check.

    The external momenta are ``(it,it,it,it,4it)``.  At reported order ``L``
    the Virasoro series is kept through level ``(L,L)`` and the NS series
    through twice-level ``(2L,2L)``.  No complex conjugation is applied to
    analytically continued momenta or structure constants.
    """

    if not 0 < t < 0.2:
        raise ValueError(
            "the equal-imaginary sufficient no-residue zone requires 0<t<0.2"
        )
    if maximum_order < 0:
        raise ValueError("maximum_order must be nonnegative")
    internal = tuple(complex(value) for value in internal_momenta)
    if len(internal) != 2:
        raise ValueError("two internal momenta are required")
    external = (1j * t,) * 4 + (4j * t,)
    form_weights = ns_fivepoint_structure_weights(
        internal_momenta=internal,
        external_momenta=external,
        precision=structure_precision,
    )
    ordinary_structure_product = liouville_fivepoint_structure_product(
        internal_momenta=internal,
        external_momenta=external,
        precision=structure_precision,
    )

    virasoro = SelfDualVirasoroSphereFivePointHRecursion(
        internal_momenta=internal,
        external_momenta=external,
        radius=radius,
        check_radius=check_radius,
        samples=samples,
    )
    virasoro_series = virasoro.series((maximum_order, maximum_order))

    ns_blocks: dict[tuple[int, int], SelfDualNSSphereFivePointHRecursion] = {}
    ns_series = {}
    maximum_twice_level = 2 * maximum_order
    for parity1 in (0, 1):
        for parity2 in (0, 1):
            routing = (parity1, parity2)
            block = SelfDualNSSphereFivePointHRecursion(
                internal_momenta=internal,
                external_momenta=external,
                component_parities=routing,
                radius=radius,
                check_radius=check_radius,
                samples=samples,
            )
            ns_blocks[routing] = block
            ns_series[routing] = block.series(
                (maximum_twice_level, maximum_twice_level)
            )

    rows: list[dict[str, Any]] = []
    previous = None
    for order in range(maximum_order + 1):
        virasoro_value = virasoro_series.descendant_value(
            q_1.conjugate(),
            q_2.conjugate(),
            maximum_levels=(order, order),
        )
        ns_value = 0.0j
        components: dict[str, dict[str, float]] = {}
        for routing, series in ns_series.items():
            component = series.descendant_value(
                q_1,
                q_2,
                maximum_levels=(2 * order, 2 * order),
            )
            weighted = ns_component_structure_product(
                form_weights, routing
            ) * component
            ns_value += weighted
            components[f"{routing[0]}{routing[1]}"] = _complex_data(weighted)
        ordinary_value = ordinary_structure_product * virasoro_value
        product = ns_value * ordinary_value
        row: dict[str, Any] = {
            "order": order,
            "ns_maximum_twice_levels": [2 * order, 2 * order],
            "virasoro_maximum_levels": [order, order],
            "ns_structure_weighted": _complex_data(ns_value),
            "virasoro_normalized": _complex_data(virasoro_value),
            "ordinary_liouville_structure_weighted": _complex_data(
                ordinary_value
            ),
            "block_product": _complex_data(product),
            "ns_components": components,
        }
        if previous is not None:
            row["product_relative_change"] = _relative_change(product, previous)
        rows.append(row)
        previous = product

    direct_order = min(maximum_order, 2)
    direct_ns = direct_ns_sphere_fivepoint_series(
        c=13.5,
        internal_weights=tuple(
            ns_liouville_weight(momentum, 1.0) for momentum in internal
        ),
        external_weights=tuple(
            ns_liouville_weight(momentum, 1.0) for momentum in external
        ),
        maximum_twice_levels=(2 * direct_order, 2 * direct_order),
        form_weights=form_weights,
    ).descendant_value(q_1, q_2)
    direct_virasoro = direct_virasoro_sphere_fivepoint_series(
        c=virasoro_central_charge(1.0),
        internal_weights=tuple(
            virasoro_liouville_weight(momentum, 1.0) for momentum in internal
        ),
        external_weights=tuple(
            virasoro_liouville_weight(momentum, 1.0) for momentum in external
        ),
        maximum_levels=(direct_order, direct_order),
    ).descendant_value(q_1.conjugate(), q_2.conjugate())
    recursive_at_direct_order = complex(
        rows[direct_order]["block_product"]["real"],
        rows[direct_order]["block_product"]["imag"],
    )
    direct_product = direct_ns * ordinary_structure_product * direct_virasoro

    maximum_diagnostic = max(
        diagnostic.relative_error
        for block in ns_blocks.values()
        for diagnostic in block.diagnostics(
            (maximum_twice_level, maximum_twice_level)
        ).values()
    )
    maximum_diagnostic = max(
        maximum_diagnostic,
        max(
            diagnostic.relative_error
            for diagnostic in virasoro.diagnostics(
                (maximum_order, maximum_order)
            ).values()
        ),
    )
    return {
        "scope": "fixed_internal_momenta_normalized_block_product_not_amplitude",
        "external_momenta": [_complex_data(value) for value in external],
        "internal_momenta": [_complex_data(value) for value in internal],
        "equal_imaginary_t": t,
        "global_no_residue_margin": 1.0 - 5.0 * t,
        "q_1": _complex_data(q_1),
        "q_2": _complex_data(q_2),
        "ordinary_liouville_structure_product": _complex_data(
            ordinary_structure_product
        ),
        "rows": rows,
        "direct_certification_order": direct_order,
        "direct_product": _complex_data(direct_product),
        "direct_recursive_relative_difference": _relative_change(
            direct_product, recursive_at_direct_order
        ),
        "maximum_finite_part_radius_diagnostic": maximum_diagnostic,
        "omissions": [
            "two internal Liouville momentum integrations",
            "five-point moduli integration and channel cover",
            "full matter/ghost PCO sum",
        ],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--t", type=float, default=0.12)
    parser.add_argument("--internal-p1", type=float, default=0.37)
    parser.add_argument("--internal-p2", type=float, default=0.61)
    parser.add_argument("--maximum-order", type=int, default=2)
    parser.add_argument("--samples", type=int, default=12)
    parser.add_argument("--output", type=Path)
    return parser


def main() -> None:
    args = _parser().parse_args()
    result = run_probe(
        t=args.t,
        internal_momenta=(args.internal_p1, args.internal_p2),
        maximum_order=args.maximum_order,
        samples=args.samples,
    )
    payload = json.dumps(result, indent=2, sort_keys=True)
    if args.output is None:
        print(payload)
    else:
        args.output.write_text(payload + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
