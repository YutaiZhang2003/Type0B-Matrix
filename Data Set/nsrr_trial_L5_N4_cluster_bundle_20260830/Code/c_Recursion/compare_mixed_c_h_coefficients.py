"""Compare NSNSRR NS-channel coefficients from c- and h-recursion.

The c-recursive curve in this diagnostic uses the ordinary scalar global
seed at every recursive call.  The pole kernel is the validated Ramond
fixed-weight c-pole kernel, but this scalar seed is known to omit Ramond
G_0-dependent non-global terms beginning at level 3/2.  The comparison is
therefore a diagnostic of the missing regular seed, not a production block.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from mixed_ramond_sphere_blocks import MixedNSExchangeSphereFourPointBlock
from ramond_c_recursive_sphere_blocks import (
    CRecursiveMixedNSExchangeSphereFourPointBlock,
)
from ramond_sphere_blocks import ramond_liouville_weight
from superconformal_blocks import central_charge, ns_liouville_weight


def _complex_record(value: complex) -> dict[str, float]:
    return {
        "real": float(value.real),
        "imag": float(value.imag),
        "abs": float(abs(value)),
    }


def coefficient_ledger(
    *,
    b: float,
    momenta: tuple[float, float, float, float],
    internal_momentum: float,
    order: int,
    sign2: int,
) -> dict[str, Any]:
    """Return one all-coefficient c-versus-h comparison."""

    c = central_charge(b)
    h1_r = ramond_liouville_weight(momenta[0], b)
    h2_r = ramond_liouville_weight(momenta[1], b)
    h3_ns = ns_liouville_weight(momenta[2], b)
    h4_ns = ns_liouville_weight(momenta[3], b)
    internal_weight = ns_liouville_weight(internal_momentum, b)

    holder: dict[str, CRecursiveMixedNSExchangeSphereFourPointBlock] = {}

    def scalar_global_seed(
        twice_level: int,
        shifted_weight: complex,
        shifted_c: complex,
        parity: str,
        sign3: int,
        recursive_sign2: int,
    ) -> complex:
        block = holder["block"]
        return block._global_trial_seed(
            twice_level=twice_level,
            internal_weight=shifted_weight,
            c=shifted_c,
            parity=parity,
            sign3=sign3,
            sign2=recursive_sign2,
        )

    c_block = CRecursiveMixedNSExchangeSphereFourPointBlock(
        c=c,
        h1_r=h1_r,
        h2_r=h2_r,
        h3_ns=h3_ns,
        h4_ns=h4_ns,
        internal_weight=internal_weight,
        sign2=sign2,
        regular_seed=scalar_global_seed,
    )
    holder["block"] = c_block
    h_block = MixedNSExchangeSphereFourPointBlock(
        b=b,
        p1_r=momenta[0],
        p2_r=momenta[1],
        p3_ns=momenta[2],
        p4_ns=momenta[3],
        internal_momentum=internal_momentum,
        sign2=sign2,
    )

    sectors: dict[str, list[dict[str, Any]]] = {}
    for parity in ("even", "odd"):
        c_coefficients = c_block.elliptic_coefficients(order, parity)
        h_coefficients = h_block.elliptic_coefficients(order, parity)
        offset = 0 if parity == "even" else 1
        rows = []
        for index in range(order):
            twice_power = 2 * index + offset
            c_value = c_coefficients[twice_power]
            h_value = h_coefficients[twice_power]
            difference = c_value - h_value
            rows.append(
                {
                    "index": index,
                    "q_power": twice_power / 2.0,
                    "c": _complex_record(c_value),
                    "h": _complex_record(h_value),
                    "difference": _complex_record(difference),
                    "relative_difference": float(
                        abs(difference) / max(abs(h_value), 1.0e-300)
                    ),
                }
            )
        sectors[parity] = rows

    return {
        "sign2": sign2,
        "sectors": sectors,
    }


def build_comparison(
    *,
    b: float = 1.27,
    momenta: tuple[float, float, float, float] = (
        0.31,
        0.41,
        0.23,
        0.37,
    ),
    internal_momentum: float = 0.70,
    order: int = 20,
) -> dict[str, Any]:
    if order < 1:
        raise ValueError("order must be positive")
    return {
        "description": (
            "NSNSRR NS-channel H(q): validated c-pole kernel with the "
            "incomplete scalar global seed versus the tested h-recursion"
        ),
        "seed_status": "diagnostic_scalar_global_seed_not_production",
        "parameters": {
            "b": b,
            "c": _complex_record(central_charge(b)),
            "momenta": list(momenta),
            "internal_momentum": internal_momentum,
            "order": order,
        },
        "branches": [
            coefficient_ledger(
                b=b,
                momenta=momenta,
                internal_momentum=internal_momentum,
                order=order,
                sign2=sign2,
            )
            for sign2 in (1, -1)
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--b", type=float, default=1.27)
    parser.add_argument("--momenta", nargs=4, type=float, default=(0.31, 0.41, 0.23, 0.37))
    parser.add_argument("--internal-momentum", type=float, default=0.70)
    parser.add_argument("--order", type=int, default=20)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = build_comparison(
        b=args.b,
        momenta=tuple(args.momenta),
        internal_momentum=args.internal_momentum,
        order=args.order,
    )
    payload = json.dumps(result, indent=2, sort_keys=True)
    if args.output is None:
        print(payload)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
        print(args.output)


if __name__ == "__main__":
    main()
