#!/usr/bin/env python3
"""Checks for the torus three-point pair-OPE block."""

from __future__ import annotations

import numpy as np

try:
    from torus_three_point_ope_blocks import (
        comb_ope_c_recursion_coefficients,
        comb_ope_direct_coefficients,
        pair_ope_c_recursion_coefficients,
        pair_ope_direct_coefficients,
        pair_ope_global_coefficient,
        pair_ope_large_c_coefficient,
    )
except ImportError:  # pragma: no cover
    from plumbing.torus_three_point_ope_blocks import (
        comb_ope_c_recursion_coefficients,
        comb_ope_direct_coefficients,
        pair_ope_c_recursion_coefficients,
        pair_ope_direct_coefficients,
        pair_ope_global_coefficient,
        pair_ope_large_c_coefficient,
    )


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def run() -> None:
    external = (0.83, 0.91, 0.88)
    internal = (1.37, 1.82, 1.51)
    direct = pair_ope_direct_coefficients(
        22.7,
        external_weights=external,
        internal_weights=internal,
        orders=(3, 3, 3),
    )
    recursive = pair_ope_c_recursion_coefficients(
        22.7,
        external_weights=external,
        internal_weights=internal,
        orders=(3, 3, 3),
    )
    relative = np.max(np.abs(direct - recursive) / np.maximum(np.abs(direct), 1.0))
    print(f"pair-OPE direct versus c-recursion maximum scaled error: {relative:.3e}")
    require(relative < 2.0e-10, "pair-OPE c-recursion disagrees with direct sewing")
    require(abs(direct[0, 0, 0] - 1.0) < 1.0e-13, "wrong primary coefficient")
    require(
        abs(
            pair_ope_large_c_coefficient(
                (1, 1, 1),
                external_weights=external,
                internal_weights=internal,
            )
            - pair_ope_global_coefficient(
                (1, 1, 1),
                external_weights=external,
                internal_weights=internal,
            )
        )
        < 1.0e-13,
        "vacuum seed entered below oscillator level two",
    )
    comb_direct = comb_ope_direct_coefficients(
        22.7,
        external_weights=external,
        internal_weights=internal,
        orders=(3, 3, 3),
    )
    comb_recursive = comb_ope_c_recursion_coefficients(
        22.7,
        external_weights=external,
        internal_weights=internal,
        orders=(3, 3, 3),
    )
    comb_relative = np.max(
        np.abs(comb_direct - comb_recursive) / np.maximum(np.abs(comb_direct), 1.0)
    )
    print(
        "comb-OPE direct versus c-recursion maximum scaled error: "
        f"{comb_relative:.3e}"
    )
    require(
        comb_relative < 2.0e-10,
        "comb-OPE c-recursion disagrees with direct sewing",
    )
    print("all torus three-point pair-OPE checks passed")


if __name__ == "__main__":
    run()
