#!/usr/bin/env python3
"""Checks for the direct torus two-point Virasoro block engine."""

from __future__ import annotations

import cmath
import math

import numpy as np

try:
    from torus_descendant_blocks import torus_one_point_descendant_coefficients
    from torus_two_point_blocks import (
        elliptic_nome,
        modular_lambda_series,
        necklace_coefficients_in_elliptic_nomes,
        necklace_descendant_coefficients,
        ope_c_recursion_coefficients,
        ope_coefficients_in_z,
        ope_descendant_coefficients,
    )
    from virasoro_blocks import TorusTwoPointVirasoroBlock
    from genus1_two_point_worldsheet import audited_h_recursion_necklace_coefficients
except ImportError:  # pragma: no cover
    from plumbing.torus_descendant_blocks import torus_one_point_descendant_coefficients
    from plumbing.torus_two_point_blocks import (
        elliptic_nome,
        modular_lambda_series,
        necklace_coefficients_in_elliptic_nomes,
        necklace_descendant_coefficients,
        ope_c_recursion_coefficients,
        ope_coefficients_in_z,
        ope_descendant_coefficients,
    )
    from plumbing.virasoro_blocks import TorusTwoPointVirasoroBlock
    from plumbing.genus1_two_point_worldsheet import (
        audited_h_recursion_necklace_coefficients,
    )


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def check_level_zero_and_one() -> None:
    c = 25.0
    h1 = 1.37
    h2 = 1.82
    d = 0.91
    necklace = necklace_descendant_coefficients(c, h1, h2, d, d, 1, 1)
    ope = ope_descendant_coefficients(c, h1, h2, d, d, 1, 1)
    print("level-zero checks")
    print(f"  necklace[0,0]={necklace[0,0]!r}")
    print(f"  ope[0,0]={ope[0,0]!r}")
    require(abs(necklace[0, 0] - 1.0) < 1.0e-13, "wrong necklace primary term")
    require(abs(ope[0, 0] - 1.0) < 1.0e-13, "wrong OPE primary term")
    require(np.all(np.isfinite(necklace)), "nonfinite necklace level-one coefficients")
    require(np.all(np.isfinite(ope)), "nonfinite OPE level-one coefficients")


def check_identity_necklace_reduces_to_character() -> None:
    c = 25.0
    h = 1.43
    order = 4
    coefficients = necklace_descendant_coefficients(c, h, h, 0.0, 0.0, order, order)
    # Identity insertions force equal descendant levels.  The diagonal is the
    # Verma-module partition number p(n).
    expected = np.array([1.0, 1.0, 2.0, 3.0, 5.0])
    diagonal = np.diag(coefficients)
    off_diagonal = coefficients - np.diag(diagonal)
    print("\nidentity necklace check")
    print(f"  diagonal={diagonal.tolist()}")
    print(f"  max off diagonal={np.max(np.abs(off_diagonal)):.6e}")
    require(np.max(np.abs(diagonal - expected)) < 2.0e-10, "wrong torus character")
    require(np.max(np.abs(off_diagonal)) < 2.0e-10, "identity changes descendant level")


def check_ope_zero_ope_level_is_torus_one_point() -> None:
    c = 25.0
    h_loop = 1.37
    h_ope = 1.82
    d = 0.91
    order = 4
    coefficients = ope_descendant_coefficients(c, h_loop, h_ope, d, d, order, 0)
    torus = torus_one_point_descendant_coefficients(c, h_loop, h_ope, (), order)
    difference = coefficients[:, 0] - np.asarray(torus)
    print("\nOPE m=0 torus one-point check")
    print(f"  max difference={np.max(np.abs(difference)):.6e}")
    require(np.max(np.abs(difference)) < 2.0e-10, "OPE loop contraction is wrong")


def check_ope_c_recursion_against_descendants() -> None:
    cases = (
        (26.215, 1.31, 1.73, 0.41, 0.56),
        (25.0, 1.37, 1.82, 0.91, 0.91),
        (25.0, 1.50, 1.50, 0.96, 0.96),
        (8.7, 1.07, 0.91, 0.31, 0.47),
    )
    largest_error = 0.0
    for c, h_loop, h_ope, d1, d2 in cases:
        direct = ope_descendant_coefficients(
            c,
            h_loop,
            h_ope,
            d1,
            d2,
            4,
            5,
        )
        recursive = ope_c_recursion_coefficients(
            c,
            h_loop,
            h_ope,
            d1,
            d2,
            4,
            5,
        )
        largest_error = max(
            largest_error,
            float(
                np.max(
                    np.abs(recursive - direct)
                    / np.maximum(1.0, np.abs(direct))
                )
            ),
        )
    print("\nOPE CCY c-recursion check")
    print(f"  max relative coefficient error={largest_error:.6e}")
    require(
        largest_error < 5.0e-11,
        "OPE c-recursion disagrees with direct descendant sewing",
    )


def check_regulated_necklace_h_recursion() -> None:
    h1 = 1.37
    h2 = 1.82
    d = 0.91
    orders = (4, 3)
    direct = necklace_descendant_coefficients(
        25.0,
        h1,
        h2,
        d,
        d,
        *orders,
    )
    regulated, audit_error, used_fallback = audited_h_recursion_necklace_coefficients(
        h1,
        h2,
        d,
        d,
        orders,
    )
    error = float(
        np.max(np.abs(regulated - direct) / np.maximum(1.0, np.abs(direct)))
    )
    print("\nregulated necklace h-recursion check")
    print(
        f"  max relative coefficient error={error:.6e}, "
        f"audit={audit_error:.6e}, fallback={used_fallback}"
    )
    require(error < 1.0e-7, "regulated h-recursion misses the c=25 necklace block")

    difficult_direct = necklace_descendant_coefficients(
        25.0,
        10.0,
        30.0,
        d,
        d,
        6,
        3,
    )
    difficult, difficult_audit, difficult_fallback = (
        audited_h_recursion_necklace_coefficients(
            10.0,
            30.0,
            d,
            d,
            (6, 3),
        )
    )
    difficult_error = float(
        np.max(
            np.abs(difficult - difficult_direct)
            / np.maximum(1.0, np.abs(difficult_direct))
        )
    )
    print(
        f"  hard-node error={difficult_error:.6e}, "
        f"audit={difficult_audit:.6e}, fallback={difficult_fallback}"
    )
    require(difficult_fallback, "unstable h-recursion node did not fall back")
    require(difficult_error < 1.0e-12, "hard-node descendant fallback is inconsistent")


def _lambda_numeric(q: complex, terms: int = 80) -> complex:
    theta2_reduced = sum(q ** (n * (n + 1)) for n in range(terms))
    theta2 = 2.0 * q ** 0.25 * theta2_reduced
    theta3 = 1.0 + 2.0 * sum(q ** (n * n) for n in range(1, terms))
    return (theta2 / theta3) ** 4


def check_elliptic_reexpansion() -> None:
    order = 8
    series = modular_lambda_series(order)
    expected_head = np.array([0.0, 16.0, -128.0, 704.0, -3072.0])
    print("\nmodular lambda check")
    print(f"  head={series[:5].tolist()}")
    require(np.max(np.abs(series[:5] - expected_head)) < 1.0e-10, "wrong lambda series")

    hat_q = 0.012 + 0.007j
    lambda_truncated = sum(series[n] * hat_q**n for n in range(order + 1))
    lambda_direct = _lambda_numeric(hat_q)
    require(abs(lambda_truncated - lambda_direct) < 2.0e-11, "lambda series evaluation failed")
    recovered = elliptic_nome(lambda_direct)
    print(f"  |E(lambda(q))-q|={abs(recovered-hat_q):.6e}")
    require(abs(recovered - hat_q) < 2.0e-11, "elliptic nome is not inverse to lambda")

    rng = np.random.default_rng(1234)
    original = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    transformed = necklace_coefficients_in_elliptic_nomes(original, 8, 8)
    q1_hat = 0.008 + 0.003j
    q2_hat = -0.006 + 0.004j
    q1 = _lambda_numeric(q1_hat)
    q2 = _lambda_numeric(q2_hat)
    direct = sum(original[n, m] * q1**n * q2**m for n in range(4) for m in range(4))
    recomposed = sum(
        transformed[n, m] * q1_hat**n * q2_hat**m
        for n in range(9)
        for m in range(9)
    )
    require(abs(direct - recomposed) < 2.0e-9, "bivariate elliptic re-expansion failed")


def check_ope_z_reexpansion() -> None:
    rng = np.random.default_rng(4321)
    original = rng.normal(size=(3, 5)) + 1j * rng.normal(size=(3, 5))
    transformed = ope_coefficients_in_z(original, 12)
    q = 0.002 + 0.001j
    z = 0.08 + 0.04j
    v = cmath.exp(-1j * z) - 1.0
    direct = sum(original[n, m] * q**n * v**m for n in range(3) for m in range(5))
    recomposed = sum(
        transformed[n, m] * q**n * z**m
        for n in range(3)
        for m in range(13)
    )
    print("\nOPE z re-expansion check")
    print(f"  difference={abs(direct-recomposed):.6e}")
    require(abs(direct - recomposed) < 2.0e-11, "OPE z re-expansion failed")


def run() -> None:
    check_level_zero_and_one()
    check_identity_necklace_reduces_to_character()
    check_ope_zero_ope_level_is_torus_one_point()
    check_ope_c_recursion_against_descendants()
    check_regulated_necklace_h_recursion()
    check_elliptic_reexpansion()
    check_ope_z_reexpansion()
    print("\nall torus two-point block checks passed")


if __name__ == "__main__":
    run()
