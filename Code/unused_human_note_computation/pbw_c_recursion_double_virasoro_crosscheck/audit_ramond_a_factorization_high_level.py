#!/usr/bin/env python3
"""Extended exact audit of Ramond A_rs and null three-point factorization."""

from __future__ import annotations

import argparse
from itertools import product
import time

import sympy as sp

from ramond_pbw_generalized_ward import (
    clean,
    contract_ramond_null,
    degenerate_null_vector,
    fixed_beta_inverse_null_norm,
    inverse_null_product_59,
    ramond_degenerate_data,
    ramond_labels_at_level,
)


def assert_zero(label: str, expression: sp.Expr) -> None:
    residual = clean(expression)
    if residual != 0:
        raise AssertionError(f"{label}: expected zero, got {residual}")


def certify_a_rs(r: int, s: int, b: sp.Expr, *, require_mismatches: bool):
    started = time.monotonic()
    data = ramond_degenerate_data(r, s, b)
    level = int(data["level"])
    chi_plus = degenerate_null_vector(r, s, b, 0)
    chi_minus = degenerate_null_vector(r, s, b, 1)
    direct_plus = fixed_beta_inverse_null_norm(r, s, b, chi_plus)
    direct_minus = fixed_beta_inverse_null_norm(r, s, b, chi_minus)
    even_product = inverse_null_product_59(r, s, b, lattice="even")
    literal_product = inverse_null_product_59(r, s, b, lattice="literal")
    odd_product = inverse_null_product_59(r, s, b, lattice="odd")

    assert_zero(f"A_{{{r},{s}}} even lattice", direct_plus - even_product)
    assert_zero(
        f"A_{{{r},{s}}} null-doublet norm phase",
        direct_minus + sp.I * direct_plus,
    )
    if require_mismatches:
        if clean(direct_plus - literal_product) == 0:
            raise AssertionError(f"A_{{{r},{s}}}: literal lattice unexpectedly passed")
        if clean(direct_plus - odd_product) == 0:
            raise AssertionError(f"A_{{{r},{s}}}: odd lattice unexpectedly passed")
    return direct_plus, time.monotonic() - started


def certify_factorization(
    r: int,
    s: int,
    b: sp.Expr,
    lambda_i: sp.Expr,
    beta_j: sp.Expr,
):
    started = time.monotonic()
    level = r * s // 2
    human_passes = 0
    component_count = 0
    for null_slot, null_ground, spectator_ground, p_phi, eta in product(
        (2, 3), (0, 1), (0, 1), (0, 1), (1, -1)
    ):
        result = contract_ramond_null(
            r=r,
            s=s,
            p_phi=p_phi,
            eta=eta,
            b=b,
            lambda_i=lambda_i,
            beta_j=beta_j,
            null_ground=null_ground,
            spectator_ground=spectator_ground,
            null_slot=null_slot,
        )
        assert_zero(
            f"factorization ({r},{s}), slot={null_slot}, alpha={null_ground}, "
            f"gamma={spectator_ground}, p_phi={p_phi}, eta={eta}",
            result["generalized_residual"],
        )
        component_count += 1
        if result["residual"] == 0:
            human_passes += 1

    # Slot 3 has no plane-coordinate sign, so its p_phi=0 components always
    # match the printed formula.  Slot 2 adds eight more at even level.
    expected_human_passes = 16 if level % 2 == 0 else 8
    if human_passes != expected_human_passes:
        raise AssertionError(
            f"({r},{s}): expected {expected_human_passes} literal (5.10) "
            f"passes, got {human_passes}"
        )
    return component_count, human_passes, time.monotonic() - started


def run(symbolic_through: int, sampled_through: int) -> None:
    if sampled_through < symbolic_through:
        raise ValueError("sampled-through must be at least symbolic-through")
    b = sp.symbols("b", nonzero=True)
    lambda_i, beta_j = sp.symbols("lambda_i beta_j", nonzero=True)
    sample_b = sp.Rational(2, 3)
    sample_lambda = sp.Rational(5, 7)
    sample_beta = sp.Rational(4, 9)

    print("Extended Ramond A_rs and three-point factorization audit")
    print(f"  symbolic generic parameters through level {symbolic_through}")
    print(f"  exact sampled checks through level {sampled_through}")
    print("  generalized plane laws (N=rs/2):")
    print(
        "    null in slot 2 (z=1): rho = (-1)^N "
        "P_rs^{R,(-1)^p_phi eta} rho_shifted"
    )
    print(
        "    null in slot 3 (z=0): rho = "
        "P_rs^{R,(-1)^p_phi eta} rho_shifted"
    )

    for level in range(1, symbolic_through + 1):
        for r, s in ramond_labels_at_level(level):
            a_value, a_seconds = certify_a_rs(
                r, s, b, require_mismatches=True
            )
            count, human_passes, factor_seconds = certify_factorization(
                r, s, b, lambda_i, beta_j
            )
            print(
                f"PASS symbolic: level={level}, (r,s)=({r},{s}), "
                f"A_rs={a_value}, A-seconds={a_seconds:.2f}, "
                f"factor-components={count}, literal-5.10-passes={human_passes}, "
                f"factor-seconds={factor_seconds:.2f}"
            )

    for level in range(symbolic_through + 1, sampled_through + 1):
        for r, s in ramond_labels_at_level(level):
            a_value, a_seconds = certify_a_rs(
                r, s, sample_b, require_mismatches=False
            )
            count, human_passes, factor_seconds = certify_factorization(
                r, s, sample_b, sample_lambda, sample_beta
            )
            print(
                f"PASS exact sample: level={level}, (r,s)=({r},{s}), "
                f"b={sample_b}, lambda={sample_lambda}, beta_j={sample_beta}, "
                f"A_rs={a_value}, A-seconds={a_seconds:.2f}, "
                f"factor-components={count}, literal-5.10-passes={human_passes}, "
                f"factor-seconds={factor_seconds:.2f}"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbolic-through", type=int, default=3)
    parser.add_argument("--sampled-through", type=int, default=4)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    run(arguments.symbolic_through, arguments.sampled_through)
