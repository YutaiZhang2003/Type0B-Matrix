#!/usr/bin/env python3
"""Focused symbolic audit of the Ramond PBW formulas (5.3), (5.6)--(5.10)."""

from __future__ import annotations

from itertools import product
from pathlib import Path
import sys

import sympy as sp


HERE = Path(__file__).resolve().parent
NSRR = HERE.parent / "nsrr"
if str(NSRR) not in sys.path:
    sys.path.insert(0, str(NSRR))

from ramond_pbw_generalized_ward import (
    GeneralizedNRRWard,
    RamondPBWModule,
    clean,
    contract_level_one_null,
    fixed_beta_inverse_null_norm,
    fusion_polynomial_510,
    inverse_null_product_59,
    pole_equation_residual,
    ramond_degenerate_data,
)


def assert_zero(label: str, expression: sp.Expr) -> None:
    residual = clean(expression)
    if residual != 0:
        raise AssertionError(f"{label}: expected zero, got {residual}")


def vector_column(basis, vector) -> sp.Matrix:
    return sp.Matrix([vector.get(state, sp.S.Zero) for state in basis])


def run_audit() -> None:
    b = sp.symbols("b", nonzero=True)
    lambda_i, beta_j = sp.symbols("lambda_i beta_j")

    print("Ramond PBW / generalized NS-R-R Ward audit")
    print("  convention: Human Notes (5.1), (5.3), (5.6)-(5.10)")
    print("  Ward system: generalized only; p_phi is mandatory")

    # (5.3): the four raw component normalizations are all one.  The
    # fixed-form combinations are shifted by p_phi, but the components are
    # not renormalized.
    for second, third in product((0, 1), repeat=2):
        assert_zero(
            "(5.3) component normalization",
            GeneralizedNRRWard.component_normalization(second, third) - 1,
        )
    print("\n(5.3) four component ground normalizations: PASS (all equal 1)")

    for r, s in ((2, 1), (1, 2)):
        pole = ramond_degenerate_data(r, s, b)
        assert_zero(f"({r},{s}) pole quadratic", pole_equation_residual(r, s, b))
        assert_zero(
            f"({r},{s}) shifted weight",
            pole["h_shifted"] - pole["h"] - sp.Rational(r * s, 2),
        )
        module = RamondPBWModule(pole["h"], pole["beta"], pole["c"])

        nulls = {}
        for parity in (0, 1):
            basis, gram = module.gram_matrix(1, parity)
            null = module.normalized_null_vector(1, parity)
            nulls[parity] = null
            residual = gram * vector_column(basis, null)
            for entry in residual:
                assert_zero(f"({r},{s}) parity-{parity} Gram null", entry)

        basis_even, gram_even = module.gram_matrix(1, 0)
        print(f"\n(5.6)-(5.8) labels (r,s)=({r},{s}): PASS")
        print(f"  beta_rs = {pole['beta']}")
        print(f"  h_rs = {pole['h']}")
        print("  even Gram basis =", [state.label() for state in basis_even])
        print("  even Gram matrix =")
        sp.print_latex(gram_even)
        print("  chi^+ =", {state.label(): value for state, value in nulls[0].items()})
        print("  chi^- =", {state.label(): value for state, value in nulls[1].items()})

        direct_a = fixed_beta_inverse_null_norm(r, s, b, nulls[0])
        literal_a = inverse_null_product_59(r, s, b, lattice="literal")
        even_a = inverse_null_product_59(r, s, b, lattice="even")
        odd_a = inverse_null_product_59(r, s, b, lattice="odd")
        assert_zero(f"({r},{s}) even-lattice (5.9)", direct_a - even_a)
        if clean(direct_a - literal_a) == 0:
            raise AssertionError("literal p+q in Z product unexpectedly passed")
        print("  (5.9) direct fixed-beta inverse norm =", direct_a)
        print("         printed literal p+q in Z       =", literal_a, "[MISMATCH]")
        print("         p+q even sublattice            =", even_a, "[PASS]")
        print("         p+q odd sublattice             =", odd_a, "[MISMATCH]")

        print("  (5.10) direct generalized-Ward contractions:")
        for p_phi, eta in product((0, 1), (1, -1)):
            result = contract_level_one_null(
                r=r,
                s=s,
                p_phi=p_phi,
                eta=eta,
                b=b,
                lambda_i=lambda_i,
                beta_j=beta_j,
            )
            parity_corrected = fusion_polynomial_510(
                r,
                s,
                lambda_i,
                beta_j,
                b,
                (-1) ** p_phi * eta,
            )
            assert_zero(
                f"({r},{s}) generalized parity pattern",
                result["direct"] + parity_corrected,
            )
            status = "PASS" if result["residual"] == 0 else "MISMATCH"
            print(
                f"    p_phi={p_phi}, eta={eta:+d}: direct={result['direct']}"
            )
            print(
                f"      printed P(eta)={result['predicted']} [{status}]; "
                f"direct=-P((-1)^p_phi eta)"
            )

    # An explicit sign anchor: changing p_phi changes epsilon even with all
    # descendant words and the third ground state held fixed.
    sample = ramond_degenerate_data(2, 1, b)
    epsilons = []
    for p_phi in (0, 1):
        evaluator = GeneralizedNRRWard(
            p_phi=p_phi,
            form_parity=p_phi,
            eta=1,
            h_ns=sp.Symbol("h_i"),
            h_second=sample["h"],
            h_third=sp.Symbol("h_j"),
            beta_second=sample["beta"],
            beta_third=beta_j,
            central_charge=sample["c"],
        )
        epsilons.append(evaluator.epsilon((), (), 0))
    assert_zero("intrinsic-primary parity flips epsilon", epsilons[0] + epsilons[1])
    print("\nGeneralized parity anchor: PASS")
    print(f"  epsilon(p_phi=0)={epsilons[0]}, epsilon(p_phi=1)={epsilons[1]}")
    print("\nAudit completed without suppressing the (5.9)/(5.10) mismatches.")


if __name__ == "__main__":
    run_audit()
