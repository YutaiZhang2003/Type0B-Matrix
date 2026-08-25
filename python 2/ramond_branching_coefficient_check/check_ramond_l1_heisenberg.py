#!/usr/bin/env python3
"""Check the Ramond physical-L1/current-descendant identity at low labels.

This check is independent of the double-Virasoro PBW solve.  It compares

    L_1 W_n

computed directly on the finite chi string with

    H_(4 n - 3)^(-2)(J_-) W_(n-1)

in the common auxiliary-fermion times physical-fermion Fock basis.  The
comparison is up to the single normalization coefficient kappa_n^epsilon.
Only the positive branch is needed; the negative branch follows by the
reflected realization.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import math
import sys

import sympy as sp


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import check_ramond_branching as branch  # noqa: E402
import decompose_physical_l1 as reduction  # noqa: E402


I = sp.I


def current_j_minus(expression, level):
    """Apply the positive-branch current mode J_-level.

    With Psi=(psi-i eta)/sqrt(2), the convention [J_0,Psi]=Psi is

        J_m = i sum_r :psi_(m-r) eta_r: .

    The two real fermions have no mutual contraction, so their normal-ordered
    cross bilinear equals the displayed ordered product.  The tensor sign is
    the sign acquired when the physical fermion crosses the auxiliary state.
    """

    level = int(level)
    if level <= 0:
        raise ValueError("A current lowering-mode level must be positive.")
    mode = -level
    answer = {}
    for full_state, outer in expression.items():
        auxiliary, physical = reduction._split_product_state(full_state)
        auxiliary_modes, auxiliary_ground = auxiliary
        _, physical_modes, _ = physical

        # These are all r for which eta_r and psi_(mode-r) can both act.
        summation_modes = set(range(mode, 1))
        summation_modes.update(physical_modes)
        summation_modes.update(mode - value for value in auxiliary_modes)
        tensor_sign = (-1) ** (
            (len(auxiliary_modes) + auxiliary_ground) % 2
        )

        for summation_mode in summation_modes:
            physical_final, physical_coefficient = branch.apply_fermion(
                summation_mode, physical, realization=-1
            )
            if not physical_coefficient:
                continue
            auxiliary_final, auxiliary_coefficient = branch.apply_auxiliary(
                mode - summation_mode, auxiliary
            )
            if not auxiliary_coefficient:
                continue
            reduction.add_term(
                answer,
                reduction._join_product_state(auxiliary_final, physical_final),
                outer
                * I
                * tensor_sign
                * auxiliary_coefficient
                * physical_coefficient,
            )
    return answer


def h_polynomial_descendant(expression, degree, alpha):
    """Apply H_degree^(alpha)(J_-) using its partition expansion."""

    degree = int(degree)
    alpha = sp.Integer(alpha)
    answer = {}
    for partition in branch.partitions(degree):
        multiplicities = Counter(partition)
        z_partition = sp.Integer(1)
        for part, count in multiplicities.items():
            z_partition *= part**count * math.factorial(count)
        coefficient = alpha ** len(partition) / z_partition
        image = expression
        for part in reversed(partition):
            image = current_j_minus(image, part)
        for state, value in image.items():
            reduction.add_term(answer, state, coefficient * value)
    return answer


def proportionality_coefficient(target, source):
    """Return target/source and certify an exact zero residual."""

    if not source:
        raise AssertionError("The proposed Heisenberg descendant is zero.")
    pivot = next(iter(sorted(source, key=str)))
    coefficient = sp.factor(sp.cancel(target.get(pivot, 0) / source[pivot]))
    states = set(target).union(source)
    residual = {
        state: sp.factor(
            sp.cancel(target.get(state, 0) - coefficient * source.get(state, 0))
        )
        for state in states
    }
    residual = {state: value for state, value in residual.items() if value != 0}
    if residual:
        raise AssertionError(
            f"The current-descendant residual has {len(residual)} nonzero terms: "
            f"{residual}"
        )
    return coefficient


def check_positive_label(branch_label, parity):
    """Check one positive-label identity and return its exact kappa."""

    branch_label = sp.Rational(branch_label)
    if branch_label < sp.Rational(3, 4):
        raise ValueError("The first nonzero case is n=3/4.")
    degree = int(4 * branch_label - 3)
    lower_label = branch_label - 1
    target = reduction.physical_l1_positive_raw(branch_label, parity)
    lower = reduction.primary_in_fixed_fock(
        lower_label,
        parity,
        q_value=branch.Q,
        momentum=branch.P,
    )
    current_descendant = h_polynomial_descendant(lower, degree, alpha=-2)
    coefficient = proportionality_coefficient(target, current_descendant)
    return degree, coefficient, len(target), len(current_descendant)


def main():
    for branch_label in (
        sp.Rational(3, 4),
        sp.Rational(5, 4),
        sp.Rational(7, 4),
    ):
        for parity in (0, 1):
            degree, coefficient, target_terms, descendant_terms = (
                check_positive_label(branch_label, parity)
            )
            print(
                f"n={branch_label}, parity={parity}, d={degree}: "
                f"L_1 W_n = ({coefficient}) H_d^(-2)(J_-) W_(n-1); "
                f"exact residual zero ({target_terms}/{descendant_terms} Fock terms)"
            )


if __name__ == "__main__":
    main()
