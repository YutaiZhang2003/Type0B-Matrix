#!/usr/bin/env python3
"""Degree and work estimates for screening reconstruction.

The estimates in this file concern the ordinary, charge-preserving
Pfaffian/Selberg chart.  They never count or construct super-Virasoro
descendants.  A generic fixed Ramond chiral form can also require a reflected
external chart.  The exact reflection recurrence lives in ``reflection/``;
if its level block has dimension ``d``, materialising that block costs
``O(d**3)`` time and ``O(d**2)`` memory.  Thus the estimates printed here are
*not* a claimed polynomial bound for the complete reflected problem.
"""

from __future__ import annotations

import argparse
from fractions import Fraction


def ns_chi_count(branch_label) -> int:
    n = abs(Fraction(branch_label))
    count = 2 * n
    if count.denominator != 1:
        raise ValueError("NS branch label must be half-integral.")
    return count.numerator


def ramond_mode_count(branch_label) -> int:
    n = abs(Fraction(branch_label))
    count = 2 * n - Fraction(1, 2)
    if count.denominator != 1 or count < 0:
        raise ValueError("Ramond branch label must lie in Z/2+1/4.")
    return count.numerator


def ramond_chi_count(branch_label, parity: int) -> int:
    """Length of the exact 2016 ordered chi string."""

    parity = int(parity)
    if parity not in (0, 1):
        raise ValueError(parity)
    base = ramond_mode_count(branch_label) + 1
    return base if base % 2 == parity else base + 1


def ns_endpoint_grade2(branch_label) -> int:
    """Twice the free-field endpoint grade of the NS chi string."""

    n = abs(Fraction(branch_label))
    grade2 = 4 * n * n
    if grade2.denominator != 1:
        raise ValueError("NS branch label must be half-integral.")
    return grade2.numerator


def ramond_endpoint_grade(branch_label) -> int:
    """Integer free-field endpoint grade of a Ramond chi string."""

    mode_count = ramond_mode_count(branch_label)
    return mode_count * (mode_count + 1) // 2


def momentum_degree_bound(n1, n2, n3) -> int:
    """NS--R--R degree bound in any one external momentum.

    In paper labels k_i=2 n_i this is

        k_1^2+k_2^2+k_3^2-1/2.

    It is integral because k_1 is integral and k_2,k_3 are half-integral.
    """

    k1, k2, k3 = (2 * Fraction(value) for value in (n1, n2, n3))
    degree = k1 * k1 + k2 * k2 + k3 * k3 - Fraction(1, 2)
    if degree.denominator != 1:
        raise ValueError((n1, n2, n3, degree))
    return degree.numerator


def estimate(n1, n2, n3, epsilon2=1, epsilon3=1):
    degree = momentum_degree_bound(n1, n2, n3)
    chi_modes = (
        ns_chi_count(n1)
        + ramond_chi_count(n2, epsilon2)
        + ramond_chi_count(n3, epsilon3)
    )
    nodes = degree + 1
    # Conservative dense arithmetic estimates.  The actual N=2 Selberg
    # backend is a linear-size Gamma product, and barycentric interpolation
    # uses O(nodes^2) arithmetic when all exact weights are constructed.
    pfaffian_ops = nodes * chi_modes**3
    interpolation_ops = nodes**2
    return {
        "degree": degree,
        "interpolation_nodes": nodes,
        "chi_modes": chi_modes,
        "ns_endpoint_grade2": ns_endpoint_grade2(n1),
        "ramond2_endpoint_grade": ramond_endpoint_grade(n2),
        "ramond3_endpoint_grade": ramond_endpoint_grade(n3),
        "dense_pfaffian_ops_upper_bound": pfaffian_ops,
        "interpolation_ops_upper_bound": interpolation_ops,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("n1", nargs="?", default="2")
    parser.add_argument("n2", nargs="?", default="7/4")
    parser.add_argument("n3", nargs="?", default="7/4")
    parser.add_argument("--epsilon2", type=int, default=1)
    parser.add_argument("--epsilon3", type=int, default=1)
    args = parser.parse_args()
    result = estimate(
        Fraction(args.n1),
        Fraction(args.n2),
        Fraction(args.n3),
        args.epsilon2,
        args.epsilon3,
    )
    for key, value in result.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
