#!/usr/bin/env python3
"""Exact, Ward-free benchmark of the resolved screening pipeline.

The benchmark compares the coefficient-determinant identity with a direct
Pfaffian quotient at rational screening positions.  ``--large`` includes
the requested ``(v_2,W_{7/4},W_{7/4})`` example (eleven screenings).
No super-Virasoro state, Gram matrix, or Ward evaluator is imported.
"""

from __future__ import annotations

import argparse
import time

import sympy as sp

from .boundary_zero_modes import (
    projected_determinant_constant,
    projected_selberg_ratio,
    projected_vandermonde_constant,
)


QUARTER = sp.Rational(1, 4)


def _run(labels, check_second_sample=False, integrate=False):
    count = int(2 * sum(labels))
    began = time.perf_counter()
    determinant = projected_determinant_constant(*labels, 0, 1)
    determinant_time = time.perf_counter() - began

    began = time.perf_counter()
    sampled = projected_vandermonde_constant(*labels, 0, 1, sample_shift=0)
    sample_time = time.perf_counter() - began
    if sp.simplify(determinant - sampled) != 0:
        raise AssertionError((labels, determinant, sampled))

    second = ""
    if check_second_sample:
        began = time.perf_counter()
        sampled_again = projected_vandermonde_constant(
            *labels, 0, 1, sample_shift=1
        )
        second_time = time.perf_counter() - began
        if sp.simplify(determinant - sampled_again) != 0:
            raise AssertionError((labels, determinant, sampled_again))
        second = f", second Pfaffian sample={second_time:.3f}s"

    print(
        f"levels={labels}, N={count}: C={determinant}; "
        f"coefficient determinant={determinant_time:.3f}s, "
        f"direct Pfaffian sample={sample_time:.3f}s{second}"
    )
    if integrate:
        # A generic nonresonant rational point.  It avoids Gamma poles in
        # the analytically continued Selberg products, so the displayed
        # value can be evaluated directly without a regulator.
        b = sp.Rational(7, 5)
        p2 = sp.Rational(2, 7)
        p3 = sp.Rational(3, 11)
        q = b + 1 / b
        A = -b * (q / 2 + p3) - sp.Rational(1, 2)
        B = -b * (q / 2 + p2) - sp.Rational(1, 2)
        g = -b * q / 2
        began = time.perf_counter()
        ratio = projected_selberg_ratio(*labels, 0, 1, A, B, g)
        integral_time = time.perf_counter() - began
        print(
            f"  exact Selberg ratio at b=7/5,P2=2/7,P3=3/11: "
            f"{sp.N(ratio, 10)} ({integral_time:.3f}s)"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--large",
        action="store_true",
        help="also run the eleven-screening v2,W7/4,W7/4 benchmark",
    )
    parser.add_argument(
        "--two-samples",
        action="store_true",
        help="audit the Vandermonde quotient at a second rational point",
    )
    parser.add_argument(
        "--selberg",
        action="store_true",
        help="also evaluate the complete exact Selberg-product ratio",
    )
    arguments = parser.parse_args()

    cases = [
        (sp.Integer(0), QUARTER, QUARTER),
        (sp.Integer(0), 3 * QUARTER, 3 * QUARTER),
        (sp.Integer(0), 5 * QUARTER, 5 * QUARTER),
    ]
    if arguments.large:
        cases.append((sp.Integer(2), 7 * QUARTER, 7 * QUARTER))
    for labels in cases:
        _run(
            labels,
            check_second_sample=arguments.two_samples,
            integrate=arguments.selberg,
        )


if __name__ == "__main__":
    main()
