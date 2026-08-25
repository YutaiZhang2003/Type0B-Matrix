#!/usr/bin/env python3
"""Colored bifundamental factors on the branch staircase diagrams.

This is an exploratory implementation of Eq. (4.1) of arXiv:1210.7454.
It keeps the two Ramond holonomies separate.  The branch diagrams are

    R:  (delta_M, delta_M),       n=(2M+1)/4,
    NS: (delta_M, delta_{M-1}),   n=M/2,

where delta_M=(M,M-1,...,1).  No state/Ward evaluator is imported.
"""

from __future__ import annotations

import itertools
import sympy as sp


def staircase(size: int) -> tuple[int, ...]:
    return tuple(range(int(size), 0, -1))


def boxes(diagram: tuple[int, ...]):
    for row, length in enumerate(diagram, start=1):
        for column in range(1, length + 1):
            yield row, column


def arm(diagram: tuple[int, ...], box: tuple[int, int]) -> int:
    row, column = box
    return diagram[row - 1] - column


def leg(diagram: tuple[int, ...], box: tuple[int, int]) -> int:
    row, column = box
    return sum(length >= column for length in diagram) - row


def e_factor(
    first: tuple[int, ...],
    second: tuple[int, ...],
    momentum,
    box: tuple[int, int],
    b,
):
    return momentum - leg(second, box) / b + (arm(first, box) + 1) * b


def z_bif(
    alpha,
    p_left,
    diagrams_left,
    charges_left,
    p_right,
    diagrams_right,
    charges_right,
    b,
):
    """Return the colored bifundamental product of arXiv:1210.7454."""

    q_background = b + 1 / b
    left_momenta = (p_left, -p_left)
    right_momenta = (p_right, -p_right)
    answer = sp.Integer(1)
    for i, j in itertools.product(range(2), repeat=2):
        y_i = diagrams_right[i]
        w_j = diagrams_left[j]
        q_i = charges_right[i]
        u_j = charges_left[j]
        for square in boxes(y_i):
            parity = leg(w_j, square) + arm(y_i, square) + 1
            if parity % 2 == (u_j - q_i) % 2:
                e = e_factor(
                    y_i,
                    w_j,
                    right_momenta[i] - left_momenta[j],
                    square,
                    b,
                )
                answer *= q_background - e - alpha
        for square in boxes(w_j):
            parity = leg(y_i, square) + arm(w_j, square) + 1
            if parity % 2 == (q_i - u_j) % 2:
                e = e_factor(
                    w_j,
                    y_i,
                    left_momenta[j] - right_momenta[i],
                    square,
                    b,
                )
                answer *= e - alpha
    return sp.factor(answer)


def ramond_pair(branch_label):
    branch_label = sp.Rational(branch_label)
    mode_count = 2 * branch_label - sp.Rational(1, 2)
    if not mode_count.is_integer or mode_count < 0:
        raise ValueError("positive Ramond branch label required")
    diagram = staircase(int(mode_count))
    return (diagram, diagram)


RAMOND_HOLONOMIES = ((0, 1), (1, 0))


def hard_matrix():
    b, p1, p2, p3 = sp.symbols("b P_1 P_2 P_3", nonzero=True)
    diagrams = ramond_pair(sp.Rational(3, 4))
    alpha = b / 2 + 1 / (2 * b) + p1
    matrix = sp.Matrix(
        [
            [
                z_bif(alpha, p2, diagrams, u, p3, diagrams, q, b)
                for q in RAMOND_HOLONOMIES
            ]
            for u in RAMOND_HOLONOMIES
        ]
    )
    return b, p1, p2, p3, matrix


if __name__ == "__main__":
    _, _, _, _, result = hard_matrix()
    print("colored hard matrix")
    sp.print_latex(result)
    for row in result.tolist():
        print(*(sp.factor(entry) for entry in row), sep="\n")
