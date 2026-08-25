"""Literal Fock-path audit of the zero-mode-resolved Pfaffian.

This validation file may import the stored 2016 path expansion.  Production
screening modules do not.  It compares fermionic screening integrands before
any Selberg integral, so a discrepancy localizes spin/Koszul errors without
mixing in bosonic normalization.
"""

from __future__ import annotations

import itertools

import sympy as sp

from python.nsrr_chi_branching.nsrr_chi_formula import ramond_fock_paths

from .boundary_zero_modes import (
    ExternalRow,
    I,
    _ground,
    _local_one_point,
    _screen_kernel,
    external_pair,
    resolved_contour_laurent,
    _fixed_boundary_correlator,
    FOCK_TO_SCBLOCK_MINUS,
)
from .core import pfaffian_recursive
from .screening_pfaffian import _row_pair_with_screening


def ns_fock_paths(branch_label):
    branch_label = sp.Rational(branch_label)
    modes = tuple(
        sp.Rational(value, 2)
        for value in range(int(4 * branch_label - 1), 0, -2)
    )
    answer = []
    for count in range(len(modes) + 1):
        for physical in itertools.combinations(modes, count):
            physical = tuple(sorted(physical, reverse=True))
            auxiliary = tuple(mode for mode in modes if mode not in physical)
            crossings = sum(p > a for p in physical for a in auxiliary)
            answer.append((auxiliary, physical, (-I) ** len(physical) * (-1) ** crossings))
    return tuple(answer)


def _pair(left, right):
    kind_left, item_left = left
    kind_right, item_right = right
    if kind_left == kind_right == "screening":
        return _screen_kernel(item_left, item_right)
    if kind_left == "external" and kind_right == "screening":
        return _row_pair_with_screening(item_left.leg, item_left.mode, item_right)
    if kind_left == "screening" and kind_right == "external":
        return _row_pair_with_screening(item_right.leg, item_right.mode, item_left)
    return external_pair(item_left.leg, item_left.mode, item_right.leg, item_right.mode)


def sector_value(objects, sector, form_parity, eta, ground2, ground3):
    objects = tuple(objects)
    if len(objects) % 2 == 0:
        matrix = [[sp.Integer(0) for _ in objects] for _ in objects]
        for left in range(len(objects)):
            for right in range(left + 1, len(objects)):
                value = _pair(objects[left], objects[right])
                matrix[left][right] = value
                matrix[right][left] = -value
        return _ground(sector, form_parity, eta, ground2, ground3) * pfaffian_recursive(matrix)

    answer = sp.Integer(0)
    for position, (kind, item) in enumerate(objects):
        if kind == "screening":
            leg, mode = "screen", 0
        else:
            leg, mode = item.leg, item.mode
        if leg == "inf":
            ground_value = _ground(sector, 1 - form_parity, eta, ground2, ground3)
        elif leg == "one":
            ground_value = _ground(sector, form_parity, eta, 1 - ground2, ground3)
        else:
            ground_value = _ground(sector, form_parity, eta, ground2, 1 - ground3)
        remaining = objects[:position] + objects[position + 1 :]
        matrix = [[sp.Integer(0) for _ in remaining] for _ in remaining]
        for left in range(len(remaining)):
            for right in range(left + 1, len(remaining)):
                value = _pair(remaining[left], remaining[right])
                matrix[left][right] = value
                matrix[right][left] = -value
        answer += (
            (-1) ** position
            * _local_one_point(leg, mode)
            * ground_value
            * pfaffian_recursive(matrix)
        )
    return answer


def literal_integrand(n1, n2, n3, form_parity, eta, screenings, grouped=False):
    n1, n2, n3 = map(sp.Rational, (n1, n2, n3))
    screenings = tuple(screenings)
    external_parity = int(2 * n1) + int(2 * n2 + sp.Rational(1, 2)) + int(
        2 * n3 + sp.Rational(1, 2)
    )
    auxiliary_form = (external_parity - int(form_parity)) % 2
    physical_screening_form = int(form_parity) % 2
    answer = sp.Integer(0)
    groups = {}
    for auxiliary1, physical1, coefficient1 in ns_fock_paths(n1):
        for state2, coefficient2 in ramond_fock_paths(n2, int(2 * n2 + sp.Rational(1, 2)) % 2):
            auxiliary2, ground_a2, physical2, ground_p2 = state2
            for state3, coefficient3 in ramond_fock_paths(n3, int(2 * n3 + sp.Rational(1, 2)) % 2):
                auxiliary3, ground_a3, physical3, ground_p3 = state3
                auxiliary_objects = (
                    tuple(("external", ExternalRow("inf", mode, 1, 0)) for mode in reversed(auxiliary1))
                    + tuple(("external", ExternalRow("one", mode, 1, 0)) for mode in auxiliary2)
                    + tuple(("external", ExternalRow("zero", mode, 1, 0)) for mode in auxiliary3)
                )
                physical_objects = (
                    tuple(("external", ExternalRow("inf", mode, 0, 1)) for mode in reversed(physical1))
                    + tuple(("external", ExternalRow("one", mode, 0, 1)) for mode in physical2)
                    + tuple(("screening", value) for value in screenings)
                    + tuple(("external", ExternalRow("zero", mode, 0, 1)) for mode in physical3)
                )
                auxiliary_value = sector_value(
                    auxiliary_objects,
                    "auxiliary",
                    auxiliary_form,
                    eta,
                    ground_a2,
                    ground_a3,
                )
                physical_value = sector_value(
                    physical_objects,
                    "physical",
                    physical_screening_form,
                    eta,
                    ground_p2,
                    ground_p3 ^ (len(screenings) % 2),
                )
                physical_parity1 = len(physical1) % 2
                physical_parity2 = (len(physical2) + ground_p2) % 2
                auxiliary_parity2 = (len(auxiliary2) + ground_a2) % 2
                auxiliary_parity3 = (len(auxiliary3) + ground_a3) % 2
                koszul = (-1) ** (
                    physical_parity1 * (auxiliary_parity2 + auxiliary_parity3)
                    + physical_parity2 * auxiliary_parity3
                )
                term = (
                    koszul
                    * coefficient1
                    * coefficient2
                    * coefficient3
                    * auxiliary_value
                    * physical_value
                    * FOCK_TO_SCBLOCK_MINUS ** (ground_p2 + ground_p3)
                )
                answer += term
                key = (ground_a2, ground_a3)
                groups[key] = groups.get(key, 0) + term
    return groups if grouped else answer


def audit():
    labels = (0, sp.Rational(3, 4), sp.Rational(3, 4))
    sample = (sp.Rational(1, 7), sp.Rational(2, 7), sp.Rational(4, 7))
    for form_parity in (0, 1):
        for eta in (1, -1):
            literal = sp.cancel(literal_integrand(*labels, form_parity, eta, sample))
            ts, compressed = resolved_contour_laurent(*labels, form_parity, eta)
            compressed = sp.cancel(compressed.subs(dict(zip(ts, sample))))
            difference = sp.simplify(literal - compressed)
            print(form_parity, eta, "literal=", literal, "difference=", difference)
            if form_parity == 0 and eta == 1:
                groups = literal_integrand(
                    *labels, form_parity, eta, sample, grouped=True
                )
                for key in itertools.product((0, 1), repeat=2):
                    ga2, ga3 = key
                    gp2, gp3 = 1 - ga2, 1 - ga3
                    maximum2 = int(2 * labels[1] - sp.Rational(1, 2))
                    maximum3 = int(2 * labels[2] - sp.Rational(1, 2))
                    ref2 = (
                        (-1) ** (maximum2 * (maximum2 + 1) // 2)
                        / sp.sqrt(2)
                        * (1 if ga2 else -I)
                    )
                    ref3 = (
                        (-1) ** (maximum3 * (maximum3 + 1) // 2)
                        / sp.sqrt(2)
                        * (1 if ga3 else -I)
                    )
                    zero = ref2 * ref3 * (-1) ** (gp2 * (maximum3 + ga3))
                    zero *= FOCK_TO_SCBLOCK_MINUS ** (gp2 + gp3)
                    fixed = _fixed_boundary_correlator(
                        *labels,
                        form_parity,
                        eta,
                        sample,
                        ga2,
                        ga3,
                    )
                    compressed_group = sp.simplify(zero * fixed)
                    print("  ", key, sp.simplify(groups.get(key, 0)), compressed_group)


if __name__ == "__main__":
    audit()
