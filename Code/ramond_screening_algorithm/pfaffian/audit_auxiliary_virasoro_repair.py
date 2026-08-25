#!/usr/bin/env python3
"""Exact impact audit for the repaired auxiliary Virasoro evaluator.

Only the difference between the repaired and legacy auxiliary kernels is
contracted.  This is algebraically identical to evaluating every enlarged
three-point function twice, but avoids recomputing all unchanged terms.
"""

from __future__ import annotations

import argparse
import itertools

import sympy as sp

from .auxiliary_ising_kernel import (
    corrected_fermion_value_virasoro,
    grid,
    legacy_fermion_value_virasoro,
    stored_auxiliary_endpoints,
)


I = sp.I
SQRT2 = sp.sqrt(2)


def repaired_endpoint_deltas():
    """Return the nonzero repaired-minus-legacy endpoint differences."""

    answer = {}
    for endpoint in stored_auxiliary_endpoints():
        difference = sp.factor(
            corrected_fermion_value_virasoro(*endpoint)
            - legacy_fermion_value_virasoro(*endpoint)
        )
        if difference != 0:
            answer[endpoint] = difference
    return answer


def enlarged_three_point_delta(labels, discrete, sample, endpoint_deltas):
    """Contract the exact repaired-minus-legacy enlarged form."""

    n1, n2, n3 = map(sp.Rational, labels)
    epsilon2, epsilon3, form_parity, eta = map(int, discrete)
    b_value, p1, p2, p3 = map(sp.sympify, sample)
    q_value = sp.cancel(b_value + 1 / b_value)
    central_charge = sp.Rational(3, 2) + 3 * q_value**2
    h1 = (q_value**2 / 4 - p1**2) / 2
    h2 = sp.Rational(1, 16) + q_value**2 / 8 - p2**2 / 2
    h3 = sp.Rational(1, 16) + q_value**2 / 8 - p3**2 / 2
    evaluator = grid.PhysicalNRREvaluator(
        form_parity,
        eta,
        h1,
        h2,
        h3,
        p2 / SQRT2,
        p3 / SQRT2,
        central_charge,
    )
    first = grid.ns_components(n1, q_value, p1)
    second = grid.ramond_components(n2, epsilon2, q_value, p2)
    third = grid.ramond_components(n3, epsilon3, q_value, p3)
    ns_parity = int(2 * n1) % 2
    auxiliary_form_parity = (
        ns_parity + epsilon2 + epsilon3 - form_parity
    ) % 2

    components = [sp.Integer(0)] * 4
    for auxiliary1, word1, coefficient1 in first:
        physical_parity1 = grid.state_parity(word1)
        for (
            auxiliary2,
            ground2,
            word2,
            physical_ground2,
            coefficient2,
        ) in second:
            physical_parity2 = grid.state_parity(word2, physical_ground2)
            auxiliary_parity2 = (len(auxiliary2) + ground2) % 2
            for (
                auxiliary3,
                ground3,
                word3,
                physical_ground3,
                coefficient3,
            ) in third:
                endpoint = (
                    auxiliary_form_parity,
                    tuple(auxiliary1),
                    tuple(auxiliary2),
                    int(ground2),
                    tuple(auxiliary3),
                    int(ground3),
                )
                auxiliary_delta = endpoint_deltas.get(endpoint)
                if auxiliary_delta is None:
                    continue
                physical = evaluator.value(
                    word1,
                    word2,
                    physical_ground2,
                    word3,
                    physical_ground3,
                )
                if physical == 0:
                    continue
                auxiliary_parity3 = (len(auxiliary3) + ground3) % 2
                tensor_exponent = (
                    physical_parity1
                    * (auxiliary_parity2 + auxiliary_parity3)
                    + physical_parity2 * auxiliary_parity3
                )
                term = (
                    (-1) ** tensor_exponent
                    * coefficient1
                    * coefficient2
                    * coefficient3
                    * auxiliary_delta
                    * physical
                )
                for index, value in enumerate(
                    grid.quadratic_number_components(term)
                ):
                    components[index] += value
    return sp.factor(
        sp.cancel(
            components[0]
            + components[1] * SQRT2
            + I * components[2]
            + I * SQRT2 * components[3]
        )
    )


def audit(full_impact=False):
    endpoint_deltas = repaired_endpoint_deltas()
    if len(endpoint_deltas) != 128:
        raise AssertionError(len(endpoint_deltas))
    print("auxiliary endpoint changes: 128/512")
    if not full_impact:
        print(
            "full 432-restriction impact: not rerun after the final "
            "BPZ/cocycle/frame corrections"
        )
        return
    for sample_index, sample in enumerate(grid.SAMPLES, start=1):
        changed = []
        changed_levels = set()
        for labels in itertools.product(
            grid.GRID_NS_LEVELS,
            grid.GRID_R_LEVELS,
            grid.GRID_R_LEVELS,
        ):
            count = 0
            for discrete in grid.DISCRETE_CHOICES:
                difference = enlarged_three_point_delta(
                    labels, discrete, sample, endpoint_deltas
                )
                if difference != 0:
                    changed.append((labels, discrete, difference))
                    count += 1
            if count:
                changed_levels.add(tuple(labels))
        print(
            f"sample {sample_index}: {len(changed)}/432 changed restrictions; "
            f"changed level triples={tuple(sorted(changed_levels))}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-impact", action="store_true")
    arguments = parser.parse_args()
    audit(arguments.full_impact)
