#!/usr/bin/env python3
"""Finite coefficient and local Ward checks; no momentum or moduli integration."""
import argparse
from fractions import Fraction
import hashlib
from itertools import product
import json
from pathlib import Path

import numpy as np

from literature_component_blocks import ComponentBlocks, PaperConstants, sewn_integrand
from literature_double_virasoro import (
    B_C3, BranchVertex, LiteratureDoubleVirasoroBlocks, auxiliary_series, branch_state,
)
from spin23_two_virasoro_ramond import _embedded_l_minus_one_action


MOMENTA = (.21, .39, .31, .43)


def compare_coefficients(family, order=4):
    reference = ComponentBlocks(family, MOMENTA, .71, order)
    candidate = LiteratureDoubleVirasoroBlocks(family, MOMENTA, .71, order)
    count, maximum, worst = 0, 0., None
    for external, parity, left, right in product(
            product((0, 1), repeat=4), (0, 1),
            (1,) if family == "mixed_ns" else (1, -1), (1, -1)):
        expected = reference.coefficients(external, parity, left, right)
        actual = candidate.coefficients(external, parity, left, right)
        error = float(np.max(abs(actual-expected)))
        if error > maximum:
            maximum, worst = error, (external, parity, left, right)
        count += 1
    return dict(family=family, component_blocks=count,
                coefficients=count*(order+1), maximum_twice_level=order,
                maximum_absolute_error=maximum, worst_component=worst)


def _acted_vector(branch, copy):
    answer = {}
    for state, coefficient in zip(branch.basis, branch.coefficients):
        for target, action in _embedded_l_minus_one_action(
                state, sector=branch.sector, copy=copy, b=B_C3,
                h=branch.super_weight, c=branch.super_central_charge).items():
            answer[target] = answer.get(target, 0)+coefficient*action
    return answer


def check_embedded_wards(order):
    """Check both actual embedded L_-1 operators at all three punctures."""
    momenta = (.31, .37, .43)
    sectors = ("NS", "R", "R") if order == "NR" else ("R", "R", "NS")
    ns_slot = sectors.index("NS")
    r_slots = [j for j, s in enumerate(sectors) if s == "R"]
    count, maximum = 0, 0.
    for ns, r1, r2, a, b, eta in product(
            (Fraction(0), Fraction(-1, 2), Fraction(1, 2)),
            (Fraction(-1, 4), Fraction(1, 4)),
            (Fraction(-1, 4), Fraction(1, 4)), (0, 1), (0, 1), (1, -1)):
        labels, parities = [None]*3, [None]*3
        labels[ns_slot] = ns
        for slot, n, bit in zip(r_slots, (r1, r2), (a, b)):
            labels[slot], parities[slot] = n, bit
        branches = tuple(branch_state(s, p, n, bit) for s, p, n, bit in
                         zip(sectors, momenta, labels, parities))
        vertex = BranchVertex(order, momenta, (int(2*ns)+a+b) % 2, eta)
        vectors = [dict(zip(v.basis, v.coefficients)) for v in branches]
        primary = vertex.evaluate_vectors(vectors)
        for copy, slot in product((1, 2), range(3)):
            h = [getattr(v.parameters, "h_"+str(copy)) for v in branches]
            factor = (h[0]+h[1]-h[2], h[0]-h[1]-h[2], h[2]+h[1]-h[0])[slot]
            changed = vectors.copy()
            changed[slot] = _acted_vector(branches[slot], copy)
            error = abs(vertex.evaluate_vectors(changed)-factor*primary)
            maximum = max(maximum, float(error))
            count += 1
    return dict(order=order, identities=count, maximum_absolute_error=maximum)


def check_physical_tensors():
    constants = PaperConstants(32)
    rows = []
    cases = {
        "mixed_ns": (1, 0, (1, 0), (0, 0)),
        "mixed_r": ((1, 0), 1, 0, (0, 0)),
        "rrrr": (1, 0, 1, 0),
    }
    encode = lambda z: [float(z.real), float(z.imag)]
    for family, external in cases.items():
        for P in (.71, -.71):
            old = ComponentBlocks(family, MOMENTA, P, 4)
            new = LiteratureDoubleVirasoroBlocks(family, MOMENTA, P, 4)
            for z in (.5+.1j, .5-.1j):
                expected = sewn_integrand(old, constants, P, z, external)
                actual = sewn_integrand(new, constants, P, z, external)
                error = float(abs(actual-expected)/max(abs(actual), abs(expected), 1e-300))
                rows.append(dict(family=family, external=external, P=P, z=encode(z),
                                 native=encode(expected), double_virasoro=encode(actual),
                                 relative_error=error))
    return rows


def run():
    coefficients = [compare_coefficients(family)
                    for family in ("mixed_ns", "mixed_r", "rrrr")]
    wards = [check_embedded_wards(order) for order in ("NR", "RN")]
    physical = check_physical_tensors()
    assert max(r["maximum_absolute_error"] for r in coefficients+wards) < 2e-11
    assert max(r["relative_error"] for r in physical) < 2e-11
    root = Path(__file__).resolve().parent
    source_names = (
        "literature_component_blocks.py", "literature_double_virasoro.py",
        "audit_literature_double_virasoro.py",
        "spin23_two_virasoro_ramond.py", "virasoro_sphere_c_recursion.py",
        "ns_algebra/ns_three_point_tensor.py",
        "ramond_algebra/ns_rr_three_point_tensor.py",
    )
    return dict(
        scope="c=3, real signed nonzero Ramond momenta, all external multiplet components",
        method="Finite highest-vector branching and ordinary Virasoro c-recursion, followed by auxiliary-block removal and the exact linear-to-native sign/dual map",
        integration="none: neither internal momentum nor worldsheet moduli integrated",
        human_note_null_formulas_used=False,
        new_crossing_campaign=False,
        b=[B_C3.real, B_C3.imag], external_momenta=MOMENTA,
        coefficients=coefficients, embedded_virasoro_wards=wards,
        physical_tensor_comparisons=physical,
        auxiliary_series={family: [float(x.real) for x in auxiliary_series(family, 3)]
                          for family in ("mixed_ns", "mixed_r", "rrrr")},
        source_sha256={name: hashlib.sha256((root/name).read_bytes()).hexdigest()
                       for name in source_names},
        limitation="Finite coefficient agreement does not certify high-order numerical stability, a generic-central-charge continuation, or the production heterotic amplitude.",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    result = run()
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(result, indent=2)+"\n")
    for row in result["coefficients"]+result["embedded_virasoro_wards"]:
        print(row)
    print("Maximum physical tensor relative error:",
          max(row["relative_error"] for row in result["physical_tensor_comparisons"]))
