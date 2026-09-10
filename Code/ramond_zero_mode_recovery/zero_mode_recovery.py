"""Zero-mode-normalized auxiliary sewing for the NSRR theta channel.

The insertion D=sqrt(2)*Theta*psi_0 acts as (-1)^ground on a Ramond
Fock state. Pi_0=(1+D)/2 fixes the auxiliary ground label on one R cut.
Its auxiliary block has constant 1 in the full parity algebra.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "double_virasoro" / "nsrr"))
from nsrr_genus2_block import (  # noqa: E402
    BRANCHING_RECURSION,
    HumanAuxiliaryThreePoint,
    ZERO_VECTOR,
    level_triples,
    ns_fermion_states,
    ramond_fermion_states,
)
from theta_star_algebra import star_multiply, theta_quadratic_sign  # noqa: E402


def auxiliary_with_zero_mode(*, maximum_total_twice_level, cut=1, insertion="fixed"):
    """Sew with identity, D, or Pi_0 on R slot 1 or 2 (NS slot is 0)."""
    if cut not in (1, 2) or insertion not in ("identity", "saturated", "fixed"):
        raise ValueError("choose an R cut and identity/saturated/fixed insertion")
    form = HumanAuxiliaryThreePoint((
        BRANCHING_RECURSION.FreeFieldModule("NS", 1.4, 0.1),
        BRANCHING_RECURSION.FreeFieldModule("R", 1.4, 0.2),
        BRANCHING_RECURSION.FreeFieldModule("R", 1.4, 0.3),
    ))
    answer = {}
    for levels in level_triples(maximum_total_twice_level):
        vector = [0j] * 8
        for ns in ns_fermion_states(levels[0]):
            for r1, g1 in ramond_fermion_states(levels[1] // 2):
                for r2, g2 in ramond_fermion_states(levels[2] // 2):
                    ground = (g1, g2)[cut - 1]
                    if insertion == "fixed" and ground:
                        continue
                    a, b, c = len(ns) % 2, (len(r1) + g1) % 2, (len(r2) + g2) % 2
                    if (a + b + c) % 2:
                        continue
                    rho = form.value((
                        tuple(int(2 * r) for r in ns),
                        (tuple(int(r) for r in r1), g1),
                        (tuple(int(r) for r in r2), g2),
                    ))
                    index = a | (b << 1) | (c << 2)
                    weight = (-1) ** ground if insertion == "saturated" else 1
                    vector[index] += weight * (-1) ** a * theta_quadratic_sign(index) * rho**2
        answer[levels] = tuple(vector)
    return answer


def recover_physical_series(enlarged_with_insertion, auxiliary_with_insertion, *, maximum_total_twice_level):
    """Formal division with auxiliary constant 1; no vertex-sector restriction.

    The numerator must have the SAME Pi_0 insertion as the denominator.
    Passing the old unsaturated enlarged block does not recover odd sectors.
    """
    ground = auxiliary_with_insertion.get((0, 0, 0), ZERO_VECTOR)
    if len(ground) != 8 or max(abs(a - b) for a, b in zip(ground, (1, 0, 0, 0, 0, 0, 0, 0))) > 1e-12:
        raise ValueError("the normalized auxiliary constant must be the full identity")
    positive = [(m, v) for m, v in auxiliary_with_insertion.items() if any(m) and any(v)]
    answer = {}
    for n in level_triples(maximum_total_twice_level):
        value = list(enlarged_with_insertion.get(n, ZERO_VECTOR))
        for m, coefficient in positive:
            previous = tuple(n[i] - m[i] for i in range(3))
            if min(previous) < 0:
                continue
            term = star_multiply(coefficient, answer.get(previous, ZERO_VECTOR))
            value = [a - b for a, b in zip(value, term)]
        answer[n] = tuple(value)
    return answer
