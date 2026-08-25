#!/usr/bin/env python3
"""Audit the reflected recurrence on the first crossed Ramond channel.

The production object is the Fock-space reflection block from
``intertwiner_recurrence``.  For an independent low-level audit only, this
file converts its output to the old abstract basis and contracts the stored
NS--R--R Ward form.  The reflected leg is *not* converted with its own
``realization=+1`` transition: it is first mapped by the new recurrence to
the ordinary ``realization=-1`` Fock chart.

The test uses

    (n1,n2,n3) = (0, 3/4, -3/4)

and the exact reflected-sector relation to the positive-sheet hard channel.
In particular the ``eta=+`` mixed-sheet member is the reflected image of
the irreducible ``eta=-`` polynomial H.  Both stored rational samples, both
form parities, both second-leg copies, and both eta values are checked.
"""

from __future__ import annotations

from collections import defaultdict
from functools import lru_cache
from pathlib import Path
import sys

import sympy as sp


ROOT = Path(__file__).resolve().parents[3]
CHI_DIR = ROOT / "python 2" / "nsrr_chi_branching"
GRID_DIR = ROOT / "python 2" / "ramond_three_point_grid"
RAMOND_DIR = ROOT / "python 2" / "ramond_branching_coefficient_check"
for directory in (CHI_DIR, GRID_DIR, RAMOND_DIR):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import nsrr_chi_formula as chi  # noqa: E402
import compute_grid as stored  # noqa: E402
import check_ramond_branching as ramond  # noqa: E402

from .intertwiner_recurrence import symbolic_level_one


I = sp.I
SQRT2 = sp.sqrt(2)
FOCK_TO_SCBLOCK_MINUS = -(1 - I) / SQRT2


@lru_cache(None)
def reflected_components(branch_label, parity, q_value, momentum):
    """Map one negative-sheet chi string through the direct reflection R."""

    branch_label = sp.Rational(branch_label)
    if branch_label >= 0:
        raise ValueError("This audit expects a negative Ramond branch.")

    grouped = defaultdict(lambda: defaultdict(lambda: sp.Integer(0)))
    for state, coefficient in chi.ramond_fock_paths(branch_label, parity):
        auxiliary_modes, auxiliary_ground, physical_modes, physical_ground = state
        grouped[(auxiliary_modes, auxiliary_ground)][
            ((), physical_modes, physical_ground)
        ] += coefficient

    answer = []
    substitutions = {ramond.Q: q_value, ramond.P: momentum}
    for auxiliary_state, physical_expression in grouped.items():
        one_state = next(iter(physical_expression))
        level = sum(one_state[0]) + sum(one_state[1])
        basis = ramond.basis(level)
        row = {state: index for index, state in enumerate(basis)}
        vector = sp.zeros(len(basis), 1)
        for state, coefficient in physical_expression.items():
            vector[row[state]] += coefficient

        if level == 0:
            reflection = sp.eye(2)
        elif level == 1:
            reflection = symbolic_level_one()
        else:
            raise ValueError("The hard-channel audit only needs levels zero and one.")
        ordinary_vector = reflection.subs(substitutions, simultaneous=True) * vector

        # Audit-only conversion.  The reflected realization never appears
        # here: the recurrence has already put the vector in the ordinary
        # Fock chart.
        _, ordinary_transition = ramond.transition(level, -1)
        ordinary_transition = ordinary_transition.subs(
            substitutions, simultaneous=True
        )
        coefficients = ordinary_transition.inv() * ordinary_vector
        for index, (virasoro_modes, supercurrent_modes, ground) in enumerate(basis):
            coefficient = sp.cancel(coefficients[index])
            if ground:
                coefficient *= FOCK_TO_SCBLOCK_MINUS
            coefficient = sp.cancel(coefficient)
            if coefficient == 0:
                continue
            word = tuple(("L", -mode) for mode in virasoro_modes)
            word += tuple(("G", -mode) for mode in supercurrent_modes)
            answer.append(
                (
                    auxiliary_state[0],
                    auxiliary_state[1],
                    word,
                    ground,
                    coefficient,
                )
            )
    return tuple(answer)


def reflected_hard_value(
    epsilon2,
    form_parity,
    eta,
    sample,
):
    """Evaluate (0,3/4,-3/4) with the third leg supplied by R."""

    b_value, p1, p2, p3 = sample
    q_value = sp.cancel(b_value + 1 / b_value)
    central_charge = sp.Rational(3, 2) + 3 * q_value**2
    n = sp.Rational(3, 4)
    physical = stored.PhysicalNRREvaluator(
        form_parity,
        eta,
        (q_value**2 / 4 - p1**2) / 2,
        sp.Rational(1, 16) + q_value**2 / 8 - p2**2 / 2,
        sp.Rational(1, 16) + q_value**2 / 8 - p3**2 / 2,
        p2 / SQRT2,
        p3 / SQRT2,
        central_charge,
    )
    first = chi.ns_path_components(0, q_value, p1)
    second = chi.ramond_path_components(n, epsilon2, q_value, p2)
    third = reflected_components(-n, 0, q_value, p3)
    auxiliary_form = (int(epsilon2) - int(form_parity)) % 2

    answer = sp.Integer(0)
    for auxiliary1, word1, coefficient1 in first:
        physical_parity1 = stored.state_parity(word1)
        for (
            auxiliary2,
            auxiliary_ground2,
            word2,
            physical_ground2,
            coefficient2,
        ) in second:
            physical_parity2 = stored.state_parity(word2, physical_ground2)
            auxiliary_parity2 = (
                len(auxiliary2) + auxiliary_ground2
            ) % 2
            for (
                auxiliary3,
                auxiliary_ground3,
                word3,
                physical_ground3,
                coefficient3,
            ) in third:
                auxiliary_parity3 = (
                    len(auxiliary3) + auxiliary_ground3
                ) % 2
                auxiliary_value = stored.fermion_value_virasoro(
                    auxiliary_form,
                    auxiliary1,
                    auxiliary2,
                    auxiliary_ground2,
                    auxiliary3,
                    auxiliary_ground3,
                )
                physical_value = physical.value(
                    word1,
                    word2,
                    physical_ground2,
                    word3,
                    physical_ground3,
                )
                koszul = (-1) ** (
                    physical_parity1
                    * (auxiliary_parity2 + auxiliary_parity3)
                    + physical_parity2 * auxiliary_parity3
                )
                answer += (
                    koszul
                    * coefficient1
                    * coefficient2
                    * coefficient3
                    * auxiliary_value
                    * physical_value
                )
    return sp.factor(sp.cancel(answer))


def audit():
    n = sp.Rational(3, 4)
    checked = 0
    h_checks = 0
    for sample in stored.SAMPLES:
        b_value, p1, p2, p3 = sample
        reflected_sample = (b_value, p1, p2, -p3)
        for form_parity in (0, 1):
            for epsilon2 in (0, 1):
                for original_eta in (1, -1):
                    mixed_eta = -original_eta
                    calculated = reflected_hard_value(
                        epsilon2,
                        form_parity,
                        mixed_eta,
                        reflected_sample,
                    )
                    master = chi.hard_crossed_master(
                        epsilon2,
                        original_eta,
                        b_value,
                        p1,
                        p2,
                        p3,
                    )
                    reduction = chi.master_reduction_factor(
                        n,
                        n,
                        epsilon2,
                        0,
                        form_parity,
                        original_eta,
                    )
                    expected = (-1) ** form_parity * master * reduction
                    residual = sp.factor(sp.cancel(calculated - expected))
                    if residual != 0:
                        raise AssertionError(
                            (
                                sample,
                                form_parity,
                                epsilon2,
                                original_eta,
                                residual,
                            )
                        )
                    checked += 1
                    if original_eta == -1:
                        h_checks += 1
    print(
        f"reflected hard channel: {checked} exact restrictions, "
        f"including {h_checks} restrictions of irreducible H"
    )


if __name__ == "__main__":
    audit()
