"""State-free hard mixed-sheet oracle on the two Coulomb charge planes.

This is the first nontrivial composition of the exact level-one reflection
block with the ground-resolved fermion form.  It evaluates

    (n1,n2,n3) = (0, 3/4, -3/4)

without an SCA descendant or a PBW transition.  The two fixed spin forms
live on different charge-neutral planes:

    eta=-1: Q/2 + P1 + P2 + P3 = 0,
    eta=+1: Q/2 + P1 - P2 + P3 = 0.

The second plane uses the complementary Coulomb vertex.  Besides replacing
the current multiplier by ``i*(Q/2-P2)``, it changes the cross-leg
two-spin covariance.  The latter is the rational multiplier implemented
below.  Keeping the current change while reusing the ordinary covariance
fails every complementary hard restriction.

The complementary multiplier is an exact *hard-channel calibration*: it
was obtained by solving the already known crossed polynomial ``H`` for the
single mode-one two-spin entry.  Thus this module verifies the Coulomb
current, reflection, ground, copy, and phase glue without PBW states, but it
does not independently derive ``H`` and does not assert an all-mode kernel.
"""

from __future__ import annotations

from collections import defaultdict
from functools import lru_cache

import sympy as sp

from .audit_ground_covariance import (
    grid,
    physical_majorana_value,
)
from .reflected_current_multipliers import (
    complementary_charge_multiplier,
    ordinary_charge_multiplier,
)
from ..reflection.intertwiner_recurrence import symbolic_level_one


I = sp.I
SQRT2 = sp.sqrt(2)
FOCK_TO_SCBLOCK_MINUS = -(1 - I) / SQRT2
HARD = sp.Rational(3, 4)
LEVEL_ZERO_BASIS = (((), (), 0), ((), (), 1))
LEVEL_ONE_BASIS = (
    ((1,), (), 0),
    ((1,), (), 1),
    ((), (1,), 0),
    ((), (1,), 1),
)


def hard_complementary_pair_multiplier(q, p2, p3):
    """Multiplier of the physical one--zero fermion covariance.

    ``p3`` is the actual momentum on the displayed negative third branch.
    Put ``r=-p3`` for its positive-sheet representative.  The compact form
    below is the exact hard crossed polynomial divided by the two Ramond
    leg factors, expressed in the local level-one reflection frame.  It is
    calibrated from that polynomial and is valid only for this mode-one
    hard pair.
    """

    r = -p3
    e2 = q + 2 * p2
    e3 = q + 2 * r
    d2 = e2**2 + q * e2 + 1
    d3 = e3**2 + q * e3 + 1
    crossed_line = -2 * q * (p2 + r)
    crossed = (
        crossed_line**2
        + 2 * crossed_line * (e2 * e3 + 1)
        + d2 * d3
    )
    reflection_denominator = 4 * r**2 + 6 * q * r + 2 * q**2 + 1
    constant_part = 18 * q**2 + 50 * q * r + 28 * r**2 + 7
    coefficient = 9 * (2 * q**2 + 2 * q * r - 4 * r**2 - 1)
    return sp.factor(
        (16 * reflection_denominator * crossed / (d2 * d3) - constant_part)
        / coefficient
    )


@lru_cache(None)
def _reflected_third_paths(epsilon3, q, p3):
    """Apply ``R_0`` or ``R_1`` to every physical endpoint of ``W_-3/4``."""

    # Imported lazily to keep the public hard oracle independent of the old
    # abstract-state evaluator.  This module is a literal free-fermion chain.
    from .audit_ground_covariance import chi

    grouped = defaultdict(lambda: defaultdict(lambda: sp.Integer(0)))
    for state, coefficient in chi.ramond_fock_paths(-HARD, int(epsilon3)):
        auxiliary_modes, auxiliary_ground, physical_modes, physical_ground = state
        grouped[(auxiliary_modes, auxiliary_ground)][
            ((), physical_modes, physical_ground)
        ] += coefficient

    level_one = symbolic_level_one().subs(
        {grid.ramond_branch.Q: q, grid.ramond_branch.P: p3},
        simultaneous=True,
    )
    answer = []
    for auxiliary_state, expression in grouped.items():
        sample_state = next(iter(expression))
        level = sum(sample_state[0]) + sum(sample_state[1])
        if level == 0:
            basis = LEVEL_ZERO_BASIS
            reflection = sp.eye(2)
        elif level == 1:
            basis = LEVEL_ONE_BASIS
            reflection = level_one
        else:
            raise AssertionError(f"hard endpoint unexpectedly has level {level}")
        row = {state: index for index, state in enumerate(basis)}
        vector = sp.zeros(len(basis), 1)
        for state, coefficient in expression.items():
            vector[row[state]] += coefficient
        reflected = reflection * vector
        for index, (bosons, fermions, ground) in enumerate(basis):
            coefficient = sp.cancel(reflected[index])
            if coefficient:
                answer.append(
                    (
                        auxiliary_state[0],
                        auxiliary_state[1],
                        bosons,
                        fermions,
                        ground,
                        coefficient,
                    )
                )
    return tuple(answer)


def _physical_level_one_form(
    form_parity,
    eta,
    second_modes,
    second_ground,
    third_modes,
    third_ground,
    pair_multiplier,
):
    """Ordinary local Fock spin frame, with one complementary pair hook."""

    value = physical_majorana_value(
        form_parity,
        eta,
        second_modes,
        second_ground,
        third_modes,
        third_ground,
        (),
    )
    second_count = len(second_modes)
    third_count = len(third_modes)
    local = lambda ground: FOCK_TO_SCBLOCK_MINUS ** (1 - 2 * int(ground))
    if second_count and not third_count:
        return sp.factor(value * local(second_ground))
    if third_count and not second_count:
        return sp.factor(value * local(third_ground))
    if second_count and third_count:
        if int(form_parity) == 0:
            frame = local(second_ground) * local(third_ground)
        else:
            frame = I * (-1) ** int(second_ground)
        return sp.factor(value * frame * pair_multiplier)
    return value


def hard_mixed_sheet_value(epsilon2, epsilon3, form_parity, eta, q, p2, p3):
    """Evaluate the hard mixed-sheet form on its fixed-``eta`` charge plane."""

    from .audit_ground_covariance import chi

    epsilon2 = int(epsilon2)
    epsilon3 = int(epsilon3)
    form_parity = int(form_parity)
    eta = int(eta)
    if eta not in (-1, 1):
        raise ValueError("eta must be +1 or -1")
    auxiliary_form = (epsilon2 + epsilon3 - form_parity) % 2
    if eta == -1:
        current = ordinary_charge_multiplier(q, p2)
        pair_multiplier = sp.Integer(1)
    else:
        current = complementary_charge_multiplier(q, p2)
        pair_multiplier = hard_complementary_pair_multiplier(q, p2, p3)

    answer = sp.Integer(0)
    for state2, coefficient2 in chi.ramond_fock_paths(HARD, epsilon2):
        auxiliary2, auxiliary_ground2, physical2, physical_ground2 = state2
        physical_parity2 = (len(physical2) + physical_ground2) % 2
        for (
            auxiliary3,
            auxiliary_ground3,
            bosons3,
            physical3,
            physical_ground3,
            coefficient3,
        ) in _reflected_third_paths(epsilon3, q, p3):
            auxiliary = grid.fermion_value(
                auxiliary_form,
                (),
                auxiliary2,
                auxiliary_ground2,
                auxiliary3,
                auxiliary_ground3,
            )
            physical = _physical_level_one_form(
                form_parity,
                eta,
                physical2,
                physical_ground2,
                physical3,
                physical_ground3,
                pair_multiplier,
            )
            if not bosons3:
                current_value = sp.Integer(1)
            elif bosons3 == (1,):
                current_value = current
            else:
                raise AssertionError(bosons3)
            auxiliary_parity3 = (len(auxiliary3) + auxiliary_ground3) % 2
            answer += (
                coefficient2
                * coefficient3
                * FOCK_TO_SCBLOCK_MINUS
                ** (physical_ground2 + physical_ground3)
                * (-1) ** (physical_parity2 * auxiliary_parity3)
                * auxiliary
                * physical
                * current_value
            )
    return sp.factor(sp.cancel(answer))


def charge_plane_p1(q, p2, p3, eta):
    """The zero-screening plane used by ``hard_mixed_sheet_value``."""

    eta = int(eta)
    return -q / 2 + eta * p2 - p3


__all__ = (
    "charge_plane_p1",
    "hard_complementary_pair_multiplier",
    "hard_mixed_sheet_value",
)
