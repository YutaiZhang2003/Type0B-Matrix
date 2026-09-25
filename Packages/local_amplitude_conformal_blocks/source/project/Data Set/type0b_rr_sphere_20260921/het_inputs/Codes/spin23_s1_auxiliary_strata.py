#!/usr/bin/env python3
r"""Formal off-shell auxiliary-field strata at the first wall resonance.

This module is deliberately combinatorial.  It does *not* assign a number to
products such as ``delta2(z_i-z_j) * abs(z_i-z_j)**kappa``.  At resonant
momenta those products require a common supersymmetric analytic extension.

In the conventions used by the five-point resonance code, the off-shell
super-Liouville auxiliary field obeys

``<F(z) F(u)>_0 = -pi * delta2(z-u)``.

The auxiliary branch of a picture-zero singlet is ``i*a_i*F``.  At first
order in the wall coupling, the wall is either the Yukawa component ``Y`` or
its auxiliary component ``F*exp(phi)``.  After dividing by the Yukawa action
coefficient ``-2*i*mu``, Wick contraction gives one positive factor

``pi * a_u * a_v * delta2(z_u-z_v)``

for every auxiliary-field edge, with wall charge ``a_w=1``.  Every allowed
term is therefore a partial matching on ``{w} union E``, where ``E`` is the
set of zero-picture external singlets.  An unmatched wall vertex denotes the
Yukawa wall; a matched wall vertex denotes the auxiliary wall component.

The returned objects are formal strata and are useful for auditing the
component catalog.  They must not be integrated independently and added with
unrelated finite-part prescriptions.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Hashable, Mapping, Sequence


WALL_LABEL = "w"
AuxiliaryLabel = Hashable
AuxiliaryEdge = tuple[AuxiliaryLabel, AuxiliaryLabel]


@dataclass(frozen=True)
class S1AuxiliaryStratum:
    """One formal delta-supported component at order one in the wall."""

    edges: tuple[AuxiliaryEdge, ...]
    relative_coefficient: complex
    wall_component: str
    external_auxiliary_legs: tuple[AuxiliaryLabel, ...]

    @property
    def delta_codimension(self) -> int:
        """Return the number of complex delta functions in the stratum."""

        return len(self.edges)


def _matchings(labels: tuple[AuxiliaryLabel, ...]) -> tuple[tuple[AuxiliaryEdge, ...], ...]:
    """Enumerate all partial matchings in a deterministic recursive order."""

    if not labels:
        return ((),)
    first = labels[0]
    tail = labels[1:]
    result: list[tuple[AuxiliaryEdge, ...]] = list(_matchings(tail))
    for index, second in enumerate(tail):
        remainder = tail[:index] + tail[index + 1 :]
        result.extend(
            ((first, second),) + submatching
            for submatching in _matchings(remainder)
        )
    return tuple(result)


def enumerate_s1_auxiliary_strata(
    eligible_external_legs: Sequence[AuxiliaryLabel],
    external_charges: Mapping[AuxiliaryLabel, complex],
    *,
    wall_label: AuxiliaryLabel = WALL_LABEL,
) -> tuple[S1AuxiliaryStratum, ...]:
    r"""Return every formal off-shell auxiliary-field stratum at ``s=1``.

    Parameters
    ----------
    eligible_external_legs:
        The zero-picture external singlets.  Vector legs and picture-minus-one
        singlets must not be included.
    external_charges:
        Their free-field exponential charges ``a_i=1+i*P_i``.
    wall_label:
        A label distinct from every external leg.  Its charge is fixed to one.

    Returns
    -------
    tuple[S1AuxiliaryStratum, ...]
        One entry per partial matching on the wall and eligible external legs.
        Coefficients are relative to the stripped Yukawa coefficient
        ``-2*i*mu`` and do not contain the delta distributions themselves.
    """

    external = tuple(eligible_external_legs)
    if len(set(external)) != len(external):
        raise ValueError("eligible external legs must be distinct")
    if wall_label in external:
        raise ValueError("the wall label must be distinct from external legs")
    missing = tuple(label for label in external if label not in external_charges)
    if missing:
        raise ValueError(f"missing external charges for {missing!r}")

    charges = {label: complex(external_charges[label]) for label in external}
    charges[wall_label] = 1.0 + 0.0j
    external_set = set(external)
    result: list[S1AuxiliaryStratum] = []
    for edges in _matchings((wall_label,) + external):
        coefficient = 1.0 + 0.0j
        auxiliary_external: set[AuxiliaryLabel] = set()
        wall_matched = False
        for first, second in edges:
            coefficient *= math.pi * charges[first] * charges[second]
            if first == wall_label or second == wall_label:
                wall_matched = True
            if first in external_set:
                auxiliary_external.add(first)
            if second in external_set:
                auxiliary_external.add(second)
        result.append(
            S1AuxiliaryStratum(
                edges=edges,
                relative_coefficient=coefficient,
                wall_component="auxiliary" if wall_matched else "yukawa",
                external_auxiliary_legs=tuple(
                    label for label in external if label in auxiliary_external
                ),
            )
        )
    return tuple(result)


__all__ = [
    "AuxiliaryEdge",
    "AuxiliaryLabel",
    "S1AuxiliaryStratum",
    "WALL_LABEL",
    "enumerate_s1_auxiliary_strata",
]
