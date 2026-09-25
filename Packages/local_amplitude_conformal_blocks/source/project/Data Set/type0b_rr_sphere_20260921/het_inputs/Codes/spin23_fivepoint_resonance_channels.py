#!/usr/bin/env python3
r"""Complete S/V channel catalogue for the heterotic sphere five-point function.

The public channel convention in this module is ``(incoming, out1, ..., out4)``.
The worldsheet code instead orders insertions as

``(out1@0, out2@z2, out3@z3, out4@1, incoming@infinity)``.

Only assignments with an even total number of SO(23) vectors can be nonzero.
There are 16 such kind assignments.  A two-vector assignment has one scalar
Kronecker coefficient, while a four-vector assignment has three independent
pair contractions.  Consequently the complete labelled calculation contains
26 scalar projections.

This file only handles states, labels, and flavor projectors.  It does not
assume the discarded higher-point hafnian proposal and it does not attach
incoming/outgoing wall-reflection phases.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Iterable, Literal, Sequence

from spin23_sphere_fivepoint_amplitude import SphereFivePointState
from spin23_sphere_fivepoint_resonance import fivepoint_resonance_momentum


KindLetter = Literal["S", "V"]
ExternalPair = tuple[int, int]
VectorPairing = tuple[ExternalPair, ExternalPair]


def _validate_kind_string(kinds: str) -> str:
    normalized = str(kinds).upper()
    if len(normalized) != 5 or any(kind not in "SV" for kind in normalized):
        raise ValueError("a channel must be a five-letter S/V string")
    return normalized


def incoming_first_to_worldsheet(kinds: str) -> str:
    """Convert ``(incoming,out1,...,out4)`` to ``(out1,...,out4,incoming)``."""

    normalized = _validate_kind_string(kinds)
    return normalized[1:] + normalized[0]


def worldsheet_to_incoming_first(kinds: str) -> str:
    """Convert ``(out1,...,out4,incoming)`` to ``(incoming,out1,...,out4)``."""

    normalized = _validate_kind_string(kinds)
    return normalized[-1] + normalized[:-1]


def even_vector_kind_strings(*, incoming_first: bool = True) -> tuple[str, ...]:
    """Return all 16 five-state kind assignments allowed by fermion parity."""

    result = tuple(
        "".join(kinds)
        for kinds in product("SV", repeat=5)
        if kinds.count("V") % 2 == 0
    )
    if incoming_first:
        return result
    return tuple(incoming_first_to_worldsheet(kinds) for kinds in result)


def _perfect_matchings(labels: Sequence[int]) -> tuple[VectorPairing, ...]:
    ordered = tuple(sorted(int(label) for label in labels))
    if len(ordered) != 4 or len(set(ordered)) != 4:
        raise ValueError("exactly four distinct vector labels are required")
    first, second, third, fourth = ordered
    return (
        ((first, second), (third, fourth)),
        ((first, third), (second, fourth)),
        ((first, fourth), (second, third)),
    )


def vector_pairings(kinds: str) -> tuple[tuple[ExternalPair, ...], ...]:
    """Return the independent SO(23) projectors for one incoming-first channel.

    External labels are ``0`` for the incoming state and ``1,...,4`` for the
    outgoing states.  A zero-vector channel is represented by the empty
    pairing; two-vector channels have their unique pair; four-vector channels
    have the three perfect matchings.
    """

    normalized = _validate_kind_string(kinds)
    labels = tuple(index for index, kind in enumerate(normalized) if kind == "V")
    if len(labels) % 2:
        return ()
    if not labels:
        return ((),)
    if len(labels) == 2:
        return (((labels[0], labels[1]),),)
    if len(labels) == 4:
        return _perfect_matchings(labels)
    # Five external states cannot contain a positive even number above four.
    raise AssertionError("unreachable vector multiplicity")


def _normalized_pairing(
    kinds: str,
    pairing: VectorPairing | ExternalPair | None,
) -> tuple[ExternalPair, ...]:
    normalized = _validate_kind_string(kinds)
    vector_labels = tuple(
        index for index, kind in enumerate(normalized) if kind == "V"
    )
    if not vector_labels:
        if pairing is not None:
            raise ValueError("the all-singlet channel has no vector pairing")
        return ()
    if len(vector_labels) == 2:
        expected = tuple(sorted(vector_labels))
        if pairing is None:
            return (expected,)  # type: ignore[return-value]
        if len(pairing) == 2 and all(isinstance(value, int) for value in pairing):
            candidate = tuple(sorted(pairing))
        elif (
            len(pairing) == 1
            and not isinstance(pairing[0], int)
            and len(pairing[0]) == 2
        ):
            candidate = tuple(sorted(pairing[0]))
        else:
            raise ValueError("a two-vector channel has one two-leg pairing")
        if candidate != expected:
            raise ValueError("the pairing does not join the two vector legs")
        return (candidate,)  # type: ignore[return-value]
    if len(vector_labels) == 4:
        if pairing is None or len(pairing) != 2:
            raise ValueError("a four-vector channel requires two vector pairs")
        pairs = tuple(tuple(sorted(pair)) for pair in pairing)
        flattened = tuple(sorted(label for pair in pairs for label in pair))
        if any(len(pair) != 2 or pair[0] == pair[1] for pair in pairs):
            raise ValueError("every vector pair must contain two distinct legs")
        if flattened != tuple(sorted(vector_labels)):
            raise ValueError("the pairing must use every vector leg exactly once")
        return tuple(sorted(pairs))  # type: ignore[return-value]
    raise ValueError("odd-vector channels have no nonzero scalar projection")


def canonical_flavors(
    kinds: str,
    pairing: VectorPairing | ExternalPair | None = None,
) -> tuple[int | None, ...]:
    """Return incoming-first flavors isolating the requested delta tensor.

    Distinct pairs receive distinct flavors.  Thus no unwanted perfect
    matching survives when a four-vector correlator is evaluated.
    """

    normalized = _validate_kind_string(kinds)
    pairs = _normalized_pairing(normalized, pairing)
    flavors: list[int | None] = [None] * 5
    for flavor, pair in enumerate(pairs):
        for label in pair:
            flavors[label] = flavor
    for label, kind in enumerate(normalized):
        if kind == "V" and flavors[label] is None:
            raise AssertionError("a vector leg was not assigned a projector flavor")
    return tuple(flavors)


@dataclass(frozen=True)
class FivePointChannelProjection:
    """One scalar coefficient in the complete five-point S/V tensor basis."""

    incoming_first_kinds: str
    vector_pairing: tuple[ExternalPair, ...]

    def __post_init__(self) -> None:
        normalized = _validate_kind_string(self.incoming_first_kinds)
        pairs = _normalized_pairing(normalized, self.vector_pairing or None)
        object.__setattr__(self, "incoming_first_kinds", normalized)
        object.__setattr__(self, "vector_pairing", pairs)

    @property
    def worldsheet_kinds(self) -> str:
        return incoming_first_to_worldsheet(self.incoming_first_kinds)

    @property
    def flavors(self) -> tuple[int | None, ...]:
        pairing: VectorPairing | ExternalPair | None
        if len(self.vector_pairing) == 0:
            pairing = None
        elif len(self.vector_pairing) == 1:
            pairing = self.vector_pairing[0]
        else:
            pairing = self.vector_pairing  # type: ignore[assignment]
        return canonical_flavors(self.incoming_first_kinds, pairing)

    @property
    def label(self) -> str:
        process = (
            self.incoming_first_kinds[0]
            + "_to_"
            + self.incoming_first_kinds[1:]
        )
        if not self.vector_pairing:
            return process
        suffix = "_".join(f"{first}{second}" for first, second in self.vector_pairing)
        return f"{process}_{suffix}"


def all_nonzero_projections() -> tuple[FivePointChannelProjection, ...]:
    """Return all 26 labelled scalar projections."""

    result: list[FivePointChannelProjection] = []
    for kinds in even_vector_kind_strings(incoming_first=True):
        labels = tuple(index for index, kind in enumerate(kinds) if kind == "V")
        if not labels:
            pairings: Iterable[tuple[ExternalPair, ...]] = ((),)
        elif len(labels) == 2:
            pairings = (((labels[0], labels[1]),),)
        else:
            pairings = _perfect_matchings(labels)
        result.extend(
            FivePointChannelProjection(kinds, tuple(pairing))
            for pairing in pairings
        )
    return tuple(result)


def resonant_worldsheet_states(
    projection: FivePointChannelProjection,
    outgoing_momenta: Sequence[complex],
    *,
    screening_number: int = 1,
) -> tuple[SphereFivePointState, ...]:
    """Build the five states in the repository's worldsheet ordering.

    ``outgoing_momenta`` are the four positive/reflected Liouville momenta.
    Their sum must equal the selected resonance momentum.  The incoming time
    momentum carries the opposite sign, while its Liouville momentum does not.
    """

    outgoing = tuple(complex(value) for value in outgoing_momenta)
    if len(outgoing) != 4:
        raise ValueError("four outgoing momenta are required")
    incoming = fivepoint_resonance_momentum(screening_number)
    if abs(sum(outgoing) - incoming) > 1.0e-10:
        raise ValueError("outgoing momenta must sum to the resonance momentum")

    kinds = projection.incoming_first_kinds
    flavors = projection.flavors
    incoming_first_momenta = (incoming,) + outgoing
    incoming_first_states: list[SphereFivePointState] = []
    for label, (kind, momentum, flavor) in enumerate(
        zip(kinds, incoming_first_momenta, flavors)
    ):
        time_momentum = -momentum if label == 0 else momentum
        if kind == "S":
            state = SphereFivePointState.singlet(momentum, time_momentum)
        else:
            state = SphereFivePointState.vector(
                momentum, time_momentum, int(flavor)
            )
        incoming_first_states.append(state)
    return tuple(incoming_first_states[1:] + incoming_first_states[:1])


def soft_regulated_worldsheet_states(
    projection: FivePointChannelProjection,
    outgoing_momenta: Sequence[complex],
    soft_momentum: complex,
) -> tuple[SphereFivePointState, ...]:
    r"""Build the five physical states before adding a sixth soft singlet.

    The additional outgoing singlet has momentum ``soft_momentum``.  Six-point
    energy conservation and linear-dilaton charge neutrality require

    ``sum(outgoing_momenta) + soft_momentum = 2i``.

    Taking ``soft_momentum -> 0`` gives the one-screening five-point
    resonance.  The returned five-state tuple is not neutral by itself; it is
    intended for :func:`evaluate_regulated_soft_singlet_integrand`.
    """

    outgoing = tuple(complex(value) for value in outgoing_momenta)
    soft = complex(soft_momentum)
    if len(outgoing) != 4:
        raise ValueError("four outgoing momenta are required")
    incoming = fivepoint_resonance_momentum(1)
    if abs(sum(outgoing) + soft - incoming) > 1.0e-10:
        raise ValueError(
            "outgoing physical momenta plus the soft singlet must sum to 2i"
        )

    kinds = projection.incoming_first_kinds
    flavors = projection.flavors
    incoming_first_momenta = (incoming,) + outgoing
    states: list[SphereFivePointState] = []
    for label, (kind, momentum, flavor) in enumerate(
        zip(kinds, incoming_first_momenta, flavors)
    ):
        time_momentum = -momentum if label == 0 else momentum
        if kind == "S":
            state = SphereFivePointState.singlet(momentum, time_momentum)
        else:
            state = SphereFivePointState.vector(
                momentum, time_momentum, int(flavor)
            )
        states.append(state)
    return tuple(states[1:] + states[:1])


__all__ = [
    "ExternalPair",
    "FivePointChannelProjection",
    "KindLetter",
    "VectorPairing",
    "all_nonzero_projections",
    "canonical_flavors",
    "even_vector_kind_strings",
    "incoming_first_to_worldsheet",
    "resonant_worldsheet_states",
    "soft_regulated_worldsheet_states",
    "vector_pairings",
    "worldsheet_to_incoming_first",
]
