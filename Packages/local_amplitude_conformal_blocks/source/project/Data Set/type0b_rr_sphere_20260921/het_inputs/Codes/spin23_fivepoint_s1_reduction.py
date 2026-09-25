#!/usr/bin/env python3
r"""Exact S/V reduction on the first five-point wall resonance.

At ``s=1`` the antiholomorphic part of the screened linear-dilaton
representative is a free-fermion correlator.  A genuine singlet on external
leg ``i`` is the common ``phi`` flavor with coefficient

``a_i = 1 + i P_i``,

and the wall (or its ordinary soft-singlet regulator) supplies one additional
fermion of the same flavor.  Vectors are the other 23 free flavors.  The
bosonic Koba--Nielsen factor and the holomorphic Pfaffian do not depend on the
S/V labels.

Consequently, every nonzero zero-vector or two-vector projection is an exact
linear combination of four-vector projections at the *same fixed-incoming
kinematics*.  The identities hold pointwise on the separated configuration
space and for nonzero soft momentum, hence also after any common meromorphic
continuation of the complete integrals.  No incoming leg is crossed or
reassigned.

External labels in this module are incoming-first: ``0`` is incoming and
``1,...,4`` are outgoing.  The fifteen seed projections have one singlet and
four vectors.  If the sole singlet is ``r``, it pairs with the soft wall
fermion; the remaining four vector legs carry one of their three perfect
matchings.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from spin23_fivepoint_resonance_channels import (
    ExternalPair,
    FivePointChannelProjection,
    all_nonzero_projections,
)
from spin23_linear_dilaton_fivepoint import (
    FIRST_SCREENING_MOMENTUM,
    liouville_charge,
)


@dataclass(frozen=True)
class S1SeedContribution:
    """One four-vector seed and its charge-polynomial coefficient."""

    seed_projection: FivePointChannelProjection
    coefficient: complex

    @property
    def seed_label(self) -> str:
        return self.seed_projection.label


def _perfect_matchings(labels: Sequence[int]) -> tuple[tuple[ExternalPair, ...], ...]:
    ordered = tuple(sorted(int(label) for label in labels))
    if len(ordered) != 4 or len(set(ordered)) != 4:
        raise ValueError("exactly four distinct external labels are required")
    first, second, third, fourth = ordered
    return (
        ((first, second), (third, fourth)),
        ((first, third), (second, fourth)),
        ((first, fourth), (second, third)),
    )


def four_vector_seed_projections() -> tuple[FivePointChannelProjection, ...]:
    """Return the fifteen one-singlet/four-vector seed projections."""

    return tuple(
        projection
        for projection in all_nonzero_projections()
        if projection.incoming_first_kinds.count("V") == 4
    )


def _external_charges(
    outgoing_momenta: Sequence[complex],
    soft_momentum: complex,
    *,
    tolerance: float = 1.0e-10,
) -> tuple[complex, complex, complex, complex, complex]:
    outgoing = tuple(complex(value) for value in outgoing_momenta)
    soft = complex(soft_momentum)
    if len(outgoing) != 4:
        raise ValueError("four physical outgoing momenta are required")
    if abs(sum(outgoing) + soft - FIRST_SCREENING_MOMENTUM) > tolerance:
        raise ValueError(
            "the physical outgoing momenta plus the soft momentum must sum to 2i"
        )
    momenta = (FIRST_SCREENING_MOMENTUM,) + outgoing
    return tuple(liouville_charge(momentum) for momentum in momenta)  # type: ignore[return-value]


def _seed_projection(
    sole_singlet: int,
    vector_pairs: Sequence[ExternalPair],
) -> FivePointChannelProjection:
    label = int(sole_singlet)
    if label not in range(5):
        raise ValueError("the sole-singlet label must lie in 0,...,4")
    kinds = "".join("S" if index == label else "V" for index in range(5))
    pairs = tuple(
        sorted(tuple(sorted((int(first), int(second)))) for first, second in vector_pairs)
    )
    return FivePointChannelProjection(kinds, pairs)


def s1_projection_reduction(
    projection: FivePointChannelProjection,
    outgoing_momenta: Sequence[complex],
    *,
    soft_momentum: complex = 0.0j,
) -> tuple[S1SeedContribution, ...]:
    r"""Reduce one allowed projection to four-vector seeds.

    ``outgoing_momenta`` are the four *physical* outgoing momenta.  For the
    ordinary soft regulator they therefore obey

    ``sum(outgoing_momenta) + soft_momentum = 2i``.

    At the exact wall representative set ``soft_momentum=0``.
    """

    charges = _external_charges(outgoing_momenta, soft_momentum)
    kinds = projection.incoming_first_kinds
    vector_labels = tuple(index for index, kind in enumerate(kinds) if kind == "V")
    singlet_labels = tuple(index for index, kind in enumerate(kinds) if kind == "S")

    if len(vector_labels) == 4:
        return (S1SeedContribution(projection, 1.0 + 0.0j),)

    if len(vector_labels) == 2:
        vector_pair = tuple(sorted(vector_labels))
        terms: list[S1SeedContribution] = []
        for wall_partner in singlet_labels:
            remaining = tuple(
                label for label in singlet_labels if label != wall_partner
            )
            seed = _seed_projection(
                wall_partner,
                (vector_pair, tuple(sorted(remaining))),
            )
            coefficient = charges[remaining[0]] * charges[remaining[1]]
            terms.append(S1SeedContribution(seed, coefficient))
        return tuple(terms)

    if len(vector_labels) == 0:
        terms = []
        for wall_partner in singlet_labels:
            remaining = tuple(
                label for label in singlet_labels if label != wall_partner
            )
            coefficient = 1.0 + 0.0j
            for label in remaining:
                coefficient *= charges[label]
            for pairing in _perfect_matchings(remaining):
                terms.append(
                    S1SeedContribution(
                        _seed_projection(wall_partner, pairing),
                        coefficient,
                    )
                )
        return tuple(terms)

    raise ValueError("only the 26 even-vector five-point projections are nonzero")


def all_s1_reduction_coefficients(
    outgoing_momenta: Sequence[complex],
    *,
    soft_momentum: complex = 0.0j,
) -> dict[str, tuple[S1SeedContribution, ...]]:
    """Return the seed decomposition of all 26 labelled projections."""

    return {
        projection.label: s1_projection_reduction(
            projection,
            outgoing_momenta,
            soft_momentum=soft_momentum,
        )
        for projection in all_nonzero_projections()
    }


def _seed_values_by_label(
    seed_values: Mapping[str | FivePointChannelProjection, complex],
) -> dict[str, complex]:
    normalized: dict[str, complex] = {}
    for raw_key, raw_value in seed_values.items():
        label = raw_key.label if isinstance(raw_key, FivePointChannelProjection) else str(raw_key)
        if label in normalized:
            raise ValueError(f"duplicate seed value for {label!r}")
        normalized[label] = complex(raw_value)
    expected = {projection.label for projection in four_vector_seed_projections()}
    missing = sorted(expected - set(normalized))
    if missing:
        raise ValueError("missing four-vector seed values: " + ", ".join(missing))
    return normalized


def reconstruct_all_s1_values(
    seed_values: Mapping[str | FivePointChannelProjection, complex],
    outgoing_momenta: Sequence[complex],
    *,
    soft_momentum: complex = 0.0j,
) -> dict[str, complex]:
    """Reconstruct all 26 projection values from the fifteen seed values."""

    values = _seed_values_by_label(seed_values)
    reductions = all_s1_reduction_coefficients(
        outgoing_momenta,
        soft_momentum=soft_momentum,
    )
    return {
        label: sum(
            (term.coefficient * values[term.seed_label] for term in terms),
            0.0j,
        )
        for label, terms in reductions.items()
    }


__all__ = [
    "S1SeedContribution",
    "all_s1_reduction_coefficients",
    "four_vector_seed_projections",
    "reconstruct_all_s1_values",
    "s1_projection_reduction",
]
