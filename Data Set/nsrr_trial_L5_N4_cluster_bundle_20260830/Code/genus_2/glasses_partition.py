r"""Parity-correct all-NS genus-two sewing in the glasses channel.

At either glasses trinion the repeated handle primary occurs twice, so its
intrinsic parity cancels.  If ``a`` is the relative three-form label and
``p_B`` is the separating-bridge primary parity, the absolute trinion parity
is therefore

    a_abs = a + p_B (mod 2).

The diagonal nonchiral term carries ``(-1)**a_abs``.  For the even NS
continuum of Type 0B this is the same even-minus-odd sign as in the theta
channel, but its derivation and locality constraint depend only on the bridge
primary, not on the sum of all three primary parities.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from genus_2.theta_partition import TYPE0B_NS_PRIMARY_PARITIES


ParityTriple = tuple[int, int, int]


def _parities(values: Sequence[int], *, name: str) -> ParityTriple:
    result = tuple(int(value) for value in values)
    if len(result) != 3 or any(value not in (0, 1) for value in result):
        raise ValueError(f"{name} must contain exactly three parity bits")
    return result  # type: ignore[return-value]


def _sector(value: int, *, name: str) -> int:
    result = int(value)
    if result not in (0, 1):
        raise ValueError(f"{name} must be zero or one")
    return result


@dataclass(frozen=True)
class GlassesSectorPair:
    holomorphic_sector: int
    antiholomorphic_sector: int
    holomorphic_primary_parities: ParityTriple
    antiholomorphic_primary_parities: ParityTriple
    absolute_parity: int
    sign: int


def glasses_sector_pair(
    holomorphic_sector: int,
    *,
    holomorphic_primary_parities: Sequence[int] = TYPE0B_NS_PRIMARY_PARITIES,
    antiholomorphic_primary_parities: Sequence[int] = TYPE0B_NS_PRIMARY_PARITIES,
) -> GlassesSectorPair:
    r"""Apply ``a+p_B=a_tilde+p_B_tilde`` and its sewing sign."""

    sector = _sector(holomorphic_sector, name="holomorphic_sector")
    parities = _parities(
        holomorphic_primary_parities,
        name="holomorphic_primary_parities",
    )
    anti_parities = _parities(
        antiholomorphic_primary_parities,
        name="antiholomorphic_primary_parities",
    )
    absolute = (sector + parities[2]) % 2
    anti_sector = (absolute + anti_parities[2]) % 2
    return GlassesSectorPair(
        holomorphic_sector=sector,
        antiholomorphic_sector=anti_sector,
        holomorphic_primary_parities=parities,
        antiholomorphic_primary_parities=anti_parities,
        absolute_parity=absolute,
        sign=-1 if absolute else 1,
    )


def glasses_diagonal_sector_contribution(
    *,
    sector: int,
    measure: float,
    structure_weight: float,
    primary_times_block: complex,
    primary_parities: Sequence[int] = TYPE0B_NS_PRIMARY_PARITIES,
) -> float:
    r"""Return ``measure*(-1)^(a+p_B)*C_a^2*|q^h F_a|^2``."""

    pair = glasses_sector_pair(
        sector,
        holomorphic_primary_parities=primary_parities,
        antiholomorphic_primary_parities=primary_parities,
    )
    return float(
        pair.sign
        * float(measure)
        * float(structure_weight)
        * abs(complex(primary_times_block)) ** 2
    )


@dataclass(frozen=True)
class GlassesNullTransport:
    edge: int
    null_parity: int
    child_sector: int
    child_lifts: tuple[int, int, int]
    edge_character: int


def glasses_null_transport(
    *,
    sector: int,
    lifts: Sequence[int],
    edge: int,
    rs: int,
) -> GlassesNullTransport:
    r"""Return the PBW-verified odd-null transport in glasses edge order.

    Edges 0 and 1 are self-loop handles and edge 2 is the separating bridge.
    An odd handle null toggles the intermediate form twice, hence leaves ``a``
    fixed, but flips the bridge lift.  This produces the toric character
    ``(-1)^a`` for even primaries.  An odd bridge null toggles both endpoint
    forms once, hence maps ``a -> a xor 1``; the endpoint Koszul factors cancel
    the apparent spectator-lift flip, so all lifts remain fixed.
    """

    current_sector = _sector(sector, name="sector")
    current_lifts = tuple(int(value) for value in lifts)
    if len(current_lifts) != 3 or any(value not in (-1, 1) for value in current_lifts):
        raise ValueError("lifts must contain exactly three signs")
    selected_edge = int(edge)
    if selected_edge not in (0, 1, 2):
        raise ValueError("edge must be zero, one, or two")
    product = int(rs)
    if product < 0:
        raise ValueError("rs must be nonnegative")
    delta = product % 2
    child_sector = current_sector
    child_lifts = current_lifts
    if delta:
        if selected_edge in (0, 1):
            child_lifts = (
                current_lifts[0],
                current_lifts[1],
                -current_lifts[2],
            )
        else:
            child_sector ^= 1
    return GlassesNullTransport(
        edge=selected_edge,
        null_parity=delta,
        child_sector=child_sector,
        child_lifts=child_lifts,
        edge_character=current_lifts[selected_edge] ** delta,
    )


__all__ = [
    "GlassesNullTransport",
    "GlassesSectorPair",
    "glasses_diagonal_sector_contribution",
    "glasses_null_transport",
    "glasses_sector_pair",
]
