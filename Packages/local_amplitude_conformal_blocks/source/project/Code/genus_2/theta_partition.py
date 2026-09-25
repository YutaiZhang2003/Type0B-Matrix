r"""Parity-correct all-NS genus-two partition assembly in the theta channel.

This module implements the nonchiral sewing formula in
``Human Notes/SCblock.tex``, around ``NSblockThetaDefinition``.  The chiral
block label is the *relative* descendant parity

``a = A + C + E (mod 2)``,

whereas the sign in the nonchiral partition function is controlled by the
absolute parity of the complete holomorphic three-point tensor,

``a_abs = a + p1 + p2 + p3 (mod 2)``.

Consequently a diagonal Type-0B NS contribution is not an unsigned sum of
the two chiral sectors.  For even NS primaries it is proportional to

``C_0**2 |F_0|**2 - C_1**2 |F_1|**2``.

The pure functions here deliberately sit between conformal-block evaluation
and momentum integration.  This makes the Koszul sign visible in local runs,
Cannon shards, and tests instead of burying it in a structure-constant phase.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


ParityTriple = tuple[int, int, int]
TYPE0B_NS_PRIMARY_PARITIES: ParityTriple = (0, 0, 0)


def _parity_triple(
    values: Sequence[int], *, name: str
) -> ParityTriple:
    if len(values) != 3:
        raise ValueError(f"{name} must contain exactly three entries")
    result = tuple(int(value) for value in values)
    if any(value not in (0, 1) for value in result):
        raise ValueError(f"{name} must contain only zeroes and ones")
    return result  # type: ignore[return-value]


def _sector(value: int, *, name: str) -> int:
    result = int(value)
    if result not in (0, 1):
        raise ValueError(f"{name} must be zero or one")
    return result


@dataclass(frozen=True)
class ThetaSectorPair:
    r"""Holomorphic/antiholomorphic sector pairing and its Koszul sign.

    Locality requires

    ``a + sum(p_i) = a_tilde + sum(p_tilde_i) (mod 2)``.

    ``absolute_parity`` is this common value.  The partition-function term
    carries ``sign=(-1)**absolute_parity``.
    """

    holomorphic_sector: int
    antiholomorphic_sector: int
    holomorphic_primary_parities: ParityTriple
    antiholomorphic_primary_parities: ParityTriple
    absolute_parity: int
    sign: int


def theta_sector_pair(
    holomorphic_sector: int,
    *,
    holomorphic_primary_parities: Sequence[int] = TYPE0B_NS_PRIMARY_PARITIES,
    antiholomorphic_primary_parities: Sequence[int] = TYPE0B_NS_PRIMARY_PARITIES,
) -> ThetaSectorPair:
    r"""Return the theta-channel sector selected by the locality constraint."""

    sector = _sector(holomorphic_sector, name="holomorphic_sector")
    parities = _parity_triple(
        holomorphic_primary_parities,
        name="holomorphic_primary_parities",
    )
    anti_parities = _parity_triple(
        antiholomorphic_primary_parities,
        name="antiholomorphic_primary_parities",
    )
    absolute = (sector + sum(parities)) % 2
    anti_sector = (absolute + sum(anti_parities)) % 2
    return ThetaSectorPair(
        holomorphic_sector=sector,
        antiholomorphic_sector=anti_sector,
        holomorphic_primary_parities=parities,
        antiholomorphic_primary_parities=anti_parities,
        absolute_parity=absolute,
        sign=-1 if absolute else 1,
    )


def theta_partition_term(
    *,
    holomorphic_sector: int,
    structure_weight: complex,
    holomorphic_primary_factor: complex,
    holomorphic_block: complex,
    antiholomorphic_primary_factor: complex,
    antiholomorphic_block: complex,
    antiholomorphic_sector: int | None = None,
    holomorphic_primary_parities: Sequence[int] = TYPE0B_NS_PRIMARY_PARITIES,
    antiholomorphic_primary_parities: Sequence[int] = TYPE0B_NS_PRIMARY_PARITIES,
) -> complex:
    r"""Assemble one term of the general nonchiral theta decomposition.

    ``structure_weight`` is the note's ``(C_123^(a))**2`` in the chosen
    operator normalization.  Primary plumbing powers are kept separate from
    the descendant blocks so callers cannot accidentally omit them.
    """

    pair = theta_sector_pair(
        holomorphic_sector,
        holomorphic_primary_parities=holomorphic_primary_parities,
        antiholomorphic_primary_parities=antiholomorphic_primary_parities,
    )
    if antiholomorphic_sector is not None:
        supplied = _sector(
            antiholomorphic_sector, name="antiholomorphic_sector"
        )
        if supplied != pair.antiholomorphic_sector:
            raise ValueError(
                "antiholomorphic sector violates "
                "a+sum(p)=a_tilde+sum(p_tilde) mod 2"
            )
    return (
        pair.sign
        * complex(structure_weight)
        * complex(holomorphic_primary_factor)
        * complex(holomorphic_block)
        * complex(antiholomorphic_primary_factor)
        * complex(antiholomorphic_block)
    )


def theta_diagonal_sector_contribution(
    *,
    sector: int,
    measure: float,
    structure_weight: float,
    primary_times_block: complex,
    primary_parities: Sequence[int] = TYPE0B_NS_PRIMARY_PARITIES,
) -> float:
    r"""Return one diagonal Type-0B NS theta contribution.

    The antiholomorphic data are the complex conjugates of the holomorphic
    data and have the same intrinsic primary parities.  The function therefore
    evaluates

    ``measure * (-1)**(a+sum(p_i)) * C_a**2 * |q**h F_a|**2``.
    """

    pair = theta_sector_pair(
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
class ThetaNullTransport:
    r"""Discrete data multiplying one theta-channel NS Kac residue."""

    edge: int
    null_parity: int
    child_sector: int
    child_lifts: tuple[int, int, int]
    edge_character: int


def theta_null_transport(
    *,
    sector: int,
    lifts: Sequence[int],
    edge: int,
    rs: int,
) -> ThetaNullTransport:
    r"""Implement the parity transport in the note's theta c-recursion.

    An even null leaves the block label and all lifts unchanged.  An odd null
    flips ``a``, contributes ``eta_edge``, and flips the two spectator lifts.
    The shifted module's intrinsic parity is thereby represented without
    mutating the caller's primary-parity metadata.
    """

    current_sector = _sector(sector, name="sector")
    current_lifts = tuple(int(value) for value in lifts)
    if len(current_lifts) != 3 or any(
        value not in (-1, 1) for value in current_lifts
    ):
        raise ValueError("lifts must contain exactly three signs")
    selected_edge = int(edge)
    if selected_edge not in (0, 1, 2):
        raise ValueError("edge must be zero, one, or two")
    product = int(rs)
    if product < 0:
        raise ValueError("rs must be nonnegative")
    null_parity = product % 2
    if null_parity:
        child_lifts = tuple(
            lift if index == selected_edge else -lift
            for index, lift in enumerate(current_lifts)
        )
    else:
        child_lifts = current_lifts
    return ThetaNullTransport(
        edge=selected_edge,
        null_parity=null_parity,
        child_sector=current_sector ^ null_parity,
        child_lifts=child_lifts,  # type: ignore[arg-type]
        edge_character=current_lifts[selected_edge] ** null_parity,
    )


__all__ = [
    "TYPE0B_NS_PRIMARY_PARITIES",
    "ThetaNullTransport",
    "ThetaSectorPair",
    "theta_diagonal_sector_contribution",
    "theta_null_transport",
    "theta_partition_term",
    "theta_sector_pair",
]
