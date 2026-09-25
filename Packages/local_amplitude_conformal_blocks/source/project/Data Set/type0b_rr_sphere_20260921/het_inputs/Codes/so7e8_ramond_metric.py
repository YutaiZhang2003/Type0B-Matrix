"""Explicit, conditional normalization of the physical Ramond inverse pairing.

Keep the stored NSNSNS and RRNS three-point functions unchanged. The working
hypothesis D_R = D_NS/2 multiplies each *nonchiral* internal R inverse pairing
by two. It is not a factor on each chiral block, external leg, or amplitude.
This scalar policy supplies neither a GSO projector nor PCO sewing tensors.
"""
from __future__ import annotations

from typing import Literal, Sequence


Sector = Literal["NS", "R"]
MetricConvention = Literal["recorded_equal", "conditional_identity"]
CONDITIONAL_RAMOND_METRIC: MetricConvention = "conditional_identity"


def _sectors(values: Sequence[Sector]) -> tuple[Sector, ...]:
    if isinstance(values, str):
        raise TypeError("supply a sequence of edge sectors, not a single string")
    result = tuple(values)
    if any(value not in ("NS", "R") for value in result):
        raise ValueError("each sector must be 'NS' or 'R'")
    return result


def inverse_pairing_factor(
    internal_sectors: Sequence[Sector], *, convention: MetricConvention
) -> int:
    """Multiplier relative to recorded equal metrics; apply exactly once.

    Requiring an explicit convention prevents silent changes in callers that
    have not opted into the conditional prescription.
    """
    sectors = _sectors(internal_sectors)
    if convention not in ("recorded_equal", CONDITIONAL_RAMOND_METRIC):
        raise ValueError("unknown Ramond metric convention")
    return 2 ** sectors.count("R") if convention == CONDITIONAL_RAMOND_METRIC else 1


def comb_internal_sectors(external_sectors: Sequence[Sector]) -> tuple[Sector, ...]:
    """Route spin sectors in ((1,2),3,...,(n-1,n)), not descendant parities.

    An internal edge is R precisely when one side contains an odd number of
    external R punctures. This bookkeeping does not determine physical
    species, spinor indices, three-point signs, or component coefficients.
    """
    external = _sectors(external_sectors)
    if len(external) < 3:
        raise ValueError("a sphere trivalent comb needs at least three external legs")
    if external.count("R") % 2:
        raise ValueError("an odd number of external Ramond punctures is not supported")
    parity = 0
    result = []
    for index, sector in enumerate(external[:-2]):
        parity ^= sector == "R"
        if index >= 1:
            result.append("R" if parity else "NS")
    return tuple(result)


def metric_metadata(convention: MetricConvention) -> dict:
    factor = inverse_pairing_factor(("R",), convention=convention)
    return dict(
        convention=convention,
        status="conditional" if factor == 2 else "recorded_equal_metric_reference",
        ramond_over_ns_pairing=1 / factor,
        inverse_pairing_factor_per_internal_R_edge=factor,
        applies_to="one physical nonchiral inverse pairing per internal R edge",
        stored_three_point_functions_changed=False,
        external_leg_rescaling_applied=False,
        fitted_to_crossing=False,
        normalization_dictionary_resolved=False,
        certifies_PCO_GSO_sewing=False,
    )
