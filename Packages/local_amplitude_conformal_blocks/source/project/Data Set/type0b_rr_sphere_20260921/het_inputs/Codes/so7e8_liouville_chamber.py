#!/usr/bin/env python3
r"""Conservative generic-energy chamber checks for SO(7) sphere integrals.

The numerical Liouville decomposition initially keeps the internal momentum
on the undeformed real half-line.  For outgoing momenta with nonnegative
imaginary parts, a sufficient first-pole condition shared by the NS and RRNS
structure constants is

``sum(Im p_out) + max(Im p_out) < 1``.

It follows from the nearest zeros of ``Upsilon_NS(1+i A)`` at arguments zero
and two.  The check is deliberately labelled *sufficient*: it certifies the
small generic complex-energy chamber used to define the amplitude before
analytic continuation, not every possible pole-free complex kinematic point.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence


@dataclass(frozen=True)
class LiouvilleChamberCheck:
    """Result of the conservative real-contour chamber test."""

    outgoing_momenta: tuple[complex, complex, complex]
    incoming_momentum: complex
    first_pole_margin: float
    nonnegative_outgoing_imaginary_parts: bool
    energy_conserving: bool

    @property
    def certified(self) -> bool:
        """Whether all sufficient chamber conditions hold strictly."""

        return (
            self.energy_conserving
            and self.nonnegative_outgoing_imaginary_parts
            and self.first_pole_margin > 0
        )


def conservative_real_p_chamber(
    outgoing_momenta: Sequence[complex],
    incoming_momentum: complex | None = None,
    *,
    tolerance: float = 1.0e-12,
) -> LiouvilleChamberCheck:
    r"""Check the standard positive-imaginary no-first-crossing chamber."""

    outgoing = tuple(complex(value) for value in outgoing_momenta)
    if len(outgoing) != 3:
        raise ValueError("outgoing_momenta must contain exactly three values")
    if not all(
        math.isfinite(value.real) and math.isfinite(value.imag)
        for value in outgoing
    ):
        raise ValueError("all outgoing momenta must be finite")
    expected_incoming = sum(outgoing, 0.0j)
    incoming = (
        expected_incoming
        if incoming_momentum is None
        else complex(incoming_momentum)
    )
    if not (math.isfinite(incoming.real) and math.isfinite(incoming.imag)):
        raise ValueError("incoming_momentum must be finite")
    scale = max(1.0, abs(incoming), *(abs(value) for value in outgoing))
    conserving = abs(incoming - expected_incoming) <= tolerance * scale
    imaginary_parts = tuple(value.imag for value in outgoing)
    nonnegative = all(value >= -tolerance for value in imaginary_parts)
    margin = 1.0 - (sum(imaginary_parts) + max(imaginary_parts))
    return LiouvilleChamberCheck(
        outgoing_momenta=outgoing,  # type: ignore[arg-type]
        incoming_momentum=incoming,
        first_pole_margin=float(margin),
        nonnegative_outgoing_imaginary_parts=nonnegative,
        energy_conserving=conserving,
    )


def require_conservative_real_p_chamber(
    outgoing_momenta: Sequence[complex],
    incoming_momentum: complex | None = None,
) -> LiouvilleChamberCheck:
    """Return a certified check or raise with the failed condition."""

    result = conservative_real_p_chamber(outgoing_momenta, incoming_momentum)
    if not result.energy_conserving:
        raise ValueError("incoming momentum must equal the sum of outgoing momenta")
    if not result.nonnegative_outgoing_imaginary_parts:
        raise ValueError(
            "this sufficient chamber requires nonnegative outgoing imaginary parts"
        )
    if result.first_pole_margin <= 0:
        raise ValueError(
            "the undeformed real-P contour is outside the conservative "
            f"no-first-crossing chamber (margin={result.first_pole_margin:.6g})"
        )
    return result


__all__ = [
    "LiouvilleChamberCheck",
    "conservative_real_p_chamber",
    "require_conservative_real_p_chamber",
]
