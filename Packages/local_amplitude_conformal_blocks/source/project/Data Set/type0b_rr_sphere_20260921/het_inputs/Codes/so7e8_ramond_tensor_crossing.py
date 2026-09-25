#!/usr/bin/env python3
r"""Tensor bookkeeping for folding two-R/two-NS sphere amplitudes.

The exterior of the modulus plane is mapped to the unit disk together with
the exchange of the NS legs at ``z`` and ``1``.  Scalar and one-vector tensor
structures are unchanged after relabelling the external leg.  For two
vectors, however,

``delta^(ba) C = delta^(ab) C`` while ``C gamma^(ba) = -C gamma^(ab)``.

The antisymmetric sign must be applied when the original and inversion-folded
pieces are assembled; it is not a conformal-block or Liouville phase.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


NSKind = Literal["S", "V"]


@dataclass(frozen=True)
class TwoRamondTensorValues:
    """The populated tensor coefficients for one ordered NS pair."""

    ns_at_z: NSKind
    ns_at_one: NSKind
    F_SS: complex | None = None
    F_SV: complex | None = None
    F_VS: complex | None = None
    F0: complex | None = None
    F2: complex | None = None

    def populated(self) -> tuple[complex, ...]:
        """Return the non-``None`` coefficients in the canonical order."""

        return tuple(
            complex(value)
            for value in (self.F_SS, self.F_SV, self.F_VS, self.F0, self.F2)
            if value is not None
        )


def _validate_shape(values: TwoRamondTensorValues) -> None:
    expected = {
        ("S", "S"): (True, False, False, False, False),
        ("S", "V"): (False, True, False, False, False),
        ("V", "S"): (False, False, True, False, False),
        ("V", "V"): (False, False, False, True, True),
    }
    key = (values.ns_at_z, values.ns_at_one)
    if key not in expected:
        raise ValueError("NS kinds must each be 'S' or 'V'")
    observed = tuple(
        value is not None
        for value in (values.F_SS, values.F_SV, values.F_VS, values.F0, values.F2)
    )
    if observed != expected[key]:
        raise ValueError(f"tensor population does not match ordered species {key}")


def fold_two_ramond_tensor_values(
    original: TwoRamondTensorValues,
    swapped: TwoRamondTensorValues,
) -> TwoRamondTensorValues:
    r"""Add an ordered unit-disk piece and its inversion-folded partner.

    ``swapped`` must be evaluated with its two NS species exchanged.  The
    result is returned in the tensor convention of ``original``.
    """

    _validate_shape(original)
    _validate_shape(swapped)
    original_kinds = (original.ns_at_z, original.ns_at_one)
    required_swapped = tuple(reversed(original_kinds))
    if (swapped.ns_at_z, swapped.ns_at_one) != required_swapped:
        raise ValueError(
            "the inversion-folded piece must exchange the NS legs at z and one"
        )

    if original_kinds == ("S", "S"):
        return TwoRamondTensorValues(
            "S", "S", F_SS=complex(original.F_SS) + complex(swapped.F_SS)
        )
    if original_kinds == ("S", "V"):
        return TwoRamondTensorValues(
            "S", "V", F_SV=complex(original.F_SV) + complex(swapped.F_VS)
        )
    if original_kinds == ("V", "S"):
        return TwoRamondTensorValues(
            "V", "S", F_VS=complex(original.F_VS) + complex(swapped.F_SV)
        )
    return TwoRamondTensorValues(
        "V",
        "V",
        F0=complex(original.F0) + complex(swapped.F0),
        # The swapped evaluator's gamma^(ba) is minus gamma^(ab).
        F2=complex(original.F2) - complex(swapped.F2),
    )


__all__ = ["TwoRamondTensorValues", "fold_two_ramond_tensor_values"]
