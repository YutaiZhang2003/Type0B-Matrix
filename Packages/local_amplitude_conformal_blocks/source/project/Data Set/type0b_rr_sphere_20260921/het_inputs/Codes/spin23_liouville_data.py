#!/usr/bin/env python3
r"""Delta-normalized ``b=1`` ordinary-Liouville sphere data.

The primary ``V_P`` has chiral weight ``h=1+P**2`` and two-point metric
``pi delta(P-P')``.  In the BRY/Xi normalization with
``Upsilon_1(x)=G(x)G(2-x)``, the renormalized DOZZ coefficient is

.. math::

   C(P_1,P_2,P_3)=\frac1{\Upsilon_1(1+iP_\Sigma)}
   \prod_{j=1}^3
   \frac{2P_j\Upsilon_1(1+2iP_j)}
        {\Upsilon_1(1+i(P_\Sigma-2P_j))}.

No momentum or structure constant is complex-conjugated under analytic
continuation.  The omitted cosmological prefactor is momentum independent
and singular before the standard ``b=1`` renormalization.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Sequence

import mpmath as mp


Number = complex | float | int


def liouville_weight(momentum: Number) -> complex:
    """Return the ordinary ``b=1`` Liouville weight ``h=1+P**2``."""

    value = complex(momentum)
    return 1.0 + value * value


def _upsilon_1_mp(value: mp.mpc) -> mp.mpc:
    return mp.barnesg(value) * mp.barnesg(2 - value)


@lru_cache(maxsize=65536)
def liouville_structure_constant(
    p_1: Number,
    p_2: Number,
    p_3: Number,
    *,
    precision: int = 40,
) -> complex:
    """Return the delta-normalized renormalized DOZZ coefficient at ``b=1``."""

    if not isinstance(precision, int) or precision < 16:
        raise ValueError("precision must be an integer of at least 16 digits")
    momenta_python = tuple(complex(value) for value in (p_1, p_2, p_3))
    if any(value == 0 for value in momenta_python):
        return 0.0j
    with mp.workdps(precision):
        momenta = tuple(mp.mpc(value) for value in momenta_python)
        total = sum(momenta)
        result = 1 / _upsilon_1_mp(1 + 1j * total)
        for momentum in momenta:
            result *= (
                2
                * momentum
                * _upsilon_1_mp(1 + 2j * momentum)
                / _upsilon_1_mp(1 + 1j * (total - 2 * momentum))
            )
        return complex(result)


def liouville_fivepoint_structure_product(
    *,
    internal_momenta: Sequence[Number],
    external_momenta: Sequence[Number],
    precision: int = 40,
) -> complex:
    """Return the product of the three DOZZ coefficients in the comb channel."""

    internal = tuple(complex(value) for value in internal_momenta)
    external = tuple(complex(value) for value in external_momenta)
    if len(internal) != 2 or len(external) != 5:
        raise ValueError("two internal and five external momenta are required")
    p_1, p_2 = internal
    d_1, d_2, d_3, d_4, d_5 = external
    return (
        liouville_structure_constant(p_1, d_2, d_1, precision=precision)
        * liouville_structure_constant(p_2, d_3, p_1, precision=precision)
        * liouville_structure_constant(d_5, d_4, p_2, precision=precision)
    )


__all__ = [
    "liouville_fivepoint_structure_product",
    "liouville_structure_constant",
    "liouville_weight",
]
