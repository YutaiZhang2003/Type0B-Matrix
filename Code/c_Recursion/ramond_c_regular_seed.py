"""Representation-theoretic regular seed for mixed Ramond c-recursion.

For sign-resolved Ramond blocks at fixed external weights, the relation

    beta_i(c)^2 = c/24 - h_i^R

makes the coefficients algebraic rather than rational functions of ``c``.
The pole data alone therefore do not reconstruct the coefficient from its
strict value at ``c=infinity``.  The missing analytic part is obtained here
directly from ordinary-c Ward/Gram sewing:

    regular = direct coefficient - sum of all internal Kac pole parts.

This is independent of the HJS h-recursion.  It is slower than a closed light
block, but it gives an exact low-level seed and is the appropriate oracle for
deriving the contracted Ramond light block.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from mixed_ns_ramond_descendant_blocks import (
    BruteForceMixedNSExchangeSphereBlock,
)
from ramond_c_recursive_sphere_blocks import (
    CRecursiveMixedNSExchangeSphereFourPointBlock,
)


Parity = Literal["even", "odd"]


class DirectMixedRamondRegularSeed:
    """Exact low-level regular part for the NS-channel NSNSRR block."""

    def __init__(
        self,
        *,
        h1_r: complex,
        h2_r: complex,
        h3_ns: complex,
        h4_ns: complex,
        pole_tolerance: float = 1.0e-12,
    ) -> None:
        self.h1 = complex(h1_r)
        self.h2 = complex(h2_r)
        self.h3 = complex(h3_ns)
        self.h4 = complex(h4_ns)
        self.pole_tolerance = float(pole_tolerance)

    @lru_cache(maxsize=None)
    def direct_coefficient(
        self,
        twice_level: int,
        internal_weight: complex,
        c: complex,
        parity: Parity,
        sign2: int,
    ) -> complex:
        block = BruteForceMixedNSExchangeSphereBlock(
            c=c,
            h1_r=self.h1,
            h2_r=self.h2,
            h3_ns=self.h3,
            h4_ns=self.h4,
            internal_weight=internal_weight,
            sign2=sign2,
        )
        return block.coefficient(twice_level, parity)

    @lru_cache(maxsize=None)
    def coefficient(
        self,
        twice_level: int,
        internal_weight: complex,
        c: complex,
        parity: Parity,
        sign2: int,
    ) -> complex:
        """Return the analytic-internal-pole regular part at one level."""

        if sign2 not in (-1, 1):
            raise ValueError("sign2 must be +1 or -1")
        result = self.direct_coefficient(
            twice_level,
            internal_weight,
            c,
            parity,
            sign2,
        )
        kernel = CRecursiveMixedNSExchangeSphereFourPointBlock(
            c=c,
            h1_r=self.h1,
            h2_r=self.h2,
            h3_ns=self.h3,
            h4_ns=self.h4,
            internal_weight=internal_weight,
            sign2=sign2,
            pole_tolerance=self.pole_tolerance,
        )
        for r in range(2, twice_level + 1):
            for s in range(1, twice_level // r + 1):
                product = r * s
                if product > twice_level or (r + s) % 2:
                    continue
                (
                    residue,
                    c_pole,
                    next_parity,
                    _,
                    next_sign2,
                ) = kernel._pole_kernel(
                    r=r,
                    s=s,
                    internal_weight=internal_weight,
                    c=c,
                    parity=parity,
                    sign3=1,
                    sign2=sign2,
                )
                denominator = c - c_pole
                scale = max(1.0, abs(c), abs(c_pole))
                if abs(denominator) <= self.pole_tolerance * scale:
                    raise ZeroDivisionError(
                        f"regular seed encountered the ({r},{s}) NS pole"
                    )
                result -= (
                    residue
                    / denominator
                    * self.direct_coefficient(
                        twice_level - product,
                        internal_weight + product / 2.0,
                        c_pole,
                        next_parity,
                        next_sign2,
                    )
                )
        return result

    def __call__(
        self,
        twice_level: int,
        internal_weight: complex,
        c: complex,
        parity: Parity,
        sign3: int,
        sign2: int,
    ) -> complex:
        if sign3 != 1:
            raise ValueError("the mixed NS channel has sign3=+1")
        return self.coefficient(
            twice_level,
            complex(internal_weight),
            complex(c),
            parity,
            sign2,
        )


__all__ = ["DirectMixedRamondRegularSeed"]
