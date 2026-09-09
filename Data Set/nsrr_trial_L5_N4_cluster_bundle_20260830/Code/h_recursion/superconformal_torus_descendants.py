"""Independent low-level descendant sewing for N=1 torus one-point blocks.

This module is a small brute-force oracle for ``superconformal_torus_blocks``.
It does not use degenerate weights, null-vector residues, fusion polynomials,
or a Zamolodchikov recursion.  Instead it contracts explicit Gram and
three-point matrices obtained from the ordinary-c super-Virasoro algebra and
the HJS/Suchanek three-form Ward identities.

The implemented levels are deliberately small:

* NS: levels 0, 1/2, and 1;
* generic long R: the even level-zero state and the level-one basis
  ``(L_-1 w+, G_-1 G_0 w+)``.

They are enough to test the first NS spin-lifted term and the first genuinely
Ramond sewing matrix, including its G_0 ground-state dependence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

from ramond_descendant_blocks import RamondVermaModule
from superconformal_torus_blocks import ramond_positive_character_coefficients


Matrix = Tuple[Tuple[complex, ...], ...]


def _trace_inverse_times(left: Matrix, right: Matrix) -> complex:
    """Return ``Tr(left^{-1} right)`` for a one- or two-dimensional basis."""

    if len(left) != len(right) or any(
        len(row) != len(left) for row in left + right
    ):
        raise ValueError(
            "Gram and vertex matrices must be square and equal-sized"
        )
    if len(left) == 1:
        return right[0][0] / left[0][0]
    if len(left) != 2:
        raise ValueError(
            "the low-level oracle supports matrices of size at most two"
        )

    a, b = left[0]
    c, d = left[1]
    determinant = a * d - b * c
    if abs(determinant) == 0.0:
        raise ZeroDivisionError("the low-level Gram matrix is singular")
    inverse = (
        (d / determinant, -b / determinant),
        (-c / determinant, a / determinant),
    )
    return sum(
        inverse[row][column] * right[column][row]
        for row in range(2)
        for column in range(2)
    )


@dataclass(frozen=True)
class BruteForceNSTorusOnePointBlock:
    """Direct NS sewing through level one for a primary external insertion."""

    internal_weight: complex
    external_weight: complex

    def __post_init__(self) -> None:
        object.__setattr__(self, "internal_weight", complex(self.internal_weight))
        object.__setattr__(self, "external_weight", complex(self.external_weight))
        if abs(self.internal_weight) == 0.0:
            raise ValueError("the low-level NS Gram matrix is singular at h=0")

    def gram_matrices(self) -> Dict[int, Matrix]:
        """Return Gram matrices keyed by twice the descendant level."""

        h = self.internal_weight
        return {
            0: ((1.0 + 0.0j,),),
            1: ((2.0 * h,),),
            2: ((2.0 * h,),),
        }

    def vertex_matrices(self) -> Dict[int, Matrix]:
        """Return Ward-reduced three-point matrices through level one."""

        h = self.internal_weight
        d = self.external_weight
        return {
            0: ((1.0 + 0.0j,),),
            # rho(G_-1/2 h, nu_d, G_-1/2 h) = 2h-d.
            1: ((2.0 * h - d,),),
            # rho(L_-1 h, nu_d, L_-1 h) = 2h+d(d-1).
            2: ((2.0 * h + d * (d - 1.0),),),
        }

    def raw_coefficients(self) -> Dict[int, complex]:
        """Plane-frame coefficients ``F_f`` keyed by twice the level."""

        gram = self.gram_matrices()
        vertex = self.vertex_matrices()
        return {
            twice_level: _trace_inverse_times(
                gram[twice_level], vertex[twice_level]
            )
            for twice_level in (0, 1, 2)
        }

    def elliptic_coefficients(self) -> Dict[int, complex]:
        """Remove the NS Verma character and return ``H_0,H_1/2,H_1``."""

        raw = self.raw_coefficients()
        # The NS character starts 1+q^(1/2)+q+O(q^(3/2)).
        h0 = raw[0]
        h_half = raw[1] - h0
        h_one = raw[2] - h0 - h_half
        return {0: h0, 1: h_half, 2: h_one}


@dataclass(frozen=True)
class BruteForceRamondTorusOnePointBlock:
    """Direct generic long-R even block through level one."""

    central_charge: complex
    internal_weight: complex
    external_weight: complex
    sign: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "central_charge", complex(self.central_charge))
        object.__setattr__(self, "internal_weight", complex(self.internal_weight))
        object.__setattr__(self, "external_weight", complex(self.external_weight))
        sign = int(self.sign)
        if sign not in (-1, 1):
            raise ValueError("sign must be +1 or -1")
        object.__setattr__(self, "sign", sign)
        if abs(self.kappa_squared) == 0.0:
            raise ValueError(
                "the generic long-R oracle excludes the short state h=c/24"
            )

    @property
    def kappa_squared(self) -> complex:
        return self.internal_weight - self.central_charge / 24.0

    @property
    def module(self) -> RamondVermaModule:
        return RamondVermaModule(
            c=self.central_charge,
            weight=self.internal_weight,
        )

    def gram_matrices(self) -> Tuple[Matrix, Matrix]:
        """Return level-zero and level-one even Gram matrices."""

        return (
            ((1.0 + 0.0j,),),
            self.module.gram_matrix(1, 0),
        )

    def vertex_matrices(self) -> Tuple[Matrix, Matrix]:
        """Return the Ward-reduced even R--NS--R matrices.

        The level-one basis is ``(L_-1 w+, G_-1 G_0 w+)``.  The off-diagonal
        and lower-right entries follow by retaining the unnormalized
        ``G_0 w+`` ground state until after applying the RR Ward identities.
        """

        c = self.central_charge
        h = self.internal_weight
        d = self.external_weight
        kappa = self.kappa_squared
        sign = self.sign
        off_diagonal = 1.5 * kappa + (d - 1.0) * kappa * (1.0 - sign)
        level_one: Matrix = (
            (
                2.0 * h + d * (d - 1.0),
                off_diagonal,
            ),
            (
                off_diagonal,
                sign
                * kappa
                * (2.0 * h + c / 4.0 - 2.0 * d),
            ),
        )
        return (((1.0 + 0.0j,),), level_one)

    def raw_even_coefficients(self) -> Tuple[complex, complex]:
        """Return the HJS plane-frame even coefficients ``F_0,F_1``."""

        gram = self.gram_matrices()
        vertex = self.vertex_matrices()
        return (
            _trace_inverse_times(gram[0], vertex[0]),
            _trace_inverse_times(gram[1], vertex[1]),
        )

    def elliptic_coefficients(self) -> Tuple[complex, complex]:
        """Return ``H_0,H_1`` in the two HJS sign sectors."""

        raw = self.raw_even_coefficients()
        if self.sign == -1:
            return raw
        character = ramond_positive_character_coefficients(1)
        return (raw[0], raw[1] - character[1] * raw[0])


__all__ = [
    "BruteForceNSTorusOnePointBlock",
    "BruteForceRamondTorusOnePointBlock",
]
