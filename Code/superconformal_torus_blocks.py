"""Genus-one N=1 superconformal blocks with explicit NS/R sewing data.

This module implements the toric elliptic recursions of

    Hadasz--Jaskolski--Suchanek, arXiv:1207.5740,

in the ordinary-central-charge convention used by BRY.  The puncture is an
NS primary.  The handle can carry either

* an NS Verma module, with half-integer levels and an explicit lift of the
  plumbing parameter; or
* a generic long-R module, with integer levels and an explicit two-state
  ground fiber.

The Ramond ground basis is deliberately unnormalized,

    e_+ = w^+,  e_- = G_0 w^+.

It remains regular as h -> c/24 and makes the shortening zero visible.  The
scalar HJS ``+`` and ``-`` toric blocks are projections of this ground-fiber
data; higher-genus code must retain the matrix until the R cycle is closed.

At rational central charge individual Kac terms are resonant.  The classes
below are the generic-b kernels.  A Type-0B production wrapper must take the
finite part of each assembled coefficient at b=1, as in
``self_dual_superconformal_blocks.py``.
"""

from __future__ import annotations

import cmath
from dataclasses import dataclass
from functools import lru_cache
import math
from typing import Dict, Literal, Tuple, Union

from mixed_ramond_sphere_blocks import (
    _ns_a_factor,
    _ns_degenerate_weight,
    _ns_ns_fusion_polynomial,
    _r_a_beta,
    _r_beta_prime,
    _r_beta_rs,
    _r_ns_fusion_polynomial,
)
from ramond_sphere_blocks import ramond_beta, ramond_liouville_weight
from self_dual_superconformal_blocks import (
    FinitePartDiagnostics,
    _dictionary_finite_part,
)
from superconformal_blocks import (
    central_charge,
    ns_liouville_weight,
)


Number = Union[complex, float]
Matrix2 = Tuple[Tuple[complex, complex], Tuple[complex, complex]]
RamondCycleInsertion = Literal["identity", "parity", "g0", "parity_g0"]
TorusEdgeSector = Literal["NS", "R"]
TorusSpinLabel = Literal["NS", "NS_tilde", "R", "R_tilde", "mixed"]


def _validate_sign(value: int, name: str) -> int:
    value = int(value)
    if value not in (-1, 1):
        raise ValueError(f"{name} must be +1 or -1")
    return value


def _matmul(left: Matrix2, right: Matrix2) -> Matrix2:
    return tuple(
        tuple(
            sum(left[row][inner] * right[inner][column] for inner in range(2))
            for column in range(2)
        )
        for row in range(2)
    )  # type: ignore[return-value]


@dataclass(frozen=True)
class NSPlumbingParameter:
    """Reduced NS plumbing parameter together with its spin lift."""

    q: complex
    lift_sign: int = 1

    def __post_init__(self) -> None:
        q = complex(self.q)
        if not 0.0 < abs(q) < 1.0:
            raise ValueError("NS plumbing q must satisfy 0 < |q| < 1")
        object.__setattr__(self, "q", q)
        object.__setattr__(
            self, "lift_sign", _validate_sign(self.lift_sign, "lift_sign")
        )

    def level_factor(self, twice_level: int) -> complex:
        """Return the lifted factor for a state at level ``twice_level/2``."""

        if not isinstance(twice_level, int) or twice_level < 0:
            raise ValueError("twice_level must be a nonnegative integer")
        return (
            self.lift_sign**twice_level
            * self.q ** (twice_level / 2.0)
        )


@dataclass(frozen=True)
class RamondPlumbingParameter:
    """Reduced R plumbing parameter and the operator closing its ground fiber."""

    q: complex
    cycle_insertion: RamondCycleInsertion = "identity"

    def __post_init__(self) -> None:
        q = complex(self.q)
        if not 0.0 < abs(q) < 1.0:
            raise ValueError("R plumbing q must satisfy 0 < |q| < 1")
        if self.cycle_insertion not in (
            "identity",
            "parity",
            "g0",
            "parity_g0",
        ):
            raise ValueError("unsupported Ramond cycle insertion")
        object.__setattr__(self, "q", q)

    def level_factor(self, level: int) -> complex:
        if not isinstance(level, int) or level < 0:
            raise ValueError("R level must be a nonnegative integer")
        return self.q**level


@dataclass(frozen=True)
class TorusTwoPointSpinStructure:
    """Spin and sector ledger for a two-edge torus necklace.

    The sectors on adjacent necklace edges determine the puncture sector:
    equal sectors require two NS punctures, while an NS--R transition
    requires a Ramond puncture.  Hence a mixed necklace necessarily carries
    two external Ramond fields.

    For two NS edges the physical temporal holonomy is the product of the
    two local NS lifts.  For two R edges it is the parity of the number of
    ``(-1)^F`` insertions.  In the mixed case the NS lift and the R-cycle
    insertion are independent sewing data and give the four spin structures
    on the twice-R-punctured torus.
    """

    edge_sector_1: TorusEdgeSector
    edge_sector_2: TorusEdgeSector
    ns_lift_sign_1: int = 1
    ns_lift_sign_2: int = 1
    r_cycle_insertion_1: RamondCycleInsertion = "identity"
    r_cycle_insertion_2: RamondCycleInsertion = "identity"

    def __post_init__(self) -> None:
        for name in ("edge_sector_1", "edge_sector_2"):
            if getattr(self, name) not in ("NS", "R"):
                raise ValueError(f"{name} must be 'NS' or 'R'")
        for index in (1, 2):
            sector = getattr(self, f"edge_sector_{index}")
            lift = _validate_sign(
                getattr(self, f"ns_lift_sign_{index}"),
                f"ns_lift_sign_{index}",
            )
            insertion = getattr(self, f"r_cycle_insertion_{index}")
            if insertion not in (
                "identity",
                "parity",
                "g0",
                "parity_g0",
            ):
                raise ValueError(
                    f"unsupported r_cycle_insertion_{index}"
                )
            object.__setattr__(self, f"ns_lift_sign_{index}", lift)
            if sector == "NS" and insertion != "identity":
                raise ValueError(
                    f"edge {index} is NS and cannot carry an R-cycle "
                    "insertion"
                )
            if sector == "R" and lift != 1:
                raise ValueError(
                    f"edge {index} is R and cannot carry an NS lift"
                )

    @property
    def edge_sectors(self) -> Tuple[TorusEdgeSector, TorusEdgeSector]:
        return (self.edge_sector_1, self.edge_sector_2)

    @property
    def external_sectors(self) -> Tuple[TorusEdgeSector, TorusEdgeSector]:
        sector: TorusEdgeSector = (
            "NS" if self.edge_sector_1 == self.edge_sector_2 else "R"
        )
        return (sector, sector)

    @property
    def spin_label(self) -> TorusSpinLabel:
        if self.edge_sectors == ("NS", "NS"):
            return (
                "NS"
                if self.ns_lift_sign_1 * self.ns_lift_sign_2 == 1
                else "NS_tilde"
            )
        if self.edge_sectors == ("R", "R"):
            odd_parity = (
                self.r_cycle_insertion_1 in ("parity", "parity_g0")
            ) ^ (
                self.r_cycle_insertion_2 in ("parity", "parity_g0")
            )
            return "R_tilde" if odd_parity else "R"
        return "mixed"

    @property
    def mixed_spin_bits(self) -> Tuple[int, int]:
        """Return ``(NS lift, R parity)`` for a mixed necklace."""

        if self.spin_label != "mixed":
            raise ValueError("mixed_spin_bits requires one NS and one R edge")
        if self.edge_sector_1 == "NS":
            lift = self.ns_lift_sign_1
            insertion = self.r_cycle_insertion_2
        else:
            lift = self.ns_lift_sign_2
            insertion = self.r_cycle_insertion_1
        return (
            lift,
            -1 if insertion in ("parity", "parity_g0") else 1,
        )

    def plumbing_parameters(
        self, q1: Number, q2: Number
    ) -> Tuple[
        Union[NSPlumbingParameter, RamondPlumbingParameter],
        Union[NSPlumbingParameter, RamondPlumbingParameter],
    ]:
        parameters = []
        for index, q in ((1, q1), (2, q2)):
            if getattr(self, f"edge_sector_{index}") == "NS":
                parameters.append(
                    NSPlumbingParameter(
                        q, getattr(self, f"ns_lift_sign_{index}")
                    )
                )
            else:
                parameters.append(
                    RamondPlumbingParameter(
                        q, getattr(self, f"r_cycle_insertion_{index}")
                    )
                )
        return (parameters[0], parameters[1])


class RamondGroundFiber:
    """Unnormalized long-R ground fiber and its elementary sewing matrices."""

    def __init__(self, *, c: Number, weight: Number) -> None:
        self.c = complex(c)
        self.weight = complex(weight)
        self.kappa_squared = self.weight - self.c / 24.0

    @property
    def gram(self) -> Matrix2:
        return (
            (1.0 + 0.0j, 0.0j),
            (0.0j, self.kappa_squared),
        )

    @property
    def g0(self) -> Matrix2:
        # Columns are G0 e_+ and G0 e_-.
        return (
            (0.0j, self.kappa_squared),
            (1.0 + 0.0j, 0.0j),
        )

    @property
    def fermion_parity(self) -> Matrix2:
        return (
            (1.0 + 0.0j, 0.0j),
            (0.0j, -1.0 + 0.0j),
        )

    def even_vertex(self, sign: int) -> Matrix2:
        """Ground matrix of the normalized HJS even R--NS--R form."""

        sign = _validate_sign(sign, "sign")
        return (
            (1.0 + 0.0j, 0.0j),
            (0.0j, sign * self.kappa_squared),
        )

    def insertion_matrix(self, insertion: RamondCycleInsertion) -> Matrix2:
        identity: Matrix2 = (
            (1.0 + 0.0j, 0.0j),
            (0.0j, 1.0 + 0.0j),
        )
        if insertion == "identity":
            return identity
        if insertion == "parity":
            return self.fermion_parity
        if insertion == "g0":
            return self.g0
        if insertion == "parity_g0":
            return _matmul(self.fermion_parity, self.g0)
        raise ValueError("unsupported Ramond cycle insertion")

    def contract(
        self,
        vertex: Matrix2,
        insertion: RamondCycleInsertion = "identity",
    ) -> complex:
        """Return Tr(B^{-1} S V) without normalizing the ground doublet."""

        if abs(self.kappa_squared) == 0.0:
            raise ValueError(
                "the long-R contraction is singular at h=c/24; "
                "project to the short quotient first"
            )
        inverse_gram: Matrix2 = (
            (1.0 + 0.0j, 0.0j),
            (0.0j, 1.0 / self.kappa_squared),
        )
        product = _matmul(
            inverse_gram,
            _matmul(self.insertion_matrix(insertion), vertex),
        )
        return product[0][0] + product[1][1]


def ns_verma_character_coefficients(
    max_twice_level: int, *, lift_sign: int = 1
) -> Dict[int, complex]:
    """Return NS Verma-character coefficients keyed by twice the level."""

    if not isinstance(max_twice_level, int) or max_twice_level < 0:
        raise ValueError("max_twice_level must be a nonnegative integer")
    lift_sign = _validate_sign(lift_sign, "lift_sign")
    coefficients = [0.0j] * (max_twice_level + 1)
    coefficients[0] = 1.0 + 0.0j

    # Bosonic L_-n oscillators.
    for mode in range(2, max_twice_level + 1, 2):
        for power in range(mode, max_twice_level + 1):
            coefficients[power] += coefficients[power - mode]

    # Fermionic G_{-(n-1/2)} oscillators.
    for mode in range(1, max_twice_level + 1, 2):
        for power in range(max_twice_level, mode - 1, -1):
            coefficients[power] += lift_sign * coefficients[power - mode]

    return {
        twice_level: value
        for twice_level, value in enumerate(coefficients)
        if value != 0
    }


def ramond_verma_character_coefficients(
    order: int, *, parity_sign: int = 1
) -> Tuple[int, ...]:
    """Normalized long-R character with an optional ``(-1)^F`` insertion."""

    if not isinstance(order, int) or order < 0:
        raise ValueError("order must be a nonnegative integer")
    parity_sign = _validate_sign(parity_sign, "parity_sign")
    coefficients = [0] * (order + 1)
    coefficients[0] = 1
    for mode in range(1, order + 1):
        for level in range(mode, order + 1):
            coefficients[level] += coefficients[level - mode]
        for level in range(order, mode - 1, -1):
            coefficients[level] += (
                parity_sign * coefficients[level - mode]
            )
    return tuple(coefficients)


def ramond_positive_character_coefficients(order: int) -> Tuple[int, ...]:
    """Dimensions of one long-R parity block through ``order``."""

    return ramond_verma_character_coefficients(order, parity_sign=1)


class NSTorusOnePointBlock:
    """NS-handle elliptic block for one external NS primary."""

    def __init__(
        self,
        *,
        b: Number,
        internal_weight: Number,
        external_weight: Number,
        pole_tolerance: float = 1.0e-12,
    ) -> None:
        self.b = complex(b)
        self.c = central_charge(self.b)
        self.internal_weight = complex(internal_weight)
        self.external_weight = complex(external_weight)
        self.pole_tolerance = float(pole_tolerance)

    @lru_cache(maxsize=None)
    def _coefficient(
        self, twice_level: int, internal_weight: complex
    ) -> complex:
        result = 1.0 + 0.0j if twice_level == 0 else 0.0j
        for r in range(1, twice_level + 1):
            for s in range(1, twice_level // r + 1):
                product = r * s
                if product > twice_level or (r + s) % 2:
                    continue
                delta_rs = _ns_degenerate_weight(self.b, r, s)
                denominator = internal_weight - delta_rs
                if abs(denominator) < self.pole_tolerance:
                    raise ZeroDivisionError(
                        f"internal weight is too close to the ({r},{s}) NS pole"
                    )
                shifted_weight = delta_rs + product / 2.0
                left = _ns_ns_fusion_polynomial(
                    b=self.b,
                    r=r,
                    s=s,
                    lower_weight=shifted_weight,
                    upper_weight=self.external_weight,
                    starred=bool(product % 2),
                )
                right = _ns_ns_fusion_polynomial(
                    b=self.b,
                    r=r,
                    s=s,
                    lower_weight=delta_rs,
                    upper_weight=self.external_weight,
                    starred=False,
                )
                # For an unstarred external NS primary, the toric residue at
                # half-integer level carries the HJS sewing sign
                # \widetilde{s}_{rs}=(-1)^{rs}.  It is essential already for
                # H_{1/2}=-Delta_lambda/(2 Delta); omitting it leaves the
                # integer-level coefficient unchanged but reverses every
                # leading half-integer Ward check.
                sewing_sign = -1.0 if product % 2 else 1.0
                result += (
                    sewing_sign
                    * _ns_a_factor(
                        self.b, r, s, self.pole_tolerance
                    )
                    * left
                    * right
                    / denominator
                    * self._coefficient(
                        twice_level - product, shifted_weight
                    )
                )
        return result

    def elliptic_coefficients(self, max_twice_level: int) -> Dict[int, complex]:
        if not isinstance(max_twice_level, int) or max_twice_level < 0:
            raise ValueError("max_twice_level must be a nonnegative integer")
        return {
            twice_level: self._coefficient(
                twice_level, self.internal_weight
            )
            for twice_level in range(max_twice_level + 1)
        }

    def raw_coefficients(self, max_twice_level: int) -> Dict[int, complex]:
        """Return plane-frame coefficients after restoring the NS character."""

        elliptic = self.elliptic_coefficients(max_twice_level)
        character = ns_verma_character_coefficients(max_twice_level)
        return {
            twice_level: sum(
                character.get(offset, 0.0j)
                * elliptic.get(twice_level - offset, 0.0j)
                for offset in range(twice_level + 1)
            )
            for twice_level in range(max_twice_level + 1)
        }

    def evaluate(
        self, plumbing: NSPlumbingParameter, max_twice_level: int
    ) -> complex:
        return sum(
            coefficient * plumbing.level_factor(twice_level)
            for twice_level, coefficient in self.elliptic_coefficients(
                max_twice_level
            ).items()
        )

    def evaluate_raw(
        self, plumbing: NSPlumbingParameter, max_twice_level: int
    ) -> complex:
        """Evaluate the plane-frame descendant series in a chosen NS lift."""

        return sum(
            coefficient * plumbing.level_factor(twice_level)
            for twice_level, coefficient in self.raw_coefficients(
                max_twice_level
            ).items()
        )

    def chiral_block(
        self, plumbing: NSPlumbingParameter, max_twice_level: int
    ) -> complex:
        r"""Return \(q^{h-c/24}\) times the plane-frame descendant series."""

        return (
            plumbing.q ** (self.internal_weight - self.c / 24.0)
            * self.evaluate_raw(plumbing, max_twice_level)
        )


class RamondTorusOnePointBlock:
    """Generic long-R toric block for one external NS primary."""

    def __init__(
        self,
        *,
        b: Number,
        internal_beta: Number,
        external_weight: Number,
        sign: int = 1,
        pole_tolerance: float = 1.0e-12,
    ) -> None:
        self.b = complex(b)
        self.c = central_charge(self.b)
        self.internal_beta = complex(internal_beta)
        if abs(self.internal_beta) <= pole_tolerance:
            raise ValueError(
                "the generic long-R toric block requires nonzero beta; "
                "construct the short quotient separately"
            )
        self.external_weight = complex(external_weight)
        self.sign = _validate_sign(sign, "sign")
        self.pole_tolerance = float(pole_tolerance)

    @lru_cache(maxsize=None)
    def _coefficient(self, level: int, internal_beta: complex) -> complex:
        result = 1.0 + 0.0j if level == 0 else 0.0j
        internal_weight = self.c / 24.0 - internal_beta**2
        for r in range(1, 2 * level + 1):
            for s in range(1, (2 * level) // r + 1):
                product = r * s
                if product > 2 * level or (r + s) % 2 != 1 or product % 2:
                    continue
                shift = product // 2
                beta_rs = _r_beta_rs(self.b, r, s)
                beta_prime = _r_beta_prime(self.b, r, s)
                delta_rs = self.c / 24.0 - beta_rs**2
                denominator = internal_weight - delta_rs
                if abs(denominator) < self.pole_tolerance:
                    raise ZeroDivisionError(
                        f"internal weight is too close to the ({r},{s}) R pole"
                    )
                # mixed_ramond_sphere_blocks stores the beta-pole residue
                # A^(beta)=-A^(h)/(2 beta_rs).
                a_weight = (
                    -2.0
                    * beta_rs
                    * _r_a_beta(
                        self.b, r, s, self.pole_tolerance
                    )
                )
                left = _r_ns_fusion_polynomial(
                    b=self.b,
                    r=r,
                    s=s,
                    ramond_beta_value=beta_prime,
                    ns_weight=self.external_weight,
                    sign=self.sign,
                )
                right = _r_ns_fusion_polynomial(
                    b=self.b,
                    r=r,
                    s=s,
                    ramond_beta_value=beta_rs,
                    ns_weight=self.external_weight,
                    sign=self.sign,
                )
                result += (
                    a_weight
                    * left
                    * right
                    / denominator
                    * self._coefficient(level - shift, beta_prime)
                )
        return result

    @property
    def internal_weight(self) -> complex:
        return self.c / 24.0 - self.internal_beta**2

    @property
    def ground_fiber(self) -> RamondGroundFiber:
        return RamondGroundFiber(
            c=self.c, weight=self.internal_weight
        )

    def elliptic_coefficients(self, order: int) -> Tuple[complex, ...]:
        if not isinstance(order, int) or order < 0:
            raise ValueError("order must be a nonnegative integer")
        return tuple(
            self._coefficient(level, self.internal_beta)
            for level in range(order + 1)
        )

    def raw_even_coefficients(self, order: int) -> Tuple[complex, ...]:
        """Return the HJS plane-frame even-block coefficients."""

        elliptic = self.elliptic_coefficients(order)
        if self.sign == -1:
            return elliptic
        character = ramond_positive_character_coefficients(order)
        return tuple(
            sum(
                character[offset] * elliptic[level - offset]
                for offset in range(level + 1)
            )
            for level in range(order + 1)
        )

    def cycle_projected_raw_coefficients(
        self, plumbing: RamondPlumbingParameter, order: int
    ) -> Tuple[complex, ...]:
        """Close the R ground fiber after forming the scalar HJS even block."""

        ground_factor = self.ground_fiber.contract(
            self.ground_fiber.even_vertex(self.sign),
            plumbing.cycle_insertion,
        )
        return tuple(
            ground_factor * coefficient
            for coefficient in self.raw_even_coefficients(order)
        )

    def evaluate_elliptic(
        self, plumbing: RamondPlumbingParameter, order: int
    ) -> complex:
        return sum(
            coefficient * plumbing.level_factor(level)
            for level, coefficient in enumerate(
                self.elliptic_coefficients(order)
            )
        )


class SelfDualNSTorusOnePointBlock:
    """Type-0B NS-handle block from coefficient-wise finite parts at b=1."""

    def __init__(
        self,
        *,
        internal_momentum: Number,
        external_momentum: Number,
        radius: float = 0.04,
        check_radius: float = 0.05,
        samples: int = 24,
    ) -> None:
        self.internal_momentum = complex(internal_momentum)
        self.external_momentum = complex(external_momentum)
        self.radius = float(radius)
        self.check_radius = float(check_radius)
        self.samples = int(samples)
        self._cache: Dict[
            int,
            Tuple[
                Dict[int, complex],
                Dict[int, FinitePartDiagnostics],
            ],
        ] = {}

    def _block_at(self, b: complex) -> NSTorusOnePointBlock:
        return NSTorusOnePointBlock(
            b=b,
            internal_weight=ns_liouville_weight(
                self.internal_momentum, b
            ),
            external_weight=ns_liouville_weight(
                self.external_momentum, b
            ),
        )

    @property
    def internal_weight(self) -> complex:
        return ns_liouville_weight(self.internal_momentum, 1.0)

    def _data(
        self, max_twice_level: int
    ) -> Tuple[
        Dict[int, complex],
        Dict[int, FinitePartDiagnostics],
    ]:
        if not isinstance(max_twice_level, int) or max_twice_level < 0:
            raise ValueError("max_twice_level must be a nonnegative integer")
        if max_twice_level not in self._cache:
            keys = tuple(range(max_twice_level + 1))
            self._cache[max_twice_level] = _dictionary_finite_part(
                lambda b: self._block_at(b).elliptic_coefficients(
                    max_twice_level
                ),
                keys=keys,
                radius=self.radius,
                check_radius=self.check_radius,
                samples=self.samples,
            )
        return self._cache[max_twice_level]

    def elliptic_coefficients(self, max_twice_level: int) -> Dict[int, complex]:
        return dict(self._data(max_twice_level)[0])

    def coefficient_diagnostics(
        self, max_twice_level: int
    ) -> Dict[int, FinitePartDiagnostics]:
        return dict(self._data(max_twice_level)[1])

    def raw_coefficients(self, max_twice_level: int) -> Dict[int, complex]:
        """Return exact-Type-0B plane-frame coefficients."""

        elliptic = self.elliptic_coefficients(max_twice_level)
        character = ns_verma_character_coefficients(max_twice_level)
        return {
            twice_level: sum(
                character.get(offset, 0.0j)
                * elliptic.get(twice_level - offset, 0.0j)
                for offset in range(twice_level + 1)
            )
            for twice_level in range(max_twice_level + 1)
        }

    def evaluate(
        self, plumbing: NSPlumbingParameter, max_twice_level: int
    ) -> complex:
        return sum(
            coefficient * plumbing.level_factor(twice_level)
            for twice_level, coefficient in self.elliptic_coefficients(
                max_twice_level
            ).items()
        )

    def evaluate_raw(
        self, plumbing: NSPlumbingParameter, max_twice_level: int
    ) -> complex:
        return sum(
            coefficient * plumbing.level_factor(twice_level)
            for twice_level, coefficient in self.raw_coefficients(
                max_twice_level
            ).items()
        )

    def chiral_block(
        self, plumbing: NSPlumbingParameter, max_twice_level: int
    ) -> complex:
        r"""Return the exact-Type-0B block \(q^{h-c/24}F(q)\)."""

        return (
            plumbing.q
            ** (self.internal_weight - central_charge(1.0) / 24.0)
            * self.evaluate_raw(plumbing, max_twice_level)
        )


class SelfDualRamondTorusOnePointBlock:
    """Type-0B long-R toric block from finite parts at exact c=27/2."""

    def __init__(
        self,
        *,
        internal_momentum: Number,
        external_momentum: Number,
        sign: int = 1,
        radius: float = 0.04,
        check_radius: float = 0.05,
        samples: int = 24,
    ) -> None:
        self.internal_momentum = complex(internal_momentum)
        if abs(self.internal_momentum) == 0.0:
            raise ValueError(
                "the generic long-R toric block requires nonzero momentum"
            )
        self.external_momentum = complex(external_momentum)
        self.sign = _validate_sign(sign, "sign")
        self.radius = float(radius)
        self.check_radius = float(check_radius)
        self.samples = int(samples)
        self._cache: Dict[
            int,
            Tuple[
                Dict[int, complex],
                Dict[int, FinitePartDiagnostics],
            ],
        ] = {}

    def _block_at(self, b: complex) -> RamondTorusOnePointBlock:
        return RamondTorusOnePointBlock(
            b=b,
            internal_beta=ramond_beta(self.internal_momentum),
            external_weight=ns_liouville_weight(
                self.external_momentum, b
            ),
            sign=self.sign,
        )

    def _data(
        self, order: int
    ) -> Tuple[
        Dict[int, complex],
        Dict[int, FinitePartDiagnostics],
    ]:
        if not isinstance(order, int) or order < 0:
            raise ValueError("order must be a nonnegative integer")
        if order not in self._cache:
            keys = tuple(range(order + 1))
            self._cache[order] = _dictionary_finite_part(
                lambda b: {
                    level: coefficient
                    for level, coefficient in enumerate(
                        self._block_at(b).elliptic_coefficients(order)
                    )
                },
                keys=keys,
                radius=self.radius,
                check_radius=self.check_radius,
                samples=self.samples,
            )
        return self._cache[order]

    @property
    def internal_weight(self) -> complex:
        return ramond_liouville_weight(
            self.internal_momentum, 1.0
        )

    @property
    def ground_fiber(self) -> RamondGroundFiber:
        return RamondGroundFiber(
            c=central_charge(1.0),
            weight=self.internal_weight,
        )

    def elliptic_coefficients(self, order: int) -> Tuple[complex, ...]:
        values = self._data(order)[0]
        return tuple(values[level] for level in range(order + 1))

    def coefficient_diagnostics(
        self, order: int
    ) -> Dict[int, FinitePartDiagnostics]:
        return dict(self._data(order)[1])

    def raw_even_coefficients(self, order: int) -> Tuple[complex, ...]:
        elliptic = self.elliptic_coefficients(order)
        if self.sign == -1:
            return elliptic
        character = ramond_positive_character_coefficients(order)
        return tuple(
            sum(
                character[offset] * elliptic[level - offset]
                for offset in range(level + 1)
            )
            for level in range(order + 1)
        )

    def cycle_projected_raw_coefficients(
        self, plumbing: RamondPlumbingParameter, order: int
    ) -> Tuple[complex, ...]:
        ground_factor = self.ground_fiber.contract(
            self.ground_fiber.even_vertex(self.sign),
            plumbing.cycle_insertion,
        )
        return tuple(
            ground_factor * coefficient
            for coefficient in self.raw_even_coefficients(order)
        )

    def evaluate_elliptic(
        self, plumbing: RamondPlumbingParameter, order: int
    ) -> complex:
        return sum(
            coefficient * plumbing.level_factor(level)
            for level, coefficient in enumerate(
                self.elliptic_coefficients(order)
            )
        )


__all__ = [
    "NSPlumbingParameter",
    "NSTorusOnePointBlock",
    "RamondGroundFiber",
    "RamondPlumbingParameter",
    "TorusTwoPointSpinStructure",
    "RamondTorusOnePointBlock",
    "SelfDualNSTorusOnePointBlock",
    "SelfDualRamondTorusOnePointBlock",
    "ns_verma_character_coefficients",
    "ramond_positive_character_coefficients",
    "ramond_verma_character_coefficients",
]
