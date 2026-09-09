"""NS h-recursive and R beta-recursive blocks on a two-punctured torus.

The two-punctured torus is assembled from two three-punctured spheres.  The
two internal NS edges have weights ``h1,h2`` and plumbing parameters
``q1,q2``; the external bottom-component primaries have weights ``d1,d2``.
In the annulus frame used by Belavin--Geiko the punctures are at ``1,q1`` and
the torus nome is ``q=q1*q2``.

``NSTorusTwoPointHRecursionBlock`` computes the generic-``b`` bivariate NS
series, and ``SelfDualNSTorusTwoPointHRecursionBlock`` takes the required
coefficient-wise finite parts at the Type-0B point.  Odd singular vectors
toggle an explicit two-state fermion routing around the necklace.

The direct NS sewing class remains a low-level oracle.  It gives every
coefficient with at most one global descendant on an edge:

    (level1,level2) = (0,0), (1,0), (0,1), (1/2,1/2).

The coefficients with exactly one half-integer level vanish by fermion
parity.  This is the controlled leading layer used to test the recursion.

For Ramond sewing each internal edge retains the unnormalized basis

    e_+ = w^+,  e_- = G_0 w^+,

until both RRNS vertices and the chosen cycle insertions have been
contracted.  The paired beta-pole recursion is implemented together with a
direct Ward/Gram regular seed through level one on each edge.  Its residue
kernel is all-level; higher-order production awaits the contracted Ramond
light seed.
"""

from __future__ import annotations

import cmath
from dataclasses import dataclass
from functools import lru_cache
import math
from typing import Callable, Dict, Literal, Tuple

from mixed_ramond_sphere_blocks import (
    _ns_a_factor,
    _ns_degenerate_weight,
    _ns_ns_fusion_polynomial,
    _r_a_beta,
    _r_beta_prime,
    _r_beta_rs,
    _r_ns_fusion_polynomial,
)
from ramond_descendant_blocks import (
    RamondThreePointWardMatrix,
    RamondVermaModule,
)
from ramond_sphere_blocks import ramond_beta, ramond_liouville_weight
from self_dual_superconformal_blocks import (
    FinitePartDiagnostics,
    _dictionary_finite_part,
)
from superconformal_blocks import central_charge, ns_liouville_weight
from superconformal_torus_blocks import (
    NSPlumbingParameter,
    RamondGroundFiber,
    RamondPlumbingParameter,
    ns_verma_character_coefficients,
    ramond_positive_character_coefficients,
    ramond_verma_character_coefficients,
)


LevelPair = Tuple[int, int]
Matrix2 = Tuple[Tuple[complex, complex], Tuple[complex, complex]]
Matrix = Tuple[Tuple[complex, ...], ...]
RamondRegularSeed = Callable[
    [int, int, complex, complex, int, int], complex
]


@lru_cache(maxsize=None)
def _ns_character_coefficient(twice_level: int) -> complex:
    return ns_verma_character_coefficients(twice_level).get(
        twice_level, 0.0j
    )


@lru_cache(maxsize=None)
def _r_character_coefficient(level: int, parity_sign: int) -> complex:
    return complex(
        ramond_verma_character_coefficients(
            level, parity_sign=parity_sign
        )[level]
    )


def _matmul2(left: Matrix2, right: Matrix2) -> Matrix2:
    return tuple(
        tuple(
            sum(
                left[row][inner] * right[inner][column]
                for inner in range(2)
            )
            for column in range(2)
        )
        for row in range(2)
    )  # type: ignore[return-value]


def _trace2(matrix: Matrix2) -> complex:
    return matrix[0][0] + matrix[1][1]


@dataclass(frozen=True)
class NSTorusTwoPointLeadingBlock:
    """Direct NS sewing of two bottom-component superconformal primaries."""

    central_charge: complex
    internal_weight_1: complex
    internal_weight_2: complex
    external_weight_1: complex
    external_weight_2: complex

    def __post_init__(self) -> None:
        for name in (
            "central_charge",
            "internal_weight_1",
            "internal_weight_2",
            "external_weight_1",
            "external_weight_2",
        ):
            object.__setattr__(self, name, complex(getattr(self, name)))
        if abs(self.internal_weight_1) == 0.0:
            raise ValueError("the first NS Gram matrix is singular at h1=0")
        if abs(self.internal_weight_2) == 0.0:
            raise ValueError("the second NS Gram matrix is singular at h2=0")

    def raw_coefficients(self) -> Dict[LevelPair, complex]:
        """Return coefficients keyed by twice the levels on the two edges."""

        h1 = self.internal_weight_1
        h2 = self.internal_weight_2
        d1 = self.external_weight_1
        d2 = self.external_weight_2

        # The integer coefficients use
        # rho(L_-1 h_i,d,h_j)=h_i+d-h_j and B_1(h_i)=2h_i.
        f10 = (h1 + d1 - h2) * (h1 + d2 - h2) / (2.0 * h1)
        f01 = (h2 + d1 - h1) * (h2 + d2 - h1) / (2.0 * h2)

        # At both vertices
        # rho(G_-1/2 h_i,d,G_-1/2 h_j)=h_i+h_j-d.
        # One isolated odd state cannot couple to two even primaries.
        f_half_half = (
            (h1 + h2 - d1)
            * (h1 + h2 - d2)
            / (4.0 * h1 * h2)
        )
        return {
            (0, 0): 1.0 + 0.0j,
            (1, 0): 0.0j,
            (0, 1): 0.0j,
            (2, 0): f10,
            (0, 2): f01,
            (1, 1): f_half_half,
        }

    def evaluate(
        self,
        plumbing_1: NSPlumbingParameter,
        plumbing_2: NSPlumbingParameter,
    ) -> complex:
        """Evaluate the displayed leading necklace series."""

        return sum(
            coefficient
            * plumbing_1.level_factor(twice_level_1)
            * plumbing_2.level_factor(twice_level_2)
            for (
                twice_level_1,
                twice_level_2,
            ), coefficient in self.raw_coefficients().items()
        )

    def chiral_block(
        self,
        plumbing_1: NSPlumbingParameter,
        plumbing_2: NSPlumbingParameter,
    ) -> complex:
        """Restore both internal-weight and Casimir sewing prefactors."""

        return (
            plumbing_1.q
            ** (self.internal_weight_1 - self.central_charge / 24.0)
            * plumbing_2.q
            ** (self.internal_weight_2 - self.central_charge / 24.0)
            * self.evaluate(plumbing_1, plumbing_2)
        )


class NSTorusTwoPointHRecursionBlock:
    r"""Generic-``b`` NS two-point necklace block from internal-weight recursion.

    The block is a bivariate series in the two plumbing parameters.  Its
    internal weights are parameterized by ``h1`` and ``a=h2-h1`` as in the
    torus-necklace recursion of Cho--Collier--Yin.  At a pole on one edge the
    adjacent weight is evaluated along this fixed-difference line.

    An odd NS singular vector changes the fermion routing between the two
    pants vertices.  ``routing=0`` has equal parities on the two necklace
    edges and has the NS Verma character as its regular part.
    ``routing=1`` has opposite parities and has zero regular part.  Keeping
    this two-state routing is essential: it removes isolated half-level
    terms and reproduces the direct level-one coefficients.

    This class is the generic irrational-``b`` kernel.  At rational ``b`` or
    coincident internal weights use
    :class:`SelfDualNSTorusTwoPointHRecursionBlock`, which takes finite parts
    only after complete bivariate coefficients have been assembled.
    """

    def __init__(
        self,
        *,
        b: complex,
        internal_weight_1: complex,
        internal_weight_2: complex,
        external_weight_1: complex,
        external_weight_2: complex,
        pole_tolerance: float = 1.0e-12,
    ) -> None:
        self.b = complex(b)
        self.c = central_charge(self.b)
        self.internal_weight_1 = complex(internal_weight_1)
        self.internal_weight_2 = complex(internal_weight_2)
        self.external_weight_1 = complex(external_weight_1)
        self.external_weight_2 = complex(external_weight_2)
        self.pole_tolerance = float(pole_tolerance)
        if not math.isfinite(self.pole_tolerance) or self.pole_tolerance <= 0:
            raise ValueError("pole_tolerance must be finite and positive")

    @staticmethod
    def _validate_twice_level(value: int, name: str) -> int:
        if not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
        return value

    @lru_cache(maxsize=None)
    def _coefficient(
        self,
        twice_level_1: int,
        twice_level_2: int,
        base_weight: complex,
        weight_difference: complex,
        routing: int,
    ) -> complex:
        """Return one coefficient on a fixed-difference recursion line."""

        if twice_level_1 < 0 or twice_level_2 < 0:
            return 0.0j
        weight_1 = complex(base_weight)
        difference = complex(weight_difference)
        weight_2 = weight_1 + difference

        result = 0.0j
        if routing == 0 and twice_level_1 == twice_level_2:
            result = _ns_character_coefficient(twice_level_1)

        for edge, available_level in (
            (1, twice_level_1),
            (2, twice_level_2),
        ):
            for r in range(1, available_level + 1):
                for s in range(1, available_level // r + 1):
                    product = r * s
                    if product > available_level or (r + s) % 2:
                        continue

                    degenerate = _ns_degenerate_weight(self.b, r, s)
                    if edge == 1:
                        denominator = weight_1 - degenerate
                        adjacent_at_pole = degenerate + difference
                    else:
                        denominator = weight_2 - degenerate
                        adjacent_at_pole = degenerate - difference
                    scale = max(
                        1.0,
                        abs(weight_1),
                        abs(weight_2),
                        abs(degenerate),
                    )
                    if abs(denominator) <= self.pole_tolerance * scale:
                        raise ZeroDivisionError(
                            "two-edge h-recursion encountered the "
                            f"({r},{s}) NS pole on edge {edge}; use a "
                            "collision-aware finite-part wrapper"
                        )

                    # The routing selects the even or odd NS three-point
                    # form at both ends of the singular edge.  This is the
                    # multipoint counterpart of the even/odd block switch
                    # in the sphere NS h-recursion.
                    fusion_1 = _ns_ns_fusion_polynomial(
                        b=self.b,
                        r=r,
                        s=s,
                        lower_weight=self.external_weight_1,
                        upper_weight=adjacent_at_pole,
                        starred=bool(routing),
                    )
                    fusion_2 = _ns_ns_fusion_polynomial(
                        b=self.b,
                        r=r,
                        s=s,
                        lower_weight=self.external_weight_2,
                        upper_weight=adjacent_at_pole,
                        starred=bool(routing),
                    )
                    residue = (
                        _ns_a_factor(
                            self.b, r, s, self.pole_tolerance
                        )
                        * fusion_1
                        * fusion_2
                    )
                    next_routing = routing ^ (product % 2)

                    if edge == 1:
                        tail = self._coefficient(
                            twice_level_1 - product,
                            twice_level_2,
                            complex(degenerate + product / 2.0),
                            complex(difference - product / 2.0),
                            next_routing,
                        )
                    else:
                        tail = self._coefficient(
                            twice_level_1,
                            twice_level_2 - product,
                            complex(degenerate - difference),
                            complex(difference + product / 2.0),
                            next_routing,
                        )
                    result += residue * tail / denominator
        return result

    def raw_coefficient(
        self, twice_level_1: int, twice_level_2: int
    ) -> complex:
        """Return the plane-frame coefficient of the requested monomial."""

        twice_level_1 = self._validate_twice_level(
            twice_level_1, "twice_level_1"
        )
        twice_level_2 = self._validate_twice_level(
            twice_level_2, "twice_level_2"
        )
        return self._coefficient(
            twice_level_1,
            twice_level_2,
            self.internal_weight_1,
            self.internal_weight_2 - self.internal_weight_1,
            0,
        )

    def raw_coefficients(
        self, max_twice_level_1: int, max_twice_level_2: int
    ) -> Dict[LevelPair, complex]:
        max_twice_level_1 = self._validate_twice_level(
            max_twice_level_1, "max_twice_level_1"
        )
        max_twice_level_2 = self._validate_twice_level(
            max_twice_level_2, "max_twice_level_2"
        )
        return {
            (twice_level_1, twice_level_2): self.raw_coefficient(
                twice_level_1, twice_level_2
            )
            for twice_level_1 in range(max_twice_level_1 + 1)
            for twice_level_2 in range(max_twice_level_2 + 1)
        }

    def evaluate(
        self,
        plumbing_1: NSPlumbingParameter,
        plumbing_2: NSPlumbingParameter,
        max_twice_level_1: int,
        max_twice_level_2: int,
    ) -> complex:
        return sum(
            coefficient
            * plumbing_1.level_factor(twice_level_1)
            * plumbing_2.level_factor(twice_level_2)
            for (
                twice_level_1,
                twice_level_2,
            ), coefficient in self.raw_coefficients(
                max_twice_level_1, max_twice_level_2
            ).items()
        )

    def chiral_block(
        self,
        plumbing_1: NSPlumbingParameter,
        plumbing_2: NSPlumbingParameter,
        max_twice_level_1: int,
        max_twice_level_2: int,
    ) -> complex:
        return (
            plumbing_1.q
            ** (self.internal_weight_1 - self.c / 24.0)
            * plumbing_2.q
            ** (self.internal_weight_2 - self.c / 24.0)
            * self.evaluate(
                plumbing_1,
                plumbing_2,
                max_twice_level_1,
                max_twice_level_2,
            )
        )


class SelfDualNSTorusTwoPointHRecursionBlock:
    """Collision-aware Type-0B wrapper for the NS two-edge h-recursion."""

    def __init__(
        self,
        *,
        internal_momentum_1: complex,
        internal_momentum_2: complex,
        external_momentum_1: complex,
        external_momentum_2: complex,
        radius: float = 0.04,
        check_radius: float = 0.05,
        samples: int = 24,
        difference_radius: float = 0.03,
        difference_samples: int = 16,
        collision_tolerance: float = 1.0e-12,
    ) -> None:
        self.internal_momentum_1 = complex(internal_momentum_1)
        self.internal_momentum_2 = complex(internal_momentum_2)
        self.external_momentum_1 = complex(external_momentum_1)
        self.external_momentum_2 = complex(external_momentum_2)
        self.radius = float(radius)
        self.check_radius = float(check_radius)
        self.samples = int(samples)
        self.difference_radius = float(difference_radius)
        self.difference_samples = int(difference_samples)
        self.collision_tolerance = float(collision_tolerance)
        if (
            not math.isfinite(self.difference_radius)
            or self.difference_radius <= 0
        ):
            raise ValueError("difference_radius must be finite and positive")
        if self.difference_samples < 8:
            raise ValueError("difference_samples must be at least eight")
        self._cache: Dict[
            Tuple[int, int],
            Tuple[
                Dict[LevelPair, complex],
                Dict[LevelPair, FinitePartDiagnostics],
            ],
        ] = {}

    @property
    def c(self) -> complex:
        return central_charge(1.0)

    @property
    def internal_weight_1(self) -> complex:
        return ns_liouville_weight(self.internal_momentum_1, 1.0)

    @property
    def internal_weight_2(self) -> complex:
        return ns_liouville_weight(self.internal_momentum_2, 1.0)

    def _block_at(
        self, b: complex, *, weight_difference_shift: complex = 0.0j
    ) -> NSTorusTwoPointHRecursionBlock:
        weight_1 = ns_liouville_weight(self.internal_momentum_1, b)
        weight_2 = (
            ns_liouville_weight(self.internal_momentum_2, b)
            + weight_difference_shift
        )
        return NSTorusTwoPointHRecursionBlock(
            b=b,
            internal_weight_1=weight_1,
            internal_weight_2=weight_2,
            external_weight_1=ns_liouville_weight(
                self.external_momentum_1, b
            ),
            external_weight_2=ns_liouville_weight(
                self.external_momentum_2, b
            ),
        )

    def _table_at(
        self,
        b: complex,
        max_twice_level_1: int,
        max_twice_level_2: int,
    ) -> Dict[LevelPair, complex]:
        momentum_difference = (
            self.internal_momentum_2 * self.internal_momentum_2
            - self.internal_momentum_1 * self.internal_momentum_1
        ) / 2.0
        if abs(momentum_difference) > self.collision_tolerance:
            return self._block_at(b).raw_coefficients(
                max_twice_level_1, max_twice_level_2
            )

        totals = {
            (twice_level_1, twice_level_2): 0.0j
            for twice_level_1 in range(max_twice_level_1 + 1)
            for twice_level_2 in range(max_twice_level_2 + 1)
        }
        for index in range(self.difference_samples):
            angle = (
                2.0
                * math.pi
                * (index + 0.5)
                / self.difference_samples
            )
            displacement = self.difference_radius * cmath.exp(1j * angle)
            values = self._block_at(
                b, weight_difference_shift=displacement
            ).raw_coefficients(
                max_twice_level_1, max_twice_level_2
            )
            for key, value in values.items():
                totals[key] += value
        return {
            key: value / self.difference_samples
            for key, value in totals.items()
        }

    def _data(
        self, max_twice_level_1: int, max_twice_level_2: int
    ) -> Tuple[
        Dict[LevelPair, complex],
        Dict[LevelPair, FinitePartDiagnostics],
    ]:
        NSTorusTwoPointHRecursionBlock._validate_twice_level(
            max_twice_level_1, "max_twice_level_1"
        )
        NSTorusTwoPointHRecursionBlock._validate_twice_level(
            max_twice_level_2, "max_twice_level_2"
        )
        cache_key = (max_twice_level_1, max_twice_level_2)
        if cache_key not in self._cache:
            keys = tuple(
                (twice_level_1, twice_level_2)
                for twice_level_1 in range(max_twice_level_1 + 1)
                for twice_level_2 in range(max_twice_level_2 + 1)
            )
            self._cache[cache_key] = _dictionary_finite_part(
                lambda b: self._table_at(
                    b, max_twice_level_1, max_twice_level_2
                ),
                keys=keys,
                radius=self.radius,
                check_radius=self.check_radius,
                samples=self.samples,
            )
        return self._cache[cache_key]

    def raw_coefficients(
        self, max_twice_level_1: int, max_twice_level_2: int
    ) -> Dict[LevelPair, complex]:
        return dict(
            self._data(max_twice_level_1, max_twice_level_2)[0]
        )

    def coefficient_diagnostics(
        self, max_twice_level_1: int, max_twice_level_2: int
    ) -> Dict[LevelPair, FinitePartDiagnostics]:
        return dict(
            self._data(max_twice_level_1, max_twice_level_2)[1]
        )

    def evaluate(
        self,
        plumbing_1: NSPlumbingParameter,
        plumbing_2: NSPlumbingParameter,
        max_twice_level_1: int,
        max_twice_level_2: int,
    ) -> complex:
        return sum(
            coefficient
            * plumbing_1.level_factor(twice_level_1)
            * plumbing_2.level_factor(twice_level_2)
            for (
                twice_level_1,
                twice_level_2,
            ), coefficient in self.raw_coefficients(
                max_twice_level_1, max_twice_level_2
            ).items()
        )

    def chiral_block(
        self,
        plumbing_1: NSPlumbingParameter,
        plumbing_2: NSPlumbingParameter,
        max_twice_level_1: int,
        max_twice_level_2: int,
    ) -> complex:
        return (
            plumbing_1.q
            ** (self.internal_weight_1 - self.c / 24.0)
            * plumbing_2.q
            ** (self.internal_weight_2 - self.c / 24.0)
            * self.evaluate(
                plumbing_1,
                plumbing_2,
                max_twice_level_1,
                max_twice_level_2,
            )
        )


@dataclass(frozen=True)
class MixedNSRamondTorusTwoPointGroundBlock:
    """Ground-fiber normalization for an NS--R necklace.

    A mixed pair of internal edges forces both external punctures to be
    Ramond.  The NS ground is one-dimensional, while the long-R edge retains
    the unnormalized basis ``(w^+, G_0 w^+)``.  Each RRNS vertex is selected
    by its HJS sign, even/odd three-form, and external Ramond ground
    component.  The internal R fiber is contracted only after its cycle
    insertion has been applied.

    This object fixes the zeroth coefficient and the spin-selection rules
    required by the future all-level mixed h-recursion; it is not a
    descendant-level evaluator.
    """

    central_charge: complex
    internal_ns_weight: complex
    internal_r_weight: complex
    vertex_sign_1: int = 1
    vertex_sign_2: int = 1
    form_parity_1: Literal["even", "odd"] = "even"
    form_parity_2: Literal["even", "odd"] = "even"
    external_ground_1: Literal["+", "-"] = "+"
    external_ground_2: Literal["+", "-"] = "+"

    def __post_init__(self) -> None:
        for name in (
            "central_charge",
            "internal_ns_weight",
            "internal_r_weight",
        ):
            object.__setattr__(self, name, complex(getattr(self, name)))
        for name in ("vertex_sign_1", "vertex_sign_2"):
            value = int(getattr(self, name))
            if value not in (-1, 1):
                raise ValueError(f"{name} must be +1 or -1")
            object.__setattr__(self, name, value)
        for name in ("form_parity_1", "form_parity_2"):
            if getattr(self, name) not in ("even", "odd"):
                raise ValueError(f"{name} must be 'even' or 'odd'")
        for name in ("external_ground_1", "external_ground_2"):
            if getattr(self, name) not in ("+", "-"):
                raise ValueError(f"{name} must be '+' or '-'")
        if abs(self.kappa_squared) == 0.0:
            raise ValueError(
                "the mixed ground block requires a generic long-R edge"
            )

    @property
    def kappa_squared(self) -> complex:
        return self.internal_r_weight - self.central_charge / 24.0

    @property
    def ground_fiber(self) -> RamondGroundFiber:
        return RamondGroundFiber(
            c=self.central_charge,
            weight=self.internal_r_weight,
        )

    @staticmethod
    def _normalized_vertex(
        form_parity: Literal["even", "odd"], sign: int
    ) -> Matrix2:
        if form_parity == "even":
            return (
                (1.0 + 0.0j, 0.0j),
                (0.0j, complex(sign)),
            )
        return (
            (0.0j, 1.0 + 0.0j),
            (1j * sign, 0.0j),
        )

    def _internal_vector(
        self,
        *,
        vertex: int,
    ) -> Tuple[complex, complex]:
        if vertex == 1:
            matrix = self._normalized_vertex(
                self.form_parity_1, self.vertex_sign_1
            )
            external_index = 0 if self.external_ground_1 == "+" else 1
            normalized = (
                matrix[0][external_index],
                matrix[1][external_index],
            )
        elif vertex == 2:
            matrix = self._normalized_vertex(
                self.form_parity_2, self.vertex_sign_2
            )
            external_index = 0 if self.external_ground_2 == "+" else 1
            normalized = (
                matrix[external_index][0],
                matrix[external_index][1],
            )
        else:
            raise ValueError("vertex must be one or two")
        kappa = cmath.sqrt(self.kappa_squared)
        return (normalized[0], kappa * normalized[1])

    def ground_coefficient(
        self,
        ns_plumbing: NSPlumbingParameter,
        r_plumbing: RamondPlumbingParameter,
    ) -> complex:
        """Contract the internal long-R ground fiber in the chosen spin sector."""

        if not isinstance(ns_plumbing, NSPlumbingParameter):
            raise TypeError("ns_plumbing must be an NS plumbing parameter")
        if not isinstance(r_plumbing, RamondPlumbingParameter):
            raise TypeError("r_plumbing must be a Ramond plumbing parameter")
        left = self._internal_vector(vertex=1)
        right = self._internal_vector(vertex=2)
        inverse_gram: Matrix2 = (
            (1.0 + 0.0j, 0.0j),
            (0.0j, 1.0 / self.kappa_squared),
        )
        insertion = self.ground_fiber.insertion_matrix(
            r_plumbing.cycle_insertion
        )
        sewing = _matmul2(inverse_gram, insertion)
        return sum(
            left[row] * sewing[row][column] * right[column]
            for row in range(2)
            for column in range(2)
        )

    def chiral_block(
        self,
        ns_plumbing: NSPlumbingParameter,
        r_plumbing: RamondPlumbingParameter,
    ) -> complex:
        return (
            ns_plumbing.q
            ** (self.internal_ns_weight - self.central_charge / 24.0)
            * r_plumbing.q
            ** (self.internal_r_weight - self.central_charge / 24.0)
            * self.ground_coefficient(ns_plumbing, r_plumbing)
        )


@dataclass(frozen=True)
class RamondTorusTwoPointGroundBlock:
    """Level-zero R--R necklace block with both ground fibers left explicit."""

    central_charge: complex
    internal_weight_1: complex
    internal_weight_2: complex
    external_weight_1: complex
    external_weight_2: complex
    vertex_sign_1: int = 1
    vertex_sign_2: int = 1

    def __post_init__(self) -> None:
        for name in (
            "central_charge",
            "internal_weight_1",
            "internal_weight_2",
            "external_weight_1",
            "external_weight_2",
        ):
            object.__setattr__(self, name, complex(getattr(self, name)))
        for name in ("vertex_sign_1", "vertex_sign_2"):
            value = int(getattr(self, name))
            if value not in (-1, 1):
                raise ValueError(f"{name} must be +1 or -1")
            object.__setattr__(self, name, value)
        if abs(self.kappa_squared_1) == 0.0:
            raise ValueError(
                "the first generic long-R ground fiber is singular at h=c/24"
            )
        if abs(self.kappa_squared_2) == 0.0:
            raise ValueError(
                "the second generic long-R ground fiber is singular at h=c/24"
            )

    @property
    def kappa_squared_1(self) -> complex:
        return self.internal_weight_1 - self.central_charge / 24.0

    @property
    def kappa_squared_2(self) -> complex:
        return self.internal_weight_2 - self.central_charge / 24.0

    @property
    def ground_fiber_1(self) -> RamondGroundFiber:
        return RamondGroundFiber(
            c=self.central_charge,
            weight=self.internal_weight_1,
        )

    @property
    def ground_fiber_2(self) -> RamondGroundFiber:
        return RamondGroundFiber(
            c=self.central_charge,
            weight=self.internal_weight_2,
        )

    def rrns_ground_vertex(self, sign: int) -> Matrix2:
        """Return the normalized even RRNS matrix from edge 2 to edge 1.

        In the HJS ``w^+,w^-`` basis the normalized even forms have ground
        matrices ``diag(1,+/-1)``.  Transferring to the unnormalized
        ``(w^+,G_0 w^+)`` fibers gives the second entry
        ``+/- kappa_1 kappa_2``.  The principal square roots agree with the
        positive Type-0B momentum branch; changing a Ramond momentum branch
        exchanges the HJS sign label.
        """

        sign = int(sign)
        if sign not in (-1, 1):
            raise ValueError("sign must be +1 or -1")
        kappa_1 = cmath.sqrt(self.kappa_squared_1)
        kappa_2 = cmath.sqrt(self.kappa_squared_2)
        return (
            (1.0 + 0.0j, 0.0j),
            (0.0j, sign * kappa_1 * kappa_2),
        )

    @staticmethod
    def _inverse_ground_gram(fiber: RamondGroundFiber) -> Matrix2:
        if abs(fiber.kappa_squared) == 0.0:
            raise ValueError(
                "the long-R contraction is singular at h=c/24; "
                "project to the short quotient first"
            )
        return (
            (1.0 + 0.0j, 0.0j),
            (0.0j, 1.0 / fiber.kappa_squared),
        )

    def ground_coefficient(
        self,
        plumbing_1: RamondPlumbingParameter,
        plumbing_2: RamondPlumbingParameter,
    ) -> complex:
        r"""Contract both ground fibers before taking the scalar trace.

        The ordering is

        ``Tr(B1^-1 S1 V1 B2^-1 S2 V2)``,

        where ``Sa`` is the cycle insertion attached to edge ``a``.
        """

        fiber_1 = self.ground_fiber_1
        fiber_2 = self.ground_fiber_2
        inverse_1 = self._inverse_ground_gram(fiber_1)
        inverse_2 = self._inverse_ground_gram(fiber_2)
        insertion_1 = fiber_1.insertion_matrix(
            plumbing_1.cycle_insertion
        )
        insertion_2 = fiber_2.insertion_matrix(
            plumbing_2.cycle_insertion
        )
        vertex_1 = self.rrns_ground_vertex(self.vertex_sign_1)
        # The ground matrix is symmetric under exchange of the two fibers.
        vertex_2 = self.rrns_ground_vertex(self.vertex_sign_2)
        product = _matmul2(
            inverse_1,
            _matmul2(
                insertion_1,
                _matmul2(
                    vertex_1,
                    _matmul2(
                        inverse_2,
                        _matmul2(insertion_2, vertex_2),
                    ),
                ),
            ),
        )
        return _trace2(product)

    def chiral_block(
        self,
        plumbing_1: RamondPlumbingParameter,
        plumbing_2: RamondPlumbingParameter,
    ) -> complex:
        """Restore both Ramond Casimir sewing prefactors."""

        return (
            plumbing_1.q
            ** (self.internal_weight_1 - self.central_charge / 24.0)
            * plumbing_2.q
            ** (self.internal_weight_2 - self.central_charge / 24.0)
            * self.ground_coefficient(plumbing_1, plumbing_2)
        )


@dataclass(frozen=True)
class BruteForceRamondTorusTwoPointBlock:
    """Direct RRNS necklace sewing through a small pair of integer levels.

    This is the coefficient oracle for the two-edge Zamolodchikov recursion.
    It keeps both long-R parity blocks until the cycle trace is taken and does
    not use Kac poles or fusion polynomials.
    """

    central_charge: complex
    internal_weight_1: complex
    internal_weight_2: complex
    external_weight_1: complex
    external_weight_2: complex
    vertex_sign_1: int = 1
    vertex_sign_2: int = 1
    cycle_insertion_1: str = "identity"
    cycle_insertion_2: str = "identity"

    def __post_init__(self) -> None:
        for name in (
            "central_charge",
            "internal_weight_1",
            "internal_weight_2",
            "external_weight_1",
            "external_weight_2",
        ):
            object.__setattr__(self, name, complex(getattr(self, name)))
        for name in ("vertex_sign_1", "vertex_sign_2"):
            value = int(getattr(self, name))
            if value not in (-1, 1):
                raise ValueError(f"{name} must be +1 or -1")
            object.__setattr__(self, name, value)
        for name in ("cycle_insertion_1", "cycle_insertion_2"):
            value = str(getattr(self, name))
            if value not in ("identity", "parity"):
                raise ValueError(
                    "the direct positive-level oracle currently supports "
                    "only identity and parity cycle insertions"
                )
            object.__setattr__(self, name, value)
        if (
            abs(self.internal_weight_1 - self.central_charge / 24.0) == 0.0
            or abs(self.internal_weight_2 - self.central_charge / 24.0) == 0.0
        ):
            raise ValueError("the direct oracle requires generic long-R modules")

    @property
    def module_1(self) -> RamondVermaModule:
        return RamondVermaModule(
            c=self.central_charge,
            weight=self.internal_weight_1,
        )

    @property
    def module_2(self) -> RamondVermaModule:
        return RamondVermaModule(
            c=self.central_charge,
            weight=self.internal_weight_2,
        )

    @staticmethod
    def _inverse_times(module: RamondVermaModule, gram: Matrix, value: Matrix) -> Matrix:
        if not value:
            return ()
        columns = len(value[0])
        solved_columns = tuple(
            module._solve(
                gram,
                tuple(value[row][column] for row in range(len(value))),
            )
            for column in range(columns)
        )
        return tuple(
            tuple(solved_columns[column][row] for column in range(columns))
            for row in range(len(value))
        )

    @staticmethod
    def _cycle_eigenvalue(insertion: str, parity: int) -> complex:
        return -1.0 + 0.0j if insertion == "parity" and parity else 1.0 + 0.0j

    def raw_coefficient(self, level_1: int, level_2: int) -> complex:
        """Return the plane-frame coefficient of ``q1^level_1 q2^level_2``."""

        if (
            not isinstance(level_1, int)
            or not isinstance(level_2, int)
            or level_1 < 0
            or level_2 < 0
        ):
            raise ValueError("Ramond levels must be nonnegative integers")
        module_1 = self.module_1
        module_2 = self.module_2
        vertex_1 = RamondThreePointWardMatrix(
            left_module=module_1,
            right_module=module_2,
            external_ns_weight=self.external_weight_1,
            sign=self.vertex_sign_1,
        )
        vertex_2 = RamondThreePointWardMatrix(
            left_module=module_2,
            right_module=module_1,
            external_ns_weight=self.external_weight_2,
            sign=self.vertex_sign_2,
        )

        result = 0.0j
        for parity in (0, 1):
            gram_1 = module_1.gram_matrix(level_1, parity)
            gram_2 = module_2.gram_matrix(level_2, parity)
            matrix_1 = vertex_1.matrix(level_1, level_2, parity)
            matrix_2 = vertex_2.matrix(level_2, level_1, parity)
            inverse_1_matrix_1 = self._inverse_times(
                module_1, gram_1, matrix_1
            )
            inverse_2_matrix_2 = self._inverse_times(
                module_2, gram_2, matrix_2
            )
            trace = sum(
                inverse_1_matrix_1[row][column]
                * inverse_2_matrix_2[column][row]
                for row in range(len(inverse_1_matrix_1))
                for column in range(len(inverse_2_matrix_2))
            )
            result += (
                self._cycle_eigenvalue(self.cycle_insertion_1, parity)
                * self._cycle_eigenvalue(self.cycle_insertion_2, parity)
                * trace
            )
        return result

    def raw_coefficients(
        self, max_level_1: int, max_level_2: int
    ) -> Dict[Tuple[int, int], complex]:
        return {
            (level_1, level_2): self.raw_coefficient(level_1, level_2)
            for level_1 in range(max_level_1 + 1)
            for level_2 in range(max_level_2 + 1)
        }

    def reduced_coefficients(
        self, max_level_1: int, max_level_2: int
    ) -> Dict[Tuple[int, int], complex]:
        """Divide by the ground trace and remove the R cycle character.

        The non-degenerate R character depends on the product ``q1*q2``.
        Its ground multiplicity is already contained in the explicit cycle
        trace, so the remaining character coefficients are
        ``ramond_positive_character_coefficients``.
        """

        raw = self.raw_coefficients(max_level_1, max_level_2)
        ground = raw[(0, 0)]
        if abs(ground) == 0.0:
            raise ValueError(
                "the chosen HJS signs and cycle projection have zero ground "
                "trace and cannot define a normalized reduced block"
            )
        character = ramond_positive_character_coefficients(
            min(max_level_1, max_level_2)
        )
        reduced: Dict[Tuple[int, int], complex] = {}
        for level_1 in range(max_level_1 + 1):
            for level_2 in range(max_level_2 + 1):
                value = raw[(level_1, level_2)] / ground
                for offset in range(1, min(level_1, level_2) + 1):
                    value -= (
                        character[offset]
                        * reduced[(level_1 - offset, level_2 - offset)]
                    )
                reduced[(level_1, level_2)] = value
        return reduced


class _IncompleteRamondTorusTwoPointHRecursionBlock:
    r"""Scratchpad for the matrix CCY recursion of an R--R necklace.

    Write ``h1=H`` and ``h2=H+a`` and continue in ``H`` at fixed ``a``.
    The normalized regular part is the R character

    ``prod_n (1 + eta Q^n)/(1 - Q^n)``, ``Q=q1*q2``,

    where ``eta`` is fixed by the temporal fermion holonomy.  The pole
    kernel below is correct, but a scalar character is not the full regular
    seed: direct large-weight checks show mixing between the two HJS ground
    branches.  The public evaluator therefore does not use this class.
    """

    def __init__(
        self,
        *,
        b: complex,
        internal_weight_1: complex,
        internal_weight_2: complex,
        external_weight_1: complex,
        external_weight_2: complex,
        vertex_sign_1: int = 1,
        vertex_sign_2: int = 1,
        cycle_insertion_1: str = "identity",
        cycle_insertion_2: str = "identity",
        pole_tolerance: float = 1.0e-12,
    ) -> None:
        self.b = complex(b)
        self.c = central_charge(self.b)
        self.internal_weight_1 = complex(internal_weight_1)
        self.internal_weight_2 = complex(internal_weight_2)
        self.external_weight_1 = complex(external_weight_1)
        self.external_weight_2 = complex(external_weight_2)
        self.vertex_sign_1 = self._validate_sign(
            vertex_sign_1, "vertex_sign_1"
        )
        self.vertex_sign_2 = self._validate_sign(
            vertex_sign_2, "vertex_sign_2"
        )
        self.cycle_insertion_1 = self._validate_cycle(
            cycle_insertion_1, "cycle_insertion_1"
        )
        self.cycle_insertion_2 = self._validate_cycle(
            cycle_insertion_2, "cycle_insertion_2"
        )
        self.pole_tolerance = float(pole_tolerance)
        if not math.isfinite(self.pole_tolerance) or self.pole_tolerance <= 0:
            raise ValueError("pole_tolerance must be finite and positive")
        odd_cycle = (
            (self.cycle_insertion_1 == "parity")
            ^ (self.cycle_insertion_2 == "parity")
        )
        self.parity_sign = -1 if odd_cycle else 1
        if self.vertex_sign_1 * self.vertex_sign_2 != self.parity_sign:
            raise ValueError(
                "the HJS sign product must equal the R-character holonomy"
            )

    @staticmethod
    def _validate_sign(value: int, name: str) -> int:
        value = int(value)
        if value not in (-1, 1):
            raise ValueError(f"{name} must be +1 or -1")
        return value

    @staticmethod
    def _validate_cycle(value: str, name: str) -> str:
        value = str(value)
        if value not in ("identity", "parity"):
            raise ValueError(f"{name} must be 'identity' or 'parity'")
        return value

    @staticmethod
    def _validate_level(value: int, name: str) -> int:
        if not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
        return value

    def _beta(self, weight: complex) -> complex:
        return cmath.sqrt(self.c / 24.0 - weight)

    def _shifted_signs(
        self,
        *,
        r: int,
        s: int,
        shifted_weight: complex,
        sign_1: int,
        sign_2: int,
    ) -> Tuple[int, int]:
        beta_prime = _r_beta_prime(self.b, r, s)
        principal = self._beta(shifted_weight)
        if abs(principal - beta_prime) <= abs(principal + beta_prime):
            return (sign_1, sign_2)
        return (-sign_1, -sign_2)

    def _fusion_product(
        self,
        *,
        r: int,
        s: int,
        adjacent_beta: complex,
        sign_1: int,
        sign_2: int,
    ) -> complex:
        return _r_ns_fusion_polynomial(
            b=self.b,
            r=r,
            s=s,
            ramond_beta_value=adjacent_beta,
            ns_weight=self.external_weight_1,
            sign=sign_1,
        ) * _r_ns_fusion_polynomial(
            b=self.b,
            r=r,
            s=s,
            ramond_beta_value=adjacent_beta,
            ns_weight=self.external_weight_2,
            sign=sign_2,
        )

    @lru_cache(maxsize=None)
    def _coefficient(
        self,
        level_1: int,
        level_2: int,
        base_weight: complex,
        weight_difference: complex,
        sign_1: int,
        sign_2: int,
    ) -> complex:
        if level_1 < 0 or level_2 < 0:
            return 0.0j
        result = 0.0j
        if level_1 == level_2:
            result = _r_character_coefficient(
                level_1, self.parity_sign
            )

        for edge, available_level in (
            (1, level_1),
            (2, level_2),
        ):
            for r in range(1, 2 * available_level + 1):
                for s in range(
                    1, (2 * available_level) // r + 1
                ):
                    product = r * s
                    if (
                        (r + s) % 2 != 1
                        or product % 2
                        or product // 2 > available_level
                    ):
                        continue
                    shift = product // 2
                    beta_rs = _r_beta_rs(self.b, r, s)
                    degenerate = self.c / 24.0 - beta_rs**2
                    if edge == 1:
                        denominator = base_weight - degenerate
                        adjacent_weight = (
                            degenerate + weight_difference
                        )
                    else:
                        denominator = (
                            base_weight
                            + weight_difference
                            - degenerate
                        )
                        adjacent_weight = (
                            degenerate - weight_difference
                        )
                    scale = max(
                        1.0,
                        abs(base_weight),
                        abs(base_weight + weight_difference),
                        abs(degenerate),
                    )
                    if abs(denominator) <= self.pole_tolerance * scale:
                        raise ZeroDivisionError(
                            "R--R simultaneous h-recursion encountered "
                            f"the ({r},{s}) pole on edge {edge}"
                        )
                    residue = (
                        -2.0
                        * beta_rs
                        * _r_a_beta(
                            self.b, r, s, self.pole_tolerance
                        )
                        * self._fusion_product(
                            r=r,
                            s=s,
                            adjacent_beta=self._beta(adjacent_weight),
                            sign_1=sign_1,
                            sign_2=sign_2,
                        )
                    )
                    shifted_weight = degenerate + shift
                    next_sign_1, next_sign_2 = self._shifted_signs(
                        r=r,
                        s=s,
                        shifted_weight=shifted_weight,
                        sign_1=sign_1,
                        sign_2=sign_2,
                    )
                    if edge == 1:
                        tail = self._coefficient(
                            level_1 - shift,
                            level_2,
                            shifted_weight,
                            weight_difference - shift,
                            next_sign_1,
                            next_sign_2,
                        )
                    else:
                        tail = self._coefficient(
                            level_1,
                            level_2 - shift,
                            degenerate - weight_difference,
                            weight_difference + shift,
                            next_sign_1,
                            next_sign_2,
                        )
                    result += residue * tail / denominator
        return result

    def normalized_coefficient(
        self, level_1: int, level_2: int
    ) -> complex:
        raise NotImplementedError(
            "the R--R CCY recursion requires the matrix-valued HJS "
            "large-weight seed"
        )
        level_1 = self._validate_level(level_1, "level_1")
        level_2 = self._validate_level(level_2, "level_2")
        return self._coefficient(
            level_1,
            level_2,
            self.internal_weight_1,
            self.internal_weight_2 - self.internal_weight_1,
            self.vertex_sign_1,
            self.vertex_sign_2,
        )

    def normalized_coefficients(
        self, max_level_1: int, max_level_2: int
    ) -> Dict[LevelPair, complex]:
        max_level_1 = self._validate_level(max_level_1, "max_level_1")
        max_level_2 = self._validate_level(max_level_2, "max_level_2")
        return {
            (level_1, level_2): self.normalized_coefficient(
                level_1, level_2
            )
            for level_1 in range(max_level_1 + 1)
            for level_2 in range(max_level_2 + 1)
        }

    def raw_coefficients(
        self, max_level_1: int, max_level_2: int
    ) -> Dict[LevelPair, complex]:
        return {
            key: 2.0 * value
            for key, value in self.normalized_coefficients(
                max_level_1, max_level_2
            ).items()
        }

    def normalized_evaluate(
        self,
        plumbing_1: RamondPlumbingParameter,
        plumbing_2: RamondPlumbingParameter,
        max_level_1: int,
        max_level_2: int,
    ) -> complex:
        if plumbing_1.cycle_insertion != self.cycle_insertion_1:
            raise ValueError("plumbing_1 cycle insertion does not match block")
        if plumbing_2.cycle_insertion != self.cycle_insertion_2:
            raise ValueError("plumbing_2 cycle insertion does not match block")
        return sum(
            coefficient
            * plumbing_1.level_factor(level_1)
            * plumbing_2.level_factor(level_2)
            for (level_1, level_2), coefficient in (
                self.normalized_coefficients(
                    max_level_1, max_level_2
                ).items()
            )
        )

    def normalized_chiral_block(
        self,
        plumbing_1: RamondPlumbingParameter,
        plumbing_2: RamondPlumbingParameter,
        max_level_1: int,
        max_level_2: int,
    ) -> complex:
        return (
            plumbing_1.q
            ** (self.internal_weight_1 - self.c / 24.0)
            * plumbing_2.q
            ** (self.internal_weight_2 - self.c / 24.0)
            * self.normalized_evaluate(
                plumbing_1,
                plumbing_2,
                max_level_1,
                max_level_2,
            )
        )


class RamondTorusTwoPointBetaRecursionBlock:
    r"""Generic-``b`` two-edge recursion for a long-R necklace.

    The natural R-channel variables are the two zero-mode parameters

    ``h_i = c/24 - beta_i**2``.

    Every R Kac label has ``r+s`` odd and ``rs`` even.  A pole on either
    edge therefore shifts an integer level by ``rs/2``.  The pole at
    ``+beta_rs`` preserves both adjacent HJS signs, while the pole at
    ``-beta_rs`` flips both signs.  This is the same matrix-valued residue
    pattern as the tested NSNSRR R-exchange sphere block.

    Closing the two R ground fibers before taking the large-momentum limit
    leaves a nontrivial analytic part.  Consequently the scalar seed is not
    the constant one.  By default this class uses
    :class:`DirectRamondTorusTwoPointRegularSeed`, which obtains the
    pole-free part from independent Ward/Gram sewing.  A future contracted
    R light block can be supplied through ``regular_seed`` without changing
    the pole recursion.

    Only identity and fermion-parity cycle insertions are supported at
    positive level.  The HJS signs must obey the corresponding nonzero
    ground-trace selection rule.
    """

    def __init__(
        self,
        *,
        b: complex,
        internal_beta_1: complex,
        internal_beta_2: complex,
        external_weight_1: complex,
        external_weight_2: complex,
        vertex_sign_1: int = 1,
        vertex_sign_2: int = 1,
        cycle_insertion_1: str = "identity",
        cycle_insertion_2: str = "identity",
        regular_seed: RamondRegularSeed | None = None,
        use_direct_regular_seed: bool = True,
        pole_tolerance: float = 1.0e-12,
    ) -> None:
        self.b = complex(b)
        self.c = central_charge(self.b)
        self.internal_beta_1 = complex(internal_beta_1)
        self.internal_beta_2 = complex(internal_beta_2)
        self.external_weight_1 = complex(external_weight_1)
        self.external_weight_2 = complex(external_weight_2)
        self.vertex_sign_1 = self._validate_sign(
            vertex_sign_1, "vertex_sign_1"
        )
        self.vertex_sign_2 = self._validate_sign(
            vertex_sign_2, "vertex_sign_2"
        )
        self.cycle_insertion_1 = self._validate_cycle(
            cycle_insertion_1, "cycle_insertion_1"
        )
        self.cycle_insertion_2 = self._validate_cycle(
            cycle_insertion_2, "cycle_insertion_2"
        )
        self.pole_tolerance = float(pole_tolerance)
        if not math.isfinite(self.pole_tolerance) or self.pole_tolerance <= 0:
            raise ValueError("pole_tolerance must be finite and positive")
        if (
            abs(self.internal_beta_1) <= self.pole_tolerance
            or abs(self.internal_beta_2) <= self.pole_tolerance
        ):
            raise ValueError(
                "the two-edge R recursion requires two generic long-R "
                "modules with nonzero beta"
            )
        cycle_parity = (
            (self.cycle_insertion_1 == "parity")
            ^ (self.cycle_insertion_2 == "parity")
        )
        wanted_product = -1 if cycle_parity else 1
        if self.vertex_sign_1 * self.vertex_sign_2 != wanted_product:
            raise ValueError(
                "the chosen HJS signs have zero ground trace for these "
                "cycle insertions"
            )

        self.regular_seed = regular_seed
        if self.regular_seed is None and use_direct_regular_seed:
            self.regular_seed = DirectRamondTorusTwoPointRegularSeed(
                b=self.b,
                external_weight_1=self.external_weight_1,
                external_weight_2=self.external_weight_2,
                cycle_insertion_1=self.cycle_insertion_1,
                cycle_insertion_2=self.cycle_insertion_2,
                pole_tolerance=self.pole_tolerance,
            ).coefficient

    @staticmethod
    def _validate_sign(value: int, name: str) -> int:
        value = int(value)
        if value not in (-1, 1):
            raise ValueError(f"{name} must be +1 or -1")
        return value

    @staticmethod
    def _validate_cycle(value: str, name: str) -> str:
        value = str(value)
        if value not in ("identity", "parity"):
            raise ValueError(
                f"{name} must be 'identity' or 'parity' in the "
                "positive-level recursion"
            )
        return value

    @staticmethod
    def _validate_level(value: int, name: str) -> int:
        if not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
        return value

    @property
    def internal_weight_1(self) -> complex:
        return self.c / 24.0 - self.internal_beta_1**2

    @property
    def internal_weight_2(self) -> complex:
        return self.c / 24.0 - self.internal_beta_2**2

    def _fusion_product(
        self,
        *,
        r: int,
        s: int,
        adjacent_beta: complex,
        sign_1: int,
        sign_2: int,
    ) -> complex:
        return _r_ns_fusion_polynomial(
            b=self.b,
            r=r,
            s=s,
            ramond_beta_value=adjacent_beta,
            ns_weight=self.external_weight_1,
            sign=sign_1,
        ) * _r_ns_fusion_polynomial(
            b=self.b,
            r=r,
            s=s,
            ramond_beta_value=adjacent_beta,
            ns_weight=self.external_weight_2,
            sign=sign_2,
        )

    def _pole_sum(
        self,
        level_1: int,
        level_2: int,
        beta_1: complex,
        beta_2: complex,
        sign_1: int,
        sign_2: int,
        tail_evaluator: RamondRegularSeed,
    ) -> complex:
        result = 0.0j
        for edge, available_level in ((1, level_1), (2, level_2)):
            for r in range(1, 2 * available_level + 1):
                for s in range(1, (2 * available_level) // r + 1):
                    product = r * s
                    if (
                        (r + s) % 2 != 1
                        or product % 2
                        or product // 2 > available_level
                    ):
                        continue
                    shift = product // 2
                    beta_rs = _r_beta_rs(self.b, r, s)
                    beta_prime = _r_beta_prime(self.b, r, s)
                    a_factor = _r_a_beta(
                        self.b, r, s, self.pole_tolerance
                    )
                    active_beta = beta_1 if edge == 1 else beta_2
                    adjacent_beta = beta_2 if edge == 1 else beta_1
                    denominator_plus = active_beta - beta_rs
                    denominator_minus = active_beta + beta_rs
                    scale = max(1.0, abs(active_beta), abs(beta_rs))
                    if (
                        abs(denominator_plus)
                        <= self.pole_tolerance * scale
                        or abs(denominator_minus)
                        <= self.pole_tolerance * scale
                    ):
                        raise ZeroDivisionError(
                            "two-edge beta recursion encountered the "
                            f"({r},{s}) R pole on edge {edge}"
                        )

                    positive_residue = a_factor * self._fusion_product(
                        r=r,
                        s=s,
                        adjacent_beta=adjacent_beta,
                        sign_1=sign_1,
                        sign_2=sign_2,
                    )
                    negative_residue = -a_factor * self._fusion_product(
                        r=r,
                        s=s,
                        adjacent_beta=adjacent_beta,
                        sign_1=-sign_1,
                        sign_2=-sign_2,
                    )
                    if edge == 1:
                        positive_tail = tail_evaluator(
                            level_1 - shift,
                            level_2,
                            beta_prime,
                            beta_2,
                            sign_1,
                            sign_2,
                        )
                        negative_tail = tail_evaluator(
                            level_1 - shift,
                            level_2,
                            beta_prime,
                            beta_2,
                            -sign_1,
                            -sign_2,
                        )
                    else:
                        positive_tail = tail_evaluator(
                            level_1,
                            level_2 - shift,
                            beta_1,
                            beta_prime,
                            sign_1,
                            sign_2,
                        )
                        negative_tail = tail_evaluator(
                            level_1,
                            level_2 - shift,
                            beta_1,
                            beta_prime,
                            -sign_1,
                            -sign_2,
                        )
                    result += (
                        positive_residue
                        * positive_tail
                        / denominator_plus
                        + negative_residue
                        * negative_tail
                        / denominator_minus
                    )
        return result

    @lru_cache(maxsize=None)
    def _coefficient(
        self,
        level_1: int,
        level_2: int,
        beta_1: complex,
        beta_2: complex,
        sign_1: int,
        sign_2: int,
    ) -> complex:
        if level_1 < 0 or level_2 < 0:
            return 0.0j
        if self.regular_seed is None:
            if level_1 == 0 and level_2 == 0:
                regular = 1.0 + 0.0j
            else:
                raise NotImplementedError(
                    "the R necklace pole recursion requires a regular seed"
                )
        else:
            regular = complex(
                self.regular_seed(
                    level_1,
                    level_2,
                    beta_1,
                    beta_2,
                    sign_1,
                    sign_2,
                )
            )
        return regular + self._pole_sum(
            level_1,
            level_2,
            beta_1,
            beta_2,
            sign_1,
            sign_2,
            self._coefficient,
        )

    def normalized_coefficient(
        self, level_1: int, level_2: int
    ) -> complex:
        """Return a coefficient divided by the nonzero ground trace two."""

        level_1 = self._validate_level(level_1, "level_1")
        level_2 = self._validate_level(level_2, "level_2")
        return self._coefficient(
            level_1,
            level_2,
            self.internal_beta_1,
            self.internal_beta_2,
            self.vertex_sign_1,
            self.vertex_sign_2,
        )

    def normalized_coefficients(
        self, max_level_1: int, max_level_2: int
    ) -> Dict[LevelPair, complex]:
        max_level_1 = self._validate_level(max_level_1, "max_level_1")
        max_level_2 = self._validate_level(max_level_2, "max_level_2")
        return {
            (level_1, level_2): self.normalized_coefficient(
                level_1, level_2
            )
            for level_1 in range(max_level_1 + 1)
            for level_2 in range(max_level_2 + 1)
        }

    def raw_coefficients(
        self, max_level_1: int, max_level_2: int
    ) -> Dict[LevelPair, complex]:
        """Restore the ground trace of the selected long-R cycle."""

        return {
            key: 2.0 * value
            for key, value in self.normalized_coefficients(
                max_level_1, max_level_2
            ).items()
        }

    def evaluate(
        self,
        plumbing_1: RamondPlumbingParameter,
        plumbing_2: RamondPlumbingParameter,
        max_level_1: int,
        max_level_2: int,
    ) -> complex:
        """Evaluate the cycle-projected series with ground trace two."""

        return 2.0 * self.normalized_evaluate(
            plumbing_1,
            plumbing_2,
            max_level_1,
            max_level_2,
        )

    def normalized_evaluate(
        self,
        plumbing_1: RamondPlumbingParameter,
        plumbing_2: RamondPlumbingParameter,
        max_level_1: int,
        max_level_2: int,
    ) -> complex:
        """Evaluate the series after dividing out the ground trace."""

        if plumbing_1.cycle_insertion != self.cycle_insertion_1:
            raise ValueError("plumbing_1 cycle insertion does not match block")
        if plumbing_2.cycle_insertion != self.cycle_insertion_2:
            raise ValueError("plumbing_2 cycle insertion does not match block")
        return sum(
            coefficient
            * plumbing_1.level_factor(level_1)
            * plumbing_2.level_factor(level_2)
            for (
                level_1,
                level_2,
            ), coefficient in self.normalized_coefficients(
                max_level_1, max_level_2
            ).items()
        )

    def chiral_block(
        self,
        plumbing_1: RamondPlumbingParameter,
        plumbing_2: RamondPlumbingParameter,
        max_level_1: int,
        max_level_2: int,
    ) -> complex:
        """Return the cycle-projected chiral block with ground trace two."""

        return 2.0 * self.normalized_chiral_block(
            plumbing_1,
            plumbing_2,
            max_level_1,
            max_level_2,
        )

    def normalized_chiral_block(
        self,
        plumbing_1: RamondPlumbingParameter,
        plumbing_2: RamondPlumbingParameter,
        max_level_1: int,
        max_level_2: int,
    ) -> complex:
        """Return the chiral block after dividing out the ground trace."""

        return (
            plumbing_1.q
            ** (self.internal_weight_1 - self.c / 24.0)
            * plumbing_2.q
            ** (self.internal_weight_2 - self.c / 24.0)
            * self.normalized_evaluate(
                plumbing_1,
                plumbing_2,
                max_level_1,
                max_level_2,
            )
        )


class DirectRamondTorusTwoPointRegularSeed:
    """Pole-free R-necklace seed extracted from direct Ward/Gram sewing.

    The underlying two-leg Ward matrix is presently validated through level
    one on each edge.  This oracle therefore deliberately stops there.  It
    separates the analytic ground-fiber part from the universal beta-pole
    recursion and is independently reusable in a future optimized
    contracted R light-block implementation.
    """

    def __init__(
        self,
        *,
        b: complex,
        external_weight_1: complex,
        external_weight_2: complex,
        cycle_insertion_1: str = "identity",
        cycle_insertion_2: str = "identity",
        pole_tolerance: float = 1.0e-12,
    ) -> None:
        self.b = complex(b)
        self.c = central_charge(self.b)
        self.external_weight_1 = complex(external_weight_1)
        self.external_weight_2 = complex(external_weight_2)
        self.cycle_insertion_1 = (
            RamondTorusTwoPointBetaRecursionBlock._validate_cycle(
                cycle_insertion_1, "cycle_insertion_1"
            )
        )
        self.cycle_insertion_2 = (
            RamondTorusTwoPointBetaRecursionBlock._validate_cycle(
                cycle_insertion_2, "cycle_insertion_2"
            )
        )
        self.pole_tolerance = float(pole_tolerance)
        self._kernel = RamondTorusTwoPointBetaRecursionBlock(
            b=self.b,
            internal_beta_1=1.0,
            internal_beta_2=1.1,
            external_weight_1=self.external_weight_1,
            external_weight_2=self.external_weight_2,
            vertex_sign_1=1,
            vertex_sign_2=(
                -1
                if (
                    (self.cycle_insertion_1 == "parity")
                    ^ (self.cycle_insertion_2 == "parity")
                )
                else 1
            ),
            cycle_insertion_1=self.cycle_insertion_1,
            cycle_insertion_2=self.cycle_insertion_2,
            use_direct_regular_seed=False,
            pole_tolerance=self.pole_tolerance,
        )

    @staticmethod
    def _canonical_flip(beta: complex) -> bool:
        # The HJS convention used by the fusion polynomials is
        # kappa=-i beta.  RamondVermaModule instead reconstructs the
        # principal kappa=sqrt(h-c/24).  Flip both adjacent vertex signs
        # whenever those two square-root branches differ.
        principal_kappa = cmath.sqrt(-beta * beta)
        hjs_kappa = -1j * beta
        return abs(principal_kappa + hjs_kappa) < abs(
            principal_kappa - hjs_kappa
        )

    @lru_cache(maxsize=None)
    def _direct_block(
        self,
        beta_1: complex,
        beta_2: complex,
        sign_1: int,
        sign_2: int,
    ) -> BruteForceRamondTorusTwoPointBlock:
        if self._canonical_flip(beta_1) ^ self._canonical_flip(beta_2):
            sign_1 = -sign_1
            sign_2 = -sign_2
        return BruteForceRamondTorusTwoPointBlock(
            central_charge=self.c,
            internal_weight_1=self.c / 24.0 - beta_1**2,
            internal_weight_2=self.c / 24.0 - beta_2**2,
            external_weight_1=self.external_weight_1,
            external_weight_2=self.external_weight_2,
            vertex_sign_1=sign_1,
            vertex_sign_2=sign_2,
            cycle_insertion_1=self.cycle_insertion_1,
            cycle_insertion_2=self.cycle_insertion_2,
        )

    @lru_cache(maxsize=None)
    def direct_coefficient(
        self,
        level_1: int,
        level_2: int,
        beta_1: complex,
        beta_2: complex,
        sign_1: int,
        sign_2: int,
    ) -> complex:
        """Return the direct coefficient normalized to ground value one."""

        if level_1 > 1 or level_2 > 1:
            raise NotImplementedError(
                "the direct two-leg R regular seed is currently validated "
                "only through level one on each edge"
            )
        return (
            self._direct_block(
                beta_1, beta_2, sign_1, sign_2
            ).raw_coefficient(level_1, level_2)
            / 2.0
        )

    @lru_cache(maxsize=None)
    def coefficient(
        self,
        level_1: int,
        level_2: int,
        beta_1: complex,
        beta_2: complex,
        sign_1: int,
        sign_2: int,
    ) -> complex:
        direct = self.direct_coefficient(
            level_1,
            level_2,
            beta_1,
            beta_2,
            sign_1,
            sign_2,
        )
        polar = self._kernel._pole_sum(
            level_1,
            level_2,
            beta_1,
            beta_2,
            sign_1,
            sign_2,
            self.direct_coefficient,
        )
        return direct - polar


class SelfDualRamondTorusTwoPointBetaRecursionBlock:
    """Exact-Type-0B finite part of the long-R two-edge recursion."""

    def __init__(
        self,
        *,
        internal_momentum_1: complex,
        internal_momentum_2: complex,
        external_momentum_1: complex,
        external_momentum_2: complex,
        vertex_sign_1: int = 1,
        vertex_sign_2: int = 1,
        cycle_insertion_1: str = "identity",
        cycle_insertion_2: str = "identity",
        radius: float = 0.04,
        check_radius: float = 0.05,
        samples: int = 24,
    ) -> None:
        self.internal_momentum_1 = complex(internal_momentum_1)
        self.internal_momentum_2 = complex(internal_momentum_2)
        if (
            abs(self.internal_momentum_1) == 0.0
            or abs(self.internal_momentum_2) == 0.0
        ):
            raise ValueError(
                "the generic long-R recursion requires nonzero momenta"
            )
        self.external_momentum_1 = complex(external_momentum_1)
        self.external_momentum_2 = complex(external_momentum_2)
        self.vertex_sign_1 = int(vertex_sign_1)
        self.vertex_sign_2 = int(vertex_sign_2)
        self.cycle_insertion_1 = str(cycle_insertion_1)
        self.cycle_insertion_2 = str(cycle_insertion_2)
        self.radius = float(radius)
        self.check_radius = float(check_radius)
        self.samples = int(samples)
        self._cache: Dict[
            Tuple[int, int],
            Tuple[
                Dict[LevelPair, complex],
                Dict[LevelPair, FinitePartDiagnostics],
            ],
        ] = {}
        # Validate signs, cycles, and their ground-trace selection now.
        self._block_at(1.1)

    @property
    def c(self) -> complex:
        return central_charge(1.0)

    @property
    def internal_weight_1(self) -> complex:
        return ramond_liouville_weight(
            self.internal_momentum_1, 1.0
        )

    @property
    def internal_weight_2(self) -> complex:
        return ramond_liouville_weight(
            self.internal_momentum_2, 1.0
        )

    def _block_at(
        self, b: complex
    ) -> RamondTorusTwoPointBetaRecursionBlock:
        return RamondTorusTwoPointBetaRecursionBlock(
            b=b,
            internal_beta_1=ramond_beta(self.internal_momentum_1),
            internal_beta_2=ramond_beta(self.internal_momentum_2),
            external_weight_1=ns_liouville_weight(
                self.external_momentum_1, b
            ),
            external_weight_2=ns_liouville_weight(
                self.external_momentum_2, b
            ),
            vertex_sign_1=self.vertex_sign_1,
            vertex_sign_2=self.vertex_sign_2,
            cycle_insertion_1=self.cycle_insertion_1,
            cycle_insertion_2=self.cycle_insertion_2,
        )

    def _data(
        self, max_level_1: int, max_level_2: int
    ) -> Tuple[
        Dict[LevelPair, complex],
        Dict[LevelPair, FinitePartDiagnostics],
    ]:
        RamondTorusTwoPointBetaRecursionBlock._validate_level(
            max_level_1, "max_level_1"
        )
        RamondTorusTwoPointBetaRecursionBlock._validate_level(
            max_level_2, "max_level_2"
        )
        key = (max_level_1, max_level_2)
        if key not in self._cache:
            levels = tuple(
                (level_1, level_2)
                for level_1 in range(max_level_1 + 1)
                for level_2 in range(max_level_2 + 1)
            )
            self._cache[key] = _dictionary_finite_part(
                lambda b: self._block_at(b).raw_coefficients(
                    max_level_1, max_level_2
                ),
                keys=levels,
                radius=self.radius,
                check_radius=self.check_radius,
                samples=self.samples,
            )
        return self._cache[key]

    def raw_coefficients(
        self, max_level_1: int, max_level_2: int
    ) -> Dict[LevelPair, complex]:
        return dict(self._data(max_level_1, max_level_2)[0])

    def normalized_coefficients(
        self, max_level_1: int, max_level_2: int
    ) -> Dict[LevelPair, complex]:
        return {
            key: value / 2.0
            for key, value in self.raw_coefficients(
                max_level_1, max_level_2
            ).items()
        }

    def coefficient_diagnostics(
        self, max_level_1: int, max_level_2: int
    ) -> Dict[LevelPair, FinitePartDiagnostics]:
        return dict(self._data(max_level_1, max_level_2)[1])

    def evaluate(
        self,
        plumbing_1: RamondPlumbingParameter,
        plumbing_2: RamondPlumbingParameter,
        max_level_1: int,
        max_level_2: int,
    ) -> complex:
        """Evaluate the cycle-projected series with ground trace two."""

        return 2.0 * self.normalized_evaluate(
            plumbing_1,
            plumbing_2,
            max_level_1,
            max_level_2,
        )

    def normalized_evaluate(
        self,
        plumbing_1: RamondPlumbingParameter,
        plumbing_2: RamondPlumbingParameter,
        max_level_1: int,
        max_level_2: int,
    ) -> complex:
        if plumbing_1.cycle_insertion != self.cycle_insertion_1:
            raise ValueError("plumbing_1 cycle insertion does not match block")
        if plumbing_2.cycle_insertion != self.cycle_insertion_2:
            raise ValueError("plumbing_2 cycle insertion does not match block")
        return sum(
            coefficient
            * plumbing_1.level_factor(level_1)
            * plumbing_2.level_factor(level_2)
            for (
                level_1,
                level_2,
            ), coefficient in self.normalized_coefficients(
                max_level_1, max_level_2
            ).items()
        )

    def chiral_block(
        self,
        plumbing_1: RamondPlumbingParameter,
        plumbing_2: RamondPlumbingParameter,
        max_level_1: int,
        max_level_2: int,
    ) -> complex:
        return 2.0 * self.normalized_chiral_block(
            plumbing_1,
            plumbing_2,
            max_level_1,
            max_level_2,
        )

    def normalized_chiral_block(
        self,
        plumbing_1: RamondPlumbingParameter,
        plumbing_2: RamondPlumbingParameter,
        max_level_1: int,
        max_level_2: int,
    ) -> complex:
        return (
            plumbing_1.q
            ** (self.internal_weight_1 - self.c / 24.0)
            * plumbing_2.q
            ** (self.internal_weight_2 - self.c / 24.0)
            * self.normalized_evaluate(
                plumbing_1,
                plumbing_2,
                max_level_1,
                max_level_2,
            )
        )


__all__ = [
    "BruteForceRamondTorusTwoPointBlock",
    "DirectRamondTorusTwoPointRegularSeed",
    "NSTorusTwoPointHRecursionBlock",
    "NSTorusTwoPointLeadingBlock",
    "MixedNSRamondTorusTwoPointGroundBlock",
    "RamondTorusTwoPointBetaRecursionBlock",
    "RamondTorusTwoPointGroundBlock",
    "SelfDualRamondTorusTwoPointBetaRecursionBlock",
    "SelfDualNSTorusTwoPointHRecursionBlock",
]
