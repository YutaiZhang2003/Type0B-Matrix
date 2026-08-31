"""Two-punctured torus blocks in the sphere--torus OPE channel.

This is the pants decomposition complementary to the necklace channel in
``superconformal_torus_two_point``.  The two external NS primaries first fuse
through an NS bridge of weight ``hs``.  That bridge is then sewn to a torus
one-point block whose handle carries weight ``hh``:

    NS(d1) --\
              (sphere pants) -- hs -- (torus one-point with handle hh)
    NS(d2) --/

The collision coordinate is ``x=1-z`` in the annulus-plane frame, with the
external insertions at ``z`` and ``1``.  The torus nome is ``q``.

The currently verified NS block sews the bridge directly through level one
and uses the complete torus one-point h-recursion on the handle.  A scalar
bridge recursion is insufficient beyond this order: odd NS null vectors
couple the even and odd toric three-form families.

An independent Ward/Gram sewing oracle is retained through bridge level one.
It includes the NS states

    |hs>, G_-1/2|hs>, L_-1|hs>.

For the even NS three-form the half-level torus one-point amplitude vanishes
by fermion parity.  The level-one amplitude follows from the plane-frame
L_0 Ward identity and is valid at every handle level.  This direct class is
used only to validate the first recursive coefficients.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
from typing import Dict, Tuple

from mixed_ns_ramond_descendant_blocks import (
    NSNSThreePointWardVector,
    NSVermaModule,
    State,
)
from mixed_ramond_sphere_blocks import (
    _ns_a_factor,
    _ns_degenerate_weight,
    _ns_ns_fusion_polynomial,
)
from ramond_sphere_blocks import ramond_beta
from self_dual_superconformal_blocks import (
    FinitePartDiagnostics,
    _dictionary_finite_part,
)
from superconformal_blocks import central_charge, ns_liouville_weight
from superconformal_torus_blocks import (
    NSPlumbingParameter,
    NSTorusOnePointBlock,
    RamondPlumbingParameter,
    RamondTorusOnePointBlock,
    SelfDualNSTorusOnePointBlock,
    SelfDualRamondTorusOnePointBlock,
)


LevelPair = Tuple[int, int]


def _validate_bridge_cutoff(max_twice_level: int) -> int:
    if not isinstance(max_twice_level, int) or not 0 <= max_twice_level <= 2:
        raise ValueError(
            "max_bridge_twice_level must be 0, 1, or 2; the direct "
            "external-descendant Ward oracle is currently exact through "
            "bridge level one"
        )
    return max_twice_level


def _validate_handle_cutoff(value: int, name: str) -> int:
    if not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _rising(value: complex, order: int) -> complex:
    result = 1.0 + 0.0j
    for offset in range(order):
        result *= value + offset
    return result


@dataclass(frozen=True)
class PlaneTorusExternalNSWardVector:
    """Torus one-point amplitudes for an external NS state through level one.

    Values are normalized by the torus one-point amplitude of the primary.
    They therefore multiply every coefficient of the handle block.

    In the annulus-plane coordinate, diagonal handle states have equal
    ``L_0`` eigenvalue.  Consequently

        <u|L_-1 V_h(1)|u> = -h <u|V_h(1)|u>.

    The one-point amplitude of ``G_-1/2 V_h`` vanishes by fermion parity.
    """

    external_weight: complex

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "external_weight", complex(self.external_weight)
        )

    def value(self, state: State) -> complex:
        if state == ():
            return 1.0 + 0.0j
        if state == (("G", -1),):
            return 0.0j
        if state == (("L", -2),):
            return -self.external_weight
        raise ValueError(
            "the torus external-state Ward vector is implemented only "
            "through NS level one"
        )

    def vector(
        self, module: NSVermaModule, twice_level: int
    ) -> Tuple[complex, ...]:
        _validate_bridge_cutoff(twice_level)
        return tuple(self.value(state) for state in module.basis(twice_level))


@dataclass(frozen=True)
class BruteForceNSOPEBridgeGroundBlock:
    """Direct Ward/Gram oracle with the torus handle at ground level.

    At handle level zero the OPE graph becomes an ordinary sphere block with
    external weights ``(d1,d2,hh,hh)``.  Contracting the two closed HJS Ward
    vectors against the full NS Gram matrix gives an independent check of
    every bridge h-recursion coefficient that is affordable by direct level
    truncation.  This class is a test oracle, not the production evaluator.
    """

    central_charge: complex
    bridge_weight: complex
    handle_weight: complex
    external_weight_1: complex
    external_weight_2: complex

    def __post_init__(self) -> None:
        for name in (
            "central_charge",
            "bridge_weight",
            "handle_weight",
            "external_weight_1",
            "external_weight_2",
        ):
            object.__setattr__(self, name, complex(getattr(self, name)))

    @property
    def module(self) -> NSVermaModule:
        return NSVermaModule(
            c=self.central_charge,
            weight=self.bridge_weight,
        )

    def coefficient(self, twice_level: int) -> complex:
        if not isinstance(twice_level, int) or twice_level < 0:
            raise ValueError("twice_level must be a nonnegative integer")
        if twice_level % 2:
            return 0.0j
        module = self.module
        sphere = NSNSThreePointWardVector(
            internal_weight=self.bridge_weight,
            central_weight=self.external_weight_1,
            right_weight=self.external_weight_2,
        ).vector(module, twice_level)
        torus = NSNSThreePointWardVector(
            internal_weight=self.bridge_weight,
            central_weight=self.handle_weight,
            right_weight=self.handle_weight,
        ).vector(module, twice_level)
        solved = module._solve(
            module.gram_matrix(twice_level),
            torus,
        )
        return sum(
            left * right for left, right in zip(sphere, solved)
        )

    def coefficients(
        self, max_twice_level: int
    ) -> Dict[int, complex]:
        if not isinstance(max_twice_level, int) or max_twice_level < 0:
            raise ValueError(
                "max_twice_level must be a nonnegative integer"
            )
        return {
            twice_level: self.coefficient(twice_level)
            for twice_level in range(max_twice_level + 1)
        }


class _BridgeSewing:
    """Common even-form sphere-to-torus bridge contraction."""

    def __init__(
        self,
        *,
        c: complex,
        bridge_weight: complex,
        external_weight_1: complex,
        external_weight_2: complex,
    ) -> None:
        self.c = complex(c)
        self.bridge_weight = complex(bridge_weight)
        self.external_weight_1 = complex(external_weight_1)
        self.external_weight_2 = complex(external_weight_2)
        self.module = NSVermaModule(
            c=self.c,
            weight=self.bridge_weight,
        )
        self.sphere = NSNSThreePointWardVector(
            internal_weight=self.bridge_weight,
            central_weight=self.external_weight_1,
            right_weight=self.external_weight_2,
        )
        self.torus = PlaneTorusExternalNSWardVector(
            external_weight=self.bridge_weight
        )

    @staticmethod
    def _orientation_factor(twice_level: int) -> complex:
        """Convert the local displacement ``z-1`` to ``x=1-z``."""

        if twice_level % 2:
            # The even block has zero odd-level coefficient through the
            # implemented range, so no square-root phase is needed here.
            return 1.0 + 0.0j
        return complex((-1) ** (twice_level // 2))

    def coefficient(self, twice_level: int) -> complex:
        _validate_bridge_cutoff(twice_level)
        gram = self.module.gram_matrix(twice_level)
        sphere = self.sphere.vector(self.module, twice_level)
        torus = self.torus.vector(self.module, twice_level)
        solved = self.module._solve(gram, torus)
        contraction = sum(
            left * right for left, right in zip(sphere, solved)
        )
        return self._orientation_factor(twice_level) * contraction

    def coefficients(self, max_twice_level: int) -> Dict[int, complex]:
        _validate_bridge_cutoff(max_twice_level)
        return {
            twice_level: self.coefficient(twice_level)
            for twice_level in range(max_twice_level + 1)
        }


class NSTorusTwoPointOPEBlock:
    """Generic-``b`` OPE-channel block with an NS handle.

    The bridge is sewn directly through level one.  The handle coefficients
    are supplied by the all-level NS torus one-point h-recursion.
    """

    def __init__(
        self,
        *,
        b: complex,
        bridge_weight: complex,
        handle_weight: complex,
        external_weight_1: complex,
        external_weight_2: complex,
    ) -> None:
        self.b = complex(b)
        self.c = central_charge(self.b)
        self.bridge_weight = complex(bridge_weight)
        self.handle_weight = complex(handle_weight)
        self.external_weight_1 = complex(external_weight_1)
        self.external_weight_2 = complex(external_weight_2)
        self.bridge = _BridgeSewing(
            c=self.c,
            bridge_weight=self.bridge_weight,
            external_weight_1=self.external_weight_1,
            external_weight_2=self.external_weight_2,
        )
        self.handle = NSTorusOnePointBlock(
            b=self.b,
            internal_weight=self.handle_weight,
            external_weight=self.bridge_weight,
        )

    def bridge_coefficients(
        self, max_bridge_twice_level: int
    ) -> Dict[int, complex]:
        return self.bridge.coefficients(max_bridge_twice_level)

    def raw_coefficients(
        self,
        max_bridge_twice_level: int,
        max_handle_twice_level: int,
    ) -> Dict[LevelPair, complex]:
        bridge = self.bridge_coefficients(max_bridge_twice_level)
        handle = self.handle.raw_coefficients(
            _validate_handle_cutoff(
                max_handle_twice_level, "max_handle_twice_level"
            )
        )
        return {
            (bridge_level, handle_level): bridge_coefficient
            * handle_coefficient
            for bridge_level, bridge_coefficient in bridge.items()
            for handle_level, handle_coefficient in handle.items()
        }

    def evaluate(
        self,
        collision_parameter: complex,
        plumbing: NSPlumbingParameter,
        max_bridge_twice_level: int,
        max_handle_twice_level: int,
    ) -> complex:
        x = complex(collision_parameter)
        return sum(
            coefficient
            * x ** (bridge_level / 2.0)
            * plumbing.level_factor(handle_level)
            for (
                bridge_level,
                handle_level,
            ), coefficient in self.raw_coefficients(
                max_bridge_twice_level,
                max_handle_twice_level,
            ).items()
        )

    def chiral_block(
        self,
        collision_parameter: complex,
        plumbing: NSPlumbingParameter,
        max_bridge_twice_level: int,
        max_handle_twice_level: int,
    ) -> complex:
        x = complex(collision_parameter)
        return (
            x
            ** (
                self.bridge_weight
                - self.external_weight_1
                - self.external_weight_2
            )
            * plumbing.q ** (self.handle_weight - self.c / 24.0)
            * self.evaluate(
                x,
                plumbing,
                max_bridge_twice_level,
                max_handle_twice_level,
            )
        )


class _IncompleteNSTorusTwoPointOPEHRecursionBlock:
    r"""Development scratchpad for the coupled bridge h-recursion.

    This is a genuine internal-weight recursion, not a descendant-level
    construction.  Recursion in the separating bridge weight ``hs`` uses
    one fusion polynomial from each endpoint of that edge:

    * ``P(d1,d2)`` from the sphere three-point vertex;
    * ``P(hh,hh)`` from the self-glued torus vertex.

    The pole-free bridge seed is the global ``osp(1|2)`` contribution.  In
    the even family its integer-level coefficients form

        (hs)_n (hs+d1-d2)_n / (n! (2hs)_n).

    This is the hypergeometric global block in ``x=1-z``.  It multiplies the
    complete torus one-point h-recursion with external weight ``hs``.
    Consequently both internal directions are recursive: handle poles are
    already resummed in the seed, while every bridge residue calls the same
    object at the shifted bridge weight.

    This scalar ansatz is deliberately private and is not used by the
    production evaluator.  It matches bridge level one but fails the
    independent sphere-block limit at higher levels because the odd toric
    three-form seed is missing.
    """

    def __init__(
        self,
        *,
        b: complex,
        bridge_weight: complex,
        handle_weight: complex,
        external_weight_1: complex,
        external_weight_2: complex,
        pole_tolerance: float = 1.0e-12,
    ) -> None:
        self.b = complex(b)
        self.c = central_charge(self.b)
        self.bridge_weight = complex(bridge_weight)
        self.handle_weight = complex(handle_weight)
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
    def _handle_raw_coefficient(
        self,
        twice_level: int,
        bridge_weight: complex,
    ) -> complex:
        handle = NSTorusOnePointBlock(
            b=self.b,
            internal_weight=self.handle_weight,
            external_weight=bridge_weight,
            pole_tolerance=self.pole_tolerance,
        )
        return handle.raw_coefficients(twice_level)[twice_level]

    def _regular_coefficient(
        self,
        bridge_twice_level: int,
        handle_twice_level: int,
        bridge_weight: complex,
        routing: int,
    ) -> complex:
        if routing != 0 or bridge_twice_level % 2:
            return 0.0j
        level = bridge_twice_level // 2
        denominator = (
            math.factorial(level)
            * _rising(2.0 * bridge_weight, level)
        )
        if abs(denominator) <= self.pole_tolerance:
            raise ZeroDivisionError(
                "the global OPE seed encountered a singular bridge norm"
            )
        global_coefficient = (
            _rising(bridge_weight, level)
            * _rising(
                bridge_weight
                + self.external_weight_1
                - self.external_weight_2,
                level,
            )
            / denominator
        )
        return global_coefficient * self._handle_raw_coefficient(
            handle_twice_level,
            bridge_weight,
        )

    @lru_cache(maxsize=None)
    def _coefficient(
        self,
        bridge_twice_level: int,
        handle_twice_level: int,
        bridge_weight: complex,
        routing: int,
    ) -> complex:
        if bridge_twice_level < 0 or handle_twice_level < 0:
            return 0.0j
        result = self._regular_coefficient(
            bridge_twice_level,
            handle_twice_level,
            bridge_weight,
            routing,
        )
        for r in range(1, bridge_twice_level + 1):
            for s in range(1, bridge_twice_level // r + 1):
                product = r * s
                if (
                    product > bridge_twice_level
                    or (r + s) % 2
                ):
                    continue
                degenerate = _ns_degenerate_weight(self.b, r, s)
                denominator = bridge_weight - degenerate
                scale = max(
                    1.0,
                    abs(bridge_weight),
                    abs(degenerate),
                )
                if abs(denominator) <= self.pole_tolerance * scale:
                    raise ZeroDivisionError(
                        "OPE bridge h-recursion encountered the "
                        f"({r},{s}) NS pole; use the self-dual "
                        "coefficient-wise finite-part wrapper"
                    )
                sphere_fusion = _ns_ns_fusion_polynomial(
                    b=self.b,
                    r=r,
                    s=s,
                    lower_weight=self.external_weight_1,
                    upper_weight=self.external_weight_2,
                    starred=bool(routing),
                )
                handle_fusion = _ns_ns_fusion_polynomial(
                    b=self.b,
                    r=r,
                    s=s,
                    lower_weight=self.handle_weight,
                    upper_weight=self.handle_weight,
                    starred=bool(routing),
                )
                residue = (
                    _ns_a_factor(
                        self.b,
                        r,
                        s,
                        self.pole_tolerance,
                    )
                    * sphere_fusion
                    * handle_fusion
                )
                next_routing = routing ^ (product % 2)
                result += (
                    residue
                    / denominator
                    * self._coefficient(
                        bridge_twice_level - product,
                        handle_twice_level,
                        complex(degenerate + product / 2.0),
                        next_routing,
                    )
                )
        return result

    def raw_coefficient(
        self,
        bridge_twice_level: int,
        handle_twice_level: int,
    ) -> complex:
        raise NotImplementedError(
            "the bridge h-recursion is matrix-valued in the even/odd "
            "toric three-form sectors; the scalar ansatz is disabled"
        )
        bridge_twice_level = self._validate_twice_level(
            bridge_twice_level, "bridge_twice_level"
        )
        handle_twice_level = self._validate_twice_level(
            handle_twice_level, "handle_twice_level"
        )
        return self._coefficient(
            bridge_twice_level,
            handle_twice_level,
            self.bridge_weight,
            0,
        )

    def raw_coefficients(
        self,
        max_bridge_twice_level: int,
        max_handle_twice_level: int,
    ) -> Dict[LevelPair, complex]:
        max_bridge_twice_level = self._validate_twice_level(
            max_bridge_twice_level, "max_bridge_twice_level"
        )
        max_handle_twice_level = self._validate_twice_level(
            max_handle_twice_level, "max_handle_twice_level"
        )
        return {
            (bridge_level, handle_level): self.raw_coefficient(
                bridge_level, handle_level
            )
            for bridge_level in range(max_bridge_twice_level + 1)
            for handle_level in range(max_handle_twice_level + 1)
        }

    def evaluate(
        self,
        collision_parameter: complex,
        plumbing: NSPlumbingParameter,
        max_bridge_twice_level: int,
        max_handle_twice_level: int,
    ) -> complex:
        x = complex(collision_parameter)
        return sum(
            coefficient
            * x ** (bridge_level / 2.0)
            * plumbing.level_factor(handle_level)
            for (
                bridge_level,
                handle_level,
            ), coefficient in self.raw_coefficients(
                max_bridge_twice_level,
                max_handle_twice_level,
            ).items()
        )

    def chiral_block(
        self,
        collision_parameter: complex,
        plumbing: NSPlumbingParameter,
        max_bridge_twice_level: int,
        max_handle_twice_level: int,
    ) -> complex:
        x = complex(collision_parameter)
        return (
            x
            ** (
                self.bridge_weight
                - self.external_weight_1
                - self.external_weight_2
            )
            * plumbing.q ** (self.handle_weight - self.c / 24.0)
            * self.evaluate(
                x,
                plumbing,
                max_bridge_twice_level,
                max_handle_twice_level,
            )
        )


class RamondHandleTorusTwoPointOPEBlock:
    """Generic-``b`` OPE-channel block with a long-R handle."""

    def __init__(
        self,
        *,
        b: complex,
        bridge_weight: complex,
        handle_beta: complex,
        external_weight_1: complex,
        external_weight_2: complex,
        sign: int = 1,
    ) -> None:
        self.b = complex(b)
        self.c = central_charge(self.b)
        self.bridge_weight = complex(bridge_weight)
        self.handle_beta = complex(handle_beta)
        self.external_weight_1 = complex(external_weight_1)
        self.external_weight_2 = complex(external_weight_2)
        self.sign = int(sign)
        self.bridge = _BridgeSewing(
            c=self.c,
            bridge_weight=self.bridge_weight,
            external_weight_1=self.external_weight_1,
            external_weight_2=self.external_weight_2,
        )
        self.handle = RamondTorusOnePointBlock(
            b=self.b,
            internal_beta=self.handle_beta,
            external_weight=self.bridge_weight,
            sign=self.sign,
        )

    @property
    def handle_weight(self) -> complex:
        return self.handle.internal_weight

    def bridge_coefficients(
        self, max_bridge_twice_level: int
    ) -> Dict[int, complex]:
        return self.bridge.coefficients(max_bridge_twice_level)

    def normalized_raw_coefficients(
        self,
        max_bridge_twice_level: int,
        max_handle_level: int,
    ) -> Dict[LevelPair, complex]:
        bridge = self.bridge_coefficients(max_bridge_twice_level)
        handle = self.handle.raw_even_coefficients(
            _validate_handle_cutoff(max_handle_level, "max_handle_level")
        )
        return {
            (bridge_level, handle_level): bridge_coefficient
            * handle_coefficient
            for bridge_level, bridge_coefficient in bridge.items()
            for handle_level, handle_coefficient in enumerate(handle)
        }

    def normalized_evaluate(
        self,
        collision_parameter: complex,
        plumbing: RamondPlumbingParameter,
        max_bridge_twice_level: int,
        max_handle_level: int,
    ) -> complex:
        x = complex(collision_parameter)
        return sum(
            coefficient
            * x ** (bridge_level / 2.0)
            * plumbing.level_factor(handle_level)
            for (
                bridge_level,
                handle_level,
            ), coefficient in self.normalized_raw_coefficients(
                max_bridge_twice_level,
                max_handle_level,
            ).items()
        )

    def normalized_chiral_block(
        self,
        collision_parameter: complex,
        plumbing: RamondPlumbingParameter,
        max_bridge_twice_level: int,
        max_handle_level: int,
    ) -> complex:
        x = complex(collision_parameter)
        return (
            x
            ** (
                self.bridge_weight
                - self.external_weight_1
                - self.external_weight_2
            )
            * plumbing.q ** (self.handle_weight - self.c / 24.0)
            * self.normalized_evaluate(
                x,
                plumbing,
                max_bridge_twice_level,
                max_handle_level,
            )
        )


class _IncompleteSelfDualNSTorusTwoPointOPEBlock:
    """Disabled wrapper around the incomplete scalar bridge recursion."""

    def __init__(
        self,
        *,
        bridge_momentum: complex,
        handle_momentum: complex,
        external_momentum_1: complex,
        external_momentum_2: complex,
        radius: float = 0.04,
        check_radius: float = 0.05,
        samples: int = 24,
    ) -> None:
        self.bridge_momentum = complex(bridge_momentum)
        self.handle_momentum = complex(handle_momentum)
        self.external_momentum_1 = complex(external_momentum_1)
        self.external_momentum_2 = complex(external_momentum_2)
        self.radius = float(radius)
        self.check_radius = float(check_radius)
        self.samples = int(samples)
        self.c = central_charge(1.0)
        self.bridge_weight = ns_liouville_weight(
            self.bridge_momentum, 1.0
        )
        self.handle_weight = ns_liouville_weight(
            self.handle_momentum, 1.0
        )
        self.external_weight_1 = ns_liouville_weight(
            self.external_momentum_1, 1.0
        )
        self.external_weight_2 = ns_liouville_weight(
            self.external_momentum_2, 1.0
        )
        self._cache: Dict[
            LevelPair,
            Tuple[
                Dict[LevelPair, complex],
                Dict[LevelPair, FinitePartDiagnostics],
            ],
        ] = {}

    def _block_at(
        self, b: complex
    ) -> _IncompleteNSTorusTwoPointOPEHRecursionBlock:
        return _IncompleteNSTorusTwoPointOPEHRecursionBlock(
            b=b,
            bridge_weight=ns_liouville_weight(
                self.bridge_momentum, b
            ),
            handle_weight=ns_liouville_weight(
                self.handle_momentum, b
            ),
            external_weight_1=ns_liouville_weight(
                self.external_momentum_1, b
            ),
            external_weight_2=ns_liouville_weight(
                self.external_momentum_2, b
            ),
        )

    def _data(
        self,
        max_bridge_twice_level: int,
        max_handle_twice_level: int,
    ) -> Tuple[
        Dict[LevelPair, complex],
        Dict[LevelPair, FinitePartDiagnostics],
    ]:
        _IncompleteNSTorusTwoPointOPEHRecursionBlock._validate_twice_level(
            max_bridge_twice_level, "max_bridge_twice_level"
        )
        _IncompleteNSTorusTwoPointOPEHRecursionBlock._validate_twice_level(
            max_handle_twice_level, "max_handle_twice_level"
        )
        key = (max_bridge_twice_level, max_handle_twice_level)
        if key not in self._cache:
            keys = tuple(
                (bridge_level, handle_level)
                for bridge_level in range(max_bridge_twice_level + 1)
                for handle_level in range(max_handle_twice_level + 1)
            )
            self._cache[key] = _dictionary_finite_part(
                lambda b: self._block_at(b).raw_coefficients(
                    max_bridge_twice_level,
                    max_handle_twice_level,
                ),
                keys=keys,
                radius=self.radius,
                check_radius=self.check_radius,
                samples=self.samples,
            )
        return self._cache[key]

    def raw_coefficients(
        self,
        max_bridge_twice_level: int,
        max_handle_twice_level: int,
    ) -> Dict[LevelPair, complex]:
        return dict(
            self._data(
                max_bridge_twice_level,
                max_handle_twice_level,
            )[0]
        )

    def bridge_coefficients(
        self, max_bridge_twice_level: int
    ) -> Dict[int, complex]:
        values = self.raw_coefficients(max_bridge_twice_level, 0)
        return {
            bridge_level: values[(bridge_level, 0)]
            for bridge_level in range(max_bridge_twice_level + 1)
        }

    def coefficient_diagnostics(
        self,
        max_bridge_twice_level: int,
        max_handle_twice_level: int,
    ) -> Dict[LevelPair, FinitePartDiagnostics]:
        return dict(
            self._data(
                max_bridge_twice_level,
                max_handle_twice_level,
            )[1]
        )

    def evaluate(
        self,
        collision_parameter: complex,
        plumbing: NSPlumbingParameter,
        max_bridge_twice_level: int,
        max_handle_twice_level: int,
    ) -> complex:
        x = complex(collision_parameter)
        return sum(
            coefficient
            * x ** (bridge_level / 2.0)
            * plumbing.level_factor(handle_level)
            for (
                bridge_level,
                handle_level,
            ), coefficient in self.raw_coefficients(
                max_bridge_twice_level,
                max_handle_twice_level,
            ).items()
        )

    def chiral_block(
        self,
        collision_parameter: complex,
        plumbing: NSPlumbingParameter,
        max_bridge_twice_level: int,
        max_handle_twice_level: int,
    ) -> complex:
        x = complex(collision_parameter)
        return (
            x
            ** (
                self.bridge_weight
                - self.external_weight_1
                - self.external_weight_2
            )
            * plumbing.q ** (self.handle_weight - self.c / 24.0)
            * self.evaluate(
                x,
                plumbing,
                max_bridge_twice_level,
                max_handle_twice_level,
            )
        )


class SelfDualNSTorusTwoPointOPEBlock:
    """Verified Type-0B NS OPE block through bridge level one."""

    def __init__(
        self,
        *,
        bridge_momentum: complex,
        handle_momentum: complex,
        external_momentum_1: complex,
        external_momentum_2: complex,
        radius: float = 0.04,
        check_radius: float = 0.05,
        samples: int = 24,
    ) -> None:
        self.bridge_momentum = complex(bridge_momentum)
        self.handle_momentum = complex(handle_momentum)
        self.external_momentum_1 = complex(external_momentum_1)
        self.external_momentum_2 = complex(external_momentum_2)
        self.c = central_charge(1.0)
        self.bridge_weight = ns_liouville_weight(
            self.bridge_momentum, 1.0
        )
        self.handle_weight = ns_liouville_weight(
            self.handle_momentum, 1.0
        )
        self.external_weight_1 = ns_liouville_weight(
            self.external_momentum_1, 1.0
        )
        self.external_weight_2 = ns_liouville_weight(
            self.external_momentum_2, 1.0
        )
        self.bridge = _BridgeSewing(
            c=self.c,
            bridge_weight=self.bridge_weight,
            external_weight_1=self.external_weight_1,
            external_weight_2=self.external_weight_2,
        )
        self.handle = SelfDualNSTorusOnePointBlock(
            internal_momentum=self.handle_momentum,
            external_momentum=self.bridge_momentum,
            radius=radius,
            check_radius=check_radius,
            samples=samples,
        )

    bridge_coefficients = NSTorusTwoPointOPEBlock.bridge_coefficients
    raw_coefficients = NSTorusTwoPointOPEBlock.raw_coefficients
    evaluate = NSTorusTwoPointOPEBlock.evaluate
    chiral_block = NSTorusTwoPointOPEBlock.chiral_block


class SelfDualRamondHandleTorusTwoPointOPEBlock:
    """Exact-Type-0B OPE-channel block with a long-R handle."""

    def __init__(
        self,
        *,
        bridge_momentum: complex,
        handle_momentum: complex,
        external_momentum_1: complex,
        external_momentum_2: complex,
        sign: int = 1,
        radius: float = 0.04,
        check_radius: float = 0.05,
        samples: int = 24,
    ) -> None:
        self.bridge_momentum = complex(bridge_momentum)
        self.handle_momentum = complex(handle_momentum)
        self.external_momentum_1 = complex(external_momentum_1)
        self.external_momentum_2 = complex(external_momentum_2)
        self.sign = int(sign)
        self.c = central_charge(1.0)
        self.bridge_weight = ns_liouville_weight(
            self.bridge_momentum, 1.0
        )
        self.external_weight_1 = ns_liouville_weight(
            self.external_momentum_1, 1.0
        )
        self.external_weight_2 = ns_liouville_weight(
            self.external_momentum_2, 1.0
        )
        self.bridge = _BridgeSewing(
            c=self.c,
            bridge_weight=self.bridge_weight,
            external_weight_1=self.external_weight_1,
            external_weight_2=self.external_weight_2,
        )
        self.handle = SelfDualRamondTorusOnePointBlock(
            internal_momentum=self.handle_momentum,
            external_momentum=self.bridge_momentum,
            sign=self.sign,
            radius=radius,
            check_radius=check_radius,
            samples=samples,
        )
        self.handle_beta = ramond_beta(self.handle_momentum)

    @property
    def handle_weight(self) -> complex:
        return self.handle.internal_weight

    bridge_coefficients = (
        RamondHandleTorusTwoPointOPEBlock.bridge_coefficients
    )
    normalized_raw_coefficients = (
        RamondHandleTorusTwoPointOPEBlock.normalized_raw_coefficients
    )
    normalized_evaluate = (
        RamondHandleTorusTwoPointOPEBlock.normalized_evaluate
    )
    normalized_chiral_block = (
        RamondHandleTorusTwoPointOPEBlock.normalized_chiral_block
    )


__all__ = [
    "BruteForceNSOPEBridgeGroundBlock",
    "NSTorusTwoPointOPEBlock",
    "PlaneTorusExternalNSWardVector",
    "RamondHandleTorusTwoPointOPEBlock",
    "SelfDualNSTorusTwoPointOPEBlock",
    "SelfDualRamondHandleTorusTwoPointOPEBlock",
]
