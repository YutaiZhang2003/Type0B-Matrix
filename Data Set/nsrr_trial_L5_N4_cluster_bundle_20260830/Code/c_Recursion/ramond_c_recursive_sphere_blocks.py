"""Fixed-weight c-pole recursion for sphere blocks with Ramond insertions.

This module is an independent validation layer for the existing fixed-c
``h``/``beta`` recursions.  It implements the NS-edge pole kernels for the
two four-point configurations

* four external Ramond primaries (RRRR);
* two Ramond and two NS primaries in the NS-exchange channel (NSNSRR).

All internal and external conformal weights are held fixed while ``c`` moves.
Consequently the Ramond parameters

    beta_i(c)^2 = c/24 - h_i^R

must be reevaluated at every recursive pole.  A continuous square-root branch
is part of the chiral convention; the implementation uses the principal
branch, which agrees with beta=iP/sqrt(2) at the physical Type-0B point.

The exchanged NS module has a scalar inverse-Gram residue, and those residues
are directly testable against the established elliptic recursion.  The
regular part is subtler.  At fixed Ramond weights,

    beta_i(c)^2 = c/24 - h_i

so external G_0 matrix elements scale as sqrt(c).  Non-global descendants
therefore survive the large-c limit, and the ordinary all-NS osp(1|2) seed is
not the Ramond seed.  A complete c-recursive block must keep the external
Ramond ground indices open and supply the corresponding matrix-valued
large-c seed.  Until that seed is provided, this module deliberately refuses
to extrapolate the scalar global formula past the independently fixed
levels 0, 1/2, and 1.

The genuinely matrix-valued c-recursion for an internal Ramond edge is also a
separate layer and is not approximated here.
"""

from __future__ import annotations

import cmath
from dataclasses import dataclass
from functools import lru_cache
import math
from typing import Callable, Dict, Literal, Sequence, Union

from mixed_ramond_sphere_blocks import (
    MixedNSExchangeSphereFourPointBlock,
    _ns_ns_fusion_polynomial,
    _rr_ns_fusion_polynomial,
)
from ramond_sphere_blocks import (
    RamondExternalSphereFourPointBlock,
    b_from_c,
)
from superconformal_blocks import (
    NSSphereFourPointBlock,
    _rising,
    _series_compose,
    _series_mul,
    _series_pow,
)


Number = Union[complex, float]
Parity = Literal["even", "odd"]
RegularSeed = Callable[
    [int, complex, complex, Parity, int, int], complex
]


@dataclass(frozen=True)
class CRecursionResidueCheck:
    """One numerical fixed-weight c-pole comparison."""

    r: int
    s: int
    twice_level: int
    parity: Parity
    c_pole: complex
    predicted: complex
    measured: complex

    @property
    def relative_error(self) -> float:
        return abs(self.predicted - self.measured) / max(
            abs(self.measured), 1.0e-300
        )


def _sign(value: int, name: str) -> int:
    value = int(value)
    if value not in (-1, 1):
        raise ValueError(f"{name} must be +1 or -1")
    return value


def _opposite(parity: Parity) -> Parity:
    return "odd" if parity == "even" else "even"


class _NSExchangeCRecursion:
    """Shared fixed-weight NS-edge c-recursion."""

    def __init__(
        self,
        *,
        c: Number,
        internal_weight: Number,
        pole_tolerance: float,
        regular_seed: RegularSeed | None,
    ) -> None:
        self.c = complex(c)
        self.internal_weight = complex(internal_weight)
        self.pole_tolerance = float(pole_tolerance)
        self.regular_seed = regular_seed

    @staticmethod
    def _beta_at(c: complex, weight: complex) -> complex:
        return cmath.sqrt(c / 24.0 - weight)

    @staticmethod
    def _validate_level(twice_level: int, parity: Parity) -> None:
        if not isinstance(twice_level, int) or twice_level < 0:
            raise ValueError("twice_level must be a nonnegative integer")
        if parity not in ("even", "odd"):
            raise ValueError("parity must be 'even' or 'odd'")
        wanted = 0 if parity == "even" else 1
        if twice_level % 2 != wanted:
            raise ValueError(
                f"{parity} block requires twice_level congruent to {wanted}"
            )

    def _pole_kernel(
        self,
        *,
        r: int,
        s: int,
        internal_weight: complex,
        c: complex,
        parity: Parity,
        sign3: int,
        sign2: int,
    ) -> tuple[complex, complex, Parity, int, int]:
        raise NotImplementedError

    def _global_trial_seed(
        self,
        *,
        twice_level: int,
        internal_weight: complex,
        c: complex,
        parity: Parity,
        sign3: int,
        sign2: int,
    ) -> complex:
        raise NotImplementedError

    def _seed(
        self,
        *,
        twice_level: int,
        internal_weight: complex,
        c: complex,
        parity: Parity,
        sign3: int,
        sign2: int,
    ) -> complex:
        # These coefficients are fixed by the primary, G_{-1/2}, and L_{-1}
        # Ward/Gram contractions and do not yet probe the Ramond large-c
        # oscillator seed.
        if (
            (parity == "even" and twice_level in (0, 2))
            or (parity == "odd" and twice_level == 1)
        ):
            return self._global_trial_seed(
                twice_level=twice_level,
                internal_weight=internal_weight,
                c=c,
                parity=parity,
                sign3=sign3,
                sign2=sign2,
            )
        if self.regular_seed is None:
            raise NotImplementedError(
                "Ramond fixed-weight c-recursion beyond level one requires "
                "the matrix-valued Ramond large-c regular seed; the ordinary "
                "scalar osp(1|2) seed is not valid"
            )
        return complex(
            self.regular_seed(
                twice_level,
                internal_weight,
                c,
                parity,
                sign3,
                sign2,
            )
        )

    def naive_global_seed(
        self, twice_level: int, parity: Parity
    ) -> complex:
        """Return the tempting but generally incomplete scalar global seed.

        This is exposed only as a diagnostic.  It is exact at levels 0, 1/2,
        and 1, but external Ramond zero modes invalidate it at higher levels.
        """

        self._validate_level(twice_level, parity)
        return self._global_trial_seed(
            twice_level=twice_level,
            internal_weight=self.internal_weight,
            c=self.c,
            parity=parity,
            sign3=self.sign3,
            sign2=self.sign2,
        )

    def _elliptic_to_local(
        self,
        elliptic_coefficients: Sequence[complex],
        parity: Parity,
        *,
        c: complex,
        internal_weight: complex,
    ) -> tuple[complex, ...]:
        """Invert the triangular local-z to elliptic-q conversion."""

        order = len(elliptic_coefficients)
        if order == 0:
            return ()
        max_power = order - 1
        theta3, ratio, z_series = NSSphereFourPointBlock._elliptic_series_data(
            max_power
        )
        vacuum_shift = (c - 1.5) / 24.0
        alpha = internal_weight - vacuum_shift
        one_minus_z = [-value for value in z_series]
        one_minus_z[0] += 1.0
        common = _series_mul(
            _series_mul(
                _series_pow(ratio, alpha, max_power),
                _series_pow(
                    one_minus_z,
                    -self._one_minus_z_exponent(c),
                    max_power,
                ),
                max_power,
            ),
            _series_pow(
                theta3,
                -self._theta_exponent(c),
                max_power,
            ),
            max_power,
        )

        def mapped(values: Sequence[complex]) -> list[complex]:
            composed = _series_compose(values, z_series, max_power)
            if parity == "even":
                return _series_mul(common, composed, max_power)
            reduced = _series_mul(
                _series_mul(
                    common,
                    _series_pow(ratio, 0.5, max_power),
                    max_power,
                ),
                composed,
                max_power,
            )
            return [4.0 * value for value in reduced]

        local: list[complex] = []
        for index, target in enumerate(elliptic_coefficients):
            zero_value = mapped(local + [0.0j])[index]
            unit_value = mapped(local + [1.0 + 0.0j])[index]
            diagonal = unit_value - zero_value
            if abs(diagonal) == 0.0:
                raise ZeroDivisionError(
                    "singular local-to-elliptic triangular conversion"
                )
            local.append((target - zero_value) / diagonal)
        return tuple(local)

    def _one_minus_z_exponent(self, c: complex) -> complex:
        raise NotImplementedError

    def _theta_exponent(self, c: complex) -> complex:
        raise NotImplementedError

    def _reference_local_coefficients(
        self,
        *,
        c: complex,
        internal_weight: complex,
        sign3: int,
        sign2: int,
        order: int,
        parity: Parity,
    ) -> tuple[complex, ...]:
        raise NotImplementedError

    def numerical_residue_check(
        self,
        *,
        r: int,
        s: int,
        twice_level: int,
        parity: Parity,
        epsilon: float = 1.0e-6,
    ) -> CRecursionResidueCheck:
        """Compare the c-pole kernel with a symmetric h-recursion limit."""

        self._validate_level(twice_level, parity)
        if r < 2 or s < 1 or (r + s) % 2 or r * s > twice_level:
            raise ValueError(
                "NS c-pole labels require r>=2, s>=1, r+s even, "
                "and rs<=twice_level"
            )
        if epsilon <= 0.0:
            raise ValueError("epsilon must be positive")
        (
            residue,
            c_pole,
            next_parity,
            next_sign3,
            next_sign2,
        ) = self._pole_kernel(
            r=r,
            s=s,
            internal_weight=self.internal_weight,
            c=self.c,
            parity=parity,
            sign3=self.sign3,
            sign2=self.sign2,
        )
        remaining = twice_level - r * s
        tail_order = remaining // 2 + 1
        tail_values = self._reference_local_coefficients(
            c=c_pole,
            internal_weight=self.internal_weight + r * s / 2.0,
            sign3=next_sign3,
            sign2=next_sign2,
            order=tail_order,
            parity=next_parity,
        )
        tail_index = remaining // 2
        predicted = residue * tail_values[tail_index]

        target_index = twice_level // 2
        target_order = target_index + 1
        plus = self._reference_local_coefficients(
            c=c_pole + epsilon,
            internal_weight=self.internal_weight,
            sign3=self.sign3,
            sign2=self.sign2,
            order=target_order,
            parity=parity,
        )[target_index]
        minus = self._reference_local_coefficients(
            c=c_pole - epsilon,
            internal_weight=self.internal_weight,
            sign3=self.sign3,
            sign2=self.sign2,
            order=target_order,
            parity=parity,
        )[target_index]
        measured = 0.5 * epsilon * (plus - minus)
        return CRecursionResidueCheck(
            r=r,
            s=s,
            twice_level=twice_level,
            parity=parity,
            c_pole=c_pole,
            predicted=predicted,
            measured=measured,
        )

    @lru_cache(maxsize=None)
    def _coefficient(
        self,
        twice_level: int,
        internal_weight: complex,
        c: complex,
        parity: Parity,
        sign3: int,
        sign2: int,
    ) -> complex:
        result = self._seed(
            twice_level=twice_level,
            internal_weight=internal_weight,
            c=c,
            parity=parity,
            sign3=sign3,
            sign2=sign2,
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
                    next_sign3,
                    next_sign2,
                ) = self._pole_kernel(
                    r=r,
                    s=s,
                    internal_weight=internal_weight,
                    c=c,
                    parity=parity,
                    sign3=sign3,
                    sign2=sign2,
                )
                denominator = c - c_pole
                scale = max(1.0, abs(c), abs(c_pole))
                if abs(denominator) <= self.pole_tolerance * scale:
                    raise ZeroDivisionError(
                        f"c-recursion encountered the ({r},{s}) NS pole"
                    )
                result += residue / denominator * self._coefficient(
                    twice_level - product,
                    internal_weight + product / 2.0,
                    c_pole,
                    next_parity,
                    next_sign3,
                    next_sign2,
                )
        return result

    def coefficient(self, twice_level: int, parity: Parity) -> complex:
        self._validate_level(twice_level, parity)
        return self._coefficient(
            twice_level,
            self.internal_weight,
            self.c,
            parity,
            self.sign3,
            self.sign2,
        )

    def local_coefficients(
        self, order: int, parity: Parity
    ) -> Dict[int, complex]:
        if not isinstance(order, int) or order < 1:
            raise ValueError("order must be a positive integer")
        if parity == "even":
            levels = range(0, 2 * order, 2)
        elif parity == "odd":
            levels = range(1, 2 * order, 2)
        else:
            raise ValueError("parity must be 'even' or 'odd'")
        return {
            twice_level: self.coefficient(twice_level, parity)
            for twice_level in levels
        }

    def elliptic_coefficients(
        self, order: int, parity: Parity
    ) -> Dict[int, complex]:
        """Convert the c-recursive local series to the HJS elliptic frame."""

        if not isinstance(order, int) or order < 1:
            raise ValueError("order must be a positive integer")
        max_power = order - 1
        theta3, ratio, z_series = NSSphereFourPointBlock._elliptic_series_data(
            max_power
        )
        vacuum_shift = (self.c - 1.5) / 24.0
        alpha = self.internal_weight - vacuum_shift
        one_minus_z = [-value for value in z_series]
        one_minus_z[0] += 1.0
        common = _series_mul(
            _series_mul(
                _series_pow(ratio, alpha, max_power),
                _series_pow(
                    one_minus_z,
                    -self._one_minus_z_exponent(self.c),
                    max_power,
                ),
                max_power,
            ),
            _series_pow(
                theta3,
                -self._theta_exponent(self.c),
                max_power,
            ),
            max_power,
        )
        local = [
            self.coefficient(
                2 * index + (1 if parity == "odd" else 0),
                parity,
            )
            for index in range(order)
        ]
        composed = _series_compose(local, z_series, max_power)
        if parity == "even":
            values = _series_mul(common, composed, max_power)
            return {2 * index: value for index, value in enumerate(values)}
        if parity == "odd":
            reduced = _series_mul(
                _series_mul(
                    common,
                    _series_pow(ratio, 0.5, max_power),
                    max_power,
                ),
                composed,
                max_power,
            )
            return {
                2 * index + 1: 4.0 * value
                for index, value in enumerate(reduced)
            }
        raise ValueError("parity must be 'even' or 'odd'")

    def z_block(self, z: Number, order: int, parity: Parity) -> complex:
        z = complex(z)
        leading = self.internal_weight - self.h1 - self.h2
        series = sum(
            value * z ** (twice_level / 2.0)
            for twice_level, value in self.local_coefficients(
                order, parity
            ).items()
        )
        return z**leading * series


class CRecursiveRamondExternalSphereFourPointBlock(_NSExchangeCRecursion):
    r"""RRRR sphere block with an internal NS module and fixed-weight c poles."""

    def __init__(
        self,
        *,
        c: Number,
        h1: Number,
        h2: Number,
        h3: Number,
        h4: Number,
        internal_weight: Number,
        sign3: int = 1,
        sign2: int = 1,
        pole_tolerance: float = 1.0e-12,
        regular_seed: RegularSeed | None = None,
    ) -> None:
        super().__init__(
            c=c,
            internal_weight=internal_weight,
            pole_tolerance=pole_tolerance,
            regular_seed=regular_seed,
        )
        self.h1 = complex(h1)
        self.h2 = complex(h2)
        self.h3 = complex(h3)
        self.h4 = complex(h4)
        self.sign3 = _sign(sign3, "sign3")
        self.sign2 = _sign(sign2, "sign2")

    def _global_trial_seed(
        self,
        *,
        twice_level: int,
        internal_weight: complex,
        c: complex,
        parity: Parity,
        sign3: int,
        sign2: int,
    ) -> complex:
        if parity == "even":
            level = twice_level // 2
            return (
                _rising(internal_weight + self.h3 - self.h4, level)
                * _rising(internal_weight + self.h2 - self.h1, level)
                / (
                    math.factorial(level)
                    * _rising(2.0 * internal_weight, level)
                )
            )

        level = (twice_level - 1) // 2
        beta1 = self._beta_at(c, self.h1)
        beta2 = self._beta_at(c, self.h2)
        beta3 = self._beta_at(c, self.h3)
        beta4 = self._beta_at(c, self.h4)
        ground_coupling = -(
            beta4 - sign3 * beta3
        ) * (
            beta1 - sign2 * beta2
        )
        return (
            ground_coupling
            * _rising(
                internal_weight + self.h3 - self.h4 + 0.5, level
            )
            * _rising(
                internal_weight + self.h2 - self.h1 + 0.5, level
            )
            / (
                math.factorial(level)
                * _rising(2.0 * internal_weight, level + 1)
            )
        )

    def _one_minus_z_exponent(self, c: complex) -> complex:
        return (c - 1.5) / 24.0 - self.h2 - self.h3

    def _theta_exponent(self, c: complex) -> complex:
        return (c - 1.5) / 2.0 - 4.0 * (
            self.h1 + self.h2 + self.h3 + self.h4
        )

    def _reference_local_coefficients(
        self,
        *,
        c: complex,
        internal_weight: complex,
        sign3: int,
        sign2: int,
        order: int,
        parity: Parity,
    ) -> tuple[complex, ...]:
        beta1 = self._beta_at(c, self.h1)
        beta2 = self._beta_at(c, self.h2)
        beta3 = self._beta_at(c, self.h3)
        beta4 = self._beta_at(c, self.h4)
        reference = RamondExternalSphereFourPointBlock(
            b=b_from_c(c),
            beta1=beta1,
            beta2=beta2,
            beta3=beta3,
            beta4=beta4,
            internal_weight=internal_weight,
            sign3=sign3,
            sign2=sign2,
            pole_tolerance=min(self.pole_tolerance, 1.0e-14),
        )
        coefficients = reference.elliptic_coefficients(order, parity)
        offset = 0 if parity == "even" else 1
        elliptic = tuple(
            coefficients[2 * index + offset] for index in range(order)
        )
        return self._elliptic_to_local(
            elliptic,
            parity,
            c=c,
            internal_weight=internal_weight,
        )

    def _pole_kernel(
        self,
        *,
        r: int,
        s: int,
        internal_weight: complex,
        c: complex,
        parity: Parity,
        sign3: int,
        sign2: int,
    ) -> tuple[complex, complex, Parity, int, int]:
        b_pole, c_pole, derivative_c = NSSphereFourPointBlock._pole_data(
            r, s, internal_weight
        )
        beta1 = self._beta_at(c_pole, self.h1)
        beta2 = self._beta_at(c_pole, self.h2)
        beta3 = self._beta_at(c_pole, self.h3)
        beta4 = self._beta_at(c_pole, self.h4)
        left = _rr_ns_fusion_polynomial(
            b=b_pole,
            r=r,
            s=s,
            lower_beta=beta4,
            upper_beta=beta3,
            upper_sign=sign3,
        )
        right = _rr_ns_fusion_polynomial(
            b=b_pole,
            r=r,
            s=s,
            lower_beta=beta1,
            upper_beta=beta2,
            upper_sign=sign2,
        )
        odd_kac = bool(r % 2)
        orientation = -1.0 if odd_kac else 1.0
        residue = (
            orientation
            * (-derivative_c)
            * NSSphereFourPointBlock._a_factor(r, s, b_pole)
            * left
            * right
        )
        if odd_kac:
            return (
                residue,
                c_pole,
                _opposite(parity),
                -sign3,
                -sign2,
            )
        return residue, c_pole, parity, sign3, sign2


class CRecursiveMixedNSExchangeSphereFourPointBlock(_NSExchangeCRecursion):
    r"""NS4 NS3 R2 R1 block with an internal NS module and c-recursion."""

    def __init__(
        self,
        *,
        c: Number,
        h1_r: Number,
        h2_r: Number,
        h3_ns: Number,
        h4_ns: Number,
        internal_weight: Number,
        sign2: int = 1,
        pole_tolerance: float = 1.0e-12,
        regular_seed: RegularSeed | None = None,
    ) -> None:
        super().__init__(
            c=c,
            internal_weight=internal_weight,
            pole_tolerance=pole_tolerance,
            regular_seed=regular_seed,
        )
        self.h1 = complex(h1_r)
        self.h2 = complex(h2_r)
        self.h3 = complex(h3_ns)
        self.h4 = complex(h4_ns)
        self.sign3 = 1
        self.sign2 = _sign(sign2, "sign2")

    def _global_trial_seed(
        self,
        *,
        twice_level: int,
        internal_weight: complex,
        c: complex,
        parity: Parity,
        sign3: int,
        sign2: int,
    ) -> complex:
        if parity == "even":
            level = twice_level // 2
            return (
                _rising(internal_weight + self.h3 - self.h4, level)
                * _rising(internal_weight + self.h2 - self.h1, level)
                / (
                    math.factorial(level)
                    * _rising(2.0 * internal_weight, level)
                )
            )

        level = (twice_level - 1) // 2
        beta1 = self._beta_at(c, self.h1)
        beta2 = self._beta_at(c, self.h2)
        ground_coupling = cmath.exp(-1j * math.pi / 4.0) * (
            beta1 - sign2 * beta2
        )
        return (
            ground_coupling
            * _rising(
                internal_weight + self.h3 - self.h4 + 0.5, level
            )
            * _rising(
                internal_weight + self.h2 - self.h1 + 0.5, level
            )
            / (
                math.factorial(level)
                * _rising(2.0 * internal_weight, level + 1)
            )
        )

    def _one_minus_z_exponent(self, c: complex) -> complex:
        return (c - 1.5) / 24.0 - self.h2 - self.h3 + 1.0 / 16.0

    def _theta_exponent(self, c: complex) -> complex:
        return (
            (c - 1.5) / 2.0
            - 4.0 * (self.h1 + self.h2 + self.h3 + self.h4)
            + 0.5
        )

    def _reference_local_coefficients(
        self,
        *,
        c: complex,
        internal_weight: complex,
        sign3: int,
        sign2: int,
        order: int,
        parity: Parity,
    ) -> tuple[complex, ...]:
        reference = MixedNSExchangeSphereFourPointBlock.from_weights(
            b=b_from_c(c),
            h1_r=self.h1,
            h2_r=self.h2,
            h3_ns=self.h3,
            h4_ns=self.h4,
            internal_weight=internal_weight,
            sign2=sign2,
            pole_tolerance=min(self.pole_tolerance, 1.0e-14),
        )
        coefficients = reference.elliptic_coefficients(order, parity)
        offset = 0 if parity == "even" else 1
        elliptic = tuple(
            coefficients[2 * index + offset] for index in range(order)
        )
        return self._elliptic_to_local(
            elliptic,
            parity,
            c=c,
            internal_weight=internal_weight,
        )

    def _pole_kernel(
        self,
        *,
        r: int,
        s: int,
        internal_weight: complex,
        c: complex,
        parity: Parity,
        sign3: int,
        sign2: int,
    ) -> tuple[complex, complex, Parity, int, int]:
        b_pole, c_pole, derivative_c = NSSphereFourPointBlock._pole_data(
            r, s, internal_weight
        )
        beta1 = self._beta_at(c_pole, self.h1)
        beta2 = self._beta_at(c_pole, self.h2)
        left = _ns_ns_fusion_polynomial(
            b=b_pole,
            r=r,
            s=s,
            lower_weight=self.h4,
            upper_weight=self.h3,
            starred=(parity == "odd"),
        )
        right = _rr_ns_fusion_polynomial(
            b=b_pole,
            r=r,
            s=s,
            lower_beta=beta1,
            upper_beta=beta2,
            upper_sign=sign2,
        )
        odd_kac = bool(r % 2)
        phase = 1.0 + 0.0j
        next_parity = parity
        next_sign2 = sign2
        if odd_kac:
            phase = cmath.exp(
                (1j if parity == "even" else -1j) * math.pi / 4.0
            )
            next_parity = _opposite(parity)
            next_sign2 = -sign2
        residue = (
            phase
            * (-derivative_c)
            * NSSphereFourPointBlock._a_factor(r, s, b_pole)
            * left
            * right
        )
        return residue, c_pole, next_parity, sign3, next_sign2


__all__ = [
    "CRecursionResidueCheck",
    "CRecursiveMixedNSExchangeSphereFourPointBlock",
    "CRecursiveRamondExternalSphereFourPointBlock",
]
