"""Direct long-R descendant sewing for the mixed sphere four-point block.

This is an independent low-level oracle for
``MixedRExchangeSphereFourPointBlock``.  It does not use Kac determinants,
fusion polynomials, or an elliptic recursion.  Instead it:

1. enumerates the positive-parity Ramond PBW basis;
2. evaluates its Gram matrix directly from the ordinary-c super-Virasoro
   algebra;
3. obtains the two N-R-R three-form vectors from the Ward identities;
4. contracts the vectors with the inverse Gram matrix;
5. converts the local z series to the elliptic H(q) convention.

The ground basis is deliberately unnormalized:

    w^+,  G_0 w^+.

Thus the level-one positive-parity basis is exactly

    L_-1 w^+,  G_-1 G_0 w^+,

matching the convention documented in ``super_zamolodchikov_recursion.tex``.
The implementation is intended for brute-force checks at low level, not as a
replacement for the much faster recursive block.
"""

from __future__ import annotations

import cmath
from functools import lru_cache
import math
from typing import Dict, Iterable, Literal, Sequence, Tuple

import mpmath

from ramond_sphere_blocks import ramond_beta, ramond_liouville_weight
from superconformal_blocks import (
    NSSphereFourPointBlock,
    _series_compose,
    _series_mul,
    _series_pow,
    central_charge,
    ns_liouville_weight,
)


Mode = Tuple[Literal["L", "G"], int]
State = Tuple[Mode, ...]


def _integer_partitions(total: int, maximum: int | None = None) -> Iterable[Tuple[int, ...]]:
    if total == 0:
        yield ()
        return
    if maximum is None or maximum > total:
        maximum = total
    for first in range(maximum, 0, -1):
        for tail in _integer_partitions(total - first, first):
            yield (first,) + tail


def _strict_partitions(total: int, maximum: int | None = None) -> Iterable[Tuple[int, ...]]:
    if total == 0:
        yield ()
        return
    if maximum is None or maximum > total:
        maximum = total
    for first in range(maximum, 0, -1):
        for tail in _strict_partitions(total - first, first - 1):
            yield (first,) + tail


def state_level(state: State) -> int:
    return sum(-index for _, index in state if index < 0)


def state_parity(state: State) -> int:
    return sum(1 for kind, _ in state if kind == "G") % 2


def _mode_parity(mode: Mode) -> int:
    return 1 if mode[0] == "G" else 0


def _zone(mode: Mode) -> int:
    if mode[1] < 0:
        return 0
    if mode[1] == 0:
        return 1
    return 2


def _half_binomial(order: int) -> complex:
    result = 1.0 + 0.0j
    for offset in range(order):
        result *= (0.5 - offset) / (offset + 1.0)
    return result


class RamondVermaModule:
    """Low-level unnormalized Ramond Verma module at fixed ``(c,h)``."""

    def __init__(self, *, c: complex, weight: complex) -> None:
        self.c = complex(c)
        self.weight = complex(weight)
        self.kappa_squared = self.weight - self.c / 24.0
        self._basis_cache: Dict[tuple[int, int], Tuple[State, ...]] = {}
        self._gram_cache: Dict[tuple[int, int], Tuple[Tuple[complex, ...], ...]] = {}
        self._action_cache: Dict[tuple[Mode, State], Dict[State, complex]] = {}

    def basis(self, level: int, parity: int = 0) -> Tuple[State, ...]:
        """Return a PBW basis at integer level and total fermion parity."""

        if not isinstance(level, int) or level < 0:
            raise ValueError("level must be a nonnegative integer")
        parity = int(parity)
        if parity not in (0, 1):
            raise ValueError("parity must be 0 or 1")
        key = (level, parity)
        if key in self._basis_cache:
            return self._basis_cache[key]

        states = []
        for g_level in range(level + 1):
            for g_partition in _strict_partitions(g_level):
                l_level = level - g_level
                for l_partition in _integer_partitions(l_level):
                    ground_g0 = (parity - len(g_partition)) % 2
                    word: State = tuple(
                        [("L", -part) for part in l_partition]
                        + [("G", -part) for part in g_partition]
                        + ([("G", 0)] if ground_g0 else [])
                    )
                    states.append(word)

        # Put the Virasoro-only state first and then use a deterministic PBW
        # order.  This reproduces (L_-1 w+, G_-1 G0 w+) at level one.
        states.sort(
            key=lambda state: (
                sum(1 for kind, index in state if kind == "G" and index < 0),
                tuple((kind, -index) for kind, index in state),
            )
        )
        self._basis_cache[key] = tuple(states)
        return self._basis_cache[key]

    @staticmethod
    def bpz(state: State) -> State:
        return tuple((kind, -index) for kind, index in reversed(state))

    def _super_bracket(self, left: Mode, right: Mode) -> Tuple[Tuple[complex, Mode | None], ...]:
        left_kind, left_index = left
        right_kind, right_index = right
        terms: list[tuple[complex, Mode | None]] = []

        if left_kind == "L" and right_kind == "L":
            terms.append(
                (
                    complex(left_index - right_index),
                    ("L", left_index + right_index),
                )
            )
            if left_index + right_index == 0:
                terms.append(
                    (
                        self.c
                        * (left_index**3 - left_index)
                        / 12.0,
                        None,
                    )
                )
        elif left_kind == "L" and right_kind == "G":
            terms.append(
                (
                    complex(left_index / 2.0 - right_index),
                    ("G", left_index + right_index),
                )
            )
        elif left_kind == "G" and right_kind == "L":
            terms.append(
                (
                    complex(left_index - right_index / 2.0),
                    ("G", left_index + right_index),
                )
            )
        else:
            terms.append((2.0 + 0.0j, ("L", left_index + right_index)))
            if left_index + right_index == 0:
                terms.append(
                    (
                        self.c * (left_index * left_index - 0.25) / 3.0,
                        None,
                    )
                )
        return tuple((coefficient, mode) for coefficient, mode in terms if coefficient != 0)

    @lru_cache(maxsize=None)
    def expectation(self, word: State) -> complex:
        """Return ``<w+| word |w+>`` by superalgebra normal ordering."""

        for index in range(len(word) - 1):
            left = word[index]
            right = word[index + 1]
            if _zone(left) <= _zone(right):
                continue

            swapped_sign = -1.0 if _mode_parity(left) and _mode_parity(right) else 1.0
            swapped = (
                word[:index]
                + (right, left)
                + word[index + 2 :]
            )
            result = swapped_sign * self.expectation(swapped)
            for coefficient, replacement in self._super_bracket(left, right):
                reduced = (
                    word[:index]
                    + (() if replacement is None else (replacement,))
                    + word[index + 2 :]
                )
                result += coefficient * self.expectation(reduced)
            return result

        if any(mode_index != 0 for _, mode_index in word):
            return 0.0j

        l0_count = sum(1 for kind, _ in word if kind == "L")
        g0_count = sum(1 for kind, _ in word if kind == "G")
        if g0_count % 2:
            return 0.0j
        return self.weight**l0_count * self.kappa_squared ** (g0_count // 2)

    def inner_product(self, left: State, right: State) -> complex:
        return self.expectation(self.bpz(left) + right)

    def gram_matrix(
        self, level: int, parity: int = 0
    ) -> Tuple[Tuple[complex, ...], ...]:
        key = (level, parity)
        if key not in self._gram_cache:
            basis = self.basis(level, parity)
            self._gram_cache[key] = tuple(
                tuple(self.inner_product(left, right) for right in basis)
                for left in basis
            )
        return self._gram_cache[key]

    @staticmethod
    def _solve(
        matrix: Sequence[Sequence[complex]], vector: Sequence[complex]
    ) -> Tuple[complex, ...]:
        size = len(vector)
        if size == 0:
            return ()
        mp_matrix = mpmath.matrix(
            [[mpmath.mpc(matrix[row][column]) for column in range(size)] for row in range(size)]
        )
        mp_vector = mpmath.matrix([mpmath.mpc(value) for value in vector])
        solution = mpmath.lu_solve(mp_matrix, mp_vector)
        return tuple(complex(solution[index]) for index in range(size))

    def mode_action(self, mode: Mode, state: State) -> Dict[State, complex]:
        """Expand ``mode * state`` in the PBW basis using Gram projection."""

        key = (mode, state)
        if key in self._action_cache:
            return dict(self._action_cache[key])

        target_level = state_level(state) - mode[1]
        target_parity = (state_parity(state) + _mode_parity(mode)) % 2
        if target_level < 0:
            self._action_cache[key] = {}
            return {}

        target_basis = self.basis(target_level, target_parity)
        overlaps = [
            self.expectation(self.bpz(test_state) + (mode,) + state)
            for test_state in target_basis
        ]
        coordinates = self._solve(
            self.gram_matrix(target_level, target_parity), overlaps
        )
        result = {
            basis_state: coefficient
            for basis_state, coefficient in zip(target_basis, coordinates)
            if abs(coefficient) > 1.0e-13
        }
        self._action_cache[key] = result
        return dict(result)


class RamondThreePointWardVector:
    """One N-R-R three-form vector with a generic long-R internal leg."""

    def __init__(
        self,
        *,
        module: RamondVermaModule,
        external_beta: complex,
        external_ramond_weight: complex,
        external_ns_weight: complex,
        sign: int,
    ) -> None:
        self.module = module
        self.external_beta = complex(external_beta)
        self.external_ramond_weight = complex(external_ramond_weight)
        self.external_ns_weight = complex(external_ns_weight)
        self.sign = int(sign)
        if self.sign not in (-1, 1):
            raise ValueError("sign must be +1 or -1")
        self.internal_beta = cmath.sqrt(-self.module.kappa_squared)
        self._cache: Dict[State, complex] = {}

    @staticmethod
    def _g0_coefficient(beta: complex, parity: int) -> complex:
        if parity == 0:
            return 1j * cmath.exp(-1j * math.pi / 4.0) * beta
        return 1j * cmath.exp(1j * math.pi / 4.0) * beta

    def value(self, state: State) -> complex:
        """Evaluate the normalized three-form on one PBW state."""

        if state in self._cache:
            return self._cache[state]
        parity = state_parity(state)
        level = state_level(state)

        if level == 0:
            if parity == 0 and state == ():
                result = 1.0 + 0.0j
            elif parity == 1 and state == (("G", 0),):
                result = (
                    -self.sign
                    * self._g0_coefficient(self.internal_beta, 0)
                )
            else:
                raise ValueError(f"unexpected Ramond ground PBW state {state!r}")
            self._cache[state] = result
            return result

        first = state[0]
        rest = state[1:]
        if first[1] >= 0:
            raise ValueError(f"non-creation mode in PBW state {state!r}")

        if first[0] == "L":
            mode_number = -first[1]
            result = (
                self.module.weight
                + state_level(rest)
                + mode_number * self.external_ramond_weight
                - self.external_ns_weight
            ) * self.value(rest)
            self._cache[state] = result
            return result

        mode_number = -first[1]
        rest_parity = state_parity(rest)
        external_g0 = self._g0_coefficient(
            self.external_beta, parity
        )
        ward_phase = 1j * ((-1.0) ** rest_parity)
        result = external_g0 * self.value(rest) / ward_phase

        # Equation (26) of Suchanek at z=1.  Every p>0 term lowers the
        # total descendant level, so the recursion is triangular.
        for p in range(1, level + 1):
            acted = self.module.mode_action(("G", -mode_number + p), rest)
            if not acted:
                continue
            lower_value = sum(
                coefficient * self.value(lower_state)
                for lower_state, coefficient in acted.items()
            )
            result -= _half_binomial(p) * ((-1.0) ** p) * lower_value

        self._cache[state] = result
        return result

    def vector(self, level: int, parity: int = 0) -> Tuple[complex, ...]:
        return tuple(
            self.value(state)
            for state in self.module.basis(level, parity)
        )


class RamondThreePointWardMatrix:
    """RRNS three-form with descendants on both generic Ramond legs.

    The row module is the bra Ramond leg and the column module is the ket
    Ramond leg.  Both use the unnormalized ground basis

        e_+ = w^+,  e_- = G_0 w^+.

    ``sign`` labels the normalized HJS even RRNS form.  The bottom-component
    ground matrix is therefore

        diag(1, sign * kappa_left * kappa_right).

    Matrix elements at positive levels are reduced with the ordinary-c
    super-Virasoro commutators.  This class is independent of any Kac
    determinant or Zamolodchikov recursion and is consequently the direct
    two-edge oracle needed by the torus necklace block.
    """

    def __init__(
        self,
        *,
        left_module: RamondVermaModule,
        right_module: RamondVermaModule,
        external_ns_weight: complex,
        sign: int,
    ) -> None:
        self.left_module = left_module
        self.right_module = right_module
        self.external_ns_weight = complex(external_ns_weight)
        self.sign = int(sign)
        if self.sign not in (-1, 1):
            raise ValueError("sign must be +1 or -1")
        if (
            abs(self.left_module.kappa_squared) == 0.0
            or abs(self.right_module.kappa_squared) == 0.0
        ):
            raise ValueError(
                "the generic RRNS Ward matrix excludes a short R ground fiber"
            )
        self.left_kappa = cmath.sqrt(self.left_module.kappa_squared)
        self.right_kappa = cmath.sqrt(self.right_module.kappa_squared)
        self._cache: Dict[tuple[State, State, int], complex] = {}

    @staticmethod
    def _ground_index(state: State) -> int:
        if state == ():
            return 0
        if state == (("G", 0),):
            return 1
        raise ValueError(f"expected an R ground state, got {state!r}")

    @staticmethod
    def _g0_action(module: RamondVermaModule) -> Tuple[Tuple[complex, ...], ...]:
        return (
            (0.0j, module.kappa_squared),
            (1.0 + 0.0j, 0.0j),
        )

    @property
    def ground_bottom(self) -> Tuple[Tuple[complex, ...], ...]:
        return (
            (1.0 + 0.0j, 0.0j),
            (
                0.0j,
                self.sign * self.left_kappa * self.right_kappa,
            ),
        )

    @property
    def ground_top(self) -> Tuple[Tuple[complex, ...], ...]:
        """Ground matrix of the upper NS component ``G_-1/2 V``."""

        left_action = self._g0_action(self.left_module)
        right_action = self._g0_action(self.right_module)
        bottom = self.ground_bottom
        # [G_0,V]=G_{-1/2}V in the BRY ordinary-c convention.
        return tuple(
            tuple(
                sum(
                    left_action[k][i] * bottom[k][j]
                    - bottom[i][k] * right_action[k][j]
                    for k in range(2)
                )
                for j in range(2)
            )
            for i in range(2)
        )

    @staticmethod
    def _last_creation_index(state: State) -> int | None:
        for index in range(len(state) - 1, -1, -1):
            if state[index][1] < 0:
                return index
        return None

    @staticmethod
    def _first_creation_index(state: State) -> int | None:
        for index, mode in enumerate(state):
            if mode[1] < 0:
                return index
        return None

    def value(
        self,
        left_state: State,
        right_state: State,
        component: int = 0,
    ) -> complex:
        """Return ``rho(left_state, V_component, right_state)``.

        ``component=0`` is the bottom NS primary and ``component=1`` is its
        upper superpartner.  The recursion first removes creation modes from
        the bra leg and then from the ket leg.
        """

        component = int(component)
        if component not in (0, 1):
            raise ValueError("component must be 0 or 1")
        key = (left_state, right_state, component)
        if key in self._cache:
            return self._cache[key]

        left_index = self._last_creation_index(left_state)
        if left_index is not None:
            mode = left_state[left_index]
            rest = left_state[:left_index] + left_state[left_index + 1 :]
            mode_number = -mode[1]
            left_weight = self.left_module.weight + state_level(rest)
            right_weight = self.right_module.weight + state_level(right_state)

            if mode[0] == "L":
                component_weight = (
                    self.external_ns_weight + component / 2.0
                )
                result = (
                    left_weight
                    - right_weight
                    + mode_number * component_weight
                ) * self.value(rest, right_state, component)
                acted = self.right_module.mode_action(
                    ("L", mode_number), right_state
                )
                result += sum(
                    coefficient
                    * self.value(rest, lower_state, component)
                    for lower_state, coefficient in acted.items()
                )
            elif component == 0:
                result = self.value(rest, right_state, 1)
                acted = self.right_module.mode_action(
                    ("G", mode_number), right_state
                )
                result += sum(
                    coefficient * self.value(rest, lower_state, 0)
                    for lower_state, coefficient in acted.items()
                )
            else:
                result = (
                    left_weight
                    - right_weight
                    + 2.0
                    * mode_number
                    * self.external_ns_weight
                ) * self.value(rest, right_state, 0)
                acted = self.right_module.mode_action(
                    ("G", mode_number), right_state
                )
                result -= sum(
                    coefficient * self.value(rest, lower_state, 1)
                    for lower_state, coefficient in acted.items()
                )

            self._cache[key] = result
            return result

        right_index = self._first_creation_index(right_state)
        if right_index is not None:
            mode = right_state[right_index]
            rest = right_state[:right_index] + right_state[right_index + 1 :]
            mode_number = -mode[1]
            left_weight = self.left_module.weight
            right_weight = self.right_module.weight + state_level(rest)

            if mode[0] == "L":
                component_weight = (
                    self.external_ns_weight + component / 2.0
                )
                result = (
                    right_weight
                    - left_weight
                    + mode_number * component_weight
                ) * self.value(left_state, rest, component)
            elif component == 0:
                result = -self.value(left_state, rest, 1)
            else:
                result = (
                    left_weight
                    - right_weight
                    - 2.0
                    * mode_number
                    * self.external_ns_weight
                ) * self.value(left_state, rest, 0)

            self._cache[key] = result
            return result

        left_ground = self._ground_index(left_state)
        right_ground = self._ground_index(right_state)
        ground = self.ground_bottom if component == 0 else self.ground_top
        result = ground[left_ground][right_ground]
        self._cache[key] = result
        return result

    def matrix(
        self,
        left_level: int,
        right_level: int,
        parity: int,
        component: int = 0,
    ) -> Tuple[Tuple[complex, ...], ...]:
        left_basis = self.left_module.basis(left_level, parity)
        right_basis = self.right_module.basis(right_level, parity)
        return tuple(
            tuple(
                self.value(left_state, right_state, component)
                for right_state in right_basis
            )
            for left_state in left_basis
        )


class BruteForceMixedRExchangeSphereBlock:
    """Direct mixed RRNSNS block with a generic long-R exchange."""

    def __init__(
        self,
        *,
        b: complex,
        p1_ns: complex,
        p2_r: complex,
        p3_r: complex,
        p4_ns: complex,
        internal_momentum: complex,
        sign3: int = 1,
        sign2: int = 1,
    ) -> None:
        self.b = complex(b)
        self.c = central_charge(self.b)
        self.h1 = ns_liouville_weight(p1_ns, self.b)
        self.h2 = ramond_liouville_weight(p2_r, self.b)
        self.h3 = ramond_liouville_weight(p3_r, self.b)
        self.h4 = ns_liouville_weight(p4_ns, self.b)
        self.beta2 = ramond_beta(p2_r)
        self.beta3 = ramond_beta(p3_r)
        self.internal_beta = ramond_beta(internal_momentum)
        if abs(self.internal_beta) <= 1.0e-14:
            raise ValueError("the brute-force block requires a generic long-R module")
        self.internal_weight = self.c / 24.0 - self.internal_beta**2
        self.module = RamondVermaModule(
            c=self.c, weight=self.internal_weight
        )
        self.left = RamondThreePointWardVector(
            module=self.module,
            external_beta=self.beta3,
            external_ramond_weight=self.h3,
            external_ns_weight=self.h4,
            sign=sign3,
        )
        self.right = RamondThreePointWardVector(
            module=self.module,
            external_beta=self.beta2,
            external_ramond_weight=self.h2,
            external_ns_weight=self.h1,
            sign=sign2,
        )

    def local_coefficients(self, order: int = 3) -> Tuple[complex, ...]:
        """Return ``1 + F_1 z + ... + F_order z^order``."""

        if not isinstance(order, int) or order < 0:
            raise ValueError("order must be a nonnegative integer")
        coefficients = []
        for level in range(order + 1):
            gram = self.module.gram_matrix(level, 0)
            left = self.left.vector(level, 0)
            right = self.right.vector(level, 0)
            solved_right = self.module._solve(gram, right)
            coefficients.append(
                sum(
                    left_value * right_value
                    for left_value, right_value in zip(left, solved_right)
                )
            )
        return tuple(coefficients)

    def elliptic_coefficients(self, order: int = 3) -> Dict[int, complex]:
        """Return the direct elliptic coefficients through ``q^order``."""

        if not isinstance(order, int) or order < 0:
            raise ValueError("order must be a nonnegative integer")
        local = self.local_coefficients(order)

        # We need lambda(q) one order beyond the target because
        # (16 q / z)^a contains z/q.
        theta3_full, _, z_full = NSSphereFourPointBlock._elliptic_series_data(
            order + 1
        )
        theta3 = theta3_full[: order + 1]
        z_series = z_full[: order + 1]
        z_over_16q = [
            z_full[power + 1] / 16.0 for power in range(order + 1)
        ]
        one_minus_z = [-value for value in z_series]
        one_minus_z[0] += 1.0

        vacuum_shift = (self.c - 1.5) / 24.0
        q_exponent = self.internal_weight - vacuum_shift - 1.0 / 16.0
        one_minus_exponent = vacuum_shift - self.h2 - self.h3
        theta_exponent = (
            (self.c - 1.5) / 2.0
            - 4.0 * (self.h1 + self.h2 + self.h3 + self.h4)
            + 0.5
        )

        reduced_prefactor = _series_mul(
            _series_mul(
                _series_pow(z_over_16q, -q_exponent, order),
                _series_pow(
                    one_minus_z, one_minus_exponent, order
                ),
                order,
            ),
            _series_pow(theta3, theta_exponent, order),
            order,
        )
        local_in_q = _series_compose(local, z_series, order)
        elliptic = _series_mul(
            local_in_q,
            _series_pow(reduced_prefactor, -1.0, order),
            order,
        )
        return {power: elliptic[power] for power in range(order + 1)}


__all__ = [
    "BruteForceMixedRExchangeSphereBlock",
    "RamondThreePointWardVector",
    "RamondThreePointWardMatrix",
    "RamondVermaModule",
    "state_level",
    "state_parity",
]
