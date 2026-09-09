"""Direct NS-channel sewing with two external Ramond ground states.

This module is an ordinary-c, low-level representation-theory oracle for the
mixed sphere block

    <NS_4 NS_3 R_2 R_1>

with an NS intermediate module.  It is deliberately independent of both the
HJS elliptic h-recursion and the fixed-weight c-pole recursion:

* :class:`NSVermaModule` constructs the NS PBW basis and Gram matrix directly
  from the super-Virasoro algebra;
* :class:`NSNSThreePointWardVector` implements the closed NS-NS-NS three-form
  of Hadasz--Jaskolski--Suchanek;
* :class:`RRNSThreePointWardMatrix` implements the Ramond Ward identities with
  the two external ground indices left open until the final BRY/HJS sign
  projection.

The code is intended for low-level derivations and regression tests.  The
recursive blocks remain the production implementation.
"""

from __future__ import annotations

import cmath
from functools import lru_cache
import math
from typing import Dict, Iterable, Literal, Sequence, Tuple

import mpmath


Mode = Tuple[Literal["L", "G"], int]
State = Tuple[Mode, ...]
Matrix2 = Tuple[Tuple[complex, complex], Tuple[complex, complex]]


def _integer_partitions(
    total: int, maximum: int | None = None
) -> Iterable[Tuple[int, ...]]:
    if total == 0:
        yield ()
        return
    if maximum is None or maximum > total:
        maximum = total
    for first in range(maximum, 0, -1):
        for tail in _integer_partitions(total - first, first):
            yield (first,) + tail


def _strict_odd_partitions(
    total: int, maximum: int | None = None
) -> Iterable[Tuple[int, ...]]:
    """Partition ``total`` into distinct positive odd integers."""

    if total == 0:
        yield ()
        return
    if maximum is None or maximum > total:
        maximum = total
    if maximum % 2 == 0:
        maximum -= 1
    for first in range(maximum, 0, -2):
        for tail in _strict_odd_partitions(total - first, first - 2):
            yield (first,) + tail


def state_twice_level(state: State) -> int:
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


def _binomial(value: float, order: int) -> complex:
    result = 1.0 + 0.0j
    for offset in range(order):
        result *= (value - offset) / (offset + 1.0)
    return result


def _matrix_add(left: Matrix2, right: Matrix2) -> Matrix2:
    return tuple(
        tuple(left[row][column] + right[row][column] for column in range(2))
        for row in range(2)
    )  # type: ignore[return-value]


def _matrix_scale(value: complex, matrix: Matrix2) -> Matrix2:
    return tuple(
        tuple(value * matrix[row][column] for column in range(2))
        for row in range(2)
    )  # type: ignore[return-value]


def _matrix_mul(left: Matrix2, right: Matrix2) -> Matrix2:
    return tuple(
        tuple(
            sum(left[row][index] * right[index][column] for index in range(2))
            for column in range(2)
        )
        for row in range(2)
    )  # type: ignore[return-value]


def _matrix_transpose(matrix: Matrix2) -> Matrix2:
    return tuple(
        tuple(matrix[column][row] for column in range(2))
        for row in range(2)
    )  # type: ignore[return-value]


class NSVermaModule:
    """Low-level NS Verma module with twice-integer mode bookkeeping.

    A Virasoro mode ``L_n`` is stored as ``("L", 2*n)`` and a supercurrent
    mode ``G_r`` as ``("G", 2*r)``.  The PBW ordering is the HJS ordering

        G_{-k_i} ... G_{-k_1} L_{-m_j} ... L_{-m_1}|h>,

    where ``k_1 > ... > k_i`` and ``m_1 >= ... >= m_j``.
    """

    def __init__(self, *, c: complex, weight: complex) -> None:
        self.c = complex(c)
        self.weight = complex(weight)
        self._basis_cache: Dict[int, Tuple[State, ...]] = {}
        self._gram_cache: Dict[int, Tuple[Tuple[complex, ...], ...]] = {}
        self._action_cache: Dict[tuple[Mode, State], Dict[State, complex]] = {}

    def basis(self, twice_level: int) -> Tuple[State, ...]:
        if not isinstance(twice_level, int) or twice_level < 0:
            raise ValueError("twice_level must be a nonnegative integer")
        if twice_level in self._basis_cache:
            return self._basis_cache[twice_level]

        states = []
        for g_twice_level in range(twice_level + 1):
            remainder = twice_level - g_twice_level
            if remainder % 2:
                continue
            for g_parts in _strict_odd_partitions(g_twice_level):
                for l_parts in _integer_partitions(remainder // 2):
                    word: State = tuple(
                        [("G", -part) for part in reversed(g_parts)]
                        + [("L", -2 * part) for part in reversed(l_parts)]
                    )
                    states.append(word)

        states.sort(
            key=lambda state: (
                sum(1 for kind, _ in state if kind == "G"),
                state,
            )
        )
        self._basis_cache[twice_level] = tuple(states)
        return self._basis_cache[twice_level]

    @staticmethod
    def bpz(state: State) -> State:
        return tuple((kind, -index) for kind, index in reversed(state))

    def _super_bracket(
        self, left: Mode, right: Mode
    ) -> Tuple[Tuple[complex, Mode | None], ...]:
        left_kind, left_twice = left
        right_kind, right_twice = right
        left_index = left_twice / 2.0
        right_index = right_twice / 2.0
        terms: list[tuple[complex, Mode | None]] = []

        if left_kind == "L" and right_kind == "L":
            terms.append(
                (
                    complex(left_index - right_index),
                    ("L", left_twice + right_twice),
                )
            )
            if left_twice + right_twice == 0:
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
                    ("G", left_twice + right_twice),
                )
            )
        elif left_kind == "G" and right_kind == "L":
            terms.append(
                (
                    complex(left_index - right_index / 2.0),
                    ("G", left_twice + right_twice),
                )
            )
        else:
            terms.append(
                (2.0 + 0.0j, ("L", left_twice + right_twice))
            )
            if left_twice + right_twice == 0:
                terms.append(
                    (
                        self.c
                        * (left_index * left_index - 0.25)
                        / 3.0,
                        None,
                    )
                )
        return tuple(
            (coefficient, mode)
            for coefficient, mode in terms
            if coefficient != 0
        )

    @lru_cache(maxsize=None)
    def expectation(self, word: State) -> complex:
        for index in range(len(word) - 1):
            left = word[index]
            right = word[index + 1]
            if _zone(left) <= _zone(right):
                continue

            swapped_sign = (
                -1.0
                if _mode_parity(left) and _mode_parity(right)
                else 1.0
            )
            swapped = (
                word[:index] + (right, left) + word[index + 2 :]
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

        if any(index != 0 for _, index in word):
            return 0.0j
        if any(kind == "G" for kind, _ in word):
            raise ValueError("the NS module has no G_0 mode")
        return self.weight ** len(word)

    def inner_product(self, left: State, right: State) -> complex:
        return self.expectation(self.bpz(left) + right)

    def gram_matrix(
        self, twice_level: int
    ) -> Tuple[Tuple[complex, ...], ...]:
        if twice_level not in self._gram_cache:
            basis = self.basis(twice_level)
            self._gram_cache[twice_level] = tuple(
                tuple(
                    self.inner_product(left, right) for right in basis
                )
                for left in basis
            )
        return self._gram_cache[twice_level]

    @staticmethod
    def _solve(
        matrix: Sequence[Sequence[complex]],
        vector: Sequence[complex],
    ) -> Tuple[complex, ...]:
        size = len(vector)
        if size == 0:
            return ()
        mp_matrix = mpmath.matrix(
            [
                [
                    mpmath.mpc(matrix[row][column])
                    for column in range(size)
                ]
                for row in range(size)
            ]
        )
        mp_vector = mpmath.matrix(
            [mpmath.mpc(value) for value in vector]
        )
        solution = mpmath.lu_solve(mp_matrix, mp_vector)
        return tuple(complex(solution[index]) for index in range(size))

    def mode_action(self, mode: Mode, state: State) -> Dict[State, complex]:
        key = (mode, state)
        if key in self._action_cache:
            return dict(self._action_cache[key])

        target_twice_level = state_twice_level(state) - mode[1]
        if target_twice_level < 0:
            self._action_cache[key] = {}
            return {}

        target_basis = self.basis(target_twice_level)
        overlaps = [
            self.expectation(self.bpz(test_state) + (mode,) + state)
            for test_state in target_basis
        ]
        coordinates = self._solve(
            self.gram_matrix(target_twice_level), overlaps
        )
        result = {
            basis_state: coefficient
            for basis_state, coefficient in zip(
                target_basis, coordinates
            )
            if abs(coefficient) > 1.0e-12
        }
        self._action_cache[key] = result
        return dict(result)


class NSNSThreePointWardVector:
    """Normalized HJS NS-NS-NS three-form on the internal NS leg."""

    def __init__(
        self,
        *,
        internal_weight: complex,
        central_weight: complex,
        right_weight: complex,
    ) -> None:
        self.internal_weight = complex(internal_weight)
        self.central_weight = complex(central_weight)
        self.right_weight = complex(right_weight)

    @staticmethod
    def _indices(state: State) -> tuple[Tuple[float, ...], Tuple[int, ...]]:
        g_indices = tuple(
            -index / 2.0 for kind, index in state if kind == "G"
        )
        l_indices = tuple(
            -index // 2 for kind, index in state if kind == "L"
        )
        # The state word contains the smallest index first; HJS labels
        # k_1,m_1 as the largest indices.
        return tuple(reversed(g_indices)), tuple(reversed(l_indices))

    @staticmethod
    def _gamma(
        base_weight: complex,
        central_weight: complex,
        right_weight: complex,
        indices: Sequence[int],
    ) -> complex:
        result = 1.0 + 0.0j
        previous = 0
        for mode in indices:
            result *= (
                base_weight
                - right_weight
                + mode * central_weight
                + previous
            )
            previous += mode
        return result

    @staticmethod
    def _eta(
        base_weight: complex,
        central_weight: complex,
        right_weight: complex,
        indices: Sequence[float],
        *,
        odd_positions: bool,
    ) -> complex:
        result = 1.0 + 0.0j
        start = 0 if odd_positions else 1
        for position in range(start, len(indices), 2):
            mode = indices[position]
            result *= (
                base_weight
                - right_weight
                + 2.0 * mode * central_weight
                + sum(indices[:position])
            )
        return result

    def value(self, state: State) -> complex:
        g_indices, l_indices = self._indices(state)
        l_level = sum(l_indices)
        if len(g_indices) % 2 == 0:
            return self._eta(
                self.internal_weight + l_level,
                self.central_weight,
                self.right_weight,
                g_indices,
                odd_positions=True,
            ) * self._gamma(
                self.internal_weight,
                self.central_weight,
                self.right_weight,
                l_indices,
            )
        return self._eta(
            self.internal_weight + l_level,
            self.central_weight,
            self.right_weight,
            g_indices,
            odd_positions=False,
        ) * self._gamma(
            self.internal_weight,
            self.central_weight + 0.5,
            self.right_weight,
            l_indices,
        )

    def vector(
        self, module: NSVermaModule, twice_level: int
    ) -> Tuple[complex, ...]:
        return tuple(
            self.value(state) for state in module.basis(twice_level)
        )


class RRNSThreePointWardMatrix:
    """RRNS three-form matrices for descendants on the NS leg.

    The row and column bases are the normalized HJS Ramond ground states
    ``(w^+,w^-)``.  ``form_parity`` selects

    ``B_even = diag(1, sign)`` or
    ``B_odd  = E_{+-} + i sign E_{-+}``.

    The matrix recursion is the second Ward identity in HJS eq. (4.3),
    specialized to Ramond ground states at ``z=1``.
    """

    def __init__(
        self,
        *,
        module: NSVermaModule,
        beta1: complex,
        beta2: complex,
        sign: int,
        form_parity: Literal["even", "odd"],
    ) -> None:
        self.module = module
        self.beta1 = complex(beta1)
        self.beta2 = complex(beta2)
        self.sign = int(sign)
        if self.sign not in (-1, 1):
            raise ValueError("sign must be +1 or -1")
        if form_parity not in ("even", "odd"):
            raise ValueError("form_parity must be 'even' or 'odd'")
        self.form_parity = form_parity
        self._cache: Dict[State, Matrix2] = {}

    @staticmethod
    def _g0_action(beta: complex) -> Matrix2:
        lower = 1j * cmath.exp(-1j * math.pi / 4.0) * beta
        upper = 1j * cmath.exp(1j * math.pi / 4.0) * beta
        return (
            (0.0j, upper),
            (lower, 0.0j),
        )

    @property
    def ground_matrix(self) -> Matrix2:
        if self.form_parity == "even":
            return (
                (1.0 + 0.0j, 0.0j),
                (0.0j, complex(self.sign)),
            )
        return (
            (0.0j, 1.0 + 0.0j),
            (1j * self.sign, 0.0j),
        )

    def _acted_matrix(self, mode: Mode, state: State) -> Matrix2:
        result: Matrix2 = ((0.0j, 0.0j), (0.0j, 0.0j))
        for lower_state, coefficient in self.module.mode_action(
            mode, state
        ).items():
            result = _matrix_add(
                result,
                _matrix_scale(coefficient, self.matrix(lower_state)),
            )
        return result

    def matrix(self, state: State) -> Matrix2:
        if state in self._cache:
            return self._cache[state]
        if not state:
            result = self.ground_matrix
            self._cache[state] = result
            return result

        first = state[0]
        rest = state[1:]
        if first[1] >= 0:
            raise ValueError(f"non-creation mode in PBW state {state!r}")

        if first[0] == "L":
            mode_number = -first[1] // 2
            coefficient = (
                self.module.weight
                + state_twice_level(rest) / 2.0
                + mode_number
                * (self.module.c / 24.0 - self.beta2**2)
                - (self.module.c / 24.0 - self.beta1**2)
            )
            result = _matrix_scale(coefficient, self.matrix(rest))
            self._cache[state] = result
            return result

        m = (-first[1] - 1) // 2
        target_twice_level = state_twice_level(state)
        result: Matrix2 = ((0.0j, 0.0j), (0.0j, 0.0j))
        for order in range(1, target_twice_level // 2 + 1):
            acted = self._acted_matrix(
                ("G", 2 * order - 2 * m - 1), rest
            )
            result = _matrix_add(
                result,
                _matrix_scale(
                    -_binomial(m + 0.5, order)
                    * ((-1.0) ** order),
                    acted,
                ),
            )

        rest_matrix = self.matrix(rest)
        if m == 0:
            result = _matrix_add(
                result,
                _matrix_mul(
                    _matrix_transpose(self._g0_action(self.beta2)),
                    rest_matrix,
                ),
            )

        column_parity: Matrix2 = (
            (1.0 + 0.0j, 0.0j),
            (0.0j, -1.0 + 0.0j),
        )
        right_action = _matrix_mul(
            _matrix_mul(rest_matrix, self._g0_action(self.beta1)),
            column_parity,
        )
        result = _matrix_add(
            result,
            _matrix_scale(
                1j
                * ((-1.0) ** (state_parity(rest) + 1 + m)),
                right_action,
            ),
        )
        self._cache[state] = result
        return result

    def value(self, state: State) -> complex:
        return self.matrix(state)[0][0]

    def vector(self, twice_level: int) -> Tuple[complex, ...]:
        return tuple(
            self.value(state)
            for state in self.module.basis(twice_level)
        )


class BruteForceMixedNSExchangeSphereBlock:
    """Direct Gram/Ward block for ``<NS_4 NS_3 R_2 R_1>``."""

    def __init__(
        self,
        *,
        c: complex,
        h1_r: complex,
        h2_r: complex,
        h3_ns: complex,
        h4_ns: complex,
        internal_weight: complex,
        sign2: int,
    ) -> None:
        self.c = complex(c)
        self.h1 = complex(h1_r)
        self.h2 = complex(h2_r)
        self.h3 = complex(h3_ns)
        self.h4 = complex(h4_ns)
        self.internal_weight = complex(internal_weight)
        self.sign2 = int(sign2)
        if self.sign2 not in (-1, 1):
            raise ValueError("sign2 must be +1 or -1")
        self.beta1 = cmath.sqrt(self.c / 24.0 - self.h1)
        self.beta2 = cmath.sqrt(self.c / 24.0 - self.h2)
        self.module = NSVermaModule(
            c=self.c, weight=self.internal_weight
        )
        self.left = NSNSThreePointWardVector(
            internal_weight=self.internal_weight,
            central_weight=self.h3,
            right_weight=self.h4,
        )

    def coefficient(
        self,
        twice_level: int,
        parity: Literal["even", "odd"],
    ) -> complex:
        if parity not in ("even", "odd"):
            raise ValueError("parity must be 'even' or 'odd'")
        if twice_level % 2 != (0 if parity == "even" else 1):
            raise ValueError("twice_level and parity are inconsistent")
        right = RRNSThreePointWardMatrix(
            module=self.module,
            beta1=self.beta1,
            beta2=self.beta2,
            sign=self.sign2,
            form_parity=parity,
        )
        gram = self.module.gram_matrix(twice_level)
        left_vector = self.left.vector(self.module, twice_level)
        right_vector = right.vector(twice_level)
        solved = self.module._solve(gram, right_vector)
        return sum(
            left_value * right_value
            for left_value, right_value in zip(left_vector, solved)
        )

    def local_coefficients(
        self,
        order: int,
        parity: Literal["even", "odd"],
    ) -> Tuple[complex, ...]:
        if not isinstance(order, int) or order < 1:
            raise ValueError("order must be a positive integer")
        offset = 0 if parity == "even" else 1
        return tuple(
            self.coefficient(2 * index + offset, parity)
            for index in range(order)
        )


__all__ = [
    "BruteForceMixedNSExchangeSphereBlock",
    "NSNSThreePointWardVector",
    "NSVermaModule",
    "RRNSThreePointWardMatrix",
    "state_parity",
    "state_twice_level",
]
