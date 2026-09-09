#!/usr/bin/env python3
"""Finite-c genus-one and genus-two NS sewing checks.

This file is an independent low-level oracle for the all-NS c-recursion.
It constructs NS PBW bases and Gram matrices directly from the ordinary-c
super-Virasoro algebra, evaluates the three-descendant NS three-form by the
Ward identities, and sews the two trinions of the genus-two theta graph.

Levels are stored as twice-levels.  The genus-two cutoff is *total* level:
``sum(twice_levels) <= 2*order``.  This is the convention used by the CCY
genus-two implementation already present in
``Code/genus_2_cross_channel``.

The recursive answer includes the necessary relative-label transport

    a_v -> a_v + rs (mod 2)

at both endpoints of an edge carrying an odd singular vector.  Omitting
this transport is a genuine error: a fixed-parity three-form cannot factor
through an odd null state into a shifted module with the same relative label.
Intrinsic primary parities ``p_i`` are not included in ``a_v``; they enter
the graded Ward tensors, sewing powers, and orientation character separately.

The theta orientation sign is exactly the human-note graded contraction sign
for the half-edge order (zero, one, infinity), with the fixed infinity-frame
sign absorbed into the lifted plumbing coordinate.  At total level three
all three two-supercurrent vacuum links therefore carry the same minus sign.

Run from the repository root with

    python3 Code/ns_genus12_finite_c_check.py

The default command reproduces the original level-six, 455-coefficient
benchmark.  Passing ``--genus-two-order 8`` checks all 969 genus-two
coefficients of total plumbing level at most eight; a genuinely generic
weight point must then be used because higher-order coincident poles are not
implemented by this scalar evaluator.
"""

from __future__ import annotations

import argparse
import cmath
from dataclasses import asdict, dataclass
from functools import lru_cache
import json
import math
from typing import Iterable, Sequence

import numpy as np

from mixed_ns_ramond_descendant_blocks import (
    Mode,
    NSVermaModule,
    State,
    state_parity,
    state_twice_level,
)
from ns_genus_c_recursion_checks import (
    ns_c_pole,
    ns_fusion_polynomial,
    ns_inverse_null_slope,
)
from ns_global_osp_block import osp_norm, osp_three_point
from ns_human_convention import (
    human_note_rho_sign,
    normalize_parity_triple,
    theta_orientation_sign as human_theta_orientation_sign,
)
from ns_regular_block import THETA_ORIENTATION, regular_coefficient


def generalized_binomial(value: float, order: int) -> complex:
    """Return the polynomial binomial ``(value choose order)``."""

    if order < 0:
        return 0.0 + 0.0j
    result = 1.0 + 0.0j
    for offset in range(order):
        result *= (complex(value) - offset) / (offset + 1)
    return result


def _mode_index(mode: Mode) -> float:
    return mode[1] / 2.0


def _is_global_state(state: State) -> bool:
    return (
        sum(1 for kind, _ in state if kind == "G") <= 1
        and all(
            (kind == "L" and index == -2)
            or (kind == "G" and index == -1)
            for kind, index in state
        )
    )


def _global_labels(state: State) -> tuple[int, int]:
    if not _is_global_state(state):
        raise ValueError("state is not in the osp(1|2) module")
    return (
        sum(1 for kind, _ in state if kind == "L"),
        sum(1 for kind, _ in state if kind == "G"),
    )


def _is_global_boundary_state(state: State) -> bool:
    """Whether a state is a bottom primary or its G_-1/2 component."""

    return state in ((), (("G", -1),))


class NumericNSVermaModule(NSVermaModule):
    """NS module with cached double-precision solves for the stress test.

    The representation-theory oracle in ``mixed_ns_ramond_descendant_blocks``
    deliberately uses an arbitrary-precision LU solve for each mode action.
    At total order eight the same target Gram matrix is solved thousands of
    times, so cache its numerical inverse here.  Gram entries and overlaps
    are still generated independently from the super-Virasoro commutators.
    """

    def __init__(self, *, c: complex, weight: complex) -> None:
        super().__init__(c=c, weight=weight)
        self._numeric_inverse_cache: dict[int, np.ndarray] = {}

    def numeric_inverse_gram(self, twice_level: int) -> np.ndarray:
        if twice_level not in self._numeric_inverse_cache:
            self._numeric_inverse_cache[twice_level] = np.linalg.inv(
                np.asarray(self.gram_matrix(twice_level), dtype=np.complex128)
            )
        return self._numeric_inverse_cache[twice_level]

    def mode_action(self, mode: Mode, state: State) -> dict[State, complex]:
        key = (mode, state)
        if key in self._action_cache:
            return dict(self._action_cache[key])
        target_level = state_twice_level(state) - mode[1]
        if target_level < 0:
            self._action_cache[key] = {}
            return {}
        target_basis = self.basis(target_level)
        if not target_basis:
            self._action_cache[key] = {}
            return {}
        overlaps = np.asarray(
            [
                self.expectation(self.bpz(test_state) + (mode,) + state)
                for test_state in target_basis
            ],
            dtype=np.complex128,
        )
        inverse = self.numeric_inverse_gram(target_level)
        # Accelerate's complex matmul can emit spurious floating-point
        # warnings even when finite inputs produce the correct finite vector.
        # Validate the result explicitly instead.
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            coordinates = inverse @ overlaps
        if not np.all(np.isfinite(coordinates)):
            raise ArithmeticError(
                "unstable numeric mode-action solve: "
                f"c={self.c!r}, h={self.weight!r}, mode={mode!r}, "
                f"state={state!r}, target_level={target_level}, "
                f"inverse_max={np.max(np.abs(inverse))!r}, "
                f"overlap_max={np.max(np.abs(overlaps))!r}"
            )
        result = {
            basis_state: complex(coefficient)
            for basis_state, coefficient in zip(target_basis, coordinates)
            if abs(coefficient) > 1.0e-12
        }
        self._action_cache[key] = result
        return dict(result)


class NSVacuumModule:
    """NS vacuum quotient with ``G_-1/2|0>`` and ``L_-1|0>`` removed."""

    def __init__(self, *, c: complex) -> None:
        self.c = complex(c)
        self.weight = 0.0 + 0.0j
        self.verma = NSVermaModule(c=self.c, weight=0.0)
        self._basis_cache: dict[int, tuple[State, ...]] = {}
        self._gram_cache: dict[int, tuple[tuple[complex, ...], ...]] = {}
        self._action_cache: dict[tuple[Mode, State], dict[State, complex]] = {}

    def basis(self, twice_level: int) -> tuple[State, ...]:
        if twice_level not in self._basis_cache:
            self._basis_cache[twice_level] = tuple(
                state
                for state in self.verma.basis(twice_level)
                if ("G", -1) not in state and ("L", -2) not in state
            )
        return self._basis_cache[twice_level]

    def gram_matrix(self, twice_level: int) -> tuple[tuple[complex, ...], ...]:
        if twice_level not in self._gram_cache:
            basis = self.basis(twice_level)
            self._gram_cache[twice_level] = tuple(
                tuple(self.verma.inner_product(left, right) for right in basis)
                for left in basis
            )
        return self._gram_cache[twice_level]

    def mode_action(self, mode: Mode, state: State) -> dict[State, complex]:
        key = (mode, state)
        if key in self._action_cache:
            return dict(self._action_cache[key])
        target_level = state_twice_level(state) - mode[1]
        if target_level < 0:
            self._action_cache[key] = {}
            return {}
        target_basis = self.basis(target_level)
        if not target_basis:
            self._action_cache[key] = {}
            return {}
        overlaps = [
            self.verma.expectation(
                self.verma.bpz(test_state) + (mode,) + state
            )
            for test_state in target_basis
        ]
        coordinates = NSVermaModule._solve(
            self.gram_matrix(target_level), overlaps
        )
        result = {
            basis_state: coefficient
            for basis_state, coefficient in zip(target_basis, coordinates)
            if abs(coefficient) > 1.0e-12
        }
        self._action_cache[key] = result
        return dict(result)


class NSDescendantThreeForm:
    """The human-note NS three-form at ``(infinity,1,0)``.

    The Ward recurrence itself uses the fixed-parity convention printed in
    ``Human Notes/SCblock.tex``; there is no caller-side convention change.

    The recursion implements Suchanek's NS Ward identities (2.23)--(2.29),
    with the Virasoro identities in exactly the CCY plane frame.
    """

    def __init__(
        self,
        *,
        c: complex,
        bra_weight: complex,
        middle_weight: complex,
        ket_weight: complex,
        vacuum: bool = False,
        primary_parities: Sequence[int] = (0, 0, 0),
    ) -> None:
        self.c = complex(c)
        self.vacuum = bool(vacuum)
        self.primary_parities = normalize_parity_triple(
            primary_parities, name="primary_parities"
        )
        self.weights = (
            complex(bra_weight),
            complex(middle_weight),
            complex(ket_weight),
        )
        if self.vacuum:
            if any(weight != 0.0 for weight in self.weights):
                raise ValueError("vacuum quotient requires zero primary weights")
            if any(self.primary_parities):
                raise ValueError("the NS vacuum primary must be even")
            self.modules = tuple(NSVacuumModule(c=self.c) for _ in range(3))
        else:
            self.modules = tuple(
                NumericNSVermaModule(c=self.c, weight=weight)
                for weight in self.weights
            )

    @staticmethod
    def _state_weight(primary_weight: complex, state: State) -> complex:
        return primary_weight + state_twice_level(state) / 2.0

    def _linear_action(
        self,
        *,
        target_states: tuple[State, State, State],
        slot: int,
        mode: Mode,
        states: tuple[State, State, State],
    ) -> complex:
        target_sign = human_note_rho_sign(
            tuple(state_parity(state) for state in target_states),
            self.primary_parities,
        )
        result = 0.0 + 0.0j
        for acted_state, coefficient in self.modules[slot].mode_action(
            mode, states[slot]
        ).items():
            changed = list(states)
            changed[slot] = acted_state
            changed_states = tuple(changed)
            changed_sign = human_note_rho_sign(
                tuple(state_parity(state) for state in changed_states),
                self.primary_parities,
            )
            result += (
                coefficient
                * target_sign
                * changed_sign
                * self.value(*changed_states)
            )
        return result

    def _reorder_leading_global_fermion(
        self, bra: State, middle: State, ket: State
    ) -> complex:
        """Move a leading G_-1/2 past the next PBW generator."""

        if len(bra) < 2 or bra[0] != ("G", -1):
            raise ValueError("expected a nontrivial leading G_-1/2 word")
        next_mode = bra[1]
        remainder = bra[2:]
        if next_mode[0] == "G":
            # G_-1/2 G_r = -G_r G_-1/2 + 2 L_(r-1/2).
            reordered = (next_mode, ("G", -1)) + remainder
            l_mode = ("L", next_mode[1] - 1)
            return (
                -self.value(reordered, middle, ket)
                + 2.0
                * self.value(
                    (l_mode,) + remainder, middle, ket
                )
            )

        # G_-1/2 L_n = L_n G_-1/2 + (-1/2-n/2) G_(n-1/2).
        # [G_{-1/2},L_{-n}]=(n-1)G_{-n-1/2}/2.  Stored mode
        # indices are twice the physical indices.
        coefficient = (-next_mode[1] - 2.0) / 4.0
        reordered = (next_mode, ("G", -1)) + remainder
        g_mode = ("G", next_mode[1] - 1)
        return (
            self.value(reordered, middle, ket)
            + coefficient
            * self.value((g_mode,) + remainder, middle, ket)
        )

    def _reflected_value(
        self, bra: State, middle: State, ket: State
    ) -> complex:
        target_states = (bra, middle, ket)
        reflected_states = (ket, middle, bra)
        target_sign = human_note_rho_sign(
            tuple(state_parity(state) for state in target_states),
            self.primary_parities,
        )
        reflected_primary_parities = (
            self.primary_parities[2],
            self.primary_parities[1],
            self.primary_parities[0],
        )
        reflected_sign = human_note_rho_sign(
            tuple(state_parity(state) for state in reflected_states),
            reflected_primary_parities,
        )
        reflected = NSDescendantThreeForm(
            c=self.c,
            bra_weight=self.weights[2],
            middle_weight=self.weights[1],
            ket_weight=self.weights[0],
            vacuum=self.vacuum,
            primary_parities=reflected_primary_parities,
        )
        return (
            target_sign
            * reflected_sign
            * reflected.value(*reflected_states)
        )

    @lru_cache(maxsize=None)
    def value(self, bra: State, middle: State, ket: State) -> complex:
        """Return the fixed-parity human-note ``rho_a``."""

        # Only the eight zero-bosonic-level component tensors are boundary
        # data.  All L_-1 chains must themselves follow from the Ward
        # identities; using a separate closed formula here would make the
        # direct oracle circular with the proposed global seed.
        if all(
            _is_global_boundary_state(state)
            for state in (bra, middle, ket)
        ):
            (n3, e3), (n2, e2), (n1, e1) = (
                _global_labels(state) for state in (bra, middle, ket)
            )
            return osp_three_point(
                n1=n3,
                n2=n2,
                n3=n1,
                epsilon1=e3,
                epsilon2=e2,
                epsilon3=e1,
                d1=self.weights[0],
                d2=self.weights[1],
                d3=self.weights[2],
                primary_parities=self.primary_parities,
            )

        states = (bra, middle, ket)

        # Removing the middle word first makes the eventual reflection of a
        # ket descendant unambiguous: the middle field is then a bottom
        # primary and carries no extra reflection sign.
        if middle:
            mode = middle[0]
            tail = middle[1:]
            if mode[0] == "G":
                k = -_mode_index(mode)
                if k < 0.5:
                    raise ValueError("expected a negative NS supercurrent mode")
                parity_13 = (state_parity(bra) + state_parity(ket)) % 2
                result = 0.0 + 0.0j

                max_first = int(
                    math.floor(state_twice_level(bra) / 2.0 - k + 1.0e-12)
                )
                for m in range(max(0, max_first) + 1):
                    coefficient = generalized_binomial(k - 1.5 + m, m)
                    result += coefficient * self._linear_action(
                        target_states=states,
                        slot=0,
                        mode=("G", int(round(2.0 * (k + m)))),
                        states=(bra, tail, ket),
                    )

                # Only m=0 can be a creation mode.  Positive modes truncate
                # once their index exceeds the ket level.
                max_second = int(
                    math.floor(state_twice_level(ket) / 2.0 + 0.5 + 1.0e-12)
                )
                # Fixed-parity human-note normalization adds one minus to the
                # ket term relative to the component-ordered S2b identity.
                ward_sign = -((-1) ** (
                    parity_13 + int(round(k + 0.5))
                ))
                for m in range(max(0, max_second) + 1):
                    coefficient = generalized_binomial(k - 1.5 + m, m)
                    result += ward_sign * coefficient * self._linear_action(
                        target_states=states,
                        slot=2,
                        mode=("G", 2 * m - 1),
                        states=(bra, tail, ket),
                    )
                return result

            n = int(round(-_mode_index(mode)))
            if n == 1:
                exponent = (
                    self._state_weight(self.weights[0], bra)
                    - self._state_weight(self.weights[1], tail)
                    - self._state_weight(self.weights[2], ket)
                )
                return exponent * self.value(bra, tail, ket)
            if n < 1:
                raise ValueError("expected a negative Virasoro mode")

            result = 0.0 + 0.0j
            max_first = max(0, state_twice_level(bra) // 2 - n)
            for m in range(max_first + 1):
                coefficient = generalized_binomial(n - 2 + m, n - 2)
                result += coefficient * self._linear_action(
                    target_states=states,
                    slot=0,
                    mode=("L", 2 * (n + m)),
                    states=(bra, tail, ket),
                )
            max_second = max(0, state_twice_level(ket) // 2 + 1)
            for m in range(max_second + 1):
                coefficient = generalized_binomial(n - 2 + m, n - 2)
                result += ((-1) ** n) * coefficient * self._linear_action(
                    target_states=states,
                    slot=2,
                    mode=("L", 2 * (m - 1)),
                    states=(bra, tail, ket),
                )
            return result

        if bra:
            mode = bra[0]
            tail = bra[1:]
            if mode == ("G", -1) and tail:
                return self._reorder_leading_global_fermion(bra, middle, ket)

            if mode[0] == "G":
                k = -_mode_index(mode)
                if k <= 0.5:
                    # With an empty middle slot, a lone global fermion can
                    # only remain together with a non-global ket.  Reflecting
                    # the two endpoint slots strictly reduces that case.
                    return self._reflected_value(bra, middle, ket)
                parity_13 = (state_parity(tail) + state_parity(ket)) % 2
                # The exponent is the parity of the full first-slot state
                # G_-k tail plus the ket, hence the explicit +1.
                result = ((-1) ** (parity_13 + 1)) * self._linear_action(
                    target_states=states,
                    slot=2,
                    mode=("G", int(round(2.0 * k))),
                    states=(tail, middle, ket),
                )
                upper = int(round(k - 0.5))
                for m in range(-1, upper + 1):
                    coefficient = generalized_binomial(k + 0.5, m + 1)
                    result += coefficient * self._linear_action(
                        target_states=states,
                        slot=1,
                        mode=("G", 2 * m + 1),
                        states=(tail, middle, ket),
                    )
                return result

            n = int(round(-_mode_index(mode)))
            result = self._linear_action(
                target_states=states,
                slot=2,
                mode=("L", 2 * n),
                states=(tail, middle, ket),
            )
            for m in range(-1, n + 1):
                coefficient = math.comb(n + 1, m + 1)
                result += coefficient * self._linear_action(
                    target_states=states,
                    slot=1,
                    mode=("L", 2 * m),
                    states=(tail, middle, ket),
                )
            return result

        if ket:
            return self._reflected_value(bra, middle, ket)

        return 1.0 + 0.0j


def theta_orientation_sign(
    twice_levels: Sequence[int],
    primary_parities: Sequence[int] = (0, 0, 0),
) -> int:
    """Return ``(-1)^Q(A+p_1,C+p_2,E+p_3)`` from the note."""

    if len(twice_levels) != 3:
        raise ValueError("theta graph has three edge levels")
    return human_theta_orientation_sign(
        tuple(int(level) % 2 for level in twice_levels),
        normalize_parity_triple(
            primary_parities, name="primary_parities"
        ),
    )


def _inverse(matrix: Sequence[Sequence[complex]]) -> np.ndarray:
    return np.linalg.inv(np.asarray(matrix, dtype=np.complex128))


class DirectThetaOracle:
    """Reusable direct finite-``c`` theta-sewing oracle.

    A level-six total-degree check visits hundreds of level triples but only
    thirteen individual edge levels.  Reusing the modules, Gram inverses, and
    Ward-identity cache changes that calculation from repeated reconstruction
    of the same algebraic data into one finite descendant computation.
    """

    def __init__(
        self,
        *,
        c: complex,
        weights: Sequence[complex],
        vacuum: bool = False,
        primary_parities: Sequence[int] = (0, 0, 0),
    ) -> None:
        if len(weights) != 3:
            raise ValueError("theta sewing requires three weights")
        self.c = complex(c)
        self.vacuum = bool(vacuum)
        self.primary_parities = normalize_parity_triple(
            primary_parities, name="primary_parities"
        )
        self.weights = tuple(complex(weight) for weight in weights)
        if self.vacuum:
            if any(weight != 0.0 for weight in self.weights):
                raise ValueError("vacuum quotient requires zero weights")
            if any(self.primary_parities):
                raise ValueError("the NS vacuum primary must be even")
            self.modules = tuple(NSVacuumModule(c=self.c) for _ in range(3))
        else:
            self.modules = tuple(
                NumericNSVermaModule(c=self.c, weight=weight)
                for weight in self.weights
            )
        self.form = NSDescendantThreeForm(
            c=self.c,
            bra_weight=self.weights[0],
            middle_weight=self.weights[1],
            ket_weight=self.weights[2],
            vacuum=self.vacuum,
            primary_parities=self.primary_parities,
        )

    @lru_cache(maxsize=None)
    def basis(self, edge: int, twice_level: int) -> tuple[State, ...]:
        return tuple(self.modules[int(edge)].basis(int(twice_level)))

    @lru_cache(maxsize=None)
    def inverse_gram(self, edge: int, twice_level: int) -> np.ndarray:
        module = self.modules[int(edge)]
        if isinstance(module, NumericNSVermaModule):
            return module.numeric_inverse_gram(int(twice_level))
        return _inverse(module.gram_matrix(int(twice_level)))

    def coefficient(
        self,
        *,
        twice_levels: Sequence[int],
        sectors: tuple[int, int],
        lifts: Sequence[int] = (1, 1, 1),
    ) -> complex:
        if len(twice_levels) != 3 or len(lifts) != 3:
            raise ValueError("theta sewing requires three levels and lifts")
        levels = tuple(int(value) for value in twice_levels)
        relative_label = sum(levels) % 2
        if any(int(sector) not in (0, 1) for sector in sectors):
            raise ValueError("sector labels must be zero or one")
        expected_sectors = (
            (0, 0)
            if self.vacuum and relative_label == 0
            else (relative_label, relative_label)
        )
        if self.vacuum and relative_label:
            return 0.0 + 0.0j
        if sectors != expected_sectors:
            return 0.0 + 0.0j

        bases = tuple(self.basis(edge, level) for edge, level in enumerate(levels))
        if any(not basis for basis in bases):
            return 0.0 + 0.0j
        inverses = tuple(
            self.inverse_gram(edge, level) for edge, level in enumerate(levels)
        )
        tensor = np.empty(tuple(len(basis) for basis in bases), dtype=np.complex128)
        for i0, state0 in enumerate(bases[0]):
            for i1, state1 in enumerate(bases[1]):
                for i2, state2 in enumerate(bases[2]):
                    tensor[i0, i1, i2] = self.form.value(state0, state1, state2)

        contracted = np.einsum(
            "abc,ad,be,cf,def->",
            tensor,
            inverses[0],
            inverses[1],
            inverses[2],
            tensor,
            optimize=True,
        )
        lift = math.prod(
            int(sign) ** ((level + primary) % 2)
            for sign, level, primary in zip(
                lifts, levels, self.primary_parities
            )
        )
        return complex(
            theta_orientation_sign(levels, self.primary_parities)
            * lift
            * contracted
        )


def direct_theta_coefficient(
    *,
    c: complex,
    weights: Sequence[complex],
    twice_levels: Sequence[int],
    sectors: tuple[int, int],
    lifts: Sequence[int] = (1, 1, 1),
    primary_parities: Sequence[int] = (0, 0, 0),
) -> complex:
    """Directly sew one finite-c theta-graph coefficient."""

    return DirectThetaOracle(
        c=c,
        weights=weights,
        primary_parities=primary_parities,
    ).coefficient(
        twice_levels=twice_levels,
        sectors=sectors,
        lifts=lifts,
    )


def global_theta_coefficient(
    *,
    weights: Sequence[complex],
    twice_levels: Sequence[int],
    sectors: tuple[int, int],
    lifts: Sequence[int] = (1, 1, 1),
    primary_parities: Sequence[int] = (0, 0, 0),
) -> complex:
    """One coefficient of the explicit global osp theta network."""

    levels = tuple(int(value) for value in twice_levels)
    primaries = normalize_parity_triple(
        primary_parities, name="primary_parities"
    )
    relative_label = sum(levels) % 2
    if sectors != (relative_label, relative_label):
        return 0.0 + 0.0j
    occupations = tuple(level // 2 for level in levels)
    fermions = tuple(level % 2 for level in levels)
    rho = osp_three_point(
        n1=occupations[0],
        n2=occupations[1],
        n3=occupations[2],
        epsilon1=fermions[0],
        epsilon2=fermions[1],
        epsilon3=fermions[2],
        d1=weights[0],
        d2=weights[1],
        d3=weights[2],
        primary_parities=primaries,
    )
    denominator = math.prod(
        osp_norm(weight, occupation, fermion)
        for weight, occupation, fermion in zip(weights, occupations, fermions)
    )
    lift = math.prod(
        int(sign) ** (fermion ^ primary)
        for sign, fermion, primary in zip(lifts, fermions, primaries)
    )
    return (
        theta_orientation_sign(levels, primaries)
        * lift
        * rho
        * rho
        / denominator
    )


# Direct large-c sewing of the NS vacuum quotient gives these coefficients in
# the ordered theta plumbing frame.  The entries are indexed by twice-levels
# and include the theta orientation sign.  They are independently regenerated
# by ``extract_theta_vacuum_seed`` below; keeping the explicit table makes the
# finite-c recursion deterministic and avoids using the generic block under
# test to define its own regular part.
THETA_VACUUM_SEED_LEVEL6: dict[tuple[int, int, int], int] = {
    (0, 0, 0): 1,
    (0, 3, 3): -1,
    (3, 0, 3): -1,
    (3, 3, 0): -1,
    (0, 3, 5): -3,
    (0, 4, 4): 1,
    (0, 5, 3): -3,
    (4, 0, 4): 1,
    (4, 4, 0): 1,
    (5, 3, 0): -3,
    (0, 3, 7): -6,
    (0, 4, 6): 4,
    (0, 5, 5): -16,
    (0, 6, 4): 4,
    (0, 7, 3): -6,
    (5, 0, 5): -1,
    (5, 5, 0): -1,
    (6, 4, 0): 4,
    (7, 3, 0): -6,
    (0, 3, 9): -10,
    (0, 4, 8): 10,
    (0, 5, 7): -50,
    (0, 6, 6): 25,
    (0, 7, 5): -50,
    (0, 8, 4): 10,
    (0, 9, 3): -10,
    (6, 0, 6): 1,
    (6, 6, 0): 1,
    (7, 5, 0): -8,
    (8, 4, 0): 10,
    (9, 3, 0): -10,
}


# The independently extracted vacuum quotient at physical total levels seven
# and eight.  Together with ``THETA_VACUUM_SEED_LEVEL6`` this is the regular
# vacuum input needed by the order-eight recursion stress test.  Every entry
# is an integer; the five-point Richardson extraction reproduces this table
# with a maximum rounding error below 3e-13.
THETA_VACUUM_SEED_LEVEL8: dict[tuple[int, int, int], int] = {
    **THETA_VACUUM_SEED_LEVEL6,
    (0, 3, 11): -15,
    (0, 4, 10): 20,
    (0, 5, 9): -120,
    (0, 6, 8): 90,
    (0, 7, 7): -226,
    (0, 8, 6): 90,
    (0, 9, 5): -120,
    (0, 10, 4): 20,
    (0, 11, 3): -15,
    (3, 3, 8): -3,
    (3, 4, 7): -1,
    (3, 7, 4): -1,
    (3, 8, 3): -3,
    (4, 3, 7): -1,
    (4, 7, 3): -1,
    (7, 0, 7): -2,
    (7, 3, 4): -1,
    (7, 4, 3): -1,
    (7, 7, 0): -2,
    (8, 3, 3): 3,
    (8, 6, 0): 10,
    (9, 5, 0): -30,
    (10, 4, 0): 20,
    (11, 3, 0): -15,
    (0, 3, 13): -21,
    (0, 4, 12): 35,
    (0, 5, 11): -245,
    (0, 6, 10): 245,
    (0, 7, 9): -742,
    (0, 8, 8): 443,
    (0, 9, 7): -742,
    (0, 10, 6): 245,
    (0, 11, 5): -245,
    (0, 12, 4): 35,
    (0, 13, 3): -21,
    (3, 3, 10): -6,
    (3, 4, 9): -4,
    (3, 5, 8): -16,
    (3, 6, 7): -4,
    (3, 7, 6): -4,
    (3, 8, 5): -16,
    (3, 9, 4): -4,
    (3, 10, 3): -6,
    (4, 3, 9): -3,
    (4, 4, 8): 2,
    (4, 5, 7): -3,
    (4, 7, 5): -3,
    (4, 8, 4): 2,
    (4, 9, 3): -3,
    (5, 3, 8): -1,
    (5, 7, 4): -3,
    (5, 8, 3): -16,
    (6, 7, 3): -4,
    (8, 0, 8): 3,
    (8, 3, 5): -1,
    (8, 4, 4): 2,
    (8, 5, 3): -1,
    (8, 8, 0): 3,
    (9, 3, 4): -3,
    (9, 4, 3): -4,
    (9, 7, 0): -22,
    (10, 3, 3): -6,
    (10, 6, 0): 45,
    (11, 5, 0): -80,
    (12, 4, 0): 35,
    (13, 3, 0): -21,
}


def extract_theta_vacuum_seed(
    max_order: int,
    *,
    c_samples: Sequence[float] = (2000.0, 4000.0, 8000.0, 16000.0, 32000.0),
) -> dict[tuple[int, int, int], complex]:
    """Independently extract the large-``c`` NS vacuum-quotient seed.

    The vacuum quotient removes ``G_-1/2|0>`` and ``L_-1|0>`` before forming
    its Gram matrices.  A Richardson fit in ``1/c`` then takes the large-c
    limit.  This calculation contains no generic internal weights and no
    c-recursion residues.
    """

    levels = [
        item for item in level_tuples(2 * int(max_order)) if sum(item) % 2 == 0
    ]
    samples = np.asarray(tuple(float(value) for value in c_samples), dtype=float)
    if len(samples) < 2:
        raise ValueError("at least two large-c samples are required")
    values = []
    for sample in samples:
        oracle = DirectThetaOracle(c=sample, weights=(0.0, 0.0, 0.0), vacuum=True)
        values.append(
            [
                oracle.coefficient(twice_levels=item, sectors=(0, 0))
                for item in levels
            ]
        )
    richardson = np.column_stack(
        [samples ** (-power) for power in range(len(samples))]
    )
    limits = np.linalg.solve(richardson, np.asarray(values))[0]
    return dict(zip(levels, limits))


def regular_theta_coefficient(
    *,
    weights: Sequence[complex],
    twice_levels: Sequence[int],
    sectors: tuple[int, int],
    lifts: Sequence[int] = (1, 1, 1),
    primary_parities: Sequence[int] = (0, 0, 0),
) -> complex:
    """Large-c coefficient from the explicit polarized Koszul formula."""

    levels = tuple(int(value) for value in twice_levels)
    primaries = normalize_parity_triple(
        primary_parities, name="primary_parities"
    )
    if sum(levels) > 16:
        raise ValueError("the explicit theta vacuum seed is truncated at level eight")
    lifted_vacuum: dict[tuple[int, int, int], complex] = {}
    for vacuum_levels, coefficient in THETA_VACUUM_SEED_LEVEL8.items():
        vacuum_lift = math.prod(
            int(lifts[edge]) ** (vacuum_levels[edge] % 2)
            for edge in range(3)
        )
        lifted_vacuum[vacuum_levels] = coefficient * vacuum_lift

    return regular_coefficient(
        level=levels,
        vacuum_coefficients=lifted_vacuum,
        global_coefficient=lambda remainder: global_theta_coefficient(
            weights=weights,
            twice_levels=remainder,
            sectors=sectors,
            lifts=lifts,
            primary_parities=primaries,
        ),
        orientation=THETA_ORIENTATION,
        global_edge_parity_offsets=primaries,
    )


def _fusion_pair(weights: Sequence[complex], edge: int) -> tuple[complex, complex]:
    # Cyclically rotate the chosen edge to the first slot.
    if edge == 0:
        return complex(weights[1]), complex(weights[2])
    if edge == 1:
        return complex(weights[2]), complex(weights[0])
    if edge == 2:
        return complex(weights[0]), complex(weights[1])
    raise ValueError("theta edge must be 0, 1, or 2")


def theta_residue_prefactor(
    *,
    r: int,
    s: int,
    edge: int,
    weights: Sequence[complex],
    sectors: tuple[int, int],
) -> tuple[complex, complex]:
    """Return ``(c_pole, J A Sigma_L Sigma_R)`` for one theta edge."""

    pole = ns_c_pole(r, s, weights[edge])
    first_weight, second_weight = _fusion_pair(weights, edge)
    polynomials = []
    for sector in sectors:
        polynomials.append(
            ns_fusion_polynomial(
                r=r,
                s=s,
                # The polynomial is indexed by the original three-form.
                # For an odd null vector it is the *shifted subblock sector*,
                # not the polynomial label, that is toggled.  This is also
                # what the direct (3,1) residue and the sphere NS recursion
                # independently require.
                a=int(sector),
                first_weight=first_weight,
                second_weight=second_weight,
                b=pole.b,
            )
        )
    residue = (
        pole.jacobian
        * ns_inverse_null_slope(r, s, pole.b)
        * polynomials[0]
        * polynomials[1]
    )
    return pole.c, residue


def _transported_sectors(
    sectors: tuple[int, int], null_parity: int
) -> tuple[int, int]:
    if not null_parity:
        return sectors
    return sectors[0] ^ 1, sectors[1] ^ 1


def recursion_theta_coefficient(
    *,
    c: complex,
    weights: Sequence[complex],
    twice_levels: Sequence[int],
    sectors: tuple[int, int],
    lifts: Sequence[int] = (1, 1, 1),
    primary_parities: Sequence[int] = (0, 0, 0),
) -> complex:
    """Evaluate the sector-coupled NS c-recursion at one multi-level."""

    initial_weights = tuple(complex(value) for value in weights)
    initial_levels = tuple(int(value) for value in twice_levels)
    lift_values = tuple(int(value) for value in lifts)
    primaries = normalize_parity_triple(
        primary_parities, name="primary_parities"
    )

    @lru_cache(maxsize=None)
    def recurse(
        current_c: complex,
        current_weights: tuple[complex, complex, complex],
        levels: tuple[int, int, int],
        current_sectors: tuple[int, int],
    ) -> complex:
        if min(levels) < 0:
            return 0.0 + 0.0j
        relative_label = sum(levels) % 2
        if current_sectors != (relative_label, relative_label):
            return 0.0 + 0.0j
        total = regular_theta_coefficient(
            weights=current_weights,
            twice_levels=levels,
            sectors=current_sectors,
            lifts=lift_values,
            primary_parities=primaries,
        )
        for edge, edge_level in enumerate(levels):
            for r in range(2, edge_level + 1):
                for s in range(1, edge_level // r + 1):
                    rs = r * s
                    if rs > edge_level or (r + s) % 2:
                        continue
                    pole_c, residue = theta_residue_prefactor(
                        r=r,
                        s=s,
                        edge=edge,
                        weights=current_weights,
                        sectors=current_sectors,
                    )
                    shifted_levels = list(levels)
                    shifted_levels[edge] -= rs
                    shifted_weights = list(current_weights)
                    shifted_weights[edge] += rs / 2.0
                    shifted_sectors = _transported_sectors(
                        current_sectors, rs % 2
                    )
                    # The removed half-integral null level carries the local
                    # plumbing lift.  Orientation/Koszul transport is already
                    # encoded by the fixed graph convention in both seeds.
                    null_lift = lift_values[edge] ** (rs % 2)
                    orientation_transport = theta_orientation_sign(
                        levels, primaries
                    ) / theta_orientation_sign(
                        shifted_levels, primaries
                    )
                    denominator = current_c - pole_c
                    if abs(denominator) < 1.0e-14:
                        raise ZeroDivisionError(
                            "coincident Kac pole requires detuning or a "
                            "higher-order Laurent recursion; unsupported by "
                            "this scalar numeric evaluator: "
                            f"c={current_c!r}, edge={edge}, (r,s)=({r},{s}), "
                            f"weights={current_weights!r}, levels={levels!r}, "
                            f"sectors={current_sectors!r}"
                        )
                    total += (
                        null_lift
                        * orientation_transport
                        * residue
                        / denominator
                        * recurse(
                            pole_c,
                            tuple(shifted_weights),
                            tuple(shifted_levels),
                            shifted_sectors,
                        )
                    )
        return total

    return recurse(
        complex(c), initial_weights, initial_levels, tuple(int(x) for x in sectors)
    )


def level_tuples(max_total_twice_level: int) -> Iterable[tuple[int, int, int]]:
    for total in range(max_total_twice_level + 1):
        for level0 in range(total + 1):
            for level1 in range(total - level0 + 1):
                yield level0, level1, total - level0 - level1


def generic_ns_character_coefficients(max_twice_level: int) -> tuple[int, ...]:
    """Return generic NS Verma character coefficients through a cutoff."""

    coefficients = [0] * (max_twice_level + 1)
    coefficients[0] = 1
    for bosonic in range(2, max_twice_level + 1, 2):
        for level in range(bosonic, max_twice_level + 1):
            coefficients[level] += coefficients[level - bosonic]
    for fermionic in range(1, max_twice_level + 1, 2):
        for level in range(max_twice_level, fermionic - 1, -1):
            coefficients[level] += coefficients[level - fermionic]
    return tuple(coefficients)


@dataclass(frozen=True)
class GenusTwoDiscrepancy:
    twice_levels: tuple[int, int, int]
    sectors: tuple[int, int]
    direct_real: float
    direct_imag: float
    recursive_real: float
    recursive_imag: float
    absolute_error: float


@dataclass(frozen=True)
class CheckSummary:
    c: complex
    weights: tuple[complex, complex, complex]
    lifts: tuple[int, int, int]
    genus_one_direct: tuple[int, ...]
    genus_one_recursive: tuple[int, ...]
    genus_one_max_error: float
    genus_two_coefficient_count: int
    genus_two_max_error: float
    genus_two_max_relative_error: float
    genus_two_max_error_by_twice_level: tuple[float, ...]
    genus_two_worst: GenusTwoDiscrepancy
    ward_global_max_error: float
    vacuum_seed_max_error: float


def _ward_global_check(
    c: complex, weights: Sequence[complex], max_twice_total: int
) -> float:
    form = NSDescendantThreeForm(
        c=c,
        bra_weight=weights[0],
        middle_weight=weights[1],
        ket_weight=weights[2],
    )
    maximum = 0.0
    for levels in level_tuples(int(max_twice_total)):
        states = []
        for level in levels:
            epsilon = level % 2
            occupation = level // 2
            states.append((("G", -1),) * epsilon + (("L", -2),) * occupation)
        expected = osp_three_point(
            n1=levels[0] // 2,
            n2=levels[1] // 2,
            n3=levels[2] // 2,
            epsilon1=levels[0] % 2,
            epsilon2=levels[1] % 2,
            epsilon3=levels[2] % 2,
            d1=weights[0],
            d2=weights[1],
            d3=weights[2],
        )
        maximum = max(maximum, abs(form.value(*states) - expected))
    return float(maximum)


def run_checks(
    *,
    c: complex = 37.25,
    weights: Sequence[complex] = (0.73, 0.91, 1.17),
    genus_one_order: int = 6,
    genus_two_order: int = 6,
    lifts: Sequence[int] = (1, 1, 1),
) -> CheckSummary:
    """Run the requested genus-one and genus-two comparisons."""

    if not 0 <= genus_two_order <= 8:
        raise ValueError("the explicit genus-two vacuum seed supports orders 0 through 8")
    if genus_one_order < 0:
        raise ValueError("genus-one order must be nonnegative")
    weights_tuple = tuple(complex(value) for value in weights)
    if len(weights_tuple) != 3:
        raise ValueError("three genus-two internal weights are required")
    lifts_tuple = tuple(int(value) for value in lifts)
    if len(lifts_tuple) != 3 or any(value not in (-1, 1) for value in lifts_tuple):
        raise ValueError("three genus-two lift signs, each +1 or -1, are required")

    # For the torus zero-point block Tr(B^{-1}B)=dim(level), at every c and
    # h.  Its c-recursion has no poles and equals the generic NS character.
    genus_one_direct = tuple(
        len(NSVermaModule(c=c, weight=weights_tuple[0]).basis(level))
        for level in range(2 * genus_one_order + 1)
    )
    genus_one_recursive = generic_ns_character_coefficients(2 * genus_one_order)
    genus_one_error = max(
        abs(left - right)
        for left, right in zip(genus_one_direct, genus_one_recursive)
    )

    discrepancies: list[GenusTwoDiscrepancy] = []
    direct_oracle = DirectThetaOracle(c=c, weights=weights_tuple)
    for levels in level_tuples(2 * genus_two_order):
        parity = sum(levels) % 2
        sectors = (parity, parity)
        direct = direct_oracle.coefficient(
            twice_levels=levels,
            sectors=sectors,
            lifts=lifts_tuple,
        )
        try:
            recursive = recursion_theta_coefficient(
                c=c,
                weights=weights_tuple,
                twice_levels=levels,
                sectors=sectors,
                lifts=lifts_tuple,
            )
        except ZeroDivisionError as error:
            raise ZeroDivisionError(
                f"order-by-order recursion failed at top twice-levels {levels!r}; "
                "use generic weights or a collision-aware finite-part evaluator"
            ) from error
        discrepancies.append(
            GenusTwoDiscrepancy(
                twice_levels=levels,
                sectors=sectors,
                direct_real=float(direct.real),
                direct_imag=float(direct.imag),
                recursive_real=float(recursive.real),
                recursive_imag=float(recursive.imag),
                absolute_error=float(abs(direct - recursive)),
            )
        )
    worst = max(discrepancies, key=lambda item: item.absolute_error)
    maximum_relative_error = max(
        item.absolute_error
        / max(1.0, abs(complex(item.direct_real, item.direct_imag)))
        for item in discrepancies
    )
    errors_by_level = tuple(
        max(
            item.absolute_error
            for item in discrepancies
            if sum(item.twice_levels) == total
        )
        for total in range(2 * genus_two_order + 1)
    )
    extracted_vacuum = extract_theta_vacuum_seed(genus_two_order)
    vacuum_error = 0.0
    for levels, value in extracted_vacuum.items():
        expected = THETA_VACUUM_SEED_LEVEL8.get(levels, 0)
        vacuum_error = max(vacuum_error, abs(value - expected))
    return CheckSummary(
        c=complex(c),
        weights=weights_tuple,
        lifts=lifts_tuple,
        genus_one_direct=genus_one_direct,
        genus_one_recursive=genus_one_recursive,
        genus_one_max_error=float(genus_one_error),
        genus_two_coefficient_count=len(discrepancies),
        genus_two_max_error=worst.absolute_error,
        genus_two_max_relative_error=float(maximum_relative_error),
        genus_two_max_error_by_twice_level=errors_by_level,
        genus_two_worst=worst,
        # The full order-eight comparison already tests the global seed at
        # every remainder level against direct Ward sewing.  Keep this
        # redundant pure-global diagnostic at the original level-six cutoff;
        # direct normal ordering of isolated level-eight global chains is much
        # more expensive than the complete 969-coefficient comparison.
        ward_global_max_error=_ward_global_check(
            c, weights_tuple, min(12, 2 * genus_two_order)
        ),
        vacuum_seed_max_error=float(vacuum_error),
    )


def _json_default(value):
    if isinstance(value, complex):
        return {"real": value.real, "imag": value.imag}
    raise TypeError(f"cannot serialize {type(value).__name__}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Direct finite-c versus c-recursive NS genus-one/two check"
    )
    parser.add_argument("--c", type=float, default=37.25)
    parser.add_argument("--weights", nargs=3, type=float, default=(0.73, 0.91, 1.17))
    parser.add_argument("--genus-one-order", type=int, default=6)
    parser.add_argument("--genus-two-order", type=int, default=6)
    parser.add_argument("--lifts", nargs=3, type=int, default=(1, 1, 1))
    args = parser.parse_args()
    summary = run_checks(
        c=args.c,
        weights=args.weights,
        genus_one_order=args.genus_one_order,
        genus_two_order=args.genus_two_order,
        lifts=args.lifts,
    )
    print(json.dumps(asdict(summary), indent=2, default=_json_default))
    if summary.ward_global_max_error > 2.0e-9:
        raise AssertionError("NS Ward recursion disagrees with the closed osp tensor")
    if summary.vacuum_seed_max_error > 2.0e-8:
        raise AssertionError("large-c vacuum quotient disagrees with the seed table")
    if summary.genus_one_max_error != 0.0:
        raise AssertionError("genus-one trace does not equal the NS character")
    if summary.genus_two_max_error > 2.0e-8:
        raise AssertionError("finite-c genus-two sewing disagrees with c-recursion")
    print("finite-c NS genus-one/two checks: PASS")


if __name__ == "__main__":
    main()
