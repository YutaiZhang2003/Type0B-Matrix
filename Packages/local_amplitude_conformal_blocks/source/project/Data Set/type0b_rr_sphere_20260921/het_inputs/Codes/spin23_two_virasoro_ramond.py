#!/usr/bin/env python3
r"""Two-Virasoro construction of Ramond genus-one two-point blocks.

The supplied derivation proves the algebra embedding

.. math::

   \mathrm{Vir}_{c^{(1)}}\oplus\mathrm{Vir}_{c^{(2)}}
   \subset F\oplus\mathrm{SVir}_c,

where :math:`F` is one auxiliary Majorana fermion.  This module implements
the finite algebraic part of that statement rather than assuming the
conditional mixed-sector finite-product formula in the note.

At a fixed branching grade, a branch highest state is obtained as the
eigenvector of :math:`L^{(1)}_0` with the analytically known branch weight.
Because that grade is the first grade of the corresponding pair of Virasoro
Verma modules, the selected eigenvector is automatically a simultaneous
highest state of both commuting Virasoro algebras at generic parameters.
The code nevertheless reports an eigen-residual and rejects a non-isolated
eigenvalue.

The oriented mixed branching coefficients are evaluated directly from the
finite branch states, the auxiliary-fermion R--NS--R form, and the exact
super-Virasoro R--NS--R Ward tensor already used by the direct block oracle.
No closed Ramond blow-up factor is inserted.  Products of left and right
branching coefficients are divided by the three branch norms, so arbitrary
eigenvector rescalings and square-root choices cancel.

This module currently provides the exact branching data needed when the cut
NS edge is at level zero or one half.  Those two coefficients give,
respectively, the primary-primary and
:math:`G_{-1/2}`--:math:`G_{-1/2}` torus two-point blocks.  The ordinary
Virasoro theta-block recursion and formal auxiliary-block division are
implemented below in this file.
"""

from __future__ import annotations

import cmath
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from itertools import combinations
import math
from typing import Callable, Literal, Mapping, Sequence

import numpy as np
import sympy as sp

from ns_algebra.ns_sca import (
    G as ns_G,
    L as ns_L,
    Word as NSWord,
    act_mode as ns_act_mode,
    descendant_inner_product as ns_inner_product,
    fermion_parity as ns_fermion_parity,
    pbw_basis as ns_pbw_basis,
    twice_level as ns_twice_level,
)
from ramond_algebra.nrr_three_point_tensor import (
    hjs_to_polynomial_ground_tensor,
    rr_three_point_from_ground_tensor,
)
from ramond_algebra.ns_rr_three_point_tensor import (
    hjs_ns_rr_polynomial_ground_tensor,
    ns_rr_three_point_from_ground_tensor,
)
from ramond_algebra.ramond_sca import (
    G as r_G,
    L as r_L,
    PBWState,
    act_mode as r_act_mode,
    descendant_inner_product as r_inner_product,
    pbw_basis as r_pbw_basis,
)
from spin23_genus1_recursion import (
    dictionary_finite_part,
    ns_liouville_weight,
    super_liouville_central_charge,
)


Sector = Literal["NS", "R"]
BranchNumber = Fraction
SuperState = NSWord | PBWState

_RR_TEMPLATE_H_INFINITY, _RR_TEMPLATE_H_NS, _RR_TEMPLATE_H_ZERO = sp.symbols(
    "h_infinity h_ns h_zero"
)
_RR_TEMPLATE_C = sp.Symbol("c")
_RR_TEMPLATE_GROUND = sp.symbols("ground_00 ground_01 ground_10 ground_11")
_RR_TEMPLATE_GROUND_MATRIX = sp.Matrix(2, 2, _RR_TEMPLATE_GROUND)


@dataclass(frozen=True)
class AuxiliaryFermionState:
    r"""One auxiliary-Majorana Fock state.

    ``twice_modes`` contains the positive doubled labels of the occupied
    creation operators :math:`f_{-r}`, in increasing order.  Ramond states
    additionally retain the parity of the two-dimensional zero-mode ground
    fiber.
    """

    sector: Sector
    twice_modes: tuple[int, ...] = ()
    ground_parity: int = 0

    def __post_init__(self) -> None:
        if self.sector not in ("NS", "R"):
            raise ValueError("sector must be 'NS' or 'R'")
        modes = tuple(int(value) for value in self.twice_modes)
        object.__setattr__(self, "twice_modes", modes)
        if modes != tuple(sorted(modes)) or len(set(modes)) != len(modes):
            raise ValueError("twice_modes must be distinct and increasing")
        if any(value <= 0 for value in modes):
            raise ValueError("occupied creation-mode labels must be positive")
        if self.sector == "NS":
            if self.ground_parity != 0:
                raise ValueError("the NS auxiliary vacuum has one ground state")
            if any(value % 2 == 0 for value in modes):
                raise ValueError("NS fermion modes must be half-integral")
        else:
            if self.ground_parity not in (0, 1):
                raise ValueError("Ramond ground_parity must be zero or one")
            if any(value % 2 for value in modes):
                raise ValueError("Ramond fermion modes must be integral")

    @property
    def twice_level(self) -> int:
        """Return twice the oscillator level above the sector ground."""

        return sum(self.twice_modes)

    @property
    def parity(self) -> int:
        """Return total fermion parity."""

        return self.ground_parity ^ (len(self.twice_modes) % 2)


def _fermion_subsets(total: int, allowed: tuple[int, ...]) -> tuple[tuple[int, ...], ...]:
    return tuple(
        subset
        for size in range(len(allowed) + 1)
        for subset in combinations(allowed, size)
        if sum(subset) == total
    )


@lru_cache(maxsize=None)
def auxiliary_fermion_basis(
    sector: Sector,
    twice_level: int,
    parity: int | None = None,
) -> tuple[AuxiliaryFermionState, ...]:
    """Return an auxiliary-fermion Fock basis at one twice-level."""

    if sector not in ("NS", "R"):
        raise ValueError("sector must be 'NS' or 'R'")
    if not isinstance(twice_level, int) or twice_level < 0:
        raise ValueError("twice_level must be a nonnegative integer")
    if parity not in (None, 0, 1):
        raise ValueError("parity must be None, zero, or one")
    if sector == "NS":
        allowed = tuple(range(1, twice_level + 1, 2))
        grounds = (0,)
    else:
        if twice_level % 2:
            return ()
        allowed = tuple(range(2, twice_level + 1, 2))
        grounds = (0, 1)
    states = tuple(
        AuxiliaryFermionState(sector, subset, ground)
        for subset in _fermion_subsets(twice_level, allowed)
        for ground in grounds
    )
    if parity is not None:
        states = tuple(state for state in states if state.parity == parity)
    return states


def _add_auxiliary_state(
    out: dict[AuxiliaryFermionState, complex],
    state: AuxiliaryFermionState,
    coefficient: complex,
) -> None:
    value = out.get(state, 0.0j) + complex(coefficient)
    if abs(value) == 0:
        out.pop(state, None)
    else:
        out[state] = value


def auxiliary_fermion_mode_action(
    twice_mode: int,
    state: AuxiliaryFermionState,
) -> dict[AuxiliaryFermionState, complex]:
    r"""Act with :math:`f_r`, where ``twice_mode`` stores :math:`2r`."""

    if not isinstance(twice_mode, int):
        raise TypeError("twice_mode must be an integer")
    if state.sector == "NS" and twice_mode % 2 == 0:
        raise ValueError("NS fermion modes must be half-integral")
    if state.sector == "R" and twice_mode % 2:
        raise ValueError("Ramond fermion modes must be integral")

    if twice_mode == 0:
        if state.sector != "R":
            raise ValueError("the NS sector has no fermion zero mode")
        # Move f_0 through all nonzero creators before acting on u_+ or u_-.
        coefficient = ((-1) ** len(state.twice_modes)) / math.sqrt(2.0)
        return {
            AuxiliaryFermionState(
                "R", state.twice_modes, 1 - state.ground_parity
            ): coefficient
        }

    if twice_mode > 0:
        if twice_mode not in state.twice_modes:
            return {}
        position = state.twice_modes.index(twice_mode)
        modes = state.twice_modes[:position] + state.twice_modes[position + 1 :]
        return {
            AuxiliaryFermionState(state.sector, modes, state.ground_parity):
            (-1) ** position
        }

    creator = -twice_mode
    if creator in state.twice_modes:
        return {}
    position = sum(existing < creator for existing in state.twice_modes)
    modes = list(state.twice_modes)
    modes.insert(position, creator)
    return {
        AuxiliaryFermionState(
            state.sector, tuple(modes), state.ground_parity
        ): (-1) ** position
    }


def auxiliary_fermion_inner_product(
    left: AuxiliaryFermionState,
    right: AuxiliaryFermionState,
) -> complex:
    r"""Return the bilinear BPZ pairing with :math:`f_r^\dagger=-f_{-r}`."""

    if left.sector != right.sector:
        return 0.0j
    if left.twice_level != right.twice_level or left.parity != right.parity:
        return 0.0j
    working: dict[AuxiliaryFermionState, complex] = {right: 1.0 + 0.0j}
    for label in left.twice_modes:
        updated: dict[AuxiliaryFermionState, complex] = {}
        for state, coefficient in working.items():
            for output, action_coefficient in auxiliary_fermion_mode_action(
                label, state
            ).items():
                _add_auxiliary_state(
                    updated, output, -coefficient * action_coefficient
                )
        working = updated
        if not working:
            return 0.0j
    total = 0.0j
    for state, coefficient in working.items():
        if state.twice_modes or state.ground_parity != left.ground_parity:
            continue
        ground_norm = -1.0 if (
            state.sector == "R" and state.ground_parity == 1
        ) else 1.0
        total += coefficient * ground_norm
    return complex(total)


def _pfaffian_from_upper_triangle(
    entries: tuple[tuple[complex, ...], ...],
) -> complex:
    """Return the Pfaffian of a small antisymmetric contraction matrix."""

    size = len(entries)
    if any(len(row) != size for row in entries):
        raise ValueError("the Pfaffian input must be square")
    if size % 2:
        return 0.0j
    if size == 0:
        return 1.0 + 0.0j
    total = 0.0j
    for partner in range(1, size):
        remaining = tuple(
            index for index in range(1, size) if index != partner
        )
        minor = tuple(
            tuple(entries[row][column] for column in remaining)
            for row in remaining
        )
        total += (
            (-1) ** (partner + 1)
            * entries[0][partner]
            * _pfaffian_from_upper_triangle(minor)
        )
    return complex(total)


@lru_cache(maxsize=32768)
def auxiliary_ns_ns_ns_three_point(
    infinity_state: AuxiliaryFermionState,
    middle_state: AuxiliaryFermionState,
    zero_state: AuxiliaryFermionState,
) -> complex:
    r"""Return the ordered free-Majorana NS three-point form.

    The punctures are at ``(infinity,1,0)``.  A mode
    :math:`f_{-n-1/2}|0\rangle` is represented by
    :math:`\partial^n f/n!`.  Wick's theorem then reduces the answer to a
    finite Pfaffian.  Self-contractions within one normal-ordered state are
    omitted.  The infinity contractions include the BPZ convention
    :math:`f_r^\dagger=-f_{-r}` used by
    :func:`auxiliary_fermion_inner_product`.

    This formula is independent of the two-Virasoro decomposition.  In
    particular, setting the middle state to the vacuum reproduces the Fock
    inner product exactly and supplies a direct normalization check.
    """

    if any(
        state.sector != "NS"
        for state in (infinity_state, middle_state, zero_state)
    ):
        raise ValueError("all three auxiliary states must be NS states")

    # Entries are (puncture, derivative order).  Reversing the infinity
    # modes implements the order reversal in BPZ conjugation.
    fields = tuple(
        [("infinity", (label - 1) // 2) for label in reversed(infinity_state.twice_modes)]
        + [("middle", (label - 1) // 2) for label in middle_state.twice_modes]
        + [("zero", (label - 1) // 2) for label in zero_state.twice_modes]
    )
    if len(fields) % 2:
        return 0.0j

    def contraction(
        left: tuple[str, int],
        right: tuple[str, int],
    ) -> complex:
        left_group, left_order = left
        right_group, right_order = right
        if left_group == right_group:
            return 0.0j
        if left_group == "infinity":
            if right_group == "middle":
                return complex(
                    -math.comb(left_order, right_order)
                    if right_order <= left_order
                    else 0
                )
            if right_group == "zero":
                return complex(-1 if left_order == right_order else 0)
        if left_group == "middle" and right_group == "zero":
            return complex(
                (-1) ** left_order
                * math.comb(left_order + right_order, left_order)
            )
        raise AssertionError("the ordered puncture list is inconsistent")

    matrix = [[0.0j for _ in fields] for _ in fields]
    for row in range(len(fields)):
        for column in range(row + 1, len(fields)):
            value = contraction(fields[row], fields[column])
            matrix[row][column] = value
            matrix[column][row] = -value
    return _pfaffian_from_upper_triangle(
        tuple(tuple(row) for row in matrix)
    )


@lru_cache(maxsize=131072)
def auxiliary_r_ns_r_three_point(
    infinity_state: AuxiliaryFermionState,
    middle_state: AuxiliaryFermionState,
    zero_state: AuxiliaryFermionState,
    *,
    form_parity: int = 0,
    ramond_parity_twist: int = 1,
) -> complex:
    r"""Return the auxiliary R--NS--R form needed by the cut theta graph.

    The cut NS edge is needed only at levels zero and one half.  The NS
    vacuum inserts the identity.  The state :math:`f_{-1/2}|0\rangle`
    inserts :math:`f(1)=\sum_{n\in\mathbb Z}f_n`; level conservation leaves
    exactly one Ramond mode in the matrix element.
    """

    if form_parity not in (0, 1):
        raise ValueError("form_parity must be zero or one")
    if ramond_parity_twist not in (-1, 1):
        raise ValueError("ramond_parity_twist must be +1 or -1")
    if infinity_state.sector != "R" or zero_state.sector != "R":
        raise ValueError("the outer auxiliary states must be Ramond")
    if middle_state.sector != "NS":
        raise ValueError("the middle auxiliary state must be NS")
    def apply_form_intertwiner(
        state: AuxiliaryFermionState,
    ) -> Mapping[AuxiliaryFermionState, complex]:
        if form_parity == 0:
            return {
                state: complex(ramond_parity_twist**state.parity)
            }
        output = AuxiliaryFermionState(
            "R",
            state.twice_modes,
            1 - state.ground_parity,
        )
        return {
            output: (
                ramond_parity_twist**output.parity / math.sqrt(2.0)
            )
        }

    if middle_state.twice_modes == ():
        total = 0.0j
        for acted, coefficient in apply_form_intertwiner(zero_state).items():
            total += coefficient * auxiliary_fermion_inner_product(
                infinity_state, acted
            )
        return complex(total)
    if middle_state.twice_modes != (1,):
        raise NotImplementedError(
            "the exact cut-edge implementation currently needs only the "
            "NS vacuum and f_-1/2 vacuum"
        )
    twice_mode = zero_state.twice_level - infinity_state.twice_level
    total = 0.0j
    for middle_acted, middle_coefficient in auxiliary_fermion_mode_action(
        twice_mode, zero_state
    ).items():
        for acted, form_coefficient in apply_form_intertwiner(
            middle_acted
        ).items():
            total += (
                middle_coefficient
                * form_coefficient
                * auxiliary_fermion_inner_product(infinity_state, acted)
            )
    return complex(total)


def _generalized_binomial(alpha: float, order: int) -> float:
    """Return the generalized binomial coefficient for integer ``order``."""

    if order < 0:
        return 0.0
    value = 1.0
    for index in range(1, order + 1):
        value *= (float(alpha) - index + 1.0) / index
    return float(value)


def auxiliary_ns_r_r_three_point(
    infinity_state: AuxiliaryFermionState,
    middle_state: AuxiliaryFermionState,
    zero_state: AuxiliaryFermionState,
    *,
    form_parity: int = 0,
    _ground_tensor: Sequence[Sequence[complex]] | None = None,
) -> complex:
    r"""Return the auxiliary form in the PDF order ``(NS,R,R)``.

    The punctures are at ``(infinity,1,0)``.  The recursion follows by
    deforming contours of

    .. math::

       \frac{w^n f(w)}{\sqrt{w(w-1)}}.

    It is the weight-one-half analogue of the NS--R Ward identities used
    for the supercurrent.  In particular, it is not obtained by permuting
    the inputs of :func:`auxiliary_r_ns_r_three_point`; such a permutation
    changes descendant local coordinates.

    The terminal tensors below include the spin-field local-coordinate
    phases appropriate to the ordered punctures.  They are therefore not the
    bare BPZ metric in equation (2.5).  The odd terminal is obtained from the
    even terminal by acting with :math:`f_0` on the Ramond state at zero.
    Consistency of that zero-mode insertion with the square-root contour
    reverses the continuation phase between the even and odd forms.

    ``_ground_tensor`` is an internal audit hook used to test the finite Ward
    system with arbitrary terminal data; production callers leave it unset.
    """

    if infinity_state.sector != "NS":
        raise ValueError("the infinity auxiliary state must be NS")
    if middle_state.sector != "R" or zero_state.sector != "R":
        raise ValueError("the middle and zero auxiliary states must be Ramond")
    if form_parity not in (0, 1):
        raise ValueError("form_parity must be zero or one")
    if _ground_tensor is None:
        if form_parity == 0:
            terminal = ((1.0 + 0.0j, 0.0j), (0.0j, 1.0j))
        else:
            inverse_sqrt_two = 1.0 / math.sqrt(2.0)
            terminal = (
                (0.0j, inverse_sqrt_two + 0.0j),
                (1j * inverse_sqrt_two, 0.0j),
            )
    else:
        if len(_ground_tensor) != 2 or any(
            len(row) != 2 for row in _ground_tensor
        ):
            raise ValueError("_ground_tensor must be a two-by-two matrix")
        terminal = tuple(
            tuple(complex(entry) for entry in row) for row in _ground_tensor
        )
    contour_phase = -1j if form_parity == 0 else 1j

    def remove_leading(state: AuxiliaryFermionState) -> AuxiliaryFermionState:
        return AuxiliaryFermionState(
            state.sector,
            state.twice_modes[1:],
            state.ground_parity,
        )

    def summation_bound(
        *states: AuxiliaryFermionState,
        shift: int = 0,
    ) -> int:
        twice_level = sum(state.twice_level for state in states)
        return twice_level // 2 + abs(int(shift)) + 4

    @lru_cache(maxsize=None)
    def evaluate(
        infinity: AuxiliaryFermionState,
        middle: AuxiliaryFermionState,
        zero: AuxiliaryFermionState,
    ) -> complex:
        def add_mode_terms(
            value: complex,
            scale: complex,
            twice_mode: int,
            state: AuxiliaryFermionState,
            slot: Literal["infinity", "middle", "zero"],
            base_infinity: AuxiliaryFermionState,
            base_middle: AuxiliaryFermionState,
            base_zero: AuxiliaryFermionState,
        ) -> complex:
            for output, coefficient in auxiliary_fermion_mode_action(
                twice_mode,
                state,
            ).items():
                if slot == "infinity":
                    term = evaluate(output, base_middle, base_zero)
                elif slot == "middle":
                    term = evaluate(base_infinity, output, base_zero)
                else:
                    term = evaluate(base_infinity, base_middle, output)
                value += complex(scale) * coefficient * term
            return complex(value)

        if infinity.twice_modes:
            twice_r = infinity.twice_modes[0]
            rest = remove_leading(infinity)
            # r is half-integral and n=r+1/2 is integral.
            n = (twice_r + 1) // 2
            bound = summation_bound(rest, middle, zero, shift=n)
            sign = (-1) ** (rest.parity + zero.parity + 1)
            value = 0.0j
            for p in range(bound + 1):
                value = add_mode_terms(
                    value,
                    _generalized_binomial(n - 0.5, p),
                    2 * p,
                    middle,
                    "middle",
                    rest,
                    middle,
                    zero,
                )
                if p > 0:
                    value = add_mode_terms(
                        value,
                        -_generalized_binomial(-0.5, p) * (-1) ** p,
                        2 * p - twice_r,
                        rest,
                        "infinity",
                        rest,
                        middle,
                        zero,
                    )
                value = add_mode_terms(
                    value,
                        contour_phase
                    * sign
                    * _generalized_binomial(-0.5, p)
                    * (-1) ** p,
                    2 * (n + p),
                    zero,
                    "zero",
                    rest,
                    middle,
                    zero,
                )
            return complex(value)

        if zero.twice_modes:
            twice_r = zero.twice_modes[0]
            r = twice_r // 2
            rest = remove_leading(zero)
            bound = summation_bound(infinity, middle, rest, shift=r)
            sign = (-1) ** (infinity.parity + rest.parity + 1)
            lhs = 0.0j
            first_rhs = 0.0j
            remainder = 0.0j
            for p in range(bound + 1):
                lhs = add_mode_terms(
                    lhs,
                    _generalized_binomial(-r - 0.5, p),
                    2 * p,
                    middle,
                    "middle",
                    infinity,
                    middle,
                    rest,
                )
                first_rhs = add_mode_terms(
                    first_rhs,
                    _generalized_binomial(-0.5, p) * (-1) ** p,
                    2 * p + twice_r + 1,
                    infinity,
                    "infinity",
                    infinity,
                    middle,
                    rest,
                )
                if p > 0:
                    remainder = add_mode_terms(
                        remainder,
                        _generalized_binomial(-0.5, p) * (-1) ** p,
                        2 * p - twice_r,
                        rest,
                        "zero",
                        infinity,
                        middle,
                        rest,
                    )
            return complex(
                contour_phase * sign * (lhs - first_rhs)
                - remainder
            )

        if middle.twice_modes:
            twice_r = middle.twice_modes[0]
            r = twice_r // 2
            rest = remove_leading(middle)
            bound = summation_bound(infinity, rest, zero, shift=r)
            value = 0.0j
            sign = (-1) ** (infinity.parity + zero.parity + 1)
            for p in range(bound + 1):
                coefficient = (
                    _generalized_binomial(-0.5 - r, p) * (-1) ** p
                )
                value = add_mode_terms(
                    value,
                    coefficient,
                    2 * p + twice_r + 1,
                    infinity,
                    "infinity",
                    infinity,
                    rest,
                    zero,
                )
                value = add_mode_terms(
                    value,
                    -contour_phase
                    * sign
                    * coefficient
                    * (-1) ** r,
                    2 * p,
                    zero,
                    "zero",
                    infinity,
                    rest,
                    zero,
                )
                if p > 0:
                    value = add_mode_terms(
                        value,
                        -_generalized_binomial(-0.5, p),
                        2 * p - twice_r,
                        rest,
                        "middle",
                        infinity,
                        rest,
                        zero,
                    )
            return complex(value)

        return complex(terminal[middle.ground_parity][zero.ground_parity])

    value = evaluate(infinity_state, middle_state, zero_state)
    if (
        infinity_state.parity + middle_state.parity + zero_state.parity
    ) % 2 != form_parity:
        if abs(value) > 1.0e-12:
            raise ArithmeticError("the auxiliary Ward recursion violated parity")
        return 0.0j
    return complex(value)


@lru_cache(maxsize=None)
def _auxiliary_ramond_edge_data(
    twice_level: int,
    lift_sign: int,
) -> tuple[tuple[AuxiliaryFermionState, ...], np.ndarray]:
    """Return one Ramond Fock basis and its lifted inverse BPZ matrix."""

    if lift_sign not in (-1, 1):
        raise ValueError("lift_sign must be +1 or -1")
    basis = auxiliary_fermion_basis("R", twice_level)
    if not basis:
        return (), np.zeros((0, 0), dtype=np.complex128)
    gram = np.asarray(
        [
            [auxiliary_fermion_inner_product(left, right) for right in basis]
            for left in basis
        ],
        dtype=np.complex128,
    )
    inverse = np.linalg.inv(gram)
    parity_lift = np.asarray(
        [lift_sign**state.parity for state in basis],
        dtype=np.complex128,
    )
    kernel = parity_lift[:, np.newaxis] * inverse
    kernel.setflags(write=False)
    return basis, kernel


@dataclass(frozen=True)
class AuxiliaryNecklaceValue:
    """One cut-edge coefficient of the auxiliary-fermion theta block."""

    value: complex
    cut_state: AuxiliaryFermionState
    q_previous: complex
    q_current: complex
    maximum_twice_level: int
    previous_lift_sign: int
    current_lift_sign: int
    form_parity: int
    include_gathering_sign: bool
    left_parity_twist: int
    right_parity_twist: int


def auxiliary_ramond_theta_coefficients(
    *,
    cut_state: AuxiliaryFermionState,
    maximum_level_2: int,
    maximum_level_3: int,
    lift_sign_2: int = 1,
    lift_sign_3: int = 1,
    form_parity: int,
) -> dict[tuple[int, int], complex]:
    r"""Return the auxiliary block coefficients in the PDF theta convention.

    Keys are the integer excitation levels ``(N2,N3)`` on the two Ramond
    edges.  The NS cut state is either the vacuum or
    :math:`f_{-1/2}|0\rangle`.  Unlike
    :func:`auxiliary_ramond_necklace_value`, both oriented vertices retain
    the global edge order ``(NS,R2,R3)`` and the complete theta gathering
    sign is inserted once.  This is the auxiliary series
    :math:`\mathcal F_g^f` that appears in equations (6.5)--(6.7) of the
    supplied note.
    """

    if cut_state.sector != "NS":
        raise ValueError("cut_state must belong to the NS auxiliary module")
    if cut_state.twice_modes not in ((), (1,)):
        raise NotImplementedError(
            "only the NS vacuum and f_-1/2 vacuum are needed for PP and GG"
        )
    for value, name in (
        (maximum_level_2, "maximum_level_2"),
        (maximum_level_3, "maximum_level_3"),
    ):
        if not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    for value, name in (
        (lift_sign_2, "lift_sign_2"),
        (lift_sign_3, "lift_sign_3"),
    ):
        if value not in (-1, 1):
            raise ValueError(f"{name} must be +1 or -1")
    if form_parity not in (0, 1):
        raise ValueError("form_parity must be zero or one")
    cut_norm = auxiliary_fermion_inner_product(cut_state, cut_state)
    if cut_norm == 0:
        raise ArithmeticError("the auxiliary cut state has zero BPZ norm")

    coefficients: dict[tuple[int, int], complex] = {}
    cut_parity = cut_state.parity
    for level_2 in range(maximum_level_2 + 1):
        basis_2, kernel_2 = _auxiliary_ramond_edge_data(
            2 * level_2,
            lift_sign_2,
        )
        for level_3 in range(maximum_level_3 + 1):
            basis_3, kernel_3 = _auxiliary_ramond_edge_data(
                2 * level_3,
                lift_sign_3,
            )
            left_vertex = np.asarray(
                [
                    [
                        auxiliary_ns_r_r_three_point(
                            cut_state,
                            state_2,
                            state_3,
                            form_parity=form_parity,
                        )
                        for state_3 in basis_3
                    ]
                    for state_2 in basis_2
                ],
                dtype=np.complex128,
            )
            # The auxiliary form has no independent HJS structure label.
            # Its right-oriented restriction uses the same global edge
            # arguments as the left restriction; the transpose required by
            # the matrix contraction is supplied by the Einstein sum below.
            right_vertex = left_vertex
            gathering = np.asarray(
                [
                    [
                        (-1)
                        ** (
                            cut_parity * (state_2.parity + state_3.parity)
                            + state_2.parity * state_3.parity
                        )
                        for state_3 in basis_3
                    ]
                    for state_2 in basis_2
                ],
                dtype=np.complex128,
            )
            coefficient = np.einsum(
                "ab,ac,bd,cd->",
                gathering * left_vertex,
                kernel_2,
                kernel_3,
                right_vertex,
                optimize=True,
            )
            coefficients[(level_2, level_3)] = complex(
                coefficient / cut_norm
            )
    return coefficients


def auxiliary_ramond_necklace_coefficients(
    *,
    cut_state: AuxiliaryFermionState,
    maximum_previous_level: int,
    maximum_current_level: int,
    previous_lift_sign: int = 1,
    current_lift_sign: int = 1,
    form_parity: int = 0,
    include_gathering_sign: bool = False,
    left_parity_twist: int = 1,
    right_parity_twist: int = 1,
) -> dict[tuple[int, int], complex]:
    r"""Return auxiliary coefficients in the direct necklace convention.

    Keys are integer excitation levels ``(N_previous,N_current)``.  The two
    local vertices are ordered as ``(previous R,cut NS,current R)`` and
    ``(current R,cut NS,previous R)``, exactly as in the cyclic contraction
    used by :func:`auxiliary_ramond_necklace_value`.
    """

    if cut_state.sector != "NS":
        raise ValueError("cut_state must belong to the NS auxiliary module")
    if cut_state.twice_modes not in ((), (1,)):
        raise NotImplementedError(
            "only the NS vacuum and f_-1/2 vacuum are needed for PP and GG"
        )
    for value, name in (
        (maximum_previous_level, "maximum_previous_level"),
        (maximum_current_level, "maximum_current_level"),
    ):
        if not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    for value, name in (
        (previous_lift_sign, "previous_lift_sign"),
        (current_lift_sign, "current_lift_sign"),
        (left_parity_twist, "left_parity_twist"),
        (right_parity_twist, "right_parity_twist"),
    ):
        if value not in (-1, 1):
            raise ValueError(f"{name} must be +1 or -1")
    if form_parity not in (0, 1):
        raise ValueError("form_parity must be zero or one")

    cut_norm = auxiliary_fermion_inner_product(cut_state, cut_state)
    if cut_norm == 0:
        raise ArithmeticError("the auxiliary cut state has zero BPZ norm")

    coefficients: dict[tuple[int, int], complex] = {}
    for previous_level in range(maximum_previous_level + 1):
        previous_basis, previous_kernel = _auxiliary_ramond_edge_data(
            2 * previous_level,
            previous_lift_sign,
        )
        for current_level in range(maximum_current_level + 1):
            current_basis, current_kernel = _auxiliary_ramond_edge_data(
                2 * current_level,
                current_lift_sign,
            )
            left_vertex = np.asarray(
                [
                    [
                        auxiliary_r_ns_r_three_point(
                            previous_state,
                            cut_state,
                            current_state,
                            form_parity=form_parity,
                            ramond_parity_twist=left_parity_twist,
                        )
                        for current_state in current_basis
                    ]
                    for previous_state in previous_basis
                ],
                dtype=np.complex128,
            )
            right_vertex = np.asarray(
                [
                    [
                        auxiliary_r_ns_r_three_point(
                            current_state,
                            cut_state,
                            previous_state,
                            form_parity=form_parity,
                            ramond_parity_twist=right_parity_twist,
                        )
                        for previous_state in previous_basis
                    ]
                    for current_state in current_basis
                ],
                dtype=np.complex128,
            )
            if include_gathering_sign:
                cut_parity = cut_state.parity
                gathering = np.asarray(
                    [
                        [
                            (-1)
                            ** (
                                cut_parity
                                * (previous_state.parity + current_state.parity)
                                + previous_state.parity * current_state.parity
                            )
                            for current_state in current_basis
                        ]
                        for previous_state in previous_basis
                    ],
                    dtype=np.complex128,
                )
                left_vertex = gathering * left_vertex
            coefficient = np.trace(
                left_vertex
                @ current_kernel
                @ right_vertex
                @ previous_kernel
            )
            coefficients[(previous_level, current_level)] = complex(
                coefficient / cut_norm
            )
    return coefficients


def auxiliary_ramond_necklace_value(
    *,
    cut_state: AuxiliaryFermionState,
    q_previous: complex,
    q_current: complex,
    maximum_twice_level: int,
    previous_lift_sign: int = 1,
    current_lift_sign: int = 1,
    form_parity: int = 0,
    include_gathering_sign: bool = False,
    left_parity_twist: int = 1,
    right_parity_twist: int = 1,
) -> AuxiliaryNecklaceValue:
    r"""Evaluate one auxiliary cut-edge coefficient through a Ramond cutoff.

    The two local vertices are ordered as
    ``(previous R, cut NS, current R)`` and
    ``(current R, cut NS, previous R)``.  The returned value includes the
    inverse BPZ norm of ``cut_state`` because it is a coefficient of the
    *sewn* auxiliary theta block.  In particular,

    .. math::

       \langle f_{-1/2}0|f_{-1/2}0\rangle=-1,

    so the level-one-half coefficient carries a minus sign.  Omitting this
    inverse norm gives an incorrect deconvolution of the super block.
    """

    if form_parity not in (0, 1):
        raise ValueError("form_parity must be zero or one")
    if cut_state.sector != "NS":
        raise ValueError("cut_state must belong to the NS auxiliary module")
    if cut_state.twice_modes not in ((), (1,)):
        raise NotImplementedError(
            "only the NS vacuum and f_-1/2 vacuum are needed for PP and GG"
        )
    if not isinstance(maximum_twice_level, int) or maximum_twice_level < 0:
        raise ValueError("maximum_twice_level must be a nonnegative integer")
    maximum_twice_level -= maximum_twice_level % 2
    for value, name in (
        (previous_lift_sign, "previous_lift_sign"),
        (current_lift_sign, "current_lift_sign"),
        (left_parity_twist, "left_parity_twist"),
        (right_parity_twist, "right_parity_twist"),
    ):
        if value not in (-1, 1):
            raise ValueError(f"{name} must be +1 or -1")

    q_previous = complex(q_previous)
    q_current = complex(q_current)
    cut_norm = auxiliary_fermion_inner_product(cut_state, cut_state)
    if cut_norm == 0:
        raise ArithmeticError("the auxiliary cut state has zero BPZ norm")

    maximum_level = maximum_twice_level // 2
    coefficients = auxiliary_ramond_necklace_coefficients(
        cut_state=cut_state,
        maximum_previous_level=maximum_level,
        maximum_current_level=maximum_level,
        previous_lift_sign=previous_lift_sign,
        current_lift_sign=current_lift_sign,
        form_parity=form_parity,
        include_gathering_sign=include_gathering_sign,
        left_parity_twist=left_parity_twist,
        right_parity_twist=right_parity_twist,
    )
    total = sum(
        coefficient
        * q_previous**previous_level
        * q_current**current_level
        for (previous_level, current_level), coefficient in coefficients.items()
    )
    return AuxiliaryNecklaceValue(
        value=complex(total),
        cut_state=cut_state,
        q_previous=q_previous,
        q_current=q_current,
        maximum_twice_level=maximum_twice_level,
        previous_lift_sign=previous_lift_sign,
        current_lift_sign=current_lift_sign,
        form_parity=form_parity,
        include_gathering_sign=bool(include_gathering_sign),
        left_parity_twist=left_parity_twist,
        right_parity_twist=right_parity_twist,
    )


@dataclass(frozen=True)
class TensorBasisState:
    """One decomposable state in ``F tensor SVir``."""

    auxiliary: AuxiliaryFermionState
    super_state: SuperState

    @property
    def parity(self) -> int:
        if isinstance(self.super_state, PBWState):
            super_parity = self.super_state.parity
        else:
            super_parity = ns_fermion_parity(self.super_state)
        return self.auxiliary.parity ^ super_parity


@dataclass(frozen=True)
class VirasoroBranchParameters:
    """Central charges and weights of one two-Virasoro branch."""

    b_super: complex
    b_1: complex
    b_2: complex
    c_1: complex
    c_2: complex
    h_1: complex
    h_2: complex


def _consistent_embedding_roots(
    b: complex,
) -> tuple[complex, complex, complex, complex]:
    """Return ``(b1,b2,d1,d2)`` on a branch satisfying the note's identities."""

    b = complex(b)
    if b == 0 or abs(b * b - 1.0) == 0:
        raise ValueError("the embedding is defined at generic b, then continued")
    b_1 = cmath.sqrt(2.0 * b * b / (1.0 - b * b))
    inverse_b_2 = cmath.sqrt(2.0 / (b * b - 1.0))
    b_2 = 1.0 / inverse_b_2
    denominator_1 = cmath.sqrt(2.0 - 2.0 * b * b)
    # Fix the second square root by the first identity in equation (3.4).
    denominator_2 = -inverse_b_2 * denominator_1 / b_1
    return b_1, b_2, denominator_1, denominator_2


def virasoro_branch_parameters(
    *,
    b: complex,
    physical_momentum: complex,
    branch_number: BranchNumber,
) -> VirasoroBranchParameters:
    """Return the two ordinary Virasoro weights for one branch label."""

    n = Fraction(branch_number)
    b_1, b_2, denominator_1, denominator_2 = _consistent_embedding_roots(b)
    inverse_b_2 = 1.0 / b_2
    q_1 = b_1 + 1.0 / b_1
    q_2 = b_2 + inverse_b_2
    # The note's P is i times the real Liouville momentum used by this code.
    momentum = 1j * complex(physical_momentum)
    p_1 = momentum / denominator_1 + float(n) * b_1
    p_2 = momentum / denominator_2 + float(n) * inverse_b_2
    h_1 = q_1 * q_1 / 4.0 - p_1 * p_1
    h_2 = q_2 * q_2 / 4.0 - p_2 * p_2
    return VirasoroBranchParameters(
        b_super=complex(b),
        b_1=b_1,
        b_2=b_2,
        c_1=1.0 + 6.0 * q_1 * q_1,
        c_2=1.0 + 6.0 * q_2 * q_2,
        h_1=h_1,
        h_2=h_2,
    )


def _branch_twice_grade(sector: Sector, branch_number: BranchNumber) -> int:
    n = Fraction(branch_number)
    if sector == "NS":
        doubled = 2 * n
        if doubled.denominator != 1:
            raise ValueError("NS branch numbers lie in one-half integers")
        return int(doubled) ** 2
    shifted = n - Fraction(1, 4)
    doubled = 2 * shifted
    if doubled.denominator != 1:
        raise ValueError("R branch numbers lie in 1/2 Z + 1/4")
    k = int(doubled)
    return k * (k + 1)


def _super_twice_level(state: SuperState) -> int:
    return (
        state.twice_descendant_level
        if isinstance(state, PBWState)
        else ns_twice_level(state)
        if state
        else 0
    )


@lru_cache(maxsize=None)
def _tensor_basis(
    sector: Sector,
    total_twice_level: int,
    parity: int,
) -> tuple[TensorBasisState, ...]:
    out: list[TensorBasisState] = []
    for auxiliary_level in range(total_twice_level + 1):
        super_level = total_twice_level - auxiliary_level
        auxiliary_basis = auxiliary_fermion_basis(
            sector, auxiliary_level
        )
        super_basis: Sequence[SuperState]
        if sector == "NS":
            super_basis = tuple(ns_pbw_basis(super_level))
        else:
            if super_level % 2:
                continue
            super_basis = tuple(r_pbw_basis(super_level))
        for auxiliary in auxiliary_basis:
            for super_state in super_basis:
                state = TensorBasisState(auxiliary, super_state)
                if state.parity == parity:
                    out.append(state)
    return tuple(out)


def _super_mode_action(
    sector: Sector,
    twice_mode: int,
    state: SuperState,
    *,
    h: complex,
    c: complex,
) -> Mapping[SuperState, sp.Expr]:
    if sector == "NS":
        if isinstance(state, PBWState):
            raise TypeError("an NS tensor state cannot contain a Ramond state")
        return ns_act_mode(
            ns_G(Fraction(twice_mode, 2)),
            {tuple(state): sp.S.One},
            h=sp.sympify(h),
            c=sp.sympify(c),
        )
    if not isinstance(state, PBWState):
        raise TypeError("a Ramond tensor state must contain a PBWState")
    if twice_mode % 2:
        raise ValueError("Ramond supercurrent modes are integral")
    return r_act_mode(
        r_G(twice_mode // 2),
        {state: sp.S.One},
        h=sp.sympify(h),
        c=sp.sympify(c),
    )


@lru_cache(maxsize=None)
def _auxiliary_l_minus_one_action(
    state: AuxiliaryFermionState,
) -> Mapping[AuxiliaryFermionState, complex]:
    r"""Act with the auxiliary stress-tensor mode :math:`L^f_{-1}`."""

    if not state.twice_modes:
        if state.sector == "NS":
            return {}
        return {
            AuxiliaryFermionState(
                "R",
                (2,),
                1 - state.ground_parity,
            ): 1.0 / (2.0 * math.sqrt(2.0))
        }

    leading = state.twice_modes[0]
    rest = AuxiliaryFermionState(
        state.sector,
        state.twice_modes[1:],
        state.ground_parity,
    )
    result: dict[AuxiliaryFermionState, complex] = {}

    # L_-1 f_-r = f_-r L_-1 + (r+1/2) f_-(r+1).
    for rest_output, rest_coefficient in _auxiliary_l_minus_one_action(
        rest
    ).items():
        for output, coefficient in auxiliary_fermion_mode_action(
            -leading,
            rest_output,
        ).items():
            _add_auxiliary_state(
                result,
                output,
                rest_coefficient * coefficient,
            )
    commutator_coefficient = (leading + 1) / 2.0
    for output, coefficient in auxiliary_fermion_mode_action(
        -(leading + 2),
        rest,
    ).items():
        _add_auxiliary_state(
            result,
            output,
            commutator_coefficient * coefficient,
        )
    return result


def _add_tensor_state(
    result: dict[TensorBasisState, complex],
    state: TensorBasisState,
    coefficient: complex,
) -> None:
    """Accumulate one decomposable tensor state."""

    value = result.get(state, 0.0j) + complex(coefficient)
    if abs(value) <= 1.0e-15:
        result.pop(state, None)
    else:
        result[state] = value


def _embedded_l_minus_one_action(
    state: TensorBasisState,
    *,
    sector: Sector,
    copy: int,
    b: complex,
    h: complex,
    c: complex,
) -> Mapping[TensorBasisState, complex]:
    r"""Act with :math:`L_{-1}^{(1)}` or :math:`L_{-1}^{(2)}`.

    This finite action is used as an audit of the branch restrictions in
    equation (4.10) of the supplied PDF.  It implements equation (2.8)
    directly, including the graded sign in
    :math:`U_{-1}=\sum_r f_{-1-r}G_r`.
    """

    if copy not in (1, 2):
        raise ValueError("copy must be one or two")
    b = complex(b)
    if b == 0 or b * b == 1:
        raise ValueError("the embedded generators require generic b")
    if copy == 1:
        denominator = 1.0 / b - b
        a = (1.0 / b) / denominator
        d = -(1.0 / b + 2.0 * b) / denominator
        e = 1.0 / denominator
    else:
        denominator = b - 1.0 / b
        a = b / denominator
        d = -(b + 2.0 / b) / denominator
        e = 1.0 / denominator

    result: dict[TensorBasisState, complex] = {}
    super_state = state.super_state
    if sector == "NS":
        if isinstance(super_state, PBWState):
            raise TypeError("an NS tensor state cannot contain a Ramond state")
        super_outputs = ns_act_mode(
            ns_L(-1),
            {tuple(super_state): sp.S.One},
            h=sp.sympify(h),
            c=sp.sympify(c),
        )
    else:
        if not isinstance(super_state, PBWState):
            raise TypeError("a Ramond tensor state must contain an NS state")
        super_outputs = r_act_mode(
            r_L(-1),
            {super_state: sp.S.One},
            h=sp.sympify(h),
            c=sp.sympify(c),
        )
    for output, coefficient in super_outputs.items():
        _add_tensor_state(
            result,
            TensorBasisState(state.auxiliary, output),
            a * complex(sp.N(coefficient, 30)),
        )

    for auxiliary, coefficient in _auxiliary_l_minus_one_action(
        state.auxiliary
    ).items():
        _add_tensor_state(
            result,
            TensorBasisState(auxiliary, super_state),
            d * coefficient,
        )

    maximum_auxiliary_mode = max(state.auxiliary.twice_modes, default=0)
    minimum_twice_r = -2 - maximum_auxiliary_mode
    maximum_twice_r = _super_twice_level(super_state)
    parity = 1 if sector == "NS" else 0
    if minimum_twice_r % 2 != parity:
        minimum_twice_r += 1
    graded_sign = -1.0 if state.auxiliary.parity else 1.0
    for twice_r in range(
        minimum_twice_r,
        maximum_twice_r + 1,
        2,
    ):
        super_outputs = _super_mode_action(
            sector,
            twice_r,
            super_state,
            h=h,
            c=c,
        )
        if not super_outputs:
            continue
        auxiliary_outputs = auxiliary_fermion_mode_action(
            -2 - twice_r,
            state.auxiliary,
        )
        for auxiliary, auxiliary_coefficient in auxiliary_outputs.items():
            for output, super_coefficient in super_outputs.items():
                _add_tensor_state(
                    result,
                    TensorBasisState(auxiliary, output),
                    e
                    * graded_sign
                    * auxiliary_coefficient
                    * complex(sp.N(super_coefficient, 30)),
                )
    return result


def _u_zero_matrix(
    basis: tuple[TensorBasisState, ...],
    *,
    sector: Sector,
    h: complex,
    c: complex,
) -> np.ndarray:
    index = {state: position for position, state in enumerate(basis)}
    matrix = np.zeros((len(basis), len(basis)), dtype=np.complex128)
    for column, tensor_state in enumerate(basis):
        auxiliary = tensor_state.auxiliary
        super_state = tensor_state.super_state
        super_level = _super_twice_level(super_state)
        if sector == "NS":
            mode_values = tuple(
                value
                for value in range(-auxiliary.twice_level, super_level + 1)
                if value % 2
            )
        else:
            mode_values = tuple(
                value
                for value in range(-auxiliary.twice_level, super_level + 1, 2)
            )
        graded_sign = -1.0 if auxiliary.parity else 1.0
        for twice_mode in mode_values:
            super_outputs = _super_mode_action(
                sector,
                twice_mode,
                super_state,
                h=h,
                c=c,
            )
            if not super_outputs:
                continue
            auxiliary_outputs = auxiliary_fermion_mode_action(
                -twice_mode, auxiliary
            )
            for auxiliary_output, auxiliary_coefficient in auxiliary_outputs.items():
                for super_output, super_coefficient in super_outputs.items():
                    output = TensorBasisState(auxiliary_output, super_output)
                    row = index.get(output)
                    if row is None:
                        raise AssertionError(
                            "U_0 left the fixed-level, fixed-parity tensor basis"
                        )
                    matrix[row, column] += (
                        graded_sign
                        * auxiliary_coefficient
                        * complex(sp.N(super_coefficient, 30))
                    )
    return matrix


def _tensor_gram_matrix(
    basis: tuple[TensorBasisState, ...],
    *,
    sector: Sector,
    h: complex,
    c: complex,
) -> np.ndarray:
    matrix = np.zeros((len(basis), len(basis)), dtype=np.complex128)
    for row, left in enumerate(basis):
        for column, right in enumerate(basis):
            auxiliary = auxiliary_fermion_inner_product(
                left.auxiliary, right.auxiliary
            )
            if auxiliary == 0:
                continue
            if sector == "NS":
                if isinstance(left.super_state, PBWState) or isinstance(
                    right.super_state, PBWState
                ):
                    raise TypeError("NS tensor basis contains a Ramond state")
                super_pairing = ns_inner_product(
                    left.super_state,
                    right.super_state,
                    h=sp.sympify(h),
                    c=sp.sympify(c),
                )
            else:
                if not isinstance(left.super_state, PBWState) or not isinstance(
                    right.super_state, PBWState
                ):
                    raise TypeError("R tensor basis contains an NS state")
                super_pairing = r_inner_product(
                    left.super_state,
                    right.super_state,
                    h=sp.sympify(h),
                    c=sp.sympify(c),
                )
            matrix[row, column] = auxiliary * complex(sp.N(super_pairing, 30))
    return matrix


@dataclass(frozen=True)
class EmbeddedBranchState:
    """One finite simultaneous two-Virasoro highest state."""

    sector: Sector
    physical_momentum: complex
    branch_number: BranchNumber
    parity: int
    parameters: VirasoroBranchParameters
    super_central_charge: complex
    super_weight: complex
    basis: tuple[TensorBasisState, ...]
    coefficients: tuple[complex, ...]
    norm: complex
    eigen_residual: float
    spectral_gap: float


@lru_cache(maxsize=8192)
def embedded_branch_state(
    *,
    b: complex,
    sector: Sector,
    physical_momentum: complex,
    branch_number: BranchNumber,
    parity: int | None = None,
    isolation_tolerance: float = 1.0e-8,
) -> EmbeddedBranchState:
    r"""Construct one branch highest state at generic :math:`b`.

    The SVD selects the null vector of
    :math:`L^{(1)}_0-h^{(1)}_n`.  ``spectral_gap`` is the ratio of the next
    singular value to the matrix scale; it certifies that the selected
    branch is one-dimensional at the sampled generic parameters.
    """

    if sector not in ("NS", "R"):
        raise ValueError("sector must be 'NS' or 'R'")
    n = Fraction(branch_number)
    twice_grade = _branch_twice_grade(sector, n)
    if sector == "NS":
        expected_parity = int(2 * n) % 2
        if parity is not None and parity != expected_parity:
            raise ValueError("NS branch parity is fixed by 2n modulo two")
        branch_parity = expected_parity
    else:
        if parity not in (0, 1):
            raise ValueError("a Ramond branch requires parity zero or one")
        branch_parity = int(parity)

    b = complex(b)
    momentum = complex(physical_momentum)
    c = super_liouville_central_charge(b)
    q_background = b + 1.0 / b
    if sector == "NS":
        h = ns_liouville_weight(momentum, b)
        auxiliary_ground_weight = 0.0
    else:
        h = c / 24.0 + momentum * momentum / 2.0
        auxiliary_ground_weight = 1.0 / 16.0
    parameters = virasoro_branch_parameters(
        b=b,
        physical_momentum=momentum,
        branch_number=n,
    )
    basis = _tensor_basis(sector, twice_grade, branch_parity)
    if not basis:
        raise AssertionError("the branch-onset tensor basis is empty")
    u_zero = _u_zero_matrix(basis, sector=sector, h=h, c=c)
    denominator = 1.0 / b - b
    a_1 = (1.0 / b) / denominator
    d_1 = -(1.0 / b + 2.0 * b) / denominator
    e_1 = 1.0 / denominator
    l_one = e_1 * u_zero
    for diagonal, state in enumerate(basis):
        super_level = _super_twice_level(state.super_state) / 2.0
        auxiliary_level = state.auxiliary.twice_level / 2.0
        l_one[diagonal, diagonal] += (
            a_1 * (h + super_level)
            + d_1 * (auxiliary_ground_weight + auxiliary_level)
        )

    shifted = l_one - parameters.h_1 * np.eye(len(basis), dtype=np.complex128)
    _, singular_values, right_vectors = np.linalg.svd(shifted)
    vector = right_vectors[-1, :].conjugate()
    scale = max(1.0, float(np.linalg.norm(l_one, ord=2)), abs(parameters.h_1))
    residual = float(np.linalg.norm(shifted @ vector) / (scale * np.linalg.norm(vector)))
    if len(singular_values) == 1:
        gap = 1.0
    else:
        gap = float(singular_values[-2] / scale)
    if residual > isolation_tolerance or gap <= isolation_tolerance:
        raise ArithmeticError(
            "failed to isolate the requested two-Virasoro branch: "
            f"residual={residual:.3e}, gap={gap:.3e}, sector={sector}, n={n}"
        )

    gram = _tensor_gram_matrix(basis, sector=sector, h=h, c=c)
    norm = complex(vector.T @ gram @ vector)
    if abs(norm) <= isolation_tolerance:
        raise ArithmeticError("the selected branch state has vanishing BPZ norm")
    return EmbeddedBranchState(
        sector=sector,
        physical_momentum=momentum,
        branch_number=n,
        parity=branch_parity,
        parameters=parameters,
        super_central_charge=c,
        super_weight=h,
        basis=basis,
        coefficients=tuple(complex(value) for value in vector),
        norm=norm,
        eigen_residual=residual,
        spectral_gap=gap,
    )


def _super_state_parity(state: SuperState) -> int:
    return state.parity if isinstance(state, PBWState) else ns_fermion_parity(state)


@lru_cache(maxsize=32768)
def _cached_hjs_polynomial_ground_tensor(
    beta_infinity: complex,
    beta_zero: complex,
    structure_sign: int,
) -> tuple[complex, complex, complex, complex]:
    """Return one numerical HJS tensor reused across all branch states."""

    tensor = hjs_to_polynomial_ground_tensor(
        sp.sympify(beta_infinity),
        sp.sympify(beta_zero),
        structure_sign=structure_sign,
    )
    return tuple(
        complex(sp.N(tensor[row, column], 30))
        for row in range(2)
        for column in range(2)
    )


@lru_cache(maxsize=None)
def _rr_three_point_numeric_template(
    infinity_state: PBWState,
    middle_word: NSWord,
    zero_state: PBWState,
) -> Callable[..., object]:
    """Compile the exact R-NS-R Ward polynomial for one state triple."""

    expression = rr_three_point_from_ground_tensor(
        infinity_state,
        middle_word,
        zero_state,
        h_infinity=_RR_TEMPLATE_H_INFINITY,
        h_ns=_RR_TEMPLATE_H_NS,
        h_zero=_RR_TEMPLATE_H_ZERO,
        c=_RR_TEMPLATE_C,
        ground_tensor=_RR_TEMPLATE_GROUND_MATRIX,
        simplify=False,
    )
    return sp.lambdify(
        (
            _RR_TEMPLATE_H_INFINITY,
            _RR_TEMPLATE_H_NS,
            _RR_TEMPLATE_H_ZERO,
            _RR_TEMPLATE_C,
            *_RR_TEMPLATE_GROUND,
        ),
        expression,
        modules="numpy",
        # Global CSE is counterproductive for the large level-three Ward
        # polynomials: compiling one template can take longer than the entire
        # subsequent numerical contraction.  The unsimplified expression is
        # still exact, and numerical values are cached separately below.
        cse=False,
        docstring_limit=0,
    )


@lru_cache(maxsize=262144)
def _rr_three_point_numeric(
    infinity_state: PBWState,
    middle_word: NSWord,
    zero_state: PBWState,
    *,
    h_infinity: complex,
    h_ns: complex,
    h_zero: complex,
    c: complex,
    ground_tensor: tuple[complex, complex, complex, complex],
) -> complex:
    """Evaluate one cached exact Ward polynomial numerically."""

    evaluator = _rr_three_point_numeric_template(
        infinity_state,
        middle_word,
        zero_state,
    )
    return complex(
        evaluator(
            h_infinity,
            h_ns,
            h_zero,
            c,
            *ground_tensor,
        )
    )


@lru_cache(maxsize=65536)
def _elementary_product_form(
    infinity: TensorBasisState,
    middle: TensorBasisState,
    zero: TensorBasisState,
    *,
    orientation: Literal["left", "right"],
    h_infinity: complex,
    h_middle: complex,
    h_zero: complex,
    c: complex,
    beta_infinity: complex,
    beta_zero: complex,
    structure_sign: int,
    super_form_parity: int | None,
    auxiliary_form_parity: int,
    auxiliary_parity_twist: int,
) -> complex:
    r"""Evaluate one product-theory form in local ``(R,NS,R)`` order.

    The state-gathering signs include the parities of the two homogeneous
    intertwining forms.  Both signs are fixed by the embedded Virasoro Ward
    identities.  Their remaining relative sign is fixed by the BPZ-oriented
    cyclic sewing and is independently checked against direct inverse-Gram
    ``PP`` and ``GG`` coefficients.
    """

    if not isinstance(infinity.super_state, PBWState):
        raise TypeError("the infinity super-state must be Ramond")
    if isinstance(middle.super_state, PBWState):
        raise TypeError("the middle super-state must be NS")
    if not isinstance(zero.super_state, PBWState):
        raise TypeError("the zero super-state must be Ramond")
    if auxiliary_form_parity not in (0, 1):
        raise ValueError("auxiliary_form_parity must be zero or one")
    mu = (
        infinity.auxiliary.parity,
        middle.auxiliary.parity,
        zero.auxiliary.parity,
    )
    nu = (
        _super_state_parity(infinity.super_state),
        _super_state_parity(middle.super_state),
        _super_state_parity(zero.super_state),
    )
    if sum(mu) % 2 != auxiliary_form_parity:
        return 0.0j
    state_super_form_parity = sum(nu) % 2
    if super_form_parity is not None:
        if super_form_parity not in (0, 1):
            raise ValueError("super_form_parity must be zero, one, or None")
        if state_super_form_parity != super_form_parity:
            return 0.0j
    auxiliary = auxiliary_r_ns_r_three_point(
        infinity.auxiliary,
        middle.auxiliary,
        zero.auxiliary,
        form_parity=auxiliary_form_parity,
        ramond_parity_twist=auxiliary_parity_twist,
    )
    if auxiliary == 0:
        return 0.0j
    ground_tensor = _cached_hjs_polynomial_ground_tensor(
        complex(beta_infinity),
        complex(beta_zero),
        int(structure_sign),
    )
    super_value = _rr_three_point_numeric(
        infinity.super_state,
        middle.super_state,
        zero.super_state,
        h_infinity=h_infinity,
        h_ns=h_middle,
        h_zero=h_zero,
        c=c,
        ground_tensor=ground_tensor,
    )
    form_parity_difference = (
        state_super_form_parity + auxiliary_form_parity
    )
    corrected_left = (
        mu[0] * nu[1]
        + form_parity_difference * (mu[0] + mu[1])
    )
    if orientation == "left":
        exponent = corrected_left
    elif orientation == "right":
        exponent = (
            mu[0] * nu[1]
            + form_parity_difference * (mu[0] + nu[1])
        )
    else:
        raise ValueError("orientation must be 'left' or 'right'")
    return complex(((-1) ** exponent) * auxiliary * super_value)


def oriented_branching_coefficient(
    infinity: EmbeddedBranchState,
    middle: EmbeddedBranchState,
    zero: EmbeddedBranchState,
    *,
    orientation: Literal["left", "right"],
    structure_sign: int,
    super_form_parity: int | None = None,
    auxiliary_form_parity: int = 0,
    auxiliary_parity_twist: int = 1,
) -> complex:
    """Evaluate one unnormalized oriented mixed branching coefficient."""

    if (infinity.sector, middle.sector, zero.sector) != ("R", "NS", "R"):
        raise ValueError("the implemented local order is (R,NS,R)")
    if structure_sign not in (-1, 1):
        raise ValueError("structure_sign must be +1 or -1")
    c = infinity.super_central_charge
    if max(
        abs(middle.super_central_charge - c),
        abs(zero.super_central_charge - c),
    ) > 1.0e-10 * max(1.0, abs(c)):
        raise ValueError("all three branches must use the same central charge")
    beta_infinity = 1j * infinity.physical_momentum / math.sqrt(2.0)
    beta_zero = 1j * zero.physical_momentum / math.sqrt(2.0)
    total = 0.0j
    for coefficient_infinity, state_infinity in zip(
        infinity.coefficients, infinity.basis
    ):
        for coefficient_middle, state_middle in zip(
            middle.coefficients, middle.basis
        ):
            for coefficient_zero, state_zero in zip(
                zero.coefficients, zero.basis
            ):
                elementary = _elementary_product_form(
                    state_infinity,
                    state_middle,
                    state_zero,
                    orientation=orientation,
                    h_infinity=infinity.super_weight,
                    h_middle=middle.super_weight,
                    h_zero=zero.super_weight,
                    c=c,
                    beta_infinity=beta_infinity,
                    beta_zero=beta_zero,
                    structure_sign=structure_sign,
                    super_form_parity=super_form_parity,
                    auxiliary_form_parity=auxiliary_form_parity,
                    auxiliary_parity_twist=auxiliary_parity_twist,
                )
                total += (
                    coefficient_infinity
                    * coefficient_middle
                    * coefficient_zero
                    * elementary
                )
    return complex(total)


@dataclass(frozen=True)
class BranchingProduct:
    """Phase-independent product of the two oriented branch coefficients."""

    value: complex
    left: complex
    right: complex
    norm_product: complex
    max_eigen_residual: float
    min_spectral_gap: float


def branching_coefficient_product(
    infinity: EmbeddedBranchState,
    middle: EmbeddedBranchState,
    zero: EmbeddedBranchState,
    *,
    left_structure_sign: int,
    right_structure_sign: int,
    super_form_parity: int | None = None,
    auxiliary_form_parity: int = 0,
    left_auxiliary_parity_twist: int = 1,
    right_auxiliary_parity_twist: int = 1,
) -> BranchingProduct:
    r"""Return :math:`B_LB_R/(\|v_1\|^2\|v_2\|^2\|v_3\|^2)`."""

    left = oriented_branching_coefficient(
        infinity,
        middle,
        zero,
        orientation="left",
        structure_sign=left_structure_sign,
        super_form_parity=super_form_parity,
        auxiliary_form_parity=auxiliary_form_parity,
        auxiliary_parity_twist=left_auxiliary_parity_twist,
    )
    right = oriented_branching_coefficient(
        infinity,
        middle,
        zero,
        orientation="right",
        structure_sign=right_structure_sign,
        super_form_parity=super_form_parity,
        auxiliary_form_parity=auxiliary_form_parity,
        auxiliary_parity_twist=right_auxiliary_parity_twist,
    )
    norm_product = infinity.norm * middle.norm * zero.norm
    if norm_product == 0:
        raise ArithmeticError("a branch norm vanished")
    return BranchingProduct(
        value=left * right / norm_product,
        left=left,
        right=right,
        norm_product=norm_product,
        max_eigen_residual=max(
            infinity.eigen_residual,
            middle.eigen_residual,
            zero.eigen_residual,
        ),
        min_spectral_gap=min(
            infinity.spectral_gap,
            middle.spectral_gap,
            zero.spectral_gap,
        ),
    )


@lru_cache(maxsize=32768)
def _theta_elementary_product_form(
    ns_edge: TensorBasisState,
    ramond_edge_2: TensorBasisState,
    ramond_edge_3: TensorBasisState,
    *,
    orientation: Literal["left", "right"],
    h_ns: complex,
    h_ramond_2: complex,
    h_ramond_3: complex,
    c: complex,
    beta_2: complex,
    beta_3: complex,
    structure_sign: int,
    super_form_parity: int,
    auxiliary_form_parity: int,
) -> complex:
    r"""Evaluate one product-form entry in the PDF's theta-edge order.

    All three constituent forms use the literal local order
    ``(NS at infinity,R2 at 1,R3 at 0)``; no descendant state is silently
    transported through a Mobius permutation.

    The signs are the form-parity-aware correction of equations (4.4) and
    (4.5) in the supplied note.  The displayed equations there reorder only
    the states and omit the parity of the homogeneous intertwining forms.
    The corrected left sign is fixed independently by requiring the
    restriction to obey both embedded Virasoro Ward identities.  The right
    sign is its complementary orientation, fixed so that the complete theta
    sewing sign still factorizes into the separate super-Virasoro and
    auxiliary-fermion sewing signs.
    """

    if not isinstance(ns_edge.super_state, tuple):
        raise TypeError("the first theta edge must contain an NS state")
    if not isinstance(ramond_edge_2.super_state, PBWState):
        raise TypeError("theta edge two must contain a Ramond state")
    if not isinstance(ramond_edge_3.super_state, PBWState):
        raise TypeError("theta edge three must contain a Ramond state")
    if super_form_parity not in (0, 1):
        raise ValueError("super_form_parity must be zero or one")
    if auxiliary_form_parity not in (0, 1):
        raise ValueError("auxiliary_form_parity must be zero or one")
    if structure_sign not in (-1, 1):
        raise ValueError("structure_sign must be +1 or -1")

    auxiliary_parities = (
        ns_edge.auxiliary.parity,
        ramond_edge_2.auxiliary.parity,
        ramond_edge_3.auxiliary.parity,
    )
    super_parities = (
        _super_state_parity(ns_edge.super_state),
        _super_state_parity(ramond_edge_2.super_state),
        _super_state_parity(ramond_edge_3.super_state),
    )
    if sum(auxiliary_parities) % 2 != auxiliary_form_parity:
        return 0.0j
    if sum(super_parities) % 2 != super_form_parity:
        return 0.0j

    auxiliary = auxiliary_ns_r_r_three_point(
        ns_edge.auxiliary,
        ramond_edge_2.auxiliary,
        ramond_edge_3.auxiliary,
        form_parity=auxiliary_form_parity,
    )
    if auxiliary == 0:
        return 0.0j

    ground_tensor = hjs_ns_rr_polynomial_ground_tensor(
        sp.sympify(beta_2),
        sp.sympify(beta_3),
        structure_sign=structure_sign,
    )
    super_value = ns_rr_three_point_from_ground_tensor(
        ns_edge.super_state,
        ramond_edge_2.super_state,
        ramond_edge_3.super_state,
        h_infinity=sp.sympify(h_ns),
        h_middle=sp.sympify(h_ramond_2),
        h_zero=sp.sympify(h_ramond_3),
        c=sp.sympify(c),
        ground_tensor=ground_tensor,
    )
    mu_1, mu_2, mu_3 = auxiliary_parities
    nu_1, nu_2, nu_3 = super_parities
    left_exponent = (
        (super_form_parity + nu_2) * mu_3
        + (1 - auxiliary_form_parity) * mu_1
    )
    if orientation == "left":
        exponent = left_exponent
    elif orientation == "right":
        note_left = nu_1 * (mu_2 + mu_3) + nu_2 * mu_3
        note_right = mu_1 * (nu_2 + nu_3) + mu_2 * nu_3
        exponent = note_left + note_right + left_exponent
    else:
        raise ValueError("orientation must be 'left' or 'right'")
    return complex(
        (-1) ** exponent
        * auxiliary
        * complex(sp.N(super_value, 30))
    )


def theta_branching_coefficient_product(
    ns_edge: EmbeddedBranchState,
    ramond_edge_2: EmbeddedBranchState,
    ramond_edge_3: EmbeddedBranchState,
    *,
    left_structure_sign: int,
    right_structure_sign: int,
    super_form_parity: int,
    auxiliary_form_parity: int,
) -> BranchingProduct:
    r"""Return the normalized branch product in equation (6.5).

    Arguments follow the global theta-edge order ``(NS,R2,R3)`` rather than
    the geometric order of the local Ward evaluator.  The explicit theta
    sewing sign

    .. math::

       (-1)^{(2n_1)\epsilon_2+(2n_1)\epsilon_3+
                    \epsilon_2\epsilon_3}

    is deliberately *not* included; the caller must insert it once, together
    with the three spin lifts, as displayed in equation (6.5).
    """

    if (ns_edge.sector, ramond_edge_2.sector, ramond_edge_3.sector) != (
        "NS",
        "R",
        "R",
    ):
        raise ValueError("theta branches must be ordered as (NS,R,R)")
    for value, name in (
        (left_structure_sign, "left_structure_sign"),
        (right_structure_sign, "right_structure_sign"),
    ):
        if value not in (-1, 1):
            raise ValueError(f"{name} must be +1 or -1")
    for value, name in (
        (super_form_parity, "super_form_parity"),
        (auxiliary_form_parity, "auxiliary_form_parity"),
    ):
        if value not in (0, 1):
            raise ValueError(f"{name} must be zero or one")
    c = ns_edge.super_central_charge
    if max(
        abs(ramond_edge_2.super_central_charge - c),
        abs(ramond_edge_3.super_central_charge - c),
    ) > 1.0e-10 * max(1.0, abs(c)):
        raise ValueError("all three branches must use the same central charge")

    beta_2 = 1j * ramond_edge_2.physical_momentum / math.sqrt(2.0)
    beta_3 = 1j * ramond_edge_3.physical_momentum / math.sqrt(2.0)

    def oriented(orientation: Literal["left", "right"], sign: int) -> complex:
        total = 0.0j
        for coefficient_1, state_1 in zip(ns_edge.coefficients, ns_edge.basis):
            for coefficient_2, state_2 in zip(
                ramond_edge_2.coefficients,
                ramond_edge_2.basis,
            ):
                for coefficient_3, state_3 in zip(
                    ramond_edge_3.coefficients,
                    ramond_edge_3.basis,
                ):
                    total += (
                        coefficient_1
                        * coefficient_2
                        * coefficient_3
                        * _theta_elementary_product_form(
                            state_1,
                            state_2,
                            state_3,
                            orientation=orientation,
                            h_ns=ns_edge.super_weight,
                            h_ramond_2=ramond_edge_2.super_weight,
                            h_ramond_3=ramond_edge_3.super_weight,
                            c=c,
                            beta_2=beta_2,
                            beta_3=beta_3,
                            structure_sign=sign,
                            super_form_parity=super_form_parity,
                            auxiliary_form_parity=auxiliary_form_parity,
                        )
                    )
        return complex(total)

    left = oriented("left", left_structure_sign)
    right = oriented("right", right_structure_sign)
    norm_product = ns_edge.norm * ramond_edge_2.norm * ramond_edge_3.norm
    if norm_product == 0:
        raise ArithmeticError("a branch norm vanished")
    return BranchingProduct(
        value=left * right / norm_product,
        left=left,
        right=right,
        norm_product=norm_product,
        max_eigen_residual=max(
            ns_edge.eigen_residual,
            ramond_edge_2.eigen_residual,
            ramond_edge_3.eigen_residual,
        ),
        min_spectral_gap=min(
            ns_edge.spectral_gap,
            ramond_edge_2.spectral_gap,
            ramond_edge_3.spectral_gap,
        ),
    )


def necklace_branching_coefficient_product(
    previous: EmbeddedBranchState,
    middle: EmbeddedBranchState,
    current: EmbeddedBranchState,
    *,
    left_structure_sign: int,
    right_structure_sign: int,
    super_form_parity: int | None = None,
    auxiliary_form_parity: int = 0,
    include_gathering_sign: bool = False,
    left_auxiliary_parity_twist: int = 1,
    right_auxiliary_parity_twist: int = 1,
) -> BranchingProduct:
    r"""Return the branch product in the direct necklace convention.

    The direct two-point oracle orders each local vertex as
    ``(previous R, external NS, current R)``.  At the second vertex the two
    Ramond edges therefore exchange their infinity and zero slots.  This is
    different from writing both theta-graph vertices in one fixed global edge
    order, and the distinction is phase-sensitive in the Ramond ground fiber.

    The cyclic trace in :mod:`spin23_genus1_blocks` does not insert the
    additional three-edge gathering sign used in equation (6.5) of the
    supplied two-Virasoro note.  Consequently that sign must *not* be added to
    the value returned here.  This convention map is verified by the complete
    Ramond ground-fiber identity for all four choices of the two HJS signs.
    """

    if (previous.sector, middle.sector, current.sector) != ("R", "NS", "R"):
        raise ValueError("the necklace branch order must be (R,NS,R)")
    left = oriented_branching_coefficient(
        previous,
        middle,
        current,
        orientation="left",
        structure_sign=left_structure_sign,
        super_form_parity=super_form_parity,
        auxiliary_form_parity=auxiliary_form_parity,
        auxiliary_parity_twist=left_auxiliary_parity_twist,
    )
    right = oriented_branching_coefficient(
        current,
        middle,
        previous,
        orientation="right",
        structure_sign=right_structure_sign,
        super_form_parity=super_form_parity,
        auxiliary_form_parity=auxiliary_form_parity,
        auxiliary_parity_twist=right_auxiliary_parity_twist,
    )
    norm_product = previous.norm * middle.norm * current.norm
    if norm_product == 0:
        raise ArithmeticError("a branch norm vanished")
    gathering_sign = (
        (-1)
        ** (
            previous.parity * middle.parity
            + previous.parity * current.parity
            + middle.parity * current.parity
        )
        if include_gathering_sign
        else 1
    )
    return BranchingProduct(
        value=gathering_sign * left * right / norm_product,
        left=left,
        right=right,
        norm_product=norm_product,
        max_eigen_residual=max(
            previous.eigen_residual,
            middle.eigen_residual,
            current.eigen_residual,
        ),
        min_spectral_gap=min(
            previous.spectral_gap,
            middle.spectral_gap,
            current.spectral_gap,
        ),
    )


__all__ = [
    "AuxiliaryFermionState",
    "AuxiliaryNecklaceValue",
    "BranchingProduct",
    "EmbeddedBranchState",
    "TensorBasisState",
    "VirasoroBranchParameters",
    "auxiliary_fermion_basis",
    "auxiliary_fermion_inner_product",
    "auxiliary_fermion_mode_action",
    "auxiliary_ns_ns_ns_three_point",
    "auxiliary_r_ns_r_three_point",
    "auxiliary_ramond_necklace_coefficients",
    "auxiliary_ramond_necklace_value",
    "auxiliary_ramond_theta_coefficients",
    "branching_coefficient_product",
    "embedded_branch_state",
    "necklace_branching_coefficient_product",
    "oriented_branching_coefficient",
    "theta_branching_coefficient_product",
    "virasoro_branch_parameters",
]
