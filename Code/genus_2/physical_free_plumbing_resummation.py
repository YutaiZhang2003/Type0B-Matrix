#!/usr/bin/env python3
r"""Direct plumbing resummation of the physical NS free superfield.

The physical fermion in this file is the Majorana field belonging to the
free ``X+psi`` denominator of ``Q_L``.  It is not the auxiliary Majorana used
in the double-Virasoro branching construction.

For the theta graph, the analytic sphere generating coefficient of NS
fermion Fock states is a Pfaffian of a three-slot contraction kernel ``K``.
The BPZ bra reversal differs by a state-size sign, which cancels between the
two theta trinions.  Sewing therefore turns the squared Pfaffian into
principal minors of ``K``.  Their all-level sum is a Fredholm determinant.
The literal Human-Note orientation sign is imposed by a four-determinant
parity filter,

    F_psi^Theta = (-D+++ + D-++ + D+-+ + D++-) / 2,

where ``D_sigma = det(1 + K W_sigma)`` and
``x_e = eta_e sqrt(q_e)``.  This construction uses only plumbing data and
the analytic pants kernel: it does not evaluate a period matrix or a Riemann
theta constant.

The free-boson oscillator is independently resummed by the plumbing
Schottky/Heisenberg primitive product.  The returned superfield oscillator
is

    |P_X^Theta F_psi^Theta|^2.

The noncompact two-loop momentum Gaussian is deliberately kept separate.  It
must be derived from the physical free-boson plumbing momentum kernel before
forming the complete noncompact partition function.
"""

from __future__ import annotations

import argparse
import cmath
from dataclasses import dataclass
import math
from math import comb
from pathlib import Path
import sys
from typing import Sequence

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
CODE_ROOT = SCRIPT_DIR.parent
PLUMBING_DIR = CODE_ROOT / "genus_2_cross_channel"
C_RECURSION_DIR = CODE_ROOT / "c_Recursion"
for dependency in (CODE_ROOT, PLUMBING_DIR, C_RECURSION_DIR):
    if str(dependency) not in sys.path:
        sys.path.insert(0, str(dependency))

from free_boson_plumbing import (  # noqa: E402
    glasses_free_boson_product,
    theta_free_boson_product,
)
from free_majorana_pair_of_pants import (  # noqa: E402
    glasses_majorana_plumbing_partition,
    theta_majorana_plumbing_partition,
)


SlotMode = tuple[int, int]


@dataclass(frozen=True)
class ThetaPhysicalFermionResummation:
    q_values: tuple[complex, complex, complex]
    lifts: tuple[int, int, int]
    max_mode: int
    chiral_value: complex
    nonchiral_value: float
    determinant_values: tuple[complex, complex, complex, complex]


@dataclass(frozen=True)
class ThetaPhysicalSuperfieldOscillator:
    q_values: tuple[complex, complex, complex]
    lifts: tuple[int, int, int]
    fermion_max_mode: int
    boson_max_word_length: int
    boson_max_mode: int
    boson_chiral: complex
    fermion_chiral: complex
    superfield_chiral_oscillator: complex
    superfield_nonchiral_oscillator: float


@dataclass(frozen=True)
class ThetaChargedBosonResummation:
    q_values: tuple[complex, complex, complex]
    max_mode: int
    alpha_zero: float
    alpha_one: float
    alpha_infinity: float
    vacuum_chiral: complex
    charged_exponent: complex
    chiral_value: complex


@dataclass(frozen=True)
class ThetaBosonLoopGaussian:
    q_values: tuple[complex, complex, complex]
    max_mode: int
    # The independent charge vector is (alpha_zero,alpha_one), with
    # alpha_infinity=-(alpha_zero+alpha_one).  In the convention
    # X X ~ -log and h(alpha)=alpha^2/2, the nonchiral momentum dependence is
    # exp(-pi alpha^T A alpha).
    charge_quadratic_matrix: tuple[tuple[float, float], tuple[float, float]]
    determinant: float
    minimum_eigenvalue: float
    charge_measure_gaussian: float


@dataclass(frozen=True)
class ThetaPhysicalSuperfieldPartition:
    oscillator: ThetaPhysicalSuperfieldOscillator
    loop_gaussian: ThetaBosonLoopGaussian
    one_superfield_value: float
    nine_superfield_value: float


@dataclass(frozen=True)
class GlassesPhysicalFermionResummation:
    q_values: tuple[complex, complex, complex]
    lifts: tuple[int, int, int]
    max_mode: int
    chiral_value: complex
    nonchiral_value: float


@dataclass(frozen=True)
class GlassesChargedBosonResummation:
    q_values: tuple[complex, complex, complex]
    max_mode: int
    alpha_left: float
    alpha_right: float
    vacuum_chiral: complex
    charged_exponent: complex
    chiral_value: complex


@dataclass(frozen=True)
class GlassesBosonLoopGaussian:
    q_values: tuple[complex, complex, complex]
    max_mode: int
    charge_quadratic_matrix: tuple[tuple[float, float], tuple[float, float]]
    determinant: float
    minimum_eigenvalue: float
    charge_measure_gaussian: float


@dataclass(frozen=True)
class GlassesPhysicalSuperfieldPartition:
    q_values: tuple[complex, complex, complex]
    lifts: tuple[int, int, int]
    boson_chiral: complex
    fermion_chiral: complex
    nonchiral_oscillator: float
    loop_gaussian: GlassesBosonLoopGaussian
    one_superfield_value: float
    nine_superfield_value: float


@dataclass(frozen=True)
class PhysicalSuperfieldPlumbingPartition:
    """Fast all-determinant physical ``X+psi`` plumbing partition.

    The same mode cutoff is used for the physical-Majorana Fredholm
    determinant, the charged-boson vacuum determinant, and the two-loop
    charge Gaussian.  No period matrix, theta constant, or auxiliary
    double-Virasoro fermion enters this object.
    """

    channel: str
    q_values: tuple[complex, complex, complex]
    lifts: tuple[int, int, int]
    max_mode: int
    boson_chiral: complex
    fermion_chiral: complex
    boson_nonchiral_oscillator: float
    fermion_nonchiral_oscillator: float
    nonchiral_oscillator: float
    charge_quadratic_matrix: tuple[tuple[float, float], tuple[float, float]]
    loop_gaussian: float
    one_superfield_value: float
    nine_superfield_value: float


def sphere_fermion_kernel(max_mode: int) -> tuple[np.ndarray, tuple[SlotMode, ...]]:
    r"""Return the analytic NS sphere kernel in slots ``(infinity,1,0)``.

    Mode ``m`` denotes ``psi_{-(m-1/2)}|0>``.  The upper-triangular entries
    are the three analytic two-field contractions in the normalized
    ``(infinity,1,0)`` pants frame.  In increasing mode order on all slots,
    the generating coefficient is the Pfaffian of the corresponding
    principal submatrix.  The physical BPZ bra reverses its fields and hence
    multiplies a bra state of size ``s`` by ``(-1)^(s(s-1)/2)``.  This sign
    cancels in theta's squared coefficient and is incorporated by the
    glasses half-edge pairing ledger below.
    """

    cutoff = int(max_mode)
    if cutoff < 1:
        raise ValueError("max_mode must be at least one")
    indices = tuple(
        (slot, mode) for slot in range(3) for mode in range(1, cutoff + 1)
    )

    def upper(left: SlotMode, right: SlotMode) -> int:
        left_slot, left_mode = left
        right_slot, right_mode = right
        if left_slot == right_slot:
            return 0
        if left_slot > right_slot:
            return -upper(right, left)
        if (left_slot, right_slot) == (0, 1):
            return (
                comb(left_mode - 1, right_mode - 1)
                if left_mode >= right_mode
                else 0
            )
        if (left_slot, right_slot) == (0, 2):
            return int(left_mode == right_mode)
        if (left_slot, right_slot) == (1, 2):
            return (-1) ** (left_mode - 1) * comb(
                left_mode + right_mode - 2,
                left_mode - 1,
            )
        raise AssertionError("unreachable sphere slot pair")

    kernel = np.asarray(
        [[upper(left, right) for right in indices] for left in indices],
        dtype=np.complex128,
    )
    if not np.array_equal(kernel.T, -kernel):
        raise AssertionError("physical fermion sphere kernel is not antisymmetric")
    return kernel, indices


def sphere_boson_kernel(max_mode: int) -> tuple[np.ndarray, tuple[SlotMode, ...]]:
    r"""Return the normalized charged-Heisenberg pants pairing kernel.

    The unnormalized elementary contractions are

    ``C_inf,1(m,n)=m binom(m-1,n-1)``,
    ``C_inf,0(m,n)=m delta(m,n)``, and
    ``C_1,0(m,n)=(-1)^(m-1) n binom(n+m-1,m-1)``.

    Dividing entry ``(slot,m),(slot',n)`` by ``sqrt(m n)`` changes to
    oscillator variables whose Fock metric is the ordinary factorial metric.
    """

    cutoff = int(max_mode)
    if cutoff < 1:
        raise ValueError("max_mode must be at least one")
    indices = tuple(
        (slot, mode) for slot in range(3) for mode in range(1, cutoff + 1)
    )

    def entry(left: SlotMode, right: SlotMode) -> float:
        left_slot, left_mode = left
        right_slot, right_mode = right
        if left_slot == right_slot:
            return 0.0
        if left_slot > right_slot:
            return entry(right, left)
        normalization = math.sqrt(left_mode * right_mode)
        if (left_slot, right_slot) == (0, 1):
            unnormalized = (
                left_mode * comb(left_mode - 1, right_mode - 1)
                if left_mode >= right_mode
                else 0
            )
        elif (left_slot, right_slot) == (0, 2):
            unnormalized = left_mode if left_mode == right_mode else 0
        elif (left_slot, right_slot) == (1, 2):
            unnormalized = (
                (-1) ** (left_mode - 1)
                * right_mode
                * comb(right_mode + left_mode - 1, left_mode - 1)
            )
        else:
            raise AssertionError("unreachable sphere slot pair")
        return float(unnormalized / normalization)

    kernel = np.asarray(
        [[entry(left, right) for right in indices] for left in indices],
        dtype=np.complex128,
    )
    if not np.array_equal(kernel.T, kernel):
        raise AssertionError("physical boson sphere kernel is not symmetric")
    return kernel, indices


def _charged_boson_linear_vector(
    indices: Sequence[SlotMode],
    *,
    alpha_zero: float,
    alpha_one: float,
) -> np.ndarray:
    r"""Return the current--exponential contractions in ``(inf,1,0)``.

    In the bra convention
    ``<alpha_inf|V_alpha_one(1)|alpha_zero>`` one has
    ``alpha_inf=alpha_one+alpha_zero``.  The three normalized one-current
    contractions are

    ``L_inf,m=alpha_one/sqrt(m)``,
    ``L_1,m=(-1)^(m-1) alpha_zero/sqrt(m)``, and
    ``L_0,m=-alpha_one/sqrt(m)``.
    """

    values = []
    for slot, mode in indices:
        if slot == 0:
            value = float(alpha_one)
        elif slot == 1:
            value = float((-1) ** (mode - 1) * alpha_zero)
        elif slot == 2:
            value = float(-alpha_one)
        else:
            raise AssertionError("unreachable sphere slot")
        values.append(value / math.sqrt(mode))
    return np.asarray(values, dtype=np.complex128)


def theta_charged_boson_resummation(
    q_values: Sequence[complex],
    *,
    alpha_zero: float,
    alpha_one: float,
    max_mode: int,
) -> ThetaChargedBosonResummation:
    r"""Resum the charged free-boson theta pants tensor.

    This is a direct Gaussian contraction of the two analytic pants tensors.
    The two tensors carry the same numerical charge labels in the inverse-BPZ
    sewing convention.  Their normalized linear source is therefore the
    symmetric source ``(ell,ell)``.  The vacuum determinant and charged Schur
    complement are

    ``P_M = det(1-B^2)^(-1/2)``,
    ``E_M = (D ell)^T (1-B)^(-1) (D ell)``,

    where ``D^2=W=diag(q_e^m)`` and ``B=D K_X D``.
    """

    q_tuple = tuple(complex(value) for value in q_values)
    if len(q_tuple) != 3 or any(abs(value) >= 1 for value in q_tuple):
        raise ValueError("three theta coordinates with |q_e|<1 are required")
    kernel, indices = sphere_boson_kernel(max_mode)
    q_slots = (q_tuple[2], q_tuple[1], q_tuple[0])
    square_weights = np.asarray(
        [cmath.sqrt(q_slots[slot] ** mode) for slot, mode in indices],
        dtype=np.complex128,
    )
    balanced = square_weights[:, None] * kernel * square_weights[None, :]
    identity = np.eye(len(indices), dtype=np.complex128)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        determinant = complex(np.linalg.det(identity - balanced @ balanced))
    if not np.isfinite(determinant.real) or not np.isfinite(determinant.imag):
        raise ArithmeticError("physical boson Fredholm determinant overflowed")
    vacuum = determinant ** (-0.5)
    # Select the branch continuously connected to the unit vacuum term.
    if abs(vacuum - 1.0) > abs(-vacuum - 1.0):
        vacuum = -vacuum
    linear = _charged_boson_linear_vector(
        indices,
        alpha_zero=float(alpha_zero),
        alpha_one=float(alpha_one),
    )
    balanced_linear = square_weights * linear
    exponent = complex(
        balanced_linear.T
        @ np.linalg.solve(identity - balanced, balanced_linear)
    )
    alpha_infinity = -(float(alpha_zero) + float(alpha_one))
    primary = (
        q_tuple[0] ** (0.5 * float(alpha_zero) ** 2)
        * q_tuple[1] ** (0.5 * float(alpha_one) ** 2)
        * q_tuple[2] ** (0.5 * alpha_infinity**2)
    )
    value = primary * vacuum * cmath.exp(exponent)
    return ThetaChargedBosonResummation(
        q_values=q_tuple,  # type: ignore[arg-type]
        max_mode=int(max_mode),
        alpha_zero=float(alpha_zero),
        alpha_one=float(alpha_one),
        alpha_infinity=float(alpha_infinity),
        vacuum_chiral=complex(vacuum),
        charged_exponent=complex(exponent),
        chiral_value=complex(value),
    )


def theta_boson_loop_gaussian(
    q_values: Sequence[complex],
    *,
    max_mode: int,
) -> ThetaBosonLoopGaussian:
    r"""Return the two-loop Gaussian directly from charged pants sewing.

    For independent charge vector ``u=(alpha_zero,alpha_one)``, define

    ``|F_X(u;q)/F_X(0;q)|^2 = exp(-pi u^T A u)``.

    The returned ``det(A)^(-1/2)`` therefore uses completeness measure
    ``d alpha_zero d alpha_one``.  A rescaled momentum convention must rescale
    both ``A`` and the completeness measure; no such constant is hidden here.
    """

    q_tuple = tuple(complex(value) for value in q_values)

    def log_ratio(alpha_zero: float, alpha_one: float) -> float:
        result = theta_charged_boson_resummation(
            q_tuple,
            alpha_zero=alpha_zero,
            alpha_one=alpha_one,
            max_mode=max_mode,
        )
        vacuum = theta_charged_boson_resummation(
            q_tuple,
            alpha_zero=0.0,
            alpha_one=0.0,
            max_mode=max_mode,
        )
        return float(2.0 * math.log(abs(result.chiral_value / vacuum.chiral_value)))

    value_10 = log_ratio(1.0, 0.0)
    value_01 = log_ratio(0.0, 1.0)
    value_11 = log_ratio(1.0, 1.0)
    a00 = -value_10 / math.pi
    a11 = -value_01 / math.pi
    a01 = -(value_11 - value_10 - value_01) / (2.0 * math.pi)
    matrix = np.asarray(((a00, a01), (a01, a11)), dtype=float)
    eigenvalues = np.linalg.eigvalsh(matrix)
    determinant = float(np.linalg.det(matrix))
    if eigenvalues[0] <= 0.0 or determinant <= 0.0:
        raise ValueError("direct plumbing loop-momentum Gaussian is not positive")
    return ThetaBosonLoopGaussian(
        q_values=q_tuple,  # type: ignore[arg-type]
        max_mode=int(max_mode),
        charge_quadratic_matrix=(
            (float(matrix[0, 0]), float(matrix[0, 1])),
            (float(matrix[1, 0]), float(matrix[1, 1])),
        ),
        determinant=determinant,
        minimum_eigenvalue=float(eigenvalues[0]),
        charge_measure_gaussian=float(determinant**-0.5),
    )


def _glasses_half_edge_index(
    vertex: int,
    slot: int,
    mode: int,
    max_mode: int,
) -> int:
    return int(vertex) * (3 * int(max_mode)) + int(slot) * int(max_mode) + int(mode) - 1


def _glasses_half_edge_scaling_and_pairing(
    q_values: Sequence[complex],
    *,
    max_mode: int,
    fermionic: bool,
    lifts: Sequence[int] = (1, 1, 1),
) -> tuple[np.ndarray, np.ndarray]:
    r"""Return diagonal half-edge scaling and the glasses pairing matrix."""

    q_tuple = tuple(complex(value) for value in q_values)
    lift_tuple = tuple(int(value) for value in lifts)
    if len(q_tuple) != 3 or any(abs(value) >= 1 for value in q_tuple):
        raise ValueError("glasses needs (q_left,q_right,q_bridge) with |q_e|<1")
    if len(lift_tuple) != 3 or any(value not in (-1, 1) for value in lift_tuple):
        raise ValueError("three physical +/-1 glasses lifts are required")
    dimension = 6 * int(max_mode)
    diagonal = np.zeros(dimension, dtype=np.complex128)
    pairing = np.zeros((dimension, dimension), dtype=np.complex128)
    for mode in range(1, int(max_mode) + 1):
        if fermionic:
            edge_weights = tuple(
                (lift * cmath.sqrt(q)) ** (2 * mode - 1)
                for lift, q in zip(lift_tuple, q_tuple)
            )
        else:
            edge_weights = tuple(q**mode for q in q_tuple)
        incidences = (
            ((0, 0), (0, 2), edge_weights[0]),
            ((1, 0), (1, 2), edge_weights[1]),
            ((0, 1), (1, 1), edge_weights[2]),
        )
        for edge_index, (left, right, weight) in enumerate(incidences):
            left_index = _glasses_half_edge_index(*left, mode, max_mode)
            right_index = _glasses_half_edge_index(*right, mode, max_mode)
            root = cmath.sqrt(weight)
            diagonal[left_index] = root
            diagonal[right_index] = root
            if fermionic:
                # In the coefficient (occupation-number) basis, the two
                # self-loop bends contribute -1.  The bridge contraction of
                # two Grassmann generating functions additionally produces
                # (-1)^{s(s-1)/2}.  Only even bridge occupation s survives,
                # so multiplying every bridge contraction by i supplies
                # i^s=(-1)^{s/2} and cancels that coefficient-basis sign.
                # This phase is physical-Majorana Fock bookkeeping; it is
                # unrelated to the auxiliary double-Virasoro fermion.
                orientation = -1 if edge_index in (0, 1) else 1j
                pairing[left_index, right_index] = orientation
                pairing[right_index, left_index] = -orientation
            else:
                pairing[left_index, right_index] = 1
                pairing[right_index, left_index] = 1
    return diagonal, pairing


def glasses_physical_fermion_fredholm(
    q_values: Sequence[complex],
    lifts: Sequence[int],
    *,
    max_mode: int,
) -> GlassesPhysicalFermionResummation:
    """Evaluate the physical glasses Majorana by one Gaussian Pfaffian."""

    q_tuple = tuple(complex(value) for value in q_values)
    lift_tuple = tuple(int(value) for value in lifts)
    local_kernel, _ = sphere_fermion_kernel(max_mode)
    zero = np.zeros_like(local_kernel)
    vertex_kernel = np.block([[local_kernel, zero], [zero, local_kernel]])
    scaling, pairing = _glasses_half_edge_scaling_and_pairing(
        q_tuple,
        max_mode=max_mode,
        fermionic=True,
        lifts=lift_tuple,
    )
    balanced_kernel = (
        scaling[:, None] * vertex_kernel * scaling[None, :]
    )
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        matrix = (
            np.eye(len(scaling), dtype=np.complex128)
            + pairing @ balanced_kernel
        )
        determinant = complex(np.linalg.det(matrix))
    if not np.isfinite(determinant.real) or not np.isfinite(determinant.imag):
        raise ArithmeticError("glasses physical fermion determinant overflowed")
    value = cmath.sqrt(determinant)
    if abs(value - 1.0) > abs(-value - 1.0):
        value = -value
    return GlassesPhysicalFermionResummation(
        q_values=q_tuple,  # type: ignore[arg-type]
        lifts=lift_tuple,  # type: ignore[arg-type]
        max_mode=int(max_mode),
        chiral_value=complex(value),
        nonchiral_value=float(abs(value) ** 2),
    )


def glasses_charged_boson_resummation(
    q_values: Sequence[complex],
    *,
    alpha_left: float,
    alpha_right: float,
    max_mode: int,
) -> GlassesChargedBosonResummation:
    """Contract the two charged glasses pants as one bosonic Gaussian."""

    q_tuple = tuple(complex(value) for value in q_values)
    local_kernel, indices = sphere_boson_kernel(max_mode)
    zero = np.zeros_like(local_kernel)
    vertex_kernel = np.block([[local_kernel, zero], [zero, local_kernel]])
    scaling, pairing = _glasses_half_edge_scaling_and_pairing(
        q_tuple,
        max_mode=max_mode,
        fermionic=False,
    )
    balanced_kernel = scaling[:, None] * vertex_kernel * scaling[None, :]
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        matrix = (
            np.eye(len(scaling), dtype=np.complex128)
            - pairing @ balanced_kernel
        )
        determinant = complex(np.linalg.det(matrix))
    if not np.isfinite(determinant.real) or not np.isfinite(determinant.imag):
        raise ArithmeticError("glasses charged-boson determinant overflowed")
    vacuum = determinant ** (-0.5)
    if abs(vacuum - 1.0) > abs(-vacuum - 1.0):
        vacuum = -vacuum
    left_linear = _charged_boson_linear_vector(
        indices,
        alpha_zero=float(alpha_left),
        alpha_one=0.0,
    )
    right_linear = _charged_boson_linear_vector(
        indices,
        alpha_zero=float(alpha_right),
        alpha_one=0.0,
    )
    linear = np.concatenate((left_linear, right_linear))
    balanced_linear = scaling * linear
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        exponent = complex(
            0.5
            * balanced_linear.T
            @ np.linalg.solve(matrix, pairing @ balanced_linear)
        )
    primary = (
        q_tuple[0] ** (0.5 * float(alpha_left) ** 2)
        * q_tuple[1] ** (0.5 * float(alpha_right) ** 2)
    )
    value = primary * vacuum * cmath.exp(exponent)
    return GlassesChargedBosonResummation(
        q_values=q_tuple,  # type: ignore[arg-type]
        max_mode=int(max_mode),
        alpha_left=float(alpha_left),
        alpha_right=float(alpha_right),
        vacuum_chiral=complex(vacuum),
        charged_exponent=complex(exponent),
        chiral_value=complex(value),
    )


def glasses_boson_loop_gaussian(
    q_values: Sequence[complex],
    *,
    max_mode: int,
) -> GlassesBosonLoopGaussian:
    """Return the direct glasses two-handle charge Gaussian."""

    q_tuple = tuple(complex(value) for value in q_values)
    vacuum = glasses_charged_boson_resummation(
        q_tuple,
        alpha_left=0.0,
        alpha_right=0.0,
        max_mode=max_mode,
    )

    def log_ratio(alpha_left: float, alpha_right: float) -> float:
        result = glasses_charged_boson_resummation(
            q_tuple,
            alpha_left=alpha_left,
            alpha_right=alpha_right,
            max_mode=max_mode,
        )
        return float(2.0 * math.log(abs(result.chiral_value / vacuum.chiral_value)))

    value_10 = log_ratio(1.0, 0.0)
    value_01 = log_ratio(0.0, 1.0)
    value_11 = log_ratio(1.0, 1.0)
    a00 = -value_10 / math.pi
    a11 = -value_01 / math.pi
    a01 = -(value_11 - value_10 - value_01) / (2.0 * math.pi)
    matrix = np.asarray(((a00, a01), (a01, a11)), dtype=float)
    eigenvalues = np.linalg.eigvalsh(matrix)
    determinant = float(np.linalg.det(matrix))
    if eigenvalues[0] <= 0.0 or determinant <= 0.0:
        raise ValueError("direct glasses loop Gaussian is not positive")
    return GlassesBosonLoopGaussian(
        q_values=q_tuple,  # type: ignore[arg-type]
        max_mode=int(max_mode),
        charge_quadratic_matrix=(
            (float(matrix[0, 0]), float(matrix[0, 1])),
            (float(matrix[1, 0]), float(matrix[1, 1])),
        ),
        determinant=determinant,
        minimum_eigenvalue=float(eigenvalues[0]),
        charge_measure_gaussian=float(determinant**-0.5),
    )


def glasses_physical_superfield_partition(
    q_values: Sequence[complex],
    lifts: Sequence[int],
    *,
    fermion_max_mode: int,
    boson_max_word_length: int,
    boson_max_mode: int,
    charged_boson_max_mode: int,
    boson_tolerance: float = 1.0e-15,
) -> GlassesPhysicalSuperfieldPartition:
    """Return one and nine physical free superfields in glasses plumbing."""

    q_tuple = tuple(complex(value) for value in q_values)
    lift_tuple = tuple(int(value) for value in lifts)
    fermion = glasses_physical_fermion_fredholm(
        q_tuple,
        lift_tuple,
        max_mode=fermion_max_mode,
    )
    boson_product = glasses_free_boson_product(
        *q_tuple,
        max_word_length=int(boson_max_word_length),
        max_mode=int(boson_max_mode),
        tolerance=float(boson_tolerance),
    )
    boson_chiral = cmath.exp(boson_product.chiral_log_product)
    oscillator = float(abs(boson_chiral * fermion.chiral_value) ** 2)
    loop = glasses_boson_loop_gaussian(
        q_tuple,
        max_mode=charged_boson_max_mode,
    )
    value = float(loop.charge_measure_gaussian * oscillator)
    return GlassesPhysicalSuperfieldPartition(
        q_values=q_tuple,  # type: ignore[arg-type]
        lifts=lift_tuple,  # type: ignore[arg-type]
        boson_chiral=complex(boson_chiral),
        fermion_chiral=complex(fermion.chiral_value),
        nonchiral_oscillator=oscillator,
        loop_gaussian=loop,
        one_superfield_value=value,
        nine_superfield_value=float(value**9),
    )


def _validate_theta_inputs(
    q_values: Sequence[complex], lifts: Sequence[int]
) -> tuple[tuple[complex, complex, complex], tuple[int, int, int]]:
    q_tuple = tuple(complex(value) for value in q_values)
    lift_tuple = tuple(int(value) for value in lifts)
    if len(q_tuple) != 3:
        raise ValueError("theta plumbing needs (q_zero,q_one,q_infinity)")
    if any(abs(value) >= 1 for value in q_tuple):
        raise ValueError("theta plumbing coordinates must satisfy |q_e|<1")
    if len(lift_tuple) != 3 or any(value not in (-1, 1) for value in lift_tuple):
        raise ValueError("three physical +/-1 plumbing lifts are required")
    return q_tuple, lift_tuple  # type: ignore[return-value]


def theta_physical_fermion_fredholm(
    q_values: Sequence[complex],
    lifts: Sequence[int],
    *,
    max_mode: int,
) -> ThetaPhysicalFermionResummation:
    r"""Evaluate the Human-Note theta Majorana block by determinants.

    The public edge order is ``(zero,one,infinity)``.  The analytic sphere
    kernel uses ``(infinity,one,zero)``, so the plumbing half-variables are
    reversed at this boundary and nowhere else.
    """

    q_tuple, lift_tuple = _validate_theta_inputs(q_values, lifts)
    kernel, indices = sphere_fermion_kernel(max_mode)
    half_variables_geometry = tuple(
        lift * cmath.sqrt(q)
        for lift, q in zip(lift_tuple, q_tuple)
    )
    half_variables_slots = (
        half_variables_geometry[2],
        half_variables_geometry[1],
        half_variables_geometry[0],
    )

    def determinant(slot_signs: tuple[int, int, int]) -> complex:
        weights = np.asarray(
            [
                slot_signs[slot]
                * half_variables_slots[slot] ** (2 * mode - 1)
                for slot, mode in indices
            ],
            dtype=np.complex128,
        )
        # det(1+K W)=det(1+sqrt(W) K sqrt(W)).  The balanced form is
        # essential numerically: the binomial entries of K grow with the
        # mode, while the plumbing weights decay.  Scaling only the columns
        # exposes large intermediate entries even when the Fredholm operator
        # itself is well conditioned.
        square_roots = np.asarray(
            [cmath.sqrt(complex(weight)) for weight in weights],
            dtype=np.complex128,
        )
        balanced = (
            np.eye(len(indices), dtype=np.complex128)
            + square_roots[:, None] * kernel * square_roots[None, :]
        )
        # NumPy 2.x can emit spurious floating-point warnings from its
        # internal determinant scaling even for a well-conditioned complex
        # matrix.  Finiteness is checked explicitly immediately afterwards.
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            value = complex(np.linalg.det(balanced))
        if not np.isfinite(value.real) or not np.isfinite(value.imag):
            raise ArithmeticError("physical fermion Fredholm determinant overflowed")
        return value

    sign_assignments = (
        (1, 1, 1),
        (-1, 1, 1),
        (1, -1, 1),
        (1, 1, -1),
    )
    determinants = tuple(determinant(signs) for signs in sign_assignments)
    value = (
        -determinants[0]
        + determinants[1]
        + determinants[2]
        + determinants[3]
    ) / 2.0
    return ThetaPhysicalFermionResummation(
        q_values=q_tuple,
        lifts=lift_tuple,
        max_mode=int(max_mode),
        chiral_value=complex(value),
        nonchiral_value=float(abs(value) ** 2),
        determinant_values=determinants,  # type: ignore[arg-type]
    )


def theta_physical_superfield_oscillator(
    q_values: Sequence[complex],
    lifts: Sequence[int],
    *,
    fermion_max_mode: int,
    boson_max_word_length: int,
    boson_max_mode: int,
    boson_tolerance: float = 1.0e-15,
) -> ThetaPhysicalSuperfieldOscillator:
    """Return the period-matrix-free theta ``X+psi`` oscillator factor."""

    q_tuple, lift_tuple = _validate_theta_inputs(q_values, lifts)
    fermion = theta_physical_fermion_fredholm(
        q_tuple,
        lift_tuple,
        max_mode=fermion_max_mode,
    )
    boson = theta_free_boson_product(
        *q_tuple,
        max_word_length=int(boson_max_word_length),
        max_mode=int(boson_max_mode),
        tolerance=float(boson_tolerance),
    )
    boson_chiral = cmath.exp(boson.chiral_log_product)
    chiral = boson_chiral * fermion.chiral_value
    return ThetaPhysicalSuperfieldOscillator(
        q_values=q_tuple,
        lifts=lift_tuple,
        fermion_max_mode=int(fermion_max_mode),
        boson_max_word_length=int(boson_max_word_length),
        boson_max_mode=int(boson_max_mode),
        boson_chiral=complex(boson_chiral),
        fermion_chiral=complex(fermion.chiral_value),
        superfield_chiral_oscillator=complex(chiral),
        superfield_nonchiral_oscillator=float(abs(chiral) ** 2),
    )


def theta_physical_superfield_partition(
    q_values: Sequence[complex],
    lifts: Sequence[int],
    *,
    fermion_max_mode: int,
    boson_max_word_length: int,
    boson_max_mode: int,
    charged_boson_max_mode: int,
    boson_tolerance: float = 1.0e-15,
) -> ThetaPhysicalSuperfieldPartition:
    r"""Return the complete theta free factor in the charge measure.

    The loop measure is ``d alpha_zero d alpha_one`` for
    ``X X ~ -log`` and ``h(alpha)=alpha^2/2``.  This convention is explicit
    because a rescaling of momentum changes the Gaussian and completeness
    measure together.
    """

    oscillator = theta_physical_superfield_oscillator(
        q_values,
        lifts,
        fermion_max_mode=fermion_max_mode,
        boson_max_word_length=boson_max_word_length,
        boson_max_mode=boson_max_mode,
        boson_tolerance=boson_tolerance,
    )
    loop = theta_boson_loop_gaussian(
        q_values,
        max_mode=charged_boson_max_mode,
    )
    value = loop.charge_measure_gaussian * oscillator.superfield_nonchiral_oscillator
    return ThetaPhysicalSuperfieldPartition(
        oscillator=oscillator,
        loop_gaussian=loop,
        one_superfield_value=float(value),
        nine_superfield_value=float(value**9),
    )


def physical_superfield_plumbing_partition(
    channel: str,
    q_values: Sequence[complex],
    lifts: Sequence[int],
    *,
    max_mode: int,
) -> PhysicalSuperfieldPlumbingPartition:
    r"""Evaluate the complete physical free denominator by plumbing matrices.

    This is the direct numerical implementation of equations (8.11) and
    (8.18) of the machine note.  In particular, the bosonic zero-mode factor
    is the Gaussian derived from charged pants sewing in the
    ``h(alpha)=alpha^2/2`` and ``d alpha_1 d alpha_2`` convention.  It is not
    replaced by ``det(Im Omega)^(-1/2)``.
    """

    q_tuple = tuple(complex(value) for value in q_values)
    lift_tuple = tuple(int(value) for value in lifts)
    cutoff = int(max_mode)
    if cutoff < 1:
        raise ValueError("max_mode must be at least one")
    if channel == "theta":
        fermion = theta_physical_fermion_fredholm(
            q_tuple,
            lift_tuple,
            max_mode=cutoff,
        )
        boson = theta_charged_boson_resummation(
            q_tuple,
            alpha_zero=0.0,
            alpha_one=0.0,
            max_mode=cutoff,
        )
        loop = theta_boson_loop_gaussian(q_tuple, max_mode=cutoff)
    elif channel == "glasses":
        fermion = glasses_physical_fermion_fredholm(
            q_tuple,
            lift_tuple,
            max_mode=cutoff,
        )
        boson = glasses_charged_boson_resummation(
            q_tuple,
            alpha_left=0.0,
            alpha_right=0.0,
            max_mode=cutoff,
        )
        loop = glasses_boson_loop_gaussian(q_tuple, max_mode=cutoff)
    else:
        raise ValueError("channel must be theta or glasses")

    boson_nonchiral = float(abs(boson.vacuum_chiral) ** 2)
    fermion_nonchiral = float(fermion.nonchiral_value)
    oscillator = float(boson_nonchiral * fermion_nonchiral)
    value = float(loop.charge_measure_gaussian * oscillator)
    return PhysicalSuperfieldPlumbingPartition(
        channel=channel,
        q_values=q_tuple,  # type: ignore[arg-type]
        lifts=lift_tuple,  # type: ignore[arg-type]
        max_mode=cutoff,
        boson_chiral=complex(boson.vacuum_chiral),
        fermion_chiral=complex(fermion.chiral_value),
        boson_nonchiral_oscillator=boson_nonchiral,
        fermion_nonchiral_oscillator=fermion_nonchiral,
        nonchiral_oscillator=oscillator,
        charge_quadratic_matrix=loop.charge_quadratic_matrix,
        loop_gaussian=float(loop.charge_measure_gaussian),
        one_superfield_value=value,
        nine_superfield_value=float(value**9),
    )


def _parse_complex(value: str) -> complex:
    return complex(value.replace("i", "j"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--channel", choices=("theta", "glasses"), default="theta")
    parser.add_argument("--q", nargs=3, type=_parse_complex, required=True)
    parser.add_argument("--lifts", nargs=3, type=int, default=(1, 1, 1))
    parser.add_argument("--fermion-max-mode", type=int, default=18)
    parser.add_argument("--direct-twice-level", type=int, default=20)
    parser.add_argument("--boson-word-length", type=int, default=10)
    parser.add_argument("--boson-max-mode", type=int, default=70)
    parser.add_argument("--charged-boson-max-mode", type=int, default=18)
    args = parser.parse_args()

    if args.channel == "theta":
        fermion = theta_physical_fermion_fredholm(
            args.q,
            args.lifts,
            max_mode=args.fermion_max_mode,
        )
        direct = theta_majorana_plumbing_partition(
            *args.q,
            max_total_twice_level=args.direct_twice_level,
            lifts=args.lifts,
        )
        oscillator = theta_physical_superfield_oscillator(
            args.q,
            args.lifts,
            fermion_max_mode=args.fermion_max_mode,
            boson_max_word_length=args.boson_word_length,
            boson_max_mode=args.boson_max_mode,
        )
        complete = theta_physical_superfield_partition(
            args.q,
            args.lifts,
            fermion_max_mode=args.fermion_max_mode,
            boson_max_word_length=args.boson_word_length,
            boson_max_mode=args.boson_max_mode,
            charged_boson_max_mode=args.charged_boson_max_mode,
        )
        boson_chiral = oscillator.boson_chiral
        superfield_chiral = oscillator.superfield_chiral_oscillator
        nonchiral_oscillator = oscillator.superfield_nonchiral_oscillator
        loop_matrix = complete.loop_gaussian.charge_quadratic_matrix
        loop_gaussian = complete.loop_gaussian.charge_measure_gaussian
    else:
        fermion = glasses_physical_fermion_fredholm(
            args.q,
            args.lifts,
            max_mode=args.fermion_max_mode,
        )
        direct = glasses_majorana_plumbing_partition(
            *args.q,
            max_total_twice_level=args.direct_twice_level,
            lifts=args.lifts,
        )
        complete = glasses_physical_superfield_partition(
            args.q,
            args.lifts,
            fermion_max_mode=args.fermion_max_mode,
            boson_max_word_length=args.boson_word_length,
            boson_max_mode=args.boson_max_mode,
            charged_boson_max_mode=args.charged_boson_max_mode,
        )
        boson_chiral = complete.boson_chiral
        superfield_chiral = complete.boson_chiral * complete.fermion_chiral
        nonchiral_oscillator = complete.nonchiral_oscillator
        loop_matrix = complete.loop_gaussian.charge_quadratic_matrix
        loop_gaussian = complete.loop_gaussian.charge_measure_gaussian

    print(f"physical {args.channel}-plumbing free superfield")
    print(f"  fermion Fredholm value: {fermion.chiral_value!r}")
    print(f"  fermion direct value:   {direct.chiral_value!r}")
    print(
        "  direct/Fredholm relative difference: "
        f"{abs(direct.chiral_value / fermion.chiral_value - 1.0):.6e}"
    )
    print(f"  boson chiral oscillator: {boson_chiral!r}")
    print(f"  X+psi chiral oscillator: {superfield_chiral!r}")
    print(
        "  X+psi nonchiral oscillator: "
        f"{nonchiral_oscillator:.16e}"
    )
    print(
        "  direct plumbing loop matrix: "
        f"{loop_matrix!r}"
    )
    print(
        "  loop Gaussian in the displayed two-charge measure: "
        f"{loop_gaussian:.16e}"
    )
    print(f"  one complete X+psi: {complete.one_superfield_value:.16e}")
    print(f"  nine complete X+psi: {complete.nine_superfield_value:.16e}")


if __name__ == "__main__":
    main()
