#!/usr/bin/env python3
r"""First two-channel genus-two NS super-Liouville partition experiment.

This module evaluates the raw plumbing-frame partition of the b=1
(``c=27/2``, or ``hat c=9``) N=1 super-Liouville theory in the theta and
glasses graphs.  The finite-c block is evaluated by a *functional*
Zamolodchikov c-recursion: ``recursion_order`` limits only the accumulated
twice-level of nested Kac residues.  Every recursion leaf is evaluated as a
direct, tolerance-controlled large-c regular block rather than as a finite
plumbing-q polynomial.

The Weyl-frame independent comparison is

    Q_L = Z_L / Z_(X+psi)^9,

where one free superfield is one noncompact real scalar and one NS Majorana
fermion, in the unit scalar zero-mode-volume convention.  The numerator and
denominator both use the raw plumbing convention (the common cylinder
Casimir factors are omitted edge by edge).

The implementation is deliberately conservative:

* theta and glasses blocks have separate graph kernels and residue ledgers;
* the glasses odd signs are fixed by the separating torus factorization;
* the two spin markings are related by the certified symplectic word, not by
  fitting the two answers;
* global osp(1|2), primitive-word, momentum-quadrature, and recursion-order
  convergence controls are recorded separately.

At order six and above the self-dual point contains confluent Kac poles.
Those orders are defined by the constant Laurent coefficient on a circle in
``t=log(b)``; two radii must be compared in any production run.
"""

from __future__ import annotations

import argparse
import cmath
import csv
from dataclasses import asdict, dataclass
from functools import lru_cache
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Iterable, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/type0b-matplotlib")

import numpy as np
from scipy.special import roots_genlaguerre


SCRIPT_DIR = Path(__file__).resolve().parent
PYTHON_DIR = SCRIPT_DIR / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from compare_ns_torus_c_h_recursion import _global_torus_block  # noqa: E402
from genus2_vacuum_blocks import (  # noqa: E402
    primitive_conjugacy_words,
    word_multiplier,
)
from ns_genus_c_recursion_checks import (  # noqa: E402
    ns_c_pole,
    ns_fusion_polynomial,
    ns_inverse_null_slope,
)
from ns_global_osp_block import osp_norm, osp_three_point  # noqa: E402
from ns_regular_block import (  # noqa: E402
    PlumbingFrameLedger,
    THETA_ORIENTATION,
)
from ns_vacuum_schottky import (  # noqa: E402
    ns_schottky_vacuum_block,
    spin_half_multiplier,
)
from plumbing_algorithms import (  # noqa: E402
    generators_for_glasses,
    generators_for_theta,
)
from super_liouville_structure_constants import (  # noqa: E402
    ns_structure_constant,
    ns_tilde_structure_constant,
)


C_HAT9 = 13.5

# Slot order is (0_L,1_L,infinity_L,0_R,1_R,infinity_R).  The self-sewn
# handles are (0_L,infinity_L) and (0_R,infinity_R), and the separating edge
# is (1_L,1_R).  The -1 transition on the right handle is forced by the
# determinant-one BPZ lift: without it the q_bridge=0 coefficient differs
# from the product of the two independently known torus kernels by
# (-1)^(right twice-level).
GLASSES_FRAME_LEDGER = PlumbingFrameLedger(
    edge_half_edges=((0, 2), (3, 5), (1, 4)),
    external_half_edges=(),
    contraction_order=(0, 2, 3, 5, 1, 4),
    half_edge_frame_signs=(1, 1, 1, 1, 1, -1),
    edge_transition_signs=(1, -1, 1),
)
GLASSES_ORIENTATION = GLASSES_FRAME_LEDGER.orientation()


@dataclass(frozen=True)
class GlobalSumDiagnostics:
    value: complex
    last_shell: complex
    max_total_occupation: int
    converged: bool


@dataclass(frozen=True)
class FreeSuperfieldDiagnostics:
    value: float
    chiral_log: complex
    det_im_omega: float
    primitive_count: int
    max_word_length: int
    max_mode: int
    previous_word_relative_change: float


@dataclass(frozen=True)
class ChannelResult:
    channel: str
    recursion_order: int
    quadrature_order: int
    value: float
    free_superfield: float
    q_l: float
    runtime_seconds: float
    global_max_occupation_used: int
    global_nonconverged_calls: int
    global_worst_last_shell_relative: float
    block_calls: int


def ns_weight(momentum: float) -> float:
    """b=1 NS continuum weight h=Q^2/8+P^2/2."""

    return 0.5 + 0.5 * float(momentum) ** 2


def _unit(edge: int) -> tuple[int, int, int]:
    return tuple(int(index == edge) for index in range(3))  # type: ignore[return-value]


def _transport_lifts(orientation, lifts: Sequence[int], edge: int) -> tuple[int, int, int]:
    """Insert the polarized orientation character for one odd null."""

    basis = _unit(edge)
    return tuple(
        int(lifts[target])
        * (-1) ** orientation.polarized_exponent(basis, _unit(target))
        for target in range(3)
    )  # type: ignore[return-value]


def _q_power(q: complex, occupation: int, fermion: int) -> complex:
    if q == 0:
        return 1.0 + 0.0j if occupation == 0 and fermion == 0 else 0.0j
    return q**occupation * (cmath.sqrt(q) if fermion else 1.0)


def _theta_schottky_data(
    q_values: Sequence[complex], edge_lifts: Sequence[int]
):
    r"""Return theta Schottky data in the stored plumbing/period marking.

    The overlap data use the two-pants coordinates of
    :func:`generators_for_theta`.  Relative to ``ccy_theta_generators``, this
    marking swaps the first and third plumbing parameters and inverts the
    second generator.  The inversion also contributes a central minus to the
    determinant-one lift selected by the two implementations.  Consequently
    the period-matched generator signs are

        (xi_zero*xi_infinity, -xi_one*xi_infinity).

    Keeping the generators and their lift signs in one helper prevents the
    Schottky oscillator product from drifting away from the period matrix.
    """

    if len(q_values) != 3 or len(edge_lifts) != 3:
        raise ValueError(
            "theta plumbing needs three coordinates and three lifts"
        )
    lifts = tuple(int(sign) for sign in edge_lifts)
    if any(sign not in (-1, 1) for sign in lifts):
        raise ValueError("theta plumbing lift signs must be +/-1")
    xi_zero, xi_one, xi_infinity = lifts
    return (
        generators_for_theta(*(complex(value) for value in q_values)),
        (xi_zero * xi_infinity, -xi_one * xi_infinity),
    )


def _occupation_shell(total: int) -> Iterable[tuple[int, int, int]]:
    for n0 in range(total + 1):
        for n1 in range(total - n0 + 1):
            yield n0, n1, total - n0 - n1


def _theta_global_term(
    weights: Sequence[complex],
    q_values: Sequence[complex],
    occupations: Sequence[int],
    fermions: Sequence[int],
    lifts: Sequence[int],
) -> complex:
    rho = osp_three_point(
        n1=int(occupations[0]),
        n2=int(occupations[1]),
        n3=int(occupations[2]),
        epsilon1=int(fermions[0]),
        epsilon2=int(fermions[1]),
        epsilon3=int(fermions[2]),
        d1=weights[0],
        d2=weights[1],
        d3=weights[2],
    )
    denominator = math.prod(
        osp_norm(weight, int(occupation), int(fermion))
        for weight, occupation, fermion in zip(weights, occupations, fermions)
    )
    plumbing = math.prod(
        _q_power(complex(q), int(occupation), int(fermion))
        * int(lift) ** int(fermion)
        for q, occupation, fermion, lift in zip(
            q_values, occupations, fermions, lifts
        )
    )
    orientation = (-1) ** THETA_ORIENTATION.exponent(fermions)
    return orientation * plumbing * rho * rho / denominator


def _glasses_global_term(
    weights: Sequence[complex],
    q_values: Sequence[complex],
    occupations: Sequence[int],
    fermions: Sequence[int],
    lifts: Sequence[int],
) -> complex:
    h_left, h_right, h_bridge = weights
    n_left, n_right, n_bridge = (int(value) for value in occupations)
    e_left, e_right, e_bridge = (int(value) for value in fermions)
    left = osp_three_point(
        n1=n_left,
        n2=n_bridge,
        n3=n_left,
        epsilon1=e_left,
        epsilon2=e_bridge,
        epsilon3=e_left,
        d1=h_left,
        d2=h_bridge,
        d3=h_left,
    )
    right = osp_three_point(
        n1=n_right,
        n2=n_bridge,
        n3=n_right,
        epsilon1=e_right,
        epsilon2=e_bridge,
        epsilon3=e_right,
        d1=h_right,
        d2=h_bridge,
        d3=h_right,
    )
    denominator = (
        osp_norm(h_left, n_left, e_left)
        * osp_norm(h_right, n_right, e_right)
        * osp_norm(h_bridge, n_bridge, e_bridge)
    )
    plumbing = math.prod(
        _q_power(complex(q), int(occupation), int(fermion))
        * int(lift) ** int(fermion)
        for q, occupation, fermion, lift in zip(
            q_values, occupations, fermions, lifts
        )
    )
    orientation = (-1) ** GLASSES_ORIENTATION.exponent(fermions)
    return orientation * plumbing * left * right / denominator


def direct_global_block(
    *,
    channel: str,
    weights: Sequence[complex],
    q_values: Sequence[complex],
    sector: int,
    lifts: Sequence[int],
    tolerance: float,
    max_total_occupation: int,
) -> GlobalSumDiagnostics:
    """Directly sum an osp graph to a requested numerical tolerance.

    This is a convergence cutoff on the direct numerical representation of
    the exact regular seed.  It is not the Zamolodchikov recursion order and
    is audited independently.
    """

    if channel not in {"theta", "glasses"}:
        raise ValueError("channel must be theta or glasses")
    if sector not in (0, 1):
        raise ValueError("sector must be 0 or 1")
    total_value = 0.0 + 0.0j
    last_shell = 0.0 + 0.0j
    small_shells = 0
    converged = False
    used = 0
    for total_occupation in range(max_total_occupation + 1):
        shell = 0.0 + 0.0j
        for occupations in _occupation_shell(total_occupation):
            for fermions in (
                (e0, e1, e2)
                for e0 in (0, 1)
                for e1 in (0, 1)
                for e2 in (0, 1)
            ):
                if channel == "theta":
                    allowed = sum(fermions) % 2 == sector
                    term_function = _theta_global_term
                else:
                    # A handle occurs twice at its trinion.  Hence the two
                    # vertex parities both equal the bridge fermion parity.
                    allowed = fermions[2] == sector
                    term_function = _glasses_global_term
                if not allowed:
                    continue
                shell += term_function(
                    weights, q_values, occupations, fermions, lifts
                )
        total_value += shell
        last_shell = shell
        used = total_occupation
        scale = max(1.0, abs(total_value))
        if total_occupation >= 5 and abs(shell) <= tolerance * scale:
            small_shells += 1
        else:
            small_shells = 0
        if small_shells >= 3:
            converged = True
            break
    return GlobalSumDiagnostics(
        value=total_value,
        last_shell=last_shell,
        max_total_occupation=used,
        converged=converged,
    )


class NSGenus2CRecursion:
    """Functional genus-two N=1 c-recursion in one plumbing channel."""

    def __init__(
        self,
        *,
        channel: str,
        q_values: Sequence[complex],
        global_tolerance: float = 2.0e-10,
        global_max_total_occupation: int = 15,
        vacuum_word_length: int = 7,
        vacuum_max_mode: int = 50,
    ) -> None:
        if channel not in {"theta", "glasses"}:
            raise ValueError("channel must be theta or glasses")
        if len(q_values) != 3 or any(not abs(complex(q)) < 1 for q in q_values):
            raise ValueError("three plumbing coordinates with |q|<1 are required")
        self.channel = channel
        self.q_values = tuple(complex(q) for q in q_values)
        self.orientation = (
            THETA_ORIENTATION if channel == "theta" else GLASSES_ORIENTATION
        )
        self.global_tolerance = float(global_tolerance)
        self.global_max_total_occupation = int(global_max_total_occupation)
        self.vacuum_word_length = int(vacuum_word_length)
        self.vacuum_max_mode = int(vacuum_max_mode)
        self.global_max_used = 0
        self.global_nonconverged_calls = 0
        self.global_worst_last_shell_relative = 0.0
        self.block_calls = 0

    @lru_cache(maxsize=None)
    def _vacuum(self, lifts: tuple[int, int, int]) -> complex:
        if self.channel == "theta":
            generators, generator_signs = _theta_schottky_data(
                self.q_values, lifts
            )
            # The certified overlap theta chart is extremely close to a
            # maximal cusp.  The product routine remains stable here, but a
            # short word cutoff is already far below the global tolerance.
        else:
            generators = generators_for_glasses(*self.q_values)
            # The bridge is a spanning-tree edge.  Its lift is a vertex-frame
            # gauge sign and the even vacuum trinion has even bridge parity;
            # the two homological spin lifts are the handle lifts.
            generator_signs = (lifts[0], lifts[1])
        return ns_schottky_vacuum_block(
            generators,
            generator_signs,
            max_word_length=self.vacuum_word_length,
            max_mode=self.vacuum_max_mode,
        ).value

    @lru_cache(maxsize=None)
    def _global(
        self,
        weights: tuple[complex, complex, complex],
        sector: int,
        lifts: tuple[int, int, int],
    ) -> complex:
        diagnostics = direct_global_block(
            channel=self.channel,
            weights=weights,
            q_values=self.q_values,
            sector=sector,
            lifts=lifts,
            tolerance=self.global_tolerance,
            max_total_occupation=self.global_max_total_occupation,
        )
        self.global_max_used = max(
            self.global_max_used, diagnostics.max_total_occupation
        )
        if not diagnostics.converged:
            self.global_nonconverged_calls += 1
        self.global_worst_last_shell_relative = max(
            self.global_worst_last_shell_relative,
            abs(diagnostics.last_shell) / max(1.0, abs(diagnostics.value)),
        )
        return diagnostics.value

    def _regular(
        self,
        weights: tuple[complex, complex, complex],
        sector: int,
        lifts: tuple[int, int, int],
    ) -> complex:
        # Resolve the vacuum/global Koszul cross sign by characters of the
        # vacuum lift variables.  The global sum is partitioned by its three
        # edge fermion parities.
        if self.channel == "glasses":
            # Q_glasses=e_bridge(e_left+e_right).  The even vacuum trinion
            # has even bridge parity, whereas every global term in sector
            # alpha has e_bridge=alpha.  Consequently the entire cross sign
            # is implemented by flipping both vacuum handle lifts once in
            # the odd sector; no component-by-component projection is needed.
            vacuum_lifts = (
                lifts[0] * (-1) ** sector,
                lifts[1] * (-1) ** sector,
                lifts[2],
            )
            return self._vacuum(vacuum_lifts) * self._global(
                weights, sector, lifts
            )
        result = 0.0 + 0.0j
        for sigma in (
            (e0, e1, e2)
            for e0 in (0, 1)
            for e1 in (0, 1)
            for e2 in (0, 1)
        ):
            if self.channel == "theta":
                if sum(sigma) % 2 != sector:
                    continue
            elif sigma[2] != sector:
                continue
            flip = tuple(
                self.orientation.polarized_exponent(_unit(edge), sigma)
                for edge in range(3)
            )
            vacuum_lifts = tuple(
                lifts[edge] * (-1) ** flip[edge] for edge in range(3)
            )
            # Compute a fixed-sigma global component directly.  Calling the
            # sector sum and projecting by lift characters would require all
            # eight characters; the explicit component is cheaper here.
            global_component = self._global_component(weights, sigma, lifts)
            result += self._vacuum(vacuum_lifts) * global_component
        return result

    @lru_cache(maxsize=None)
    def _global_component(
        self,
        weights: tuple[complex, complex, complex],
        sigma: tuple[int, int, int],
        lifts: tuple[int, int, int],
    ) -> complex:
        total_value = 0.0 + 0.0j
        small_shells = 0
        converged = False
        used = 0
        last_shell = 0.0 + 0.0j
        for total_occupation in range(self.global_max_total_occupation + 1):
            shell = 0.0 + 0.0j
            for occupations in _occupation_shell(total_occupation):
                if self.channel == "theta":
                    shell += _theta_global_term(
                        weights, self.q_values, occupations, sigma, lifts
                    )
                else:
                    shell += _glasses_global_term(
                        weights, self.q_values, occupations, sigma, lifts
                    )
            total_value += shell
            last_shell = shell
            used = total_occupation
            if total_occupation >= 5 and abs(shell) <= self.global_tolerance * max(
                1.0, abs(total_value)
            ):
                small_shells += 1
            else:
                small_shells = 0
            if small_shells >= 3:
                converged = True
                break
        self.global_max_used = max(self.global_max_used, used)
        if not converged:
            self.global_nonconverged_calls += 1
        self.global_worst_last_shell_relative = max(
            self.global_worst_last_shell_relative,
            abs(last_shell) / max(1.0, abs(total_value)),
        )
        return total_value

    def _residue(
        self,
        *,
        r: int,
        s: int,
        edge: int,
        weights: tuple[complex, complex, complex],
        sector: int,
    ) -> tuple[complex, complex, int]:
        rs = r * s
        pole = ns_c_pole(r, s, weights[edge])
        if self.channel == "theta":
            other = (
                (weights[1], weights[2]),
                (weights[2], weights[0]),
                (weights[0], weights[1]),
            )[edge]
            polynomials = (
                ns_fusion_polynomial(
                    r=r,
                    s=s,
                    alpha=sector,
                    first_weight=other[0],
                    second_weight=other[1],
                    b=pole.b,
                ),
            ) * 2
            child_sector = sector ^ (rs % 2)
        elif edge in (0, 1):
            # A tadpole edge meets the same trinion twice.  Factoring an odd
            # null toggles the intermediate three-form once at each endpoint,
            # so the final child sector is unchanged.  The two polynomials
            # have labels alpha and alpha+rs and see the unshifted/shifted
            # copy of the self-glued handle, as in the independent torus
            # c-recursion.
            bridge = weights[2]
            handle = weights[edge]
            polynomials = (
                ns_fusion_polynomial(
                    r=r,
                    s=s,
                    alpha=sector,
                    first_weight=bridge,
                    second_weight=handle,
                    b=pole.b,
                ),
                ns_fusion_polynomial(
                    r=r,
                    s=s,
                    alpha=sector ^ (rs % 2),
                    first_weight=bridge,
                    second_weight=handle + rs / 2.0,
                    b=pole.b,
                ),
            )
            child_sector = sector
        else:
            polynomials = (
                ns_fusion_polynomial(
                    r=r,
                    s=s,
                    alpha=sector,
                    first_weight=weights[0],
                    second_weight=weights[0],
                    b=pole.b,
                ),
                ns_fusion_polynomial(
                    r=r,
                    s=s,
                    alpha=sector,
                    first_weight=weights[1],
                    second_weight=weights[1],
                    b=pole.b,
                ),
            )
            child_sector = sector ^ (rs % 2)
        residue = (
            pole.jacobian
            * ns_inverse_null_slope(r, s, pole.b)
            * polynomials[0]
            * polynomials[1]
        )
        return pole.c, residue, child_sector

    def block(
        self,
        *,
        weights: Sequence[complex],
        sector: int,
        recursion_order: int,
        lifts: Sequence[int],
        central_charge: complex = C_HAT9,
    ) -> complex:
        if recursion_order < 0 or recursion_order > 8:
            raise ValueError(
                "this evaluator supports recursion orders 0..8"
            )
        weight_tuple = tuple(complex(value) for value in weights)
        lift_tuple = tuple(int(value) for value in lifts)
        if len(weight_tuple) != 3 or len(lift_tuple) != 3:
            raise ValueError("three weights and three lifts are required")
        if any(value not in (-1, 1) for value in lift_tuple):
            raise ValueError("lifts must be +/-1")

        @lru_cache(maxsize=None)
        def recurse(
            remaining: int,
            current_c: complex,
            current_weights: tuple[complex, complex, complex],
            current_sector: int,
            current_lifts: tuple[int, int, int],
        ) -> complex:
            self.block_calls += 1
            total = self._regular(current_weights, current_sector, current_lifts)
            for edge in range(3):
                for r in range(2, remaining + 1):
                    for s in range(1, remaining // r + 1):
                        rs = r * s
                        if rs > remaining or (r + s) % 2:
                            continue
                        pole_c, residue, child_sector = self._residue(
                            r=r,
                            s=s,
                            edge=edge,
                            weights=current_weights,
                            sector=current_sector,
                        )
                        denominator = current_c - pole_c
                        if abs(denominator) < 1.0e-12:
                            raise ZeroDivisionError(
                                "confluent c-pole encountered below the order-four bar"
                            )
                        shifted = list(current_weights)
                        shifted[edge] += rs / 2.0
                        if rs % 2:
                            child_lifts = _transport_lifts(
                                self.orientation, current_lifts, edge
                            )
                            orientation_constant = (-1) ** self.orientation.exponent(
                                _unit(edge)
                            )
                        else:
                            child_lifts = current_lifts
                            orientation_constant = 1
                        total += (
                            residue
                            / denominator
                            * complex(self.q_values[edge]) ** (rs / 2.0)
                            * current_lifts[edge] ** (rs % 2)
                            * orientation_constant
                            * recurse(
                                remaining - rs,
                                complex(pole_c),
                                tuple(shifted),
                                child_sector,
                                tuple(child_lifts),
                            )
                        )
            return total

        return recurse(
            int(recursion_order),
            complex(central_charge),
            weight_tuple,  # type: ignore[arg-type]
            int(sector),
            lift_tuple,  # type: ignore[arg-type]
        )

    def finite_part_block(
        self,
        *,
        momenta: Sequence[float],
        sector: int,
        recursion_order: int,
        lifts: Sequence[int],
        radius: float = 0.035,
        samples: int = 24,
    ) -> complex:
        r"""Return the b=1 constant Laurent coefficient of one block.

        The contour is ``b=exp(radius*exp(i theta))`` with midpoint angular
        sampling.  The initial central charge and all three continuum weights
        move together with b.  This is the same finite-part prescription used
        by the independently checked torus c-recursion; it is not a small real
        detuning.  Radius stability is intentionally the caller's concern so
        a production workflow can record two independent contours.
        """

        if recursion_order < 0 or recursion_order > 8:
            raise ValueError("finite-part recursion order must lie in 0..8")
        if len(momenta) != 3:
            raise ValueError("three continuum momenta are required")
        if not radius > 0 or samples < 4:
            raise ValueError("finite-part radius must be positive and samples >= 4")
        total = 0.0 + 0.0j
        for index in range(int(samples)):
            angle = 2.0 * math.pi * (index + 0.5) / int(samples)
            b_value = cmath.exp(float(radius) * cmath.exp(1j * angle))
            background = b_value + 1.0 / b_value
            central_charge = 1.5 + 3.0 * background * background
            weights = tuple(
                background * background / 8.0 + float(momentum) ** 2 / 2.0
                for momentum in momenta
            )
            total += self.block(
                weights=weights,
                sector=sector,
                recursion_order=recursion_order,
                lifts=lifts,
                central_charge=central_charge,
            )
        return total / int(samples)


def _free_superfield_chiral_log(
    generators,
    generator_lift_signs: Sequence[int],
    *,
    max_word_length: int,
    max_mode: int,
) -> tuple[complex, int]:
    """Raw chiral oscillator log for one scalar-Majorana superfield."""

    value = 0.0j
    words = _primitive_conjugacy_words_cached(
        len(generators), int(max_word_length)
    )
    for word in words:
        multiplier = word_multiplier(generators, word)
        half = spin_half_multiplier(generators, word, generator_lift_signs)
        if not abs(multiplier) < 1:
            raise ValueError("non-loxodromic primitive multiplier")
        for mode in range(1, int(max_mode) + 1):
            value -= cmath.log(1.0 - multiplier**mode)
            value += cmath.log(1.0 + half * multiplier ** (mode - 1))
    return value, len(words)


@lru_cache(maxsize=None)
def _primitive_conjugacy_words_cached(
    generator_count: int, max_word_length: int
) -> tuple[tuple[int, ...], ...]:
    """Cache the geometry-independent primitive-word enumeration."""

    return tuple(
        primitive_conjugacy_words(int(generator_count), int(max_word_length))
    )


@lru_cache(maxsize=None)
def _free_superfield_partition_cached(
    *,
    channel: str,
    q_values: tuple[complex, complex, complex],
    omega_entries: tuple[complex, complex, complex, complex],
    physical_lifts: tuple[int, int, int],
    max_word_length: int,
    max_mode: int,
) -> FreeSuperfieldDiagnostics:
    if channel == "theta":
        generators, generator_signs = _theta_schottky_data(
            q_values, physical_lifts
        )
    elif channel == "glasses":
        generators = generators_for_glasses(*q_values)
        generator_signs = (int(physical_lifts[0]), int(physical_lifts[1]))
    else:
        raise ValueError("channel must be theta or glasses")
    chiral_log, primitive_count = _free_superfield_chiral_log(
        generators,
        generator_signs,
        max_word_length=max_word_length,
        max_mode=max_mode,
    )
    previous_log, _ = _free_superfield_chiral_log(
        generators,
        generator_signs,
        max_word_length=max(1, int(max_word_length) - 1),
        max_mode=max_mode,
    )
    omega = np.asarray(omega_entries, dtype=np.complex128).reshape(2, 2)
    det_y = float(np.linalg.det(omega.imag))
    if det_y <= 0:
        raise ValueError("period matrix must have positive Im determinant")
    value = det_y**-0.5 * math.exp(2.0 * chiral_log.real)
    previous_value = det_y**-0.5 * math.exp(2.0 * previous_log.real)
    return FreeSuperfieldDiagnostics(
        value=float(value),
        chiral_log=chiral_log,
        det_im_omega=det_y,
        primitive_count=primitive_count,
        max_word_length=int(max_word_length),
        max_mode=int(max_mode),
        previous_word_relative_change=float(
            abs(value - previous_value) / max(1.0e-300, abs(value))
        ),
    )


def free_superfield_partition(
    *,
    channel: str,
    q_values: Sequence[complex],
    omega: np.ndarray,
    physical_lifts: Sequence[int],
    max_word_length: int,
    max_mode: int,
) -> FreeSuperfieldDiagnostics:
    """Cached public wrapper for the channel's free-superfield denominator."""

    q_tuple = tuple(complex(value) for value in q_values)
    lift_tuple = tuple(int(value) for value in physical_lifts)
    omega_tuple = tuple(
        complex(value) for value in np.asarray(omega, dtype=np.complex128).reshape(-1)
    )
    return _free_superfield_partition_cached(
        channel=channel,
        q_values=q_tuple,  # type: ignore[arg-type]
        omega_entries=omega_tuple,  # type: ignore[arg-type]
        physical_lifts=lift_tuple,  # type: ignore[arg-type]
        max_word_length=int(max_word_length),
        max_mode=int(max_mode),
    )


def _primary_gaussian_rule(q: complex, order: int) -> tuple[np.ndarray, np.ndarray]:
    r"""Gauss-Laguerre rule for exp(-|log q| P^2) dP/pi."""

    log_abs = math.log(abs(complex(q)))
    if not log_abs < 0:
        raise ValueError("primary Gaussian rule requires 0<|q|<1")
    u, w = roots_genlaguerre(int(order), -0.5)
    scale = 1.0 / math.sqrt(-log_abs)
    nodes = scale * np.sqrt(u)
    weights = scale * w * np.exp(u) / (2.0 * math.pi)
    return np.asarray(nodes, dtype=float), np.asarray(weights, dtype=float)


def _structure_weight(
    channel: str,
    sector: int,
    momenta: Sequence[float],
    precision: int,
) -> float:
    constant = ns_structure_constant if sector == 0 else ns_tilde_structure_constant
    if channel == "theta":
        value = constant(*momenta, precision=precision) ** 2
    else:
        left, right, bridge = momenta
        value = constant(left, bridge, left, precision=precision) * constant(
            right, bridge, right, precision=precision
        )
    if abs(value.imag) > 2.0e-9 * max(1.0, abs(value.real)):
        raise ArithmeticError(f"structure weight is not real: {value!r}")
    return float(value.real)


def evaluate_channel(
    *,
    channel: str,
    q_values: Sequence[complex],
    omega: np.ndarray,
    physical_lifts: Sequence[int],
    recursion_order: int,
    quadrature_order: int,
    structure_precision: int,
    global_tolerance: float,
    global_max_total_occupation: int,
    vacuum_word_length: int,
    vacuum_max_mode: int,
    free_word_length: int,
    free_max_mode: int,
) -> tuple[ChannelResult, FreeSuperfieldDiagnostics]:
    started = time.time()
    rules = [_primary_gaussian_rule(q, quadrature_order) for q in q_values]
    recursion = NSGenus2CRecursion(
        channel=channel,
        q_values=q_values,
        global_tolerance=global_tolerance,
        global_max_total_occupation=global_max_total_occupation,
        vacuum_word_length=vacuum_word_length,
        vacuum_max_mode=vacuum_max_mode,
    )
    total = 0.0
    structure_cache: dict[tuple[int, float, float, float], float] = {}
    for i0, p0 in enumerate(rules[0][0]):
        for i1, p1 in enumerate(rules[1][0]):
            for i2, p2 in enumerate(rules[2][0]):
                momenta = (float(p0), float(p1), float(p2))
                weights = tuple(ns_weight(momentum) for momentum in momenta)
                primary = cmath.exp(
                    sum(
                        weight * cmath.log(complex(q))
                        for weight, q in zip(weights, q_values)
                    )
                )
                measure = (
                    rules[0][1][i0] * rules[1][1][i1] * rules[2][1][i2]
                )
                sector_total = 0.0
                for sector in (0, 1):
                    structure_key = (sector,) + momenta
                    if structure_key not in structure_cache:
                        structure_cache[structure_key] = _structure_weight(
                            channel,
                            sector,
                            momenta,
                            structure_precision,
                        )
                    block = recursion.block(
                        weights=weights,
                        sector=sector,
                        recursion_order=recursion_order,
                        lifts=physical_lifts,
                    )
                    sector_total += structure_cache[structure_key] * abs(
                        primary * block
                    ) ** 2
                total += float(measure) * sector_total
    free = free_superfield_partition(
        channel=channel,
        q_values=q_values,
        omega=omega,
        physical_lifts=physical_lifts,
        max_word_length=free_word_length,
        max_mode=free_max_mode,
    )
    q_l = total / free.value**9
    return (
        ChannelResult(
            channel=channel,
            recursion_order=int(recursion_order),
            quadrature_order=int(quadrature_order),
            value=float(total),
            free_superfield=float(free.value),
            q_l=float(q_l),
            runtime_seconds=float(time.time() - started),
            global_max_occupation_used=recursion.global_max_used,
            global_nonconverged_calls=recursion.global_nonconverged_calls,
            global_worst_last_shell_relative=float(
                recursion.global_worst_last_shell_relative
            ),
            block_calls=recursion.block_calls,
        ),
        free,
    )


def _omega(row: dict[str, str], prefix: str) -> np.ndarray:
    return np.asarray(
        [
            [
                complex(float(row[f"{prefix}_11_real"]), float(row[f"{prefix}_11_imag"])),
                complex(float(row[f"{prefix}_12_real"]), float(row[f"{prefix}_12_imag"])),
            ],
            [
                complex(float(row[f"{prefix}_12_real"]), float(row[f"{prefix}_12_imag"])),
                complex(float(row[f"{prefix}_22_real"]), float(row[f"{prefix}_22_imag"])),
            ],
        ],
        dtype=np.complex128,
    )


def _complex_json(value: complex) -> dict[str, float]:
    return {"real": float(value.real), "imag": float(value.imag)}


def run_internal_checks() -> dict[str, object]:
    """Independent sign and separating-limit checks used before integration."""

    if GLASSES_ORIENTATION.edge_linear_bits != (0, 0, 0):
        raise AssertionError("glasses frame did not derive zero linear bits")
    exponent_table = {
        bits: GLASSES_ORIENTATION.exponent(bits)
        for bits in (
            (e0, e1, e2)
            for e0 in (0, 1)
            for e1 in (0, 1)
            for e2 in (0, 1)
        )
    }
    if any(
        exponent != (bits[2] * (bits[0] + bits[1])) % 2
        for bits, exponent in exponent_table.items()
    ):
        raise AssertionError("glasses orientation is not bridge*(left+right)")

    q_left = 0.11 + 0.006j
    q_right = 0.14 - 0.004j
    h_left, h_right, h_bridge = 0.73, 0.91, 0.62
    global_glasses = direct_global_block(
        channel="glasses",
        weights=(h_left, h_right, h_bridge),
        q_values=(q_left, q_right, 0.0),
        sector=0,
        lifts=(1, 1, 1),
        tolerance=2.0e-13,
        max_total_occupation=18,
    )
    torus_product = complex(
        _global_torus_block(q_left, 1, h_left, h_bridge)
        * _global_torus_block(q_right, 1, h_right, h_bridge)
    )
    factorization_error = abs(global_glasses.value - torus_product) / max(
        1.0, abs(torus_product)
    )
    if factorization_error > 2.0e-10:
        raise AssertionError(
            f"glasses global separating factorization failed: {factorization_error:.3e}"
        )

    # The self-sewn handle residue must reduce to the independently checked
    # torus c-recursion residue.  This catches the otherwise easy-to-miss
    # odd-null endpoint sign.
    from compare_ns_torus_c_h_recursion import (
        _c_pole,
        _ns_a_factor,
        _ns_ns_fusion_polynomial,
    )

    recursion = NSGenus2CRecursion(
        channel="glasses",
        q_values=(q_left, q_right, 1.0e-3),
    )
    pole_c, handle_residue, child_sector = recursion._residue(
        r=3,
        s=1,
        edge=0,
        weights=(h_left, h_right, h_bridge),
        sector=0,
    )
    torus_b, torus_c, torus_jacobian = _c_pole(h_left, 3, 1)
    torus_residue = (
        -torus_jacobian
        * _ns_a_factor(torus_b, 3, 1)
        * _ns_ns_fusion_polynomial(
            b=torus_b,
            r=3,
            s=1,
            lower_weight=h_left + 1.5,
            upper_weight=h_bridge,
            starred=True,
        )
        * _ns_ns_fusion_polynomial(
            b=torus_b,
            r=3,
            s=1,
            lower_weight=h_left,
            upper_weight=h_bridge,
            starred=False,
        )
    )
    handle_residue_error = max(
        abs(pole_c - complex(torus_c)),
        abs(handle_residue - complex(torus_residue)),
    ) / max(1.0, abs(complex(torus_residue)))
    if handle_residue_error > 2.0e-12 or child_sector != 0:
        raise AssertionError("glasses handle residue failed the torus reduction")

    # The modular word maps glasses characteristic [00|00] to theta [00|11].
    symplectic_matrix = np.asarray(
        [[2, -1, -1, -1], [-2, 1, 0, -1], [1, 0, 0, 0], [-1, 1, 0, 0]],
        dtype=int,
    )
    A, B = symplectic_matrix[:2, :2], symplectic_matrix[:2, 2:]
    C, D = symplectic_matrix[2:, :2], symplectic_matrix[2:, 2:]
    theta_alpha = tuple((np.diag(C @ D.T) % 2).tolist())
    theta_beta = tuple((np.diag(A @ B.T) % 2).tolist())
    if (theta_alpha, theta_beta) != ((0, 0), (1, 1)):
        raise AssertionError("spin characteristic transport changed")
    _, theta_generator_signs = _theta_schottky_data(
        (0.11, 0.12, 0.13), (1, -1, -1)
    )
    if theta_generator_signs != (-1, -1):
        raise AssertionError("theta edge lifts do not realize [00|11]")
    return {
        "glasses_orientation_bits": list(GLASSES_ORIENTATION.edge_linear_bits),
        "glasses_orientation_formula": "e_bridge*(e_left+e_right) mod 2",
        "separating_global_relative_error": float(factorization_error),
        "separating_global_sum_converged": global_glasses.converged,
        "handle_residue_torus_relative_error": float(handle_residue_error),
        "spin_source_characteristic": {"alpha": [0, 0], "beta": [0, 0]},
        "spin_target_characteristic": {"alpha": [0, 0], "beta": [1, 1]},
        "glasses_edge_lifts": [1, 1, 1],
        "theta_edge_lifts": [1, -1, -1],
        "symplectic_matrix": symplectic_matrix.tolist(),
    }


def _json_default(value: object) -> object:
    if isinstance(value, complex):
        return _complex_json(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"cannot JSON encode {type(value).__name__}")


def _plot(path: Path, rows: Sequence[dict[str, object]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(8.8, 5.6), constrained_layout=True)
    grouped: dict[str, list[dict[str, object]]] = {"theta": [], "glasses": []}
    for row in rows:
        grouped[str(row["channel"])].append(row)
    styles = {"theta": ("#176b87", "o"), "glasses": ("#c65d37", "s")}
    for channel, values in grouped.items():
        values.sort(key=lambda row: (int(row["recursion_order"]), int(row["quadrature_order"])))
        color, marker = styles[channel]
        x = np.arange(len(values))
        axis.plot(
            x,
            [float(row["q_l"]) for row in values],
            color=color,
            marker=marker,
            linewidth=1.6,
            label=channel,
        )
        for position, row in zip(x, values):
            axis.annotate(
                f"R{row['recursion_order']}/N{row['quadrature_order']}",
                (position, float(row["q_l"])),
                xytext=(0, 7),
                textcoords="offset points",
                ha="center",
                fontsize=8,
            )
    axis.set_yscale("log")
    axis.set_ylabel(r"$Q_L=Z_L/Z_{X+\psi}^{9}$")
    axis.set_xlabel("convergence setting (R=recursion order, N=quadrature order)")
    axis.set_title("Genus-two NS super-Liouville: independent plumbing channels")
    axis.grid(alpha=0.22, which="both")
    axis.legend(frameon=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=190)
    fig.savefig(path.with_suffix(".svg"))
    plt.close(fig)


def run(argv: Sequence[str] | None = None) -> dict[str, object]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--overlap-csv",
        type=Path,
        default=Path(
            "../Project/StringMC/plumbing/results/genus2_plumbing_moduli_samples/"
            "direct_bulk_N128_overlap_N32/overlap_samples.csv"
        ),
    )
    parser.add_argument("--overlap-id", default="o0026")
    parser.add_argument(
        "--settings",
        nargs="+",
        default=("0:4", "3:4", "4:4", "4:6"),
        help="recursion:quadrature pairs",
    )
    parser.add_argument("--structure-precision", type=int, default=30)
    parser.add_argument("--global-tolerance", type=float, default=2.0e-9)
    parser.add_argument("--global-max-occupation", type=int, default=18)
    parser.add_argument("--vacuum-word-length", type=int, default=6)
    parser.add_argument("--vacuum-max-mode", type=int, default=36)
    parser.add_argument("--free-word-length", type=int, default=13)
    parser.add_argument("--free-max-mode", type=int, default=50)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("Data Set/ns_genus2_theta_glasses_hatc9.json"),
    )
    parser.add_argument(
        "--output-plot",
        type=Path,
        default=Path("Data Set/ns_genus2_theta_glasses_hatc9.png"),
    )
    parser.add_argument("--skip-plot", action="store_true")
    args = parser.parse_args(argv)

    checks = run_internal_checks()
    source_rows = list(csv.DictReader(args.overlap_csv.open()))
    matches = [row for row in source_rows if row["overlap_id"] == args.overlap_id]
    if len(matches) != 1 or matches[0].get("status") != "ok":
        raise RuntimeError("requested certified overlap row is unavailable")
    source = matches[0]
    q_values = {
        channel: tuple(
            complex(source[f"{channel}_q{index}"]) for index in range(1, 4)
        )
        for channel in ("theta", "glasses")
    }
    omegas = {
        channel: _omega(source, f"{channel}_omega")
        for channel in ("theta", "glasses")
    }
    lifts = {"theta": (1, -1, -1), "glasses": (1, 1, 1)}
    settings = []
    for text in args.settings:
        recursion_text, quadrature_text = text.split(":", 1)
        settings.append((int(recursion_text), int(quadrature_text)))

    result_rows: list[dict[str, object]] = []
    free_diagnostics: dict[str, object] = {}
    for recursion_order, quadrature_order in settings:
        for channel in ("theta", "glasses"):
            print(
                f"{channel}: recursion={recursion_order} quadrature={quadrature_order}",
                flush=True,
            )
            result, free = evaluate_channel(
                channel=channel,
                q_values=q_values[channel],
                omega=omegas[channel],
                physical_lifts=lifts[channel],
                recursion_order=recursion_order,
                quadrature_order=quadrature_order,
                structure_precision=args.structure_precision,
                global_tolerance=args.global_tolerance,
                global_max_total_occupation=args.global_max_occupation,
                vacuum_word_length=args.vacuum_word_length,
                vacuum_max_mode=args.vacuum_max_mode,
                free_word_length=args.free_word_length,
                free_max_mode=args.free_max_mode,
            )
            row = asdict(result)
            result_rows.append(row)
            free_diagnostics[channel] = asdict(free)
            print(
                f"  ZL={result.value:.10e} Zfree={result.free_superfield:.10e} "
                f"QL={result.q_l:.10e} time={result.runtime_seconds:.1f}s",
                flush=True,
            )

    comparisons = []
    for recursion_order, quadrature_order in settings:
        selected = {
            str(row["channel"]): row
            for row in result_rows
            if int(row["recursion_order"]) == recursion_order
            and int(row["quadrature_order"]) == quadrature_order
        }
        ratio = float(selected["theta"]["q_l"]) / float(selected["glasses"]["q_l"])
        comparisons.append(
            {
                "recursion_order": recursion_order,
                "quadrature_order": quadrature_order,
                "theta_over_glasses": ratio,
                "relative_difference": ratio - 1.0,
            }
        )

    source_subset = {
        key: value
        for key, value in source.items()
        if key
        in {
            "overlap_id",
            "theta_word",
            "theta_q1",
            "theta_q2",
            "theta_q3",
            "glasses_q1",
            "glasses_q2",
            "glasses_q3",
            "both_q_max",
            "theta_period_residual",
            "theta_validation_residual",
            "theta_map_stability",
            "glasses_seam_residual",
            "glasses_symmetry_error",
        }
    }
    output = {
        "scope": "First genus-two NS super-Liouville theta/glasses c-recursion comparison",
        "quantity": "Q_L = Z_L / Z_(one free scalar + one NS Majorana)^9",
        "normalization": (
            "raw plumbing cylinder convention; common Casimir factors omitted; "
            "unit noncompact scalar zero-mode volume"
        ),
        "central_charge": C_HAT9,
        "hat_c": 9,
        "source_csv": str(args.overlap_csv),
        "source_row": source_subset,
        "checks": checks,
        "numerics": {
            "structure_precision": args.structure_precision,
            "global_tolerance": args.global_tolerance,
            "global_max_total_occupation": args.global_max_occupation,
            "vacuum_word_length": args.vacuum_word_length,
            "vacuum_max_mode": args.vacuum_max_mode,
            "free_word_length": args.free_word_length,
            "free_max_mode": args.free_max_mode,
            "warning": (
                "recursion order is the only conformal recursion cutoff; the osp and "
                "primitive products have separately reported numerical convergence cutoffs"
            ),
        },
        "free_superfield_diagnostics": free_diagnostics,
        "results": result_rows,
        "comparisons": comparisons,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(output, indent=2, default=_json_default) + "\n"
    )
    if not args.skip_plot:
        _plot(args.output_plot, result_rows)
    return output


if __name__ == "__main__":
    run()
