#!/usr/bin/env python3
r"""First two-channel genus-two NS super-Liouville partition experiment.

This module evaluates the raw plumbing-frame partition of the b=1
(``c=27/2``, or ``hat c=9``) N=1 super-Liouville theory in the theta and
glasses graphs.  The finite-c block is evaluated by a *functional*
Zamolodchikov c-recursion: ``recursion_order`` limits only the accumulated
twice-level of nested Kac residues.  Every recursion leaf is evaluated as a
large-c regular block rather than as a finite plumbing-q polynomial.  The
glasses global block is resummed exactly into three factorized one-variable
kernels built from Gauss hypergeometric functions.  In the theta graph the
complete non-tiny middle-edge family is resummed into a Gauss function, leaving
only a separately audited adaptive sum on its two near-cusp endpoint edges.

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

import mpmath
import numpy as np
from scipy.special import roots_genlaguerre


SCRIPT_DIR = Path(__file__).resolve().parent
PYTHON_DIR = SCRIPT_DIR / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from compare_ns_torus_c_h_recursion import _global_torus_block  # noqa: E402
from free_boson_plumbing import riemann_theta_constant_genus2  # noqa: E402
from genus2_vacuum_blocks import (  # noqa: E402
    primitive_conjugacy_words,
    word_multiplier,
)
from ccy_genus2_block import PartialFractionInC  # noqa: E402
from ns_genus_c_recursion_checks import (  # noqa: E402
    ns_c_pole,
    ns_fusion_polynomial,
    ns_inverse_null_slope,
)
from ns_recursion_recipe import (  # noqa: E402
    ns_ordinary_edge_scalar_kernel_mp,
    ns_self_loop_scalar_kernel_mp,
)
from ns_global_osp_block import (  # noqa: E402
    osp_norm,
    osp_three_point,
    osp_two_chain_kernel,
)
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


# Keep the two conventions explicit.  The block and Gram-matrix APIs always
# receive the ordinary super-Virasoro central charge; ``hat c`` is metadata.
HAT_C_TARGET = 9.0
C_ORDINARY_AT_HAT_C_9 = 1.5 * HAT_C_TARGET
# Higher orders are enabled for pointwise convergence diagnostics.  Production
# locality comparisons remain gated on an explicit order-to-order audit.
MAX_RECURSION_ORDER = 24

# The overlap table, Schottky generators, primary propagators, and spin lifts
# are all edge-labelled in geometric theta order.  The CCY trinion tensor has
# a different positional convention: its slots are (infinity, one, zero).
# Keep the conversion local to calls into that tensor; changing the stored
# edge order would instead misattach the primary and Schottky factors.
THETA_GEOMETRY_EDGE_ORDER = ("q_zero", "q_one", "q_infinity")
THETA_CCY_DESCENDANT_EDGE_ORDER = ("q_infinity", "q_one", "q_zero")
GLASSES_GEOMETRY_EDGE_ORDER = ("q_left", "q_right", "q_bridge")
GLASSES_CCY_DESCENDANT_EDGE_ORDER = GLASSES_GEOMETRY_EDGE_ORDER

# The stored theta periods include this final integer branch shift relative to
# the unbranched modular word.  With Omega -> Omega + N, the symplectic factor
# is [[I,N],[0,I]], so its sign is fixed by direct matrix composition.
THETA_INTEGER_BRANCH = ((-1, 1), (1, -1))
GLASSES_TO_THETA_UNBRANCHED = (
    (2, -1, -1, -1),
    (-2, 1, 0, -1),
    (1, 0, 0, 0),
    (-1, 1, 0, 0),
)
GLASSES_TO_THETA_BRANCH_COMPOSED = (
    (0, 0, -1, -1),
    (0, 0, 0, -1),
    (1, 0, 0, 0),
    (-1, 1, 0, 0),
)

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


class _MPPartialFractionInC:
    """Arbitrary-precision partial fractions used at confluent NS poles."""

    _CONFLUENT_ANCHOR = mpmath.mpf("1.5")
    _MAX_MOMENT_RATIO = mpmath.mpf("0.2")
    _MAX_MOMENT_TERMS = 512
    _MIXED_FAMILY = object()

    def __init__(self, constant=0, *, moment_diagnostics=None) -> None:
        self.constant = mpmath.mpc(constant)
        self.poles: list[tuple[mpmath.mpc, list[mpmath.mpc], object]] = []
        self.moment_diagnostics = moment_diagnostics

    def _matching_index(self, pole, tolerance) -> int | None:
        pole = mpmath.mpc(pole)
        for index, (existing, _, _) in enumerate(self.poles):
            if abs(existing - pole) < tolerance:
                return index
        return None

    def add_pole_coefficient(
        self,
        pole,
        order,
        coefficient,
        tolerance,
        *,
        family_key: int | None = None,
    ) -> None:
        if order <= 0:
            raise ValueError("partial-fraction pole order must be positive")
        coefficient = mpmath.mpc(coefficient)
        if coefficient == 0:
            return
        pole = mpmath.mpc(pole)
        index = self._matching_index(pole, tolerance)
        if index is None:
            self.poles.append((pole, [], family_key))
            index = len(self.poles) - 1
        existing_pole, coefficients, existing_family = self.poles[index]
        if existing_family is self._MIXED_FAMILY:
            merged_family = existing_family
        elif existing_family is None:
            merged_family = family_key
        elif family_key is None or family_key == existing_family:
            merged_family = existing_family
        else:
            # An exact exceptional-locus collision can merge algebraically
            # distinct fixed-(r-s) families.  Keep the Laurent pole combined,
            # but do not assign it to either near-confluent moment family.
            merged_family = self._MIXED_FAMILY
        self.poles[index] = (existing_pole, coefficients, merged_family)
        while len(coefficients) < order:
            coefficients.append(mpmath.mpc(0))
        coefficients[order - 1] += coefficient

    def _record_ratio(self, key, numerator, denominator) -> None:
        if self.moment_diagnostics is None or numerator == 0:
            return
        ratio = numerator / abs(denominator) if denominator != 0 else mpmath.inf
        self.moment_diagnostics[key] = max(
            self.moment_diagnostics.get(key, mpmath.mpf(0)), ratio
        )

    def _direct_group_value(self, entries, evaluation_point, tolerance):
        value = mpmath.mpc(0)
        absolute_sum = mpmath.mpf(0)
        for pole, coefficients, _ in entries:
            delta = evaluation_point - pole
            if abs(delta) < tolerance:
                raise ZeroDivisionError(
                    "requested central charge lies on an uncancelled NS pole"
                )
            for order, coefficient in enumerate(coefficients, start=1):
                term = coefficient / delta**order
                value += term
                absolute_sum += abs(term)
        self._record_ratio(
            "max_direct_cancellation_ratio", absolute_sum, value
        )
        return value

    def _fixed_difference_moment_value(
        self, entries, evaluation_point, tolerance
    ):
        r"""Combine one fixed-``(r-s)`` Kac family before pole evaluation.

        All positive fixed-``(r-s)`` NS Kac families meet at ``c=3/2``
        when the internal weight approaches ``(r-s)^2/8``.  For
        ``Delta_j = c_j-3/2`` and ``C = c-3/2``, each Laurent term is

        ``a[j,m] / (C-Delta_j)^m
          = sum_n binomial(m+n-1,n) a[j,m] Delta_j^n / C^(m+n)``.

        The sum over the family is formed at every ``n`` before division by
        ``C``.  This is the confluent-pole analogue of evaluating a common
        numerator and prevents the individually enormous ``r-s=2`` terms
        from being rounded before they cancel.  It is an expansion in pole
        displacement, not in a plumbing coordinate.
        """

        anchor = mpmath.mpc(self._CONFLUENT_ANCHOR)
        center_delta = evaluation_point - anchor
        if center_delta == 0:
            return None
        pole_deltas = [pole - anchor for pole, _, _ in entries]
        max_ratio = max(
            (abs(delta / center_delta) for delta in pole_deltas),
            default=mpmath.mpf(0),
        )
        if max_ratio > self._MAX_MOMENT_RATIO:
            return None

        value = mpmath.mpc(0)
        absolute_moment_sum = mpmath.mpf(0)
        small_terms = 0
        terms_used = 0
        # A common numerator with N Laurent coefficients can have its first
        # N-1 confluent moments vanish identically.  Do not mistake those
        # algebraic zeros for a converged tail; inspect at least three terms
        # beyond the complete coefficient count.
        minimum_terms = sum(
            len(coefficients) for _, coefficients, _ in entries
        ) + 3
        for moment_order in range(self._MAX_MOMENT_TERMS):
            term = mpmath.mpc(0)
            absolute_term_sum = mpmath.mpf(0)
            for (_, coefficients, _), pole_delta in zip(
                entries, pole_deltas
            ):
                for pole_order, coefficient in enumerate(
                    coefficients, start=1
                ):
                    summand = (
                        mpmath.binomial(
                            pole_order + moment_order - 1, moment_order
                        )
                        * coefficient
                        * pole_delta**moment_order
                        / center_delta ** (pole_order + moment_order)
                    )
                    term += summand
                    absolute_term_sum += abs(summand)
            self._record_ratio(
                "max_moment_cancellation_ratio", absolute_term_sum, term
            )
            value += term
            absolute_moment_sum += abs(term)
            terms_used = moment_order + 1
            scale = max(mpmath.mpf(1), abs(value))
            if terms_used >= minimum_terms and abs(term) <= tolerance * scale:
                small_terms += 1
            else:
                small_terms = 0
            if small_terms >= 3:
                if self.moment_diagnostics is not None:
                    self.moment_diagnostics["moment_groups"] += 1
                    self.moment_diagnostics["max_moment_terms"] = max(
                        self.moment_diagnostics["max_moment_terms"],
                        terms_used,
                    )
                    self.moment_diagnostics["max_moment_ratio"] = max(
                        self.moment_diagnostics["max_moment_ratio"],
                        max_ratio,
                    )
                    self._record_ratio(
                        "max_moment_series_cancellation_ratio",
                        absolute_moment_sum,
                        value,
                    )
                return value
        raise RuntimeError(
            "fixed-(r-s) confluent moment sum did not converge: "
            f"ratio={max_ratio!r}, terms={terms_used}"
        )

    def _regular_value_at(self, evaluation_point, tolerance, *, exclude=None):
        evaluation_point = mpmath.mpc(evaluation_point)
        groups: dict[object, list] = {}
        for entry in self.poles:
            pole, _, family_key = entry
            if exclude is not None and abs(pole - exclude) < tolerance:
                continue
            groups.setdefault(family_key, []).append(entry)

        value = mpmath.mpc(self.constant)
        absolute_group_sum = abs(value)
        for family_key, entries in groups.items():
            family_value = None
            if (
                isinstance(family_key, int)
                and family_key > 0
                and len(entries) >= 2
            ):
                family_value = self._fixed_difference_moment_value(
                    entries, evaluation_point, tolerance
                )
            if family_value is None:
                if (
                    self.moment_diagnostics is not None
                    and isinstance(family_key, int)
                    and len(entries) >= 2
                ):
                    self.moment_diagnostics["direct_groups"] += 1
                family_value = self._direct_group_value(
                    entries, evaluation_point, tolerance
                )
            value += family_value
            absolute_group_sum += abs(family_value)
        self._record_ratio(
            "max_total_cancellation_ratio", absolute_group_sum, value
        )
        return value

    def finite_part_at(self, pole, tolerance):
        pole = mpmath.mpc(pole)
        entries = [
            entry
            for entry in self.poles
            if abs(entry[0] - pole) >= tolerance
        ]
        return self.constant + self._direct_group_value(
            entries, pole, tolerance
        )

    def pole_coefficients_at(self, pole, tolerance):
        index = self._matching_index(pole, tolerance)
        if index is None:
            return ()
        return tuple(self.poles[index][1])

    def add_residue_times_laurent_at(
        self,
        *,
        pole,
        residue,
        subblock,
        tolerance,
        family_key: int | None = None,
    ) -> None:
        residue = mpmath.mpc(residue)
        self.add_pole_coefficient(
            pole,
            1,
            residue * subblock.finite_part_at(pole, tolerance),
            tolerance,
            family_key=family_key,
        )
        for order, coefficient in enumerate(
            subblock.pole_coefficients_at(pole, tolerance), start=1
        ):
            self.add_pole_coefficient(
                pole,
                order + 1,
                residue * coefficient,
                tolerance,
                family_key=family_key,
            )

    def value(self, central_charge, tolerance):
        return self._regular_value_at(central_charge, tolerance)


@dataclass(frozen=True)
class FreeSuperfieldDiagnostics:
    value: float
    chiral_log: complex
    scalar_chiral_log_real: float
    scalar_chiral_log_imag: float
    characteristic_alpha: tuple[int, int]
    characteristic_beta: tuple[int, int]
    theta_abs: float
    det_im_omega: float
    primitive_count: int
    max_word_length: int
    max_mode: int
    previous_word_relative_change: float
    fermion_method: str


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


def _theta_geometry_to_ccy_order(values: Sequence[object]) -> tuple[object, object, object]:
    """Reverse theta edge-labelled values into CCY ``(infinity,one,zero)`` slots."""

    if len(values) != 3:
        raise ValueError("theta edge ordering requires exactly three values")
    return values[2], values[1], values[0]


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
    # q_values, weights, and lifts remain in geometric (zero,one,infinity)
    # edge order throughout the graph recursion.  Only the positional inputs
    # of the CCY trinion tensor are reversed.
    ccy_occupations = _theta_geometry_to_ccy_order(occupations)
    ccy_fermions = _theta_geometry_to_ccy_order(fermions)
    ccy_weights = _theta_geometry_to_ccy_order(weights)
    rho = osp_three_point(
        n1=int(ccy_occupations[0]),
        n2=int(ccy_occupations[1]),
        n3=int(ccy_occupations[2]),
        epsilon1=int(ccy_fermions[0]),
        epsilon2=int(ccy_fermions[1]),
        epsilon3=int(ccy_fermions[2]),
        d1=ccy_weights[0],
        d2=ccy_weights[1],
        d3=ccy_weights[2],
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


def _resummed_theta_endpoint_term(
    weights: Sequence[complex],
    q_values: Sequence[complex],
    endpoint_occupations: Sequence[int],
    fermions: Sequence[int],
    lifts: Sequence[int],
) -> complex:
    r"""Sum the complete middle-edge family for fixed theta endpoints.

    Geometry order is ``(zero, one, infinity)``.  In the CCY trinion tensor,
    the middle-chain occupation appears only through

    ``falling(x, n_one)``,

    while the two-chain kernel is independent of ``n_one``.  Squaring the
    trinion and dividing by the middle-edge norm therefore produces a Gauss
    hypergeometric function with parameters ``(-x,-x;2*h_one+e_one)``.
    """

    if len(endpoint_occupations) != 2:
        raise ValueError("theta endpoint occupations must be (n_zero,n_infinity)")
    h_zero, h_one, h_infinity = (complex(value) for value in weights)
    q_zero, q_one, q_infinity = (complex(value) for value in q_values)
    n_zero, n_infinity = (int(value) for value in endpoint_occupations)
    e_zero, e_one, e_infinity = (int(value) for value in fermions)

    two_chain = osp_two_chain_kernel(
        k=n_infinity,
        m=n_zero,
        epsilon1=e_infinity,
        epsilon2=e_one,
        epsilon3=e_zero,
        d1=h_infinity,
        d2=h_one,
        d3=h_zero,
    )
    exponent = (
        h_infinity
        - h_one
        - h_zero
        + 0.5 * (e_infinity - e_one - e_zero)
        + n_infinity
        - n_zero
    )
    middle_norm_primary = 1.0 + 0.0j if e_one == 0 else 2 * h_one
    middle_family = (
        (int(lifts[1]) * cmath.sqrt(q_one)) ** e_one
        / middle_norm_primary
        * complex(
            mpmath.fp.hyp2f1(
                -exponent,
                -exponent,
                2 * h_one + e_one,
                q_one,
            )
        )
    )
    endpoint_norm = osp_norm(h_zero, n_zero, e_zero) * osp_norm(
        h_infinity, n_infinity, e_infinity
    )
    endpoint_plumbing = (
        _q_power(q_zero, n_zero, e_zero) * int(lifts[0]) ** e_zero
        * _q_power(q_infinity, n_infinity, e_infinity)
        * int(lifts[2]) ** e_infinity
    )
    orientation = (-1) ** THETA_ORIENTATION.exponent(fermions)
    return complex(
        orientation
        * endpoint_plumbing
        * two_chain**2
        / endpoint_norm
        * middle_family
    )


def resummed_theta_global_block(
    *,
    weights: Sequence[complex],
    q_values: Sequence[complex],
    sector: int,
    lifts: Sequence[int],
    tolerance: float,
    max_total_endpoint_occupation: int,
) -> GlobalSumDiagnostics:
    r"""Evaluate the theta global block with its middle edge resummed.

    The full occupation sum on the CCY ``one`` edge is evaluated by
    :math:`{}_2F_1`.  Only the two endpoint occupations remain in the
    adaptive shell sum.  At the certified five-point overlap these endpoint
    plumbing parameters are of order ``1e-9``; their last shell is recorded
    independently rather than hidden inside the c-recursion order.
    """

    if len(weights) != 3 or len(q_values) != 3 or len(lifts) != 3:
        raise ValueError("three weights, plumbing coordinates, and lifts are required")
    if sector not in (0, 1):
        raise ValueError("sector must be 0 or 1")
    if tolerance <= 0 or max_total_endpoint_occupation < 0:
        raise ValueError("positive tolerance and endpoint occupation ceiling required")
    lift_tuple = tuple(int(value) for value in lifts)
    if any(value not in (-1, 1) for value in lift_tuple):
        raise ValueError("lifts must be +/-1")
    if any(not abs(complex(q)) < 1 for q in q_values):
        raise ValueError("the theta plumbing coordinates must satisfy |q| < 1")

    total_value = 0.0 + 0.0j
    last_shell = 0.0 + 0.0j
    small_shells = 0
    converged = False
    used = 0
    for endpoint_total in range(max_total_endpoint_occupation + 1):
        shell = 0.0 + 0.0j
        for n_zero in range(endpoint_total + 1):
            n_infinity = endpoint_total - n_zero
            for fermions in (
                (e_zero, e_one, e_infinity)
                for e_zero in (0, 1)
                for e_one in (0, 1)
                for e_infinity in (0, 1)
            ):
                if sum(fermions) % 2 != sector:
                    continue
                shell += _resummed_theta_endpoint_term(
                    weights,
                    q_values,
                    (n_zero, n_infinity),
                    fermions,
                    lift_tuple,
                )
        total_value += shell
        last_shell = shell
        used = endpoint_total
        scale = max(1.0, abs(total_value))
        if endpoint_total >= 3 and abs(shell) <= tolerance * scale:
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


def resummed_theta_global_component(
    *,
    weights: Sequence[complex],
    q_values: Sequence[complex],
    fermions: Sequence[int],
    lifts: Sequence[int],
    tolerance: float,
    max_total_endpoint_occupation: int,
) -> GlobalSumDiagnostics:
    """Middle-edge-resummed theta block for one fixed fermion triple."""

    if len(fermions) != 3 or any(int(value) not in (0, 1) for value in fermions):
        raise ValueError("fermions must contain three bits")
    if len(weights) != 3 or len(q_values) != 3 or len(lifts) != 3:
        raise ValueError("three weights, plumbing coordinates, and lifts are required")
    if tolerance <= 0 or max_total_endpoint_occupation < 0:
        raise ValueError("positive tolerance and endpoint occupation ceiling required")
    lift_tuple = tuple(int(value) for value in lifts)
    if any(value not in (-1, 1) for value in lift_tuple):
        raise ValueError("lifts must be +/-1")
    if any(not abs(complex(q)) < 1 for q in q_values):
        raise ValueError("the theta plumbing coordinates must satisfy |q| < 1")
    sigma = tuple(int(value) for value in fermions)

    total_value = 0.0 + 0.0j
    last_shell = 0.0 + 0.0j
    small_shells = 0
    converged = False
    used = 0
    for endpoint_total in range(max_total_endpoint_occupation + 1):
        shell = sum(
            (
                _resummed_theta_endpoint_term(
                    weights,
                    q_values,
                    (n_zero, endpoint_total - n_zero),
                    sigma,
                    lift_tuple,
                )
                for n_zero in range(endpoint_total + 1)
            ),
            0.0 + 0.0j,
        )
        total_value += shell
        last_shell = shell
        used = endpoint_total
        scale = max(1.0, abs(total_value))
        if endpoint_total >= 3 and abs(shell) <= tolerance * scale:
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


def _resummed_glasses_handle(
    q: complex,
    lift: int,
    handle_weight: complex,
    bridge_weight: complex,
    bridge_parity: int,
) -> complex:
    r"""Resum one self-sewn handle at fixed bridge parity.

    Translation covariance factorizes the middle-chain occupation from the
    two endpoint chains.  What remains on each handle is a torus global
    :math:`\mathfrak{osp}(1|2)` block.  For even bridge parity this is the
    standard torus one-point block.  For odd bridge parity the effective
    external weight is shifted by one half and the odd handle primary carries
    the converted fixed-parity trilinear coefficient ``2h+d-1/2``.
    """

    if lift not in (-1, 1):
        raise ValueError("lift must be +1 or -1")
    if bridge_parity not in (0, 1):
        raise ValueError("bridge_parity must be 0 or 1")
    q_mp = mpmath.mpc(q)
    if not abs(q_mp) < 1:
        raise ValueError("the glasses plumbing coordinates must satisfy |q| < 1")
    h = mpmath.mpc(handle_weight)
    d = mpmath.mpc(bridge_weight)
    if bridge_parity == 0:
        return complex(_global_torus_block(q_mp, lift, h, d))

    shifted_external = d + mpmath.mpf("0.5")
    common = (1 - q_mp) ** (-shifted_external)
    even = common * mpmath.hyp2f1(
        2 * h - shifted_external,
        1 - shifted_external,
        2 * h,
        q_mp,
    )
    odd = (
        lift
        * mpmath.sqrt(q_mp)
        * (2 * h + d - mpmath.mpf("0.5"))
        / (2 * h)
        * common
        * mpmath.hyp2f1(
            2 * h + 1 - shifted_external,
            1 - shifted_external,
            2 * h + 1,
            q_mp,
        )
    )
    return complex(even + odd)


def _resummed_glasses_bridge(
    q: complex,
    lift: int,
    bridge_weight: complex,
    bridge_parity: int,
) -> complex:
    r"""Resum the separating-chain occupation at fixed fermion parity."""

    if lift not in (-1, 1):
        raise ValueError("lift must be +1 or -1")
    if bridge_parity not in (0, 1):
        raise ValueError("bridge_parity must be 0 or 1")
    q_mp = mpmath.mpc(q)
    if not abs(q_mp) < 1:
        raise ValueError("the glasses plumbing coordinates must satisfy |q| < 1")
    d = mpmath.mpc(bridge_weight)
    alpha = int(bridge_parity)
    normalization = mpmath.rf(2 * d, alpha)
    primary = (lift * mpmath.sqrt(q_mp)) ** alpha / normalization
    return complex(
        primary
        * mpmath.hyp2f1(
            d + mpmath.mpf(alpha) / 2,
            d + mpmath.mpf(alpha) / 2,
            2 * d + alpha,
            q_mp,
        )
    )


def resummed_glasses_global_block(
    *,
    weights: Sequence[complex],
    q_values: Sequence[complex],
    sector: int,
    lifts: Sequence[int],
    working_precision: int = 30,
) -> GlobalSumDiagnostics:
    r"""Evaluate the glasses global block without an occupation cutoff.

    At fixed bridge parity ``alpha=sector``, the orientation character is

    ``(-1)**(alpha*(e_left+e_right))``.

    It is therefore absorbed by flipping both handle lifts in the odd sector.
    The remaining three occupation sums factorize into one-variable kernels
    built from Gauss hypergeometric functions.  This is an exact resummation of
    :func:`direct_global_block`, not a fit or an assumption of locality.
    """

    if len(weights) != 3 or len(q_values) != 3 or len(lifts) != 3:
        raise ValueError("three weights, plumbing coordinates, and lifts are required")
    if sector not in (0, 1):
        raise ValueError("sector must be 0 or 1")
    lift_tuple = tuple(int(value) for value in lifts)
    if any(value not in (-1, 1) for value in lift_tuple):
        raise ValueError("lifts must be +/-1")
    h_left, h_right, h_bridge = (complex(value) for value in weights)
    q_left, q_right, q_bridge = (complex(value) for value in q_values)
    alpha = int(sector)
    effective_left_lift = lift_tuple[0] * (-1) ** alpha
    effective_right_lift = lift_tuple[1] * (-1) ** alpha

    with mpmath.workdps(int(working_precision)):
        value = (
            _resummed_glasses_handle(
                q_left,
                effective_left_lift,
                h_left,
                h_bridge,
                alpha,
            )
            * _resummed_glasses_handle(
                q_right,
                effective_right_lift,
                h_right,
                h_bridge,
                alpha,
            )
            * _resummed_glasses_bridge(
                q_bridge,
                lift_tuple[2],
                h_bridge,
                alpha,
            )
        )
    return GlobalSumDiagnostics(
        value=complex(value),
        last_shell=0.0 + 0.0j,
        max_total_occupation=0,
        converged=True,
    )


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
        global_method: str = "auto",
        global_tolerance: float = 2.0e-10,
        global_max_total_occupation: int = 15,
        vacuum_word_length: int = 7,
        vacuum_max_mode: int = 50,
    ) -> None:
        if channel not in {"theta", "glasses"}:
            raise ValueError("channel must be theta or glasses")
        if len(q_values) != 3 or any(not abs(complex(q)) < 1 for q in q_values):
            raise ValueError("three plumbing coordinates with |q|<1 are required")
        if global_method not in {"auto", "direct", "resummed"}:
            raise ValueError("global_method must be auto, direct, or resummed")
        self.channel = channel
        self.q_values = tuple(complex(q) for q in q_values)
        self.global_method = str(global_method)
        self.effective_global_method = (
            "resummed" if global_method != "direct" else "direct"
        )
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
        self.global_resummed_calls = 0
        self.block_calls = 0
        self.confluent_moment_groups = 0
        self.confluent_direct_groups = 0
        self.confluent_max_moment_terms = 0
        self.confluent_max_moment_ratio = 0.0
        self.confluent_max_direct_cancellation_ratio = 0.0
        self.confluent_max_moment_cancellation_ratio = 0.0
        self.confluent_max_moment_series_cancellation_ratio = 0.0
        self.confluent_max_total_cancellation_ratio = 0.0

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
        use_resummation = self.effective_global_method == "resummed"
        if use_resummation:
            if self.channel == "theta":
                diagnostics = resummed_theta_global_block(
                    weights=weights,
                    q_values=self.q_values,
                    sector=sector,
                    lifts=lifts,
                    tolerance=self.global_tolerance,
                    max_total_endpoint_occupation=(
                        self.global_max_total_occupation
                    ),
                )
            else:
                diagnostics = resummed_glasses_global_block(
                    weights=weights,
                    q_values=self.q_values,
                    sector=sector,
                    lifts=lifts,
                )
            self.global_resummed_calls += 1
        else:
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
        self.global_worst_last_shell_relative = max(
            self.global_worst_last_shell_relative,
            abs(diagnostics.last_shell) / max(1.0, abs(diagnostics.value)),
        )
        if not diagnostics.converged:
            self.global_nonconverged_calls += 1
            raise RuntimeError(
                "global block failed its pointwise convergence test: "
                f"channel={self.channel}, weights={weights!r}, "
                f"sector={sector}, q_values={self.q_values!r}, "
                f"max_occupation={diagnostics.max_total_occupation}, "
                f"last_shell={diagnostics.last_shell!r}, "
                f"value={diagnostics.value!r}"
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
        if self.channel == "theta" and self.effective_global_method == "resummed":
            diagnostics = resummed_theta_global_component(
                weights=weights,
                q_values=self.q_values,
                fermions=sigma,
                lifts=lifts,
                tolerance=self.global_tolerance,
                max_total_endpoint_occupation=self.global_max_total_occupation,
            )
            self.global_resummed_calls += 1
            self.global_max_used = max(
                self.global_max_used, diagnostics.max_total_occupation
            )
            self.global_worst_last_shell_relative = max(
                self.global_worst_last_shell_relative,
                abs(diagnostics.last_shell) / max(1.0, abs(diagnostics.value)),
            )
            if not diagnostics.converged:
                self.global_nonconverged_calls += 1
                raise RuntimeError(
                    "theta global component failed its pointwise convergence test: "
                    f"weights={weights!r}, fermions={sigma!r}, "
                    f"q_values={self.q_values!r}, "
                    f"max_occupation={diagnostics.max_total_occupation}, "
                    f"last_shell={diagnostics.last_shell!r}, "
                    f"value={diagnostics.value!r}"
                )
            return diagnostics.value

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
        self.global_worst_last_shell_relative = max(
            self.global_worst_last_shell_relative,
            abs(last_shell) / max(1.0, abs(total_value)),
        )
        if not converged:
            self.global_nonconverged_calls += 1
            raise RuntimeError(
                "direct global component failed its pointwise convergence test: "
                f"channel={self.channel}, weights={weights!r}, "
                f"fermions={sigma!r}, q_values={self.q_values!r}, "
                f"max_occupation={used}, last_shell={last_shell!r}, "
                f"value={total_value!r}"
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
        toric_sign = 1
        if self.channel == "theta":
            # Geometry edges are (zero, one, infinity), whereas the CCY
            # tensor slots are (infinity, one, zero).  These are the ordered
            # endpoint pairs obtained by mapping the CCY residue ledger back
            # to geometric edge labels.  Swapping the first two pairs happens
            # to cancel after their product, but keeping the literal order is
            # essential for a reusable genus-(g,n) convention ledger.
            other = (
                (weights[2], weights[1]),
                (weights[0], weights[2]),
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
            # A tadpole edge meets the same trinion twice.  Factoring the first
            # incidence of an odd null toggles the intermediate three-form;
            # factoring the second toggles it back, so the final child sector
            # is unchanged.  The two sequential polynomials
            # have labels alpha and alpha+rs and see the unshifted/shifted
            # copy of the self-glued handle, as in the independent torus
            # c-recursion.  The toric reflection factor is
            # S_rs^alpha = (-1)^(alpha*rs).
            bridge = weights[2]
            handle = weights[edge]
            toric_sign = (-1) ** (sector * rs)
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
            toric_sign
            * pole.jacobian
            * ns_inverse_null_slope(r, s, pole.b)
            * polynomials[0]
            * polynomials[1]
        )
        return pole.c, residue, child_sector

    def _residue_mp(
        self,
        *,
        r: int,
        s: int,
        edge: int,
        weights: tuple[complex, complex, complex],
        sector: int,
    ):
        """High-precision local residue from the shared graph recipe."""

        rs = r * s
        if self.channel == "theta":
            other = (
                (weights[2], weights[1]),
                (weights[0], weights[2]),
                (weights[0], weights[1]),
            )[edge]
            pole, residue, children = ns_ordinary_edge_scalar_kernel_mp(
                r=r,
                s=s,
                internal_weight=weights[edge],
                left_weights=other,
                right_weights=other,
                left_sector=sector,
                right_sector=sector,
            )
            if children[0] != children[1]:  # pragma: no cover
                raise AssertionError("theta endpoint sectors diverged")
            child_sector = children[0]
        elif edge in (0, 1):
            pole, residue, child_sector = ns_self_loop_scalar_kernel_mp(
                r=r,
                s=s,
                handle_weight=weights[edge],
                external_weight=weights[2],
                sector=sector,
            )
        else:
            pole, residue, children = ns_ordinary_edge_scalar_kernel_mp(
                r=r,
                s=s,
                internal_weight=weights[edge],
                left_weights=(weights[0], weights[0]),
                right_weights=(weights[1], weights[1]),
                left_sector=sector,
                right_sector=sector,
            )
            if children[0] != children[1]:  # pragma: no cover
                raise AssertionError("glasses bridge endpoint sectors diverged")
            child_sector = children[0]
        return pole.c, residue, child_sector

    def block(
        self,
        *,
        weights: Sequence[complex],
        sector: int,
        recursion_order: int,
        lifts: Sequence[int],
        central_charge: complex = C_ORDINARY_AT_HAT_C_9,
    ) -> complex:
        if recursion_order < 0 or recursion_order > MAX_RECURSION_ORDER:
            raise ValueError(
                "this evaluator supports recursion orders "
                f"0..{MAX_RECURSION_ORDER}"
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
                                "confluent c-pole requires a higher-order "
                                "Laurent recursion: "
                                f"remaining={remaining}, edge={edge}, "
                                f"(r,s)=({r},{s}), current_c={current_c!r}, "
                                f"pole_c={pole_c!r}, weights={current_weights!r}"
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

    def collision_aware_block(
        self,
        *,
        weights: Sequence[complex],
        sector: int,
        recursion_order: int,
        lifts: Sequence[int],
        central_charge: complex = C_ORDINARY_AT_HAT_C_9,
        pole_tolerance: float = 1.0e-10,
    ) -> complex:
        r"""Evaluate the block after analytically combining confluent poles.

        Each recursion node is represented as

        ``regular + sum_{p,k} coefficient[p,k] / (c-p)**k``.

        If a child pole coincides with the parent Kac pole, its Laurent
        coefficients are promoted to higher pole order before any numerical
        evaluation.  This is the NS analogue of the collision-aware bosonic
        CCY recursion and avoids subtracting separately divergent branches on
        a detuned ``b`` contour.
        """

        if recursion_order < 0 or recursion_order > MAX_RECURSION_ORDER:
            raise ValueError(
                "this evaluator supports recursion orders "
                f"0..{MAX_RECURSION_ORDER}"
            )
        if pole_tolerance <= 0:
            raise ValueError("pole_tolerance must be positive")
        weight_tuple = tuple(complex(value) for value in weights)
        lift_tuple = tuple(int(value) for value in lifts)
        if len(weight_tuple) != 3 or len(lift_tuple) != 3:
            raise ValueError("three weights and three lifts are required")
        if any(value not in (-1, 1) for value in lift_tuple):
            raise ValueError("lifts must be +/-1")

        @lru_cache(maxsize=None)
        def recurse(
            remaining: int,
            current_weights: tuple[complex, complex, complex],
            current_sector: int,
            current_lifts: tuple[int, int, int],
        ) -> PartialFractionInC:
            self.block_calls += 1
            total = PartialFractionInC(
                constant=self._regular(
                    current_weights, current_sector, current_lifts
                )
            )
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
                        child = recurse(
                            remaining - rs,
                            tuple(shifted),  # type: ignore[arg-type]
                            child_sector,
                            tuple(child_lifts),
                        )
                        total.add_residue_times_laurent_at(
                            pole=pole_c,
                            residue=(
                                residue
                                * complex(self.q_values[edge]) ** (rs / 2.0)
                                * current_lifts[edge] ** (rs % 2)
                                * orientation_constant
                            ),
                            subblock=child,
                            pole_tolerance=pole_tolerance,
                        )
            return total

        partial_fraction = recurse(
            int(recursion_order),
            weight_tuple,  # type: ignore[arg-type]
            int(sector),
            lift_tuple,  # type: ignore[arg-type]
        )
        return partial_fraction.value(
            complex(central_charge), pole_tolerance=pole_tolerance
        )

    def collision_aware_block_mp(
        self,
        *,
        weights: Sequence[complex],
        sector: int,
        recursion_order: int,
        lifts: Sequence[int],
        central_charge: complex = C_ORDINARY_AT_HAT_C_9,
        working_precision: int = 60,
    ) -> complex:
        """Arbitrary-precision collision-aware NS c-recursion."""

        if recursion_order < 0 or recursion_order > MAX_RECURSION_ORDER:
            raise ValueError(
                "this evaluator supports recursion orders "
                f"0..{MAX_RECURSION_ORDER}"
            )
        if working_precision < 30:
            raise ValueError("working_precision must be at least 30 digits")
        weight_tuple = tuple(complex(value) for value in weights)
        lift_tuple = tuple(int(value) for value in lifts)
        if len(weight_tuple) != 3 or len(lift_tuple) != 3:
            raise ValueError("three weights and three lifts are required")
        if any(value not in (-1, 1) for value in lift_tuple):
            raise ValueError("lifts must be +/-1")

        with mpmath.workdps(int(working_precision)):
            tolerance = mpmath.mpf(10) ** (-(int(working_precision) // 2))
            moment_diagnostics = {
                "moment_groups": 0,
                "direct_groups": 0,
                "max_moment_terms": 0,
                "max_moment_ratio": mpmath.mpf(0),
                "max_direct_cancellation_ratio": mpmath.mpf(0),
                "max_moment_cancellation_ratio": mpmath.mpf(0),
                "max_moment_series_cancellation_ratio": mpmath.mpf(0),
                "max_total_cancellation_ratio": mpmath.mpf(0),
            }

            @lru_cache(maxsize=None)
            def recurse(
                remaining: int,
                current_weights: tuple[complex, complex, complex],
                current_sector: int,
                current_lifts: tuple[int, int, int],
            ) -> _MPPartialFractionInC:
                self.block_calls += 1
                total = _MPPartialFractionInC(
                    self._regular(
                        current_weights, current_sector, current_lifts
                    ),
                    moment_diagnostics=moment_diagnostics,
                )
                for edge in range(3):
                    for r in range(2, remaining + 1):
                        for s in range(1, remaining // r + 1):
                            rs = r * s
                            if rs > remaining or (r + s) % 2:
                                continue
                            pole_c, residue, child_sector = self._residue_mp(
                                r=r,
                                s=s,
                                edge=edge,
                                weights=current_weights,
                                sector=current_sector,
                            )
                            shifted = list(current_weights)
                            shifted[edge] += rs / 2.0
                            if rs % 2:
                                child_lifts = _transport_lifts(
                                    self.orientation, current_lifts, edge
                                )
                                orientation_constant = (
                                    -1
                                ) ** self.orientation.exponent(_unit(edge))
                            else:
                                child_lifts = current_lifts
                                orientation_constant = 1
                            child = recurse(
                                remaining - rs,
                                tuple(shifted),  # type: ignore[arg-type]
                                child_sector,
                                tuple(child_lifts),
                            )
                            total.add_residue_times_laurent_at(
                                pole=pole_c,
                                residue=(
                                    residue
                                    * mpmath.mpc(self.q_values[edge])
                                    ** (mpmath.mpf(rs) / 2)
                                    * current_lifts[edge] ** (rs % 2)
                                    * orientation_constant
                                ),
                                subblock=child,
                                tolerance=tolerance,
                                family_key=r - s,
                            )
                return total

            partial_fraction = recurse(
                int(recursion_order),
                weight_tuple,  # type: ignore[arg-type]
                int(sector),
                lift_tuple,  # type: ignore[arg-type]
            )
            value = partial_fraction.value(
                mpmath.mpc(central_charge), tolerance
            )
            self.confluent_moment_groups += int(
                moment_diagnostics["moment_groups"]
            )
            self.confluent_direct_groups += int(
                moment_diagnostics["direct_groups"]
            )
            self.confluent_max_moment_terms = max(
                self.confluent_max_moment_terms,
                int(moment_diagnostics["max_moment_terms"]),
            )
            self.confluent_max_moment_ratio = max(
                self.confluent_max_moment_ratio,
                float(moment_diagnostics["max_moment_ratio"]),
            )
            self.confluent_max_direct_cancellation_ratio = max(
                self.confluent_max_direct_cancellation_ratio,
                float(moment_diagnostics["max_direct_cancellation_ratio"]),
            )
            self.confluent_max_moment_cancellation_ratio = max(
                self.confluent_max_moment_cancellation_ratio,
                float(moment_diagnostics["max_moment_cancellation_ratio"]),
            )
            self.confluent_max_moment_series_cancellation_ratio = max(
                self.confluent_max_moment_series_cancellation_ratio,
                float(
                    moment_diagnostics[
                        "max_moment_series_cancellation_ratio"
                    ]
                ),
            )
            self.confluent_max_total_cancellation_ratio = max(
                self.confluent_max_total_cancellation_ratio,
                float(moment_diagnostics["max_total_cancellation_ratio"]),
            )
            return complex(value)

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

        if recursion_order < 0 or recursion_order > MAX_RECURSION_ORDER:
            raise ValueError(
                "finite-part recursion order must lie in "
                f"0..{MAX_RECURSION_ORDER}"
            )
        if len(momenta) != 3:
            raise ValueError("three continuum momenta are required")
        if not radius > 0 or samples < 4:
            raise ValueError("finite-part radius must be positive and samples >= 4")
        if samples < 2 * recursion_order:
            raise ValueError(
                "finite-part angular sampling must satisfy "
                "samples >= 2 * recursion_order to prevent Laurent-harmonic "
                f"aliasing; received samples={samples}, "
                f"recursion_order={recursion_order}"
            )
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


def _free_scalar_chiral_log(
    generators,
    *,
    max_word_length: int,
    max_mode: int,
) -> tuple[complex, int]:
    r"""Raw chiral oscillator log for one noncompact free scalar.

    This is the exact Zograf/Heisenberg primitive product in the plumbing
    frame.  It is kept separate from the legacy half-multiplier product:
    beyond genus one the Majorana determinant is fixed by bosonization and is
    not obtained by simply appending one fermionic factor per primitive word.
    """

    value = 0.0j
    words = _primitive_conjugacy_words_cached(
        len(generators), int(max_word_length)
    )
    for word in words:
        multiplier = word_multiplier(generators, word)
        if not abs(multiplier) < 1:
            raise ValueError("non-loxodromic primitive multiplier")
        for mode in range(1, int(max_mode) + 1):
            value -= cmath.log(1.0 - multiplier**mode)
    return value, len(words)


def _spin_characteristic_from_lifts(
    channel: str,
    q_values: Sequence[complex],
    physical_lifts: Sequence[int],
) -> tuple[tuple[int, int], tuple[int, int]]:
    r"""Return the even genus-two characteristic in the channel period basis.

    All computations here use NS representations on the Schottky A-cycles,
    hence ``alpha=(0,0)``.  The beta bits are determined by the chosen
    determinant-one generator lifts and by the BPZ half-edge frame.  In the
    glasses marking ``+`` is beta zero.  In the fixed branch-composed theta
    marking the independent direct Majorana pants-sewing oracle gives

    ``(s1,s2)=(+,+),(+,-),(-,+),(-,-)``
    ``-> beta=(1,1),(1,0),(0,1),(0,0)``.

    The first theta bit contains the BPZ affine shift that was absent from the
    older generator-sign-only ledger.  Consequently physical theta lifts
    ``(-,+,+)`` (equivalently ``(+,-,-)``) realize ``[00|00]`` and match
    glasses lifts ``(+,+,+)``.  Theta ``(+,+,+)`` instead realizes
    ``[00|10]``.
    """

    lifts = tuple(int(value) for value in physical_lifts)
    if len(lifts) != 3 or any(value not in (-1, 1) for value in lifts):
        raise ValueError("three physical plumbing lifts, each +/-1, are required")
    if channel == "glasses":
        generator_signs = (lifts[0], lifts[1])
        beta = tuple(int(sign < 0) for sign in generator_signs)
    elif channel == "theta":
        _, generator_signs = _theta_schottky_data(q_values, lifts)
        beta = (
            int(generator_signs[0] > 0),
            int(generator_signs[1] > 0),
        )
    else:
        raise ValueError("channel must be theta or glasses")
    return (0, 0), beta  # type: ignore[return-value]


def _transport_spin_characteristic(
    symplectic_matrix: Sequence[Sequence[int]],
    characteristic: tuple[tuple[int, int], tuple[int, int]],
) -> tuple[tuple[int, int], tuple[int, int]]:
    r"""Transport a genus-two half-characteristic with the affine action.

    The period convention is ``Omega'=(A Omega+B)(C Omega+D)^(-1)``.  For a
    binary characteristic ``[alpha|beta]`` the corresponding action is

    ``alpha' = D alpha - C beta + diag(C D^T)``,
    ``beta'  =-B alpha + A beta + diag(A B^T)`` modulo two.

    Keeping the diagonal affine terms here is essential: a purely linear
    transport can silently select a different spin structure.
    """

    matrix = np.asarray(symplectic_matrix, dtype=int)
    if matrix.shape != (4, 4):
        raise ValueError("genus-two spin transport requires a 4x4 matrix")
    identity = np.eye(2, dtype=int)
    symplectic_form = np.block(
        [
            [np.zeros((2, 2), dtype=int), identity],
            [-identity, np.zeros((2, 2), dtype=int)],
        ]
    )
    if not np.array_equal(matrix.T @ symplectic_form @ matrix, symplectic_form):
        raise ValueError("spin transport matrix is not symplectic")

    A, B = matrix[:2, :2], matrix[:2, 2:]
    C, D = matrix[2:, :2], matrix[2:, 2:]
    alpha = np.asarray(characteristic[0], dtype=int)
    beta = np.asarray(characteristic[1], dtype=int)
    if alpha.shape != (2,) or beta.shape != (2,):
        raise ValueError("genus-two characteristic must have two alpha and beta bits")
    transported_alpha = (
        D @ alpha - C @ beta + np.diag(C @ D.T)
    ) % 2
    transported_beta = (
        -B @ alpha + A @ beta + np.diag(A @ B.T)
    ) % 2
    return (
        tuple(int(value) for value in transported_alpha),
        tuple(int(value) for value in transported_beta),
    )  # type: ignore[return-value]


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
        generators, _ = _theta_schottky_data(
            q_values, physical_lifts
        )
    elif channel == "glasses":
        generators = generators_for_glasses(*q_values)
    else:
        raise ValueError("channel must be theta or glasses")
    scalar_log, primitive_count = _free_scalar_chiral_log(
        generators,
        max_word_length=max_word_length,
        max_mode=max_mode,
    )
    previous_scalar_log, _ = _free_scalar_chiral_log(
        generators,
        max_word_length=max(1, int(max_word_length) - 1),
        max_mode=max_mode,
    )
    omega = np.asarray(omega_entries, dtype=np.complex128).reshape(2, 2)
    det_y = float(np.linalg.det(omega.imag))
    if det_y <= 0:
        raise ValueError("period matrix must have positive Im determinant")
    characteristic = _spin_characteristic_from_lifts(
        channel, q_values, physical_lifts
    )
    theta = riemann_theta_constant_genus2(
        omega, characteristic, tol=1.0e-14
    )
    if theta == 0.0:
        raise ValueError("the selected spin structure has a fermion zero mode")

    # Bosonization for a complex fermion gives Z_psi^2 = theta_delta Z_M.
    # One real Majorana therefore contributes sqrt(theta_delta P_bos) in the
    # chiral theory.  Multiplying by the real scalar and both chiralities gives
    #
    #   Z_(X+psi) = det(Im Omega)^(-1/2) |theta_delta| |P_bos|^3.
    chiral_log = 1.5 * scalar_log + 0.5 * cmath.log(theta)
    previous_chiral_log = (
        1.5 * previous_scalar_log + 0.5 * cmath.log(theta)
    )
    value = det_y**-0.5 * math.exp(2.0 * chiral_log.real)
    previous_value = det_y**-0.5 * math.exp(2.0 * previous_chiral_log.real)
    return FreeSuperfieldDiagnostics(
        value=float(value),
        chiral_log=chiral_log,
        scalar_chiral_log_real=float(scalar_log.real),
        scalar_chiral_log_imag=float(scalar_log.imag),
        characteristic_alpha=characteristic[0],
        characteristic_beta=characteristic[1],
        theta_abs=float(abs(theta)),
        det_im_omega=det_y,
        primitive_count=primitive_count,
        max_word_length=int(max_word_length),
        max_mode=int(max_mode),
        previous_word_relative_change=float(
            abs(value - previous_value) / max(1.0e-300, abs(value))
        ),
        fermion_method="exact theta-function bosonization resummation",
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

    if tuple(reversed(THETA_GEOMETRY_EDGE_ORDER)) != THETA_CCY_DESCENDANT_EDGE_ORDER:
        raise AssertionError("theta geometry/CCY descendant edge ledger changed")

    theta_weights = (0.71, 1.23, 0.94)
    theta_q = (0.073 + 0.004j, 0.121 - 0.006j, 0.097 + 0.003j)
    theta_occupations = (0, 1, 0)
    theta_fermions = (0, 0, 0)
    theta_lifts = (1, 1, -1)
    theta_term = _theta_global_term(
        theta_weights,
        theta_q,
        theta_occupations,
        theta_fermions,
        theta_lifts,
    )
    ccy_occupations = _theta_geometry_to_ccy_order(theta_occupations)
    ccy_fermions = _theta_geometry_to_ccy_order(theta_fermions)
    ccy_weights = _theta_geometry_to_ccy_order(theta_weights)
    correct_rho = osp_three_point(
        n1=int(ccy_occupations[0]),
        n2=int(ccy_occupations[1]),
        n3=int(ccy_occupations[2]),
        epsilon1=int(ccy_fermions[0]),
        epsilon2=int(ccy_fermions[1]),
        epsilon3=int(ccy_fermions[2]),
        d1=ccy_weights[0],
        d2=ccy_weights[1],
        d3=ccy_weights[2],
    )
    wrong_rho = osp_three_point(
        n1=theta_occupations[0],
        n2=theta_occupations[1],
        n3=theta_occupations[2],
        epsilon1=theta_fermions[0],
        epsilon2=theta_fermions[1],
        epsilon3=theta_fermions[2],
        d1=theta_weights[0],
        d2=theta_weights[1],
        d3=theta_weights[2],
    )
    theta_denominator = math.prod(
        osp_norm(weight, occupation, fermion)
        for weight, occupation, fermion in zip(
            theta_weights, theta_occupations, theta_fermions
        )
    )
    theta_plumbing = math.prod(
        _q_power(q, occupation, fermion) * lift**fermion
        for q, occupation, fermion, lift in zip(
            theta_q, theta_occupations, theta_fermions, theta_lifts
        )
    )
    theta_expected = (
        (-1) ** THETA_ORIENTATION.exponent(theta_fermions)
        * theta_plumbing
        * correct_rho
        * correct_rho
        / theta_denominator
    )
    theta_wrong = (
        (-1) ** THETA_ORIENTATION.exponent(theta_fermions)
        * theta_plumbing
        * wrong_rho
        * wrong_rho
        / theta_denominator
    )
    theta_order_error = abs(theta_term - theta_expected) / max(
        1.0, abs(theta_expected)
    )
    theta_old_order_displacement = abs(theta_wrong - theta_expected) / max(
        1.0, abs(theta_expected)
    )
    if theta_order_error > 2.0e-14:
        raise AssertionError("theta descendant tensor did not receive CCY slot order")
    if theta_old_order_displacement < 1.0e-2:
        raise AssertionError("theta descendant-order regression sample is not discriminating")

    theta_resummation_errors = []
    theta_resummation_endpoint_shells = []
    for alpha in (0, 1):
        theta_direct = direct_global_block(
            channel="theta",
            weights=theta_weights,
            q_values=theta_q,
            sector=alpha,
            lifts=theta_lifts,
            tolerance=2.0e-13,
            max_total_occupation=40,
        )
        theta_resummed = resummed_theta_global_block(
            weights=theta_weights,
            q_values=theta_q,
            sector=alpha,
            lifts=theta_lifts,
            tolerance=2.0e-13,
            max_total_endpoint_occupation=40,
        )
        error = abs(theta_direct.value - theta_resummed.value) / max(
            1.0, abs(theta_resummed.value)
        )
        if (
            not theta_direct.converged
            or not theta_resummed.converged
            or error > 2.0e-11
        ):
            raise AssertionError(
                "theta middle-edge hypergeometric resummation failed: "
                f"sector={alpha} error={error:.3e}"
            )
        theta_resummation_errors.append(float(error))
        theta_resummation_endpoint_shells.append(
            int(theta_resummed.max_total_occupation)
        )

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

    # Validate the complete nonseparating resummation, including the odd
    # bridge sector and its induced flips of both handle lifts, against the
    # independent occupation-shell implementation.
    resummation_errors = []
    q_probe = (q_left, q_right, 0.047 + 0.003j)
    for alpha in (0, 1):
        direct_probe = direct_global_block(
            channel="glasses",
            weights=(h_left, h_right, h_bridge),
            q_values=q_probe,
            sector=alpha,
            lifts=(-1, 1, -1),
            tolerance=2.0e-13,
            max_total_occupation=24,
        )
        resummed_probe = resummed_glasses_global_block(
            weights=(h_left, h_right, h_bridge),
            q_values=q_probe,
            sector=alpha,
            lifts=(-1, 1, -1),
        )
        error = abs(direct_probe.value - resummed_probe.value) / max(
            1.0, abs(resummed_probe.value)
        )
        if not direct_probe.converged or error > 2.0e-11:
            raise AssertionError(
                "glasses global hypergeometric resummation failed: "
                f"sector={alpha} converged={direct_probe.converged} error={error:.3e}"
            )
        resummation_errors.append(float(error))

    # The self-sewn handle residue must reduce to the independently checked
    # sequential torus c-recursion residue.  Check both an odd and an even
    # null: the former catches the intermediate sector toggle and toric sign,
    # while the latter still requires the shifted h + rs/2 argument.
    from compare_ns_torus_c_h_recursion import (
        _c_pole,
        _ns_a_factor,
        _ns_ns_fusion_polynomial,
    )

    recursion = NSGenus2CRecursion(
        channel="glasses",
        q_values=(q_left, q_right, 1.0e-3),
    )
    handle_residue_errors: dict[str, float] = {}
    for r, s in ((3, 1), (2, 2)):
        rs = r * s
        for sector in (0, 1):
            pole_c, handle_residue, child_sector = recursion._residue(
                r=r,
                s=s,
                edge=0,
                weights=(h_left, h_right, h_bridge),
                sector=sector,
            )
            torus_b, torus_c, torus_jacobian = _c_pole(h_left, r, s)
            torus_residue = (
                (-1 if rs % 2 else 1)
                * (-1) ** (sector * rs)
                * torus_jacobian
                * _ns_a_factor(torus_b, r, s)
                * _ns_ns_fusion_polynomial(
                    b=torus_b,
                    r=r,
                    s=s,
                    lower_weight=h_left + rs / 2.0,
                    upper_weight=h_bridge,
                    starred=bool(sector ^ (rs % 2)),
                )
                * _ns_ns_fusion_polynomial(
                    b=torus_b,
                    r=r,
                    s=s,
                    lower_weight=h_left,
                    upper_weight=h_bridge,
                    starred=bool(sector),
                )
            )
            error = max(
                abs(pole_c - complex(torus_c)),
                abs(handle_residue - complex(torus_residue)),
            ) / max(1.0, abs(complex(torus_residue)))
            key = f"{r},{s},sector={sector}"
            handle_residue_errors[key] = float(error)
            if error > 2.0e-12 or child_sector != sector:
                raise AssertionError(
                    "glasses handle residue failed the sequential torus reduction: "
                    f"(r,s)=({r},{s}), sector={sector}"
                )
    handle_residue_error = max(handle_residue_errors.values())

    # Use the complete glasses-to-theta word, including the final integer
    # branch transformation used by the stored theta period matrices.  It
    # maps the selected glasses characteristic [00|00] to [00|00].  The
    # independent Majorana pants-sewing oracle fixes which theta plumbing
    # lifts realize that transported characteristic: (-,+,+), not (+,+,+).
    unbranched_symplectic_matrix = np.asarray(
        GLASSES_TO_THETA_UNBRANCHED, dtype=int
    )
    theta_integer_branch = np.asarray(THETA_INTEGER_BRANCH, dtype=int)
    identity = np.eye(2, dtype=int)
    branch_symplectic_matrix = np.block(
        [
            [identity, theta_integer_branch],
            [np.zeros((2, 2), dtype=int), identity],
        ]
    )
    symplectic_matrix = np.asarray(
        GLASSES_TO_THETA_BRANCH_COMPOSED, dtype=int
    )
    if not np.array_equal(
        branch_symplectic_matrix @ unbranched_symplectic_matrix,
        symplectic_matrix,
    ):
        raise AssertionError("theta integer branch does not compose the modular word")
    source_characteristic = ((0, 0), (0, 0))
    target_characteristic = _transport_spin_characteristic(
        symplectic_matrix, source_characteristic
    )
    if target_characteristic != ((0, 0), (0, 0)):
        raise AssertionError("spin characteristic transport changed")
    same_spin_theta_lifts = (-1, 1, 1)
    _, theta_generator_signs = _theta_schottky_data(
        (0.11, 0.12, 0.13), same_spin_theta_lifts
    )
    if theta_generator_signs != (-1, -1):
        raise AssertionError("theta edge lifts do not realize [00|00]")
    theta_characteristic = _spin_characteristic_from_lifts(
        "theta", (0.11, 0.12, 0.13), same_spin_theta_lifts
    )
    glasses_characteristic = _spin_characteristic_from_lifts(
        "glasses", (0.11, 0.12, 0.13), (1, 1, 1)
    )
    if theta_characteristic != ((0, 0), (0, 0)):
        raise AssertionError("theta edge lifts have the wrong characteristic")
    if glasses_characteristic != ((0, 0), (0, 0)):
        raise AssertionError("glasses edge lifts have the wrong characteristic")
    stale_theta_characteristic = _spin_characteristic_from_lifts(
        "theta", (0.11, 0.12, 0.13), (1, 1, 1)
    )
    if stale_theta_characteristic != ((0, 0), (1, 0)):
        raise AssertionError("theta BPZ affine spin shift changed")
    return {
        "theta_geometry_edge_order": list(THETA_GEOMETRY_EDGE_ORDER),
        "theta_ccy_descendant_edge_order": list(
            THETA_CCY_DESCENDANT_EDGE_ORDER
        ),
        "theta_descendant_order_relative_error": float(theta_order_error),
        "theta_old_order_relative_displacement": float(
            theta_old_order_displacement
        ),
        "theta_global_method": "middle-edge hypergeometric resummation",
        "theta_resummation_direct_relative_errors": theta_resummation_errors,
        "theta_resummation_endpoint_shells": theta_resummation_endpoint_shells,
        "glasses_geometry_edge_order": list(GLASSES_GEOMETRY_EDGE_ORDER),
        "glasses_ccy_descendant_edge_order": list(
            GLASSES_CCY_DESCENDANT_EDGE_ORDER
        ),
        "glasses_orientation_bits": list(GLASSES_ORIENTATION.edge_linear_bits),
        "glasses_orientation_formula": "e_bridge*(e_left+e_right) mod 2",
        "same_spin_theta_lifts": list(same_spin_theta_lifts),
        "same_spin_glasses_lifts": [1, 1, 1],
        "same_spin_characteristic": {
            "alpha": [0, 0],
            "beta": [0, 0],
        },
        "separating_global_relative_error": float(factorization_error),
        "separating_global_sum_converged": global_glasses.converged,
        "glasses_global_method": "three-factor hypergeometric resummation",
        "glasses_resummation_direct_relative_errors": resummation_errors,
        "handle_residue_torus_relative_error": float(handle_residue_error),
        "handle_residue_torus_relative_errors": handle_residue_errors,
        "spin_source_characteristic": {"alpha": [0, 0], "beta": [0, 0]},
        "spin_target_characteristic": {"alpha": [0, 0], "beta": [0, 0]},
        "glasses_edge_lifts": [1, 1, 1],
        "theta_edge_lifts": list(same_spin_theta_lifts),
        "theta_integer_branch": theta_integer_branch.tolist(),
        "unbranched_symplectic_matrix": unbranched_symplectic_matrix.tolist(),
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
    # The two channel lifts must realize the same intrinsic spin structure.
    # The BPZ affine shift in the branch-composed theta frame makes (-,+,+),
    # rather than the formerly used (+,+,-), the [00|00] representative.
    lifts = {"theta": (-1, 1, 1), "glasses": (1, 1, 1)}
    spin_characteristics = {
        channel: _spin_characteristic_from_lifts(
            channel, q_values[channel], lifts[channel]
        )
        for channel in ("theta", "glasses")
    }
    if len(set(spin_characteristics.values())) != 1:
        raise RuntimeError(
            "theta and glasses plumbing lifts select different spin structures"
        )
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
        "central_charge_convention": "ordinary c; hat_c=2c/3",
        "central_charge": C_ORDINARY_AT_HAT_C_9,
        "hat_c": HAT_C_TARGET,
        "source_csv": str(args.overlap_csv),
        "source_row": source_subset,
        "physical_lifts": {channel: list(value) for channel, value in lifts.items()},
        "spin_characteristics": {
            channel: {
                "alpha": list(characteristic[0]),
                "beta": list(characteristic[1]),
            }
            for channel, characteristic in spin_characteristics.items()
        },
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
                "recursion order is the only conformal recursion cutoff; the theta "
                "endpoint sum and primitive products have separately reported numerical "
                "convergence controls, while the glasses global block is fully resummed"
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
