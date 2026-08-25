#!/usr/bin/env python3
r"""Exploratory BRY-convention all-NS sphere five-tachyon integrand.

This module is deliberately matrix-model blind.  It constructs the genus-zero
five-point matter density for the Type-0B tachyon process

``T^+_(omega_in) -> T^-_(omega_1) ... T^-_(omega_4)``

with ``omega_in=sum(omega_i)``.  Labels are fixed in the gauge

``(z_0,z_1,z_2,z_3,z_4)=(infinity,1,0,z,w)``.

The three fixed labels 0, 1, and 2 are raised to picture zero.  In BRY
conventions their
holomorphic matter insertion is

``U_i = Lambda_i - k_i psi_i V_i``,

where ``k=(omega_in,-omega_1,...,-omega_4)`` is the signed timelike momentum.
Only zero or two free-fermion selections survive in each chiral half.  The
remaining Liouville components are evaluated by the multipoint fixed-weight
``c``-recursion, including independent holomorphic and antiholomorphic
``G_-1/2`` markings.

The returned density is the PCO component sum ``I_NS(z,w)`` normalized so
that the BRY-style inference from the all-tachyon diagram to the full RHS
amplitude would read

``A_(1R->4R) = (i/4) g_s^5 C_(S2) delta(E) integral d2z d2w I_NS``.

That last inference requires equality of the sixteen even-axion diagrams and
is not used anywhere in this file.  The literal all-tachyon contribution is
``i/64`` times the same integral before the common topology and delta
factors.  Matrix-model comparison belongs in a separate post-freeze module.
"""

from __future__ import annotations

import cmath
from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations, permutations, product
import math
from pathlib import Path
import sys
from typing import Iterable, Mapping, Sequence, Union

import mpmath
import numpy as np
from scipy.stats import qmc


SCRIPT_DIR = Path(__file__).resolve().parent
CODE_ROOT = SCRIPT_DIR.parents[1]
C_RECURSION = CODE_ROOT / "c_Recursion"
REFERENCE_PLUMBING = (
    CODE_ROOT
    / "bosonic_c1_one_to_n_reference"
    / "reference_implementation"
    / "plumbing"
)
for dependency in (SCRIPT_DIR, C_RECURSION, REFERENCE_PLUMBING):
    if str(dependency) not in sys.path:
        sys.path.insert(0, str(dependency))

from ns_multipoint_c_recursion import NSSphereLinearCRecursion  # noqa: E402
from sphere_five_point_liouville import (  # noqa: E402
    INFINITY,
    LinearChannel,
    ProjectivePoint,
    best_linear_channels,
    linear_channel_complex_jacobian_to_chart,
    linear_channel_from_ordering,
    linear_channel_positions_by_label,
    mobius_to_zero_one_infinity,
    oriented_tree_orderings,
)
from super_liouville_structure_constants import (  # noqa: E402
    ns_structure_constant,
    ns_structure_constant_mp,
    ns_tilde_structure_constant,
    ns_tilde_structure_constant_mp,
)


Number = Union[complex, float]
SectorAssignment = tuple[int, int, int]

# One global gauge/picture convention.  Keeping these labels centralized is
# important: a relabelling changes not only the PCO component sum but also the
# superghost correlator and the picture-dependent powers on every boundary of
# Mbar_0,5.
FIXED_INFINITY_LABEL = 0
FIXED_ONE_LABEL = 1
FIXED_ZERO_LABEL = 2
MOVING_LABELS = (3, 4)
PICTURE_ZERO_LABELS = (0, 1, 2)
MINUS_ONE_LABELS = (3, 4)
MINIMAL_SUBTRACTION_T_MAX = (25.0 + math.sqrt(545.0)) / 80.0
MINIMAL_SUBTRACTION_OPTIMAL_T = (1.0 + math.sqrt(2.0)) / 4.0

# Channel-to-channel proposal-density ratios may use any common affine
# reference chart.  This reference keeps a non-degenerate determinant in the
# deep-corner regression where the physical fixed triple (0,1,2) degenerates;
# it does not change the physical gauge or picture assignment.
ATLAS_REFERENCE_FIXED_INFINITY_LABEL = 0
ATLAS_REFERENCE_FIXED_ONE_LABEL = 1
ATLAS_REFERENCE_FIXED_ZERO_LABEL = 4
ATLAS_REFERENCE_MOVING_LABELS = (2, 3)

# Component-to-scalar phases at the two ends of the standard comb.  They are
# named constants so arbitrary-channel crossing, rather than reversal alone,
# can pin the common BPZ convention.
STANDARD_ZERO_DESCENDANT_PHASE = -1.0 + 0.0j
STANDARD_INFINITY_DESCENDANT_PHASE = -1.0j


def _finite_complex(name: str, value: Number) -> complex:
    result = complex(value)
    if not math.isfinite(result.real) or not math.isfinite(result.imag):
        raise ValueError(f"{name} must be finite")
    return result


def _validate_positions(
    positions: Sequence[ProjectivePoint],
) -> tuple[ProjectivePoint, ...]:
    if len(positions) != 5:
        raise ValueError("positions must contain five projective punctures")
    normalized = tuple(
        None if value is None else _finite_complex(f"positions[{index}]", value)
        for index, value in enumerate(positions)
    )
    homogeneous = [
        (1.0 + 0.0j, 0.0 + 0.0j)
        if value is None
        else (complex(value), 1.0 + 0.0j)
        for value in normalized
    ]
    for left in range(5):
        for right in range(left):
            determinant = (
                homogeneous[left][0] * homogeneous[right][1]
                - homogeneous[left][1] * homogeneous[right][0]
            )
            # Boundary-focused importance sampling legitimately produces
            # separations far below 1e-14.  Reject only an actual floating-
            # point collision; a tolerance here cuts out a finite and, for
            # small convergence margins, numerically important collar.
            if determinant == 0.0:
                raise ValueError("sphere punctures must be pairwise distinct")
    return normalized


def _odd_sector_assignments() -> tuple[SectorAssignment, ...]:
    return tuple(
        assignment
        for assignment in product((0, 1), repeat=3)
        if sum(assignment) % 2 == 1
    )


ODD_SECTOR_ASSIGNMENTS = _odd_sector_assignments()


def pco_safe_linear_channels(
    positions: Sequence[ProjectivePoint],
) -> tuple[LinearChannel, ...]:
    r"""Return convergent combs with the two unraised fields at the endpoints.

    Endpoint ``G_-1/2`` components require an additional BPZ spin-frame
    convention that is absent from the scalar BRY four-point blocks.  The
    physical gauge puts the two unraised integrated fields at labels 3 and 4,
    so no such convention is needed: order the three raised fixed fields in
    a comb ``(3,a,b,c,4)``.  One of the six permutations has
    ``|q1|<=1`` and ``|q2|<=1`` everywhere away from equal-radius walls.
    """

    normalized = _validate_positions(positions)
    candidates = tuple(
        linear_channel_from_ordering(normalized, (3, *middle, 4))
        for middle in permutations(PICTURE_ZERO_LABELS)
    )
    convergent = tuple(channel for channel in candidates if channel.score < 1.0)
    if not convergent:
        raise RuntimeError(
            "no PCO-safe five-point comb lies strictly inside the plumbing bidisc"
        )
    return tuple(sorted(convergent, key=lambda item: item.score))


def incoming_endpoint_linear_channels(
    positions: Sequence[ProjectivePoint],
) -> tuple[LinearChannel, ...]:
    """Return convergent combs with the incoming label 0 at the left endpoint."""

    normalized = _validate_positions(positions)
    candidates = tuple(
        linear_channel_from_ordering(normalized, (0, *tail))
        for tail in permutations((1, 2, 3, 4))
    )
    convergent = tuple(channel for channel in candidates if channel.score < 1.0)
    if not convergent:
        raise RuntimeError(
            "no incoming-endpoint five-point comb lies inside the plumbing bidisc"
        )
    return tuple(sorted(convergent, key=lambda item: item.score))


def incoming_endpoint_tree_orderings() -> tuple[tuple[int, ...], ...]:
    """Return oriented combs in which the incoming label is not the middle leaf."""

    return tuple(
        ordering for ordering in oriented_tree_orderings() if ordering.index(0) != 2
    )


@dataclass(frozen=True)
class PCOChiralTerm:
    """One free-fermion/Liouville-component term in three raised vertices."""

    timelike_labels: tuple[int, ...]
    liouville_descendants: tuple[int, int, int, int, int]
    coefficient: complex


@dataclass(frozen=True)
class CrossedNSStructurePole:
    """One simple endpoint pole crossed by an NS structure constant."""

    family: str
    momentum: complex
    wall: float
    contour_coefficient: complex = -2.0j
    initial_momentum: complex | None = None
    crossing_parameter: float = 0.0


@dataclass(frozen=True)
class ContinuedMomentumDensity:
    """Continuous and residue strata at one point of ``M_0,5``."""

    continuous: complex
    left_residues: complex
    right_residues: complex
    nested_residues: complex
    # Diagnostic subset already included in ``right_residues``.
    middle_line_residues: complex = 0.0 + 0.0j

    @property
    def total(self) -> complex:
        return (
            self.continuous
            + self.left_residues
            + self.right_residues
            + self.nested_residues
        )


@dataclass(frozen=True)
class MovingMiddleResidueTerm:
    """One quadrature-node contribution from a moving middle-trinion pole."""

    first_momentum: complex
    second_pole: CrossedNSStructurePole
    sectors: SectorAssignment
    value: complex


@dataclass(frozen=True)
class MovingMiddleCornerTerm:
    """Leading double-plumbing coefficient for one moving-middle term."""

    first_momentum: complex
    second_pole: CrossedNSStructurePole
    sectors: SectorAssignment
    left_beta: complex
    right_beta: complex
    coefficient: complex


@dataclass(frozen=True)
class MovingMiddleFaceTerm:
    """Leading normal coefficient of one moving-middle face term."""

    first_momentum: complex
    second_pole: CrossedNSStructurePole
    sectors: SectorAssignment
    right_beta: complex
    coefficient: complex


def crossed_ns_structure_poles(
    first_imaginary_momentum: float,
    second_imaginary_momentum: float,
    sector: int,
    *,
    wall_tolerance: float = 1.0e-12,
) -> tuple[CrossedNSStructurePole, ...]:
    r"""List positive-imaginary poles crossed from the real NS contour.

    The external momenta are ``i*A`` and ``i*B`` with ``A,B>0``.  For the
    ``C`` sector, ``Upsilon_NS`` gives walls at odd positive integers; for
    ``tilde C``, ``Upsilon_R`` gives walls at even positive integers.  Both
    sum and absolute-difference pole families are included.
    """

    first = float(first_imaginary_momentum)
    second = float(second_imaginary_momentum)
    selected_sector = int(sector)
    if (
        not math.isfinite(first)
        or not math.isfinite(second)
        or first <= 0.0
        or second <= 0.0
    ):
        raise ValueError("imaginary momentum magnitudes must be positive and finite")
    if selected_sector not in (0, 1):
        raise ValueError("sector must be 0 (C) or 1 (tilde C)")
    initial_wall = 1.0 if selected_sector == 0 else 2.0
    combinations_by_family = (
        ("sum", first + second),
        ("difference", abs(first - second)),
    )
    result: list[CrossedNSStructurePole] = []
    for family, combination in combinations_by_family:
        wall = initial_wall
        while wall <= combination + wall_tolerance:
            distance = combination - wall
            if abs(distance) <= wall_tolerance:
                raise ValueError(
                    f"the requested kinematics lies on a {family} Liouville wall"
                )
            if distance > 0.0:
                result.append(
                    CrossedNSStructurePole(
                        family=family,
                        momentum=1.0j * distance,
                        wall=wall,
                    )
                )
            wall += 2.0
    return tuple(
        sorted(result, key=lambda item: (item.momentum.imag, item.family))
    )


def crossed_ns_structure_poles_complex(
    first_momentum: Number,
    second_momentum: Number,
    sector: int,
    *,
    wall_tolerance: float = 1.0e-12,
) -> tuple[CrossedNSStructurePole, ...]:
    r"""List poles crossed by a vertical continuation of complex momenta.

    Starting from the real parts of ``P_a`` and ``P_b``, continue both
    momenta vertically to their supplied values.  A sum-family pole is
    ``P=P_a+P_b-i*m``.  The positive-imaginary difference family is
    ``P=+(P_a-P_b)-i*m`` or ``P=+(P_b-P_a)-i*m``.  The wall labels ``m``
    are positive odd integers for ``C`` and positive even integers for
    ``tilde C``.  Retaining the pole's real part is essential on the
    subtraction-free five-point path.
    """

    first = _finite_complex("first_momentum", first_momentum)
    second = _finite_complex("second_momentum", second_momentum)
    selected_sector = int(sector)
    if selected_sector not in (0, 1):
        raise ValueError("sector must be 0 (C) or 1 (tilde C)")
    if wall_tolerance <= 0.0 or not math.isfinite(wall_tolerance):
        raise ValueError("wall_tolerance must be positive and finite")

    initial_wall = 1.0 if selected_sector == 0 else 2.0
    candidates: list[tuple[str, complex]] = [("sum", first + second)]
    difference = first - second
    if difference.imag > wall_tolerance:
        candidates.append(("difference", difference))
    elif difference.imag < -wall_tolerance:
        candidates.append(("difference", -difference))

    result: list[CrossedNSStructurePole] = []
    for family, combination in candidates:
        wall = initial_wall
        while wall <= combination.imag + wall_tolerance:
            distance = combination.imag - wall
            if abs(distance) <= wall_tolerance:
                raise ValueError(
                    f"the requested kinematics lies on a {family} Liouville wall"
                )
            if distance > 0.0:
                result.append(
                    CrossedNSStructurePole(
                        family=family,
                        momentum=combination - 1.0j * wall,
                        wall=wall,
                    )
                )
            wall += 2.0
    return tuple(
        sorted(
            result,
            key=lambda item: (
                item.momentum.imag,
                item.momentum.real,
                item.family,
            ),
        )
    )


def _positive_contour_structure_poles(
    first_momentum: Number,
    second_momentum: Number,
    sector: int,
    *,
    real_tolerance: float = 1.0e-12,
) -> tuple[CrossedNSStructurePole, ...]:
    r"""Fold crossed poles to the BRY nonnegative-real quotient contour.

    The unfolded integrand is even in every internal momentum: both adjacent
    NS structure constants are odd under ``P -> -P``, while the block depends
    on ``P^2``.  A raw pole with positive real part crosses the positive half
    contour upward and carries ``-2i``.  If its real part is negative, its
    reflected pole crosses the positive half contour downward; the canonical
    quotient representative is therefore ``-P`` and carries ``+2i``.  It is
    incorrect simply to discard the latter crossing.

    Purely imaginary poles use the upward ``-2i`` endpoint convention.
    """

    tolerance = float(real_tolerance)
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("real_tolerance must be positive and finite")
    first = _finite_complex("first_momentum", first_momentum)
    second = _finite_complex("second_momentum", second_momentum)
    return _quotient_structure_poles_along_paths(
        complex(first.real, 0.0),
        first,
        complex(second.real, 0.0),
        second,
        sector,
        real_tolerance=tolerance,
    )


def _path_value(initial: complex, final: complex, parameter: float) -> complex:
    return complex(initial + float(parameter) * (final - initial))


def _quotient_structure_poles_along_paths(
    first_initial: Number,
    first_final: Number,
    second_initial: Number,
    second_final: Number,
    sector: int,
    *,
    start_parameter: float = 0.0,
    real_tolerance: float = 1.0e-12,
    wall_tolerance: float = 1.0e-12,
) -> tuple[CrossedNSStructurePole, ...]:
    r"""Return quotient-contour crossings along two affine momentum paths.

    External momenta follow ``P(s)=Re(P)+i*s*Im(P)``.  A residue momentum is
    itself an affine path with a nonzero wall offset, so nested continuation
    must retain its true value at ``s=0`` and may begin only after the parent
    pole's crossing parameter.  This helper implements that path ledger for
    the sum and both signed-difference pole families.
    """

    a0 = _finite_complex("first_initial", first_initial)
    a1 = _finite_complex("first_final", first_final)
    b0 = _finite_complex("second_initial", second_initial)
    b1 = _finite_complex("second_final", second_final)
    selected_sector = int(sector)
    if selected_sector not in (0, 1):
        raise ValueError("sector must be 0 (C) or 1 (tilde C)")
    start = float(start_parameter)
    if not 0.0 <= start < 1.0:
        raise ValueError("start_parameter must lie in [0,1)")

    combinations = (
        ("sum", a0 + b0, a1 + b1),
        ("difference", a0 - b0, a1 - b1),
        ("difference", b0 - a0, b1 - a1),
    )
    maximum_imaginary = max(
        abs(value.imag)
        for _, initial, final in combinations
        for value in (initial, final)
    )
    initial_wall = 1 if selected_sector == 0 else 2
    maximum_wall = int(math.ceil(maximum_imaginary)) + 3
    result: list[CrossedNSStructurePole] = []
    for family, combination_initial, combination_final in combinations:
        wall = initial_wall
        while wall <= maximum_wall:
            raw_initial = combination_initial - 1.0j * wall
            raw_final = combination_final - 1.0j * wall
            at_start = _path_value(raw_initial, raw_final, start)
            y0 = at_start.imag
            y1 = raw_final.imag
            if abs(y1) <= wall_tolerance:
                raise ValueError(
                    f"the requested kinematics lies on a {family} Liouville wall"
                )
            if y0 * y1 < -wall_tolerance**2:
                local_fraction = -y0 / (y1 - y0)
                crossing = start + (1.0 - start) * local_fraction
                canonical_initial = raw_initial
                canonical_final = raw_final
                canonical_start = at_start
                reflected = canonical_start.real < -real_tolerance
                if reflected:
                    canonical_initial = -canonical_initial
                    canonical_final = -canonical_final
                    canonical_start = -canonical_start
                upward = canonical_start.imag < canonical_final.imag
                pole = CrossedNSStructurePole(
                    family=(f"reflected-{family}" if reflected else family),
                    momentum=complex(canonical_final),
                    wall=float(wall),
                    contour_coefficient=(-2.0j if upward else 2.0j),
                    initial_momentum=complex(canonical_initial),
                    crossing_parameter=float(crossing),
                )
                if not any(
                    abs(pole.momentum - previous.momentum) < 1.0e-10
                    and abs(pole.crossing_parameter - previous.crossing_parameter)
                    < 1.0e-10
                    for previous in result
                ):
                    result.append(pole)
            wall += 2
    return tuple(
        sorted(
            result,
            key=lambda item: (
                item.crossing_parameter,
                item.momentum.real,
                item.momentum.imag,
                item.family,
            ),
        )
    )


def _nested_quotient_structure_poles(
    parent: CrossedNSStructurePole,
    second_momentum: Number,
    sector: int,
) -> tuple[CrossedNSStructurePole, ...]:
    """Continue a middle-trinion pole only after its parent residue exists."""

    second = _finite_complex("second_momentum", second_momentum)
    parent_initial = (
        complex(parent.momentum.real, 0.0)
        if parent.initial_momentum is None
        else complex(parent.initial_momentum)
    )
    return _quotient_structure_poles_along_paths(
        parent_initial,
        parent.momentum,
        complex(second.real, 0.0),
        second,
        sector,
        start_parameter=max(parent.crossing_parameter, 0.0),
    )


def balanced_equal_energy(t_value: float) -> complex:
    r"""Return the endpoint-channel balanced energy ``x(t)+i*t``.

    With all three PCOs on the fixed punctures, the two tight exponents in
    the first open chamber above ``t=3/5`` are the continuous face of the
    two raised outgoing fields and the wall-one sum residue on an incoming--
    raised-outgoing face.  Writing ``u=t^2-x^2``, their margins are
    ``4*u-1`` and ``-16*u+10*t-2``.  Equating them gives

    ``u=t/2-1/20`` and ``x=sqrt(t^2-t/2+1/20)``.

    The endpoint-channel margin is ``2*t-6/5``.  This curve is retained as a
    diagnostic, but is *not* a globally subtraction-free path: a channel
    with the incoming field in the middle has a difference-pole residue line
    whose mixed-picture boundary diverges.  The global audit below records
    that obstruction explicitly.
    """

    t = float(t_value)
    if not math.isfinite(t) or not 0.6 < t < 2.0 / 3.0:
        raise ValueError("balanced equal-energy path requires 3/5<t<2/3")
    x_squared = t * t - 0.5 * t + 0.05
    if x_squared <= 0.0:
        raise ArithmeticError("the balanced path produced a non-positive x^2")
    return complex(math.sqrt(x_squared), t)


def equal_complex_energy_convergence_audit(omega: Number) -> dict[str, object]:
    r"""Audit every endpoint-tree stratum with its actual picture threshold.

    If a boundary cherry contains ``r`` picture-zero external vertices, its
    leading density is ``|q|^{-1-r+P_e^2-K_e^2}``; radial convergence is
    therefore ``Re(P_e^2-K_e^2)>r-1``.  The audit runs this condition over
    all oriented endpoint trees and every continuous, endpoint-residue, and
    nested-residue edge in the first chamber above the third incoming--
    outgoing structure wall, ``3/5<t<2/3``.  It also includes the decisive
    middle-incoming difference-pole line, which proves that the endpoint-
    balanced curve is not a global subtraction-free domain.
    """

    value = _finite_complex("omega", omega)
    t = value.imag
    u = t * t - value.real * value.real
    if not 0.6 < t < 2.0 / 3.0:
        raise ValueError("the three-fixed-PCO residue audit requires 3/5<t<2/3")

    records: list[dict[str, object]] = []

    def add_record(
        name: str,
        edge: str,
        momentum: complex,
        channel_energy: complex,
        sectors: SectorAssignment | None,
        pair: tuple[int, int],
    ) -> None:
        delta = float((momentum * momentum - channel_energy * channel_energy).real)
        raised_count = sum(int(label in PICTURE_ZERO_LABELS) for label in pair)
        threshold = float(
            1
            if frozenset(pair) == frozenset(MINUS_ONE_LABELS)
            else raised_count - 1
        )
        records.append(
            {
                "name": name,
                "edge": edge,
                "momentum": {"real": momentum.real, "imag": momentum.imag},
                "channel_energy": {
                    "real": channel_energy.real,
                    "imag": channel_energy.imag,
                },
                "sectors": None if sectors is None else list(sectors),
                "boundary_pair": list(pair),
                "raised_picture_zero_count": raised_count,
                "integrability_threshold": threshold,
                "real_P2_minus_K2": delta,
                "integrability_margin": delta - threshold,
            }
        )

    external = (4.0 * value, value, value, value, value)
    signed = (4.0 * value, -value, -value, -value, -value)
    for ordering in oriented_tree_orderings():
        if 0 not in ordering[:2]:
            continue
        left_pair = (ordering[0], ordering[1])
        right_pair = (ordering[3], ordering[4])
        pa, pb, pc, pd, pe = (external[label] for label in ordering)
        left_energy = sum(signed[label] for label in left_pair)
        right_energy = sum(signed[label] for label in right_pair)
        add_record("continuous", "left", 0.0j, left_energy, None, left_pair)
        add_record("continuous", "right", 0.0j, right_energy, None, right_pair)

        for sectors in ODD_SECTOR_ASSIGNMENTS:
            sector_left, sector_middle, sector_right = sectors
            left_poles = crossed_ns_structure_poles_complex(pa, pb, sector_left)
            right_poles = crossed_ns_structure_poles_complex(pd, pe, sector_right)
            for pole in left_poles:
                add_record(
                    f"left-{pole.family}-m{int(pole.wall)}",
                    "left",
                    pole.momentum,
                    left_energy,
                    sectors,
                    left_pair,
                )
            for pole in right_poles:
                add_record(
                    f"right-{pole.family}-m{int(pole.wall)}",
                    "right",
                    pole.momentum,
                    right_energy,
                    sectors,
                    right_pair,
                )
            for left_pole in left_poles:
                middle_poles = crossed_ns_structure_poles_complex(
                    left_pole.momentum, pc, sector_middle
                )
                for pole in middle_poles:
                    add_record(
                        (
                            f"nested-left-{left_pole.family}-m{int(left_pole.wall)}"
                            f"-middle-{pole.family}-m{int(pole.wall)}"
                        ),
                        "right",
                        pole.momentum,
                        right_energy,
                        sectors,
                        right_pair,
                    )
                for pole in right_poles:
                    add_record(
                        (
                            f"nested-left-{left_pole.family}-m{int(left_pole.wall)}"
                            f"-right-{pole.family}-m{int(pole.wall)}"
                        ),
                        "right",
                        pole.momentum,
                        right_energy,
                        sectors,
                        right_pair,
                    )

    endpoint_minimum = min(
        float(record["integrability_margin"]) for record in records
    )

    # In the mixed/mixed middle-incoming corner (1,3 | 0 | 2,4), continuing
    # the middle Liouville momentum 4*omega crosses the C difference pole
    # P2=4*omega-P1-i for 0<P1<4*x.  At the positive-contour endpoint
    # P1->4*x its real part vanishes, so this residue line approaches
    # P2=i*(4*t-1).  The right pair (2,4) contains one picture-zero and one
    # picture-minus-one field, hence threshold zero.  This is an open-corner
    # obstruction, not a measure-zero endpoint effect.
    if value.real <= 0.0:
        raise ValueError("the three-fixed-PCO audit requires Re(omega)>0")
    add_record(
        "middle-difference-m1-P1-to-4x",
        "right",
        1.0j * (4.0 * t - 1.0),
        -2.0 * value,
        (0, 0, 1),
        (2, 4),
    )

    minimum = min(float(record["integrability_margin"]) for record in records)
    mixed_middle_margin = float(
        records[-1]["integrability_margin"]
    )
    split_middle_margin = mixed_middle_margin + 1.0
    tilde_mixed_middle_margin = float(
        4.0 * u - (4.0 * t - 2.0) ** 2
    )
    minimal_subtraction_chamber = (
        endpoint_minimum > 0.0
        and split_middle_margin > 0.0
        and tilde_mixed_middle_margin > 0.0
    )
    return {
        "omega": {"real": value.real, "imag": value.imag},
        "t": t,
        "u": u,
        "fixed_residue_chamber": "3/5<t<2/3",
        "integrability_condition": "Re(P_e^2-K_e^2)>r_e-1",
        "all_moduli_boundaries_absolutely_convergent": minimum > 0.0,
        "minimum_integrability_margin": minimum,
        "endpoint_channel_minimum_integrability_margin": endpoint_minimum,
        "global_middle_channel_obstruction": (
            "P2=4*omega-P1-i in the mixed/mixed corner "
            "(1,3|0|2,4)"
        ),
        "middle_difference_m1_mixed_margin": mixed_middle_margin,
        "middle_difference_m1_split_margin": split_middle_margin,
        "middle_difference_m2_mixed_margin": tilde_mixed_middle_margin,
        "minimal_subtraction_chamber": minimal_subtraction_chamber,
        "minimal_subtraction_t_domain_on_balanced_path": (
            "3/5<t<(25+sqrt(545))/80"
        ),
        "required_polynomial_subtraction_orbits": (
            [
                {
                    "representative_corner": [1, 3, 0, 2, 4],
                    "multiplicity": 2,
                    "residue_family": "middle C difference wall m=1",
                    "pole_line": "P2=4*omega-P1-i, 0<P1<4*Re(omega)",
                    "boundary_pair_picture_zero_count": 1,
                    "integrability_margin": mixed_middle_margin,
                }
            ]
            if minimal_subtraction_chamber
            else None
        ),
        "records": records,
    }


def three_fixed_pco_subtraction_free_no_go(
    omega: Number,
) -> dict[str, object]:
    r"""Certify the equal-energy no-go for a subtraction-free global region.

    The proof uses only two unavoidable local strata and is independent of
    all higher Liouville walls.

    * The raised outgoing pair ``{1,2}`` on the continuous contour has
      margin ``M_cont=4*(t^2-x^2)-1``.  For ``0<t<=1/2`` this is non-positive.
    * For ``t>1/2``, the middle-incoming mixed/mixed corner
      ``{1,3}|{0}|{2,4}`` contains the crossed wall-one ``C`` residue line.
      One of its sum/difference branches approaches
      ``P2=i*(4*t-1)`` on the nonnegative quotient contour, giving
      ``M_mid=4*(t^2-x^2)-(4*t-1)^2``.  Since ``x^2>=0``,
      ``M_mid<=-(6*t-1)*(2*t-1)<0``.

    Strict positivity is required for absolute radial convergence.  The two
    cases exhaust every ``t>0``, including ``t=1/2`` where the continuum face
    is at best logarithmic.  Thus no real ``x`` produces a globally
    subtraction-free open region with equal outgoing energies.
    """

    value = _finite_complex("omega", omega)
    x_value = value.real
    t_value = value.imag
    if t_value <= 0.0:
        raise ValueError("the no-go certificate assumes Im(omega)=t>0")
    u_value = t_value * t_value - x_value * x_value
    continuous_margin = 4.0 * u_value - 1.0
    middle_margin = 4.0 * u_value - (4.0 * t_value - 1.0) ** 2
    continuous_upper_bound = 4.0 * t_value * t_value - 1.0
    middle_upper_bound = -(
        (6.0 * t_value - 1.0) * (2.0 * t_value - 1.0)
    )

    tolerance = 1.0e-14
    if x_value > tolerance:
        contour_branch = "difference"
        contour_endpoint = 4.0 * x_value
        contour_limit = "P1->4*x from below"
    elif x_value < -tolerance:
        contour_branch = "sum"
        contour_endpoint = -4.0 * x_value
        contour_limit = "P1->-4*x from above"
    else:
        contour_branch = "sum"
        contour_endpoint = 0.0
        contour_limit = "P1->0 from above"

    if t_value <= 0.5:
        obstructing_stratum = "continuous raised pair {1,2}"
        obstructing_margin = continuous_margin
        certified_upper_bound = continuous_upper_bound
        bound_factorization = "M_cont<=4*t^2-1<=(or =)0"
    else:
        obstructing_stratum = (
            "middle C wall-1 residue in {1,3}|{0}|{2,4}"
        )
        obstructing_margin = middle_margin
        certified_upper_bound = middle_upper_bound
        bound_factorization = "M_mid<=-(6*t-1)*(2*t-1)<0"

    if certified_upper_bound > 2.0e-13:
        raise AssertionError("the analytic no-go bound was evaluated inconsistently")
    return {
        "omega": {"real": x_value, "imag": t_value},
        "equal_outgoing_energies": True,
        "picture_zero_labels": list(PICTURE_ZERO_LABELS),
        "picture_minus_one_labels": list(MINUS_ONE_LABELS),
        "u": u_value,
        "continuous_raised_pair_margin": continuous_margin,
        "middle_wall_one_mixed_pair_margin": middle_margin,
        "continuous_margin_upper_bound": continuous_upper_bound,
        "middle_margin_upper_bound": middle_upper_bound,
        "middle_pole_branch_on_positive_contour": contour_branch,
        "middle_pole_contour_endpoint_P1": contour_endpoint,
        "middle_pole_contour_limit": contour_limit,
        "obstructing_stratum": obstructing_stratum,
        "obstructing_margin": obstructing_margin,
        "certified_upper_bound_on_obstructing_margin": certified_upper_bound,
        "bound_factorization": bound_factorization,
        "strict_absolute_convergence": False,
        "globally_subtraction_free_region_exists": False,
        "scope": (
            "real x, t>0, omega_1=...=omega_4=x+i*t, fixed picture-zero "
            "labels 0,1,2, vertical BRY Liouville continuation"
        ),
    }


@dataclass(frozen=True)
class NSFivePointQMCResult:
    """Target-blind RQMC estimate of the raw all-NS moduli integral."""

    outgoing_energies: tuple[complex, complex, complex, complex]
    estimates: tuple[complex, ...]
    mean: complex
    standard_error_real: float
    standard_error_imag: float
    samples_per_replicate: int
    replicates: int
    radial_power: float
    seed: int
    recursion_max_twice_level: int | None
    global_max_twice_levels: tuple[int, int]
    momentum_orders: tuple[int, int]
    momentum_maximum: float

    def to_json(self) -> dict[str, object]:
        return {
            "schema": "type0b-ns-sphere-five-tachyon-worldsheet-blind-v1",
            "blind_freeze": False,
            "blind_freeze_statement": (
                "The estimator and reported integral do not evaluate or import "
                "a matrix-model target."
            ),
            "outgoing_energies": [
                {"real": value.real, "imag": value.imag}
                for value in self.outgoing_energies
            ],
            "incoming_energy": {
                "real": sum(self.outgoing_energies).real,
                "imag": sum(self.outgoing_energies).imag,
            },
            "integral_mean": {"real": self.mean.real, "imag": self.mean.imag},
            "standard_error_real": self.standard_error_real,
            "standard_error_imag": self.standard_error_imag,
            "replicate_estimates": [
                {"real": value.real, "imag": value.imag}
                for value in self.estimates
            ],
            "samples_per_replicate": self.samples_per_replicate,
            "replicates": self.replicates,
            "radial_power": self.radial_power,
            "seed": self.seed,
            "recursion_max_twice_level": self.recursion_max_twice_level,
            "global_max_twice_levels": list(self.global_max_twice_levels),
            "momentum_orders": list(self.momentum_orders),
            "momentum_maximum": self.momentum_maximum,
            "matrix_model_used": False,
        }


@dataclass(frozen=True)
class NSFivePointFinitePartQMCResult:
    """Target-blind BRY finite-part estimate of the all-NS moduli integral."""

    outgoing_energies: tuple[complex, complex, complex, complex]
    estimates: tuple[complex, ...]
    bulk_estimates: tuple[complex, ...]
    face_estimates: tuple[complex, ...]
    corner_contribution: complex
    mean: complex
    standard_error_real: float
    standard_error_imag: float
    collar_radius: float
    projection_radius: float
    bulk_samples_per_replicate: int
    face_samples_per_replicate: int
    replicates: int
    radial_power: float
    seed: int
    recursion_max_twice_level: int | None
    global_max_twice_levels: tuple[int, int]
    momentum_orders: tuple[int, int]
    momentum_maximum: float
    extreme_bulk_weights: tuple[dict[str, object], ...] = ()
    subtraction_scheme: str = (
        "leading local radial finite-part forest on all 10 faces and "
        "15 compatible corners"
    )

    def to_json(self) -> dict[str, object]:
        return {
            "schema": "type0b-ns-sphere-five-tachyon-worldsheet-finite-part-v1",
            "blind_freeze": False,
            "blind_freeze_statement": (
                "No matrix-model formula, target value, or fit coefficient was "
                "evaluated in constructing this worldsheet estimate."
            ),
            "subtraction_scheme": self.subtraction_scheme,
            "outgoing_energies": [
                {"real": value.real, "imag": value.imag}
                for value in self.outgoing_energies
            ],
            "incoming_energy": {
                "real": sum(self.outgoing_energies).real,
                "imag": sum(self.outgoing_energies).imag,
            },
            "integral_mean": {"real": self.mean.real, "imag": self.mean.imag},
            "standard_error_real": self.standard_error_real,
            "standard_error_imag": self.standard_error_imag,
            "replicate_estimates": [
                {"real": value.real, "imag": value.imag}
                for value in self.estimates
            ],
            "bulk_estimates": [
                {"real": value.real, "imag": value.imag}
                for value in self.bulk_estimates
            ],
            "face_estimates": [
                {"real": value.real, "imag": value.imag}
                for value in self.face_estimates
            ],
            "corner_contribution": {
                "real": self.corner_contribution.real,
                "imag": self.corner_contribution.imag,
            },
            "collar_radius": self.collar_radius,
            "projection_radius": self.projection_radius,
            "bulk_samples_per_replicate": self.bulk_samples_per_replicate,
            "face_samples_per_replicate": self.face_samples_per_replicate,
            "replicates": self.replicates,
            "radial_power": self.radial_power,
            "seed": self.seed,
            "recursion_max_twice_level": self.recursion_max_twice_level,
            "global_max_twice_levels": list(self.global_max_twice_levels),
            "momentum_orders": list(self.momentum_orders),
            "momentum_maximum": self.momentum_maximum,
            "extreme_bulk_weights": list(self.extreme_bulk_weights),
            "matrix_model_used": False,
        }


@dataclass(frozen=True)
class NSFivePointContinuedQMCResult:
    """Target-blind raw-moduli estimate with the Liouville residue forest."""

    outgoing_energies: tuple[complex, complex, complex, complex]
    estimates: tuple[complex, ...]
    continuous_estimates: tuple[complex, ...]
    left_residue_estimates: tuple[complex, ...]
    right_residue_estimates: tuple[complex, ...]
    nested_residue_estimates: tuple[complex, ...]
    mean: complex
    standard_error_real: float
    standard_error_imag: float
    samples_per_replicate: int
    replicates: int
    radial_power: float
    seed: int
    recursion_max_twice_level: int | None
    global_max_twice_levels: tuple[int, int]
    momentum_orders: tuple[int, int]
    momentum_maximum: float
    extreme_sample_diagnostics: tuple[dict[str, object], ...] = ()

    def to_json(self) -> dict[str, object]:
        def encoded(values: Sequence[complex]) -> list[dict[str, float]]:
            return [
                {"real": value.real, "imag": value.imag} for value in values
            ]

        return {
            "schema": "type0b-ns-sphere-five-tachyon-continued-worldsheet-v1",
            "blind_freeze": False,
            "blind_freeze_statement": (
                "No matrix-model formula, target value, or fit coefficient was "
                "evaluated in constructing this worldsheet estimate."
            ),
            "liouville_contour": (
                "real double continuum plus the complete iterated -2i residue "
                "forest for a fixed vertical-continuation chamber"
            ),
            "outgoing_energies": encoded(self.outgoing_energies),
            "incoming_energy": {
                "real": sum(self.outgoing_energies).real,
                "imag": sum(self.outgoing_energies).imag,
            },
            "integral_mean": {"real": self.mean.real, "imag": self.mean.imag},
            "standard_error_real": self.standard_error_real,
            "standard_error_imag": self.standard_error_imag,
            "replicate_estimates": encoded(self.estimates),
            "continuous_estimates": encoded(self.continuous_estimates),
            "left_residue_estimates": encoded(self.left_residue_estimates),
            "right_residue_estimates": encoded(self.right_residue_estimates),
            "nested_residue_estimates": encoded(self.nested_residue_estimates),
            "samples_per_replicate": self.samples_per_replicate,
            "replicates": self.replicates,
            "radial_power": self.radial_power,
            "seed": self.seed,
            "recursion_max_twice_level": self.recursion_max_twice_level,
            "global_max_twice_levels": list(self.global_max_twice_levels),
            "momentum_orders": list(self.momentum_orders),
            "momentum_maximum": self.momentum_maximum,
            "extreme_sample_diagnostics": list(
                self.extreme_sample_diagnostics
            ),
            "matrix_model_used": False,
        }


def imaginary_energy_chamber_audit(
    outgoing_energies: Sequence[Number],
) -> dict[str, object]:
    r"""Audit raw-moduli convergence versus the undeformed NS contour.

    For ``omega_i=i t_i`` with positive ``t_i``, a collision of two raised
    tachyons has BRY threshold

    ``P_*^2 = 1-(t_i+t_j)^2``.

    The raw moduli integral is locally integrable only when every such
    threshold is non-positive.  On the other hand, the first ``C``-type
    super-Liouville contour pinch occurs when

    ``t_in + max(t_i) = 1``.

    These requirements are incompatible for the present placement of three
    PCOs.  In equal kinematics they read respectively ``t>=1/2`` and
    ``t<1/5``.
    """

    if len(outgoing_energies) != 4:
        raise ValueError("outgoing_energies must contain four values")
    normalized = tuple(
        _finite_complex(f"outgoing_energies[{index}]", value)
        for index, value in enumerate(outgoing_energies)
    )
    if any(abs(value.real) > 1.0e-13 or value.imag <= 0.0 for value in normalized):
        raise ValueError(
            "the chamber audit requires positive purely imaginary outgoing energies"
        )
    t_values = tuple(value.imag for value in normalized)
    signed_t_values = (sum(t_values), *(-value for value in t_values))
    boundary_pairs = tuple(combinations(range(5), 2))
    thresholds = {
        f"{left},{right}": 1.0
        - (signed_t_values[left] + signed_t_values[right]) ** 2
        for left, right in boundary_pairs
    }
    contour_wall_parameter = sum(t_values) + max(t_values)
    raw_moduli_convergent = all(value <= 0.0 for value in thresholds.values())
    undeformed_contour_valid = contour_wall_parameter < 1.0
    return {
        "outgoing_imaginary_parts": list(t_values),
        "boundary_pair_threshold_squared": thresholds,
        "raw_moduli_convergent_without_pco_subtraction": raw_moduli_convergent,
        "first_C_contour_wall_parameter": contour_wall_parameter,
        "undeformed_positive_real_liouville_contour_valid": undeformed_contour_valid,
        "simultaneously_subtraction_and_residue_free": (
            raw_moduli_convergent and undeformed_contour_valid
        ),
    }


def _fermion_pair(
    positions: Sequence[ProjectivePoint], left: int, right: int
) -> complex:
    r"""Return ``<psi^0(z_left) psi^0(z_right)>`` in the displayed order.

    The local coordinate at infinity is ``1/z`` and the normalized field is
    ``psi(infinity)=lim_(Z->infinity) Z psi(Z)``.  The timelike target metric
    supplies the minus sign in ``<psi^0(z)psi^0(w)>=-1/(z-w)``.  This sign is
    what makes the two-PCO specialization reproduce BRY's ``-omega_2
    omega_3 J`` term.
    """

    if left == right:
        raise ValueError("a free-fermion contraction needs distinct labels")
    first = positions[left]
    second = positions[right]
    if first is None and second is None:
        raise ValueError("two distinct punctures cannot both be infinity")
    if first is None:
        return -1.0 + 0.0j
    if second is None:
        return 1.0 + 0.0j
    separation = complex(first) - complex(second)
    if separation == 0.0:
        raise ZeroDivisionError("the free-fermion contraction hit a collision")
    return -1.0 / separation


def _superghost_pair_factor(
    positions: Sequence[ProjectivePoint],
) -> float:
    r"""Return the nonchiral correlator of the two picture-minus-one fields."""

    normalized = _validate_positions(positions)
    left, right = MINUS_ONE_LABELS
    if normalized[left] is None or normalized[right] is None:
        return 1.0
    separation = abs(complex(normalized[left]) - complex(normalized[right]))
    if separation == 0.0:
        raise ZeroDivisionError("the two picture-minus-one vertices collided")
    return float(separation**-2)


def pco_chiral_terms(
    *,
    positions: Sequence[ProjectivePoint],
    signed_energies: Sequence[Number],
    raised_labels: Sequence[int] = PICTURE_ZERO_LABELS,
    operator_order: Sequence[int] = (0, 1, 2, 3, 4),
) -> tuple[PCOChiralTerm, ...]:
    r"""Expand ``prod_(i in raised) (Lambda_i-k_i psi_i V_i)``.

    The free Majorana expectation value removes odd selections.  With three
    raised vertices, the surviving subsets contain zero or two timelike
    fermions.  ``coefficient`` includes the free-fermion contraction and the
    graded sign required to separate the timelike and Liouville chiral CFTs.
    """

    normalized_positions = _validate_positions(positions)
    if len(signed_energies) != 5:
        raise ValueError("signed_energies must contain five values")
    energies = tuple(
        _finite_complex(f"signed_energies[{index}]", value)
        for index, value in enumerate(signed_energies)
    )
    raised = tuple(int(label) for label in raised_labels)
    order = tuple(int(label) for label in operator_order)
    if len(set(raised)) != len(raised) or any(label not in range(5) for label in raised):
        raise ValueError("raised_labels must be distinct labels in range(5)")
    if len(order) != 5 or set(order) != set(range(5)):
        raise ValueError("operator_order must permute labels 0,...,4")
    rank = {label: index for index, label in enumerate(order)}

    terms: list[PCOChiralTerm] = []
    allowed_subsets: Iterable[tuple[int, ...]] = ((), *combinations(raised, 2))
    for selected in allowed_subsets:
        selected_set = set(selected)
        descendants = tuple(
            int(label in raised and label not in selected_set)
            for label in range(5)
        )
        coefficient = complex(
            math.prod(-energies[label] for label in selected)
        )
        if len(selected) == 2:
            ordered_selected = tuple(sorted(selected, key=rank.__getitem__))
            coefficient *= _fermion_pair(
                normalized_positions,
                ordered_selected[0],
                ordered_selected[1],
            )

        # Move all timelike-fermion factors to the left of all Liouville
        # factors while preserving the displayed operator order.
        crossings = 0
        for earlier in order:
            if not descendants[earlier]:
                continue
            crossings += sum(
                int(later in selected_set)
                for later in order[rank[earlier] + 1 :]
            )
        if crossings % 2:
            coefficient = -coefficient
        terms.append(
            PCOChiralTerm(
                timelike_labels=tuple(sorted(selected, key=rank.__getitem__)),
                liouville_descendants=descendants,  # type: ignore[arg-type]
                coefficient=coefficient,
            )
        )
    return tuple(terms)


@lru_cache(maxsize=None)
def _legendre_interval(order: int, upper: float) -> tuple[tuple[float, float], ...]:
    order = int(order)
    upper = float(upper)
    if order < 2 or not math.isfinite(upper) or upper <= 0.0:
        raise ValueError("momentum quadrature order and endpoint must be positive")
    nodes, weights = np.polynomial.legendre.leggauss(order)
    return tuple(
        (float(0.5 * upper * (node + 1.0)), float(0.5 * upper * weight))
        for node, weight in zip(nodes, weights)
    )


class BRYNSFiveTachyonIntegrand:
    r"""Evaluate the BRY-normalized all-NS five-point PCO density.

    This class contains no matrix-model formula, fit coefficient, or target
    polynomial.  External Liouville momenta may be complex.  Internal
    Liouville momenta remain on the positive real BRY contour with measure
    ``dP_1 dP_2 / pi^2``.
    """

    def __init__(
        self,
        *,
        outgoing_energies: Sequence[Number],
        recursion_max_twice_level: int | None = None,
        global_max_twice_levels: Sequence[int] = (8, 4),
        global_max_total_twice_level: int | None = 8,
        momentum_orders: Sequence[int] = (4, 5),
        momentum_maximum: float = 4.0,
        structure_precision: int = 24,
        central_charge_shift: float = 1.0e-5,
        block_working_precision: int = 50,
        pole_tolerance: float = 1.0e-28,
    ) -> None:
        if len(outgoing_energies) != 4:
            raise ValueError("outgoing_energies must contain four values")
        self.outgoing_energies = tuple(
            _finite_complex(f"outgoing_energies[{index}]", value)
            for index, value in enumerate(outgoing_energies)
        )
        self.incoming_energy = sum(self.outgoing_energies)
        self.signed_energies = (
            self.incoming_energy,
            *(-value for value in self.outgoing_energies),
        )
        self.external_momenta = (
            self.incoming_energy,
            *self.outgoing_energies,
        )
        if recursion_max_twice_level is not None and (
            not isinstance(recursion_max_twice_level, int)
            or recursion_max_twice_level < 0
        ):
            raise ValueError(
                "recursion_max_twice_level must be non-negative or None"
            )
        maxima = tuple(int(value) for value in global_max_twice_levels)
        if len(maxima) != 2 or any(value < 0 for value in maxima):
            raise ValueError("global_max_twice_levels must contain two non-negative values")
        orders = tuple(int(value) for value in momentum_orders)
        if len(orders) != 2 or any(value < 2 for value in orders):
            raise ValueError("momentum_orders must contain two integers at least two")
        if orders[0] == orders[1]:
            raise ValueError("stagger the two momentum orders to avoid the P1=P2 diagonal")
        if global_max_total_twice_level is not None and (
            not isinstance(global_max_total_twice_level, int)
            or global_max_total_twice_level < 0
        ):
            raise ValueError("global_max_total_twice_level must be non-negative or None")
        if not math.isfinite(momentum_maximum) or momentum_maximum <= 0.0:
            raise ValueError("momentum_maximum must be positive and finite")
        if structure_precision < 15 or block_working_precision < 30:
            raise ValueError("insufficient structure or block working precision")
        if central_charge_shift < 0.0 or not math.isfinite(central_charge_shift):
            raise ValueError("central_charge_shift must be finite and non-negative")
        if pole_tolerance <= 0.0 or not math.isfinite(pole_tolerance):
            raise ValueError("pole_tolerance must be finite and positive")

        self.recursion_max_twice_level = recursion_max_twice_level
        self.global_max_twice_levels = maxima
        self.global_max_total_twice_level = global_max_total_twice_level
        self.momentum_orders = orders
        self.momentum_maximum = float(momentum_maximum)
        self.structure_precision = int(structure_precision)
        self.central_charge_shift = float(central_charge_shift)
        self.block_working_precision = int(block_working_precision)
        self.pole_tolerance = float(pole_tolerance)
        self._structure_cache: dict[tuple[object, ...], complex] = {}
        self._structure_residue_cache: dict[tuple[object, ...], complex] = {}
        self._structure_laurent_cache: dict[
            tuple[object, ...], tuple[complex, ...]
        ] = {}
        self._block_cache: dict[
            tuple[object, ...], NSSphereLinearCRecursion
        ] = {}
        self._middle_corner_projection_cache: dict[
            tuple[object, ...], tuple[MovingMiddleCornerTerm, ...]
        ] = {}
        self._middle_face_projection_cache: dict[
            tuple[object, ...], tuple[MovingMiddleFaceTerm, ...]
        ] = {}

    @property
    def block_central_charge(self) -> float:
        return 13.5 + self.central_charge_shift

    def block_weight(self, momentum: Number) -> complex:
        value = _finite_complex("momentum", momentum)
        q_squared = self.block_central_charge / 3.0 - 0.5
        return complex(0.5 * (q_squared / 4.0 + value * value))

    @property
    def external_weights(self) -> tuple[complex, ...]:
        return tuple(self.block_weight(value) for value in self.external_momenta)

    def fixed_gauge_positions(self, z: Number, w: Number) -> tuple[ProjectivePoint, ...]:
        return _validate_positions((INFINITY, 1.0 + 0.0j, 0.0 + 0.0j, z, w))

    def _structure_constant(
        self, first: Number, second: Number, third: Number, sector: int
    ) -> complex:
        momenta = (complex(first), complex(second), complex(third))
        symmetric_momenta = tuple(
            sorted(momenta, key=lambda value: (value.real, value.imag))
        )
        key = (int(sector), symmetric_momenta)
        if key not in self._structure_cache:
            function = ns_structure_constant if sector == 0 else ns_tilde_structure_constant
            self._structure_cache[key] = function(
                *symmetric_momenta, self.structure_precision
            )
        return self._structure_cache[key]

    def _structure_constant_residue(
        self,
        first: Number,
        second: Number,
        pole: Number,
        sector: int,
    ) -> complex:
        r"""Return ``Res_P C_sector(first,second,P)`` by a contour coefficient.

        At ``b=1`` the higher Upsilon zeros coalesce, so poles beyond the
        first wall need not be simple.  A symmetric two-point extraction
        then mixes the cubic (and higher) Laurent coefficients into a false
        divergent answer.  The circular Fourier coefficient below isolates
        the true ``(P-pole)^-1`` term for any pole order.
        """

        first_value = complex(first)
        second_value = complex(second)
        pole_value = complex(pole)
        key = (int(sector), first_value, second_value, pole_value)
        if key in self._structure_residue_cache:
            return self._structure_residue_cache[key]
        function = (
            ns_structure_constant_mp
            if int(sector) == 0
            else ns_tilde_structure_constant_mp
        )
        precision = max(60, self.structure_precision + 30)
        point_count = 32

        def coefficient(radius_text: str) -> mpmath.mpc:
            radius = mpmath.mpf(radius_text)
            total = mpmath.mpc(0)
            for index in range(point_count):
                phase = mpmath.e ** (
                    2j * mpmath.pi * mpmath.mpf(index) / point_count
                )
                offset = radius * phase
                total += (
                    function(
                        mpmath.mpc(first_value),
                        mpmath.mpc(second_value),
                        mpmath.mpc(pole_value) + offset,
                    )
                    * offset
                )
            return total / point_count

        with mpmath.workdps(precision):
            coarse = coefficient("0.002")
            fine = coefficient("0.001")
            scale = max(abs(coarse), abs(fine), mpmath.mpf("1e-80"))
            if abs(coarse - fine) > mpmath.mpf("1e-24") * scale:
                raise ArithmeticError(
                    "structure-constant contour residue failed its radius check"
                )
            residue = complex(fine)
        if not math.isfinite(residue.real) or not math.isfinite(residue.imag):
            raise ArithmeticError("structure-constant residue is non-finite")
        self._structure_residue_cache[key] = residue
        return residue

    def _structure_laurent_coefficients(
        self,
        first: Number,
        second: Number,
        pole: CrossedNSStructurePole,
        sector: int,
    ) -> tuple[complex, ...]:
        r"""Return ``(a_-1,...,a_-r)`` at a crossed b=1 pole.

        In the BRY ``b=1`` specialization, the pole order equals the wall
        label: odd wall ``m`` for ``C`` and even wall ``m`` for ``tilde C``.
        A discrete Fourier coefficient on a small circle extracts every
        negative Laurent coefficient without assuming a simple pole.
        """

        first_value = complex(first)
        second_value = complex(second)
        pole_value = complex(pole.momentum)
        order = int(round(pole.wall))
        if order < 1 or order > 3:
            raise ValueError(
                "the production path supports Laurent pole orders through three; "
                "stay in the chamber 3/5<t<2/3"
            )
        key = (int(sector), first_value, second_value, pole_value, order)
        if key in self._structure_laurent_cache:
            return self._structure_laurent_cache[key]
        function = (
            ns_structure_constant_mp
            if int(sector) == 0
            else ns_tilde_structure_constant_mp
        )
        precision = max(60, self.structure_precision + 30)
        point_count = 24
        with mpmath.workdps(precision):
            radius = mpmath.mpf("0.001")
            coefficients = [mpmath.mpc(0) for _ in range(order)]
            for index in range(point_count):
                phase = mpmath.e ** (
                    2j * mpmath.pi * mpmath.mpf(index) / point_count
                )
                offset = radius * phase
                value = function(
                    mpmath.mpc(first_value),
                    mpmath.mpc(second_value),
                    mpmath.mpc(pole_value) + offset,
                )
                for negative_order in range(1, order + 1):
                    coefficients[negative_order - 1] += (
                        value * offset**negative_order / point_count
                    )
            result = tuple(complex(value) for value in coefficients)
        if any(
            not math.isfinite(value.real) or not math.isfinite(value.imag)
            for value in result
        ):
            raise ArithmeticError("a structure Laurent coefficient is non-finite")
        self._structure_laurent_cache[key] = result
        return result

    @staticmethod
    def _analytic_first_derivative(function, point: complex) -> complex:
        """Return a Richardson-improved analytic derivative."""

        def centered(step: float):
            return (function(point + step) - function(point - step)) / (2.0 * step)

        coarse = centered(2.0e-4)
        fine = centered(1.0e-4)
        return (4.0 * fine - coarse) / 3.0

    @staticmethod
    def _analytic_second_derivative(function, point: complex) -> complex:
        """Return a Richardson-improved analytic second derivative."""

        center = function(point)

        def centered(step: float):
            return (
                function(point + step) - 2.0 * center + function(point - step)
            ) / (step * step)

        coarse = centered(4.0e-4)
        fine = centered(2.0e-4)
        return (4.0 * fine - coarse) / 3.0

    @staticmethod
    def _numeric_contour_residue(function, pole: complex) -> complex:
        """Extract the full integrand residue, including block derivatives."""

        radius = 5.0e-4
        point_count = 12
        total = 0.0 + 0.0j
        for index in range(point_count):
            phase = cmath.exp(2.0j * math.pi * index / point_count)
            offset = radius * phase
            total += function(pole + offset) * offset
        return total / point_count

    def _structure_product(
        self,
        ordered_external_momenta: Sequence[Number],
        internal_momenta: tuple[complex, complex],
        sectors: SectorAssignment,
    ) -> complex:
        external = tuple(complex(value) for value in ordered_external_momenta)
        p1, p2 = internal_momenta
        return (
            self._structure_constant(external[0], external[1], p1, sectors[0])
            * self._structure_constant(p1, external[2], p2, sectors[1])
            * self._structure_constant(p2, external[3], external[4], sectors[2])
        )

    @staticmethod
    def _spin_local_scales(
        channel: LinearChannel,
        positions: Sequence[ProjectivePoint],
    ) -> tuple[complex, ...]:
        r"""Choose the coherent genus-zero spin lift of the channel map.

        A descendant ``G_-1/2 V`` needs a square root of the local-coordinate
        derivative.  Independent principal square roots at the five
        punctures do not define a lift of a Mobius map and give spurious
        factors of ``i`` between oriented charts.  Writing
        ``f(z)=(az+b)/(cz+d)`` and ``s^2=ad-bc`` fixes all five roots from the
        single choice ``s=sqrt(det f)``.  The factors of ``i`` are the
        canonical-coordinate conversions when exactly one endpoint is at
        infinity.
        """

        transform = channel.mobius
        root_determinant = cmath.sqrt(complex(transform.determinant))
        result: list[complex] = []
        for ordered_index, label in enumerate(channel.ordering):
            source = positions[label]
            target_is_infinity = ordered_index == 4
            if source is None:
                if target_is_infinity:
                    root = root_determinant / complex(transform.a)
                else:
                    root = 1.0j * root_determinant / complex(transform.c)
            else:
                z_value = complex(source)
                if target_is_infinity:
                    root = (
                        1.0j
                        * root_determinant
                        / (complex(transform.a) * z_value + complex(transform.b))
                    )
                else:
                    root = root_determinant / (
                        complex(transform.c) * z_value + complex(transform.d)
                    )
            local_scale = complex(channel.local_scales[ordered_index])
            if abs(root * root - local_scale) > 2.0e-10 * max(
                1.0, abs(local_scale)
            ):
                raise ArithmeticError("incoherent Mobius spin lift")
            result.append(complex(root))
        return tuple(result)

    @classmethod
    def _component_covariance(
        cls,
        channel: LinearChannel,
        positions: Sequence[ProjectivePoint],
        ordered_weights: Sequence[complex],
        ordered_descendants: Sequence[int],
        *,
        antiholomorphic: bool,
    ) -> complex:
        logarithm = 0.0 + 0.0j
        spin_scales = cls._spin_local_scales(channel, positions)
        spin_factor = 1.0 + 0.0j
        for scale, spin_scale, weight, descendant in zip(
            channel.local_scales,
            spin_scales,
            ordered_weights,
            ordered_descendants,
        ):
            # Keep the antiholomorphic logarithm on the branch conjugate to
            # the holomorphic one.  Calling ``log(conj(scale))`` directly is
            # wrong on the negative-real cut (both principal logs would use
            # +i*pi), and produces spurious channel phases for nonintegral
            # weights.
            log_scale = cmath.log(complex(scale))
            if antiholomorphic:
                log_scale = log_scale.conjugate()
            logarithm += complex(weight) * log_scale
            if descendant:
                spin_factor *= (
                    spin_scale.conjugate()
                    if antiholomorphic
                    else spin_scale
                )
        return mpmath.exp(mpmath.mpc(logarithm)) * mpmath.mpc(spin_factor)

    def _chiral_block(
        self,
        channel: LinearChannel,
        positions: Sequence[ProjectivePoint],
        internal_momenta: tuple[complex, complex],
        sectors: SectorAssignment,
        descendants_by_label: Sequence[int],
        *,
        antiholomorphic: bool,
        omit_leading_edge: int | None = None,
        only_leading_edge: int | None = None,
    ) -> complex:
        ordered_weights = tuple(
            self.external_weights[label] for label in channel.ordering
        )
        ordered_descendants = tuple(
            int(descendants_by_label[label]) for label in channel.ordering
        )
        internal_weights = tuple(
            self.block_weight(momentum) for momentum in internal_momenta
        )
        block_key = (
            ordered_weights,
            internal_momenta,
            sectors,
            ordered_descendants,
        )
        if block_key not in self._block_cache:
            self._block_cache[block_key] = NSSphereLinearCRecursion(
                central_charge=self.block_central_charge,
                external_weights=ordered_weights,
                external_descendants=ordered_descendants,
                internal_weights=internal_weights,
                vertex_sectors=sectors,
                working_precision=self.block_working_precision,
                pole_tolerance=self.pole_tolerance,
            )
        block = self._block_cache[block_key]
        parities = block.compatible_level_parities()
        if parities is None:  # pragma: no cover - a sphere is a tree
            raise AssertionError("sphere block did not fix its edge parities")
        minimum_levels = [0, 0]
        accumulated_maxima: list[int | None] = [None, None]
        evaluation_maxima = list(self.global_max_twice_levels)
        if omit_leading_edge is not None and only_leading_edge is not None:
            raise ValueError("cannot omit and select the same leading edge state")
        if omit_leading_edge is not None:
            selected_edge = int(omit_leading_edge)
            if selected_edge not in (0, 1):
                raise ValueError("omit_leading_edge must be zero, one, or None")
            minimum_levels[selected_edge] = parities[selected_edge] + 2
        if only_leading_edge is not None:
            selected_edge = int(only_leading_edge)
            if selected_edge not in (0, 1):
                raise ValueError("only_leading_edge must be zero, one, or None")
            accumulated_maxima[selected_edge] = parities[selected_edge]
            evaluation_maxima[selected_edge] = parities[selected_edge]
        holomorphic_logs = tuple(
            cmath.log(value) for value in (channel.q1, channel.q2)
        )
        q_logs = tuple(
            value.conjugate() for value in holomorphic_logs
        ) if antiholomorphic else holomorphic_logs
        q_values = (
            channel.q1.conjugate(),
            channel.q2.conjugate(),
        ) if antiholomorphic else (channel.q1, channel.q2)
        with mpmath.workdps(self.block_working_precision):
            if self.recursion_max_twice_level is None:
                reduced = block.series_value(
                    q_values,
                    evaluation_maxima,
                    max_total_twice_level=self.global_max_total_twice_level,
                    q_log_values=q_logs,
                    minimum_twice_levels=minimum_levels,
                )
            else:
                reduced = block.recursive_series_value(
                    q_values,
                    self.recursion_max_twice_level,
                    self.global_max_twice_levels,
                    global_max_total_twice_level=self.global_max_total_twice_level,
                    q_log_values=q_logs,
                    minimum_twice_levels=minimum_levels,
                    maximum_accumulated_twice_levels=accumulated_maxima,
                )
            cumulative = mpmath.mpc(
                ordered_weights[0] + 0.5 * ordered_descendants[0]
            )
            leading = mpmath.mpc(1)
            for edge, (q_log, internal_weight) in enumerate(
                zip(q_logs, internal_weights)
            ):
                cumulative += mpmath.mpc(
                    ordered_weights[edge + 1]
                    + 0.5 * ordered_descendants[edge + 1]
                )
                leading *= mpmath.exp(
                    mpmath.mpc(q_log) * (internal_weight - cumulative)
                )
            # BRY's scalar component blocks differ from the human-note
            # fixed-parity trilinear blocks by one minus sign per odd internal
            # edge.  For four points this is exactly the conversion used by
            # sphere_four_point.py for both unstarred and doubly-starred odd
            # blocks.
            scalar_phase = -1 if sum(parities) % 2 else 1
            # The BRY scalar component and the human-note algebraic state
            # differ at the two comb endpoints.  A descendant at zero has
            # phase -1; at infinity it has phase -i holomorphically and +i
            # antiholomorphically.  Their relative ratio is the coherent BPZ
            # spin lift fixed by channel reversal.  The common endpoint sign
            # is fixed independently by the BRY two-PCO OPE square
            # (c0+omega_i*omega_j)^2.
            endpoint_phase = (
                STANDARD_ZERO_DESCENDANT_PHASE ** ordered_descendants[0]
                * (
                    STANDARD_INFINITY_DESCENDANT_PHASE.conjugate()
                    if antiholomorphic
                    else STANDARD_INFINITY_DESCENDANT_PHASE
                ) ** ordered_descendants[-1]
            )
            covariance = self._component_covariance(
                channel,
                positions,
                ordered_weights,
                ordered_descendants,
                antiholomorphic=antiholomorphic,
            )
            return (
                scalar_phase
                * endpoint_phase
                * covariance
                * leading
                * reduced
            )

    @staticmethod
    def _timelike_boson_factor(
        positions: Sequence[ProjectivePoint], signed_energies: Sequence[complex]
    ) -> complex:
        if len(positions) != len(signed_energies):
            raise ValueError("positions and signed_energies must have equal length")
        logarithm = 0.0 + 0.0j
        for left in range(len(positions)):
            if positions[left] is None:
                continue
            for right in range(left + 1, len(positions)):
                if positions[right] is None:
                    continue
                separation = abs(complex(positions[left]) - complex(positions[right]))
                if separation == 0.0:
                    raise ZeroDivisionError("the timelike boson factor hit a collision")
                logarithm -= (
                    2.0
                    * complex(signed_energies[left])
                    * complex(signed_energies[right])
                    * math.log(separation)
                )
        return mpmath.exp(mpmath.mpc(logarithm))

    def _fourpoint_single_w_chiral_block(
        self,
        *,
        external_momenta: Sequence[Number],
        internal_momentum: Number,
        sectors: tuple[int, int],
        descendant_slot: int,
        z: complex,
        antiholomorphic: bool,
    ) -> complex:
        """Four-point block with one external ``G_-1/2`` component.

        This lower-point block is the factor left on a five-point PCO face.
        It is evaluated in the standard ``(0,z,1,infinity)`` frame.
        """

        momenta = tuple(complex(value) for value in external_momenta)
        if len(momenta) != 4:
            raise ValueError("external_momenta must contain four values")
        if descendant_slot not in range(4):
            raise ValueError("descendant_slot must lie in range(4)")
        descendants = tuple(int(index == descendant_slot) for index in range(4))
        weights = tuple(self.block_weight(value) for value in momenta)
        internal = _finite_complex("internal_momentum", internal_momentum)
        internal_weight = self.block_weight(internal)
        block_key = (
            "face-fourpoint",
            weights,
            internal,
            sectors,
            descendants,
        )
        if block_key not in self._block_cache:
            self._block_cache[block_key] = NSSphereLinearCRecursion(
                central_charge=self.block_central_charge,
                external_weights=weights,
                external_descendants=descendants,
                internal_weights=(internal_weight,),
                vertex_sectors=sectors,
                working_precision=self.block_working_precision,
                pole_tolerance=self.pole_tolerance,
            )
        block = self._block_cache[block_key]
        holomorphic_log = cmath.log(z)
        argument_log = (
            holomorphic_log.conjugate()
            if antiholomorphic
            else holomorphic_log
        )
        argument = z.conjugate() if antiholomorphic else z
        with mpmath.workdps(self.block_working_precision):
            if self.recursion_max_twice_level is None:
                reduced = block.series_value(
                    (argument,),
                    (self.global_max_twice_levels[1],),
                    max_total_twice_level=self.global_max_total_twice_level,
                    q_log_values=(argument_log,),
                )
            else:
                reduced = block.recursive_series_value(
                    (argument,),
                    self.recursion_max_twice_level,
                    (self.global_max_twice_levels[1],),
                    global_max_total_twice_level=self.global_max_total_twice_level,
                    q_log_values=(argument_log,),
                )
            exponent = internal_weight - (
                weights[0]
                + 0.5 * descendants[0]
                + weights[1]
                + 0.5 * descendants[1]
            )
            edge_parity = block.compatible_level_parities()[0]
            scalar_phase = -1 if edge_parity else 1
            endpoint_phase = (
                STANDARD_ZERO_DESCENDANT_PHASE ** descendants[0]
                * (
                    STANDARD_INFINITY_DESCENDANT_PHASE.conjugate()
                    if antiholomorphic
                    else STANDARD_INFINITY_DESCENDANT_PHASE
                ) ** descendants[-1]
            )
            return complex(
                scalar_phase
                * endpoint_phase
                * mpmath.exp(mpmath.mpc(argument_log) * exponent)
                * reduced
            )

    def _fourpoint_single_w_sector_kernel(
        self,
        *,
        external_momenta: Sequence[Number],
        signed_energies: Sequence[Number],
        internal_momentum: Number,
        sectors: tuple[int, int],
        descendant_slot: int,
        z: Number,
    ) -> complex:
        """Return one lower four-point block product without structures."""

        momenta = tuple(complex(value) for value in external_momenta)
        energies = tuple(complex(value) for value in signed_energies)
        if len(momenta) != 4 or len(energies) != 4:
            raise ValueError("four external momenta and signed energies are required")
        if sectors not in ((0, 1), (1, 0)):
            raise ValueError("a one-W four-point block needs sectors (0,1) or (1,0)")
        z_value = _finite_complex("z", z)
        holomorphic = self._fourpoint_single_w_chiral_block(
            external_momenta=momenta,
            internal_momentum=internal_momentum,
            sectors=sectors,
            descendant_slot=descendant_slot,
            z=z_value,
            antiholomorphic=False,
        )
        antiholomorphic = self._fourpoint_single_w_chiral_block(
            external_momenta=momenta,
            internal_momentum=internal_momentum,
            sectors=sectors,
            descendant_slot=descendant_slot,
            z=z_value,
            antiholomorphic=True,
        )
        return complex(
            self._timelike_boson_factor(
                (0.0 + 0.0j, z_value, 1.0 + 0.0j, INFINITY),
                energies,
            )
            * holomorphic
            * antiholomorphic
        )

    def fourpoint_single_w_momentum_integrand(
        self,
        *,
        external_momenta: Sequence[Number],
        signed_energies: Sequence[Number],
        internal_momentum: float,
        descendant_slot: int,
        z: Number,
    ) -> complex:
        r"""Return the lower four-point one-``W`` density, including ``dR/pi``."""

        momenta = tuple(complex(value) for value in external_momenta)
        energies = tuple(complex(value) for value in signed_energies)
        if len(momenta) != 4 or len(energies) != 4:
            raise ValueError("four external momenta and signed energies are required")
        if abs(sum(energies)) > 2.0e-12:
            raise ValueError("the lower four-point signed energies must sum to zero")
        z_value = _finite_complex("z", z)
        if z_value in (0.0, 1.0):
            raise ValueError("the lower four-point modulus must avoid 0 and 1")
        internal = float(internal_momentum)
        if not math.isfinite(internal) or internal < 0.0:
            raise ValueError("internal_momentum must be non-negative and finite")
        total = 0.0 + 0.0j
        descendants = tuple(int(index == descendant_slot) for index in range(4))
        for sectors in ((0, 1), (1, 0)):
            structures = (
                self._structure_constant(momenta[0], momenta[1], internal, sectors[0])
                * self._structure_constant(internal, momenta[2], momenta[3], sectors[1])
            )
            total += structures * self._fourpoint_single_w_sector_kernel(
                external_momenta=momenta,
                signed_energies=energies,
                internal_momentum=internal,
                sectors=sectors,
                descendant_slot=descendant_slot,
                z=z_value,
            )
        return total / math.pi

    def fourpoint_single_w_continued_integrand(
        self,
        *,
        external_momenta: Sequence[Number],
        signed_energies: Sequence[Number],
        descendant_slot: int,
        z: Number,
    ) -> complex:
        r"""Integrate the lower one-``W`` four-point contour plus residues."""

        momenta = tuple(complex(value) for value in external_momenta)
        energies = tuple(complex(value) for value in signed_energies)
        if len(momenta) != 4 or len(energies) != 4:
            raise ValueError("four external momenta and signed energies are required")
        if abs(sum(energies)) > 2.0e-12:
            raise ValueError("the lower four-point signed energies must sum to zero")
        z_value = _finite_complex("z", z)
        nodes = _legendre_interval(
            self.momentum_orders[1], self.momentum_maximum
        )
        total = 0.0 + 0.0j

        def apply_laurent_residue(
            coefficients: tuple[complex, ...], regular_function, pole: complex
        ) -> complex:
            value = coefficients[0] * regular_function(pole)
            if len(coefficients) >= 2:
                value += coefficients[1] * self._analytic_first_derivative(
                    regular_function, pole
                )
            if len(coefficients) >= 3:
                value += 0.5 * coefficients[2] * self._analytic_second_derivative(
                    regular_function, pole
                )
            return value

        for sectors in ((0, 1), (1, 0)):
            left_sector, right_sector = sectors

            def kernel(momentum: complex) -> complex:
                return self._fourpoint_single_w_sector_kernel(
                    external_momenta=momenta,
                    signed_energies=energies,
                    internal_momentum=momentum,
                    sectors=sectors,
                    descendant_slot=descendant_slot,
                    z=z_value,
                )

            for momentum, weight in nodes:
                p_value = complex(momentum)
                total += (
                    weight
                    * self._structure_constant(
                        momenta[0], momenta[1], p_value, left_sector
                    )
                    * self._structure_constant(
                        p_value, momenta[2], momenta[3], right_sector
                    )
                    * kernel(p_value)
                    / math.pi
                )

            left_poles = _positive_contour_structure_poles(
                momenta[0], momenta[1], left_sector
            )
            right_poles = _positive_contour_structure_poles(
                momenta[2], momenta[3], right_sector
            )
            if any(
                abs(left.momentum - right.momentum) < 1.0e-9
                for left in left_poles
                for right in right_poles
            ):
                raise ArithmeticError(
                    "the lower four-point contour has a coincident residue pole"
                )
            for pole in left_poles:
                coefficients = self._structure_laurent_coefficients(
                    momenta[0], momenta[1], pole, left_sector
                )

                def left_regular(value: complex) -> complex:
                    return (
                        self._structure_constant(
                            value, momenta[2], momenta[3], right_sector
                        )
                        * kernel(value)
                    )

                total += pole.contour_coefficient * apply_laurent_residue(
                    coefficients, left_regular, pole.momentum
                )
            for pole in right_poles:
                coefficients = self._structure_laurent_coefficients(
                    momenta[2], momenta[3], pole, right_sector
                )

                def right_regular(value: complex) -> complex:
                    return (
                        self._structure_constant(
                            momenta[0], momenta[1], value, left_sector
                        )
                        * kernel(value)
                    )

                total += pole.contour_coefficient * apply_laurent_residue(
                    coefficients, right_regular, pole.momentum
                )
        return complex(total)

    def two_pco_face_asymptotic_momentum_density(
        self,
        *,
        ordering: Sequence[int],
        normal_momentum: float,
        remaining_momentum: float,
        remaining_modulus: Number,
    ) -> complex:
        r"""Coefficient of ``|q_normal|^beta`` on a raised-pair face.

        The returned quantity is a density with respect to
        ``dP dR d2q_normal d2q_remaining`` and includes both ``1/pi``
        continuum measures.  The ordering must start with the two raised
        colliding labels.  Its final three entries select one of the six
        four-point crossing cells on that boundary divisor.
        """

        selected = tuple(int(label) for label in ordering)
        if len(selected) != 5 or set(selected) != set(range(5)):
            raise ValueError("ordering must permute labels 0,...,4")
        colliding = set(selected[:2])
        if not colliding <= set(PICTURE_ZERO_LABELS) or len(colliding) != 2:
            raise ValueError("the first two labels must be a raised PCO pair")
        remaining_raised = (set(PICTURE_ZERO_LABELS) - colliding).pop()
        descendant_slot = selected[2:].index(remaining_raised) + 1
        p_normal = float(normal_momentum)
        p_remaining = float(remaining_momentum)
        modulus = _finite_complex("remaining_modulus", remaining_modulus)
        if modulus in (0.0, 1.0):
            raise ValueError("remaining_modulus must avoid 0 and 1")
        left, right, third, fourth, fifth = selected
        omega_sum = self.external_momenta[left] + self.external_momenta[right]
        q_squared_over_four = (self.block_central_charge / 3.0 - 0.5) / 4.0
        beta = (
            -2.0
            - q_squared_over_four
            + p_normal * p_normal
            - omega_sum * omega_sum
        )
        c0 = (
            self.block_weight(self.external_momenta[left])
            + self.block_weight(self.external_momenta[right])
            - self.block_weight(p_normal)
        )
        pco_ope_square = (
            c0
            + self.external_momenta[left] * self.external_momenta[right]
        ) ** 2
        bubble = (
            self._structure_constant(
                self.external_momenta[left],
                self.external_momenta[right],
                p_normal,
                0,
            )
            * pco_ope_square
            / math.pi
        )
        lower = self.fourpoint_single_w_momentum_integrand(
            external_momenta=(
                p_normal,
                self.external_momenta[third],
                self.external_momenta[fourth],
                self.external_momenta[fifth],
            ),
            signed_energies=(
                self.signed_energies[left] + self.signed_energies[right],
                self.signed_energies[third],
                self.signed_energies[fourth],
                self.signed_energies[fifth],
            ),
            internal_momentum=p_remaining,
            descendant_slot=descendant_slot,
            z=modulus,
        )
        return complex(
            bubble
            * cmath.exp((complex(beta) + 2.0) * math.log(abs(modulus)))
            * lower
        )

    def two_pco_face_continued_asymptotic_normal_density(
        self,
        *,
        ordering: Sequence[int],
        normal_momentum: float,
        remaining_modulus: Number,
    ) -> complex:
        r"""Continued face coefficient at fixed normal continuum momentum.

        The normal ``D_12`` momentum remains on the positive real contour.
        The lower four-point function includes its own continuum and every
        crossed quotient-contour residue.  The result already contains the
        normal ``dP/pi`` bubble measure but not its Gauss weight.
        """

        selected = tuple(int(label) for label in ordering)
        if len(selected) != 5 or set(selected) != set(range(5)):
            raise ValueError("ordering must permute labels 0,...,4")
        colliding = set(selected[:2])
        if colliding != {1, 2}:
            raise ValueError("the corrected one-divisor face must be D_12")
        remaining_raised = (set(PICTURE_ZERO_LABELS) - colliding).pop()
        descendant_slot = selected[2:].index(remaining_raised) + 1
        p_normal = float(normal_momentum)
        if not math.isfinite(p_normal) or p_normal < 0.0:
            raise ValueError("normal_momentum must be non-negative and finite")
        modulus = _finite_complex("remaining_modulus", remaining_modulus)
        if modulus in (0.0, 1.0):
            raise ValueError("remaining_modulus must avoid 0 and 1")
        left, right, third, fourth, fifth = selected
        beta = self._two_pco_face_beta(selected, p_normal)
        c0 = (
            self.block_weight(self.external_momenta[left])
            + self.block_weight(self.external_momenta[right])
            - self.block_weight(p_normal)
        )
        pco_ope_square = (
            c0
            + self.external_momenta[left] * self.external_momenta[right]
        ) ** 2
        bubble = (
            self._structure_constant(
                self.external_momenta[left],
                self.external_momenta[right],
                p_normal,
                0,
            )
            * pco_ope_square
            / math.pi
        )
        lower = self.fourpoint_single_w_continued_integrand(
            external_momenta=(
                p_normal,
                self.external_momenta[third],
                self.external_momenta[fourth],
                self.external_momenta[fifth],
            ),
            signed_energies=(
                self.signed_energies[left] + self.signed_energies[right],
                self.signed_energies[third],
                self.signed_energies[fourth],
                self.signed_energies[fifth],
            ),
            descendant_slot=descendant_slot,
            z=modulus,
        )
        return complex(
            bubble
            * cmath.exp((beta + 2.0) * math.log(abs(modulus)))
            * lower
        )

    def _two_pco_face_beta(
        self, ordering: Sequence[int], normal_momentum: float
    ) -> complex:
        """Return the radial power in ``|q_normal|^beta``."""

        selected = tuple(int(label) for label in ordering)
        if len(selected) != 5 or set(selected) != set(range(5)):
            raise ValueError("ordering must permute labels 0,...,4")
        if (
            len(set(selected[:2])) != 2
            or not set(selected[:2]) <= set(PICTURE_ZERO_LABELS)
        ):
            raise ValueError("the first two labels must be a raised PCO pair")
        momentum = float(normal_momentum)
        if not math.isfinite(momentum) or momentum < 0.0:
            raise ValueError("normal_momentum must be non-negative and finite")
        omega_sum = (
            self.external_momenta[selected[0]]
            + self.external_momenta[selected[1]]
        )
        q_squared_over_four = (self.block_central_charge / 3.0 - 0.5) / 4.0
        return complex(
            -2.0
            - q_squared_over_four
            + momentum * momentum
            - omega_sum * omega_sum
        )

    def two_pco_face_counterterm_q_density(
        self,
        *,
        ordering: Sequence[int],
        normal_coordinate: Number,
        remaining_modulus: Number,
    ) -> complex:
        r"""Integrate the leading face density over both Liouville momenta.

        The result multiplies ``d2q_normal d2q_remaining`` in the selected
        linear plumbing chart.  It is the pointwise counterterm subtracted
        inside the PCO collar before its normal radial integral is restored
        by analytic finite part.
        """

        normal = _finite_complex("normal_coordinate", normal_coordinate)
        if normal == 0.0:
            raise ValueError("normal_coordinate must be nonzero")
        modulus = _finite_complex("remaining_modulus", remaining_modulus)
        node_sets = tuple(
            _legendre_interval(order, self.momentum_maximum)
            for order in self.momentum_orders
        )
        total = 0.0 + 0.0j
        for first, second in product(*node_sets):
            p_normal, weight_normal = first
            p_remaining, weight_remaining = second
            beta = self._two_pco_face_beta(ordering, p_normal)
            coefficient = self.two_pco_face_asymptotic_momentum_density(
                ordering=ordering,
                normal_momentum=p_normal,
                remaining_momentum=p_remaining,
                remaining_modulus=modulus,
            )
            total += (
                weight_normal
                * weight_remaining
                * coefficient
                * cmath.exp(beta * math.log(abs(normal)))
            )
        return complex(total)

    def two_pco_face_finite_part_density(
        self,
        *,
        ordering: Sequence[int],
        remaining_modulus: Number,
        collar_radius: float,
    ) -> complex:
        r"""Apply the BRY radial finite part normal to one PCO face.

        The returned density multiplies ``d2q_remaining``.  For
        ``alpha=(beta+2)/2`` the normal integral is continued as
        ``pi*rho^(2 alpha)/alpha``; its logarithmic finite term is
        ``2*pi*log(rho)`` at ``alpha=0``.
        """

        modulus = _finite_complex("remaining_modulus", remaining_modulus)
        radius = float(collar_radius)
        if not math.isfinite(radius) or not 0.0 < radius < 1.0:
            raise ValueError("collar_radius must lie in (0,1)")
        node_sets = tuple(
            _legendre_interval(order, self.momentum_maximum)
            for order in self.momentum_orders
        )
        total = 0.0 + 0.0j
        for first, second in product(*node_sets):
            p_normal, weight_normal = first
            p_remaining, weight_remaining = second
            beta = self._two_pco_face_beta(ordering, p_normal)
            radial_finite_part = _complex_radial_finite_part(beta, radius)
            coefficient = self.two_pco_face_asymptotic_momentum_density(
                ordering=ordering,
                normal_momentum=p_normal,
                remaining_momentum=p_remaining,
                remaining_modulus=modulus,
            )
            total += (
                weight_normal
                * weight_remaining
                * radial_finite_part
                * coefficient
            )
        return complex(total)

    def two_pco_face_continued_counterterm_q_density(
        self,
        *,
        ordering: Sequence[int],
        normal_coordinate: Number,
        remaining_modulus: Number,
    ) -> complex:
        """Return the continued ``D_12`` counterterm in linear-q measure."""

        normal = _finite_complex("normal_coordinate", normal_coordinate)
        if normal == 0.0:
            raise ValueError("normal_coordinate must be nonzero")
        total = 0.0 + 0.0j
        for momentum, weight in _legendre_interval(
            self.momentum_orders[0], self.momentum_maximum
        ):
            beta = self._two_pco_face_beta(ordering, momentum)
            coefficient = self.two_pco_face_continued_asymptotic_normal_density(
                ordering=ordering,
                normal_momentum=momentum,
                remaining_modulus=remaining_modulus,
            )
            total += (
                weight
                * coefficient
                * cmath.exp(beta * math.log(abs(normal)))
            )
        return complex(total)

    def two_pco_face_continued_finite_part_density(
        self,
        *,
        ordering: Sequence[int],
        remaining_modulus: Number,
        collar_radius: float,
    ) -> complex:
        """Apply the complex normal finite part to the continued D_12 face."""

        radius = float(collar_radius)
        if not math.isfinite(radius) or not 0.0 < radius < 1.0:
            raise ValueError("collar_radius must lie in (0,1)")
        total = 0.0 + 0.0j
        for momentum, weight in _legendre_interval(
            self.momentum_orders[0], self.momentum_maximum
        ):
            beta = self._two_pco_face_beta(ordering, momentum)
            coefficient = self.two_pco_face_continued_asymptotic_normal_density(
                ordering=ordering,
                normal_momentum=momentum,
                remaining_modulus=remaining_modulus,
            )
            total += (
                weight
                * coefficient
                * _complex_radial_finite_part(beta, radius)
            )
        return complex(total)

    def boundary_radial_beta(
        self,
        ordering: Sequence[int],
        internal_momentum: float,
        *,
        side: str = "left",
    ) -> complex:
        r"""Return the universal NS tachyon face power ``|q|^beta``.

        ``side='left'`` uses the first cherry and the first internal
        momentum.  ``side='right'`` uses the final cherry and the second
        internal momentum.  The channel energy is the sum of *signed*
        timelike momenta, which is essential for an incoming--outgoing face.
        """

        selected = tuple(int(label) for label in ordering)
        if len(selected) != 5 or set(selected) != set(range(5)):
            raise ValueError("ordering must permute labels 0,...,4")
        momentum = float(internal_momentum)
        if not math.isfinite(momentum) or momentum < 0.0:
            raise ValueError("internal_momentum must be non-negative and finite")
        if side == "left":
            pair = selected[:2]
        elif side == "right":
            pair = selected[-2:]
        else:
            raise ValueError("side must be 'left' or 'right'")
        channel_energy = sum(self.signed_energies[label] for label in pair)
        q_squared_over_four = (self.block_central_charge / 3.0 - 0.5) / 4.0
        return complex(
            -2.0
            - q_squared_over_four
            + momentum * momentum
            - channel_energy * channel_energy
        )

    def _continued_boundary_radial_beta(
        self,
        ordering: Sequence[int],
        internal_momentum: Number,
        *,
        side: str,
        edge_sector: int,
    ) -> complex:
        """Return the complex continued exponent for one residue term."""

        selected = tuple(int(label) for label in ordering)
        if len(selected) != 5 or set(selected) != set(range(5)):
            raise ValueError("ordering must permute labels 0,...,4")
        momentum = _finite_complex("internal_momentum", internal_momentum)
        if edge_sector not in (0, 1):
            raise ValueError("edge_sector must be zero or one")
        if side == "left":
            pair = selected[:2]
        elif side == "right":
            pair = selected[-2:]
        else:
            raise ValueError("side must be 'left' or 'right'")
        channel_energy = sum(self.signed_energies[label] for label in pair)
        q_squared_over_four = (self.block_central_charge / 3.0 - 0.5) / 4.0
        return complex(
            -2.0
            - q_squared_over_four
            + momentum * momentum
            - channel_energy * channel_energy
            + int(edge_sector)
        )

    def linear_q_momentum_density(
        self,
        *,
        ordering: Sequence[int],
        q1: Number,
        q2: Number,
        internal_momenta: Sequence[float],
    ) -> complex:
        r"""Return the fixed-momentum density multiplying ``d2q1 d2q2``."""

        selected = tuple(int(label) for label in ordering)
        first = _finite_complex("q1", q1)
        second = _finite_complex("q2", q2)
        if not 0.0 < abs(first) < 1.0 or not 0.0 < abs(second) < 1.0:
            raise ValueError("linear plumbing coordinates must lie in the punctured bidisc")
        positions = _to_fixed_gauge(first, second, selected)
        channel = linear_channel_from_ordering(positions, selected)
        jacobian = linear_channel_complex_jacobian_to_chart(
            first,
            second,
            selected,
            fixed_zero=FIXED_ZERO_LABEL,
            fixed_one=FIXED_ONE_LABEL,
            fixed_infinity=FIXED_INFINITY_LABEL,
            moving_labels=MOVING_LABELS,
        )
        return complex(
            self.momentum_integrand(
                positions, internal_momenta, channel=channel
            )
            * abs(jacobian) ** 2
        )

    def boundary_face_leading_momentum_coefficient(
        self,
        *,
        ordering: Sequence[int],
        normal_momentum: float,
        remaining_momentum: float,
        remaining_modulus: Number,
        projection_radius: float = 1.0e-5,
    ) -> complex:
        r"""Project the spin-zero leading coefficient on any boundary face.

        This target-blind projection uses the same complete PCO component
        sum as the bulk integrand.  ``projection_radius`` is exposed so its
        limit can be certified independently; the face QMC keeps the
        remaining modulus outside the finite-part collar, making the limit
        uniform.
        """

        modulus = _finite_complex("remaining_modulus", remaining_modulus)
        radius = float(projection_radius)
        if not math.isfinite(radius) or radius <= 0.0:
            raise ValueError("projection_radius must be positive and finite")
        if radius >= min(0.01, 0.1 * abs(modulus)):
            raise ValueError(
                "projection_radius must be small compared with the remaining modulus"
            )
        beta = self.boundary_radial_beta(
            ordering, normal_momentum, side="left"
        )
        if abs(beta.imag) > 2.0e-12:
            raise ValueError("the boundary projection requires a real radial power")
        density = self.linear_q_momentum_density(
            ordering=ordering,
            q1=radius,
            q2=modulus,
            internal_momenta=(normal_momentum, remaining_momentum),
        )
        return complex(density / radius**beta.real)

    def boundary_face_finite_part_density(
        self,
        *,
        ordering: Sequence[int],
        remaining_modulus: Number,
        collar_radius: float,
        projection_radius: float = 1.0e-5,
    ) -> complex:
        """Return the leading normal finite part multiplying ``d2q_remaining``."""

        node_sets = tuple(
            _legendre_interval(order, self.momentum_maximum)
            for order in self.momentum_orders
        )
        total = 0.0 + 0.0j
        for first, second in product(*node_sets):
            p_normal, weight_normal = first
            p_remaining, weight_remaining = second
            beta = self.boundary_radial_beta(
                ordering, p_normal, side="left"
            )
            if abs(beta.imag) > 2.0e-12:
                raise ValueError("the face finite part requires a real radial power")
            coefficient = self.boundary_face_leading_momentum_coefficient(
                ordering=ordering,
                normal_momentum=p_normal,
                remaining_momentum=p_remaining,
                remaining_modulus=remaining_modulus,
                projection_radius=projection_radius,
            )
            total += (
                weight_normal
                * weight_remaining
                * _radial_finite_part(beta.real, collar_radius)
                * coefficient
            )
        return complex(total)

    def boundary_corner_leading_momentum_coefficient(
        self,
        *,
        ordering: Sequence[int],
        left_momentum: float,
        right_momentum: float,
        projection_radius: float = 1.0e-5,
    ) -> complex:
        """Project the double-leading coefficient at a compatible corner."""

        radius = float(projection_radius)
        if not math.isfinite(radius) or not 1.0e-7 <= radius < 1.0e-3:
            raise ValueError("corner projection_radius must lie in [1e-7,1e-3)")
        left_beta = self.boundary_radial_beta(
            ordering, left_momentum, side="left"
        )
        right_beta = self.boundary_radial_beta(
            ordering, right_momentum, side="right"
        )
        if abs(left_beta.imag) > 2.0e-12 or abs(right_beta.imag) > 2.0e-12:
            raise ValueError("the corner projection requires real radial powers")
        density = self.linear_q_momentum_density(
            ordering=ordering,
            q1=radius,
            q2=radius,
            internal_momenta=(left_momentum, right_momentum),
        )
        return complex(
            density / radius ** (left_beta.real + right_beta.real)
        )

    def boundary_corner_finite_part(
        self,
        *,
        ordering: Sequence[int],
        collar_radius: float,
        projection_radius: float = 1.0e-5,
    ) -> complex:
        """Apply both commuting radial finite parts at one boundary corner."""

        node_sets = tuple(
            _legendre_interval(order, self.momentum_maximum)
            for order in self.momentum_orders
        )
        total = 0.0 + 0.0j
        for first, second in product(*node_sets):
            left_momentum, left_weight = first
            right_momentum, right_weight = second
            left_beta = self.boundary_radial_beta(
                ordering, left_momentum, side="left"
            )
            right_beta = self.boundary_radial_beta(
                ordering, right_momentum, side="right"
            )
            coefficient = self.boundary_corner_leading_momentum_coefficient(
                ordering=ordering,
                left_momentum=left_momentum,
                right_momentum=right_momentum,
                projection_radius=projection_radius,
            )
            total += (
                left_weight
                * right_weight
                * _radial_finite_part(left_beta.real, collar_radius)
                * _radial_finite_part(right_beta.real, collar_radius)
                * coefficient
            )
        return complex(total)

    def _sector_component_kernel(
        self,
        positions: Sequence[ProjectivePoint],
        internal_momenta: tuple[complex, complex],
        sectors: SectorAssignment,
        channel: LinearChannel,
    ) -> complex:
        """Return timelike times chiral blocks, with no structures or measure."""

        normalized_positions = _validate_positions(positions)
        holomorphic_terms = pco_chiral_terms(
            positions=normalized_positions,
            signed_energies=self.signed_energies,
            operator_order=channel.ordering,
        )
        antiholomorphic_terms = pco_chiral_terms(
            positions=tuple(
                None if value is None else complex(value).conjugate()
                for value in normalized_positions
            ),
            signed_energies=self.signed_energies,
            operator_order=channel.ordering,
        )
        holomorphic_values = tuple(
            term.coefficient
            * self._chiral_block(
                channel,
                normalized_positions,
                internal_momenta,
                sectors,
                term.liouville_descendants,
                antiholomorphic=False,
            )
            for term in holomorphic_terms
        )
        antiholomorphic_values = tuple(
            term.coefficient
            * self._chiral_block(
                channel,
                normalized_positions,
                internal_momenta,
                sectors,
                term.liouville_descendants,
                antiholomorphic=True,
            )
            for term in antiholomorphic_terms
        )
        # BRY's nonchiral picture-zero vertex does not factor into two
        # independent chiral polynomials once three PCOs are present.  If a
        # vertex supplies both psi^0 and tilde-psi^0, their cocycle/reordering
        # contributes one minus sign.  Hence a holomorphic timelike subset
        # S and an antiholomorphic subset T carry (-1)^|S intersect T|.
        # With only two PCOs the allowed subsets are empty or the full pair,
        # so this sign is always +1; that is why the BRY four-point G/H/J
        # regression alone cannot detect it.
        pco_component_sum = mpmath.mpc(0)
        for holomorphic_term, holomorphic_value in zip(
            holomorphic_terms, holomorphic_values
        ):
            holomorphic_timelike = set(holomorphic_term.timelike_labels)
            for antiholomorphic_term, antiholomorphic_value in zip(
                antiholomorphic_terms, antiholomorphic_values
            ):
                overlap = len(
                    holomorphic_timelike.intersection(
                        antiholomorphic_term.timelike_labels
                    )
                )
                pco_component_sum += (
                    (-1 if overlap % 2 else 1)
                    * holomorphic_value
                    * antiholomorphic_value
                )
        return (
            self._timelike_boson_factor(
                normalized_positions, self.signed_energies
            )
            * pco_component_sum
        )

    def _sector_component_kernel_boundary_remainder(
        self,
        positions: Sequence[ProjectivePoint],
        internal_momenta: tuple[complex, complex],
        sectors: SectorAssignment,
        channel: LinearChannel,
        *,
        boundary_edge: int,
    ) -> complex:
        r"""Return the block with one endpoint primary state removed.

        This evaluates the descendant remainder directly from the all-c
        coefficient series.  It is therefore stable arbitrarily deep in a
        collar, where evaluating the full primary and subtracting its
        factorized face coefficient would lose the remainder to catastrophic
        cancellation.  ``boundary_edge`` is zero for the left cherry and one
        for the right cherry.
        """

        selected_edge = int(boundary_edge)
        if selected_edge not in (0, 1):
            raise ValueError("boundary_edge must be zero or one")
        normalized_positions = _validate_positions(positions)
        holomorphic_terms = pco_chiral_terms(
            positions=normalized_positions,
            signed_energies=self.signed_energies,
            operator_order=channel.ordering,
        )
        antiholomorphic_terms = pco_chiral_terms(
            positions=tuple(
                None if value is None else complex(value).conjugate()
                for value in normalized_positions
            ),
            signed_energies=self.signed_energies,
            operator_order=channel.ordering,
        )

        def values(terms, *, antiholomorphic: bool, remainder: bool):
            return tuple(
                term.coefficient
                * self._chiral_block(
                    channel,
                    normalized_positions,
                    internal_momenta,
                    sectors,
                    term.liouville_descendants,
                    antiholomorphic=antiholomorphic,
                    omit_leading_edge=(selected_edge if remainder else None),
                )
                for term in terms
            )

        holomorphic_full = values(
            holomorphic_terms, antiholomorphic=False, remainder=False
        )
        holomorphic_remainder = values(
            holomorphic_terms, antiholomorphic=False, remainder=True
        )
        antiholomorphic_full = values(
            antiholomorphic_terms, antiholomorphic=True, remainder=False
        )
        antiholomorphic_remainder = values(
            antiholomorphic_terms, antiholomorphic=True, remainder=True
        )

        component_sum = mpmath.mpc(0)
        for holomorphic_term, full_h, remainder_h in zip(
            holomorphic_terms, holomorphic_full, holomorphic_remainder
        ):
            holomorphic_timelike = set(holomorphic_term.timelike_labels)
            for antiholomorphic_term, full_a, remainder_a in zip(
                antiholomorphic_terms,
                antiholomorphic_full,
                antiholomorphic_remainder,
            ):
                overlap = len(
                    holomorphic_timelike.intersection(
                        antiholomorphic_term.timelike_labels
                    )
                )
                # full_h*full_a - leading_h*leading_a, written without
                # subtracting two nearly equal nonchiral products.
                stable_remainder = (
                    remainder_h * full_a
                    + (full_h - remainder_h) * remainder_a
                )
                component_sum += (
                    (-1 if overlap % 2 else 1) * stable_remainder
                )
        return (
            self._timelike_boson_factor(
                normalized_positions, self.signed_energies
            )
            * component_sum
        )

    def _sector_component_kernel_boundary_primary(
        self,
        positions: Sequence[ProjectivePoint],
        internal_momenta: tuple[complex, complex],
        sectors: SectorAssignment,
        channel: LinearChannel,
        *,
        boundary_edge: int,
    ) -> complex:
        """Return only the leading continuum state on one endpoint edge."""

        selected_edge = int(boundary_edge)
        if selected_edge not in (0, 1):
            raise ValueError("boundary_edge must be zero or one")
        normalized_positions = _validate_positions(positions)
        holomorphic_terms = pco_chiral_terms(
            positions=normalized_positions,
            signed_energies=self.signed_energies,
            operator_order=channel.ordering,
        )
        antiholomorphic_terms = pco_chiral_terms(
            positions=tuple(
                None if value is None else complex(value).conjugate()
                for value in normalized_positions
            ),
            signed_energies=self.signed_energies,
            operator_order=channel.ordering,
        )

        def values(terms, *, antiholomorphic: bool):
            return tuple(
                term.coefficient
                * self._chiral_block(
                    channel,
                    normalized_positions,
                    internal_momenta,
                    sectors,
                    term.liouville_descendants,
                    antiholomorphic=antiholomorphic,
                    only_leading_edge=selected_edge,
                )
                for term in terms
            )

        holomorphic_values = values(
            holomorphic_terms, antiholomorphic=False
        )
        antiholomorphic_values = values(
            antiholomorphic_terms, antiholomorphic=True
        )
        component_sum = mpmath.mpc(0)
        for holomorphic_term, holomorphic_value in zip(
            holomorphic_terms, holomorphic_values
        ):
            holomorphic_timelike = set(holomorphic_term.timelike_labels)
            for antiholomorphic_term, antiholomorphic_value in zip(
                antiholomorphic_terms, antiholomorphic_values
            ):
                overlap = len(
                    holomorphic_timelike.intersection(
                        antiholomorphic_term.timelike_labels
                    )
                )
                component_sum += (
                    (-1 if overlap % 2 else 1)
                    * holomorphic_value
                    * antiholomorphic_value
                )
        return (
            self._timelike_boson_factor(
                normalized_positions, self.signed_energies
            )
            * component_sum
        )

    def momentum_integrand(
        self,
        positions: Sequence[ProjectivePoint],
        internal_momenta: Sequence[Number],
        *,
        channel: LinearChannel | None = None,
    ) -> complex:
        """Return the complete PCO density at fixed ``(P1,P2)``.

        The result includes ``dP1 dP2 / pi^2`` but not the quadrature weights.
        """

        normalized_positions = _validate_positions(positions)
        momenta = tuple(
            _finite_complex(f"internal_momenta[{index}]", value)
            for index, value in enumerate(internal_momenta)
        )
        if len(momenta) != 2:
            raise ValueError("internal_momenta must contain two finite values")
        active_channel = (
            best_linear_channels(normalized_positions, limit=1)[0]
            if channel is None
            else channel
        )
        if active_channel.score >= 1.0:
            raise ValueError("the supplied channel is outside |q_i|<1")
        ordered_external_momenta = tuple(
            self.external_momenta[label] for label in active_channel.ordering
        )
        total = 0.0 + 0.0j
        for sectors in ODD_SECTOR_ASSIGNMENTS:
            total += (
                self._structure_product(
                    ordered_external_momenta,
                    momenta,  # type: ignore[arg-type]
                    sectors,
                )
                * self._sector_component_kernel(
                    normalized_positions,
                    momenta,  # type: ignore[arg-type]
                    sectors,
                    active_channel,
                )
            )
        return complex(total / math.pi**2)

    def integrand_positions(
        self,
        positions: Sequence[ProjectivePoint],
        *,
        channel: LinearChannel | None = None,
    ) -> complex:
        """Integrate the two momenta in the matter density at fixed punctures.

        This deliberately excludes the two-picture-minus-one superghost
        correlator.  Use ``fixed_gauge_integrand_positions`` for the full
        density multiplying ``d2z d2w`` in the physical gauge.
        """

        normalized_positions = _validate_positions(positions)
        active_channel = (
            best_linear_channels(normalized_positions, limit=1)[0]
            if channel is None
            else channel
        )
        if active_channel.score >= 1.0:
            raise ValueError("the supplied channel is outside |q_i|<1")
        node_sets = tuple(
            _legendre_interval(order, self.momentum_maximum)
            for order in self.momentum_orders
        )
        total = 0.0 + 0.0j
        for first, second in product(*node_sets):
            p1, weight1 = first
            p2, weight2 = second
            total += weight1 * weight2 * self.momentum_integrand(
                normalized_positions,
                (p1, p2),
                channel=active_channel,
            )
        return complex(total)

    def fixed_gauge_integrand_positions(
        self,
        positions: Sequence[ProjectivePoint],
        *,
        channel: LinearChannel | None = None,
    ) -> complex:
        """Return the full ``(infinity,1,0,z,w)`` density."""

        normalized = _validate_positions(positions)
        if (
            normalized[FIXED_INFINITY_LABEL] is not None
            or abs(complex(normalized[FIXED_ONE_LABEL]) - 1.0) > 2.0e-12
            or abs(complex(normalized[FIXED_ZERO_LABEL])) > 2.0e-12
        ):
            raise ValueError("positions are not in the (infinity,1,0,z,w) gauge")
        return complex(
            _superghost_pair_factor(normalized)
            * self.integrand_positions(normalized, channel=channel)
        )

    def continued_middle_line_terms_positions(
        self,
        positions: Sequence[ProjectivePoint],
        *,
        channel: LinearChannel,
    ) -> tuple[MovingMiddleResidueTerm, ...]:
        r"""Return the individual moving-middle residue-line contributions.

        Each value includes the first-momentum Gauss weight, ``dP1/pi``, and
        BRY's ``-2i`` contour coefficient, but excludes the superghost and
        moduli-coordinate Jacobian.  Keeping the terms separate exposes the
        internal momenta needed for the analytic all-``c`` corner finite
        part.
        """

        normalized_positions = _validate_positions(positions)
        active_channel = channel
        if active_channel.ordering.index(0) in (3, 4):
            active_channel = linear_channel_from_ordering(
                normalized_positions, tuple(reversed(active_channel.ordering))
            )
        if active_channel.score >= 1.0 or active_channel.ordering.index(0) > 2:
            raise ValueError("the residue channel is not a convergent oriented comb")

        ordered_external = tuple(
            self.external_momenta[label] for label in active_channel.ordering
        )
        pa, pb, pc, pd, pe = ordered_external
        first_nodes = _legendre_interval(
            self.momentum_orders[0], self.momentum_maximum
        )
        result: list[MovingMiddleResidueTerm] = []

        def apply_laurent_residue(
            coefficients: tuple[complex, ...],
            regular_function,
            pole_value: complex,
        ) -> complex:
            value = coefficients[0] * regular_function(pole_value)
            if len(coefficients) >= 2:
                value += coefficients[1] * self._analytic_first_derivative(
                    regular_function, pole_value
                )
            if len(coefficients) >= 3:
                value += 0.5 * coefficients[2] * self._analytic_second_derivative(
                    regular_function, pole_value
                )
            return value

        for sectors in ODD_SECTOR_ASSIGNMENTS:
            sector_left, sector_middle, sector_right = sectors
            for first_node in first_nodes:
                p1, weight1 = first_node
                p1_value = complex(p1)
                for middle_pole in _positive_contour_structure_poles(
                    p1_value, pc, sector_middle
                ):
                    p2 = middle_pole.momentum
                    middle_laurent = self._structure_laurent_coefficients(
                        p1_value,
                        pc,
                        middle_pole,
                        sector_middle,
                    )

                    def middle_regular(p2_value: complex) -> complex:
                        return (
                            self._structure_constant(
                                pa, pb, p1_value, sector_left
                            )
                            * self._structure_constant(
                                p2_value, pd, pe, sector_right
                            )
                            * self._sector_component_kernel(
                                normalized_positions,
                                (p1_value, p2_value),
                                sectors,
                                active_channel,
                            )
                        )

                    residue = apply_laurent_residue(
                        middle_laurent, middle_regular, p2
                    )
                    result.append(
                        MovingMiddleResidueTerm(
                            first_momentum=p1_value,
                            second_pole=middle_pole,
                            sectors=sectors,
                            value=(
                                weight1
                                * middle_pole.contour_coefficient
                                * residue
                                / math.pi
                            ),
                        )
                    )
        return tuple(result)

    def continued_integrand_components_positions(
        self,
        positions: Sequence[ProjectivePoint],
        *,
        channel: LinearChannel | None = None,
        excluded_middle_walls: Sequence[float] = (),
        subtracted_continuum_edge: int | None = None,
        primary_continuum_edge: int | None = None,
    ) -> ContinuedMomentumDensity:
        r"""Integrate the general vertical-continuation residue forest.

        Arbitrary finite outgoing momenta are supported.  Momentum
        conservation fixes the incoming momentum to their sum.  If the
        incoming leg is in the right cherry the comb is reversed; left-cherry
        and middle-incoming channels are then both supported.  The two-dimensional
        momentum forest is evaluated in the order ``P1`` then ``P2``:
        continuous, all ``P2`` residues (right endpoint and moving middle-
        trinion lines), left-endpoint residue integrals, and their nested
        intersections.  Every crossed quotient-contour pole carries BRY's
        coefficient ``-2i``.
        """

        normalized_positions = _validate_positions(positions)
        active_channel = (
            best_linear_channels(normalized_positions, limit=1)[0]
            if channel is None
            else channel
        )
        if active_channel.ordering.index(0) in (3, 4):
            active_channel = linear_channel_from_ordering(
                normalized_positions, tuple(reversed(active_channel.ordering))
            )
        if active_channel.score >= 1.0 or active_channel.ordering.index(0) > 2:
            raise ValueError("the residue channel is not a convergent oriented comb")
        if subtracted_continuum_edge not in (None, 0, 1):
            raise ValueError("subtracted_continuum_edge must be zero, one, or None")
        if primary_continuum_edge not in (None, 0, 1):
            raise ValueError("primary_continuum_edge must be zero, one, or None")
        if (
            subtracted_continuum_edge is not None
            and primary_continuum_edge is not None
        ):
            raise ValueError("select either the primary or its remainder, not both")

        ordered_external = tuple(
            self.external_momenta[label] for label in active_channel.ordering
        )
        pa, pb, pc, pd, pe = ordered_external
        middle_momentum = pc
        excluded_walls = tuple(float(value) for value in excluded_middle_walls)

        node_sets = tuple(
            _legendre_interval(order, self.momentum_maximum)
            for order in self.momentum_orders
        )
        continuous = 0.0 + 0.0j
        left_residues = 0.0 + 0.0j
        right_residues = 0.0 + 0.0j
        nested_residues = 0.0 + 0.0j
        middle_line_residues = 0.0 + 0.0j

        def sector_kernel(
            p1_value: complex,
            p2_value: complex,
            sectors: SectorAssignment,
            subtract_edge: int | None,
            primary_edge: int | None,
        ):
            momenta = (complex(p1_value), complex(p2_value))
            if subtract_edge is None and primary_edge is None:
                return self._sector_component_kernel(
                    normalized_positions, momenta, sectors, active_channel
                )
            if primary_edge is not None:
                return self._sector_component_kernel_boundary_primary(
                    normalized_positions,
                    momenta,
                    sectors,
                    active_channel,
                    boundary_edge=primary_edge,
                )
            return self._sector_component_kernel_boundary_remainder(
                normalized_positions,
                momenta,
                sectors,
                active_channel,
                boundary_edge=subtract_edge,
            )

        for sectors in ODD_SECTOR_ASSIGNMENTS:
            sector_left, sector_middle, sector_right = sectors
            left_poles = _positive_contour_structure_poles(
                pa, pb, sector_left
            )
            right_poles = _positive_contour_structure_poles(
                pd, pe, sector_right
            )

            def component_with_structures(p1_value: complex, p2_value: complex) -> complex:
                momenta = (complex(p1_value), complex(p2_value))
                return (
                    self._structure_constant(pa, pb, p1_value, sector_left)
                    * self._structure_constant(p1_value, pc, p2_value, sector_middle)
                    * self._structure_constant(p2_value, pd, pe, sector_right)
                    * sector_kernel(
                        momenta[0],
                        momenta[1],
                        sectors,
                        subtracted_continuum_edge,
                        primary_continuum_edge,
                    )
                )

            def apply_laurent_residue(
                coefficients: tuple[complex, ...],
                regular_function,
                pole_value: complex,
            ) -> complex:
                value = coefficients[0] * regular_function(pole_value)
                if len(coefficients) == 2:
                    value += coefficients[1] * self._analytic_first_derivative(
                        regular_function, pole_value
                    )
                elif len(coefficients) == 3:
                    value += coefficients[1] * self._analytic_first_derivative(
                        regular_function, pole_value
                    )
                    value += 0.5 * coefficients[2] * self._analytic_second_derivative(
                        regular_function, pole_value
                    )
                return value

            for first_node, second_node in product(*node_sets):
                p1, weight1 = first_node
                p2, weight2 = second_node
                continuous += (
                    weight1
                    * weight2
                    * component_with_structures(complex(p1), complex(p2))
                    / math.pi**2
                )

            for right_pole in right_poles:
                p2 = right_pole.momentum
                right_laurent = self._structure_laurent_coefficients(
                    pd, pe, right_pole, sector_right
                )
                for first_node in node_sets[0]:
                    p1, weight1 = first_node
                    p1_value = complex(p1)

                    def right_regular(p2_value: complex) -> complex:
                        return (
                            self._structure_constant(pa, pb, p1_value, sector_left)
                            * self._structure_constant(
                                p1_value, pc, p2_value, sector_middle
                            )
                            * sector_kernel(
                                p1_value,
                                p2_value,
                                sectors,
                                0 if subtracted_continuum_edge == 0 else None,
                                0 if primary_continuum_edge == 0 else None,
                            )
                        )

                    right_residue = apply_laurent_residue(
                        right_laurent, right_regular, p2
                    )
                    right_residues += (
                        weight1
                        * right_pole.contour_coefficient
                        * right_residue
                        / math.pi
                    )

            # When the incoming field occupies the middle leaf, continuing
            # C(P1,4*omega,P2) crosses moving P2 sum/difference poles for each
            # real P1.  (For an outgoing middle leaf in this chamber the list
            # is empty.)  These are one-dimensional residue lines, not
            # endpoint residues, but belong to the same second-contour layer.
            for first_node in node_sets[0]:
                p1, weight1 = first_node
                p1_value = complex(p1)
                middle_poles = _positive_contour_structure_poles(
                    p1_value, middle_momentum, sector_middle
                )
                for middle_pole in middle_poles:
                    if any(
                        abs(middle_pole.wall - excluded) < 1.0e-12
                        for excluded in excluded_walls
                    ):
                        continue
                    p2 = middle_pole.momentum
                    middle_laurent = self._structure_laurent_coefficients(
                        p1_value,
                        middle_momentum,
                        middle_pole,
                        sector_middle,
                    )

                    def middle_regular(p2_value: complex) -> complex:
                        return (
                            self._structure_constant(
                                pa, pb, p1_value, sector_left
                            )
                            * self._structure_constant(
                                p2_value, pd, pe, sector_right
                            )
                            * sector_kernel(
                                p1_value,
                                p2_value,
                                sectors,
                                0 if subtracted_continuum_edge == 0 else None,
                                0 if primary_continuum_edge == 0 else None,
                            )
                        )

                    middle_residue = apply_laurent_residue(
                        middle_laurent, middle_regular, p2
                    )
                    middle_contribution = (
                        weight1
                        * middle_pole.contour_coefficient
                        * middle_residue
                        / math.pi
                    )
                    right_residues += middle_contribution
                    middle_line_residues += middle_contribution

            for left_pole in left_poles:
                p1 = left_pole.momentum
                left_laurent = self._structure_laurent_coefficients(
                    pa, pb, left_pole, sector_left
                )

                def left_full_residue(
                    p2_value: complex,
                    *,
                    subtract_right_continuum: bool = False,
                    select_primary_right: bool = False,
                ) -> complex:
                    def left_regular(p1_value: complex) -> complex:
                        return (
                            self._structure_constant(
                                p1_value, pc, p2_value, sector_middle
                            )
                            * self._structure_constant(
                                p2_value, pd, pe, sector_right
                            )
                            * sector_kernel(
                                p1_value,
                                p2_value,
                                sectors,
                                1 if subtract_right_continuum else None,
                                1 if select_primary_right else None,
                            )
                        )

                    return apply_laurent_residue(
                        left_laurent, left_regular, p1
                    )

                for second_node in node_sets[1]:
                    p2, weight2 = second_node
                    left_residues += (
                        weight2
                        * left_pole.contour_coefficient
                        * left_full_residue(
                            complex(p2),
                            subtract_right_continuum=(
                                subtracted_continuum_edge == 1
                            ),
                            select_primary_right=(primary_continuum_edge == 1),
                        )
                        / math.pi
                    )

                nested_poles: list[CrossedNSStructurePole] = list(right_poles)
                nested_poles.extend(
                    _nested_quotient_structure_poles(
                        left_pole, middle_momentum, sector_middle
                    )
                )
                distinct_nested: list[CrossedNSStructurePole] = []
                for pole in nested_poles:
                    if any(
                        abs(pole.momentum - other_pole.momentum) < 1.0e-10
                        for other_pole in distinct_nested
                    ):
                        continue
                    distinct_nested.append(pole)
                for pole in distinct_nested:
                    nested_residues += (
                        left_pole.contour_coefficient
                        * pole.contour_coefficient
                        * self._numeric_contour_residue(
                            lambda value: left_full_residue(
                                value,
                                subtract_right_continuum=False,
                                select_primary_right=False,
                            ),
                            pole.momentum,
                        )
                    )

        return ContinuedMomentumDensity(
            continuous=continuous,
            left_residues=left_residues,
            right_residues=right_residues,
            nested_residues=nested_residues,
            middle_line_residues=middle_line_residues,
        )

    def continued_integrand_positions(
        self,
        positions: Sequence[ProjectivePoint],
        *,
        channel: LinearChannel | None = None,
        excluded_middle_walls: Sequence[float] = (),
    ) -> complex:
        """Return the residue-continued fixed-gauge moduli density."""

        return self.continued_integrand_components_positions(
            positions,
            channel=channel,
            excluded_middle_walls=excluded_middle_walls,
        ).total

    def continued_linear_q_components(
        self,
        q1: Number,
        q2: Number,
        ordering: Sequence[int],
        *,
        evaluation_orderings: Sequence[Sequence[int]] | None = None,
        excluded_middle_walls: Sequence[float] = (),
        subtracted_continuum_boundary_pair: Sequence[int] | None = None,
        primary_continuum_boundary_pair: Sequence[int] | None = None,
    ) -> ContinuedMomentumDensity:
        r"""Return the continued density with respect to ``d2q1 d2q2``.

        The channel's own ``(0,q1*q2,q2,1,infinity)`` frame remains stable in
        arbitrarily deep collars.  The factor ``|q2|^2`` is the Jacobian from
        the two finite punctures ``(q1*q2,q2)``.  The two unraised vertices
        supply the nonchiral superghost correlator ``|z_3-z_4|^-2`` (equal to
        one when either puncture is at infinity).  Together these convert the
        matter correlator to the invariant gauge-fixed moduli density.
        """

        first = _finite_complex("q1", q1)
        second = _finite_complex("q2", q2)
        sampled_ordering = tuple(int(label) for label in ordering)
        if not 0.0 < abs(first) < 1.0 or not 0.0 < abs(second) < 1.0:
            raise ValueError("linear plumbing coordinates must lie in the punctured bidisc")
        positions = linear_channel_positions_by_label(
            first, second, sampled_ordering
        )
        if evaluation_orderings is None:
            selected = sampled_ordering
            if selected.index(0) in (3, 4):
                selected = tuple(reversed(selected))
            if selected.index(0) > 2:
                raise ValueError(
                    "the continued channel must have incoming label 0 on the left "
                    "side or at the middle leaf"
                )
            active_channel = linear_channel_from_ordering(positions, selected)
        else:
            candidates: list[LinearChannel] = []
            for candidate_ordering in evaluation_orderings:
                selected = tuple(int(label) for label in candidate_ordering)
                if len(selected) != 5 or set(selected) != set(range(5)):
                    raise ValueError(
                        "every evaluation ordering must permute labels 0,...,4"
                    )
                try:
                    candidate = linear_channel_from_ordering(
                        positions, selected
                    )
                except (ValueError, ZeroDivisionError):
                    continue
                if candidate.score < 1.0:
                    candidates.append(candidate)
            if not candidates:
                raise ArithmeticError(
                    "the all-c-recursion split found no convergent certified channel"
                )
            # The Voronoi decision is purely geometric and therefore sees
            # the complete 120-chart all-c atlas.  Only *after* that decision
            # do we reverse a right-incoming comb for the endpoint-residue
            # convention.  A comb and its reversal have the same rho, so
            # this normalization changes neither the cell nor the block
            # convergence radius.
            geometric_channel = min(candidates, key=lambda item: item.score)
            selected = geometric_channel.ordering
            if selected.index(0) in (3, 4):
                selected = tuple(reversed(selected))
            active_channel = linear_channel_from_ordering(positions, selected)
        if active_channel.score >= 1.0:
            raise ArithmeticError("the sampled residue channel is invalid")
        if (
            subtracted_continuum_boundary_pair is not None
            and primary_continuum_boundary_pair is not None
        ):
            raise ValueError("select either a boundary primary or its remainder")

        def boundary_edge(pair_labels: Sequence[int] | None) -> int | None:
            if pair_labels is None:
                return None
            pair = tuple(
                sorted(int(label) for label in pair_labels)
            )
            if len(pair) != 2 or pair[0] == pair[1]:
                raise ValueError(
                    "a continuum boundary pair must contain two labels"
                )
            if set(active_channel.ordering[:2]) == set(pair):
                return 0
            if set(active_channel.ordering[3:]) == set(pair):
                return 1
            raise ValueError("the selected boundary pair is not an endpoint cherry")

        subtracted_edge = boundary_edge(subtracted_continuum_boundary_pair)
        primary_edge = boundary_edge(primary_continuum_boundary_pair)
        components = self.continued_integrand_components_positions(
            positions,
            channel=active_channel,
            excluded_middle_walls=excluded_middle_walls,
            subtracted_continuum_edge=subtracted_edge,
            primary_continuum_edge=primary_edge,
        )
        superghost = _superghost_pair_factor(positions)
        factor = float(superghost * abs(second) ** 2)
        return ContinuedMomentumDensity(
            continuous=components.continuous * factor,
            left_residues=components.left_residues * factor,
            right_residues=components.right_residues * factor,
            nested_residues=components.nested_residues * factor,
            middle_line_residues=components.middle_line_residues * factor,
        )

    def continued_linear_q_middle_line_terms(
        self,
        q1: Number,
        q2: Number,
        ordering: Sequence[int],
    ) -> tuple[MovingMiddleResidueTerm, ...]:
        """Return moving-middle terms with respect to ``d2q1 d2q2``."""

        first = _finite_complex("q1", q1)
        second = _finite_complex("q2", q2)
        selected = tuple(int(label) for label in ordering)
        if not 0.0 < abs(first) < 1.0 or not 0.0 < abs(second) < 1.0:
            raise ValueError("linear plumbing coordinates must lie in the punctured bidisc")
        positions = linear_channel_positions_by_label(first, second, selected)
        active_channel = linear_channel_from_ordering(positions, selected)
        raw_terms = self.continued_middle_line_terms_positions(
            positions, channel=active_channel
        )
        factor = _superghost_pair_factor(positions) * abs(second) ** 2
        return tuple(
            MovingMiddleResidueTerm(
                first_momentum=term.first_momentum,
                second_pole=term.second_pole,
                sectors=term.sectors,
                value=term.value * factor,
            )
            for term in raw_terms
        )

    def continued_middle_line_face_terms(
        self,
        q1: Number,
        ordering: Sequence[int],
        *,
        projection_radius: float = 1.0e-5,
        wall: float = 1.0,
    ) -> tuple[MovingMiddleFaceTerm, ...]:
        """Project the leading normal coefficient as a function of ``q1``."""

        first = _finite_complex("q1", q1)
        selected = tuple(int(label) for label in ordering)
        radius = float(projection_radius)
        selected_wall = float(wall)
        if not 0.0 < abs(first) < 1.0:
            raise ValueError("the remaining face modulus must satisfy 0<|q1|<1")
        if not 1.0e-8 <= radius < 1.0e-3:
            raise ValueError("projection_radius must lie in [1e-8,1e-3)")
        key = (first, selected, radius, selected_wall)
        cached = self._middle_face_projection_cache.get(key)
        if cached is not None:
            return cached
        terms = self.continued_linear_q_middle_line_terms(
            first, radius, selected
        )
        projected: list[MovingMiddleFaceTerm] = []
        for term in terms:
            if abs(term.second_pole.wall - selected_wall) > 1.0e-12:
                continue
            right_beta = self._continued_boundary_radial_beta(
                selected,
                term.second_pole.momentum,
                side="right",
                edge_sector=term.sectors[2],
            )
            leading_power = mpmath.exp(
                mpmath.mpc(right_beta) * math.log(radius)
            )
            projected.append(
                MovingMiddleFaceTerm(
                    first_momentum=term.first_momentum,
                    second_pole=term.second_pole,
                    sectors=term.sectors,
                    right_beta=right_beta,
                    coefficient=term.value / leading_power,
                )
            )
        result = tuple(projected)
        self._middle_face_projection_cache[key] = result
        return result

    def continued_middle_line_face_finite_part_density(
        self,
        q1: Number,
        ordering: Sequence[int],
        *,
        collar_radius: float,
        projection_radius: float = 1.0e-5,
        wall: float = 1.0,
    ) -> complex:
        """Return the normal finite part multiplying ``d2q1``."""

        radius = float(collar_radius)
        if not 0.0 < radius < 0.2:
            raise ValueError("collar_radius must lie in (0,0.2)")
        total = mpmath.mpc(0)
        for term in self.continued_middle_line_face_terms(
            q1,
            ordering,
            projection_radius=projection_radius,
            wall=wall,
        ):
            total += mpmath.mpc(term.coefficient) * _complex_radial_finite_part(
                term.right_beta, radius
            )
        return complex(total)

    def continued_middle_line_corner_terms(
        self,
        ordering: Sequence[int],
        *,
        projection_radius: float = 1.0e-5,
        wall: float = 1.0,
    ) -> tuple[MovingMiddleCornerTerm, ...]:
        r"""Project the leading double-corner coefficient term by term.

        The endpoint trinion sector adds one unit to the radial exponent on
        an odd edge.  This is the same parity shift carried by the lowest
        nonzero coefficient of the all-``c`` block.  Only the selected wall
        is returned, so integrable higher moving lines remain in the raw
        numerical remainder.
        """

        selected = tuple(int(label) for label in ordering)
        radius = float(projection_radius)
        selected_wall = float(wall)
        if not 1.0e-8 <= radius < 1.0e-3:
            raise ValueError("projection_radius must lie in [1e-8,1e-3)")
        key = (selected, radius, selected_wall)
        cached = self._middle_corner_projection_cache.get(key)
        if cached is not None:
            return cached
        terms = self.continued_linear_q_middle_line_terms(
            radius, radius, selected
        )
        projected: list[MovingMiddleCornerTerm] = []
        for term in terms:
            if abs(term.second_pole.wall - selected_wall) > 1.0e-12:
                continue
            left_beta = self._continued_boundary_radial_beta(
                selected,
                term.first_momentum,
                side="left",
                edge_sector=term.sectors[0],
            )
            right_beta = self._continued_boundary_radial_beta(
                selected,
                term.second_pole.momentum,
                side="right",
                edge_sector=term.sectors[2],
            )
            leading_power = mpmath.exp(
                mpmath.mpc(left_beta + right_beta) * math.log(radius)
            )
            projected.append(
                MovingMiddleCornerTerm(
                    first_momentum=term.first_momentum,
                    second_pole=term.second_pole,
                    sectors=term.sectors,
                    left_beta=left_beta,
                    right_beta=right_beta,
                    coefficient=term.value / leading_power,
                )
            )
        result = tuple(projected)
        self._middle_corner_projection_cache[key] = result
        return result

    def continued_middle_line_corner_counterterm(
        self,
        q1: Number,
        q2: Number,
        ordering: Sequence[int],
        *,
        projection_radius: float = 1.0e-5,
        wall: float = 1.0,
    ) -> complex:
        """Evaluate the projected spin-zero corner counterterm."""

        first = _finite_complex("q1", q1)
        second = _finite_complex("q2", q2)
        if first == 0.0 or second == 0.0:
            raise ValueError("corner counterterm requires nonzero plumbing coordinates")
        log_first = math.log(abs(first))
        log_second = math.log(abs(second))
        total = mpmath.mpc(0)
        for term in self.continued_middle_line_corner_terms(
            ordering,
            projection_radius=projection_radius,
            wall=wall,
        ):
            total += mpmath.mpc(term.coefficient) * mpmath.exp(
                mpmath.mpc(term.left_beta) * log_first
                + mpmath.mpc(term.right_beta) * log_second
            )
        return complex(total)

    def continued_middle_line_corner_finite_part(
        self,
        ordering: Sequence[int],
        *,
        collar_radius: float,
        projection_radius: float = 1.0e-5,
        wall: float = 1.0,
    ) -> complex:
        """Apply the two analytic radial finite parts to the corner line."""

        radius = float(collar_radius)
        if not 0.0 < radius < 0.2:
            raise ValueError("collar_radius must lie in (0,0.2)")
        total = mpmath.mpc(0)
        for term in self.continued_middle_line_corner_terms(
            ordering,
            projection_radius=projection_radius,
            wall=wall,
        ):
            total += (
                mpmath.mpc(term.coefficient)
                * _complex_radial_finite_part(term.left_beta, radius)
                * _complex_radial_finite_part(term.right_beta, radius)
            )
        return complex(total)

    def integrand(self, z: Number, w: Number) -> complex:
        """Return the fixed-gauge density multiplying ``d2z d2w``."""

        return self.fixed_gauge_integrand_positions(
            self.fixed_gauge_positions(z, w)
        )


def _power_disk_sample(
    radial_uniform: float,
    angular_uniform: float,
    radial_power: float,
) -> complex:
    if not 0.0 < radial_power <= 2.0:
        raise ValueError("radial_power must lie in (0,2]")
    lower = np.nextafter(0.0, 1.0)
    upper = np.nextafter(1.0, 0.0)
    uniform = min(max(float(radial_uniform), lower), upper)
    radius = uniform ** (1.0 / radial_power)
    angle = 2.0 * math.pi * float(angular_uniform)
    return complex(radius * cmath.exp(1.0j * angle))


def _oriented_bidisc_mixture_density(
    positions: Sequence[ProjectivePoint], *, radial_power: float
) -> float:
    """Return the 120-chart proposal density in the supplied fixed gauge."""

    total = 0.0
    orderings = oriented_tree_orderings()
    for ordering in orderings:
        try:
            channel = linear_channel_from_ordering(positions, ordering)
        except (ValueError, ZeroDivisionError):
            continue
        radius1 = abs(channel.q1)
        radius2 = abs(channel.q2)
        if not (0.0 < radius1 < 1.0 and 0.0 < radius2 < 1.0):
            continue
        jacobian = linear_channel_complex_jacobian_to_chart(
            channel.q1,
            channel.q2,
            ordering,
            fixed_zero=FIXED_ZERO_LABEL,
            fixed_one=FIXED_ONE_LABEL,
            fixed_infinity=FIXED_INFINITY_LABEL,
            moving_labels=MOVING_LABELS,
        )
        area_jacobian = abs(jacobian) ** 2
        if area_jacobian <= 0.0 or not math.isfinite(area_jacobian):
            continue
        density1 = radial_power / (2.0 * math.pi) * radius1 ** (
            radial_power - 2.0
        )
        density2 = radial_power / (2.0 * math.pi) * radius2 ** (
            radial_power - 2.0
        )
        total += density1 * density2 / area_jacobian
    density = total / len(orderings)
    if density <= 0.0 or not math.isfinite(density):
        raise ArithmeticError("the five-point atlas mixture density is non-positive")
    return float(density)


def _projective_channel_coordinates(
    positions: Sequence[ProjectivePoint], ordering: Sequence[int]
) -> tuple[complex, complex]:
    """Return ``(q1,q2)`` from homogeneous cross ratios without tolerances."""

    selected = tuple(int(label) for label in ordering)
    if len(positions) != 5 or len(selected) != 5 or set(selected) != set(range(5)):
        raise ValueError("positions and ordering must describe five labelled punctures")

    def homogeneous(point: ProjectivePoint) -> tuple[complex, complex]:
        return (1.0 + 0.0j, 0.0 + 0.0j) if point is None else (complex(point), 1.0 + 0.0j)

    def determinant(
        left: tuple[complex, complex], right: tuple[complex, complex]
    ) -> complex:
        return left[0] * right[1] - left[1] * right[0]

    points = tuple(homogeneous(point) for point in positions)
    a, b, c, d, e = (points[label] for label in selected)
    q1_denominator = determinant(b, e) * determinant(c, a)
    q2_denominator = determinant(c, e) * determinant(d, a)
    if q1_denominator == 0.0 or q2_denominator == 0.0:
        raise ZeroDivisionError("the requested projective channel is degenerate")
    q1 = determinant(b, a) * determinant(c, e) / q1_denominator
    q2 = determinant(c, a) * determinant(d, e) / q2_denominator
    return complex(q1), complex(q2)


def _oriented_bidisc_mixture_density_in_channel(
    q1: complex,
    q2: complex,
    sampled_ordering: Sequence[int],
    *,
    radial_power: float,
    orderings: Sequence[Sequence[int]] | None = None,
    pair_radial_powers: Mapping[tuple[int, int], float] | None = None,
    return_log_density: bool = False,
) -> float:
    """Return the 120-chart proposal density with respect to sampled ``d4q``.

    This form never maps a deep plumbing collar to a fixed gauge, where two
    distinct punctures can become equal at double precision.  Transition
    Jacobians are ratios of the exact analytic Jacobians from each channel
    to the same fixed-label chart.
    """

    selected = tuple(int(label) for label in sampled_ordering)
    positions = linear_channel_positions_by_label(q1, q2, selected)
    sampled_jacobian = linear_channel_complex_jacobian_to_chart(
        q1,
        q2,
        selected,
        fixed_zero=ATLAS_REFERENCE_FIXED_ZERO_LABEL,
        fixed_one=ATLAS_REFERENCE_FIXED_ONE_LABEL,
        fixed_infinity=ATLAS_REFERENCE_FIXED_INFINITY_LABEL,
        moving_labels=ATLAS_REFERENCE_MOVING_LABELS,
    )
    if sampled_jacobian == 0.0:
        raise ArithmeticError("the sampled channel has a zero chart Jacobian")

    def powers(ordering: Sequence[int]) -> tuple[float, float]:
        if pair_radial_powers is None:
            return radial_power, radial_power
        left_pair = tuple(sorted((int(ordering[0]), int(ordering[1]))))
        right_pair = tuple(sorted((int(ordering[3]), int(ordering[4]))))
        try:
            return (
                float(pair_radial_powers[left_pair]),
                float(pair_radial_powers[right_pair]),
            )
        except KeyError as error:
            raise ValueError(
                f"missing radial power for boundary pair {error.args[0]}"
            ) from error

    log_terms: list[float] = []
    atlas_orderings = (
        oriented_tree_orderings()
        if orderings is None
        else tuple(tuple(int(label) for label in ordering) for ordering in orderings)
    )
    if not atlas_orderings:
        raise ValueError("the atlas ordering collection must be nonempty")
    for ordering in atlas_orderings:
        try:
            other_q1, other_q2 = _projective_channel_coordinates(positions, ordering)
        except (ValueError, ZeroDivisionError):
            continue
        radius1 = abs(other_q1)
        radius2 = abs(other_q2)
        if not (0.0 < radius1 < 1.0 and 0.0 < radius2 < 1.0):
            continue
        other_jacobian = linear_channel_complex_jacobian_to_chart(
            other_q1,
            other_q2,
            ordering,
            fixed_zero=ATLAS_REFERENCE_FIXED_ZERO_LABEL,
            fixed_one=ATLAS_REFERENCE_FIXED_ONE_LABEL,
            fixed_infinity=ATLAS_REFERENCE_FIXED_INFINITY_LABEL,
            moving_labels=ATLAS_REFERENCE_MOVING_LABELS,
        )
        if other_jacobian == 0.0:
            continue
        power1, power2 = powers(ordering)
        log_density1 = (
            math.log(power1 / (2.0 * math.pi))
            + (power1 - 2.0) * math.log(radius1)
        )
        log_density2 = (
            math.log(power2 / (2.0 * math.pi))
            + (power2 - 2.0) * math.log(radius2)
        )
        if tuple(ordering) == selected:
            # The identity chart transition is exactly one.  Computing it as
            # a quotient of two separately extreme Jacobians loses this fact
            # in deep collars through inf/inf or subnormal rounding.
            log_transition_squared = 0.0
        else:
            sampled_absolute = abs(sampled_jacobian)
            other_absolute = abs(other_jacobian)
            if (
                not math.isfinite(sampled_absolute)
                or not math.isfinite(other_absolute)
                or sampled_absolute == 0.0
                or other_absolute == 0.0
            ):
                continue
            log_transition_squared = 2.0 * (
                math.log(sampled_absolute) - math.log(other_absolute)
            )
        log_terms.append(
            log_density1 + log_density2 + log_transition_squared
        )
    if not log_terms:
        raise ArithmeticError("the channel-space atlas mixture density is empty")
    maximum_log = max(log_terms)
    log_density = (
        maximum_log
        + math.log(sum(math.exp(value - maximum_log) for value in log_terms))
        - math.log(len(atlas_orderings))
    )
    if not math.isfinite(log_density):
        raise ArithmeticError("the logarithmic atlas mixture density is non-finite")
    if return_log_density:
        return float(log_density)
    density = math.exp(log_density)
    if density <= 0.0 or not math.isfinite(density):
        raise ArithmeticError("the channel-space atlas mixture density is non-positive")
    return float(density)


def _channel_density_transition(
    q1: complex,
    q2: complex,
    sampled_ordering: Sequence[int],
    target_ordering: Sequence[int],
) -> tuple[complex, complex, float]:
    """Return target coordinates and ``d4q_target/d4q_sample``."""

    sampled = tuple(int(label) for label in sampled_ordering)
    target = tuple(int(label) for label in target_ordering)
    positions = linear_channel_positions_by_label(q1, q2, sampled)
    target_q1, target_q2 = _projective_channel_coordinates(positions, target)
    sampled_jacobian = linear_channel_complex_jacobian_to_chart(
        q1,
        q2,
        sampled,
        fixed_zero=ATLAS_REFERENCE_FIXED_ZERO_LABEL,
        fixed_one=ATLAS_REFERENCE_FIXED_ONE_LABEL,
        fixed_infinity=ATLAS_REFERENCE_FIXED_INFINITY_LABEL,
        moving_labels=ATLAS_REFERENCE_MOVING_LABELS,
    )
    target_jacobian = linear_channel_complex_jacobian_to_chart(
        target_q1,
        target_q2,
        target,
        fixed_zero=ATLAS_REFERENCE_FIXED_ZERO_LABEL,
        fixed_one=ATLAS_REFERENCE_FIXED_ONE_LABEL,
        fixed_infinity=ATLAS_REFERENCE_FIXED_INFINITY_LABEL,
        moving_labels=ATLAS_REFERENCE_MOVING_LABELS,
    )
    sampled_absolute = abs(sampled_jacobian)
    target_absolute = abs(target_jacobian)
    if (
        sampled_absolute == 0.0
        or target_absolute == 0.0
        or not math.isfinite(sampled_absolute)
        or not math.isfinite(target_absolute)
    ):
        raise ArithmeticError("the channel density transition is singular")
    log_factor = 2.0 * (
        math.log(sampled_absolute) - math.log(target_absolute)
    )
    if log_factor > math.log(np.finfo(float).max):
        raise OverflowError("the channel density transition overflowed")
    factor = math.exp(log_factor)
    return target_q1, target_q2, factor


def _to_fixed_gauge(
    q1: complex, q2: complex, ordering: Sequence[int]
) -> tuple[ProjectivePoint, ...]:
    """Map a sampled channel to the physical fixed gauge without tolerances.

    ``MobiusMap.__call__`` intentionally identifies very small denominators
    with infinity.  That is convenient for ordinary chart work but aliases
    distinct punctures in the deep collars required here.  Homogeneous
    determinants distinguish a true infinity (zero denominator) from an
    arbitrarily large finite coordinate.
    """

    sampled_positions = linear_channel_positions_by_label(q1, q2, ordering)

    def homogeneous(point: ProjectivePoint) -> tuple[complex, complex]:
        return (1.0 + 0.0j, 0.0 + 0.0j) if point is None else (complex(point), 1.0 + 0.0j)

    def determinant(
        left: tuple[complex, complex], right: tuple[complex, complex]
    ) -> complex:
        return left[0] * right[1] - left[1] * right[0]

    points = tuple(homogeneous(point) for point in sampled_positions)
    zero_point = points[FIXED_ZERO_LABEL]
    one_point = points[FIXED_ONE_LABEL]
    infinity_point = points[FIXED_INFINITY_LABEL]
    denominator = determinant(one_point, zero_point)
    if denominator == 0.0:
        raise ValueError(
            "the sampled chart collapsed fixed labels "
            f"{FIXED_ONE_LABEL} and {FIXED_ZERO_LABEL}"
        )
    normalization = determinant(one_point, infinity_point) / denominator
    transformed: list[ProjectivePoint] = []
    for point in points:
        numerator = normalization * determinant(point, zero_point)
        denominator = determinant(point, infinity_point)
        transformed.append(None if denominator == 0.0 else complex(numerator / denominator))
    return _validate_positions(tuple(transformed))


def _pco_face_sector_orderings() -> tuple[tuple[int, ...], ...]:
    """Return the six crossing cells on each of the three PCO faces."""

    result: list[tuple[int, ...]] = []
    labels = set(range(5))
    for pair in combinations(PICTURE_ZERO_LABELS, 2):
        remaining = tuple(sorted(labels - set(pair)))
        for tail in permutations(remaining):
            result.append((*pair, *tail))
    return tuple(result)


PCO_FACE_SECTOR_ORDERINGS = _pco_face_sector_orderings()


def _boundary_face_sector_orderings() -> tuple[tuple[int, ...], ...]:
    """Return six four-point crossing cells on each of all ten faces."""

    labels = set(range(5))
    result: list[tuple[int, ...]] = []
    for pair in combinations(range(5), 2):
        remaining = tuple(sorted(labels - set(pair)))
        for tail in permutations(remaining):
            result.append((*pair, *tail))
    return tuple(result)


BOUNDARY_FACE_SECTOR_ORDERINGS = _boundary_face_sector_orderings()


def _boundary_face_raised_orbits() -> tuple[tuple[tuple[int, ...], int], ...]:
    """Compress equal-energy faces under the picture-preserving S2 x S2."""

    groups: dict[tuple[int, ...], list[tuple[int, ...]]] = {}
    for ordering in BOUNDARY_FACE_SECTOR_ORDERINGS:
        signature = tuple(
            5 if label in (1, 2) else 6 if label in (3, 4) else label
            for label in ordering
        )
        groups.setdefault(signature, []).append(ordering)
    return tuple((members[0], len(members)) for members in groups.values())


BOUNDARY_FACE_RAISED_ORBITS = _boundary_face_raised_orbits()


def _boundary_corner_orderings() -> tuple[tuple[int, ...], ...]:
    """Return one oriented chart for each of the fifteen compatible corners."""

    labels = set(range(5))
    divisors = tuple(combinations(range(5), 2))
    result: list[tuple[int, ...]] = []
    for index, left in enumerate(divisors):
        for right in divisors[index + 1 :]:
            if set(left) & set(right):
                continue
            middle = tuple(labels - set(left) - set(right))
            if len(middle) != 1:
                raise AssertionError("compatible five-point faces leave one label")
            result.append((*left, middle[0], *right))
    if len(result) != 15:
        raise AssertionError("Mbar_0,5 must have fifteen boundary corners")
    return tuple(result)


BOUNDARY_CORNER_ORDERINGS = _boundary_corner_orderings()


def _incoming_outgoing_face_orderings() -> tuple[tuple[int, ...], ...]:
    """Return six lower crossing cells on each incoming--outgoing face."""

    result: list[tuple[int, ...]] = []
    for adjacent in (1, 2, 3, 4):
        remaining = tuple(label for label in range(1, 5) if label != adjacent)
        for tail in permutations(remaining):
            result.append((0, adjacent, *tail))
    return tuple(result)


INCOMING_OUTGOING_FACE_ORDERINGS = _incoming_outgoing_face_orderings()


def _incoming_outgoing_corner_orderings() -> tuple[tuple[int, ...], ...]:
    """Return the twelve compatible IO/OO corners in incoming-first frames."""

    result: list[tuple[int, ...]] = []
    outgoing = set(range(1, 5))
    for adjacent in sorted(outgoing):
        remaining = outgoing - {adjacent}
        for right_pair in combinations(sorted(remaining), 2):
            middle = tuple(remaining - set(right_pair))
            result.append((0, adjacent, middle[0], *right_pair))
    if len(result) != 12:
        raise AssertionError("equal-energy 1->4 must have twelve IO/OO corners")
    return tuple(result)


INCOMING_OUTGOING_CORNER_ORDERINGS = _incoming_outgoing_corner_orderings()


def _four_point_fundamental_cell(value: complex) -> bool:
    r"""Return the six-sector cell ``|z-1|<1, 0<Re(z)<1/2``."""

    modulus = complex(value)
    return abs(modulus - 1.0) < 1.0 and 0.0 < modulus.real < 0.5


def _four_point_fundamental_cell_sample(
    horizontal: float, vertical: float
) -> tuple[complex, float]:
    """Map a unit square exactly onto the bounded six-sector cell."""

    lower = np.nextafter(0.0, 1.0)
    upper = np.nextafter(1.0, 0.0)
    u_value = min(max(float(horizontal), lower), upper)
    v_value = min(max(float(vertical), lower), upper)
    real_part = 0.5 * u_value
    height = math.sqrt(max(0.0, 1.0 - (real_part - 1.0) ** 2))
    imaginary_part = (2.0 * v_value - 1.0) * height
    modulus = complex(real_part, imaginary_part)
    if not _four_point_fundamental_cell(modulus):
        raise ArithmeticError("the direct four-point cell map left its domain")
    # dx/du=1/2 and dy/dv=2*height.
    return modulus, height


ONE_DIVISOR_FACE_ORDERINGS = tuple(
    ordering
    for ordering in PCO_FACE_SECTOR_ORDERINGS
    if tuple(ordering[:2]) == (1, 2)
)


def _one_divisor_face_channel_in_sampled_chart(
    q1: complex,
    q2: complex,
    sampled_ordering: Sequence[int],
    collar_radius: float,
) -> tuple[tuple[int, ...], complex, complex, float] | None:
    """Locate the unique ``D_12`` collar cell from a sampled all-c chart."""

    matches: list[tuple[tuple[int, ...], complex, complex, float]] = []
    for ordering in ONE_DIVISOR_FACE_ORDERINGS:
        try:
            target_q1, target_q2, transition = _channel_density_transition(
                q1, q2, sampled_ordering, ordering
            )
        except (ArithmeticError, OverflowError, ZeroDivisionError):
            continue
        if (
            0.0 < abs(target_q1) < collar_radius
            and _four_point_fundamental_cell(target_q2)
        ):
            matches.append((ordering, target_q1, target_q2, transition))
    if len(matches) > 1:
        raise ArithmeticError(
            "the D_12 four-point collar cells overlap; decrease the collar"
        )
    return matches[0] if matches else None


def _plane_map(
    radial_coordinate: float, angular_coordinate: float
) -> tuple[complex, float]:
    """Map a unit square to the complex plane with its area Jacobian."""

    radial = float(radial_coordinate)
    angular = float(angular_coordinate)
    radius = math.tan(0.5 * math.pi * radial)
    angle = 2.0 * math.pi * angular
    jacobian = math.pi * math.pi * radius * (1.0 + radius * radius)
    return complex(radius * cmath.exp(1.0j * angle)), jacobian


def _radial_finite_part(beta: float, collar_radius: float) -> float:
    r"""Continue ``int d2q |q|^beta`` over a circular collar."""

    exponent = float(beta)
    radius = float(collar_radius)
    if not math.isfinite(exponent):
        raise ValueError("beta must be finite")
    if not math.isfinite(radius) or not 0.0 < radius < 1.0:
        raise ValueError("collar_radius must lie in (0,1)")
    alpha = 0.5 * (exponent + 2.0)
    if abs(alpha) < 1.0e-13:
        return 2.0 * math.pi * math.log(radius)
    return math.pi * math.exp(2.0 * alpha * math.log(radius)) / alpha


def _complex_radial_finite_part(beta: Number, collar_radius: float):
    r"""Continue ``int d2q |q|^beta`` for a complex radial exponent."""

    exponent = mpmath.mpc(beta)
    radius = float(collar_radius)
    if not mpmath.isfinite(exponent):
        raise ValueError("beta must be finite")
    if not math.isfinite(radius) or not 0.0 < radius < 1.0:
        raise ValueError("collar_radius must lie in (0,1)")
    shifted = exponent + 2
    if abs(shifted) < mpmath.mpf("1e-20"):
        return 2 * mpmath.pi * math.log(radius)
    return (
        2
        * mpmath.pi
        * mpmath.exp(shifted * math.log(radius))
        / shifted
    )


def _pco_face_chart(
    positions: Sequence[ProjectivePoint], collar_radius: float
) -> tuple[tuple[int, ...], LinearChannel] | None:
    """Locate a point in one of the disjoint raised-pair face collars."""

    normalized = _validate_positions(positions)
    radius = float(collar_radius)
    if not math.isfinite(radius) or not 0.0 < radius < 0.2:
        raise ValueError("collar_radius must lie in (0,0.2)")
    matches: list[tuple[tuple[int, ...], LinearChannel]] = []
    for ordering in PCO_FACE_SECTOR_ORDERINGS:
        channel = linear_channel_from_ordering(normalized, ordering)
        if (
            abs(channel.q1) < radius
            and _four_point_fundamental_cell(channel.q2)
        ):
            matches.append((ordering, channel))
    if len(matches) > 1:
        raise ArithmeticError(
            "PCO face collars overlap; decrease collar_radius before integration"
        )
    return matches[0] if matches else None


def _finite_part_remainder_integrand(
    kernel: BRYNSFiveTachyonIntegrand,
    positions: Sequence[ProjectivePoint],
    collar_radius: float,
) -> complex:
    """Return ``I-S_face I`` in a PCO collar and ``I`` elsewhere."""

    face = _pco_face_chart(positions, collar_radius)
    if face is None:
        return kernel.fixed_gauge_integrand_positions(positions)
    ordering, channel = face
    raw = kernel.fixed_gauge_integrand_positions(positions, channel=channel)
    counterterm_q = kernel.two_pco_face_counterterm_q_density(
        ordering=ordering,
        normal_coordinate=channel.q1,
        remaining_modulus=channel.q2,
    )
    jacobian = linear_channel_complex_jacobian_to_chart(
        channel.q1,
        channel.q2,
        ordering,
        fixed_zero=FIXED_ZERO_LABEL,
        fixed_one=FIXED_ONE_LABEL,
        fixed_infinity=FIXED_INFINITY_LABEL,
        moving_labels=MOVING_LABELS,
    )
    area_jacobian = abs(jacobian) ** 2
    if not math.isfinite(area_jacobian) or area_jacobian <= 0.0:
        raise ArithmeticError("PCO face chart has a non-positive Jacobian")
    return complex(
        raw
        - _superghost_pair_factor(positions) * counterterm_q / area_jacobian
    )


def integrate_imaginary_energy_atlas_qmc(
    kernel: BRYNSFiveTachyonIntegrand,
    *,
    sobol_power: int = 3,
    replicates: int = 2,
    radial_power: float = 0.15,
    seed: int = 20260825,
) -> NSFivePointQMCResult:
    r"""Integrate an absolutely convergent raw density over ``M_0,5``.

    It performs no boundary subtraction or Liouville-contour residue.  The
    chamber audit is therefore enforced rather than assumed.  For the present
    BRY collocated-PCO prescription there is no positive equal-imaginary
    energy satisfying both requirements; a faithful production calculation
    must add either the PCO finite-part layers or the crossed-pole residues.
    """

    sobol_power = int(sobol_power)
    replicates = int(replicates)
    radial_power = float(radial_power)
    if sobol_power < 1 or replicates < 2:
        raise ValueError("sobol_power must be positive and replicates at least two")
    if not 0.0 < radial_power <= 2.0:
        raise ValueError("radial_power must lie in (0,2]")
    audit = imaginary_energy_chamber_audit(kernel.outgoing_energies)
    if not audit["raw_moduli_convergent_without_pco_subtraction"]:
        raise ValueError(
            "the raw five-point moduli integral has a two-PCO collision "
            "power divergence; BRY finite-part/vertical-integration layers "
            "are required"
        )
    if not audit["undeformed_positive_real_liouville_contour_valid"]:
        raise ValueError(
            "super-Liouville structure-constant poles have crossed the "
            "positive-real internal-momentum contour; discrete residues are "
            "required"
        )
    orderings = oriented_tree_orderings()
    estimates: list[complex] = []
    for replicate in range(replicates):
        sampler = qmc.Sobol(d=5, scramble=True, seed=int(seed) + replicate)
        samples = sampler.random_base2(sobol_power)
        values: list[complex] = []
        for sample in samples:
            q1 = _power_disk_sample(sample[0], sample[1], radial_power)
            q2 = _power_disk_sample(sample[2], sample[3], radial_power)
            ordering_index = min(
                int(sample[4] * len(orderings)), len(orderings) - 1
            )
            positions = _to_fixed_gauge(q1, q2, orderings[ordering_index])
            density = _oriented_bidisc_mixture_density(
                positions, radial_power=radial_power
            )
            values.append(
                kernel.fixed_gauge_integrand_positions(positions) / density
            )
        estimates.append(complex(np.mean(np.asarray(values, dtype=complex))))
    array = np.asarray(estimates, dtype=complex)
    mean = complex(np.mean(array))
    return NSFivePointQMCResult(
        outgoing_energies=kernel.outgoing_energies,  # type: ignore[arg-type]
        estimates=tuple(estimates),
        mean=mean,
        standard_error_real=float(np.std(array.real, ddof=1) / math.sqrt(replicates)),
        standard_error_imag=float(np.std(array.imag, ddof=1) / math.sqrt(replicates)),
        samples_per_replicate=2**sobol_power,
        replicates=replicates,
        radial_power=radial_power,
        seed=int(seed),
        recursion_max_twice_level=kernel.recursion_max_twice_level,
        global_max_twice_levels=kernel.global_max_twice_levels,
        momentum_orders=kernel.momentum_orders,
        momentum_maximum=kernel.momentum_maximum,
    )


def integrate_equal_complex_energy_continued_atlas_qmc(
    kernel: BRYNSFiveTachyonIntegrand,
    *,
    sobol_power: int = 3,
    replicates: int = 2,
    radial_power: float = 0.2,
    seed: int = 20260825,
) -> NSFivePointContinuedQMCResult:
    r"""Integrate the three-fixed-PCO complex chamber ``3/5<t<2/3``."""

    sobol_power = int(sobol_power)
    replicates = int(replicates)
    radial_power = float(radial_power)
    if sobol_power < 1 or replicates < 2:
        raise ValueError("sobol_power must be positive and replicates at least two")
    if not 0.0 < radial_power <= 2.0:
        raise ValueError("radial_power must lie in (0,2]")
    first = kernel.outgoing_energies[0]
    if (
        not 0.6 < first.imag < 2.0 / 3.0
        or max(abs(value - first) for value in kernel.outgoing_energies) > 1.0e-13
    ):
        raise ValueError(
            "the continued atlas driver requires equal omega_i=x+i*t with 3/5<t<2/3"
        )
    audit = equal_complex_energy_convergence_audit(first)
    if not audit["all_moduli_boundaries_absolutely_convergent"]:
        raise ValueError(
            "the complex kinematics does not pass the conservative residue-forest "
            f"boundary audit: minimum margin={audit['minimum_integrability_margin']}"
        )

    orderings = incoming_endpoint_tree_orderings()
    continuous_estimates: list[complex] = []
    left_estimates: list[complex] = []
    right_estimates: list[complex] = []
    nested_estimates: list[complex] = []
    estimates: list[complex] = []
    for replicate in range(replicates):
        sampler = qmc.Sobol(d=5, scramble=True, seed=int(seed) + replicate)
        component_values: list[ContinuedMomentumDensity] = []
        for sample in sampler.random_base2(sobol_power):
            q1 = _power_disk_sample(sample[0], sample[1], radial_power)
            q2 = _power_disk_sample(sample[2], sample[3], radial_power)
            ordering_index = min(
                int(sample[4] * len(orderings)), len(orderings) - 1
            )
            ordering = orderings[ordering_index]
            proposal_density = _oriented_bidisc_mixture_density_in_channel(
                q1,
                q2,
                ordering,
                radial_power=radial_power,
                orderings=orderings,
            )
            value = kernel.continued_linear_q_components(
                q1,
                q2,
                ordering,
            )
            component_values.append(
                ContinuedMomentumDensity(
                    continuous=value.continuous / proposal_density,
                    left_residues=value.left_residues / proposal_density,
                    right_residues=value.right_residues / proposal_density,
                    nested_residues=value.nested_residues / proposal_density,
                )
            )
        def stable_mean(values):
            selected = tuple(mpmath.mpc(value) for value in values)
            return mpmath.mpc(
                mpmath.fsum(value.real for value in selected) / len(selected),
                mpmath.fsum(value.imag for value in selected) / len(selected),
            )

        continuous = stable_mean(
            value.continuous for value in component_values
        )
        left = stable_mean(
            value.left_residues for value in component_values
        )
        right = stable_mean(
            value.right_residues for value in component_values
        )
        nested = stable_mean(
            value.nested_residues for value in component_values
        )
        continuous_estimates.append(continuous)
        left_estimates.append(left)
        right_estimates.append(right)
        nested_estimates.append(nested)
        def component_total(value):
            selected = tuple(
                mpmath.mpc(item)
                for item in (
                    value.continuous,
                    value.left_residues,
                    value.right_residues,
                    value.nested_residues,
                )
            )
            return mpmath.mpc(
                mpmath.fsum(item.real for item in selected),
                mpmath.fsum(item.imag for item in selected),
            )

        total = stable_mean(component_total(value) for value in component_values)
        estimates.append(complex(total))

    array = np.asarray(estimates, dtype=complex)
    overall_mean = complex(
        math.fsum(value.real for value in estimates) / len(estimates),
        math.fsum(value.imag for value in estimates) / len(estimates),
    )
    return NSFivePointContinuedQMCResult(
        outgoing_energies=kernel.outgoing_energies,  # type: ignore[arg-type]
        estimates=tuple(estimates),
        continuous_estimates=tuple(continuous_estimates),
        left_residue_estimates=tuple(left_estimates),
        right_residue_estimates=tuple(right_estimates),
        nested_residue_estimates=tuple(nested_estimates),
        mean=overall_mean,
        standard_error_real=float(
            np.std(array.real, ddof=1) / math.sqrt(replicates)
        ),
        standard_error_imag=float(
            np.std(array.imag, ddof=1) / math.sqrt(replicates)
        ),
        samples_per_replicate=2**sobol_power,
        replicates=replicates,
        radial_power=radial_power,
        seed=int(seed),
        recursion_max_twice_level=kernel.recursion_max_twice_level,
        global_max_twice_levels=kernel.global_max_twice_levels,
        momentum_orders=kernel.momentum_orders,
        momentum_maximum=kernel.momentum_maximum,
    )


def integrate_complex_energy_continued_atlas_qmc(
    kernel: BRYNSFiveTachyonIntegrand,
    *,
    orderings: Sequence[Sequence[int]],
    sobol_power: int = 3,
    replicates: int = 2,
    radial_power: float = 0.2,
    pair_radial_powers: Mapping[tuple[int, int], float] | None = None,
    stratify_orderings: bool = True,
    adaptive_c_channel: bool = True,
    seed: int = 20260825,
) -> NSFivePointContinuedQMCResult:
    r"""Integrate a subtraction-free complex chamber in a supplied atlas.

    The caller is responsible for certifying that the supplied kinematics are
    absolutely convergent.  The natural fixed-weight ``c``-recursion atlas
    contains all 120 oriented combs.  A sampled chart may put the incoming
    label on the right; only the residue evaluation is then reflected to the
    equivalent orientation with the incoming label on the left.  Keeping the
    convergence audit outside this matrix-blind kernel avoids a circular
    import with :mod:`type0b_ns_five_tachyon_domain`.
    """

    sobol_power = int(sobol_power)
    replicates = int(replicates)
    radial_power = float(radial_power)
    atlas_orderings = tuple(
        tuple(int(label) for label in ordering) for ordering in orderings
    )
    if sobol_power < 0 or replicates < 2:
        raise ValueError(
            "sobol_power must be non-negative and replicates at least two"
        )
    if not 0.0 < radial_power <= 2.0:
        raise ValueError("radial_power must lie in (0,2]")
    selected_pair_powers = (
        None
        if pair_radial_powers is None
        else {
            tuple(sorted((int(pair[0]), int(pair[1])))): float(value)
            for pair, value in pair_radial_powers.items()
        }
    )
    if selected_pair_powers is not None and any(
        not 0.0 < value <= 2.0 for value in selected_pair_powers.values()
    ):
        raise ValueError("every pair radial power must lie in (0,2]")
    if not atlas_orderings:
        raise ValueError("the continued atlas must contain at least one ordering")
    for ordering in atlas_orderings:
        if len(ordering) != 5 or set(ordering) != set(range(5)):
            raise ValueError("every atlas ordering must permute labels 0,...,4")

    continuous_estimates: list[complex] = []
    left_estimates: list[complex] = []
    right_estimates: list[complex] = []
    nested_estimates: list[complex] = []
    estimates: list[complex] = []
    extreme_sample_diagnostics: list[dict[str, object]] = []
    for replicate in range(replicates):
        component_values: list[ContinuedMomentumDensity] = []
        replicate_extrema: dict[str, dict[str, object]] = {}
        if stratify_orderings:
            sampled = (
                (ordering, sample)
                for ordering_index, ordering in enumerate(atlas_orderings)
                for sample in qmc.Sobol(
                    d=4,
                    scramble=True,
                    seed=int(seed) + 10000 * replicate + ordering_index,
                ).random_base2(sobol_power)
            )
        else:
            sampled = (
                (
                    atlas_orderings[
                        min(
                            int(sample[4] * len(atlas_orderings)),
                            len(atlas_orderings) - 1,
                        )
                    ],
                    sample[:4],
                )
                for sample in qmc.Sobol(
                    d=5, scramble=True, seed=int(seed) + replicate
                ).random_base2(sobol_power)
            )
        for ordering, sample in sampled:
            if selected_pair_powers is None:
                power1 = radial_power
                power2 = radial_power
            else:
                left_pair = tuple(sorted((ordering[0], ordering[1])))
                right_pair = tuple(sorted((ordering[3], ordering[4])))
                try:
                    power1 = selected_pair_powers[left_pair]
                    power2 = selected_pair_powers[right_pair]
                except KeyError as error:
                    raise ValueError(
                        f"missing radial power for boundary pair {error.args[0]}"
                    ) from error
            q1 = _power_disk_sample(sample[0], sample[1], power1)
            q2 = _power_disk_sample(sample[2], sample[3], power2)
            log_proposal_density = _oriented_bidisc_mixture_density_in_channel(
                q1,
                q2,
                ordering,
                radial_power=radial_power,
                orderings=atlas_orderings,
                pair_radial_powers=selected_pair_powers,
                return_log_density=True,
            )
            value = kernel.continued_linear_q_components(
                q1,
                q2,
                ordering,
                evaluation_orderings=(
                    atlas_orderings if adaptive_c_channel else None
                ),
            )

            def weighted(component, component_name: str) -> complex:
                selected = mpmath.mpc(component)
                magnitude = abs(selected)
                if not mpmath.isfinite(magnitude):
                    raise ArithmeticError(
                        "non-finite continued component before importance "
                        f"weighting: ordering={ordering}, "
                        f"log|q|=({math.log(abs(q1))},{math.log(abs(q2))})"
                    )
                if magnitude == 0.0:
                    return 0.0 + 0.0j
                log_magnitude = (
                    float(mpmath.log(magnitude)) - log_proposal_density
                )
                if log_magnitude < math.log(np.nextafter(0.0, 1.0)):
                    return 0.0 + 0.0j
                if log_magnitude > math.log(np.finfo(float).max):
                    raise OverflowError(
                        "the importance-weighted worldsheet density overflowed"
                    )
                if log_magnitude > 100.0:
                    raise ArithmeticError(
                        "importance weight exceeded the diagnostic scale: "
                        f"log|weight|={log_magnitude}, ordering={ordering}, "
                        f"log|q|=({math.log(abs(q1))},{math.log(abs(q2))})"
                    )
                previous = replicate_extrema.get(component_name)
                if (
                    previous is None
                    or log_magnitude > float(previous["log_absolute_weight"])
                ):
                    replicate_extrema[component_name] = {
                        "component": component_name,
                        "log_absolute_weight": log_magnitude,
                        "ordering": list(ordering),
                        "log_absolute_q": [
                            math.log(abs(q1)),
                            math.log(abs(q2)),
                        ],
                        "q": [
                            {"real": q1.real, "imag": q1.imag},
                            {"real": q2.real, "imag": q2.imag},
                        ],
                        "radial_powers": [power1, power2],
                        "log_proposal_density": log_proposal_density,
                    }
                result = complex(
                    selected / magnitude * math.exp(log_magnitude)
                )
                if not math.isfinite(result.real) or not math.isfinite(result.imag):
                    raise ArithmeticError(
                        "importance weighting produced a non-finite phase: "
                        f"log|weight|={log_magnitude}, ordering={ordering}, "
                        f"log|q|=({math.log(abs(q1))},{math.log(abs(q2))})"
                    )
                return result

            component_values.append(
                ContinuedMomentumDensity(
                    continuous=weighted(value.continuous, "continuous"),
                    left_residues=weighted(
                        value.left_residues, "left_residues"
                    ),
                    right_residues=weighted(
                        value.right_residues, "right_residues"
                    ),
                    nested_residues=weighted(
                        value.nested_residues, "nested_residues"
                    ),
                )
            )
        def stable_mean(values) -> complex:
            selected = tuple(complex(value) for value in values)
            return complex(
                math.fsum(value.real for value in selected) / len(selected),
                math.fsum(value.imag for value in selected) / len(selected),
            )

        continuous = stable_mean(
            value.continuous for value in component_values
        )
        left = stable_mean(
            value.left_residues for value in component_values
        )
        right = stable_mean(
            value.right_residues for value in component_values
        )
        nested = stable_mean(
            value.nested_residues for value in component_values
        )
        continuous_estimates.append(continuous)
        left_estimates.append(left)
        right_estimates.append(right)
        nested_estimates.append(nested)
        def component_total(value) -> complex:
            selected = (
                complex(value.continuous),
                complex(value.left_residues),
                complex(value.right_residues),
                complex(value.nested_residues),
            )
            return complex(
                math.fsum(item.real for item in selected),
                math.fsum(item.imag for item in selected),
            )

        total = stable_mean(component_total(value) for value in component_values)
        estimates.append(total)
        extreme_sample_diagnostics.append(
            {
                "replicate": replicate,
                "components": replicate_extrema,
            }
        )

    array = np.asarray(estimates, dtype=complex)
    overall_mean = complex(
        math.fsum(value.real for value in estimates) / len(estimates),
        math.fsum(value.imag for value in estimates) / len(estimates),
    )
    return NSFivePointContinuedQMCResult(
        outgoing_energies=kernel.outgoing_energies,  # type: ignore[arg-type]
        estimates=tuple(estimates),
        continuous_estimates=tuple(continuous_estimates),
        left_residue_estimates=tuple(left_estimates),
        right_residue_estimates=tuple(right_estimates),
        nested_residue_estimates=tuple(nested_estimates),
        mean=overall_mean,
        standard_error_real=float(
            np.std(array.real, ddof=1) / math.sqrt(replicates)
        ),
        standard_error_imag=float(
            np.std(array.imag, ddof=1) / math.sqrt(replicates)
        ),
        samples_per_replicate=(
            len(atlas_orderings) * 2**sobol_power
            if stratify_orderings
            else 2**sobol_power
        ),
        replicates=replicates,
        radial_power=radial_power,
        seed=int(seed),
        recursion_max_twice_level=kernel.recursion_max_twice_level,
        global_max_twice_levels=kernel.global_max_twice_levels,
        momentum_orders=kernel.momentum_orders,
        momentum_maximum=kernel.momentum_maximum,
        extreme_sample_diagnostics=tuple(extreme_sample_diagnostics),
    )


def integrate_imaginary_energy_continued_atlas_qmc(
    kernel: BRYNSFiveTachyonIntegrand,
    *,
    sobol_power: int = 3,
    replicates: int = 2,
    radial_power: float = 0.2,
    seed: int = 20260825,
) -> NSFivePointContinuedQMCResult:
    """Backward-compatible name for the equal-complex-energy driver."""

    return integrate_equal_complex_energy_continued_atlas_qmc(
        kernel,
        sobol_power=sobol_power,
        replicates=replicates,
        radial_power=radial_power,
        seed=seed,
    )


def integrate_complex_energy_one_divisor_qmc(
    kernel: BRYNSFiveTachyonIntegrand,
    *,
    orderings: Sequence[Sequence[int]],
    collar_radius: float = 0.05,
    bulk_sobol_power: int = 1,
    face_sobol_power: int = 2,
    replicates: int = 2,
    radial_power: float = 0.1,
    pair_radial_powers: Mapping[tuple[int, int], float] | None = None,
    normal_correction_order: int = 2,
    normal_correction_angular_order: int = 8,
    seed: int = 20260825,
) -> NSFivePointFinitePartQMCResult:
    r"""Integrate the corrected chamber with one ``D_12`` finite part.

    The full 120-chart mixture is used for sampling and the smallest all-c
    channel for ordinary evaluation.  Inside the unique ``D_12`` collar the
    complete continued five-point density is evaluated in one of its six
    lower four-point crossing cells and its factorized primary normal term is
    subtracted pointwise.  The same term, including the continued lower
    four-point residue contour, is restored by the complex radial finite
    part.  No h-recursive face block is used.
    """

    atlas_orderings = tuple(
        tuple(int(label) for label in ordering) for ordering in orderings
    )
    collar = float(collar_radius)
    bulk_power = int(bulk_sobol_power)
    face_power = int(face_sobol_power)
    replicate_count = int(replicates)
    correction_order = int(normal_correction_order)
    correction_angular_order = int(normal_correction_angular_order)
    if len(atlas_orderings) != 120 or len(set(atlas_orderings)) != 120:
        raise ValueError("the one-divisor driver requires the full 120-chart atlas")
    if not 0.0 < collar < 0.2:
        raise ValueError("collar_radius must lie in (0,0.2)")
    if bulk_power < 0 or face_power < 0 or replicate_count < 2:
        raise ValueError("Sobol powers must be nonnegative and replicates at least two")
    if correction_order < 2:
        raise ValueError("normal_correction_order must be at least two")
    if correction_angular_order < 4 or correction_angular_order % 2:
        raise ValueError(
            "normal_correction_angular_order must be an even integer at least four"
        )
    if not 0.0 < radial_power <= 2.0:
        raise ValueError("radial_power must lie in (0,2]")
    selected_pair_powers = (
        None
        if pair_radial_powers is None
        else {
            tuple(sorted((int(pair[0]), int(pair[1])))): float(value)
            for pair, value in pair_radial_powers.items()
        }
    )
    if selected_pair_powers is not None:
        required_pairs = set(combinations(range(5), 2))
        if set(selected_pair_powers) != required_pairs:
            raise ValueError("pair_radial_powers must specify all ten boundary pairs")
        if any(
            not 0.0 < value <= 2.0
            for value in selected_pair_powers.values()
        ):
            raise ValueError("every pair radial power must lie in (0,2]")

    bulk_estimates: list[complex] = []
    face_estimates: list[complex] = []
    estimates: list[complex] = []
    extreme_bulk_weights: list[dict[str, object]] = []
    for replicate in range(replicate_count):
        bulk_values: list[complex] = []
        replicate_extreme: dict[str, object] | None = None
        for ordering_index, ordering in enumerate(atlas_orderings):
            sampler = qmc.Sobol(
                d=4,
                scramble=True,
                seed=int(seed) + 10000 * replicate + ordering_index,
            )
            for sample in sampler.random_base2(bulk_power):
                if selected_pair_powers is None:
                    power1 = float(radial_power)
                    power2 = float(radial_power)
                else:
                    power1 = selected_pair_powers[
                        tuple(sorted((ordering[0], ordering[1])))
                    ]
                    power2 = selected_pair_powers[
                        tuple(sorted((ordering[3], ordering[4])))
                    ]
                q1 = _power_disk_sample(sample[0], sample[1], power1)
                q2 = _power_disk_sample(sample[2], sample[3], power2)
                log_density = _oriented_bidisc_mixture_density_in_channel(
                    q1,
                    q2,
                    ordering,
                    radial_power=float(radial_power),
                    orderings=atlas_orderings,
                    pair_radial_powers=selected_pair_powers,
                    return_log_density=True,
                )
                face = _one_divisor_face_channel_in_sampled_chart(
                    q1, q2, ordering, collar
                )
                if face is None:
                    components = kernel.continued_linear_q_components(
                        q1,
                        q2,
                        ordering,
                        evaluation_orderings=atlas_orderings,
                    )
                    value = components.total
                    in_subtracted_cell = False
                    face_ordering = None
                else:
                    face_ordering, target_q1, target_q2, transition = face
                    raw = kernel.continued_linear_q_components(
                        target_q1,
                        target_q2,
                        face_ordering,
                        subtracted_continuum_boundary_pair=(1, 2),
                    ).total
                    # The primary is omitted coefficient-by-coefficient in
                    # the all-c block series.  This is algebraically the same
                    # face subtraction as raw-counterterm, but remains stable
                    # when |q_normal| is hundreds of decades below one.
                    value = raw * transition
                    in_subtracted_cell = True

                selected_value = mpmath.mpc(value)
                magnitude = abs(selected_value)
                if magnitude == 0.0:
                    bulk_values.append(0.0j)
                    continue
                log_weight = float(mpmath.log(magnitude)) - log_density
                if log_weight > math.log(np.finfo(float).max):
                    raise OverflowError("the one-divisor bulk weight overflowed")
                if (
                    replicate_extreme is None
                    or log_weight > float(replicate_extreme["log_absolute_weight"])
                ):
                    replicate_extreme = {
                        "replicate": replicate,
                        "log_absolute_weight": log_weight,
                        "sampled_ordering": list(ordering),
                        "log_absolute_sampled_q": [
                            math.log(abs(q1)),
                            math.log(abs(q2)),
                        ],
                        "in_subtracted_cell": in_subtracted_cell,
                        "face_ordering": (
                            None if face_ordering is None else list(face_ordering)
                        ),
                    }
                bulk_values.append(
                    complex(selected_value / magnitude * math.exp(log_weight))
                )
        bulk_estimate = complex(
            math.fsum(value.real for value in bulk_values) / len(bulk_values),
            math.fsum(value.imag for value in bulk_values) / len(bulk_values),
        )

        face_values: list[complex] = []
        face_sampler = qmc.Sobol(
            d=2,
            scramble=True,
            seed=int(seed) + 500000 + replicate,
        )
        for sample in face_sampler.random_base2(face_power):
            modulus, area_jacobian = _four_point_fundamental_cell_sample(
                sample[0], sample[1]
            )
            density = 0.0 + 0.0j
            fixture_correction = mpmath.mpc(0)
            for face_ordering in ONE_DIVISOR_FACE_ORDERINGS:
                positions = linear_channel_positions_by_label(
                    0.5, modulus, face_ordering
                )
                density += (
                    _superghost_pair_factor(positions)
                    * kernel.two_pco_face_continued_finite_part_density(
                        ordering=face_ordering,
                        remaining_modulus=modulus,
                        collar_radius=collar,
                    )
                )
                # The bulk collar omits the exact leading state of the
                # selected all-c block.  Its asymptotic q-primary is restored
                # by the finite part above.  Add back their integrable fixture
                # difference.  The symmetric angular rule cancels the O(q)
                # and O(qbar) modes; the angular average starts at |q|^2.
                for normal_radius, normal_weight in _legendre_interval(
                    correction_order, collar
                ):
                    angular_sum = mpmath.mpc(0)
                    for angular_index in range(correction_angular_order):
                        angle = (
                            2.0
                            * math.pi
                            * (angular_index + 0.5)
                            / correction_angular_order
                        )
                        normal = normal_radius * cmath.exp(1.0j * angle)
                        actual_primary = kernel.continued_linear_q_components(
                            normal,
                            modulus,
                            face_ordering,
                            primary_continuum_boundary_pair=(1, 2),
                        ).total
                        normal_positions = linear_channel_positions_by_label(
                            normal, modulus, face_ordering
                        )
                        asymptotic_primary = (
                            _superghost_pair_factor(normal_positions)
                            * kernel.two_pco_face_continued_counterterm_q_density(
                                ordering=face_ordering,
                                normal_coordinate=normal,
                                remaining_modulus=modulus,
                            )
                        )
                        angular_sum += actual_primary - asymptotic_primary
                    fixture_correction += (
                        normal_weight
                        * 2.0
                        * math.pi
                        * normal_radius
                        * angular_sum
                        / correction_angular_order
                    )
            face_values.append(
                complex(area_jacobian * (density + fixture_correction))
            )
        face_estimate = complex(
            math.fsum(value.real for value in face_values) / len(face_values),
            math.fsum(value.imag for value in face_values) / len(face_values),
        )
        bulk_estimates.append(bulk_estimate)
        face_estimates.append(face_estimate)
        estimates.append(bulk_estimate + face_estimate)
        extreme_bulk_weights.append(replicate_extreme or {"replicate": replicate})

    array = np.asarray(estimates, dtype=complex)
    return NSFivePointFinitePartQMCResult(
        outgoing_energies=kernel.outgoing_energies,  # type: ignore[arg-type]
        estimates=tuple(estimates),
        bulk_estimates=tuple(bulk_estimates),
        face_estimates=tuple(face_estimates),
        corner_contribution=0.0j,
        mean=complex(np.mean(array)),
        standard_error_real=float(
            np.std(array.real, ddof=1) / math.sqrt(replicate_count)
        ),
        standard_error_imag=float(
            np.std(array.imag, ddof=1) / math.sqrt(replicate_count)
        ),
        collar_radius=collar,
        projection_radius=0.0,
        bulk_samples_per_replicate=len(atlas_orderings) * 2**bulk_power,
        face_samples_per_replicate=2**face_power,
        replicates=replicate_count,
        radial_power=float(radial_power),
        seed=int(seed),
        recursion_max_twice_level=kernel.recursion_max_twice_level,
        global_max_twice_levels=kernel.global_max_twice_levels,
        momentum_orders=kernel.momentum_orders,
        momentum_maximum=kernel.momentum_maximum,
        extreme_bulk_weights=tuple(extreme_bulk_weights),
        subtraction_scheme=(
            "one D_12 continuum-primary finite part plus the ordinary "
            "all-c leading-fixture correction; no corner subtraction"
        ),
    )


def integrate_complex_energy_minimal_subtraction_qmc(
    kernel: BRYNSFiveTachyonIntegrand,
    *,
    orderings: Sequence[Sequence[int]],
    target_ordering: Sequence[int] = (1, 2, 0, 3, 4),
    collar_radius: float = 0.05,
    projection_radius: float = 1.0e-5,
    bulk_sobol_power: int = 2,
    face_sobol_power: int = 2,
    replicates: int = 2,
    radial_power: float = 0.2,
    pair_radial_powers: Mapping[tuple[int, int], float] | None = None,
    seed: int = 20260825,
) -> NSFivePointFinitePartQMCResult:
    r"""Legacy one-line finite-part experiment in a supplied all-c atlas.

    In the target four-point crossing cell, the complete wall-one moving
    middle contribution is omitted for ``|q2|<collar_radius``.  Its leading
    normal coefficient is projected with the same all-``c`` block and its
    complex radial power is restored analytically.  The discarded subleading
    collar remainder vanishes as the collar shrinks and is certified by an
    explicit collar/projection-radius scan.  This routine is retained for
    local counterterm tests only: the historical ray used with it has been
    rejected by the complete reflected-pole ledger.  Production must use a
    stable-divisor/corner forest over the symmetric all-c Voronoi split.
    """

    atlas_orderings = tuple(
        tuple(int(label) for label in ordering) for ordering in orderings
    )
    target = tuple(int(label) for label in target_ordering)
    collar = float(collar_radius)
    projection = float(projection_radius)
    bulk_power = int(bulk_sobol_power)
    face_power = int(face_sobol_power)
    replicate_count = int(replicates)
    if not atlas_orderings:
        raise ValueError("the all-c atlas must be nonempty")
    if len(target) != 5 or set(target) != set(range(5)) or target[2] != 0:
        raise ValueError("target_ordering must be a middle-incoming comb")
    if not 0.0 < collar < 0.2:
        raise ValueError("collar_radius must lie in (0,0.2)")
    if not 1.0e-8 <= projection < min(1.0e-3, 0.1 * collar):
        raise ValueError(
            "projection_radius must lie in [1e-8,min(1e-3,0.1*collar))"
        )
    if bulk_power < 0 or face_power < 0 or replicate_count < 2:
        raise ValueError("Sobol powers must be nonnegative and replicates at least two")
    selected_pair_powers = (
        None
        if pair_radial_powers is None
        else {
            tuple(sorted((int(pair[0]), int(pair[1])))): float(value)
            for pair, value in pair_radial_powers.items()
        }
    )

    bulk_estimates: list[complex] = []
    face_estimates: list[complex] = []
    estimates: list[complex] = []
    extreme_bulk_weights: list[dict[str, object]] = []
    for replicate in range(replicate_count):
        bulk_values: list[complex] = []
        replicate_extreme: dict[str, object] | None = None
        for ordering_index, ordering in enumerate(atlas_orderings):
            sampler = qmc.Sobol(
                d=4,
                scramble=True,
                seed=int(seed) + 10000 * replicate + ordering_index,
            )
            for sample in sampler.random_base2(bulk_power):
                if selected_pair_powers is None:
                    power1 = float(radial_power)
                    power2 = float(radial_power)
                else:
                    power1 = selected_pair_powers[
                        tuple(sorted((ordering[0], ordering[1])))
                    ]
                    power2 = selected_pair_powers[
                        tuple(sorted((ordering[3], ordering[4])))
                    ]
                q1 = _power_disk_sample(sample[0], sample[1], power1)
                q2 = _power_disk_sample(sample[2], sample[3], power2)
                log_density = _oriented_bidisc_mixture_density_in_channel(
                    q1,
                    q2,
                    ordering,
                    radial_power=float(radial_power),
                    orderings=atlas_orderings,
                    pair_radial_powers=selected_pair_powers,
                    return_log_density=True,
                )

                in_subtracted_cell = False
                try:
                    target_q1, target_q2, transition = _channel_density_transition(
                        q1, q2, ordering, target
                    )
                    in_subtracted_cell = (
                        0.0 < abs(target_q2) < collar
                        and 0.0 < abs(target_q1) < 1.0
                    )
                except (ArithmeticError, OverflowError, ZeroDivisionError):
                    in_subtracted_cell = False

                if in_subtracted_cell:
                    components = kernel.continued_linear_q_components(
                        target_q1,
                        target_q2,
                        target,
                        excluded_middle_walls=(1.0,),
                    )
                    value = components.total * transition
                else:
                    components = kernel.continued_linear_q_components(
                        q1,
                        q2,
                        ordering,
                        evaluation_orderings=atlas_orderings,
                    )
                    value = components.total

                selected_value = mpmath.mpc(value)
                magnitude = abs(selected_value)
                if magnitude == 0:
                    bulk_values.append(0.0j)
                    continue
                log_weight = float(mpmath.log(magnitude)) - log_density
                if log_weight > math.log(np.finfo(float).max):
                    raise OverflowError("the finite-part bulk weight overflowed")
                if (
                    replicate_extreme is None
                    or log_weight
                    > float(replicate_extreme["log_absolute_weight"])
                ):
                    replicate_extreme = {
                        "replicate": replicate,
                        "log_absolute_weight": log_weight,
                        "sampled_ordering": list(ordering),
                        "log_absolute_sampled_q": [
                            math.log(abs(q1)),
                            math.log(abs(q2)),
                        ],
                        "in_subtracted_cell": in_subtracted_cell,
                        "target_q": (
                            [
                                {"real": target_q1.real, "imag": target_q1.imag},
                                {"real": target_q2.real, "imag": target_q2.imag},
                            ]
                            if in_subtracted_cell
                            else None
                        ),
                    }
                bulk_values.append(
                    complex(selected_value / magnitude * math.exp(log_weight))
                )
        bulk_estimate = complex(
            math.fsum(value.real for value in bulk_values) / len(bulk_values),
            math.fsum(value.imag for value in bulk_values) / len(bulk_values),
        )

        face_values: list[complex] = []
        face_sampler = qmc.Sobol(
            d=2,
            scramble=True,
            seed=int(seed) + 500000 + replicate,
        )
        for sample in face_sampler.random_base2(face_power):
            modulus = _power_disk_sample(sample[0], sample[1], 2.0)
            density = kernel.continued_middle_line_face_finite_part_density(
                modulus,
                target,
                collar_radius=collar,
                projection_radius=projection,
                wall=1.0,
            )
            face_values.append(complex(math.pi * density))
        face_estimate = complex(
            math.fsum(value.real for value in face_values) / len(face_values),
            math.fsum(value.imag for value in face_values) / len(face_values),
        )
        bulk_estimates.append(bulk_estimate)
        face_estimates.append(face_estimate)
        estimates.append(bulk_estimate + face_estimate)
        extreme_bulk_weights.append(replicate_extreme or {"replicate": replicate})

    array = np.asarray(estimates, dtype=complex)
    return NSFivePointFinitePartQMCResult(
        outgoing_energies=kernel.outgoing_energies,  # type: ignore[arg-type]
        estimates=tuple(estimates),
        bulk_estimates=tuple(bulk_estimates),
        face_estimates=tuple(face_estimates),
        corner_contribution=0.0j,
        mean=complex(np.mean(array)),
        standard_error_real=float(
            np.std(array.real, ddof=1) / math.sqrt(replicate_count)
        ),
        standard_error_imag=float(
            np.std(array.imag, ddof=1) / math.sqrt(replicate_count)
        ),
        collar_radius=collar,
        projection_radius=projection,
        bulk_samples_per_replicate=len(atlas_orderings) * 2**bulk_power,
        face_samples_per_replicate=2**face_power,
        replicates=replicate_count,
        radial_power=float(radial_power),
        seed=int(seed),
        recursion_max_twice_level=kernel.recursion_max_twice_level,
        global_max_twice_levels=kernel.global_max_twice_levels,
        momentum_orders=kernel.momentum_orders,
        momentum_maximum=kernel.momentum_maximum,
        extreme_bulk_weights=tuple(extreme_bulk_weights),
    )


def integrate_imaginary_energy_finite_part_qmc(
    kernel: BRYNSFiveTachyonIntegrand,
    *,
    collar_radius: float = 0.08,
    bulk_sobol_power: int = 4,
    face_sobol_power: int = 4,
    replicates: int = 2,
    radial_power: float = 0.5,
    projection_radius: float = 1.0e-5,
    seed: int = 20260825,
) -> NSFivePointFinitePartQMCResult:
    r"""Integrate the imaginary-energy worldsheet density by local finite part.

    This is the leading local forest on ``Mbar_0,5``.  The bulk excludes a
    circular collar around every one of the ten boundary divisors.  The
    leading spin-zero normal coefficient on each face is integrated by BRY's
    radial finite part over six four-point crossing cells, with the remaining
    collar excised.  Finally both commuting radial finite parts are applied
    at all fifteen compatible corners.  The finite collar and projection
    radii are explicit certification parameters.
    """

    collar_radius = float(collar_radius)
    bulk_sobol_power = int(bulk_sobol_power)
    face_sobol_power = int(face_sobol_power)
    replicates = int(replicates)
    radial_power = float(radial_power)
    projection_radius = float(projection_radius)
    if not math.isfinite(collar_radius) or not 0.0 < collar_radius < 0.2:
        raise ValueError("collar_radius must lie in (0,0.2)")
    if bulk_sobol_power < 1 or face_sobol_power < 1 or replicates < 2:
        raise ValueError("Sobol powers must be positive and replicates at least two")
    if not 0.0 < radial_power <= 2.0:
        raise ValueError("radial_power must lie in (0,2]")
    if not 1.0e-7 <= projection_radius < min(1.0e-3, 0.1 * collar_radius):
        raise ValueError(
            "projection_radius must lie in [1e-7,min(1e-3,0.1*collar))"
        )
    audit = imaginary_energy_chamber_audit(kernel.outgoing_energies)
    if not audit["undeformed_positive_real_liouville_contour_valid"]:
        raise ValueError(
            "super-Liouville poles have crossed the positive-real momentum "
            "contour; this residue-free finite-part driver is not valid"
        )

    orderings = oriented_tree_orderings()
    equal_outgoing = max(
        abs(value - kernel.outgoing_energies[0])
        for value in kernel.outgoing_energies
    ) < 1.0e-13
    face_orbits = (
        BOUNDARY_FACE_RAISED_ORBITS
        if equal_outgoing
        else tuple((ordering, 1) for ordering in BOUNDARY_FACE_SECTOR_ORDERINGS)
    )
    corner_contribution = complex(
        sum(
            kernel.boundary_corner_finite_part(
                ordering=ordering,
                collar_radius=collar_radius,
                projection_radius=projection_radius,
            )
            for ordering in BOUNDARY_CORNER_ORDERINGS
        )
    )
    bulk_estimates: list[complex] = []
    face_estimates: list[complex] = []
    estimates: list[complex] = []
    for replicate in range(replicates):
        bulk_sampler = qmc.Sobol(
            d=5, scramble=True, seed=int(seed) + replicate
        )
        bulk_values: list[complex] = []
        for sample in bulk_sampler.random_base2(bulk_sobol_power):
            q1 = _power_disk_sample(sample[0], sample[1], radial_power)
            q2 = _power_disk_sample(sample[2], sample[3], radial_power)
            ordering_index = min(
                int(sample[4] * len(orderings)), len(orderings) - 1
            )
            positions = _to_fixed_gauge(q1, q2, orderings[ordering_index])
            best = best_linear_channels(positions, limit=1)[0]
            if (
                abs(best.q1) < collar_radius
                or abs(best.q2) < collar_radius
            ):
                bulk_values.append(0.0 + 0.0j)
                continue
            proposal_density = _oriented_bidisc_mixture_density(
                positions, radial_power=radial_power
            )
            bulk_values.append(
                kernel.fixed_gauge_integrand_positions(positions)
                / proposal_density
            )
        bulk_estimate = complex(
            np.mean(np.asarray(bulk_values, dtype=complex))
        )

        face_sampler = qmc.Sobol(
            d=2,
            scramble=True,
            seed=int(seed) + 10000 + replicate,
        )
        face_values: list[complex] = []
        for sample in face_sampler.random_base2(face_sobol_power):
            modulus, area_jacobian = _plane_map(sample[0], sample[1])
            if (
                not _four_point_fundamental_cell(modulus)
                or abs(modulus) < collar_radius
            ):
                face_values.append(0.0 + 0.0j)
                continue
            density = sum(
                multiplicity
                * kernel.boundary_face_finite_part_density(
                    ordering=ordering,
                    remaining_modulus=modulus,
                    collar_radius=collar_radius,
                    projection_radius=projection_radius,
                )
                for ordering, multiplicity in face_orbits
            )
            face_values.append(complex(area_jacobian * density))
        face_estimate = complex(
            np.mean(np.asarray(face_values, dtype=complex))
        )

        bulk_estimates.append(bulk_estimate)
        face_estimates.append(face_estimate)
        estimates.append(bulk_estimate + face_estimate + corner_contribution)

    array = np.asarray(estimates, dtype=complex)
    return NSFivePointFinitePartQMCResult(
        outgoing_energies=kernel.outgoing_energies,  # type: ignore[arg-type]
        estimates=tuple(estimates),
        bulk_estimates=tuple(bulk_estimates),
        face_estimates=tuple(face_estimates),
        corner_contribution=corner_contribution,
        mean=complex(np.mean(array)),
        standard_error_real=float(
            np.std(array.real, ddof=1) / math.sqrt(replicates)
        ),
        standard_error_imag=float(
            np.std(array.imag, ddof=1) / math.sqrt(replicates)
        ),
        collar_radius=collar_radius,
        projection_radius=projection_radius,
        bulk_samples_per_replicate=2**bulk_sobol_power,
        face_samples_per_replicate=2**face_sobol_power,
        replicates=replicates,
        radial_power=radial_power,
        seed=int(seed),
        recursion_max_twice_level=kernel.recursion_max_twice_level,
        global_max_twice_levels=kernel.global_max_twice_levels,
        momentum_orders=kernel.momentum_orders,
        momentum_maximum=kernel.momentum_maximum,
    )


__all__ = [
    "BOUNDARY_CORNER_ORDERINGS",
    "BOUNDARY_FACE_SECTOR_ORDERINGS",
    "BRYNSFiveTachyonIntegrand",
    "ContinuedMomentumDensity",
    "CrossedNSStructurePole",
    "MovingMiddleCornerTerm",
    "MovingMiddleFaceTerm",
    "MovingMiddleResidueTerm",
    "NSFivePointContinuedQMCResult",
    "NSFivePointFinitePartQMCResult",
    "NSFivePointQMCResult",
    "ODD_SECTOR_ASSIGNMENTS",
    "ONE_DIVISOR_FACE_ORDERINGS",
    "PCO_FACE_SECTOR_ORDERINGS",
    "PCOChiralTerm",
    "balanced_equal_energy",
    "crossed_ns_structure_poles_complex",
    "equal_complex_energy_convergence_audit",
    "integrate_imaginary_energy_finite_part_qmc",
    "integrate_imaginary_energy_continued_atlas_qmc",
    "integrate_equal_complex_energy_continued_atlas_qmc",
    "integrate_complex_energy_continued_atlas_qmc",
    "integrate_complex_energy_one_divisor_qmc",
    "integrate_complex_energy_minimal_subtraction_qmc",
    "integrate_imaginary_energy_atlas_qmc",
    "imaginary_energy_chamber_audit",
    "incoming_endpoint_linear_channels",
    "crossed_ns_structure_poles",
    "pco_safe_linear_channels",
    "pco_chiral_terms",
]
