#!/usr/bin/env python3
r"""Hybrid-recursion Type-0B sphere four-point amplitude.

The external labels and fixed gauge are

``(z_0,z_1,z_2,z_3)=(infinity,1,z,0)``

with momenta ``(omega_in,omega_3,omega_2,omega_1)`` and signed timelike
energies ``(omega_in,-omega_3,-omega_2,-omega_1)``.  Labels 1 and 2 are in
picture zero; labels 0 and 3 are in picture minus one.  The implementation
uses the fixed-difference ``h`` recursion at ordinary moduli points and the
fixed-weight ``c`` recursion in local plumbing collars.

The external momenta are analytically continued from their real parts.  All
super-Liouville poles crossed by that continuation are added on the BRY
positive-half momentum contour. The convergence audit checks the continuum
endpoint and every crossed residue separately at all three divisors of
``Mbar_0,4``. Thus a positive complete-contour audit margin certifies an
ordinary, unsubtracted moduli integral; it is not merely a statement about
the continuum term. The equal pure-imaginary fixed-contour path omits those
residues and is diagnostic only. Separate experimental routines implement a
meromorphic radial finite part of the crossed-pole term, following the
bosonic ``c=1`` reference strategy.
"""

from __future__ import annotations

import cmath
from dataclasses import asdict, dataclass
from functools import lru_cache
from itertools import permutations
import math
from pathlib import Path
import sys
from typing import Literal, Sequence, Union

import mpmath
import numpy as np
from scipy.stats import qmc


HERE = Path(__file__).resolve().parent
CODE_ROOT = HERE.parent
FIVE_POINT = (
    CODE_ROOT / "higher_point_amplitude_attempts" / "type0b_ns_five_tachyon"
)
REFERENCE_PLUMBING = (
    CODE_ROOT
    / "bosonic_c1_one_to_n_reference"
    / "reference_implementation"
    / "plumbing"
)
for dependency in (HERE, FIVE_POINT, REFERENCE_PLUMBING):
    if str(dependency) not in sys.path:
        sys.path.insert(0, str(dependency))

from ns_multipoint_c_recursion import NSSphereLinearCRecursion  # noqa: E402
from ns_multipoint_h_recursion import NSSphereLinearHRecursion  # noqa: E402
from sphere_four_point import BRYFourTachyonSphere  # noqa: E402
from superconformal_blocks import (  # noqa: E402
    NSSphereFourPointBlock,
    _series_compose,
    _series_mul,
    _series_pow,
    elliptic_nome,
)
from sphere_five_point_liouville import (  # noqa: E402
    MobiusMap,
    ProjectivePoint,
    mobius_to_zero_one_infinity,
)
from super_liouville_structure_constants import (  # noqa: E402
    ns_structure_constant,
    ns_structure_constant_mp,
    ns_tilde_structure_constant,
    ns_tilde_structure_constant_mp,
)
from type0b_ns_five_tachyon import (  # noqa: E402
    CrossedNSStructurePole,
    _positive_contour_structure_poles,
)


Number = Union[complex, float]
BlockBackend = Literal["hybrid", "h", "c"]
BlockRegion = Literal["auto", "bulk", "corner"]
ContourPrescription = Literal["fixed", "continued"]
MomentumRule = Literal["global", "wall-one-30"]

PICTURE_ZERO_LABELS = (1, 2)
MINUS_ONE_LABELS = (0, 3)
SECTOR_ASSIGNMENTS = ((0, 0), (1, 1))
STANDARD_ZERO_DESCENDANT_PHASE = -1.0 + 0.0j
STANDARD_INFINITY_DESCENDANT_PHASE = -1.0j

# A deliberately non-coincident ray.  In the rectangle below every stable
# boundary contribution has a positive radial margin and the crossed b=1
# poles have order at most two.
CONVERGENT_RAY_COEFFICIENTS = (1.0, 0.98, 0.92)
CONVERGENT_RAY_RECTANGLE = ((0.495, 0.505), (0.745, 0.755))
CONVERGENT_RAY_REFERENCE = -0.5 + 0.75j
LARGE_RESIDUE_RAY_RECTANGLE = ((0.965, 1.055), (1.185, 1.265))
LARGE_RESIDUE_RAY_REFERENCE = -1.01 + 1.225j
WALL_ONE_RAY_COEFFICIENTS = (0.1, 1.0, 1.0)
WALL_ONE_RAY_RECTANGLE = ((0.238, 0.304), (0.596, 0.628))
WALL_TWO_RAY_RECTANGLE = ((0.493, 0.515), (0.742, 0.761))
WALL_THREE_RAY_RECTANGLE = ((0.695, 0.7575), (0.93, 0.9825))
WALL_ONE_MOMENTUM_INTERVALS = (
    (0.0, 0.3, 3),
    (0.3, 0.7, 7),
    (0.7, 1.1, 14),
    (1.1, 1.5, 3),
    (1.5, 3.0, 3),
)


def _finite_complex(name: str, value: Number) -> complex:
    result = complex(value)
    if not math.isfinite(result.real) or not math.isfinite(result.imag):
        raise ValueError(f"{name} must be finite")
    return result


def _validate_positions(
    positions: Sequence[ProjectivePoint],
) -> tuple[ProjectivePoint, ...]:
    if len(positions) != 4:
        raise ValueError("positions must contain four projective punctures")
    result = tuple(
        None if value is None else _finite_complex(f"positions[{index}]", value)
        for index, value in enumerate(positions)
    )
    homogeneous = tuple(
        (1.0 + 0.0j, 0.0 + 0.0j)
        if value is None
        else (complex(value), 1.0 + 0.0j)
        for value in result
    )
    for left in range(4):
        for right in range(left):
            if (
                homogeneous[left][0] * homogeneous[right][1]
                - homogeneous[left][1] * homogeneous[right][0]
            ) == 0.0:
                raise ValueError("sphere punctures must be pairwise distinct")
    return result


@dataclass(frozen=True)
class FourPointChannel:
    """One oriented four-leaf plumbing frame ``(0,q,1,infinity)``."""

    ordering: tuple[int, int, int, int]
    q: complex
    mobius: MobiusMap
    local_scales: tuple[complex, complex, complex, complex]
    score: float

    @property
    def cherry(self) -> tuple[int, int]:
        return self.ordering[0], self.ordering[1]


def four_point_channel_from_ordering(
    positions: Sequence[ProjectivePoint], ordering: Sequence[int]
) -> FourPointChannel:
    normalized = _validate_positions(positions)
    selected = tuple(int(label) for label in ordering)
    if len(selected) != 4 or set(selected) != set(range(4)):
        raise ValueError("ordering must permute labels 0,...,3")
    a, b, c, d = selected
    transform = mobius_to_zero_one_infinity(
        normalized[a], normalized[c], normalized[d]
    )
    local = transform(normalized[b])
    if local is None or local == 0.0 or local == 1.0:
        raise ValueError("the ordered four-point channel is degenerate")
    scales = tuple(transform.local_scale(normalized[label]) for label in selected)
    return FourPointChannel(
        ordering=selected,
        q=complex(local),
        mobius=transform,
        local_scales=scales,
        score=float(abs(local)),
    )


def best_four_point_channel(
    positions: Sequence[ProjectivePoint],
) -> FourPointChannel:
    """Return the fastest of the six anharmonic plumbing frames."""

    candidates: list[FourPointChannel] = []
    for ordering in permutations(range(4)):
        try:
            channel = four_point_channel_from_ordering(positions, ordering)
        except (ValueError, ZeroDivisionError):
            continue
        if channel.score < 1.0 + 1.0e-13:
            candidates.append(channel)
    if not candidates:
        raise RuntimeError("no convergent four-point channel was found")
    return min(candidates, key=lambda value: (value.score, value.ordering))


@dataclass(frozen=True)
class FourPointBoundaryMargin:
    partition: tuple[int, int]
    side: tuple[int, int]
    kind: str
    sector: int | None
    momentum: complex
    channel_energy: complex
    threshold: int
    margin: float

    def to_json(self) -> dict[str, object]:
        result = asdict(self)
        result["partition"] = list(self.partition)
        result["side"] = list(self.side)
        result["momentum"] = {
            "real": self.momentum.real,
            "imag": self.momentum.imag,
        }
        result["channel_energy"] = {
            "real": self.channel_energy.real,
            "imag": self.channel_energy.imag,
        }
        return result


@dataclass(frozen=True)
class FourPointConvergenceAudit:
    outgoing_energies: tuple[complex, complex, complex]
    incoming_energy: complex
    records: tuple[FourPointBoundaryMargin, ...]
    minimum_margin: float
    minimum_wall_clearance: float
    convergent: bool

    def to_json(self) -> dict[str, object]:
        return {
            "outgoing_energies": [
                {"real": value.real, "imag": value.imag}
                for value in self.outgoing_energies
            ],
            "incoming_energy": {
                "real": self.incoming_energy.real,
                "imag": self.incoming_energy.imag,
            },
            "records": [record.to_json() for record in self.records],
            "minimum_margin": self.minimum_margin,
            "minimum_wall_clearance": self.minimum_wall_clearance,
            "convergent": self.convergent,
        }


@dataclass(frozen=True)
class FourPointCrossingFrameValue:
    """One channel evaluation entering a pointwise crossing audit."""

    frame: int
    q: complex
    backend: str
    continuous: complex
    residues: complex

    @property
    def total(self) -> complex:
        return self.continuous + self.residues

    def to_json(self) -> dict[str, object]:
        def encode(value: complex) -> dict[str, float]:
            return {"real": value.real, "imag": value.imag}

        return {
            "frame": self.frame,
            "q": encode(self.q),
            "backend": self.backend,
            "continuous": encode(self.continuous),
            "residues": encode(self.residues),
            "total": encode(self.total),
        }


@dataclass(frozen=True)
class FourPointCrossingAudit:
    """Pointwise equality test for several channel decompositions."""

    z: complex
    values: tuple[FourPointCrossingFrameValue, ...]
    maximum_absolute_difference: float
    relative_spread: float
    relative_tolerance: float
    passed: bool

    def to_json(self) -> dict[str, object]:
        return {
            "z": {"real": self.z.real, "imag": self.z.imag},
            "values": [value.to_json() for value in self.values],
            "maximum_absolute_difference": self.maximum_absolute_difference,
            "relative_spread": self.relative_spread,
            "relative_tolerance": self.relative_tolerance,
            "passed": self.passed,
        }


@dataclass(frozen=True)
class FourPointRayRectangleCertificate:
    x_interval: tuple[float, float]
    t_interval: tuple[float, float]
    minimum_margin_lower_bound: float
    minimum_wall_clearance: float
    chamber_record_count: int
    certified: bool

    def to_json(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class FourPointResidueWallCertificate:
    wall: int
    record_count: int
    minimum_margin_lower_bound: float
    maximum_combined_pole_order: int
    maximum_logarithm_power: int

    def to_json(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class FourPointResidueDomainCertificate:
    ray_coefficients: tuple[float, float, float]
    x_interval: tuple[float, float]
    t_interval: tuple[float, float]
    continuum_record_count: int
    continuum_minimum_margin_lower_bound: float
    residue_walls: tuple[FourPointResidueWallCertificate, ...]
    minimum_wall_clearance: float
    chamber_record_count: int
    chamber_stable: bool
    certified: bool

    def to_json(self) -> dict[str, object]:
        result = asdict(self)
        result["residue_walls"] = [
            wall.to_json() for wall in self.residue_walls
        ]
        return result


def _wall_clearance(first: complex, second: complex, sector: int) -> float:
    start = 1 if sector == 0 else 2
    combinations = [first + second, first - second, second - first]
    maximum = max(abs(value.imag) for value in combinations)
    walls = range(start, int(math.ceil(maximum)) + 4, 2)
    return min(abs(value.imag - wall) for value in combinations for wall in walls)


def audit_four_point_convergence(
    outgoing_energies: Sequence[Number],
    *,
    include_residues: bool = True,
) -> FourPointConvergenceAudit:
    r"""Audit ``Re(P^2-K^2)>tau`` on all continuum/residue strata.

    The raised--raised partition ``{1,2}|{0,3}`` has ``tau=1``.  The other
    two partitions are mixed-picture and have ``tau=0``.
    """

    if len(outgoing_energies) != 3:
        raise ValueError("outgoing_energies must contain three values")
    outgoing = tuple(
        _finite_complex(f"outgoing_energies[{index}]", value)
        for index, value in enumerate(outgoing_energies)
    )
    incoming = sum(outgoing)
    external = (incoming, outgoing[2], outgoing[1], outgoing[0])
    signed = (incoming, -outgoing[2], -outgoing[1], -outgoing[0])
    partitions = ((0, 1), (0, 2), (1, 2))
    records: list[FourPointBoundaryMargin] = []
    clearances: list[float] = []
    for partition in partitions:
        complement = tuple(label for label in range(4) if label not in partition)
        threshold = int(frozenset(partition) == frozenset(PICTURE_ZERO_LABELS))
        for side in (partition, complement):
            energy = signed[side[0]] + signed[side[1]]
            records.append(
                FourPointBoundaryMargin(
                    partition=partition,
                    side=side,  # type: ignore[arg-type]
                    kind="continuous",
                    sector=None,
                    momentum=0.0 + 0.0j,
                    channel_energy=energy,
                    threshold=threshold,
                    margin=float((-energy * energy).real - threshold),
                )
            )
            for sector in (0, 1):
                clearances.append(
                    _wall_clearance(external[side[0]], external[side[1]], sector)
                )
                if not include_residues:
                    continue
                for pole in _positive_contour_structure_poles(
                    external[side[0]], external[side[1]], sector
                ):
                    records.append(
                        FourPointBoundaryMargin(
                            partition=partition,
                            side=side,  # type: ignore[arg-type]
                            kind=f"residue:{pole.family}:wall-{pole.wall:g}",
                            sector=sector,
                            momentum=pole.momentum,
                            channel_energy=energy,
                            threshold=threshold,
                            margin=float(
                                (
                                    pole.momentum * pole.momentum
                                    - energy * energy
                                ).real
                                - threshold
                            ),
                        )
                    )
    minimum = min(record.margin for record in records)
    clearance = min(clearances)
    return FourPointConvergenceAudit(
        outgoing_energies=outgoing,  # type: ignore[arg-type]
        incoming_energy=incoming,
        records=tuple(records),
        minimum_margin=minimum,
        minimum_wall_clearance=clearance,
        convergent=minimum > 0.0 and clearance > 1.0e-10,
    )


def convergent_ray_energies(base: Number) -> tuple[complex, complex, complex]:
    value = _finite_complex("base", base)
    return tuple(value * coefficient for coefficient in CONVERGENT_RAY_COEFFICIENTS)  # type: ignore[return-value]


def certify_convergent_ray_rectangle(
    x_interval: Sequence[float] = CONVERGENT_RAY_RECTANGLE[0],
    t_interval: Sequence[float] = CONVERGENT_RAY_RECTANGLE[1],
) -> FourPointRayRectangleCertificate:
    r"""Certify the advertised open pole chamber by exact quadratic minima.

    On a fixed pole chamber every momentum is ``A*lambda-i*m`` up to the
    harmless quotient reflection, while ``K=B*lambda``.  Consequently each
    margin is a separable quadratic in ``x`` and ``t``.  We minimize that
    quadratic on the full rectangle, not on a sampling grid.
    """

    if len(x_interval) != 2 or len(t_interval) != 2:
        raise ValueError("each interval must contain two endpoints")
    x0, x1 = (float(value) for value in x_interval)
    t0, t1 = (float(value) for value in t_interval)
    if not 0.0 < x0 < x1 or not 0.0 < t0 < t1:
        raise ValueError("rectangle endpoints must be positive and increasing")
    center_x = 0.5 * (x0 + x1)
    center_t = 0.5 * (t0 + t1)
    center_lambda = complex(-center_x, center_t)
    audits = tuple(
        audit_four_point_convergence(convergent_ray_energies(complex(-x, t)))
        for x in (x0, x1)
        for t in (t0, t1)
    )
    center = audit_four_point_convergence(
        convergent_ray_energies(center_lambda)
    )

    def signature(audit: FourPointConvergenceAudit) -> tuple[object, ...]:
        return tuple(
            (record.partition, record.side, record.kind, record.sector)
            for record in audit.records
        )

    chamber_stable = all(signature(audit) == signature(center) for audit in audits)
    lower_bounds: list[float] = []
    for record in center.records:
        b_coefficient = complex(record.channel_energy / center_lambda).real
        if record.kind == "continuous":
            a_coefficient = 0.0
            wall = 0.0
        else:
            a_coefficient = abs(record.momentum.real / center_lambda.real)
            wall = float(record.kind.rsplit("-", 1)[1])
        coefficient = a_coefficient**2 - b_coefficient**2
        x_value = x0 if coefficient >= 0.0 else x1
        t_candidates = [t0, t1]
        # The t-quadratic is convex only when coefficient is negative.
        if coefficient < 0.0:
            stationary = a_coefficient * wall / coefficient
            if t0 < stationary < t1:
                t_candidates.append(stationary)
        lower_bounds.append(
            min(
                coefficient * (x_value**2 - t_value**2)
                + 2.0 * a_coefficient * wall * t_value
                - wall**2
                - record.threshold
                for t_value in t_candidates
            )
        )
    minimum_margin = min(lower_bounds)
    minimum_clearance = min(audit.minimum_wall_clearance for audit in audits)
    return FourPointRayRectangleCertificate(
        x_interval=(x0, x1),
        t_interval=(t0, t1),
        minimum_margin_lower_bound=float(minimum_margin),
        minimum_wall_clearance=float(minimum_clearance),
        chamber_record_count=len(center.records),
        certified=(
            chamber_stable
            and minimum_margin > 0.0
            and minimum_clearance > 1.0e-10
        ),
    )


def _ray_record_rectangle_lower_bound(
    record: FourPointBoundaryMargin,
    center_lambda: complex,
    x_interval: tuple[float, float],
    t_interval: tuple[float, float],
) -> float:
    """Minimize one fixed-ledger radial margin on a ray rectangle."""

    b_coefficient = complex(record.channel_energy / center_lambda).real
    if record.kind == "continuous":
        a_coefficient = 0.0
        wall = 0.0
    else:
        a_coefficient = abs(record.momentum.real / center_lambda.real)
        wall = float(record.kind.rsplit("-", 1)[1])
    coefficient = a_coefficient**2 - b_coefficient**2
    x_value = x_interval[0] if coefficient >= 0.0 else x_interval[1]
    return float(
        min(
            coefficient * (x_value**2 - t_value**2)
            + 2.0 * a_coefficient * wall * t_value
            - wall**2
            - record.threshold
            for t_value in t_interval
        )
    )


def _grouped_residue_orders(
    outgoing_energies: Sequence[complex],
) -> dict[int, int]:
    """Return the largest full-product pole order involving each wall."""

    outgoing = tuple(complex(value) for value in outgoing_energies)
    incoming = sum(outgoing)
    external = (incoming, outgoing[2], outgoing[1], outgoing[0])
    result: dict[int, int] = {}
    for partition in ((0, 1), (0, 2), (1, 2)):
        complement = tuple(
            label for label in range(4) if label not in partition
        )
        for sector in (0, 1):
            groups: list[list[CrossedNSStructurePole]] = []
            for pair in (partition, complement):
                for pole in _positive_contour_structure_poles(
                    external[pair[0]], external[pair[1]], sector
                ):
                    for group in groups:
                        if abs(pole.momentum - group[0].momentum) < 1.0e-9:
                            group.append(pole)
                            break
                    else:
                        groups.append([pole])
            for group in groups:
                combined_order = sum(
                    int(round(pole.wall)) for pole in group
                )
                for wall in {int(round(pole.wall)) for pole in group}:
                    result[wall] = max(result.get(wall, 0), combined_order)
    return result


def certify_residue_convergent_ray_rectangle(
    x_interval: Sequence[float] = LARGE_RESIDUE_RAY_RECTANGLE[0],
    t_interval: Sequence[float] = LARGE_RESIDUE_RAY_RECTANGLE[1],
    *,
    ray_coefficients: Sequence[float] = CONVERGENT_RAY_COEFFICIENTS,
    ray_real_sign: int = -1,
) -> FourPointResidueDomainCertificate:
    r"""Certify continuum and wall-1,... residue convergence separately.

    The pole ledger is required to agree at all four corners. Within that
    chamber each radial margin is an explicit quadratic in ``x`` and ``t``;
    the returned bounds are analytic rectangle minima, not grid samples.
    Pole multiplicities from coincident left/right trinion poles are also
    recorded because an order-``k`` residue can generate logarithms through
    ``(log|q|)^(k-1)``.
    """

    if len(x_interval) != 2 or len(t_interval) != 2:
        raise ValueError("each interval must contain two endpoints")
    if len(ray_coefficients) != 3:
        raise ValueError("ray_coefficients must contain three values")
    coefficients = tuple(float(value) for value in ray_coefficients)
    if any(value <= 0.0 or not math.isfinite(value) for value in coefficients):
        raise ValueError("ray coefficients must be positive and finite")
    real_sign = int(ray_real_sign)
    if real_sign not in (-1, 1):
        raise ValueError("ray_real_sign must be -1 or +1")
    x0, x1 = (float(value) for value in x_interval)
    t0, t1 = (float(value) for value in t_interval)
    if not 0.0 < x0 < x1 or not 0.0 < t0 < t1:
        raise ValueError("rectangle endpoints must be positive and increasing")
    x_pair = (x0, x1)
    t_pair = (t0, t1)
    center_lambda = complex(
        real_sign * 0.5 * (x0 + x1), 0.5 * (t0 + t1)
    )

    def energies(value: complex) -> tuple[complex, complex, complex]:
        return tuple(value * coefficient for coefficient in coefficients)  # type: ignore[return-value]

    center = audit_four_point_convergence(
        energies(center_lambda), include_residues=True
    )
    corners = tuple(
        audit_four_point_convergence(
            energies(complex(real_sign * x_value, t_value)),
            include_residues=True,
        )
        for x_value in x_pair
        for t_value in t_pair
    )

    def signature(audit: FourPointConvergenceAudit) -> tuple[object, ...]:
        return tuple(
            (record.partition, record.side, record.kind, record.sector)
            for record in audit.records
        )

    chamber_stable = all(
        signature(corner) == signature(center) for corner in corners
    )
    continuum_bounds: list[float] = []
    residue_bounds: dict[int, list[float]] = {}
    for record in center.records:
        lower_bound = _ray_record_rectangle_lower_bound(
            record, center_lambda, x_pair, t_pair
        )
        if record.kind == "continuous":
            continuum_bounds.append(lower_bound)
        else:
            wall = int(round(float(record.kind.rsplit("-", 1)[1])))
            residue_bounds.setdefault(wall, []).append(lower_bound)
    grouped_orders = _grouped_residue_orders(
        energies(center_lambda)
    )
    wall_certificates = tuple(
        FourPointResidueWallCertificate(
            wall=wall,
            record_count=len(bounds),
            minimum_margin_lower_bound=min(bounds),
            maximum_combined_pole_order=grouped_orders[wall],
            maximum_logarithm_power=grouped_orders[wall] - 1,
        )
        for wall, bounds in sorted(residue_bounds.items())
    )
    continuum_minimum = min(continuum_bounds)
    clearance = min(corner.minimum_wall_clearance for corner in corners)
    all_residues_positive = all(
        wall.minimum_margin_lower_bound > 0.0
        for wall in wall_certificates
    )
    return FourPointResidueDomainCertificate(
        ray_coefficients=coefficients,  # type: ignore[arg-type]
        x_interval=x_pair,
        t_interval=t_pair,
        continuum_record_count=len(continuum_bounds),
        continuum_minimum_margin_lower_bound=continuum_minimum,
        residue_walls=wall_certificates,
        minimum_wall_clearance=clearance,
        chamber_record_count=len(center.records),
        chamber_stable=chamber_stable,
        certified=(
            chamber_stable
            and continuum_minimum > 0.0
            and all_residues_positive
            and clearance > 1.0e-10
        ),
    )


@dataclass(frozen=True)
class FourPointDensityComponents:
    continuous: complex
    residues: complex

    @property
    def total(self) -> complex:
        return self.continuous + self.residues


@dataclass(frozen=True)
class FourPointFinitePartCounterterm:
    """One spin-zero radial term integrated by meromorphic continuation."""

    frame: int
    radial_power: float
    coefficient: complex
    collar_radius: float
    fit_residual: float

    @property
    def analytic_contribution(self) -> complex:
        return complex(
            2.0
            * math.pi
            * self.coefficient
            * self.collar_radius**self.radial_power
            / self.radial_power
        )

    def to_json(self) -> dict[str, object]:
        return {
            "frame": self.frame,
            "radial_power": self.radial_power,
            "coefficient": {
                "real": self.coefficient.real,
                "imag": self.coefficient.imag,
            },
            "collar_radius": self.collar_radius,
            "fit_residual": self.fit_residual,
            "analytic_contribution": {
                "real": self.analytic_contribution.real,
                "imag": self.analytic_contribution.imag,
            },
        }


@dataclass(frozen=True)
class FourPointAmplitudeResult:
    outgoing_energies: tuple[complex, complex, complex]
    incoming_energy: complex
    backend: str
    estimates: tuple[complex, ...]
    mean: complex
    standard_error_real: float
    standard_error_imag: float
    samples_per_replicate: int
    replicates: int
    radial_power: float
    convergence_margin: float
    finite_part_counterterms: tuple[FourPointFinitePartCounterterm, ...] = ()

    def to_json(self) -> dict[str, object]:
        result: dict[str, object] = {
            "outgoing_energies": [
                {"real": value.real, "imag": value.imag}
                for value in self.outgoing_energies
            ],
            "incoming_energy": {
                "real": self.incoming_energy.real,
                "imag": self.incoming_energy.imag,
            },
            "backend": self.backend,
            "estimates": [
                {"real": value.real, "imag": value.imag}
                for value in self.estimates
            ],
            "mean": {"real": self.mean.real, "imag": self.mean.imag},
            "standard_error_real": self.standard_error_real,
            "standard_error_imag": self.standard_error_imag,
            "samples_per_replicate": self.samples_per_replicate,
            "replicates": self.replicates,
            "radial_power": self.radial_power,
            "convergence_margin": self.convergence_margin,
        }
        if self.finite_part_counterterms:
            result["finite_part_counterterms"] = [
                counterterm.to_json()
                for counterterm in self.finite_part_counterterms
            ]
        return result


@dataclass(frozen=True)
class FourPointContinuedAmplitudeResult:
    """Continuum/residue ledger for an analytically continued amplitude."""

    outgoing_energies: tuple[complex, complex, complex]
    incoming_energy: complex
    backend: str
    continuous_estimates: tuple[complex, ...]
    residue_estimates: tuple[complex, ...]
    radial_power: float
    convergence_margin: float
    samples_per_replicate: int

    @property
    def estimates(self) -> tuple[complex, ...]:
        return tuple(
            continuous + residues
            for continuous, residues in zip(
                self.continuous_estimates, self.residue_estimates
            )
        )

    @staticmethod
    def _mean(values: tuple[complex, ...]) -> complex:
        return complex(np.mean(np.asarray(values, dtype=complex)))

    @staticmethod
    def _standard_errors(values: tuple[complex, ...]) -> tuple[float, float]:
        array = np.asarray(values, dtype=complex)
        count = len(values)
        return (
            float(np.std(array.real, ddof=1) / math.sqrt(count)),
            float(np.std(array.imag, ddof=1) / math.sqrt(count)),
        )

    @property
    def mean(self) -> complex:
        return self._mean(self.estimates)

    @property
    def continuous_mean(self) -> complex:
        return self._mean(self.continuous_estimates)

    @property
    def residue_mean(self) -> complex:
        return self._mean(self.residue_estimates)

    @property
    def standard_errors(self) -> tuple[float, float]:
        return self._standard_errors(self.estimates)

    def to_json(self) -> dict[str, object]:
        def encode(value: complex) -> dict[str, float]:
            return {"real": value.real, "imag": value.imag}

        total_errors = self.standard_errors
        continuous_errors = self._standard_errors(self.continuous_estimates)
        residue_errors = self._standard_errors(self.residue_estimates)
        return {
            "outgoing_energies": [
                encode(value) for value in self.outgoing_energies
            ],
            "incoming_energy": encode(self.incoming_energy),
            "backend": self.backend,
            "estimates": [encode(value) for value in self.estimates],
            "mean": encode(self.mean),
            "standard_error_real": total_errors[0],
            "standard_error_imag": total_errors[1],
            "continuous": {
                "estimates": [
                    encode(value) for value in self.continuous_estimates
                ],
                "mean": encode(self.continuous_mean),
                "standard_error_real": continuous_errors[0],
                "standard_error_imag": continuous_errors[1],
            },
            "residues": {
                "estimates": [
                    encode(value) for value in self.residue_estimates
                ],
                "mean": encode(self.residue_mean),
                "standard_error_real": residue_errors[0],
                "standard_error_imag": residue_errors[1],
            },
            "samples_per_replicate": self.samples_per_replicate,
            "replicates": len(self.estimates),
            "radial_power": self.radial_power,
            "convergence_margin": self.convergence_margin,
        }


@lru_cache(maxsize=None)
def _legendre_interval(order: int, upper: float) -> tuple[tuple[float, float], ...]:
    if order < 2 or upper <= 0.0 or not math.isfinite(upper):
        raise ValueError("invalid momentum quadrature")
    nodes, weights = np.polynomial.legendre.leggauss(int(order))
    return tuple(
        (float(0.5 * upper * (node + 1.0)), float(0.5 * upper * weight))
        for node, weight in zip(nodes, weights)
    )


@lru_cache(maxsize=1)
def wall_one_momentum_rule() -> tuple[tuple[float, float], ...]:
    """Thirty-node composite Gauss rule for the wall-1-only chamber."""

    result: list[tuple[float, float]] = []
    for lower, upper, order in WALL_ONE_MOMENTUM_INTERVALS:
        nodes, weights = np.polynomial.legendre.leggauss(order)
        result.extend(
            (
                float(lower + 0.5 * (upper - lower) * (node + 1.0)),
                float(0.5 * (upper - lower) * weight),
            )
            for node, weight in zip(nodes, weights)
        )
    if len(result) != 30:
        raise ArithmeticError("the wall-one momentum rule must contain 30 nodes")
    return tuple(result)


class Type0BSphereFourPointHybrid:
    """Continued Type-0B four-tachyon density in a six-frame block atlas."""

    def __init__(
        self,
        *,
        outgoing_energies: Sequence[Number],
        block_backend: BlockBackend = "hybrid",
        contour_prescription: ContourPrescription = "continued",
        hybrid_corner_radius: float = 0.15,
        hybrid_elliptic_nome_threshold: float = 0.3,
        recursion_max_twice_level: int = 6,
        momentum_order: int = 6,
        momentum_maximum: float = 5.0,
        momentum_rule: MomentumRule = "global",
        structure_precision: int = 24,
        central_charge_shift: float = 1.0e-5,
        block_working_precision: int = 50,
        pole_tolerance: float = 1.0e-28,
        allow_finite_part: bool = False,
    ) -> None:
        if len(outgoing_energies) != 3:
            raise ValueError("outgoing_energies must contain three values")
        self.outgoing_energies = tuple(
            _finite_complex(f"outgoing_energies[{index}]", value)
            for index, value in enumerate(outgoing_energies)
        )
        self.incoming_energy = sum(self.outgoing_energies)
        # Label order in the fixed (infinity,1,z,0) gauge.
        self.external_momenta = (
            self.incoming_energy,
            self.outgoing_energies[2],
            self.outgoing_energies[1],
            self.outgoing_energies[0],
        )
        self.signed_energies = (
            self.incoming_energy,
            -self.outgoing_energies[2],
            -self.outgoing_energies[1],
            -self.outgoing_energies[0],
        )
        if block_backend not in ("hybrid", "h", "c"):
            raise ValueError("block_backend must be 'hybrid', 'h', or 'c'")
        if contour_prescription not in ("fixed", "continued"):
            raise ValueError("contour_prescription must be 'fixed' or 'continued'")
        if not 0.0 < hybrid_corner_radius < 1.0:
            raise ValueError("hybrid_corner_radius must lie in (0,1)")
        if not 0.0 < hybrid_elliptic_nome_threshold < 1.0:
            raise ValueError("hybrid_elliptic_nome_threshold must lie in (0,1)")
        if recursion_max_twice_level < 0:
            raise ValueError("recursion_max_twice_level must be non-negative")
        if momentum_order < 2 or momentum_maximum <= 0.0:
            raise ValueError("invalid momentum quadrature")
        if momentum_rule not in ("global", "wall-one-30"):
            raise ValueError("momentum_rule must be 'global' or 'wall-one-30'")
        if momentum_rule == "wall-one-30" and (
            momentum_order != 30 or abs(momentum_maximum - 3.0) > 1.0e-14
        ):
            raise ValueError(
                "wall-one-30 requires momentum_order=30 and momentum_maximum=3"
            )
        self.block_backend = block_backend
        self.contour_prescription = contour_prescription
        self.hybrid_corner_radius = float(hybrid_corner_radius)
        self.hybrid_elliptic_nome_threshold = float(
            hybrid_elliptic_nome_threshold
        )
        self.recursion_max_twice_level = int(recursion_max_twice_level)
        self.momentum_order = int(momentum_order)
        self.momentum_maximum = float(momentum_maximum)
        self.momentum_rule = momentum_rule
        self.structure_precision = int(structure_precision)
        self.central_charge_shift = float(central_charge_shift)
        self.block_working_precision = int(block_working_precision)
        self.pole_tolerance = float(pole_tolerance)
        self.allow_finite_part = bool(allow_finite_part)
        # ``allow_finite_part`` only disables the subtraction-free gate.  It
        # never changes the density itself: the caller must supply and audit
        # every boundary counterterm explicitly.  This is needed both for the
        # continued-contour finite-part experiments below and for BRY's
        # fixed-contour polynomial subtraction benchmark.
        self._block_cache: dict[tuple[object, ...], object] = {}
        self._elliptic_coefficient_cache: dict[
            tuple[object, ...], tuple[object, ...]
        ] = {}
        self._structure_cache: dict[tuple[object, ...], complex] = {}
        self._laurent_cache: dict[tuple[object, ...], tuple[complex, ...]] = {}
        self.audit = audit_four_point_convergence(
            self.outgoing_energies,
            include_residues=(self.contour_prescription == "continued"),
        )
        if not self.audit.convergent and not self.allow_finite_part:
            raise ValueError(
                "the selected omega point is not subtraction-free: "
                f"minimum corner margin={self.audit.minimum_margin:.6g}"
            )

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

    @staticmethod
    def fixed_positions(z: Number) -> tuple[ProjectivePoint, ...]:
        value = _finite_complex("z", z)
        if value in (0.0, 1.0):
            raise ValueError("z must avoid 0 and 1")
        return (None, 1.0 + 0.0j, value, 0.0 + 0.0j)

    def _selected_backend(self, channel: FourPointChannel, region: BlockRegion) -> str:
        if region not in ("auto", "bulk", "corner"):
            raise ValueError("block_region must be 'auto', 'bulk', or 'corner'")
        if self.block_backend != "hybrid":
            return self.block_backend
        if region != "auto":
            return "c" if region == "corner" else "h"
        # The verified h-recursion component has the two picture-zero fields
        # in the middle slots.  Outside that routing, or once its elliptic nome
        # leaves the declared |q_ell| patch, use the local c-recursion chart.
        middle_labels = frozenset(channel.ordering[1:3])
        nome_magnitude = abs(elliptic_nome(channel.q))
        if (
            middle_labels == frozenset(PICTURE_ZERO_LABELS)
            and nome_magnitude < self.hybrid_elliptic_nome_threshold
        ):
            return "h"
        return "c"

    def _structure_constant(
        self, first: Number, second: Number, third: Number, sector: int
    ) -> complex:
        values = tuple(
            sorted(
                (complex(first), complex(second), complex(third)),
                key=lambda value: (value.real, value.imag),
            )
        )
        key = (int(sector), values)
        if key not in self._structure_cache:
            function = ns_structure_constant if sector == 0 else ns_tilde_structure_constant
            self._structure_cache[key] = function(*values, self.structure_precision)
        return self._structure_cache[key]

    def _structure_laurent_coefficients(
        self,
        first: complex,
        second: complex,
        pole: CrossedNSStructurePole,
        sector: int,
    ) -> tuple[complex, ...]:
        order = int(round(pole.wall))
        if order < 1 or order > 4:
            raise ValueError("crossed b=1 pole order exceeds the supported chamber")
        key = (int(sector), first, second, pole.momentum, order)
        if key in self._laurent_cache:
            return self._laurent_cache[key]
        function = ns_structure_constant_mp if sector == 0 else ns_tilde_structure_constant_mp
        point_count = max(24, 8 * order)
        with mpmath.workdps(max(60, self.structure_precision + 35)):
            radius = mpmath.mpf("0.001")
            coefficients = [mpmath.mpc(0) for _ in range(order)]
            for index in range(point_count):
                phase = mpmath.e ** (2j * mpmath.pi * index / point_count)
                offset = radius * phase
                value = function(
                    mpmath.mpc(first),
                    mpmath.mpc(second),
                    mpmath.mpc(pole.momentum) + offset,
                )
                for negative_order in range(1, order + 1):
                    coefficients[negative_order - 1] += (
                        value * offset**negative_order / point_count
                    )
            result = tuple(complex(value) for value in coefficients)
        self._laurent_cache[key] = result
        return result

    def _structure_product_laurent_coefficients(
        self,
        ordered_external: tuple[complex, complex, complex, complex],
        pole: complex,
        sector: int,
        order: int,
    ) -> tuple[complex, ...]:
        r"""Return negative Laurent coefficients of both trinion factors.

        Energy conservation makes a sum pole of one trinion coincide with a
        reflected-difference pole of the other.  Extracting the product at
        once is essential: treating the two labels as separate residues
        double-counts the contour and misses the regular Laurent terms in
        the residue of the resulting double pole.
        """

        if order < 1 or order > 2:
            raise ValueError("the production chamber supports product pole order two")
        key = ("product", int(sector), ordered_external, complex(pole), int(order))
        if key in self._laurent_cache:
            return self._laurent_cache[key]
        function = ns_structure_constant_mp if sector == 0 else ns_tilde_structure_constant_mp
        pa, pb, pc, pd = ordered_external
        point_count = 32
        with mpmath.workdps(max(65, self.structure_precision + 40)):
            radius = mpmath.mpf("0.001")
            coefficients = [mpmath.mpc(0) for _ in range(order)]
            for index in range(point_count):
                phase = mpmath.e ** (2j * mpmath.pi * index / point_count)
                offset = radius * phase
                momentum = mpmath.mpc(pole) + offset
                value = function(
                    mpmath.mpc(pa), mpmath.mpc(pb), momentum
                ) * function(momentum, mpmath.mpc(pc), mpmath.mpc(pd))
                for negative_order in range(1, order + 1):
                    coefficients[negative_order - 1] += (
                        value * offset**negative_order / point_count
                    )
            result = tuple(complex(value) for value in coefficients)
        self._laurent_cache[key] = result
        return result

    @staticmethod
    def _fermion_pair(
        positions: Sequence[ProjectivePoint], left: int, right: int
    ) -> complex:
        first, second = positions[left], positions[right]
        if first is None:
            return -1.0 + 0.0j
        if second is None:
            return 1.0 + 0.0j
        return -1.0 / (complex(first) - complex(second))

    def _pco_terms(
        self,
        positions: Sequence[ProjectivePoint],
        operator_order: Sequence[int],
    ) -> tuple[tuple[tuple[int, int, int, int], complex], ...]:
        rank = {label: index for index, label in enumerate(operator_order)}
        raised = tuple(sorted(PICTURE_ZERO_LABELS, key=rank.__getitem__))
        # The all-Liouville-superdescendant term.
        descendants = tuple(int(label in PICTURE_ZERO_LABELS) for label in range(4))
        # The term with the two timelike fermions.
        coefficient = math.prod(-self.signed_energies[label] for label in raised)
        coefficient *= self._fermion_pair(positions, raised[0], raised[1])
        return (
            (descendants, 1.0 + 0.0j),
            ((0, 0, 0, 0), complex(coefficient)),
        )

    @staticmethod
    def _spin_local_scales(
        channel: FourPointChannel,
        positions: Sequence[ProjectivePoint],
    ) -> tuple[complex, ...]:
        transform = channel.mobius
        root_determinant = cmath.sqrt(complex(transform.determinant))
        result: list[complex] = []
        for ordered_index, label in enumerate(channel.ordering):
            source = positions[label]
            target_is_infinity = ordered_index == 3
            if source is None:
                root = (
                    root_determinant / complex(transform.a)
                    if target_is_infinity
                    else 1.0j * root_determinant / complex(transform.c)
                )
            else:
                z_value = complex(source)
                root = (
                    1.0j
                    * root_determinant
                    / (complex(transform.a) * z_value + complex(transform.b))
                    if target_is_infinity
                    else root_determinant
                    / (complex(transform.c) * z_value + complex(transform.d))
                )
            expected = complex(channel.local_scales[ordered_index])
            if abs(root * root - expected) > 2.0e-10 * max(1.0, abs(expected)):
                raise ArithmeticError("incoherent Mobius spin lift")
            result.append(complex(root))
        return tuple(result)

    def _component_covariance(
        self,
        channel: FourPointChannel,
        positions: Sequence[ProjectivePoint],
        ordered_weights: Sequence[complex],
        ordered_descendants: Sequence[int],
        *,
        antiholomorphic: bool,
    ) -> complex:
        logarithm = 0.0 + 0.0j
        spin_factor = 1.0 + 0.0j
        spin_scales = self._spin_local_scales(channel, positions)
        for scale, spin, weight, descendant in zip(
            channel.local_scales,
            spin_scales,
            ordered_weights,
            ordered_descendants,
        ):
            log_scale = cmath.log(complex(scale))
            logarithm += complex(weight) * (
                log_scale.conjugate() if antiholomorphic else log_scale
            )
            if descendant:
                spin_factor *= spin.conjugate() if antiholomorphic else spin
        return complex(mpmath.exp(mpmath.mpc(logarithm)) * spin_factor)

    def _chiral_block(
        self,
        *,
        channel: FourPointChannel,
        positions: Sequence[ProjectivePoint],
        internal_momentum: complex,
        sectors: tuple[int, int],
        descendants_by_label: Sequence[int],
        antiholomorphic: bool,
        block_region: BlockRegion,
    ) -> complex:
        ordering = channel.ordering
        ordered_weights = tuple(self.external_weights[label] for label in ordering)
        ordered_descendants = tuple(int(descendants_by_label[label]) for label in ordering)
        internal_weight = self.block_weight(internal_momentum)
        backend = self._selected_backend(channel, block_region)
        key = (
            backend,
            ordered_weights,
            internal_momentum,
            sectors,
            ordered_descendants,
        )
        if key not in self._block_cache:
            block_type = NSSphereLinearHRecursion if backend == "h" else NSSphereLinearCRecursion
            self._block_cache[key] = block_type(
                central_charge=self.block_central_charge,
                external_weights=ordered_weights,
                external_descendants=ordered_descendants,
                internal_weights=(internal_weight,),
                vertex_sectors=sectors,
                working_precision=self.block_working_precision,
                pole_tolerance=self.pole_tolerance,
            )
        block = self._block_cache[key]
        q_log = cmath.log(channel.q)
        argument_log = q_log.conjugate() if antiholomorphic else q_log
        argument = channel.q.conjugate() if antiholomorphic else channel.q
        with mpmath.workdps(self.block_working_precision):
            if key not in self._elliptic_coefficient_cache:
                parity = block.compatible_level_parities()[0]
                effective_weights = tuple(
                    weight + mpmath.mpf(descendant) / 2
                    for weight, descendant in zip(
                        ordered_weights, ordered_descendants
                    )
                )
                coefficient_count = self.recursion_max_twice_level // 2 + 1
                maximum_power = coefficient_count - 1
                theta3_series, ratio, z_series = (
                    NSSphereFourPointBlock._elliptic_series_data(maximum_power)
                )
                q_squared = mpmath.mpc(
                    self.block_central_charge / 3.0 - 0.5
                )
                alpha = internal_weight - q_squared / 8
                beta = (
                    q_squared / 8
                    - effective_weights[1]
                    - effective_weights[2]
                )
                gamma = 1.5 * q_squared - 4 * sum(effective_weights)
                one_minus_z = [-value for value in z_series]
                one_minus_z[0] += 1
                common = _series_mul(
                    _series_mul(
                        _series_pow(ratio, alpha, maximum_power),
                        _series_pow(one_minus_z, -beta, maximum_power),
                        maximum_power,
                    ),
                    _series_pow(theta3_series, -gamma, maximum_power),
                    maximum_power,
                )
                z_coefficients = [
                    block.coefficient(
                        (2 * level if parity == 0 else 2 * level + 1,)
                    )
                    for level in range(coefficient_count)
                ]
                composed = _series_compose(
                    z_coefficients, z_series, maximum_power
                )
                if parity == 0:
                    nome_coefficients = _series_mul(
                        common, composed, maximum_power
                    )
                else:
                    nome_coefficients = _series_mul(
                        _series_mul(
                            common,
                            _series_pow(ratio, 0.5, maximum_power),
                            maximum_power,
                        ),
                        composed,
                        maximum_power,
                    )
                self._elliptic_coefficient_cache[key] = (
                    parity,
                    effective_weights,
                    q_squared,
                    alpha,
                    gamma,
                    tuple(nome_coefficients),
                )
            (
                parity,
                effective_weights,
                q_squared,
                alpha,
                gamma,
                nome_coefficients,
            ) = self._elliptic_coefficient_cache[key]
            nome = elliptic_nome(argument)
            if parity == 0:
                elliptic_part = sum(
                    coefficient * nome**power
                    for power, coefficient in enumerate(nome_coefficients)
                )
            else:
                elliptic_part = sum(
                    4 * coefficient * nome ** (power + 0.5)
                    for power, coefficient in enumerate(nome_coefficients)
                )
            theta3 = mpmath.jtheta(3, 0, mpmath.mpc(nome))
            one_minus_log = cmath.log(1.0 - channel.q)
            if antiholomorphic:
                one_minus_log = one_minus_log.conjugate()
            full_block = (
                mpmath.exp(mpmath.mpc(cmath.log(16.0 * nome)) * alpha)
                * mpmath.exp(
                    mpmath.mpc(argument_log)
                    * (q_squared / 8 - effective_weights[0] - effective_weights[1])
                )
                * mpmath.exp(
                    mpmath.mpc(one_minus_log)
                    * (q_squared / 8 - effective_weights[1] - effective_weights[2])
                )
                * theta3**gamma
                * elliptic_part
            )
            scalar_phase = -1 if parity else 1
            endpoint_phase = (
                STANDARD_ZERO_DESCENDANT_PHASE ** ordered_descendants[0]
                * (
                    STANDARD_INFINITY_DESCENDANT_PHASE.conjugate()
                    if antiholomorphic
                    else STANDARD_INFINITY_DESCENDANT_PHASE
                )
                ** ordered_descendants[-1]
            )
            return complex(
                scalar_phase
                * endpoint_phase
                * self._component_covariance(
                    channel,
                    positions,
                    ordered_weights,
                    ordered_descendants,
                    antiholomorphic=antiholomorphic,
                )
                * full_block
            )

    @staticmethod
    def _timelike_boson_factor(
        positions: Sequence[ProjectivePoint], signed_energies: Sequence[complex]
    ) -> complex:
        logarithm = 0.0 + 0.0j
        for left in range(4):
            if positions[left] is None:
                continue
            for right in range(left + 1, 4):
                if positions[right] is None:
                    continue
                separation = abs(complex(positions[left]) - complex(positions[right]))
                logarithm -= (
                    2.0
                    * signed_energies[left]
                    * signed_energies[right]
                    * math.log(separation)
                )
        return complex(mpmath.exp(mpmath.mpc(logarithm)))

    def _sector_component_kernel(
        self,
        positions: Sequence[ProjectivePoint],
        internal_momentum: complex,
        sectors: tuple[int, int],
        channel: FourPointChannel,
        *,
        block_region: BlockRegion,
    ) -> complex:
        holomorphic_terms = self._pco_terms(positions, channel.ordering)
        antiholomorphic_positions = tuple(
            None if value is None else complex(value).conjugate()
            for value in positions
        )
        antiholomorphic_terms = self._pco_terms(
            antiholomorphic_positions, channel.ordering
        )
        holomorphic = sum(
            coefficient
            * self._chiral_block(
                channel=channel,
                positions=positions,
                internal_momentum=internal_momentum,
                sectors=sectors,
                descendants_by_label=descendants,
                antiholomorphic=False,
                block_region=block_region,
            )
            for descendants, coefficient in holomorphic_terms
        )
        antiholomorphic = sum(
            coefficient
            * self._chiral_block(
                channel=channel,
                positions=positions,
                internal_momentum=internal_momentum,
                sectors=sectors,
                descendants_by_label=descendants,
                antiholomorphic=True,
                block_region=block_region,
            )
            for descendants, coefficient in antiholomorphic_terms
        )
        return complex(
            self._timelike_boson_factor(positions, self.signed_energies)
            * holomorphic
            * antiholomorphic
        )

    def fixed_momentum_density(
        self,
        z: Number,
        internal_momentum: Number,
        *,
        channel: FourPointChannel | None = None,
        block_region: BlockRegion = "auto",
    ) -> complex:
        """Return the uncontinued ``dP`` density, including ``1/pi``."""

        positions = self.fixed_positions(z)
        active = best_four_point_channel(positions) if channel is None else channel
        momentum = _finite_complex("internal_momentum", internal_momentum)
        ordered = tuple(self.external_momenta[label] for label in active.ordering)
        total = 0.0 + 0.0j
        for sectors in SECTOR_ASSIGNMENTS:
            sector = sectors[0]
            total += (
                self._structure_constant(ordered[0], ordered[1], momentum, sector)
                * self._structure_constant(momentum, ordered[2], ordered[3], sector)
                * self._sector_component_kernel(
                    positions,
                    momentum,
                    sectors,
                    active,
                    block_region=block_region,
                )
            )
        return complex(total / math.pi)

    @staticmethod
    def _first_derivative(function, point: complex) -> complex:
        coarse = (function(point + 2.0e-4) - function(point - 2.0e-4)) / 4.0e-4
        fine = (function(point + 1.0e-4) - function(point - 1.0e-4)) / 2.0e-4
        return complex((4.0 * fine - coarse) / 3.0)

    def _residue_density(
        self,
        positions: Sequence[ProjectivePoint],
        channel: FourPointChannel,
        *,
        block_region: BlockRegion,
    ) -> complex:
        ordered = tuple(self.external_momenta[label] for label in channel.ordering)
        pa, pb, pc, pd = ordered
        total = 0.0 + 0.0j
        for sectors in SECTOR_ASSIGNMENTS:
            sector = sectors[0]
            left_poles = _positive_contour_structure_poles(pa, pb, sector)
            right_poles = _positive_contour_structure_poles(pc, pd, sector)
            groups: list[list[CrossedNSStructurePole]] = []
            for candidate in (*left_poles, *right_poles):
                for group in groups:
                    if abs(candidate.momentum - group[0].momentum) < 1.0e-9:
                        group.append(candidate)
                        break
                else:
                    groups.append([candidate])
            for group in groups:
                pole = group[0]
                if any(
                    abs(item.contour_coefficient - pole.contour_coefficient) > 1.0e-12
                    for item in group[1:]
                ):
                    raise ValueError("coincident poles cross the quotient contour oppositely")
                order = sum(int(round(item.wall)) for item in group)
                coefficients = self._structure_product_laurent_coefficients(
                    ordered, pole.momentum, sector, order
                )

                def regular(momentum: complex) -> complex:
                    return self._sector_component_kernel(
                        positions,
                        momentum,
                        sectors,
                        channel,
                        block_region=block_region,
                    )

                residue = coefficients[0] * regular(pole.momentum)
                if len(coefficients) >= 2:
                    residue += coefficients[1] * self._first_derivative(
                        regular, pole.momentum
                    )
                total += pole.contour_coefficient * residue
        # ``contour_coefficient`` is already the quotient-contour factor
        # ``(+/- 2*pi*i)/pi = +/- 2*i`` for the ``dP/pi`` measure.  Dividing
        # by pi again would undercount every crossed-pole contribution and
        # violates channel crossing after analytic continuation.
        return complex(total)

    def density_components(
        self,
        z: Number,
        *,
        channel: FourPointChannel | None = None,
        block_region: BlockRegion = "auto",
    ) -> FourPointDensityComponents:
        positions = self.fixed_positions(z)
        active_channel = (
            best_four_point_channel(positions) if channel is None else channel
        )
        continuous = 0.0 + 0.0j
        momentum_nodes = (
            wall_one_momentum_rule()
            if self.momentum_rule == "wall-one-30"
            else _legendre_interval(
                self.momentum_order, self.momentum_maximum
            )
        )
        for momentum, weight in momentum_nodes:
            continuous += weight * self.fixed_momentum_density(
                z,
                momentum,
                channel=active_channel,
                block_region=block_region,
            )
        residues = (
            self._residue_density(
                positions, active_channel, block_region=block_region
            )
            if self.contour_prescription == "continued"
            else 0.0 + 0.0j
        )
        return FourPointDensityComponents(
            continuous=complex(continuous), residues=residues
        )

    def density(
        self,
        z: Number,
        *,
        channel: FourPointChannel | None = None,
        block_region: BlockRegion = "auto",
    ) -> complex:
        return self.density_components(
            z, channel=channel, block_region=block_region
        ).total


class Type0BFixedContourFourPointElliptic:
    r"""Fixed-contour four-point kernel in the original BRY gauge.

    Plane-series coefficients come from h-recursion in the bulk and
    c-recursion in the three collars.  Both are converted to the elliptic
    nome before evaluation, which is the efficient four-point specialization
    of the general plumbing atlas.
    """

    def __init__(
        self,
        *,
        outgoing_energies: Sequence[Number],
        block_backend: BlockBackend = "hybrid",
        hybrid_corner_radius: float = 0.15,
        block_order: int = 8,
        momentum_order: int = 10,
        momentum_maximum: float = 6.0,
        structure_precision: int = 24,
        central_charge_shift: float = 1.0e-5,
        block_working_precision: int = 50,
    ) -> None:
        if len(outgoing_energies) != 3:
            raise ValueError("outgoing_energies must contain three values")
        self.outgoing_energies = tuple(
            _finite_complex(f"outgoing_energies[{index}]", value)
            for index, value in enumerate(outgoing_energies)
        )
        self.incoming_energy = sum(self.outgoing_energies)
        self.audit = audit_four_point_convergence(
            self.outgoing_energies, include_residues=False
        )
        if not self.audit.convergent:
            raise ValueError(
                "the fixed-contour moduli integral is not convergent: "
                f"minimum OPE margin={self.audit.minimum_margin:.6g}"
            )
        self.momentum_order = int(momentum_order)
        self.momentum_maximum = float(momentum_maximum)
        self.sphere = BRYFourTachyonSphere(
            omega=self.incoming_energy,
            omega1=self.outgoing_energies[0],
            omega2=self.outgoing_energies[1],
            omega3=self.outgoing_energies[2],
            block_order=int(block_order),
            structure_precision=int(structure_precision),
            central_charge_shift=float(central_charge_shift),
            block_working_precision=int(block_working_precision),
            block_backend=block_backend,
            hybrid_corner_radius=float(hybrid_corner_radius),
        )

    @property
    def block_backend(self) -> str:
        return self.sphere.liouville.block_backend

    @block_backend.setter
    def block_backend(self, value: BlockBackend) -> None:
        if value not in ("hybrid", "h", "c"):
            raise ValueError("block_backend must be 'hybrid', 'h', or 'c'")
        self.sphere.liouville.block_backend = value

    def density(self, z: Number) -> complex:
        return self.sphere.reduced_integrand(
            z,
            p_max=self.momentum_maximum,
            quadrature_order=self.momentum_order,
        )


def canonical_chart_channel(
    positions: Sequence[ProjectivePoint], chart: int
) -> FourPointChannel:
    """Return the canonical channel for ``z``, ``1-z``, or ``1/z``."""

    orderings = (
        (3, 2, 1, 0),  # q=z
        (2, 1, 0, 3),  # q=1-z
        (0, 2, 1, 3),  # q=1/z
    )
    selected = int(chart)
    if selected not in (0, 1, 2):
        raise ValueError("chart must be 0, 1, or 2")
    channel = four_point_channel_from_ordering(positions, orderings[selected])
    if channel.score >= 1.0 + 1.0e-12:
        raise ValueError("the selected canonical chart lies outside the unit disc")
    return channel


def _logsumexp(values: Sequence[float]) -> float:
    largest = max(values)
    return largest + math.log(sum(math.exp(value - largest) for value in values))


def _power_disk_sample(first: float, second: float, radial_power: float) -> complex:
    value = min(
        max(float(first), np.nextafter(0.0, 1.0)),
        np.nextafter(1.0, 0.0),
    )
    radius = math.exp(math.log(value) / radial_power)
    angle = 2.0 * math.pi * float(second)
    return complex(radius * cmath.exp(1.0j * angle))


def three_chart_log_mixture_density(z: complex, radial_power: float) -> float:
    z = complex(z)
    base = math.log(radial_power / (2.0 * math.pi))
    terms: list[float] = []
    if 0.0 < abs(z) < 1.0:
        terms.append(base + (radial_power - 2.0) * math.log(abs(z)))
    if 0.0 < abs(1.0 - z) < 1.0:
        terms.append(base + (radial_power - 2.0) * math.log(abs(1.0 - z)))
    if abs(z) > 1.0:
        terms.append(base + (-radial_power - 2.0) * math.log(abs(z)))
    if not terms:
        raise ArithmeticError("the three-chart proposal failed to cover z")
    return _logsumexp(terms) - math.log(3.0)


def integrate_subtraction_free_four_point(
    kernel: Type0BSphereFourPointHybrid,
    *,
    sobol_power: int = 6,
    replicates: int = 4,
    radial_power: float | None = None,
    seed: int = 20260827,
) -> FourPointAmplitudeResult:
    """Integrate the continued density over the three-chart ``M_0,4`` atlas."""

    if sobol_power < 1 or replicates < 2:
        raise ValueError("sobol_power must be positive and replicates at least two")
    if radial_power is None:
        radial_power = min(1.0, 0.9 * kernel.audit.minimum_margin)
    radial_power = float(radial_power)
    if not 0.0 < radial_power <= 2.0:
        raise ValueError("radial_power must lie in (0,2]")
    estimates: list[complex] = []
    for replicate in range(replicates):
        sampler = qmc.Sobol(d=3, scramble=True, seed=seed + replicate)
        samples: list[complex] = []
        for point in sampler.random_base2(sobol_power):
            local = _power_disk_sample(point[0], point[1], radial_power)
            chart = min(int(point[2] * 3), 2)
            z = local if chart == 0 else (1.0 - local if chart == 1 else 1.0 / local)
            proposal = three_chart_log_mixture_density(z, radial_power)
            positions = kernel.fixed_positions(z)
            channel = canonical_chart_channel(positions, chart)
            samples.append(
                kernel.density(z, channel=channel) * math.exp(-proposal)
            )
        estimates.append(complex(np.mean(np.asarray(samples, dtype=complex))))
    values = np.asarray(estimates, dtype=complex)
    return FourPointAmplitudeResult(
        outgoing_energies=kernel.outgoing_energies,  # type: ignore[arg-type]
        incoming_energy=kernel.incoming_energy,
        backend=kernel.block_backend,
        estimates=tuple(estimates),
        mean=complex(np.mean(values)),
        standard_error_real=float(np.std(values.real, ddof=1) / math.sqrt(replicates)),
        standard_error_imag=float(np.std(values.imag, ddof=1) / math.sqrt(replicates)),
        samples_per_replicate=2**sobol_power,
        replicates=replicates,
        radial_power=radial_power,
        convergence_margin=kernel.audit.minimum_margin,
    )


def integrate_subtraction_free_four_point_component_stratified_qmc(
    kernel: Type0BSphereFourPointHybrid,
    *,
    sobol_power: int = 6,
    replicates: int = 4,
    radial_power: float | None = None,
    seed: int = 20260827,
) -> FourPointContinuedAmplitudeResult:
    r"""Stratify the valid three-chart mixture and retain its residue ledger.

    The charts ``z=w``, ``z=1-w``, and ``z=1/w`` use frames 0, 1, and 2.
    Each chart receives its own scrambled Sobol net and the three chart means
    are averaged exactly.  The balance-heuristic denominator remains the
    full three-chart mixture, so overlap regions are counted once without a
    randomly sampled chart label.
    """

    sobol_power = int(sobol_power)
    replicates = int(replicates)
    if sobol_power < 1 or replicates < 2:
        raise ValueError("sobol_power must be positive and replicates at least two")
    if radial_power is None:
        radial_power = min(1.0, 0.9 * kernel.audit.minimum_margin)
    radial_power = float(radial_power)
    if not 0.0 < radial_power <= 2.0:
        raise ValueError("radial_power must lie in (0,2]")

    continuous_estimates: list[complex] = []
    residue_estimates: list[complex] = []
    for replicate in range(replicates):
        continuous_chart_means: list[complex] = []
        residue_chart_means: list[complex] = []
        for chart in range(3):
            sampler = qmc.Sobol(
                d=2,
                scramble=True,
                seed=seed + 3 * replicate + chart,
            )
            continuous_samples: list[complex] = []
            residue_samples: list[complex] = []
            for point in sampler.random_base2(sobol_power):
                local = _power_disk_sample(
                    point[0], point[1], radial_power
                )
                z = (
                    local
                    if chart == 0
                    else (1.0 - local if chart == 1 else 1.0 / local)
                )
                proposal = three_chart_log_mixture_density(z, radial_power)
                positions = kernel.fixed_positions(z)
                channel = canonical_chart_channel(positions, chart)
                components = kernel.density_components(z, channel=channel)
                inverse_proposal = math.exp(-proposal)
                continuous_samples.append(
                    components.continuous * inverse_proposal
                )
                residue_samples.append(components.residues * inverse_proposal)
            continuous_chart_means.append(
                complex(np.mean(np.asarray(continuous_samples, dtype=complex)))
            )
            residue_chart_means.append(
                complex(np.mean(np.asarray(residue_samples, dtype=complex)))
            )
        continuous_estimates.append(
            complex(sum(continuous_chart_means) / 3.0)
        )
        residue_estimates.append(complex(sum(residue_chart_means) / 3.0))

    return FourPointContinuedAmplitudeResult(
        outgoing_energies=kernel.outgoing_energies,  # type: ignore[arg-type]
        incoming_energy=kernel.incoming_energy,
        backend=f"{kernel.block_backend}:continued-stratified-three-chart",
        continuous_estimates=tuple(continuous_estimates),
        residue_estimates=tuple(residue_estimates),
        radial_power=radial_power,
        convergence_margin=kernel.audit.minimum_margin,
        samples_per_replicate=3 * 2**sobol_power,
    )


def _crossing_cell_inverse(q_value: complex, frame: int) -> tuple[complex, complex]:
    """Return ``(z(q),dz/dq)`` for one of the six anharmonic frames."""

    q_value = complex(q_value)
    selected = int(frame)
    if selected == 0:
        return q_value, 1.0 + 0.0j
    if selected == 1:
        return 1.0 - q_value, -1.0 + 0.0j
    if selected == 2:
        return 1.0 / q_value, -1.0 / q_value**2
    if selected == 3:
        return 1.0 - 1.0 / q_value, 1.0 / q_value**2
    if selected == 4:
        return q_value / (q_value - 1.0), -1.0 / (q_value - 1.0) ** 2
    if selected == 5:
        return 1.0 / (1.0 - q_value), 1.0 / (1.0 - q_value) ** 2
    raise ValueError("frame must lie in range(6)")


def _crossing_cell_channel(
    positions: Sequence[ProjectivePoint], frame: int
) -> FourPointChannel:
    orderings = (
        (3, 2, 1, 0),
        (2, 1, 0, 3),
        (0, 2, 1, 3),
        (0, 2, 3, 1),
        (0, 1, 3, 2),
        (0, 3, 1, 2),
    )
    selected = int(frame)
    if selected not in range(6):
        raise ValueError("frame must lie in range(6)")
    return four_point_channel_from_ordering(positions, orderings[selected])


def audit_four_point_crossing(
    kernel: Type0BSphereFourPointHybrid,
    z: Number,
    *,
    frames: Sequence[int] = (0, 1),
    block_region: BlockRegion = "corner",
    relative_tolerance: float = 5.0e-3,
) -> FourPointCrossingAudit:
    r"""Compare channel decompositions before any moduli integration.

    The default forces the c-recursive local representation in every frame,
    so this audit tests the continued contour and channel covariance rather
    than the h/c routing decision.  A production integration must not proceed
    when this test fails at its converged numerical settings.
    """

    point = _finite_complex("z", z)
    selected_frames = tuple(int(frame) for frame in frames)
    if len(selected_frames) < 2 or len(set(selected_frames)) != len(selected_frames):
        raise ValueError("frames must contain at least two distinct values")
    if any(frame not in range(6) for frame in selected_frames):
        raise ValueError("crossing frames must lie in range(6)")
    tolerance = float(relative_tolerance)
    if tolerance <= 0.0 or not math.isfinite(tolerance):
        raise ValueError("relative_tolerance must be positive and finite")

    positions = kernel.fixed_positions(point)
    values: list[FourPointCrossingFrameValue] = []
    for frame in selected_frames:
        channel = _crossing_cell_channel(positions, frame)
        if channel.score >= 1.0 + 1.0e-12:
            raise ValueError(
                f"frame {frame} has |q|={channel.score:.6g} outside its unit-disc patch"
            )
        components = kernel.density_components(
            point,
            channel=channel,
            block_region=block_region,
        )
        values.append(
            FourPointCrossingFrameValue(
                frame=frame,
                q=channel.q,
                backend=kernel._selected_backend(channel, block_region),
                continuous=components.continuous,
                residues=components.residues,
            )
        )

    differences = [
        abs(values[left].total - values[right].total)
        for left in range(len(values))
        for right in range(left)
    ]
    maximum_difference = float(max(differences))
    scale = max(abs(value.total) for value in values)
    relative_spread = float(
        maximum_difference / scale if scale > 0.0 else maximum_difference
    )
    return FourPointCrossingAudit(
        z=point,
        values=tuple(values),
        maximum_absolute_difference=maximum_difference,
        relative_spread=relative_spread,
        relative_tolerance=tolerance,
        passed=relative_spread <= tolerance,
    )


def _equal_pure_imaginary_residue_power(
    kernel: Type0BSphereFourPointHybrid,
) -> float:
    r"""Return the radial power of the crossed ``P=i(4t-1)`` term.

    For equal outgoing momenta ``omega_a=i*t`` the raised--raised divisor has

    ``x=(2t)^2-(4t-1)^2-1``.

    The first production chamber ``1/2<t<2/3`` has ``-2<x<0``.  The leading
    spin-zero term therefore needs one radial finite part, while the next
    nonzero diagonal NS level, at ``x+2``, is integrable.
    """

    outgoing = tuple(complex(value) for value in kernel.outgoing_energies)
    if any(abs(value.real) > 1.0e-12 for value in outgoing):
        raise ValueError("the equal-energy finite part requires pure-imaginary omega")
    t_value = outgoing[0].imag
    if any(abs(value.imag - t_value) > 1.0e-12 for value in outgoing[1:]):
        raise ValueError("the finite-part implementation requires equal outgoing omega")
    if not 0.5 < t_value < 2.0 / 3.0:
        raise ValueError("the implemented residue chamber is 1/2<t<2/3")
    q_squared = kernel.block_central_charge / 3.0 - 0.5
    radial_power = (
        (2.0 * t_value) ** 2
        - (4.0 * t_value - 1.0) ** 2
        - q_squared / 4.0
    )
    if not -2.0 < radial_power < 0.0:
        raise ArithmeticError("unexpected number of divergent radial levels")
    return float(radial_power)


def _crossing_frame_residue_density(
    kernel: Type0BSphereFourPointHybrid,
    q_value: complex,
    frame: int,
) -> complex:
    z_value, derivative = _crossing_cell_inverse(q_value, frame)
    positions = kernel.fixed_positions(z_value)
    channel = _crossing_cell_channel(positions, frame)
    return complex(
        kernel.density_components(
            z_value,
            channel=channel,
            block_region="corner",
        ).residues
        * abs(derivative) ** 2
    )


def estimate_equal_imaginary_residue_counterterms(
    kernel: Type0BSphereFourPointHybrid,
    *,
    collar_radius: float = 0.1,
    fit_radii: Sequence[float] = (0.008, 0.006, 0.0045, 0.0032, 0.0024),
    angular_order: int = 12,
    fit_order: int = 2,
) -> tuple[FourPointFinitePartCounterterm, ...]:
    r"""Compute the two crossing-frame coefficients of the divergent residue.

    Frames 1 and 5 are the two oriented representatives of ``z -> 1``.  The
    full PCO component sum removes the naively divergent leading term in the
    other four frames.  The coefficient is obtained directly from the
    Laurent residue of the two structure constants and the leading BRY PCO
    square.  The sampling arguments remain accepted for API compatibility;
    no small-radius extrapolation defines the counterterm.
    """

    if kernel.contour_prescription != "continued" or not kernel.allow_finite_part:
        raise ValueError("counterterm extraction requires an enabled continued finite part")
    radius = float(collar_radius)
    if not 0.0 < radius < 0.75:
        raise ValueError("collar_radius must lie in (0,3/4)")
    _ = tuple(float(value) for value in fit_radii)
    _ = int(angular_order)
    _ = int(fit_order)
    radial_power = _equal_pure_imaginary_residue_power(kernel)
    t_value = kernel.outgoing_energies[0].imag
    pole = 1.0j * (4.0 * t_value - 1.0)
    picture_momenta = tuple(
        kernel.external_momenta[label] for label in PICTURE_ZERO_LABELS
    )
    leading_pco = (
        kernel.block_weight(picture_momenta[0])
        + kernel.block_weight(picture_momenta[1])
        - kernel.block_weight(pole)
        + picture_momenta[0] * picture_momenta[1]
    ) ** 2
    counterterms: list[FourPointFinitePartCounterterm] = []
    for frame in (1, 5):
        sample_z, _ = _crossing_cell_inverse(0.01 + 0.002j, frame)
        channel = _crossing_cell_channel(kernel.fixed_positions(sample_z), frame)
        ordered = tuple(
            kernel.external_momenta[label] for label in channel.ordering
        )
        laurent_residue = kernel._structure_product_laurent_coefficients(
            ordered,
            pole,
            0,
            1,
        )[0]
        coefficient = -2.0j * laurent_residue * leading_pco / math.pi
        counterterms.append(
            FourPointFinitePartCounterterm(
                frame=frame,
                radial_power=radial_power,
                coefficient=complex(coefficient),
                collar_radius=radius,
                fit_residual=0.0,
            )
        )
    scale = max(1.0, abs(counterterms[0].coefficient))
    if abs(counterterms[0].coefficient - counterterms[1].coefficient) > 2.0e-8 * scale:
        raise ArithmeticError("the two z=1 residue frames violate crossing symmetry")
    return tuple(counterterms)


def _finite_part_crossing_cell_estimate(
    kernel: Type0BSphereFourPointHybrid,
    counterterms: Sequence[FourPointFinitePartCounterterm],
    *,
    radial_order: int,
    angular_order: int,
    radial_sampling_power: float,
) -> complex:
    radial_nodes_raw, radial_weights_raw = np.polynomial.legendre.leggauss(
        radial_order
    )
    radial_nodes = 0.5 * (radial_nodes_raw + 1.0)
    radial_weights = 0.5 * radial_weights_raw
    angular_nodes, angular_weights = np.polynomial.legendre.leggauss(angular_order)
    by_frame = {counterterm.frame: counterterm for counterterm in counterterms}
    intervals = ((0.0, math.pi / 3.0), (math.pi / 3.0, math.pi))
    total = 0.0 + 0.0j
    for lower, upper in intervals:
        midpoint = 0.5 * (lower + upper)
        half_width = 0.5 * (upper - lower)
        for angular_node, angular_weight in zip(angular_nodes, angular_weights):
            theta = midpoint + half_width * float(angular_node)
            cosine = math.cos(theta)
            maximum_radius = min(
                1.0,
                0.5 / cosine if cosine > 0.0 else math.inf,
            )
            phase = cmath.exp(1.0j * theta)
            for radial_node, radial_weight in zip(radial_nodes, radial_weights):
                radius = maximum_radius * math.exp(
                    math.log(float(radial_node)) / radial_sampling_power
                )
                q_value = radius * phase
                radial_jacobian = (
                    radius * radius
                    / (radial_sampling_power * float(radial_node))
                )
                frame_sum = 0.0 + 0.0j
                for frame in range(6):
                    z_value, derivative = _crossing_cell_inverse(q_value, frame)
                    positions = kernel.fixed_positions(z_value)
                    channel = _crossing_cell_channel(positions, frame)
                    density = kernel.density(z_value, channel=channel) * abs(derivative) ** 2
                    counterterm = by_frame.get(frame)
                    if counterterm is not None and radius < counterterm.collar_radius:
                        density -= counterterm.coefficient * radius ** (
                            counterterm.radial_power - 2.0
                        )
                    frame_sum += density
                total += (
                    half_width
                    * float(angular_weight)
                    * float(radial_weight)
                    * radial_jacobian
                    * frame_sum
                )
    analytic = sum(
        (counterterm.analytic_contribution for counterterm in counterterms),
        0.0 + 0.0j,
    )
    return complex(2.0 * total.real + analytic)


def integrate_equal_imaginary_continued_finite_part_cells(
    kernel: Type0BSphereFourPointHybrid,
    *,
    radial_order: int = 5,
    angular_order: int = 6,
    ladder_steps: int = 3,
    radial_sampling_power: float = 0.5,
    collar_radius: float = 0.1,
    counterterm_angular_order: int = 12,
) -> FourPointAmplitudeResult:
    """Integrate the equal-imaginary continued amplitude including residues."""

    radial_order = int(radial_order)
    angular_order = int(angular_order)
    ladder_steps = int(ladder_steps)
    radial_sampling_power = float(radial_sampling_power)
    if radial_order < 2 or angular_order < 2 or ladder_steps < 2:
        raise ValueError("orders and ladder_steps must each be at least two")
    if not 0.0 < radial_sampling_power <= 2.0:
        raise ValueError("radial_sampling_power must lie in (0,2]")
    counterterms = estimate_equal_imaginary_residue_counterterms(
        kernel,
        collar_radius=collar_radius,
        angular_order=counterterm_angular_order,
    )
    estimates = tuple(
        _finite_part_crossing_cell_estimate(
            kernel,
            counterterms,
            radial_order=radial_order + step,
            angular_order=angular_order + 2 * step,
            radial_sampling_power=radial_sampling_power,
        )
        for step in range(ladder_steps)
    )
    finest = estimates[-1]
    previous = estimates[-2]
    positive_margins = [
        record.margin for record in kernel.audit.records if record.margin > 0.0
    ]
    remainder_margin = min(
        min(positive_margins),
        counterterms[0].radial_power + 2.0,
    )
    finest_radial = radial_order + ladder_steps - 1
    finest_angular = angular_order + 2 * (ladder_steps - 1)
    return FourPointAmplitudeResult(
        outgoing_energies=kernel.outgoing_energies,  # type: ignore[arg-type]
        incoming_energy=kernel.incoming_energy,
        backend=f"{kernel.block_backend}:continued-finite-part-cells",
        estimates=estimates,
        mean=finest,
        standard_error_real=abs((finest - previous).real),
        standard_error_imag=abs((finest - previous).imag),
        samples_per_replicate=6 * finest_radial * finest_angular * 2,
        replicates=ladder_steps,
        radial_power=radial_sampling_power,
        convergence_margin=float(remainder_margin),
        finite_part_counterterms=counterterms,
    )


def integrate_equal_imaginary_continued_finite_part_qmc(
    kernel: Type0BSphereFourPointHybrid,
    *,
    sobol_power: int = 8,
    replicates: int = 4,
    radial_sampling_power: float = 0.5,
    collar_radius: float = 0.1,
    counterterm_angular_order: int = 12,
    seed: int = 20260827,
) -> FourPointAmplitudeResult:
    r"""Three-chart QMC evaluation of the continued equal-energy finite part.

    Unlike the folded-disk representation, the three-chart atlas contains a
    complete round collar around ``z=1``.  The single local subtraction can
    therefore be restored exactly as ``2*pi*A*rho^x/x``.
    """

    sobol_power = int(sobol_power)
    replicates = int(replicates)
    radial_sampling_power = float(radial_sampling_power)
    if sobol_power < 1 or replicates < 2:
        raise ValueError("sobol_power must be positive and replicates at least two")
    if not 0.0 < radial_sampling_power <= 2.0:
        raise ValueError("radial_sampling_power must lie in (0,2]")
    oriented = estimate_equal_imaginary_residue_counterterms(
        kernel,
        collar_radius=collar_radius,
        angular_order=counterterm_angular_order,
    )
    coefficient = 0.5 * (oriented[0].coefficient + oriented[1].coefficient)
    counterterm = FourPointFinitePartCounterterm(
        frame=1,
        radial_power=oriented[0].radial_power,
        coefficient=coefficient,
        collar_radius=float(collar_radius),
        fit_residual=max(item.fit_residual for item in oriented),
    )
    estimates: list[complex] = []
    for replicate in range(replicates):
        sampler = qmc.Sobol(d=3, scramble=True, seed=int(seed) + replicate)
        samples: list[complex] = []
        for point in sampler.random_base2(sobol_power):
            local = _power_disk_sample(
                point[0], point[1], radial_sampling_power
            )
            chart = min(int(point[2] * 3), 2)
            z_value = (
                local
                if chart == 0
                else (1.0 - local if chart == 1 else 1.0 / local)
            )
            proposal = three_chart_log_mixture_density(
                z_value, radial_sampling_power
            )
            positions = kernel.fixed_positions(z_value)
            channel = canonical_chart_channel(positions, chart)
            density = kernel.density(z_value, channel=channel)
            local_radius = abs(1.0 - z_value)
            if local_radius < counterterm.collar_radius:
                density -= counterterm.coefficient * local_radius ** (
                    counterterm.radial_power - 2.0
                )
            samples.append(density * math.exp(-proposal))
        estimates.append(
            complex(np.mean(np.asarray(samples, dtype=complex)))
            + counterterm.analytic_contribution
        )
    values = np.asarray(estimates, dtype=complex)
    positive_margins = [
        record.margin for record in kernel.audit.records if record.margin > 0.0
    ]
    remainder_margin = min(
        min(positive_margins),
        counterterm.radial_power + 2.0,
    )
    return FourPointAmplitudeResult(
        outgoing_energies=kernel.outgoing_energies,  # type: ignore[arg-type]
        incoming_energy=kernel.incoming_energy,
        backend=f"{kernel.block_backend}:continued-finite-part-qmc",
        estimates=tuple(estimates),
        mean=complex(np.mean(values)),
        standard_error_real=float(
            np.std(values.real, ddof=1) / math.sqrt(replicates)
        ),
        standard_error_imag=float(
            np.std(values.imag, ddof=1) / math.sqrt(replicates)
        ),
        samples_per_replicate=2**sobol_power,
        replicates=replicates,
        radial_power=radial_sampling_power,
        convergence_margin=float(remainder_margin),
        finite_part_counterterms=(counterterm,),
    )


def _finite_part_crossing_cell_collar_estimate(
    kernel: Type0BSphereFourPointHybrid,
    counterterms: Sequence[FourPointFinitePartCounterterm],
    *,
    inner_radial_order: int,
    inner_angular_order: int,
    outer_radial_order: int,
    outer_angular_order: int,
    radial_sampling_power: float,
    angular_shift: float,
) -> complex:
    """Project collar spins before continuing the spin-zero radial term."""

    by_frame = {counterterm.frame: counterterm for counterterm in counterterms}
    collar_radius = min(counterterm.collar_radius for counterterm in counterterms)
    inner_raw, inner_weights_raw = np.polynomial.legendre.leggauss(
        inner_radial_order
    )
    inner_nodes = 0.5 * (inner_raw + 1.0)
    inner_weights = 0.5 * inner_weights_raw
    total = 0.0 + 0.0j

    # For r<rho the crossing cell is a complete disk.  A periodic angular
    # rule annihilates every nonzero Fourier spin before the singular radial
    # limit is sampled.
    for angular_index in range(inner_angular_order):
        theta = 2.0 * math.pi * (
            (angular_index + float(angular_shift)) / inner_angular_order
        )
        phase = cmath.exp(1.0j * theta)
        for radial_node, radial_weight in zip(inner_nodes, inner_weights):
            radius = collar_radius * math.exp(
                math.log(float(radial_node)) / radial_sampling_power
            )
            q_value = radius * phase
            radial_jacobian = (
                radius * radius
                / (radial_sampling_power * float(radial_node))
            )
            frame_sum = 0.0 + 0.0j
            for frame in range(6):
                z_value, derivative = _crossing_cell_inverse(q_value, frame)
                positions = kernel.fixed_positions(z_value)
                channel = _crossing_cell_channel(positions, frame)
                density = kernel.density(
                    z_value, channel=channel, block_region="corner"
                ) * abs(derivative) ** 2
                counterterm = by_frame.get(frame)
                if counterterm is not None:
                    density -= counterterm.coefficient * radius ** (
                        counterterm.radial_power - 2.0
                    )
                frame_sum += density
            total += (
                2.0
                * math.pi
                / inner_angular_order
                * float(radial_weight)
                * radial_jacobian
                * frame_sum
            )

    outer_raw, outer_weights = np.polynomial.legendre.leggauss(
        outer_radial_order
    )
    angular_raw, angular_weights = np.polynomial.legendre.leggauss(
        outer_angular_order
    )
    angular_intervals = (
        (-math.pi / 3.0, math.pi / 3.0),
        (math.pi / 3.0, math.pi),
        (math.pi, 5.0 * math.pi / 3.0),
    )
    for angular_lower, angular_upper in angular_intervals:
        angular_midpoint = 0.5 * (angular_lower + angular_upper)
        angular_scale = 0.5 * (angular_upper - angular_lower)
        for angular_node, angular_weight in zip(angular_raw, angular_weights):
            theta = angular_midpoint + angular_scale * float(angular_node)
            cosine = math.cos(theta)
            maximum_radius = min(
                1.0,
                0.5 / cosine if cosine > 0.0 else math.inf,
            )
            if maximum_radius <= collar_radius:
                continue
            radial_midpoint = 0.5 * (collar_radius + maximum_radius)
            radial_scale = 0.5 * (maximum_radius - collar_radius)
            phase = cmath.exp(1.0j * theta)
            for radial_node, radial_weight in zip(outer_raw, outer_weights):
                radius = radial_midpoint + radial_scale * float(radial_node)
                q_value = radius * phase
                frame_sum = 0.0 + 0.0j
                for frame in range(6):
                    z_value, derivative = _crossing_cell_inverse(q_value, frame)
                    positions = kernel.fixed_positions(z_value)
                    channel = _crossing_cell_channel(positions, frame)
                    frame_sum += (
                        kernel.density(z_value, channel=channel)
                        * abs(derivative) ** 2
                    )
                total += (
                    angular_scale
                    * float(angular_weight)
                    * radial_scale
                    * float(radial_weight)
                    * radius
                    * frame_sum
                )
    return complex(
        total
        + sum(
            (counterterm.analytic_contribution for counterterm in counterterms),
            0.0 + 0.0j,
        )
    )


def integrate_equal_imaginary_continued_finite_part_collars(
    kernel: Type0BSphereFourPointHybrid,
    *,
    inner_radial_order: int = 3,
    inner_angular_order: int = 8,
    outer_radial_order: int = 3,
    outer_angular_order: int = 4,
    ladder_steps: int = 2,
    radial_sampling_power: float = 0.5,
    collar_radius: float = 0.1,
    counterterm_angular_order: int = 12,
) -> FourPointAmplitudeResult:
    """BRY collar finite part with deterministic spin projection."""

    orders = (
        int(inner_radial_order),
        int(inner_angular_order),
        int(outer_radial_order),
        int(outer_angular_order),
    )
    ladder_steps = int(ladder_steps)
    radial_sampling_power = float(radial_sampling_power)
    if min(orders) < 2 or ladder_steps < 2:
        raise ValueError("all orders and ladder_steps must be at least two")
    if not 0.0 < radial_sampling_power <= 2.0:
        raise ValueError("radial_sampling_power must lie in (0,2]")
    counterterms = estimate_equal_imaginary_residue_counterterms(
        kernel,
        collar_radius=collar_radius,
        angular_order=counterterm_angular_order,
    )
    estimates = tuple(
        _finite_part_crossing_cell_collar_estimate(
            kernel,
            counterterms,
            inner_radial_order=orders[0] + step,
            inner_angular_order=orders[1] + 2 * step,
            outer_radial_order=orders[2] + step,
            outer_angular_order=orders[3] + 2 * step,
            radial_sampling_power=radial_sampling_power,
            angular_shift=(step + 0.5) / (ladder_steps + 1.0),
        )
        for step in range(ladder_steps)
    )
    finest = estimates[-1]
    previous = estimates[-2]
    positive_margins = [
        record.margin for record in kernel.audit.records if record.margin > 0.0
    ]
    remainder_margin = min(
        min(positive_margins),
        counterterms[0].radial_power + 2.0,
    )
    inner_count = (
        6
        * (orders[0] + ladder_steps - 1)
        * (orders[1] + 2 * (ladder_steps - 1))
    )
    outer_count = (
        18
        * (orders[2] + ladder_steps - 1)
        * (orders[3] + 2 * (ladder_steps - 1))
    )
    return FourPointAmplitudeResult(
        outgoing_energies=kernel.outgoing_energies,  # type: ignore[arg-type]
        incoming_energy=kernel.incoming_energy,
        backend=f"{kernel.block_backend}:continued-finite-part-collars",
        estimates=estimates,
        mean=finest,
        standard_error_real=abs((finest - previous).real),
        standard_error_imag=abs((finest - previous).imag),
        samples_per_replicate=inner_count + outer_count,
        replicates=ladder_steps,
        radial_power=radial_sampling_power,
        convergence_margin=float(remainder_margin),
        finite_part_counterterms=counterterms,
    )


def _density_in_crossing_frame(
    kernel: Type0BSphereFourPointHybrid,
    z_value: complex,
    frame: int,
    *,
    block_region: BlockRegion = "auto",
) -> complex:
    positions = kernel.fixed_positions(z_value)
    channel = _crossing_cell_channel(positions, frame)
    return kernel.density(
        z_value,
        channel=channel,
        block_region=block_region,
    )


def _folded_density_outside_one_collar(
    kernel: Type0BSphereFourPointHybrid,
    folded_coordinate: complex,
    collar_radius: float,
) -> complex:
    """Fold the full plane to ``|u|<1`` while excising ``|1-z|<rho``."""

    value = complex(folded_coordinate)
    if not 0.0 < abs(value) < 1.0:
        raise ValueError("folded_coordinate must lie in the punctured unit disk")
    total = 0.0 + 0.0j
    if abs(1.0 - value) >= collar_radius:
        near_one = abs(1.0 - value) < kernel.hybrid_corner_radius
        frame = 1 if near_one else 0
        total += _density_in_crossing_frame(
            kernel,
            value,
            frame,
            block_region="auto",
        )
    outside = 1.0 / value
    if abs(1.0 - outside) >= collar_radius:
        near_one = abs(1.0 - outside) < kernel.hybrid_corner_radius
        frame = 5 if near_one else 2
        total += abs(value) ** -4 * _density_in_crossing_frame(
            kernel,
            outside,
            frame,
            block_region="auto",
        )
    return complex(total)


def _folded_origin_collar(
    kernel: Type0BSphereFourPointHybrid,
    *,
    collar_radius: float,
    origin_radius: float,
    radial_order: int,
    angular_order: int,
    radial_sampling_power: float,
    angular_shift: float,
) -> complex:
    """Integrate the folded ``z=0, infinity`` collar deterministically."""

    raw_nodes, raw_weights = np.polynomial.legendre.leggauss(radial_order)
    nodes = 0.5 * (raw_nodes + 1.0)
    weights = 0.5 * raw_weights
    total = 0.0 + 0.0j
    for angular_index in range(angular_order):
        theta = 2.0 * math.pi * (
            (angular_index + float(angular_shift)) / angular_order
        )
        phase = cmath.exp(1.0j * theta)
        for node, weight in zip(nodes, weights):
            radius = origin_radius * math.exp(
                math.log(float(node)) / radial_sampling_power
            )
            radial_jacobian = (
                radius * radius
                / (radial_sampling_power * float(node))
            )
            total += (
                2.0
                * math.pi
                / angular_order
                * float(weight)
                * radial_jacobian
                * _folded_density_outside_one_collar(
                    kernel,
                    radius * phase,
                    collar_radius,
                )
            )
    return complex(total)


def _one_collar_finite_part(
    kernel: Type0BSphereFourPointHybrid,
    counterterm: FourPointFinitePartCounterterm,
    *,
    radial_order: int,
    angular_order: int,
    radial_sampling_power: float,
    angular_shift: float,
) -> complex:
    """Integrate the complete ``z=1`` disk after the spin-zero subtraction."""

    raw_nodes, raw_weights = np.polynomial.legendre.leggauss(radial_order)
    nodes = 0.5 * (raw_nodes + 1.0)
    weights = 0.5 * raw_weights
    total = 0.0 + 0.0j
    for angular_index in range(angular_order):
        theta = 2.0 * math.pi * (
            (angular_index + float(angular_shift)) / angular_order
        )
        phase = cmath.exp(1.0j * theta)
        for node, weight in zip(nodes, weights):
            radius = counterterm.collar_radius * math.exp(
                math.log(float(node)) / radial_sampling_power
            )
            z_value = 1.0 - radius * phase
            density = _density_in_crossing_frame(
                kernel,
                z_value,
                1,
                block_region="corner",
            )
            density -= counterterm.coefficient * radius ** (
                counterterm.radial_power - 2.0
            )
            radial_jacobian = (
                radius * radius
                / (radial_sampling_power * float(node))
            )
            total += (
                2.0
                * math.pi
                / angular_order
                * float(weight)
                * radial_jacobian
                * density
            )
    return complex(total + counterterm.analytic_contribution)


def integrate_equal_imaginary_continued_finite_part_folded(
    kernel: Type0BSphereFourPointHybrid,
    *,
    sobol_power: int = 7,
    replicates: int = 4,
    origin_collar_radius: float = 0.35,
    origin_radial_order: int = 4,
    origin_angular_order: int = 12,
    origin_radial_power: float = 0.5,
    collar_radial_order: int = 4,
    collar_angular_order: int = 8,
    collar_radial_power: float = 0.5,
    collar_radius: float = 0.1,
    counterterm_angular_order: int = 12,
    seed: int = 20260827,
) -> FourPointAmplitudeResult:
    r"""Non-overlapping continued amplitude: one OPE disk plus folded bulk."""

    sobol_power = int(sobol_power)
    replicates = int(replicates)
    origin_collar_radius = float(origin_collar_radius)
    origin_radial_order = int(origin_radial_order)
    origin_angular_order = int(origin_angular_order)
    origin_radial_power = float(origin_radial_power)
    collar_radial_order = int(collar_radial_order)
    collar_angular_order = int(collar_angular_order)
    collar_radial_power = float(collar_radial_power)
    if sobol_power < 1 or replicates < 2:
        raise ValueError("sobol_power must be positive and replicates at least two")
    if collar_radial_order < 2 or collar_angular_order < 4:
        raise ValueError("the collar orders are too small")
    if origin_radial_order < 2 or origin_angular_order < 4:
        raise ValueError("the origin-collar orders are too small")
    if not 0.0 < origin_collar_radius < 1.0 - collar_radius:
        raise ValueError("the two deterministic collars must be disjoint")
    if not 0.0 < origin_radial_power <= 2.0:
        raise ValueError("origin_radial_power must lie in (0,2]")
    if not 0.0 < collar_radial_power <= 2.0:
        raise ValueError("collar_radial_power must lie in (0,2]")
    oriented = estimate_equal_imaginary_residue_counterterms(
        kernel,
        collar_radius=collar_radius,
        angular_order=counterterm_angular_order,
    )
    counterterm = oriented[0]
    collar = _one_collar_finite_part(
        kernel,
        counterterm,
        radial_order=collar_radial_order,
        angular_order=collar_angular_order,
        radial_sampling_power=collar_radial_power,
        angular_shift=0.371,
    )
    origin_collar = _folded_origin_collar(
        kernel,
        collar_radius=float(collar_radius),
        origin_radius=origin_collar_radius,
        radial_order=origin_radial_order,
        angular_order=origin_angular_order,
        radial_sampling_power=origin_radial_power,
        angular_shift=0.219,
    )
    estimates: list[complex] = []
    for replicate in range(replicates):
        sampler = qmc.Sobol(d=2, scramble=True, seed=int(seed) + replicate)
        samples: list[complex] = []
        for point in sampler.random_base2(sobol_power):
            radius = math.sqrt(
                origin_collar_radius**2
                + (1.0 - origin_collar_radius**2) * float(point[0])
            )
            value = radius * cmath.exp(2.0j * math.pi * float(point[1]))
            proposal = 1.0 / (
                math.pi * (1.0 - origin_collar_radius**2)
            )
            samples.append(
                _folded_density_outside_one_collar(
                    kernel, value, float(collar_radius)
                )
                / proposal
            )
        outer = complex(np.mean(np.asarray(samples, dtype=complex)))
        estimates.append(outer + origin_collar + collar)
    values = np.asarray(estimates, dtype=complex)
    positive_margins = [
        record.margin for record in kernel.audit.records if record.margin > 0.0
    ]
    remainder_margin = min(
        min(positive_margins),
        counterterm.radial_power + 2.0,
    )
    return FourPointAmplitudeResult(
        outgoing_energies=kernel.outgoing_energies,  # type: ignore[arg-type]
        incoming_energy=kernel.incoming_energy,
        backend=f"{kernel.block_backend}:continued-finite-part-folded",
        estimates=tuple(estimates),
        mean=complex(np.mean(values)),
        standard_error_real=float(
            np.std(values.real, ddof=1) / math.sqrt(replicates)
        ),
        standard_error_imag=float(
            np.std(values.imag, ddof=1) / math.sqrt(replicates)
        ),
        samples_per_replicate=2**sobol_power,
        replicates=replicates,
        radial_power=origin_radial_power,
        convergence_margin=float(remainder_margin),
        finite_part_counterterms=(counterterm,),
    )


def _folded_annulus_term_quadrature(
    kernel: Type0BSphereFourPointHybrid,
    *,
    origin_radius: float,
    collar_radius: float,
    inverted: bool,
    radial_order: int,
    angular_order: int,
    angular_shift: float,
) -> complex:
    """Integrate one folded-plane image with the disk boundary resolved."""

    onset = (
        1.0 / (1.0 + collar_radius)
        if inverted
        else 1.0 - collar_radius
    )
    breakpoints = sorted(
        {
            float(origin_radius),
            float(min(1.0, max(origin_radius, onset))),
            1.0,
        }
    )
    raw_radial_nodes, raw_radial_weights = np.polynomial.legendre.leggauss(
        radial_order
    )
    raw_angular_nodes, raw_angular_weights = np.polynomial.legendre.leggauss(
        angular_order
    )
    total = 0.0 + 0.0j
    for lower, upper in zip(breakpoints[:-1], breakpoints[1:]):
        if upper <= lower:
            continue
        midpoint = 0.5 * (lower + upper)
        half_width = 0.5 * (upper - lower)
        for raw_radius, raw_radial_weight in zip(
            raw_radial_nodes, raw_radial_weights
        ):
            radius = midpoint + half_width * float(raw_radius)
            radial_weight = half_width * float(raw_radial_weight) * radius
            if inverted:
                boundary_cosine = (
                    1.0 + (1.0 - collar_radius**2) * radius**2
                ) / (2.0 * radius)
            else:
                boundary_cosine = (
                    1.0 + radius**2 - collar_radius**2
                ) / (2.0 * radius)
            if boundary_cosine >= 1.0:
                angular_samples = (
                    (
                        2.0
                        * math.pi
                        * (index + angular_shift)
                        / angular_order,
                        2.0 * math.pi / angular_order,
                    )
                    for index in range(angular_order)
                )
            else:
                theta_minimum = math.acos(max(-1.0, boundary_cosine))
                angular_midpoint = math.pi
                angular_half_width = math.pi - theta_minimum
                angular_samples = (
                    (
                        angular_midpoint
                        + angular_half_width * float(raw_angle),
                        angular_half_width * float(raw_angular_weight),
                    )
                    for raw_angle, raw_angular_weight in zip(
                        raw_angular_nodes, raw_angular_weights
                    )
                )
            for theta, angular_weight in angular_samples:
                folded = radius * cmath.exp(1.0j * theta)
                if inverted:
                    z_value = 1.0 / folded
                    density = radius ** -4 * _density_in_crossing_frame(
                        kernel,
                        z_value,
                        2,
                        block_region="auto",
                    )
                else:
                    density = _density_in_crossing_frame(
                        kernel,
                        folded,
                        0,
                        block_region="auto",
                    )
                total += radial_weight * angular_weight * density
    return complex(total)


def integrate_equal_imaginary_continued_finite_part_quadrature(
    kernel: Type0BSphereFourPointHybrid,
    *,
    radial_order: int = 6,
    angular_order: int = 12,
    ladder_steps: int = 2,
    origin_collar_radius: float = 0.35,
    origin_radial_order: int = 4,
    origin_angular_order: int = 12,
    origin_radial_power: float = 0.5,
    collar_radial_order: int = 4,
    collar_angular_order: int = 12,
    collar_radial_power: float = 0.5,
    collar_radius: float = 0.1,
    counterterm_angular_order: int = 12,
) -> FourPointAmplitudeResult:
    r"""Deterministic finite-part integral with all disk edges resolved."""

    radial_order = int(radial_order)
    angular_order = int(angular_order)
    ladder_steps = int(ladder_steps)
    if radial_order < 2 or angular_order < 4 or ladder_steps < 2:
        raise ValueError("radial_order>=2, angular_order>=4, ladder_steps>=2 required")
    if not 0.0 < origin_collar_radius < 1.0 - collar_radius:
        raise ValueError("the two deterministic collars must be disjoint")
    oriented = estimate_equal_imaginary_residue_counterterms(
        kernel,
        collar_radius=collar_radius,
        angular_order=counterterm_angular_order,
    )
    counterterm = oriented[0]
    collar = _one_collar_finite_part(
        kernel,
        counterterm,
        radial_order=int(collar_radial_order),
        angular_order=int(collar_angular_order),
        radial_sampling_power=float(collar_radial_power),
        angular_shift=0.371,
    )
    origin_collar = _folded_origin_collar(
        kernel,
        collar_radius=float(collar_radius),
        origin_radius=float(origin_collar_radius),
        radial_order=int(origin_radial_order),
        angular_order=int(origin_angular_order),
        radial_sampling_power=float(origin_radial_power),
        angular_shift=0.219,
    )
    estimates: list[complex] = []
    for step in range(ladder_steps):
        scale = 2 ** (step - ladder_steps + 1)
        current_radial = max(2, int(round(radial_order * scale)))
        current_angular = max(4, int(round(angular_order * scale)))
        annulus = 0.0 + 0.0j
        for inverted, shift in ((False, 0.137), (True, 0.293)):
            annulus += _folded_annulus_term_quadrature(
                kernel,
                origin_radius=float(origin_collar_radius),
                collar_radius=float(collar_radius),
                inverted=inverted,
                radial_order=current_radial,
                angular_order=current_angular,
                angular_shift=shift,
            )
        estimates.append(complex(collar + origin_collar + annulus))
    finest = estimates[-1]
    previous = estimates[-2]
    positive_margins = [
        record.margin for record in kernel.audit.records if record.margin > 0.0
    ]
    remainder_margin = min(
        min(positive_margins),
        counterterm.radial_power + 2.0,
    )
    return FourPointAmplitudeResult(
        outgoing_energies=kernel.outgoing_energies,  # type: ignore[arg-type]
        incoming_energy=kernel.incoming_energy,
        backend=f"{kernel.block_backend}:continued-finite-part-quadrature",
        estimates=tuple(estimates),
        mean=finest,
        standard_error_real=abs((finest - previous).real),
        standard_error_imag=abs((finest - previous).imag),
        samples_per_replicate=(
            2 * radial_order * angular_order
            + int(origin_radial_order) * int(origin_angular_order)
            + int(collar_radial_order) * int(collar_angular_order)
        ),
        replicates=ladder_steps,
        radial_power=float(origin_radial_power),
        convergence_margin=float(remainder_margin),
        finite_part_counterterms=(counterterm,),
    )


def integrate_subtraction_free_four_point_cells(
    kernel: Type0BSphereFourPointHybrid,
    *,
    radial_order: int = 6,
    angular_order: int = 12,
    replicates: int = 3,
    radial_power: float | None = None,
) -> FourPointAmplitudeResult:
    r"""Deterministically integrate the six crossing cells of ``M_0,4``.

    The common cell is ``|q|<=1`` and ``Re(q)<=1/2``.  It contains only the
    OPE endpoint ``q=0``.  Gauss--Legendre quadrature is used after the power
    map ``r=r_max(theta)*u^(1/radial_power)``; independent angular offsets
    provide a conservative convergence diagnostic without stochastic noise.
    """

    radial_order = int(radial_order)
    angular_order = int(angular_order)
    replicates = int(replicates)
    if radial_order < 2 or angular_order < 4 or replicates < 2:
        raise ValueError(
            "radial_order>=2, angular_order>=4, and replicates>=2 are required"
        )
    if radial_power is None:
        radial_power = min(1.0, 0.9 * kernel.audit.minimum_margin)
    radial_power = float(radial_power)
    if not 0.0 < radial_power <= 2.0:
        raise ValueError("radial_power must lie in (0,2]")
    nodes, weights = np.polynomial.legendre.leggauss(radial_order)
    radial_nodes = 0.5 * (nodes + 1.0)
    radial_weights = 0.5 * weights
    estimates: list[complex] = []
    for replicate in range(replicates):
        total = 0.0 + 0.0j
        angular_shift = replicate / (replicates * angular_order)
        for angular_index in range(angular_order):
            theta = 2.0 * math.pi * (
                angular_index / angular_order + angular_shift
            )
            cosine = math.cos(theta)
            maximum_radius = min(
                1.0,
                0.5 / cosine if cosine > 0.0 else math.inf,
            )
            phase = cmath.exp(1.0j * theta)
            for node, weight in zip(radial_nodes, radial_weights):
                radius = maximum_radius * math.exp(
                    math.log(float(node)) / radial_power
                )
                q_value = radius * phase
                radial_jacobian = (
                    radius * radius / (radial_power * float(node))
                )
                frame_sum = 0.0 + 0.0j
                for frame in range(6):
                    z, derivative = _crossing_cell_inverse(q_value, frame)
                    positions = kernel.fixed_positions(z)
                    channel = _crossing_cell_channel(positions, frame)
                    if abs(channel.q - q_value) > 2.0e-10 * max(1.0, abs(q_value)):
                        raise ArithmeticError("crossing-cell channel mismatch")
                    frame_sum += (
                        kernel.density(z, channel=channel) * abs(derivative) ** 2
                    )
                total += float(weight) * radial_jacobian * frame_sum
        estimates.append(complex(total * (2.0 * math.pi / angular_order)))
    values = np.asarray(estimates, dtype=complex)
    return FourPointAmplitudeResult(
        outgoing_energies=kernel.outgoing_energies,  # type: ignore[arg-type]
        incoming_energy=kernel.incoming_energy,
        backend=kernel.block_backend,
        estimates=tuple(estimates),
        mean=complex(np.mean(values)),
        standard_error_real=float(np.std(values.real, ddof=1) / math.sqrt(replicates)),
        standard_error_imag=float(np.std(values.imag, ddof=1) / math.sqrt(replicates)),
        samples_per_replicate=6 * radial_order * angular_order,
        replicates=replicates,
        radial_power=radial_power,
        convergence_margin=kernel.audit.minimum_margin,
    )


def integrate_subtraction_free_four_point_component_cells(
    kernel: Type0BSphereFourPointHybrid,
    *,
    radial_order: int = 6,
    angular_order: int = 12,
    replicates: int = 3,
    radial_power: float | None = None,
) -> FourPointContinuedAmplitudeResult:
    r"""Integrate six geometric cells through the verified three-chart atlas.

    The six anharmonic maps provide a non-overlapping tessellation of moduli
    space.  They are used only for geometry and Jacobians.  At every image
    point the conformal block is evaluated in the fastest available member
    of the crossing-connected canonical atlas (frames 0, 1, and 2).  Thus
    the currently rejected orientations 3 and 4 never enter the integrand.
    Continuum and crossed-pole contributions share identical quadrature
    nodes, so their covariance and any cancellation remain visible.
    """

    radial_order = int(radial_order)
    angular_order = int(angular_order)
    replicates = int(replicates)
    if radial_order < 2 or angular_order < 4 or replicates < 2:
        raise ValueError(
            "radial_order>=2, angular_order>=4, and replicates>=2 are required"
        )
    if radial_power is None:
        radial_power = min(1.0, 0.9 * kernel.audit.minimum_margin)
    radial_power = float(radial_power)
    if not 0.0 < radial_power <= 2.0:
        raise ValueError("radial_power must lie in (0,2]")

    nodes, weights = np.polynomial.legendre.leggauss(radial_order)
    radial_nodes = 0.5 * (nodes + 1.0)
    radial_weights = 0.5 * weights
    continuous_estimates: list[complex] = []
    residue_estimates: list[complex] = []
    for replicate in range(replicates):
        continuous_total = 0.0 + 0.0j
        residue_total = 0.0 + 0.0j
        angular_shift = replicate / (replicates * angular_order)
        for angular_index in range(angular_order):
            theta = 2.0 * math.pi * (
                angular_index / angular_order + angular_shift
            )
            cosine = math.cos(theta)
            maximum_radius = min(
                1.0,
                0.5 / cosine if cosine > 0.0 else math.inf,
            )
            phase = cmath.exp(1.0j * theta)
            for node, weight in zip(radial_nodes, radial_weights):
                radius = maximum_radius * math.exp(
                    math.log(float(node)) / radial_power
                )
                q_value = radius * phase
                radial_jacobian = (
                    radius * radius / (radial_power * float(node))
                )
                continuous_frames = 0.0 + 0.0j
                residue_frames = 0.0 + 0.0j
                for frame in range(6):
                    z, derivative = _crossing_cell_inverse(q_value, frame)
                    positions = kernel.fixed_positions(z)
                    candidates = tuple(
                        channel
                        for canonical_frame in range(3)
                        for channel in (
                            _crossing_cell_channel(
                                positions, canonical_frame
                            ),
                        )
                        if channel.score < 1.0 + 1.0e-12
                    )
                    if not candidates:
                        raise ArithmeticError(
                            "the verified three-chart atlas failed to cover z"
                        )
                    channel = min(
                        candidates,
                        key=lambda candidate: (
                            abs(elliptic_nome(candidate.q)),
                            candidate.ordering,
                        ),
                    )
                    components = kernel.density_components(
                        z, channel=channel
                    )
                    jacobian = abs(derivative) ** 2
                    continuous_frames += components.continuous * jacobian
                    residue_frames += components.residues * jacobian
                factor = float(weight) * radial_jacobian
                continuous_total += factor * continuous_frames
                residue_total += factor * residue_frames
        angular_factor = 2.0 * math.pi / angular_order
        continuous_estimates.append(
            complex(angular_factor * continuous_total)
        )
        residue_estimates.append(complex(angular_factor * residue_total))

    return FourPointContinuedAmplitudeResult(
        outgoing_energies=kernel.outgoing_energies,  # type: ignore[arg-type]
        incoming_energy=kernel.incoming_energy,
        backend=f"{kernel.block_backend}:continued-component-cells",
        continuous_estimates=tuple(continuous_estimates),
        residue_estimates=tuple(residue_estimates),
        radial_power=radial_power,
        convergence_margin=kernel.audit.minimum_margin,
        samples_per_replicate=6 * radial_order * angular_order,
    )


def _smooth_crossing_cell_estimate(
    kernel: Type0BSphereFourPointHybrid | Type0BFixedContourFourPointElliptic,
    *,
    radial_order: int,
    angular_order: int,
    radial_power: float,
) -> complex:
    """One tensor estimate with the angular kinks split explicitly."""

    radial_nodes_raw, radial_weights_raw = np.polynomial.legendre.leggauss(
        radial_order
    )
    radial_nodes = 0.5 * (radial_nodes_raw + 1.0)
    radial_weights = 0.5 * radial_weights_raw
    angular_nodes, angular_weights = np.polynomial.legendre.leggauss(angular_order)
    pure_imaginary = all(abs(value.real) < 1.0e-14 for value in kernel.outgoing_energies)
    intervals = (
        ((0.0, math.pi / 3.0), (math.pi / 3.0, math.pi))
        if pure_imaginary
        else (
            (-math.pi / 3.0, math.pi / 3.0),
            (math.pi / 3.0, math.pi),
            (math.pi, 5.0 * math.pi / 3.0),
        )
    )
    total = 0.0 + 0.0j
    for lower, upper in intervals:
        midpoint = 0.5 * (lower + upper)
        half_width = 0.5 * (upper - lower)
        for angular_node, angular_weight in zip(angular_nodes, angular_weights):
            theta = midpoint + half_width * float(angular_node)
            cosine = math.cos(theta)
            maximum_radius = min(
                1.0,
                0.5 / cosine if cosine > 0.0 else math.inf,
            )
            phase = cmath.exp(1.0j * theta)
            for radial_node, radial_weight in zip(radial_nodes, radial_weights):
                radius = maximum_radius * math.exp(
                    math.log(float(radial_node)) / radial_power
                )
                q_value = radius * phase
                radial_jacobian = (
                    radius * radius / (radial_power * float(radial_node))
                )
                frame_sum = 0.0 + 0.0j
                for frame in range(6):
                    z, derivative = _crossing_cell_inverse(q_value, frame)
                    if isinstance(kernel, Type0BFixedContourFourPointElliptic):
                        density = kernel.density(z)
                    else:
                        positions = kernel.fixed_positions(z)
                        channel = _crossing_cell_channel(positions, frame)
                        density = kernel.density(z, channel=channel)
                    frame_sum += density * abs(derivative) ** 2
                total += (
                    half_width
                    * float(angular_weight)
                    * float(radial_weight)
                    * radial_jacobian
                    * frame_sum
                )
    if pure_imaginary:
        return complex(2.0 * total.real)
    return complex(total)


def integrate_subtraction_free_four_point_cell_ladder(
    kernel: Type0BSphereFourPointHybrid | Type0BFixedContourFourPointElliptic,
    *,
    radial_order: int = 4,
    angular_order: int = 4,
    ladder_steps: int = 2,
    radial_power: float | None = None,
) -> FourPointAmplitudeResult:
    """Crossing-cell Gauss ladder; the finest estimate is the reported mean."""

    radial_order = int(radial_order)
    angular_order = int(angular_order)
    ladder_steps = int(ladder_steps)
    if radial_order < 2 or angular_order < 2 or ladder_steps < 2:
        raise ValueError("orders and ladder_steps must each be at least two")
    if radial_power is None:
        radial_power = min(1.0, 0.9 * kernel.audit.minimum_margin)
    radial_power = float(radial_power)
    estimates = tuple(
        _smooth_crossing_cell_estimate(
            kernel,
            radial_order=radial_order + step,
            angular_order=angular_order + 2 * step,
            radial_power=radial_power,
        )
        for step in range(ladder_steps)
    )
    finest = estimates[-1]
    previous = estimates[-2]
    pure_imaginary = all(abs(value.real) < 1.0e-14 for value in kernel.outgoing_energies)
    finest_radial = radial_order + ladder_steps - 1
    finest_angular = angular_order + 2 * (ladder_steps - 1)
    angular_piece_count = 2 if pure_imaginary else 3
    return FourPointAmplitudeResult(
        outgoing_energies=kernel.outgoing_energies,  # type: ignore[arg-type]
        incoming_energy=kernel.incoming_energy,
        backend=kernel.block_backend,
        estimates=estimates,
        mean=finest,
        standard_error_real=abs((finest - previous).real),
        standard_error_imag=abs((finest - previous).imag),
        samples_per_replicate=(
            6 * finest_radial * finest_angular * angular_piece_count
        ),
        replicates=ladder_steps,
        radial_power=radial_power,
        convergence_margin=kernel.audit.minimum_margin,
    )


def folded_unit_disk_density(
    kernel: Type0BSphereFourPointHybrid,
    z: Number,
    *,
    corner_radius: float | None = None,
) -> complex:
    r"""Return the inside plus inversion image on ``0<|z|<1``.

    Away from collars, frames 0 and 2 keep the fixed picture-zero fields in
    the two middle slots and therefore use h-recursion.  Near ``z=1`` the
    local frames 1 and 5 are supplied by c-recursion; the origin collar uses
    c-recursion in frames 0 and 2.  This avoids using the presently
    unverified endpoint-descendant h-recursion in the production bulk.
    """

    value = _finite_complex("z", z)
    if not 0.0 < abs(value) < 1.0 + 1.0e-14:
        raise ValueError("the folded coordinate must lie in the punctured unit disk")
    radius = kernel.hybrid_corner_radius if corner_radius is None else float(corner_radius)
    if not 0.0 < radius < 0.5:
        raise ValueError("corner_radius must lie in (0,1/2)")
    origin_corner = abs(value) < radius
    one_corner = abs(1.0 - value) < radius

    inside_positions = kernel.fixed_positions(value)
    inside_frame = 1 if one_corner else 0
    inside_channel = _crossing_cell_channel(inside_positions, inside_frame)
    local_region: BlockRegion = (
        "corner" if origin_corner or one_corner else "auto"
    )
    inside = kernel.density(
        value,
        channel=inside_channel,
        block_region=local_region,
    )

    outside_point = 1.0 / value
    outside_positions = kernel.fixed_positions(outside_point)
    outside_frame = 5 if one_corner else 2
    outside_channel = _crossing_cell_channel(outside_positions, outside_frame)
    outside = kernel.density(
        outside_point,
        channel=outside_channel,
        block_region=local_region,
    )
    return complex(inside + abs(value) ** -4 * outside)


def integrate_folded_unit_disk_qmc(
    kernel: Type0BSphereFourPointHybrid,
    *,
    sobol_power: int = 9,
    replicates: int = 4,
    one_corner_power: float | None = None,
    seed: int = 20260827,
) -> FourPointAmplitudeResult:
    """Integrate the h-dominant folded disk with an OPE-cap mixture."""

    sobol_power = int(sobol_power)
    replicates = int(replicates)
    if sobol_power < 1 or replicates < 2:
        raise ValueError("sobol_power must be positive and replicates at least two")
    if one_corner_power is None:
        raised_records = tuple(
            record.margin
            for record in kernel.audit.records
            if frozenset(record.partition) == frozenset(PICTURE_ZERO_LABELS)
        )
        one_corner_power = 0.9 * min(raised_records)
    alpha = float(one_corner_power)
    if not 0.0 < alpha <= 2.0:
        raise ValueError("one_corner_power must lie in (0,2]")
    estimates: list[complex] = []
    for replicate in range(replicates):
        sampler = qmc.Sobol(d=3, scramble=True, seed=seed + replicate)
        samples: list[complex] = []
        for point in sampler.random_base2(sobol_power):
            component = min(int(2 * point[2]), 1)
            if component == 0:
                radius = math.sqrt(float(point[0]))
                angle = 2.0 * math.pi * float(point[1])
                value = radius * cmath.exp(1.0j * angle)
            else:
                radius = math.exp(math.log(max(float(point[0]), 1.0e-300)) / alpha)
                angle = 2.0 * math.pi * float(point[1])
                value = 1.0 - radius * cmath.exp(1.0j * angle)
            if not 0.0 < abs(value) < 1.0:
                samples.append(0.0 + 0.0j)
                continue
            uniform_density = 1.0 / math.pi
            cap_radius = abs(1.0 - value)
            cap_density = (
                alpha / (2.0 * math.pi) * cap_radius ** (alpha - 2.0)
                if 0.0 < cap_radius < 1.0
                else 0.0
            )
            mixture_density = 0.5 * (uniform_density + cap_density)
            samples.append(
                folded_unit_disk_density(kernel, value) / mixture_density
            )
        estimates.append(complex(np.mean(np.asarray(samples, dtype=complex))))
    values = np.asarray(estimates, dtype=complex)
    return FourPointAmplitudeResult(
        outgoing_energies=kernel.outgoing_energies,  # type: ignore[arg-type]
        incoming_energy=kernel.incoming_energy,
        backend=f"{kernel.block_backend}:folded-disk",
        estimates=tuple(estimates),
        mean=complex(np.mean(values)),
        standard_error_real=float(np.std(values.real, ddof=1) / math.sqrt(replicates)),
        standard_error_imag=float(np.std(values.imag, ddof=1) / math.sqrt(replicates)),
        samples_per_replicate=2**sobol_power,
        replicates=replicates,
        radial_power=alpha,
        convergence_margin=kernel.audit.minimum_margin,
    )


__all__ = [
    "CONVERGENT_RAY_COEFFICIENTS",
    "CONVERGENT_RAY_RECTANGLE",
    "CONVERGENT_RAY_REFERENCE",
    "LARGE_RESIDUE_RAY_RECTANGLE",
    "LARGE_RESIDUE_RAY_REFERENCE",
    "WALL_ONE_RAY_COEFFICIENTS",
    "WALL_ONE_RAY_RECTANGLE",
    "WALL_TWO_RAY_RECTANGLE",
    "WALL_THREE_RAY_RECTANGLE",
    "WALL_ONE_MOMENTUM_INTERVALS",
    "FourPointAmplitudeResult",
    "FourPointContinuedAmplitudeResult",
    "FourPointBoundaryMargin",
    "FourPointChannel",
    "FourPointConvergenceAudit",
    "FourPointCrossingAudit",
    "FourPointCrossingFrameValue",
    "FourPointFinitePartCounterterm",
    "FourPointRayRectangleCertificate",
    "FourPointResidueDomainCertificate",
    "FourPointResidueWallCertificate",
    "FourPointDensityComponents",
    "Type0BSphereFourPointHybrid",
    "Type0BFixedContourFourPointElliptic",
    "audit_four_point_convergence",
    "audit_four_point_crossing",
    "best_four_point_channel",
    "canonical_chart_channel",
    "certify_convergent_ray_rectangle",
    "certify_residue_convergent_ray_rectangle",
    "convergent_ray_energies",
    "four_point_channel_from_ordering",
    "estimate_equal_imaginary_residue_counterterms",
    "integrate_equal_imaginary_continued_finite_part_cells",
    "integrate_equal_imaginary_continued_finite_part_collars",
    "integrate_equal_imaginary_continued_finite_part_folded",
    "integrate_equal_imaginary_continued_finite_part_quadrature",
    "integrate_equal_imaginary_continued_finite_part_qmc",
    "integrate_subtraction_free_four_point",
    "integrate_subtraction_free_four_point_component_stratified_qmc",
    "integrate_subtraction_free_four_point_cells",
    "integrate_subtraction_free_four_point_component_cells",
    "integrate_subtraction_free_four_point_cell_ladder",
    "folded_unit_disk_density",
    "integrate_folded_unit_disk_qmc",
    "three_chart_log_mixture_density",
    "wall_one_momentum_rule",
]
