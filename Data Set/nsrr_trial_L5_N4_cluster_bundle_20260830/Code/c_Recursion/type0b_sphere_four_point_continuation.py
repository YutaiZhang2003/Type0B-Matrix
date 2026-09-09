#!/usr/bin/env python3
r"""Certified analytic-continuation chamber search for the Type-0B four point.

The amplitude kernel in :mod:`type0b_sphere_four_point_hybrid` knows how to
enumerate every continuum and crossed-pole contribution at the three boundary
divisors of ``Mbar_0,4``.  This module turns that pointwise audit into an
optimization problem on ray rectangles

    omega_j = a_j (sigma*x + i*t),

where ``sigma=+1`` is the BRY continuation from positive real energies.  A
candidate rectangle is accepted only when its crossed-pole ledger is constant
and every continuum and residue radial exponent has a positive analytic lower
bound on the *whole* rectangle.

Convergence alone is not enough for a useful numerical integral.  If a radial
exponent is ``Delta+i*Phi``, importance sampling removes the power
``r**Delta`` but leaves the log-radial oscillation ``exp(i*Phi*log(r))``.  We
therefore certify an upper bound for ``|Phi|`` and rank domains using their
convergence margin, area, oscillation burden, and residue cost.  Mathematical
certification is kept separate from production readiness: the current four-
point residue evaluator supports coincident product poles only through order
two.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Iterable, Sequence

from type0b_sphere_four_point_hybrid import (
    FourPointBoundaryMargin,
    FourPointConvergenceAudit,
    FourPointResidueDomainCertificate,
    audit_four_point_convergence,
    certify_residue_convergent_ray_rectangle,
)


@dataclass(frozen=True)
class FourPointExponentBound:
    """Uniform bounds for one fixed-ledger plumbing exponent."""

    partition: tuple[int, int]
    side: tuple[int, int]
    kind: str
    sector: int | None
    threshold: int
    minimum_real_part: float
    maximum_absolute_imaginary_part: float
    phase_to_margin_upper_bound: float

    @property
    def signature(self) -> tuple[object, ...]:
        return self.partition, self.side, self.kind, self.sector

    def to_json(self) -> dict[str, object]:
        result = asdict(self)
        result["partition"] = list(self.partition)
        result["side"] = list(self.side)
        return result


@dataclass(frozen=True)
class FourPointContinuationCertificate:
    """Certified kinematic rectangle and its numerical quality metrics."""

    ray_coefficients: tuple[float, float, float]
    ray_real_sign: int
    x_interval: tuple[float, float]
    t_interval: tuple[float, float]
    path_formula: str
    domain_certificate: FourPointResidueDomainCertificate
    exponent_bounds: tuple[FourPointExponentBound, ...]
    limiting_signature: tuple[object, ...]
    minimum_margin_lower_bound: float
    maximum_phase_upper_bound: float
    maximum_phase_to_margin_upper_bound: float
    rectangle_area: float
    residue_wall_count: int
    residue_record_count: int
    residue_cost: int
    maximum_combined_pole_order: int
    maximum_logarithm_power: int
    required_minimum_margin: float
    required_wall_clearance: float
    supported_product_pole_order: int
    mathematically_certified: bool
    production_ready: bool
    quality_score: float

    @property
    def center_base(self) -> complex:
        return complex(
            self.ray_real_sign * 0.5 * sum(self.x_interval),
            0.5 * sum(self.t_interval),
        )

    @property
    def center_outgoing_energies(self) -> tuple[complex, complex, complex]:
        return tuple(
            coefficient * self.center_base
            for coefficient in self.ray_coefficients
        )  # type: ignore[return-value]

    @property
    def crossed_walls(self) -> tuple[int, ...]:
        return tuple(wall.wall for wall in self.domain_certificate.residue_walls)

    def ranking_key(self) -> tuple[float, ...]:
        """Lexicographic key used after the hard production filters."""

        return (
            float(self.production_ready),
            float(self.mathematically_certified),
            self.quality_score,
            -float(self.residue_wall_count),
            -float(self.residue_cost),
            self.minimum_margin_lower_bound,
            self.rectangle_area,
        )

    def to_json(self, *, include_exponent_bounds: bool = True) -> dict[str, object]:
        result: dict[str, object] = {
            "ray_coefficients": list(self.ray_coefficients),
            "ray_real_sign": self.ray_real_sign,
            "x_interval": list(self.x_interval),
            "t_interval": list(self.t_interval),
            "center_base": {
                "real": self.center_base.real,
                "imag": self.center_base.imag,
            },
            "center_outgoing_energies": [
                {"real": value.real, "imag": value.imag}
                for value in self.center_outgoing_energies
            ],
            "path_formula": self.path_formula,
            "crossed_walls": list(self.crossed_walls),
            "limiting_signature": _signature_to_json(self.limiting_signature),
            "minimum_margin_lower_bound": self.minimum_margin_lower_bound,
            "maximum_phase_upper_bound": self.maximum_phase_upper_bound,
            "maximum_phase_to_margin_upper_bound": (
                self.maximum_phase_to_margin_upper_bound
            ),
            "rectangle_area": self.rectangle_area,
            "residue_wall_count": self.residue_wall_count,
            "residue_record_count": self.residue_record_count,
            "residue_cost": self.residue_cost,
            "maximum_combined_pole_order": self.maximum_combined_pole_order,
            "maximum_logarithm_power": self.maximum_logarithm_power,
            "required_minimum_margin": self.required_minimum_margin,
            "required_wall_clearance": self.required_wall_clearance,
            "supported_product_pole_order": self.supported_product_pole_order,
            "mathematically_certified": self.mathematically_certified,
            "production_ready": self.production_ready,
            "quality_score": self.quality_score,
            "domain_certificate": self.domain_certificate.to_json(),
        }
        if include_exponent_bounds:
            result["exponent_bounds"] = [
                bound.to_json() for bound in self.exponent_bounds
            ]
        return result


@dataclass(frozen=True)
class FourPointContinuationSearchResult:
    """Ranked output of a deterministic chamber search."""

    candidates_evaluated: int
    mathematically_certified_count: int
    production_ready_count: int
    ranked_candidates: tuple[FourPointContinuationCertificate, ...]

    @property
    def best(self) -> FourPointContinuationCertificate | None:
        return self.ranked_candidates[0] if self.ranked_candidates else None

    def to_json(self, *, include_exponent_bounds: bool = False) -> dict[str, object]:
        return {
            "candidates_evaluated": self.candidates_evaluated,
            "mathematically_certified_count": self.mathematically_certified_count,
            "production_ready_count": self.production_ready_count,
            "best": (
                None
                if self.best is None
                else self.best.to_json(
                    include_exponent_bounds=include_exponent_bounds
                )
            ),
            "ranked_candidates": [
                candidate.to_json(
                    include_exponent_bounds=include_exponent_bounds
                )
                for candidate in self.ranked_candidates
            ],
        }


def _signature(record: FourPointBoundaryMargin) -> tuple[object, ...]:
    return record.partition, record.side, record.kind, record.sector


def _signature_to_json(signature: tuple[object, ...]) -> list[object]:
    partition, side, kind, sector = signature
    return [list(partition), list(side), kind, sector]  # type: ignore[arg-type]


def _audit_signature(audit: FourPointConvergenceAudit) -> tuple[tuple[object, ...], ...]:
    return tuple(_signature(record) for record in audit.records)


def _record_affine_parameters(
    record: FourPointBoundaryMargin,
    center_base: complex,
) -> tuple[float, float, float]:
    r"""Return real ``(alpha,beta,c)`` with ``P=alpha*l+i*c``, ``K=beta*l``."""

    if abs(center_base.real) < 1.0e-14:
        raise ValueError("the ray base must have a nonzero real part")
    beta_value = record.channel_energy / center_base
    if abs(beta_value.imag) > 2.0e-9:
        raise ArithmeticError("channel energy is not real-linear on the ray")
    beta = float(beta_value.real)
    if record.kind == "continuous":
        alpha = 0.0
        offset = 0.0
    else:
        alpha = float(record.momentum.real / center_base.real)
        residual = record.momentum - alpha * center_base
        if abs(residual.real) > 2.0e-9:
            raise ArithmeticError("residue momentum left the affine ray ledger")
        offset = float(residual.imag)
    reconstructed = alpha * center_base + 1j * offset
    if abs(reconstructed - record.momentum) > 5.0e-9:
        raise ArithmeticError("failed to reconstruct the residue momentum")
    return alpha, beta, offset


def _record_rectangle_bound(
    record: FourPointBoundaryMargin,
    center_base: complex,
    x_interval: tuple[float, float],
    t_interval: tuple[float, float],
    real_sign: int,
) -> FourPointExponentBound:
    r"""Minimize ``Re(P^2-K^2-tau)`` and bound its phase analytically."""

    alpha, beta, offset = _record_affine_parameters(record, center_base)
    coefficient = alpha * alpha - beta * beta
    x_value = x_interval[0] if coefficient >= 0.0 else x_interval[1]
    t_candidates = [t_interval[0], t_interval[1]]
    # The t quadratic -coefficient*t^2-2*alpha*offset*t is convex
    # precisely when coefficient is negative.
    if coefficient < 0.0:
        stationary = -alpha * offset / coefficient
        if t_interval[0] < stationary < t_interval[1]:
            t_candidates.append(stationary)

    def real_part(x_value: float, t_value: float) -> float:
        return float(
            coefficient * (x_value * x_value - t_value * t_value)
            - 2.0 * alpha * offset * t_value
            - offset * offset
            - record.threshold
        )

    minimum_real = min(real_part(x_value, value) for value in t_candidates)
    maximum_phase = max(
        abs(
            2.0
            * real_sign
            * x_value
            * (coefficient * t_value + alpha * offset)
        )
        for x_value in x_interval
        for t_value in t_interval
    )
    ratio = (
        float(maximum_phase / minimum_real)
        if minimum_real > 0.0
        else math.inf
    )
    return FourPointExponentBound(
        partition=record.partition,
        side=record.side,
        kind=record.kind,
        sector=record.sector,
        threshold=record.threshold,
        minimum_real_part=float(minimum_real),
        maximum_absolute_imaginary_part=float(maximum_phase),
        phase_to_margin_upper_bound=ratio,
    )


def certify_four_point_continuation_rectangle(
    x_interval: Sequence[float],
    t_interval: Sequence[float],
    *,
    ray_coefficients: Sequence[float],
    ray_real_sign: int = 1,
    required_minimum_margin: float = 0.0,
    required_wall_clearance: float = 1.0e-8,
    supported_product_pole_order: int = 2,
) -> FourPointContinuationCertificate:
    r"""Certify and score a vertical analytic-continuation ray rectangle.

    The path at each point of the rectangle is

    ``lambda(s)=sigma*x+i*s*t``, ``0<=s<=1``.

    The crossed-pole routine uses this monotone continuation from the real
    axis.  Requiring a fixed endpoint ledger on the rectangle fixes the set
    of residues, while the exponent bounds certify the ordinary unsubtracted
    moduli integral for the continuum and every residue term.
    """

    if len(x_interval) != 2 or len(t_interval) != 2:
        raise ValueError("each interval must contain two endpoints")
    if len(ray_coefficients) != 3:
        raise ValueError("ray_coefficients must contain three values")
    coefficients = tuple(float(value) for value in ray_coefficients)
    if any(value <= 0.0 or not math.isfinite(value) for value in coefficients):
        raise ValueError("ray coefficients must be positive and finite")
    sign = int(ray_real_sign)
    if sign not in (-1, 1):
        raise ValueError("ray_real_sign must be -1 or +1")
    x_pair = tuple(float(value) for value in x_interval)
    t_pair = tuple(float(value) for value in t_interval)
    if not 0.0 < x_pair[0] < x_pair[1] or not 0.0 < t_pair[0] < t_pair[1]:
        raise ValueError("rectangle endpoints must be positive and increasing")
    if required_minimum_margin < 0.0:
        raise ValueError("required_minimum_margin must be nonnegative")
    if required_wall_clearance <= 0.0:
        raise ValueError("required_wall_clearance must be positive")
    if supported_product_pole_order < 1:
        raise ValueError("supported_product_pole_order must be positive")

    domain = certify_residue_convergent_ray_rectangle(
        x_pair,
        t_pair,
        ray_coefficients=coefficients,
        ray_real_sign=sign,
    )
    center_base = complex(sign * 0.5 * sum(x_pair), 0.5 * sum(t_pair))
    center_energies = tuple(value * center_base for value in coefficients)
    center = audit_four_point_convergence(
        center_energies, include_residues=True
    )
    corner_audits = tuple(
        audit_four_point_convergence(
            tuple(value * complex(sign * x_value, t_value) for value in coefficients),
            include_residues=True,
        )
        for x_value in x_pair
        for t_value in t_pair
    )
    signature = _audit_signature(center)
    if domain.chamber_stable and any(
        _audit_signature(corner) != signature for corner in corner_audits
    ):
        raise ArithmeticError("domain and explicit chamber-stability audits disagree")

    bounds = tuple(
        _record_rectangle_bound(record, center_base, x_pair, t_pair, sign)
        for record in center.records
    )
    limiting = min(bounds, key=lambda item: item.minimum_real_part)
    minimum_margin = limiting.minimum_real_part
    maximum_phase = max(
        bound.maximum_absolute_imaginary_part for bound in bounds
    )
    maximum_ratio = max(
        bound.phase_to_margin_upper_bound for bound in bounds
    )

    residue_wall_count = len(domain.residue_walls)
    residue_record_count = sum(wall.record_count for wall in domain.residue_walls)
    residue_cost = sum(
        wall.record_count * (1 + wall.maximum_logarithm_power)
        for wall in domain.residue_walls
    )
    maximum_order = max(
        (wall.maximum_combined_pole_order for wall in domain.residue_walls),
        default=0,
    )
    maximum_logarithm = max(
        (wall.maximum_logarithm_power for wall in domain.residue_walls),
        default=0,
    )
    domain_minimum = min(
        domain.continuum_minimum_margin_lower_bound,
        *(
            wall.minimum_margin_lower_bound
            for wall in domain.residue_walls
        ),
    )
    if abs(domain_minimum - minimum_margin) > 2.0e-9:
        raise ArithmeticError("independent analytic margin certificates disagree")
    area = (x_pair[1] - x_pair[0]) * (t_pair[1] - t_pair[0])
    mathematically_certified = bool(
        domain.certified
        and minimum_margin >= required_minimum_margin
        and domain.minimum_wall_clearance >= required_wall_clearance
    )
    production_ready = bool(
        mathematically_certified
        and maximum_order <= supported_product_pole_order
    )
    quality = (
        minimum_margin
        * math.sqrt(area)
        / ((1.0 + maximum_phase) * (1.0 + residue_cost))
        if mathematically_certified
        else 0.0
    )
    sheet = "positive" if sign > 0 else "negative"
    return FourPointContinuationCertificate(
        ray_coefficients=coefficients,  # type: ignore[arg-type]
        ray_real_sign=sign,
        x_interval=x_pair,  # type: ignore[arg-type]
        t_interval=t_pair,  # type: ignore[arg-type]
        path_formula=(
            f"{sheet}-real sheet: lambda(s)={sign:+d}*x+i*s*t, 0<=s<=1"
        ),
        domain_certificate=domain,
        exponent_bounds=bounds,
        limiting_signature=limiting.signature,
        minimum_margin_lower_bound=float(minimum_margin),
        maximum_phase_upper_bound=float(maximum_phase),
        maximum_phase_to_margin_upper_bound=float(maximum_ratio),
        rectangle_area=float(area),
        residue_wall_count=residue_wall_count,
        residue_record_count=residue_record_count,
        residue_cost=residue_cost,
        maximum_combined_pole_order=maximum_order,
        maximum_logarithm_power=maximum_logarithm,
        required_minimum_margin=float(required_minimum_margin),
        required_wall_clearance=float(required_wall_clearance),
        supported_product_pole_order=int(supported_product_pole_order),
        mathematically_certified=mathematically_certified,
        production_ready=production_ready,
        quality_score=float(quality),
    )


def search_four_point_continuation_rectangles(
    *,
    ray_candidates: Iterable[Sequence[float]],
    rectangles: Iterable[tuple[Sequence[float], Sequence[float]]],
    ray_real_sign: int = 1,
    required_minimum_margin: float = 0.0,
    required_wall_clearance: float = 1.0e-8,
    supported_product_pole_order: int = 2,
    keep: int = 20,
) -> FourPointContinuationSearchResult:
    """Certify and rank a deterministic collection of candidate domains."""

    if keep < 1:
        raise ValueError("keep must be positive")
    rays = tuple(tuple(float(value) for value in ray) for ray in ray_candidates)
    boxes = tuple((tuple(x), tuple(t)) for x, t in rectangles)
    candidates: list[FourPointContinuationCertificate] = []
    for ray in rays:
        for x_interval, t_interval in boxes:
            candidates.append(
                certify_four_point_continuation_rectangle(
                    x_interval,
                    t_interval,
                    ray_coefficients=ray,
                    ray_real_sign=ray_real_sign,
                    required_minimum_margin=required_minimum_margin,
                    required_wall_clearance=required_wall_clearance,
                    supported_product_pole_order=supported_product_pole_order,
                )
            )
    ranked = tuple(
        sorted(candidates, key=lambda item: item.ranking_key(), reverse=True)[:keep]
    )
    return FourPointContinuationSearchResult(
        candidates_evaluated=len(candidates),
        mathematically_certified_count=sum(
            candidate.mathematically_certified for candidate in candidates
        ),
        production_ready_count=sum(candidate.production_ready for candidate in candidates),
        ranked_candidates=ranked,
    )


def centered_rectangles(
    x_centers: Iterable[float],
    t_centers: Iterable[float],
    *,
    x_half_width: float,
    t_half_width: float,
) -> tuple[tuple[tuple[float, float], tuple[float, float]], ...]:
    """Construct positive ray rectangles from center grids."""

    if x_half_width <= 0.0 or t_half_width <= 0.0:
        raise ValueError("rectangle half widths must be positive")
    result = []
    for x_center in x_centers:
        for t_center in t_centers:
            x_value = float(x_center)
            t_value = float(t_center)
            if x_value <= x_half_width or t_value <= t_half_width:
                raise ValueError("rectangle centers must exceed their half widths")
            result.append(
                (
                    (x_value - x_half_width, x_value + x_half_width),
                    (t_value - t_half_width, t_value + t_half_width),
                )
            )
    return tuple(result)


__all__ = [
    "FourPointContinuationCertificate",
    "FourPointContinuationSearchResult",
    "FourPointExponentBound",
    "centered_rectangles",
    "certify_four_point_continuation_rectangle",
    "search_four_point_continuation_rectangles",
]
