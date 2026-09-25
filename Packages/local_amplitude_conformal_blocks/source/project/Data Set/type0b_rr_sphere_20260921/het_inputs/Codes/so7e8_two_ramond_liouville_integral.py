#!/usr/bin/env python3
r"""Full finite-level Liouville quadrature for two-R/two-NS amplitudes.

This evaluator is deliberately restricted to a genuinely convergent generic
complex-energy chamber.  The default uses the locally audited crossed internal-NS
decomposition over the full folded plane. The full correlator remains
provisional: the corrected RN orientation, ordered PCO signs, and one-star
theta seed pass local Ward and inversion tests, but integrated picture
agreement still needs endpoint/order refinement. A mixed local-channel atlas is
implemented for auditing, but its direct internal-R sewing is deliberately
opt-in until a P-integrated direct/crossed overlap test passes.

The modulus plane is folded to the unit disk and partitioned into the same
six geometric pieces used by the SO(23) calculation: original/swapped bulk,
original/swapped disk around zero, and original/swapped lens around one.
Here every piece is an ordinary convergent quadrature.  The blocks are
algebraically reorganized using their sector-specific elliptic prefactors.
The crossed expansion is slow near ``z=0``; adjacent-level stability must
therefore be checked numerically rather than inferred from a finite cutoff.

The returned values omit the common sphere normalization, heterotic coupling,
energy delta function, and external reflection phases.  Internal momentum is
normalized as ``integral_0^infinity dP/pi`` (or its explicitly requested
finite-cutoff approximation).
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Mapping, Sequence

import numpy as np

from spin23_singlet_amplitudes import (
    _momentum_quadrature,
    _stable_complex_fsum,
    _stable_complex_sum,
)
from heterotic_so23_1to3_vvvv_fit_bundle import heterotic_so23_1to3_fast as fast
from ramond_sphere_uniformization import (
    EllipticPrefactoredRamondSphereSeries,
    UniformizedRamondSphereSeries,
    elliptically_prefactor_internal_ramond_series,
    elliptically_prefactor_two_ramond_internal_ns_series,
    uniformize_ramond_sphere_series,
)
from so7e8_liouville_chamber import (
    LiouvilleChamberCheck,
    require_conservative_real_p_chamber,
)
from so7e8_ramond_cocycles import RamondFamily
from so7e8_ramond_sphere_integrand import (
    CrossedBlockBackend,
    FixedPTwoRamondKernel,
    NSKind,
    build_fixed_p_two_ramond_kernel,
)
from so7e8_mixed_elliptic_recursion import MixedEllipticBlockSeries
from sphere_block_uniformization import CutSide
from so7e8_sphere_branches import sphere_channel_coordinates


@dataclass(frozen=True)
class CollisionConvergenceCheck:
    r"""Leading fixed-P=0 radial margins at the three pair collisions."""

    zero: float
    one: float
    infinity: float

    @property
    def minimum(self) -> float:
        return min(self.zero, self.one, self.infinity)

    @property
    def convergent(self) -> bool:
        return self.minimum > 0.0


@dataclass(frozen=True)
class TwoRamondLiouvilleValues:
    """Integrated SO(7) tensor coefficients and their six atlas pieces."""

    ns_at_z: NSKind
    ns_at_one: NSKind
    coefficients: Mapping[str, complex]
    pieces: Mapping[str, Mapping[str, complex]]


@dataclass(frozen=True)
class TwoRamondLiouvilleEvaluation:
    """One convergent-chamber evaluation with numerical diagnostics."""

    values: TwoRamondLiouvilleValues
    external_liouville_momenta: tuple[complex, complex, complex, complex]
    time_momenta: tuple[complex, complex, complex, complex]
    chamber: LiouvilleChamberCheck
    collision_convergence: CollisionConvergenceCheck
    maximum_twice_level: int
    momentum_nodes: int
    p_max: float
    elliptic_prefactor: bool
    crossed_block_backend: CrossedBlockBackend
    used_uncertified_direct_channel: bool
    build_seconds: float
    integration_seconds: float
    production_certified: bool = False
    picture_transport_certified: bool = False
    momentum_quadrature: Mapping[str, object] | None = None


@dataclass(frozen=True)
class _UniformizedTerm:
    pco_branch: str
    coefficient: complex
    holomorphic: (
        UniformizedRamondSphereSeries
        | EllipticPrefactoredRamondSphereSeries
        | MixedEllipticBlockSeries
    )
    antiholomorphic: (
        UniformizedRamondSphereSeries
        | EllipticPrefactoredRamondSphereSeries
        | MixedEllipticBlockSeries
    )


@dataclass(frozen=True)
class _UniformizedKernel:
    source: FixedPTwoRamondKernel
    terms: tuple[_UniformizedTerm, ...]

    def evaluate_many(self, z: np.ndarray, *, cut_side: CutSide = "upper") -> Mapping[str, np.ndarray]:
        """Evaluate all populated tensor coefficients on a geometric slice."""

        points = np.asarray(z, dtype=np.complex128)
        if points.ndim != 1:
            raise ValueError("the modulus grid must be one-dimensional")
        if np.any((points == 0.0) | (points == 1.0)):
            raise ValueError("quadrature nodes must avoid the punctures")
        frame = sphere_channel_coordinates(points, channel=self.source.channel, cut_side=cut_side)
        points, zbar, local, local_bar = frame.z, frame.zbar, frame.local, frame.local_bar
        scalar = np.zeros_like(points)
        for term in self.terms:
            holomorphic = np.asarray(
                term.holomorphic.value(local, cut_side=frame.holomorphic_cut_side),
                dtype=np.complex128,
            )
            antiholomorphic = np.asarray(
                term.antiholomorphic.value(local_bar, cut_side=frame.antiholomorphic_cut_side),
                dtype=np.complex128,
            )
            # At moving picture zero, K_psi(z) supplies z**(-1/2) only to
            # the psi0 branch.  At fixed picture zero, the moving picture
            # minus-one NS superghost supplies the same factor to both
            # branches.
            if term.pco_branch == "psi0" or self.source.picture_zero_at == "one":
                kpsi = np.exp(-0.5 * np.log(points))
            else:
                kpsi = 1.0
            scalar += term.coefficient * kpsi * holomorphic * antiholomorphic

        k1, k2, k3, _ = self.source.time_momenta
        time_factor = np.exp(
            -2.0 * k1 * k2 * np.log(np.abs(points))
            -2.0 * k2 * k3 * np.log(np.abs(1.0 - points))
        )
        common = time_factor * scalar
        species = (self.source.ns_at_z, self.source.ns_at_one)
        if species == ("S", "S"):
            return {"F_SS": common}
        if species == ("S", "V"):
            return {"F_SV": common / math.sqrt(2.0)}

        root = np.exp(-0.5 * np.log(zbar))
        if species == ("V", "S"):
            return {"F_VS": common * root / math.sqrt(2.0)}
        return {
            "F0": common
            * root
            * (1.0 + zbar)
            / (2.0 * (1.0 - zbar)),
            "F2": -0.5 * common * root,
        }


def collision_convergence_margins(
    external_liouville_momenta: Sequence[complex],
) -> CollisionConvergenceCheck:
    r"""Return ``-Re[(p_i+p_j)^2]`` in all three degeneration channels."""

    momenta = tuple(complex(value) for value in external_liouville_momenta)
    if len(momenta) != 4:
        raise ValueError("external_liouville_momenta must have length four")
    p1, p2, p3, _ = momenta
    return CollisionConvergenceCheck(
        zero=float(-((p1 + p2) ** 2).real),
        one=float(-((p2 + p3) ** 2).real),
        infinity=float(-((p1 + p3) ** 2).real),
    )


def _uniformize_kernel(
    kernel: FixedPTwoRamondKernel,
    *,
    elliptic_prefactor: bool,
) -> _UniformizedKernel:
    def populated(series: object) -> bool:
        if isinstance(series, (MixedEllipticBlockSeries, EllipticPrefactoredRamondSphereSeries)):
            return bool(series.h_coefficients) and any(series.h_coefficients)
        coefficients = getattr(series, "coefficients")
        return bool(coefficients) and any(coefficients.values())

    def adapt(series: object):
        if isinstance(series, EllipticPrefactoredRamondSphereSeries):
            return series if elliptic_prefactor else uniformize_ramond_sphere_series(series.source)
        if isinstance(series, MixedEllipticBlockSeries):
            return series
        if not elliptic_prefactor:
            adapter = uniformize_ramond_sphere_series
        elif kernel.channel == "direct":
            adapter = elliptically_prefactor_internal_ramond_series
        else:
            adapter = elliptically_prefactor_two_ramond_internal_ns_series
        return adapter(series)

    terms: list[_UniformizedTerm] = []
    for term in kernel.terms:
        # An unavailable parity component at a deliberately tiny cutoff is
        # identically zero and should not be sent to the nonempty-series API.
        if not populated(term.holomorphic_series):
            continue
        if not populated(term.antiholomorphic_series):
            continue
        terms.append(
            _UniformizedTerm(
                pco_branch=term.pco_branch,
                coefficient=complex(term.coefficient),
                holomorphic=adapt(term.holomorphic_series),
                antiholomorphic=adapt(term.antiholomorphic_series),
            )
        )
    return _UniformizedKernel(kernel, tuple(terms))


def _integrate_grid(
    kernel: _UniformizedKernel,
    z: np.ndarray,
    area_weights: np.ndarray,
) -> Mapping[str, complex]:
    values = kernel.evaluate_many(z)
    return {
        name: _stable_complex_sum(area_weights * coefficient)
        for name, coefficient in values.items()
    }


def _swapped_to_original(
    values: Mapping[str, complex],
    original_species: tuple[NSKind, NSKind],
) -> Mapping[str, complex]:
    if original_species == ("S", "S"):
        return {"F_SS": complex(values["F_SS"])}
    if original_species == ("S", "V"):
        # The single-vector Spin(7) intertwiner reverses orientation under
        # the NS2/NS3 inversion. The complete fixed-P densities obey this
        # sign after the ordered PCO and RN endpoint corrections.
        return {"F_SV": -complex(values["F_VS"])}
    if original_species == ("V", "S"):
        return {"F_VS": -complex(values["F_SV"])}
    return {
        "F0": complex(values["F0"]),
        # The swapped correlator is written with gamma^(ba).
        "F2": -complex(values["F2"]),
    }


def _add_weighted(
    accumulator: dict[str, list[complex]],
    values: Mapping[str, complex],
    weight: float,
) -> None:
    for name, value in values.items():
        accumulator.setdefault(name, []).append(float(weight) * complex(value))


def evaluate_two_ramond_liouville_convergent(
    external_liouville_momenta: Sequence[complex],
    *,
    ns_at_z: NSKind,
    ns_at_one: NSKind,
    zero_ramond_family: RamondFamily = "Psi_tilde",
    infinity_ramond_family: RamondFamily = "Psi",
    maximum_twice_level: int = 5,
    p_nodes: int | str | Sequence[int] = 4,
    p_max: float = 3.0,
    p_cut: float = 0.03,
    momentum_scheme: str = "cutoff",
    infinite_gauss_scale: float = 1.0,
    momentum_threshold_options=None,
    epsilon0: float = 0.08,
    epsilon1: float = 0.06,
    theta_orders: Sequence[int] = (8, 8, 16),
    radial_order: int = 10,
    disk_radial_order: int = 10,
    disk_angular_order: int = 24,
    disk_radial_power: float = 3.0,
    lens_radial_order: int = 10,
    lens_angular_order: int = 24,
    lens_radial_power: float = 3.0,
    digits: int = 40,
    condition_limit: float = 1.0e13,
    elliptic_prefactor: bool = True,
    crossed_block_backend: CrossedBlockBackend = "double_virasoro",
    allow_uncertified_direct_channel: bool = False,
) -> TwoRamondLiouvilleEvaluation:
    r"""Evaluate a two-R amplitude by the full Liouville spectral integral.

    This entry point intentionally refuses kinematics outside the strict
    ordinary-convergence chamber. By default it evaluates the locally audited
    crossed NS-internal decomposition on every atlas piece. This converges
    slowly near ``z=0`` but does not import the presently rejected direct
    internal-R nonchiral projector.  The direct channel remains available
    only through the explicitly named audit flag.

    Mixed coefficients default to the double-Virasoro branch sum and ordinary
    c-recursion, followed by elliptic-H truncation. ``inverse_gram`` is an
    explicit low-order benchmark option, not an automatic fallback.

    Select ``momentum_scheme='threshold_weighted'`` to cover the endpoint,
    bulk and infinite tail with a P**beta exp(-a*P**2+s*P) envelope. The
    thresholds/envelope can be varied with ``momentum_threshold_options``.
    Returned weights retain the original dP/pi measure. In this mode p_max
    is a compatibility argument, NOT a physical integration cutoff.

    This result is diagnostic, not a certified string amplitude. The folded
    full numerical convergence still requires independent checks; see
    SO7E8_RAMOND_EVALUATION_20260911.md.
    """

    momenta = tuple(complex(value) for value in external_liouville_momenta)
    if len(momenta) != 4:
        raise ValueError("external_liouville_momenta must contain (p1,p2,p3,p4)")
    p1, p2, p3, p4 = momenta
    chamber = require_conservative_real_p_chamber((p1, p2, p3), p4)
    collision = collision_convergence_margins(momenta)
    if not collision.convergent:
        raise ValueError(
            "ordinary crossed-channel integration requires all collision "
            "margins -Re[(pi+pj)^2] to be positive; "
            f"received minimum {collision.minimum:.6g}"
        )
    if maximum_twice_level < 1:
        raise ValueError("maximum_twice_level must be at least one")
    if not 0 < epsilon0 < 1 or not 0 < epsilon1 < 1:
        raise ValueError("epsilon0 and epsilon1 must lie in (0,1)")
    if epsilon0 + epsilon1 >= 1:
        raise ValueError("the zero disk and one lens must not overlap")
    if p_max <= 0:
        raise ValueError("p_max must be positive")

    times = (p1, p2, p3, -p4)
    swapped_momenta = (p1, p3, p2, p4)
    swapped_times = (p1, p3, p2, -p4)
    original_species: tuple[NSKind, NSKind] = (ns_at_z, ns_at_one)
    swapped_species: tuple[NSKind, NSKind] = (ns_at_one, ns_at_z)

    internal_momenta, momentum_weights = _momentum_quadrature(
        p_nodes,
        p_max,
        p_cut,
        scheme=momentum_scheme,
        infinite_gauss_scale=infinite_gauss_scale,
        momentum_threshold_options=momentum_threshold_options,
    )
    internal_momenta = np.asarray(internal_momenta, dtype=float)
    momentum_weights = np.asarray(momentum_weights, dtype=float) / math.pi

    bulk_z, bulk_weights, _, _ = fast.annulus_excluding_lens(
        epsilon0, epsilon1, theta_orders, radial_order
    )
    disk_z, disk_weights = fast.disk_grid(
        epsilon0,
        disk_radial_order,
        disk_angular_order,
        disk_radial_power,
    )
    lens_w, lens_weights = fast.ref._lens_grid(
        epsilon1,
        lens_radial_order,
        lens_angular_order,
        lens_radial_power,
    )
    lens_z = 1.0 - lens_w

    piece_names = (
        "original_bulk",
        "swapped_bulk",
        "original_z0",
        "swapped_z0",
        "original_z1",
        "swapped_z1",
    )
    accumulated: dict[str, dict[str, list[complex]]] = {
        name: {} for name in piece_names
    }
    build_seconds = 0.0
    integration_seconds = 0.0

    for internal, p_weight in zip(
        internal_momenta, momentum_weights, strict=True
    ):
        build_started = time.perf_counter()
        original_crossed = _uniformize_kernel(
            build_fixed_p_two_ramond_kernel(
                float(internal),
                external_liouville_momenta=momenta,
                time_momenta=times,
                ns_at_z=original_species[0],
                ns_at_one=original_species[1],
                zero_ramond_family=zero_ramond_family,
                infinity_ramond_family=infinity_ramond_family,
                channel="crossed",
                picture_zero_at="z",
                maximum_twice_level=maximum_twice_level,
                digits=digits,
                condition_limit=condition_limit,
                crossed_block_backend=crossed_block_backend,
            ),
            elliptic_prefactor=elliptic_prefactor,
        )
        swapped_crossed = _uniformize_kernel(
            build_fixed_p_two_ramond_kernel(
                float(internal),
                external_liouville_momenta=swapped_momenta,
                time_momenta=swapped_times,
                ns_at_z=swapped_species[0],
                ns_at_one=swapped_species[1],
                zero_ramond_family=zero_ramond_family,
                infinity_ramond_family=infinity_ramond_family,
                channel="crossed",
                picture_zero_at="one",
                maximum_twice_level=maximum_twice_level,
                digits=digits,
                condition_limit=condition_limit,
                crossed_block_backend=crossed_block_backend,
            ),
            elliptic_prefactor=elliptic_prefactor,
        )
        if allow_uncertified_direct_channel:
            original_direct = _uniformize_kernel(
                build_fixed_p_two_ramond_kernel(
                    float(internal),
                    external_liouville_momenta=momenta,
                    time_momenta=times,
                    ns_at_z=original_species[0],
                    ns_at_one=original_species[1],
                    zero_ramond_family=zero_ramond_family,
                    infinity_ramond_family=infinity_ramond_family,
                    channel="direct",
                    allow_uncertified_direct_projector=True,
                    picture_zero_at="z",
                    maximum_twice_level=maximum_twice_level,
                    digits=digits,
                    condition_limit=condition_limit,
                ),
                elliptic_prefactor=elliptic_prefactor,
            )
            swapped_direct = _uniformize_kernel(
                build_fixed_p_two_ramond_kernel(
                    float(internal),
                    external_liouville_momenta=swapped_momenta,
                    time_momenta=swapped_times,
                    ns_at_z=swapped_species[0],
                    ns_at_one=swapped_species[1],
                    zero_ramond_family=zero_ramond_family,
                    infinity_ramond_family=infinity_ramond_family,
                    channel="direct",
                    allow_uncertified_direct_projector=True,
                    # Preserve the physical picture-zero NS operator under
                    # the exterior NS2/NS3 exchange.
                    picture_zero_at="one",
                    maximum_twice_level=maximum_twice_level,
                    digits=digits,
                    condition_limit=condition_limit,
                ),
                elliptic_prefactor=elliptic_prefactor,
            )
        else:
            original_direct = original_crossed
            swapped_direct = swapped_crossed
        build_seconds += time.perf_counter() - build_started

        integration_started = time.perf_counter()
        original_values = (
            _integrate_grid(original_direct, bulk_z, bulk_weights),
            _integrate_grid(original_direct, disk_z, disk_weights),
            _integrate_grid(original_crossed, lens_z, lens_weights),
        )
        swapped_values = tuple(
            _swapped_to_original(values, original_species)
            for values in (
                _integrate_grid(swapped_direct, bulk_z, bulk_weights),
                _integrate_grid(swapped_direct, disk_z, disk_weights),
                _integrate_grid(swapped_crossed, lens_z, lens_weights),
            )
        )
        integration_seconds += time.perf_counter() - integration_started

        for name, values in zip(
            ("original_bulk", "original_z0", "original_z1"),
            original_values,
            strict=True,
        ):
            _add_weighted(accumulated[name], values, float(p_weight))
        for name, values in zip(
            ("swapped_bulk", "swapped_z0", "swapped_z1"),
            swapped_values,
            strict=True,
        ):
            _add_weighted(accumulated[name], values, float(p_weight))

    pieces: dict[str, dict[str, complex]] = {}
    for piece_name, tensor_terms in accumulated.items():
        pieces[piece_name] = {
            tensor_name: _stable_complex_fsum(terms)
            for tensor_name, terms in tensor_terms.items()
        }
    tensor_names = tuple(next(iter(pieces.values())).keys())
    coefficients = {
        tensor_name: _stable_complex_fsum(
            pieces[piece][tensor_name] for piece in piece_names
        )
        for tensor_name in tensor_names
    }
    return TwoRamondLiouvilleEvaluation(
        values=TwoRamondLiouvilleValues(
            ns_at_z=original_species[0],
            ns_at_one=original_species[1],
            coefficients=coefficients,
            pieces=pieces,
        ),
        external_liouville_momenta=momenta,  # type: ignore[arg-type]
        time_momenta=times,  # type: ignore[arg-type]
        chamber=chamber,
        collision_convergence=collision,
        maximum_twice_level=maximum_twice_level,
        momentum_nodes=len(internal_momenta),
        p_max=float(p_max),
        elliptic_prefactor=bool(elliptic_prefactor),
        crossed_block_backend=crossed_block_backend,
        used_uncertified_direct_channel=bool(allow_uncertified_direct_channel),
        build_seconds=build_seconds,
        integration_seconds=integration_seconds,
        momentum_quadrature=(
            fast.ref.threshold_weighted_rule(p_nodes, momentum_threshold_options).metadata()
            if momentum_scheme == "threshold_weighted" else
            {"scheme": momentum_scheme, "node_count": len(internal_momenta)}
        ),
    )


__all__ = [
    "CollisionConvergenceCheck",
    "TwoRamondLiouvilleEvaluation",
    "TwoRamondLiouvilleValues",
    "collision_convergence_margins",
    "evaluate_two_ramond_liouville_convergent",
]
