#!/usr/bin/env python3
"""Checks for the explicit genus-two ``c=1`` string integrand."""

from __future__ import annotations

import json
import math
import sys
import warnings
from pathlib import Path

import numpy as np

try:
    from conformal_frame_labels import (
        GLASSES_PLUMBING_FRAME,
        THETA_PLUMBING_FRAME,
        UNIT_AREA_BERGMAN_FRAME,
    )
    from free_boson_pair_of_pants import (
        glasses_heisenberg_plumbing_partition,
        theta_heisenberg_plumbing_partition,
    )
    from free_boson_plumbing import (
        bergman_scalar_partition_candidate,
        noncompact_scalar_loop_momentum_factor,
        noncompact_scalar_zero_mode_factor,
        xi_noncompact_scalar_loop_momentum_factor,
    )
    from genus2_bc_ghost import canonical_bc_ghost_density
    from genus2_c1_string_integrand import (
        DIMENSIONLESS_SEWING_SCALAR_NORMALIZATION,
        SameFrameMatterPartitions,
        XI_PHYSICAL_MOMENTUM_SCALAR_NORMALIZATION,
        compact_boson_winding_evaluation_genus2,
        compact_boson_winding_sum_genus2,
        genus2_c1_string_integrand_density,
        genus2_c1_string_integrand_density_from_geometry,
        genus2_c1_string_integrand_geometry,
        plumbing_matter_to_bergman,
    )
    from genus2_integrand_normalization import (
        RAW_PRODUCT_FACTORIZATION_NORMALIZATION,
        STRING_NOTE_INTEGRATION_KERNEL_CONVENTION,
        c1_sphere_normalized_genus2_kernel_multiplier,
        xi_compact_target_zero_mode,
        xi_full_replacement_over_dimensionless,
        xi_genus2_scalar_over_dimensionless,
    )
except ImportError:  # pragma: no cover - supports package-style execution
    from plumbing.conformal_frame_labels import (
        GLASSES_PLUMBING_FRAME,
        THETA_PLUMBING_FRAME,
        UNIT_AREA_BERGMAN_FRAME,
    )
    from plumbing.free_boson_pair_of_pants import (
        glasses_heisenberg_plumbing_partition,
        theta_heisenberg_plumbing_partition,
    )
    from plumbing.free_boson_plumbing import (
        bergman_scalar_partition_candidate,
        noncompact_scalar_loop_momentum_factor,
        noncompact_scalar_zero_mode_factor,
        xi_noncompact_scalar_loop_momentum_factor,
    )
    from plumbing.genus2_bc_ghost import canonical_bc_ghost_density
    from plumbing.genus2_c1_string_integrand import (
        DIMENSIONLESS_SEWING_SCALAR_NORMALIZATION,
        SameFrameMatterPartitions,
        XI_PHYSICAL_MOMENTUM_SCALAR_NORMALIZATION,
        compact_boson_winding_evaluation_genus2,
        compact_boson_winding_sum_genus2,
        genus2_c1_string_integrand_density,
        genus2_c1_string_integrand_density_from_geometry,
        genus2_c1_string_integrand_geometry,
        plumbing_matter_to_bergman,
    )
    from plumbing.genus2_integrand_normalization import (
        RAW_PRODUCT_FACTORIZATION_NORMALIZATION,
        STRING_NOTE_INTEGRATION_KERNEL_CONVENTION,
        c1_sphere_normalized_genus2_kernel_multiplier,
        xi_compact_target_zero_mode,
        xi_full_replacement_over_dimensionless,
        xi_genus2_scalar_over_dimensionless,
    )


OMEGA = np.asarray(
    [[0.12 + 1.15j, 0.08 + 0.04j], [0.08 + 0.04j, -0.09 + 0.92j]],
    dtype=np.complex128,
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _relative_error(value: float, target: float) -> float:
    return abs(float(value) - float(target)) / max(abs(float(target)), 1.0e-300)


def _period_matrix(data: dict[str, object], key: str) -> np.ndarray:
    rows = data[key]
    return np.asarray([[complex(value) for value in row] for row in rows], dtype=np.complex128)


def check_weyl_factor_cancellation() -> None:
    canonical_scalar = 0.73
    canonical_liouville = 1.17
    frame_factors = (0.0416214, 1.37)
    canonical_matter = SameFrameMatterPartitions(
        conformal_frame=UNIT_AREA_BERGMAN_FRAME,
        liouville_partition=canonical_liouville,
        noncompact_scalar_partition=canonical_scalar,
    )
    canonical = genus2_c1_string_integrand_density(
        OMEGA,
        1.2,
        matter_partitions=canonical_matter,
        lattice_nmax=7,
        theta_nmax=10,
    )

    print("algebraic plumbing-frame cancellation")
    for index, frame_factor in enumerate(frame_factors):
        plumbing_scalar = frame_factor * canonical_scalar
        plumbing_liouville = frame_factor**25 * canonical_liouville
        plumbing_matter = SameFrameMatterPartitions(
            conformal_frame=THETA_PLUMBING_FRAME if index == 0 else GLASSES_PLUMBING_FRAME,
            liouville_partition=plumbing_liouville,
            noncompact_scalar_partition=plumbing_scalar,
        )
        conversion = plumbing_matter_to_bergman(
            plumbing_matter,
            bergman_scalar_partition=canonical_scalar,
        )
        plumbing = genus2_c1_string_integrand_density(
            OMEGA,
            1.2,
            matter_partitions=plumbing_matter,
            lattice_nmax=7,
            theta_nmax=10,
        )
        reconstructed = genus2_c1_string_integrand_density(
            OMEGA,
            1.2,
            matter_partitions=conversion.bergman_matter,
            lattice_nmax=7,
            theta_nmax=10,
        )
        error = _relative_error(plumbing.density, canonical.density)
        print(f"  W={frame_factor:.7g}: relative density error = {error:.6e}")
        _require(error < 5.0e-14, "the scalar Weyl factor did not cancel")
        _require(
            _relative_error(reconstructed.density, plumbing.density) < 5.0e-14,
            "explicit Bergman reconstruction disagrees with the plumbing quotient",
        )
        _require(conversion.quotient_relative_error < 5.0e-14, "matter quotient changed frames")

    try:
        SameFrameMatterPartitions(
            conformal_frame="unspecified",  # type: ignore[arg-type]
            liouville_partition=1.0,
            noncompact_scalar_partition=1.0,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("an unspecified matter frame was accepted")


def check_shared_integrand_geometry() -> None:
    """Low/high matter quotients must reuse one unchanged geometry object."""

    matter_low = SameFrameMatterPartitions(
        conformal_frame=THETA_PLUMBING_FRAME,
        liouville_partition=1.17,
        noncompact_scalar_partition=0.73,
    )
    matter_high = SameFrameMatterPartitions(
        conformal_frame=THETA_PLUMBING_FRAME,
        liouville_partition=1.19,
        noncompact_scalar_partition=0.731,
    )
    geometry = genus2_c1_string_integrand_geometry(
        OMEGA,
        1.2,
        lattice_nmax=7,
        theta_nmax=10,
    )
    low = genus2_c1_string_integrand_density_from_geometry(
        geometry,
        matter_partitions=matter_low,
    )
    high = genus2_c1_string_integrand_density_from_geometry(
        geometry,
        matter_partitions=matter_high,
    )
    direct = genus2_c1_string_integrand_density(
        OMEGA,
        1.2,
        matter_partitions=matter_low,
        lattice_nmax=7,
        theta_nmax=10,
    )
    _require(low.log_density == direct.log_density, "shared geometry changed the density")
    _require(low.chi10 == high.chi10, "low/high matter changed the shared chi10")
    _require(
        low.compact_winding_sum == high.compact_winding_sum,
        "low/high matter changed the shared compact lattice sum",
    )
    expected_log_ratio = (
        math.log(matter_high.liouville_partition / matter_low.liouville_partition)
        - 25.0
        * math.log(
            matter_high.noncompact_scalar_partition
            / matter_low.noncompact_scalar_partition
        )
    )
    _require(
        abs((high.log_density - low.log_density) - expected_log_ratio) < 2.0e-14,
        "shared geometry did not isolate the low/high matter quotient",
    )


def check_xi_free_scalar_normalization() -> None:
    r"""Check Xi's ``dk/(2 pi)`` measure and the correlated replacement."""

    alpha_prime = 1.7
    radius = 1.2
    scalar_dimensionless = 0.73
    liouville = 1.17
    scalar_xi = (
        scalar_dimensionless
        * xi_genus2_scalar_over_dimensionless(alpha_prime)
    )

    loop_dimensionless = noncompact_scalar_loop_momentum_factor(OMEGA)
    loop_xi = xi_noncompact_scalar_loop_momentum_factor(
        OMEGA,
        alpha_prime=alpha_prime,
    )
    _require(
        _relative_error(
            loop_xi,
            loop_dimensionless / (4.0 * math.pi**2 * alpha_prime),
        )
        < 2.0e-15,
        "Xi's genus-two scalar loop measure has the wrong normalization",
    )

    dimensionless_input = genus2_c1_string_integrand_density(
        OMEGA,
        radius,
        alpha_prime=alpha_prime,
        matter_partitions=SameFrameMatterPartitions(
            conformal_frame=UNIT_AREA_BERGMAN_FRAME,
            liouville_partition=liouville,
            noncompact_scalar_partition=scalar_dimensionless,
            noncompact_scalar_normalization=(
                DIMENSIONLESS_SEWING_SCALAR_NORMALIZATION
            ),
        ),
        lattice_nmax=7,
        theta_nmax=10,
    )
    xi_input = genus2_c1_string_integrand_density(
        OMEGA,
        radius,
        alpha_prime=alpha_prime,
        matter_partitions=SameFrameMatterPartitions(
            conformal_frame=UNIT_AREA_BERGMAN_FRAME,
            liouville_partition=liouville,
            noncompact_scalar_partition=scalar_xi,
            noncompact_scalar_normalization=(
                XI_PHYSICAL_MOMENTUM_SCALAR_NORMALIZATION
            ),
        ),
        lattice_nmax=7,
        theta_nmax=10,
    )
    _require(
        _relative_error(dimensionless_input.density, xi_input.density) < 2.0e-14,
        "dimensionless and Xi-normalized scalar inputs give different densities",
    )
    _require(
        _relative_error(
            dimensionless_input.noncompact_scalar_partition,
            scalar_xi,
        )
        < 2.0e-15,
        "the integrand did not expose Xi's scalar partition",
    )

    old_log_density = (
        2.0 * math.log(2.0 * math.pi)
        - 13.0 * math.log(dimensionless_input.det_im_omega)
        - 2.0 * dimensionless_input.chi10_log_abs
        + math.log(radius * dimensionless_input.compact_winding_sum)
        + math.log(liouville)
        - 25.0 * math.log(scalar_dimensionless)
    )
    observed_net_factor = math.exp(
        dimensionless_input.log_density - old_log_density
    )
    expected_net_factor = xi_full_replacement_over_dimensionless(alpha_prime)
    print("\nXi free-scalar zero-mode normalization")
    print(f"  Z_X^Xi / Z_X^p       = {scalar_xi / scalar_dimensionless:.16e}")
    print(f"  full Xi/old factor   = {observed_net_factor:.16e}")
    print(f"  expected             = {expected_net_factor:.16e}")
    _require(
        _relative_error(observed_net_factor, expected_net_factor) < 3.0e-14,
        "the correlated Xi scalar conversion does not reduce to 1/(2 pi sqrt(alpha'))",
    )
    _require(
        _relative_error(
            dimensionless_input.compact_target_zero_mode,
            2.0 * math.pi * math.sqrt(alpha_prime) * radius,
        )
        < 2.0e-15,
        "the Xi compact target zero mode is not 2 pi R_phys",
    )
    _require(
        _relative_error(
            dimensionless_input.string_note_kernel_multiplier,
            64.0 * math.pi**4,
        )
        < 2.0e-15,
        "the final c=1 kernel did not apply the complete 64*pi^4 multiplier",
    )


def check_factorization_normalization_exposure() -> None:
    matter = SameFrameMatterPartitions(
        conformal_frame=UNIT_AREA_BERGMAN_FRAME,
        liouville_partition=1.17,
        noncompact_scalar_partition=0.73,
    )
    product = genus2_c1_string_integrand_density(
        OMEGA,
        1.2,
        matter_partitions=matter,
        lattice_nmax=7,
        theta_nmax=10,
        chi10_normalization="product",
    )
    normalized_cusp = genus2_c1_string_integrand_density(
        OMEGA,
        1.2,
        matter_partitions=matter,
        lattice_nmax=7,
        theta_nmax=10,
        chi10_normalization="string_note_2^-12",
    )

    print("\nseparating Mumford-residue normalization")
    print(f"  raw product multiplier       = {product.factorization_normalization:.16e}")
    print(
        "  normalized/raw density      = "
        f"{product.factorization_normalized_density / product.density:.16e}"
    )
    print(
        "  cusp-convention agreement   = "
        f"{normalized_cusp.density / product.factorization_normalized_density:.16e}"
    )
    print(
        "  string-note kernel/local CFT= "
        f"{product.string_note_kernel_density / product.factorization_normalized_density:.16e}"
    )
    _require(
        product.factorization_normalization == RAW_PRODUCT_FACTORIZATION_NORMALIZATION,
        "the raw theta-product integrand has the wrong Mumford-residue multiplier",
    )
    _require(
        _relative_error(
            product.factorization_normalized_density,
            RAW_PRODUCT_FACTORIZATION_NORMALIZATION * product.density,
        )
        < 2.0e-14,
        "the exposed Mumford-residue-normalized density is inconsistent",
    )
    _require(
        _relative_error(normalized_cusp.density, product.factorization_normalized_density) < 2.0e-13,
        "the normalized cusp form and explicit product multiplier disagree",
    )
    _require(
        normalized_cusp.factorization_normalization == 1.0,
        "the Mumford-residue-normalized cusp form retained an extra multiplier",
    )
    _require(
        product.string_note_kernel_convention
        == normalized_cusp.string_note_kernel_convention
        == STRING_NOTE_INTEGRATION_KERNEL_CONVENTION,
        "the string-note kernel convention is not explicit",
    )
    _require(
        _relative_error(
            product.string_note_kernel_density,
            c1_sphere_normalized_genus2_kernel_multiplier()
            * product.factorization_normalized_density,
        )
        < 2.0e-14,
        "the string-note real-measure multiplier is missing from the integrand",
    )


def check_compact_sum_in_saved_markings() -> None:
    period_path = Path("plumbing/results/theta_glasses_period_precision_w8_radial_q01500.json")
    data = json.loads(period_path.read_text())
    omega_glasses = _period_matrix(data, "omega_glasses")
    omega_theta = _period_matrix(data, "omega_theta_target")

    print("\ncompact winding sum in the saved symplectic markings")
    for radius in (0.8, 1.0, 1.3):
        glasses = compact_boson_winding_sum_genus2(omega_glasses, radius, lattice_nmax=12)
        theta = compact_boson_winding_sum_genus2(omega_theta, radius, lattice_nmax=12)
        error = _relative_error(theta, glasses)
        print(f"  R={radius:.1f}: theta/glasses={theta / glasses:.16e}, error={error:.6e}")
        _require(error < 2.0e-12, "compact winding sum is inconsistent between markings")


def check_genus2_t_duality_scaling() -> None:
    radius = 1.25
    direct = radius * compact_boson_winding_sum_genus2(OMEGA, radius, lattice_nmax=12)
    dual_radius = 1.0 / radius
    dual = dual_radius * compact_boson_winding_sum_genus2(OMEGA, dual_radius, lattice_nmax=12)
    actual_ratio = direct / dual
    expected_ratio = radius**-2
    error = _relative_error(actual_ratio, expected_ratio)

    print("\ngenus-two compact-boson T-duality scaling")
    print(f"  Z(R) / Z(1/R) = {actual_ratio:.16e}")
    print(f"  expected R^(-2) = {expected_ratio:.16e}")
    print(f"  relative error   = {error:.6e}")
    _require(error < 2.0e-12, "the compact zero-mode factor has the wrong genus-two T-duality weight")


def check_poisson_resummation() -> None:
    print("\nPoisson-resummed compact-boson theta function")
    for radius in (0.8, 1.0, 1.3, 2.0):
        direct = compact_boson_winding_sum_genus2(
            OMEGA,
            radius,
            lattice_nmax=14,
            algorithm="direct",
        )
        poisson = compact_boson_winding_evaluation_genus2(
            OMEGA,
            radius,
            tolerance=1.0e-14,
            algorithm="poisson",
        )
        error = _relative_error(poisson.value, direct)
        print(
            f"  R={radius:.1f}: direct/resummed error={error:.6e}, "
            f"cutoffs=({poisson.momentum_nmax},{poisson.winding_nmax})"
        )
        _require(error < 2.0e-12, "Poisson resummation disagrees with the direct theta sum")

    deep_cusp = np.asarray(
        [
            [0.13 + 4.4j, 0.08 + 0.03j],
            [0.08 + 0.03j, -0.11 + 1.411039e6j],
        ],
        dtype=np.complex128,
    )
    radius = 1.6
    direct_radius = compact_boson_winding_evaluation_genus2(
        deep_cusp,
        radius,
        tolerance=1.0e-13,
    )
    dual_radius = compact_boson_winding_evaluation_genus2(
        deep_cusp,
        1.0 / radius,
        tolerance=1.0e-13,
    )
    actual_ratio = radius * direct_radius.value / ((1.0 / radius) * dual_radius.value)
    duality_error = _relative_error(actual_ratio, radius**-2)
    print(
        "  deep cusp: "
        f"algorithm={direct_radius.algorithm}, "
        f"cutoffs=({direct_radius.momentum_nmax},{direct_radius.winding_nmax}), "
        f"T-duality error={duality_error:.6e}"
    )
    _require(
        direct_radius.algorithm == dual_radius.algorithm == "poisson",
        "the automatic evaluator did not route the deep cusp to Poisson resummation",
    )
    _require(
        max(
            direct_radius.momentum_nmax,
            direct_radius.winding_nmax,
            dual_radius.momentum_nmax,
            dual_radius.winding_nmax,
        )
        <= 6,
        "the resummed deep-cusp cutoff is unexpectedly large",
    )
    _require(duality_error < 2.0e-12, "the resummed theta sum violates T-duality")

    # This is the extreme nonseparating-cusp period matrix from production
    # sample r004-c01-p00000118.  Diagonalizing its assembled 4x4 winding
    # quadratic form is ill-conditioned and produced a platform-dependent
    # negative eigenvalue on the cluster even though Im(Omega) is positive.
    production_cusp = np.asarray(
        [
            [
                0.28335479367524385 + 1.112566439249399j,
                -0.33641934487968683 + 0.1718225172857308j,
            ],
            [
                -0.33641934487968683 + 0.1718225172857308j,
                0.29582383669912815 + 6896429848.499076j,
            ],
        ],
        dtype=np.complex128,
    )
    production_evaluation = compact_boson_winding_evaluation_genus2(
        production_cusp,
        1.0,
        tolerance=1.0e-12,
    )
    print(
        "  production cusp: "
        f"algorithm={production_evaluation.algorithm}, "
        f"cutoffs=({production_evaluation.momentum_nmax},"
        f"{production_evaluation.winding_nmax}), "
        f"value={production_evaluation.value:.16e}"
    )
    _require(
        production_evaluation.algorithm == "poisson",
        "the production cusp was not routed to Poisson resummation",
    )
    _require(
        max(
            production_evaluation.momentum_nmax,
            production_evaluation.winding_nmax,
        )
        <= 6,
        "the production-cusp Poisson cutoff is unexpectedly large",
    )
    _require(
        math.isfinite(production_evaluation.value) and production_evaluation.value > 0.0,
        "the production-cusp compact-boson theta value is not positive and finite",
    )

    # This R=15 production geometry previously evaluated phases for terms
    # whose winding weights had already underflowed to zero.  The value was
    # finite, but NumPy emitted misleading matmul overflow warnings.
    large_radius_cusp = np.asarray(
        [
            [
                0.36468237 + 1.20736049j,
                -0.29718603 + 0.357496592j,
            ],
            [
                -0.29718603 + 0.357496592j,
                0.06395721 + 12712.1738j,
            ],
        ],
        dtype=np.complex128,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        large_radius_evaluation = compact_boson_winding_evaluation_genus2(
            large_radius_cusp,
            15.0,
            tolerance=1.0e-13,
            algorithm="poisson",
        )
    _require(
        math.isfinite(large_radius_evaluation.value)
        and large_radius_evaluation.value > 0.0,
        "the R=15 deep-cusp compact-boson theta value is not positive and finite",
    )


def check_saved_full_measure_ratio() -> None:
    period_data = json.loads(
        Path("plumbing/results/theta_glasses_period_precision_w8_radial_q01500.json").read_text()
    )
    integration_data = json.loads(
        Path(
            "plumbing/results/theta_glasses_precision_w8_radial_q01500_block12_gaussian_q12.json"
        ).read_text()
    )
    omega_glasses = _period_matrix(period_data, "omega_glasses")
    omega_theta = _period_matrix(period_data, "omega_theta_target")

    unit_bergman_matter = SameFrameMatterPartitions(
        conformal_frame=UNIT_AREA_BERGMAN_FRAME,
        liouville_partition=1.0,
        noncompact_scalar_partition=1.0,
    )
    glasses = genus2_c1_string_integrand_density(
        omega_glasses,
        1.0,
        matter_partitions=unit_bergman_matter,
        lattice_nmax=12,
        theta_nmax=12,
    )
    theta = genus2_c1_string_integrand_density(
        omega_theta,
        1.0,
        matter_partitions=unit_bergman_matter,
        lattice_nmax=12,
        theta_nmax=12,
    )

    corrected_matter_ratio = float(integration_data["rows"][0]["full_free_boson_corrected"])
    automorphy = math.sqrt(glasses.det_im_omega / theta.det_im_omega)
    period_volume_ratio = automorphy**-6
    common_factor_ratio = (
        theta.critical_mumford_density
        * theta.compact_lattice_factor
        / (glasses.critical_mumford_density * glasses.compact_lattice_factor)
    )
    full_measure_ratio = corrected_matter_ratio * common_factor_ratio * period_volume_ratio

    print("\nsaved theta/glasses full measure comparison")
    print(f"  corrected matter ratio       = {corrected_matter_ratio:.16e}")
    print(f"  common density ratio         = {common_factor_ratio:.16e}")
    print(f"  period-volume Jacobian       = {period_volume_ratio:.16e}")
    print(f"  full modular measure ratio   = {full_measure_ratio:.16e}")
    _require(
        _relative_error(common_factor_ratio * period_volume_ratio, 1.0) < 2.0e-10,
        "the common compact/Mumford factors did not cancel with the period-volume Jacobian",
    )
    _require(
        _relative_error(full_measure_ratio, corrected_matter_ratio) < 2.0e-10,
        "assembling the full measure changed the period-matched channel ratio",
    )
    _require(
        abs(full_measure_ratio - 1.0) < 2.0e-3,
        "saved full-measure comparison is outside its error budget",
    )


def check_saved_explicit_bergman_reconstruction() -> None:
    period_data = json.loads(
        Path("plumbing/results/theta_glasses_period_precision_w8_radial_q01500.json").read_text()
    )
    integration_data = json.loads(
        Path(
            "plumbing/results/theta_glasses_precision_w8_radial_q01500_block12_gaussian_q12.json"
        ).read_text()
    )
    omega_common = _period_matrix(period_data, "omega_glasses")
    omega_theta_forward = _period_matrix(period_data, "omega_theta_forward")
    glasses_q = tuple(
        complex(period_data[key])
        for key in ("glasses_q_left", "glasses_q_right", "glasses_q_bridge")
    )
    theta_q = tuple(
        complex(period_data[key])
        for key in ("theta_q1", "theta_q2", "theta_q3")
    )
    direct_level = int(integration_data["direct_sewing_level"])
    glasses_oscillator = glasses_heisenberg_plumbing_partition(
        *glasses_q,
        max_total_level=direct_level,
    ).nonchiral_value
    theta_oscillator = theta_heisenberg_plumbing_partition(
        *theta_q,
        max_total_level=direct_level,
    ).nonchiral_value
    glasses_scalar_plumbing = glasses_oscillator * noncompact_scalar_zero_mode_factor(omega_common)
    theta_scalar_plumbing = theta_oscillator * noncompact_scalar_zero_mode_factor(omega_theta_forward)
    row = integration_data["rows"][0]
    glasses_matter_plumbing = SameFrameMatterPartitions(
        conformal_frame=GLASSES_PLUMBING_FRAME,
        liouville_partition=float(row["glasses_abs"]),
        noncompact_scalar_partition=glasses_scalar_plumbing,
    )
    theta_matter_plumbing = SameFrameMatterPartitions(
        conformal_frame=THETA_PLUMBING_FRAME,
        liouville_partition=float(row["theta_abs"]),
        noncompact_scalar_partition=theta_scalar_plumbing,
    )

    bergman_candidate = bergman_scalar_partition_candidate(
        omega_common,
        theta_nmax=12,
    ).partition_candidate
    glasses_conversion = plumbing_matter_to_bergman(
        glasses_matter_plumbing,
        bergman_scalar_partition=bergman_candidate,
    )
    theta_conversion = plumbing_matter_to_bergman(
        theta_matter_plumbing,
        bergman_scalar_partition=bergman_candidate,
    )
    glasses_canonical = genus2_c1_string_integrand_density(
        omega_common,
        1.0,
        matter_partitions=glasses_conversion.bergman_matter,
        lattice_nmax=12,
        theta_nmax=12,
    )
    theta_canonical = genus2_c1_string_integrand_density(
        omega_common,
        1.0,
        matter_partitions=theta_conversion.bergman_matter,
        lattice_nmax=12,
        theta_nmax=12,
    )
    canonical_ratio = theta_canonical.density / glasses_canonical.density
    saved_ratio = float(row["full_free_boson_corrected"])

    ghost = canonical_bc_ghost_density(
        omega_common,
        bergman_determinant_constant=1.0,
        theta_nmax=12,
    )
    winding_sum = compact_boson_winding_sum_genus2(omega_common, 1.0, lattice_nmax=12)
    compact_bergman = (
        xi_compact_target_zero_mode(1.0)
        * xi_genus2_scalar_over_dimensionless()
        * ghost.scalar_partition_without_target_zero_mode
        * winding_sum
    )
    glasses_component_product = (
        ghost.bc_ghost_density
        * compact_bergman
        * glasses_conversion.bergman_matter.liouville_partition
    )
    theta_component_product = (
        ghost.bc_ghost_density
        * compact_bergman
        * theta_conversion.bergman_matter.liouville_partition
    )
    glasses_ghost_plumbing = (
        ghost.bc_ghost_density
        / glasses_conversion.plumbing_over_bergman_scalar_factor**26
    )
    theta_ghost_plumbing = (
        ghost.bc_ghost_density
        / theta_conversion.plumbing_over_bergman_scalar_factor**26
    )
    glasses_plumbing_product = (
        glasses_ghost_plumbing
        * (
            xi_compact_target_zero_mode(1.0)
            * xi_genus2_scalar_over_dimensionless()
            * glasses_scalar_plumbing
            * winding_sum
        )
        * glasses_matter_plumbing.liouville_partition
    )
    theta_plumbing_product = (
        theta_ghost_plumbing
        * (
            xi_compact_target_zero_mode(1.0)
            * xi_genus2_scalar_over_dimensionless()
            * theta_scalar_plumbing
            * winding_sum
        )
        * theta_matter_plumbing.liouville_partition
    )

    print("\nsaved explicit conversion to one Bergman frame")
    print(f"  common Z_X^B (C_B=1) = {bergman_candidate:.16e}")
    print(f"  Z_X^pl glasses        = {glasses_scalar_plumbing:.16e}")
    print(f"  Z_X^pl theta          = {theta_scalar_plumbing:.16e}")
    print(
        "  W_pl/B glasses,theta = "
        f"{glasses_conversion.plumbing_over_bergman_scalar_factor:.16e}, "
        f"{theta_conversion.plumbing_over_bergman_scalar_factor:.16e}"
    )
    print(
        "  Z_L^B glasses,theta   = "
        f"{glasses_conversion.bergman_matter.liouville_partition:.16e}, "
        f"{theta_conversion.bergman_matter.liouville_partition:.16e}"
    )
    print(f"  G_bc^B                 = {ghost.bc_ghost_density:.16e}")
    print(
        "  G_bc^pl glasses,theta = "
        f"{glasses_ghost_plumbing:.16e}, {theta_ghost_plumbing:.16e}"
    )
    print(f"  canonical theta/glasses = {canonical_ratio:.16e}")
    print(f"  saved corrected ratio    = {saved_ratio:.16e}")
    _require(
        _relative_error(bergman_candidate, ghost.scalar_partition_without_target_zero_mode) < 2.0e-14,
        "ghost and scalar modules do not use the same Bergman determinant",
    )
    _require(
        ghost.conformal_frame == glasses_conversion.bergman_matter.conformal_frame
        == theta_conversion.bergman_matter.conformal_frame
        == UNIT_AREA_BERGMAN_FRAME,
        "ghost, scalar, and reconstructed Liouville factors do not share the unit-area Bergman label",
    )
    _require(
        _relative_error(glasses_component_product, glasses_canonical.density) < 2.0e-13,
        "glasses equation (72) factors are not in one Bergman frame",
    )
    _require(
        _relative_error(theta_component_product, theta_canonical.density) < 2.0e-13,
        "theta equation (72) factors are not in one Bergman frame",
    )
    _require(
        _relative_error(glasses_plumbing_product, glasses_component_product) < 2.0e-13,
        "glasses ghost Weyl factor has the wrong direction or central-charge power",
    )
    _require(
        _relative_error(theta_plumbing_product, theta_component_product) < 2.0e-13,
        "theta ghost Weyl factor has the wrong direction or central-charge power",
    )
    _require(
        _relative_error(canonical_ratio, saved_ratio) < 5.0e-12,
        "explicit Bergman reconstruction disagrees with the saved channel comparison",
    )

    for determinant_constant in (0.37, 1.0, 2.4):
        scalar_bergman = bergman_candidate / math.sqrt(determinant_constant)
        glasses_shifted = plumbing_matter_to_bergman(
            glasses_matter_plumbing,
            bergman_scalar_partition=scalar_bergman,
        )
        shifted_density = genus2_c1_string_integrand_density(
            omega_common,
            1.0,
            matter_partitions=glasses_shifted.bergman_matter,
            lattice_nmax=12,
            theta_nmax=12,
        ).density
        shifted_ghost = canonical_bc_ghost_density(
            omega_common,
            bergman_determinant_constant=determinant_constant,
            theta_nmax=12,
        )
        shifted_component_product = (
            shifted_ghost.bc_ghost_density
            * (
                xi_compact_target_zero_mode(1.0)
                * xi_genus2_scalar_over_dimensionless()
                * shifted_ghost.scalar_partition_without_target_zero_mode
                * winding_sum
            )
            * glasses_shifted.bergman_matter.liouville_partition
        )
        _require(
            _relative_error(shifted_density, glasses_canonical.density) < 2.0e-13,
            "the unknown Bergman determinant constant did not cancel",
        )
        _require(
            _relative_error(shifted_component_product, glasses_canonical.density) < 2.0e-13,
            "the same-frame ghost, compact boson, and Liouville product retained C_B dependence",
        )
        _require(
            shifted_ghost.conformal_frame == UNIT_AREA_BERGMAN_FRAME,
            "changing C_B changed the declared ghost conformal frame",
        )


def main() -> None:
    check_weyl_factor_cancellation()
    check_shared_integrand_geometry()
    check_xi_free_scalar_normalization()
    check_factorization_normalization_exposure()
    check_compact_sum_in_saved_markings()
    check_genus2_t_duality_scaling()
    check_poisson_resummation()
    check_saved_full_measure_ratio()
    check_saved_explicit_bergman_reconstruction()
    print("\ngenus-two c=1 string integrand checks passed")


if __name__ == "__main__":
    if sys.argv[1:] == ["--compact-theta-only"]:
        check_poisson_resummation()
        print("\ncompact-boson theta checks passed")
    else:
        main()
