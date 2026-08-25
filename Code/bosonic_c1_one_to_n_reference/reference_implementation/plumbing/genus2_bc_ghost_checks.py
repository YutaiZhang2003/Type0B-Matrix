#!/usr/bin/env python3
"""Checks for the canonical genus-two ``bc`` ghost density."""

from __future__ import annotations

import json
import math
from pathlib import Path

import mpmath as mp
import numpy as np

try:
    from conformal_frame_labels import UNIT_AREA_BERGMAN_FRAME
    from genus2_bc_ghost import canonical_bc_ghost_density, period_volume_jacobian_abs_squared
    from liouville_genus2_modular_check import (
        SymplecticTransform,
        named_transform,
        sp4_generator_transforms,
    )
except ImportError:  # pragma: no cover - supports package-style execution
    from plumbing.conformal_frame_labels import UNIT_AREA_BERGMAN_FRAME
    from plumbing.genus2_bc_ghost import canonical_bc_ghost_density, period_volume_jacobian_abs_squared
    from plumbing.liouville_genus2_modular_check import (
        SymplecticTransform,
        named_transform,
        sp4_generator_transforms,
    )


OMEGA = np.asarray(
    [[0.12 + 1.15j, 0.08 + 0.04j], [0.08 + 0.04j, -0.09 + 0.92j]],
    dtype=np.complex128,
)
OMEGA_ALTERNATE = np.asarray(
    [[0.21 + 1.32j, -0.17 + 0.18j], [-0.17 + 0.18j, 0.07 + 0.88j]],
    dtype=np.complex128,
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _relative_error(value: float, target: float) -> float:
    return abs(float(value) - float(target)) / max(abs(float(target)), 1.0e-300)


def _symplectic_matrix(transform: SymplecticTransform) -> np.ndarray:
    return np.block([[transform.a, transform.b], [transform.c, transform.d]])


def _compose_transforms(name: str, *transforms: SymplecticTransform) -> SymplecticTransform:
    """Return the transform obtained by applying ``transforms`` from left to right."""

    matrix = np.eye(4, dtype=np.int64)
    for transform in transforms:
        matrix = _symplectic_matrix(transform) @ matrix
    return SymplecticTransform(
        name=name,
        a=matrix[:2, :2],
        b=matrix[:2, 2:],
        c=matrix[2:, :2],
        d=matrix[2:, 2:],
    )


def _mcg_test_transforms() -> tuple[SymplecticTransform, ...]:
    generators = sp4_generator_transforms()
    inverses = tuple(transform.inverse() for transform in generators)
    words = (
        _compose_transforms(
            "T11 full-s T22",
            named_transform("T11"),
            named_transform("full-s"),
            named_transform("T22"),
        ),
        _compose_transforms(
            "gl-shear-12 T12 full-s",
            named_transform("gl-shear-12"),
            named_transform("T12"),
            named_transform("full-s"),
        ),
        _compose_transforms(
            "full-s T11 full-s T22",
            named_transform("full-s"),
            named_transform("T11"),
            named_transform("full-s"),
            named_transform("T22"),
        ),
        _compose_transforms(
            "T12 gl-shear-12 bridge-sign full-s",
            named_transform("T12"),
            named_transform("gl-shear-12"),
            named_transform("bridge-sign"),
            named_transform("full-s"),
        ),
    )
    return generators + inverses + words


def check_critical_factorization() -> None:
    result = canonical_bc_ghost_density(OMEGA, theta_nmax=10)
    reconstructed = result.bc_ghost_density * result.scalar_partition_without_target_zero_mode**26
    factorization_error = _relative_error(reconstructed, result.critical_matter_ghost_density)
    simplification_error = _relative_error(result.bc_ghost_density, result.simplified_bc_ghost_density)

    print("critical Mumford factorization")
    print(f"  critical density       = {result.critical_matter_ghost_density:.16e}")
    print(f"  ghost times 26 scalars = {reconstructed:.16e}")
    print(f"  factorization error    = {factorization_error:.6e}")
    print(f"  simplification error   = {simplification_error:.6e}")
    print(f"  conformal frame        = {result.conformal_frame}")
    _require(result.conformal_frame == UNIT_AREA_BERGMAN_FRAME, "ghost frame label is not Bergman")
    _require(result.worldsheet_area == 1.0, "ghost Bergman metric is not unit area")
    _require(factorization_error < 2.0e-14, "ghost/scalar factorization failed")
    _require(simplification_error < 2.0e-14, "closed ghost formula disagrees with the quotient")


def check_separating_mumford_factorization() -> None:
    r"""Test the absolute Mumford residue without a production normalizer.

    In the string-note convention, ``chi10_note=2^-12 Psi10`` and
    ``q=2*pi*i*epsilon``.  Equations (4.108)--(4.109) then require the
    chiral residue of ``(2*pi*i)/chi10_note`` to be one relative to the two
    genus-one Mumford forms.  The eta functions below are evaluated with
    mpmath, independently of the genus-two theta-product implementation.
    """

    tau_left = 0.17 + 1.13j
    tau_right = -0.21 + 0.91j
    eta_left = complex(mp.eta(mp.mpc(tau_left)))
    eta_right = complex(mp.eta(mp.mpc(tau_right)))
    bare_torus_product = (
        tau_left.imag**-13
        * abs(eta_left) ** -48
        * tau_right.imag**-13
        * abs(eta_right) ** -48
    )
    completed_punctured_torus_product = (2.0 * math.pi) ** 4 * bare_torus_product
    epsilon_values = (0.025j, 0.0125j, 0.00625j, 0.003125j, 0.0015625j)
    theta_errors = []
    bare_residue_errors = []
    punctured_ratios = []

    print("\nseparating Mumford normalization (independent eta evaluation)")
    for epsilon in epsilon_values:
        omega = np.asarray(
            [[tau_left, epsilon], [epsilon, tau_right]],
            dtype=np.complex128,
        )
        result = canonical_bc_ghost_density(
            omega,
            mumford_prefactor=2j * math.pi,
            theta_nmax=12,
            chi10_normalization="product",
        )
        q_bridge = 2j * math.pi * epsilon
        expected_raw_product = (
            2.0**12
            * q_bridge**2
            * eta_left**24
            * eta_right**24
        )
        theta_error = abs(result.chi10 / expected_raw_product - 1.0)

        # Convert d epsilon to d q, strip |dq/q^2|^2, and apply the
        # nonchiral 2^24 conversion from Psi10 to chi10_note.
        normalized_density_in_epsilon = (
            2.0**24 * result.critical_matter_ghost_density
        )
        stripped_density_in_q = (
            normalized_density_in_epsilon
            * abs(1.0 / (2j * math.pi)) ** 2
            * abs(q_bridge) ** 4
        )
        bare_ratio = stripped_density_in_q / bare_torus_product
        punctured_ratio = (
            stripped_density_in_q / completed_punctured_torus_product
        )
        theta_errors.append(float(theta_error))
        bare_residue_errors.append(abs(float(bare_ratio) - 1.0))
        punctured_ratios.append(float(punctured_ratio))
        print(
            f"  |epsilon|={abs(epsilon):.7f}: "
            f"Psi10 asymptotic error={theta_error:.3e}, "
            f"residue/bare={bare_ratio:.12e}, "
            f"residue/punctured={punctured_ratio:.12e}"
        )

    expected_punctured_ratio = (2.0 * math.pi) ** -4
    _require(
        all(later < earlier for earlier, later in zip(theta_errors, theta_errors[1:])),
        "raw theta-product separating asymptotic does not converge",
    )
    _require(
        all(
            later < earlier
            for earlier, later in zip(bare_residue_errors, bare_residue_errors[1:])
        ),
        "critical Mumford residue does not converge to the genus-one product",
    )
    _require(theta_errors[-1] < 1.0e-5, "raw theta-product normalization is incorrect")
    _require(
        bare_residue_errors[-1] < 2.0e-5,
        "critical separating residue is not one in the bare Mumford convention",
    )
    _require(
        _relative_error(punctured_ratios[-1], expected_punctured_ratio) < 2.0e-5,
        "the punctured-torus comparison does not expose the expected B-form factor",
    )


def check_modular_s_covariance() -> None:
    transformed = -np.linalg.inv(OMEGA)
    original = canonical_bc_ghost_density(OMEGA, theta_nmax=10)
    image = canonical_bc_ghost_density(transformed, theta_nmax=10)
    automorphy = abs(np.linalg.det(OMEGA))

    chi_ratio = abs(image.chi10) / abs(original.chi10)
    expected_chi_ratio = automorphy**10
    ghost_ratio = image.bc_ghost_density / original.bc_ghost_density
    expected_ghost_ratio = automorphy**6
    volume_ratio = period_volume_jacobian_abs_squared(np.linalg.det(OMEGA))
    invariant_ratio = ghost_ratio * volume_ratio
    scalar_ratio = image.scalar_partition_without_target_zero_mode / original.scalar_partition_without_target_zero_mode

    print("\nS modular covariance")
    print(f"  |chi10' / chi10|      = {chi_ratio:.16e}")
    print(f"  expected weight 10    = {expected_chi_ratio:.16e}")
    print(f"  ghost density ratio   = {ghost_ratio:.16e}")
    print(f"  expected weight 6     = {expected_ghost_ratio:.16e}")
    print(f"  period-volume ratio   = {volume_ratio:.16e}")
    print(f"  invariant product     = {invariant_ratio:.16e}")
    print(f"  scalar partition ratio= {scalar_ratio:.16e}")
    _require(_relative_error(chi_ratio, expected_chi_ratio) < 2.0e-12, "chi10 weight-10 law failed")
    _require(_relative_error(ghost_ratio, expected_ghost_ratio) < 2.0e-12, "ghost weight-6 law failed")
    _require(_relative_error(invariant_ratio, 1.0) < 2.0e-12, "ghost measure is not modular invariant")
    _require(_relative_error(scalar_ratio, 1.0) < 2.0e-12, "Bergman scalar partition is not invariant")


def check_full_mcg_covariance() -> None:
    r"""Check the induced genus-two MCG action through ``Sp(4,Z)``.

    The scalar coefficient of the critical integrand has modular weight
    ``(3,3)``.  Its product with the nonchiral period-coordinate volume is
    therefore invariant.
    """

    maxima = {
        "chi10 complex weight-10": 0.0,
        "det Im weight -2": 0.0,
        "critical coefficient weight 6": 0.0,
        "critical measure invariance": 0.0,
        "Bergman F invariance": 0.0,
        "scalar partition invariance": 0.0,
        "symmetric-coordinate Jacobian": 0.0,
    }
    worst = {key: "" for key in maxima}
    coordinate_basis = (
        np.asarray([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex128),
        np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128),
        np.asarray([[0.0, 0.0], [0.0, 1.0]], dtype=np.complex128),
    )
    transforms = _mcg_test_transforms()

    for point_index, omega in enumerate((OMEGA, OMEGA_ALTERNATE), start=1):
        original = canonical_bc_ghost_density(omega, theta_nmax=10)
        for transform in transforms:
            label = f"point {point_index}, {transform.name}"
            image = canonical_bc_ghost_density(transform.transform_omega(omega), theta_nmax=10)
            automorphy = transform.det_factor(omega)
            abs_automorphy = abs(automorphy)
            volume_ratio = period_volume_jacobian_abs_squared(automorphy)

            errors = {
                "chi10 complex weight-10": abs(
                    image.chi10 / (automorphy**10 * original.chi10) - 1.0
                ),
                "det Im weight -2": _relative_error(
                    image.det_im_omega / original.det_im_omega,
                    abs_automorphy**-2,
                ),
                "critical coefficient weight 6": _relative_error(
                    image.critical_matter_ghost_density
                    / original.critical_matter_ghost_density,
                    abs_automorphy**6,
                ),
                "critical measure invariance": _relative_error(
                    image.critical_matter_ghost_density
                    / original.critical_matter_ghost_density
                    * volume_ratio,
                    1.0,
                ),
                "Bergman F invariance": _relative_error(
                    image.petersson_norm_delta2,
                    original.petersson_norm_delta2,
                ),
                "scalar partition invariance": _relative_error(
                    image.scalar_partition_without_target_zero_mode,
                    original.scalar_partition_without_target_zero_mode,
                ),
            }

            denominator_inverse = np.linalg.inv(
                transform.c.astype(np.complex128) @ omega
                + transform.d.astype(np.complex128)
            )
            columns = []
            for variation in coordinate_basis:
                image_variation = denominator_inverse.T @ variation @ denominator_inverse
                columns.append(
                    (image_variation[0, 0], image_variation[0, 1], image_variation[1, 1])
                )
            coordinate_jacobian = np.asarray(columns, dtype=np.complex128).T
            errors["symmetric-coordinate Jacobian"] = abs(
                np.linalg.det(coordinate_jacobian) / automorphy**-3 - 1.0
            )

            for key, error in errors.items():
                if error > maxima[key]:
                    maxima[key] = float(error)
                    worst[key] = label

    print("\nfull genus-two MCG / Sp(4,Z) covariance")
    print(f"  period matrices tested = 2")
    print(f"  transformations tested = {len(transforms)} per matrix")
    for key, error in maxima.items():
        print(f"  max {key:34s} = {error:.6e}  ({worst[key]})")

    _require(maxima["chi10 complex weight-10"] < 2.0e-11, "chi10 complex modular law failed")
    _require(maxima["det Im weight -2"] < 2.0e-13, "det Im(Omega) modular law failed")
    _require(
        maxima["critical coefficient weight 6"] < 5.0e-11,
        "critical bosonic-string coefficient does not have weight (3,3)",
    )
    _require(
        maxima["critical measure invariance"] < 5.0e-11,
        "critical bosonic-string measure is not MCG invariant",
    )
    _require(maxima["Bergman F invariance"] < 2.0e-11, "Bergman Petersson norm is not invariant")
    _require(maxima["scalar partition invariance"] < 2.0e-11, "Bergman scalar partition is not invariant")
    _require(
        maxima["symmetric-coordinate Jacobian"] < 2.0e-13,
        "symmetric period-coordinate Jacobian has the wrong modular weight",
    )


def check_symmetric_period_coordinate_jacobian() -> None:
    inverse = np.linalg.inv(OMEGA)
    coordinate_basis = (
        np.asarray([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex128),
        np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128),
        np.asarray([[0.0, 0.0], [0.0, 1.0]], dtype=np.complex128),
    )
    columns = []
    for variation in coordinate_basis:
        image_variation = inverse @ variation @ inverse
        columns.append((image_variation[0, 0], image_variation[0, 1], image_variation[1, 1]))
    jacobian = np.asarray(columns, dtype=np.complex128).T
    determinant = complex(np.linalg.det(jacobian))
    expected = complex(np.linalg.det(OMEGA) ** -3)
    relative_error = abs(determinant / expected - 1.0)
    abs_squared = abs(determinant) ** 2
    expected_abs_squared = period_volume_jacobian_abs_squared(np.linalg.det(OMEGA))

    print("\nsymmetric period-coordinate Jacobian")
    print(f"  det dOmega'/dOmega = {determinant!r}")
    print(f"  det(Omega)^(-3)     = {expected!r}")
    print(f"  relative error      = {relative_error:.6e}")
    _require(relative_error < 2.0e-14, "off-diagonal period-coordinate convention is inconsistent")
    _require(
        _relative_error(abs_squared, expected_abs_squared) < 2.0e-14,
        "period-volume absolute-square Jacobian is inconsistent",
    )


def check_saved_theta_glasses_markings() -> None:
    path = Path("plumbing/results/theta_glasses_period_precision_w8_radial_q01500.json")
    data = json.loads(path.read_text())

    def matrix(key: str) -> np.ndarray:
        return np.asarray([[complex(value) for value in row] for row in data[key]], dtype=np.complex128)

    omega_glasses = matrix("omega_glasses")
    omega_theta = matrix("omega_theta_target")
    glasses = canonical_bc_ghost_density(omega_glasses, theta_nmax=12)
    theta = canonical_bc_ghost_density(omega_theta, theta_nmax=12)
    automorphy = math.sqrt(glasses.det_im_omega / theta.det_im_omega)
    ghost_ratio = theta.bc_ghost_density / glasses.bc_ghost_density
    expected_ratio = automorphy**6
    invariant_ratio = ghost_ratio * automorphy**-6
    petersson_ratio = theta.petersson_norm_delta2 / glasses.petersson_norm_delta2

    print("\nsaved theta/glasses period-matched markings")
    print(f"  ghost theta/glasses   = {ghost_ratio:.16e}")
    print(f"  expected weight 6     = {expected_ratio:.16e}")
    print(f"  with volume Jacobian  = {invariant_ratio:.16e}")
    print(f"  Bergman F ratio       = {petersson_ratio:.16e}")
    _require(_relative_error(ghost_ratio, expected_ratio) < 2.0e-10, "saved ghost automorphy factor failed")
    _require(_relative_error(invariant_ratio, 1.0) < 2.0e-10, "saved ghost measures do not agree")
    _require(_relative_error(petersson_ratio, 1.0) < 2.0e-10, "saved period matrices do not give common F")


def check_normalization_scaling() -> None:
    baseline = canonical_bc_ghost_density(OMEGA, theta_nmax=10)
    changed = canonical_bc_ghost_density(
        OMEGA,
        mumford_prefactor=3.0 - 4.0j,
        bergman_determinant_constant=1.25,
        theta_nmax=10,
    )
    expected_ratio = (abs(3.0 - 4.0j) / (2.0 * math.pi)) ** 2 * 1.25**13
    actual_ratio = changed.bc_ghost_density / baseline.bc_ghost_density

    print("\nmoduli-independent normalization scaling")
    print(f"  actual ratio   = {actual_ratio:.16e}")
    print(f"  expected ratio = {expected_ratio:.16e}")
    _require(_relative_error(actual_ratio, expected_ratio) < 2.0e-14, "normalization powers are inconsistent")


def main() -> None:
    check_critical_factorization()
    check_separating_mumford_factorization()
    check_modular_s_covariance()
    check_full_mcg_covariance()
    check_symmetric_period_coordinate_jacobian()
    check_saved_theta_glasses_markings()
    check_normalization_scaling()
    print("\ncanonical genus-two bc ghost checks passed")


if __name__ == "__main__":
    main()
