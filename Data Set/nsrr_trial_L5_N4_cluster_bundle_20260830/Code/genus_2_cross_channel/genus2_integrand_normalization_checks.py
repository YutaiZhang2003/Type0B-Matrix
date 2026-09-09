#!/usr/bin/env python3
"""Exact checks for the genus-two normalization ledger.

The component identities here are supplemented by the combined lower-genus
sewing check in ``genus2_integrand_factorization_audit.py``.  Together they
certify the local matter-plus-ghost identities used by the modular driver.
The BRY/Xi genus-two topology and real-measure bridge is a separate open
normalization check.
"""

from __future__ import annotations

import math

import numpy as np

try:
    from free_boson_plumbing import noncompact_scalar_zero_mode_factor
    from genus2_c1_string_integrand import compact_boson_winding_sum_genus2
    from genus2_integrand_normalization import (
        ALGEBRAIC_IGUSA_CHI10,
        BRY_XI_FULL_GENUS2_AMPLITUDE_DICTIONARY_RECONCILED,
        BRY_XI_GENUS2_TOPOLOGY_DICTIONARY_RECONCILED,
        BRY_XI_LOCAL_LIOUVILLE_CFT_DICTIONARY_RECONCILED,
        BRY_XI_STRING_COUPLING_DICTIONARY_RECONCILED,
        C1_SPHERE_TOPOLOGY_CORRECTION_APPLIED_TO_PRODUCTION_KERNEL,
        C1_SPHERE_TOPOLOGY_NORMALIZATION_AUDITED,
        CODE_SCALAR_OVER_DHP_PER_VOLUME,
        CODE_GENUS1_CRITICAL_STACK_PREFACTOR,
        DHP_GENUS1_CRITICAL_STACK_PREFACTOR,
        FACTORIZATION_NORMALIZED_CHI10,
        FULL_CFT_FACTORIZATION_CERTIFIED,
        GENUS2_C1_ABSOLUTE_NORMALIZATION_CERTIFIED,
        GENUS2_VACUUM_GAUGE_FIXING_NORMALIZATION,
        GENUS2_GENERIC_STACK_WEIGHT,
        RAW_PRODUCT_CHIRAL_RESIDUE,
        RAW_PRODUCT_FACTORIZATION_NORMALIZATION,
        RAW_PRODUCT_NONCHIRAL_RESIDUE,
        RAW_THETA_PRODUCT,
        STRING_NOTE_GENUS2_REAL_FORM_FACTOR,
        WORLDSHEET_GAUGE_FIXING_NORMALIZATION_CERTIFIED,
        bry_genus2_topology_normalization,
        bry_genus2_topology_normalization_from_mqm,
        bry_string_coupling_from_mqm_fermi_level,
        bry_xi_bare_convention_map,
        c1_genus_topology_correction,
        c1_genus2_topology_correction,
        c1_reduced_sphere_metric,
        c1_sphere_normalized_genus2_kernel_multiplier,
        c1_timelike_sphere_constant,
        c1_matrix_model_genus2_coefficient,
        coarse_genus2_fundamental_domain_target,
        coarse_string_note_integration_kernel_target,
        critical_prefactor_ratio_to_dhp,
        factorization_normalized_moduli_integral_target,
        genus1_critical_prefactor_ratio_to_dhp,
        genus2_worldsheet_gauge_fixing_normalization,
        genus2_topology_normalization,
        mqm_fermi_level_from_bry_string_coupling,
        mqm_fermi_level_from_xi_string_coupling,
        mumford_factorization_normalization,
        raw_product_bry_overall_normalization,
        raw_product_bry_normalization_from_mqm,
        raw_product_moduli_integral_target,
        separating_mumford_normalization,
        sphere_state_metric_normalization,
        string_note_genus2_complex_form_real_factor,
        string_note_genus2_full_kernel_multiplier,
        string_note_genus2_kernel_multiplier,
        string_note_integration_kernel_target,
        worldsheet_gauge_fixing_normalization,
        xi_compact_target_zero_mode,
        xi_full_replacement_over_dimensionless,
        xi_genus2_scalar_over_dimensionless,
        xi_string_coupling_from_mqm_fermi_level,
    )
except ImportError:  # pragma: no cover - supports package-style execution
    from plumbing.free_boson_plumbing import noncompact_scalar_zero_mode_factor
    from plumbing.genus2_c1_string_integrand import compact_boson_winding_sum_genus2
    from plumbing.genus2_integrand_normalization import (
        ALGEBRAIC_IGUSA_CHI10,
        BRY_XI_FULL_GENUS2_AMPLITUDE_DICTIONARY_RECONCILED,
        BRY_XI_GENUS2_TOPOLOGY_DICTIONARY_RECONCILED,
        BRY_XI_LOCAL_LIOUVILLE_CFT_DICTIONARY_RECONCILED,
        BRY_XI_STRING_COUPLING_DICTIONARY_RECONCILED,
        C1_SPHERE_TOPOLOGY_CORRECTION_APPLIED_TO_PRODUCTION_KERNEL,
        C1_SPHERE_TOPOLOGY_NORMALIZATION_AUDITED,
        CODE_SCALAR_OVER_DHP_PER_VOLUME,
        CODE_GENUS1_CRITICAL_STACK_PREFACTOR,
        DHP_GENUS1_CRITICAL_STACK_PREFACTOR,
        FACTORIZATION_NORMALIZED_CHI10,
        FULL_CFT_FACTORIZATION_CERTIFIED,
        GENUS2_C1_ABSOLUTE_NORMALIZATION_CERTIFIED,
        GENUS2_VACUUM_GAUGE_FIXING_NORMALIZATION,
        GENUS2_GENERIC_STACK_WEIGHT,
        RAW_PRODUCT_CHIRAL_RESIDUE,
        RAW_PRODUCT_FACTORIZATION_NORMALIZATION,
        RAW_PRODUCT_NONCHIRAL_RESIDUE,
        RAW_THETA_PRODUCT,
        STRING_NOTE_GENUS2_REAL_FORM_FACTOR,
        WORLDSHEET_GAUGE_FIXING_NORMALIZATION_CERTIFIED,
        bry_genus2_topology_normalization,
        bry_genus2_topology_normalization_from_mqm,
        bry_string_coupling_from_mqm_fermi_level,
        bry_xi_bare_convention_map,
        c1_genus_topology_correction,
        c1_genus2_topology_correction,
        c1_reduced_sphere_metric,
        c1_sphere_normalized_genus2_kernel_multiplier,
        c1_timelike_sphere_constant,
        c1_matrix_model_genus2_coefficient,
        coarse_genus2_fundamental_domain_target,
        coarse_string_note_integration_kernel_target,
        critical_prefactor_ratio_to_dhp,
        factorization_normalized_moduli_integral_target,
        genus1_critical_prefactor_ratio_to_dhp,
        genus2_worldsheet_gauge_fixing_normalization,
        genus2_topology_normalization,
        mqm_fermi_level_from_bry_string_coupling,
        mqm_fermi_level_from_xi_string_coupling,
        mumford_factorization_normalization,
        raw_product_bry_overall_normalization,
        raw_product_bry_normalization_from_mqm,
        raw_product_moduli_integral_target,
        separating_mumford_normalization,
        sphere_state_metric_normalization,
        string_note_genus2_complex_form_real_factor,
        string_note_genus2_full_kernel_multiplier,
        string_note_genus2_kernel_multiplier,
        string_note_integration_kernel_target,
        worldsheet_gauge_fixing_normalization,
        xi_compact_target_zero_mode,
        xi_full_replacement_over_dimensionless,
        xi_genus2_scalar_over_dimensionless,
        xi_string_coupling_from_mqm_fermi_level,
    )


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _relative_error(value: float, target: float) -> float:
    return abs(float(value) / float(target) - 1.0)


def check_exact_conversion_constants() -> None:
    print("exact cusp-form conversion constants")
    print(f"  raw chiral residue          = {RAW_PRODUCT_CHIRAL_RESIDUE:.16e}")
    print(f"  raw nonchiral residue       = {RAW_PRODUCT_NONCHIRAL_RESIDUE:.16e}")
    print(f"  raw density normalization   = {RAW_PRODUCT_FACTORIZATION_NORMALIZATION:.16e}")
    _require(
        RAW_PRODUCT_CHIRAL_RESIDUE**2 == RAW_PRODUCT_NONCHIRAL_RESIDUE,
        "chiral and nonchiral raw residues are inconsistent",
    )
    _require(
        RAW_PRODUCT_NONCHIRAL_RESIDUE * RAW_PRODUCT_FACTORIZATION_NORMALIZATION == 1.0,
        "raw theta-product normalization does not give unit Mumford residue",
    )
    _require(
        mumford_factorization_normalization(FACTORIZATION_NORMALIZED_CHI10) == 1.0,
        "Mumford-residue-normalized chi10 should require no multiplier",
    )
    _require(
        mumford_factorization_normalization(ALGEBRAIC_IGUSA_CHI10) == 1.0 / 16.0,
        "algebraic Igusa normalization has the wrong conversion",
    )
    genus1_dhp_ratio = genus1_critical_prefactor_ratio_to_dhp()
    genus2_dhp_ratio = critical_prefactor_ratio_to_dhp()
    expected_dhp_ratio = CODE_SCALAR_OVER_DHP_PER_VOLUME**26
    print(f"  scalar code / DHP per volume= {CODE_SCALAR_OVER_DHP_PER_VOLUME:.16e}")
    print(f"  genus-one code / DHP        = {genus1_dhp_ratio:.16e}")
    print(f"  genus-two code / DHP        = {genus2_dhp_ratio:.16e}")
    _require(
        _relative_error(genus1_dhp_ratio, expected_dhp_ratio) < 2.0e-15,
        "the genus-one D'Hoker--Phong conversion is inconsistent",
    )
    _require(
        _relative_error(genus2_dhp_ratio, expected_dhp_ratio) < 2.0e-15,
        "the D'Hoker--Phong critical-measure convention ledger is inconsistent",
    )
    _require(
        _relative_error(genus1_dhp_ratio, genus2_dhp_ratio) < 2.0e-15,
        "the scalar volume convention changes between genus one and genus two",
    )
    _require(
        _relative_error(
            CODE_GENUS1_CRITICAL_STACK_PREFACTOR
            / DHP_GENUS1_CRITICAL_STACK_PREFACTOR,
            genus1_dhp_ratio,
        )
        < 2.0e-15,
        "the explicit genus-one prefactors do not reproduce the conversion",
    )

    alpha_prime = 1.7
    radius = 1.2
    scalar_factor = xi_genus2_scalar_over_dimensionless(alpha_prime)
    compact_zero_mode = xi_compact_target_zero_mode(radius, alpha_prime)
    net_factor = xi_full_replacement_over_dimensionless(alpha_prime)
    direct_cancellation = (
        (4.0 * math.pi**2 * alpha_prime) ** -26
        * (4.0 * math.pi**2 * alpha_prime) ** 25
        * (compact_zero_mode / radius)
    )
    print(f"  Xi scalar / dimensionless   = {scalar_factor:.16e}")
    print(f"  Xi full replacement factor  = {net_factor:.16e}")
    _require(
        _relative_error(
            scalar_factor,
            1.0 / (4.0 * math.pi**2 * alpha_prime),
        )
        < 2.0e-15,
        "Xi's genus-two dk/(2 pi) scalar factor is incorrect",
    )
    _require(
        _relative_error(direct_cancellation, net_factor) < 3.0e-15,
        "the 26-scalar Xi replacement factors do not cancel correctly",
    )


def check_topology_factorization() -> None:
    bry_string_coupling = 0.37
    sphere = 2.0 * math.pi / bry_string_coupling**2
    topology = genus2_topology_normalization(
        sphere_normalization=sphere,
        torus_normalization=1.0,
    )
    bry_topology = bry_genus2_topology_normalization(bry_string_coupling)
    combined = raw_product_bry_overall_normalization(bry_string_coupling)

    print("\nclosed-string topology factorization")
    print(f"  C_S2                         = {sphere:.16e}")
    print(f"  C_T2                         = {1.0:.16e}")
    print(f"  C_Sigma2                     = {topology:.16e}")
    print(
        "  normalized raw product / (g_s^BRY)^2 = "
        f"{combined / bry_string_coupling**2:.16e}"
    )
    _require(
        _relative_error(topology * sphere, 1.0) < 2.0e-15,
        "separating topology constants do not factorize",
    )
    _require(
        _relative_error(topology, bry_topology) < 2.0e-15,
        "BRY genus-two topology normalization is inconsistent",
    )
    _require(
        _relative_error(combined, RAW_PRODUCT_FACTORIZATION_NORMALIZATION * topology) < 2.0e-15,
        "the cusp-form and topology normalization layers were combined incorrectly",
    )


def check_polyakov_gauge_fixing_recurrence() -> None:
    """Check the string-note sewing recurrences and real-measure conversion."""

    alpha_prime = 1.0
    sphere_metric = sphere_state_metric_normalization(alpha_prime)
    sewing_scale = 8.0 * math.pi / (alpha_prime * sphere_metric)
    n_20 = worldsheet_gauge_fixing_normalization(2, 0)
    n_11 = worldsheet_gauge_fixing_normalization(1, 1)
    n_12 = worldsheet_gauge_fixing_normalization(1, 2)
    separating_right = -1j * n_20 * sewing_scale
    nonseparating_right = -1j * n_20 * sewing_scale
    real_form_factor = string_note_genus2_complex_form_real_factor()
    kernel_multiplier = string_note_genus2_kernel_multiplier(alpha_prime)
    c1_sphere_constant = c1_timelike_sphere_constant(alpha_prime)
    c1_sphere_metric = c1_reduced_sphere_metric(alpha_prime)
    genus1_topology_correction = c1_genus_topology_correction(1, alpha_prime)
    genus2_topology_correction = c1_genus2_topology_correction(alpha_prime)
    c1_kernel_multiplier = c1_sphere_normalized_genus2_kernel_multiplier(
        alpha_prime
    )

    print("\nPolyakov gauge-fixing normalization")
    print(f"  K_S2                         = {sphere_metric:.16e}")
    print(f"  N_(2,0)                      = {n_20}")
    print(f"  N_(1,1)^2                    = {n_11**2}")
    print(f"  -i N_(2,0) 8pi/(alpha' K)   = {separating_right}")
    print(f"  N_(1,2)                      = {n_12}")
    print(f"  complex six-form -> d^6Omega = {real_form_factor:.16e}")
    print(f"  critical g_s^2-stripped mult = {kernel_multiplier:.16e}")
    print(f"  K_tilde_S2 c=1               = {c1_sphere_constant:.16e}")
    print(f"  Khat_S2 c=1                  = {c1_sphere_metric:.16e}")
    print(f"  genus-one topology correction= {genus1_topology_correction:.16e}")
    print(f"  genus-two topology correction= {genus2_topology_correction:.16e}")
    print(f"  final c=1 kernel multiplier  = {c1_kernel_multiplier:.16e}")
    _require(
        n_20 == GENUS2_VACUUM_GAUGE_FIXING_NORMALIZATION == -1j,
        "the formal genus-two vacuum gauge-fixing phase is not -i",
    )
    _require(
        genus2_worldsheet_gauge_fixing_normalization() == n_20,
        "the genus-two gauge-fixing helper disagrees with N_(h,n)",
    )
    _require(
        abs(n_11**2 - separating_right) < 1.0e-15,
        "the separating plumbing recurrence for N_(h,n) failed",
    )
    _require(
        abs(n_12 - nonseparating_right) < 1.0e-15,
        "the nonseparating plumbing recurrence for N_(h,n) failed",
    )
    _require(
        real_form_factor == STRING_NOTE_GENUS2_REAL_FORM_FACTOR == 8.0,
        "the genus-two complex-form conversion is not eight",
    )
    _require(
        _relative_error(kernel_multiplier, alpha_prime / math.pi) < 2.0e-15,
        "the inherited critical-string kernel multiplier is not alpha'/pi",
    )
    _require(
        _relative_error(c1_sphere_metric, 4.0 * math.pi) < 2.0e-15,
        "the reduced c=1 sphere metric is not 4*pi",
    )
    _require(
        genus1_topology_correction == 1.0,
        "the sphere-topology replacement must leave the genus-one vacuum unchanged",
    )
    _require(
        _relative_error(genus2_topology_correction, 2.0 / alpha_prime)
        < 2.0e-15,
        "the c=1 genus-two topology correction is not 2/alpha'",
    )
    _require(
        _relative_error(c1_kernel_multiplier, 2.0 / math.pi) < 2.0e-15,
        "the sphere-normalized c=1 kernel multiplier is not 2/pi",
    )
    _require(
        WORLDSHEET_GAUGE_FIXING_NORMALIZATION_CERTIFIED,
        "the local string-note real-measure algebra is not certified",
    )
    _require(
        C1_SPHERE_TOPOLOGY_NORMALIZATION_AUDITED,
        "the c=1 sphere-topology correction must be recorded as audited",
    )
    _require(
        C1_SPHERE_TOPOLOGY_CORRECTION_APPLIED_TO_PRODUCTION_KERNEL
        and not BRY_XI_GENUS2_TOPOLOGY_DICTIONARY_RECONCILED
        and not BRY_XI_FULL_GENUS2_AMPLITUDE_DICTIONARY_RECONCILED
        and not GENUS2_C1_ABSOLUTE_NORMALIZATION_CERTIFIED
        and not FULL_CFT_FACTORIZATION_CERTIFIED,
        "the production kernel must include 2/alpha' while the independent "
        "absolute genus-two certification remains false",
    )


def check_bry_xi_bare_map() -> None:
    """Check only the BRY/Xi factors stated explicitly in the two sources."""

    torus_two_point = bry_xi_bare_convention_map(1, 2)
    genus_two_vacuum = bry_xi_bare_convention_map(2, 0)
    print("\nBRY/Xi bare convention map")
    print(
        "  torus two-point measure/coupling = "
        f"{torus_two_point.xi_over_bry_real_measure_factor:g} * "
        f"{torus_two_point.xi_over_bry_coupling_weight:g}"
    )
    print(
        "  genus-two vacuum measure/coupling = "
        f"{genus_two_vacuum.xi_over_bry_real_measure_factor:g} * "
        f"{genus_two_vacuum.xi_over_bry_coupling_weight:g}"
    )
    _require(
        BRY_XI_LOCAL_LIOUVILLE_CFT_DICTIONARY_RECONCILED,
        "the identical BRY/Xi local Liouville data should be certified",
    )
    _require(
        BRY_XI_STRING_COUPLING_DICTIONARY_RECONCILED,
        "the BRY/Xi string-coupling dictionary should be certified",
    )
    _require(
        torus_two_point.xi_over_bry_known_product == 1.0,
        "the two torus moduli must cancel the genus-one two-point coupling map",
    )
    _require(
        genus_two_vacuum.xi_over_bry_real_measure_factor == 8.0
        and genus_two_vacuum.xi_over_bry_coupling_weight == 0.25
        and genus_two_vacuum.xi_over_bry_known_product == 2.0,
        "the genus-two bare BRY/Xi factors are inconsistent",
    )


def check_mqm_dictionary() -> None:
    fermi_level = 2.7
    xi_string_coupling = xi_string_coupling_from_mqm_fermi_level(fermi_level)
    bry_string_coupling = bry_string_coupling_from_mqm_fermi_level(fermi_level)
    matrix_coupling = 2.0 * math.pi * bry_string_coupling
    sphere = 2.0 * math.pi / bry_string_coupling**2
    topology = bry_genus2_topology_normalization_from_mqm(fermi_level)
    raw_normalization = raw_product_bry_normalization_from_mqm(fermi_level)
    radius = 1.0
    matrix_coefficient = c1_matrix_model_genus2_coefficient(radius)
    normalized_integral = factorization_normalized_moduli_integral_target(radius)
    coarse_integral = coarse_genus2_fundamental_domain_target(radius)
    string_note_integral = string_note_integration_kernel_target(radius)
    coarse_string_note_integral = coarse_string_note_integration_kernel_target(radius)
    string_note_full_multiplier = (
        xi_string_coupling**2
        * c1_sphere_normalized_genus2_kernel_multiplier()
    )
    raw_integral = raw_product_moduli_integral_target(radius)
    unconverted_bry_over_xi = (
        topology * coarse_integral
        / (matrix_coefficient / fermi_level**2)
    )

    print("\nBRY/Xi matrix-model dictionary")
    print(f"  mu                            = {fermi_level:.16e}")
    print(f"  g_s^Xi                        = {xi_string_coupling:.16e}")
    print(f"  g_s^BRY                       = {bry_string_coupling:.16e}")
    print(f"  (g_s^BRY)^3 C_S2             = {bry_string_coupling**3 * sphere:.16e}")
    print(f"  BRY matrix coupling g          = {matrix_coupling:.16e}")
    print(f"  C_Sigma2                      = {topology:.16e}")
    print(f"  raw normalization             = {raw_normalization:.16e}")
    print(f"  unit-Mumford stack target R=1  = {normalized_integral:.16e}")
    print(f"  coarse-domain target R=1     = {coarse_integral:.16e}")
    print(f"  string-note kernel target R=1 = {string_note_integral:.16e}")
    print(f"  string-note full multiplier   = {string_note_full_multiplier:.16e}")
    print(f"  raw integral target R=1       = {raw_integral:.16e}")
    print(f"  unconverted BRY/Xi ratio      = {unconverted_bry_over_xi:.16e}")
    _require(
        _relative_error(
            mqm_fermi_level_from_xi_string_coupling(xi_string_coupling),
            fermi_level,
        )
        < 2.0e-15,
        "the Xi g_s-mu dictionary is not invertible",
    )
    _require(
        _relative_error(
            mqm_fermi_level_from_bry_string_coupling(bry_string_coupling),
            fermi_level,
        )
        < 2.0e-15,
        "the BRY g_s-mu dictionary is not invertible",
    )
    _require(
        _relative_error(bry_string_coupling / xi_string_coupling, 2.0)
        < 2.0e-15,
        "the BRY/Xi string-coupling conversion is not two",
    )
    _require(
        _relative_error(matrix_coupling, 1.0 / fermi_level) < 2.0e-15,
        "the BRY matrix coupling does not equal mu^-1",
    )
    _require(
        _relative_error(bry_string_coupling**3 * sphere, 1.0 / fermi_level)
        < 2.0e-15,
        "the BRY genus-zero normalization does not reproduce g=2 pi g_s=mu^-1",
    )
    _require(
        _relative_error(topology, 1.0 / (8.0 * math.pi**3 * fermi_level**2))
        < 2.0e-15,
        "the genus-two topology factor has the wrong MQM normalization",
    )
    _require(
        _relative_error(
            raw_normalization,
            2.0**21 / (math.pi**3 * fermi_level**2),
        )
        < 2.0e-15,
        "the raw-product MQM normalization is inconsistent",
    )
    _require(
        _relative_error(
            raw_product_bry_overall_normalization(bry_string_coupling),
            raw_normalization,
        )
        < 2.0e-15,
        "the g_s and MQM forms of the raw normalization disagree",
    )
    _require(
        _relative_error(normalized_integral, math.pi**3 / 30.0) < 2.0e-15,
        "the self-dual normalized moduli-integral target is incorrect",
    )
    _require(
        _relative_error(
            GENUS2_GENERIC_STACK_WEIGHT * coarse_integral,
            normalized_integral,
        )
        < 2.0e-15,
        "the coarse genus-two domain is missing its generic stack weight",
    )
    _require(
        _relative_error(
            string_note_full_multiplier * normalized_integral,
            matrix_coefficient / fermi_level**2,
        )
        < 2.0e-15,
        "the string-note stack integral does not reproduce the MQM coefficient",
    )
    _require(
        _relative_error(unconverted_bry_over_xi, 2.0) < 2.0e-15,
        "the exposed BRY/Xi mismatch should be two before converting the "
        "moduli differential and stack conventions",
    )
    _require(
        _relative_error(string_note_integral, math.pi**2 / 15.0) < 2.0e-15,
        "the self-dual string-note integration-kernel target is incorrect",
    )
    _require(
        _relative_error(
            GENUS2_GENERIC_STACK_WEIGHT * coarse_string_note_integral,
            string_note_integral,
        )
        < 2.0e-15,
        "the string-note target applies the stack weight incorrectly",
    )
    _require(
        _relative_error(raw_integral * RAW_PRODUCT_FACTORIZATION_NORMALIZATION, normalized_integral)
        < 2.0e-15,
        "raw and Mumford-residue-normalized targets disagree",
    )


def _compact_boson_winding_sum_genus1(tau: complex, radius: float, nmax: int) -> float:
    integers = range(-int(nmax), int(nmax) + 1)
    return float(
        sum(
            math.exp(-math.pi * radius**2 * abs(m + tau * n) ** 2 / tau.imag)
            for m in integers
            for n in integers
        )
    )


def check_scalar_separating_factorization() -> None:
    tau_left = 0.17 + 1.13j
    tau_right = -0.21 + 0.91j
    omega = np.asarray([[tau_left, 0.0], [0.0, tau_right]], dtype=np.complex128)
    radius = 1.23
    nmax = 9

    scalar_genus2 = noncompact_scalar_zero_mode_factor(omega)
    scalar_product = (tau_left.imag * tau_right.imag) ** -0.5
    theta_genus2 = compact_boson_winding_sum_genus2(
        omega,
        radius,
        lattice_nmax=nmax,
    )
    theta_left = _compact_boson_winding_sum_genus1(tau_left, radius, nmax)
    theta_right = _compact_boson_winding_sum_genus1(tau_right, radius, nmax)
    compact_genus2 = radius * scalar_genus2 * theta_genus2
    compact_left = radius * tau_left.imag**-0.5 * theta_left
    compact_right = radius * tau_right.imag**-0.5 * theta_right
    sewn_compact = compact_left * compact_right / radius

    print("\nscalar separating factorization")
    print(f"  noncompact Gaussian ratio = {scalar_genus2 / scalar_product:.16e}")
    print(f"  winding-sum ratio         = {theta_genus2 / (theta_left * theta_right):.16e}")
    print(f"  compact sewing ratio      = {compact_genus2 / sewn_compact:.16e}")
    _require(
        _relative_error(scalar_genus2, scalar_product) < 2.0e-15,
        "the noncompact scalar Gaussian does not factorize",
    )
    _require(
        _relative_error(theta_genus2, theta_left * theta_right) < 2.0e-13,
        "the compact winding sum does not factorize",
    )
    _require(
        _relative_error(compact_genus2, sewn_compact) < 2.0e-13,
        "the compact scalar retained an extra genus-two Gaussian constant",
    )


def check_numerical_separating_limit() -> None:
    tau_left = 0.17 + 1.13j
    tau_right = -0.21 + 0.91j
    epsilon_values = (0.025j, 0.0125j, 0.00625j, 0.003125j)
    expected_residues = {
        RAW_THETA_PRODUCT: RAW_PRODUCT_NONCHIRAL_RESIDUE,
        FACTORIZATION_NORMALIZED_CHI10: 1.0,
        ALGEBRAIC_IGUSA_CHI10: 16.0,
    }

    print("\nnumerical separating Mumford residue")
    for normalization, expected in expected_residues.items():
        errors = []
        for epsilon in epsilon_values:
            result = separating_mumford_normalization(
                tau_left,
                tau_right,
                epsilon,
                chi10_normalization=normalization,
                theta_nmax=9,
            )
            error = _relative_error(result.nonchiral_residue, expected)
            errors.append(error)
            print(
                f"  {normalization:>14s}, |epsilon|={abs(epsilon):.5f}: "
                f"residue={result.nonchiral_residue:.16e}, "
                f"normalized={result.normalized_nonchiral_residue:.16e}"
            )
        _require(errors[-1] < errors[0], f"{normalization} residue did not converge")
        _require(errors[-1] < 7.0e-5, f"{normalization} residue is outside its limit error")


def main() -> None:
    check_exact_conversion_constants()
    check_polyakov_gauge_fixing_recurrence()
    check_bry_xi_bare_map()
    check_topology_factorization()
    check_mqm_dictionary()
    check_scalar_separating_factorization()
    check_numerical_separating_limit()
    print("\ngenus-two integrand normalization checks passed")


if __name__ == "__main__":
    main()
