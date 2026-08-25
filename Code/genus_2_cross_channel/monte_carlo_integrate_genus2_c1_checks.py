#!/usr/bin/env python3
"""Fast algebraic checks for the genus-two c=1 Monte Carlo estimator."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

try:
    from genus2_integrand_normalization import (
        GENUS2_GENERIC_STACK_WEIGHT,
        LEGACY_STRING_NOTE_DIMENSIONLESS_SCALAR_KERNEL_CONVENTION,
        PRE_SPHERE_XI_STRING_NOTE_INTEGRATION_KERNEL_CONVENTION,
        STRING_NOTE_INTEGRATION_KERNEL_CONVENTION,
        c1_genus2_topology_correction,
        c1_sphere_normalized_genus2_kernel_multiplier,
        string_note_genus2_kernel_multiplier,
        xi_full_replacement_over_dimensionless,
    )
    from genus2_moduli_rqmc import generate_rqmc_design
    from genus2_moduli_physical_mixture_rqmc import (
        PHYSICAL_MIXTURE_SAMPLING_SCHEME,
        generate_physical_mixture_design,
    )
    from genus2_siegel_fundamental_domain import SIEGEL_VOLUME_G2
    from liouville_genus2_ccy import liouville_genus2_ccy_partition
    from liouville_genus2_glasses import liouville_genus2_glasses_partition
    from monte_carlo_integrate_genus2_c1 import (
        RescaledLiouvillePartition,
        _load_resume_rows,
        _resume_config,
        _write_csv,
        cached_period_solution_objects,
        canonicalize_string_note_kernel_row,
        estimate_from_transformed_values,
        evaluate_liouville_rescaled,
        evaluate_noncompact_scalar,
        kernel_det_im_power,
        node_stack_integration_weight,
        omega_from_csv_row,
        sampling_scheme_for_rows,
        summarize_rqmc_rows,
        validate_covariant_period_frames,
    )
except ImportError:  # pragma: no cover
    from plumbing.genus2_integrand_normalization import (
        GENUS2_GENERIC_STACK_WEIGHT,
        LEGACY_STRING_NOTE_DIMENSIONLESS_SCALAR_KERNEL_CONVENTION,
        PRE_SPHERE_XI_STRING_NOTE_INTEGRATION_KERNEL_CONVENTION,
        STRING_NOTE_INTEGRATION_KERNEL_CONVENTION,
        c1_genus2_topology_correction,
        c1_sphere_normalized_genus2_kernel_multiplier,
        string_note_genus2_kernel_multiplier,
        xi_full_replacement_over_dimensionless,
    )
    from plumbing.genus2_moduli_rqmc import generate_rqmc_design
    from plumbing.genus2_moduli_physical_mixture_rqmc import (
        PHYSICAL_MIXTURE_SAMPLING_SCHEME,
        generate_physical_mixture_design,
    )
    from plumbing.genus2_siegel_fundamental_domain import SIEGEL_VOLUME_G2
    from plumbing.liouville_genus2_ccy import liouville_genus2_ccy_partition
    from plumbing.liouville_genus2_glasses import liouville_genus2_glasses_partition
    from plumbing.monte_carlo_integrate_genus2_c1 import (
        RescaledLiouvillePartition,
        _load_resume_rows,
        _resume_config,
        _write_csv,
        cached_period_solution_objects,
        canonicalize_string_note_kernel_row,
        estimate_from_transformed_values,
        evaluate_liouville_rescaled,
        evaluate_noncompact_scalar,
        kernel_det_im_power,
        node_stack_integration_weight,
        omega_from_csv_row,
        sampling_scheme_for_rows,
        summarize_rqmc_rows,
        validate_covariant_period_frames,
    )


def run_checks() -> None:
    with TemporaryDirectory() as temporary_directory:
        root = Path(temporary_directory)
        source_csv = root / "source.csv"
        source_csv.write_text("rqmc_node_id,sample_index\nn0,0\nn1,1\n")
        selected = [
            {"rqmc_node_id": "n0", "sample_index": "0"},
            {"rqmc_node_id": "n1", "sample_index": "1"},
        ]
        args = argparse.Namespace(
            source_csv=source_csv,
            out_dir=root / "output",
            resume=False,
            rqmc_replicate=0,
            block_order_low=8,
            block_order_high=8,
        )
        config = _resume_config(args, selected)
        config_path = root / "run_config.json"
        samples_path = root / "samples.csv"
        if _load_resume_rows(
            csv_path=samples_path,
            config_path=config_path,
            expected_config=config,
            resume=False,
        ):
            raise AssertionError("fresh run unexpectedly loaded resume rows")
        _write_csv(
            samples_path,
            [{"rqmc_node_id": "n0", "sample_index": 0, "status": "ok"}],
        )
        resumed = _load_resume_rows(
            csv_path=samples_path,
            config_path=config_path,
            expected_config=config,
            resume=True,
        )
        if resumed["rqmc:n0"]["status"] != "ok":
            raise AssertionError("successful resume row was not recovered")
        changed_args = argparse.Namespace(**vars(args))
        changed_args.block_order_high = 10
        changed_config = _resume_config(changed_args, selected)
        try:
            _load_resume_rows(
                csv_path=samples_path,
                config_path=config_path,
                expected_config=changed_config,
                resume=True,
            )
        except ValueError as exc:
            if "do not match" not in str(exc):
                raise
        else:
            raise AssertionError("resume accepted a changed CFT order")

    rescaled = RescaledLiouvillePartition(
        scaled_partition=2.5,
        log_threshold_factor=math.log(1.0e-200),
    )
    if not math.isclose(rescaled.log_partition, math.log(2.5e-200), rel_tol=1.0e-15):
        raise AssertionError("Liouville logarithmic threshold restoration is incorrect")
    if not math.isclose(rescaled.raw_partition, 2.5e-200, rel_tol=1.0e-13):
        raise AssertionError("representable Liouville threshold factor was not restored")

    legacy = canonicalize_string_note_kernel_row(
        {
            "transformed_integrand_low": math.pi,
            "transformed_integrand_high": 2.0 * math.pi,
        }
    )
    if legacy["integration_kernel_convention"] != STRING_NOTE_INTEGRATION_KERNEL_CONVENTION:
        raise AssertionError("legacy row did not acquire the string-note convention")
    expected_legacy = (
        xi_full_replacement_over_dimensionless()
        * c1_sphere_normalized_genus2_kernel_multiplier()
        * math.pi
    )
    if not math.isclose(float(legacy["transformed_integrand_low"]), expected_legacy):
        raise AssertionError(
            "legacy unit-Mumford row did not acquire the Xi and final c=1 factors"
        )
    if not math.isclose(
        float(legacy["string_note_kernel_multiplier"]),
        c1_sphere_normalized_genus2_kernel_multiplier(),
    ):
        raise AssertionError("saved string-note kernel multiplier is incorrect")
    current = canonicalize_string_note_kernel_row(
        {
            "integration_kernel_convention": STRING_NOTE_INTEGRATION_KERNEL_CONVENTION,
            "transformed_integrand_low": 1.0,
            "transformed_integrand_high": 2.0,
        }
    )
    if float(current["transformed_integrand_high"]) != 2.0:
        raise AssertionError("current string-note row was rescaled twice")

    pre_sphere_xi = canonicalize_string_note_kernel_row(
        {
            "integration_kernel_convention": (
                PRE_SPHERE_XI_STRING_NOTE_INTEGRATION_KERNEL_CONVENTION
            ),
            "transformed_integrand_low": 1.0,
            "transformed_integrand_high": 2.0,
        }
    )
    if not math.isclose(
        float(pre_sphere_xi["transformed_integrand_low"]),
        c1_genus2_topology_correction(),
    ):
        raise AssertionError("Xi pre-sphere row did not acquire exactly the topology factor")

    old_string_note = canonicalize_string_note_kernel_row(
        {
            "integration_kernel_convention": (
                LEGACY_STRING_NOTE_DIMENSIONLESS_SCALAR_KERNEL_CONVENTION
            ),
            "transformed_integrand_low": 1.0,
            "transformed_integrand_high": 2.0,
            "factorized_density_low": math.pi,
            "factorized_log_density_low": math.log(math.pi),
        }
    )
    if not math.isclose(
        float(old_string_note["transformed_integrand_low"]),
        xi_full_replacement_over_dimensionless()
        * c1_genus2_topology_correction(),
    ):
        raise AssertionError("old string-note row did not acquire Xi and topology factors")
    if not math.isclose(
        float(old_string_note["factorized_density_low"]),
        math.pi * xi_full_replacement_over_dimensionless(),
    ):
        raise AssertionError("old unit-Mumford density did not acquire Xi's scalar normalization")

    common = {
        "b": 1.0,
        "block_order": 2,
        "quadrature_order": 2,
        "quadrature_scheme": "primary-gaussian",
        "dps": 24,
        "include_vacuum_seed": False,
        "store_samples": False,
    }
    theta_q = (0.07 + 0.01j, -0.08 + 0.005j, 0.06 - 0.007j)
    theta_raw = liouville_genus2_ccy_partition(
        q1=theta_q[0], q2=theta_q[1], q3=theta_q[2], propagator_shift=0.0, **common
    ).value
    theta_scaled = liouville_genus2_ccy_partition(
        q1=theta_q[0], q2=theta_q[1], q3=theta_q[2], propagator_shift=1.0, **common
    ).value
    theta_restored = theta_scaled * abs(theta_q[0] * theta_q[1] * theta_q[2]) ** 2
    if abs(theta_restored / theta_raw - 1.0) > 1.0e-12:
        raise AssertionError("theta log-domain threshold factorization changed the partition")

    glasses_q = (0.09 - 0.006j, 0.065 + 0.008j, -0.075 + 0.004j)
    glasses_raw = liouville_genus2_glasses_partition(
        q_left=glasses_q[0],
        q_right=glasses_q[1],
        q_bridge=glasses_q[2],
        propagator_shift=0.0,
        **common,
    ).value
    glasses_scaled = liouville_genus2_glasses_partition(
        q_left=glasses_q[0],
        q_right=glasses_q[1],
        q_bridge=glasses_q[2],
        propagator_shift=1.0,
        **common,
    ).value
    glasses_restored = glasses_scaled * abs(
        glasses_q[0] * glasses_q[1] * glasses_q[2]
    ) ** 2
    if abs(glasses_restored / glasses_raw - 1.0) > 1.0e-12:
        raise AssertionError("glasses log-domain threshold factorization changed the partition")

    cusp_logs = (-5.0 + 0.0j, -3606.0 + 0.0j, -4.0 + 0.0j)
    cusp_q = tuple(math.exp(max(value.real, -690.0)) + 0.0j for value in cusp_logs)
    cusp_liouville = evaluate_liouville_rescaled(
        "theta",
        cusp_q,
        log_q_values=cusp_logs,
        block_order=0,
        quadrature_order=1,
        quadrature_scheme="primary-gaussian",
        dps=18,
        vacuum_word_length=2,
        vacuum_oscillator_level=8,
    )
    if not math.isfinite(cusp_liouville.scaled_partition) or cusp_liouville.scaled_partition <= 0.0:
        raise AssertionError("logarithmic-cusp Liouville evaluation is not finite")
    if cusp_liouville.log_threshold_factor != 2.0 * sum(value.real for value in cusp_logs):
        raise AssertionError("logarithmic-cusp threshold used the surrogate q")

    two_pi = 2.0 * math.pi
    cusp_omega = 1.0j * np.asarray(
        [[9.0 / two_pi, 4.0 / two_pi], [4.0 / two_pi, 3610.0 / two_pi]],
        dtype=np.complex128,
    )
    cusp_scalar, _tail, primitive_count = evaluate_noncompact_scalar(
        "theta",
        cusp_q,
        cusp_omega,
        log_q_values=cusp_logs,
        word_length=4,
        max_mode=40,
        tolerance=1.0e-14,
    )
    if not math.isfinite(cusp_scalar) or cusp_scalar <= 0.0 or primitive_count != 1:
        raise AssertionError("logarithmic-cusp scalar did not reduce to the surviving handle")

    double_cusp_logs = (-800.0 + 0.0j, -900.0 + 0.0j, -4.0 + 0.0j)
    double_cusp_q = tuple(
        math.exp(max(value.real, -690.0)) + 0.0j for value in double_cusp_logs
    )
    double_cusp_scalar, _tail, primitive_count = evaluate_noncompact_scalar(
        "theta",
        double_cusp_q,
        cusp_omega,
        log_q_values=double_cusp_logs,
        word_length=4,
        max_mode=40,
        tolerance=1.0e-14,
    )
    expected_zero_mode = float(np.linalg.det(cusp_omega.imag) ** -0.5)
    if primitive_count != 0 or not math.isclose(
        double_cusp_scalar,
        expected_zero_mode,
        rel_tol=1.0e-14,
    ):
        raise AssertionError("double theta cusp did not reduce to the rank-zero scalar limit")

    constant = 2.75
    result = estimate_from_transformed_values([constant] * 16)
    expected = GENUS2_GENERIC_STACK_WEIGHT * SIEGEL_VOLUME_G2 * constant
    if not math.isclose(float(result["estimate"]), expected, rel_tol=1.0e-15):
        raise AssertionError("invariant-measure importance factor is incorrect")
    if float(result["standard_error"]) != 0.0:
        raise AssertionError("constant transformed integrand acquired Monte Carlo variance")
    if not math.isclose(float(result["coarse_domain_estimate_before_stack_weight"]), 2.0 * expected):
        raise AssertionError("generic genus-two stack weight was not applied exactly once")
    if not math.isclose(float(result["effective_sample_size"]), 16.0, rel_tol=1.0e-15):
        raise AssertionError("effective sample size is wrong for constant weights")
    if not math.isclose(float(result["largest_sample_fraction"]), 1.0 / 16.0):
        raise AssertionError("largest-sample fraction is wrong")

    uneven = estimate_from_transformed_values([0.0, 0.0, 1.0, 9.0])
    if not math.isclose(float(uneven["effective_sample_size"]), 100.0 / 82.0):
        raise AssertionError("effective sample size is wrong for uneven weights")
    if not math.isclose(float(uneven["largest_sample_fraction"]), 0.9):
        raise AssertionError("heavy-tail diagnostic is wrong")

    design, _summaries = generate_rqmc_design(
        replicate_count=8,
        power=6,
        base_seed=4411,
        marginal_bins=8,
    )
    for row in design:
        row["status"] = "ok"
        row["test_value"] = constant
        if not math.isclose(
            node_stack_integration_weight(row),
            float(row["rqmc_stack_integration_weight"]),
            rel_tol=1.0e-15,
        ):
            raise AssertionError("RQMC node weight was not preserved")
    if sampling_scheme_for_rows(design) != "scrambled_sobol_minkowski_importance":
        raise AssertionError("RQMC sampling scheme was not detected")
    rqmc = summarize_rqmc_rows(design, value_key="test_value")
    if rqmc is None:
        raise AssertionError("complete RQMC scrambles were not summarized")
    if not math.isclose(
        float(rqmc["volume_calibrated_estimate"]),
        expected,
        rel_tol=2.0e-15,
    ):
        raise AssertionError("RQMC exact-volume control failed for a constant")
    if abs(float(rqmc["estimate"]) - expected) > 5.0 * float(rqmc["standard_error"]):
        raise AssertionError("raw RQMC estimate misses the constant control")
    if summarize_rqmc_rows(design[:-1], value_key="test_value") is not None:
        raise AssertionError("an incomplete RQMC scramble produced a headline estimate")

    physical_design, _physical_summaries = generate_physical_mixture_design(
        replicate_count=4,
        power=6,
        base_seed=77103,
    )
    cached_q = (0.1 + 0.0j, 0.08 + 0.0j, 0.06 + 0.0j)
    cached_logs = tuple(complex(np.log(value)) for value in cached_q)
    recovery_row = {
        "sample_index": physical_design[0]["sample_index"],
        "rqmc_node_id": physical_design[0]["rqmc_node_id"],
        "status": "ok",
        "topology": "theta",
        "symplectic_word": "I",
        "chart_status": "reference-q-envelope",
        "atlas_period_algorithm": "holomorphic-form-collocation",
        "atlas_period_map_region": "holomorphic-bulk",
        "atlas_plumbing_geometry_margin": "0.2",
        "atlas_period_overlap_residual": "",
        "atlas_inverse_seed_source": "unit-test",
        "atlas_q_max": "0.1",
        "atlas_period_residual": "2e-8",
        "atlas_period_map_stability": "3e-8",
        "period_algorithm": "holomorphic-form-collocation",
        "period_map_region": "holomorphic-bulk",
        "period_overlap_residual": "",
        "period_agreement_tolerance": "1e-5",
        "period_certified_error_bound": "3e-8",
        "fixed_q_period_residual": "2e-8",
        "fixed_q_period_map_step": "3e-8",
        "q_refined": "False",
        "final_period_residual": "2e-8",
        "final_period_map_step": "3e-8",
        "period_map_low_order": "20",
        "period_map_high_order": "24",
        "period_map_validation_order": "24",
        "period_seam_residual": "1e-7",
        "period_symmetry_error": "1e-10",
        "reinverse_message": "cached",
        "reinverse_nfev": "0",
        "max_tau_shift": "0",
    }
    for i in range(4):
        for j in range(4):
            recovery_row[f"symplectic_m{i}{j}"] = str(int(i == j))
    for index, (q_value, log_value) in enumerate(
        zip(cached_q, cached_logs), start=1
    ):
        recovery_row[f"atlas_q{index}"] = str(q_value)
        recovery_row[f"atlas_log_q{index}"] = str(log_value)
        recovery_row[f"final_q{index}"] = str(q_value)
        recovery_row[f"final_log_q{index}"] = str(log_value)
    cached_chart, cached_certificate = cached_period_solution_objects(
        physical_design[0], recovery_row, tolerance=1.0e-5
    )
    if cached_chart.word != "I" or cached_certificate.q != cached_q:
        raise AssertionError("cached period solution was not rehydrated exactly")
    try:
        cached_period_solution_objects(
            physical_design[0],
            {**recovery_row, "final_period_residual": "2e-5"},
            tolerance=1.0e-5,
        )
    except ValueError as exc:
        if "exceeds" not in str(exc):
            raise
    else:
        raise AssertionError("cached period solution bypassed its numerical bar")
    omega_fundamental = omega_from_csv_row(physical_design[0])
    marking = np.asarray(
        [[1, 0, 1, 0], [0, 1, 0, -1], [0, 0, 1, 0], [0, 0, 0, 1]],
        dtype=np.int64,
    )
    omega_plumbing = omega_fundamental + np.asarray([[1, 0], [0, -1]])
    validate_covariant_period_frames(
        omega_fundamental,
        marking,
        omega_plumbing,
    )
    try:
        validate_covariant_period_frames(
            omega_fundamental,
            marking,
            omega_fundamental,
        )
    except ValueError as exc:
        if "not the saved" not in str(exc):
            raise
    else:
        raise AssertionError("covariant frame guard accepted the wrong plumbing matrix")
    for row in physical_design:
        row["status"] = "ok"
        row["test_value"] = float(row["det_im_omega"]) ** -3
        if kernel_det_im_power(row) != 0:
            raise AssertionError("physical RQMC row retained a det(Y)^3 observable")
    if sampling_scheme_for_rows(physical_design) != PHYSICAL_MIXTURE_SAMPLING_SCHEME:
        raise AssertionError("physical-mixture sampling scheme was not detected")
    physical_summary = summarize_rqmc_rows(
        physical_design,
        value_key="test_value",
    )
    if physical_summary is None:
        raise AssertionError("complete physical-mixture scrambles were not summarized")
    coarse_replicate_controls = []
    for replicate in sorted({int(row["rqmc_replicate"]) for row in physical_design}):
        replicate_rows = [
            row
            for row in physical_design
            if int(row["rqmc_replicate"]) == replicate
        ]
        coarse_replicate_controls.append(
            sum(
                float(row["rqmc_stack_integration_weight"])
                * float(row["test_value"])
                / GENUS2_GENERIC_STACK_WEIGHT
                for row in replicate_rows
            )
        )
    coarse_control_estimate = float(np.mean(coarse_replicate_controls))
    if not math.isclose(
        float(physical_summary["estimate"]),
        GENUS2_GENERIC_STACK_WEIGHT * coarse_control_estimate,
        rel_tol=2.0e-15,
    ):
        raise AssertionError("physical-mixture stack weight was not applied exactly once")
    physical_control = GENUS2_GENERIC_STACK_WEIGHT * SIEGEL_VOLUME_G2
    if abs(float(physical_summary["estimate"]) - physical_control) > (
        6.0 * float(physical_summary["standard_error"])
    ):
        raise AssertionError("physical-mixture invariant-volume control failed")
    if physical_summary["volume_calibrated_estimate"] is not None:
        raise AssertionError("physical-measure rows received invariant-volume calibration")
    if summarize_rqmc_rows(physical_design[:-1], value_key="test_value") is not None:
        raise AssertionError("an incomplete physical mixture produced a headline estimate")

    print("monte_carlo_integrate_genus2_c1 checks passed")


if __name__ == "__main__":
    run_checks()
