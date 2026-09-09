#!/usr/bin/env python3
"""Checks for strict RQMC CFT assembly and nested reuse."""

from __future__ import annotations

import math

try:
    from assemble_genus2_c1_rqmc import assemble_rows, index_evaluations
    from genus2_integrand_normalization import (
        GENUS2_GENERIC_STACK_WEIGHT,
        STRING_NOTE_INTEGRATION_KERNEL_CONVENTION,
        c1_sphere_normalized_genus2_kernel_multiplier,
        xi_full_replacement_over_dimensionless,
    )
    from genus2_moduli_rqmc import generate_rqmc_design
    from genus2_moduli_physical_mixture_rqmc import generate_physical_mixture_design
    from genus2_siegel_fundamental_domain import SIEGEL_VOLUME_G2
except ImportError:  # pragma: no cover
    from plumbing.assemble_genus2_c1_rqmc import assemble_rows, index_evaluations
    from plumbing.genus2_integrand_normalization import (
        GENUS2_GENERIC_STACK_WEIGHT,
        STRING_NOTE_INTEGRATION_KERNEL_CONVENTION,
        c1_sphere_normalized_genus2_kernel_multiplier,
        xi_full_replacement_over_dimensionless,
    )
    from plumbing.genus2_moduli_rqmc import generate_rqmc_design
    from plumbing.genus2_moduli_physical_mixture_rqmc import (
        generate_physical_mixture_design,
    )
    from plumbing.genus2_siegel_fundamental_domain import SIEGEL_VOLUME_G2


def run_checks() -> None:
    design, _summaries = generate_rqmc_design(
        replicate_count=8,
        power=6,
        base_seed=90210,
        marginal_bins=8,
    )
    evaluations = []
    constant = 2.75
    for row in design:
        evaluation = dict(row)
        evaluation.update(
            {
                "status": "ok",
                "error": "",
                "integration_kernel_convention": STRING_NOTE_INTEGRATION_KERNEL_CONVENTION,
                "transformed_integrand_low": constant,
                "transformed_integrand_high": constant,
            }
        )
        evaluations.append(evaluation)
    combined, diagnostics = assemble_rows(design, evaluations)
    if not diagnostics["complete"] or not diagnostics["headline_available"]:
        raise AssertionError("complete synthetic design did not produce a headline")
    high = diagnostics["high_order_summary"]
    expected = GENUS2_GENERIC_STACK_WEIGHT * SIEGEL_VOLUME_G2 * constant
    if not math.isclose(
        float(high["volume_calibrated_estimate"]), expected, rel_tol=2.0e-15
    ):
        raise AssertionError("assembled exact-volume constant control failed")
    if len(combined) != len(design):
        raise AssertionError("assembler changed the design node count")

    legacy_row = dict(evaluations[0])
    legacy_row.pop("integration_kernel_convention")
    legacy_row["transformed_integrand_low"] = math.pi
    legacy_row["transformed_integrand_high"] = 2.0 * math.pi
    legacy_node_id = str(legacy_row["rqmc_node_id"])
    canonical_legacy = index_evaluations([legacy_row])[legacy_node_id]
    multiplier = c1_sphere_normalized_genus2_kernel_multiplier()
    if not math.isclose(
        float(canonical_legacy["transformed_integrand_high"]),
        2.0 * math.pi * multiplier * xi_full_replacement_over_dimensionless(),
        rel_tol=2.0e-15,
    ):
        raise AssertionError(
            "assembler did not convert a legacy row by the Xi and final c=1 factors"
        )

    failed_then_retried = [dict(evaluations[0]), dict(evaluations[0])]
    failed_then_retried[0].update({"status": "failed", "error": "first attempt"})
    retried = index_evaluations(failed_then_retried)
    if retried[str(evaluations[0]["rqmc_node_id"])]["status"] != "ok":
        raise AssertionError("a certified retry did not replace the failed attempt")

    _combined, incomplete = assemble_rows(design, evaluations[:-1])
    if incomplete["headline_available"] or incomplete["missing_count"] != 1:
        raise AssertionError("assembler did not reject an incomplete scramble")

    wrong_coordinate = [dict(row) for row in evaluations]
    wrong_coordinate[0]["x11"] = float(wrong_coordinate[0]["x11"]) + 1.0e-8
    try:
        assemble_rows(design, wrong_coordinate)
    except ValueError as exc:
        if "evaluation disagrees in x11" not in str(exc):
            raise AssertionError("unexpected coordinate-mismatch diagnostic") from exc
    else:
        raise AssertionError("assembler accepted a node-id match at the wrong Omega")

    extended, _extended_summaries = generate_rqmc_design(
        replicate_count=8,
        power=7,
        base_seed=90210,
        marginal_bins=8,
    )
    reused, nested = assemble_rows(extended, evaluations)
    if nested["headline_available"]:
        raise AssertionError("nested extension reported before new nodes were evaluated")
    if nested["nested_reused_node_count"] != len(evaluations):
        raise AssertionError("nested extension did not recognize all reusable old nodes")
    old = {str(row["rqmc_node_id"]): row for row in evaluations}
    for row in reused:
        node_id = str(row["rqmc_node_id"])
        if node_id in old and row["status"] != "ok":
            raise AssertionError("nested extension lost a reusable old CFT value")

    physical_design, _physical_summaries = generate_physical_mixture_design(
        replicate_count=4,
        power=6,
        base_seed=92017,
    )
    physical_evaluations = []
    for row in physical_design:
        evaluation = dict(row)
        control = float(row["det_im_omega"]) ** -3
        evaluation.update(
            {
                "status": "ok",
                "error": "",
                "integration_kernel_convention": STRING_NOTE_INTEGRATION_KERNEL_CONVENTION,
                "transformed_integrand_low": control,
                "transformed_integrand_high": control,
            }
        )
        physical_evaluations.append(evaluation)
    _physical_combined, physical_diagnostics = assemble_rows(
        physical_design,
        physical_evaluations,
    )
    physical_high = physical_diagnostics["high_order_summary"]
    if physical_high is None or physical_high["volume_calibrated_estimate"] is not None:
        raise AssertionError("assembler misidentified the physical-measure estimator")
    physical_expected = GENUS2_GENERIC_STACK_WEIGHT * SIEGEL_VOLUME_G2
    if abs(float(physical_high["estimate"]) - physical_expected) > (
        6.0 * float(physical_high["standard_error"])
    ):
        raise AssertionError("assembled physical invariant-volume control failed")

    print("assemble_genus2_c1_rqmc checks passed")


if __name__ == "__main__":
    run_checks()
