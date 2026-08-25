#!/usr/bin/env python3
"""End-to-end normalization checks for the compact genus-two release."""

from __future__ import annotations

import math

try:
    from export_genus2_free_energy_release import (
        DEFAULT_INPUT,
        DEFAULT_OUTPUT,
        _load_csv,
        _release_rows,
    )
    from genus2_integrand_normalization import (
        LEGACY_STRING_NOTE_DIMENSIONLESS_SCALAR_KERNEL_CONVENTION,
        PRE_SPHERE_XI_STRING_NOTE_INTEGRATION_KERNEL_CONVENTION,
        STRING_NOTE_INTEGRATION_KERNEL_CONVENTION,
        integration_kernel_scale_to_current,
    )
except ImportError:  # pragma: no cover
    from plumbing.export_genus2_free_energy_release import (
        DEFAULT_INPUT,
        DEFAULT_OUTPUT,
        _load_csv,
        _release_rows,
    )
    from plumbing.genus2_integrand_normalization import (
        LEGACY_STRING_NOTE_DIMENSIONLESS_SCALAR_KERNEL_CONVENTION,
        PRE_SPHERE_XI_STRING_NOTE_INTEGRATION_KERNEL_CONVENTION,
        STRING_NOTE_INTEGRATION_KERNEL_CONVENTION,
        integration_kernel_scale_to_current,
    )


def _source_rows() -> list[dict[str, str]]:
    rows = []
    for index in range(39):
        radius = 0.5 + 1.5 * index / 38.0
        rows.append(
            {
                "radius": repr(radius),
                "free_energy_over_gs_squared": "1.0",
                "rqmc_scramble_standard_error": "0.1",
                "normalized_worldsheet_shape": "1.0",
                "normalized_worldsheet_shape_jackknife_se": "0.0",
                "contribution_effective_sample_size": "10.0",
                "largest_node_fraction": "0.1",
            }
        )
    return rows


def run_checks() -> None:
    if not math.isclose(
        integration_kernel_scale_to_current(
            LEGACY_STRING_NOTE_DIMENSIONLESS_SCALAR_KERNEL_CONVENTION
        ),
        1.0 / math.pi,
        rel_tol=2.0e-15,
    ):
        raise AssertionError("legacy dimensionless kernel does not migrate by 1/pi")
    if integration_kernel_scale_to_current(
        PRE_SPHERE_XI_STRING_NOTE_INTEGRATION_KERNEL_CONVENTION
    ) != 2.0:
        raise AssertionError("pre-sphere Xi kernel does not migrate by two")
    if integration_kernel_scale_to_current(
        STRING_NOTE_INTEGRATION_KERNEL_CONVENTION
    ) != 1.0:
        raise AssertionError("current kernel migration is not idempotent")

    release = _release_rows(
        _source_rows(),
        source_kernel_convention=(
            LEGACY_STRING_NOTE_DIMENSIONLESS_SCALAR_KERNEL_CONVENTION
        ),
    )
    radius_one = release[19]
    if not math.isclose(
        float(radius_one["connected_logZ_genus2_over_gs_squared"]),
        1.0 / math.pi,
        rel_tol=2.0e-15,
    ):
        raise AssertionError("release omitted the source-to-v3 kernel migration")
    if not math.isclose(
        float(radius_one["coarse_domain_integral_K2_c1"]),
        2.0 / math.pi,
        rel_tol=2.0e-15,
    ):
        raise AssertionError("release did not undo the stack weight exactly once")
    if not math.isclose(
        float(radius_one["thermal_free_energy_over_gs_squared"]),
        -1.0 / (2.0 * math.pi**2 * float(radius_one["radius"])),
        rel_tol=2.0e-15,
    ):
        raise AssertionError("release thermal-circle conversion is incorrect")
    if radius_one["integration_kernel_convention"] != (
        STRING_NOTE_INTEGRATION_KERNEL_CONVENTION
    ):
        raise AssertionError("release did not record the final kernel convention")

    archived = next(
        row for row in _load_csv(DEFAULT_INPUT)
        if math.isclose(float(row["radius"]), 1.0)
    )
    published = next(
        row for row in _load_csv(DEFAULT_OUTPUT / "radius_sweep.csv")
        if math.isclose(float(row["radius"]), 1.0)
    )
    archived_stack_value = float(archived["free_energy_over_gs_squared"])
    published_stack_value = float(
        published["connected_logZ_genus2_over_gs_squared"]
    )
    if not math.isclose(
        published_stack_value,
        archived_stack_value / math.pi,
        rel_tol=2.0e-15,
    ):
        raise AssertionError("published R=1 value did not migrate the archive by 1/pi")
    if not math.isclose(
        float(published["coarse_domain_integral_K2_c1"]),
        2.0 * published_stack_value,
        rel_tol=2.0e-15,
    ):
        raise AssertionError("published R=1 value applies the stack factor incorrectly")
    if published["integration_kernel_convention"] != (
        STRING_NOTE_INTEGRATION_KERNEL_CONVENTION
    ):
        raise AssertionError("published table does not identify the v3 convention")

    print("export_genus2_free_energy_release checks passed")


if __name__ == "__main__":
    run_checks()
