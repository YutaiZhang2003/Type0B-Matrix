#!/usr/bin/env python3
"""Independently audit the production sampling measure with Siegel volume.

The production proposal samples ``d^3X dt1 dt3 dr``.  This audit reconstructs
the physical Jacobian, equal-mixture density, determinant, proposal-count
normalization, and genus-two stack weight directly from the saved coordinates.
It then integrates ``det(Im Omega)^(-3)`` and compares the coarse-domain result
with ``pi^3/270``.  No CFT value is used.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DESIGN = ROOT / (
    "plumbing/results/genus2_c1_moduli_mc/physical_mixture_R8_C4_M256"
)
COARSE_SIEGEL_VOLUME_G2 = math.pi**3 / 270.0
GENUS2_STACK_WEIGHT = 0.5
SQRT3_OVER_2 = math.sqrt(3.0) / 2.0


def _relative_error(value: float, target: float) -> float:
    return abs(float(value) / float(target) - 1.0)


def _mean_and_se(values: list[float]) -> tuple[float, float]:
    if len(values) < 2:
        raise ValueError("at least two complete scrambles are required")
    mean = math.fsum(values) / len(values)
    variance = math.fsum((value - mean) ** 2 for value in values) / (
        len(values) - 1
    )
    return mean, math.sqrt(variance / len(values))


def audit_design(design_dir: Path) -> dict[str, object]:
    nodes_path = design_dir / "domain_nodes.csv"
    summary_path = design_dir / "summary.json"
    with nodes_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if not rows:
        raise ValueError("production domain-node table is empty")

    components = tuple(
        (
            float(component["rate_t1"]),
            float(component["rate_t3"]),
        )
        for component in summary["components"]
    )
    component_count = len(components)
    if component_count == 0:
        raise ValueError("production mixture has no components")

    grouped: dict[int, list[dict[str, str]]] = {}
    maximum_residuals = {
        "physical_jacobian_weight": 0.0,
        "log_mixture_density": 0.0,
        "log_physical_weight": 0.0,
        "det_im_omega": 0.0,
        "stack_node_weight": 0.0,
    }
    for row in rows:
        replicate = int(row["rqmc_replicate"])
        grouped.setdefault(replicate, []).append(row)

        t1 = float(row["rqmc_t1"])
        t3 = float(row["rqmc_t3"])
        r = float(row["rqmc_r"])
        jacobian = 0.5 * SQRT3_OVER_2**3 * math.exp(3.0 * t1 + t3)
        mixture_density = math.fsum(
            rate_t1
            * rate_t3
            * math.exp(-rate_t1 * t1 - rate_t3 * t3)
            for rate_t1, rate_t3 in components
        ) / component_count
        reconstructed_weight = jacobian / mixture_density
        stored_weight = float(row["rqmc_physical_measure_weight"])

        y11 = float(row["y11"])
        y12 = float(row["y12"])
        y22 = float(row["y22"])
        determinant_from_matrix = y11 * y22 - y12 * y12
        determinant_from_coordinates = (
            SQRT3_OVER_2**2
            * math.exp(2.0 * t1)
            * (math.exp(t3) - 0.25 * r * r)
        )
        stored_determinant = float(row["det_im_omega"])
        proposal_count = int(row["rqmc_proposal_count"])
        reconstructed_stack_weight = (
            GENUS2_STACK_WEIGHT * reconstructed_weight / proposal_count
        )

        maximum_residuals["physical_jacobian_weight"] = max(
            maximum_residuals["physical_jacobian_weight"],
            _relative_error(stored_weight, reconstructed_weight),
        )
        maximum_residuals["log_mixture_density"] = max(
            maximum_residuals["log_mixture_density"],
            abs(float(row["rqmc_log_mixture_density"]) - math.log(mixture_density)),
        )
        maximum_residuals["log_physical_weight"] = max(
            maximum_residuals["log_physical_weight"],
            abs(float(row["rqmc_log_physical_measure_weight"]) - math.log(reconstructed_weight)),
        )
        maximum_residuals["det_im_omega"] = max(
            maximum_residuals["det_im_omega"],
            _relative_error(stored_determinant, determinant_from_matrix),
            _relative_error(stored_determinant, determinant_from_coordinates),
        )
        maximum_residuals["stack_node_weight"] = max(
            maximum_residuals["stack_node_weight"],
            _relative_error(
                float(row["rqmc_stack_integration_weight"]),
                reconstructed_stack_weight,
            ),
        )

    coarse_replicates: list[float] = []
    stack_replicates: list[float] = []
    accepted_counts: list[int] = []
    proposal_counts: list[int] = []
    for replicate in sorted(grouped):
        replicate_rows = grouped[replicate]
        accepted_count = int(replicate_rows[0]["rqmc_domain_count"])
        proposal_count = int(replicate_rows[0]["rqmc_proposal_count"])
        if len(replicate_rows) != accepted_count:
            raise AssertionError(
                f"replicate {replicate} has {len(replicate_rows)} saved domain "
                f"rows but declares {accepted_count}"
            )
        if any(
            int(row["rqmc_proposal_count"]) != proposal_count
            for row in replicate_rows
        ):
            raise AssertionError(f"replicate {replicate} mixes proposal counts")

        coarse = math.fsum(
            float(row["rqmc_physical_measure_weight"])
            / float(row["det_im_omega"]) ** 3
            / proposal_count
            for row in replicate_rows
        )
        stack = math.fsum(
            float(row["rqmc_stack_integration_weight"])
            / float(row["det_im_omega"]) ** 3
            for row in replicate_rows
        )
        if not math.isclose(
            stack,
            GENUS2_STACK_WEIGHT * coarse,
            rel_tol=3.0e-15,
            abs_tol=0.0,
        ):
            raise AssertionError(f"replicate {replicate} applies the stack incorrectly")
        coarse_replicates.append(coarse)
        stack_replicates.append(stack)
        accepted_counts.append(accepted_count)
        proposal_counts.append(proposal_count)

    coarse_mean, coarse_se = _mean_and_se(coarse_replicates)
    stack_mean, stack_se = _mean_and_se(stack_replicates)
    z_score = (coarse_mean - COARSE_SIEGEL_VOLUME_G2) / coarse_se

    recorded_replicates = {
        int(item["replicate"]): float(item["invariant_volume_control"])
        for item in summary["replicates"]
    }
    for replicate, value in zip(sorted(grouped), coarse_replicates):
        if not math.isclose(
            value,
            recorded_replicates[replicate],
            rel_tol=3.0e-15,
            abs_tol=0.0,
        ):
            raise AssertionError(
                f"replicate {replicate} disagrees with its saved volume control"
            )

    if any(value > 5.0e-13 for value in maximum_residuals.values()):
        raise AssertionError(f"saved sampling fields are inconsistent: {maximum_residuals}")
    if abs(z_score) > 5.0:
        raise AssertionError("production hyperbolic-volume control misses by over 5 sigma")

    return {
        "scope": "independent reconstruction from saved production coordinates",
        "integrand": "det(Im Omega)^(-3)",
        "coarse_domain_exact_volume": COARSE_SIEGEL_VOLUME_G2,
        "coarse_domain_estimate": coarse_mean,
        "coarse_domain_standard_error": coarse_se,
        "coarse_domain_z_score": z_score,
        "stack_weight": GENUS2_STACK_WEIGHT,
        "stack_volume_exact": GENUS2_STACK_WEIGHT * COARSE_SIEGEL_VOLUME_G2,
        "stack_volume_estimate": stack_mean,
        "stack_volume_standard_error": stack_se,
        "replicate_count": len(coarse_replicates),
        "proposal_counts": proposal_counts,
        "accepted_domain_counts": accepted_counts,
        "coarse_replicate_estimates": coarse_replicates,
        "maximum_saved_field_residuals": maximum_residuals,
    }


def run(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--design-dir", type=Path, default=DEFAULT_DESIGN)
    parser.add_argument("--out-json", type=Path, default=None)
    args = parser.parse_args(list(argv) if argv is not None else None)

    result = audit_design(args.design_dir)
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print("Genus-two production hyperbolic-volume sampling audit")
    print(
        "  coarse volume = "
        f"{result['coarse_domain_estimate']:.16e} +/- "
        f"{result['coarse_domain_standard_error']:.16e}"
    )
    print(f"  exact pi^3/270 = {result['coarse_domain_exact_volume']:.16e}")
    print(f"  z score = {result['coarse_domain_z_score']:.6f}")
    print(
        "  stack volume = "
        f"{result['stack_volume_estimate']:.16e} +/- "
        f"{result['stack_volume_standard_error']:.16e}"
    )
    print(f"  exact stack volume = {result['stack_volume_exact']:.16e}")
    print(
        "  maximum saved-field residual = "
        f"{max(result['maximum_saved_field_residuals'].values()):.3e}"
    )
    if args.out_json is not None:
        print(f"  wrote {args.out_json}")


if __name__ == "__main__":
    run()
