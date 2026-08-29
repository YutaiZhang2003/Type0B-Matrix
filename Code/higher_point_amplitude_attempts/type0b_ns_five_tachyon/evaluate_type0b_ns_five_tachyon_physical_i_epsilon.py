#!/usr/bin/env python3
"""Matrix-blind direct-subtraction driver near physical 1->4 kinematics."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Sequence

from type0b_ns_five_tachyon import (
    BRYNSFiveTachyonIntegrand,
    integrate_physical_i_epsilon_finite_part_qmc,
)
from type0b_ns_five_tachyon_domain import (
    physical_i_epsilon_frequencies,
    physical_i_epsilon_subtraction_audit,
)


SCRIPT_DIR = Path(__file__).resolve().parent
C_RECURSION_DIR = SCRIPT_DIR.parents[1] / "c_Recursion"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the Type-0B NS sphere five-tachyon worldsheet integral "
            "at physical positive energies plus a small +i-epsilon, using "
            "direct BRY polynomial subtraction and no remote continuation."
        )
    )
    parser.add_argument(
        "--energies",
        type=float,
        nargs=4,
        default=(0.25, 0.25, 0.25, 0.25),
        metavar=("E1", "E2", "E3", "E4"),
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.02,
        help="positive Feynman regulator; the physical value is epsilon->0+",
    )
    parser.add_argument(
        "--epsilon-weights",
        type=float,
        nargs=4,
        default=(1.0, 1.0, 1.0, 1.0),
        metavar=("NU1", "NU2", "NU3", "NU4"),
    )
    parser.add_argument(
        "--block-backend",
        choices=("hybrid", "h", "c"),
        default="h",
        help=(
            "production uses regulated h-recursion in the best CCY chart; "
            "c and hybrid are overlap-audit modes"
        ),
    )
    parser.add_argument("--hybrid-q-threshold", type=float, default=0.3)
    parser.add_argument(
        "--recursion-max-twice-level",
        type=int,
        default=-1,
        help=(
            "-1 uses direct coefficient-series truncation; explicit h or "
            "hybrid audits also require matched h/c cutoffs"
        ),
    )
    parser.add_argument(
        "--global-max-twice-levels", type=int, nargs=2, default=(4, 4)
    )
    parser.add_argument("--global-max-total-twice-level", type=int, default=6)
    parser.add_argument(
        "--momentum-orders",
        type=int,
        nargs=2,
        default=(5, 7),
        help=(
            "normal-threshold and smooth-continuum Gauss orders per fixed panel"
        ),
    )
    parser.add_argument("--momentum-maximum", type=float, default=2.0)
    parser.add_argument(
        "--momentum-refinement-shells",
        type=int,
        default=-1,
        help=(
            "-1 automatically spans each finite-part threshold with fixed "
            "factor-four panels; a nonnegative value requests that many "
            "shoulders (zero is one global Gauss panel)"
        ),
    )
    parser.add_argument(
        "--disable-momentum-singularity-subtraction",
        action="store_true",
        help=(
            "disable the production f(P*) subtraction and use only the "
            "threshold-centered composite Gauss rule"
        ),
    )
    parser.add_argument("--structure-precision", type=int, default=22)
    parser.add_argument("--central-charge-shift", type=float, default=1.0e-5)
    parser.add_argument(
        "--h-regulator-eta",
        type=float,
        default=None,
        help=(
            "for h-recursion, evaluate at b=exp(eta) while keeping the "
            "physical self-dual weights fixed"
        ),
    )
    parser.add_argument(
        "--h-regulator-etas",
        type=float,
        nargs="+",
        default=(0.16, 0.13, 0.10, 0.075, 0.055),
        help=(
            "log(b) nodes used to extrapolate every h-recursion coefficient "
            "at fixed physical weights"
        ),
    )
    parser.add_argument(
        "--h-regulator-polynomial-degree", type=int, default=3
    )
    parser.add_argument(
        "--h-regulator-comparison-degree", type=int, default=2
    )
    parser.add_argument(
        "--h-fit-variant",
        choices=("production", "comparison"),
        default="production",
    )
    parser.add_argument("--block-working-precision", type=int, default=45)
    parser.add_argument("--collar-radius", type=float, default=0.08)
    parser.add_argument(
        "--collar-radii",
        type=float,
        nargs="+",
        default=None,
        help=(
            "evaluate several collars with one kernel and common Sobol points; "
            "when supplied this replaces --collar-radius"
        ),
    )
    parser.add_argument(
        "--include-comparison-fit",
        action="store_true",
        help=(
            "also integrate the lower-degree coefficient fit using the same "
            "kernel and Sobol points"
        ),
    )
    parser.add_argument("--projection-radius", type=float, default=1.0e-5)
    parser.add_argument(
        "--face-collar-relative-tolerance",
        type=float,
        default=0.05,
        help=(
            "maximum sampled relative disagreement between the degree-zero "
            "face CFT polynomial and the full c-recursive density at the collar"
        ),
    )
    parser.add_argument(
        "--face-collar-absolute-tolerance", type=float, default=1.0e-10
    )
    parser.add_argument(
        "--face-collar-samples-per-orbit", type=int, default=3
    )
    parser.add_argument(
        "--face-collar-normal-angle-count", type=int, default=2
    )
    parser.add_argument(
        "--face-collar-reference-backend",
        choices=("h", "c"),
        default="c",
        help="near-collar full-CFT recursion used by the truncation certificate",
    )
    parser.add_argument(
        "--face-collar-reference-max-twice-levels",
        type=int,
        nargs=2,
        default=(6, 6),
        metavar=("LEFT", "RIGHT"),
        help="minimum direct c-series levels used by the collar certificate",
    )
    parser.add_argument(
        "--face-collar-reference-max-total-twice-level", type=int, default=10
    )
    parser.add_argument(
        "--face-collar-previous-reference-max-twice-levels",
        type=int,
        nargs=2,
        default=(4, 4),
        metavar=("LEFT", "RIGHT"),
    )
    parser.add_argument(
        "--face-collar-previous-reference-max-total-twice-level",
        type=int,
        default=6,
    )
    parser.add_argument(
        "--face-collar-reference-convergence-relative-tolerance",
        type=float,
        default=0.01,
    )
    parser.add_argument(
        "--face-collar-certificate-seed", type=int, default=20260830
    )
    parser.add_argument(
        "--enforce-face-collar-certificate",
        action="store_true",
        help=(
            "fail on the optional boundary diagnostic; normally it is recorded "
            "but not enforced because the full c block contains higher normal powers"
        ),
    )
    parser.add_argument(
        "--skip-face-collar-diagnostic",
        action="store_true",
        help="skip the non-blocking collar diagnostic (useful for independent array chunks)",
    )
    parser.add_argument(
        "--skip-corner-contribution",
        action="store_true",
        help=(
            "omit the deterministic corner finite part in this shard; the "
            "cluster reducer must restore a separately computed common value"
        ),
    )
    parser.add_argument("--bulk-sobol-power", type=int, default=4)
    parser.add_argument("--face-sobol-power", type=int, default=4)
    parser.add_argument("--replicates", type=int, default=4)
    parser.add_argument("--radial-power", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.block_backend == "hybrid":
        if abs(args.hybrid_q_threshold - 0.3) > 1.0e-15:
            raise ValueError(
                "the legacy hybrid audit is fixed at strict |q|<0.3"
            )
        if args.recursion_max_twice_level != -1:
            raise ValueError(
                "hybrid production requires matched plumbing-series cutoffs; "
                "use --recursion-max-twice-level=-1"
            )
    recursion_cutoff = (
        None
        if args.recursion_max_twice_level < 0
        else args.recursion_max_twice_level
    )
    audit = physical_i_epsilon_subtraction_audit(
        args.energies,
        args.epsilon,
        epsilon_weights=args.epsilon_weights,
        central_charge_shift=args.central_charge_shift,
    )
    if not audit["undeformed_positive_real_liouville_contours"]:
        raise ValueError(
            "epsilon is outside the residue-free physical chamber; reduce it"
        )
    if not audit["all_required_modes_degree_zero"]:
        raise NotImplementedError(
            "these energies require positive-degree diagonal counterterms"
        )

    outgoing = physical_i_epsilon_frequencies(
        args.energies,
        args.epsilon,
        epsilon_weights=args.epsilon_weights,
    )
    kernel = BRYNSFiveTachyonIntegrand(
        outgoing_energies=outgoing,
        block_backend=args.block_backend,
        hybrid_q_threshold=args.hybrid_q_threshold,
        recursion_max_twice_level=recursion_cutoff,
        global_max_twice_levels=args.global_max_twice_levels,
        global_max_total_twice_level=args.global_max_total_twice_level,
        momentum_orders=args.momentum_orders,
        momentum_maximum=args.momentum_maximum,
        structure_precision=args.structure_precision,
        central_charge_shift=args.central_charge_shift,
        h_regulator_eta=args.h_regulator_eta,
        h_regulator_etas=args.h_regulator_etas,
        h_regulator_polynomial_degree=args.h_regulator_polynomial_degree,
        h_regulator_comparison_degree=args.h_regulator_comparison_degree,
        h_fit_variant=args.h_fit_variant,
        block_working_precision=args.block_working_precision,
    )
    collar_radii = tuple(
        float(value)
        for value in (
            (args.collar_radius,)
            if args.collar_radii is None
            else args.collar_radii
        )
    )
    if not collar_radii or len(set(collar_radii)) != len(collar_radii):
        raise ValueError("collar radii must be a nonempty collection of distinct values")
    fit_variants = (
        ("production", "comparison")
        if args.include_comparison_fit and args.block_backend == "h"
        else (args.h_fit_variant,)
    )
    result_payloads: list[dict[str, object]] = []
    for fit_variant in fit_variants:
        kernel.set_h_fit_variant(fit_variant)
        for radius_index, radius in enumerate(collar_radii):
            run_diagnostic = (
                not args.skip_face_collar_diagnostic
                and fit_variant == "production"
            )
            result = integrate_physical_i_epsilon_finite_part_qmc(
                kernel,
                real_outgoing_energies=args.energies,
                epsilon=args.epsilon,
                epsilon_weights=args.epsilon_weights,
                face_collar_relative_tolerance=args.face_collar_relative_tolerance,
                face_collar_absolute_tolerance=args.face_collar_absolute_tolerance,
                face_collar_samples_per_orbit=args.face_collar_samples_per_orbit,
                face_collar_normal_angle_count=args.face_collar_normal_angle_count,
                face_collar_reference_backend=args.face_collar_reference_backend,
                face_collar_reference_max_twice_levels=(
                    args.face_collar_reference_max_twice_levels
                ),
                face_collar_reference_max_total_twice_level=(
                    args.face_collar_reference_max_total_twice_level
                ),
                face_collar_previous_reference_max_twice_levels=(
                    args.face_collar_previous_reference_max_twice_levels
                ),
                face_collar_previous_reference_max_total_twice_level=(
                    args.face_collar_previous_reference_max_total_twice_level
                ),
                face_collar_reference_convergence_relative_tolerance=(
                    args.face_collar_reference_convergence_relative_tolerance
                ),
                face_collar_certificate_seed=(
                    args.face_collar_certificate_seed + radius_index
                ),
                run_face_collar_diagnostic=run_diagnostic,
                enforce_face_collar_certificate=(
                    args.enforce_face_collar_certificate and run_diagnostic
                ),
                compute_corner_contribution=not args.skip_corner_contribution,
                collar_radius=radius,
                projection_radius=args.projection_radius,
                bulk_sobol_power=args.bulk_sobol_power,
                face_sobol_power=args.face_sobol_power,
                replicates=args.replicates,
                radial_power=args.radial_power,
                momentum_refinement_shells=args.momentum_refinement_shells,
                momentum_singularity_subtraction=(
                    not args.disable_momentum_singularity_subtraction
                ),
                seed=args.seed,
            )
            encoded = result.to_json()
            encoded["h_fit_variant"] = fit_variant
            encoded["radius_index"] = radius_index
            result_payloads.append(encoded)

    common_payload = {
        "status": (
            "worldsheet_coefficient_extrapolated_not_frozen"
            if args.block_backend == "h"
            else "worldsheet_recursion_audit_not_frozen"
        ),
        "prescription": "direct physical-domain +i-epsilon boundary value",
        "large_remote_analytic_continuation_used": False,
        "liouville_residue_forest_used": False,
        "subtraction_audit": audit,
        "block_backend_evaluation_counts": dict(
            kernel._block_backend_evaluation_counts
        ),
        "chart_atlas": {
            "oriented_linear_charts": 120,
            "unoriented_trivalent_trees": 15,
            "selection_score": "minimize max(|q1|,|q2|)",
            "production_convergence_condition": "|q1|<1 and |q2|<1",
        },
        "production_block_policy": (
            "coefficient-wise self-dual h-recursion in the best CCY chart"
            if args.block_backend == "h"
            else "non-production recursion-overlap audit"
        ),
        "h_recursion_role": "production coefficient-wise self-dual limit",
        "c_recursion_role": "low-order descendant-validated overlap check",
        "self_dual_regulator_eta": args.h_regulator_eta,
        "self_dual_coefficient_fit": kernel.h_self_dual_fit_diagnostics(),
        "q_variable_convention": (
            "ordinary CCY sphere-linear plumbing coordinates q1=z1/z2 and q2=z2"
        ),
        "global_max_total_twice_level": args.global_max_total_twice_level,
        "structure_precision": args.structure_precision,
        "central_charge_shift": args.central_charge_shift,
        "face_collar_diagnostic_enforced": bool(
            args.enforce_face_collar_certificate
        ),
        "face_collar_diagnostic_interpretation": (
            "non-strict: disagreement includes higher normal powers retained "
            "in the numerical forest remainder"
        ),
        "block_working_precision": args.block_working_precision,
        "worldsheet_normalization": {
            "stored_quantity": "integral d2z d2w I_NS(z,w)",
            "literal_all_tachyon_diagram": (
                "(i/64) g_s^5 C_S2 delta(E) times stored_quantity"
            ),
            "matrix_model_comparison_performed": False,
        },
        "source_sha256": {
            "type0b_ns_five_tachyon.py": _sha256(
                SCRIPT_DIR / "type0b_ns_five_tachyon.py"
            ),
            "type0b_ns_five_tachyon_domain.py": _sha256(
                SCRIPT_DIR / "type0b_ns_five_tachyon_domain.py"
            ),
            "ns_multipoint_c_recursion.py": _sha256(
                C_RECURSION_DIR / "ns_multipoint_c_recursion.py"
            ),
            "ns_multipoint_h_recursion.py": _sha256(
                C_RECURSION_DIR / "ns_multipoint_h_recursion.py"
            ),
            Path(__file__).name: _sha256(Path(__file__)),
        },
        "matrix_model_used": False,
    }
    if len(result_payloads) == 1:
        payload = {**result_payloads[0], **common_payload}
    else:
        payload = {
            "schema": "type0b-ns-fivepoint-coupled-collar-fit-bundle-v1",
            **common_payload,
            "collar_radii": list(collar_radii),
            "fit_variants": list(fit_variants),
            "common_random_numbers": True,
            "results": result_payloads,
        }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(output)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
