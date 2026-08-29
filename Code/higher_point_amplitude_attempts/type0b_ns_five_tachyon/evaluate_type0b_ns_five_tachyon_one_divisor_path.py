#!/usr/bin/env python3
"""Exploratory matrix-blind scan on the proposed Type-0B D12 ray."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Sequence

from type0b_ns_five_tachyon import (
    BRYNSFiveTachyonIntegrand,
    integrate_complex_energy_one_divisor_qmc,
)
from type0b_ns_five_tachyon_domain import (
    hybrid_atlas_orderings,
    general_complex_energy_convergence_audit,
    is_one_divisor_subtraction_record,
    one_divisor_ray_certificate,
    one_divisor_ray_frequencies,
)


SCRIPT_DIR = Path(__file__).resolve().parent
C_RECURSION_DIR = SCRIPT_DIR.parents[1] / "c_Recursion"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _encode(value: complex) -> dict[str, float]:
    return {"real": float(value.real), "imag": float(value.imag)}


def _atomic_write(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _parser() -> argparse.ArgumentParser:
    certificate = one_divisor_ray_certificate()
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the matrix-blind all-NS Type-0B sphere five-point "
            "integral on the certified one-divisor hybrid-recursion ray."
        )
    )
    parser.add_argument(
        "--t-values",
        type=float,
        nargs="+",
        default=certificate["ten_sampling_parameters"],
    )
    parser.add_argument(
        "--recursion-max-twice-level",
        type=int,
        default=-1,
        help=(
            "use -1 for the matched finite plumbing-series truncation required "
            "by the hybrid production run"
        ),
    )
    parser.add_argument(
        "--block-backend",
        choices=("hybrid", "h", "c"),
        default="c",
    )
    parser.add_argument("--hybrid-q-threshold", type=float, default=0.3)
    parser.add_argument(
        "--global-max-twice-levels", type=int, nargs=2, default=(4, 4)
    )
    parser.add_argument("--global-max-total-twice-level", type=int, default=8)
    parser.add_argument("--momentum-orders", type=int, nargs=2, default=(3, 4))
    parser.add_argument("--momentum-maximum", type=float, default=2.0)
    parser.add_argument("--structure-precision", type=int, default=20)
    parser.add_argument("--block-working-precision", type=int, default=40)
    parser.add_argument("--central-charge-shift", type=float, default=1.0e-5)
    parser.add_argument("--collar-radius", type=float, default=0.05)
    parser.add_argument("--bulk-sobol-power", type=int, default=2)
    parser.add_argument("--face-sobol-power", type=int, default=2)
    parser.add_argument("--replicates", type=int, default=4)
    parser.add_argument("--normal-correction-order", type=int, default=3)
    parser.add_argument("--normal-correction-angular-order", type=int, default=8)
    parser.add_argument("--radial-power-cap", type=float, default=0.3)
    parser.add_argument("--radial-power-margin-factor", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _pair_remainder_margins(
    records: Sequence[dict[str, object]],
) -> dict[tuple[int, int], float]:
    margins: dict[tuple[int, int], float] = {}
    subtraction_count = 0
    for record in records:
        pair = tuple(sorted(int(label) for label in record["pair"]))
        margin = float(record["margin"])
        if is_one_divisor_subtraction_record(record):
            subtraction_count += 1
            # The first allowed NS descendant raises the nonchiral radial
            # power by one relative to the removed primary.
            margin += 1.0
        margins[pair] = min(margins.get(pair, math.inf), margin)
    if subtraction_count != 1 or set(margins) != {
        tuple(pair)
        for pair in (
            (0, 1), (0, 2), (0, 3), (0, 4), (1, 2),
            (1, 3), (1, 4), (2, 3), (2, 4), (3, 4),
        )
    }:
        raise AssertionError("the audit did not produce the certified ten-pair ledger")
    if min(margins.values()) <= 0.0:
        raise AssertionError("the subtracted remainder is not integrable")
    return margins


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    certificate = one_divisor_ray_certificate()
    if not certificate["one_divisor_interval_certified"]:
        raise RuntimeError("the one-divisor ray certificate failed")
    lower = float(certificate["lower_endpoint"])
    upper = float(certificate["upper_endpoint"])
    t_values = tuple(float(value) for value in args.t_values)
    if len(set(t_values)) != len(t_values):
        raise ValueError("t-values must be distinct")
    if any(not lower <= value <= upper for value in t_values):
        raise ValueError(f"every t must lie in [{lower},{upper}]")
    recursion_cutoff = (
        None
        if args.recursion_max_twice_level < 0
        else args.recursion_max_twice_level
    )
    if args.block_backend == "hybrid" and recursion_cutoff is not None:
        raise ValueError(
            "hybrid production requires --recursion-max-twice-level -1 so "
            "the h- and c-recursive regions use the same finite plumbing-series cutoff"
        )
    cap = float(args.radial_power_cap)
    margin_factor = float(args.radial_power_margin_factor)
    if not 0.0 < cap <= 2.0:
        raise ValueError("radial-power-cap must lie in (0,2]")
    if not 0.0 < margin_factor < 2.0:
        raise ValueError("radial-power-margin-factor must lie in (0,2)")

    payload: dict[str, object] = {
        "schema": "type0b-ns-five-tachyon-one-divisor-worldsheet-v1",
        "status": "running",
        "blind_freeze": False,
        "matrix_model_used": False,
        "blind_statement": (
            "No matrix-model formula, target value, or fitted coefficient is "
            "imported or evaluated by this driver."
        ),
        "ray_certificate": certificate,
        "finite_part": {
            "stable_divisor": "D_12",
            "scheme": (
                "omit the continuum endpoint primary coefficient-by-coefficient "
                "in the selected recursive boundary block; restore its asymptotic coefficient "
                "by the complex radial finite part and restore the integrable "
                "fixture difference by symmetric angular quadrature"
            ),
            "corner_subtractions": 0,
        },
        "settings": {
            "block_backend": args.block_backend,
            "hybrid_q_threshold": args.hybrid_q_threshold,
            "recursion_max_twice_level": recursion_cutoff,
            "global_max_twice_levels": list(args.global_max_twice_levels),
            "global_max_total_twice_level": args.global_max_total_twice_level,
            "momentum_orders": list(args.momentum_orders),
            "momentum_maximum": args.momentum_maximum,
            "structure_precision": args.structure_precision,
            "block_working_precision": args.block_working_precision,
            "central_charge_shift": args.central_charge_shift,
            "collar_radius": args.collar_radius,
            "bulk_sobol_power": args.bulk_sobol_power,
            "face_sobol_power": args.face_sobol_power,
            "replicates": args.replicates,
            "normal_correction_order": args.normal_correction_order,
            "normal_correction_angular_order": (
                args.normal_correction_angular_order
            ),
            "radial_power_cap": cap,
            "radial_power_margin_factor": margin_factor,
            "seed": args.seed,
            "boundary_split": (
                "symmetric 120-chart Voronoi atlas; after minimizing "
                "max(|q1|,|q2|), use c-recursion in the selected chart; ten stable divisors "
                "and fifteen compatible corners are the analytic ledger; the "
                "incoming-leg residue orientation is normalized only after the "
                "geometric chart is selected"
            ),
        },
        "source_sha256": {
            name: _sha256(
                (
                    SCRIPT_DIR
                    if name.startswith("type0b_") or name == Path(__file__).name
                    else C_RECURSION_DIR
                )
                / name
            )
            for name in (
                "type0b_ns_five_tachyon.py",
                "type0b_ns_five_tachyon_domain.py",
                "ns_multipoint_c_recursion.py",
                "ns_multipoint_h_recursion.py",
                "super_liouville_structure_constants.py",
                Path(__file__).name,
            )
        },
        "points": [],
    }
    output = args.output.resolve()
    _atomic_write(output, payload)

    points: list[dict[str, object]] = []
    for index, t_value in enumerate(t_values):
        outgoing = one_divisor_ray_frequencies(t_value)
        audit = general_complex_energy_convergence_audit(outgoing)
        negative = [
            record for record in audit["records"] if float(record["margin"]) <= 0.0
        ]
        if len(negative) != 1 or not is_one_divisor_subtraction_record(negative[0]):
            raise AssertionError("the requested point left the one-divisor chamber")
        pair_margins = _pair_remainder_margins(audit["records"])
        pair_powers = {
            pair: min(cap, margin_factor * margin)
            for pair, margin in pair_margins.items()
        }
        if any(
            power >= 2.0 * pair_margins[pair]
            for pair, power in pair_powers.items()
        ):
            raise AssertionError("a channel proposal has infinite remainder variance")

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
            block_working_precision=args.block_working_precision,
        )
        result = integrate_complex_energy_one_divisor_qmc(
            kernel,
            orderings=hybrid_atlas_orderings(outgoing),
            collar_radius=args.collar_radius,
            bulk_sobol_power=args.bulk_sobol_power,
            face_sobol_power=args.face_sobol_power,
            replicates=args.replicates,
            radial_power=min(pair_powers.values()),
            pair_radial_powers=pair_powers,
            normal_correction_order=args.normal_correction_order,
            normal_correction_angular_order=(
                args.normal_correction_angular_order
            ),
            seed=args.seed + 1000 * index,
        )
        literal_amplitude = 1.0j * result.mean / 64.0
        point = {
            "t": t_value,
            "outgoing_omegas": [_encode(value) for value in outgoing],
            "incoming_omega": _encode(sum(outgoing)),
            "audit_minimum_raw_margin": audit["minimum_integrability_margin"],
            "audit_minimum_remainder_margin": min(pair_margins.values()),
            "pair_remainder_margins": {
                f"{pair[0]},{pair[1]}": value
                for pair, value in sorted(pair_margins.items())
            },
            "pair_radial_powers": {
                f"{pair[0]},{pair[1]}": value
                for pair, value in sorted(pair_powers.items())
            },
            "worldsheet_integral": _encode(result.mean),
            "worldsheet_standard_error": {
                "real": result.standard_error_real,
                "imag": result.standard_error_imag,
            },
            "replicate_integrals": [_encode(value) for value in result.estimates],
            "bulk_replicates": [_encode(value) for value in result.bulk_estimates],
            "finite_part_face_replicates": [
                _encode(value) for value in result.face_estimates
            ],
            "literal_all_ns_stripped_amplitude": _encode(literal_amplitude),
            "literal_all_ns_stripped_standard_error": {
                "real": result.standard_error_imag / 64.0,
                "imag": result.standard_error_real / 64.0,
            },
            "extreme_bulk_weights": list(result.extreme_bulk_weights),
            "block_backend_evaluation_counts": dict(
                kernel._block_backend_evaluation_counts
            ),
        }
        points.append(point)
        payload["points"] = points
        _atomic_write(output, payload)
        print(
            f"t={t_value:.9f} I={result.mean.real:+.8e}"
            f"{result.mean.imag:+.8e}i",
            flush=True,
        )

    payload["status"] = "complete_unfrozen"
    _atomic_write(output, payload)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
