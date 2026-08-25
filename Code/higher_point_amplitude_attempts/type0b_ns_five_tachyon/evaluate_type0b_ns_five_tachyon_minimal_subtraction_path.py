#!/usr/bin/env python3
"""Exploratory blind Type-0B five-tachyon finite-part driver."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Sequence

from type0b_ns_five_tachyon import (
    BRYNSFiveTachyonIntegrand,
    integrate_complex_energy_minimal_subtraction_qmc,
)
from type0b_ns_five_tachyon_domain import (
    all_c_atlas_orderings,
    general_complex_energy_convergence_audit,
    is_unavoidable_three_fixed_pco_record,
    minimal_subtraction_ray_certificate,
    minimal_subtraction_ray_frequencies,
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
    certificate = minimal_subtraction_ray_certificate()
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the matrix-blind Type-0B all-NS sphere five-point "
            "integral on the certified one-corner finite-part ray."
        )
    )
    parser.add_argument(
        "--t-values",
        type=float,
        nargs="+",
        default=certificate["ten_sampling_parameters"],
    )
    parser.add_argument("--recursion-max-twice-level", type=int, default=2)
    parser.add_argument("--global-max-twice-levels", type=int, nargs=2, default=(2, 2))
    parser.add_argument("--global-max-total-twice-level", type=int, default=4)
    parser.add_argument("--momentum-orders", type=int, nargs=2, default=(3, 4))
    parser.add_argument("--momentum-maximum", type=float, default=2.0)
    parser.add_argument("--structure-precision", type=int, default=20)
    parser.add_argument("--block-working-precision", type=int, default=35)
    parser.add_argument("--central-charge-shift", type=float, default=0.0)
    parser.add_argument("--collar-radius", type=float, default=0.05)
    parser.add_argument("--projection-radius", type=float, default=1.0e-5)
    parser.add_argument("--bulk-sobol-power", type=int, default=3)
    parser.add_argument("--face-sobol-power", type=int, default=4)
    parser.add_argument("--replicates", type=int, default=4)
    parser.add_argument("--radial-power-cap", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    certificate = minimal_subtraction_ray_certificate()
    if not certificate["minimal_subtraction_interval_certified"]:
        raise RuntimeError("the one-corner ray certificate failed")
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
    cap = float(args.radial_power_cap)
    if not 0.0 < cap <= 2.0:
        raise ValueError("radial-power-cap must lie in (0,2]")

    payload: dict[str, object] = {
        "schema": "type0b-ns-five-tachyon-one-corner-worldsheet-v1",
        "status": "running",
        "blind_freeze": False,
        "matrix_model_used": False,
        "blind_statement": (
            "No matrix-model formula, target value, or fitted coefficient is "
            "imported or evaluated by this driver."
        ),
        "ray_certificate": certificate,
        "finite_part": {
            "target_ordering": [1, 2, 0, 3, 4],
            "wall": 1,
            "scheme": (
                "omit the full wall-one moving-middle term in its q2 collar; "
                "restore its all-c leading normal coefficient by the complex "
                "radial finite part"
            ),
        },
        "settings": {
            "recursion_max_twice_level": recursion_cutoff,
            "global_max_twice_levels": list(args.global_max_twice_levels),
            "global_max_total_twice_level": args.global_max_total_twice_level,
            "momentum_orders": list(args.momentum_orders),
            "momentum_maximum": args.momentum_maximum,
            "structure_precision": args.structure_precision,
            "block_working_precision": args.block_working_precision,
            "central_charge_shift": args.central_charge_shift,
            "collar_radius": args.collar_radius,
            "projection_radius": args.projection_radius,
            "bulk_sobol_power": args.bulk_sobol_power,
            "face_sobol_power": args.face_sobol_power,
            "replicates": args.replicates,
            "radial_power_cap": cap,
            "seed": args.seed,
            "boundary_split": (
                "symmetric 120-chart all-c Voronoi atlas; residue evaluation "
                "normalizes the incoming leg to the left or middle only after "
                "the geometric chart minimizing max(|q1|,|q2|) is selected"
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
        outgoing = minimal_subtraction_ray_frequencies(t_value)
        audit = general_complex_energy_convergence_audit(outgoing)
        negative = [record for record in audit["records"] if record["margin"] <= 0.0]
        if len(negative) != 1 or not is_unavoidable_three_fixed_pco_record(negative[0]):
            raise AssertionError("the requested point left the one-corner chamber")
        pair_margins: dict[tuple[int, int], float] = {}
        for record in audit["records"]:
            if is_unavoidable_three_fixed_pco_record(record):
                continue
            pair = tuple(sorted(int(label) for label in record["pair"]))
            pair_margins[pair] = min(
                pair_margins.get(pair, math.inf), float(record["margin"])
            )
        pair_powers = {
            pair: min(cap, margin) for pair, margin in pair_margins.items()
        }
        if any(power >= 2.0 * pair_margins[pair] for pair, power in pair_powers.items()):
            raise AssertionError("a channel proposal has infinite remainder variance")

        kernel = BRYNSFiveTachyonIntegrand(
            outgoing_energies=outgoing,
            recursion_max_twice_level=recursion_cutoff,
            global_max_twice_levels=args.global_max_twice_levels,
            global_max_total_twice_level=args.global_max_total_twice_level,
            momentum_orders=args.momentum_orders,
            momentum_maximum=args.momentum_maximum,
            structure_precision=args.structure_precision,
            central_charge_shift=args.central_charge_shift,
            block_working_precision=args.block_working_precision,
        )
        atlas = all_c_atlas_orderings(outgoing)
        result = integrate_complex_energy_minimal_subtraction_qmc(
            kernel,
            orderings=atlas,
            collar_radius=args.collar_radius,
            projection_radius=args.projection_radius,
            bulk_sobol_power=args.bulk_sobol_power,
            face_sobol_power=args.face_sobol_power,
            replicates=args.replicates,
            radial_power=min(pair_powers.values()),
            pair_radial_powers=pair_powers,
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
            "remainder_replicates": [_encode(value) for value in result.bulk_estimates],
            "finite_part_face_replicates": [_encode(value) for value in result.face_estimates],
            "literal_all_ns_stripped_amplitude": _encode(literal_amplitude),
            "literal_all_ns_stripped_standard_error": {
                "real": result.standard_error_imag / 64.0,
                "imag": result.standard_error_real / 64.0,
            },
            "extreme_bulk_weights": list(result.extreme_bulk_weights),
        }
        points.append(point)
        payload["points"] = points
        _atomic_write(output, payload)
        print(
            f"t={t_value:.9f} I={result.mean.real:+.8e}"
            f"{result.mean.imag:+.8e}i"
        )

    payload["status"] = "complete"
    payload["blind_freeze"] = False
    _atomic_write(output, payload)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
