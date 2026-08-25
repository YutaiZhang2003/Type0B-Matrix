#!/usr/bin/env python3
"""Exploratory obsolete scanner for the Type-0B NS five-tachyon path.

The corrected superghost boundary count proves that no subtraction-free
domain exists with PCOs on all three picture-zero punctures.  The driver now
fails closed until the unavoidable all-c corner finite part is implemented.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Sequence

from type0b_ns_five_tachyon import (
    BRYNSFiveTachyonIntegrand,
    integrate_complex_energy_continued_atlas_qmc,
)
from type0b_ns_five_tachyon_domain import (
    CERTIFIED_RAY_COEFFICIENTS,
    certified_ray_atlas_orderings,
    certified_ray_frequencies,
    certified_ray_interval,
    general_complex_energy_convergence_audit,
    three_fixed_pco_subtraction_free_no_go,
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


def _encode_component_diagnostic(value) -> dict[str, object]:
    real = float(value.real)
    imaginary = float(value.imag)
    if math.isfinite(real) and math.isfinite(imaginary):
        return {"real": real, "imag": imaginary}
    magnitude = abs(value)
    return {
        "representation": "log-polar",
        "log_absolute_value": float(math.log(float(magnitude)))
        if math.isfinite(float(magnitude))
        else float(__import__("mpmath").log(magnitude)),
        "phase": float(__import__("mpmath").arg(value)),
    }


def _atomic_write(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the BRY-convention all-NS sphere 1->4 worldsheet "
            "integral on the subtraction-free complex path.  This program "
            "does not import or evaluate a matrix-model answer."
        )
    )
    parser.add_argument(
        "--t-values",
        type=float,
        nargs="+",
        default=(
            0.65020,
            0.65034,
            0.65048,
            0.65062,
            0.65076,
            0.65090,
            0.65104,
            0.65118,
            0.65132,
            0.65146,
        ),
    )
    parser.add_argument("--recursion-max-twice-level", type=int, default=0)
    parser.add_argument("--global-max-twice-levels", type=int, nargs=2, default=(2, 2))
    parser.add_argument("--global-max-total-twice-level", type=int, default=4)
    parser.add_argument("--momentum-orders", type=int, nargs=2, default=(2, 3))
    parser.add_argument("--momentum-maximum", type=float, default=2.0)
    parser.add_argument("--structure-precision", type=int, default=20)
    parser.add_argument("--block-working-precision", type=int, default=35)
    parser.add_argument("--central-charge-shift", type=float, default=0.0)
    parser.add_argument("--sobol-power", type=int, default=4)
    parser.add_argument("--replicates", type=int, default=4)
    parser.add_argument(
        "--radial-power",
        type=float,
        default=0.0,
        help="positive fixed power; zero uses the minimum boundary margin at each t",
    )
    parser.add_argument(
        "--minimum-radial-power",
        type=float,
        default=0.012,
        help=(
            "floating-point floor for channel-local powers when --radial-power=0; "
            "it must stay below twice the smallest selected boundary margin"
        ),
    )
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    recursion_cutoff = (
        None
        if args.recursion_max_twice_level < 0
        else args.recursion_max_twice_level
    )
    t_values = tuple(float(value) for value in args.t_values)
    if len(set(t_values)) != len(t_values):
        raise ValueError("t-values must be distinct")
    no_go = three_fixed_pco_subtraction_free_no_go(
        certified_ray_frequencies(t_values[0])
    )
    raise RuntimeError(
        "the subtraction-free scan is disabled: the two (-1)-picture "
        "vertices contribute an extra |q|^-2 superghost singularity, and "
        "the incoming-middle first-wall residue gives an unavoidable "
        "threshold-one corner divergence. Implement the all-c corner "
        f"finite part before resuming production. Certificate: {no_go}"
    )

    # Unreachable historical scanner retained below as the starting point
    # for the finite-part implementation.
    ray_certificate = certified_ray_interval()
    lower = float(ray_certificate["lower_endpoint"])
    upper = float(ray_certificate["upper_endpoint"])
    if any(not lower < value < upper for value in t_values):
        raise ValueError(
            "every t-value must lie strictly inside the certified ray interval "
            f"({lower}, {upper})"
        )

    payload: dict[str, object] = {
        "schema": "type0b-ns-sphere-five-tachyon-complex-path-worldsheet-v1",
        "status": "running",
        "blind_freeze": False,
        "matrix_model_used": False,
        "blind_statement": (
            "No matrix-model formula, target value, or fitted coefficient is "
            "imported or evaluated by this driver."
        ),
        "kinematics": {
            "outgoing": "omega_a(t)=c_a*t with four separated complex c_a",
            "ray_coefficients": [
                _encode(value) for value in CERTIFIED_RAY_COEFFICIENTS
            ],
            "incoming": "omega_in(t)=sum_a omega_a(t)",
            "path": "certified separated-frequency ray",
            "t_domain": [lower, upper],
            "ray_certificate": ray_certificate,
            "picture_assignment": (
                "labels 0,1,2 fixed at infinity,1,0 in picture zero; "
                "labels 3,4 integrated at z,w in picture -1"
            ),
        },
        "stored_worldsheet_quantity": "integral over M_0,5 of I_NS",
        "literal_all_ns_amplitude": (
            "A_T5/[g_s^5 C_S2 delta(E)] = i*integral(I_NS)/64"
        ),
        "right_mode_inference_not_used_during_worldsheet_stage": (
            "A_R5=16*A_T5, if the sixteen even-axion diagrams are equal"
        ),
        "settings": {
            "recursion_max_twice_level": recursion_cutoff,
            "global_max_twice_levels": list(args.global_max_twice_levels),
            "global_max_total_twice_level": args.global_max_total_twice_level,
            "momentum_orders": list(args.momentum_orders),
            "momentum_maximum": args.momentum_maximum,
            "structure_precision": args.structure_precision,
            "block_working_precision": args.block_working_precision,
            "central_charge_shift": args.central_charge_shift,
            "sobol_power": args.sobol_power,
            "replicates": args.replicates,
            "atlas_ordering_stratification": True,
            "boundary_split": (
                "all-c-recursion Voronoi cells: evaluate each proposal sample "
                "in the certified channel minimizing max(|q1|,|q2|)"
            ),
            "requested_radial_power": args.radial_power,
            "minimum_radial_power": args.minimum_radial_power,
            "seed": args.seed,
        },
        "source_sha256": {
            "type0b_ns_five_tachyon.py": _sha256(
                SCRIPT_DIR / "type0b_ns_five_tachyon.py"
            ),
            "ns_multipoint_c_recursion.py": _sha256(
                C_RECURSION_DIR / "ns_multipoint_c_recursion.py"
            ),
            "super_liouville_structure_constants.py": _sha256(
                C_RECURSION_DIR / "super_liouville_structure_constants.py"
            ),
            "evaluate_type0b_ns_five_tachyon_path.py": _sha256(Path(__file__)),
            "type0b_ns_five_tachyon_domain.py": _sha256(
                SCRIPT_DIR / "type0b_ns_five_tachyon_domain.py"
            ),
        },
        "points": [],
    }
    output = args.output.resolve()
    _atomic_write(output, payload)

    points: list[dict[str, object]] = []
    for index, t_value in enumerate(t_values):
        outgoing = certified_ray_frequencies(t_value)
        incoming = sum(outgoing)
        audit = general_complex_energy_convergence_audit(outgoing)
        if not audit["strictly_subtraction_free"]:
            raise AssertionError(f"path point failed its boundary audit: {audit}")
        minimum_margin = float(audit["minimum_integrability_margin"])
        pair_margins: dict[tuple[int, int], float] = {}
        for record in audit["records"]:
            pair = tuple(sorted(int(label) for label in record["pair"]))
            pair_margins[pair] = min(
                pair_margins.get(pair, math.inf), float(record["margin"])
            )
        if args.radial_power > 0.0:
            radial_power = float(args.radial_power)
            pair_radial_powers = {
                pair: radial_power for pair in pair_margins
            }
        else:
            floor = float(args.minimum_radial_power)
            if not 0.0 < floor < 2.0 * minimum_margin:
                raise ValueError(
                    "minimum-radial-power must be positive and below twice "
                    f"the smallest margin {minimum_margin}"
                )
            pair_radial_powers = {
                pair: min(0.2, max(floor, margin))
                for pair, margin in pair_margins.items()
            }
            radial_power = min(pair_radial_powers.values())
        if not 0.0 < radial_power <= 2.0:
            raise ValueError("the selected radial power is outside (0,2]")
        pair_variance_margins = {
            pair: 2.0 * pair_margins[pair] - power
            for pair, power in pair_radial_powers.items()
        }
        if min(pair_variance_margins.values()) <= 0.0:
            raise ValueError(
                "the selected proposal has infinite boundary variance: "
                "every pair power must be strictly below twice its audited "
                "integrability margin"
            )

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
        orderings = certified_ray_atlas_orderings(t_value)
        result = integrate_complex_energy_continued_atlas_qmc(
            kernel,
            orderings=orderings,
            sobol_power=args.sobol_power,
            replicates=args.replicates,
            radial_power=radial_power,
            pair_radial_powers=pair_radial_powers,
            stratify_orderings=True,
            adaptive_c_channel=True,
            seed=args.seed + 1000 * index,
        )
        stripped_amplitude = 1.0j * result.mean / 64.0
        inferred_right_mode_amplitude = 16.0 * stripped_amplitude
        point = {
            "t": t_value,
            "outgoing_omegas": [_encode(value) for value in outgoing],
            "incoming_omega": _encode(incoming),
            "radial_power": radial_power,
            "pair_integrability_margins": {
                f"{pair[0]},{pair[1]}": margin
                for pair, margin in sorted(pair_margins.items())
            },
            "pair_radial_powers": {
                f"{pair[0]},{pair[1]}": power
                for pair, power in sorted(pair_radial_powers.items())
            },
            "pair_variance_margins": {
                f"{pair[0]},{pair[1]}": margin
                for pair, margin in sorted(pair_variance_margins.items())
            },
            "atlas_ordering_count": len(orderings),
            "convergence_audit": audit,
            "worldsheet_integral": _encode(result.mean),
            "worldsheet_integral_standard_error": {
                "real": result.standard_error_real,
                "imag": result.standard_error_imag,
            },
            "literal_all_ns_stripped_amplitude": _encode(stripped_amplitude),
            "literal_all_ns_stripped_standard_error": {
                "real": result.standard_error_imag / 64.0,
                "imag": result.standard_error_real / 64.0,
            },
            "right_mode_amplitude_if_16_even_axion_diagrams_equal": _encode(
                inferred_right_mode_amplitude
            ),
            "right_mode_standard_error_if_16_even_axion_diagrams_equal": {
                "real": result.standard_error_imag / 4.0,
                "imag": result.standard_error_real / 4.0,
            },
            "replicate_integrals": [_encode(value) for value in result.estimates],
            "component_replicates": {
                "continuous": [
                    _encode_component_diagnostic(value)
                    for value in result.continuous_estimates
                ],
                "left_residues": [
                    _encode_component_diagnostic(value)
                    for value in result.left_residue_estimates
                ],
                "right_residues": [
                    _encode_component_diagnostic(value)
                    for value in result.right_residue_estimates
                ],
                "nested_residues": [
                    _encode_component_diagnostic(value)
                    for value in result.nested_residue_estimates
                ],
            },
            "extreme_sample_diagnostics": list(
                result.extreme_sample_diagnostics
            ),
        }
        checked_values = (
            *(
                (f"replicate_{replicate}", value)
                for replicate, value in enumerate(result.estimates)
            ),
            ("mean", result.mean),
            ("stripped_amplitude", stripped_amplitude),
        )
        for name, value in checked_values:
            if not math.isfinite(value.real) or not math.isfinite(value.imag):
                raise ArithmeticError(
                    "the worldsheet scan produced a non-finite value: "
                    f"{name}={value!r}"
                )
        points.append(point)
        payload["points"] = points
        _atomic_write(output, payload)
        print(
            f"t={t_value:.8f} "
            f"omega_in={incoming.real:+.8f}{incoming.imag:+.8f}i "
            f"integral={result.mean.real:+.8e}{result.mean.imag:+.8e}i"
        )

    payload["status"] = "complete"
    _atomic_write(output, payload)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
