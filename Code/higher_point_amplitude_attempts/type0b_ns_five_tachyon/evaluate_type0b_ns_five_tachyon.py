#!/usr/bin/env python3
"""Exploratory target-blind equal-energy Type-0B NS five-point driver."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Sequence

from type0b_ns_five_tachyon import (
    BRYNSFiveTachyonIntegrand,
    imaginary_energy_chamber_audit,
    integrate_imaginary_energy_atlas_qmc,
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
            "Evaluate and freeze the BRY-convention all-NS sphere 1->4 "
            "worldsheet integral without loading a matrix-model target."
        )
    )
    parser.add_argument("--t", type=float, default=0.10)
    parser.add_argument(
        "--block-backend",
        choices=("hybrid", "h", "c"),
        default="c",
        help=(
            "production uses c-recursion in the best chart; h and hybrid "
            "are recursion-overlap audits"
        ),
    )
    parser.add_argument("--hybrid-q-threshold", type=float, default=0.3)
    parser.add_argument("--recursion-max-twice-level", type=int, default=-1)
    parser.add_argument("--global-max-twice-levels", type=int, nargs=2, default=(6, 6))
    parser.add_argument("--global-max-total-twice-level", type=int, default=8)
    parser.add_argument("--momentum-orders", type=int, nargs=2, default=(3, 4))
    parser.add_argument("--momentum-maximum", type=float, default=4.0)
    parser.add_argument("--structure-precision", type=int, default=20)
    parser.add_argument("--block-working-precision", type=int, default=45)
    parser.add_argument("--sobol-power", type=int, default=4)
    parser.add_argument("--replicates", type=int, default=4)
    parser.add_argument("--radial-power", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.t <= 0.0:
        raise ValueError("t must be positive")
    audit = imaginary_energy_chamber_audit((1.0j * args.t,) * 4)
    if not audit["simultaneously_subtraction_and_residue_free"]:
        raise ValueError(
            "no raw blind freeze is possible in this collocated-PCO gauge: "
            f"chamber audit={audit}"
        )
    recursion_cutoff = (
        None
        if args.recursion_max_twice_level < 0
        else args.recursion_max_twice_level
    )
    kernel = BRYNSFiveTachyonIntegrand(
        outgoing_energies=(1.0j * args.t,) * 4,
        block_backend=args.block_backend,
        hybrid_q_threshold=args.hybrid_q_threshold,
        recursion_max_twice_level=recursion_cutoff,
        global_max_twice_levels=args.global_max_twice_levels,
        global_max_total_twice_level=args.global_max_total_twice_level,
        momentum_orders=args.momentum_orders,
        momentum_maximum=args.momentum_maximum,
        structure_precision=args.structure_precision,
        block_working_precision=args.block_working_precision,
    )
    result = integrate_imaginary_energy_atlas_qmc(
        kernel,
        sobol_power=args.sobol_power,
        replicates=args.replicates,
        radial_power=args.radial_power,
        seed=args.seed,
    )
    payload = result.to_json()
    payload.update(
        {
            "blind_freeze": True,
            "blind_freeze_date": "2026-08-25",
            "kinematics": "omega_1=...=omega_4=i*t, omega_in=4*i*t",
            "equal_imaginary_t": args.t,
            "chamber_audit": audit,
            "block_evaluation": args.block_backend,
            "global_max_total_twice_level": args.global_max_total_twice_level,
            "structure_precision": args.structure_precision,
            "block_working_precision": args.block_working_precision,
            "worldsheet_normalization": {
                "stored_quantity": "integral d2z d2w I_NS(z,w)",
                "literal_all_tachyon_diagram": (
                    "(i/64) g_s^5 C_S2 delta(E) times stored_quantity"
                ),
                "full_R_mode_if_16_even_axion_diagrams_are_equal": (
                    "(i/4) g_s^5 C_S2 delta(E) times stored_quantity"
                ),
                "diagram_equality_used_in_computation": False,
            },
            "source_sha256": {
                "type0b_ns_five_tachyon.py": _sha256(
                    SCRIPT_DIR / "type0b_ns_five_tachyon.py"
                ),
                "ns_multipoint_c_recursion.py": _sha256(
                    C_RECURSION_DIR / "ns_multipoint_c_recursion.py"
                ),
                "ns_multipoint_h_recursion.py": _sha256(
                    C_RECURSION_DIR / "ns_multipoint_h_recursion.py"
                ),
                "evaluate_type0b_ns_five_tachyon.py": _sha256(Path(__file__)),
            },
        }
    )
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(output)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
