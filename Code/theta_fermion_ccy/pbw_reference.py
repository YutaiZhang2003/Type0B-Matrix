"""Independent PBW side of the user's requested level-five comparison.

This module is never imported by the production CCY pipeline. It reuses the
existing high-precision physical SCA Gram/Ward implementation, without any
enlarged or auxiliary block calculation.
"""

from __future__ import annotations

import argparse
from fractions import Fraction
import json
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "ramond_zero_mode_recovery"))


def encode(value):
    return {"real": value.x.real.mid().str(105, radius=False),
            "imag": value.x.imag.mid().str(105, radius=False)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--level", type=int, default=5)
    parser.add_argument("--f", type=int, choices=(0, 1), default=0)
    parser.add_argument("--p", type=int, choices=(0, 1), default=0)
    parser.add_argument("--eta", type=int, choices=(-1, 1), default=1)
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args()
    if not 0 <= args.level <= 5:
        parser.error("The requested PBW comparison is limited to total level 5")

    from check_level10_complex import load_runner
    runner = load_runner()
    b = Fraction(7, 5)
    momenta = tuple(map(Fraction, ("11/23", "13/29", "17/31")))
    started = time.perf_counter()
    oracle = runner.Check(b, momenta)
    report = {
        "status": "running", "algorithm": "independent physical SCA PBW",
        "total_q_level": args.level, "b": str(b),
        "momenta": [str(x) for x in momenta], "p": args.p, "f": args.f,
        "etas": [args.eta, -args.eta], "precision_bits": 384,
        "coefficients": [],
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    for levels in runner.level_triples(2*args.level):
        coefficient = oracle.physical_coefficient(
            levels, args.p, args.f, (args.eta, -args.eta))
        report["coefficients"].append({
            "twice_levels": levels, "values": [encode(x) for x in coefficient]})
        report["elapsed_seconds"] = time.perf_counter() - started
        args.json.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"pbw_levels": levels,
                          "elapsed_seconds": report["elapsed_seconds"]}), flush=True)
    report["status"] = "reference computed; comparison pending"
    args.json.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
