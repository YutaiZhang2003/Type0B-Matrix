#!/usr/bin/env python3
"""Independent L5 equal-sign PBW check; does not modify any block kernel."""
import argparse
import hashlib
from pathlib import Path
import time

import sympy as sp

import nsrr_factorized_sign_trial as trial
from nsrr_genus2_block import HumanNSRRThetaOracle, level_triples


def run(output):
    start = time.monotonic()
    b = sp.Rational(7, 5)
    bg = b+1/b
    momenta = (.31, .43, .57)
    p = [sp.Rational(str(x)) for x in momenta]
    runtime = trial.dv.NSRRDoubleVirasoroTheta(b=1.4, physical_momenta=momenta, cutoff=5,
                                             completion="none")
    rows = []
    for f in (0, 1):
        for eta in (1, -1):
            reference = HumanNSRRThetaOracle(
                central_charge=sp.Rational(3, 2)+3*bg**2, h_ns=bg**2/8+p[0]**2/2,
                beta_r1=sp.I*p[1]/sp.sqrt(2), beta_r2=sp.I*p[2]/sp.sqrt(2),
                form_parity=f, primary_parity=0, etas=(eta, eta))
            components = runtime.physical_components(f, eta, eta)
            error = 0.
            for e in level_triples(10):
                expected = reference.coefficient_components(e[0], e[1]//2, e[2]//2)
                for actual, target in zip(components[e], expected):
                    error = max(error, abs(complex(actual)-complex(target))/max(1., abs(complex(target))))
            row = {"form_parity": f, "eta": eta, "coefficient_count": len(components),
                   "maximum_scaled_error": error}
            rows.append(row)
            print(row, flush=True)
    if max(r["maximum_scaled_error"] for r in rows) > 2e-8:
        raise ArithmeticError("L5 double-Virasoro/PBW equality failed")
    result = {"max_level": 5, "momenta_slots": momenta, "rows": rows,
              "branching_ward_residual": runtime.ward_residual_maximum,
              "protected_kernel_sha256": trial.protected_hashes(),
              "audit_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "elapsed_seconds": time.monotonic()-start}
    trial.save(output, result)
    print({"saved": str(output), "elapsed_seconds": result["elapsed_seconds"]}, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
