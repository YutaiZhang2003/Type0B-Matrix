"""Finite-level checks of the open Ramond-edge sewing construction."""
from __future__ import annotations

import argparse
from itertools import product
import json
from pathlib import Path
import time

import numpy as np
import sympy as sp

from mixed_blocks import MixedNSRamondPlumbingBlock, HumanNSRRThetaOracle, RamondState


def pair(value):
    return [float(np.real(value)), float(np.imag(value))]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-twice-level", type=int, default=4)
    args = parser.parse_args()
    started = time.monotonic()
    block = MixedNSRamondPlumbingBlock(p_ns=sp.Rational(3, 5),
        p_r=sp.Rational(7, 10), omega=sp.I/4)
    residuals, samples = [], []
    for f, eta1, eta2 in product((0, 1), (1, -1), (1, -1)):
        oracle = HumanNSRRThetaOracle(central_charge=block.c, h_ns=block.h_ns,
            beta_r1=block.beta_r, beta_r2=block.beta_ext,
            form_parity=f, primary_parity=0, etas=(eta1, eta2))
        # Closing ground and level-one puncture states probes positive R
        # modes, the external descendant, and the external Gram contraction.
        for n in range(args.max_twice_level+1):
            for r in range((args.max_twice_level-n)//2+1):
                for e in range(min(1, (args.max_twice_level-n-2*r)//2)+1):
                    expected = oracle.coefficient_components(n, r, e)
                    for rp, ep in product((0, 1), repeat=2):
                        found = block.closed_theta_component(n, r, e, rp, ep,
                            form_parity=f, etas=(eta1, eta2))
                        component = n % 2 + 2*rp + 4*ep
                        error = abs(found-expected[component])
                        residuals.append(error)
                        if error > 2e-11*max(1, abs(expected[component])):
                            raise ArithmeticError((n, r, e, f, eta1, eta2, component, error))
        for word in ((), (("G", -1),)):
            state1 = RamondState(word, 0)
            state2 = RamondState((), 0)
            # A single G insertion changes the homogeneous three-form
            # routing; equal forms with equal external grounds would vanish.
            forms = (f, 1-f) if word else (f, f)
            series = block.series(args.max_twice_level, state1, state2,
                                  forms=forms, etas=(eta1, eta2))
            if word and not any(abs(v)>1e-12 for v in series.values()):
                raise ArithmeticError("The routed G_-1 sample unexpectedly vanished")
            samples.append(dict(forms=list(forms), etas=[eta1, eta2], external_word=list(word),
                external_grounds=[0, 0], coefficients=[dict(ns_twice_level=n,
                    r_level=r, value=pair(v)) for (n, r), v in series.items()]))
    # Virasoro global coefficient on the NS edge is fixed without R Ward
    # recursion; the chosen ++ ground normalization is one.
    plus = RamondState((), 0)
    f00 = block.coefficient(0, 0, plus, plus)
    f10 = block.coefficient(2, 0, plus, plus)
    expected = complex((block.h_ns+block.h_r-block.h_ext)**2/(2*block.h_ns))
    if abs(f00-1) > 1e-14 or abs(f10-expected) > 1e-13:
        raise ArithmeticError("Global NS L_-1 normalization failed")
    # Complex continuation must be bilinear, never absolute-square sewing.
    complex_block = MixedNSRamondPlumbingBlock(p_ns=sp.Rational(3, 5),
        p_r=sp.Rational(7, 10), omega=sp.Rational(1, 5)+sp.I/4)
    complex_coefficient = complex_block.coefficient(2, 0, plus, plus)
    complex_expected = complex((complex_block.h_ns+complex_block.h_r-complex_block.h_ext)**2/
                              (2*complex_block.h_ns))
    assert abs(complex_coefficient-complex_expected) < 1e-13
    assert abs(complex_coefficient.imag) > 1e-5
    result = dict(schema="type0b-rr-open-sewing-check-v1", scope="marked chiral plumbing blocks only",
        string_integral_computed=False, max_total_descendant_level=args.max_twice_level/2,
        check_count=len(residuals), maximum_theta_closure_absolute_residual=max(residuals),
        ns_global_level_one_residual=abs(f10-expected),
        complex_continuation_residual=abs(complex_coefficient-complex_expected),
        complex_continuation_coefficient=pair(complex_coefficient),
        external_energy=[0, .25], internal_momenta=[.6, .7],
        samples=samples, seconds=time.monotonic()-started)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k!="samples"}, indent=2))


if __name__ == "__main__":
    main()
