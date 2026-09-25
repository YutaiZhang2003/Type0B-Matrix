#!/usr/bin/env python3
"""Focused checks of b-dual extrapolation, analytic continuation and CCY blocks."""
import argparse
import hashlib
from itertools import product
import json
from pathlib import Path

import numpy as np

from literature_component_blocks import ComponentBlocks
from literature_double_virasoro import LiteratureDoubleVirasoroBlocks
from literature_self_dual_blocks import ExtrapolatedLiteratureBlocks, SelfDualLiteratureBlocks


REAL_P = (.21, .39, .31, .43)
COMPLEX_P = (.02+.22j, .03+.23j, .04+.24j, .09+.69j)


def direct_reference(family, order, b=1):
    """Use literal real-slice Ward/Gram identities at the specified real c.

    Only the SCA representation's c,h are changed before its first action;
    all three-point identities and the native Hermitian frame are unchanged.
    This helper deliberately does not analytically continue the bra convention.
    """
    block = ComponentBlocks(family, REAL_P, .71, order)
    q = b+1/b
    for m in (*block.modules, block.internal, block.left.m3, block.left.m2, block.left.m1,
              block.right.m3, block.right.m2, block.right.m1):
        m.c = 1.5+3*q*q
        m.h = q*q/8+m.p*m.p/2+(1/16 if m.sector == "R" else 0)
    return block


def run():
    rows = []
    for family in ("mixed_ns", "mixed_r", "rrrr"):
        # Generic real-b continuation is checked before taking its b=1 limit.
        a = direct_reference(family, 4, .83)
        b = LiteratureDoubleVirasoroBlocks(family, REAL_P, .71, 4, b=.83)
        error = 0.
        for e, k, sl, sr in product(((0,0,0,0), (1,0,1,1)), (0,1), (1,-1), (1,-1)):
            x, y = a.coefficients(e,k,sl,sr), b.coefficients(e,k,sl,sr)
            error = max(error, float(np.max(abs(x-y))/max(1., np.max(abs(x)))))
        rows.append(dict(check="generic b=.83 vs independent Ward/Gram", family=family,
                         maximum_twice_level=4, maximum_scaled_error=error, tolerance=2e-11))

        a = direct_reference(family, 10)
        b = ExtrapolatedLiteratureBlocks(family, REAL_P, .71, 10)
        error = 0.
        for k, sl, sr in product((0,1), (1,-1), (1,-1)):
            x, y = a.coefficients((0,0,0,0),k,sl,sr), b.coefficients((0,0,0,0),k,sl,sr)
            error = max(error, float(np.max(abs(x-y))/max(1., np.max(abs(x)))))
        rows.append(dict(check="Type0B extrapolation vs direct b=1 Ward/Gram", family=family,
                         maximum_twice_level=10, maximum_scaled_error=error, tolerance=1e-6))

        a = ExtrapolatedLiteratureBlocks(family, COMPLEX_P, .71, 4)
        b = SelfDualLiteratureBlocks(family, COMPLEX_P, .71, 4)
        error = 0.
        duality = 0.
        for e,k in (((0,0,0,0),0), ((1,0,1,1),1)):
            x, y = a._native_components(e,k), b._native_components(e,k)
            error = max(error, float(np.max(abs(x-y))/max(1., np.max(abs(y)))))
            generic = LiteratureDoubleVirasoroBlocks(family, COMPLEX_P, .71, 4, b=1.01)
            reciprocal = LiteratureDoubleVirasoroBlocks(family, COMPLEX_P, .71, 4, b=1/1.01)
            x, y = generic._native_components(e,k), reciprocal._native_components(e,k)
            duality = max(duality, float(np.max(abs(x-y))/max(1., np.max(abs(y)))))
        rows.append(dict(check="complex momenta: extrapolation vs two-contour b=1 limit", family=family,
                         maximum_twice_level=4, maximum_scaled_error=error, tolerance=1e-6))
        rows.append(dict(check="b <-> 1/b assembled-table symmetry at complex momenta", family=family,
                         maximum_twice_level=4, maximum_scaled_error=duality, tolerance=1e-10))
    root = Path(__file__).parent
    names = ("literature_double_virasoro.py", "literature_self_dual_blocks.py",
             "literature_component_blocks.py", "virasoro_sphere_c_recursion.py",
             "audit_literature_self_dual.py")
    return dict(status="passed" if all(r["maximum_scaled_error"] <= r["tolerance"] for r in rows) else "failed",
                rows=rows, epsilon=.01, finite_b_samples=[1.01,1.02],
                method="Type0B extrapolation applied after full double-Virasoro assembly",
                production_superconformal_gram_fallback=False,
                scope="finite coefficients; neither momentum nor moduli integral",
                sources={n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in names})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))
    if result["status"] != "passed":
        raise SystemExit(1)
