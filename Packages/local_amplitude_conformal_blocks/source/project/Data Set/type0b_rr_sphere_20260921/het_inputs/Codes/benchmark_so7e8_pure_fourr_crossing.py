"""Independent pure-SLD R+^4 crossing control, without heterotic sewing.

Uses Suchanek, arXiv:1012.2974v2, section 4.1: sum the even-even and
odd-odd blocks over both HJS signs.  Complex external momenta are continued
analytically on both sides; they are not complex conjugated.
"""
import argparse
from itertools import product
import json
import math
from pathlib import Path
import sys
import time
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Codes"))
from benchmark_so7e8_four_ramond_convergence import MOMENTA
from so7e8_four_ramond_elliptic_recursion import general_four_ramond_sld_elliptic_h_series
from spin23_singlet_amplitudes import _momentum_quadrature
from spin23_super_liouville_data import rr_ns_chiral_structure_constant


def pure_fourr(momenta, z, order, nodes):
    ps, pw = _momentum_quadrature(nodes, 4, .03, scheme="infinite_gauss")
    result = np.zeros(len(z), complex)
    for p, weight in zip(ps, pw):
        for sl, sr in product((-1, 1), repeat=2):
            structure = .25*rr_ns_chiral_structure_constant(
                momenta[3], momenta[2], float(p), structure_sign=sl, precision=80
            )*rr_ns_chiral_structure_constant(
                momenta[1], momenta[0], -float(p), structure_sign=sr, precision=80
            )
            for parity in ("even", "odd"):
                values = []
                for chirality in ("holomorphic", "antiholomorphic"):
                    block = general_four_ramond_sld_elliptic_h_series(
                        float(p), external_momenta=momenta,
                        external_ground_parities=(0,0,0,0),
                        maximum_twice_level=order, component=parity,
                        left_structure_sign=sl, right_structure_sign=sr,
                        chirality=chirality, digits=max(80,4*order+40),
                    )
                    hol = chirality == "holomorphic"
                    values.append(block.value(z if hol else z.conjugate(),
                        cut_side="upper" if hol else "lower"))
                result += weight*structure*values[0]*values[1]/math.pi
    return result


def run(order, nodes):
    start = time.perf_counter()
    z = np.array([.37+.23j, .68-.15j, 2+1e-7j, 2-1e-7j])
    m = MOMENTA
    direct = pure_fourr(m, z, order, nodes)
    crossed = pure_fourr((m[2],m[1],m[0],m[3]), 1-z, order, nodes)
    pairs = lambda a: np.stack([np.asarray(a).real,np.asarray(a).imag], axis=-1).tolist()
    return dict(order=order, p_nodes=nodes, momenta=pairs(m), z=pairs(z),
        direct=pairs(direct), crossed=pairs(crossed),
        relative_crossing_defect=(abs(direct-crossed)/np.maximum(abs(direct),abs(crossed))).tolist(),
        relative_cut_jump=float(abs(direct[2]-direct[3])/abs(direct[2])),
        seconds=time.perf_counter()-start,
        scope="Pure super-Liouville R+^4; does not certify heterotic component sewing.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--order", type=int, default=11)
    parser.add_argument("--nodes", type=int, default=48)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.order,args.nodes)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))
