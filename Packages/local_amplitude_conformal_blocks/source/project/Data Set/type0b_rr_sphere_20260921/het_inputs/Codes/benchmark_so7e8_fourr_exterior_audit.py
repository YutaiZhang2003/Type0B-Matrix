#!/usr/bin/env python3
"""P-integrated crossing audit, keeping exterior vertex labels unchanged."""
import argparse
import json
from pathlib import Path
import sys
import time
from dataclasses import replace
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Codes"))
from benchmark_so7e8_four_ramond_convergence import MOMENTA, FAMILIES
from so7e8_direct_domain import spin7_global_grid
from so7e8_four_ramond_assembly import (build_fixed_p_four_ramond_kernel,
                                       spin7_reexpress_pairing_coefficients)
from so7e8_four_ramond_liouville_integral import _evaluate_uniform_kernel
from spin23_singlet_amplitudes import _momentum_quadrature


def run(order, nodes):
    started = time.perf_counter()
    u = np.array([.37+.23j, .68-.15j])
    z = 1/u
    spin_u = spin7_global_grid(tuple(u.conjugate()))
    spin_z = spin7_global_grid(tuple(z.conjugate()))
    direct = np.zeros((len(u), 4), complex)
    folded = np.zeros_like(direct)
    separated = {side: {c: np.zeros_like(direct) for c in ("identity","fermion")}
                 for side in ("direct","folded")}
    cut_points=np.array([-.5+1e-7j,-.5-1e-7j,2+1e-7j,2-1e-7j])
    spin_cuts=spin7_global_grid(tuple(cut_points.conjugate()))
    cuts=np.zeros((4,4),complex)
    ps, weights = _momentum_quadrature(nodes, 4, .03, scheme="infinite_gauss")
    for p, pw in zip(ps, weights):
        common = dict(maximum_twice_level=order, digits=max(80, 4*order+40),
                      spin7_maximum_order=12, sld_block_backend="elliptic_recursion")
        m = MOMENTA
        original = build_fixed_p_four_ramond_kernel(float(p),
            external_liouville_momenta=m, time_momenta=(*m[:3], -m[3]),
            families=FAMILIES, **common)
        m = (m[0], m[2], m[1], m[3])
        swapped = build_fixed_p_four_ramond_kernel(float(p),
            external_liouville_momenta=m, time_momenta=(*m[:3], -m[3]),
            families=(FAMILIES[0], FAMILIES[2], FAMILIES[1], FAMILIES[3]), **common)
        for c in ("identity","fermion"):
            original_part=replace(original, terms=tuple(t for t in original.terms if t.time_channel==c))
            swapped_part=replace(swapped, terms=tuple(t for t in swapped.terms if t.time_channel==c))
            separated["direct"][c] += pw/np.pi * _evaluate_uniform_kernel(original_part, z, {}, spin_z)/abs(u[:, None])**4
            values = _evaluate_uniform_kernel(swapped_part, u, {}, spin_u)
            separated["folded"][c] += pw/np.pi * np.array([-spin7_reexpress_pairing_coefficients(
                row, source_pairing="02|13", target_pairing="01|23") for row in values])
        cuts += pw/np.pi*_evaluate_uniform_kernel(original,cut_points,{},spin_cuts)
    direct=sum(separated["direct"].values())
    folded=sum(separated["folded"].values())
    d0=separated["direct"]["identity"]-separated["folded"]["identity"]
    d1=separated["direct"]["fermion"]-separated["folded"]["fermion"]
    tau=-np.vdot(d1,d0)/np.vdot(d1,d1)
    def pairs(a):
        return np.stack([np.asarray(a).real, np.asarray(a).imag], axis=-1).tolist()
    return dict(order=order, p_nodes=nodes, momenta=pairs(MOMENTA),
        u=pairs(u), direct=pairs(direct), folded=pairs(folded),
        relative_defect=(np.linalg.norm(direct-folded, axis=1)/np.linalg.norm(direct, axis=1)).tolist(),
        unconstrained_best_tau=pairs(tau),
        best_tau_residual=float(np.linalg.norm(d0+tau*d1)/np.linalg.norm(direct)),
        cut_points=pairs(cut_points),cut_values=pairs(cuts),
        cut_jump_relative=[float(np.linalg.norm(cuts[i]-cuts[i+1])/np.linalg.norm(cuts[i])) for i in (0,2)],
        seconds=time.perf_counter()-started, production_certified=False)


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--order", type=int, default=15)
    parser.add_argument("--nodes", type=int, default=24)
    parser.add_argument("--output", type=Path, required=True)
    args=parser.parse_args()
    result=run(args.order,args.nodes)
    args.output.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))
