"""Independent direct Ward/Gram modular diagnostic for the recursive VV banks.

Validation only: no production coefficient replacement. Uses full-half-line,
geometry-adapted Laguerre quadrature rather than the fixed bank momentum grid.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import math
from pathlib import Path
import time

import numpy as np
from scipy.special import roots_genlaguerre

from spin23_genus1_vv_bank import VVGeometry
from spin23_genus1_banked_integral import complex_json


def node(args):
    momenta, weight, omega, q, lifts, cutoff = args
    from spin23_genus1_amplitude import G_MINUS_HALF as G
    from spin23_genus1_ns_point_sewing import paired_momentum_integrand as ns
    from spin23_genus1_ramond_point_sewing import paired_momentum_integrand as ramond
    from spin23_genus1_ns_point_sewing import vertex as nv
    from spin23_genus1_ramond_point_sewing import vertex_pair as rv
    result = np.zeros((3, 2), complex)
    for spin in range(3):
        lift = list(lifts)
        if spin == 1: lift[-1] *= -1
        for ci, words in enumerate(((G, G), ((), ()))):
            kw = dict(external_momenta=(omega, omega), plumbing_parameters=q,
                holomorphic_words=words, antiholomorphic_words=((), ()),
                maximum_twice_levels=cutoff, strip_internal_gaussian=True)
            if spin != 2: kw['edge_lift_signs'] = tuple(lift)
            result[spin, ci] = weight*(ramond if spin == 2 else ns)(momenta, **kw)
    nv.cache_clear(); rv.cache_clear()
    return result


def audit(*, omega=.5, cutoff=8, order=11, workers=4, tau=1j, z=.249+.5j):
    transformed_tau = -1/tau; transformed_z = z/tau
    transformed_z -= math.floor(transformed_z.imag/transformed_tau.imag)*transformed_tau
    transformed_z -= math.floor(transformed_z.real)
    result = []; started = time.perf_counter()
    with ProcessPoolExecutor(workers) as pool:
        for t, point in ((tau, z), (transformed_tau, transformed_z)):
            geometry = VVGeometry([t], [[0, point]])
            q = tuple(geometry.plumbing[0]); scales = -np.log(abs(geometry.plumbing[0]))
            x, w = roots_genlaguerre(order, -.5)
            args = [((float(np.sqrt(x[i]/scales[0])), float(np.sqrt(x[j]/scales[1]))),
                float(w[i]*w[j]/(4*math.pi**2*np.sqrt(scales.prod()))), complex(omega), q,
                tuple(map(int, geometry.lifts[0])), cutoff) for i in range(order) for j in range(order)]
            spectral = sum(pool.map(node, args))
            result.append(spectral*geometry.multipliers(omega)[0])
    a = result[0]; b = result[1][(0, 2, 1), :]*abs(tau)**-6
    return dict(schema='spin23-vv-direct-modular-diagnostic-v1',
        backend='independent direct Ward/Gram; validation only',
        quadrature='full-half-line generalized Laguerre', order=order,
        per_edge_twice_cutoff=cutoff, omega=[complex(omega).real, complex(omega).imag],
        tau=[tau.real, tau.imag], z=[z.real, z.imag], original=complex_json(a),
        pulled_back=complex_json(b), relative_component_difference=(abs(a-b)/np.maximum(abs(a), 1e-300)).tolist(),
        total_ratio=[complex(b.sum()/a.sum()).real, complex(b.sum()/a.sum()).imag],
        seconds=time.perf_counter()-started)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--order', type=int, default=11)
    parser.add_argument('--cutoff', type=int, default=8)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--omega', type=complex, default=.5)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = audit(omega=args.omega, cutoff=args.cutoff, order=args.order, workers=args.workers)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__': main()
