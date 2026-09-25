#!/usr/bin/env python3
"""Uncertified two-R/two-NS integral with no exterior vertex permutation.

Uses the original ordered correlator at z=1/u with d2z=|u|^-4 d2u.
This provides an independent check of the corrected inversion transport.
Local Ward tests now pass; integrated endpoint convergence must still be
checked separately. All mixed blocks use the double-Virasoro c-recursion
coefficient generator followed by elliptic-H truncation, including SS.
The internal channel is NS throughout, despite "direct" in this filename:
"direct" here refers to evaluating the ORIGINAL exterior correlator.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"Codes"))
from benchmark_so7e8_four_ramond_convergence import MOMENTA, MODULI_GRIDS
from heterotic_so23_1to3_vvvv_fit_bundle import heterotic_so23_1to3_fast as fast
from so7e8_direct_domain import exterior_grid
from so7e8_liouville_chamber import require_conservative_real_p_chamber
from so7e8_two_ramond_liouville_integral import _uniformize_kernel, collision_convergence_margins
from so7e8_ramond_sphere_integrand import build_fixed_p_two_ramond_kernel
from spin23_singlet_amplitudes import _momentum_quadrature, _stable_complex_sum
from heterotic_so23_1to3_vvvv_fit_bundle.liouville_momentum_quadrature import threshold_weighted_rule
from sphere_block_uniformization import elliptic_nome


def run(order, nodes, grid, species, picture, swapped=False, momenta=None,
        momentum_scheme="threshold_weighted", momentum_threshold_options=None):
    started=time.perf_counter()
    momenta=tuple(MOMENTA if momenta is None else momenta)
    if swapped:
        momenta=(momenta[0],momenta[2],momenta[1],momenta[3])
    require_conservative_real_p_chamber(momenta[:3],momenta[3])
    if not collision_convergence_margins(momenta).convergent:
        raise ValueError("outside collision convergence chamber")
    opts=MODULI_GRIDS[grid]
    bulk,bw,*_=fast.annulus_excluding_lens(.12,.10,opts["theta_orders"],opts["radial_order"])
    zero,zw=fast.disk_grid(.12,opts["disk_radial_order"],opts["disk_angular_order"],opts.get("disk_radial_power",3))
    w,lw=fast.ref._lens_grid(.10,opts["lens_radial_order"],opts["lens_angular_order"],opts.get("lens_radial_power",3))
    inside=(("bulk",bulk,bw),("zero",zero,zw),("one",1-w,lw))
    grids=[("inside_"+name,z,weights) for name,z,weights in inside]
    grids += [("outside_"+name,*exterior_grid(z,weights)) for name,z,weights in inside]
    ps,pw=_momentum_quadrature(nodes,4,.03,scheme=momentum_scheme,
        momentum_threshold_options=momentum_threshold_options)
    pieces={}
    species_list=("SS","SV","VS","VV") if species=="all" else (species,)
    backend="double_virasoro"
    for labels in species_list:
        totals={name:{} for name,_,_ in grids}
        for p,weight in zip(ps,pw):
            kernel=build_fixed_p_two_ramond_kernel(float(p),
                external_liouville_momenta=momenta,time_momenta=(*momenta[:3],-momenta[3]),
                ns_at_z=labels[0],ns_at_one=labels[1],
                zero_ramond_family="Psi_tilde",infinity_ramond_family="Psi",
                channel="crossed",picture_zero_at=picture,maximum_twice_level=order,
                digits=max(80,4*order+40),crossed_block_backend=backend)
            uniform=_uniformize_kernel(kernel,elliptic_prefactor=True)
            for name,z,weights in grids:
                for tensor,values in uniform.evaluate_many(z).items():
                    contribution=weight/np.pi*_stable_complex_sum(weights*values)
                    if not np.isfinite(contribution):
                        raise ArithmeticError(f"nonfinite {labels} {name} P={p}")
                    totals[name][tensor]=totals[name].get(tensor,0j)+contribution
        pieces[labels]=totals
    coefficients={labels:{key:sum(values.get(key,0j) for values in totals.values())
                        for key in next(iter(totals.values()))}
                  for labels,totals in pieces.items()}
    def encode(x):
        if isinstance(x,dict):return {k:encode(v) for k,v in x.items()}
        if isinstance(x,(complex,np.complexfloating)):return [float(x.real),float(x.imag)]
        return x
    source_files=[Path(__file__),*(ROOT/"Codes").glob("so7e8_*.py"),
        ROOT/"Codes/virasoro_sphere_c_recursion.py",
        ROOT/"Codes/ramond_sphere_uniformization.py",
        ROOT/"Codes/heterotic_so23_1to3_vvvv_fit_bundle/liouville_momentum_quadrature.py"]
    hashes={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in source_files}
    return dict(order=order,p_nodes=nodes,grid=grid,momenta=[[p.real,p.imag] for p in momenta],
        species=species,picture_zero_at=picture,swapped_ns_momenta=swapped,
        coefficients=encode(coefficients),pieces=encode(pieces),
        max_abs_nome=max(float(np.max(abs(elliptic_nome(1-z)))) for _,z,_ in grids),
        backend=backend,two_star_inverse_gram_fallback=False,full_pbw_block_used=False,
        momentum_quadrature=(threshold_weighted_rule(nodes,momentum_threshold_options).metadata()
            if momentum_scheme=="threshold_weighted" else
            {"scheme":momentum_scheme,"node_count":len(ps)}),
        exterior_method="original correlator at 1/u with |u|^-4 area Jacobian",
        seconds=time.perf_counter()-started,source_sha256=hashes,production_certified=False)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--order",type=int,default=7)
    parser.add_argument("--nodes",type=int,default=16)
    parser.add_argument("--grid",choices=tuple(MODULI_GRIDS),default="fine")
    parser.add_argument("--species",choices=("all","SS","SV","VS","VV"),default="all")
    parser.add_argument("--picture",choices=("z","one"),default="z")
    parser.add_argument("--swapped",action="store_true")
    parser.add_argument("--momentum-scheme",choices=("threshold_weighted","infinite_gauss","cutoff"),
        default="threshold_weighted")
    parser.add_argument("--threshold-options",type=json.loads,default=None)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    result=run(args.order,args.nodes,args.grid,args.species,args.picture,args.swapped,
        momentum_scheme=args.momentum_scheme,momentum_threshold_options=args.threshold_options)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))
