"""Crossing and cut audit for the reconstructed nonchiral four-R kernel."""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"Codes"))
from benchmark_so7e8_four_ramond_convergence import MOMENTA,FAMILIES
from so7e8_four_ramond_nonchiral import build_nonchiral_fixed_p_four_ramond_kernel
from so7e8_four_ramond_assembly import SPIN7_FIERZ_02_TO_01,SPIN7_BILINEAR_EXCHANGE_SIGNS
from so7e8_direct_domain import spin7_global_grid
from so7e8_four_ramond_liouville_integral import _evaluate_uniform_kernel
from spin23_singlet_amplitudes import _momentum_quadrature


def density(m,z,order,nodes):
    spin = spin7_global_grid(tuple(z.conjugate()))
    ps,pw = _momentum_quadrature(nodes,4,.03,scheme="infinite_gauss")
    result = np.zeros((len(z),4),complex)
    for p,weight in zip(ps,pw):
        kernel = build_nonchiral_fixed_p_four_ramond_kernel(float(p),
            external_liouville_momenta=m,time_momenta=(*m[:3],-m[3]),
            families=FAMILIES,maximum_twice_level=order,spin7_maximum_order=12,
            sld_block_backend="elliptic_recursion",digits=max(80,4*order+40))
        result += weight/np.pi*_evaluate_uniform_kernel(kernel,z,{},spin)
    return result


def run(order,nodes,momenta=MOMENTA,cut_epsilon=1e-7):
    started = time.perf_counter()
    u = np.array([.37+.23j,.68-.15j,.25-.4j,.71+.4j,.51+.31j,.24+.12j])
    cuts = np.array([x+lip*cut_epsilon*1j for x in (1.3,2,3) for lip in (1,-1)])
    m = momenta
    original = density(m,np.r_[1/u,1-u,cuts],order,nodes)
    inv = density((m[0],m[2],m[1],m[3]),u,order,nodes)
    refl = density((m[2],m[1],m[0],m[3]),u,order,nodes)
    d = np.diag(SPIN7_BILINEAR_EXCHANGE_SIGNS)
    f = SPIN7_FIERZ_02_TO_01
    a = original[:len(u)]/abs(u[:,None])**4
    b = -inv@f.T
    c = original[len(u):2*len(u)]
    e = -refl@(d@f@d).T
    cut_values = original[2*len(u):]
    pairs = lambda a:np.stack([np.asarray(a).real,np.asarray(a).imag],axis=-1).tolist()
    relative = lambda a,b:(np.linalg.norm(a-b,axis=1)/np.linalg.norm(a,axis=1)).tolist()
    return dict(order=order,p_nodes=nodes,momenta=pairs(m),u=pairs(u),
        inversion_defect=relative(a,b),reflection_defect=relative(c,e),
        cut_epsilon=cut_epsilon,cut_points=pairs(cuts),
        cut_jump=relative(cut_values[::2],cut_values[1::2]),
        exterior=pairs(a),inverted=pairs(b),reflected_direct=pairs(c),reflected_local=pairs(e),
        cut_values=pairs(cut_values),seconds=time.perf_counter()-started,
        sewing_basis="nonchiral_spin_fields",production_certified=False)


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--order",type=int,default=19)
    parser.add_argument("--nodes",type=int,default=48)
    parser.add_argument("--cut-epsilon",type=float,default=1e-7)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    report=run(args.order,args.nodes,cut_epsilon=args.cut_epsilon)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))
