"""Prepare reusable OPE, cusp-collision, or elliptic-tail banks offline."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import numpy as np

from spin23_genus1_vv_tail_bank import tail_endpoint_rule,prepare_tail_spectral
from spin23_genus1_vv_ope import OPEBank,prepare_ope_node,merge_ope_nodes
from spin23_genus1_vv_matching import prepare_cusp_collision


def ope_task(task):
    path,p,omega,cutoff,identity=task
    if path.exists():
        bank=OPEBank.load(path,expected=dict(preparation_id=identity))
        if not np.array_equal(bank.momenta,[p]): raise ValueError('cached momentum mismatch')
    else:
        bank=prepare_ope_node(p,omega,cutoff=cutoff)
        bank.metadata['preparation_id']=identity;bank.save(path)
    return path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('kind',choices=('ope','cusp','tail'))
    parser.add_argument('--omega',type=complex,default=.25j)
    parser.add_argument('--order',type=int,help='bridge order (OPE/cusp) or short-edge order (tail)')
    parser.add_argument('--loop-order',type=int)
    parser.add_argument('--loop-max',type=float,default=2.)
    parser.add_argument('--cutoff',type=int,default=4,help='twice handle level for OPE')
    parser.add_argument('--q-order',type=int,default=8,help='elliptic order for tail')
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.omega.real!=0 or not 0<args.omega.imag<.5:
        raise ValueError('this target uses omega=i*kappa with 0<kappa<1/2')
    defaults={'ope':(24,16),'cusp':(40,32),'tail':(28,32)}
    order=args.order or defaults[args.kind][0];loop_order=args.loop_order or defaults[args.kind][1]
    momenta,weights=tail_endpoint_rule(order,loop_order,args.loop_max)
    names=('Codes/prepare_spin23_genus1_vv_completion.py','Codes/spin23_genus1_vv_ope.py',
        'Codes/spin23_genus1_vv_matching.py','Codes/spin23_genus1_vv_tail_bank.py',
        'Codes/spin23_genus1_onepoint_recursion.py','Codes/spin23_genus1_recursive_sewing.py',
        'Codes/spin23_genus1_branch_recursion.py','Codes/audit_spin23_genus1_vv_collision.py',
        'Codes/spin23_super_liouville_data.py')
    root=Path(__file__).resolve().parents[1]
    hashes={n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in names}
    config=dict(kind=args.kind,energy=[args.omega.real,args.omega.imag],order=order,
        loop_order=loop_order,loop_max=args.loop_max,
        cutoff=args.cutoff if args.kind=='ope' else args.q_order if args.kind=='tail' else 1,
        q_order=args.q_order,
        source_hashes=hashes)
    identity=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
    config['preparation_id']=identity
    args.output.parent.mkdir(parents=True,exist_ok=True)
    if args.output.exists():
        with np.load(args.output,allow_pickle=False) as a:
            old=json.loads(str(a['metadata']))
        if old.get('preparation_id')!=identity: raise ValueError('output belongs to another preparation')
        print('already prepared',args.output);return
    if args.kind=='ope':
        cache=args.output.parent/(args.output.stem+'_nodes');cache.mkdir(exist_ok=True)
        tasks=[(cache/f'{identity[:16]}_{i}.npz',p,args.omega,args.cutoff,identity) for i,p in enumerate(momenta)]
        files=[]
        with ProcessPoolExecutor(args.workers) as pool:
            for i,path in enumerate(pool.map(ope_task,tasks)):
                files.append(path)
                if i%32==0: print('prepared',i+1,'/',len(tasks),flush=True)
        bank=merge_ope_nodes([OPEBank.load(f) for f in files],momenta,weights,config)
        bank.save(args.output)
    elif args.kind=='tail':
        bank=prepare_tail_spectral(args.omega,momenta,weights,q_order=args.q_order)
        bank.metadata.update(config);bank.save(args.output)
    else:
        bank=prepare_cusp_collision(args.omega,momenta,weights)
        arrays=dict(momenta=bank.momenta,weights=bank.weights,coefficients=bank.coefficients)
        if not all(np.isfinite(v).all() for v in arrays.values()): raise ArithmeticError('nonfinite cusp bank')
        config['array_sha256']={k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in arrays.items()}
        temporary=args.output.with_suffix('.pending.npz')
        np.savez(temporary,metadata=json.dumps(config),**arrays);temporary.replace(args.output)
    if hashes!={n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in names}:
        raise RuntimeError('source changed during preparation; output is not validated')
    print('saved',args.output,flush=True)


if __name__=='__main__':main()
