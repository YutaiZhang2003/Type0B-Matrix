#!/usr/bin/env python3
"""Convert the historical frozen physical inputs to the paper's notation.

This boundary reverses geometry order (0,1,infinity) to paper edges (1,2,3),
sets C_1=i*Ctilde, and names the NSRR coefficients C_{f,eta}. It reads no
saved block coefficients or densities. Special functions are not recomputed.
"""
import argparse
from decimal import Decimal
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text(), parse_float=str)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2)+'\n')


def components(x):
    if isinstance(x, dict):
        return str(x['real']), str(x['imag'])
    if isinstance(x, list):
        return str(x[0]), str(x[1])
    return str(x), '0'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args=parser.parse_args()
    if (args.output/'provenance.json').exists():
        raise FileExistsError('Input conversion already complete; use a fresh directory')
    old=read(args.input/'config.json')
    point=old['point']
    config={'schema':'paper-partition-fixed-spin-input-v2', 'b':old['b'], 'edge_order':[1,2,3],
            'point':{key:point[key] for key in ('source_free','target_free')}}
    for key in ('q_source','q_target'):
        config['point'][key]=list(reversed(point[key]))
    config['source_marked_spins']=[[[1,1],[0,0]],[[1,1],[1,1]]]
    config['target_marked_spins']=[[[0,0],[0,0]],[[0,0],[1,0]]]
    # One physical plumbing-sign assignment per marked spin, in paper order.
    config['source_eta_e']=[[1,-1,1],[-1,-1,1]]
    config['target_eta_e']=[[-1,1,-1],[-1,1,1]]
    config['source_period_branch']=[[0,0],[0,1]]
    config['target_period_branch']=[[-1,-1],[-1,0]]
    config['source_to_target']=[[-1,0,0,0],[1,-1,0,-2],[0,0,-1,-1],[-1,1,0,1]]
    config['provenance']={'input':str(args.input), 'policy':'physical inputs only; no saved blocks or densities'}
    write(args.output/'config.json',config)
    hashes={}
    for channel,N in [('source',7),('target',10)]:
        paths=sorted((args.input/channel/f'N{N}').glob('node-*.json'))
        if len(paths)!=N**3:
            raise ValueError(f'Incomplete {channel} grid')
        for path in paths:
            node=read(path)
            result={key:node[key] for key in ('index','N','measure')}
            result.update(schema='paper-partition-node-input-v1',channel=channel,
                          momenta=list(reversed(node['momenta'])))
            C=[components(x) for x in node['constants']]
            if any(abs(float(imag))>1e-30*max(1e-300,abs(float(real))) for real,imag in C):
                raise ValueError('Historical raw coefficients must be real')
            if channel=='target':
                result['C_a']=[list(C[0]),[str(Decimal(C[1][1]).copy_negate()),C[1][0]]]
            else:
                result['C_f_eta']=[[list(c) for c in C] for f in (0,1)]
            write(args.output/channel/path.name,result)
            hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    write(args.output/'provenance.json',hashes)
    print(json.dumps({'config':str(args.output/'config.json'),'source_nodes':343,'target_nodes':1000}))

if __name__=='__main__':
    main()
