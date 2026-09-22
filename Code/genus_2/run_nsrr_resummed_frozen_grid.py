#!/usr/bin/env python3
"""Resummed NSRR worker/reducer for the existing frozen Liouville grid.

No grid is generated or enlarged. --index evaluates one existing momentum
node; --reduce requires every frozen node and keeps the saved all-NS target.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import numpy as np

from nsrr_resummed_sewing import resummed_integrand, RUNTIME
from nsrr_resummed_backend import executable
from nsrr_normalization import LOCAL, NORMALIZATIONS, normalization_metadata

ROOT=Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def prepare(args):
    source=json.loads((args.bundle/'config.json').read_text())
    binary=executable()
    kernels=[Path(__file__),RUNTIME/'nsrr_resummed_backend.py',
             ROOT/'Code/genus_2/nsrr_resummed_sewing.py',
             ROOT/'Code/genus_2/nsrr_bilinear_sewing.py',
             ROOT/'Code/genus_2/nsrr_normalization.py',
             ROOT/'Code/genus_2/nsrr_plumbing_adapter.py',ROOT/'Code/genus_2/physical_nsrr_sewing.py']
    run=dict(schema='nsrr-resummed-frozen-grid-v1',source_config_sha256=sha(args.bundle/'config.json'),
             binary_sha256=sha(binary),implementation_sha256={str(p.relative_to(ROOT)):sha(p) for p in kernels},
             point_id=source['point_id'],nodes=len(source['nodes']),momentum_rule='unchanged frozen inputs',
             target_policy='unchanged saved all-NS target',branch_level=args.branch_level,
             branch_truncation=args.branch_truncation,recursion_order=args.recursion_order,
             global_tolerance=args.global_tolerance,global_max_shell=args.global_max_shell,dps=40,
             sewing_convention=args.sewing_convention,physical_lifts_slots=args.physical_lifts_slots,
             normalization_policy=(normalization_metadata(args.normalization)
                 if args.sewing_convention=='human-bilinear' else
                 dict(normalization='legacy-times-four',normalization_factor=4,
                      normalization_status='legacy candidate',global_normalization_verified=False)))
    path=args.output/'run.json'
    # Multiple array indices may initialize the same run. Publish a complete
    # file with an exclusive hard link so readers never see a partial write.
    path.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w',dir=path.parent,delete=False) as stream:
        temporary=Path(stream.name)
        json.dump(run,stream,indent=2,allow_nan=False)
    try:
        try: os.link(temporary,path)
        except FileExistsError: pass
    finally: temporary.unlink()
    if json.loads(path.read_text())!=run:
        raise ValueError('Output belongs to different parameters, inputs or implementation')
    return source,run


def evaluate(args,source,run):
    index=args.index
    if not 0<=index<len(source['nodes']): raise ValueError('index is outside the frozen grid')
    spec=source['nodes'][index]
    assert spec['index']==index
    path=args.bundle/spec['input']
    if sha(path)!=spec['sha256']: raise ArithmeticError('frozen momentum input changed')
    with np.load(path,allow_pickle=False) as z:
        momenta=z['momenta'].tolist();constants=z['norms'].tolist()
    if momenta!=spec['momenta']: raise ArithmeticError('momentum slot mismatch')
    output=args.output/'nodes'/f'node-{index:05d}.json'
    if output.exists():
        old=json.loads(output.read_text())
        if old['run_digest']!=digest(run) or old['index']!=index or old['status']!='complete':
            raise ArithmeticError('inconsistent completed node')
        return old
    value=resummed_integrand(b=source['b'],momenta_geometry=momenta,
             q_geometry=tuple(map(complex,source['source']['q_values'])),
             lifts_geometry=source['source']['lifts'],bry_constants=constants,
             branch_level=args.branch_level,branch_truncation=args.branch_truncation,
             recursion_order=args.recursion_order,global_tolerance=args.global_tolerance,
             global_max_shell=args.global_max_shell,dps=40,
             sewing_convention=args.sewing_convention,physical_lifts_slots=args.physical_lifts_slots,
             normalization=args.normalization,
             output_directory=args.output/'native'/f'node-{index:05d}')
    result=dict(status='complete',run_digest=digest(run),index=index,input_sha256=spec['sha256'],
                momenta=momenta,measure=spec['measure'],integrand=value['integrand'],
                blocks={str(k):str(v) for k,v in value['blocks'].items()},
                descendant_blocks={str(k):str(v) for k,v in value['descendant_blocks'].items()},
                primary_prefactor=str(value['primary_prefactor']),
                primary_weights_slots=list(value['primary_weights_slots']),
                log_q_slots=list(map(str,value['log_q_slots'])),
                sewing_convention=value['sewing_convention'],
                human_bilinear_pairing_verified=value['human_bilinear_pairing_verified'],
                local_bilinear_kernel_verified=value['local_bilinear_kernel_verified'],
                normalization=value['normalization'],normalization_factor=value['normalization_factor'],
                normalization_status=value['normalization_status'],
                global_normalization_verified=value['global_normalization_verified'],
                marked_spin_transport_verified=value['marked_spin_transport_verified'],
                diagnostics=value['diagnostics'])
    save(output,result)
    return result


def reduce(args,source,run):
    files=[args.output/'nodes'/f'node-{i:05d}.json' for i in range(len(source['nodes']))]
    missing=[i for i,p in enumerate(files) if not p.is_file()]
    if missing:
        raise RuntimeError(f'No partial-grid ratio: {len(missing)} of {len(files)} momentum nodes are missing')
    nodes=[json.loads(p.read_text()) for p in files]
    for i,(spec,node) in enumerate(zip(source['nodes'],nodes)):
        if (node['status']!='complete' or node['run_digest']!=digest(run) or node['index']!=i
            or node['input_sha256']!=spec['sha256'] or sha(args.bundle/spec['input'])!=spec['sha256']
            or node['momenta']!=spec['momenta'] or node['measure']!=spec['measure']):
            raise ArithmeticError(f'Frozen-grid node {i} provenance mismatch')
    z=math.fsum(n['measure']*n['integrand']['total'] for n in nodes)
    source_q=z/source['source']['Z_free']**source['kappa']
    result=dict(status='complete',run=run,completed_nodes=len(nodes),source_Z=z,source_Q=source_q,
                target_Q=source['target_Q'],source_over_target=source_q/source['target_Q'],
                baseline_source_Z=source['baseline_source_Z'],baseline_ratio=source['baseline_ratio'],
                relative_source_change=z/source['baseline_source_Z']-1,
                weighted_absolute_integrand_change=math.fsum(
                    abs(n['measure'])*abs(n['integrand']['total']-s['baseline_Z_node'])
                    for s,n in zip(source['nodes'],nodes))/abs(source['baseline_source_Z']))
    save(args.output/'summary.json',result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    action=parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--index',type=int);action.add_argument('--reduce',action='store_true')
    parser.add_argument('--branch-level',type=int,default=8)
    parser.add_argument('--branch-truncation',choices=['total','per-edge'],default='per-edge')
    parser.add_argument('--recursion-order',type=int,default=8)
    parser.add_argument('--global-tolerance',type=float,default=1e-13)
    parser.add_argument('--global-max-shell',type=int,default=96)
    parser.add_argument('--sewing-convention',choices=['human-bilinear','legacy-times-four'],default='human-bilinear')
    parser.add_argument('--normalization',choices=NORMALIZATIONS,default=LOCAL,
                        help='provisional-times-four applies the user-requested assumption to the new bilinear M')
    parser.add_argument('--physical-lifts-slots',type=int,nargs=3,choices=[-1,1],metavar=('NS','R1','R0'))
    args=parser.parse_args();args.bundle=args.bundle.resolve();args.output=args.output.resolve()
    if args.sewing_convention=='human-bilinear' and args.physical_lifts_slots is None:
        parser.error('--physical-lifts-slots NS R1 R0 is required for the physical bilinear matrix')
    if args.sewing_convention!='human-bilinear' and args.normalization!=LOCAL:
        parser.error('the provisional normalization requires --sewing-convention human-bilinear')
    if min(args.branch_level,args.recursion_order)<0 or not 0<args.global_tolerance<1 or not 4<=args.global_max_shell<=256:
        parser.error('invalid independent truncation or accuracy controls')
    source,run=prepare(args)
    result=reduce(args,source,run) if args.reduce else evaluate(args,source,run)
    print(json.dumps(result,indent=2,allow_nan=False))


if __name__=='__main__': main()
