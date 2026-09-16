#!/usr/bin/env python3
"""Numerical cross-channel test of the proposed NSRR bilinear pairing.

Recompute the complex all-NS lift vector, apply the explicit parity basis
conversion, and transport the two RR spin components separately. This is
a test of the proposal, not an assertion that free spin transport proves
the interacting off-diagonal fusion/pairing identity.

The five-point N=3 run compares equal quadrature orders. The central N=4
run tests quadrature variation. Source L=3/5 and target recursion orders
8/12 provide separate truncation controls. No multiplicative fit is used.
"""
from __future__ import annotations

import argparse
import cmath
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
for directory in ('Code', 'Code/genus_2', 'Code/c_Recursion', 'Code/genus_2_cross_channel'):
    sys.path.insert(0, str(ROOT/directory))
import numpy as np
from all_ns_reflected_sewing import LIFTS, lift_conversion
from compare_nsrr_nsnsns_theta import NSGenus2CRecursion, GenericSuperLiouvilleConstants, _rules, _measure
from fixed_spin_free_plumbing import fixed_spin_chiral_partition
from spin_structure import SpinCharacteristic
from nsrr_bilinear_sewing import CHANNELS, contract_nsrr_bilinear, BASE_PROJECTION_LIFTS
from nsrr_plumbing_adapter import NSRRPlumbingInputs, GEOMETRY_SECTORS
from recombine_saved_genus2_coefficient_ledger import Inputs, decode, encode, csum, object_digest, relative, write_csv

DATA = ROOT/'Data Set'
OUTPUT = DATA/'nsrr_bilinear_cross_channel_20260915'
FREE = DATA/'fixed_spin_free_NSrr_20260830/summary.json'
TARGET = DATA/'nsrr_nsnsns_target_R8_R12_R16_N5_20260830/config.json'


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def source_rows(inputs):
    sums = {}
    primary_error = 0.
    for name in ('nsrr_factorized_sign_trial_L3_N5_20260830', 'nsrr_trial_L5_N3_local_20260830'):
        directory = DATA/name
        config = inputs.read(directory/'summary.json')['config']
        digest = object_digest(config)
        assert config['channels'] == list(map(list, CHANNELS))
        tasks = [(n, i) for n in config['quadrature_orders'] for i in range(n**3)]
        paths = sorted((directory/'shards').glob('node-*.json'))
        assert len(paths) == len(tasks)
        for index, (path, task) in enumerate(zip(paths, tasks)):
            shard = inputs.read(path)
            assert (shard['index'], shard['quadrature_order'], shard['node'], shard['config_digest']) == (index, *task, digest)
            constants = tuple(decode(z) for z in shard['C_BRY'])
            rows = {(r['t'], float(r['level']), tuple(r['lifts_geometry'])): r for r in shard['rows']}
            for point in config['points']:
                primary = NSRRPlumbingInputs(tuple(map(complex, point['q_geometry'])),
                    (1,1,1), GEOMETRY_SECTORS).primary(config['b'], shard['momenta_geometry'])
                for level in (3., 5.):
                    if level not in config['levels']: continue
                    pair = [rows[point['t'], level, lift] for lift in BASE_PROJECTION_LIFTS]
                    assert decode(pair[0]['primary']) == decode(pair[1]['primary'])
                    primary_error = max(primary_error, relative(primary, decode(pair[0]['primary'])))
                    f = {label: csum(decode(row['blocks'][i]) for row in pair)/math.sqrt(2)
                         for i, label in enumerate(CHANNELS)}
                    density = np.zeros((2,2), complex)
                    for j, eta in enumerate((1,-1)):
                        d = np.array([f[0,eta,eta], 1j*f[1,eta,eta]])
                        density += constants[j]**2/4*np.outer(d,d.conjugate())
                    density *= shard['measure']*abs(primary)**2
                    for sign in (1,-1):
                        z = contract_nsrr_bilinear(descendant_blocks=f,
                            antiholomorphic_blocks={k:v.conjugate() for k,v in f.items()},
                            left_bry=constants,right_bry=constants,physical_lifts_slots=(sign,-1,1),
                            primary=primary,antiholomorphic_primary=primary.conjugate())['total']*shard['measure']
                        v = np.array([1,-1j*sign])/math.sqrt(2)
                        assert abs(z-v@density@v.conjugate()) < 2e-13*max(abs(z),np.max(abs(density)),1e-300)
                    key = (name, task[0], level, point['t'])
                    sums.setdefault(key, []).append(density)
    assert primary_error < 2e-13
    result=[]
    for (name,n,level,t), matrices in sorted(sums.items()):
        density=np.array([[csum(m[i,j] for m in matrices) for j in range(2)] for i in range(2)])
        result.append(dict(dataset=name,N=n,L=level,t=t,
                           density=[[encode(z) for z in row] for row in density]))
    return result, primary_error


def geometry_and_phases(inputs):
    free=inputs.read(FREE)
    output=[]
    for point in free['points']:
        spin_rows=[]
        for beta in ((0,0),(1,1)):
            ss=SpinCharacteristic((1,1),beta)
            ts=ss.transport(free['source_to_target'])
            assert ts.alpha == (0,0)
            values={}
            for channel,spin in (('source_NSrr',ss),('target_NSnsns',ts)):
                d=point[channel]
                omega=[[complex(z)-k for z,k in zip(row,branch)]
                       for row,branch in zip(d['omega_charge'],d['period_branch'])]
                value=fixed_spin_chiral_partition(tuple(map(complex,d['q_values'])),omega,spin.pairs,
                    period_branch=d['period_branch'],max_mode=32,radial_steps=32)
                values[channel]=value
            target_charge=ts.charge_frame(point['target_NSnsns']['period_branch'])
            ratio=values['source_NSrr']['majorana_chiral']/values['target_NSnsns']['majorana_chiral']
            spin_rows.append(dict(source_marked=ss.pairs,target_marked=ts.pairs,
                target_charge=target_charge.pairs,target_raw_lift=target_charge.all_ns_determinant_lifts(),
                majorana_source=encode(values['source_NSrr']['majorana_chiral']),
                majorana_target=encode(values['target_NSnsns']['majorana_chiral']),
                chiral_frame_ratio=encode(ratio),
                free_frame_ratio=values['source_NSrr']['Z_free_nonchiral']/values['target_NSnsns']['Z_free_nonchiral']))
        relative_phase=decode(spin_rows[1]['chiral_frame_ratio'])/decode(spin_rows[0]['chiral_frame_ratio'])
        assert abs(relative_phase-1)<5e-8
        assert abs(spin_rows[1]['free_frame_ratio']/spin_rows[0]['free_frame_ratio']-1)<5e-8
        # The independently fixed degeneration branches give relative phase
        # +1 on the entire family; do not fit a phase to the interacting Z.
        output.append(dict(t=point['t'],spin_rows=spin_rows,
            measured_relative_spin_phase=encode(relative_phase),transport_relative_phase=[1.,0.],
            source_free=point['source_NSrr']['Z_free'],target_free=point['target_NSnsns']['Z_free'],
            q_source=point['source_NSrr']['q_values'],q_target=point['target_NSnsns']['q_values']))
    # The other two RR characteristics transport to an R-containing target,
    # so comparing r=+1 with an all-NS target would compare different sectors.
    excluded=[dict(source=SpinCharacteristic((1,1),beta).pairs,
        target=SpinCharacteristic((1,1),beta).transport(free['source_to_target']).pairs)
        for beta in ((0,1),(1,0))]
    return output,free['source_to_target'],excluded


def prepare(output):
    inputs=Inputs()
    old=inputs.read(TARGET)['baseline_config']
    geometry,matrix,excluded=geometry_and_phases(inputs)
    sources,error=source_rows(inputs)
    source_files=('Code/genus_2/test_nsrr_bilinear_cross_channel.py',
        'Code/genus_2/nsrr_bilinear_sewing.py','Code/genus_2/all_ns_reflected_sewing.py',
        'Code/c_Recursion/ns_genus2_partition.py','Code/c_Recursion/generic_super_liouville_structure_constants.py')
    for name in source_files: inputs.bytes(ROOT/name)
    config=dict(schema='nsrr-bilinear-cross-channel-v1',b=old['parameters']['b'],
        parameters=old['parameters'],target_envelope=old['quadrature_reference_abs_q']['target_nsnsns'],
        geometry=geometry,source_to_target=matrix,excluded_from_all_NS=excluded,
        target_orders=[8,12],quadrature_design={'3':[p['t'] for p in geometry],'4':[.6]},
        block_precision=50,structure_precision=40,global_tolerance=2e-9,
        primary_policy='Independent channel q^h outside descendant blocks and pairing matrices',
        transport_policy='Transport each raw spin separately; relative phase fixed by independent continued Majorana roots. Test the corresponding minimal interacting spin-basis proposal; no fitted normalization.',
        source_primary_max_relative_error=error,
        implementation_sha256={name:inputs.files[name] for name in source_files})
    path=output/'config.json'
    if path.exists() and json.loads(path.read_text())!=config:
        raise ValueError('Output directory has a different calculation design or implementation')
    save(path,config);save(output/'source.json',sources);save(output/'provenance.json',inputs.files)
    return config


def target_node(output,n,index):
    config=json.loads((output/'config.json').read_text())
    digest=object_digest(config)
    path=output/'target'/f'N{n}'/f'node-{index:03d}.json'
    if path.exists():
        old=json.loads(path.read_text())
        assert (old['config_digest'],old['N'],old['index'])==(digest,n,index)
        return
    start=time.monotonic()
    rules=_rules(config['target_envelope'],n)
    indices=np.unravel_index(index,(n,)*3)
    momenta=tuple(float(rules[e][0][j]) for e,j in enumerate(indices))
    measure=float(_measure(rules,indices))
    b=config['b'];Q=b+1/b;c=1.5+3*Q*Q
    weights=tuple(Q*Q/8+p*p/2 for p in momenta)
    constants=GenericSuperLiouvilleConstants(b,dps=config['structure_precision'],
        mu=complex(config['parameters']['mu']),
        include_cosmological_prefactor=config['parameters']['include_cosmological_prefactor'])
    couplings=constants.ns_constants(*momenta)
    rows=[]
    for point in config['geometry']:
        if point['t'] not in config['quadrature_design'][str(n)]: continue
        q=tuple(map(complex,point['q_target']))
        primary=cmath.exp(sum(h*cmath.log(z) for h,z in zip(weights,q)))
        recursion=NSGenus2CRecursion(channel='theta',q_values=q,global_method='resummed',
            global_tolerance=config['global_tolerance'],global_max_total_occupation=36,
            vacuum_word_length=7,vacuum_max_mode=50)
        raw_indices=[LIFTS.index(tuple(row['target_raw_lift'])) for row in point['spin_rows']]
        for order in config['target_orders']:
            density=np.zeros((2,2),complex);literal=np.zeros(4,complex)
            sectors=[]
            for sector in (0,1):
                f=np.array([recursion.collision_aware_block_mp(weights=weights,sector=sector,
                    recursion_order=order,lifts=lift,central_charge=c,
                    working_precision=config['block_precision']) for lift in LIFTS],complex)
                u=np.asarray(lift_conversion(sector))
                h=u@f
                d=h[raw_indices]
                coefficient=measure*abs(primary)**2*couplings[sector]**2
                density+=coefficient*np.outer(d,d.conjugate())
                literal+=coefficient*f*f.conjugate()
                sectors.append(dict(sector=sector,literal_blocks=list(map(encode,f)),
                                    raw_blocks=list(map(encode,h))))
            if recursion.global_nonconverged_calls: raise ArithmeticError('All-NS global sum failed')
            rows.append(dict(t=point['t'],order=order,primary=encode(primary),weights=weights,
                log_q=list(map(lambda z:encode(cmath.log(z)),q)),sectors=sectors,
                density=[[encode(z) for z in row] for row in density],
                literal_weighted=list(map(encode,literal)),
                global_nonconverged_calls=recursion.global_nonconverged_calls))
    save(path,dict(config_digest=digest,N=n,index=index,momenta=momenta,measure=measure,
        constants=list(map(encode,couplings)),rows=rows,seconds=time.monotonic()-start))


def run(output,workers):
    config=json.loads((output/'config.json').read_text())
    tasks=[(int(n),i) for n in config['quadrature_design'] for i in range(int(n)**3)]
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    def dispatch(task):
        n,i=task
        log=output/'logs'/f'N{n}-node-{i:03d}.log';log.parent.mkdir(parents=True,exist_ok=True)
        with log.open('w') as stream:
            result=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--output',str(output),
                '--node',str(n),str(i)],env=env,stdout=stream,stderr=subprocess.STDOUT)
        if result.returncode: raise RuntimeError(f'Node {task} failed; see {log}')
        return task
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(dispatch,task) for task in tasks]
        for done,future in enumerate(as_completed(futures),1):
            task=future.result()
            print(f'Completed {done}/{len(tasks)} target nodes; N={task[0]}, index={task[1]}.',flush=True)


def reduce(output):
    config=json.loads((output/'config.json').read_text());digest=object_digest(config)
    source=json.loads((output/'source.json').read_text())
    geometry={p['t']:p for p in config['geometry']}
    inputs=Inputs()
    totals={};literal_totals={}
    for n,points in config['quadrature_design'].items():
        n=int(n)
        for index in range(n**3):
            node=inputs.read(output/'target'/f'N{n}'/f'node-{index:03d}.json')
            assert (node['config_digest'],node['N'],node['index'])==(digest,n,index)
            assert len(node['rows'])==len(points)*len(config['target_orders'])
            for row in node['rows']:
                assert row['global_nonconverged_calls']==0
                key=(n,row['t'],row['order'])
                totals.setdefault(key,[]).append(np.array([[decode(z) for z in r] for r in row['density']]))
                literal_totals.setdefault(key,[]).append([decode(z) for z in row['literal_weighted']])
    def matrix_sum(values):
        return np.array([[csum(m[i,j] for m in values) for j in range(2)] for i in range(2)])
    totals={k:matrix_sum(v) for k,v in totals.items()}
    target_rows=[dict(N=n,t=t,order=order,density=[[encode(z) for z in r] for r in mat],
        literal_Z=[encode(csum(v[i] for v in literal_totals[n,t,order])) for i in range(4)])
        for (n,t,order),mat in sorted(totals.items())]
    save(output/'target_summary.json',target_rows)
    kappa=1+2*(config['b']+1/config['b'])**2
    rows=[]
    for sr in source:
        n,t,L=sr['N'],sr['t'],sr['L']
        source_matrix=np.array([[decode(z) for z in r] for r in sr['density']])
        for order in config['target_orders']:
            if (n,t,order) not in totals: continue
            target_matrix=totals[n,t,order]
            frame=(geometry[t]['source_free']/geometry[t]['target_free'])**kappa
            for label,v in [('s=+1',np.array([1,-1j])/math.sqrt(2)),
                            ('s=-1',np.array([1,1j])/math.sqrt(2)),
                            ('resolved_00',np.array([1,0])),('resolved_11',np.array([0,1]))]:
                zs=complex(v@source_matrix@v.conjugate());zt=complex(v@target_matrix@v.conjugate())
                ratio=zs/zt/frame
                assert abs(zs.imag)<1e-10*abs(zs) and abs(zt.imag)<1e-10*abs(zt)
                rows.append(dict(t=t,N=n,source_L=L,target_R=order,comparison=label,
                    source_Z=zs.real,target_Z=zt.real,free_frame_power=frame,
                    ratio_real=ratio.real,ratio_imag=ratio.imag,norm_ratio=abs(ratio),
                    phase_difference_rad=cmath.phase(ratio),relative_gap=abs(ratio-1)))
    write_csv(output/'comparison.csv',rows)
    save(output/'comparison.json',dict(config_digest=digest,kappa=kappa,rows=rows,
        conclusion='Test the proposed physical NSRR matrix with explicit minimal spin-basis transport; finite-cutoff results, no normalization fit.',
        target_nodes=sum(int(n)**3 for n in config['quadrature_design']),
        target_global_failures=0))
    save(output/'target_provenance.json',inputs.files)
    for row in rows:
        if row['target_R']==12 and (row['source_L']==5 or row['N']==4):
            print({k:row[k] for k in ('t','N','source_L','comparison','norm_ratio')},flush=True)
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=OUTPUT)
    parser.add_argument('--workers',type=int,default=3)
    parser.add_argument('--node',nargs=2,type=int)
    parser.add_argument('--reduce',action='store_true')
    args=parser.parse_args();args.output=args.output.resolve()
    if args.node: target_node(args.output,*args.node)
    elif args.reduce: reduce(args.output)
    else:
        prepare(args.output);run(args.output,args.workers);reduce(args.output)


if __name__=='__main__': main()
