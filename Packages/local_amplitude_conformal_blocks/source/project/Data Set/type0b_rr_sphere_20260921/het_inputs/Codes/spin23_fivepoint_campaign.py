"""One-energy, source-bound Cannon campaign with reusable CFT banks.

All 120 charts remain in the full-support estimator. Momentum-rule changes
use common random numbers, and separate fixed-moduli probes exhaust the
quadrature sums. Sampling a finite rule is not deterministic quadrature.
No incomplete sum or unmeasured systematic is an accuracy certificate.
"""
import argparse
from dataclasses import asdict, replace
import hashlib
import json
import math
from pathlib import Path
from time import perf_counter

import mpmath as mp
import numpy as np
from scipy.stats import qmc

from spin23_fivepoint_bank import atomic_json, digest, build_bank, load_bank
from spin23_fivepoint_integration import (ThresholdConfig, spectral_rules,
    spectral_proposal, continuation_audit, PatchSettings, _stratum_draw)
from spin23_fivepoint_quadrature import gaussian_spectral_rules
from spin23_fivepoint_type0b_reference import layers, source_manifest


def packed(value):
    a = np.asarray(value, complex)
    if not np.isfinite(a).all():
        raise ValueError('nonfinite numerical artifact')
    return dict(real=a.real.tolist(), imag=a.imag.tolist())


def unpacked(value):
    return np.asarray(value['real'])+1j*np.asarray(value['imag'])


def stats(replicates):
    a = np.asarray(replicates, complex)
    mean = a.mean(axis=0)
    se = np.sqrt(np.sum(abs(a-mean)**2, axis=0)/(len(a)*(len(a)-1)))
    return dict(mean=packed(mean), standard_error=se.tolist(),
        relative_two_se=np.divide(2*se, abs(mean), out=np.full_like(se, np.inf),
            where=abs(mean)>0).tolist())


def numerical_sources():
    root = Path(__file__).parent
    names = ('spin23_fivepoint_bank.py', 'spin23_fivepoint_campaign.py',
        'spin23_fivepoint_c_elliptic.py', 'spin23_fivepoint_density.py',
        'spin23_fivepoint_integration.py', 'spin23_fivepoint_type0b_reference.py',
        'spin23_fivepoint_quadrature.py',
        'spin23_sphere_fivepoint.py', 'spin23_super_liouville_data.py',
        'heterotic_so23_1to3_vvvv_fit_bundle/liouville_momentum_quadrature.py')
    return dict(local={n: hashlib.sha256((root/n).read_bytes()).hexdigest() for n in names},
        reference=source_manifest()['sha256'])


def rules_for_config(config):
    counts = config['momentum_counts']
    if config.get('momentum_scheme') == 'threshold_gaussian':
        if config.get('campaign_mode') != 'baseline':
            raise ValueError('whole-line Gaussian campaign currently requires baseline mode')
        return {'baseline': gaussian_spectral_rules(counts, config.get('momentum_envelopes'))}
    if config.get('momentum_scheme', 'threshold_weighted') != 'threshold_weighted':
        raise ValueError('unknown momentum scheme')
    a = ThresholdConfig()
    b = replace(a, endpoint=.187, tail=3.27,
        bulk_breakpoints=tuple(1.017*x for x in a.bulk_breakpoints))
    variants = {'baseline': (counts, (a, b))}
    for axis in range(2):
        for region, label in enumerate(('endpoint', 'bulk', 'tail')):
            changed = [list(c) for c in counts]
            changed[axis][region] += config['momentum_increments'][region]
            variants[f'{label}{axis}'] = (changed, (a, b))
    variants['envelope'] = (counts, tuple(replace(c, a=c.a*.7, s=.15) for c in (a, b)))
    return {name: spectral_rules(c, configs) for name, (c, configs) in variants.items()}


def prepare(run, config):
    run = Path(run)
    if run.exists():
        raise FileExistsError('use a new campaign directory')
    if config.get('campaign_mode') == 'baseline':
        if (config.get('momentum_scheme') != 'threshold_gaussian'
                or config['momentum_counts'] != [8, 8]):
            raise ValueError('baseline requires exactly 8x8 whole-half-line momentum nodes')
        if config['orders'] != [8] and not config.get('test_only'):
            raise ValueError('baseline starts at physical total-level 8; refinements are separate')
    elif config['orders'] != [8, 10, 12] and not config.get('test_only'):
        raise ValueError('legacy refinement campaign uses physical total-level 8/10/12')
    energies = tuple(complex(*v) for v in config['energies'])
    if config.get('frequency_domain') == 'pure_imaginary' and any(z.real != 0 for z in energies):
        raise ValueError('pure-imaginary campaign forbids nonzero real frequencies')
    if config.get('frequency_pattern') == 'equal_outgoing':
        t = config.get('t')
        if (type(t) not in (int, float) or not math.isfinite(t) or t <= 0
                or energies != (complex(0, t),)*4):
            raise ValueError('equal-outgoing campaign requires four identical frequencies i*t with finite t>0')
    audit = continuation_audit(energies)
    if not audit['straight_real_spectral_contours_safe']:
        raise ValueError('external contour requires a separate residue audit')
    sources = numerical_sources()
    source_id = digest(sources)
    rules = rules_for_config(config)
    names = tuple(rules)
    specs, ids = [], {}
    uses = []

    def bank_index(ordering, momenta):
        spec = dict(energies=config['energies'], ordering=list(ordering), momenta=list(momenta),
            order=max(config['orders']), precision=config['precision'],
            verification_precision=config['verification_precision'], source_id=source_id)
        key = digest(spec)
        if key not in ids:
            ids[key] = len(specs)
            specs.append(spec)
            uses.append([])
        return ids[key]

    geometry = []
    probabilities = {n: tuple(spectral_proposal(r) for r in rs) for n, rs in rules.items()}
    for replicate in range(config['replicates']):
        points = qmc.Sobol(11, scramble=True, seed=config['seed']+104729*replicate).random_base2(config['sample_power'])
        for index, u in enumerate(points):
            chart = min(int(120*u[0]), 119)
            geometry.append(dict(replicate=replicate, index=index, chart=chart, u=u[3:].tolist()))
            for variant, name in enumerate(names):
                rs, ps = rules[name], probabilities[name]
                js = [min(int(np.searchsorted(np.cumsum(p), v, side='right')), len(p)-1)
                    for p, v in zip(ps, u[1:3])]
                momenta = [float(r.momenta[j]) for r, j in zip(rs, js)]
                weight = 120*math.prod(float(r.weights[j]/p[j]) for r, p, j in zip(rs, ps, js))/math.pi**2
                bank = bank_index(layers().atlas.ORDERINGS[chart], momenta)
                uses[bank].append(dict(kind='amplitude', replicate=replicate, index=index,
                    variant=variant, u=u[3:].tolist(), weight=weight))

    x, y = complex(.17, .08), complex(.52, -.11)
    probes = [(tuple(range(5)), x, y, 1.), ((2, 3, 4, 1, 0), 1-y, x/(x-1), abs(x-1)**-4)]
    # These are complete deterministic continuum-quadrature sums at fixed moduli.
    for channel, (ordering, z, t, jacobian) in enumerate(probes):
        for variant, name in enumerate(names):
            r0, r1 = rules[name]
            for i, pa in enumerate(r0.momenta):
                for j, pb in enumerate(r1.momenta):
                    bank = bank_index(ordering, (float(pa), float(pb)))
                    uses[bank].append(dict(kind='probe', channel=channel, variant=variant,
                        coordinates=[[z.real,z.imag],[t.real,t.imag]],
                        weight=float(r0.weights[i]*r1.weights[j]*jacobian/math.pi**2),
                        region=[int(r0.regions[i]),int(r1.regions[j])]))
    manifest = dict(schema='hetso23-fivepoint-cannon-v1', config=config, sources=sources,
        source_id=source_id, continuation=audit, variants=list(names), banks=specs,
        uses=uses, geometry=geometry, quadrature={n:[r.metadata() for r in rs] for n,rs in rules.items()},
        bank_count=len(specs), amplitude_draws=len(geometry),
        normalization='raw-S reduced moduli integral; dPa*dPb/pi^2 included; external string factors excluded',
        truncation='elliptic joint total physical descendant level; sum(twice_levels)<=2*order; parity retained',
        amplitude_qualified=False, error_target=.03)
    manifest['manifest_sha256'] = digest(manifest)
    atomic_json(run/'manifest.json', manifest)
    return dict(bank_count=len(specs), amplitude_draws=len(geometry), variants=list(names),
        build_shards=config['shards'], concurrent_workers=config['concurrency'],
        baseline_momentum_nodes=[len(r.momenta) for r in rules['baseline']],
        baseline_momentum_pairs=math.prod(len(r.momenta) for r in rules['baseline']),
        physical_block_orders=config['orders'],
        probe_only_banks=sum(all(u['kind']=='probe' for u in us) for us in uses))


def read_manifest(run):
    m = json.loads((Path(run)/'manifest.json').read_text())
    signature = m.pop('manifest_sha256')
    if digest(m) != signature or m['sources'] != numerical_sources():
        raise ValueError('campaign source or manifest changed')
    m['manifest_sha256'] = signature
    return m


def bank_path(run, spec):
    key = digest(spec)
    return Path(run)/'banks'/key[:2]/(key+'.json')


def build_shard(run, task, tasks):
    m = read_manifest(run)
    if not 0 <= task < tasks:
        raise ValueError('invalid build shard')
    done = []
    for index in range(task, len(m['banks']), tasks):
        spec = m['banks'][index]
        outcome = build_bank(bank_path(run, spec), spec)
        done.append(index)
        print(json.dumps(dict(event='bank_complete', index=index, total=len(m['banks']), **outcome)), flush=True)
    atomic_json(Path(run)/'build_receipts'/f'{task:04d}.json',
        dict(manifest_sha256=m['manifest_sha256'], task=task, tasks=tasks, indices=done))


def evaluate_bank(run, m, index):
    output = Path(run)/'evaluated'/f'{index:06d}.json'
    if output.exists():
        old = json.loads(output.read_text())
        if old['manifest_sha256'] != m['manifest_sha256'] or old['index'] != index:
            raise ValueError('evaluation checkpoint identity mismatch')
        return
    spec = m['banks'][index]
    bank = load_bank(bank_path(run, spec), spec)
    config = m['config']
    prefixes = {order: bank.prefix(order) for order in config['orders']}
    settings = PatchSettings(sample_power=config['sample_power'], replicates=config['replicates'],
        collar=config['collar'], maximum_increment=config['ope_depth'],
        atlas_power=config['atlas_power'], seed=config['seed'])
    records = []
    started = perf_counter()
    for use in m['uses'][index]:
        record = dict(use)
        if use['kind'] == 'probe':
            x,y = (complex(*z) for z in use['coordinates'])
            values = np.asarray([b.values(x,y) for b in prefixes.values()], complex)*use['weight']
            record['values'] = packed(values)
            # Absolute AFTER each physical nonchiral node increment, BEFORE spectral sum.
            record['absolute_order_increments'] = abs(np.diff(values,axis=0)).tolist()
        else:
            values = np.asarray([_stratum_draw(b,use['u'],settings) for b in prefixes.values()])*use['weight']
            record['parts'] = packed(values)
            if use['variant'] == 0:
                record['collar_parts'] = packed(_stratum_draw(prefixes[max(prefixes)],use['u'],
                    replace(settings,collar=config['collar_control']))*use['weight'])
                record['depth_parts'] = packed(_stratum_draw(prefixes[max(prefixes)],use['u'],
                    replace(settings,maximum_increment=config['ope_depth_control']))*use['weight'])
        records.append(record)
    atomic_json(output, dict(index=index, manifest_sha256=m['manifest_sha256'], records=records,
        elapsed_seconds=perf_counter()-started))


def evaluate_shard(run, task, tasks):
    m = read_manifest(run)
    if not 0 <= task < tasks:
        raise ValueError('invalid evaluation shard')
    for index in range(task,len(m['banks']),tasks):
        evaluate_bank(run,m,index)
        print(json.dumps(dict(event='evaluated_bank',index=index,total=len(m['banks']))),flush=True)


def reduce(run):
    m = read_manifest(run)
    c = m['config']
    r,n,v,l = c['replicates'],2**c['sample_power'],len(m['variants']),len(c['orders'])
    amplitudes = np.zeros((r,n,v,l,3,4),complex)
    seen = np.zeros((r,n,v),bool)
    collars,depths = np.zeros((r,n,3,4),complex),np.zeros((r,n,3,4),complex)
    probes = np.zeros((2,v,l,3),complex)
    increments = np.zeros((2,v,l-1,3))
    regions = np.zeros((2,v,3,3,l,3),complex)
    for index in range(len(m['banks'])):
        path = Path(run)/'evaluated'/f'{index:06d}.json'
        row = json.loads(path.read_text())  # missing data is a failure, not zero
        if row['manifest_sha256'] != m['manifest_sha256'] or row['index'] != index:
            raise ValueError('result identity mismatch')
        if len(row['records']) != len(m['uses'][index]):
            raise ValueError('incomplete bank-use coverage')
        for original,record in zip(m['uses'][index],row['records']):
            if any(record[k] != value for k,value in original.items()):
                raise ValueError('result use differs from frozen sample manifest')
            if record['kind'] == 'amplitude':
                key = record['replicate'],record['index'],record['variant']
                if seen[key]:raise ValueError('duplicate original sample')
                seen[key]=True
                amplitudes[key]=unpacked(record['parts'])
                if key[2]==0:
                    collars[key[:2]]=unpacked(record['collar_parts'])
                    depths[key[:2]]=unpacked(record['depth_parts'])
            else:
                key=record['channel'],record['variant']
                value=unpacked(record['values'])
                probes[key]+=value
                increments[key]+=np.asarray(record['absolute_order_increments']).reshape(l-1,3)
                regions[key+tuple(record['region'])]+=value
    if not seen.all():raise ValueError('incomplete amplitude sample coverage')
    values=amplitudes.sum(axis=-1)
    rep=values.mean(axis=1)
    baseline=rep[:,0,-1]
    differences={name:stats(rep[:,j,-1]-baseline) for j,name in enumerate(m['variants']) if j}
    differences['collar']=stats(collars.sum(axis=-1).mean(axis=1)-baseline)
    differences['ope_depth']=stats(depths.sum(axis=-1).mean(axis=1)-baseline)
    differences['sobol_prefix']=stats(values[:,:n//2,0,-1].mean(axis=1)-baseline)
    order_changes={f'{a}_to_{b}':stats(rep[:,0,j+1]-rep[:,0,j])
        for j,(a,b) in enumerate(zip(c['orders'][:-1],c['orders'][1:]))}
    denominator=abs(probes[:,:,-1])
    relative=np.divide(increments,denominator[:,:,None,:],out=np.full_like(increments,np.inf),
        where=denominator[:,:,None,:]>0)
    result=dict(schema='hetso23-fivepoint-cannon-result-v1', manifest_sha256=m['manifest_sha256'],
        energies=c['energies'], process='S_to_VVVV', complete=True, amplitude_qualified=False,
        error_target_met=False, target=.03, normalization=m['normalization'],
        orders=c['orders'],variants=m['variants'], sample_count=r*n,
        reduced_integral=stats(baseline), order_changes=order_changes, paired_controls=differences,
        all_order_variant_replicate_values=packed(rep), deterministic_probes=packed(probes),
        deterministic_probe_regions=packed(regions), probe_absolute_node_increments=increments.tolist(),
        probe_relative_absolute_node_increments=relative.tolist(),
        probe_cross_channel_absolute_gap=abs(probes[1]-probes[0]).tolist(),
        convergence_diagnostics_are_not_remainder_bounds=True,
        campaign_mode=c.get('campaign_mode','full_refinement'),
        block_refinement_available=l>1, momentum_refinement_available=v>1,
        missing_refinements=(['block_order'] if l==1 else [])
            +(['momentum_order','momentum_envelope'] if v==1 else []),
        remaining_qualification=['physical adapter normalization/factorization',
            'global systematic error budget including noisy control differences',
            'holdout moduli/energy coverage and tail/threshold variation'])
    # JSON must never silently serialize nonfinite diagnostics.
    def finite(x):
        if isinstance(x,float) and not math.isfinite(x):return None
        if isinstance(x,list):return [finite(y) for y in x]
        if isinstance(x,dict):return {k:finite(y) for k,y in x.items()}
        return x
    atomic_json(Path(run)/'result.json',finite(result))
    print(json.dumps(finite(result['reduced_integral'])),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=('prepare','build','evaluate','reduce','preflight'))
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--config',type=Path)
    p.add_argument('--task',type=int,default=0)
    p.add_argument('--tasks',type=int,default=1)
    args=p.parse_args()
    if args.stage=='prepare':
        print(json.dumps(prepare(args.run,json.loads(args.config.read_text()))),flush=True)
    elif args.stage=='build':build_shard(args.run,args.task,args.tasks)
    elif args.stage=='evaluate':evaluate_shard(args.run,args.task,args.tasks)
    elif args.stage=='reduce':reduce(args.run)
    else:
        m=read_manifest(args.run)
        # First real bank is retained as production work, not a discarded timing table.
        build_bank(bank_path(args.run,m['banks'][0]),m['banks'][0])
        evaluate_bank(args.run,m,0)
        atomic_json(args.run/'preflight.json',dict(passed=True,bank_index=0,
            manifest_sha256=m['manifest_sha256'],production_bank_retained=True))


if __name__=='__main__':main()
