#!/usr/bin/env python3
"""Use analytic OPE collars at all three degenerations, and a smaller bulk grid."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

import so7e8_analytic_crossed_lenses as L


def patch_check(I,rows,radius,degree):
    from resolve_so7e8_literature_atlas import channel_order_errors
    from so7e8_literature_atlas import pointwise_adjacent_change
    points=radius*np.exp(2j*np.pi*(np.arange(48)+.5)/48)
    changes,_=channel_order_errors(rows,points)
    geometry=I.BlockGeometry.build(points)
    local_error=0.
    for task,payload in rows:
        for block in I.native_pair(task,payload,5):
            series=I.local_blocks(block,degree)
            exact=I.block_grid_values(block,geometry)
            approximation=np.zeros_like(exact)
            for index,terms in series.items():
                approximation[index]=sum(t.value(points) for t in terms)
            local_error=max(local_error,float(max(pointwise_adjacent_change(exact,approximation))))
    return dict(boundary_radius=radius,degree=degree,maximum_block_change=float(max(changes)),
                maximum_component_local_series_change=local_error,
                sample_count=48,all_nodes_and_both_chiralities=True)


def compute(a):
    started=time.monotonic()
    observables=('mixed','rrrr') if a.observable=='both' else (a.observable,)
    I,integrator_hash=L.load_integrator(a.frozen_root,a.baseline)
    # The preflight, disks, and two lens charts reuse identical local series.
    # Cache by complete numerical block contents, never just by momentum index.
    uncached_local=I.local_blocks
    expansion_cache={}
    cache_hits=0
    def cached_local(block,degree):
        nonlocal cache_hits
        key=(block.family,block.p,complex(block.internal.p),degree,block.table.shape,block.table.tobytes())
        if key not in expansion_cache:
            expansion_cache[key]=uncached_local(block,degree)
        else:
            cache_hits+=1
        return expansion_cache[key]
    I.local_blocks=cached_local
    from resolve_so7e8_literature_atlas import resolve_energy
    manifest,records,_=I.load_records(a.run,a.energy)
    mom=tuple(I.unpairs(manifest['energies'][a.energy]))
    constants=I.SelfDualConstants(manifest['config']['precision'])
    reuse=None
    if a.reuse_collars:
        reuse=json.loads(a.reuse_collars.read_text())
        if (reuse['energy']!=a.energy or reuse['source_manifest_sha256']!=manifest['manifest_sha256']
                or reuse['integrator_sha256']!=integrator_hash
                or reuse['analytic_lens_sha256']!=hashlib.sha256(Path(L.__file__).read_bytes()).hexdigest()
                or reuse['settings']['ope_radius']!=a.radius
                or reuse['settings']['crossed_ope_radius']!=a.radius
                or reuse['settings']['disk_total_order']!=a.local_degree
                or reuse['status']!='enlarged_analytic_collars_computed'):
            raise ValueError('incompatible analytic collar receipt')
        for observable in observables:
            expected=({f'{s}/{p}' for s in I.SPECIES for p in I.PICTURES}
                      if observable=='mixed' else {'rrrr/reference'})
            if set(reuse['mixed' if observable=='mixed' else 'four_ramond'])!=expected:
                raise ValueError('incomplete analytic collar receipt')
    settings=dict(manifest['config']['moduli']['settings'],ope_radius=a.radius,
        crossed_ope_radius=a.radius,z_radial_nodes=a.radial,z_angular_nodes=a.angular,
        disk_total_order=a.local_degree)
    if not 0<a.radius<.4:
        raise ValueError('require separated collars with radius below 0.4')
    grids,_=I.four_ns_moduli_layout(settings)
    grids={k:v for k,v in grids.items() if k.endswith('_bulk')}
    names=np.concatenate([np.repeat(k,len(v[0])) for k,v in grids.items()])
    x=np.concatenate([v[0] for v in grids.values()])
    weights=np.concatenate([v[1] for v in grids.values()])
    report=dict(schema='so7e8-enlarged-analytic-collars-v1',energy=a.energy,momenta=I.pairs(mom),
        settings=settings,numerical_moduli_points=len(x),orders=[4,5],momentum_nodes=32,
        new_recursion=False,new_cannon_jobs=False,formula_used_for_selection=False,
        source_manifest_sha256=manifest['manifest_sha256'],integrator_sha256=integrator_hash,
        analytic_lens_sha256=hashlib.sha256(Path(L.__file__).read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        total_numerical_error_certified=False,physical_amplitude_certified=False,
        requested_observables=list(observables),observables_completed=[],
        analytic_regions=['original_zero_disk','exterior_zero_disk','original_lens','exterior_lens'],
        mixed={},four_ramond={},coverage={},patch_checks={})
    a.output.mkdir(parents=True,exist_ok=True)
    destination=a.output/(a.energy+'.json')
    for observable in observables:
        # Fixed analytic charts: zero, infinity, and crossed collar.
        zero='t' if observable=='mixed' else 's'
        crossed='s' if observable=='mixed' else 't'
        checks={}
        if reuse:
            checks=reuse['patch_checks'][observable]
        else:
            for chart,radius in ((zero,a.radius),('u',a.radius),(crossed,a.radius/(1-a.radius))):
                checks[chart]=patch_check(I,records[observable,chart],radius,a.local_degree)
        report['patch_checks'][observable]=checks
        # Local degree is independently checked; its threshold is much tighter
        # than the unchanged physical adjacent-block-order criterion.
        if any(c['maximum_block_change']>.02 or c['maximum_component_local_series_change']>1e-6 for c in checks.values()):
            report['status']='collar_preflight_failed'
            I.atomic_json(destination,report)
            raise ArithmeticError('enlarged collar requires a smaller radius or higher local degree')
        z=1-x if observable=='mixed' else x
        owner,_=I.geometric_assignment(z,observable)
        coordinates=I.channel_coordinates(z,observable)
        indices={c:np.flatnonzero(owner==j) for j,c in enumerate(I.CHARTS)}
        pointsets={observable+'/'+c:coordinates[c][indices[c]] for c in I.CHARTS}
        coverage=resolve_energy(observable,pointsets,{c:records[observable,c] for c in I.CHARTS},tolerance=.02)
        report['coverage'][observable]=coverage
        if coverage['unresolved']:
            report['status']='bulk_preflight_failed'
            I.atomic_json(destination,report)
            raise ArithmeticError('unresolved bulk queries')
        for o in coverage['overrides']:
            index=indices[o['original_channel']][o['original_point_index']]
            if abs(z[index]-complex(*o['canonical_z']))>1e-12:
                raise ValueError('override coordinate mismatch')
            owner[index]=I.CHARTS.index(o['selected_channel'])
        keys=list(I.product(I.SPECIES,I.PICTURES)) if observable=='mixed' else [('rrrr','reference')]
        pieces={n:{k:{name:np.zeros(2 if k[0]=='VV' else 4 if observable=='rrrr' else 1,complex)
                     for name in grids} for k in keys} for n in (4,5)}
        times=(mom[0],-mom[3],mom[2],mom[1])
        for j,chart in enumerate(I.CHARTS):
            mask=owner==j
            if not np.any(mask):
                continue
            zz=z[mask];native=coordinates[chart][mask]
            if observable=='rrrr':
                raw=I.four_r_grid(records[observable,chart],native,constants,a.output.parent/'spectators')
                values={n:{keys[0]:np.array([I.four_ramond_channel_to_base(v,p,chart) for v,p in zip(raw[n],zz)])} for n in (4,5)}
            else:
                outer={k:I.chart_outer(k[0][::-1],times,chart,k[1]) for k in keys}
                values=I.mixed_grid(records[observable,chart],native,outer,constants)
                if chart=='u':
                    for n in (4,5):
                        for k in keys:
                            values[n][k]=I.exchange_mixed_ns_tensor(values[n][k],k[0][::-1])/abs(1-zz)[:,None]**4
            for n in (4,5):
                for k in keys:
                    for name in grids:
                        use=names[mask]==name
                        pieces[n][k][name]+=np.sum(weights[mask][use,None]*values[n][k][use],axis=0)
        print(a.energy,observable,'bulk integrated',len(x),'points',flush=True)
        diagnostics={}
        if reuse:
            previous=reuse['mixed' if observable=='mixed' else 'four_ramond']
            diagnostics=reuse['analytic_diagnostics'][observable]
            for n in (4,5):
                for k in keys:
                    for name in diagnostics:
                        pieces[n][k][name]=I.unpairs(previous['/'.join(k)]['orders'][str(n)]['pieces'][name])
            report['reused_collar_receipt_sha256']=hashlib.sha256(a.reuse_collars.read_bytes()).hexdigest()
        else:
            for exterior,chart in ((False,zero),(True,'u')):
                name='exterior_zero_disk' if exterior else 'original_zero_disk'
                if observable=='rrrr':
                    vals,diag=I.four_r_disk(records[observable,chart],constants,a.radius,a.local_degree)
                    vals={n:{keys[0]:I.four_ramond_channel_to_base(v,1j,'u') if exterior else v} for n,v in vals.items()}
                else:
                    outer={k:I.chart_outer(k[0][::-1],times,chart,k[1]) for k in keys}
                    vals,diag=I.mixed_disk(records[observable,chart],outer,constants,a.radius,a.local_degree)
                    if exterior:
                        vals={n:{k:I.exchange_mixed_ns_tensor(v,k[0][::-1]) for k,v in vv.items()} for n,vv in vals.items()}
                for n in (4,5):
                    for k in keys:
                        pieces[n][k][name]=vals[n][k]
                diagnostics[name]=diag
            lenses=L.integrate_lenses(I,records,constants,a.radius,a.local_degree,observable=observable,momenta=mom)
            for name,row in lenses.items():
                diagnostics[name]=row['diagnostic']
                for n in (4,5):
                    for k in keys:
                        pieces[n][k][name]=row['values'][n]['/'.join(k)]
        target=report['mixed' if observable=='mixed' else 'four_ramond']
        for k in keys:
            values={str(n):I.summarize_pieces(pieces[n][k]) for n in (4,5)}
            hi,lo=(I.unpairs(values[str(n)]['value']) for n in (5,4))
            target['/'.join(k)]=dict(orders=values,component_adjacent_changes=(abs(hi-lo)/np.maximum(abs(hi),1e-300)).tolist())
        report.setdefault('analytic_diagnostics',{})[observable]=diagnostics
        report['seconds']=time.monotonic()-started
        report['local_expansion_cache']=dict(entries=len(expansion_cache),hits=cache_hits)
        report['observables_completed'].append(observable)
        report['status']=('enlarged_analytic_collars_computed'
                          if len(report['observables_completed'])==len(observables)
                          else 'enlarged_analytic_collars_partial')
        I.atomic_json(destination,report)
        print(a.energy,observable,'analytic collars integrated; elapsed',round(report['seconds'],1),flush=True)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('frozen-root','run','baseline','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--energy',required=True)
    p.add_argument('--observable',choices=('both','mixed','rrrr'),default='both')
    p.add_argument('--radius',type=float,default=.18)
    p.add_argument('--radial',type=int,default=16)
    p.add_argument('--angular',type=int,default=40)
    p.add_argument('--local-degree',type=int,default=23)
    p.add_argument('--reuse-collars',type=Path)
    compute(p.parse_args())
