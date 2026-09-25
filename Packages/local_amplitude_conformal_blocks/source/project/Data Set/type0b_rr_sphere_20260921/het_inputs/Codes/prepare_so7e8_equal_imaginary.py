#!/usr/bin/env python3
"""Freeze a nine-point equal-imaginary scan using verified literature sources."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import tarfile

import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source-run','frozen-root','baseline','output'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();repo=Path(__file__).resolve().parents[1]
    if a.output.exists():
        raise FileExistsError('choose a new versioned release')
    sys.path.insert(0,str(a.frozen_root.resolve()/'Codes'))
    from so7e8_literature_campaign import digest,pairs,atomic_json
    from so7e8_literature_atlas import CHARTS,four_ns_moduli_layout,geometric_assignment,channel_coordinates
    from so7e8_liouville_chamber import require_conservative_real_p_chamber
    original=json.loads((a.source_run/'manifest.json').read_text())
    check=dict(original);signature=check.pop('manifest_sha256');assert digest(check)==signature
    config=copy.deepcopy(original['config'])
    config['moduli']['settings'].update(ope_radius=.18,crossed_ope_radius=.18,
        disk_total_order=23,z_radial_nodes=16,z_angular_nodes=40)
    config['cluster'].update(walltime='00:30:00',maximum_concurrency=128)
    energies={};slice_points=[]
    for k in range(9):
        t=(4+3*k)/40;u=1j*t/3;mom=(u,u,u,3*u)
        chamber=require_conservative_real_p_chamber(mom[:3],mom[3])
        name=f't{round(1000*t):04d}'
        energies[name]=pairs(mom)
        slice_points.append(dict(energy=name,t_in=t,t_out=t/3,first_pole_margin=chamber.first_pole_margin))
    grids,_=four_ns_moduli_layout(config['moduli']['settings'])
    x=np.concatenate([v[0] for k,v in grids.items() if k.endswith('_bulk')])
    roles={};geometry=dict(numerical_points_per_observable=len(x),analytic_collar_radius=.18,
        local_degree=23,boundary_probe_points_per_chart=48,moduli_lenses='analytic')
    theta=2*np.pi*(np.arange(48)+.5)/48
    for obs in CHARTS:
        z=1-x if obs=='mixed' else x;owner,_=geometric_assignment(z,obs)
        coords=channel_coordinates(z,obs)
        crossed='s' if obs=='mixed' else 't'
        for j,chart in enumerate(CHARTS[obs]):
            radius=.18/(1-.18) if chart==crossed else .18
            roles[obs+'/'+chart]=np.concatenate((coords[chart][owner==j],radius*np.exp(1j*theta)))
    assignments=[];tasks=[];indices={};task_points={}
    for energy,values in energies.items():
        for obs,charts in CHARTS.items():
            base=[values[j] for j in (0,3,2,1)] if obs=='mixed' else values
            for chart,(family,sector,perm) in charts.items():
                momenta=[base[j] for j in perm]
                rule=original['momentum_rules'][sector]
                for j,(P,w) in enumerate(zip(rule['momenta'],rule['dP_weights'])):
                    role=dict(energy=energy,observable=obs,channel=chart,family=family,sector=sector,
                        momenta=momenta,momentum_index=j,P=P,dP_weight=w,point_set=obs+'/'+chart)
                    key=digest(dict(family=family,momenta=momenta,P=P,dP_weight=w))
                    if key not in indices:
                        indices[key]=len(tasks)
                        tasks.append(dict(role,point_set='native/'+family,seed_file=None))
                    assignments.append(dict(role,native_bank=indices[key]))
                    task_points.setdefault(family,set()).add(obs+'/'+chart)
    assert len(tasks)==864 and len(assignments)==1728
    native_points={'native/'+family:np.unique(np.concatenate([roles[r] for r in sorted(names)]))
                   for family,names in task_points.items()}
    source_files={}
    with tarfile.open(a.source_run/'sources.tar.gz','r:gz') as archive:
        for name,expected in original['source_sha256'].items():
            raw=archive.extractfile(name).read()
            if hashlib.sha256(raw).hexdigest()!=expected:raise ValueError('source archive changed: '+name)
            source_files[name]=raw
    for name in ('Codes/prepare_so7e8_equal_imaginary.py','Codes/so7e8_equal_imaginary_campaign.py',
                 'cluster/submit_so7e8_equal_imaginary.slurm','cluster/submit_so7e8_equal_imaginary.py'):
        source_files[name]=(repo/name).read_bytes()
    frozen_collars=repo/'data_exports/so7e8_literature_accuracy_20260921/source/enlarged_collars_v1'
    for name in ('integrate_so7e8_enlarged_collars.py','so7e8_analytic_crossed_lenses.py'):
        source_files['Codes/'+name]=(frozen_collars/name).read_bytes()
    source_files['reference/moduli/source/integrate_so7e8_literature_banks.py']=(a.baseline/'source/integrate_so7e8_literature_banks.py').read_bytes()
    source_files['reference/moduli/provenance.json']=(a.baseline/'provenance.json').read_bytes()
    for name,raw in source_files.items():
        path=a.output/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
    points_raw=(json.dumps({k:pairs(v) for k,v in native_points.items()},separators=(',',':'))+'\n').encode()
    roles_raw=(json.dumps({k:pairs(v) for k,v in roles.items()},separators=(',',':'))+'\n').encode()
    data=dict(schema='so7e8-equal-imaginary-native-v1',config=config,energies=energies,
        momentum_rules=original['momentum_rules'],bank_tasks=tasks,assignments=assignments,
        slice_points=slice_points,geometry=geometry,
        point_sets_file='point_sets.json',point_sets_sha256=hashlib.sha256(points_raw).hexdigest(),
        role_point_sets_file='role_point_sets.json',role_point_sets_sha256=hashlib.sha256(roles_raw).hexdigest(),
        source_sha256={k:hashlib.sha256(v).hexdigest() for k,v in source_files.items()},
        prior_frozen_manifest_sha256=signature,prior_source_archive_sha256=hashlib.sha256((a.source_run/'sources.tar.gz').read_bytes()).hexdigest(),
        reuse_policy='Exact native family, ordered external momenta, P and dP weight; all physical channel roles retained.',
        physical_amplitude_certified=False,formula_used_for_bank_or_channel_selection=False)
    data['manifest_sha256']=digest(data)
    atomic_json(a.output/'native/manifest.json',data)
    (a.output/'native/point_sets.json').write_bytes(points_raw)
    (a.output/'native/role_point_sets.json').write_bytes(roles_raw)
    report=dict(status='prepared_not_submitted',native_banks=len(tasks),channel_node_assignments=len(assignments),
        energies=len(energies),momentum_nodes=32,manifest_sha256=data['manifest_sha256'],
        tasks_reused_by_exact_equal_momenta=len(assignments)-len(tasks),remote_jobs_submitted=False)
    atomic_json(a.output/'prepared.json',report)
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
