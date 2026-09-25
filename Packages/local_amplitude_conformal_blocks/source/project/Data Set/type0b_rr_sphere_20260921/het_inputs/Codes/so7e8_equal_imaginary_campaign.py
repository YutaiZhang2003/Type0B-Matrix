#!/usr/bin/env python3
"""Expand exact native aliases, gate channels, integrate, and audit this scan."""
import argparse
import copy
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path

import numpy as np

from so7e8_literature_campaign import ROOT,read_manifest,digest,atomic_json,_matching_record,_save_record,unpairs


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_once(path,value):
    path=Path(path)
    if path.exists():
        if json.loads(path.read_text())!=value:raise ValueError('immutable derived input changed: '+str(path))
    else:atomic_json(path,value)


def expand(root):
    root=Path(root);native=read_manifest(root/'native')
    payloads=[];native_hashes={}
    for index,task in enumerate(native['bank_tasks']):
        path=root/'native/banks'/f'{index:04d}.json'
        payload=_matching_record(path,native,task)
        table=unpairs(payload['coefficients'])
        assert payload['order']==5 and table.shape[0]==2 and table.shape[-1]==11 and np.isfinite(table).all()
        payloads.append(payload);native_hashes[str(index)]=sha(path)
    parent=dict(schema='so7e8-exact-native-aliases-v1',config=native['config'],energies=native['energies'],
        momentum_rules=native['momentum_rules'],bank_tasks=native['assignments'],source_sha256=native['source_sha256'],
        native_manifest_sha256=native['manifest_sha256'],native_bank_sha256=native_hashes,
        alias_policy=native['reuse_policy'])
    parent['manifest_sha256']=digest(parent)
    run=root/'run';write_once(run/'parent/manifest.json',parent)
    hashes={};records=[]
    for index,task in enumerate(parent['bank_tasks']):
        source=task['native_bank'];original=native['bank_tasks'][source]
        for key in ('energy','family','sector','momenta','momentum_index','P','dP_weight'):
            if original[key]!=task[key]:raise ValueError('unsafe native alias: '+key)
        path=run/'parent/banks'/f'{index:04d}.json'
        payload=copy.deepcopy(payloads[source])
        payload['exact_native_alias']=dict(index=source,source_sha256=native_hashes[str(source)])
        if path.exists():assert _matching_record(path,parent,task)==payload
        else:_save_record(path,parent,task,payload)
        hashes[str(index)]=sha(path)
        records.append(dict(parent=True,index=index,sha256=hashes[str(index)],native_bank=source,order=5))
    points=root/'native'/native['role_point_sets_file']
    if sha(points)!=native['role_point_sets_sha256']:raise ValueError('role geometry changed')
    target=run/'point_sets.json'
    if target.exists():assert target.read_bytes()==points.read_bytes()
    else:target.write_bytes(points.read_bytes())
    manifest=dict(schema='so7e8-literature-regional-atlas-cannon-v1',config=native['config'],energies=native['energies'],
        momentum_rules=native['momentum_rules'],bank_tasks=[],source_sha256=native['source_sha256'],
        parent_manifest_sha256=parent['manifest_sha256'],parent_bank_sha256=hashes,
        point_sets_file='point_sets.json',point_sets_sha256=sha(target),geometry=native['geometry'],
        native_manifest_sha256=native['manifest_sha256'],physical_amplitude_certified=False)
    manifest['manifest_sha256']=digest(manifest);write_once(run/'manifest.json',manifest)
    from resolve_so7e8_literature_atlas import reduce_regional_atlas
    result=reduce_regional_atlas(run)
    audit=dict(status='passed',manifest_sha256=manifest['manifest_sha256'],
        native_manifest_sha256=native['manifest_sha256'],records=records,
        bank_count=len(records),distinct_native_bank_count=len(payloads),
        complete_energy_channel_node_coverage=True,finite_complete_component_tables=True,
        atlas_completion_sha256=sha(run/'atlas_completion.json'),
        moduli_grid_adjacent_orders_pass=result['moduli_grid_adjacent_orders_pass'])
    atomic_json(run/'integrity_audit.json',audit)
    if not result['moduli_grid_adjacent_orders_pass']:
        raise ArithmeticError('uncovered moduli points require targeted refinement; unchanged physical cap is 10')
    return dict(status='all_channels_available_and_regional_selection_passed',
        maximum_selected_change=result['maximum_selected_change'],overrides=result['override_count'])


def integrate(root,index):
    root=Path(root);native=read_manifest(root/'native');energy=list(native['energies'])[index]
    destination=root/'amplitudes'/(energy+'.json')
    if destination.exists():
        old=json.loads(destination.read_text())
        if old['status']=='enlarged_analytic_collars_computed' and len(old['mixed'])==8 and len(old['four_ramond'])==1:
            assert old['source_manifest_sha256']==read_manifest(root/'run')['manifest_sha256']
            return dict(status='reused_completed_amplitude',energy=energy,sha256=sha(destination))
    from integrate_so7e8_enlarged_collars import compute
    args=argparse.Namespace(frozen_root=root,run=root/'run',baseline=root/'reference/moduli',
        output=root/'amplitudes',energy=energy,observable='both',radius=.18,radial=16,angular=40,
        local_degree=23,reuse_collars=None)
    result=compute(args)
    return dict(status=result['status'],energy=energy,seconds=result['seconds'],sha256=sha(destination))


def finish(root):
    root=Path(root);native=read_manifest(root/'native');manifest=read_manifest(root/'run');rows=[]
    for point in native['slice_points']:
        path=root/'amplitudes'/(point['energy']+'.json');r=json.loads(path.read_text())
        assert r['status']=='enlarged_analytic_collars_computed' and r['source_manifest_sha256']==manifest['manifest_sha256']
        assert set(r['mixed'])=={f'{s}/{p}' for s in ('SS','SV','VS','VV') for p in ('one','infinity')}
        assert set(r['four_ramond'])=={'rrrr/reference'}
        for obs in ('mixed','rrrr'):
            assert r['coverage'][obs]['complete_order_resolved_coverage']
            for c in r['patch_checks'][obs].values():
                assert c['maximum_block_change']<=.02 and c['maximum_component_local_series_change']<=1e-6
        for values in (r['mixed'],r['four_ramond']):
            for v in values.values():
                for order in ('4','5'):
                    a=unpairs(v['orders'][order]['value']);pieces=v['orders'][order]['pieces']
                    assert len(pieces)==6 and np.isfinite(a).all()
                    assert np.allclose(sum(unpairs(x) for x in pieces.values()),a,rtol=1e-13,atol=1e-11)
        rows.append(dict(point,result_sha256=sha(path),maximum_selected_change=max(c['maximum_selected_change'] for c in r['coverage'].values())))
    result=dict(status='nine_point_moduli_scan_completed',verified_at=datetime.now(timezone.utc).isoformat(),
        native_manifest_sha256=native['manifest_sha256'],manifest_sha256=manifest['manifest_sha256'],
        rows=rows,energies=9,amplitude_cases=45,tensor_coefficients=81,
        momentum_nodes=32,moduli_integration_performed=True,physical_amplitude_certified=False,
        total_numerical_error_certified=False,formula_used_for_selection=False)
    atomic_json(root/'completion.json',result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=('expand','integrate','finish'))
    p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--task',type=int,default=0)
    a=p.parse_args()
    result=integrate(a.root,a.task) if a.stage=='integrate' else globals()[a.stage](a.root)
    print(json.dumps(result,indent=2))
