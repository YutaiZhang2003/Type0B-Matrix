#!/usr/bin/env python3
"""Spot-check saved L5 source blocks with the current 40-digit native engine.

Retain actual discrepancies and their weighted integral impact, including
the extreme momentum tail. This is not a replacement full-grid calculation.
"""
from pathlib import Path
import hashlib
import json
import math
import sys

ROOT=Path(__file__).resolve().parents[2]
for directory in ('Code/genus_2','Code/full_ramond_block_runtime','Code/c_Recursion','Code'):
    sys.path.insert(0,str(ROOT/directory))
from nsrr_cpp_backend import NativeNSRR,implementation_hashes
from recombine_saved_genus2_coefficient_ledger import decode,encode
from nsrr_bilinear_sewing import projected_components,contract_nsrr_bilinear,CHANNELS
from generic_super_liouville_structure_constants import GenericSuperLiouvilleConstants


def main():
    folder=ROOT/'Data Set/nsrr_trial_L5_N3_local_20260830'
    output=ROOT/'Data Set/nsrr_bilinear_cross_channel_20260915/fresh_source_checks.json'
    config=json.loads((folder/'summary.json').read_text())['config']
    paths=sorted((folder/'shards').glob('node-*.json'))
    report={'rows':[],'weighted_changes':[],'native_implementation':implementation_hashes(),
            'input_sha256':{},'scope':'Three representative nodes, not a full native grid rerun'}
    for index in (0,13,26):
        path=paths[index];shard=json.loads(path.read_text())
        report['input_sha256'][str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
        mom=shard['momenta_geometry'];runtime=NativeNSRR(config['b'],mom[::-1],5,dps=40)
        constants=GenericSuperLiouvilleConstants(config['b'],dps=40).rr_ns_constants(mom[1],mom[0],mom[2])
        error=max(abs(a/decode(b)-1) for a,b in zip(constants,shard['C_BRY']))
        assert error<3e-12
        components={(f,e,e):runtime.physical_components(f,e,e) for f in(0,1) for e in(1,-1)}
        for point in config['points']:
            q=tuple(map(complex,point['q_geometry']))[::-1]
            for level in (3.,5.):
                rows=[next(r for r in shard['rows'] if r['t']==point['t'] and r['level']==level
                           and r['lifts_geometry']==list(lift)) for lift in((1,1,1),(1,-1,1))]
                old={k:sum(decode(row['blocks'][i]) for row in rows)/math.sqrt(2) for i,k in enumerate(CHANNELS)}
                new=dict(old)
                for label,table in components.items():
                    new[label]=sum(projected_components(v)*math.prod(z**(n/2) for z,n in zip(q,exponent))
                                   for exponent,v in table.items() if sum(exponent)<=2*level)
                    scaled=abs(old[label]-new[label])/max(1,abs(old[label]),abs(new[label]))
                    report['rows'].append(dict(node=index,t=point['t'],L=level,channel=label,
                        saved=encode(old[label]),native=encode(new[label]),scaled_error=scaled,
                        constant_relative_error=error))
                primary=decode(rows[0]['primary'])
                for sign in (1,-1):
                    values=[]
                    for blocks in (old,new):
                        values.append(shard['measure']*contract_nsrr_bilinear(descendant_blocks=blocks,
                            antiholomorphic_blocks={k:z.conjugate() for k,z in blocks.items()},
                            left_bry=constants,right_bry=constants,physical_lifts_slots=(sign,-1,1),
                            primary=primary,antiholomorphic_primary=primary.conjugate())['total'])
                    report['weighted_changes'].append(dict(node=index,t=point['t'],L=level,s=sign,
                        saved_weighted=encode(values[0]),native_weighted=encode(values[1]),
                        absolute_change=abs(values[1]-values[0])))
        output.write_text(json.dumps(report,indent=2)+'\n')
        print('Native source check',index,'max block error',max(r['scaled_error'] for r in report['rows'] if r['node']==index),flush=True)
    report['checks']=len(report['rows'])
    report['max_block_error']=max(r['scaled_error'] for r in report['rows'])
    report['maximum_weighted_change']=max(r['absolute_change'] for r in report['weighted_changes'])
    output.write_text(json.dumps(report,indent=2)+'\n')
    print('Finished',report['checks'],'comparisons; max weighted change',report['maximum_weighted_change'])


if __name__=='__main__': main()
