"""Directed tests of the requested unphased vertex / removed physical NS sign."""
import hashlib
import json
import subprocess
import sys
import time
from decimal import Decimal, localcontext
from pathlib import Path
from compare import compare

ROOT=Path(__file__).resolve().parent
CPP=ROOT.parents[1]
RESULTS=ROOT/'results'

def run(name,command):
    started=time.perf_counter()
    result=subprocess.run(list(map(str,command)),capture_output=True,text=True)
    record=dict(name=name,command=list(map(str,command)),exit_code=result.returncode,
                wall_seconds=time.perf_counter()-started,stderr=result.stderr)
    (RESULTS/f'{name}.log').write_text(result.stderr)
    print(name,'exit',result.returncode,'wall',round(record['wall_seconds'],3),flush=True)
    if result.returncode:
        raise RuntimeError(result.stderr)
    return record

if __name__=='__main__':
    RESULTS.mkdir(exist_ok=True)
    records=[]
    for mode in ['ordinary','inserted']:
        for vertex,sign in [('production','keep'),('unphased','keep'),('unphased','remove')]:
            name=f'{mode}_{vertex}_{sign}_L3'
            records.append(run(name,[ROOT/'ramond_unphased','--mode',mode,'--level','3',
                '--dps','40','--truncation','total','--vertex',vertex,'--ns-branch-sign',sign,
                '--sector-policy','record','--json',RESULTS/f'{name}.json']))
        name=f'{mode}_literal_pbw_L3'
        records.append(run(name,[ROOT/'literal_pbw','3','0',mode,RESULTS/f'{name}.json']))
        name=f'{mode}_physical_pbw_L3'
        records.append(run(name,[CPP/'bin/pbw','--level','3','--f','0','--mode',mode,
                                 '--dps','40','--json',RESULTS/f'{name}.json']))
    records.append(run('vertex_ward',[ROOT/'vertex_probe',RESULTS/'vertex_ward.json']))
    comparisons=[]
    with localcontext() as context:
        context.prec=65
        for mode in ['ordinary','inserted']:
            for method in ['production_keep','unphased_keep','unphased_remove','literal_pbw']:
                comparisons.append(compare(RESULTS/f'{mode}_{method}_L3.json',
                                           RESULTS/f'{mode}_physical_pbw_L3.json'))
    (RESULTS/'comparison.json').write_text(json.dumps(comparisons,indent=2)+'\n')
    sources=list((ROOT/'include/ramond').glob('*.hpp'))+list(ROOT.glob('*.cpp'))+list(ROOT.glob('*.py'))
    sources+=list((CPP/'include/ramond').glob('*.hpp'))+[CPP/'drivers/pbw_main.cpp',ROOT/'Makefile']
    hashes={str(p.relative_to(CPP)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    summary=dict(scope='total physical level <=3, p=f=0, eta=1, eta_prime=+/-1',dps=40,
                 runs=records,comparisons=comparisons,source_sha256=hashes,
                 definition_check_passes=all(x['failed_components']==0 for x in comparisons if 'literal_pbw' in x['candidate']),
                 production_control_passes=all(x['failed_components']==0 for x in comparisons if 'production_keep' in x['candidate']),
                 unphased_DV_candidate_passes=all(x['failed_components']==0 for x in comparisons if 'unphased' in x['candidate']))
    (RESULTS/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print({k:v for k,v in summary.items() if k.endswith('passes')})
    if not summary['definition_check_passes'] or not summary['production_control_passes']:
        sys.exit(1)
