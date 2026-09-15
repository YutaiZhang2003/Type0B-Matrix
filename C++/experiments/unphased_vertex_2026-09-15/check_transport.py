"""Test the corrected explicit vertex transport through CCY and physical recovery."""
import hashlib
import json
import subprocess
from decimal import localcontext
from pathlib import Path
from compare import compare
from run_probes import run,ROOT,CPP,RESULTS

if __name__=='__main__':
    records=[];comparisons=[]
    for f in [0,1]:
        for mode in ['ordinary','inserted']:
            name=f'{mode}_transported_f{f}_L3'
            records.append(run(name,[ROOT/'ramond_unphased','--mode',mode,'--level','3',
                '--dps','40','--truncation','total','--f',str(f),'--vertex','transported',
                '--ns-branch-sign','keep','--sector-policy','record','--json',RESULTS/f'{name}.json']))
            reference=RESULTS/f'{mode}_physical_pbw_L3.json'
            if f:
                ref_name=f'{mode}_physical_pbw_f{f}_L3'
                reference=RESULTS/f'{ref_name}.json'
                records.append(run(ref_name,[CPP/'bin/pbw','--level','3','--f',str(f),'--mode',mode,
                    '--dps','40','--json',reference]))
            with localcontext() as context:
                context.prec=65
                result=compare(RESULTS/f'{name}.json',reference)
            comparisons.append(result)
            print(name,'max_scaled',result['max_scaled'],'failures',result['failed_components'],flush=True)
    records.append(run('vertex_ward_transported',[ROOT/'vertex_probe',RESULTS/'vertex_ward_transported.json']))
    sources=list((ROOT/'include/ramond').glob('*.hpp'))+list(ROOT.glob('*.cpp'))+list(ROOT.glob('*.py'))
    sources+=list((CPP/'include/ramond').glob('*.hpp'))+[CPP/'drivers/pbw_main.cpp',ROOT/'Makefile']
    hashes={str(p.relative_to(CPP)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    summary=dict(scope='total physical level <=3, p=0, f=0,1, eta=1, eta_prime=+/-1',dps=40,
                 phase='(-i)^(A mod 2) (-1)^[f(A+B+alpha)+p1(mathsfB+mathsfb)] times the literal Koszul product',
                 runs=records,comparisons=comparisons,source_sha256=hashes,
                 corrected_DV_passes=all(x['failed_components']==0 for x in comparisons))
    (RESULTS/'corrected_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    if not summary['corrected_DV_passes']:
        raise SystemExit('Corrected double-Virasoro comparison failed')
