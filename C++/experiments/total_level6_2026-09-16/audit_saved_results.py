"""Check complete total-degree coverage and physical loop sectors from saved data.

This does not recompute a block or use a recovered coefficient as a reference.
"""
from pathlib import Path
from decimal import Decimal as D, localcontext
from itertools import product
from collections import defaultdict
import hashlib
import json
import time

ROOT = Path(__file__).resolve().parent
ZERO = (D(0), D(0))
def sign(n): return -1 if n % 2 else 1
def phase(z, p):
    a,b=z
    return [(a,b),(-b,a),(-a,-b),(b,-a)][p%4]
def scaled(x,y):
    error=((x[0]-y[0])**2+(x[1]-y[1])**2).sqrt()
    nx=(x[0]**2+x[1]**2).sqrt(); ny=(y[0]**2+y[1]**2).sqrt()
    return error/max(D(1),nx,ny)
def domain(steps,budget):
    if not steps:
        yield (); return
    for n in range(0,budget+1,steps[0]):
        for tail in domain(steps[1:],budget-n):yield (n,)+tail

def graph_sign(mask,slots,pairs):
    ordered=sum((list(x) for x in pairs),[])
    return sign(sum(((mask>>slots[a//3][a%3])&1)*((mask>>slots[b//3][b%3])&1)
                    for j,a in enumerate(ordered) for b in ordered[j+1:] if a>b))

def audit(name):
    result=json.loads((ROOT/f'{name}_L6_results.json').read_text())
    if name=='mercedes':
        steps=[1,1,1,2,2,2]; cases=64
        slots=[[0,1,2],[0,3,5],[1,4,3],[2,5,4]]
        pairs=[[0,3],[1,6],[2,9],[4,8],[7,11],[10,5]]
        loops=[(56,3)]
        def eigen(ci,loop):return sign((ci%8).bit_count())
    elif name=='theta_ns':
        steps=[1,1,1];cases=2;slots=[[0,1,2],[0,1,2]];pairs=[[0,3],[1,4],[2,5]];loops=[]
        def eigen(ci,loop):return 1
    else:
        steps=[1,2 if name=='glasses_rr' else 1,1 if name=='glasses_nn' else 2]
        cases={'glasses_rr':8,'glasses_nr':4,'glasses_nn':2}[name]
        slots=[[0,1,1],[0,2,2]];pairs=[[0,3],[1,2],[4,5]]
        loops=([(2,1),(4,1)] if name=='glasses_rr' else [(4,1)] if name=='glasses_nr' else [])
        def eigen(ci,loop):return sign((ci//2)%2 if loop==2 else ci%2)
    expected=set(domain(steps,12));seen=defaultdict(set)
    split_components=0
    for case in result['cases']:
        marked=case['insertions']
        # Marked edges are Ramond in every suite. Counting by maximum segment
        # level gives 2*n+1 pairs with max(left,right)=n.
        distribution={0:1};remaining_marked=marked
        for step in steps:
            split=step==2 and remaining_marked>0
            remaining_marked-=int(split)
            weights={k:(k+1 if split else 1) for k in range(0,13,step)}
            distribution={k:sum(v*weights.get(k-j,0) for j,v in distribution.items()) for k in range(13)}
        expanded=sum(distribution.values())
        count=(expanded-len(expected))*(1<<len(steps))
        assert case['unequal_split']['components']==count,(name,case['case'],'incomplete split domain')
        assert case['forward']['components']==len(expected)*(1<<len(steps))
        split_components+=count
    kernel={}
    for loop,_ in loops:
        for p in range(1<<len(steps)):
            local=sum(((loop>>v[1])&1)*((p>>v[2])&1) for v in slots)
            kernel[loop,p]=graph_sign(loop^p,slots,pairs)*graph_sign(loop,slots,pairs)*graph_sign(p,slots,pairs)*sign(local)
    worst=D(0);failed=0;components=0
    for line in (ROOT/f'{name}_L6_coefficients.jsonl').open():
        row=json.loads(line);ci=row['case'];k=tuple(row['level2'])
        assert k in expected and k not in seen[ci],(name,ci,k)
        seen[ci].add(k)
        physical={p:(D(a),D(b)) for p,a,b in row['physical']}
        for loop,length in loops:
            components+=1<<len(steps)
            for p in set(physical)|{p^loop for p in physical}:
                x=phase(physical.get(p^loop,ZERO),-length)
                factor=graph_sign(loop,slots,pairs)*kernel[loop,p^loop]*eigen(ci,loop)
                x=(factor*x[0],factor*x[1]);error=scaled(x,physical.get(p,ZERO))
                worst=max(worst,error);failed+=error>D('1e-18')
    assert len(seen)==cases,(name,'missing cases',len(seen),cases)
    for ci in range(cases):assert seen[ci]==expected,(name,ci,'incomplete degree coverage')
    assert result['failed_cases']==0,(name,'block comparison failed')
    return dict(cases=cases,multidegrees_per_case=len(expected),coefficient_rows=sum(map(len,seen.values())),
                complete_total_level_6=True,unequal_split_components=split_components,loop_sector_components=components,
                loop_sector_max_scaled=str(worst),loop_sector_failed=failed,passed=not failed)

def main():
    start=time.perf_counter()
    results={name:audit(name) for name in ['theta_ns','glasses_rr','glasses_nr','glasses_nn','mercedes']}
    theta=json.loads((ROOT/'theta_summary.json').read_text());assert theta['passed']
    expected=set(domain([1,2,2],12))
    for run in theta['runs']:
        data=json.loads((ROOT/(run['name']+'.json')).read_text())
        seen={(x['exponents'][0],2*x['exponents'][1],x['exponents'][3]) for x in data['direct_physical_pbw']}
        assert seen==expected,(run['name'],'theta coverage incomplete')
    results['theta_nsrr']=dict(cases=len(theta['runs']),multidegrees_per_case=len(expected),complete_total_level_6=True,passed=True)
    result=dict(results=results,runtime_seconds=time.perf_counter()-start,passed=all(x['passed'] for x in results.values()))
    (ROOT/'coverage_and_sector_results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':
    with localcontext() as context:
        context.prec=65
        main()
